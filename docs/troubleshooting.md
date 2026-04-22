# Troubleshooting Notes

## SIP 등록 타임아웃 (TimeoutError: Registering on SIP Server timed out)

### 증상

```
TimeoutError: Registering on SIP Server timed out
```

pyVoIP가 Asterisk에 SIP 등록 시도 시 30초 타임아웃 발생.
Zoiper 소프트폰은 동일 머신에서 동일 서버에 정상 등록됨.

### 환경

- 클라이언트: macOS (192.168.35.103, LAN IP)
- VPN 연결: 172.31.90.0/24 대역 VPN을 통해 Asterisk 서버 접근
- Asterisk 서버: 172.31.79.202:5060

### 원인

`config.py`의 `get_local_ip()`가 `8.8.8.8`(인터넷)로 연결해 로컬 IP를 감지했기 때문에
LAN 인터페이스 IP(`192.168.35.103`)를 반환했음.

그러나 Asterisk 서버는 VPN을 통해서만 접근 가능하므로,
실제 SIP 패킷은 VPN 인터페이스(`172.31.90.x`)를 통해 나가고
Asterisk의 응답도 VPN 인터페이스로 돌아옴.

pyVoIP가 `192.168.35.103`에 소켓을 바인딩한 상태에서는
VPN 인터페이스로 들어오는 응답을 수신할 수 없어 타임아웃 발생.

```
[기존 흐름 - 실패]
pyVoIP socket bind → 192.168.35.103:5080 (LAN)
REGISTER 전송 → VPN 인터페이스를 통해 172.31.79.202:5060 도달
Asterisk 응답 → VPN IP(172.31.90.x)로 반환
소켓 수신 불가 (LAN 소켓이 VPN 패킷 못 받음) → TimeoutError

[수정 후 흐름 - 성공]
pyVoIP socket bind → 172.31.90.x:5080 (VPN)
REGISTER 전송 → 172.31.79.202:5060 도달
Asterisk 응답 → 172.31.90.x로 반환, 소켓 수신 성공 → 등록 완료
```

Zoiper는 내부적으로 `0.0.0.0`에 바인딩하거나 올바른 인터페이스를 자동 감지하므로
VPN 환경에서도 정상 동작.

### 수정 내용

**`config.py` — `get_local_ip()` 수정**

SIP 서버(`172.31.79.202`) 방향으로 실제 사용되는 인터페이스 IP를 감지하도록 변경.

```python
# 수정 전
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.connect(("8.8.8.8", 80))          # 인터넷 방향 → LAN IP 반환
    return s.getsockname()[0]

# 수정 후
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.connect((SIP_SERVER, SIP_PORT))   # SIP 서버 방향 → VPN IP 반환
    return s.getsockname()[0]
```

`socket.connect()`는 UDP 소켓에서 실제 연결하지 않고 라우팅 테이블만 조회하므로
부작용 없이 인터페이스 IP를 알 수 있음.

### 등록 성공 확인 (Asterisk 서버 로그)

```
From: "2001" <sip:2001@172.31.79.202>;tag=c5cb428c
To: "2001" <sip:2001@172.31.79.202>
CSeq: 2 REGISTER
Contact: <sip:2001@172.31.90.1:5080;transport=UDP>;expires=119
Expires: 120
Server: Asterisk PBX GIT-master-1b932f188
```

Contact에 VPN IP(`172.31.90.1:5080`)가 올바르게 등록됨.

### 추가 변경: 로컬 SIP 포트 설정 (.env)

```ini
# .env
SIP_LOCAL_PORT=5080   # 기본값 5060 대신 5080 사용 권장
                      # NAT/VPN 환경에서 5060 사용 시 충돌 가능성
```

`VoIPPhone(sipPort=cfg.SIP_LOCAL_PORT, ...)` 으로 `main.py`에 적용됨.

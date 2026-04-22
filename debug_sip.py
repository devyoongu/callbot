"""
debug_sip.py — Asterisk SIP 등록 응답 진단 스크립트

pyVoIP 없이 직접 UDP 소켓으로 SIP REGISTER를 보내고
서버 응답을 출력합니다.

사용:
    python debug_sip.py
"""
import socket
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
import config as cfg

LOCAL_IP  = cfg.get_local_ip()
LOCAL_PORT = 15060  # 임의 포트 사용
SERVER     = cfg.SIP_SERVER
PORT       = cfg.SIP_PORT
USERNAME   = cfg.SIP_USERNAME
TIMEOUT    = 5  # 초


def build_register():
    branch = "z9hG4bKdiag001"
    call_id = "debug-callid-001@" + LOCAL_IP
    tag = "debugtag001"

    lines = [
        f"REGISTER sip:{SERVER} SIP/2.0",
        f"Via: SIP/2.0/UDP {LOCAL_IP}:{LOCAL_PORT};branch={branch};rport",
        f'From: "{USERNAME}" <sip:{USERNAME}@{SERVER}>;tag={tag}',
        f'To: "{USERNAME}" <sip:{USERNAME}@{SERVER}>',
        f"Call-ID: {call_id}",
        "CSeq: 1 REGISTER",
        f"Contact: <sip:{USERNAME}@{LOCAL_IP}:{LOCAL_PORT};transport=UDP>",
        "Max-Forwards: 70",
        "Expires: 60",
        "Content-Length: 0",
        "",
        "",
    ]
    return "\r\n".join(lines).encode()


def main():
    print(f"[debug_sip] Local  : {LOCAL_IP}:{LOCAL_PORT}")
    print(f"[debug_sip] Server : {SERVER}:{PORT}")
    print(f"[debug_sip] User   : {USERNAME}")
    print()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LOCAL_IP, LOCAL_PORT))
    sock.settimeout(TIMEOUT)

    packet = build_register()
    print("── SENDING REGISTER ─────────────────────────────────")
    print(packet.decode())

    try:
        sock.sendto(packet, (SERVER, PORT))
        print(f"── WAITING FOR RESPONSE (timeout={TIMEOUT}s) ────────")
        data, addr = sock.recvfrom(4096)
        print(f"← Response from {addr}:")
        print(data.decode(errors="replace"))
    except socket.timeout:
        print("✗ No response received within", TIMEOUT, "seconds")
        print()
        print("원인 후보:")
        print("  1. Asterisk pjsip.conf에 2001 endpoint가 없음")
        print("  2. 방화벽이 응답을 차단 중")
        print()
        print("Asterisk 서버에서 확인:")
        print("  sudo asterisk -rx 'pjsip show endpoints'")
        print("  sudo asterisk -rx 'pjsip show registrations'")
    finally:
        sock.close()


if __name__ == "__main__":
    main()

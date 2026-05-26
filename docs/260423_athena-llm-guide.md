# Athena LLM 연동 가이드

다른 Python 콜봇 프로젝트에서 두뇌 역할을 하는 **Athena(LLM)** 를 연동하기 위한 가이드입니다.  
세션 시작(`start`) · 종료(`end`) · 질의(`query`) 세 가지 인터페이스를 중심으로 설명합니다.

---

## 목차

1. [개요](#1-개요)
2. [사전 준비](#2-사전-준비)
3. [인터페이스 설계](#3-인터페이스-설계)
4. [start — 세션 시작](#4-start--세션-시작)
5. [query — 질의 스트리밍](#5-query--질의-스트리밍)
6. [end — 세션 종료](#6-end--세션-종료)
7. [SSE 응답 구조](#7-sse-응답-구조)
8. [완전한 독립 예제 코드](#8-완전한-독립-예제-코드)
9. [설정 레퍼런스](#9-설정-레퍼런스)

---

## 1. 개요

### Athena 역할

```
[사용자 발화] → STT → Athena(LLM) → TTS → [음성 응답]
                           ↑
                      두뇌 역할
                  - 시나리오 기반 응답 생성
                  - 통화 제어 명령 (end_call / transfer_call)
                  - SSE 스트리밍으로 실시간 토큰 전달
```

### 통화 흐름과 인터페이스 매핑

```
통화 시작 ─→ start(#start) ─→ chat_threads_id 발급
    │
    └─→ query(발화 텍스트) ─→ SSE 스트리밍 응답
           │
           ├── type: "reply"        → TTS로 재생
           ├── type: "meta_status"  → 진행 상태 표시
           └── type: "command"      → end_call / transfer_call
    │
통화 종료 ─→ end(#end) ─→ 세션 정리
```

---

## 2. 사전 준비

### 패키지 설치

```bash
pip install aiohttp requests
```

### 필요 설정값

| 설정 키 | 설명 | 예시 |
|---------|------|------|
| `ATHENA_SITE` | Athena API 베이스 URL | `https://athena.example.com` |
| `ATHENA_AUTH` | Bearer 인증 토큰 | `eyJhbGci...` |
| `ATHENA_USER_ID` | API 사용자 ID | `42` |
| `ATHENA_CHAT_ROOMS_ID` | 채팅 룸 ID (서비스 ID) | `1001` |
| `ATHENA_SCENARIOS_ID` | 시나리오 ID | `55` |

---

## 3. 인터페이스 설계

세 가지 핵심 인터페이스:

| 인터페이스 | 메서드 | 방식 | 역할 |
|-----------|--------|------|------|
| **start** | `start(uui, voc_types)` | 동기 (SSE 내부 처리) | 세션 초기화, `chat_threads_id` 발급 |
| **query** | `query(text, dialog_count)` | 비동기 generator | 발화 텍스트 전달, 응답 스트리밍 수신 |
| **end** | `end(uui)` | 동기 (SSE 내부 처리) | 세션 종료, 통화 이력 저장 |

> `chat_threads_id` : 대화 연속성(컨텍스트)을 위한 스레드 식별자.  
> `start` 호출 시 발급되며, 이후 `query`·`end` 호출에 자동으로 포함됩니다.

---

## 4. start — 세션 시작

### 역할
- 통화 시작 시 Athena에 `#start` 신호를 보내 세션을 초기화합니다.
- 응답으로 받은 `chat_threads_id`를 내부에 저장하여 이후 `query`에서 컨텍스트를 유지합니다.
- `get_init_info()`로 서비스 설정(인사말 등)을 미리 받아올 수 있습니다.

### 요청 스펙

```
POST {ATHENA_SITE}/chats/m1/queries
Content-Type: application/json
Authorization: Bearer {ATHENA_AUTH}
Accept: text/event-stream
```

```json
{
  "app_type": "browser",
  "device_type": "pc",
  "users_id": 42,
  "chat_rooms_id": 1001,
  "scenarios_id": 55,
  "queries": {
    "type": "text",
    "text": "#start"
  },
  "agent_info": {
    "uui": "550e8400-e29b-41d4-a716-446655440000",
    "voc_types": ["inquiry", "complaint"]
  }
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `agent_info.uui` | string | 통화 고유 식별자 (UUID). 없으면 자동 생성 |
| `agent_info.voc_types` | list[str] | 통화 유형 분류 태그 (VOC 분류 등) |

### 응답에서 추출할 값

```json
{
  "data": {
    "documents": [
      {
        "chat_threads_id": 9901
      }
    ]
  }
}
```

### 코드

```python
import uuid
import json
import requests


def start(self, uui: str = None, voc_types: list = None) -> int | None:
    """
    Athena 세션 시작

    Args:
        uui: 통화 고유 식별자 (UUID 문자열). None이면 자동 생성
        voc_types: 통화 유형 태그 리스트 (예: ["inquiry"])

    Returns:
        chat_threads_id (int) — 성공 시
        None                  — 실패 시
    """
    if not uui:
        uui = str(uuid.uuid4())

    url = self.api_url + "/chats/m1/queries"
    headers = {
        "Authorization": f"Bearer {self.auth_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    body = {
        "app_type": "browser",
        "device_type": "pc",
        "users_id": int(self.users_id),
        "chat_rooms_id": self.chat_rooms_id,
        "scenarios_id": self.scenarios_id,
        "queries": {"type": "text", "text": "#start"},
        "agent_info": {
            "uui": uui,
            "voc_types": voc_types or [],
        },
    }

    response = requests.post(url, headers=headers, json=body, stream=True)

    for line in response.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8")
        data_str = line_str[6:] if line_str.startswith("data: ") else line_str
        try:
            data = json.loads(data_str)
            docs = data.get("data", {}).get("documents", [])
            if docs:
                chat_threads_id = docs[0].get("chat_threads_id")
                if chat_threads_id:
                    self.chat_threads_id = chat_threads_id
                    print(f"[Athena] Session started: chat_threads_id={chat_threads_id}")
                    return chat_threads_id
        except Exception as e:
            print(f"[Athena] start parse error: {e} | line: {line_str}")

    print("[Athena] start failed: chat_threads_id not received")
    return None
```

### 서비스 초기 정보 조회 (선택)

통화 시작 전 서비스 설정(인사말 메시지 등)을 미리 조회할 수 있습니다.

```python
def get_init_info(self) -> dict:
    """
    서비스 초기 정보 조회 (인사말 메시지 등)

    Returns:
        {
            "svc_id": int,
            "greeting_message": str,
            "scenarios_sid": str,   # 성공 시만 포함
            "svc_id_error": bool,   # 실패 시만 포함
        }
    """
    url = f"{self.api_url}/chats/m1/rooms/list?filter=servicesId:{self.chat_rooms_id}"
    headers = {
        "Authorization": f"Bearer {self.auth_token}",
        "x-user-id": str(self.users_id),
    }
    ret = {"svc_id": self.chat_rooms_id, "greeting_message": ""}

    try:
        response = requests.get(url, headers=headers)
        docs = response.json().get("data", {}).get("documents", [])

        for doc in docs:
            if doc.get("services_id") == self.chat_rooms_id:
                ret["scenarios_sid"]    = doc.get("scenarios_sid", "")
                ret["greeting_message"] = doc.get("greeting_message", "")[:200]
                break
    except Exception as e:
        ret["svc_id_error"] = True
        ret["greeting_message"] = f"서비스 정보 조회 오류: {e}"

    return ret
```

---

## 5. query — 질의 스트리밍

### 역할
- 사용자 발화 텍스트를 Athena에 전달하고, SSE 스트리밍으로 응답을 실시간 수신합니다.
- **async generator** 방식으로 동작하며, 응답 토큰이 도착할 때마다 `yield`합니다.
- `chat_threads_id`가 있으면 자동으로 이전 대화 컨텍스트가 유지됩니다.

### 요청 스펙

```
POST {ATHENA_SITE}/chats/m1/queries
Content-Type: application/json
Authorization: Bearer {ATHENA_AUTH}
Accept: text/event-stream
```

```json
{
  "app_type": "browser",
  "device_type": "pc",
  "users_id": 42,
  "chat_rooms_id": 1001,
  "scenarios_id": 55,
  "chat_threads_id": 9901,
  "sse_status_enabled": true,
  "queries": {
    "type": "text",
    "text": "요금제 변경하고 싶어요"
  }
}
```

> `chat_threads_id`: `start` 이후 발급된 값. 처음 응답에도 포함될 수 있으므로 자동 갱신됩니다.  
> `sse_status_enabled`: `true`로 설정하면 `meta_status` 이벤트를 수신합니다.

### yield 이벤트 타입

| `type` | 포함 필드 | 설명 |
|--------|-----------|------|
| `"reply"` | `text` | TTS로 재생할 응답 텍스트 |
| `"meta_status"` | `text` | 진행 상태 (예: `"in_progress"`, `"completed"`) |
| `"command"` | `text`, `command`, `dest_number` | 통화 제어 명령 |

#### command 값

| `command` | 설명 | `dest_number` |
|-----------|------|---------------|
| `"end_call"` | 통화 종료 요청 | `"00000"` (고정) |
| `"transfer_call"` | 상담원 연결 요청 | 연결할 전화번호 |

### 코드

```python
import aiohttp
import json


async def query(self, text: str, dialog_count: int = 0):
    """
    Athena에 발화 텍스트 전달 및 SSE 스트리밍 응답 수신

    Args:
        text: 사용자 발화 텍스트 (STT 결과)
        dialog_count: 현재 대화 턴 번호 (로깅용)

    Yields:
        dict: 이벤트 객체
            {"type": "reply",       "text": "응답 텍스트"}
            {"type": "meta_status", "text": "in_progress"}
            {"type": "command",     "text": "...", "command": "end_call",      "dest_number": "00000"}
            {"type": "command",     "text": "...", "command": "transfer_call", "dest_number": "010XXXXXXXX"}
    """
    headers = {
        "Authorization": f"Bearer {self.auth_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    body = {
        "app_type": "browser",
        "device_type": "pc",
        "users_id": int(self.users_id),
        "chat_rooms_id": self.chat_rooms_id,
        "scenarios_id": self.scenarios_id,
        "sse_status_enabled": True,
        "queries": {"type": "text", "text": text},
    }
    if self.chat_threads_id:
        body["chat_threads_id"] = self.chat_threads_id

    async with aiohttp.ClientSession() as session:
        async with session.post(
            self.api_url + "/chats/m1/queries", headers=headers, json=body
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"Athena API error {resp.status}: {error_text}")

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue

                try:
                    data = json.loads(line[6:])
                    docs = data.get("data", {}).get("documents", [])

                    if not docs:
                        # 빈 documents → 검색 결과 없음
                        yield {"type": "reply", "text": "검색된 결과가 없습니다. 다른 질문 사항은 없으신가요?"}
                        break

                    doc = docs[0]

                    # chat_threads_id 자동 갱신
                    if not self.chat_threads_id and "chat_threads_id" in doc:
                        self.chat_threads_id = doc["chat_threads_id"]

                    reply_text = doc.get("replies", {}).get("text", "")

                    # 통화 제어 명령 처리
                    if "agent_controls" in doc:
                        ctrl = doc["agent_controls"][0]
                        command = ctrl.get("command", "")
                        dest_number = ctrl.get("parameter", "00000") if command == "transfer_call" else "00000"
                        yield {
                            "type": "command",
                            "text": reply_text,
                            "command": command,
                            "dest_number": dest_number,
                        }

                    # 일반 응답 텍스트
                    elif reply_text:
                        yield {"type": "reply", "text": reply_text}

                    # 메타 상태 이벤트
                    meta_status = doc.get("meta", {}).get("status", "")
                    if meta_status:
                        yield {"type": "meta_status", "text": meta_status}

                    # 스트림 종료
                    if doc.get("is_sse_finished"):
                        break

                except Exception as e:
                    print(f"[Athena] query parse error: {e} | line: {line}")
```

### 사용 예시

```python
dialog_count = 0

async for event in athena.query("요금제를 변경하고 싶어요", dialog_count):
    event_type = event["type"]

    if event_type == "reply":
        tts_text = event["text"]
        # → TTS 재생
        print(f"[TTS] {tts_text}")

    elif event_type == "meta_status":
        status = event["text"]
        # → 진행 상태 처리 (선택)
        print(f"[Status] {status}")

    elif event_type == "command":
        command = event["command"]

        if command == "end_call":
            # → 통화 종료 처리
            print("[Command] 통화를 종료합니다.")

        elif command == "transfer_call":
            dest = event["dest_number"]
            # → 상담원 연결 처리
            print(f"[Command] 상담원 연결: {dest}")

dialog_count += 1
```

---

## 6. end — 세션 종료

### 역할
- 통화 종료 시 Athena에 `#end` 신호를 보내 세션을 닫습니다.
- 내부적으로 통화 이력 저장 등 정리 작업이 수행됩니다.
- `is_sse_finished: true`인 최종 응답을 반환합니다.

### 요청 스펙

`#start`와 동일한 엔드포인트/헤더를 사용하며, 본문만 다릅니다.

```json
{
  "app_type": "browser",
  "device_type": "pc",
  "users_id": 42,
  "chat_rooms_id": 1001,
  "scenarios_id": 55,
  "queries": {
    "type": "text",
    "text": "#end"
  },
  "agent_info": {
    "uui": "550e8400-e29b-41d4-a716-446655440000",
    "voc_types": []
  }
}
```

> `#end` 요청 시 `voc_types`는 빈 배열로 고정합니다.

### 응답에서 확인할 값

```json
{
  "data": {
    "documents": [
      {
        "is_sse_finished": true
      }
    ]
  }
}
```

### 코드

```python
def end(self, uui: str = None) -> dict | None:
    """
    Athena 세션 종료

    Args:
        uui: 통화 고유 식별자 (start()에 사용한 것과 동일한 값)

    Returns:
        최종 SSE 응답 dict (is_sse_finished=True 인 응답)
        None — 응답 없을 때
    """
    if not uui:
        uui = str(uuid.uuid4())

    url = self.api_url + "/chats/m1/queries"
    headers = {
        "Authorization": f"Bearer {self.auth_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    body = {
        "app_type": "browser",
        "device_type": "pc",
        "users_id": int(self.users_id),
        "chat_rooms_id": self.chat_rooms_id,
        "scenarios_id": self.scenarios_id,
        "queries": {"type": "text", "text": "#end"},
        "agent_info": {
            "uui": uui,
            "voc_types": [],   # #end 시 항상 빈 배열
        },
    }

    response = requests.post(url, headers=headers, json=body, stream=True)

    for line in response.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8")
        data_str = line_str[6:] if line_str.startswith("data: ") else line_str
        try:
            data = json.loads(data_str)
            docs = data.get("data", {}).get("documents", [])
            if docs and docs[0].get("is_sse_finished", False):
                print(f"[Athena] Session ended successfully")
                self.chat_threads_id = None   # 세션 초기화
                return data
        except Exception as e:
            print(f"[Athena] end parse error: {e} | line: {line_str}")

    print("[Athena] end: no final response received")
    return None
```

---

## 7. SSE 응답 구조

Athena는 모든 API 응답을 **Server-Sent Events (SSE)** 형식으로 반환합니다.

### SSE 라인 형식

```
data: {"data": {"documents": [...]}}
data: {"data": {"documents": [...]}}
...
```

### documents[0] 필드 설명

| 필드 | 타입 | 설명 |
|------|------|------|
| `chat_threads_id` | int | 대화 스레드 ID (컨텍스트 유지용) |
| `replies.text` | str | 응답 텍스트 (TTS 재생 대상) |
| `meta.status` | str | 진행 상태 (`"in_progress"`, `"completed"` 등) |
| `agent_controls` | list | 통화 제어 명령 배열 |
| `agent_controls[0].command` | str | `"end_call"` 또는 `"transfer_call"` |
| `agent_controls[0].parameter` | str | 전달 번호 (`transfer_call` 시) |
| `is_sse_finished` | bool | `true`이면 스트림 종료 |
| `passages` | list | 검색 문서 (로깅 제외 대상 — 크기가 큼) |

### SSE 파싱 패턴

```python
for raw_line in response.iter_lines():
    line = raw_line.decode("utf-8")
    if not line.startswith("data: "):
        continue                         # 빈 줄 또는 comment 스킵
    
    data = json.loads(line[6:])          # "data: " 이후만 파싱
    docs = data.get("data", {}).get("documents", [])
    
    if not docs:
        break                            # 빈 배열 → 결과 없음

    doc = docs[0]
    
    # passages 포함 응답은 로그 제외 (크기 과대)
    if "passages" in doc:
        pass
    
    if doc.get("is_sse_finished"):
        break                            # 스트림 종료
```

---

## 8. 완전한 독립 예제 코드

아래 코드는 현재 콜봇의 내부 의존성 없이 Athena를 단독으로 사용하는 완전한 예제입니다.

```python
#!/usr/bin/env python3
"""
Athena LLM 독립 사용 예제
start → query(반복) → end 흐름 시연
"""

import uuid
import json
import asyncio
import requests
import aiohttp


class AthenaClient:
    """
    Athena LLM 클라이언트

    사용법:
        client = AthenaClient(
            api_url="https://athena.example.com",
            auth_token="Bearer-token",
            users_id=42,
            chat_rooms_id=1001,
            scenarios_id=55,
        )

        uui = str(uuid.uuid4())

        # 1. 세션 시작
        thread_id = client.start(uui=uui, voc_types=["inquiry"])

        # 2. 질의 (여러 번 반복 가능)
        async for event in client.query("요금제 변경하고 싶어요"):
            if event["type"] == "reply":
                print("응답:", event["text"])
            elif event["type"] == "command":
                print("명령:", event["command"], event["dest_number"])

        # 3. 세션 종료
        client.end(uui=uui)
    """

    def __init__(
        self,
        api_url: str,
        auth_token: str,
        users_id: int,
        chat_rooms_id: int,
        scenarios_id: int,
    ):
        self.api_url = api_url.rstrip("/")
        self.auth_token = auth_token
        self.users_id = users_id
        self.chat_rooms_id = chat_rooms_id
        self.scenarios_id = scenarios_id
        self.chat_threads_id = None  # start() 이후 자동 설정

    # ------------------------------------------------------------------ #
    #  공통 헤더
    # ------------------------------------------------------------------ #

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

    def _base_body(self) -> dict:
        return {
            "app_type": "browser",
            "device_type": "pc",
            "users_id": int(self.users_id),
            "chat_rooms_id": self.chat_rooms_id,
            "scenarios_id": self.scenarios_id,
        }

    # ------------------------------------------------------------------ #
    #  start
    # ------------------------------------------------------------------ #

    def start(self, uui: str = None, voc_types: list = None) -> int | None:
        """
        세션 시작 (#start)

        Returns:
            chat_threads_id (int) — 성공
            None                  — 실패
        """
        if not uui:
            uui = str(uuid.uuid4())

        body = self._base_body()
        body.update({
            "queries": {"type": "text", "text": "#start"},
            "agent_info": {"uui": uui, "voc_types": voc_types or []},
        })

        resp = requests.post(
            self.api_url + "/chats/m1/queries",
            headers=self._headers(),
            json=body,
            stream=True,
        )

        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8")
            data_str = line[6:] if line.startswith("data: ") else line
            try:
                data = json.loads(data_str)
                docs = data.get("data", {}).get("documents", [])
                if docs:
                    chat_threads_id = docs[0].get("chat_threads_id")
                    if chat_threads_id:
                        self.chat_threads_id = chat_threads_id
                        print(f"[Athena] start OK — chat_threads_id={chat_threads_id}")
                        return chat_threads_id
            except Exception as e:
                print(f"[Athena] start parse error: {e}")

        print("[Athena] start FAILED — no chat_threads_id")
        return None

    # ------------------------------------------------------------------ #
    #  query
    # ------------------------------------------------------------------ #

    async def query(self, text: str, dialog_count: int = 0):
        """
        발화 텍스트 전달 및 SSE 스트리밍 응답 수신 (async generator)

        Yields:
            {"type": "reply",       "text": str}
            {"type": "meta_status", "text": str}
            {"type": "command",     "text": str, "command": str, "dest_number": str}
        """
        body = self._base_body()
        body.update({
            "sse_status_enabled": True,
            "queries": {"type": "text", "text": text},
        })
        if self.chat_threads_id:
            body["chat_threads_id"] = self.chat_threads_id

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url + "/chats/m1/queries",
                headers=self._headers(),
                json=body,
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"Athena API {resp.status}: {await resp.text()}")

                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue

                    try:
                        data = json.loads(line[6:])
                        docs = data.get("data", {}).get("documents", [])

                        if not docs:
                            yield {"type": "reply", "text": "검색된 결과가 없습니다. 다른 질문 사항은 없으신가요?"}
                            break

                        doc = docs[0]

                        # chat_threads_id 자동 갱신
                        if not self.chat_threads_id and "chat_threads_id" in doc:
                            self.chat_threads_id = doc["chat_threads_id"]

                        reply_text = doc.get("replies", {}).get("text", "")

                        # 통화 제어 명령
                        if "agent_controls" in doc:
                            ctrl = doc["agent_controls"][0]
                            command = ctrl.get("command", "")
                            dest = ctrl.get("parameter", "00000") if command == "transfer_call" else "00000"
                            yield {"type": "command", "text": reply_text, "command": command, "dest_number": dest}

                        # 일반 응답
                        elif reply_text:
                            yield {"type": "reply", "text": reply_text}

                        # 메타 상태
                        meta_status = doc.get("meta", {}).get("status", "")
                        if meta_status:
                            yield {"type": "meta_status", "text": meta_status}

                        if doc.get("is_sse_finished"):
                            break

                    except Exception as e:
                        print(f"[Athena] query parse error: {e} | line: {line}")

    # ------------------------------------------------------------------ #
    #  end
    # ------------------------------------------------------------------ #

    def end(self, uui: str = None) -> dict | None:
        """
        세션 종료 (#end)

        Returns:
            최종 SSE 응답 dict — 성공
            None               — 응답 없음
        """
        if not uui:
            uui = str(uuid.uuid4())

        body = self._base_body()
        body.update({
            "queries": {"type": "text", "text": "#end"},
            "agent_info": {"uui": uui, "voc_types": []},
        })

        resp = requests.post(
            self.api_url + "/chats/m1/queries",
            headers=self._headers(),
            json=body,
            stream=True,
        )

        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8")
            data_str = line[6:] if line.startswith("data: ") else line
            try:
                data = json.loads(data_str)
                docs = data.get("data", {}).get("documents", [])
                if docs and docs[0].get("is_sse_finished", False):
                    self.chat_threads_id = None   # 세션 초기화
                    print("[Athena] end OK")
                    return data
            except Exception as e:
                print(f"[Athena] end parse error: {e}")

        print("[Athena] end: no final response")
        return None

    # ------------------------------------------------------------------ #
    #  get_init_info (선택)
    # ------------------------------------------------------------------ #

    def get_init_info(self) -> dict:
        """
        서비스 초기 정보 조회 (인사말 등)

        Returns:
            {"svc_id": int, "greeting_message": str, "scenarios_sid": str}
        """
        url = f"{self.api_url}/chats/m1/rooms/list?filter=servicesId:{self.chat_rooms_id}"
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "x-user-id": str(self.users_id),
        }
        ret = {"svc_id": self.chat_rooms_id, "greeting_message": ""}
        try:
            resp = requests.get(url, headers=headers)
            docs = resp.json().get("data", {}).get("documents", [])
            for doc in docs:
                if doc.get("services_id") == self.chat_rooms_id:
                    ret["scenarios_sid"]    = doc.get("scenarios_sid", "")
                    ret["greeting_message"] = doc.get("greeting_message", "")[:200]
                    break
        except Exception as e:
            ret["svc_id_error"] = True
            ret["greeting_message"] = f"서비스 정보 조회 오류: {e}"
        return ret


# ──────────────────────────────────────────────────────────────────────── #
#  실행 예제
# ──────────────────────────────────────────────────────────────────────── #

async def main():
    client = AthenaClient(
        api_url       = "https://athena.example.com",
        auth_token    = "your-bearer-token",
        users_id      = 42,
        chat_rooms_id = 1001,
        scenarios_id  = 55,
    )

    uui = str(uuid.uuid4())

    # 0. 서비스 초기 정보 조회 (선택)
    info = client.get_init_info()
    print(f"인사말: {info.get('greeting_message')}")

    # 1. 세션 시작
    thread_id = client.start(uui=uui, voc_types=["inquiry"])
    if not thread_id:
        print("세션 시작 실패")
        return

    # 2. 질의 루프
    questions = [
        "요금제를 변경하고 싶어요",
        "LTE 요금제 중 가장 저렴한 게 뭐예요?",
    ]

    for turn, question in enumerate(questions):
        print(f"\n[Turn {turn + 1}] 질문: {question}")

        async for event in client.query(question, dialog_count=turn):
            etype = event["type"]

            if etype == "reply":
                print(f"  [응답] {event['text']}")
                # → tts.synthesize(event["text"]) 호출

            elif etype == "meta_status":
                print(f"  [상태] {event['text']}")

            elif etype == "command":
                cmd = event["command"]
                print(f"  [명령] {cmd} → dest: {event['dest_number']}")

                if cmd == "end_call":
                    print("  → 통화 종료")
                    break
                elif cmd == "transfer_call":
                    print(f"  → 상담원 연결: {event['dest_number']}")
                    break

    # 3. 세션 종료
    client.end(uui=uui)
    print("\n[Athena] 세션 종료 완료")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 9. 설정 레퍼런스

### 환경변수로 설정하는 경우

```python
import os

client = AthenaClient(
    api_url       = os.environ["ATHENA_SITE"],
    auth_token    = os.environ["ATHENA_AUTH"],
    users_id      = int(os.environ["ATHENA_USER_ID"]),
    chat_rooms_id = int(os.environ["ATHENA_CHAT_ROOMS_ID"]),
    scenarios_id  = int(os.environ["ATHENA_SCENARIOS_ID"]),
)
```

### `.env` 파일 예시

```dotenv
ATHENA_SITE=https://athena.example.com
ATHENA_AUTH=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
ATHENA_USER_ID=42
ATHENA_CHAT_ROOMS_ID=1001
ATHENA_SCENARIOS_ID=55
```

### 주요 동작 정리

| 상황 | 동작 |
|------|------|
| `start` 응답에 `chat_threads_id` 없음 | 세션 초기화 실패로 간주, `None` 반환 |
| `query` 응답 `docs` 빈 배열 | `"검색된 결과가 없습니다..."` yield 후 종료 |
| `query` 응답 `agent_controls` 포함 | `command` 이벤트 yield, `reply` 이벤트는 skip |
| `is_sse_finished: true` | 스트림 즉시 종료 |
| `passages` 필드 | 크기가 크므로 로깅 제외 권장 |
| `end` 호출 후 | `chat_threads_id = None` 으로 초기화 |

"""
callbot/athena.py — Athena LLM 클라이언트

260423_athena-llm-guide.md 의 AthenaClient 클래스 기반.
query_sync(): async query()를 동기 컨텍스트(AGI/callbot 스레드)에서 사용하기 위한 래퍼.
"""
import uuid
import json
import asyncio
import requests
import aiohttp
from typing import Optional


class AthenaClient:
    """
    Athena LLM 클라이언트

    사용법:
        client = AthenaClient(api_url=..., auth_token=..., ...)
        uui = str(uuid.uuid4())
        thread_id = client.start(uui=uui, voc_types=["inquiry"])

        # 동기 방식 (callbot 스레드에서 사용)
        events = client.query_sync("요금제 변경하고 싶어요")
        for event in events:
            if event["type"] == "reply":
                print(event["text"])

        client.end(uui=uui)
    """

    def __init__(
        self,
        api_url: str,
        auth_token: str,
        users_id: int,
        chat_rooms_id: int,
        scenarios_id,  # int 또는 str (e.g. "robi-gpt-dev:workflow_xxx")
    ):
        self.api_url       = api_url.rstrip("/")
        self.auth_token    = auth_token
        self.users_id      = users_id
        self.chat_rooms_id = chat_rooms_id
        self.scenarios_id  = scenarios_id
        self.chat_threads_id: Optional[int] = None

    # ── 공통 헬퍼 ─────────────────────────────────────────────────────

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type":  "application/json",
            "Accept":        "text/event-stream",
        }

    def _base_body(self) -> dict:
        return {
            "app_type":     "browser",
            "device_type":  "pc",
            "users_id":     int(self.users_id),
            "chat_rooms_id": self.chat_rooms_id,
            "scenarios_id": self.scenarios_id,
        }

    # ── start ──────────────────────────────────────────────────────────

    def start(self, uui: str = None, voc_types: list = None) -> Optional[int]:
        """
        Athena 세션 시작 (#start)

        Returns:
            chat_threads_id (int) — 성공
            None                  — 실패
        """
        if not uui:
            uui = str(uuid.uuid4())

        body = self._base_body()
        body.update({
            "queries":    {"type": "text", "text": "#start"},
            "agent_info": {"uui": uui, "voc_types": voc_types or []},
        })

        resp = requests.post(
            self.api_url + "/chats/m1/queries",
            headers=self._headers(),
            json=body,
            stream=True,
            timeout=30,
        )

        for raw in resp.iter_lines():
            if not raw:
                continue
            line     = raw.decode("utf-8")
            data_str = line[6:] if line.startswith("data: ") else line
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
                print(f"[Athena] start parse error: {e} | line: {line}")

        print("[Athena] start FAILED — no chat_threads_id")
        return None

    # ── query (async) ──────────────────────────────────────────────────

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
                timeout=aiohttp.ClientTimeout(total=60),
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
                            ctrl       = doc["agent_controls"][0]
                            command    = ctrl.get("command", "")
                            dest       = ctrl.get("parameter", "00000") if command == "transfer_call" else "00000"
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

    def query_sync(self, text: str, dialog_count: int = 0, on_event=None) -> list:
        """
        query()의 동기 래퍼 — callbot 스레드에서 사용

        asyncio.run()으로 이벤트 루프를 생성하고 전체 응답을 수집합니다.
        (각 통화는 독립 프로세스/스레드이므로 기존 이벤트 루프 없음)

        Args:
            text:         사용자 발화
            dialog_count: 대화 턴 인덱스 (로깅용)
            on_event:     이벤트 도착 시 호출되는 콜백 (event_dict) -> None.
                          메타 상태(meta_status)를 stream 즉시 TTS 재생 등에 사용.
                          콜백이 블로킹하면 SSE 소비도 그만큼 지연됨.

        Returns:
            list of event dicts
        """
        async def _collect():
            events = []
            async for event in self.query(text, dialog_count):
                events.append(event)
                print(f"[Athena] Event: {event['type']} — {event.get('text', '')[:50]}")
                if on_event is not None:
                    try:
                        on_event(event)
                    except Exception as e:
                        print(f"[Athena] on_event callback error: {e}")
            return events

        return asyncio.run(_collect())

    # ── end ────────────────────────────────────────────────────────────

    def end(self, uui: str = None) -> Optional[dict]:
        """
        Athena 세션 종료 (#end)

        Returns:
            최종 SSE 응답 dict — 성공
            None               — 응답 없음
        """
        if not uui:
            uui = str(uuid.uuid4())

        body = self._base_body()
        body.update({
            "queries":    {"type": "text", "text": "#end"},
            "agent_info": {"uui": uui, "voc_types": []},
        })

        try:
            resp = requests.post(
                self.api_url + "/chats/m1/queries",
                headers=self._headers(),
                json=body,
                stream=True,
                timeout=15,
            )
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line     = raw.decode("utf-8")
                data_str = line[6:] if line.startswith("data: ") else line
                try:
                    data = json.loads(data_str)
                    docs = data.get("data", {}).get("documents", [])
                    if docs and docs[0].get("is_sse_finished", False):
                        self.chat_threads_id = None
                        print("[Athena] Session ended")
                        return data
                except Exception as e:
                    print(f"[Athena] end parse error: {e}")
        except Exception as e:
            print(f"[Athena] end request error: {e}")

        return None

    # ── get_init_info ─────────────────────────────────────────────────

    def get_init_info(self) -> dict:
        """
        서비스 초기 정보 조회 (인사말 메시지 등)

        Returns:
            {"svc_id": int, "greeting_message": str}
        """
        url = f"{self.api_url}/chats/m1/rooms/list?filter=servicesId:{self.chat_rooms_id}"
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "x-user-id":     str(self.users_id),
        }
        ret = {"svc_id": self.chat_rooms_id, "greeting_message": ""}
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            docs = resp.json().get("data", {}).get("documents", [])
            for doc in docs:
                if doc.get("services_id") == self.chat_rooms_id:
                    ret["scenarios_sid"]    = doc.get("scenarios_sid", "")
                    ret["greeting_message"] = doc.get("greeting_message", "")[:200]
                    break
        except Exception as e:
            ret["svc_id_error"]    = True
            ret["greeting_message"] = ""
            print(f"[Athena] get_init_info error: {e}")
        return ret

"""
callbot/stt_qwen3.py — Qwen3-ASR WebSocket 스트리밍 STT 드라이버.

docs/260519_websocket_client_guide.md 의 프로토콜을 그대로 따른다:
  1. ws://host:port/api/ws 접속
  2. {"type":"config","language":"Korean", ...} 텍스트 송신
  3. {"type":"ready",...} 수신 후 16kHz float32 LE PCM 바이너리 스트림
  4. 클라이언트 VAD 로 silence 500ms 감지 시 {"type":"end"} 송신
  5. {"type":"result","is_final":true,...} 수신 → _final_transcript

GoogleSTTV2 와 동일한 인터페이스를 duck-type 으로 제공:
  initialize() / start_streaming(audio_source) / finalize() / stop()
  get_speech_started() / get_and_consume_eos() / get_final_transcript()
  _result_event / _transcript_ready / _final_transcript / _stt_error
  sample_rate / speech_started* / eos_* / final_time / last_speech_end_time
"""
import json
import threading
import time
from typing import Optional, Tuple

import numpy as np

try:
    from websockets.sync.client import connect
    from websockets.exceptions import ConnectionClosed
except ImportError as e:
    raise ImportError("pip install websockets>=12") from e


# 콜봇 audio_source 의 chunk: 16kHz LINEAR16 (=int16) 4000-byte (125ms) 단위.
# Qwen3 서버는 16kHz float32 LE 를 요구하므로 청크별로 int16→float32 변환 후 전송.
_CHUNK_MS              = 125
_EXPECTED_CHUNK_BYTES  = int(16000 * 0.125 * 2)   # 4000
_RMS_THRESHOLD         = 200                       # callbot 환경 측정값 (echo 192 < x < 발화 1886)
_SILENCE_FRAMES_EOS    = 4                         # 4×125ms = 500ms silence → EOS
_EOS_TRAIL_CHUNKS      = 2                         # EOS 후 250ms 더 송신 (자연 종료)
_PRE_SPEECH_RING       = 3                         # 375ms ring buffer (onset 보호)


class Qwen3WebSocketSTT:
    def __init__(
        self,
        url: str,
        language: str = "Korean",
        context: str = "",
        chunk_size_sec: float = 1.0,
        unfixed_chunk_num: int = 4,
        unfixed_token_num: int = 5,
        recv_timeout_sec: float = 20.0,
    ):
        self.url               = url
        self.language          = language
        self.context           = context
        self.chunk_size_sec    = chunk_size_sec
        self.unfixed_chunk_num = unfixed_chunk_num
        self.unfixed_token_num = unfixed_token_num
        self.recv_timeout_sec  = recv_timeout_sec
        # audio_source 가 self.sample_rate 를 읽고 16kHz 업샘플링 한다.
        self.sample_rate       = 16000

        # 호환용 attribute (GoogleSTTV2 와 동일 이름)
        self.client            = None
        self._result_event     = threading.Event()
        self._transcript_ready = threading.Event()
        self._final_transcript = ""
        self._stt_error: Optional[str]      = None
        self._stop             = False
        self._streaming_thread: Optional[threading.Thread] = None
        self._audio_source                  = None

        self.speech_started                  = False
        self.speech_started_time: Optional[float] = None
        self.eos_done                        = False
        self.eos_time: Optional[float]       = None
        self.last_speech_end_time: Optional[float] = None
        self.final_time: Optional[float]     = None

        self._speech_started_consumed        = False
        self._eos_consumed                   = False
        self._yielded_chunk_count            = 0
        self._last_interim_text              = ""
        self._session_id                     = ""

    # ── 공개 API (GoogleSTTV2 와 동일 시그니처) ──────────────────────────

    def initialize(self):
        # 연결은 start_streaming 의 background thread 에서 lazy 수행 (latency 분리).
        print(f"[STT] Qwen3 driver ready (url={self.url}, lang={self.language})")

    def start_streaming(self, audio_source):
        self._stop                    = False
        self._final_transcript        = ""
        self._stt_error               = None
        self._result_event.clear()
        self._transcript_ready.clear()
        self.speech_started           = False
        self.speech_started_time      = None
        self.eos_done                 = False
        self.eos_time                 = None
        self.last_speech_end_time     = None
        self.final_time               = None
        self._speech_started_consumed = False
        self._eos_consumed            = False
        self._yielded_chunk_count     = 0
        self._last_interim_text       = ""
        self._session_id              = ""

        self._audio_source = audio_source
        self._streaming_thread = threading.Thread(target=self._run, daemon=True)
        self._streaming_thread.start()

    def wait_for_result(self, timeout: float = 30.0) -> Tuple[bool, str]:
        done = self._result_event.wait(timeout=timeout)
        if not done:
            return False, "timeout"
        if self._stt_error:
            return False, self._stt_error
        return True, self._final_transcript

    def get_speech_started(self) -> Tuple[bool, Optional[float]]:
        if self.speech_started and not self._speech_started_consumed:
            self._speech_started_consumed = True
            return True, self.speech_started_time
        return False, None

    def get_and_consume_eos(self) -> Tuple[bool, Optional[float]]:
        if self.eos_done and not self._eos_consumed:
            self._eos_consumed = True
            return True, self.eos_time
        return False, None

    def get_final_transcript(self) -> str:
        if self._result_event.is_set() and not self._stt_error:
            return self._final_transcript
        return ""

    def stop(self):
        self._stop = True

    def finalize(self):
        self.stop()
        if self._streaming_thread and self._streaming_thread.is_alive():
            self._streaming_thread.join(timeout=3.0)
        self.client = None

    # ── 내부 ──────────────────────────────────────────────────────────

    def _run(self):
        ws = None
        try:
            ws = connect(self.url, max_size=None, open_timeout=10)
        except Exception as e:
            self._stt_error = f"ws connect failed: {e}"
            print(f"[STT] {self._stt_error}")
            self._result_event.set()
            self._transcript_ready.set()
            return

        recv_thread = None
        try:
            # 1. config 송신
            ws.send(json.dumps({
                "type":              "config",
                "context":           self.context,
                "language":          self.language,
                "chunk_size_sec":    self.chunk_size_sec,
                "unfixed_chunk_num": self.unfixed_chunk_num,
                "unfixed_token_num": self.unfixed_token_num,
            }, ensure_ascii=False))

            # 2. ready 대기
            try:
                first = ws.recv(timeout=10)
            except TimeoutError:
                self._stt_error = "ready timeout"
                return
            try:
                ready = json.loads(first)
            except Exception:
                self._stt_error = f"non-JSON first msg: {first!r}"
                return
            if ready.get("type") == "error":
                self._stt_error = f"{ready.get('code')}: {ready.get('message')}"
                return
            if ready.get("type") != "ready":
                self._stt_error = f"unexpected first msg: {ready}"
                return
            self._session_id = ready.get("session_id", "")
            print(f"[STT] Qwen3 session={self._session_id[:8]} sr={ready.get('sample_rate')}")

            # 3. 수신 스레드 시작
            recv_thread = threading.Thread(
                target=self._recv_loop, args=(ws,), daemon=True
            )
            recv_thread.start()

            # 4. 송신 루프 + VAD
            spoke_at_all = self._send_loop(ws)

            # 5. end 송신
            try:
                ws.send(json.dumps({"type": "end"}))
            except Exception as e:
                print(f"[STT] end send failed: {e}")

            # 6. 발화 없었으면 final 기다리지 않고 즉시 non_voice 처리
            if not spoke_at_all:
                self._final_transcript = "non_voice"
                self.final_time = time.time()
                self._transcript_ready.set()
                self._result_event.set()
                print("[STT] No speech detected → non_voice")
                return

            # 7. recv 가 is_final 받을 때까지 대기
            if recv_thread.is_alive():
                recv_thread.join(timeout=self.recv_timeout_sec)

            if not self._transcript_ready.is_set():
                # 시간 내 final 도착 안 함 — 마지막 interim 으로 fallback
                if self._last_interim_text:
                    self._final_transcript = self._last_interim_text.strip()
                    print(f"[STT] FINAL missed → using last interim: '{self._final_transcript}'")
                else:
                    self._final_transcript = "non_voice"
                    print("[STT] FINAL missed and no interim → non_voice")
                self.final_time = time.time()
                self._transcript_ready.set()

            self._result_event.set()
        except Exception as e:
            self._stt_error = f"streaming error: {e}"
            print(f"[STT] {self._stt_error}")
        finally:
            try:
                ws.close()
            except Exception:
                pass
            if not self._result_event.is_set():
                self._result_event.set()
            if not self._transcript_ready.is_set():
                self._transcript_ready.set()

    def _send_loop(self, ws) -> bool:
        """audio_source 청크를 받아 ws 로 송신. 클라이언트 VAD 로 EOS 결정.
        Returns: True if any speech was sent (used to decide non_voice 단축 경로)."""
        speech_detected   = False
        silence_frames    = 0
        high_energy_count = 0
        eos_buffered      = 0
        eos_committed     = False
        pre_speech_ring: "list[bytes]" = []
        speech_sent       = False

        for chunk in self._audio_source:
            if self._stop:
                break
            if not chunk:
                continue
            self._yielded_chunk_count += 1

            # EOS 후 trailing audio
            if eos_committed:
                self._send_pcm(ws, chunk)
                eos_buffered += 1
                if eos_buffered >= _EOS_TRAIL_CHUNKS:
                    break
                continue

            # speech 시작 전 ring buffer
            if not speech_detected:
                pre_speech_ring.append(chunk)
                if len(pre_speech_ring) > _PRE_SPEECH_RING:
                    pre_speech_ring.pop(0)

            if len(chunk) == _EXPECTED_CHUNK_BYTES:
                samples = np.frombuffer(chunk, dtype=np.int16)
                rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
                is_speech = rms >= _RMS_THRESHOLD

                if is_speech:
                    high_energy_count += 1
                    if high_energy_count >= 2 and not speech_detected:
                        speech_detected = True
                        silence_frames  = 0
                        # ring 의 onset chunks 먼저 송신
                        for c in pre_speech_ring:
                            self._send_pcm(ws, c)
                            speech_sent = True
                        pre_speech_ring.clear()
                        if not self.speech_started:
                            self.speech_started      = True
                            self.speech_started_time = time.time()
                            print(f"[STT] Speech started (rms={rms:.0f})")
                    if speech_detected:
                        silence_frames = 0
                else:
                    high_energy_count = 0
                    if speech_detected:
                        silence_frames += 1
                        if silence_frames >= _SILENCE_FRAMES_EOS:
                            print(
                                f"[STT] EOS triggered ({silence_frames}×125ms="
                                f"{silence_frames*125}ms silence)"
                            )
                            eos_committed     = True
                            self.eos_done     = True
                            self.eos_time     = time.time()
                            self.last_speech_end_time = self.eos_time

            if speech_detected and not eos_committed:
                self._send_pcm(ws, chunk)
                speech_sent = True

        return speech_sent

    def _send_pcm(self, ws, int16_bytes: bytes):
        arr = np.frombuffer(int16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        ws.send(arr.tobytes())

    def _recv_loop(self, ws):
        while not self._stop:
            try:
                msg = ws.recv(timeout=self.recv_timeout_sec)
            except TimeoutError:
                print("[STT] recv timeout")
                break
            except ConnectionClosed:
                break
            except Exception as e:
                print(f"[STT] recv error: {e}")
                break

            try:
                evt = json.loads(msg)
            except Exception:
                continue

            t = evt.get("type")
            if t == "result":
                text   = (evt.get("text") or "").strip()
                if evt.get("is_final"):
                    self._final_transcript = text if text else "non_voice"
                    self.final_time = time.time()
                    self._transcript_ready.set()
                    self._result_event.set()
                    eos_t = self.eos_time or self.last_speech_end_time
                    latency_str = ""
                    if eos_t is not None:
                        latency_str = f" (clientEOS→Final {(self.final_time-eos_t)*1000:.0f}ms)"
                    print(f"[STT] Final (qwen3): '{self._final_transcript}'{latency_str}")
                else:
                    self._last_interim_text = text
                    cid = evt.get("chunk_id", "?")
                    print(f"[STT] partial cid={cid}: {text[:60]}")
            elif t == "closed":
                reason = evt.get("reason", "")
                print(f"[STT] Server closed ({reason})")
                break
            elif t == "error":
                self._stt_error = f"{evt.get('code')}: {evt.get('message')}"
                print(f"[STT] Server error: {self._stt_error}")
                self._result_event.set()
                self._transcript_ready.set()
                break

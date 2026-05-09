"""
callbot/call_handler.py — 통화 상태 머신

pyVoIP callCallback에서 호출됩니다 (별도 스레드).
Athena 세션 관리, STT/TTS 루프, 통화 종료 처리를 담당합니다.

TTS 합성/재생은 TTSPipeline 클래스가 백그라운드 스레드에서 처리:
  - on_event 콜백/메인 핸들러는 enqueue() 만 호출 (non-blocking)
  - 합성 worker(executor)와 재생 worker(play_thread)가 병렬 동작
  - call 단위 lifecycle: handle_call 진입 시 1회 생성, finally에서 1회 shutdown
"""
import os
import queue
import threading
import time
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import config as cfg
from audio_source import CallAudioSource
from stt import GoogleSTTV2, create_stt
from tts import synthesize_pcm_8bit_unsigned
from athena import AthenaClient

logger = logging.getLogger("callbot")

# TTS 재생 청크 크기 — pyVoIP encode_pcmu(width=1) 입력 단위
# 160 samples × 1 byte = 20ms at 8kHz
_CHUNK_SIZE = 160
_SLEEP_SEC  = 0.018  # 20ms보다 약간 짧게 → 버퍼 유지
_SILENCE    = b"\x80" * _CHUNK_SIZE  # pyVoIP 무음 sentinel (8-bit unsigned 중심값)


def _send_priming_silence(call, num_chunks: int = 25):
    """
    RTP 스트림 프라이밍 — 무음 약 0.5초 전송으로 RTP 경로 개통.
    call 단위로 한 번만 호출 (TTSPipeline 이 _primed flag 로 관리).
    """
    from pyVoIP.VoIP import CallState
    for _ in range(num_chunks):
        if call.state != CallState.ANSWERED:
            return
        try:
            call.writeAudio(_SILENCE)
        except Exception:
            return
        time.sleep(_SLEEP_SEC)


def _send_pcm_8bit(call, pcm_8bit: bytes, label: str,
                   stop_event: Optional[threading.Event] = None) -> int:
    """
    8-bit unsigned PCM 청크를 pyVoIP writeAudio() 로 실시간 송출.
    매 청크마다 call.state 와 stop_event 확인 → 즉시 중단 가능.
    Returns: 송출한 chunk 개수.
    """
    from pyVoIP.VoIP import CallState
    sent = 0
    for i in range(0, len(pcm_8bit), _CHUNK_SIZE):
        if stop_event is not None and stop_event.is_set():
            break
        if call.state != CallState.ANSWERED:
            break

        chunk = pcm_8bit[i:i + _CHUNK_SIZE]
        if len(chunk) < _CHUNK_SIZE:
            chunk = chunk + b"\x80" * (_CHUNK_SIZE - len(chunk))

        try:
            call.writeAudio(chunk)
            sent += 1
        except Exception as e:
            logger.warning(f"[TTS] writeAudio error ({label[:30]!r}): {e}")
            break

        time.sleep(_SLEEP_SEC)
    return sent


class TTSPipeline:
    """
    한 통화 동안 TTS 합성과 재생을 파이프라이닝.

    - enqueue(text):  non-blocking. ThreadPoolExecutor에 합성 submit, FIFO 큐에 적재.
    - 백그라운드 play_thread: 큐에서 (text, future) pop → future.result() blocking → 재생.
    - call.state != ANSWERED 또는 stop_event 감지 시 즉시 종료.
    - wait_drained(): 큐 비고 재생 완료 대기 (turn 경계에서 호출 → STT 시작 전 echo 방지).
    - clear_pending(): 큐에 쌓인 미재생 항목 drop (Athena 예외 → fallback 재생 직전).
    - shutdown(): poison pill + executor shutdown + thread join.

    Lifecycle: call 단위 1회 생성/소멸. turn마다 재생성하지 않음.
    """

    def __init__(self, call, call_id: str):
        self.call         = call
        self.call_id      = call_id
        self.executor     = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix=f"tts-synth-{call_id[:8]}",
        )
        self.queue: "queue.Queue" = queue.Queue()  # of (text, future) tuples or None (poison)
        self.stop_event   = threading.Event()
        self._primed      = False
        self._idle_event  = threading.Event()
        self._idle_event.set()
        self._inflight    = 0
        self._inflight_lock = threading.Lock()
        self.play_thread  = threading.Thread(
            target=self._play_loop,
            name=f"tts-play-{call_id[:8]}",
            daemon=True,
        )
        self.play_thread.start()

    def enqueue(self, text: str):
        text = (text or "").strip()
        if not text or self.stop_event.is_set():
            return
        logger.info(f"[{self.call_id[:8]}] TTS enqueue: {text[:60]!r}")
        # CALLBOT_TTS_DISABLED=1: 합성/재생 skip. 'TTS enqueue' 로그는 그대로
        # 남으므로 server.py SSE 가 봇 응답 텍스트는 정상 push (브라우저 chat
        # 에는 표시됨). STT 만 검증할 때 RTP 잡음/echo 제거 목적.
        if os.environ.get("CALLBOT_TTS_DISABLED") == "1":
            return
        with self._inflight_lock:
            self._inflight += 1
            self._idle_event.clear()
        future = self.executor.submit(synthesize_pcm_8bit_unsigned, text)
        self.queue.put((text, future))

    def wait_drained(self, timeout: float = 60.0) -> bool:
        """모든 enqueue 항목의 재생이 끝날 때까지 대기. timeout 시 False."""
        return self._idle_event.wait(timeout=timeout)

    def clear_pending(self):
        """
        큐에 쌓인 (아직 재생 시작 전) 항목들을 drop.
        진행 중인 재생은 stop_event 로 별도 중단. 보통은 통화 종료 직전 사용.
        """
        dropped = 0
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                # poison 다시 넣지 말고 drop — shutdown 에서 다시 put
                continue
            dropped += 1
            with self._inflight_lock:
                self._inflight -= 1
                if self._inflight == 0:
                    self._idle_event.set()
        if dropped:
            logger.info(f"[{self.call_id[:8]}] TTS pipeline cleared {dropped} pending items")

    def shutdown(self, timeout: float = 5.0):
        """클린 종료 — 큐 잔여 drop, play_thread join, executor shutdown."""
        self.stop_event.set()
        self.queue.put(None)  # poison
        self.play_thread.join(timeout=timeout)
        if self.play_thread.is_alive():
            logger.warning(f"[{self.call_id[:8]}] TTS play_thread did not exit within {timeout}s")
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # cancel_futures 는 Python 3.9+
            self.executor.shutdown(wait=False)

    def _play_loop(self):
        from pyVoIP.VoIP import CallState
        while not self.stop_event.is_set():
            try:
                item = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                break
            text, future = item
            try:
                if self.call.state != CallState.ANSWERED:
                    logger.info(
                        f"[{self.call_id[:8]}] TTS skip (call not answered): {text[:40]!r}"
                    )
                    continue
                try:
                    pcm_8bit = future.result(timeout=15.0)
                except Exception as e:
                    logger.error(
                        f"[{self.call_id[:8]}] TTS synth failed for {text[:40]!r}: {e}"
                    )
                    continue
                if not self._primed:
                    _send_priming_silence(self.call)
                    self._primed = True
                    logger.info(f"[{self.call_id[:8]}] TTS RTP primed (one-shot per call)")
                logger.info(
                    f"[{self.call_id[:8]}] TTS play start "
                    f"({len(pcm_8bit)} bytes, {len(pcm_8bit)//_CHUNK_SIZE} chunks): "
                    f"{text[:60]!r}"
                )
                sent = _send_pcm_8bit(self.call, pcm_8bit, text, self.stop_event)
                logger.info(
                    f"[{self.call_id[:8]}] TTS play done ({sent} chunks): {text[:40]!r}"
                )
            except Exception as e:
                logger.error(f"[{self.call_id[:8]}] TTS pipeline error: {e}", exc_info=True)
            finally:
                with self._inflight_lock:
                    self._inflight -= 1
                    if self._inflight == 0 and self.queue.empty():
                        self._idle_event.set()


def handle_call(call):
    """
    pyVoIP callCallback — 인커밍 콜 처리

    Args:
        call: pyVoIP VoIPCall 인스턴스
    """
    from pyVoIP.VoIP import CallState

    call_id  = str(uuid.uuid4())
    pipeline: Optional[TTSPipeline] = None
    athena   = None
    dialog_count = 0
    logger.info(f"[{call_id[:8]}] Incoming call received")

    try:
        call.answer()
        pipeline = TTSPipeline(call, call_id)

        # ── RTP 수신 소켓 NAT 핀홀 개통 ──────────────────────────────
        # pyVoIP는 sin(수신)과 sout(송신) 소켓을 분리합니다.
        # TTS 프라이밍은 sout→Asterisk 경로만 열어주므로,
        # sin←Asterisk 방향은 sin이 먼저 더미 패킷을 보내야 핀홀이 열립니다.
        for rtp_client in call.RTPClients:
            try:
                rtp_client.sin.sendto(b'\x00', (rtp_client.outIP, rtp_client.outPort))
                logger.info(
                    f"[RTP] Hole punch: sin(:{rtp_client.inPort}) → "
                    f"{rtp_client.outIP}:{rtp_client.outPort}"
                )
            except Exception as e:
                logger.warning(f"[RTP] Hole punch failed: {e}")

        time.sleep(1.5)  # SIP/RTP 미디어 스트림 안정화 대기 (VPN 환경 고려)

        # ── 1. Athena 세션 시작 ──────────────────────────────────────
        greeting = cfg.FALLBACK_GREETING

        if cfg.athena_configured():
            try:
                athena = AthenaClient(
                    api_url=cfg.ATHENA_SITE,
                    auth_token=cfg.ATHENA_AUTH,
                    users_id=int(cfg.ATHENA_USER_ID),
                    chat_rooms_id=int(cfg.ATHENA_CHAT_ROOMS_ID),
                    scenarios_id=cfg.ATHENA_SCENARIOS_ID,
                )
                thread_id = athena.start(uui=call_id, voc_types=[])
                if thread_id:
                    info     = athena.get_init_info()
                    greeting = info.get("greeting_message") or greeting
                    logger.info(f"[{call_id[:8]}] Athena session started (thread={thread_id})")
                else:
                    logger.warning(f"[{call_id[:8]}] Athena start failed, using fallback mode")
                    athena = None
            except Exception as e:
                logger.error(f"[{call_id[:8]}] Athena init error: {e}")
                athena = None
        else:
            logger.info(f"[{call_id[:8]}] Athena not configured, using fallback mode")

        # ── 2. 인사말 TTS 재생 ────────────────────────────────────────
        # 인사말은 STT 시작 전 완전히 재생되어야 echo 가 잡히지 않으므로 wait_drained.
        pipeline.enqueue(greeting)
        pipeline.wait_drained()

        # ── 3. 대화 루프 ──────────────────────────────────────────────
        no_input_count = 0

        while call.state == CallState.ANSWERED:
            logger.info(f"[{call_id[:8]}] Turn {dialog_count}: Listening...")

            # TTS 직후 즉시 STT 시작 — pmin에 누적된 큐(에코 + 사용자 발화)를
            # 통째로 받아 STT 버퍼에 쌓는다. 이전 _drain_rtp_buffer는 silence streak를
            # 기다리느라 사용자 발화를 통째로 삼키는 부작용이 있어 제거.
            # 에코는 _has_interim_content 가드(stt.py)가 VAD 단계에서 필터링.

            # 3a. STT 스트리밍 청취
            audio_src = CallAudioSource(call, timeout_sec=cfg.LISTEN_TIMEOUT_SEC)
            stt       = create_stt(cfg.CREDENTIALS_DIR)

            try:
                stt.initialize()
                stt.start_streaming(audio_src)
                # 블로킹 wait_for_result() 대신 능동 폴링 루프 사용
                # (robi-t-callbot ai_handler.detect_voice() 폴링 패턴 적용)
                success, transcript = _poll_stt_result(
                    stt, audio_src, call, call_id, dialog_count
                )
            except Exception as e:
                logger.error(f"[{call_id[:8]}] STT error: {e}")
                success, transcript = False, "error"
            finally:
                audio_src.stop()
                stt.finalize()

            logger.info(f"[{call_id[:8]}] STT result: success={success}, transcript={transcript!r}")

            # 3b. 무음/인식 실패 처리
            if not success or transcript in ("non_voice", "", "timeout", "error"):
                no_input_count += 1
                logger.info(f"[{call_id[:8]}] No input ({no_input_count}/{cfg.MAX_NO_INPUT})")

                if no_input_count >= cfg.MAX_NO_INPUT:
                    pipeline.enqueue(cfg.FALLBACK_GOODBYE)
                    pipeline.wait_drained()
                    break

                pipeline.enqueue(cfg.FALLBACK_NO_INPUT)
                pipeline.wait_drained()
                continue

            no_input_count = 0

            # 3c. Athena LLM 질의 — SSE stream → TTSPipeline enqueue
            #
            # meta_status: 본 답변(reply) 도착 전 안내 멘트 — 즉시 enqueue.
            #              종류가 2~3개로 한정 → 캐시 적중률 ~100%.
            #
            # reply/command: 본 답변. 토큰 chunk 단위로 도착하므로 "||" 경계가
            #                들어올 때마다 한 문장씩 enqueue. stream 종료 후
            #                buffer 잔여분은 마지막 문장으로 한 번 더 enqueue.
            #
            # on_event 는 enqueue만 호출(non-blocking) → SSE 소비가 멈추지 않고,
            # 합성/재생은 pipeline 내부 worker 가 병렬로 진행. sentence N 재생 중
            # sentence N+1 이 미리 합성되어 gap 거의 0.
            athena_streamed = False
            if athena:
                sentence_buffer = [""]
                reply_started   = [False]

                def on_event(event):
                    etype = event.get("type")
                    text  = event.get("text", "")
                    if etype == "meta_status":
                        if text and not reply_started[0]:
                            logger.info(f"[{call_id[:8]}] Athena meta_status: {text!r}")
                            pipeline.enqueue(text)
                        return
                    if etype in ("reply", "command") and text:
                        reply_started[0] = True
                        sentence_buffer[0] += text
                        while "||" in sentence_buffer[0]:
                            sentence, _, rest = sentence_buffer[0].partition("||")
                            pipeline.enqueue(sentence)
                            sentence_buffer[0] = rest

                try:
                    events = athena.query_sync(transcript, dialog_count, on_event=on_event)
                    # stream 종료 후 buffer 잔여분(마지막 문장; "||" 미포함) enqueue
                    if sentence_buffer[0].strip():
                        pipeline.enqueue(sentence_buffer[0])
                        sentence_buffer[0] = ""
                    athena_streamed = True
                except Exception as e:
                    logger.error(f"[{call_id[:8]}] Athena query error: {e}")
                    # 큐에 부분 합성된 reply 가 있으면 fallback 멘트와 충돌 — 제거
                    pipeline.clear_pending()
                    events = [{"type": "reply", "text": cfg.FALLBACK_GOODBYE}]
            else:
                # Athena 미설정: fallback 종료
                events = [
                    {"type": "command", "command": "end_call",
                     "text": cfg.FALLBACK_GOODBYE, "dest_number": "00000"}
                ]

            # 3d. 응답 텍스트 수집 — stream 경로면 이미 enqueue 완료, fallback만 enqueue
            reply_parts   = [
                e["text"] for e in events
                if e["type"] in ("reply", "command") and e.get("text")
            ]
            command_event = next(
                (e for e in events if e["type"] == "command"), None
            )

            if reply_parts and not athena_streamed:
                full_reply = "".join(reply_parts)
                logger.info(f"[{call_id[:8]}] Reply (fallback): {full_reply[:60]}...")
                pipeline.enqueue(full_reply)
            elif reply_parts:
                full_reply = "".join(reply_parts)
                logger.info(f"[{call_id[:8]}] Reply (streamed): {full_reply[:80]}...")

            # 다음 turn STT 시작 전 모든 재생 완료 대기 — echo 방지
            pipeline.wait_drained()

            # 3e. 명령 처리
            if command_event:
                cmd = command_event.get("command", "")
                logger.info(f"[{call_id[:8]}] Command: {cmd}")

                if cmd == "end_call":
                    break
                elif cmd == "transfer_call":
                    dest = command_event.get("dest_number", "")
                    logger.info(f"[{call_id[:8]}] Transfer to: {dest}")
                    # pyVoIP에서 직접 transfer 지원이 제한적이므로 통화 종료
                    # 향후 AMI를 통한 transfer 구현 가능
                    break

            dialog_count += 1

    except Exception as e:
        logger.error(f"[{call_id[:8]}] Unhandled error in call: {e}", exc_info=True)

    finally:
        # ── 4. TTS 파이프라인 종료 ───────────────────────────────────
        if pipeline is not None:
            try:
                pipeline.shutdown(timeout=5.0)
            except Exception as e:
                logger.warning(f"[{call_id[:8]}] TTS pipeline shutdown error: {e}")

        # ── 5. Athena 세션 종료 ───────────────────────────────────────
        if athena:
            try:
                athena.end(uui=call_id)
            except Exception as e:
                logger.warning(f"[{call_id[:8]}] Athena end error: {e}")

        # ── 6. 통화 종료 ──────────────────────────────────────────────
        try:
            from pyVoIP.VoIP import CallState
            if call.state == CallState.ANSWERED:
                call.hangup()
        except Exception:
            pass

        logger.info(f"[{call_id[:8]}] Call ended (turns={dialog_count})")


def _poll_stt_result(
    stt: GoogleSTTV2,
    audio_src: CallAudioSource,
    call,
    call_id: str,
    dialog_count: int,
) -> Tuple[bool, str]:
    """
    robi-t-callbot ai_handler.detect_voice() 의 폴링 루프 패턴을 callbot에 적용.

    블로킹 wait_for_result() 대신 50ms 주기 폴링으로 STT 상태를 모니터링.

    흐름:
      ① speech_started 감지 → no-speech 타임아웃 카운터 리셋
      ② EOS 감지 → audio_src.stop() (generator는 이미 break 완료)
      ③ final transcript 감지 → 반환
      ④ no-speech 타임아웃: LISTEN_TIMEOUT_SEC 동안 발화 없으면 "non_voice"
      ⑤ post-EOS 타임아웃: EOS 후 8s 내 final result 없으면 "timeout"
    """
    from pyVoIP.VoIP import CallState

    POLL_INTERVAL       = 0.05                                      # 50ms (robi-t-callbot 동일)
    MAX_NO_SPEECH_TICKS = int(cfg.LISTEN_TIMEOUT_SEC / POLL_INTERVAL)
    MAX_POST_EOS_TICKS  = int(8.0 / POLL_INTERVAL)                  # EOS 후 최대 8s 대기
    # EOS flush: 16 chunks × 125ms = 2s + Google 처리 대기 = 8s 여유 필요

    speech_started  = False
    eos_detected    = False
    no_speech_ticks = 0
    post_eos_ticks  = 0
    prev_transcript = ""

    while True:
        # 통화 종료 체크
        try:
            if call.state != CallState.ANSWERED:
                break
        except Exception:
            break

        # STT 내부 오류 즉시 반환
        if stt._stt_error:
            return False, "error"

        # ① speech_started 폴링
        started, _ = stt.get_speech_started()
        if started and not speech_started:
            speech_started  = True
            no_speech_ticks = 0
            logger.debug(f"[{call_id[:8]}] Turn {dialog_count}: Speech started")

        # ② EOS 폴링 — robi-t-callbot handle_eos_detected() 패턴
        # generator는 이미 break되어 있거나 break 직전이므로
        # audio_src.stop()은 안전망 역할 (flush 없이 즉시 종료)
        eos, eos_t = stt.get_and_consume_eos()
        if eos and not eos_detected:
            eos_detected   = True
            post_eos_ticks = 0
            audio_src.stop()
            logger.info(
                f"[{call_id[:8]}] Turn {dialog_count}: EOS detected"
                + (f" at {eos_t:.3f}" if eos_t else "")
                + ", audio stopped"
            )

        # ③ final transcript 폴링
        transcript = stt.get_final_transcript()
        if transcript and transcript != prev_transcript:
            prev_transcript = transcript
            return True, transcript

        # result_event 최종 확인 (스트림 종료 fallback)
        if stt._result_event.is_set():
            if stt._stt_error:
                return False, stt._stt_error
            return True, stt._final_transcript

        # ④ no-speech 타임아웃
        if not speech_started and not eos_detected:
            no_speech_ticks += 1
            if no_speech_ticks > MAX_NO_SPEECH_TICKS:
                logger.info(
                    f"[{call_id[:8]}] Turn {dialog_count}: "
                    f"No speech timeout ({cfg.LISTEN_TIMEOUT_SEC}s)"
                )
                return False, "non_voice"

        # ⑤ post-EOS 타임아웃
        if eos_detected:
            post_eos_ticks += 1
            if post_eos_ticks > MAX_POST_EOS_TICKS:
                logger.warning(
                    f"[{call_id[:8]}] Turn {dialog_count}: "
                    f"Post-EOS final result timeout after {post_eos_ticks * POLL_INTERVAL:.1f}s"
                )
                return False, "timeout"

        # 이벤트 기반 대기 — busy-wait 없이 최대 POLL_INTERVAL 대기
        stt._transcript_ready.wait(timeout=POLL_INTERVAL)
        if stt._transcript_ready.is_set():
            stt._transcript_ready.clear()

    return False, "non_voice"



"""
callbot/call_handler.py — 통화 상태 머신

pyVoIP callCallback에서 호출됩니다 (별도 스레드).
Athena 세션 관리, STT/TTS 루프, 통화 종료 처리를 담당합니다.
"""
import time
import uuid
import logging

import config as cfg
from audio_source import CallAudioSource
from stt import create_stt
from tts import synthesize_pcm_8k
from athena import AthenaClient

logger = logging.getLogger("callbot")


def handle_call(call):
    """
    pyVoIP callCallback — 인커밍 콜 처리

    Args:
        call: pyVoIP VoIPCall 인스턴스
    """
    from pyVoIP.VoIP import CallState

    call_id = str(uuid.uuid4())
    logger.info(f"[{call_id[:8]}] Incoming call received")

    try:
        call.answer()
        time.sleep(1.5)  # SIP/RTP 미디어 스트림 안정화 대기 (VPN 환경 고려)

        # ── 1. Athena 세션 시작 ──────────────────────────────────────
        athena  = None
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
                thread_id = athena.start(uui=call_id, voc_types=["inquiry"])
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
        _play_tts(call, greeting)

        # ── 3. 대화 루프 ──────────────────────────────────────────────
        no_input_count = 0
        dialog_count   = 0

        while call.state == CallState.ANSWERED:
            logger.info(f"[{call_id[:8]}] Turn {dialog_count}: Listening...")

            # 3a. STT 스트리밍 청취
            audio_src = CallAudioSource(call, timeout_sec=cfg.LISTEN_TIMEOUT_SEC)
            stt       = create_stt(cfg.CREDENTIALS_DIR)

            try:
                stt.initialize()
                stt.start_streaming(audio_src)
                success, transcript = stt.wait_for_result(
                    timeout=cfg.LISTEN_TIMEOUT_SEC + 5
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
                    _play_tts(call, cfg.FALLBACK_GOODBYE)
                    break

                _play_tts(call, cfg.FALLBACK_NO_INPUT)
                continue

            no_input_count = 0

            # 3c. Athena LLM 질의
            if athena:
                try:
                    events = athena.query_sync(transcript, dialog_count)
                except Exception as e:
                    logger.error(f"[{call_id[:8]}] Athena query error: {e}")
                    events = [{"type": "reply", "text": cfg.FALLBACK_GOODBYE}]
            else:
                # Athena 미설정: fallback 종료
                events = [
                    {"type": "command", "command": "end_call",
                     "text": cfg.FALLBACK_GOODBYE, "dest_number": "00000"}
                ]

            # 3d. 응답 텍스트 수집 및 TTS 재생
            reply_parts   = [
                e["text"] for e in events
                if e["type"] in ("reply", "command") and e.get("text")
            ]
            command_event = next(
                (e for e in events if e["type"] == "command"), None
            )

            if reply_parts:
                full_reply = " ".join(reply_parts)
                logger.info(f"[{call_id[:8]}] Reply: {full_reply[:60]}...")
                _play_tts(call, full_reply)

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
        # ── 4. Athena 세션 종료 ───────────────────────────────────────
        if athena:
            try:
                athena.end(uui=call_id)
            except Exception as e:
                logger.warning(f"[{call_id[:8]}] Athena end error: {e}")

        # ── 5. 통화 종료 ──────────────────────────────────────────────
        try:
            from pyVoIP.VoIP import CallState
            if call.state == CallState.ANSWERED:
                call.hangup()
        except Exception:
            pass

        logger.info(f"[{call_id[:8]}] Call ended (turns={dialog_count})")


def _play_tts(call, text: str):
    """
    TTS 합성 후 pyVoIP writeAudio()로 재생

    320 bytes (20ms, 8kHz 16-bit) 단위로 전송합니다.
    """
    if not text or not text.strip():
        return

    from pyVoIP.VoIP import CallState

    # 320 bytes = 160 samples × 2 bytes = 20ms at 8kHz
    CHUNK_SIZE = 320
    SLEEP_SEC  = 0.018  # 20ms보다 약간 짧게 → 버퍼 유지
    silence    = b"\x00" * CHUNK_SIZE

    # RTP 스트림 프라이밍: 무음 0.5초 전송으로 RTP 경로 개통
    for _ in range(25):
        if call.state != CallState.ANSWERED:
            return
        try:
            call.writeAudio(silence)
        except Exception:
            return
        time.sleep(SLEEP_SEC)

    try:
        pcm_8k = synthesize_pcm_8k(text)
    except Exception as e:
        logger.error(f"[TTS] Synthesis failed: {e}")
        return

    logger.info(f"[TTS] Playing {len(pcm_8k)} bytes ({len(pcm_8k)//320} chunks): {text[:40]!r}")
    sent = 0
    for i in range(0, len(pcm_8k), CHUNK_SIZE):
        if call.state != CallState.ANSWERED:
            break

        chunk = pcm_8k[i:i + CHUNK_SIZE]
        # 마지막 청크가 짧으면 무음으로 패딩
        if len(chunk) < CHUNK_SIZE:
            chunk = chunk + b"\x00" * (CHUNK_SIZE - len(chunk))

        try:
            call.writeAudio(chunk)
            sent += 1
        except Exception as e:
            logger.warning(f"[TTS] writeAudio error: {e}")
            break

        time.sleep(SLEEP_SEC)

    logger.info(f"[TTS] Done ({sent} chunks sent)")

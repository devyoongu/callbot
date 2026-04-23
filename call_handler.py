"""
callbot/call_handler.py — 통화 상태 머신

pyVoIP callCallback에서 호출됩니다 (별도 스레드).
Athena 세션 관리, STT/TTS 루프, 통화 종료 처리를 담당합니다.
"""
import audioop
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
        _play_tts(call, greeting)

        # ── 3. 대화 루프 ──────────────────────────────────────────────
        no_input_count = 0
        dialog_count   = 0

        while call.state == CallState.ANSWERED:
            logger.info(f"[{call_id[:8]}] Turn {dialog_count}: Listening...")

            # TTS 에코 소거: 직전 TTS 재생 후 RTP 버퍼에 남은 에코를 버린다.
            # 0.5초간 오디오를 읽되 STT에 전달하지 않음.
            _drain_audio(call, seconds=0.5)

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
                full_reply = "".join(reply_parts)
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


def _drain_audio(call, seconds: float = 0.5):
    """
    TTS 재생 직후 RTP 에코 소거용 오디오 버퍼 드레인

    pyVoIP의 RTP 수신 버퍼를 비워 이전 TTS 에코가 STT에 유입되지 않도록 합니다.
    read_audio()로 실제 읽되 데이터는 버립니다.
    """
    from pyVoIP.VoIP import CallState
    deadline = time.time() + seconds
    while time.time() < deadline:
        if call.state != CallState.ANSWERED:
            break
        try:
            call.read_audio(160, False)
        except Exception:
            break
        time.sleep(0.01)


def _play_tts(call, text: str):
    """
    TTS 합성 후 pyVoIP writeAudio()로 재생

    pyVoIP encode_pcmu()는 8-bit unsigned PCM (width=1)을 기대합니다.
    160 bytes (20ms, 8kHz 8-bit) 단위로 전송합니다.
    """
    if not text or not text.strip():
        return

    from pyVoIP.VoIP import CallState

    # pyVoIP 내부 encode_pcmu는 width=1 (8-bit) 입력을 기대함
    # 160 samples × 1 byte = 20ms at 8kHz
    CHUNK_SIZE = 160
    SLEEP_SEC  = 0.018  # 20ms보다 약간 짧게 → 버퍼 유지
    # pyVoIP 무음: 0x80 = 128 (8-bit unsigned에서 0 중심값)
    silence    = b"\x80" * CHUNK_SIZE

    # RTP 스트림 프라이밍: 무음 0.5초 전송으로 RTP 경로 개통
    # 프라이밍 중에도 수신 버퍼 드레인 (에코 누적 방지)
    for _ in range(25):
        if call.state != CallState.ANSWERED:
            return
        try:
            call.writeAudio(silence)
            call.read_audio(160, False)  # 수신 버퍼 실시간 소거
        except Exception:
            return
        time.sleep(SLEEP_SEC)

    try:
        pcm_16k = synthesize_pcm_8k(text)  # 16-bit signed PCM at 8kHz
    except Exception as e:
        logger.error(f"[TTS] Synthesis failed: {e}")
        return

    # 16-bit signed → 8-bit signed → 8-bit unsigned (pyVoIP가 기대하는 포맷)
    pcm_8bit = audioop.lin2lin(pcm_16k, 2, 1)    # 16-bit → 8-bit signed
    pcm_8bit = audioop.bias(pcm_8bit, 1, 128)     # signed → unsigned (0~255)

    logger.info(f"[TTS] Playing {len(pcm_8bit)} bytes ({len(pcm_8bit)//160} chunks): {text[:40]!r}")
    sent = 0
    for i in range(0, len(pcm_8bit), CHUNK_SIZE):
        if call.state != CallState.ANSWERED:
            break

        chunk = pcm_8bit[i:i + CHUNK_SIZE]
        # 마지막 청크가 짧으면 무음(0x80)으로 패딩
        if len(chunk) < CHUNK_SIZE:
            chunk = chunk + b"\x80" * (CHUNK_SIZE - len(chunk))

        try:
            call.writeAudio(chunk)
            call.read_audio(160, False)  # TTS 재생 중 에코 실시간 소거
            sent += 1
        except Exception as e:
            logger.warning(f"[TTS] writeAudio error: {e}")
            break

        time.sleep(SLEEP_SEC)

    logger.info(f"[TTS] Done ({sent} chunks sent)")

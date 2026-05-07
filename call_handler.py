"""
callbot/call_handler.py — 통화 상태 머신

pyVoIP callCallback에서 호출됩니다 (별도 스레드).
Athena 세션 관리, STT/TTS 루프, 통화 종료 처리를 담당합니다.
"""
import audioop
import time
import uuid
import logging
from typing import Tuple

import config as cfg
from audio_source import CallAudioSource
from stt import GoogleSTTV2, create_stt
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
    for _ in range(25):
        if call.state != CallState.ANSWERED:
            return
        try:
            call.writeAudio(silence)
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
            sent += 1
        except Exception as e:
            logger.warning(f"[TTS] writeAudio error: {e}")
            break

        time.sleep(SLEEP_SEC)

    logger.info(f"[TTS] Done ({sent} chunks sent)")

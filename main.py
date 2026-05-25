"""
callbot/main.py — AI 콜봇 진입점

SIP 내선 2001로 Asterisk에 등록하고 인커밍 콜을 대기합니다.
1001 소프트폰에서 2001로 전화를 걸면 call_handler.handle_call()이 실행됩니다.

실행:
    cd callbot/
    python main.py                       # SIP 등록 후 콜 대기 (실제 운영)
    python main.py --wav wav/sentence_05.wav  # Asterisk 없이 WAV로 단일턴 테스트
"""
import argparse
import audioop
import time
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

# callbot/ 디렉터리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import os
import config as cfg
from call_handler import handle_call

# ── 로깅 설정 ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s [%(threadName)s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("callbot")


def run_wav_mode(wav_path: str):
    """
    Asterisk 우회 — WAV 파일을 첫 턴 사용자 발화로 주입해 handle_call 전체 경로 실행.

    스코프: WAV 1개 → 1턴. 인사말 TTS는 실제 GCP TTS로 합성되며 MockCall이 swallow.
    첫 턴 STT/Athena/응답 TTS 후 두 번째 턴 진입 시 MockCall이 ENDED 전환되어 종료.
    """
    from test_stt_wav import load_wav_as_16k_pcm
    from test_stt_pipeline import MockCall

    if not Path(wav_path).exists():
        logger.error(f"WAV file not found: {wav_path}")
        sys.exit(2)

    logger.info("=" * 60)
    logger.info("AI Callbot — WAV mode (Asterisk bypass)")
    logger.info(f"  WAV file   : {wav_path}")
    logger.info(f"  STT Model  : {cfg.GCP_STT_MODEL}")
    logger.info(f"  Athena     : {'configured' if cfg.athena_configured() else 'NOT configured (fallback mode)'}")
    logger.info("=" * 60)

    # 1) WAV → 16kHz 16-bit signed PCM
    pcm_16k_16bit = load_wav_as_16k_pcm(wav_path)

    # 2) 16kHz → 8kHz 16-bit signed (anti-aliasing 포함)
    arr16k        = np.frombuffer(pcm_16k_16bit, dtype=np.int16).astype(np.float32)
    arr8k         = resample_poly(arr16k, up=1, down=2)
    pcm_8k_16bit  = np.clip(arr8k, -32768, 32767).astype(np.int16).tobytes()

    # 3) 16-bit signed → 8-bit signed → 8-bit unsigned (pyVoIP 포맷)
    pcm_8bit_signed   = audioop.lin2lin(pcm_8k_16bit, 2, 1)
    pcm_8bit_unsigned = audioop.bias(pcm_8bit_signed, 1, 128)

    duration_sec = len(pcm_8k_16bit) / (8000 * 2)
    logger.info(f"  Audio prep : 8kHz 8-bit unsigned, {len(pcm_8bit_unsigned)} bytes, {duration_sec:.2f}s")

    # 4) MockCall 구성
    #   pre_silence_reads=10  → 짧은 무음 선행 (호환용; drain 제거 후엔 0이어도 무방)
    #   post_silence_reads=300 → ~6s 무음 후 ENDED. 첫 턴 STT/응답 TTS 동안 충분.
    mock = MockCall(
        pcm_8bit_unsigned,
        post_silence_reads=300,
        pre_silence_reads=10,
    )

    handle_call(mock)
    logger.info(f"WAV mode finished (TTS chunks swallowed: {mock._tts_chunks_written})")


def main():
    parser = argparse.ArgumentParser(description="AI Callbot")
    parser.add_argument(
        "--wav",
        help="Bypass SIP — feed WAV file as first user utterance through handle_call",
    )
    args = parser.parse_args()

    if args.wav:
        run_wav_mode(args.wav)
        return

    local_ip = cfg.get_local_ip()

    logger.info("=" * 60)
    logger.info("AI Callbot Starting")
    logger.info(f"  SIP Server : {cfg.SIP_SERVER}:{cfg.SIP_PORT}")
    logger.info(f"  Local Port : {cfg.SIP_LOCAL_PORT}")
    logger.info(f"  Extension  : {cfg.SIP_USERNAME}")
    logger.info(f"  Local IP   : {local_ip}")
    if cfg.STT_PROVIDER == "qwen3-asr":
        logger.info(f"  STT        : qwen3-asr ({cfg.QWEN_ASR_URL}, lang={cfg.QWEN_ASR_LANGUAGE})")
    else:
        logger.info(f"  STT        : google ({cfg.GCP_STT_MODEL})")
    logger.info(f"  TTS Voice  : {cfg.GCP_TTS_VOICE}")
    logger.info(f"  Athena     : {'configured' if cfg.athena_configured() else 'NOT configured (fallback mode)'}")
    logger.info("=" * 60)

    try:
        from pyVoIP.VoIP import VoIPPhone
    except ImportError:
        logger.error("pyVoIP not installed. Run: pip install pyVoIP==1.6.4")
        sys.exit(1)

    # RTP 포트 범위를 좁혀서 방화벽 관리 및 디버깅 용이하게 설정
    RTP_PORT_LOW  = int(os.environ.get("RTP_PORT_LOW",  "16000"))
    RTP_PORT_HIGH = int(os.environ.get("RTP_PORT_HIGH", "16100"))
    logger.info(f"  RTP Range  : {RTP_PORT_LOW}-{RTP_PORT_HIGH}")

    phone = VoIPPhone(
        server=cfg.SIP_SERVER,
        port=cfg.SIP_PORT,
        username=cfg.SIP_USERNAME,
        password=cfg.SIP_PASSWORD,
        myIP=local_ip,
        sipPort=cfg.SIP_LOCAL_PORT,
        callCallback=handle_call,
        rtpPortLow=RTP_PORT_LOW,
        rtpPortHigh=RTP_PORT_HIGH,
    )

    phone.start()
    logger.info(f"Registered as {cfg.SIP_USERNAME}@{cfg.SIP_SERVER}. Waiting for calls... (Ctrl+C to stop)")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        phone.stop()
        logger.info("Callbot stopped.")


if __name__ == "__main__":
    main()

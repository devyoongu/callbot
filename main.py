"""
callbot/main.py — AI 콜봇 진입점

SIP 내선 2001로 Asterisk에 등록하고 인커밍 콜을 대기합니다.
1001 소프트폰에서 2001로 전화를 걸면 call_handler.handle_call()이 실행됩니다.

실행:
    cd callbot/
    python main.py

서버 설정 (Asterisk):
    pjsip.conf   : 2001 endpoint/auth/aor 추가
    extensions.conf: exten => 2001,1,Dial(PJSIP/2001,30)
"""
import time
import logging
import sys
from pathlib import Path

# callbot/ 디렉터리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import config as cfg
from call_handler import handle_call

# ── 로깅 설정 ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("callbot")


def main():
    local_ip = cfg.get_local_ip()

    logger.info("=" * 60)
    logger.info("AI Callbot Starting")
    logger.info(f"  SIP Server : {cfg.SIP_SERVER}:{cfg.SIP_PORT}")
    logger.info(f"  Local Port : {cfg.SIP_LOCAL_PORT}")
    logger.info(f"  Extension  : {cfg.SIP_USERNAME}")
    logger.info(f"  Local IP   : {local_ip}")
    logger.info(f"  STT Model  : {cfg.GCP_STT_MODEL}")
    logger.info(f"  TTS Voice  : {cfg.GCP_TTS_VOICE}")
    logger.info(f"  Athena     : {'configured' if cfg.athena_configured() else 'NOT configured (fallback mode)'}")
    logger.info("=" * 60)

    try:
        from pyVoIP.VoIP import VoIPPhone
    except ImportError:
        logger.error("pyVoIP not installed. Run: pip install pyVoIP==1.6.4")
        sys.exit(1)

    phone = VoIPPhone(
        server=cfg.SIP_SERVER,
        port=cfg.SIP_PORT,
        username=cfg.SIP_USERNAME,
        password=cfg.SIP_PASSWORD,
        myIP=local_ip,
        sipPort=cfg.SIP_LOCAL_PORT,
        callCallback=handle_call,
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

"""
callbot/config.py — 환경변수 로더 및 전역 설정

callbot/.env 파일에서 값을 읽어옵니다.
"""
import os
import socket
from pathlib import Path
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).parent.resolve()
load_dotenv(SCRIPT_DIR / ".env")

# ── SIP 등록 ──────────────────────────────────────────────────────────
SIP_SERVER      = os.getenv("SIP_SERVER", "172.31.79.202")
SIP_PORT        = int(os.getenv("SIP_PORT", "5060"))
SIP_LOCAL_PORT  = int(os.getenv("SIP_LOCAL_PORT", "5060"))
SIP_USERNAME    = os.getenv("SIP_USERNAME", "2001")
SIP_PASSWORD    = os.getenv("SIP_PASSWORD", "secret2001")
_LOCAL_IP       = os.getenv("LOCAL_IP", "")

# ── Google Cloud ───────────────────────────────────────────────────────
GCP_PROJECT_ID   = os.getenv("GCP_PROJECT_ID", "gen-lang-client-0665942228")
GCP_STT_MODEL    = os.getenv("GCP_STT_MODEL", "chirp_3")
GCP_STT_LANGUAGE = os.getenv("GCP_STT_LANGUAGE", "ko-KR")
GCP_TTS_VOICE    = os.getenv("GCP_TTS_VOICE", "chirp3-hd-achernar")
# STT 인식 모드. "sync" 는 audio 전체를 한 번에 recognize (잘림 0, latency ~2.5s).
# "streaming" 은 streaming_recognize 사용 + is_final 누적 (chirp_3 의 mid-utterance
# is_final 을 끄지 못하므로 모두 누적해 join). latency 절감 시도용 — 회귀 시
# 환경변수로 즉시 sync 복귀.
GCP_STT_MODE     = os.getenv("GCP_STT_MODE", "sync")

_cred_env = os.getenv("CREDENTIALS_DIR", "")
CREDENTIALS_DIR = _cred_env if _cred_env else str(SCRIPT_DIR / "crendential")

# ── Athena LLM ────────────────────────────────────────────────────────
ATHENA_SITE          = os.getenv("ATHENA_SITE")
ATHENA_AUTH          = os.getenv("ATHENA_AUTH")
ATHENA_USER_ID       = os.getenv("ATHENA_USER_ID")
ATHENA_CHAT_ROOMS_ID = os.getenv("ATHENA_CHAT_ROOMS_ID")
ATHENA_SCENARIOS_ID  = os.getenv("ATHENA_SCENARIOS_ID")

# ── 대화 튜닝 ─────────────────────────────────────────────────────────
LISTEN_TIMEOUT_SEC = int(os.getenv("LISTEN_TIMEOUT_SEC", "25"))
MAX_NO_INPUT       = int(os.getenv("MAX_NO_INPUT", "3"))
FALLBACK_GREETING  = os.getenv("FALLBACK_GREETING", "안녕하세요. 무엇을 도와드릴까요?")
FALLBACK_NO_INPUT  = os.getenv("FALLBACK_NO_INPUT", "잘 들리지 않습니다. 다시 말씀해 주세요.")
FALLBACK_GOODBYE   = os.getenv("FALLBACK_GOODBYE", "감사합니다. 안녕히 계세요.")
# EOS 직후 즉시 enqueue 되는 pre-roll 멘트 (실험 옵션). Athena 의 meta_status
# ("잠시만 기다려 주세요." 등) 와 중복되어 두 번 들리는 어색함이 있어 default
# 비활성. 활성화 예: PREROLL_MESSAGE="네, 잠시만요."
# Athena meta_status 를 client 측에서 대체하는 디자인을 적용할 때 다시 활성화.
PREROLL_MESSAGE    = os.getenv("PREROLL_MESSAGE", "")


def athena_configured() -> bool:
    """Athena LLM 연동에 필요한 모든 환경변수가 설정됐는지 확인"""
    return all([ATHENA_SITE, ATHENA_AUTH, ATHENA_USER_ID,
                ATHENA_CHAT_ROOMS_ID, ATHENA_SCENARIOS_ID])


def get_local_ip() -> str:
    """로컬 네트워크 IP 반환. LOCAL_IP 환경변수가 설정되면 그 값 사용.

    SIP 서버 방향으로 실제 사용되는 인터페이스 IP를 감지합니다.
    VPN 등 복수 네트워크 인터페이스 환경에서 정확한 IP를 반환합니다.
    """
    if _LOCAL_IP:
        return _LOCAL_IP
    # SIP 서버 방향으로 나가는 인터페이스 IP 감지 (소켓은 실제 연결하지 않음)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((SIP_SERVER, SIP_PORT))
            return s.getsockname()[0]
    except Exception:
        pass
    # fallback: 인터넷 연결 기반 감지
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "0.0.0.0"

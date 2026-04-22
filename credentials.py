"""
callbot/credentials.py — Google Cloud 인증 로더

crendential/ 디렉터리의 JSON 키 파일을 round-robin으로 선택합니다.
복수의 서비스 계정 키를 배치하면 쿼터 분산이 가능합니다.
"""
import glob
import os
import threading
from typing import Optional

_rr_lock = threading.Lock()
_rr_index = 0
_rr_credentials_list: list = []


def _discover_credentials(credentials_dir: str) -> list:
    return sorted(glob.glob(os.path.join(credentials_dir, "*.json")))


def _get_next_round_robin(credentials_dir: str) -> Optional[str]:
    """Thread-safe round-robin으로 credentials 파일 경로 반환"""
    global _rr_index, _rr_credentials_list
    with _rr_lock:
        _rr_credentials_list = _discover_credentials(credentials_dir)
        if not _rr_credentials_list:
            return None
        path = _rr_credentials_list[_rr_index % len(_rr_credentials_list)]
        _rr_index += 1
        return path


def load_google_credentials(credentials_dir: str = None):
    """
    Google Cloud Credentials 객체 로드

    Args:
        credentials_dir: JSON 키 파일이 있는 디렉터리 경로.
                         None이면 config.CREDENTIALS_DIR 사용.
    Returns:
        google.oauth2.service_account.Credentials
    Raises:
        FileNotFoundError: credentials 파일이 없을 때
    """
    from google.oauth2 import service_account

    if credentials_dir is None:
        from config import CREDENTIALS_DIR
        credentials_dir = CREDENTIALS_DIR

    credentials_path = _get_next_round_robin(credentials_dir)
    if not credentials_path:
        raise FileNotFoundError(f"No credentials JSON found in: {credentials_dir}")

    print(f"[Auth] Using credentials: {os.path.basename(credentials_path)}")
    return service_account.Credentials.from_service_account_file(credentials_path)

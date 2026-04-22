"""
callbot/audio_utils.py — PCM 오디오 리샘플링 유틸리티

scipy.signal.resample_poly 사용 (anti-aliasing 필터 내장)
"""
import numpy as np
from scipy.signal import resample_poly


def upsample_8k_to_16k(pcm: bytes) -> bytes:
    """
    8kHz signed 16-bit PCM → 16kHz (2x upsample)
    resample_poly(up=2, down=1): anti-aliasing 필터 적용됨
    """
    if not pcm:
        return b""
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    out = resample_poly(arr, up=2, down=1)
    return np.clip(out, -32768, 32767).astype(np.int16).tobytes()


def downsample_24k_to_8k(pcm: bytes) -> bytes:
    """
    24kHz signed 16-bit PCM → 8kHz (3x downsample)
    resample_poly(up=1, down=3): anti-aliasing 필터로 aliasing 방지
    """
    if not pcm:
        return b""
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    out = resample_poly(arr, up=1, down=3)
    return np.clip(out, -32768, 32767).astype(np.int16).tobytes()


def pcm_chunks(pcm: bytes, chunk_size: int):
    """PCM bytes를 chunk_size 크기로 나누어 yield"""
    for i in range(0, len(pcm), chunk_size):
        yield pcm[i:i + chunk_size]

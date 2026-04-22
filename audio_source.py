"""
callbot/audio_source.py — pyVoIP RTP 오디오 → GoogleSTTV2 스트리밍 입력 변환

pyVoIP call.readAudio() 는 8kHz 16-bit PCM을 ~320 bytes 단위로 반환합니다.
GoogleSTTV2는 16kHz PCM 4000 bytes 청크를 기대합니다.

이 클래스가 수행하는 변환:
  readAudio() (~320 bytes, 8kHz) 누적 →
  2000 bytes (125ms, 8kHz) 묶음 →
  upsample_8k_to_16k() →
  4000 bytes (125ms, 16kHz) 청크 yield
"""
import time
import threading
from audio_utils import upsample_8k_to_16k

# 8kHz에서 125ms = 2000 bytes (1000 samples × 2 bytes)
_TARGET_8K_BYTES = 2000


class CallAudioSource:
    """
    pyVoIP 통화 오디오를 STT 스트리밍 입력으로 변환하는 이터러블

    사용법:
        audio_src = CallAudioSource(call, timeout_sec=15)
        stt.start_streaming(audio_src)
        ...
        audio_src.stop()  # 외부에서 강제 종료 시
    """

    def __init__(self, call, timeout_sec: int = 15):
        """
        Args:
            call: pyVoIP VoIPCall 인스턴스
            timeout_sec: 최대 청취 시간 (초). 초과 시 generator 종료.
        """
        self._call        = call
        self._timeout_sec = timeout_sec
        self._stop_event  = threading.Event()
        self._buffer      = b""

    def stop(self):
        """generator 루프를 외부에서 종료"""
        self._stop_event.set()

    def __iter__(self):
        """
        16kHz PCM 청크를 yield하는 generator.
        STT 클래스의 _audio_generator()가 별도 스레드에서 이 이터러블을 소비합니다.
        """
        from pyVoIP.VoIP import CallState

        deadline = time.time() + self._timeout_sec
        self._buffer = b""

        while not self._stop_event.is_set() and time.time() < deadline:
            # 통화가 끊어지면 종료
            try:
                state = self._call.state
            except Exception:
                break
            if state != CallState.ANSWERED:
                break

            # readAudio() — pyVoIP 내부에서 ~20ms 단위로 블로킹
            try:
                raw = self._call.readAudio()
            except Exception as e:
                print(f"[AudioSource] readAudio error: {e}")
                break

            if raw:
                self._buffer += raw

            # 2000 bytes (125ms at 8kHz) 누적 시 업샘플 후 yield
            while len(self._buffer) >= _TARGET_8K_BYTES:
                chunk_8k = self._buffer[:_TARGET_8K_BYTES]
                self._buffer = self._buffer[_TARGET_8K_BYTES:]
                chunk_16k = upsample_8k_to_16k(chunk_8k)
                yield chunk_16k

            # 오디오가 없는 경우 짧은 대기 (busy-loop 방지)
            if not raw:
                time.sleep(0.01)

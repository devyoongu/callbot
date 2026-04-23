"""
test_stt_pipeline.py — 실제 통화 경로와 동일한 STT 파이프라인 테스트

기존 test_stt_wav.py는 CallAudioSource를 우회하여 16kHz PCM을 GoogleSTTV2에 직접 주입.
이 스크립트는 실제 통화와 동일한 경로를 사용:

  TTS → 8kHz 8bit unsigned → MockCall.read_audio()
      → CallAudioSource.__iter__()
      → GoogleSTTV2._audio_generator()  ← client VAD 포함
      → Google STT V2

따라서 _has_interim_content 처리, client VAD EOS 감지 등을 실제와 동일하게 테스트 가능.

사용법:
    python test_stt_pipeline.py                    # 기본 테스트 케이스 4개
    python test_stt_pipeline.py "원하는 문장"        # 임의 문장 단일 테스트
"""
import audioop
import sys
import time
import threading

from pyVoIP.VoIP import CallState

import config as cfg
from audio_source import CallAudioSource
from stt import create_stt
from tts import synthesize_pcm_8k


# ── MockCall ──────────────────────────────────────────────────────────────────

class MockCall:
    """
    pyVoIP VoIPCall 최소 인터페이스를 구현하는 테스트용 mock.

    CallAudioSource가 요구하는 인터페이스:
      call.state                    → CallState.ANSWERED / ENDED
      call.read_audio(160, False)   → 8kHz 8-bit unsigned PCM 160 bytes
    """

    def __init__(self, pcm_8bit_unsigned: bytes, post_silence_reads: int = 60):
        """
        Args:
            pcm_8bit_unsigned: 8kHz 8-bit unsigned PCM (pyVoIP 포맷)
            post_silence_reads: 오디오 소진 후 ENDED 전환까지 무음 반환 횟수
                                 60회 × 20ms = 1.2초 무음 후 종료
        """
        self._data               = pcm_8bit_unsigned
        self._pos                = 0
        self._post_silence       = 0
        self._post_silence_limit = post_silence_reads
        self._state              = CallState.ANSWERED
        self._lock               = threading.Lock()

    @property
    def state(self):
        with self._lock:
            return self._state

    def read_audio(self, num_bytes: int, blocking: bool) -> bytes:
        """
        160 bytes의 8-bit unsigned PCM 반환.
        오디오 소진 후 무음(0x80)을 반환하다가 post_silence_limit 초과 시 ENDED 전환.
        """
        with self._lock:
            if self._pos < len(self._data):
                chunk = self._data[self._pos:self._pos + num_bytes]
                self._pos += num_bytes
                if len(chunk) < num_bytes:
                    chunk += b"\x80" * (num_bytes - len(chunk))
                return chunk
            else:
                self._post_silence += 1
                if self._post_silence >= self._post_silence_limit:
                    self._state = CallState.ENDED
                return b"\x80" * num_bytes


# ── 오디오 변환 ────────────────────────────────────────────────────────────────

def make_mock_pcm(text: str) -> bytes:
    """
    TTS 텍스트 → MockCall용 8kHz 8-bit unsigned PCM

    변환 경로:
      synthesize_pcm_8k(text)     → 8kHz 16-bit signed (tts.py:246)
      audioop.lin2lin(pcm, 2, 1)  → 8kHz 8-bit signed
      audioop.bias(pcm, 1, 128)   → 8kHz 8-bit unsigned

    CallAudioSource 내부가 이 과정을 역으로 수행하므로 원음이 복원됨.
    """
    print(f"  [TTS] 합성 중: '{text}'")
    pcm_8k_16bit     = synthesize_pcm_8k(text)
    pcm_8bit_signed  = audioop.lin2lin(pcm_8k_16bit, 2, 1)
    pcm_8bit_unsigned = audioop.bias(pcm_8bit_signed, 1, 128)
    duration_ms = len(pcm_8k_16bit) // (8000 * 2) * 1000
    print(f"  [TTS] 완료: {len(pcm_8k_16bit)} bytes, ~{len(pcm_8k_16bit) / (8000*2):.2f}s")
    return pcm_8bit_unsigned


# ── 테스트 실행 ────────────────────────────────────────────────────────────────

def run_test(label: str, text: str, expected_keywords: list) -> bool:
    """
    단일 테스트 케이스 실행.

    Args:
        label: 테스트 라벨 (예: "T1")
        text: TTS로 합성할 텍스트
        expected_keywords: transcript에 포함되어야 할 키워드 목록

    Returns:
        True if PASS, False if FAIL
    """
    print(f"\n{'='*60}")
    print(f"[{label}] {text}")
    print(f"{'='*60}")

    # 1. TTS → MockCall PCM
    t0 = time.time()
    mock_pcm  = make_mock_pcm(text)
    mock_call = MockCall(mock_pcm, post_silence_reads=60)

    # 2. CallAudioSource (실제 통화와 동일한 경로)
    audio_src = CallAudioSource(mock_call, timeout_sec=10)

    # 3. STT
    stt = create_stt()
    stt.initialize()

    print(f"  [TEST] STT 스트리밍 시작...")
    stt.start_streaming(audio_src)

    # 4. 결과 대기
    success, transcript = stt.wait_for_result(timeout=20.0)
    stt.finalize()

    elapsed = time.time() - t0

    # 5. 검증
    print(f"\n  [RESULT] success={success}, transcript='{transcript}', elapsed={elapsed:.1f}s")

    if not success:
        print(f"  ✗ FAIL  오류: {transcript}")
        return False

    if transcript in ("non_voice", "timeout"):
        print(f"  ✗ FAIL  인식 없음: {transcript}")
        return False

    missing = [kw for kw in expected_keywords if kw not in transcript]
    if missing:
        print(f"  ✗ FAIL  누락 키워드: {missing}")
        return False

    print(f"  ✓ PASS  모든 키워드 확인: {expected_keywords}")
    return True


# ── 기본 테스트 케이스 ─────────────────────────────────────────────────────────

DEFAULT_CASES = [
    # (label, text, expected_keywords)
    ("T1", "거기 위치 어디야",               ["거기", "위치"]),
    ("T2", "예약을 취소하고 싶습니다",         ["예약", "취소"]),
    ("T3", "안녕하세요",                     ["안녕"]),
    ("T4", "진료 예약을 다음 주로 변경하고 싶은데요", ["예약", "변경"]),
]


def main():
    if len(sys.argv) > 1:
        # 임의 문장 단일 테스트
        text  = " ".join(sys.argv[1:])
        cases = [("CUSTOM", text, [])]
    else:
        cases = DEFAULT_CASES

    print(f"\n{'#'*60}")
    print("STT 파이프라인 통합 테스트")
    print(f"{'#'*60}")
    print(f"파이프라인: TTS → MockCall → CallAudioSource → GoogleSTTV2")
    print(f"테스트 수: {len(cases)}")

    results = []
    for label, text, keywords in cases:
        passed = run_test(label, text, keywords)
        results.append((label, text, passed))

    # 최종 요약
    print(f"\n{'#'*60}")
    print("테스트 결과 요약")
    print(f"{'#'*60}")
    passed_count = sum(1 for _, _, p in results if p)
    for label, text, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  [{label}] {text}")
    print(f"\n  총 {passed_count}/{len(results)} 통과")

    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

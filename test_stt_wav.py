"""
test_stt_wav.py — WAV 파일로 GoogleSTTV2 STT 테스트

사용법:
    python test_stt_wav.py [wav_file]

기본값: wav/ramp_n1_0.wav

WAV 포맷 요구사항:
    - mono (1채널)
    - 16-bit signed PCM
    - 임의 샘플레이트 (자동으로 16kHz로 리샘플링)
"""
import sys
import wave
import numpy as np
from scipy.signal import resample_poly

# 프로젝트 모듈
from stt import create_stt

# GoogleSTTV2가 기대하는 청크 크기: 4000 bytes = 125ms at 16kHz 16-bit
_STT_CHUNK_SIZE = 4000
_STT_SAMPLE_RATE = 16000


def load_wav_as_16k_pcm(wav_path: str) -> bytes:
    """WAV 파일을 읽어 16kHz 16-bit signed mono PCM으로 변환"""
    with wave.open(wav_path, "rb") as wf:
        channels    = wf.getnchannels()
        sampwidth   = wf.getsampwidth()
        framerate   = wf.getframerate()
        nframes     = wf.getnframes()
        raw_data    = wf.readframes(nframes)

    print(f"[WAV] {wav_path}")
    print(f"      channels={channels}, sampwidth={sampwidth} bytes, "
          f"framerate={framerate} Hz, frames={nframes}, "
          f"duration={nframes/framerate:.2f}s")

    # 16-bit signed PCM으로 변환
    if sampwidth == 1:
        # 8-bit unsigned → 16-bit signed
        import audioop
        raw_data = audioop.bias(raw_data, 1, -128)
        raw_data = audioop.lin2lin(raw_data, 1, 2)
    elif sampwidth == 2:
        pass  # 이미 16-bit
    elif sampwidth == 3:
        # 24-bit → 16-bit (상위 2바이트만 사용)
        arr24 = np.frombuffer(raw_data, dtype=np.uint8).reshape(-1, 3)
        arr16 = (arr24[:, 1].astype(np.int16) | (arr24[:, 2].astype(np.int16) << 8))
        raw_data = arr16.tobytes()
    else:
        raise ValueError(f"Unsupported sampwidth: {sampwidth}")

    arr = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)

    # 스테레오 → 모노 (채널 평균)
    if channels == 2:
        arr = arr.reshape(-1, 2).mean(axis=1)

    # 샘플레이트 변환 → 16kHz
    if framerate != _STT_SAMPLE_RATE:
        from math import gcd
        g   = gcd(framerate, _STT_SAMPLE_RATE)
        up  = _STT_SAMPLE_RATE // g
        down = framerate // g
        print(f"[WAV] Resampling {framerate} Hz → {_STT_SAMPLE_RATE} Hz "
              f"(up={up}, down={down})")
        arr = resample_poly(arr, up=up, down=down)

    pcm = np.clip(arr, -32768, 32767).astype(np.int16).tobytes()
    print(f"[WAV] 16kHz PCM size: {len(pcm)} bytes "
          f"({len(pcm)/(_STT_SAMPLE_RATE*2):.2f}s)")
    return pcm


def wav_audio_generator(pcm_16k: bytes):
    """16kHz PCM을 4000-byte 청크로 yield하는 generator"""
    total = len(pcm_16k)
    sent  = 0
    while sent < total:
        chunk = pcm_16k[sent:sent + _STT_CHUNK_SIZE]
        # 마지막 청크가 4000 bytes 미만이면 0으로 패딩
        if len(chunk) < _STT_CHUNK_SIZE:
            chunk = chunk + b"\x00" * (_STT_CHUNK_SIZE - len(chunk))
        yield chunk
        sent += _STT_CHUNK_SIZE


def main():
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "wav/ramp_n1_0.wav"

    print("=" * 60)
    print("GoogleSTTV2 WAV 테스트")
    print("=" * 60)

    # 1. WAV → 16kHz PCM 변환
    pcm_16k = load_wav_as_16k_pcm(wav_path)

    # 2. STT 인스턴스 생성 및 초기화
    stt = create_stt()
    stt.initialize()

    # 3. 스트리밍 인식 시작
    print("\n[TEST] 스트리밍 STT 시작...")
    audio_gen = wav_audio_generator(pcm_16k)
    stt.start_streaming(audio_gen)

    # 4. 결과 대기 (최대 30초)
    timeout = 30.0
    success, transcript = stt.wait_for_result(timeout=timeout)
    stt.finalize()

    # 5. 결과 출력
    print("\n" + "=" * 60)
    if success:
        if transcript == "non_voice":
            print("[RESULT] 음성 미감지 (non_voice)")
            print("[RESULT] STT 인식 실패 — WAV 파일에 음성이 없거나 인식 불가")
        else:
            print(f"[RESULT] 인식 성공: '{transcript}'")
    else:
        print(f"[RESULT] 실패: {transcript}")
    print("=" * 60)

    return 0 if (success and transcript != "non_voice") else 1


if __name__ == "__main__":
    sys.exit(main())

"""
test_stt_athena.py — WAV 파일로 STT(server VAD) → Athena LLM 전체 흐름 테스트

흐름:
  WAV 파일 로드 → 16kHz PCM 변환
  → GoogleSTTV2 스트리밍 (server VAD EOS 감지)
  → Athena #start (세션 초기화)
  → Athena query (STT transcript 전달)
  → Athena #end (세션 종료)

사용법:
    python test_stt_athena.py [wav_file]

기본값: wav/ramp_n1_0.wav
"""
import sys
import uuid
import wave
import numpy as np
from scipy.signal import resample_poly
from math import gcd

import config as cfg
from stt import create_stt
from athena import AthenaClient

_STT_CHUNK_SIZE  = 4000   # bytes — 125ms at 16kHz 16-bit
_STT_SAMPLE_RATE = 16000


# ── WAV 로더 ──────────────────────────────────────────────────────────────

def load_wav_as_16k_pcm(wav_path: str) -> bytes:
    """WAV 파일 → 16kHz 16-bit signed mono PCM bytes"""
    with wave.open(wav_path, "rb") as wf:
        channels  = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes   = wf.getnframes()
        raw_data  = wf.readframes(nframes)

    print(f"[WAV] {wav_path}")
    print(f"      channels={channels}, sampwidth={sampwidth}B, "
          f"framerate={framerate}Hz, duration={nframes/framerate:.2f}s")

    # sampwidth 변환 → 16-bit signed
    if sampwidth == 1:
        import audioop
        raw_data = audioop.bias(raw_data, 1, -128)
        raw_data = audioop.lin2lin(raw_data, 1, 2)
    elif sampwidth == 2:
        pass
    elif sampwidth == 3:
        arr24 = np.frombuffer(raw_data, dtype=np.uint8).reshape(-1, 3)
        arr16 = (arr24[:, 1].astype(np.int16) | (arr24[:, 2].astype(np.int16) << 8))
        raw_data = arr16.tobytes()
    else:
        raise ValueError(f"Unsupported sampwidth: {sampwidth}")

    arr = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)

    # 스테레오 → 모노
    if channels == 2:
        arr = arr.reshape(-1, 2).mean(axis=1)

    # 샘플레이트 → 16kHz
    if framerate != _STT_SAMPLE_RATE:
        g    = gcd(framerate, _STT_SAMPLE_RATE)
        up   = _STT_SAMPLE_RATE // g
        down = framerate // g
        print(f"[WAV] Resampling {framerate}Hz → {_STT_SAMPLE_RATE}Hz "
              f"(up={up}, down={down})")
        arr = resample_poly(arr, up=up, down=down)

    pcm = np.clip(arr, -32768, 32767).astype(np.int16).tobytes()
    print(f"[WAV] 16kHz PCM: {len(pcm)} bytes ({len(pcm)/(_STT_SAMPLE_RATE*2):.2f}s)")
    return pcm


def wav_audio_generator(pcm_16k: bytes):
    """16kHz PCM → 4000-byte 청크 generator"""
    sent = 0
    while sent < len(pcm_16k):
        chunk = pcm_16k[sent:sent + _STT_CHUNK_SIZE]
        if len(chunk) < _STT_CHUNK_SIZE:
            chunk = chunk + b"\x00" * (_STT_CHUNK_SIZE - len(chunk))
        yield chunk
        sent += _STT_CHUNK_SIZE


# ── STT ───────────────────────────────────────────────────────────────────

def run_stt(wav_path: str) -> tuple[bool, str]:
    """WAV → STT 실행, (success, transcript) 반환"""
    pcm_16k = load_wav_as_16k_pcm(wav_path)

    stt = create_stt()
    stt.initialize()

    print("\n[STT] 스트리밍 인식 시작...")
    stt.start_streaming(wav_audio_generator(pcm_16k))
    success, transcript = stt.wait_for_result(timeout=30.0)
    stt.finalize()

    return success, transcript


# ── Athena ────────────────────────────────────────────────────────────────

def build_athena_client() -> AthenaClient:
    return AthenaClient(
        api_url       = cfg.ATHENA_SITE,
        auth_token    = cfg.ATHENA_AUTH,
        users_id      = int(cfg.ATHENA_USER_ID),
        chat_rooms_id = int(cfg.ATHENA_CHAT_ROOMS_ID),
        scenarios_id  = cfg.ATHENA_SCENARIOS_ID,
    )


def run_athena(transcript: str) -> list[dict]:
    """Athena start → query → end, 이벤트 목록 반환"""
    client = build_athena_client()
    uui    = str(uuid.uuid4())

    print(f"\n[Athena] 세션 시작 (uui={uui})")
    thread_id = client.start(uui=uui, voc_types=[])
    if not thread_id:
        return [{"type": "error", "text": "Athena start 실패 — thread_id 없음"}]

    print(f"[Athena] query 전송: '{transcript}'")
    events = client.query_sync(transcript)

    print(f"\n[Athena] 세션 종료")
    client.end(uui=uui)

    return events


# ── main ─────────────────────────────────────────────────────────────────

def main():
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "wav/ramp_n1_0.wav"

    print("=" * 60)
    print("STT → Athena 통합 테스트")
    print("=" * 60)

    # ── STEP 1: STT ───────────────────────────────────────────────────
    print("\n[STEP 1] STT")
    print("-" * 40)
    stt_ok, transcript = run_stt(wav_path)

    print(f"\n[STT 결과] success={stt_ok}, transcript='{transcript}'")

    if not stt_ok:
        print("[ABORT] STT 실패 — Athena 호출 건너뜀")
        return 1

    if transcript == "non_voice":
        print("[ABORT] 음성 미감지 — Athena 호출 건너뜀")
        return 1

    if transcript == "timeout":
        print("[ABORT] STT 타임아웃 — Athena 호출 건너뜀")
        return 1

    # ── STEP 2: Athena ────────────────────────────────────────────────
    if not cfg.athena_configured():
        print("\n[STEP 2] Athena — 환경변수 미설정, 건너뜀")
        print("         .env에 ATHENA_SITE / ATHENA_AUTH / ATHENA_USER_ID /")
        print("         ATHENA_CHAT_ROOMS_ID / ATHENA_SCENARIOS_ID 설정 필요")
        return 0

    print("\n[STEP 2] Athena LLM 호출")
    print("-" * 40)
    events = run_athena(transcript)

    # ── 최종 결과 출력 ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("최종 결과")
    print("=" * 60)
    print(f"  STT 인식: '{transcript}'")
    print(f"  Athena 이벤트 수: {len(events)}")
    for i, ev in enumerate(events, 1):
        ev_type = ev.get("type", "?")
        ev_text = ev.get("text", "")
        if ev_type == "command":
            print(f"  [{i}] {ev_type}: text='{ev_text}' | command={ev.get('command')} dest={ev.get('dest_number')}")
        else:
            print(f"  [{i}] {ev_type}: '{ev_text}'")
    print("=" * 60)

    # reply 이벤트가 1개 이상이면 성공
    replies = [e for e in events if e.get("type") == "reply"]
    return 0 if replies else 1


if __name__ == "__main__":
    sys.exit(main())

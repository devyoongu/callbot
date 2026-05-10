"""
test_stt_dump.py — wav/_test_set/ 의 5개 wav 일괄 STT 회귀 테스트

목적: SIP/RTP 없이 callbot 내부에서 STT 인식률 변동을 빠르게 측정.
e2e 테스트가 1통화 ~100s 걸리는 것에 비해 초당 5턴 평가 가능.

입력: callbot/wav/_test_set/manifest.json + q1.wav~q5.wav
   - wav 파일은 DEBUG_DUMP_RTP=1 로 한 번 캡처해 둔 STT 입력 (STT 가
     실제로 본 데이터 = audio_source.py 의 _dump_out 누적)
   - manifest 의 expected 텍스트와 비교해 CER (Levenshtein 기반) 계산

출력 (stdout):
  Q1: 'final transcript' [audio=Nms, vad_b=N, vad_e=N] CER=NN.N% (TRUNCATED?)
  ...
  AVG accuracy=NN.N% (1 - avg CER), avg audio=Nms

사용:
  python test_stt_dump.py                    # 모든 turn
  python test_stt_dump.py q1                 # 특정 turn 만
  REPEAT=3 python test_stt_dump.py           # 같은 set 3회 반복 평균
"""
import os
import sys
import json
import time
import wave
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

from stt import create_stt, GoogleSTTV2

_HERE         = Path(__file__).parent.resolve()
_TEST_DIR     = _HERE / "wav" / "_test_set"
_MANIFEST     = _TEST_DIR / "manifest.json"
_CHUNK_MS     = 125    # STT 입력 단위 (audio_source.py 와 동일)
_RESULT_TO    = 30.0   # final 대기 timeout
# 청크 간격 sleep — 실시간 스트리밍 흉내 (Google STT 가 너무 빠른 입력에
# 다른 finalization 을 보낼 수 있어 e2e 와 일관성 유지). 0 이면 burst.
_PACING_SEC   = float(os.environ.get("PACING_SEC", "0.125"))


def load_wav(path: Path, target_sr: int) -> bytes:
    """wav 를 mono 16-bit signed PCM 으로 로드. 필요 시 리샘플."""
    with wave.open(str(path), "rb") as wf:
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if ch == 2:
        arr = arr.reshape(-1, 2).mean(axis=1)
    if sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g    = gcd(sr, target_sr)
        up   = target_sr // g
        down = sr // g
        arr = resample_poly(arr, up=up, down=down)
    return np.clip(arr, -32768, 32767).astype(np.int16).tobytes()


def chunked_generator(pcm: bytes, chunk_size: int, pacing_sec: float):
    """pcm 을 chunk_size 바이트씩 yield. pacing_sec > 0 이면 실시간 시뮬레이션."""
    sent = 0
    total = len(pcm)
    while sent < total:
        chunk = pcm[sent:sent + chunk_size]
        if len(chunk) < chunk_size:
            chunk = chunk + b"\x00" * (chunk_size - len(chunk))
        yield chunk
        sent += chunk_size
        if pacing_sec > 0:
            time.sleep(pacing_sec)


def cer(expected: str, recognized: str) -> float:
    """문자 (음절 codepoint) 단위 Levenshtein 거리 / max(len)."""
    a = list(expected)
    b = list(recognized)
    if not a:
        return 0.0 if not b else 1.0
    if not b:
        return 1.0
    # 1D rolling DP
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j-1] + 1,      # insertion
                prev[j-1] + (0 if ca == cb else 1),  # subst
            )
        prev = curr
    dist = prev[-1]
    return dist / max(len(a), len(b))


def run_one(stt: GoogleSTTV2, wav_path: Path, expected: str) -> Dict:
    """단일 wav → STT → 결과 dict."""
    pcm = load_wav(wav_path, stt.sample_rate)
    chunk_bytes = int(stt.sample_rate * (_CHUNK_MS / 1000.0)) * 2
    audio_dur_ms = (len(pcm) / 2) / stt.sample_rate * 1000

    stt.initialize()
    gen = chunked_generator(pcm, chunk_bytes, _PACING_SEC)
    t0 = time.time()
    stt.start_streaming(gen)
    success, transcript = stt.wait_for_result(timeout=_RESULT_TO)
    elapsed = (time.time() - t0) * 1000
    stt.finalize()

    if not success or transcript == "non_voice":
        recognized = ""
    else:
        recognized = transcript

    err = cer(expected, recognized)
    return {
        "wav": wav_path.name,
        "expected": expected,
        "recognized": recognized,
        "cer": err,
        "accuracy": 1.0 - err,
        "audio_ms": audio_dur_ms,
        "elapsed_ms": elapsed,
        "vad_begin": stt._vad_begin_count,
        "vad_end": stt._vad_end_count,
        "yielded_chunks": stt._yielded_chunk_count,
        "truncated": len(recognized) < len(expected) * 0.7,  # 70% 미만이면 cut-off 의심
    }


def main():
    if not _MANIFEST.exists():
        print(f"[ERR] manifest 없음: {_MANIFEST}", file=sys.stderr)
        return 1
    items: List[Dict] = json.loads(_MANIFEST.read_text())
    filt = sys.argv[1] if len(sys.argv) > 1 else None
    if filt:
        items = [it for it in items if it["wav"].startswith(filt)]
    repeat = int(os.environ.get("REPEAT", "1"))

    print("=" * 78)
    print(f"STT 단위 테스트 — {len(items)} turns × {repeat} repeat, pacing={_PACING_SEC}s")
    print("=" * 78)

    stt = create_stt()
    print(f"[STT] model={stt.model}, sample_rate={stt.sample_rate}, region={stt.region}")
    print()

    results: List[Dict] = []
    for r in range(repeat):
        for it in items:
            wav_path = _TEST_DIR / it["wav"]
            if not wav_path.exists():
                print(f"[SKIP] {wav_path.name} 없음")
                continue
            expected = it["expected"]
            try:
                res = run_one(stt, wav_path, expected)
            except Exception as e:
                print(f"[FAIL] {wav_path.name}: {type(e).__name__}: {e}")
                continue
            res["round"] = r + 1
            results.append(res)
            trunc = " TRUNC" if res["truncated"] else ""
            print(
                f"R{r+1} {res['wav']}: '{res['recognized']}'\n"
                f"  expected: '{expected}'\n"
                f"  CER={res['cer']*100:.1f}%, accuracy={res['accuracy']*100:.1f}%, "
                f"audio={res['audio_ms']:.0f}ms, elapsed={res['elapsed_ms']:.0f}ms, "
                f"vad_b={res['vad_begin']}, vad_e={res['vad_end']}{trunc}"
            )

    if not results:
        print("[ERR] 결과 없음")
        return 1

    print()
    print("-" * 78)
    avg_acc  = sum(r["accuracy"] for r in results) / len(results)
    avg_cer  = sum(r["cer"]      for r in results) / len(results)
    avg_audio = sum(r["audio_ms"] for r in results) / len(results)
    avg_elap = sum(r["elapsed_ms"] for r in results) / len(results)
    n_trunc  = sum(1 for r in results if r["truncated"])
    print(f"avg accuracy={avg_acc*100:.1f}% (CER={avg_cer*100:.1f}%) over n={len(results)}")
    print(f"avg audio={avg_audio:.0f}ms, avg elapsed={avg_elap:.0f}ms, "
          f"truncated={n_trunc}/{len(results)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

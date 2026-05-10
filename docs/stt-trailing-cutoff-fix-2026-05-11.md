# STT 뒤쪽 잘림 (trailing cut-off) critical fix (2026-05-11)

## 증상

5턴 e2e 회귀에서 발화 후반부가 통째로 잘리는 현상이 반복.

```
사용자: "파견보안관제와 원격관제의 차이는 무엇인가요"
STT:    "파견 보안관제요."  (24자 → 7자, 70% 손실)

사용자: "중소기업도 인증이 꼭 필요한가요"
STT:    "중소기업도"  (16자 → 5자, 70% 손실)
```

같은 코드 / 같은 5턴 회귀에서 run-by-run 변동도 큼:
- Run A: 75.6%
- Run B: 80.1%
- Run C: 67.9%

사용자 의견: "오인식은 그럴 수 있다 — **뒤쪽 잘림이 critical**".

---

## 근본 원인

`stt.py` 의 `_run_streaming` 이 `client.streaming_recognize()` 를 사용. chirp_3
모델은 streaming 모드에서 **발화 중간 짧은 휴지에도 `is_final` 을 발행**한다.

기존 코드가 첫 `is_final` 을 즉시 채택하고 종료해 발화 후반부가 STT 에 도달하기
전에 결과 확정됨. chirp_3 의 EOS 판정은 모델 내부 로직이라 client config 으로는
우회 불가.

### 시도해본 client-side 우회 (모두 실패)

| 접근 | 결과 |
|---|---|
| `voice_activity_timeout.speech_end_timeout=2s` | 효과 없음, 일부 turn 500 에러 |
| `enable_voice_activity_events=False` | 효과 없음 |
| `chirp_2` (us-central1) 로 모델 swap | -3.8pp (오히려 후퇴) |
| `long`, `short` 모델 | 더 심하게 잘림 |
| Trailing silence 1.5s padding | 효과 없음 |
| Multi-stream restart (첫 is_final 후 새 stream) | unit test +1.8pp / e2e -7pp 회귀 |

### 핵심 발견 — sync 와 streaming 의 결정적 차이

같은 wav 를 `client.recognize()` (sync, non-streaming) 로 보내면 truncation 없음:

```
streaming chirp_3:  "파견 보안관제요." (21.7%)
sync     chirp_3:   "파견보안관제, 원격관제 차이는 무엇인가요?" (~97%)

streaming chirp_3:  "중소기업도" (29.4%)
sync     chirp_3:   "중소기업도 인증이 꼭 필요한가요?" (100%)
```

sync 는 audio 전체를 한 번에 평가하므로 mid-utterance EOS 판정이 없다.

---

## 해결 방법

### 핵심 변경 — `_run_streaming` 을 sync recognize 기반으로 재작성

**Phase 1 — `_buffer_audio_with_vad`**
- audio_source 의 청크를 buffer 에 누적
- RMS-only VAD 로 발화/silence 감지 (`SILENCE_THRESHOLD=5` × 125ms = 625ms)
- silence threshold 도달 시 EOS 커밋
- EOS 후 추가로 4 청크 (500ms) 더 버퍼해 자연 종료 silence 까지 포함
  → chirp_3 sync 가 끝부분 음절 인식 안정성 ↑

**Phase 2 — sync recognize 1회 호출**
- 누적된 audio 를 `client.recognize()` 1회 호출
- `response.results[*].alternatives[0].transcript` 들을 join 해 final 로 채택

```python
audio_buf = bytearray()
self._buffer_audio_with_vad(audio_buf)  # ← VAD-driven buffering

if self.speech_started and len(audio_buf):
    request = cloud_speech.RecognizeRequest(
        recognizer=recognizer_path,
        config=recognition_config,    # adaptation 포함 그대로
        content=bytes(audio_buf),
    )
    response = self.client.recognize(request=request, timeout=30.0)
    transcripts = [r.alternatives[0].transcript.strip()
                   for r in response.results if r.alternatives]
    self._final_transcript = " ".join(transcripts) or "non_voice"
```

### 부수적 변경 — webrtcvad 의존 제거

`webrtcvad` 가 `setuptools 82+` 에서 import 실패 (`pkg_resources` 제거됨).
RMS-only 판정으로 단순화. callbot 환경 (TTS-as-mic + telephony codec) 에서
잡음 < 192, 발화 > 1886 의 margin 이 커서 RMS=200 threshold 만으로 충분.

### Dead code 정리

- `webrtcvad` import block 제거
- `_client_vad_check` 메서드 제거 (RMS-only 가 inline)
- `_handle_vad_event` 는 `_handle_vad_event_unused` 로 rename (streaming 회귀 시
  참조 보존용)

---

## 측정

### 단위 테스트 (`test_stt_dump.py`, n=15 across 3 repeats)

| 메트릭 | baseline (streaming) | new (sync) | 변화 |
|---|---|---|---|
| 평균 인식률 | 75.6% | **93.4%** | **+17.8pp** |
| 평균 CER | 24.4% | **6.6%** | -17.8pp |
| Truncated count | 6/15 | **0/15** | — |

3회 반복 결과가 **완전 결정적** (동일 wav 의 sync recognize 는 같은 결과 반환).

### e2e 회귀 (5턴, 메트릭 패널)

| 메트릭 | baseline | new | 변화 |
|---|---|---|---|
| 평균 인식률 | 67-75% (변동 큼) | **92.1%** | **+17-25pp** |
| 평균 CER | ~25% | **7.9%** | -17pp |
| STT 평균 latency | 570ms | 2.42s | +1.85s |

per-turn:

| Q | baseline | new | 변화 |
|---|---|---|---|
| 1 ("파견보안관제와 원격관제의 차이는 무엇인가요") | 21.7% | **92.0%** | **+70.3pp** |
| 2 ("중소기업도 인증이 꼭 필요한가요") | 29.4% | **94.4%** | **+65.0pp** |
| 3 ("제로트러스트 보안이란 무엇인가요") | 58.8% | 83.3% | +24.5pp |
| 4 ("랜섬웨어 사전 예방 방법은 무엇인가요") | 95.2% | 95.2% | ±0 |
| 5 ("얼굴인식기는 실외에도 설치 가능한가요") | 70.0% | **95.2%** | **+25.2pp** |

### Trade-off

- **STT latency: 570ms → 2.42s** (+1.85s/turn) — 이후 commit 2bec88e 의
  VAD 튜닝 (`SILENCE_THRESHOLD` 5→4, `EOS_TRAIL_CHUNKS` 4→2) 으로 -375ms
  회복, 단위 테스트 elapsed 9063→8688ms (인식률 93.4% 그대로 유지).
  - 잔여 floor: sync API call 2.0-2.7s (us 리전, 줄일 수 없음) + VAD
    silence 500ms + trail buffer 250ms.
- **인식률 +17-25pp**, 트레일링 잘림 0/15
- 사용자 우선순위 (트레일링 잘림이 critical) 에 부합

---

## 호환성

- `call_handler.py` 의 polling 인터페이스 변경 없음 (`get_speech_started`,
  `get_and_consume_eos`, `get_final_transcript`, `_result_event`)
- `server.py` SSE 가 파싱하는 라인 (TTS enqueue, STT result, …) 영향 없음
- `audio_source.py` 의 chunk 단위 (16kHz × 125ms × 2 bytes) 그대로
- `DEBUG_DUMP_RTP=1` 영향 없음 (dump 는 audio_source 측 로직)
- `stt_phrases.py` adaptation phrase set 그대로 사용 (RecognitionConfig.adaptation
  은 sync 도 동일 지원)

---

## 단위 테스트 인프라

`test_stt_dump.py` + `wav/_test_set/` —
DEBUG_DUMP_RTP 로 캡처한 5개 wav (manifest.json 의 expected 텍스트와 매핑) 을
SIP/RTP 없이 callbot 내부 STT 로 직접 평가. 결정적이라 코드 변경 효과를 즉시
측정 가능.

```bash
# 1회 실행
python test_stt_dump.py

# 3회 반복 평균
REPEAT=3 python test_stt_dump.py

# 다른 모델 시험
GCP_STT_MODEL=chirp_2 python test_stt_dump.py

# 특정 turn 만
python test_stt_dump.py q1
```

향후 다른 fix 시도 (Google 신모델 출시 등) 시 회귀 테스트로 재활용.

---

## 관련 commit

- `517552b` chirp_3 + speech adaptation 으로 STT 인식률 +12.2pp
- `ed2334b` chirp_3 의 STT latency 메트릭 n=0 fix
- `2ab6c49` multi-stream restart + 단위 테스트 (이후 production 회귀로 stt.py 만 revert)
- `50d3ca9` multi-stream restart revert
- `e0aac80` chirp_2 모델 옵션 추가
- `0325294` **streaming → sync recognize 전환 — 뒤쪽 잘림 critical fix (+25pp)** ← 본 fix

---

## 백로그

- chirp_3 의 streaming 모드 자체 개선이 Google 측에서 도입되면 sync 로의 switch
  비용 (latency +1.85s) 을 줄이기 위해 streaming 회귀 검토 가능 — 이 때 본 문서
  의 단위 테스트로 즉시 비교 가능
- TTS playback 직후 RTP queue 에 누적된 echo 가 다음 turn 의 silence 감지를
  지연시키는 케이스 별도 관찰 필요 (turn 5 에서 종종 non_voice timeout)

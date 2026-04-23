# STT 인식 불량 원인 분석 및 해결 (2026-04-23)

## 증상

실제 전화 통화에서 2~3번 발화해야 겨우 몇 단어만 인식되는 현상.

```
[STT] Interim: '거기'
[STT] VAD: Speech END — real speech, stopping generator   ← 조기 종료
[STT] Final: '거기'   ← 전체 문장이 아닌 단어 하나만 인식
```

사용자가 "거기 위치 어디야"라고 발화했지만 "거기"만 인식됨.

---

## 근본 원인

### 버그 1: `SPEECH_ACTIVITY_END` 시 generator 조기 종료

서버 VAD(`SPEECH_ACTIVITY_END`)가 발화하면 `_process_audio = False`로 설정하여 audio generator를 즉시 멈추는 방식을 사용했다.

문제는 Google telephony 모델의 서버 VAD가 **단어 사이 300ms 미만의 짧은 쉼에도 END를 발화**한다는 것. 사용자가 계속 말하는 중임에도 generator가 종료되어 이후 발화가 잘린다.

**증거**: 로그에서 `SPEECH_ACTIVITY_END` 직후 두 번째 `SPEECH_ACTIVITY_BEGIN`이 등장 — 사용자가 계속 발화 중이었음을 Google 자신이 인정.

### 버그 2: `SPEECH_ACTIVITY_BEGIN` 시 `_has_interim_content` 리셋

`_handle_vad_event()`의 BEGIN 핸들러에서 `self._has_interim_content = False`를 수행했다.

이 리셋이 문장 중간 쉼 시나리오에서 치명적인 연쇄를 유발:

```
BEGIN → Interim "거기" → _has_interim_content = True
END   → (버그 1 수정 후) 계속 스트리밍
BEGIN → _has_interim_content = False  ← 리셋!
"위치 어디야" 발화 (interim 없거나 미미)
END   → _has_interim_content=False → client VAD echo reset → EOS 발화 안 됨
→ 15초 timeout → non_voice
```

---

## 해결 방법

### 수정 1: `SPEECH_ACTIVITY_END`에서 generator를 멈추지 않음

`stt.py/_handle_vad_event()`:

```python
elif event_type in (SPEECH_ACTIVITY_END, END_OF_SINGLE_UTTERANCE):
    if self._has_interim_content:
        # generator 멈추지 않음 — client VAD가 625ms 침묵 후 EOS 담당
        print("[STT] VAD: Speech END — real speech, continuing (client VAD handles EOS)")
    else:
        # interim 없음 = echo/noise → speech_started 리셋 후 계속 청취
        print("[STT] VAD: Speech END — no content (echo/noise), continuing")
        self.speech_started = False
        self.speech_started_time = None
```

EOS 판단을 client VAD(625ms 침묵 감지)에게 위임. gRPC clean EOF 시 Google이 버퍼된 오디오를 처리하여 `is_final`을 반환하는 메커니즘은 그대로 유지.

### 수정 2: `SPEECH_ACTIVITY_BEGIN` 시 `_has_interim_content` 리셋 제거

```python
if event_type == SpeechEventType.SPEECH_ACTIVITY_BEGIN:
    print("[STT] VAD: Speech BEGIN")
    self.speech_started      = True
    self.speech_started_time = time.time()
    # _has_interim_content 리셋 안 함:
    # 문장 중간 BEGIN 반복 시 이전 interim 상태 보존 필요.
```

한 번 `True`로 설정된 `_has_interim_content`는 세션 내내 유지되어,
client VAD가 625ms 침묵 감지 시 올바르게 EOS를 발화할 수 있게 됨.

---

## 수정 후 정상 동작 흐름

```
BEGIN
  → Interim: "거기"  →  _has_interim_content = True
END   (짧은 쉼)      →  "continuing" — generator 유지
BEGIN (계속 발화)    →  _has_interim_content 유지 (리셋 안 함)
  → Interim: "위치"  →  _has_interim_content 여전히 True
END   (발화 완료)    →  "continuing"
client VAD: 625ms 침묵 + _has_interim_content=True → EOS → generator break
Google: is_final → "거기 위치 어디야"  ✓
```

---

## 검증

### 자동 테스트 스크립트

기존 `test_stt_wav.py`는 `CallAudioSource`를 우회하여 실제 통화 경로를 테스트하지 못했다.
신규 `test_stt_pipeline.py`는 실제 통화와 동일한 경로를 사용:

```
TTS → 8kHz 8bit unsigned → MockCall → CallAudioSource → GoogleSTTV2
```

```bash
cd callbot
venv/bin/python3 test_stt_pipeline.py              # 전체 케이스
venv/bin/python3 test_stt_pipeline.py "원하는 문장"  # 임의 문장
```

### 테스트 결과

| 케이스 | 입력 | 인식 결과 | 소요 |
|--------|------|---------|------|
| T1 | 거기 위치 어디야 | `거기 위치 어디야` | 1.9s |
| T2 | 예약을 취소하고 싶습니다 | `예약을 취소하고 싶습니다` | 1.8s |
| T3 | 안녕하세요 | `안녕하세요` | 1.5s |
| T4 | 진료 예약을 다음 주로 변경하고 싶은데요 | `주요 예약을 다음 주로 변경하고 싶은데요` | 1.9s |

4/4 통과, 실제 전화 통화에서도 전체 문장 인식 확인.

---

## 수정된 파일

| 파일 | 변경 내용 |
|------|---------|
| `callbot/stt.py` | `_handle_vad_event()`: END 시 generator 유지, BEGIN 시 `_has_interim_content` 리셋 제거 |
| `callbot/test_stt_pipeline.py` | 신규: 실제 통화 경로 전체를 테스트하는 통합 테스트 |

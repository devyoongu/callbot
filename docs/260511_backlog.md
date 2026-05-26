# Callbot 개선 백로그

이번 STT 작업 (2026-05-10 ~ 11) 이후 남은 개선 항목. 우선순위 순.
실측 출처가 있는 항목만 — 가설/추측은 제외.

---

## P0 — 사용자 체감에 큰 영향

### 1. STT latency 단축 (2.42s → 1.0s 목표)

**진행**: VAD 상수 튜닝으로 -375ms 단축 (commit 2bec88e, 2026-05-11).
- `SILENCE_THRESHOLD` 5→4 (-125ms), `EOS_TRAIL_CHUNKS` 4→2 (-250ms)
- 단위 테스트 (n=15): accuracy 93.4% 유지, elapsed 9063→8688ms, truncated 0/15
- `SILENCE_THRESHOLD=3` 시도는 q3 인식률 -11.9pp 회귀로 후퇴

**잔여**: 약 2.0s — 주된 floor 는 sync API call (us 리전, 2.0–2.7s).

**남은 아이디어**:
- (a) **TTS pre-roll 으로 체감 dead air 가리기 — 인프라만 보존, default 비활성**
  (2026-05-11). Athena 의 `meta_status` ("잠시만 기다려 주세요." 등) 와 멘트가
  중복되어 "네, 잠시만요. 잠시만 기다려 주세요." 가 매 turn 반복되는 어색함
  관찰 → `cfg.PREROLL_MESSAGE` default `""` (비활성). 인프라 (audio_source
  echo masking + `_eos_preroll` 콜백) 는 유지. 진짜 활성화를 위해서는 callbot
  측에서 Athena `meta_status` 를 client 멘트로 대체하는 디자인 (옵션 2) 가
  먼저 필요.
- (b) **chirp_3 의 `streaming` 회귀** — `GCP_STT_MODE=streaming` 옵션 추가됨
  (default sync). 단일 stream + is_final 누적 시도 (2026-05-11) 시 단위 테스트
  accuracy 75.6% / truncated 6/15 — sync 93.4% 대비 -17.8pp 회귀.
  근본 원인: 첫 `is_final` 후 audio 더 보내도 chirp_3 가 새 `is_final` 을
  추가 발행하지 않음 (turn 당 `is_final_count=1`). Google 측 모델 개선 (mid-utterance
  EOS 제어) 또는 새 streaming 모델 출시 시 같은 단위 테스트로 즉시 재비교 가능.
- (c) **asia 리전 recognizer** (Google 이 chirp_3 를 asia 출시 시) — sync API
  의 us 왕복 latency (~300-500ms) 단축 가능.

**측정 방법**: `test_stt_dump.py` 의 `avg elapsed` 비교 + 메트릭 패널의 `STT
평균 latency`.

---

### 2. Q3 ("제로트러스트 보안이란 무엇인가요") 인식률 83% → 90%+

**현황**: 5턴 e2e 에서 Q1/Q2/Q4/Q5 는 92~95% 도달했으나 Q3 만 83.3%. sync
`recognize()` 가 종종 "무엇인가요" → "무엇일까요" 로 어미 substitution.

**아이디어**:
- adaptation phrase set 에 "보안이란 무엇인가요" 같은 짧은 의문문 어미 추가.
  현재 `stt_phrases.py` 에 의문문 어미는 주석 처리됨 — 활성화 후 단위 테스트.

---

## P1 — 안정성/품질

### 3. 첫 turn (turn 0) RTP 큐 잡음으로 시작이 매끄럽지 않은 케이스

**현황**: 일부 통화에서 turn 0 이 시작될 때 `[AudioSource] First audio
received after 3279 reads` (3.3s 대기) 같이 RTP 도착이 늦거나, 직전 greeting
TTS 의 echo 가 RTP queue 에 쌓여 STT 가 비음성 인식으로 헷갈림.

**아이디어**:
- greeting TTS 종료 직후 250–500ms drain 후 STT 시작 (call_handler 의 turn 0
  분기에 sleep 추가)
- audio_source 의 leading-silence skip 로직을 turn 0 에 한해 더 보수적으로
  (예: 첫 5 청크 무시).

**측정**: `[AudioSource] First audio received after N reads` 로그 통계.

---

### 4. TTS playback 후 첫 chunk 의 echo — 해결됨

**조치** (2026-05-11): `TTSPipeline.last_chunk_played_at` 기록 + `CallAudioSource`
가 `tts_last_play_at` callable 받아 `last_play + _ECHO_MASK_SEC (200ms)` 이내
inbound RTP 를 silence 로 치환.

**효과** (e2e 5턴): 각 turn `echo_masked=107~138 chunks` (≈2-3s 봇 응답 echo
차단). audio_buf 크기가 이전 17-24s → 5-8s 로 감소 (sync API 부담 ↓). 매 turn
첫 STT 의 false-EOS non_voice 회귀 해결 — pre-roll TTS 활성화 가능.

---

### 5. 단위 테스트 set 다양화

**현황**: `wav/_test_set/q1.wav~q5.wav` 5개만 — 실제 통화 분포의 일부.

**아이디어**:
- DEBUG_DUMP_RTP=1 으로 떨어진 `wav/_dump/` 의 다양한 발화 길이/내용을
  `_test_set/` 로 추가 (짧은 단답 ("네", "아니요"), 모호한 발음, 도메인 외 질의).
- manifest.json 의 expected 텍스트는 사람이 청취 후 작성.

---

## P2 — 코드 hygiene / 운영

### 6. 미사용 추가 recognizer (`callbot-telephony-ko`) 정리

**현황**: adaptation 검증 중 `projects/.../locations/global/recognizers/
callbot-telephony-ko` 를 생성했으나 default `_` 가 boost 미지원이라 사용 안
함. orphan 상태.

**조치**: `client.delete_recognizer(name=...)` 로 정리 또는 `chirp_3` (us) 도
같은 패턴으로 만들지 결정.

---

### 7. `wav/_dump/` 누적 정리 정책

**현황**: 현재 564 파일, 누적 ~13GB. 매 turn 마다 2개 wav (`turn_<ts>.wav`,
`turn_<ts>_<sr>k.wav`).

**아이디어**:
- 기본은 OFF, `DEBUG_DUMP_RTP=1` 일 때만 ON (현재 동작 그대로)
- audio_source 가 N 일 이상 오래된 dump 자동 삭제 (예: `find ... -mtime +7
  -delete`) — 매 통화 시작 시 cleanup 한 번.

---

### 8. `chirp_2` 모델 옵션 — 유지 vs 제거 결정

**현황**: e0aac80 에서 옵션으로 추가했으나 단위 테스트 71.8% < chirp_3 75-93%.
실 사용 안 함.

**조치**: 1–2주 후에도 사용처가 없으면 `MODEL_CONFIG` 에서 제거.

---

### 9. `_handle_vad_event_unused` 정리

**현황**: 0325294 에서 streaming 모드 dead code 의 일부 (server VAD handler).
streaming 회귀 시 참조용으로 보존.

**조치**: 1개월 후 sync 가 안정적이면 제거.

---

### 10. STT latency 메트릭의 의미 재정의

**현황**: 메트릭 패널의 "STT 평균 latency" 가 `serverEND→Final` (sync 의 경우
client-side EOS 커밋부터 sync API 응답까지) 을 표시. 새 수치 (2.42s) 는 이전
streaming 의 ~570ms 와 비교가 의미 없음.

**아이디어**:
- 라벨을 "STT EOS→Final" 또는 "STT API latency" 로 명확히
- 또는 두 metrics 분리: VAD silence detection time vs sync API call time

---

## P3 — 장기

### 11. Google STT 의 새로운 streaming 모드 출시 시 sync vs streaming 비교

`260511_stt-trailing-cutoff-fix.md` 의 단위 테스트로 즉시 비교 가능.
sync 의 +1.85s latency 를 줄일 수 있으면 streaming 회귀 검토.

---

### 12. Adaptation phrase set 의 production 로그 기반 자동 확장

**아이디어**:
- 매주 production 로그에서 STT 결과 텍스트의 빈출 미인식 단어/구를 추출
- 사람 검토 후 `stt_phrases.py` 에 추가
- 단위 테스트로 회귀 검증

---

### 13. 다중 recognizer 병렬 비교 (정확도 vs 비용)

같은 audio 를 chirp_3 + telephony 동시 보내고 길이/confidence 로 voting.
비용 2배지만 critical query (예: 금액, 주소, 이름) 에서 사용 고려.

---

## 측정 / 회귀 도구

- **단위 테스트**: `test_stt_dump.py` (REPEAT=N 으로 평균)
- **e2e 회귀**: web-rtc 의 5턴 multi-query — 메트릭 패널로 실시간 측정
- **로그 분석**: `[STT] Final`, `[STT] Turn summary`, `[STT] EOS triggered`
  라인을 grep — `260511_stt-trailing-cutoff-fix.md` 의 진단 로그 그대로

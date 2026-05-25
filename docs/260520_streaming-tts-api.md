# Streaming TTS API 연동 가이드

vLLM-Omni 기반 Qwen3-TTS 서버를 다른 클라이언트에서 연동할 때 참고하는 인터페이스 문서입니다. 실시간 음성 합성(스트리밍 PCM)과 voice 관리 API 를 포함합니다.

---

## 1. 서비스 개요

| 항목 | 값 |
|---|---|
| Base URL | `http://172.31.79.203:30000` (RTX 3090, 사내망) |
| 모델 | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` |
| API 스타일 | OpenAI Audio API 호환 (확장 필드 포함) |
| 인증 | 없음 (사내망 한정) |
| 아키텍처 | 2-stage (Thinker → Code2Wav), `--omni` 모드 |
| 오디오 출력 | **24 kHz / mono / int16 little-endian PCM** (고정) |

성능 참고치 (RTX 3090, OMP_NUM_THREADS=16, 한국어 60자 입력):

| 지표 | 값 |
|---|---|
| TTFA (Time To First Audio) | ~460 ms |
| RTF (생성 속도 / 재생 시간) | ~1.2x |
| 동시 요청 안정 한계 | 6 req 이상 시 stage 간 max_num_seqs 조정 필요 |

---

## 2. 엔드포인트 목록

| Method | Path | 용도 |
|---|---|---|
| `POST` | `/v1/audio/speech` | TTS 합성 (단일 텍스트, 스트리밍 지원) |
| `POST` | `/v1/audio/speech/batch` | 다중 텍스트 일괄 합성 |
| `WS`   | `/v1/audio/speech/stream` | 텍스트 점진 입력 스트리밍 (WebSocket) |
| `GET`  | `/v1/audio/voices` | 등록된 voice 목록 |
| `POST` | `/v1/audio/voices` | voice 신규 등록 (오디오 샘플 or 임베딩) |
| `DELETE` | `/v1/audio/voices/{name}` | voice 삭제 |
| `GET`  | `/health` | 헬스체크 |

---

## 3. 등록된 Voice

서버 기동 시 `wav/voices.json` 기반으로 `entrypoint.sh` 가 자동 등록합니다.

```bash
curl http://172.31.79.203:30000/v1/audio/voices
```

응답:
```json
{
  "voices": ["femail_achernar", "mail_achird"],
  "uploaded_voices": [
    {
      "name": "femail_achernar",
      "consent": "agreed",
      "created_at": 1746000000,
      "file_size": 245760,
      "mime_type": "audio/wav",
      "embedding_source": "audio",
      "embedding_dim": null
    },
    { "name": "mail_achird", "...": "..." }
  ]
}
```

---

## 4. POST `/v1/audio/speech` — 메인 합성 엔드포인트

### 4.1 Request Body (JSON)

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `input` | string | ✅ | 합성할 텍스트 |
| `model` | string | ⛔ | 모델 ID (생략 가능, 서버 단일 모델) |
| `voice` | string | ✅ | 등록된 voice 이름 (예: `femail_achernar`) |
| `response_format` | enum | ⛔ | `wav` (기본) / `pcm` / `flac` / `mp3` / `aac` / `opus` |
| `stream` | bool | ⛔ | `true` 시 청크 단위 전송 (기본 `false`) |
| `speed` | float | ⛔ | 0.25 ~ 4.0 (기본 1.0). **streaming 모드에선 1.0 고정** |
| `instructions` | string | ⛔ | 보이스 스타일/감정 지시 (Qwen3-TTS `instruct` 매핑) |
| `language` | string | ⛔ | `Chinese` / `English` / `Auto` 등 |
| `task_type` | enum | ⛔ | `CustomVoice` / `VoiceDesign` / `Base` |
| `ref_audio` | string | ⛔ | 즉석 voice clone 용 (URL/base64/file URI) |
| `ref_text` | string | ⛔ | `ref_audio` 의 텍스트 |
| `speaker_embedding` | float[] | ⛔ | 사전계산 임베딩 (1024-dim @0.6B). `ref_audio` 와 배타 |
| `max_new_tokens` | int | ⛔ | 생성 토큰 상한 |
| `initial_codec_chunk_frames` | int | ⛔ | 초기 청크 프레임 수 오버라이드 (지연 튜닝) |

**제약**
- `stream=true` 면 `response_format` 은 `pcm` 또는 `wav` 만 허용
- `stream=true` 일 때 `speed != 1.0` 이면 400 에러

### 4.2 응답 모드

| 모드 | Content-Type | 본문 |
|---|---|---|
| 비스트리밍 (`stream=false`) | `audio/wav` 등 | 인코딩 완료된 단일 바이너리 |
| 스트리밍 (`stream=true`, `pcm`) | `audio/pcm` | 24kHz mono int16 raw PCM, chunked transfer |
| 스트리밍 (`stream=true`, `wav`) | `audio/wav` | WAV 헤더 + PCM chunked |

### 4.3 예시 — curl

**저지연 PCM 스트리밍 (가장 권장):**
```bash
curl -N -X POST http://172.31.79.203:30000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "voice": "femail_achernar",
    "input": "안녕하세요. 스트리밍 합성 테스트입니다.",
    "response_format": "pcm",
    "stream": true
  }' \
  --output - | ffplay -f s16le -ar 24000 -ac 1 -nodisp -autoexit -
```

**비스트리밍 WAV 저장:**
```bash
curl -X POST http://172.31.79.203:30000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "voice": "femail_achernar",
    "input": "비스트리밍 모드 예시"
  }' \
  --output out.wav
```

### 4.4 예시 — Python (httpx + sounddevice)

`tests/test_streaming_kdnavien.py` 가 그대로 동작 예제입니다. 핵심 부분만 발췌:

```python
import httpx, queue, threading, time, sounddevice as sd

SERVER = "http://172.31.79.203:30000"
payload = {
    "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "voice": "femail_achernar",
    "input": "실시간 합성 테스트",
    "response_format": "pcm",
    "stream": True,
}

q: queue.Queue = queue.Queue(maxsize=64)

def fetch():
    leftover = bytearray()
    with httpx.stream("POST", f"{SERVER}/v1/audio/speech",
                      json=payload, timeout=None) as r:
        for chunk in r.iter_bytes():
            leftover.extend(chunk)
            aligned = (len(leftover) // 2) * 2  # int16 정렬
            if aligned:
                q.put(bytes(leftover[:aligned]))
                del leftover[:aligned]
    q.put(None)

threading.Thread(target=fetch, daemon=True).start()

with sd.RawOutputStream(samplerate=24000, channels=1,
                        dtype="int16", blocksize=2400) as stream:
    while (chunk := q.get()) is not None:
        stream.write(chunk)
```

**핵심 주의사항**
- PCM 청크는 항상 **int16 정렬(2바이트 단위)** 이어야 재생 가능. HTTP 청크 경계는 샘플 경계와 일치하지 않으므로 위처럼 leftover 버퍼를 둬야 함.
- 첫 청크 수신 직후 재생 시작해도 underrun 위험이 있으면 `blocksize` 를 50~100ms 로 두는 게 안전.

### 4.5 예시 — Node.js (fetch + Speaker)

```javascript
import fetch from "node-fetch";
import Speaker from "speaker";

const res = await fetch("http://172.31.79.203:30000/v1/audio/speech", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    voice: "femail_achernar",
    input: "Node.js 스트리밍 합성",
    response_format: "pcm",
    stream: true,
  }),
});

const speaker = new Speaker({ channels: 1, bitDepth: 16, sampleRate: 24000 });
res.body.pipe(speaker);
```

---

## 5. POST `/v1/audio/speech/batch` — 일괄 합성

여러 텍스트를 한 번에 처리. 비스트리밍 전용.

```json
{
  "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
  "voice": "femail_achernar",
  "response_format": "wav",
  "items": [
    { "input": "첫 번째 문장입니다." },
    { "input": "두 번째 문장입니다.", "voice": "mail_achird" },
    { "input": "세 번째 문장입니다." }
  ]
}
```

배치 최상위 필드는 기본값, `items[i]` 의 동일 필드가 우선합니다. 응답은 항목별 base64 오디오를 담은 JSON.

---

## 6. WebSocket `/v1/audio/speech/stream` — 점진 입력

LLM 출력처럼 텍스트가 점진적으로 도착하는 시나리오용. 문장 경계마다 합성된 오디오를 바이너리 프레임으로 push.

### Client → Server (JSON)

```json
// 1. 세션 설정 (최초 1회)
{
  "type": "session.config",
  "voice": "femail_achernar",
  "response_format": "wav",
  "language": "Auto"
}

// 2. 텍스트 청크 (반복)
{ "type": "input.text", "text": "안녕하세요. " }
{ "type": "input.text", "text": "오늘 날씨가 좋네요." }

// 3. 입력 종료
{ "type": "input.done" }
```

### Server → Client

| 메시지 | 타입 | 설명 |
|---|---|---|
| `{type: "audio.start", sentence_index, sentence_text, format}` | JSON | 문장 합성 시작 |
| `<binary frame>` | Bytes | 오디오 바이트 (`format` 에 따라 WAV/PCM) |
| `{type: "audio.done", sentence_index}` | JSON | 문장 종료 |
| `{type: "session.done", total_sentences}` | JSON | 전체 종료 |
| `{type: "error", message}` | JSON | 에러 |

**타임아웃**
- `session.config` 미수신 시 10초 후 종료
- 메시지 간 idle 30초 초과 시 종료

---

## 7. Voice 관리

### 7.1 등록 (오디오 샘플)

```bash
curl -X POST http://172.31.79.203:30000/v1/audio/voices \
  -F "name=my_voice" \
  -F "consent=agreed" \
  -F "ref_text=안녕하세요 음성 클로닝 샘플 입니다" \
  -F "audio_sample=@/path/to/sample.wav"
```

| 필드 | 필수 | 설명 |
|---|---|---|
| `name` | ✅ | voice 식별자 (이후 `voice=` 로 참조) |
| `consent` | ✅ | 동의 ID/문자열 (`agreed` 등) |
| `audio_sample` | ⚠️ | 오디오 파일 (max 10MB). `speaker_embedding` 과 배타 |
| `speaker_embedding` | ⚠️ | JSON 인코딩된 float 리스트 (0.6B → 1024-dim) |
| `ref_text` | ⛔ | 샘플 오디오의 전사 텍스트 (있으면 품질 향상) |

### 7.2 등록 (사전계산 임베딩)

오디오 인코딩을 클라이언트에서 미리 한 경우:

```bash
curl -X POST http://172.31.79.203:30000/v1/audio/voices \
  -F "name=embedded_voice" \
  -F "consent=agreed" \
  -F 'speaker_embedding=[0.123, -0.456, ...]'   # 1024 floats
```

이 방식은 즉시 사용 가능 (서버에서 추출 단계 생략).

### 7.3 삭제

```bash
curl -X DELETE http://172.31.79.203:30000/v1/audio/voices/my_voice
```

---

## 8. 에러 응답 포맷

OpenAI 스타일 에러:

```json
{
  "object": "error",
  "message": "Either 'audio_sample' or 'speaker_embedding' must be provided",
  "type": "BadRequestError",
  "code": 400
}
```

| HTTP | 의미 |
|---|---|
| 400 | 잘못된 요청 (스트리밍 + speed≠1, embedding 차원 불일치 등) |
| 404 | 모델이 Speech API 미지원 / voice 없음 |
| 500 | 합성 실패 (모델 OOM, 내부 에러) |

---

## 9. 클라이언트 구현 권장 사항

**저지연 콜봇 / 실시간 응답**
- `response_format=pcm`, `stream=true` 사용
- 첫 청크 도착 즉시 재생 시작 (TTFA 최소화)
- 24kHz int16 입력을 받는 오디오 디바이스 사용 (또는 리샘플)
- 네트워크 수신 ≠ 재생 속도 → **별도 스레드 + 큐** 패턴 필수
- HTTP 청크는 int16 경계 무시 → leftover 버퍼링 필수

**배치 / 파일 저장**
- `stream=false`, `response_format=wav` 또는 `mp3`
- `/v1/audio/speech/batch` 로 다수 항목 일괄 처리 시 RPS 절감

**LLM 연동 (TTLM → TTS)**
- WebSocket `/v1/audio/speech/stream` 사용
- LLM 토큰 단위로 `input.text` push, 서버가 문장 경계 자동 분할
- 문장 단위 audio 가 도착하므로 첫 문장 합성 지연만 보면 됨

**리소스 예측**
- 0.6B 모델: GPU 메모리 ~16GB (gpu-memory-utilization 0.9 기준)
- 동시 요청은 stage 간 `max_num_seqs` 균형이 중요 (6req 이상 시 튜닝 필요)

---

## 10. 헬스체크 및 모니터링

```bash
# 서버 살아있는지
curl http://172.31.79.203:30000/health

# 등록된 voice 확인
curl http://172.31.79.203:30000/v1/audio/voices

# Prometheus metrics (vllm 표준)
curl http://172.31.79.203:30000/metrics
```

---

## 11. 참고 자료

- 실제 동작 예제: `tests/test_streaming_kdnavien.py`
- 서버 배포 가이드: `guides/docker-deployment.md`
- vLLM-Omni 프로젝트 개요: `CLAUDE.md`
- OpenAI Speech API 명세 (호환 기준): https://platform.openai.com/docs/api-reference/audio

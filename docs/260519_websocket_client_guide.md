# Qwen3-ASR WebSocket 클라이언트 연동 가이드

`qwen-asr-serve-async` 가 제공하는 WebSocket 스트리밍 인터페이스로 STT 결과를
실시간으로 받기 위한 가이드입니다. 서버 운용 절차는 [SERVER_GUIDE.md](../SERVER_GUIDE.md)
를 참고하세요.

---

## 1. 한눈에 보기

| 항목 | 값 |
|---|---|
| 엔드포인트 | `ws://<host>:<port>/api/ws` (운영: `ws://172.31.79.202:30000/api/ws`) |
| 오디오 형식 | float32 little-endian PCM, **16,000 Hz**, mono, len % 4 == 0 |
| 오디오 전송 | WebSocket **바이너리** 프레임 (임의 크기) |
| 제어/결과 전송 | WebSocket **텍스트(JSON)** 프레임 |
| 첫 메시지 | `{"type":"config", ...}` (필수) |
| 마지막 메시지 | `{"type":"end"}` (서버가 final 응답 후 close) |
| 세션 수명 | WebSocket 연결과 동일 (별도 TTL 없음) |
| 인증 | 없음 (사설망 전제) |

---

## 2. 메시지 프로토콜

### 2-1. Client → Server

#### (1) `config` (텍스트 JSON, 첫 메시지 필수)

```json
{
  "type": "config",
  "context": "",
  "language": "Korean",
  "chunk_size_sec": 1.0,
  "unfixed_chunk_num": 4,
  "unfixed_token_num": 5
}
```

| 필드 | 타입 | 기본 | 설명 |
|---|---|---|---|
| `type` | string | — | 반드시 `"config"` |
| `context` | string | `""` | 프롬프트 시스템 메시지로 들어가는 컨텍스트 (도메인 정보, 화자 정보 등) |
| `language` | string\|null | `null`(자동감지) | 인식 언어 **풀네임**. ISO 코드 불가. 지원 목록은 §6 참고 |
| `chunk_size_sec` | float | `1.0` | 서버 측 청크 크기(초). 작을수록 첫 응답이 빠르나 정확도↓ |
| `unfixed_chunk_num` | int | `4` | 초반 N개 청크는 prefix 미사용 (모델이 자유 디코딩) |
| `unfixed_token_num` | int | `5` | 청크 경계 흔들림 방지를 위해 매 청크마다 롤백할 토큰 수 |

> 모든 필드는 optional. 누락 시 서버 CLI 기본값 사용.

#### (2) 오디오 (바이너리 프레임)

- float32 little-endian PCM, 16kHz, mono
- 길이 % 4 == 0 (float32 정렬)
- 크기 제약 없음 — 200ms/1초/원하는 단위로 보내도 됨
- 서버가 `chunk_size_sec` 만큼 자동으로 누적·분할

#### (3) `end` (텍스트 JSON)

```json
{"type": "end"}
```

남은 버퍼를 flush, FINAL 결과 송신, `closed` 송신 후 WebSocket close.

### 2-2. Server → Client

#### (1) `ready` (config 검증 성공 직후 1회)

```json
{
  "type": "ready",
  "session_id": "37465704fdbb4a1e8c2340d19afd438e",
  "sample_rate": 16000,
  "chunk_size_sec": 1.0,
  "unfixed_chunk_num": 4,
  "unfixed_token_num": 5
}
```

#### (2) `result` (청크 처리/종료마다)

```json
{
  "type": "result",
  "is_final": false,
  "chunk_id": 3,
  "language": "Korean",
  "text": "안녕하세요. 포지큐브 고객센터 AI 콜봇입니다."
}
```

- `is_final: false` — interim/partial 결과. 다음 partial 에서 일부가 정정될 수 있음
- `is_final: true` — `end` 처리 직후 보내는 최종 결과. 한 세션당 1회

#### (3) `error`

```json
{"type": "error", "code": "invalid_audio", "message": "float32 bytes length not multiple of 4"}
```

| code | 발생 시점 | close 여부 |
|---|---|---|
| `invalid_config` | 첫 메시지가 텍스트 JSON 이 아니거나 `type != "config"` | close |
| `invalid_audio` | 바이너리 길이가 4의 배수가 아님 | 유지 |
| `invalid_message` | 텍스트가 JSON 이 아니거나 알 수 없는 `type` | 유지 |
| `internal_error` | 서버 내부 예외 | close |

#### (4) `closed` (close 직전 마지막)

```json
{"type": "closed", "reason": "end"}   // 또는 "error"
```

---

## 3. 전형적인 흐름

```
Client                                  Server
  │  WebSocket connect                    │
  │ ─────────────────────────────────────▶│
  │  {"type":"config","language":"Korean"}│
  │ ─────────────────────────────────────▶│
  │                                       │ ◀── ready
  │  <16000×4B float32 PCM>               │
  │ ─────────────────────────────────────▶│
  │                                       │ ◀── result is_final=false cid=1
  │  <16000×4B>                           │
  │ ─────────────────────────────────────▶│
  │                                       │ ◀── result is_final=false cid=2
  │  ...                                  │
  │  {"type":"end"}                       │
  │ ─────────────────────────────────────▶│
  │                                       │ ◀── result is_final=true cid=N
  │                                       │ ◀── closed reason=end
  │                                       │ ◀── (WebSocket close)
```

---

## 4. 최소 Python 클라이언트

```python
import asyncio
import json
import numpy as np
import soundfile as sf
import websockets

SERVER = "ws://172.31.79.202:30000/api/ws"

async def transcribe(wav_path: str) -> None:
    wav, sr = sf.read(wav_path, dtype="float32")
    assert sr == 16000 and wav.ndim == 1, "16kHz mono float32 만 지원"

    async with websockets.connect(SERVER, max_size=None) as ws:
        # 1. config
        await ws.send(json.dumps({"type": "config", "language": "Korean"}))
        ready = json.loads(await ws.recv())
        print("READY:", ready["session_id"])

        # 2. 오디오를 200ms 청크로 송신 (실시간 페이싱하려면 sleep 추가)
        step = int(0.2 * 16000)
        for i in range(0, len(wav), step):
            await ws.send(wav[i : i + step].tobytes())

        # 3. 종료 신호
        await ws.send(json.dumps({"type": "end"}))

        # 4. 결과 수신
        async for msg in ws:
            evt = json.loads(msg)
            t = evt.get("type")
            if t == "result":
                tag = "FINAL" if evt["is_final"] else "partial"
                print(f"[{tag}] {evt['text']}")
            elif t == "closed":
                break
            elif t == "error":
                raise RuntimeError(f"{evt['code']}: {evt['message']}")

asyncio.run(transcribe("wav/google_tts.wav"))
```

> 실시간 마이크 입력 시뮬레이션 / 다중 동시 세션 / 페이싱 측정 예시는
> [`examples/example_qwen3_asr_vllm_streaming_ws.py`](../examples/example_qwen3_asr_vllm_streaming_ws.py)
> 와 [`tests/test_asr_ws_sequential.py`](../tests/test_asr_ws_sequential.py) 참고.

---

## 5. 브라우저(JavaScript) 클라이언트 예시

```js
const ws = new WebSocket("ws://172.31.79.202:30000/api/ws");
ws.binaryType = "arraybuffer";

ws.onopen = () => {
  ws.send(JSON.stringify({ type: "config", language: "Korean" }));
};

ws.onmessage = (e) => {
  const evt = JSON.parse(e.data);   // 서버는 항상 텍스트만 보냄
  if (evt.type === "result") {
    console.log(evt.is_final ? "FINAL" : "partial", evt.text);
  } else if (evt.type === "closed") {
    ws.close();
  } else if (evt.type === "error") {
    console.error(evt.code, evt.message);
  }
};

// AudioWorklet / MediaRecorder 등으로 16kHz Float32Array 를 얻은 뒤:
function sendAudio(float32Array) {
  ws.send(float32Array.buffer);     // 그대로 바이너리 송신
}

function finish() {
  ws.send(JSON.stringify({ type: "end" }));
}
```

> 브라우저는 보통 마이크가 44.1 / 48 kHz라서 별도로 16kHz 리샘플 필요.
> [`AudioContext({ sampleRate: 16000 })`](https://developer.mozilla.org/en-US/docs/Web/API/AudioContext)
> + AudioWorklet, 또는 `OfflineAudioContext` 로 리샘플링.

---

## 6. 오디오 형식 변환 (예시)

### 24kHz/44.1kHz/48kHz → 16kHz

```python
import numpy as np, soundfile as sf

wav, sr = sf.read(path, dtype="float32", always_2d=False)
if sr != 16000:
    dur = len(wav) / sr
    n = int(round(dur * 16000))
    x_old = np.linspace(0, dur, num=len(wav), endpoint=False)
    x_new = np.linspace(0, dur, num=n,        endpoint=False)
    wav = np.interp(x_new, x_old, wav).astype(np.float32)
```

> 정확한 리샘플링이 필요하면 `librosa.resample` 또는 `scipy.signal.resample_poly` 추천.

### int16 → float32

```python
audio_float32 = audio_int16.astype(np.float32) / 32768.0
```

### 스테레오 → 모노

```python
if wav.ndim == 2:
    wav = wav.mean(axis=1).astype(np.float32)
```

---

## 7. 지원 언어

`config.language` 에 넘길 수 있는 **풀네임**입니다. (ISO 코드는 거부됨)

```
Chinese, English, Cantonese, Arabic, German, French, Spanish,
Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese,
Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish,
Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian
```

언어를 지정하지 않으면 (`language` 필드 omit 또는 null) 모델이 자동 감지합니다.

---

## 8. 성능 / 지연 가이드

`chunk_size_sec=1.0`, 200ms 청크 페이싱 기준 실측값입니다 (16개 한국어 TTS 파일, 평균값).

| 항목 | 값 | 설명 |
|---|---|---|
| 첫 partial 도착 (TTFT) | **~0.9s** | 클라이언트가 첫 바이트 보낸 후 첫 텍스트가 돌아오기까지 |
| └ 그중 청크 누적 대기 | ~0.8s | `chunk_size_sec=1.0` 이라 1초 분량이 모일 때까지 |
| └ 그중 서버 처리 + 왕복 | ~88ms | 순수 모델 추론 + 네트워크 |
| 청크별 처리 시간 | ~80ms / 1초 청크 | 처리 능력 RTF ≈ **0.08** |
| end → FINAL 도착 | ~70~90ms | 마지막 flush 응답 |

**튜닝 포인트**

- `chunk_size_sec` ↓ (예: 0.5) → TTFT 절반 가까이 단축, 대신 컨텍스트가 짧아져 정확도 소폭 하락
- `unfixed_chunk_num` ↑ → 초반 자유 디코딩 길어져 첫 partial 안정성↑, 청크 단위 응답속도↓
- `unfixed_token_num` ↑ → 청크 경계 흔들림 정정 강화, 컨텍스트 손실↑

---

## 9. 흔한 실수

| 증상 | 원인 | 해결 |
|---|---|---|
| 첫 메시지부터 `invalid_config` 후 close | `language: "ko"` 같은 ISO 코드 사용 | `"Korean"` 풀네임 사용 |
| 동일 증상 | 첫 메시지를 바이너리로 보냄 | 첫 메시지는 반드시 텍스트 JSON `{"type":"config"}` |
| `invalid_audio` 반복 | float32 가 아닌 다른 타입(int16 등) 전송 | `numpy.float32` 캐스팅 후 `tobytes()` |
| 동일 증상 | 청크 길이가 4 의 배수가 아님 | 16kHz 샘플 단위(=4B)로 정렬 |
| FINAL 이 영원히 안 옴 | `{"type":"end"}` 미전송 | 마지막에 반드시 end 송신 |
| `ModuleNotFoundError: websockets` | 클라이언트에 websockets 미설치 | `pip install websockets` |
| 첫 partial 까지 매우 김 | 음성 데이터를 한 번에 모아서 보냄 + `chunk_size_sec=1.0` | 실시간성이 필요하면 작게 쪼개서 흘려보내기 + chunk_size 조정 |
| 텍스트에 `�` (replacement char) | 청크 경계에서 멀티바이트 문자 깨짐 | 정상 동작 — partial 단계에서만 잠시 보임. 서버가 자동 복구 (FINAL 에서는 사라짐) |

---

## 10. 관련 파일

| 파일 | 용도 |
|---|---|
| [`qwen_asr/cli/serve_async.py`](../qwen_asr/cli/serve_async.py) | WS 핸들러 구현 (`/api/ws`) |
| [`examples/example_qwen3_asr_vllm_streaming_ws.py`](../examples/example_qwen3_asr_vllm_streaming_ws.py) | 풀 기능 클라이언트 예시 (페이싱·다중 이벤트 처리) |
| [`tests/test_asr_ws_sequential.py`](../tests/test_asr_ws_sequential.py) | 10개 파일 순차 스트리밍 성능 측정 |
| [`SERVER_GUIDE.md`](../SERVER_GUIDE.md) | 서버 배포·운영 절차 (§4-1 에 WS 프로토콜 명세 요약) |

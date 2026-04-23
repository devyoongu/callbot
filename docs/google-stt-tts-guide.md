# Google STT / TTS 통합 가이드

다른 Python 콜봇 프로젝트에서 Google Cloud STT(Speech-to-Text V2)와 TTS(Text-to-Speech)를 독립적으로 사용하기 위한 가이드입니다.

---

## 목차

1. [사전 준비](#1-사전-준비)
2. [인증 설정](#2-인증-설정)
3. [Google STT V2](#3-google-stt-v2)
4. [Google TTS](#4-google-tts)
5. [오디오 포맷 규격](#5-오디오-포맷-규격)
6. [완전한 독립 예제 코드](#6-완전한-독립-예제-코드)

---

## 1. 사전 준비

### 패키지 설치

```bash
pip install google-cloud-speech google-cloud-texttospeech google-auth numpy

# Client-side VAD 사용 시 (선택)
pip install webrtcvad
```

### Google Cloud 프로젝트 설정

1. [Google Cloud Console](https://console.cloud.google.com)에서 프로젝트 생성
2. **Speech-to-Text API** 및 **Text-to-Speech API** 활성화
3. 서비스 계정 생성 후 **JSON 키 파일** 다운로드

---

## 2. 인증 설정

### 인증 파일 배치

서비스 계정 JSON 키 파일을 프로젝트 내 `credentials/` 디렉터리에 배치합니다.

```
your_project/
├── credentials/
│   └── your-service-account-key.json   ← 여기에 배치
├── stt_example.py
└── tts_example.py
```

> **복수 키 지원 (Round-Robin):** `credentials/` 디렉터리에 JSON 파일을 여러 개 배치하면 API 호출 시 자동으로 순환 선택됩니다. 쿼터 분산 목적으로 활용할 수 있습니다.

### 인증 로더 유틸리티

아래 유틸리티를 프로젝트에 복사하여 사용합니다.

```python
# google_credentials.py

import glob
import os
import threading
from typing import Optional

_rr_lock = threading.Lock()
_rr_index = 0
_rr_credentials_list: list = []


def _discover_credentials(credentials_dir: str) -> list:
    return sorted(glob.glob(os.path.join(credentials_dir, "*.json")))


def _get_next_round_robin(credentials_dir: str) -> Optional[str]:
    """Thread-safe round-robin으로 credentials 파일 경로 반환"""
    global _rr_index, _rr_credentials_list
    with _rr_lock:
        _rr_credentials_list = _discover_credentials(credentials_dir)
        if not _rr_credentials_list:
            return None
        path = _rr_credentials_list[_rr_index % len(_rr_credentials_list)]
        _rr_index += 1
        return path


def load_google_credentials(credentials_dir: str = None):
    """
    Google Cloud Credentials 객체 로드

    Args:
        credentials_dir: JSON 키 파일이 있는 디렉터리 경로.
                         None이면 스크립트 기준 'credentials/' 탐색.
    Returns:
        google.oauth2.service_account.Credentials
    Raises:
        FileNotFoundError: credentials 파일 없을 때
    """
    from google.oauth2 import service_account

    if credentials_dir is None:
        credentials_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credentials")

    credentials_path = _get_next_round_robin(credentials_dir)
    if not credentials_path:
        raise FileNotFoundError(f"No credentials JSON found in: {credentials_dir}")

    print(f"[Auth] Using credentials: {os.path.basename(credentials_path)}")
    return service_account.Credentials.from_service_account_file(credentials_path)
```

---

## 3. Google STT V2

### 지원 모델

| 모델 | 리전 | 엔드포인트 | 특징 |
|------|------|-----------|------|
| `chirp_3` | `us` | `us-speech.googleapis.com` | 범용 음성 인식, 다중 리전 |
| `telephony` | `global` | `speech.googleapis.com` | 전화 통화 최적화 (8kHz/16kHz) |

### VAD (음성 감지) 방식

| 방식 | 설명 |
|------|------|
| **Server VAD** | Google 서버에서 음성 시작/끝 감지. 정확하나 네트워크 왕복 지연 있음 |
| **Client VAD** | WebRTC VAD + RMS 에너지 이중 판단으로 로컬에서 EOS 감지. 지연 최소화 |

### 오디오 입력 규격

- **포맷:** LINEAR16 PCM
- **샘플레이트:** 16,000 Hz
- **채널:** 1 (mono)
- **청크 크기:** 4,000 bytes (125ms 분량)

### 스트리밍 STT 예제

```python
# stt_example.py

import threading
import time
import queue
import numpy as np
from typing import Optional, Tuple
from google_credentials import load_google_credentials

try:
    from google.cloud.speech_v2 import SpeechClient
    from google.cloud.speech_v2.types import cloud_speech as cloud_speech_types
    from google.api_core.client_options import ClientOptions
except ImportError:
    raise ImportError("pip install google-cloud-speech")

try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False


class GoogleSTTV2:
    """
    Google Cloud Speech-to-Text V2 스트리밍 인식기

    사용법:
        stt = GoogleSTTV2(project_id="your-gcp-project", model="chirp_3")
        stt.initialize()
        stt.start_streaming(audio_source_func)
        transcript = stt.wait_for_result()
        stt.finalize()
    """

    # 모델별 설정
    MODEL_CONFIG = {
        "chirp_3": {
            "region": "us",
            "endpoint": "us-speech.googleapis.com",
            "sample_rate": 16000,
        },
        "telephony": {
            "region": "global",
            "endpoint": "speech.googleapis.com",
            "sample_rate": 16000,
        },
    }

    def __init__(
        self,
        project_id: str,
        model: str = "chirp_3",
        language: str = "ko-KR",
        use_server_vad: bool = True,
        credentials_dir: str = None,
    ):
        """
        Args:
            project_id: GCP 프로젝트 ID
            model: STT 모델 ('chirp_3' 또는 'telephony')
            language: 언어 코드 (예: 'ko-KR', 'en-US')
            use_server_vad: True=서버 VAD, False=클라이언트 VAD(webrtcvad 필요)
            credentials_dir: credentials JSON 디렉터리 경로
        """
        self.project_id = project_id
        self.model = model
        self.language = language
        self.use_server_vad = use_server_vad
        self.credentials_dir = credentials_dir

        cfg = self.MODEL_CONFIG.get(model, self.MODEL_CONFIG["chirp_3"])
        self.region = cfg["region"]
        self.endpoint = cfg["endpoint"]
        self.sample_rate = cfg["sample_rate"]

        self.client: Optional[SpeechClient] = None
        self._result_event = threading.Event()
        self._final_transcript = ""
        self._stt_error: Optional[str] = None
        self._stop = False
        self._streaming_thread: Optional[threading.Thread] = None

        # VAD 상태
        self.speech_started = False
        self.eos_done = False
        self._process_audio = True

    # ------------------------------------------------------------------ #
    #  공개 API
    # ------------------------------------------------------------------ #

    def initialize(self):
        """클라이언트 초기화. start_streaming() 전에 반드시 호출."""
        credentials = load_google_credentials(self.credentials_dir)
        self.client = SpeechClient(
            client_options=ClientOptions(api_endpoint=self.endpoint),
            credentials=credentials,
        )
        print(f"[GoogleSTT] Client ready (model={self.model}, endpoint={self.endpoint})")

    def start_streaming(self, audio_source):
        """
        백그라운드 스레드에서 스트리밍 인식 시작

        Args:
            audio_source: 오디오 청크를 yield하는 제너레이터 또는 호출 가능 객체.
                          각 청크는 bytes 타입 LINEAR16 PCM이어야 합니다.
                          청크 크기: 4000 bytes (125ms) 권장.
        """
        self._stop = False
        self._process_audio = True
        self._final_transcript = ""
        self._stt_error = None
        self._result_event.clear()
        self.speech_started = False
        self.eos_done = False

        self._audio_source = audio_source
        self._streaming_thread = threading.Thread(
            target=self._run_streaming, daemon=True
        )
        self._streaming_thread.start()

    def wait_for_result(self, timeout: float = 30.0) -> Tuple[bool, str]:
        """
        STT 결과 대기

        Returns:
            (success, transcript): 성공 여부와 인식 결과 문자열
                                   빈 음성이면 transcript='non_voice'
        """
        done = self._result_event.wait(timeout=timeout)
        if not done:
            return False, "timeout"
        if self._stt_error:
            return False, self._stt_error
        return True, self._final_transcript

    def stop(self):
        """스트리밍 강제 중지"""
        self._stop = True
        self._process_audio = False

    def finalize(self):
        """리소스 정리"""
        self.stop()
        if self._streaming_thread and self._streaming_thread.is_alive():
            self._streaming_thread.join(timeout=3.0)
        self.client = None

    # ------------------------------------------------------------------ #
    #  내부 구현
    # ------------------------------------------------------------------ #

    def _run_streaming(self):
        try:
            recognizer_path = (
                f"projects/{self.project_id}/locations/{self.region}/recognizers/_"
            )

            recognition_config = cloud_speech_types.RecognitionConfig(
                explicit_decoding_config=cloud_speech_types.ExplicitDecodingConfig(
                    encoding=cloud_speech_types.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.sample_rate,
                    audio_channel_count=1,
                ),
                language_codes=[self.language],
                model=self.model,
            )

            streaming_features = cloud_speech_types.StreamingRecognitionFeatures(
                interim_results=True,
                enable_voice_activity_events=self.use_server_vad,
            )

            streaming_config = cloud_speech_types.StreamingRecognitionConfig(
                config=recognition_config,
                streaming_features=streaming_features,
            )

            config_request = cloud_speech_types.StreamingRecognizeRequest(
                recognizer=recognizer_path,
                streaming_config=streaming_config,
            )

            def requests_gen():
                yield config_request
                yield from self._audio_generator()

            responses = self.client.streaming_recognize(requests=requests_gen())

            for response in responses:
                if self._stop:
                    break

                # Server VAD 이벤트 처리
                if self.use_server_vad and response.speech_event_type:
                    self._handle_vad_event(response.speech_event_type)

                for result in response.results:
                    if not result.alternatives:
                        continue
                    transcript = result.alternatives[0].transcript

                    if result.is_final:
                        self._final_transcript = transcript.strip() or "non_voice"
                        print(f"[GoogleSTT] Final: '{self._final_transcript}'")
                        self._result_event.set()
                        return  # 완료

        except Exception as e:
            print(f"[GoogleSTT] Error: {e}")
            self._stt_error = str(e)
            self._result_event.set()

    def _audio_generator(self):
        """
        audio_source에서 오디오를 읽어 StreamingRecognizeRequest로 yield

        audio_source는 bytes를 yield하는 제너레이터이거나,
        get_chunk() 메서드를 제공하는 객체여야 합니다.
        """
        # Client VAD 초기화
        vad = None
        if not self.use_server_vad and WEBRTC_VAD_AVAILABLE:
            vad = webrtcvad.Vad(2)  # Aggressiveness 0-3 (2=중간)

        VAD_FRAME_SIZE = 640      # 20ms frame (320 samples × 2 bytes)
        RMS_THRESHOLD = 200       # 소음/발화 경계 RMS 값
        SILENCE_THRESHOLD = 5     # EOS 판정에 필요한 연속 무음 청크 수

        speech_detected = False
        silence_frames = 0
        high_energy_count = 0

        for chunk in self._audio_source:
            if self._stop:
                break
            if not chunk:
                continue

            # Client VAD 처리 (use_server_vad=False일 때)
            if vad and len(chunk) == 4000:
                is_speech = self._client_vad_check(
                    vad, chunk, VAD_FRAME_SIZE, RMS_THRESHOLD
                )

                if is_speech:
                    high_energy_count += 1
                    if high_energy_count >= 2 and not speech_detected:
                        speech_detected = True
                        self.speech_started = True
                        print("[GoogleSTT] Speech started (client VAD)")
                        silence_frames = 0
                else:
                    high_energy_count = 0

                if speech_detected:
                    if not is_speech:
                        silence_frames += 1
                        if silence_frames >= SILENCE_THRESHOLD:
                            self.eos_done = True
                            self._process_audio = False
                            print("[GoogleSTT] EOS detected (client VAD)")
                            yield cloud_speech_types.StreamingRecognizeRequest(audio=chunk)
                            break
                    else:
                        silence_frames = 0

            yield cloud_speech_types.StreamingRecognizeRequest(audio=chunk)

    def _client_vad_check(self, vad, chunk: bytes, frame_size: int, rms_threshold: float) -> bool:
        """WebRTC VAD + RMS 에너지 이중 판단"""
        num_frames = len(chunk) // frame_size
        speech_frames = 0

        for i in range(num_frames):
            frame = chunk[i * frame_size:(i + 1) * frame_size]
            if len(frame) == frame_size and vad.is_speech(frame, self.sample_rate):
                speech_frames += 1

        vad_positive = speech_frames >= (num_frames * 0.66)

        samples = np.frombuffer(chunk, dtype=np.int16)
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        energy_positive = rms >= rms_threshold

        return vad_positive and energy_positive

    def _handle_vad_event(self, event_type):
        """Server VAD 이벤트 핸들러"""
        SpeechEventType = cloud_speech_types.StreamingRecognizeResponse.SpeechEventType

        if event_type == SpeechEventType.SPEECH_ACTIVITY_BEGIN:
            print("[GoogleSTT] VAD: Speech BEGIN")
            self.speech_started = True

        elif event_type in (
            SpeechEventType.SPEECH_ACTIVITY_END,
            SpeechEventType.END_OF_SINGLE_UTTERANCE,
        ):
            print("[GoogleSTT] VAD: Speech END / EOS")
            self.eos_done = True
            self._process_audio = False
            self._stop = True  # Generator 종료 → Final result 수신
```

### STT 사용 예제

```python
# 마이크 또는 오디오 큐에서 읽는 제너레이터 예시
import queue

audio_queue = queue.Queue()  # 외부에서 PCM 청크를 넣는 큐

def audio_source_from_queue():
    """큐에서 오디오 청크를 읽어 yield"""
    while True:
        try:
            chunk = audio_queue.get(timeout=0.5)
            if chunk is None:  # 종료 신호
                break
            yield chunk
        except queue.Empty:
            continue

# STT 초기화 및 실행
stt = GoogleSTTV2(
    project_id="your-gcp-project-id",
    model="chirp_3",          # 또는 "telephony"
    language="ko-KR",
    use_server_vad=True,      # False이면 webrtcvad 필요
    credentials_dir="./credentials",
)
stt.initialize()
stt.start_streaming(audio_source_from_queue())

# 결과 대기
success, transcript = stt.wait_for_result(timeout=30.0)
print(f"STT 결과: {transcript}")   # "안녕하세요" 또는 "non_voice"

stt.finalize()
```

---

## 4. Google TTS

### 지원 음성 (한국어)

| 키 | Voice ID | 성별 | 특징 |
|----|---------|------|------|
| `chirp3-hd-achernar` | `ko-KR-Chirp3-HD-Achernar` | 여성 | **권장** - 고음질 |
| `chirp3-hd-achird` | `ko-KR-Chirp3-HD-Achird` | 남성 | 고음질 |
| `neural2-a` | `ko-KR-Neural2-A` | 여성 | Neural2 |
| `neural2-b` | `ko-KR-Neural2-B` | 여성 | Neural2 |
| `neural2-c` | `ko-KR-Neural2-C` | 남성 | Neural2 |

### 오디오 출력 규격

- **포맷:** LINEAR16 PCM
- **샘플레이트:** 24,000 Hz
- **채널:** 1 (mono)
- **비트 깊이:** 16-bit

> **참고:** 콜봇에서 8kHz 또는 16kHz가 필요한 경우 리샘플링이 필요합니다.

### TTS 핸들러 코드

```python
# tts_example.py

import struct
import numpy as np
from typing import Iterator, Optional
from google_credentials import load_google_credentials

try:
    from google.cloud import texttospeech
    from google.cloud.texttospeech_v1.types import (
        StreamingSynthesizeRequest,
        StreamingSynthesizeConfig,
        StreamingSynthesisInput,
        VoiceSelectionParams,
    )
except ImportError:
    raise ImportError("pip install google-cloud-texttospeech")


# 한국어 음성 매핑
GOOGLE_TTS_VOICES = {
    "chirp3-hd-achernar": "ko-KR-Chirp3-HD-Achernar",
    "chirp3-hd-achird":   "ko-KR-Chirp3-HD-Achird",
    "neural2-a":          "ko-KR-Neural2-A",
    "neural2-b":          "ko-KR-Neural2-B",
    "neural2-c":          "ko-KR-Neural2-C",
}

# 오디오 품질 설정 (조정 가능)
TRIM_START_MS  = 20   # 앞부분 artifact 제거 (ms)
FADE_OUT_MS    = 50   # 끝부분 fade-out 길이 (ms)
PADDING_MS     = 200  # 무음 패딩 길이 (ms)
SAMPLE_RATE    = 24000


class GoogleTTS:
    """
    Google Cloud Text-to-Speech 핸들러

    사용법:
        tts = GoogleTTS(voice="chirp3-hd-achernar", credentials_dir="./credentials")

        # 동기 방식 (전체 오디오 한 번에)
        pcm_bytes = tts.synthesize(text)

        # 스트리밍 방식 (청크 단위 yield)
        for chunk in tts.synthesize_streaming(text):
            send_to_phone(chunk)

        # WAV 파일 저장
        tts.save_wav(text, "output.wav")
    """

    def __init__(
        self,
        voice: str = "chirp3-hd-achernar",
        language_code: str = "ko-KR",
        credentials_dir: str = None,
    ):
        """
        Args:
            voice: 음성 ID 또는 단축 키 (GOOGLE_TTS_VOICES 참조)
            language_code: 언어 코드
            credentials_dir: credentials JSON 디렉터리 경로
        """
        self.voice_name = GOOGLE_TTS_VOICES.get(voice.lower(), voice)
        self.language_code = language_code
        self.credentials_dir = credentials_dir
        self._client: Optional[texttospeech.TextToSpeechClient] = None

    def _get_client(self) -> texttospeech.TextToSpeechClient:
        """Lazy 초기화 - 첫 호출 시 클라이언트 생성"""
        if self._client is None:
            credentials = load_google_credentials(self.credentials_dir)
            self._client = texttospeech.TextToSpeechClient(credentials=credentials)
        return self._client

    # ------------------------------------------------------------------ #
    #  동기 방식
    # ------------------------------------------------------------------ #

    def synthesize(self, text: str) -> bytes:
        """
        텍스트 → LINEAR16 PCM (24kHz, mono, 16-bit)

        artifact 제거(trim/fade-out/padding) 처리 포함

        Args:
            text: 변환할 텍스트
        Returns:
            bytes: 처리된 LINEAR16 PCM 오디오 데이터
        """
        client = self._get_client()

        response = client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=text),
            voice=texttospeech.VoiceSelectionParams(
                language_code=self.language_code,
                name=self.voice_name,
            ),
            audio_config=texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=SAMPLE_RATE,
            ),
        )

        pcm = response.audio_content
        pcm = _trim_start(pcm, TRIM_START_MS, SAMPLE_RATE)
        pcm = _apply_fade_out(pcm, FADE_OUT_MS, SAMPLE_RATE)
        pcm = _add_silence_padding(pcm, PADDING_MS, SAMPLE_RATE)
        return pcm

    def save_wav(self, text: str, output_path: str) -> str:
        """
        텍스트 → WAV 파일 저장

        Args:
            text: 변환할 텍스트
            output_path: 저장 경로 (.wav)
        Returns:
            str: 저장된 파일 경로
        """
        pcm = self.synthesize(text)
        wav = _wrap_pcm_in_wav(pcm, SAMPLE_RATE)
        with open(output_path, "wb") as f:
            f.write(wav)
        print(f"[GoogleTTS] Saved: {output_path}")
        return output_path

    # ------------------------------------------------------------------ #
    #  스트리밍 방식
    # ------------------------------------------------------------------ #

    def synthesize_streaming(
        self,
        text: str,
        min_chunk_chars: int = 10,
        max_chunk_chars: int = 200,
    ) -> Iterator[bytes]:
        """
        텍스트 → LINEAR16 PCM 청크 스트리밍 (generator)

        내부적으로 텍스트를 문장 단위로 분할하여 Google TTS Streaming API 호출.
        각 청크는 즉시 yield되므로 첫 번째 오디오까지의 지연(TTFB)이 짧습니다.

        Args:
            text: 변환할 텍스트
            min_chunk_chars: 텍스트 청크 최소 길이
            max_chunk_chars: 텍스트 청크 최대 길이
        Yields:
            bytes: LINEAR16 PCM 오디오 청크 (24kHz, mono, 16-bit)
        """
        client = self._get_client()
        text_chunks = _chunk_text(text, min_chunk_chars, max_chunk_chars)

        if not text_chunks:
            return

        def request_generator():
            # 첫 요청: 음성 설정
            yield StreamingSynthesizeRequest(
                streaming_config=StreamingSynthesizeConfig(
                    voice=VoiceSelectionParams(
                        language_code=self.language_code,
                        name=self.voice_name,
                    )
                    # Chirp3-HD는 streaming_audio_config 미지원 → 기본값 사용
                )
            )
            # 이후 요청: 텍스트 청크 전송
            for chunk in text_chunks:
                yield StreamingSynthesizeRequest(
                    input=StreamingSynthesisInput(text=chunk)
                )

        responses = client.streaming_synthesize(requests=request_generator())
        for response in responses:
            if response.audio_content:
                yield response.audio_content


# ------------------------------------------------------------------ #
#  오디오 처리 헬퍼 함수
# ------------------------------------------------------------------ #

def _trim_start(pcm: bytes, duration_ms: int, sample_rate: int) -> bytes:
    """앞부분 artifact 제거"""
    trim_bytes = int(sample_rate * 2 * duration_ms / 1000)
    return pcm[trim_bytes:] if len(pcm) > trim_bytes else pcm


def _apply_fade_out(pcm: bytes, duration_ms: int, sample_rate: int) -> bytes:
    """끝부분 fade-out 적용 (비프음 제거)"""
    arr = np.frombuffer(pcm, dtype=np.int16).copy()
    fade_samples = int(sample_rate * duration_ms / 1000)
    fade_samples = min(fade_samples, len(arr))
    curve = np.linspace(1.0, 0.0, fade_samples)
    arr[-fade_samples:] = (arr[-fade_samples:] * curve).astype(np.int16)
    return arr.tobytes()


def _add_silence_padding(pcm: bytes, duration_ms: int, sample_rate: int) -> bytes:
    """끝에 무음 패딩 추가 (스트림 종료 비프음 방지)"""
    silence_samples = int(sample_rate * duration_ms / 1000)
    silence = np.zeros(silence_samples, dtype=np.int16).tobytes()
    return pcm + silence


def _wrap_pcm_in_wav(pcm: bytes, sample_rate: int) -> bytes:
    """LINEAR16 PCM → WAV 포맷으로 래핑"""
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(pcm),
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        len(pcm),
    )
    return header + pcm


def _chunk_text(text: str, min_chars: int = 10, max_chars: int = 200) -> list:
    """한국어 문장 단위 텍스트 청킹"""
    import re

    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks, current = [], ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            words, temp = sentence.split(), ""
            for word in words:
                if len(temp) + len(word) + 1 <= max_chars:
                    temp += word + " "
                else:
                    if temp:
                        chunks.append(temp.strip())
                    temp = word + " "
            if temp:
                chunks.append(temp.strip())
        elif len(current) + len(sentence) + 1 <= max_chars:
            current += sentence + " "
        else:
            if current:
                chunks.append(current.strip())
            current = sentence + " "

    if current:
        chunks.append(current.strip())

    # 최소 크기 미만 청크 병합
    final, i = [], 0
    while i < len(chunks):
        chunk = chunks[i]
        if len(chunk) < min_chars and i + 1 < len(chunks):
            merged = chunk + " " + chunks[i + 1]
            if len(merged) <= max_chars:
                chunk = merged
                i += 1
        final.append(chunk)
        i += 1

    return final
```

### TTS 사용 예제

```python
# 동기 방식
tts = GoogleTTS(
    voice="chirp3-hd-achernar",   # 또는 "ko-KR-Chirp3-HD-Achernar" 직접 입력
    language_code="ko-KR",
    credentials_dir="./credentials",
)

# PCM bytes 반환 (24kHz, mono, 16-bit)
pcm_data = tts.synthesize("안녕하세요, 무엇을 도와드릴까요?")
print(f"Generated {len(pcm_data)} bytes of PCM")

# WAV 파일로 저장
tts.save_wav("안녕하세요.", "output.wav")

# 스트리밍 방식 (실시간 전송)
for chunk in tts.synthesize_streaming("긴 텍스트를 스트리밍으로 전송합니다."):
    # 청크를 전화 회선이나 오디오 버퍼에 전달
    send_audio_to_phone(chunk)
```

---

## 5. 오디오 포맷 규격

### STT 입력

| 항목 | 값 |
|------|----|
| 포맷 | LINEAR16 PCM (부호 있는 16-bit 정수, Little-Endian) |
| 샘플레이트 | 16,000 Hz |
| 채널 | 1 (mono) |
| 권장 청크 크기 | 4,000 bytes = 125ms |

### TTS 출력

| 항목 | 값 |
|------|----|
| 포맷 | LINEAR16 PCM (부호 있는 16-bit 정수, Little-Endian) |
| 샘플레이트 | 24,000 Hz |
| 채널 | 1 (mono) |
| Trim (앞) | 20ms 제거 (artifact 방지) |
| Fade-out (뒤) | 50ms (비프음 방지) |
| Silence Padding | 200ms 추가 (스트림 종료 비프음 방지) |

### 샘플레이트 변환 (필요 시)

TTS 출력이 24kHz이고, 콜봇이 8kHz 또는 16kHz를 사용한다면 리샘플링이 필요합니다.

```bash
pip install librosa
# 또는
pip install resampy
```

```python
import numpy as np
import librosa

def resample_pcm(pcm_bytes: bytes, src_rate: int, dst_rate: int) -> bytes:
    """LINEAR16 PCM 리샘플링"""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    resampled = librosa.resample(samples, orig_sr=src_rate, target_sr=dst_rate)
    return (resampled * 32768.0).astype(np.int16).tobytes()

# 24kHz → 16kHz 변환
pcm_16k = resample_pcm(pcm_24k, src_rate=24000, dst_rate=16000)
```

---

## 6. 완전한 독립 예제 코드

다음은 STT + TTS를 모두 포함한 최소 동작 예제입니다.

```python
#!/usr/bin/env python3
"""
Google STT + TTS 독립 실행 예제
- STT: 오디오 파일 → 텍스트
- TTS: 텍스트 → WAV 파일
"""

import wave
import numpy as np


def read_wav_as_pcm_chunks(wav_path: str, chunk_bytes: int = 4000):
    """WAV 파일을 4000 bytes 청크로 yield (STT 테스트용)"""
    with wave.open(wav_path, "rb") as wf:
        while True:
            data = wf.readframes(chunk_bytes // (wf.getsampwidth() * wf.getnchannels()))
            if not data:
                break
            yield data


def main():
    CREDENTIALS_DIR = "./credentials"
    GCP_PROJECT_ID  = "your-gcp-project-id"

    # ---- TTS 테스트 ----
    from tts_example import GoogleTTS
    tts = GoogleTTS(credentials_dir=CREDENTIALS_DIR)
    tts.save_wav("안녕하세요, 테스트입니다.", "test_output.wav")
    print("TTS 완료: test_output.wav")

    # ---- STT 테스트 (위에서 생성한 WAV를 16kHz로 변환 후 사용) ----
    # 참고: test_output.wav는 24kHz이므로 실제 STT 테스트 시
    # 16kHz WAV 파일을 준비하거나 리샘플링 후 사용하세요.
    from stt_example import GoogleSTTV2

    stt = GoogleSTTV2(
        project_id=GCP_PROJECT_ID,
        model="chirp_3",
        language="ko-KR",
        use_server_vad=True,
        credentials_dir=CREDENTIALS_DIR,
    )
    stt.initialize()

    # 16kHz WAV 파일로 테스트
    audio_gen = read_wav_as_pcm_chunks("test_input_16k.wav")
    stt.start_streaming(audio_gen)

    success, transcript = stt.wait_for_result(timeout=30.0)
    print(f"STT 결과 (success={success}): {transcript}")

    stt.finalize()


if __name__ == "__main__":
    main()
```

---

## 참고

- **GCP 쿼터 확인:** Speech-to-Text V2는 동시 스트리밍 연결 수 제한이 있습니다. 트래픽이 많은 경우 여러 서비스 계정 키를 `credentials/` 디렉터리에 배치하여 Round-Robin으로 분산하세요.
- **Chirp3-HD 스트리밍 제한:** `streaming_audio_config` 옵션을 지원하지 않으므로, 스트리밍 TTS 시 샘플레이트는 기본값(24kHz)으로 고정됩니다.
- **telephony 모델:** 전화 품질 오디오에 최적화되어 있으며, 일반 마이크 녹음에는 `chirp_3` 사용을 권장합니다.
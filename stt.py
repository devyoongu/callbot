"""
callbot/stt.py — Google Cloud Speech-to-Text V2 스트리밍 인식기

google-stt-tts-guide.md 의 GoogleSTTV2 클래스 기반.
telephony 모델: 전화 품질 오디오(8kHz) 최적화, 16kHz PCM 입력 지원.
"""
import threading
from typing import Optional, Tuple

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

import numpy as np
from credentials import load_google_credentials
import config as cfg


class GoogleSTTV2:
    """
    Google Cloud Speech-to-Text V2 스트리밍 인식기

    사용법:
        stt = GoogleSTTV2(project_id="your-gcp-project", model="telephony")
        stt.initialize()
        stt.start_streaming(audio_source_generator)
        success, transcript = stt.wait_for_result()
        stt.finalize()
    """

    # 모델별 설정
    # telephony: 전화 품질 오디오 최적화 (16kHz로 업샘플된 8kHz 전화음 입력)
    MODEL_CONFIG = {
        "chirp_3": {
            "region": "us",
            "endpoint": "us-speech.googleapis.com",
            "sample_rate": 16000,
        },
        "telephony": {
            "region": "global",
            "endpoint": "speech.googleapis.com",
            "sample_rate": 16000,  # 8kHz를 2x 업샘플해서 입력
        },
    }

    def __init__(
        self,
        project_id: str,
        model: str = "telephony",
        language: str = "ko-KR",
        use_server_vad: bool = True,
        credentials_dir: str = None,
    ):
        self.project_id = project_id
        self.model = model
        self.language = language
        self.use_server_vad = use_server_vad
        self.credentials_dir = credentials_dir

        model_cfg = self.MODEL_CONFIG.get(model, self.MODEL_CONFIG["telephony"])
        self.region      = model_cfg["region"]
        self.endpoint    = model_cfg["endpoint"]
        self.sample_rate = model_cfg["sample_rate"]

        self.client: Optional[SpeechClient] = None
        self._result_event  = threading.Event()
        self._final_transcript = ""
        self._stt_error: Optional[str] = None
        self._stop = False
        self._streaming_thread: Optional[threading.Thread] = None

        self.speech_started  = False
        self.eos_done        = False
        self._process_audio  = True

    # ── 공개 API ──────────────────────────────────────────────────────

    def initialize(self):
        """클라이언트 초기화. start_streaming() 전에 반드시 호출."""
        credentials = load_google_credentials(self.credentials_dir)
        self.client = SpeechClient(
            client_options=ClientOptions(api_endpoint=self.endpoint),
            credentials=credentials,
        )
        print(f"[STT] Client ready (model={self.model}, endpoint={self.endpoint})")

    def start_streaming(self, audio_source):
        """
        백그라운드 스레드에서 스트리밍 인식 시작

        Args:
            audio_source: bytes를 yield하는 generator.
                          각 청크는 16kHz LINEAR16 PCM 이어야 합니다.
                          권장 청크 크기: 4000 bytes (125ms)
        """
        self._stop              = False
        self._process_audio     = True
        self._final_transcript  = ""
        self._stt_error         = None
        self._result_event.clear()
        self.speech_started     = False
        self.eos_done           = False

        self._audio_source = audio_source
        self._streaming_thread = threading.Thread(
            target=self._run_streaming, daemon=True
        )
        self._streaming_thread.start()

    def wait_for_result(self, timeout: float = 30.0) -> Tuple[bool, str]:
        """
        STT 결과 대기

        Returns:
            (success, transcript)
            transcript == "non_voice"  → 음성 없음
            transcript == "timeout"    → 타임아웃
        """
        done = self._result_event.wait(timeout=timeout)
        if not done:
            return False, "timeout"
        if self._stt_error:
            return False, self._stt_error
        return True, self._final_transcript

    def stop(self):
        """스트리밍 강제 중지"""
        self._stop          = True
        self._process_audio = False

    def finalize(self):
        """리소스 정리"""
        self.stop()
        if self._streaming_thread and self._streaming_thread.is_alive():
            self._streaming_thread.join(timeout=3.0)
        self.client = None

    # ── 내부 구현 ─────────────────────────────────────────────────────

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
                        print(f"[STT] Final: '{self._final_transcript}'")
                        self._result_event.set()
                        return

        except Exception as e:
            print(f"[STT] Error: {e}")
            self._stt_error = str(e)
            self._result_event.set()

    def _audio_generator(self):
        """audio_source에서 오디오를 읽어 StreamingRecognizeRequest로 yield"""
        # Client VAD 초기화 (use_server_vad=False일 때만 사용)
        vad = None
        if not self.use_server_vad and WEBRTC_VAD_AVAILABLE:
            vad = webrtcvad.Vad(2)

        VAD_FRAME_SIZE    = 640      # 20ms frame at 16kHz (640 bytes = 320 samples × 2)
        RMS_THRESHOLD     = 200
        SILENCE_THRESHOLD = 5

        speech_detected  = False
        silence_frames   = 0
        high_energy_count = 0

        for chunk in self._audio_source:
            if self._stop:
                break
            if not chunk:
                continue

            # Client VAD (use_server_vad=False 일 때)
            if vad and len(chunk) == 4000:
                is_speech = self._client_vad_check(
                    vad, chunk, VAD_FRAME_SIZE, RMS_THRESHOLD
                )
                if is_speech:
                    high_energy_count += 1
                    if high_energy_count >= 2 and not speech_detected:
                        speech_detected  = True
                        self.speech_started = True
                        silence_frames   = 0
                else:
                    high_energy_count = 0

                if speech_detected:
                    if not is_speech:
                        silence_frames += 1
                        if silence_frames >= SILENCE_THRESHOLD:
                            self.eos_done        = True
                            self._process_audio  = False
                            yield cloud_speech_types.StreamingRecognizeRequest(audio=chunk)
                            break
                    else:
                        silence_frames = 0

            yield cloud_speech_types.StreamingRecognizeRequest(audio=chunk)

    def _client_vad_check(self, vad, chunk: bytes, frame_size: int, rms_threshold: float) -> bool:
        """WebRTC VAD + RMS 에너지 이중 판단"""
        num_frames   = len(chunk) // frame_size
        speech_frames = 0
        for i in range(num_frames):
            frame = chunk[i * frame_size:(i + 1) * frame_size]
            if len(frame) == frame_size and vad.is_speech(frame, self.sample_rate):
                speech_frames += 1
        vad_positive    = speech_frames >= (num_frames * 0.66)
        samples         = np.frombuffer(chunk, dtype=np.int16)
        rms             = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        energy_positive = rms >= rms_threshold
        return vad_positive and energy_positive

    def _handle_vad_event(self, event_type):
        """Server VAD 이벤트 핸들러"""
        SpeechEventType = cloud_speech_types.StreamingRecognizeResponse.SpeechEventType

        if event_type == SpeechEventType.SPEECH_ACTIVITY_BEGIN:
            print("[STT] VAD: Speech BEGIN")
            self.speech_started = True

        elif event_type in (
            SpeechEventType.SPEECH_ACTIVITY_END,
            SpeechEventType.END_OF_SINGLE_UTTERANCE,
        ):
            print("[STT] VAD: Speech END / EOS")
            self.eos_done       = True
            self._process_audio = False
            self._stop          = True


def create_stt(credentials_dir: str = None) -> GoogleSTTV2:
    """매 대화 턴마다 새 GoogleSTTV2 인스턴스 생성"""
    return GoogleSTTV2(
        project_id=cfg.GCP_PROJECT_ID,
        model=cfg.GCP_STT_MODEL,
        language=cfg.GCP_STT_LANGUAGE,
        use_server_vad=True,
        credentials_dir=credentials_dir or cfg.CREDENTIALS_DIR,
    )

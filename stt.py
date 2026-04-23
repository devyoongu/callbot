"""
callbot/stt.py — Google Cloud Speech-to-Text V2 스트리밍 인식기

google-stt-tts-guide.md 의 GoogleSTTV2 클래스 기반.
telephony 모델: 전화 품질 오디오(8kHz) 최적화, 16kHz PCM 입력 지원.
"""
import threading
import time
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

        self.speech_started           = False
        self.speech_started_time: Optional[float] = None
        self.eos_done                 = False
        self.eos_time: Optional[float] = None
        self._process_audio           = True
        self._has_interim_content     = False

        # 폴링 루프용 소비형 플래그 (get_speech_started / get_and_consume_eos 에서 사용)
        self._speech_started_consumed = False
        self._eos_consumed            = False
        # 폴링 루프에서 바쁜 대기(busy-wait) 없이 final result를 기다리기 위한 이벤트
        self._transcript_ready        = threading.Event()

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
        self._stop                    = False
        self._process_audio           = True
        self._final_transcript        = ""
        self._stt_error               = None
        self._result_event.clear()
        self.speech_started           = False
        self.speech_started_time      = None
        self.eos_done                 = False
        self.eos_time                 = None
        self._has_interim_content     = False
        self._speech_started_consumed = False
        self._eos_consumed            = False
        self._transcript_ready.clear()

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

    def get_speech_started(self) -> Tuple[bool, Optional[float]]:
        """
        음성 시작 여부 반환 (소비형 — 1회 반환 후 리셋).
        robi-t-callbot SpeechDetectionHandler.poll_speech_started() 대응.
        """
        if self.speech_started and not self._speech_started_consumed:
            self._speech_started_consumed = True
            return True, self.speech_started_time
        return False, None

    def get_and_consume_eos(self) -> Tuple[bool, Optional[float]]:
        """
        EOS 감지 여부 반환 및 소비 (1회 반환 후 리셋).
        robi-t-callbot SpeechDetectionHandler.poll_eos() 대응.
        """
        if self.eos_done and not self._eos_consumed:
            self._eos_consumed = True
            return True, self.eos_time
        return False, None

    def get_final_transcript(self) -> str:
        """
        result_event가 세트된 경우 final transcript 반환, 아니면 빈 문자열.
        robi-t-callbot SpeechDetectionHandler.poll_transcript() 대응.
        """
        if self._result_event.is_set() and not self._stt_error:
            return self._final_transcript
        return ""

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

            # robi-t-callbot과 동일한 간단 설정 — VoiceActivityTimeout 미사용
            # (Google 기본값 사용, 커스텀 timeout이 오히려 부작용 유발 가능)
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

                # ① results 먼저 처리 — interim이 있으면 _has_interim_content 설정
                #    (VAD 이벤트보다 먼저 처리해야 같은 response에 SPEECH_END와
                #     interim result가 함께 올 때 race condition이 없음)
                for result in response.results:
                    if not result.alternatives:
                        continue
                    transcript = result.alternatives[0].transcript

                    if result.is_final:
                        self._final_transcript = transcript.strip() or "non_voice"
                        print(f"[STT] Final: '{self._final_transcript}'")
                        self._stop = True
                        self._transcript_ready.set()
                        self._result_event.set()
                        return
                    elif transcript.strip():
                        if not self._has_interim_content:
                            print(f"[STT] Interim: '{transcript.strip()[:30]}'")
                            # 첫 번째 interim 결과로 speech_started 보장
                            # (client VAD가 감지하지 못한 경우 fallback)
                            if not self.speech_started:
                                self.speech_started = True
                                self.speech_started_time = time.time()
                        self._has_interim_content = True

                # ② VAD 이벤트 처리 — results 이후에 평가
                if self.use_server_vad and response.speech_event_type:
                    self._handle_vad_event(response.speech_event_type)

            # 스트림 정상 종료: responses 소진 후 final result가 없으면 non_voice 처리
            # (음성이 너무 짧아서 Google STT가 인식 불가한 경우)
            if not self._result_event.is_set():
                print("[STT] Stream ended without final result → non_voice")
                self._final_transcript = "non_voice"
                self._transcript_ready.set()
                self._result_event.set()

        except Exception as e:
            print(f"[STT] Error: {e}")
            self._stt_error = str(e)
            self._result_event.set()

    def _audio_generator(self):
        """
        audio_source에서 오디오를 읽어 StreamingRecognizeRequest로 yield.

        robi-t-callbot google_stt_v2_driver.audio_generator() 패턴:
          - EOS 시 flush 없이 즉시 break → gRPC clean EOF → Google is_final 반환
          - Client VAD: WebRTC VAD + RMS 에너지 이중 판단 (server VAD의 fallback)
          - _has_interim_content 가드로 echo/noise 오탐 방지
        """
        # Client VAD: server VAD의 보완/fallback으로 항상 초기화
        vad = None
        if WEBRTC_VAD_AVAILABLE:
            vad = webrtcvad.Vad(2)

        VAD_FRAME_SIZE    = 640   # 20ms frame at 16kHz (640 bytes = 320 samples × 2)
        RMS_THRESHOLD     = 200   # robi-t-callbot 동일 (노이즈 < 192, 발화 > 1886)
        SILENCE_THRESHOLD = 5     # 5 × 125ms = 625ms — robi-t-callbot 동일

        speech_detected   = False
        silence_frames    = 0
        high_energy_count = 0

        for chunk in self._audio_source:
            if self._stop:
                break

            # EOS 커밋 후 즉시 종료 (flush 없음 — robi-t-callbot 패턴)
            # gRPC가 clean end-of-stream 전송 → Google이 버퍼된 오디오로 is_final 반환
            if not self._process_audio:
                break

            if not chunk:
                continue

            # ── Client-side VAD (robi-t-callbot 패턴) ──────────────────────────
            # WebRTC VAD + RMS 이중 판단으로 발화 감지 및 EOS 커밋.
            # server VAD가 이미 EOS를 커밋했으면 (eos_done=True) 건너뜀.
            if vad and len(chunk) == 4000 and not self.eos_done:
                is_speech = self._client_vad_check(vad, chunk, VAD_FRAME_SIZE, RMS_THRESHOLD)

                if is_speech:
                    high_energy_count += 1
                    if high_energy_count >= 2 and not speech_detected:
                        speech_detected = True
                        silence_frames  = 0
                        if not self.speech_started:
                            self.speech_started      = True
                            self.speech_started_time = time.time()
                else:
                    high_energy_count = 0
                    if speech_detected:
                        silence_frames += 1
                        if silence_frames >= SILENCE_THRESHOLD:
                            if self._has_interim_content:
                                # Google interim 확인 → EOS 커밋 후 즉시 종료
                                print("[STT] Client VAD: EOS (625ms silence)")
                                self.eos_done       = True
                                self.eos_time       = time.time()
                                self._process_audio = False
                                break  # 즉시 종료 (robi-t-callbot 패턴)
                            else:
                                # interim 없음 = echo/noise → 리셋 후 계속 청취
                                print("[STT] Client VAD: 625ms silence, no interim — echo reset")
                                speech_detected   = False
                                silence_frames    = 0
                                high_energy_count = 0

                if speech_detected and is_speech:
                    silence_frames = 0
            # ────────────────────────────────────────────────────────────────────

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
            self.speech_started      = True
            self.speech_started_time = time.time()
            # _has_interim_content 리셋 안 함:
            # 문장 중간 짧은 쉼 → BEGIN → END 반복 시 interim 상태 보존 필요.
            # 리셋하면 "거기 [쉼] 위치 어디야" 발화 시 두 번째 BEGIN에서 상태 초기화되어
            # client VAD가 EOS를 발화하지 못하고 15초 timeout 발생.

        elif event_type in (
            SpeechEventType.SPEECH_ACTIVITY_END,
            SpeechEventType.END_OF_SINGLE_UTTERANCE,
        ):
            if self._has_interim_content:
                # 서버 VAD는 단어 사이 쉬는 구간(~300ms)에도 END를 발화하여 문장이 잘림.
                # generator를 멈추지 않고 계속 오디오 전송 — client VAD가 EOS 담당.
                # (log 증거: END 직후 두 번째 BEGIN이 오는 것 = 사용자가 계속 말하는 중)
                print("[STT] VAD: Speech END — real speech, continuing (client VAD handles EOS)")
            else:
                # interim 없음 = 에코/잡음 → speech_started 리셋 후 계속 청취
                print("[STT] VAD: Speech END — no content (echo/noise), continuing")
                self.speech_started      = False
                self.speech_started_time = None


def create_stt(credentials_dir: str = None) -> GoogleSTTV2:
    """매 대화 턴마다 새 GoogleSTTV2 인스턴스 생성"""
    return GoogleSTTV2(
        project_id=cfg.GCP_PROJECT_ID,
        model=cfg.GCP_STT_MODEL,
        language=cfg.GCP_STT_LANGUAGE,
        # use_server_vad=True: telephony 모델에 필수 (False 시 인식 결과 미반환)
        # EOS 시 flush 없이 generator 즉시 break → gRPC clean EOF → Google is_final
        # echo 방지는 _has_interim_content 가드가 담당
        use_server_vad=True,
        credentials_dir=credentials_dir or cfg.CREDENTIALS_DIR,
    )

"""
callbot/stt.py — Google Cloud Speech-to-Text V2 스트리밍 인식기

260423_google-stt-tts-guide.md 의 GoogleSTTV2 클래스 기반.
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

import numpy as np
from credentials import load_google_credentials
from stt_phrases import build_adaptation, supports_adaptation
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
    # telephony: 전화 품질 오디오 최적화 (narrowband, 8kHz native 입력)
    # chirp_3:   wideband (16kHz)
    MODEL_CONFIG = {
        "chirp_3": {
            "region": "us",
            "endpoint": "us-speech.googleapis.com",
            "sample_rate": 16000,
        },
        "chirp_2": {
            # us-central1 리전에서 adaptation+boost 지원. chirp_3 와 finalization
            # 동작이 다를 수 있어 뒤쪽 잘림 회귀에 대안으로 시험.
            "region": "us-central1",
            "endpoint": "us-central1-speech.googleapis.com",
            "sample_rate": 16000,
        },
        "telephony": {
            "region": "global",
            "endpoint": "speech.googleapis.com",
            "sample_rate": 8000,  # 모델 훈련 분포와 일치 — upsample 시 4–8kHz 가 0-에너지로 차서 분포 어긋남
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
        # 가장 최근 server VAD SPEECH_END 시각 — client VAD EOS 가 트리거되지 않은
        # 경우 (Google STT 가 더 빨리 is_final 반환) latency 측정용 fallback.
        self.last_speech_end_time: Optional[float] = None
        self.final_time: Optional[float] = None  # is_final 도착 시각 (EOS→Final latency 계산용)
        self._process_audio           = True
        self._has_interim_content     = False

        # 폴링 루프용 소비형 플래그 (get_speech_started / get_and_consume_eos 에서 사용)
        self._speech_started_consumed = False
        self._eos_consumed            = False
        # 폴링 루프에서 바쁜 대기(busy-wait) 없이 final result를 기다리기 위한 이벤트
        self._transcript_ready        = threading.Event()

        # 진단용 — turn 별로 reset (start_streaming 에서 0/""로)
        self._last_interim_text       = ""
        self._yielded_chunk_count     = 0
        self._vad_begin_count         = 0
        self._vad_end_count           = 0

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
        self.last_speech_end_time     = None
        self.final_time               = None
        self._has_interim_content     = False
        self._speech_started_consumed = False
        self._eos_consumed            = False
        self._transcript_ready.clear()
        self._last_interim_text       = ""
        self._yielded_chunk_count     = 0
        self._vad_begin_count         = 0
        self._vad_end_count           = 0

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
        """
        cfg.GCP_STT_MODE 에 따라 sync 또는 streaming 모드로 분기. 외부 호출자
        (start_streaming) 는 method 이름만 호출하면 됨.
        """
        if cfg.GCP_STT_MODE == "streaming":
            return self._run_stream_accumulate()
        return self._run_sync()

    def _run_sync(self):
        """
        sync (non-streaming) Recognize 기반 인식 — default.

        chirp_3 의 streaming 모드는 발화 중간 짧은 휴지에도 is_final 을 발행해
        뒤쪽이 통째로 잘림. 같은 wav 라도:
          streaming chirp_3:  "파견 보안관제요." (21.7%)
          sync     chirp_3:   "파견보안관제, 원격관제 차이는 무엇인가요?" (~97%)
        — sync 는 audio 전체를 한 번에 평가하므로 mid-utterance EOS 판정 없음.

        flow:
          1) audio_source 에서 청크 받아 buffer 누적 + client VAD 로 발화/silence
             감지 (기존 streaming 모드의 VAD 로직과 동일).
          2) silence_frames >= SILENCE_THRESHOLD (500ms) 도달 시 EOS 커밋 +
             버퍼링 종료.
          3) 버퍼 audio 를 sync recognize() 1회 호출.
          4) response.results[*].alternatives[0].transcript 들 join 해 final 로.
        """
        try:
            recognizer_path = (
                f"projects/{self.project_id}/locations/{self.region}/recognizers/_"
            )
            config_kwargs = dict(
                explicit_decoding_config=cloud_speech_types.ExplicitDecodingConfig(
                    encoding=cloud_speech_types.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.sample_rate,
                    audio_channel_count=1,
                ),
                language_codes=[self.language],
                model=self.model,
            )
            if supports_adaptation(self.model):
                config_kwargs["adaptation"] = build_adaptation()
                print(f"[STT] adaptation enabled for model={self.model}")
            recognition_config = cloud_speech_types.RecognitionConfig(**config_kwargs)

            # ── Phase 1: audio buffering + EOS 감지 ─────────────────────────
            audio_buf = bytearray()
            self._buffer_audio_with_vad(audio_buf)

            # ── Phase 2: 발화가 있었으면 sync recognize ─────────────────────
            if not self.speech_started or len(audio_buf) == 0:
                print(f"[STT] No speech detected → non_voice (audio_buf={len(audio_buf)} bytes)")
                self._final_transcript = "non_voice"
                self._transcript_ready.set()
                self._result_event.set()
                return

            recognize_t0 = time.time()
            request = cloud_speech_types.RecognizeRequest(
                recognizer=recognizer_path,
                config=recognition_config,
                content=bytes(audio_buf),
            )
            response = self.client.recognize(request=request, timeout=30.0)
            api_ms = (time.time() - recognize_t0) * 1000

            transcripts = []
            for result in response.results:
                if result.alternatives:
                    t = result.alternatives[0].transcript.strip()
                    if t:
                        transcripts.append(t)

            self.final_time = time.time()
            if transcripts:
                self._final_transcript = " ".join(transcripts)
            else:
                self._final_transcript = "non_voice"

            eos_t = self.eos_time or self.last_speech_end_time
            latency_str = ""
            if eos_t is not None:
                latency_ms = (self.final_time - eos_t) * 1000
                src = "clientEOS" if self.eos_time else "serverEND"
                latency_str = f" ({src}→Final {latency_ms:.0f}ms)"
            audio_ms = (len(audio_buf) // 2) / self.sample_rate * 1000
            print(f"[STT] Final (sync, api={api_ms:.0f}ms): '{self._final_transcript}'{latency_str}")
            print(
                f"[STT] Turn summary: audio={audio_ms:.0f}ms, "
                f"vad_begin={self._vad_begin_count}, vad_end={self._vad_end_count}, "
                f"final_len={len(self._final_transcript)}"
            )

            self._transcript_ready.set()
            self._result_event.set()

        except Exception as e:
            print(f"[STT] Error: {e}")
            self._stt_error = str(e)
            self._result_event.set()

    def _run_stream_accumulate(self):
        """
        streaming_recognize 사용 + is_final 누적 (회귀 시도).

        chirp_3 가 mid-utterance is_final 을 발행해도 stream 을 끊지 않고 audio
        를 계속 보내면서 들어오는 모든 is_final 텍스트를 누적해 join. ⤳ 발화
        후반부가 별 utterance 로 인식되어 추가 is_final 로 도착하면 자연스럽게
        합쳐짐. 첫 is_final 에서 끊던 기존 streaming 의 trailing-cutoff 회피.

        latency 이점: 마지막 is_final 은 사용자 발화 직후 (~수백 ms) 에 도착하므로
        sync 의 +1.85s VAD+API call 보다 빠르다 — 단, mid-utterance is_final 들이
        실제로 audio 의 모든 segment 를 커버해야 join 결과가 정확.

        flow:
          1) request_iter() 가 첫 request 로 streaming_config 보내고,
             이후 audio_source 청크를 StreamingRecognizeRequest(audio=...) 로 yield.
          2) 동시에 client VAD (RMS) 로 silence 추적 → SILENCE_THRESHOLD 시 EOS
             커밋 + iterator 종료 (stream close).
          3) responses 루프에서 is_final transcript 들 누적. join 후 final 채택.
        """
        try:
            recognizer_path = (
                f"projects/{self.project_id}/locations/{self.region}/recognizers/_"
            )
            config_kwargs = dict(
                explicit_decoding_config=cloud_speech_types.ExplicitDecodingConfig(
                    encoding=cloud_speech_types.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.sample_rate,
                    audio_channel_count=1,
                ),
                language_codes=[self.language],
                model=self.model,
            )
            if supports_adaptation(self.model):
                config_kwargs["adaptation"] = build_adaptation()
                print(f"[STT] adaptation enabled for model={self.model}")
            recognition_config = cloud_speech_types.RecognitionConfig(**config_kwargs)

            streaming_config = cloud_speech_types.StreamingRecognitionConfig(
                config=recognition_config,
                streaming_features=cloud_speech_types.StreamingRecognitionFeatures(
                    interim_results=True,
                    enable_voice_activity_events=False,
                ),
            )

            def request_iter():
                # 첫 request: recognizer + streaming_config
                yield cloud_speech_types.StreamingRecognizeRequest(
                    recognizer=recognizer_path,
                    streaming_config=streaming_config,
                )

                # client VAD constants — sync 와 동일
                EXPECTED_CHUNK_BYTES = int(self.sample_rate * 0.125 * 2)
                RMS_THRESHOLD     = 200
                SILENCE_THRESHOLD = 4   # 4 × 125ms = 500ms

                speech_detected   = False
                silence_frames    = 0
                high_energy_count = 0

                for chunk in self._audio_source:
                    if self._stop:
                        return
                    if not chunk:
                        continue
                    self._yielded_chunk_count += 1
                    yield cloud_speech_types.StreamingRecognizeRequest(audio=chunk)

                    if len(chunk) != EXPECTED_CHUNK_BYTES:
                        continue
                    samples = np.frombuffer(chunk, dtype=np.int16)
                    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
                    is_speech = rms >= RMS_THRESHOLD

                    if is_speech:
                        high_energy_count += 1
                        if high_energy_count >= 2 and not speech_detected:
                            speech_detected = True
                            silence_frames  = 0
                            if not self.speech_started:
                                self.speech_started      = True
                                self.speech_started_time = time.time()
                                print(f"[STT] Speech started (rms={rms:.0f})")
                        if speech_detected:
                            silence_frames = 0
                    else:
                        high_energy_count = 0
                        if speech_detected:
                            silence_frames += 1
                            if silence_frames >= SILENCE_THRESHOLD:
                                print(
                                    f"[STT] EOS triggered (streaming, "
                                    f"{silence_frames}×125ms={silence_frames*125}ms silence) — "
                                    f"closing stream"
                                )
                                self.eos_done = True
                                self.eos_time = time.time()
                                return

            responses = self.client.streaming_recognize(
                requests=request_iter(),
                timeout=30.0,
            )

            transcripts = []
            last_interim = ""
            for response in responses:
                if self._stop:
                    break
                for result in response.results:
                    if not result.alternatives:
                        continue
                    t = result.alternatives[0].transcript.strip()
                    if result.is_final:
                        if t:
                            transcripts.append(t)
                            print(f"[STT] is_final: '{t}'")
                    elif t:
                        last_interim = t

            if not self.speech_started:
                print(f"[STT] No speech detected → non_voice")
                self._final_transcript = "non_voice"
                self._transcript_ready.set()
                self._result_event.set()
                return

            if transcripts:
                self._final_transcript = " ".join(transcripts)
            elif last_interim:
                self._final_transcript = last_interim
                print(f"[STT] No is_final — using last interim: '{last_interim}'")
            else:
                self._final_transcript = "non_voice"

            self.final_time = time.time()
            eos_t = self.eos_time or self.last_speech_end_time
            latency_str = ""
            if eos_t is not None:
                latency_ms = (self.final_time - eos_t) * 1000
                src = "clientEOS" if self.eos_time else "serverEND"
                latency_str = f" ({src}→Final {latency_ms:.0f}ms)"
            print(
                f"[STT] Final (streaming, accumulated {len(transcripts)} is_final): "
                f"'{self._final_transcript}'{latency_str}"
            )
            print(
                f"[STT] Turn summary: yielded={self._yielded_chunk_count}, "
                f"is_final_count={len(transcripts)}, final_len={len(self._final_transcript)}"
            )

            self._transcript_ready.set()
            self._result_event.set()

        except Exception as e:
            print(f"[STT] Error (streaming): {e}")
            self._stt_error = str(e)
            self._result_event.set()

    def _buffer_audio_with_vad(self, audio_buf: bytearray):
        """
        audio_source 에서 청크를 받아 audio_buf 에 누적. client VAD 로 발화 감지
        및 silence (500ms) 시 EOS 커밋. EOS 후 추가로 EOS_TRAIL_CHUNKS 만큼 더
        버퍼해 발화 끝부분 silence 도 sync recognize 에 포함 (chirp_3 가 자연
        종료 audio 를 받으면 인식 정확도 향상).

        speech_started 전의 chunk 는 audio_buf 에 누적하지 않음 — leading silence
        나 봇 응답 echo 잔재가 sync API 의 인식 컨텍스트를 흐리는 회귀 차단.
        speech_started 직전 PRE_SPEECH_RING_CHUNKS 만큼은 ring buffer 로 보존 후
        speech 시작 시 audio_buf 앞에 prepend (onset 음절 손실 방지).

        loop 종료 시 EOS 가 트리거 안 됐으면 audio_buf 끝에 trailing silence 를
        명시적으로 padding — audio_source 의 deadline 으로 강제 종료된 경우에도
        sync API 가 자연 종료된 utterance 로 인식하게 함 (Turn 0 잘림 fix).
        """
        EXPECTED_CHUNK_BYTES = int(self.sample_rate * 0.125 * 2)  # 125ms — 2000@8k / 4000@16k
        RMS_THRESHOLD        = 200
        SILENCE_THRESHOLD    = 4     # 4 × 125ms = 500ms
        EOS_TRAIL_CHUNKS     = 2     # 250ms — EOS 후 trailing audio 추가 버퍼
        PRE_SPEECH_RING_CHUNKS = 3   # 375ms — speech 시작 직전 보존 (onset 보호)
        TRAIL_PADDING_CHUNKS = 4     # 500ms — EOS-less 종료 시 명시적 silence padding

        # webrtcvad 의 setuptools 82+ 호환 이슈 (pkg_resources 제거) 로 의존
        # 제거. RMS-only 판정. callbot 환경 (TTS-as-mic + telephony codec) 에서
        # 잡음 < 192, 발화 > 1886 의 margin 큼 (robi-t-callbot 측정값) → 충분.

        speech_detected   = False
        silence_frames    = 0
        high_energy_count = 0
        eos_buffered      = 0
        eos_committed     = False
        pre_speech_ring: "list[bytes]" = []  # speech 시작 전 마지막 N chunks

        for chunk in self._audio_source:
            if self._stop:
                break
            if not chunk:
                continue
            self._yielded_chunk_count += 1

            # EOS 후 trailing audio 모음
            if eos_committed:
                audio_buf.extend(chunk)
                eos_buffered += 1
                if eos_buffered >= EOS_TRAIL_CHUNKS:
                    break
                continue

            # speech 시작 전: ring buffer 에만 보관 (audio_buf 에 누적 안 함)
            if not speech_detected:
                pre_speech_ring.append(chunk)
                if len(pre_speech_ring) > PRE_SPEECH_RING_CHUNKS:
                    pre_speech_ring.pop(0)

            if len(chunk) == EXPECTED_CHUNK_BYTES:
                samples = np.frombuffer(chunk, dtype=np.int16)
                rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
                is_speech = rms >= RMS_THRESHOLD

                if is_speech:
                    high_energy_count += 1
                    if high_energy_count >= 2 and not speech_detected:
                        speech_detected = True
                        silence_frames  = 0
                        # ring buffer 의 직전 chunks 를 audio_buf 앞에 prepend
                        for c in pre_speech_ring:
                            audio_buf.extend(c)
                        pre_speech_ring.clear()
                        if not self.speech_started:
                            self.speech_started      = True
                            self.speech_started_time = time.time()
                            print(f"[STT] Speech started (rms={rms:.0f})")
                    if speech_detected:
                        silence_frames = 0
                else:
                    high_energy_count = 0
                    if speech_detected:
                        silence_frames += 1
                        if silence_frames >= SILENCE_THRESHOLD:
                            audio_ms = (len(audio_buf) // 2) / self.sample_rate * 1000
                            print(
                                f"[STT] EOS triggered ({silence_frames}×125ms="
                                f"{silence_frames*125}ms silence, audio={audio_ms:.0f}ms) — "
                                f"buffering {EOS_TRAIL_CHUNKS} trailing chunks then sync recognize"
                            )
                            eos_committed = True
                            self.eos_done = True
                            self.eos_time = time.time()

            # speech_detected 후의 chunk 는 audio_buf 에 누적 (EOS 트리거 후엔 위
            # if eos_committed 분기에서 처리됨)
            if speech_detected and not eos_committed:
                audio_buf.extend(chunk)

        # audio_source 가 EOS 트리거 없이 종료된 경우 (deadline / stop) — 발화는
        # 됐는데 trailing silence 가 부족하면 sync API 가 발화 후반부를 미완성
        # utterance 로 보고 잘라낼 수 있음. 명시적 silence padding 추가.
        if speech_detected and not eos_committed:
            silence_chunk = b"\x00" * EXPECTED_CHUNK_BYTES
            for _ in range(TRAIL_PADDING_CHUNKS):
                audio_buf.extend(silence_chunk)
            self.eos_done = True
            self.eos_time = time.time()
            audio_ms = (len(audio_buf) // 2) / self.sample_rate * 1000
            print(
                f"[STT] EOS missed — trail-padded {TRAIL_PADDING_CHUNKS}×125ms "
                f"silence (audio={audio_ms:.0f}ms) for sync recognize"
            )

    def _handle_vad_event_unused(self, event_type):
        """Server VAD 이벤트 핸들러 — 기존 streaming 모드 잔재 (sync 전환 후 dead).
        삭제하지 않고 _unused 접미사로 보존: 추후 streaming 회귀 시 참조용.
        """
        SpeechEventType = cloud_speech_types.StreamingRecognizeResponse.SpeechEventType

        if event_type == SpeechEventType.SPEECH_ACTIVITY_BEGIN:
            self._vad_begin_count += 1
            print(f"[STT] VAD: Speech BEGIN (#{self._vad_begin_count})")
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
            self._vad_end_count += 1
            # latency 측정용 — interim 유무와 무관하게 최신 END 시각 기록.
            # chirp_3 는 interim 거의 emit 하지 않아 _has_interim_content 가 False
            # 인 상태로 END 가 도착함. 이 시각이 없으면 STT latency 메트릭이 n=0
            # 으로 표시됨. 마지막 END 가 진짜 EOS 후보이므로 매번 갱신.
            self.last_speech_end_time = time.time()
            if self._has_interim_content:
                # 서버 VAD는 단어 사이 쉬는 구간(~300ms)에도 END를 발화하여 문장이 잘림.
                # generator를 멈추지 않고 계속 오디오 전송 — client VAD가 EOS 담당.
                # (log 증거: END 직후 두 번째 BEGIN이 오는 것 = 사용자가 계속 말하는 중)
                print(f"[STT] VAD: Speech END (#{self._vad_end_count}) — real speech, continuing (client VAD handles EOS)")
            else:
                # interim 없음 = telephony 에코/잡음 또는 chirp_3 의 정상 EOS.
                # speech_started 만 리셋 (echo 방지) — last_speech_end_time 은 위에서
                # 이미 기록됐으므로 latency 측정에 영향 없음.
                print(f"[STT] VAD: Speech END (#{self._vad_end_count}) — no interim (chirp_3 silent-final or echo), continuing")
                self.speech_started      = False
                self.speech_started_time = None


def create_stt(credentials_dir: str = None):
    """매 대화 턴마다 새 STT 인스턴스 생성.

    cfg.STT_PROVIDER 에 따라 드라이버 분기:
      - "google"     → GoogleSTTV2 (GCP, default)
      - "qwen3-asr"  → Qwen3WebSocketSTT (사설 WS, 16kHz float32)
    """
    provider = cfg.STT_PROVIDER
    if provider == "qwen3-asr":
        from stt_qwen3 import Qwen3WebSocketSTT
        return Qwen3WebSocketSTT(
            url=cfg.QWEN_ASR_URL,
            language=cfg.QWEN_ASR_LANGUAGE,
            context=cfg.QWEN_ASR_CONTEXT,
        )
    if provider != "google":
        print(f"[STT] Unknown STT_PROVIDER={provider!r}, falling back to google")
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

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
        # chirp_3 의 premature is_final 대응 — 첫 is_final 시 즉시 stop 하지 않고
        # audio 를 더 흘려 후속 is_final 을 누적. _audio_generator 가 이 시각으로
        # post-final grace timer 체크.
        self._first_final_time: Optional[float] = None
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
        self._first_final_time        = None
        # 다중 stream 재시작 — chirp_3 의 premature finalization 우회용.
        # _audio_generator 가 이 flag 를 보고 break → gRPC EOF → 새 stream 시작.
        self._inner_stream_should_end = False

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
        Multi-stream chirp_3 finalization fix —

        chirp_3 는 발화 중간 짧은 휴지 만으로도 is_final 을 발행해 발화의 뒷부분이
        통째로 잘림 (실측: "파견보안관제와 원격관제의 차이는 무엇인가요" → "파견
        보안관제 원격"). VoiceActivityTimeout / enable_voice_activity_events 등
        client 측 setting 으로는 우회 안 됨 — 모델 내부 EOS 판정.

        해결: is_final 도착 시 stream 을 종료 (gRPC EOF) 하고 audio source 의
        남은 청크로 새 streaming_recognize 시작. transcripts 누적 후 join.
        audio source 는 한 번 생성된 generator 라 stream 사이에 iteration 위치가
        보존됨 → 청크 손실 없이 발화 끝까지 전부 STT 처리.

        outer 종료 조건:
          (a) self._stop True (외부 호출자가 강제 종료)
          (b) 한 stream 이 is_final 없이 자연 종료 (audio source 고갈)
          (c) MAX_ITER 안전 한계 도달
        """
        # MAX_ITER=2 — 첫 stream + 1회 restart. 3번째 stream 은 보통 트레일링
        # silence 만 처리해 환각 ("아.", "아니면 무엇입니까?") 만 추가하므로 차단.
        MAX_ITER = 2
        # 너무 짧은 final 은 환각/잡음 fragment 가능성 — 본문 누적 시 제외 (그러나
        # 첫 결과 (단 1개 final만 있는 short utterance) 는 손실 방지 위해 항상 보존).
        MIN_FRAGMENT_LEN = 3
        transcripts: list[str] = []
        iter_count = 0

        try:
            recognizer_path = (
                f"projects/{self.project_id}/locations/{self.region}/recognizers/_"
            )

            # 도메인 phrase boost — adaptation 지원 모델 (chirp_3) 에만 주입.
            # telephony global 은 V2 default recognizer 가 speech_adaptation_boost
            # 미지원 (실측: "Recognizer does not support feature" 400 에러). 모델별 분기.
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

            streaming_features = cloud_speech_types.StreamingRecognitionFeatures(
                interim_results=True,
                enable_voice_activity_events=self.use_server_vad,
            )
            streaming_config = cloud_speech_types.StreamingRecognitionConfig(
                config=recognition_config,
                streaming_features=streaming_features,
            )

            while not self._stop and iter_count < MAX_ITER:
                iter_count += 1
                self._inner_stream_should_end = False
                stream_got_final = False

                config_request = cloud_speech_types.StreamingRecognizeRequest(
                    recognizer=recognizer_path,
                    streaming_config=streaming_config,
                )

                def requests_gen():
                    yield config_request
                    yield from self._audio_generator()

                try:
                    responses = self.client.streaming_recognize(requests=requests_gen())

                    for response in responses:
                        if self._stop:
                            break

                        for result in response.results:
                            if not result.alternatives:
                                continue
                            transcript = result.alternatives[0].transcript

                            if result.is_final:
                                text = transcript.strip()
                                if text and text != "non_voice":
                                    # 첫 final 은 항상 보존 (전체 인식이 짧은 발화일
                                    # 가능성). 2번째 이상은 MIN_FRAGMENT_LEN 미만이면
                                    # 트레일링 silence 의 환각 가능성 높아 skip.
                                    accept = (iter_count == 1) or (len(text) >= MIN_FRAGMENT_LEN)
                                    if accept:
                                        transcripts.append(text)
                                    self.final_time = time.time()
                                    if self._first_final_time is None:
                                        self._first_final_time = self.final_time
                                    eos_t = self.eos_time or self.last_speech_end_time
                                    latency_str = ""
                                    if eos_t is not None:
                                        latency_ms = (self.final_time - eos_t) * 1000
                                        src = "clientEOS" if self.eos_time else "serverEND"
                                        latency_str = f" ({src}→Final {latency_ms:.0f}ms)"
                                    skip_str = "" if accept else " [SKIP — 환각 의심]"
                                    print(f"[STT] Final #{iter_count}: '{text}'{latency_str}{skip_str}")
                                # is_final 도착 → 이 stream 을 종료시키고 새 stream 시작.
                                stream_got_final = True
                                self._inner_stream_should_end = True
                                continue

                            text_stripped = transcript.strip()
                            if not text_stripped:
                                continue
                            if text_stripped != self._last_interim_text:
                                tag = "1st" if not self._has_interim_content else "..."
                                print(f"[STT] Interim ({tag}): '{text_stripped[:60]}'")
                                self._last_interim_text = text_stripped
                            if not self._has_interim_content:
                                if not self.speech_started:
                                    self.speech_started = True
                                    self.speech_started_time = time.time()
                            self._has_interim_content = True

                        if self.use_server_vad and response.speech_event_type:
                            self._handle_vad_event(response.speech_event_type)

                    # stream 종료. is_final 없으면 audio source 고갈 — outer loop 도 종료.
                    if not stream_got_final:
                        if iter_count == 1:
                            print(f"[STT] Stream #{iter_count} ended without is_final → non_voice")
                        else:
                            print(f"[STT] Stream #{iter_count} ended without is_final — done")
                        break

                except Exception as e:
                    print(f"[STT] Stream #{iter_count} error: {e}")
                    self._stt_error = str(e)
                    break

            # 모든 stream 종료. transcripts 합치기 (공백 join).
            # 동일 또는 prefix 중복 제거: chirp_3 가 가끔 같은 phrase 의 더 긴 버전을
            # 다음 stream 에서 처음부터 재출력하기도 함.
            joined = self._dedupe_join(transcripts)

            if joined:
                self._final_transcript = joined
                audio_ms  = self._yielded_chunk_count * 125
                last_len  = len(self._last_interim_text)
                final_len = len(self._final_transcript)
                truncated = "TRUNCATED" if final_len < last_len else "ok"
                print(
                    f"[STT] Combined ({iter_count} streams, {len(transcripts)} finals): "
                    f"'{self._final_transcript}'"
                )
                print(
                    f"[STT] Turn summary: audio={audio_ms}ms, "
                    f"vad_begin={self._vad_begin_count}, vad_end={self._vad_end_count}, "
                    f"streams={iter_count}, finals={len(transcripts)}, "
                    f"final_len={final_len} ({truncated})"
                )
            else:
                if not self._final_transcript:
                    self._final_transcript = "non_voice"
                print(f"[STT] No content recognized → '{self._final_transcript}'")

            self._transcript_ready.set()
            self._result_event.set()

        except Exception as e:
            print(f"[STT] Error: {e}")
            self._stt_error = str(e)
            self._result_event.set()

    @staticmethod
    def _dedupe_join(parts: list) -> str:
        """
        chirp_3 multi-stream finals 을 합치되, prefix 중복 제거.
        예: ['파견 보안관제 원격', '의 차이는 무엇인가요'] → '파견 보안관제 원격 의 차이는 무엇인가요'
        예: ['중소기업도 인증이', '중소기업도 인증이 꼭 필요한가요'] → '중소기업도 인증이 꼭 필요한가요'
        예: ['제로트러스트 보안이란', '무엇인가요'] → '제로트러스트 보안이란 무엇인가요'
        """
        if not parts:
            return ""
        out = parts[0].strip()
        for p in parts[1:]:
            p = p.strip()
            if not p:
                continue
            # p 가 out 의 끝 부분과 겹치면 (chirp_3 가 같은 구절을 반복하는 경우)
            # 또는 out 이 p 의 prefix 면 (다음 stream 이 더 긴 결과 출력) → 더 긴 것으로 교체.
            if out.startswith(p) or p in out:
                continue  # already covered
            if p.startswith(out):
                out = p
                continue
            # overlap 검사: out 의 suffix 와 p 의 prefix 중복 부분 트림
            max_overlap = min(len(out), len(p), 30)
            trim = 0
            for k in range(max_overlap, 0, -1):
                if out.endswith(p[:k]):
                    trim = k
                    break
            out = out + " " + (p[trim:] if trim else p)
        return out

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

        # 16-bit PCM 기준. sample_rate 따라 byte 수가 달라지므로 동적 계산.
        EXPECTED_CHUNK_BYTES = int(self.sample_rate * 0.125 * 2)  # 125ms 청크 — 2000@8k / 4000@16k
        VAD_FRAME_SIZE       = int(self.sample_rate * 0.020 * 2)  # 20ms 프레임 — 320@8k / 640@16k
        RMS_THRESHOLD     = 200   # robi-t-callbot 동일 (노이즈 < 192, 발화 > 1886). 진폭 기준이라 sr 무관.
        SILENCE_THRESHOLD = 5     # 5 × 125ms = 625ms — robi-t-callbot 동일
        EOS_FLUSH_CHUNKS  = 5     # EOS 트리거 후 추가로 흘릴 청크 수 (~625ms)
                                  # gRPC를 즉시 끊으면 Google이 final을 못 보내고
                                  # 스트림이 종료되어 non_voice 폴백되는 문제 회피.

        speech_detected   = False
        silence_frames    = 0
        high_energy_count = 0
        flush_remaining   = 0     # >0 이면 VAD 평가 건너뛰고 그대로 흘려보냄

        for chunk in self._audio_source:
            if self._stop:
                break

            # multi-stream restart — outer 루프가 새 stream 시작 하도록 generator break.
            # gRPC clean EOF → for response 루프 종료 → outer while 다음 iteration.
            if self._inner_stream_should_end:
                break

            # EOS 커밋 후 종료 (flush 5청크 완료 후에만 _process_audio=False가 됨)
            if not self._process_audio:
                break

            if not chunk:
                continue

            # ── EOS flush 진행 중: VAD 무시하고 그대로 yield ─────────────────
            # 이 단계는 audio_src.stop()이 아직 호출되지 않은 상태에서만 동작.
            # (외부 폴링 루프는 eos_done이 세트돼야 stop을 호출하므로,
            #  flush 동안 audio_src는 계속 청크를 yield 한다.)
            if flush_remaining > 0:
                flush_remaining -= 1
                yield cloud_speech_types.StreamingRecognizeRequest(audio=chunk)
                self._yielded_chunk_count += 1
                if flush_remaining == 0:
                    print(f"[STT] Client VAD: EOS flush done — committing")
                    self.eos_done       = True
                    self.eos_time       = time.time()
                    self._process_audio = False
                    break
                continue

            # ── Client-side VAD (robi-t-callbot 패턴) ──────────────────────────
            # WebRTC VAD + RMS 이중 판단으로 발화 감지 및 EOS 트리거.
            # server VAD가 이미 EOS를 커밋했으면 (eos_done=True) 건너뜀.
            if vad and len(chunk) == EXPECTED_CHUNK_BYTES and not self.eos_done:
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
                                # 625ms 무음 + interim 존재 → flush 단계 진입.
                                # 즉시 break하지 않고 5청크(~625ms)를 더 흘려서
                                # Google이 is_final을 emit할 시간을 확보한다.
                                # 진단: yielded_chunk_count = 지금까지 STT 로 흘려보낸 청크 수
                                # (×125ms = 이번 turn 의 audio duration). last_interim 은 이 시점의
                                # 가장 긴 interim — final 이 이거보다 짧으면 STT 가 잘린 것.
                                audio_ms = self._yielded_chunk_count * 125
                                print(
                                    f"[STT] Client VAD: EOS triggered ({silence_frames}×125ms="
                                    f"{silence_frames*125}ms silence, audio={audio_ms}ms, "
                                    f"last_interim='{self._last_interim_text[:40]}') — "
                                    f"flushing {EOS_FLUSH_CHUNKS} more chunks"
                                )
                                flush_remaining = EOS_FLUSH_CHUNKS
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
            self._yielded_chunk_count += 1

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

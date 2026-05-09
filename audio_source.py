"""
callbot/audio_source.py — pyVoIP RTP 오디오 → GoogleSTTV2 스트리밍 입력 변환

pyVoIP call.read_audio() 는 8kHz 8-bit unsigned PCM을 160 bytes 단위로 반환합니다.
  - encode_pcmu/parse_pcmu 내부가 audioop width=1 (8-bit) 기준
  - 반환값: 0~255 unsigned (무음=0x80)

GoogleSTTV2는 16kHz PCM 4000 bytes 청크를 기대합니다.

이 클래스가 수행하는 변환:
  read_audio() (160 bytes, 8kHz 8-bit unsigned, non-blocking) →
  8-bit unsigned → 16-bit signed 변환 →
  2000 bytes (125ms, 8kHz 16-bit) 묶음 →
  upsample_8k_to_16k() →
  4000 bytes (125ms, 16kHz 16-bit) 청크 yield

진단 옵션:
  - 1초 주기로 `[AudioSrc N=...]` 로그 출력 (정수값/RMS/peak/샘플 미리보기)
  - DEBUG_DUMP_RTP=1 환경변수일 때 wav/_dump/turn_<ts>.wav 와 ..._16k.wav 덤프
"""
import audioop
import math
import os
import time
import threading
import wave

import numpy as np

from audio_utils import upsample_8k_to_16k

# 8kHz에서 125ms = 2000 bytes (1000 samples × 2 bytes, 16-bit 변환 후)
_TARGET_8K_BYTES = 2000

# pyVoIP 무음 sentinel (8-bit unsigned, 중심값=0x80)
_SILENCE_160 = b"\x80" * 160

# 진단 로그 주기 — 매 N번째 read마다 한 줄 (20ms × 50 = 1초)
_LOG_EVERY_N_READS = 50

# DEBUG_DUMP_RTP=1일 때 WAV가 저장될 디렉터리
_DUMP_DIR = "wav/_dump"


class CallAudioSource:
    """
    pyVoIP 통화 오디오를 STT 스트리밍 입력으로 변환하는 이터러블

    사용법:
        audio_src = CallAudioSource(call, timeout_sec=15)
        stt.start_streaming(audio_src)
        ...
        audio_src.stop()  # 외부에서 강제 종료 시
    """

    def __init__(self, call, timeout_sec: int = 15):
        """
        Args:
            call: pyVoIP VoIPCall 인스턴스
            timeout_sec: 최대 청취 시간 (초). 초과 시 generator 종료.
        """
        self._call        = call
        self._timeout_sec = timeout_sec
        self._stop_event  = threading.Event()
        self._buffer      = b""

        # 진단용 WAV 덤프 (DEBUG_DUMP_RTP=1)
        self._dump_enabled = os.environ.get("DEBUG_DUMP_RTP") == "1"
        self._dump_8k      = bytearray() if self._dump_enabled else None
        self._dump_16k     = bytearray() if self._dump_enabled else None

    def stop(self):
        """generator 루프를 외부에서 종료"""
        self._stop_event.set()

    def __iter__(self):
        """
        16kHz PCM 청크를 yield하는 generator.
        STT 클래스의 _audio_generator()가 별도 스레드에서 이 이터러블을 소비합니다.

        Leading-silence 스킵: 첫 non-silence 가 도착하기 전까지는 STT 로 chunk
        를 보내지 않는다. WebRTC→Asterisk→pyVoIP 경로에서 RTP 가 늦게 도착할 때
        leading silence padding 이 Google STT 의 인식 컨텍스트를 손상시켜
        음소를 잘못 매칭하던 현상을 차단. 트레일링/중간 silence 는 그대로
        유지해 Google STT 의 stream 을 keepalive (silence-skip 을 모든 구간에
        하면 409 "Stream timed out after receiving no more client requests"
        에러 발생).
        """
        from pyVoIP.VoIP import CallState

        deadline = time.time() + self._timeout_sec
        self._buffer = b""

        total_reads      = 0
        non_empty_reads  = 0
        # 첫 non-silence 도착 전엔 STT 로의 yield 를 보류한다.
        # _dump_8k 에는 leading silence 도 그대로 기록 (네트워크 진단용),
        # _dump_16k / yield 는 started 이후만 (STT 가 본 것 그대로).
        started = False

        while not self._stop_event.is_set() and time.time() < deadline:
            # 통화가 끊어지면 종료
            try:
                state = self._call.state
            except Exception:
                break
            if state != CallState.ANSWERED:
                break

            # non-blocking read: 패킷이 늦으면 _SILENCE_160 패딩 반환.
            try:
                raw = self._call.read_audio(160, False)
            except Exception as e:
                print(f"[AudioSource] readAudio error: {e}")
                break

            total_reads += 1
            is_silence = (raw == _SILENCE_160)

            if not is_silence:
                non_empty_reads += 1
                if not started:
                    print(f"[AudioSource] First audio received ({len(raw)} bytes) after {total_reads} reads — STT streaming begins")
                    started = True

            # 8-bit unsigned → 16-bit signed 변환 (pyVoIP parse_pcmu width=1 역변환)
            raw_signed = audioop.bias(raw, 1, -128)      # 0~255 → -128~127
            raw_16bit  = audioop.lin2lin(raw_signed, 1, 2)  # 8-bit → 16-bit signed

            # 진단 dump 8k: 네트워크 RTP 타이밍 그대로 (leading silence 포함)
            if self._dump_enabled:
                self._dump_8k.extend(raw_16bit)

            # 1초 주기 진단 로그
            if total_reads % _LOG_EVERY_N_READS == 0:
                self._log_audio_stats(raw, raw_16bit, total_reads)

            # 첫 audio 도착 전: STT 입력 버퍼/yield 보류, 짧은 대기 후 다음 read.
            if not started:
                if total_reads == 100 and non_empty_reads == 0:
                    print(f"[AudioSource] WARNING: No RTP audio received after {total_reads} reads — "
                          f"check Asterisk pjsip.conf direct_media setting")
                if is_silence:
                    time.sleep(0.01)
                continue

            # 첫 audio 이후: 정상 누적 + yield (트레일링 silence 도 STT 에 흘려보냄).
            # 모든 silence 를 skip 하면 Google STT 가 client request 부재로
            # 409 timeout 이 발생함 — 그래서 트레일링 silence 는 keepalive 로 유지.
            self._buffer += raw_16bit

            # 2000 bytes (125ms at 8kHz 16-bit) 누적 시 업샘플 후 yield
            while len(self._buffer) >= _TARGET_8K_BYTES:
                chunk_8k = self._buffer[:_TARGET_8K_BYTES]
                self._buffer = self._buffer[_TARGET_8K_BYTES:]
                chunk_16k = upsample_8k_to_16k(chunk_8k)
                if self._dump_enabled:
                    self._dump_16k.extend(chunk_16k)
                yield chunk_16k

            # 무음(RTP 미도착)이면 짧은 대기 (busy-loop 방지)
            if is_silence:
                time.sleep(0.01)

        print(f"[AudioSource] Done: {non_empty_reads}/{total_reads} reads had audio (started={started})")

        if self._dump_enabled:
            self._save_dump_wavs()

    def _log_audio_stats(self, raw: bytes, raw_16bit: bytes, total_reads: int):
        """
        한 청크의 정수값을 두 가지 해석으로 동시에 출력.
        (raw 바이트는 0~255 unsigned 가정이지만 실제 포맷이 의심되면
         이 로그로 mean이 128 근처인지 0 근처인지 확인할 수 있다.)
        """
        if not raw or not raw_16bit:
            return
        u8         = np.frombuffer(raw, dtype=np.uint8)
        mn, mx     = int(u8.min()), int(u8.max())
        mean       = float(u8.mean())

        # signed-as-int8 해석 (silence가 0 근처라고 가정했을 때의 RMS)
        s8         = u8.view(np.int8).astype(np.float32)
        rms_signed = math.sqrt(float(np.mean(s8 * s8)))

        # unsigned-bias 해석 (silence가 128 근처라고 가정했을 때의 RMS — 정상 경로)
        ub         = u8.astype(np.float32) - 128.0
        rms_unsigned = math.sqrt(float(np.mean(ub * ub)))

        # 변환된 16-bit signed PCM 통계
        i16        = np.frombuffer(raw_16bit, dtype=np.int16)
        peak16     = int(np.max(np.abs(i16))) if len(i16) else 0
        rms16      = math.sqrt(float(np.mean(i16.astype(np.float32) ** 2))) if len(i16) else 0.0
        preview    = i16[:8].tolist()

        print(
            f"[AudioSrc N={total_reads}] "
            f"raw[min={mn} max={mx} mean={mean:.1f}] "
            f"rms_s={rms_signed:.1f} rms_u={rms_unsigned:.1f} | "
            f"16bit[peak={peak16} rms={rms16:.1f} preview={preview}]"
        )

    def _save_dump_wavs(self):
        """DEBUG_DUMP_RTP=1 시 8kHz / 16kHz 두 버전 WAV 저장."""
        if not self._dump_8k and not self._dump_16k:
            return
        try:
            os.makedirs(_DUMP_DIR, exist_ok=True)
            ts        = int(time.time())
            path_8k   = os.path.join(_DUMP_DIR, f"turn_{ts}.wav")
            path_16k  = os.path.join(_DUMP_DIR, f"turn_{ts}_16k.wav")

            with wave.open(path_8k, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(8000)
                wf.writeframes(bytes(self._dump_8k))

            with wave.open(path_16k, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(bytes(self._dump_16k))

            dur_8k  = len(self._dump_8k)  / (8000 * 2)
            dur_16k = len(self._dump_16k) / (16000 * 2)
            print(
                f"[AudioSrc] Dumped: {path_8k} ({dur_8k:.2f}s 8k) | "
                f"{path_16k} ({dur_16k:.2f}s 16k)"
            )
        except Exception as e:
            print(f"[AudioSrc] WAV dump failed: {e}")

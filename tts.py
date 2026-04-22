"""
callbot/tts.py — Google Cloud Text-to-Speech 핸들러

google-stt-tts-guide.md 의 GoogleTTS 클래스 기반.
synthesize_pcm_8k(): 텍스트 → 8kHz PCM (pyVoIP writeAudio 전달용)
"""
import struct
import numpy as np
from typing import Iterator, Optional

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

from credentials import load_google_credentials
from audio_utils import downsample_24k_to_8k
import config as cfg


# 한국어 음성 매핑
GOOGLE_TTS_VOICES = {
    "chirp3-hd-achernar": "ko-KR-Chirp3-HD-Achernar",
    "chirp3-hd-achird":   "ko-KR-Chirp3-HD-Achird",
    "neural2-a":          "ko-KR-Neural2-A",
    "neural2-b":          "ko-KR-Neural2-B",
    "neural2-c":          "ko-KR-Neural2-C",
}

# 오디오 품질 설정
TRIM_START_MS = 20    # 앞부분 artifact 제거 (ms)
FADE_OUT_MS   = 50    # 끝부분 fade-out 길이 (ms)
PADDING_MS    = 200   # 무음 패딩 (ms)
SAMPLE_RATE   = 24000


class GoogleTTS:
    """
    Google Cloud Text-to-Speech 핸들러

    사용법:
        tts = GoogleTTS(voice="chirp3-hd-achernar")
        pcm_24k = tts.synthesize("안녕하세요.")   # 24kHz PCM
        tts.save_wav("안녕하세요.", "out.wav")
    """

    def __init__(
        self,
        voice: str = "chirp3-hd-achernar",
        language_code: str = "ko-KR",
        credentials_dir: str = None,
    ):
        self.voice_name    = GOOGLE_TTS_VOICES.get(voice.lower(), voice)
        self.language_code = language_code
        self.credentials_dir = credentials_dir
        self._client: Optional[texttospeech.TextToSpeechClient] = None

    def _get_client(self) -> texttospeech.TextToSpeechClient:
        """Lazy 초기화 — 첫 호출 시 클라이언트 생성"""
        if self._client is None:
            credentials  = load_google_credentials(self.credentials_dir)
            self._client = texttospeech.TextToSpeechClient(credentials=credentials)
        return self._client

    def synthesize(self, text: str) -> bytes:
        """
        텍스트 → 24kHz LINEAR16 PCM

        artifact 제거(trim/fade-out/padding) 처리 포함.

        Returns:
            bytes: 24kHz 16-bit mono PCM
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
        """텍스트 → WAV 파일 저장 (24kHz)"""
        pcm = self.synthesize(text)
        wav = _wrap_pcm_in_wav(pcm, SAMPLE_RATE)
        with open(output_path, "wb") as f:
            f.write(wav)
        print(f"[TTS] Saved: {output_path}")
        return output_path

    def synthesize_streaming(
        self,
        text: str,
        min_chunk_chars: int = 10,
        max_chunk_chars: int = 200,
    ) -> Iterator[bytes]:
        """텍스트 → 24kHz PCM 청크 스트리밍 (generator)"""
        client      = self._get_client()
        text_chunks = _chunk_text(text, min_chunk_chars, max_chunk_chars)

        if not text_chunks:
            return

        def request_generator():
            yield StreamingSynthesizeRequest(
                streaming_config=StreamingSynthesizeConfig(
                    voice=VoiceSelectionParams(
                        language_code=self.language_code,
                        name=self.voice_name,
                    )
                )
            )
            for chunk in text_chunks:
                yield StreamingSynthesizeRequest(
                    input=StreamingSynthesisInput(text=chunk)
                )

        responses = client.streaming_synthesize(requests=request_generator())
        for response in responses:
            if response.audio_content:
                yield response.audio_content


# ── 오디오 처리 헬퍼 ──────────────────────────────────────────────────

def _trim_start(pcm: bytes, duration_ms: int, sample_rate: int) -> bytes:
    trim_bytes = int(sample_rate * 2 * duration_ms / 1000)
    return pcm[trim_bytes:] if len(pcm) > trim_bytes else pcm


def _apply_fade_out(pcm: bytes, duration_ms: int, sample_rate: int) -> bytes:
    arr          = np.frombuffer(pcm, dtype=np.int16).copy()
    fade_samples = int(sample_rate * duration_ms / 1000)
    fade_samples = min(fade_samples, len(arr))
    curve        = np.linspace(1.0, 0.0, fade_samples)
    arr[-fade_samples:] = (arr[-fade_samples:] * curve).astype(np.int16)
    return arr.tobytes()


def _add_silence_padding(pcm: bytes, duration_ms: int, sample_rate: int) -> bytes:
    silence_samples = int(sample_rate * duration_ms / 1000)
    silence         = np.zeros(silence_samples, dtype=np.int16).tobytes()
    return pcm + silence


def _wrap_pcm_in_wav(pcm: bytes, sample_rate: int) -> bytes:
    num_channels    = 1
    bits_per_sample = 16
    byte_rate       = sample_rate * num_channels * bits_per_sample // 8
    block_align     = num_channels * bits_per_sample // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(pcm), b"WAVE", b"fmt ",
        16, 1, num_channels, sample_rate,
        byte_rate, block_align, bits_per_sample,
        b"data", len(pcm),
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


# ── 콜봇 전용 헬퍼 ────────────────────────────────────────────────────

_tts_instance: Optional[GoogleTTS] = None


def _get_tts() -> GoogleTTS:
    """프로세스 내 TTS 싱글톤 (통화 내 재사용)"""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = GoogleTTS(
            voice=cfg.GCP_TTS_VOICE,
            language_code="ko-KR",
            credentials_dir=cfg.CREDENTIALS_DIR,
        )
    return _tts_instance


def synthesize_pcm_8k(text: str) -> bytes:
    """
    텍스트 → 8kHz 16-bit mono PCM

    pyVoIP call.writeAudio() 에 직접 전달 가능한 포맷.
    24kHz TTS 출력을 3x 다운샘플하여 8kHz로 변환합니다.
    """
    tts     = _get_tts()
    pcm_24k = tts.synthesize(text)
    return downsample_24k_to_8k(pcm_24k)

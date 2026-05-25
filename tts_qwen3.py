"""
callbot/tts_qwen3.py — Qwen3-TTS (vLLM-Omni, OpenAI 호환) HTTP 스트리밍 드라이버

GoogleTTS 와 같은 시그니처(voice_name attribute, synthesize, synthesize_streaming)
를 노출해 tts.py 의 `synthesize_pcm_8bit_unsigned_streaming` 파이프라인이 그대로
사용할 수 있도록 한다. 24kHz 16-bit signed mono raw PCM 을 chunk 단위로 yield.

서버 계약 (docs/260520_streaming-tts-api.md §4):
    POST {url}/v1/audio/speech
    Body: {"input": <text>, "voice": <voice>, "response_format": "pcm", "stream": true}
    응답: Content-Type: audio/pcm, 24kHz mono int16 LE, chunked transfer.

주의:
    - HTTP 청크 경계 ≠ int16 샘플 경계. leftover 1-byte 버퍼링 필수.
    - streaming 일 때 speed 는 1.0 만 허용 (서버 400 회피).
    - 사내망 IP — 도달 불가 시 raise (call_handler 가 잡고 TTS 만 skip).
"""
from typing import Iterator

import requests


class Qwen3TTS:
    """
    Qwen3-TTS HTTP 드라이버. GoogleTTS 와 duck-type 호환.

    사용법:
        tts = Qwen3TTS(url="http://172.31.79.203:30000")
        for chunk in tts.synthesize_streaming("안녕하세요"):  # 24kHz int16 PCM
            ...
    """

    def __init__(
        self,
        url: str,
        voice: str = "femail_achernar",
        language: str = "Auto",
        timeout: float = 30.0,
    ):
        self._endpoint = url.rstrip("/") + "/v1/audio/speech"
        self.voice     = voice
        self.language  = language
        self._timeout  = timeout

    @property
    def voice_name(self) -> str:
        # tts.py:_cache_key 가 voice_name 을 cache 해시 입력으로 사용 — Google 의
        # voice_name (e.g. "ko-KR-Chirp3-HD-Achernar") 과 다른 값을 노출해야 두
        # provider 의 캐시가 자동 분리된다.
        return self.voice

    def _payload(self, text: str) -> dict:
        body = {
            "input": text,
            "voice": self.voice,
            "response_format": "pcm",
            "stream": True,
        }
        if self.language and self.language.lower() != "auto":
            body["language"] = self.language
        return body

    def synthesize_streaming(self, text: str) -> Iterator[bytes]:
        """24kHz 16-bit signed mono PCM chunk generator.

        int16 샘플 경계로 정렬된 chunk 만 yield. HTTP chunk 가 홀수 byte 로
        끝나면 마지막 byte 는 leftover 에 들고 다음 chunk 와 합쳐 처리.
        """
        leftover = bytearray()
        with requests.post(
            self._endpoint,
            json=self._payload(text),
            stream=True,
            timeout=self._timeout,
        ) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                leftover.extend(chunk)
                aligned = (len(leftover) // 2) * 2
                if aligned:
                    yield bytes(leftover[:aligned])
                    del leftover[:aligned]

    def synthesize(self, text: str) -> bytes:
        """non-streaming — 전체 PCM 모음 (24kHz 16-bit). test 경로 호환용."""
        return b"".join(self.synthesize_streaming(text))

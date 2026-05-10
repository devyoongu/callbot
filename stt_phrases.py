"""
callbot/stt_phrases.py — Google STT V2 speech adaptation phrase boost

도메인 (SK쉴더스 / ADT캡스 보안 서비스) 의 복합어/고유명사 인식률을 끌어올리기
위한 phrase boost 사전. Google STT 모델은 일반 한국어로 훈련되어 있어
"파견보안관제", "원격관제", "제로트러스트" 같은 복합어를 종종 잘게 쪼개거나
오인식한다 (실측: "파견보안관제와 원격관제의 차이…" → "파변보안 브랜드").

각 Phrase 의 boost 값은 0-20 범위에서 클수록 강한 가중치.
  - 20: 매우 distinctive 한 고유명사/약어 (ADT캡스, ISMS-P, 제로트러스트)
  - 15: 도메인 복합어 (파견보안관제, 원격관제, 얼굴인식기)
  - 10: 일반적인 도메인 단어 (인증, 보안, 사이버)

지원 모델 — V2 streaming 에서 adaptation+boost 가 받아지는 조합:
  - chirp_3   (us region, us-speech.googleapis.com)   ✓
  - telephony (global region)                         ✗ (400 에러)

비지원 조합에서는 build_adaptation() 호출하지 말 것 — 호출자 (stt.py) 가
모델별로 분기.
"""
from typing import List, Tuple

# (phrase, boost) 튜플의 리스트
_PHRASES: List[Tuple[str, float]] = [
    # ── 회사/제품 고유명사 ────────────────────────────────────
    ("SK쉴더스", 20.0),
    ("ADT캡스", 20.0),
    ("쉴더스", 15.0),
    ("캡스", 15.0),

    # ── 인증/규격 약어 ───────────────────────────────────────
    ("ISMS-P", 20.0),
    ("ISMS", 18.0),
    ("ISO", 15.0),

    # ── 보안 관제 서비스 (복합어) ─────────────────────────────
    ("파견보안관제", 20.0),
    ("원격관제", 18.0),
    ("출동보안관제", 18.0),
    ("관제", 12.0),
    ("파견", 12.0),
    ("보안관제", 15.0),
    ("무인경비", 15.0),
    ("침입탐지", 15.0),
    ("보안카메라", 15.0),
    ("출입통제", 15.0),
    ("CCTV", 15.0),

    # ── 인증기/디바이스 ──────────────────────────────────────
    ("얼굴인식기", 20.0),
    ("지문인식기", 18.0),
    ("카드인식기", 18.0),
    ("얼굴인식", 15.0),
    ("지문인식", 15.0),
    ("실외", 15.0),
    ("실내", 12.0),

    # ── 사이버보안 / 위협 ────────────────────────────────────
    ("제로트러스트", 20.0),
    ("랜섬웨어", 18.0),
    ("멀웨어", 15.0),
    ("피싱", 15.0),
    ("사이버보안", 15.0),
    ("사이버 위협", 12.0),
    ("보안 패러다임", 12.0),
    ("사전 예방", 12.0),
    ("예방 방법", 12.0),
    ("보안 솔루션", 12.0),
    ("보안 업데이트", 12.0),
    ("패치 적용", 12.0),
    ("백업 체계", 12.0),
    ("보안 교육", 10.0),
    ("모의훈련", 12.0),

    # ── 자주 등장하는 일반 도메인 단어 ────────────────────────
    ("인증", 10.0),
    ("보안", 10.0),
    ("정보보호", 12.0),
    ("취약점", 12.0),
    ("자산", 10.0),
    ("중소기업", 12.0),
]


def build_adaptation():
    """
    SpeechAdaptation 객체를 반환. _run_streaming 에서 RecognitionConfig 의
    adaptation 필드에 직접 전달.

    speech_v2 미설치 환경 (개발/테스트) 대비 import 는 함수 안에서.
    """
    from google.cloud.speech_v2.types import cloud_speech as cs

    phrases = [cs.PhraseSet.Phrase(value=v, boost=b) for v, b in _PHRASES]
    return cs.SpeechAdaptation(
        phrase_sets=[
            cs.SpeechAdaptation.AdaptationPhraseSet(
                inline_phrase_set=cs.PhraseSet(phrases=phrases)
            )
        ]
    )


# 모델별 adaptation 지원 여부 — stt.py 가 분기 시 사용
ADAPTATION_SUPPORTED_MODELS = {"chirp_3", "chirp_2"}


def supports_adaptation(model: str) -> bool:
    """주어진 모델 이름이 adaptation+boost 를 받을 수 있으면 True."""
    return model in ADAPTATION_SUPPORTED_MODELS

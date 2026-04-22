# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an AI callbot that registers as a SIP extension on an Asterisk PBX server and handles inbound calls. The bot performs speech-to-text (STT), queries an LLM (Athena), and responds with text-to-speech (TTS) — all in real time over a SIP/RTP audio stream.

## Running the Bot

```bash
# Activate the virtualenv first
source venv/bin/activate

# Run the bot (registers SIP extension and waits for calls)
python main.py
```

The bot requires `.env` to be configured (copy from `.env.example`).

## Environment Setup

```bash
cp .env.example .env
# Then fill in SIP credentials, GCP project, and optionally Athena LLM settings

pip install -r requirements.txt
```

Google Cloud credentials (service account JSON files) go in `crendential/` (note the typo — this is the actual directory name). Multiple JSON files are supported; `credentials.py` distributes calls across them via round-robin for quota balancing.

## Architecture

Call flow through the modules:

```
main.py
  └── VoIPPhone (pyVoIP) registers as SIP extension
        └── call_handler.handle_call()  [per-call thread]
              ├── AthenaClient.start()   — session init, gets greeting
              ├── _play_tts()            — tts.synthesize_pcm_8k() → call.writeAudio()
              └── [conversation loop]
                    ├── CallAudioSource  — reads call.readAudio(), upsamples 8kHz→16kHz
                    ├── GoogleSTTV2      — streams to Google STT V2, returns transcript
                    ├── AthenaClient.query_sync()  — SSE streaming LLM response
                    └── _play_tts()      — speaks the reply
```

### Key modules

- **`call_handler.py`** — Per-call state machine. Each inbound call runs `handle_call()` in a separate pyVoIP thread. Manages the STT→LLM→TTS loop, no-input counting, and `end_call`/`transfer_call` commands from Athena.

- **`athena.py`** — HTTP SSE client for the Athena LLM API. `start()` and `end()` use synchronous `requests` with streaming. `query()` is an async generator using `aiohttp`; `query_sync()` wraps it with `asyncio.run()` for use in pyVoIP's synchronous call threads. Athena is optional — if env vars are not set, the bot falls back to hardcoded messages.

- **`stt.py`** — Google Cloud Speech-to-Text V2 streaming client. Uses server-side VAD by default (`enable_voice_activity_events=True`). The `telephony` model receives 16kHz PCM (upsampled from 8kHz call audio). A new `GoogleSTTV2` instance is created per conversation turn via `create_stt()`.

- **`tts.py`** — Google Cloud TTS client. Synthesizes at 24kHz, then downsamples to 8kHz for pyVoIP. `synthesize_pcm_8k()` is the main entry point. `GoogleTTS` is a process-level singleton (lazy-initialized on first call).

- **`audio_source.py`** — `CallAudioSource` adapts pyVoIP's `readAudio()` (8kHz, ~320 bytes chunks) to the format expected by the STT streamer (16kHz, 4000-byte chunks). Accumulates 2000 bytes at 8kHz, upsamples to 4000 bytes at 16kHz, then yields.

- **`audio_utils.py`** — `upsample_8k_to_16k()` and `downsample_24k_to_8k()` using `scipy.signal.resample_poly` with built-in anti-aliasing.

- **`credentials.py`** — Loads GCP service account JSON from `crendential/` directory. Thread-safe round-robin rotation across multiple credential files.

- **`config.py`** — Loads all settings from `.env`. Call `athena_configured()` to check if Athena env vars are set before using `AthenaClient`.

## Audio Pipeline Details

- pyVoIP delivers 8kHz 16-bit PCM in ~320-byte (20ms) chunks via `readAudio()`
- `CallAudioSource` accumulates to 2000-byte (125ms) chunks, upsamples 2× to 4000-byte 16kHz chunks for STT
- TTS outputs 24kHz 16-bit PCM, downsampled 3× to 8kHz for `writeAudio()`
- TTS playback sends 320-byte chunks with 18ms sleep intervals to maintain RTP timing

## Asterisk Configuration

The bot registers as SIP extension `2001`. Required Asterisk config:
- `pjsip.conf`: add `2001` endpoint/auth/aor
- `extensions.conf`: `exten => 2001,1,Dial(PJSIP/2001,30)`

## Fallback Mode

If Athena env vars are not configured, the bot runs in fallback mode: greets with `FALLBACK_GREETING`, responds to any input with `FALLBACK_GOODBYE`, then hangs up. Useful for testing the SIP/STT/TTS pipeline without the LLM backend.
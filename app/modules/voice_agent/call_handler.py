import asyncio
import audioop
import json
import uuid
import base64
import logging
import os
import time
import wave
import io
from collections import Counter
from datetime import datetime
from typing import Optional
from sarvamai import AsyncSarvamAI
from fastapi import WebSocket, WebSocketDisconnect
import numpy as np
from app.modules.voice_agent import config
from app.modules.voice_agent import database as db
from app.modules.voice_agent.models import (
    CallSessionData, CallState,
    LeadData, LeadStatus,
    CompanyScriptData, CompanyData,
)
from app.modules.voice_agent.services import (
    session_save, session_delete,
    llm_respond, sarvam_tts,
    send_sms, get_calendar_slots, create_calendar_event,
    score_to_status, build_interview_sms, build_recall_sms,
    build_sarvam_stt_url, mulaw_to_pcm16,
    SARVAM_STT_TRANSCRIPT_TYPES, SARVAM_STT_PARTIAL_TYPES,
)

logger = logging.getLogger("voice_agent.call_handler")

# ─────────────────────────────────────────────────────────────────────────────
# Audio constants
# ─────────────────────────────────────────────────────────────────────────────

# Minimum RMS for a 16kHz PCM16 chunk to be considered real speech.
# Telephony at 8kHz upsampled to 16kHz — normal speech RMS is 300–5000.
# Comfort noise / silence from Vobiz sits at RMS 1–15.
# This threshold gates silence from being forwarded to Sarvam STT.
SPEECH_RMS_THRESHOLD = 120

# G.711 µ-law silence codewords: 0x7F = positive silence, 0x7E / 0x80 = adjacent
# If >85% of a mulaw frame is these bytes, it is CN/silence — not speech.
MULAW_SILENCE_BYTES = {0x7E, 0x7F, 0x80}
MULAW_SILENCE_RATIO_THRESHOLD = 0.85

# STT chunk size: 640 bytes = 320 samples @ 16kHz = 20ms
FRAME_SIZE = 3200

# Audio dump for diagnostics — set AUDIO_DUMP_DIR in env to enable
# e.g. AUDIO_DUMP_DIR=/tmp/audio_debug
AUDIO_DUMP_DIR = os.environ.get("AUDIO_DUMP_DIR", "")


def _pcm16_to_mulaw(pcm16_bytes: bytes) -> bytes:
    """
    Convert LINEAR16 PCM audio (from Sarvam TTS) to G.711 µ-law.
    Vobiz WebSocket media stream requires µ-law encoded audio.

    PCM16: 2 bytes/sample, 8000 Hz → 16000 bytes/sec
    mulaw: 1 byte/sample, 8000 Hz →  8000 bytes/sec
    """
    return audioop.lin2ulaw(pcm16_bytes, 2)  # 2 = 16-bit (2 bytes per sample)


def _is_mulaw_silence(mulaw_bytes: bytes) -> bool:
    """
    Returns True if the µ-law frame is comfort noise / silence.

    G.711 silence = byte value 0x7F (127). Adjacent values 0x7E / 0x80
    are the lowest-energy non-zero codewords. If >85% of a frame is
    these three values, Vobiz is not delivering real inbound audio.

    This check happens BEFORE ulaw2lin conversion so we catch silence
    at the source rather than after it has been resampled.
    """
    if not mulaw_bytes:
        return True
    silence_count = sum(1 for b in mulaw_bytes if b in MULAW_SILENCE_BYTES)
    return (silence_count / len(mulaw_bytes)) > MULAW_SILENCE_RATIO_THRESHOLD


def _mulaw_diag(mulaw_bytes: bytes) -> dict:
    """
    Return diagnostic dict for a µ-law buffer.
    Used for structured logging — no side effects.
    """
    counts = Counter(mulaw_bytes)
    top5 = counts.most_common(5)
    silence_count = sum(1 for b in mulaw_bytes if b in MULAW_SILENCE_BYTES)
    return {
        "len": len(mulaw_bytes),
        "unique": len(counts),
        "silence_ratio": round(silence_count / max(len(mulaw_bytes), 1), 3),
        "top5_bytes": top5,
        "first8": list(mulaw_bytes[:8]),
    }


class CallHandler:
    """
    Manages one outbound call from start to finish.

    Audio pipeline:
      Vobiz → WS (mulaw 8kHz)
        → silence gate (drop CN frames)
        → audioop.ulaw2lin() → PCM8k
        → audioop.ratecv() → PCM16k
        → 640-byte chunks
        → Sarvam STT WS (pcm_s16le @ 16kHz)
        → transcript events
        → Groq/Gemini LLM
        → Sarvam TTS (PCM16 @ 8kHz)
        → audioop.lin2ulaw() → mulaw
        → base64 → Vobiz WS playAudio frame

    Concurrency:
      _vobiz_listener()      — reads WS frames from Vobiz
      _sarvam_stt_reader()   — reads transcripts from Sarvam STT WS
      _utterance_processor() — consumes transcripts, drives LLM + TTS
      _tts_sender()          — drains TTS queue → writes mulaw to Vobiz WS
    """

    def __init__(
        self,
        websocket: WebSocket,
        lead: LeadData,
        company: CompanyData,
        script: CompanyScriptData,
    ):
        self.ws = websocket
        self.lead = lead
        self.company = company
        self.script = script

        self.call_id = str(uuid.uuid4())
        self.session = CallSessionData(
            call_id=self.call_id,
            lead_id=lead.id,
            lead_phone=lead.phone,
            lead_name=lead.name,
            company_id=company.id,
            script_id=script.id,
        )

        # Sarvam STT WebSocket handle
        self.sarvam_stt_ws = None

        # audioop.ratecv filter state — must persist across calls
        self._rate_state = None
        # Accumulate PCM16k bytes until we have a full FRAME_SIZE chunk
        self._pcm_buffer = b""

        # Audio chunks buffered before STT is ready
        self._prebuffer = []
        self.stt_ready = False

        # Utterance state
        self.utterance_buffer: str = ""
        self.utterance_ready: asyncio.Event = asyncio.Event()

        # TTS queue — mulaw bytes or None (None = barge-in flush sentinel)
        self.tts_queue: asyncio.Queue = asyncio.Queue()

        # stream_sid event — _tts_sender waits for this before sending any audio
        self._stream_sid_ready: asyncio.Event = asyncio.Event()

        # Global flags
        self.call_ended: bool = False
        self.stream_sid: Optional[str] = None
        self.simulation_mode: bool = getattr(config, "SIMULATION_MODE", False)

        # Track which slot index lead selected (0-based)
        self._chosen_slot_index: int = 0

        self._speech_active = False
        self._last_transcript = ""

        # ── Diagnostics ────────────────────────────────────────────────────
        # Counters for log-rate-limiting
        self._silence_frame_count = 0      # consecutive silence frames received
        self._speech_frame_count = 0       # consecutive speech frames received
        self._total_frames_received = 0    # total media frames seen
        self._stt_chunks_sent = 0          # chunks actually forwarded to STT

        # Audio dump file handle (None = disabled)
        self._mulaw_dump: Optional[io.RawIOBase] = None
        if AUDIO_DUMP_DIR:
            try:
                os.makedirs(AUDIO_DUMP_DIR, exist_ok=True)
                dump_path = os.path.join(AUDIO_DUMP_DIR, f"vobiz_{self.call_id}.mulaw")
                self._mulaw_dump = open(dump_path, "wb")
                logger.info(f"[{self.call_id}] 🎙 Audio dump → {dump_path}")
            except Exception as e:
                logger.warning(f"[{self.call_id}] Could not open audio dump: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Entry point
    # ─────────────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        await self.ws.accept()
        await db.update_lead(self.lead.id, {"status": LeadStatus.CALLING})
        await session_save(self.session)
        logger.info(
            f"[{self.call_id}] WS accepted | lead={self.lead.name} ({self.lead.phone})"
        )
        try:
            await asyncio.gather(
                self._vobiz_listener(),
                self._tts_sender(),
            )
        except Exception as e:
            logger.error(f"[{self.call_id}] gather error: {e}", exc_info=True)
        finally:
            await self._end_call()

    # ─────────────────────────────────────────────────────────────────────────
    # Vobiz WS listener
    # ─────────────────────────────────────────────────────────────────────────

    async def _vobiz_listener(self) -> None:
        try:
            async for raw in self.ws.iter_text():

                logger.info(f"[{self.call_id}] RAW VOBIZ EVENT: {raw[:1000]}")

                if self.call_ended:
                    break
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                event = msg.get("event")

                # ── connected ────────────────────────────────────────────────
                if event == "connected":
                    logger.info(f"[{self.call_id}] Vobiz WS connected | full_msg={msg}")

                # ── start ────────────────────────────────────────────────────
                elif event == "start":
                    start_data = msg.get("start", {})
                    logger.critical(
                        f"[{self.call_id}] START EVENT FULL: {json.dumps(msg, indent=2)}"
                    )

                    self.stream_sid = (
                        start_data.get("streamId")
                        or msg.get("streamSid")
                        or msg.get("stream_id")
                    )
                    logger.info(
                        f"[{self.call_id}] Stream started | sid={self.stream_sid}"
                    )

                    # Log all start metadata — helps diagnose bidirectional config
                    logger.info(
                        f"[{self.call_id}] START METADATA | "
                        f"mediaFormat={start_data.get('mediaFormat')} | "
                        f"tracks={start_data.get('tracks')} | "
                        f"customParameters={start_data.get('customParameters')}"
                    )

                    self._stream_sid_ready.set()
                    asyncio.create_task(self._start_sarvam_stt())
                    asyncio.create_task(self._on_call_start())

                # ── media ────────────────────────────────────────────────────
                elif event == "media":
                    await self._handle_media_frame(msg)

                # ── stop ─────────────────────────────────────────────────────
                elif event == "stop":
                    logger.info(
                        f"[{self.call_id}] Vobiz stop | "
                        f"total_frames={self._total_frames_received} | "
                        f"stt_chunks_sent={self._stt_chunks_sent}"
                    )
                    break

                else:
                    # Log unknown events — could be dtmf, mark, etc.
                    logger.debug(f"[{self.call_id}] Unknown Vobiz event: {event} | {msg}")

        except WebSocketDisconnect:
            logger.info(f"[{self.call_id}] Vobiz WS disconnected")
        except Exception as e:
            logger.error(f"[{self.call_id}] _vobiz_listener error: {e}", exc_info=True)
        finally:
            await self._end_call()

    async def _handle_media_frame(self, msg: dict) -> None:

        media = msg.get("media", {})
        payload = media.get("payload", "")
        if not payload:
            return

        self._total_frames_received += 1

        # ── 1. Track type diagnosis ──────────────────────────────────────────
        track = media.get("track", "unknown")

        logger.info(
            f"[{self.call_id}] MEDIA TRACK={track} "
            f"chunk={media.get('chunkId')} "
            f"payload_size={len(payload)}"
        )

        if self._total_frames_received <= 5:
            # Log first 5 frames verbosely — reveals if Vobiz is sending inbound
            logger.info(
                f"[{self.call_id}] MEDIA FRAME #{self._total_frames_received} | "
                f"track={track} | timestamp={media.get('timestamp')} | "
                f"chunk={media.get('chunk')}"
            )
        if track == "outbound":
            # Vobiz is echoing our TTS back — bidirectional not configured correctly
            if self._total_frames_received == 1:
                logger.error(
                    f"[{self.call_id}] ❌ VOBIZ SENDING OUTBOUND TRACK ONLY — "
                    f"inbound (caller) audio not enabled. "
                    f"Check Stream XML: bidirectional='true' and Vobiz account settings."
                )
            return  # never forward outbound audio to STT

        # ── 2. Decode mulaw ──────────────────────────────────────────────────
        try:
            mulaw_bytes = base64.b64decode(payload)
            logger.info(
                f"[{self.call_id}] AUDIO BYTES "
                f"len={len(mulaw_bytes)} "
                f"first10={list(mulaw_bytes[:10])}"
            )
        except Exception as e:
            logger.warning(f"[{self.call_id}] base64 decode failed: {e}")
            return

        # ── 3. Raw dump ──────────────────────────────────────────────────────
        if self._mulaw_dump and not self._mulaw_dump.closed:
            try:
                self._mulaw_dump.write(mulaw_bytes)
            except Exception:
                pass

        # Real audio arrived — reset silence counter
        if self._silence_frame_count > 0:
            logger.info(
                f"[{self.call_id}] ✅ Real audio after {self._silence_frame_count} silence frames"
            )
        self._silence_frame_count = 0
        self._speech_frame_count += 1

        if self._speech_frame_count == 1:
            diag = _mulaw_diag(mulaw_bytes)
            logger.critical(
                f"[{self.call_id}] 🔥 INBOUND AUDIO CONFIRMED FROM VOBIZ "
                f"(first non-silence frame received)"
            )

        # ── 5. ulaw → PCM8k ─────────────────────────────────────────────────
        try:
            pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
        except Exception as e:
            logger.warning(f"[{self.call_id}] ulaw2lin failed: {e}")
            return

        # ── 6. Resample 8k → 16k ────────────────────────────────────────────
        try:
            pcm_16k, self._rate_state = audioop.ratecv(
                pcm_8k,
                2,       # sample width bytes (16-bit)
                1,       # channels (mono)
                8000,    # input rate
                16000,   # output rate
                self._rate_state,
            )
        except Exception as e:
            logger.warning(f"[{self.call_id}] ratecv failed: {e}")
            return

        # ── 7. Accumulate buffer ─────────────────────────────────────────────
        self._pcm_buffer += pcm_16k

        # ── 8. Emit 640-byte chunks to STT ───────────────────────────────────
        while len(self._pcm_buffer) >= FRAME_SIZE:
            chunk = self._pcm_buffer[:FRAME_SIZE]
            self._pcm_buffer = self._pcm_buffer[FRAME_SIZE:]

            # FRAME_SIZE=640 is always even, but guard defensively
            if len(chunk) % 2 != 0:
                chunk = chunk[:-1]
                if not chunk:
                    continue

            # Per-chunk RMS gate — final protection against low-energy frames
            pcm_chunk = np.frombuffer(chunk, dtype=np.int16)
            chunk_rms = float(np.sqrt(np.mean(pcm_chunk.astype(np.float32) ** 2)))

            if chunk_rms < SPEECH_RMS_THRESHOLD and not self._speech_active:
                # Below threshold AND no active speech — skip silently
                logger.debug(
                    f"[{self.call_id}] Chunk below RMS threshold | "
                    f"rms={chunk_rms:.1f} < {SPEECH_RMS_THRESHOLD} | skipping"
                )
                continue

            # If STT not ready yet — buffer the chunk
            if not self.stt_ready or self.sarvam_stt_ws is None:
                logger.debug(f"[{self.call_id}] STT not ready → prebuffering chunk")
                self._prebuffer.append(chunk)
                # Cap prebuffer to avoid memory growth during long STT connect delay
                if len(self._prebuffer) > 200:
                    self._prebuffer = self._prebuffer[-100:]
                continue

            # Forward chunk to Sarvam STT
            try:
                b64_audio = base64.b64encode(chunk).decode("utf-8")
                logger.critical(
                    f"[{self.call_id}] FORCED SEND TO STT "
                    f"rms={chunk_rms} "
                    f"bytes={len(pcm_16k)}"
                )
                await self.sarvam_stt_ws.transcribe(
                    audio=b64_audio,
                    sample_rate=16000
                )
                self._stt_chunks_sent += 1
                

                if self._stt_chunks_sent % 50 == 0:
                    logger.info(
                        f"[{self.call_id}] SENT AUDIO TO STT | "
                        f"chunks_sent={self._stt_chunks_sent} | "
                        f"rms={chunk_rms:.1f}"
                    )
                    
            except Exception as e:
                logger.warning(
                    f"[{self.call_id}] STT send error: {type(e).__name__}: {e}"
                )
                asyncio.create_task(self._reconnect_stt())
                # Put chunk back so it doesn't get lost
                self._prebuffer.insert(0, chunk)
                break  # stop sending until reconnected

    # ─────────────────────────────────────────────────────────────────────────
    # Call start — speak greeting
    # ─────────────────────────────────────────────────────────────────────────

    async def _on_call_start(self) -> None:
        logger.info(f"[{self.call_id}] _on_call_start | sim={self.simulation_mode}")

        if self.simulation_mode:
            asyncio.create_task(self._inject_simulation_utterance())

        greeting = self.script.steps[0]["question"].replace("{name}", self.lead.name)
        logger.info(f"[{self.call_id}] Speaking greeting: {greeting[:60]}")
        await self._speak(greeting)

    async def _inject_simulation_utterance(self) -> None:
        """Simulation mode only — inject a canned user response after greeting."""
        await asyncio.sleep(5)
        if not self.call_ended:
            self.utterance_buffer = "ஆமா, நான் வேலை தேடுகிறேன்"
            self.utterance_ready.set()

    # ─────────────────────────────────────────────────────────────────────────
    # Sarvam STT — streaming WebSocket
    # ─────────────────────────────────────────────────────────────────────────

    async def _start_sarvam_stt(self) -> None:
        """
        Open Sarvam streaming STT WebSocket and keep it alive for the call.

        Connection params:
          model=saaras:v3        — latest, best accuracy
          mode=transcribe        — output in Tamil (source language)
          language_code=ta-IN    — Tamil
          sample_rate=16000      — we upsample mulaw 8k → PCM 16k before sending
          input_audio_codec=pcm_s16le  — raw signed 16-bit little-endian PCM
          high_vad_sensitivity=True    — triggers on quieter speech
          vad_signals=True             — receive speech_start / speech_end events
        """
        # Start utterance processor once per call
        asyncio.create_task(self._utterance_processor())

        if not config.SARVAM_API_KEY:
            logger.error(f"[{self.call_id}] SARVAM_API_KEY missing — STT disabled")
            return

        try:
            client = AsyncSarvamAI(api_subscription_key=config.SARVAM_API_KEY)

            logger.info(f"[{self.call_id}] 🚀 Connecting to Sarvam STT...")

            async with client.speech_to_text_streaming.connect(
                model="saaras:v3",
                mode="transcribe",
                language_code="ta-IN",
                sample_rate=16000,
                input_audio_codec="pcm_s16le",
                high_vad_sensitivity=True,
                vad_signals=True,
            ) as ws:
                self.sarvam_stt_ws = ws
                self.stt_ready = True
                logger.info(f"[{self.call_id}] ✅ STT CONNECTED & READY")

                # Start reader task first — must be running before we send audio
                reader_task = asyncio.create_task(self._sarvam_stt_reader())

                # Flush any audio that arrived before STT was ready
                if self._prebuffer:
                    logger.info(
                        f"[{self.call_id}] 🔄 Flushing {len(self._prebuffer)} prebuffered chunks"
                    )
                    for chunk in self._prebuffer:
                        if self.call_ended:
                            break
                        try:
                            b64_audio = base64.b64encode(chunk).decode("utf-8")
                            await self.sarvam_stt_ws.transcribe(
                                audio=b64_audio,
                                sample_rate=16000
                            )
                        except Exception as e:
                            logger.warning(f"[{self.call_id}] Prebuffer flush error: {e}")
                            break
                    self._prebuffer.clear()

                # Keep the async-with block alive until call ends
                while not self.call_ended:
                    await asyncio.sleep(0.1)

                reader_task.cancel()

        except Exception as e:
            logger.error(
                f"[{self.call_id}] ❌ STT connect FAILED: {type(e).__name__}: {e}",
                exc_info=True,
            )
            self.stt_ready = False
            self.sarvam_stt_ws = None
            asyncio.create_task(self._reconnect_stt())

    async def _reconnect_stt(self) -> None:
        if self.call_ended:
            return
        logger.warning(f"[{self.call_id}] 🔄 Reconnecting STT in 1s...")
        self.stt_ready = False
        self.sarvam_stt_ws = None
        await asyncio.sleep(1)
        if not self.call_ended:
            asyncio.create_task(self._start_sarvam_stt())

    async def _sarvam_stt_reader(self) -> None:
        """
        Read all events from the Sarvam STT WebSocket.

        Saaras v3 event types (with vad_signals=True):
          speech_start  — VAD detected speech onset
          events        — heartbeat / keepalive (text is empty, this is normal)
          speech_end    — VAD detected silence / end of utterance
          transcript    — final recognised text for the utterance

        The sequence for one utterance is always:
          speech_start → [events...] → speech_end → transcript

        If transcript never arrives after speech_end, we use _last_transcript
        (the most recent partial/events text) as a fallback.
        """
        try:
            async for message in self.sarvam_stt_ws:
                if self.call_ended:
                    break

                logger.critical(f"[{self.call_id}] RAW STT MESSAGE: {message}")

                # Normalise: handle both dict and object responses
                if isinstance(message, dict):
                    msg_type = message.get("type", "")
                    # Saaras v3 uses "transcript" key; legacy models use "text"
                    text = (
                        message.get("transcript")
                        or message.get("text")
                        or ""
                    ).strip()
                    is_final = message.get("is_final", False)
                else:
                    msg_type = getattr(message, "type", "")
                    text = (
                        getattr(message, "transcript", None)
                        or getattr(message, "text", None)
                        or ""
                    ).strip()
                    is_final = getattr(message, "is_final", False)

                logger.info(
                    f"[{self.call_id}] STT EVENT | "
                    f"type={msg_type!r} | final={is_final} | text={text!r}"
                )

                # ── speech_start: VAD triggered ──────────────────────────────
                if msg_type == "speech_start":
                    self._speech_active = True
                    logger.info(f"[{self.call_id}] 🎤 Speech started (VAD)")

                    if self.session.tts_playing:
                        logger.info(f"[{self.call_id}] 🔴 BARGE-IN → flushing TTS queue")
                        self.session.tts_playing = False
                        await session_save(self.session)
                        await self.tts_queue.put(None)  # sentinel = flush

                # ── events: keepalive heartbeat ──────────────────────────────
                # These fire continuously while audio is being received.
                # Empty text is normal — Sarvam emits these even during silence.
                # If text is present, it is an intermediate (partial) result.
                elif msg_type == "events":
                    if text:
                        self._last_transcript = text
                        logger.debug(f"[{self.call_id}] PARTIAL: {text!r}")
                    # No action needed for empty events — they confirm WS is alive

                # ── partial: intermediate result (some SDK versions) ──────────
                elif msg_type == "partial":
                    if text:
                        self._last_transcript = text
                        logger.debug(f"[{self.call_id}] PARTIAL: {text!r}")

                # ── transcript: FINAL result ─────────────────────────────────
                elif msg_type in SARVAM_STT_TRANSCRIPT_TYPES:
                    if not text:
                        logger.debug(f"[{self.call_id}] Empty transcript — ignoring")
                        continue

                    logger.info(f"[{self.call_id}] ✅ FINAL TRANSCRIPT: {text!r}")
                    self._speech_active = False
                    self._last_transcript = ""
                    self.utterance_buffer = text
                    self.utterance_ready.set()

                # ── speech_end: VAD silence detected ─────────────────────────
                elif msg_type == "speech_end":
                    self._speech_active = False
                    logger.info(f"[{self.call_id}] 🔇 Speech ended (VAD)")

                    # Fallback: if transcript hasn't arrived yet but we have a partial,
                    # use it. The final transcript may arrive a moment later — we
                    # also handle it above so there's no double-fire risk because
                    # utterance_ready.clear() is called in _utterance_processor.
                    if self._last_transcript:
                        logger.warning(
                            f"[{self.call_id}] ⚠️ Using partial as final fallback: "
                            f"{self._last_transcript!r}"
                        )
                        self.utterance_buffer = self._last_transcript
                        self.utterance_ready.set()
                        self._last_transcript = ""

                # ── error: Sarvam sent an error event ───────────────────────
                elif msg_type == "error":
                    logger.error(f"[{self.call_id}] ❌ STT ERROR EVENT: {message}")

                else:
                    # Log unknown types — helps catch new event types from Sarvam
                    logger.warning(
                        f"[{self.call_id}] Unknown STT event type: {msg_type!r} | "
                        f"full={message}"
                    )

        except Exception as e:
            if not self.call_ended:
                logger.warning(
                    f"[{self.call_id}] ⚠️ STT reader dropped: {type(e).__name__}: {e}",
                    exc_info=True,
                )
                self.stt_ready = False
                self.sarvam_stt_ws = None
                asyncio.create_task(self._reconnect_stt())

    # ─────────────────────────────────────────────────────────────────────────
    # Utterance processor — LLM → script advance → speak
    # ─────────────────────────────────────────────────────────────────────────

    async def _utterance_processor(self) -> None:
        steps = self.script.steps
        logger.info(f"[{self.call_id}] utterance_processor started | {len(steps)} steps")

        while not self.call_ended:
            try:
                await asyncio.wait_for(self.utterance_ready.wait(), timeout=60.0)
            except asyncio.TimeoutError:
                logger.info(f"[{self.call_id}] 60s silence → closing")
                await self._closing_sequence()
                return

            self.utterance_ready.clear()
            text = self.utterance_buffer.strip()
            self.utterance_buffer = ""

            if not text:
                continue

            logger.info(f"[{self.call_id}] Processing utterance: {text[:80]!r}")

            self.session.history.append({"role": "user", "content": text})
            self.session.transcript_full += f"\nLead: {text}"
            await session_save(self.session)

            # ── LLM ──────────────────────────────────────────────────────────
            result = await llm_respond(self.session, self.script, self.company)

            speech       = result.get("speech", "")
            new_score    = result.get("lead_score", self.session.lead_score)
            confidence   = result.get("score_confidence", 0)
            intent_flags = result.get("intent_flags", [])
            advance      = result.get("advance_script", False)
            should_end   = result.get("should_end_call", False)

            self.session.lead_score       = new_score
            self.session.score_confidence = confidence

            for flag in intent_flags:
                if flag not in self.session.intent_flags:
                    self.session.intent_flags.append(flag)

            if self.session.proposed_slots:
                self._chosen_slot_index = self._detect_slot_choice(
                    text, len(self.session.proposed_slots)
                )

            if advance and self.session.script_pos < len(steps) - 1:
                self.session.script_pos += 1
                logger.info(
                    f"[{self.call_id}] Script advanced → pos {self.session.script_pos}"
                )

            self.session.history.append({"role": "assistant", "content": speech})
            self.session.transcript_full += f"\nAgent: {speech}"
            await session_save(self.session)

            # ── Route to state machine ────────────────────────────────────────
            if (
                "interview_requested" in self.session.intent_flags
                and self.session.state != CallState.SCHEDULING
                and not self.session.proposed_slots
            ):
                self.session.state = CallState.SCHEDULING
                await self._handle_scheduling(speech)

            elif should_end or self.session.script_pos >= len(steps) - 1:
                await self._speak(speech)
                await asyncio.sleep(1)
                await self._closing_sequence()
                return

            else:
                await self._speak(speech)

    # ─────────────────────────────────────────────────────────────────────────
    # Scheduling — offer interview slots
    # ─────────────────────────────────────────────────────────────────────────

    async def _handle_scheduling(self, pre_speech: str) -> None:
        await self._speak(pre_speech)

        slots = await get_calendar_slots(config.INTERVIEW_SLOTS_LOOKAHEAD_DAYS)
        if not slots:
            await self._speak(
                "நேர்காணலுக்கான நேரம் இப்போது இல்லை. விரைவில் திரும்ப அழைக்கிறோம்."
            )
            return

        self.session.proposed_slots = [s.isoformat() for s in slots]

        from datetime import timedelta
        slot_lines = []
        for i, slot in enumerate(slots[:3]):
            ist = slot + timedelta(hours=5, minutes=30)
            slot_lines.append(f"விருப்பம் {i + 1}: {ist.strftime('%d/%m %I:%M %p')}")

        await self._speak(
            f"நேர்காணலுக்கு இந்த நேரங்கள் இருக்கின்றன: {', '.join(slot_lines)}. "
            f"எந்த நேரம் உங்களுக்கு வசதியாக இருக்கும்? "
            f"ஒன்று, இரண்டு அல்லது மூன்று என்று சொல்லுங்கள்."
        )
        await session_save(self.session)

    @staticmethod
    def _detect_slot_choice(text: str, num_slots: int) -> int:
        import re
        ta_map = {
            "ஒன்று": 0, "முதல்": 0, "first": 0,
            "இரண்டு": 1, "second": 1,
            "மூன்று": 2, "third": 2,
        }
        text_lower = text.lower()
        for word, idx in ta_map.items():
            if word in text_lower and idx < num_slots:
                return idx
        m = re.search(r'\b([123])\b', text)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < num_slots:
                return idx
        return 0

    # ─────────────────────────────────────────────────────────────────────────
    # Closing
    # ─────────────────────────────────────────────────────────────────────────

    async def _closing_sequence(self) -> None:
        self.session.state = CallState.CLOSING
        name  = self.lead.name
        score = self.session.lead_score

        if score == "hot":
            msg = self.script.closing_hot.replace("{name}", name)
        elif score == "warm":
            msg = self.script.closing_warm.replace("{name}", name)
        else:
            msg = self.script.closing_cold.replace("{name}", name)

        await self._speak(msg)
        await asyncio.sleep(4)
        await self._end_call()

    # ─────────────────────────────────────────────────────────────────────────
    # TTS — speak text → convert to mulaw → queue for sending
    # ─────────────────────────────────────────────────────────────────────────

    async def _speak(self, text: str) -> None:
        """
        Fetch TTS audio from Sarvam (PCM16 @ 8kHz),
        convert to µ-law, enqueue for _tts_sender.
        """
        if not text or self.call_ended:
            return

        self.session.tts_playing = True
        await session_save(self.session)

        try:
            pcm16_bytes = await sarvam_tts(text)
            if not pcm16_bytes:
                logger.warning(
                    f"[{self.call_id}] TTS returned empty audio for: {text[:40]!r}"
                )
                self.session.tts_playing = False
                await session_save(self.session)
                return

            mulaw_bytes = _pcm16_to_mulaw(pcm16_bytes)
            logger.info(
                f"[{self.call_id}] TTS ready | "
                f"pcm16={len(pcm16_bytes)}B → mulaw={len(mulaw_bytes)}B | "
                f"text={text[:40]!r}"
            )
            await self.tts_queue.put(mulaw_bytes)

        except Exception as e:
            logger.error(f"[{self.call_id}] _speak error: {e}", exc_info=True)
            self.session.tts_playing = False
            await session_save(self.session)

    # ─────────────────────────────────────────────────────────────────────────
    # TTS sender — drain queue → write mulaw chunks to Vobiz WS
    # ─────────────────────────────────────────────────────────────────────────

    async def _tts_sender(self) -> None:
        """
        Drain the TTS queue and write µ-law audio to Vobiz in 40ms chunks.

        µ-law: 1 byte/sample × 8000 samples/sec = 8000 bytes/sec
        40ms chunk = 8000 × 0.040 = 320 bytes per chunk.

        Waits for stream_sid before sending — Vobiz rejects frames before start.
        """
        try:
            await asyncio.wait_for(self._stream_sid_ready.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.error(
                f"[{self.call_id}] stream_sid never arrived — aborting TTS sender"
            )
            return

        logger.info(f"[{self.call_id}] TTS sender ready | sid={self.stream_sid}")

        CHUNK_SIZE = 320  # 40ms of µ-law @ 8kHz

        while not self.call_ended:
            try:
                audio = await asyncio.wait_for(self.tts_queue.get(), timeout=1.0)

                # None sentinel = barge-in flush
                if audio is None:
                    # Drain any remaining queued audio
                    while not self.tts_queue.empty():
                        try:
                            self.tts_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    self.session.tts_playing = False
                    await session_save(self.session)
                    logger.info(f"[{self.call_id}] TTS queue flushed (barge-in)")
                    continue

                loop = asyncio.get_event_loop()
                next_send_time = loop.time()

                for i in range(0, len(audio), CHUNK_SIZE):
                    if self.call_ended:
                        break
                    # Stop sending if barge-in sentinel arrived
                    if self.tts_queue.qsize() > 0:
                        try:
                            peek = self.tts_queue.queue[0]
                            if peek is None:
                                break
                        except (IndexError, AttributeError):
                            break

                    chunk = audio[i: i + CHUNK_SIZE]

                    try:
                        await self.ws.send_text(json.dumps({
                            "event": "playAudio",
                            "media": {
                                "contentType": "audio/x-mulaw",
                                "sampleRate": 8000,
                                "payload": base64.b64encode(chunk).decode("ascii"),
                            },
                        }))

                        next_send_time += 0.04  # 40ms pacing
                        now = loop.time()
                        await asyncio.sleep(max(0, next_send_time - now))

                    except Exception as send_err:
                        logger.warning(
                            f"[{self.call_id}] WS send error: {send_err}"
                        )
                        break

                self.session.tts_playing = False
                await session_save(self.session)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.warning(f"[{self.call_id}] _tts_sender error: {e}")
                break

    # ─────────────────────────────────────────────────────────────────────────
    # End call — idempotent
    # ─────────────────────────────────────────────────────────────────────────

    async def _end_call(self) -> None:
        if self.call_ended:
            return
        self.call_ended = True
        logger.info(
            f"[{self.call_id}] Ending call | lead={self.lead.name} | "
            f"total_frames={self._total_frames_received} | "
            f"stt_chunks_sent={self._stt_chunks_sent} | "
            f"silence_frames={self._silence_frame_count}"
        )

        self._stream_sid_ready.set()

        # Close audio dump file
        if self._mulaw_dump and not self._mulaw_dump.closed:
            try:
                self._mulaw_dump.close()
                logger.info(f"[{self.call_id}] Audio dump closed")
            except Exception:
                pass

        # Close Sarvam STT WS gracefully
        if self.sarvam_stt_ws and not self.simulation_mode:
            try:
                await self.sarvam_stt_ws.close()
            except Exception:
                pass

        await self._post_call_actions()
        await session_delete(self.call_id)

        try:
            await self.ws.close()
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Post-call — update DB, calendar, SMS
    # ─────────────────────────────────────────────────────────────────────────

    async def _post_call_actions(self) -> None:
        session = self.session
        lead    = self.lead
        company = self.company

        new_status  = score_to_status(session.lead_score)
        update_data = {
            "status": new_status,
            "notes": session.transcript_full[-2000:],
            "score": session.score_confidence,
        }

        # Case 1: Interview scheduled
        if "interview_requested" in session.intent_flags and session.proposed_slots:
            from datetime import datetime as dt
            slot_idx = min(self._chosen_slot_index, len(session.proposed_slots) - 1)
            slot = dt.fromisoformat(session.proposed_slots[slot_idx])

            event_id = await create_calendar_event(lead, company, slot, session.call_id)
            update_data["status"]                 = LeadStatus.SCHEDULED
            update_data["scheduled_interview_at"] = slot

            await db.create_interview_slot(
                lead_id=lead.id,
                call_id=session.call_id,
                company_id=company.id,
                scheduled_at=slot,
                calendar_event_id=event_id,
                sms_sent=True,
            )
            logger.info(f"[{self.call_id}] Interview @ {slot.isoformat()}")

        # Case 2: Warm / callback
        elif session.lead_score == "warm" or "callback_requested" in session.intent_flags:
            from datetime import datetime as dt, timedelta
            recall_at = dt.utcnow() + timedelta(hours=config.RECALL_AFTER_HOURS)
            update_data["status"]       = LeadStatus.RECALL
            update_data["next_call_at"] = recall_at
            logger.info(f"[{self.call_id}] Recall @ {recall_at.isoformat()}")

        await db.update_lead(lead.id, update_data)
        await db.increment_call_attempts(lead.id)
        logger.info(
            f"[{self.call_id}] Post-call done | "
            f"status={update_data['status']} | score={session.lead_score}"
        )
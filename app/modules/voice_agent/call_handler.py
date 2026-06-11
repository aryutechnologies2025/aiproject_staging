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

# STT chunk size: 640 bytes = 320 samples @ 16kHz = 20ms
#
# FIX (Bug 3): The original code had FRAME_SIZE = 3200 despite the comment
# saying 640 bytes / 20ms. 3200 bytes = 200ms chunks. This caused:
#   1. 200ms buffering latency before any audio reached Sarvam
#   2. VAD could not detect speech onset because frames were too coarse
#   3. Possible silent drops inside the SDK's internal buffers
#
# Vobiz sends 20ms mulaw frames (160 bytes @ 8kHz).
# After ulaw2lin → PCM8k (320 bytes) → ratecv to 16kHz → 640 bytes.
# So FRAME_SIZE=640 matches exactly one Vobiz input frame at the output rate.
FRAME_SIZE = 640  # 320 samples × 2 bytes = 20ms @ 16kHz  ← WAS 3200, WRONG

# Minimum RMS for a 16kHz PCM16 chunk to block purely digital silence.
# FIX (Bug 4): Original threshold was 120, applied before _speech_active was
# ever set to True (because _speech_active depended on speech_start events
# that were never parsed due to Bug 2). This caused ALL pre-speech audio to be
# dropped, preventing Sarvam's server-side VAD from ever seeing speech onset.
# Lower to 30 to pass only true digital zero-padding.
# The server-side VAD in saaras:v3 handles speech/silence discrimination.
SPEECH_RMS_THRESHOLD = 30  # was 120 — too aggressive, blocked speech onset

# G.711 µ-law silence codewords: 0x7F = positive silence, 0x7E / 0x80 = adjacent
MULAW_SILENCE_BYTES = {0x7E, 0x7F, 0x80}
MULAW_SILENCE_RATIO_THRESHOLD = 0.85

# Audio dump for diagnostics — set AUDIO_DUMP_DIR in env to enable
AUDIO_DUMP_DIR = os.environ.get("AUDIO_DUMP_DIR", "")


def _pcm16_to_mulaw(pcm16_bytes: bytes) -> bytes:
    """
    Convert LINEAR16 PCM audio (from Sarvam TTS) to G.711 µ-law.
    Vobiz WebSocket media stream requires µ-law encoded audio.

    PCM16: 2 bytes/sample, 8000 Hz → 16000 bytes/sec
    mulaw: 1 byte/sample, 8000 Hz →  8000 bytes/sec
    """
    return audioop.lin2ulaw(pcm16_bytes, 2)


def _is_mulaw_silence(mulaw_bytes: bytes) -> bool:
    """
    Returns True if the µ-law frame is comfort noise / silence.
    """
    if not mulaw_bytes:
        return True
    silence_count = sum(1 for b in mulaw_bytes if b in MULAW_SILENCE_BYTES)
    return (silence_count / len(mulaw_bytes)) > MULAW_SILENCE_RATIO_THRESHOLD


def _mulaw_diag(mulaw_bytes: bytes) -> dict:
    """Return diagnostic dict for a µ-law buffer."""
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


def _parse_sarvam_stt_message(message) -> tuple[str, str]:
    """
    Parse a Sarvam STT WebSocket message and return (event_name, text).

    FIX (Bug 2): The WebSocket AsyncAPI spec (WebSocket doc, lines 455–465)
    defines the response envelope as:
        {
          "type": "data" | "error" | "events",   ← ResponseType enum
          "data": { ... }                          ← SpeechToTextResponseData
        }

    The top-level "type" field is NEVER "speech_start", "transcript",
    "speech_end" — those strings do not appear in the ResponseType enum.
    The original code checked message.get("type") for those strings, which
    meant every message fell through to the "Unknown STT event type" branch.

    Correct parsing:
      - type == "data"   → data is SpeechToTextTranscriptionData
                           → data["transcript"] contains the final text
                           → return ("transcript", text)

      - type == "events" → data is EventsData
                           → data["signal_type"] is "START_SPEECH" or "END_SPEECH"
                             (EventsDataSignalType enum, WebSocket doc lines 421–427)
                           → return ("speech_start" | "speech_end", "")

      - type == "error"  → data is ErrorData
                           → return ("error", error_message)

    Returns a tuple of (normalised_event_name, text_content).
    normalised_event_name values: "transcript", "speech_start", "speech_end", "error", "unknown"
    """
    # Normalise to dict — SDK may return object or dict depending on version
    if not isinstance(message, dict):
        try:
            message = vars(message)
        except Exception:
            message = {}

    top_type = message.get("type", "")       # "data", "error", "events"
    data = message.get("data", {}) or {}

    # Normalise data to dict as well
    if not isinstance(data, dict):
        try:
            data = vars(data)
        except Exception:
            data = {}

    if top_type == "data":
        # SpeechToTextTranscriptionData — final transcript
        # WebSocket doc, SpeechToTextTranscriptionData schema, lines 352–407:
        #   transcript: string — "Transcript of the provided speech in original language"
        text = (data.get("transcript") or "").strip()
        return ("transcript", text)

    elif top_type == "events":
        # EventsData — VAD signal
        # WebSocket doc, EventsData schema, lines 428–448:
        #   signal_type enum: "START_SPEECH" | "END_SPEECH"
        signal_type = data.get("signal_type", "")
        if signal_type == "START_SPEECH":
            return ("speech_start", "")
        elif signal_type == "END_SPEECH":
            return ("speech_end", "")
        else:
            # Other event_type values — heartbeat / unknown
            return ("events", "")

    elif top_type == "error":
        # ErrorData schema, lines 408–420:
        #   error: string, code: string
        error_msg = data.get("error", str(data))
        return ("error", error_msg)

    else:
        # Completely unexpected envelope — log for diagnosis
        return ("unknown", "")


class CallHandler:
    """
    Manages one outbound call from start to finish.

    Audio pipeline:
      Vobiz → WS (mulaw 8kHz)
        → silence gate (drop true CN frames only)
        → audioop.ulaw2lin() → PCM8k
        → audioop.ratecv() → PCM16k
        → 640-byte chunks (20ms)
        → Sarvam STT WS (pcm_s16le @ 16kHz, sample_rate set at connection)
        → transcript events (type="data") / VAD events (type="events")
        → Groq/Gemini LLM
        → Sarvam TTS (PCM16 @ 8kHz)
        → audioop.lin2ulaw() → mulaw
        → base64 → Vobiz WS playAudio frame
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

        self._last_barge_in = 0.0

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

        self._speech_start_time = 0

        # Tiered STT Fallback State
        self._stt_timeout_task: Optional[asyncio.Task] = None
        self._stt_audio_buffer = b""
        self._local_speech_active = False
        self._consecutive_silence_chunks = 0

        # Diagnostics
        self._silence_frame_count = 0
        self._speech_frame_count = 0
        self._total_frames_received = 0
        self._stt_chunks_sent = 0

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

                if event == "connected":
                    logger.info(f"[{self.call_id}] Vobiz WS connected | full_msg={msg}")

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
                    logger.info(
                        f"[{self.call_id}] START METADATA | "
                        f"mediaFormat={start_data.get('mediaFormat')} | "
                        f"tracks={start_data.get('tracks')} | "
                        f"customParameters={start_data.get('customParameters')}"
                    )

                    self._stream_sid_ready.set()
                    asyncio.create_task(self._start_sarvam_stt())
                    asyncio.create_task(self._on_call_start())

                elif event == "media":
                    await self._handle_media_frame(msg)

                elif event == "stop":
                    logger.info(
                        f"[{self.call_id}] Vobiz stop | "
                        f"total_frames={self._total_frames_received} | "
                        f"stt_chunks_sent={self._stt_chunks_sent}"
                    )
                    break

                else:
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

        track = media.get("track", "unknown")

        logger.info(
            f"[{self.call_id}] MEDIA TRACK={track} "
            f"chunk={media.get('chunkId')} "
            f"payload_size={len(payload)}"
        )

        if self._total_frames_received <= 5:
            logger.info(
                f"[{self.call_id}] MEDIA FRAME #{self._total_frames_received} | "
                f"track={track} | timestamp={media.get('timestamp')} | "
                f"chunk={media.get('chunk')}"
            )

        if track == "outbound":
            if self._total_frames_received == 1:
                logger.error(
                    f"[{self.call_id}] ❌ VOBIZ SENDING OUTBOUND TRACK ONLY — "
                    f"inbound (caller) audio not enabled. "
                    f"Check Stream XML: bidirectional='true' and Vobiz account settings."
                )
            return

        # ── Decode mulaw ─────────────────────────────────────────────────────
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

        # ── Raw dump ─────────────────────────────────────────────────────────
        if self._mulaw_dump and not self._mulaw_dump.closed:
            try:
                self._mulaw_dump.write(mulaw_bytes)
            except Exception:
                pass

        # Skip pure comfort-noise frames (Vobiz sends 0x7F fill when no audio)
        if _is_mulaw_silence(mulaw_bytes):
            self._silence_frame_count += 1
            if self._silence_frame_count % 100 == 1:
                logger.debug(
                    f"[{self.call_id}] Silence frame #{self._silence_frame_count} (mulaw CN gate)"
                )
            return

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
                f"(first non-silence frame received) | diag={diag}"
            )

        # ── ulaw → PCM8k ─────────────────────────────────────────────────────
        try:
            pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
        except Exception as e:
            logger.warning(f"[{self.call_id}] ulaw2lin failed: {e}")
            return

        # ── Resample 8k → 16k ────────────────────────────────────────────────
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

        # ── Accumulate buffer ─────────────────────────────────────────────────
        self._pcm_buffer += pcm_16k

        # ── Emit 640-byte (20ms) chunks to STT ───────────────────────────────
        #
        # FIX (Bug 3): FRAME_SIZE is now 640 (was 3200).
        # Each chunk is exactly 20ms of PCM16 at 16kHz.
        # This matches Vobiz's 20ms input cadence after upsampling.
        while len(self._pcm_buffer) >= FRAME_SIZE:
            chunk = self._pcm_buffer[:FRAME_SIZE]
            self._pcm_buffer = self._pcm_buffer[FRAME_SIZE:]

            if len(chunk) % 2 != 0:
                chunk = chunk[:-1]
                if not chunk:
                    continue

            # Per-chunk RMS gate — block only true digital silence (zeros)
            pcm_chunk = np.frombuffer(chunk, dtype=np.int16)
            chunk_rms = float(np.sqrt(np.mean(pcm_chunk.astype(np.float32) ** 2)))

            # Layer 1: Verification & Local Buffering
            if chunk_rms >= SPEECH_RMS_THRESHOLD:
                if not self._local_speech_active:
                    self._local_speech_active = True
                    self._stt_audio_buffer = chunk
                    # Cancel any leftover timeout
                    if self._stt_timeout_task and not self._stt_timeout_task.done():
                        self._stt_timeout_task.cancel()
                else:
                    self._stt_audio_buffer += chunk
                self._consecutive_silence_chunks = 0
            else:
                if self._local_speech_active:
                    self._consecutive_silence_chunks += 1
                    self._stt_audio_buffer += chunk
                    # 40 chunks * 20ms = 800ms of consecutive local silence
                    if self._consecutive_silence_chunks >= 40:
                        self._local_speech_active = False
                        # Speech ended locally -> Start STT Tier 1 timeout
                        self._start_tier1_timeout()

            # Prevent sending padding zeroes when user is purely silent.
            # But DO NOT block intra-speech and trailing silence, as STT server needs it to trigger END_SPEECH!
            if not self._local_speech_active and chunk_rms < SPEECH_RMS_THRESHOLD:
                continue

            # If STT not ready yet — buffer the chunk
            if not self.stt_ready or self.sarvam_stt_ws is None:
                logger.debug(f"[{self.call_id}] STT not ready → prebuffering chunk")
                self._prebuffer.append(chunk)
                if len(self._prebuffer) > 200:
                    self._prebuffer = self._prebuffer[-100:]
                continue

            # Forward chunk to Sarvam STT
            #
            # FIX (Bug 1): The Sarvam STT SDK transcribe() method maps its
            # "audio" parameter to AudioData.data (the base64 payload).
            # The WebSocket AsyncAPI spec (WebSocket doc, AudioDataSampleRate
            # schema, lines 466–486) defines sample_rate as a STRING enum:
            #   enum: ['16000', '22050', '24000']
            # and explicitly states "8kHz is only supported via connection
            # parameter, not in AudioData messages."
            #
            # Therefore:
            #   1. Do NOT pass sample_rate in the per-message transcribe() call.
            #      The connection-level sample_rate=16000 declared in connect()
            #      is the correct and only place to set this.
            #   2. The SDK's internal default for the encoding field is "audio/wav"
            #      which is the only valid AudioDataEncoding value. Since we set
            #      input_audio_codec="pcm_s16le" at connection time, the server
            #      already knows the codec. The per-message encoding field does
            #      not need to be set.
            #
            # Passing sample_rate=16000 (Python int) in the per-message call
            # causes the SDK to send the integer 16000 on the wire, which does
            # not match the string enum "16000", potentially causing the server
            # to silently reject or misprocess the audio message.
            try:
                b64_audio = base64.b64encode(chunk).decode("utf-8")
                logger.critical(
                    f"[{self.call_id}] SEND TO STT "
                    f"rms={chunk_rms:.1f} "
                    f"bytes={len(chunk)}"
                )
                logger.critical(
                    f"[{self.call_id}] STT SEND CHUNK (RAW PCM) | "
                    f"len_chunk={len(chunk)} | rms={chunk_rms:.1f} | "
                    f"base64_preview={b64_audio[:80]}..."
                )
                await self.sarvam_stt_ws.transcribe(audio=b64_audio)
                logger.info(
                    f"audio payload: {b64_audio}"
                )
                self._stt_chunks_sent += 1

                if self._stt_chunks_sent == 0:
                    logger.critical(f"[{self.call_id}] 🚀 FIRST AUDIO CHUNK SENT TO STT | rms={chunk_rms:.1f} len={len(chunk)} raw_header={base64.b64encode(chunk[:44]).decode('ascii')}")

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
                self._prebuffer.insert(0, chunk)
                break

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
                sample_rate="16000",
                input_audio_codec="pcm_s16le",
                high_vad_sensitivity=True,
                vad_signals=True,
            ) as ws:
                self.sarvam_stt_ws = ws
                self.stt_ready = True
                
                logger.info(f"[{self.call_id}] ✅ STT CONNECTED & READY")
                logger.info(f"[{self.call_id}] STT connection opened, remote={ws._connection.remote_address if hasattr(ws,'_connection') else 'unknown'}")
                # Also log the exact parameters sent
                logger.info(f"[{self.call_id}] STT connect params: model=saaras:v3, language_code=ta-IN, sample_rate=16000, input_audio_codec=pcm_s16le")

                reader_task = asyncio.create_task(self._sarvam_stt_reader())

                # Flush prebuffered audio
                if self._prebuffer:
                    logger.info(
                        f"[{self.call_id}] 🔄 Flushing {len(self._prebuffer)} prebuffered chunks (RAW PCM)"
                    )
                    for chunk in self._prebuffer:
                        if self.call_ended:
                            break
                        try:
                            # FIX: Send raw PCM16 bytes directly as base64
                            b64_audio = base64.b64encode(chunk).decode("utf-8")
                            await self.sarvam_stt_ws.transcribe(audio=b64_audio)
                            
                            # Tiny sleep to avoid overwhelming server with burst
                            await asyncio.sleep(0.01)
                        except Exception as e:
                            logger.warning(f"[{self.call_id}] Prebuffer flush error: {e}")
                            break
                    self._prebuffer.clear()

                if hasattr(ws, '_connection') and ws._connection.closed:
                    logger.critical(f"STT WS closed with code {ws._connection.close_code}, reason {ws._connection.close_reason}")

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

        try:
            async for message in self.sarvam_stt_ws:
                if self.call_ended:
                    break

                logger.critical(f"[{self.call_id}] RAW STT MESSAGE: {message}")

                # Parse using the corrected envelope-aware parser
                event_name, text = _parse_sarvam_stt_message(message)
                
                logger.info(
                    f"[{self.call_id}] STT EVENT | "
                    f"event={event_name!r} | text={text!r}"
                )

                # ── speech_start: VAD triggered ──────────────────────────────
                if event_name == "speech_start":
                    self._speech_active = True
                    self._speech_start_time = time.time()
                    logger.info(f"[{self.call_id}] 🎤 Speech started (VAD START_SPEECH)")
                    logger.info(f"🎤 Speech timing [{self._speech_start_time}]")

                    now = time.time()

                    # prevent repeated flush storms
                    if (
                        self.session.tts_playing
                        and (now - self._last_barge_in) > 1.5
                    ):
                        self._last_barge_in = now

                        logger.info(
                            f"[{self.call_id}] 🔴 REAL BARGE-IN DETECTED → flushing TTS"
                        )

                        try:
                            await self.ws.send_text(json.dumps({
                                # "event": "clearAudio",
                                "streamId": self.stream_sid
                            }))
                        except Exception as e:
                            logger.warning(
                                f"[{self.call_id}] clearAudio failed: {e}"
                            )

                        self.session.tts_playing = False
                        await session_save(self.session)

                        # ONLY clear CURRENT playback
                        # await self.tts_queue.put(None)

                # ── transcript: FINAL result ─────────────────────────────────
                elif event_name == "transcript":
                    if not text:
                        logger.debug(f"[{self.call_id}] Empty transcript — ignoring")
                        continue

                    if self._stt_timeout_task and not self._stt_timeout_task.done():
                        self._stt_timeout_task.cancel()

                    logger.info(f"[{self.call_id}] ✅ FINAL TRANSCRIPT: {text!r}")
                    self._speech_active = False
                    self._last_transcript = ""
                    self.utterance_buffer = text
                    self.utterance_ready.set()

                # ── speech_end: VAD silence detected ─────────────────────────
                elif event_name == "speech_end":
                    self._speech_active = False
                    logger.info(f"[{self.call_id}] 🔇 Speech ended (VAD END_SPEECH)")
                    speech_duration = time.time() - self._speech_start_time

                    logger.info(
                        f"[{self.call_id}] speech_duration={speech_duration:.2f}s"
                    )

                    # Ignore tiny bursts (AI echo / click / noise)
                    if speech_duration < 0.4:
                        logger.info(
                            f"[{self.call_id}] Ignoring short speech burst"
                        )
                        self._last_transcript = ""
                        continue

                    # Fallback: use partial if final transcript hasn't arrived yet
                    if self._last_transcript:
                        logger.warning(
                            f"[{self.call_id}] ⚠️ Using partial as final fallback: "
                            f"{self._last_transcript!r}"
                        )
                        self.utterance_buffer = self._last_transcript
                        self.utterance_ready.set()
                        self._last_transcript = ""

                # ── events: heartbeat / other ────────────────────────────────
                elif event_name == "events":
                    # Heartbeat or unrecognised signal_type — no action needed
                    logger.debug(f"[{self.call_id}] STT heartbeat event")

                # ── error: Sarvam sent an error event ───────────────────────
                elif event_name == "error":
                    logger.error(f"[{self.call_id}] ❌ STT ERROR EVENT: {text}")

                else:
                    logger.warning(
                        f"[{self.call_id}] Unknown parsed STT event: {event_name!r} | "
                        f"raw={message}"
                    )

        except Exception as e:
            logger.critical(f"[{self.call_id}] STT reader CRASHED: {e}", exc_info=True)
            logger.critical(f"[{self.call_id}] STT reader exception: {type(e).__name__}: {e}")
            
            if not self.call_ended:
                logger.warning(
                    f"[{self.call_id}] ⚠️ STT reader dropped: {type(e).__name__}: {e}",
                    exc_info=True,
                )
                self.stt_ready = False
                self.sarvam_stt_ws = None
                asyncio.create_task(self._reconnect_stt())

    # ─────────────────────────────────────────────────────────────────────────
    # STT Tiered Fallback Logic (Tiers 1, 2, 3)
    # ─────────────────────────────────────────────────────────────────────────

    def _start_tier1_timeout(self) -> None:
        """Tier 1: Wait for transcript. If not received within short window, proceed to fallback."""
        if self._stt_timeout_task and not self._stt_timeout_task.done():
            self._stt_timeout_task.cancel()
        
        buffer_copy = self._stt_audio_buffer
        self._stt_audio_buffer = b""
        self._stt_timeout_task = asyncio.create_task(self._tier1_timeout_worker(buffer_copy))

    async def _tier1_timeout_worker(self, audio_data: bytes) -> None:
        try:
            # 1.5 second wait for normal streaming STT to return
            await asyncio.sleep(1.5)
            logger.warning(f"[{self.call_id}] ⚠️ Tier 1 Timeout: No valid STT response received.")
            
            # Trigger both Tier 2 and Tier 3 concurrently
            asyncio.create_task(self._trigger_tier2_fallback())
            asyncio.create_task(self._trigger_tier3_retry(audio_data))
        except asyncio.CancelledError:
            logger.debug(f"[{self.call_id}] Tier 1 timeout cancelled (transcript received).")

    async def _trigger_tier2_fallback(self) -> None:
        """Tier 2: Generate and speak a polite apology."""
        logger.info(f"[{self.call_id}] 🛡️ Tier 2: Generating 'I'm Sorry' prompt fallback.")
        
        apology_text = "மன்னிக்கவும், நீங்கள் சொன்னது சரியாக கேட்கவில்லை. மீண்டும் ஒரு முறை சொல்ல முடியுமா?"
        
        system_prompt = (
            "SYSTEM: You are a voice agent. You did not hear the caller clearly "
            "due to a technical issue. Generate a generic, professional, and polite "
            "apology prompt in Tamil asking the caller to repeat themselves. "
            "Keep it brief (under 10 seconds of audio). "
            "RESPOND STRICTLY IN TAMIL TEXT ONLY. DO NOT output any English text, translation, or pronunciations."
        )
        
        try:
            if config.LLM_PROVIDER == "groq":
                import groq
                client = groq.AsyncGroq(api_key=config.GROQ_API_KEY)
                resp = await client.chat.completions.create(
                    model=config.GROQ_MODEL,
                    messages=[{"role": "system", "content": system_prompt}],
                    temperature=0.3,
                    max_tokens=100
                )
                apology_text = resp.choices[0].message.content.strip()
            else:
                import google.generativeai as genai
                model = genai.GenerativeModel(config.GEMINI_MODEL)
                resp = await asyncio.to_thread(
                    model.generate_content,
                    system_prompt
                )
                apology_text = resp.text.strip()
        except Exception as e:
            logger.error(f"[{self.call_id}] LLM apology generation failed: {e}")
            
        logger.info(f"[{self.call_id}] Tier 2 Fallback text: {apology_text}")
        await self._speak(apology_text)

    async def _trigger_tier3_retry(self, audio_data: bytes) -> None:
        """Tier 3: In the background, retry STT using the REST API."""
        from app.modules.voice_agent.services import sarvam_stt_rest
        logger.info(f"[{self.call_id}] 🔄 Tier 3 Retry: Sending {len(audio_data)} bytes to REST STT.")
        try:
            transcript = await sarvam_stt_rest(audio_data, 16000)
            if transcript and transcript.strip():
                logger.info(f"[{self.call_id}] ✅ Tier 3 Success: Got transcript: {transcript!r}")
                
                # Interrupt the Tier 2 fallback TTS
                now = time.time()
                self._last_barge_in = now
                self.session.tts_playing = False
                await session_save(self.session)
                
                try:
                    await self.ws.send_text(json.dumps({"streamId": self.stream_sid}))
                except Exception as ex:
                    logger.warning(f"[{self.call_id}] clearAudio for Tier 3 failed: {ex}")
                
                # Deliver the real response
                self.utterance_buffer = transcript.strip()
                self.utterance_ready.set()
            else:
                logger.warning(f"[{self.call_id}] ❌ Tier 3 Failed: Empty transcript returned.")
        except Exception as e:
            logger.error(f"[{self.call_id}] Tier 3 Error: {e}")

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
    # Scheduling
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
    # TTS
    # ─────────────────────────────────────────────────────────────────────────

    async def _speak(self, text: str) -> None:
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
    # TTS sender
    # ─────────────────────────────────────────────────────────────────────────

    async def _tts_sender(self) -> None:
        """
        Drain the TTS queue and write µ-law audio to Vobiz in 40ms chunks.

        µ-law: 1 byte/sample × 8000 samples/sec = 8000 bytes/sec
        40ms chunk = 8000 × 0.040 = 320 bytes per chunk.
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

                if audio is None:
                    logger.info(
                        f"[{self.call_id}] TTS interrupted by barge-in"
                    )

                    self.session.tts_playing = False
                    await session_save(self.session)

                    continue

                loop = asyncio.get_event_loop()
                next_send_time = loop.time()

                for i in range(0, len(audio), CHUNK_SIZE):
                    if self.call_ended:
                        break
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

                        next_send_time += 0.04
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
    # End call
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

        if self._mulaw_dump and not self._mulaw_dump.closed:
            try:
                self._mulaw_dump.close()
                logger.info(f"[{self.call_id}] Audio dump closed")
            except Exception:
                pass

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
    # Post-call
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
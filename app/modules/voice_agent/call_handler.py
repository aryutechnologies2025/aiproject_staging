import asyncio
import audioop
import json
import uuid
import base64
import logging
from datetime import datetime
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

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
    SARVAM_STT_CONFIG_FRAME,
)

logger = logging.getLogger("voice_agent.call_handler")


def _pcm16_to_mulaw(pcm16_bytes: bytes) -> bytes:
    """
    Convert LINEAR16 PCM audio (from Sarvam TTS) to G.711 µ-law.
    Vobiz WebSocket media stream requires µ-law encoded audio.
    
    PCM16: 2 bytes/sample, 8000 Hz → 16000 bytes/sec
    mulaw: 1 byte/sample, 8000 Hz →  8000 bytes/sec
    """
    return audioop.lin2ulaw(pcm16_bytes, 2)  # 2 = 16-bit (2 bytes per sample)


class CallHandler:
    """
    Manages one outbound call from start to finish.

    Audio pipeline (corrected):
      Vobiz → WS (mulaw 8kHz) → Sarvam STT WS → transcript
      transcript → Groq LLM → response text → Sarvam TTS (PCM16)
      PCM16 → audioop.lin2ulaw() → mulaw → base64 → Vobiz WS media frame

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
                if self.call_ended:
                    break
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                event = msg.get("event")

                if event == "connected":
                    # Vobiz always sends this first — just log it
                    logger.info(f"[{self.call_id}] Vobiz WS connected")

                elif event == "start":
                    self.stream_sid = (
                        msg.get("streamSid")
                        or msg.get("start", {}).get("streamSid", "")
                    )
                    logger.info(f"[{self.call_id}] Stream started | sid={self.stream_sid}")
                    # Unblock _tts_sender — it has been waiting for stream_sid
                    self._stream_sid_ready.set()
                    # Start STT + greeting (non-blocking — don't await)
                    asyncio.create_task(self._on_call_start())

                elif event == "media":
                    payload = msg.get("media", {}).get("payload", "")
                    if payload and not self.simulation_mode:
                        if self.sarvam_stt_ws:
                            try:
                                await self.sarvam_stt_ws.send(
                                    base64.b64decode(payload)
                                )
                            except Exception as e:
                                logger.warning(f"[{self.call_id}] STT send error: {e}")
                                asyncio.create_task(self._reconnect_stt())

                elif event == "stop":
                    logger.info(f"[{self.call_id}] Vobiz stop event received")
                    break

        except WebSocketDisconnect:
            logger.info(f"[{self.call_id}] Vobiz WS disconnected")
        except Exception as e:
            logger.error(f"[{self.call_id}] _vobiz_listener error: {e}", exc_info=True)
        finally:
            await self._end_call()

    # ─────────────────────────────────────────────────────────────────────────
    # Call start — init STT, speak greeting
    # ─────────────────────────────────────────────────────────────────────────

    async def _on_call_start(self) -> None:
        logger.info(f"[{self.call_id}] _on_call_start | sim={self.simulation_mode}")

        if self.simulation_mode:
            asyncio.create_task(self._inject_simulation_utterance())
        else:
            # Start STT WS — utterance_processor launched inside regardless of result
            await self._start_sarvam_stt()

        # Speak greeting — this will block until TTS audio is queued
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
        Connect to Sarvam STT WebSocket.
        IMPORTANT: _utterance_processor is started here unconditionally.
        Even if STT fails, the utterance processor must run (for timeouts + closing).
        """
        # Always start the utterance processor — STT failing must not kill the call
        asyncio.create_task(self._utterance_processor())

        if not config.SARVAM_API_KEY:
            logger.error(f"[{self.call_id}] SARVAM_API_KEY is empty — STT disabled")
            return

        try:
            import websockets

            self.sarvam_stt_ws = await websockets.connect(
                config.SARVAM_STT_WS_URL,
                additional_headers={"API-Subscription-Key": config.SARVAM_API_KEY},
                ping_interval=20,
                ping_timeout=10,
            )
            # Send config frame — required by Sarvam protocol before any audio
            await self.sarvam_stt_ws.send(json.dumps(SARVAM_STT_CONFIG_FRAME))
            logger.info(f"[{self.call_id}] Sarvam STT WS connected ✓")

            asyncio.create_task(self._sarvam_stt_reader())

        except Exception as e:
            logger.error(
                f"[{self.call_id}] Sarvam STT connect FAILED: {e} "
                f"— call continues in TTS-only mode"
            )
            self.sarvam_stt_ws = None
            # utterance_processor is already running — it will timeout after 60s
            # and end the call gracefully via _closing_sequence

    async def _reconnect_stt(self) -> None:
        if self.call_ended:
            return
        logger.warning(f"[{self.call_id}] STT WS dropped — reconnecting in 1s")
        try:
            if self.sarvam_stt_ws:
                await self.sarvam_stt_ws.close()
        except Exception:
            pass
        self.sarvam_stt_ws = None
        await asyncio.sleep(1.0)
        if not self.call_ended:
            await self._start_sarvam_stt()

    async def _sarvam_stt_reader(self) -> None:
        """
        Read transcript events from Sarvam STT WS.

        Sarvam event types:
          {"type": "partial", "transcript": "..."}  — user still speaking → barge-in
          {"type": "final",   "transcript": "..."}  — utterance complete
          {"type": "error",   "message": "..."}      — STT error
        """
        try:
            async for raw in self.sarvam_stt_ws:
                if self.call_ended:
                    break
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue

                msg_type = msg.get("type", "")
                transcript = msg.get("transcript", "").strip()

                if msg_type == "partial" and transcript:
                    # Barge-in: user speaking → stop TTS immediately
                    if self.session.tts_playing:
                        logger.info(f"[{self.call_id}] Barge-in → flushing TTS")
                        self.session.tts_playing = False
                        await session_save(self.session)
                        await self.tts_queue.put(None)  # flush sentinel

                elif msg_type == "final" and transcript:
                    logger.info(f"[{self.call_id}] STT final: {transcript[:80]}")
                    self.utterance_buffer += " " + transcript
                    self.utterance_ready.set()

                elif msg_type == "error":
                    logger.error(f"[{self.call_id}] STT error: {msg.get('message')}")

        except Exception as e:
            if not self.call_ended:
                logger.warning(f"[{self.call_id}] _sarvam_stt_reader error: {e}")
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

            logger.info(f"[{self.call_id}] Processing: {text[:80]}")

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

            # Detect which slot the lead chose (if in scheduling state)
            if self.session.proposed_slots:
                self._chosen_slot_index = self._detect_slot_choice(
                    text, len(self.session.proposed_slots)
                )

            if advance and self.session.script_pos < len(steps) - 1:
                self.session.script_pos += 1
                logger.info(f"[{self.call_id}] Script advanced → pos {self.session.script_pos}")

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
        # Wait for TTS to finish playing before we close
        await asyncio.sleep(4)
        await self._end_call()

    # ─────────────────────────────────────────────────────────────────────────
    # TTS — speak text → convert to mulaw → queue for sending
    # ─────────────────────────────────────────────────────────────────────────

    async def _speak(self, text: str) -> None:
        """
        Fetch TTS audio from Sarvam, convert PCM16 → mulaw, enqueue.
        Sarvam returns LINEAR16 PCM. Vobiz needs µ-LAW (G.711).
        """
        if not text or self.call_ended:
            return

        self.session.tts_playing = True
        await session_save(self.session)

        try:
            # Sarvam TTS returns LINEAR16 PCM bytes
            pcm16_bytes = await sarvam_tts(text)
            if not pcm16_bytes:
                logger.warning(f"[{self.call_id}] TTS returned empty audio for: {text[:40]}")
                self.session.tts_playing = False
                await session_save(self.session)
                return

            # CRITICAL: Convert PCM16 → µ-law before sending to Vobiz
            mulaw_bytes = _pcm16_to_mulaw(pcm16_bytes)
            logger.info(
                f"[{self.call_id}] TTS ready | "
                f"pcm16={len(pcm16_bytes)}B → mulaw={len(mulaw_bytes)}B | "
                f"text={text[:40]}"
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
        Drain the TTS queue and write µ-law audio to Vobiz in 20ms chunks.

        µ-law encoding: 1 byte/sample × 8000 samples/sec = 8000 bytes/sec
        20ms chunk = 8000 × 0.020 = 160 bytes per chunk.

        MUST wait for stream_sid before sending any audio —
        Vobiz will reject frames with a missing streamSid.
        """
        # Wait until Vobiz sends the "start" event with streamSid
        try:
            await asyncio.wait_for(self._stream_sid_ready.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.error(f"[{self.call_id}] stream_sid never arrived — aborting TTS sender")
            return

        logger.info(f"[{self.call_id}] TTS sender ready | sid={self.stream_sid}")

        # µ-law: 1 byte/sample, 8kHz, 20ms = 160 bytes/chunk
        CHUNK_SIZE = 160

        while not self.call_ended:
            try:
                audio = await asyncio.wait_for(self.tts_queue.get(), timeout=1.0)

                if audio is None:
                    # Barge-in flush sentinel — drain queue and stop
                    while not self.tts_queue.empty():
                        try:
                            self.tts_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    self.session.tts_playing = False
                    await session_save(self.session)
                    continue

                # Send in 160-byte µ-law chunks with 20ms pacing
                for i in range(0, len(audio), CHUNK_SIZE):
                    if self.call_ended:
                        break
                    # Barge-in check: if new item in queue, stop mid-stream
                    if self.tts_queue.qsize() > 0:
                        break

                    chunk = audio[i: i + CHUNK_SIZE]
                    try:
                        await self.ws.send_text(json.dumps({
                            "event": "media",
                            "streamSid": self.stream_sid,
                            "media": {
                                "payload": base64.b64encode(chunk).decode("ascii")
                            },
                        }))
                    except Exception as send_err:
                        logger.warning(f"[{self.call_id}] WS send error: {send_err}")
                        break

                    await asyncio.sleep(0.020)  # 20ms pacing = real-time audio

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
        logger.info(f"[{self.call_id}] Ending call | lead={self.lead.name}")

        # Signal _tts_sender to stop waiting if it's still blocked on stream_sid
        self._stream_sid_ready.set()

        # Close Sarvam STT WS gracefully
        if self.sarvam_stt_ws and not self.simulation_mode:
            try:
                await self.sarvam_stt_ws.send(json.dumps({"type": "end_of_stream"}))
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

            # await send_sms(lead.phone, build_interview_sms(lead.name, company.name, slot))
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
            # await send_sms(lead.phone, build_recall_sms(lead.name, company.name, recall_at))
            logger.info(f"[{self.call_id}] Recall @ {recall_at.isoformat()}")

        await db.update_lead(lead.id, update_data)
        await db.increment_call_attempts(lead.id)
        logger.info(
            f"[{self.call_id}] Post-call done | "
            f"status={update_data['status']} | score={session.lead_score}"
        )
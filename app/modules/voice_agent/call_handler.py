"""
call_handler.py — WebSocket call handler

STT: Sarvam AI streaming (saarika:v2)
TTS: Sarvam AI (bulbul:v2)
LLM: Groq / Gemini

Audio flow:
  Vobiz → WebSocket (mulaw 8kHz) → Sarvam STT WS → transcript
  transcript → LLM → response text → Sarvam TTS → PCM16
  PCM16 → base64 → Vobiz WebSocket → phone speaker

Key production fixes in this version:
  1. Sarvam STT replaces Deepgram — better Tamil ASR
  2. Barge-in: user speaking while TTS playing → flush TTS queue immediately
  3. Correct slot selection — tracks which slot lead confirmed (not always slot 0)
  4. Clean simulation mode — no Sarvam STT WS in simulation (uses fake utterance)
  5. Robust _end_call — idempotent, handles partial cleanup gracefully
  6. STT reconnect on WS drop
"""

import asyncio
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


class CallHandler:
    """
    Manages one outbound call from start to finish.

    Concurrency model:
      • _vobiz_listener()   — reads WS frames from Vobiz, routes audio to STT
      • _sarvam_stt_reader() — reads transcripts from Sarvam STT WS
      • _utterance_processor() — consumes final transcripts, calls LLM, speaks
      • _tts_sender()        — drains the TTS queue and writes audio to Vobiz WS
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

        # Sarvam STT WebSocket
        self.sarvam_stt_ws = None

        # Utterance state
        self.utterance_buffer: str = ""
        self.utterance_ready: asyncio.Event = asyncio.Event()

        # TTS queue — audio_bytes or None (None = barge-in flush signal)
        self.tts_queue: asyncio.Queue = asyncio.Queue()

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
            f"[{self.call_id}] Call session started | "
            f"lead={self.lead.name} ({self.lead.phone})"
        )
        try:
            await asyncio.gather(
                self._vobiz_listener(),
                self._tts_sender(),
            )
        except Exception as e:
            logger.error(f"[{self.call_id}] Unhandled error in call gather: {e}")
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

                if event == "start":
                    self.stream_sid = msg.get("streamSid", "")
                    await self._on_call_start()

                elif event == "media":
                    payload = msg.get("media", {}).get("payload", "")
                    if payload and self.sarvam_stt_ws and not self.simulation_mode:
                        try:
                            # Send raw mulaw bytes to Sarvam STT WS
                            await self.sarvam_stt_ws.send(
                                base64.b64decode(payload)
                            )
                        except Exception:
                            # STT WS dropped — attempt reconnect
                            await self._reconnect_stt()

                elif event == "stop":
                    logger.info(f"[{self.call_id}] Vobiz sent stop event")
                    break

        except WebSocketDisconnect:
            logger.info(f"[{self.call_id}] Vobiz WS disconnected")
        except Exception as e:
            logger.error(f"[{self.call_id}] _vobiz_listener error: {e}")
        finally:
            await self._end_call()

    # ─────────────────────────────────────────────────────────────────────────
    # Call start: init STT, send greeting
    # ─────────────────────────────────────────────────────────────────────────

    async def _on_call_start(self) -> None:
        logger.info(f"[{self.call_id}] Call started | simulation={self.simulation_mode}")

        if not self.simulation_mode:
            await self._start_sarvam_stt()
        else:
            # Simulation: fake an utterance after greeting so the loop runs
            asyncio.create_task(self._inject_simulation_utterance())

        # Always speak the greeting
        greeting = self.script.steps[0]["question"].replace("{name}", self.lead.name)
        await self._speak(greeting)

    async def _inject_simulation_utterance(self) -> None:
        """
        In simulation mode there is no real audio.
        Wait for the greeting TTS to finish, then inject a canned response
        so the utterance processor fires and the script advances.
        Only used for local testing.
        """
        await asyncio.sleep(4)
        self.utterance_buffer = "ஆமா, நான் வேலை தேடுகிறேன்"
        self.utterance_ready.set()
        asyncio.create_task(self._utterance_processor())

    # ─────────────────────────────────────────────────────────────────────────
    # Sarvam STT — streaming WebSocket
    # ─────────────────────────────────────────────────────────────────────────

    async def _start_sarvam_stt(self) -> None:
        import websockets

        url = config.SARVAM_STT_WS_URL
        headers = {"API-Subscription-Key": config.SARVAM_API_KEY}

        try:
            self.sarvam_stt_ws = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10,
            )
            # Send config frame first — required by Sarvam protocol
            await self.sarvam_stt_ws.send(json.dumps(SARVAM_STT_CONFIG_FRAME))
            logger.info(f"[{self.call_id}] Sarvam STT WS connected")

            asyncio.create_task(self._sarvam_stt_reader())
            asyncio.create_task(self._utterance_processor())

        except Exception as e:
            logger.error(f"[{self.call_id}] Sarvam STT connect failed: {e}")
            # Fallback: continue without STT (agent will still speak greeting)

    async def _reconnect_stt(self) -> None:
        """Attempt to reconnect Sarvam STT WS after a drop."""
        if self.call_ended:
            return
        logger.warning(f"[{self.call_id}] STT WS dropped — reconnecting")
        try:
            if self.sarvam_stt_ws:
                await self.sarvam_stt_ws.close()
        except Exception:
            pass
        await asyncio.sleep(0.5)
        await self._start_sarvam_stt()

    async def _sarvam_stt_reader(self) -> None:
        """
        Reads transcript events from Sarvam STT WebSocket.

        Event types from Sarvam:
          {"type": "partial",  "transcript": "..."}  — interim (user still speaking)
          {"type": "final",    "transcript": "..."}  — utterance complete
          {"type": "error",    "message": "..."}      — STT error

        Barge-in: when we get a "partial" event while TTS is playing,
        we flush the TTS queue immediately (agent stops speaking).
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
                    # Barge-in: user started speaking → flush TTS
                    if self.session.tts_playing:
                        logger.info(
                            f"[{self.call_id}] Barge-in detected — flushing TTS"
                        )
                        self.session.tts_playing = False
                        await session_save(self.session)
                        # None is the flush sentinel
                        await self.tts_queue.put(None)

                elif msg_type == "final" and transcript:
                    logger.info(
                        f"[{self.call_id}] Final transcript: {transcript[:80]}"
                    )
                    self.utterance_buffer += " " + transcript
                    self.utterance_ready.set()

                elif msg_type == "error":
                    logger.error(
                        f"[{self.call_id}] Sarvam STT error: {msg.get('message')}"
                    )

        except Exception as e:
            if not self.call_ended:
                logger.warning(f"[{self.call_id}] _sarvam_stt_reader dropped: {e}")
                await self._reconnect_stt()

    # ─────────────────────────────────────────────────────────────────────────
    # Utterance processor — LLM → speech → script advance
    # ─────────────────────────────────────────────────────────────────────────

    async def _utterance_processor(self) -> None:
        steps = self.script.steps

        while not self.call_ended:
            try:
                await asyncio.wait_for(self.utterance_ready.wait(), timeout=60.0)
            except asyncio.TimeoutError:
                # No speech for 60s — end call gracefully
                logger.info(f"[{self.call_id}] 60s silence — ending call")
                await self._closing_sequence()
                return

            self.utterance_ready.clear()
            text = self.utterance_buffer.strip()
            self.utterance_buffer = ""

            if not text:
                continue

            logger.info(f"[{self.call_id}] Processing utterance: {text[:80]}")

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

            # ── Check if lead mentioned a slot number ─────────────────────────
            self._chosen_slot_index = self._detect_slot_choice(
                text, len(self.session.proposed_slots)
            )

            # ── Advance script position ───────────────────────────────────────
            if advance and self.session.script_pos < len(steps) - 1:
                self.session.script_pos += 1

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
    # Scheduling
    # ─────────────────────────────────────────────────────────────────────────

    async def _handle_scheduling(self, pre_speech: str) -> None:
        """Offer available interview slots to the lead."""
        await self._speak(pre_speech)

        slots = await get_calendar_slots(config.INTERVIEW_SLOTS_LOOKAHEAD_DAYS)
        if not slots:
            await self._speak(
                "நேர்காணலுக்கான நேரம் இப்போது இல்லை. "
                "விரைவில் திரும்ப அழைக்கிறோம்."
            )
            return

        self.session.proposed_slots = [s.isoformat() for s in slots]

        slot_lines = []
        for i, slot in enumerate(slots[:3]):
            from datetime import timedelta
            ist = slot + timedelta(hours=5, minutes=30)
            slot_lines.append(
                f"விருப்பம் {i + 1}: {ist.strftime('%d/%m %I:%M %p')}"
            )

        slot_prompt = (
            f"நேர்காணலுக்கு இந்த நேரங்கள் இருக்கின்றன: "
            f"{', '.join(slot_lines)}. "
            f"எந்த நேரம் உங்களுக்கு வசதியாக இருக்கும்? "
            f"ஒன்று, இரண்டு அல்லது மூன்று என்று சொல்லுங்கள்."
        )
        await self._speak(slot_prompt)
        await session_save(self.session)

    @staticmethod
    def _detect_slot_choice(text: str, num_slots: int) -> int:
        """
        Heuristic: detect which slot number the lead said.
        Returns 0-based index (default 0 if unclear).
        """
        import re
        # Tamil number words
        ta_map = {
            "ஒன்று": 0, "முதல்": 0, "first": 0,
            "இரண்டு": 1, "second": 1,
            "மூன்று": 2, "third": 2,
        }
        text_lower = text.lower()
        for word, idx in ta_map.items():
            if word in text_lower and idx < num_slots:
                return idx
        # Digit mentions: "1", "2", "3"
        m = re.search(r'\b([123])\b', text)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < num_slots:
                return idx
        return 0

    # ─────────────────────────────────────────────────────────────────────────
    # Closing sequence
    # ─────────────────────────────────────────────────────────────────────────

    async def _closing_sequence(self) -> None:
        self.session.state = CallState.CLOSING
        score = self.session.lead_score
        name = self.lead.name

        if score == "hot":
            msg = self.script.closing_hot.replace("{name}", name)
        elif score == "warm":
            msg = self.script.closing_warm.replace("{name}", name)
        else:
            msg = self.script.closing_cold.replace("{name}", name)

        await self._speak(msg)
        # Give TTS time to finish before hanging up
        await asyncio.sleep(3)
        await self._end_call()

    # ─────────────────────────────────────────────────────────────────────────
    # TTS → Vobiz audio sender
    # ─────────────────────────────────────────────────────────────────────────

    async def _speak(self, text: str) -> None:
        if not text or self.call_ended:
            return
        self.session.tts_playing = True
        await session_save(self.session)
        try:
            audio_bytes = await sarvam_tts(text)
            if audio_bytes:
                await self.tts_queue.put(audio_bytes)
        except Exception as e:
            logger.error(f"[{self.call_id}] TTS error: {e}")
            self.session.tts_playing = False
            await session_save(self.session)

    async def _tts_sender(self) -> None:
        """
        Drains tts_queue and writes audio chunks to Vobiz WS.
        None in the queue = barge-in flush — discard current audio and stop.
        """
        while not self.call_ended:
            try:
                audio = await asyncio.wait_for(self.tts_queue.get(), timeout=1.0)

                if audio is None:
                    # Barge-in: drain remaining items, stop playback
                    while not self.tts_queue.empty():
                        try:
                            self.tts_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    self.session.tts_playing = False
                    await session_save(self.session)
                    continue

                # Stream audio in 20ms chunks (320 bytes @ 8kHz 16-bit)
                chunk_size = 320
                for i in range(0, len(audio), chunk_size):
                    if self.call_ended or self.tts_queue.qsize() > 0:
                        # Stop mid-stream if barge-in or call ended
                        break
                    chunk = audio[i: i + chunk_size]
                    try:
                        await self.ws.send_text(json.dumps({
                            "event": "media",
                            "streamSid": self.stream_sid,
                            "media": {"payload": base64.b64encode(chunk).decode()},
                        }))
                    except Exception:
                        break
                    await asyncio.sleep(0.02)

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
        logger.info(f"[{self.call_id}] Ending call for {self.lead.name}")

        # Close Sarvam STT WS
        if self.sarvam_stt_ws and not self.simulation_mode:
            try:
                await self.sarvam_stt_ws.send(
                    json.dumps({"type": "end_of_stream"})
                )
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
    # Post-call: update lead, calendar, SMS
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

        # ── Case 1: Interview scheduled ────────────────────────────────────
        if (
            "interview_requested" in session.intent_flags
            and session.proposed_slots
        ):
            from datetime import datetime as dt
            # Use the slot the lead actually chose (not always index 0)
            slot_idx = min(self._chosen_slot_index, len(session.proposed_slots) - 1)
            slot = dt.fromisoformat(session.proposed_slots[slot_idx])

            event_id = await create_calendar_event(
                lead, company, slot, session.call_id
            )
            update_data["status"]                = LeadStatus.SCHEDULED
            update_data["scheduled_interview_at"] = slot

            # await send_sms(
            #     lead.phone,
            #     build_interview_sms(lead.name, company.name, slot),
            # )
            await db.create_interview_slot(
                lead_id=lead.id,
                call_id=session.call_id,
                company_id=company.id,
                scheduled_at=slot,
                calendar_event_id=event_id,
                sms_sent=True,
            )
            logger.info(
                f"[{self.call_id}] Interview scheduled for {lead.name} @ "
                f"{slot.isoformat()}"
            )

        # ── Case 2: Warm lead or callback requested ────────────────────────
        elif (
            session.lead_score == "warm"
            or "callback_requested" in session.intent_flags
        ):
            from datetime import datetime as dt, timedelta
            recall_at = dt.utcnow() + timedelta(hours=config.RECALL_AFTER_HOURS)
            update_data["status"]       = LeadStatus.RECALL
            update_data["next_call_at"] = recall_at
            # await send_sms(
            #     lead.phone,
            #     build_recall_sms(lead.name, company.name, recall_at),
            # )
            logger.info(
                f"[{self.call_id}] Recall scheduled for {lead.name} @ "
                f"{recall_at.isoformat()}"
            )

        await db.update_lead(lead.id, update_data)
        await db.increment_call_attempts(lead.id)
        logger.info(
            f"[{self.call_id}] Post-call done | status={update_data['status']} | "
            f"score={session.lead_score} ({session.score_confidence})"
        )
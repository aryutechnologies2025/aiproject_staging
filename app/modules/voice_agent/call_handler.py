import asyncio
import json
import uuid
import base64
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
)

_DG_LISTEN_URL = "wss://api.deepgram.com/v1/listen"


class CallHandler:
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
        self.dg_ws = None
        self.utterance_buffer = ""
        self.utterance_ready = asyncio.Event()
        self.tts_queue: asyncio.Queue = asyncio.Queue()
        self.call_ended = False
        self.stream_sid: Optional[str] = None

    async def start(self):
        await self.ws.accept()
        await session_save(self.session)
        await asyncio.gather(self._vobiz_listener(), self._tts_sender())

    async def _vobiz_listener(self):
        try:
            async for raw in self.ws.iter_text():
                if self.call_ended:
                    break
                msg = json.loads(raw)
                event = msg.get("event")

                if event == "start":
                    self.stream_sid = msg.get("streamSid", "")
                    await self._on_call_start()
                elif event == "media":
                    payload = msg.get("media", {}).get("payload", "")
                    if payload and self.dg_ws:
                        await self.dg_ws.send(base64.b64decode(payload))
                elif event == "stop":
                    await self._end_call()
        except WebSocketDisconnect:
            await self._end_call()
        except Exception:
            await self._end_call()

    async def _on_call_start(self):
        await self._start_deepgram()
        greeting = self.script.steps[0]["question"].replace("{name}", self.lead.name)
        await self._speak(greeting)

    def _build_dg_url(self) -> str:
        lang = self.company.language or "ta"
        params = (
            f"model=nova-2"
            f"&language={lang}"
            f"&punctuate=true"
            f"&interim_results=true"
            f"&endpointing={config.DEEPGRAM_ENDPOINTING_MS}"
            f"&smart_format=true"
            f"&encoding=mulaw"
            f"&sample_rate=8000"
        )
        return f"{_DG_LISTEN_URL}?{params}"

    async def _start_deepgram(self):
        import websockets

        url = self._build_dg_url()
        headers = {"Authorization": f"Token {config.DEEPGRAM_API_KEY}"}
        self.dg_ws = await websockets.connect(url, additional_headers=headers)
        asyncio.create_task(self._deepgram_receiver())
        asyncio.create_task(self._utterance_processor())

    async def _deepgram_receiver(self):
        try:
            async for raw in self.dg_ws:
                if self.call_ended:
                    break
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "Results":
                    alternatives = msg.get("channel", {}).get("alternatives", [])
                    if not alternatives:
                        continue
                    transcript = alternatives[0].get("transcript", "").strip()
                    if not transcript:
                        continue

                    is_final = msg.get("is_final", False)
                    speech_final = msg.get("speech_final", False)

                    if is_final and speech_final:
                        self.utterance_buffer += " " + transcript
                        self.utterance_ready.set()
                    elif not is_final:
                        if self.session.tts_playing:
                            self.session.tts_playing = False
                            await session_save(self.session)
                            await self.tts_queue.put(None)

                elif msg_type == "SpeechStarted":
                    if self.session.tts_playing:
                        self.session.tts_playing = False
                        await session_save(self.session)
                        await self.tts_queue.put(None)

                elif msg_type == "UtteranceEnd":
                    if self.utterance_buffer.strip():
                        self.utterance_ready.set()

        except Exception:
            pass

    async def _utterance_processor(self):
        steps = self.script.steps

        while not self.call_ended:
            await self.utterance_ready.wait()
            self.utterance_ready.clear()

            text = self.utterance_buffer.strip()
            self.utterance_buffer = ""
            if not text:
                continue

            self.session.history.append({"role": "user", "content": text})
            self.session.transcript_full += f"\nLead: {text}"
            await session_save(self.session)

            result = await llm_respond(self.session, self.script, self.company)

            speech = result.get("speech", "")
            self.session.lead_score = result.get("lead_score", self.session.lead_score)
            self.session.score_confidence = result.get("score_confidence", 0)

            for f in result.get("intent_flags", []):
                if f not in self.session.intent_flags:
                    self.session.intent_flags.append(f)

            if result.get("advance_script") and self.session.script_pos < len(steps) - 1:
                self.session.script_pos += 1

            self.session.history.append({"role": "assistant", "content": speech})
            self.session.transcript_full += f"\nAgent: {speech}"

            if "interview_requested" in self.session.intent_flags and self.session.state != CallState.SCHEDULING:
                self.session.state = CallState.SCHEDULING
                await self._handle_scheduling(speech)
            elif result.get("should_end_call") or self.session.script_pos >= len(steps) - 1:
                await self._speak(speech)
                await asyncio.sleep(1)
                await self._closing_sequence()
            else:
                await self._speak(speech)

            await session_save(self.session)

    async def _handle_scheduling(self, pre_speech: str):
        await self._speak(pre_speech)

        slots = await get_calendar_slots(config.INTERVIEW_SLOTS_LOOKAHEAD_DAYS)
        if not slots:
            await self._speak("நேர்காணலுக்கான நேரம் இப்போது இல்லை. விரைவில் திரும்ப அழைக்கிறோம்.")
            return

        self.session.proposed_slots = [s.isoformat() for s in slots]
        from datetime import timedelta
        slot_lines = []
        for i, slot in enumerate(slots[:3]):
            ist = slot + timedelta(hours=5, minutes=30)
            slot_lines.append(f"விருப்பம் {i+1}: {ist.strftime('%d/%m %I:%M %p')}")

        slot_prompt = (
            f"நேர்காணலுக்கு இந்த நேரங்கள் இருக்கின்றன: {', '.join(slot_lines)}. "
            f"எந்த நேரம் உங்களுக்கு வசதியாக இருக்கும்?"
        )
        await self._speak(slot_prompt)
        await session_save(self.session)

    async def _closing_sequence(self):
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
        await asyncio.sleep(2)
        await self._end_call()

    async def _speak(self, text: str):
        if not text:
            return
        self.session.tts_playing = True
        await session_save(self.session)
        try:
            audio_bytes = await sarvam_tts(text)
            await self.tts_queue.put(audio_bytes)
        except Exception:
            self.session.tts_playing = False
            await session_save(self.session)

    async def _tts_sender(self):
        while not self.call_ended:
            try:
                audio = await asyncio.wait_for(self.tts_queue.get(), timeout=1.0)
                if audio is None:
                    self.session.tts_playing = False
                    await session_save(self.session)
                    continue
                chunk_size = 320
                for i in range(0, len(audio), chunk_size):
                    if self.call_ended:
                        break
                    chunk = audio[i:i+chunk_size]
                    await self.ws.send_text(json.dumps({
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {"payload": base64.b64encode(chunk).decode()},
                    }))
                    await asyncio.sleep(0.02)
                self.session.tts_playing = False
                await session_save(self.session)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break

    async def _end_call(self):
        if self.call_ended:
            return
        self.call_ended = True
        if self.dg_ws:
            try:
                await self.dg_ws.send(json.dumps({"type": "CloseStream"}))
                await self.dg_ws.close()
            except Exception:
                pass
        await self._post_call_actions()
        await session_delete(self.call_id)
        try:
            await self.ws.close()
        except Exception:
            pass

    async def _post_call_actions(self):
        session = self.session
        lead = self.lead
        company = self.company

        new_status = score_to_status(session.lead_score)
        update_data = {
            "status": new_status,
            "notes": session.transcript_full[-2000:],
            "score": session.score_confidence,
        }

        if "interview_requested" in session.intent_flags and session.proposed_slots:
            from datetime import datetime as dt
            slot = dt.fromisoformat(session.proposed_slots[0])
            event_id = await create_calendar_event(lead, company, slot, session.call_id)

            update_data["status"] = LeadStatus.SCHEDULED
            update_data["scheduled_interview_at"] = slot

            await send_sms(lead.phone, build_interview_sms(lead.name, company.name, slot))
            await db.create_interview_slot(
                lead_id=lead.id,
                call_id=session.call_id,
                company_id=company.id,
                scheduled_at=slot,
                calendar_event_id=event_id,
                sms_sent=True,
            )

        elif session.lead_score == "warm" or "callback_requested" in session.intent_flags:
            from datetime import datetime as dt, timedelta
            recall_at = dt.utcnow() + timedelta(hours=config.RECALL_AFTER_HOURS)
            update_data["status"] = LeadStatus.RECALL
            update_data["next_call_at"] = recall_at
            await send_sms(lead.phone, build_recall_sms(lead.name, company.name, recall_at))

        await db.update_lead(lead.id, update_data)
        await db.increment_call_attempts(lead.id)
        
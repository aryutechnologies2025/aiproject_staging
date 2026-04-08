from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

# --- Incoming Webhook Schemas (Vapi/Retell) ---

class VapiMessage(BaseModel):
    # Depending on Vapi's exact webhook structure; this is a generalized version
    type: str
    call: Dict[str, Any]
    transcript: Optional[str] = None

class VapiWebhookPayload(BaseModel):
    message: VapiMessage

# --- Gemini Structured Output Schemas ---

class LeadAnalysisResult(BaseModel):
    lead_score: str = Field(description="Must be exactly 'Hot', 'Warm', or 'Cold'")
    summary: str = Field(description="A concise 2-sentence summary of the conversation and user intent.")
    follow_up_date: Optional[str] = Field(description="ISO 8601 formatted datetime string if a follow-up is needed, else null.")

# --- API Response Schemas ---

class WebhookResponse(BaseModel):
    status: str
    message: str

    
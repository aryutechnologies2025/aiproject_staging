from pydantic import BaseModel

class PromptBase(BaseModel):
    agent_name: str
    description: str | None = None
    system_prompt: str

class PromptCreate(PromptBase):
    pass

class PromptUpdate(BaseModel):
    description: str | None = None
    system_prompt: str | None = None

class PromptResponse(PromptBase):
    id: int

    model_config = {
        "from_attributes": True
    }

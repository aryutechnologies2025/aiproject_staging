# hrms_ai/llm/adapter_loader.py

class LLMAdapter:

    def __init__(self, adapter_name: str = "default"):
        self.adapter_name = adapter_name

    def get_adapter(self):
        return self.adapter_name

from typing import List, Optional
from pydantic import BaseModel

class OAChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionsRequest(BaseModel):
    model: Optional[str] = None
    messages: List[OAChatMessage]
    max_tokens: Optional[int] = 300
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False

class IngestTextRequest(BaseModel):
    source: str = "poc_doc"
    text: str

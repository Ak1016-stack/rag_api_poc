from typing import Any, Dict, Literal, Union
from pydantic import BaseModel, Field, ConfigDict

class ToolCall(BaseModel):
    model_config  = ConfigDict(extra="forbid")
    type: Literal["tool_call"]
    tool: Literal["calc"]
    args: Dict[str, Any] = Field(default_factory=dict)

class ToolCallAlt(BaseModel):
    model_config  = ConfigDict(extra="forbid")
    type: Literal["calc"]
    args: Dict[str, Any] = Field(default_factory=dict)

class FinalAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["final"]
    answer: Literal["use_rag"]

RouterRaw = Union[ToolCall, ToolCallAlt, FinalAnswer]
RouterOutput = Union[ToolCall, FinalAnswer]

def normalize_router_output(obj: Dict[str, Any]) -> RouterOutput:
    parsed = None
    for cls in (ToolCall, ToolCallAlt, FinalAnswer):
        try:
            parsed = cls.model_validate(obj)
            break
        except Exception:
            pass
    if parsed is None:
        raise ValueError("Router output invalid")

    if isinstance(parsed, ToolCallAlt):
        return ToolCall(type="tool_call", tool="calc", args=parsed.args)
    return parsed

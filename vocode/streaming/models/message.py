from enum import Enum
from .model import TypedModel
from typing import Optional
from enum import Enum


class MessageType(str, Enum):
    BASE = "message_base"
    SSML = "message_ssml"


class BaseMessage(TypedModel, type=MessageType.BASE):
    text: Optional[str] = None
    intent: Optional[str] = None

class JSONStrMessage:
    def __init__(self, json_str):
        self.json_str = json_str

class SSMLMessage(BaseMessage, type=MessageType.SSML):
    ssml: str

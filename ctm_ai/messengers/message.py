import uuid
from typing import Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    pk: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: Optional[str] = Field(default=None)

    class Config:
        extra = 'allow'

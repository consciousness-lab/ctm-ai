import uuid
from typing import List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    pk: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str = Field(default='assistant')
    query: Optional[str] = Field(default=None)
    content: Optional[str] = Field(default=None)
    relevance: Optional[float] = Field(default=None)
    confidence: Optional[float] = Field(default=None)
    surprise: Optional[float] = Field(default=None)
    weight: Optional[float] = Field(default=None)
    gist: Optional[str] = Field(default=None)
    gists: Optional[List[str]] = Field(default=[])

    class Config:
        extra = 'allow'

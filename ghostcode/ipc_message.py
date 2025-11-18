# ghostcode.ipc_message
# Message types for the inter process communication layer
from typing import *
from pydantic import BaseModel, Field, model_validator, TypeAdapter
from . import types

class IPCMessageBase(BaseModel):
    """Abstract base class for IPC messages."""
    type: str

    # this doesn't have any methods yet, but we are leaving it open for the future
class IPCNotification(IPCMessageBase):
    """Represents a simple message notification that might be displayed to the user."""
    type: Literal["IPCNotification"] = "IPCNotification"
    text: str = Field(
        description = "The text of the notification."
    )
    
class IPCCoderQuery(IPCMessageBase):
    """Represents a request to query the ghostcoder model."""

    query_coder_action: types.ActionQueryCoder = Field(
        description = "The action that will be used to query the coder."
    )

# sum type
IPCMessage = Annotated[
    IPCNotification | IPCCoderQuery,
    Field(discriminator="type")
]

IPCMessageAdapter: TypeAdapter[IPCMessage] = TypeAdapter(IPCMessage) 

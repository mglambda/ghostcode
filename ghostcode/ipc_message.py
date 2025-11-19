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
    client: str = Field(
        default = "",
        description = "The displayable name of the client that sent the notification."
    )
    
    text: str = Field(
        description = "The text of the notification."
    )
    
class IPCActions(IPCMessageBase):
    """Represents a request to put an action on the action queue."""
    type: Literal["IPCAction"] = "IPCAction"
    
    client: str = Field(
        default = "",
        description = "Displayable name of the client that sent the message. Used for user feedback."
    )

    text: str = Field(
        default = "",
        description = "Additional text accompanying the acions. This will be displayed to the user to explain what is happening."
    )
    
    actions: List[types.Action] = Field(
        description = "The actions that will be put on the action queue."
    )

# sum type
IPCMessage = Annotated[
    IPCNotification | IPCActions,
    Field(discriminator="type")
]

IPCMessageAdapter: TypeAdapter[IPCMessage] = TypeAdapter(IPCMessage) 

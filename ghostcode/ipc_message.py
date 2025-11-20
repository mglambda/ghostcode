# ghostcode.ipc_message
# Message types for the inter process communication layer
from typing import *
from pydantic import BaseModel, Field, model_validator, TypeAdapter
from . import types
from .nag_sources import NagSource, NagCheckResult
from .utility import timestamp_now_iso8601

class IPCMessageBase(BaseModel):
    """Abstract base class for IPC messages."""
    type: str

    timestamp: str = Field(
        default_factory = timestamp_now_iso8601,
        description = "The date the mesage was constructed."
    )


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
class ProblematicSourceReport(BaseModel):
    source: NagSource
    result: NagCheckResult
    
class IPCNag(IPCMessageBase):
    """Represents a status report by the nag subcommand, containing recent errors from various nag sources."""
    type: Literal["IPCNag"] = "IPCNag"

    problematic_sources: List[ProblematicSourceReport] = Field(
        default_factory = list,
        description = "List containing nag sources and their respective check results."
    )
    
# sum type
type IPCMessage = Annotated[
    IPCNotification | IPCActions | IPCNag,
    Field(discriminator="type")
]

IPCMessageAdapter: TypeAdapter[IPCMessage] = TypeAdapter(IPCMessage) 


class IPCResponseBase(BaseModel):
    """ABC for results that are returned by the IPC server as a response to IPCMessages."""
    type: str
    timestamp: str = Field(
        default_factory = timestamp_now_iso8601,
        description = "The date the mesage was constructed."
    )

class IPCROk(IPCResponseBase):
    """Represents a result that signals that the request was received and successfully processed."""
    type: Literal["IPCROk"] = "IPCROk"
    
class IPCRContextFiles(IPCResponseBase):
    """Returns the files that are currently in context."""
    type: Literal["IPCRContextFiles"] = "IPCRContextFiles"
    context_files: types.ContextFiles

type IPCResponse = Annotated[
    IPCROk | IPCRContextFiles,
    Field(discriminator = "type")
]

IPCResponseAdapter: TypeAdapter[IPCResponse] = TypeAdapter(IPCResponse)

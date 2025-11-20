# ghostcode.ipc_message
# Message types for the inter process communication layer
from typing import *
from pydantic import BaseModel, Field, model_validator, TypeAdapter
from . import types
from .nag_sources import NagSource, NagCheckResult, NagSourceFile
from .utility import timestamp_now_iso8601, show_model_nt

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
from ..utility import quoted_if_nonempty, show_model, language_from_extension

class ProblematicSourceReport(BaseModel):
    source: NagSource
    result: NagCheckResult

    def show(self, subheading_level: int) -> str:
        """Returns a human-readable string representation of the report.
        This is intended to be used in construction of prompts for LLMs and should include all the relevant information to fix the problem, and should contain it in an economic manner.
        A subheading level can be provided. We assume that a heading is printed by the caller with level of subheading_level - 1."""
        subheading_prefix = "#" * subheading_level
        subsubheading_prefix = "#" * (subheading_level + 1)
        
        output_parts = []
        
        # 1. Basic info about the source
        output_parts.append(f"{subheading_prefix} Source: {self.source.display_name} (Type: {self.source.type})")
        
        # 2. Problem Status
        output_parts.append(f"{subsubheading_prefix} Problem Status")
        if self.result.has_problem:
            output_parts.append("Status: Problem detected.")
        else:
            # This case should ideally not be reached if the report is truly for problematic sources
            output_parts.append("Status: No problem detected (unexpected for a problematic report).")
        
        if self.result.error_while_checking:
            output_parts.append(f"Error during check: {self.result.error_while_checking}")

            # hash is not relevant to solve problem and just wastes tokens, so we skipt it here
        #if self.result.hash:
            #output_parts.append(f"Content Hash: {self.result.hash}")

        # 3. Source Configuration Details (for LLM to understand the source itself)
        output_parts.append(f"{subsubheading_prefix} Source Configuration")
        # Use show_model for a structured YAML representation of the NagSource object
        # Pass an empty string for heading to avoid duplicate headings
        output_parts.append(show_model_nt(self.source, heading="", abridge=None))
        
        # 4. The actual problematic content
        if self.result.source_content:
            # Determine language for syntax highlighting if applicable
            language = "text"
            if isinstance(self.source, NagSourceFile):
                language = language_from_extension(self.source.filepath)
            # For other NagSource types, content is typically raw output, so 'text' is appropriate.
            
            output_parts.append(f"{subsubheading_prefix} Problematic Content")
            output_parts.append(quoted_if_nonempty(text=self.result.source_content, type_tag=language, heading_level=subheading_level + 2))
        
        return "\n\n".join(output_parts)
        
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

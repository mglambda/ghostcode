# ghostcode.nag_sources
from typing import *
from .types import *
import os
import json
from pydantic import BaseModel, Field

class NagSourceFile(NagSourceBase):
    """Represents a file that should be periodically read to be monitored and potentially nagged about."""
    
    type: Literal["NagSourceFile"] = "NagSourceFile"

    nag_interval_seconds: int = Field(
        default = 60,
        description = "Number of seconds between file reads. 1 minute is the default because we don't expect files to change super frequently."
    )

    

from typing import *
from pydantic import BaseModel, Field


class ContextFile(BaseModel):
    """Abstract representation of a filepath along with metadata. Context files are sent to the cloud LLM for prompts."""

    filepath: str = Field(
        description= "The filepath to the context file. This is always relative to the directory that the .ghostcode directory is in."
        )

    rag: bool = Field(
        default = False,
        description = "Wether to enable retrieval augmented generation for this file. Enabling rag means the local LLM will retrieve only parts of the file if it deems it necessary with the given prompt. This is usually done for large text files or documentation, and helps to avoid huge token counts for the cloud LLM."
        )
    
class ContextFiles(BaseModel):
    """Encapsulates the files (both code and otherwise) that are tracked by the project. Tracked files are sent to the cloud LLM with the prompt.
    To keep it simple and human-readable, filepaths are stored in .ghostcode/context_files, one per line, all relative to the directory that the .ghostcode directory is in. This file may change frequently.
    A seperate .ghostcode/context_files_options stores metadata about the individual files. This file is mostly persistent."""

    data: List[ContextFile]
    
class ProjectMetadata(BaseModel):
    name: str
    description: str
    
class Project(BaseModel):
    """Basic context for a ghostcode project.
    By convention, all Fields of this type are found in the .ghostcode directory at the project root, with their Field names also being the filename."""

    worker_llm_config: Dict[str, Any] = Field(
    default_factory = dict,
    description = "Configuration options sent to ghostbox for the worker LLM. This is a json file stored in .ghostcode/worker_llm_config."
    )

    coder_llm_config: Dict[str, Any] = Field(
    default_factory = dict,
    description = "Configuration options sent to ghostbox for the coder LLM. This is a json file stored in .ghostcode/worker_llm_config."
    )
   
    context_files: ContextFiles
    
    directory_file: str = Field(
    description = "A directory file is an automatically generated markdown file intended to keep LLMs from introducing changes that break the project. It usually contains a project overview and important information about the project, such as module dependencies, architectural limitations, technology choices, and much more."
    )
    project_metadata: Optional[ProjectMetadata] = None

    

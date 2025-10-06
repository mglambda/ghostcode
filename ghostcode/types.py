# ghostcode/types.py
from typing import *
from pydantic import BaseModel, Field
import os
import json
import yaml
import logging

# --- Logging Setup ---
# Configure a basic logger for the ghostcode project
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ghostcode.types')

# --- Default Configurations ---
# Default configuration for the coder LLM (e.g., for planning, complex reasoning)
DEFAULT_CODER_LLM_CONFIG = {
    "backend": "google",
    "model": "models/gemini-1.5-flash", # A good default for coding tasks
    "temperature": 0.2,
    "max_length": 4096,
    "top_p": 0.9,
    "stop": ["```", "Action:", "Observation:"], # Common stop tokens for coding/tool use
    "use_tools": True, # Coder LLM will likely use tools
    "log_time": True,
    "verbose": False,
    "chat_ai": "GhostCoder",
}

# Default configuration for the worker LLM (e.g., for generating code snippets, answering questions)
DEFAULT_WORKER_LLM_CONFIG = {
    "backend": "generic", # Compatible with many local/remote OpenAI-like endpoints
    "endpoint": "http://localhost:8080", # Common default for local LLMs (e.g., llama.cpp)
    "model": "llama3", # Placeholder, user should configure their local model
    "temperature": 0.7,
    "max_length": 1024,
    "top_p": 0.95,
    "log_time": True,
    "verbose": False,
    "chat_ai": "GhostWorker",
}

# Default project metadata
DEFAULT_PROJECT_METADATA = {
    "name": "New Ghostcode Project",
    "description": "A new project initialized with ghostcode. Edit this description to provide an overview of your project.",
}

# --- Type Definitions ---
class ContextFile(BaseModel):
    """Abstract representation of a filepath along with metadata. Context files are sent to the cloud LLM for prompts."""

    filepath: str = Field(
        description= "The filepath to the context file. This is always relative to the directory that the .ghostcode directory is in."
        )

    rag: bool = Field(
        default = False,
        description = "Whether to enable retrieval augmented generation for this file. Enabling RAG means the local LLM will retrieve only parts of the file if it deems it necessary with the given prompt. This is usually done for large text files or documentation, and helps to avoid huge token counts for the cloud LLM."
        )
    
class ContextFiles(BaseModel):
    """Encapsulates the files (both code and otherwise) that are tracked by the project. Tracked files are sent to the cloud LLM with the prompt.
    To keep it simple and human-readable, filepaths are stored in .ghostcode/context_files, one per line, all relative to the directory that the .ghostcode directory is in. This file may change frequently.
    A separate .ghostcode/context_files_options stores metadata about the individual files."""

    data: List[ContextFile] = Field(default_factory=list)

    def to_plaintext(self) -> str:
        """Serializes the list of context file paths to a plaintext string, one path per line."""
        return "\n".join([cf.filepath for cf in self.data])

    @staticmethod
    def from_plaintext(content: str) -> 'ContextFiles':
        """Deserializes a plaintext string into a ContextFiles object.
        Each line is treated as a filepath."""
        filepaths = [line.strip() for line in content.splitlines() if line.strip()]
        # For now, RAG is always False when loaded from plaintext, as there's no metadata in this format.
        # A future enhancement might involve a separate file for options or a more complex format.
        context_files = [ContextFile(filepath=fp, rag=False) for fp in filepaths]
        return ContextFiles(data=context_files)
    
class ProjectMetadata(BaseModel):
    name: str
    description: str
    
class Project(BaseModel):
    """Basic context for a ghostcode project.
    By convention, all Fields of this type are found in the .ghostcode directory at the project root, with their Field names also being the filename.
    """

    worker_llm_config: Dict[str, Any] = Field(
        default_factory = dict,
        description = "Configuration options sent to ghostbox for the worker LLM. This is a json file stored in .ghostcode/worker_llm_config.json."
    )

    coder_llm_config: Dict[str, Any] = Field(
        default_factory = dict,
        description = "Configuration options sent to ghostbox for the coder LLM. This is a json file stored in .ghostcode/coder_llm_config.json."
    )
   
    context_files: ContextFiles = Field(default_factory=ContextFiles)
    
    directory_file: str = Field(
        default = "",
        description = "A directory file is an automatically generated markdown file intended to keep LLMs from introducing changes that break the project. It usually contains a project overview and important information about the project, such as module dependencies, architectural limitations, technology choices, and much more. Stored in .ghostcode/directory_file.md."
    )
    project_metadata: Optional[ProjectMetadata] = Field(default_factory=lambda: ProjectMetadata(**DEFAULT_PROJECT_METADATA))

    # --- File names within the .ghostcode directory ---
    _GHOSTCODE_DIR: ClassVar[str] = ".ghostcode"
    _WORKER_CONFIG_FILE: ClassVar[str] = "worker_llm_config.json"
    _CODER_CONFIG_FILE: ClassVar[str] = "coder_llm_config.json"
    _CONTEXT_FILES_FILE: ClassVar[str] = "context_files" # Plaintext list of filepaths
    _DIRECTORY_FILE: ClassVar[str] = "directory_file.md" # Plaintext markdown
    _PROJECT_METADATA_FILE: ClassVar[str] = "project_metadata.yaml" # YAML format

    @staticmethod
    def _get_ghostcode_path(root: str) -> str:
        """Helper to get the full path to the .ghostcode directory."""
        return os.path.join(root, Project._GHOSTCODE_DIR)

    @staticmethod
    def find_project_root(start_path: str = '.') -> Optional[str]:
        """
        Traverses up the directory tree from start_path to find the .ghostcode directory.

        Args:
            start_path (str): The directory to start searching from.

        Returns:
            Optional[str]: The absolute path to the project root (parent of .ghostcode), or None if not found.
        """
        current_path = os.path.abspath(start_path)
        while True:
            ghostcode_path = os.path.join(current_path, Project._GHOSTCODE_DIR)
            if os.path.isdir(ghostcode_path):
                logger.debug(f"Found .ghostcode directory at {ghostcode_path}. Project root is {current_path}")
                return current_path
            
            parent_path = os.path.dirname(current_path)
            if parent_path == current_path: # Reached filesystem root
                logger.debug(f"No .ghostcode directory found up to filesystem root from {start_path}")
                return None
            current_path = parent_path

    @staticmethod
    def init(root: str) -> None:
        """
        Sets up a .ghostcode directory in the given root directory with all the necessary default files.

        Args:
            root (str): The root directory of the project where .ghostcode should be created.
        """
        ghostcode_dir = Project._get_ghostcode_path(root)
        logger.info(f"Initializing .ghostcode directory at: {ghostcode_dir}")

        try:
            os.makedirs(ghostcode_dir, exist_ok=True)
            logger.info(f"Ensured directory {ghostcode_dir} exists.")

            # 1. Create worker_llm_config.json
            worker_config_path = os.path.join(ghostcode_dir, Project._WORKER_CONFIG_FILE)
            with open(worker_config_path, 'w') as f:
                json.dump(DEFAULT_WORKER_LLM_CONFIG, f, indent=4)
            logger.info(f"Created default worker LLM config at {worker_config_path}")

            # 2. Create coder_llm_config.json
            coder_config_path = os.path.join(ghostcode_dir, Project._CODER_CONFIG_FILE)
            with open(coder_config_path, 'w') as f:
                json.dump(DEFAULT_CODER_LLM_CONFIG, f, indent=4)
            logger.info(f"Created default coder LLM config at {coder_config_path}")

            # 3. Create context_files (plaintext, initially empty)
            context_files_path = os.path.join(ghostcode_dir, Project._CONTEXT_FILES_FILE)
            with open(context_files_path, 'w') as f:
                f.write("") # Empty file
            logger.info(f"Created empty context files list at {context_files_path}")

            # 4. Create directory_file.md (plaintext, initially empty)
            directory_file_path = os.path.join(ghostcode_dir, Project._DIRECTORY_FILE)
            with open(directory_file_path, 'w') as f:
                f.write("") # Empty file
            logger.info(f"Created empty directory file at {directory_file_path}")

            # 5. Create project_metadata.yaml (YAML, with default metadata)
            project_metadata_path = os.path.join(ghostcode_dir, Project._PROJECT_METADATA_FILE)
            with open(project_metadata_path, 'w') as f:
                yaml.dump(DEFAULT_PROJECT_METADATA, f, indent=2, sort_keys=False)
            logger.info(f"Created default project metadata at {project_metadata_path}")

            logger.info(f"Ghostcode project initialized successfully in {root}.")

        except OSError as e:
            logger.error(f"Failed to create .ghostcode directory or files in {root}: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during initialization in {root}: {e}", exc_info=True)
            raise

    @staticmethod
    def from_root(root: str) -> 'Project':
        """
        Looks for a .ghostcode folder in the given root directory, loads all the necessary files,
        constructs the respective types, and then returns a new Project instance.

        Args:
            root (str): The root directory of the project.

        Returns:
            Project: A new Project instance populated with data from the .ghostcode directory.

        Raises:
            FileNotFoundError: If the .ghostcode directory does not exist.
        """
        ghostcode_dir = Project._get_ghostcode_path(root)
        logger.info(f"Loading ghostcode project from: {ghostcode_dir}")

        if not os.path.isdir(ghostcode_dir):
            logger.error(f"'.ghostcode' directory not found at {ghostcode_dir}. Please initialize the project first using `ghostcode init '{root}'`.")
            raise FileNotFoundError(f"'.ghostcode' directory not found at {ghostcode_dir}")

        worker_llm_config = {}
        coder_llm_config = {}
        context_files = ContextFiles()
        directory_file_content = ""
        project_metadata = ProjectMetadata(**DEFAULT_PROJECT_METADATA)

        # 1. Load worker_llm_config.json
        worker_config_path = os.path.join(ghostcode_dir, Project._WORKER_CONFIG_FILE)
        try:
            with open(worker_config_path, 'r') as f:
                worker_llm_config = json.load(f)
            logger.debug(f"Loaded worker LLM config from {worker_config_path}")
        except FileNotFoundError:
            logger.warning(f"Worker LLM config file not found at {worker_config_path}. Using default configuration.")
            worker_llm_config = DEFAULT_WORKER_LLM_CONFIG
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {worker_config_path}: {e}. Using default configuration.", exc_info=True)
            worker_llm_config = DEFAULT_WORKER_LLM_CONFIG
        except Exception as e:
            logger.error(f"An unexpected error occurred loading {worker_config_path}: {e}. Using default configuration.", exc_info=True)
            worker_llm_config = DEFAULT_WORKER_LLM_CONFIG

        # 2. Load coder_llm_config.json
        coder_config_path = os.path.join(ghostcode_dir, Project._CODER_CONFIG_FILE)
        try:
            with open(coder_config_path, 'r') as f:
                coder_llm_config = json.load(f)
            logger.debug(f"Loaded coder LLM config from {coder_config_path}")
        except FileNotFoundError:
            logger.warning(f"Coder LLM config file not found at {coder_config_path}. Using default configuration.")
            coder_llm_config = DEFAULT_CODER_LLM_CONFIG
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {coder_config_path}: {e}. Using default configuration.", exc_info=True)
            coder_llm_config = DEFAULT_CODER_LLM_CONFIG
        except Exception as e:
            logger.error(f"An unexpected error occurred loading {coder_config_path}: {e}. Using default configuration.", exc_info=True)
            coder_llm_config = DEFAULT_CODER_LLM_CONFIG

        # 3. Load context_files
        context_files_path = os.path.join(ghostcode_dir, Project._CONTEXT_FILES_FILE)
        try:
            with open(context_files_path, 'r') as f:
                content = f.read()
                context_files = ContextFiles.from_plaintext(content)
            logger.debug(f"Loaded context files from {context_files_path}")
        except FileNotFoundError:
            logger.warning(f"Context files list not found at {context_files_path}. Using empty list.")
            context_files = ContextFiles(data=[])
        except Exception as e:
            logger.error(f"An unexpected error occurred loading {context_files_path}: {e}. Using empty list.", exc_info=True)
            context_files = ContextFiles(data=[])

        # 4. Load directory_file.md
        directory_file_path = os.path.join(ghostcode_dir, Project._DIRECTORY_FILE)
        try:
            with open(directory_file_path, 'r') as f:
                directory_file_content = f.read()
            logger.debug(f"Loaded directory file from {directory_file_path}")
        except FileNotFoundError:
            logger.warning(f"Directory file not found at {directory_file_path}. Using empty content.")
            directory_file_content = ""
        except Exception as e:
            logger.error(f"An unexpected error occurred loading {directory_file_path}: {e}. Using empty content.", exc_info=True)
            directory_file_content = ""

        # 5. Load project_metadata.yaml
        project_metadata_path = os.path.join(ghostcode_dir, Project._PROJECT_METADATA_FILE)
        try:
            with open(project_metadata_path, 'r') as f:
                metadata_dict = yaml.safe_load(f)
                if metadata_dict:
                    project_metadata = ProjectMetadata(**metadata_dict)
                else:
                    logger.warning(f"Project metadata file {project_metadata_path} is empty. Using default metadata.")
                    project_metadata = ProjectMetadata(**DEFAULT_PROJECT_METADATA)
            logger.debug(f"Loaded project metadata from {project_metadata_path}")
        except FileNotFoundError:
            logger.warning(f"Project metadata file not found at {project_metadata_path}. Using default metadata.")
            project_metadata = ProjectMetadata(**DEFAULT_PROJECT_METADATA)
        except yaml.YAMLError as e:
            logger.error(f"Error decoding YAML from {project_metadata_path}: {e}. Using default metadata.", exc_info=True)
            project_metadata = ProjectMetadata(**DEFAULT_PROJECT_METADATA)
        except Exception as e:
            logger.error(f"An unexpected error occurred loading {project_metadata_path}: {e}. Using default metadata.", exc_info=True)
            project_metadata = ProjectMetadata(**DEFAULT_PROJECT_METADATA)

        logger.info(f"Ghostcode project loaded successfully from {root}.")
        return Project(
            worker_llm_config=worker_llm_config,
            coder_llm_config=coder_llm_config,
            context_files=context_files,
            directory_file=directory_file_content,
            project_metadata=project_metadata,
        )

    def save_to_root(self, root: str) -> None:
        """
        Serializes all the contained types and saves them to the .ghostcode folder
        within the given root directory.

        Args:
            root (str): The root directory of the project.
        """
        ghostcode_dir = Project._get_ghostcode_path(root)
        logger.info(f"Saving ghostcode project to: {ghostcode_dir}")

        if not os.path.isdir(ghostcode_dir):
            logger.warning(f"'.ghostcode' directory not found at {ghostcode_dir}. Attempting to create it.")
            try:
                os.makedirs(ghostcode_dir)
            except OSError as e:
                logger.error(f"Failed to create .ghostcode directory at {ghostcode_dir}: {e}")
                raise

        # 1. Save worker_llm_config.json
        worker_config_path = os.path.join(ghostcode_dir, Project._WORKER_CONFIG_FILE)
        try:
            with open(worker_config_path, 'w') as f:
                json.dump(self.worker_llm_config, f, indent=4)
            logger.debug(f"Saved worker LLM config to {worker_config_path}")
        except Exception as e:
            logger.error(f"Failed to save worker LLM config to {worker_config_path}: {e}", exc_info=True)
            raise

        # 2. Save coder_llm_config.json
        coder_config_path = os.path.join(ghostcode_dir, Project._CODER_CONFIG_FILE)
        try:
            with open(coder_config_path, 'w') as f:
                json.dump(self.coder_llm_config, f, indent=4)
            logger.debug(f"Saved coder LLM config to {coder_config_path}")
        except Exception as e:
            logger.error(f"Failed to save coder LLM config to {coder_config_path}: {e}", exc_info=True)
            raise

        # 3. Save context_files
        context_files_path = os.path.join(ghostcode_dir, Project._CONTEXT_FILES_FILE)
        try:
            with open(context_files_path, 'w') as f:
                f.write(self.context_files.to_plaintext())
            logger.debug(f"Saved context files to {context_files_path}")
        except Exception as e:
            logger.error(f"Failed to save context files to {context_files_path}: {e}", exc_info=True)
            raise

        # 4. Save directory_file.md
        directory_file_path = os.path.join(ghostcode_dir, Project._DIRECTORY_FILE)
        try:
            with open(directory_file_path, 'w') as f:
                f.write(self.directory_file)
            logger.debug(f"Saved directory file to {directory_file_path}")
        except Exception as e:
            logger.error(f"Failed to save directory file to {directory_file_path}: {e}", exc_info=True)
            raise

        # 5. Save project_metadata.yaml
        project_metadata_path = os.path.join(ghostcode_dir, Project._PROJECT_METADATA_FILE)
        if self.project_metadata:
            try:
                with open(project_metadata_path, 'w') as f:
                    yaml.dump(self.project_metadata.model_dump(), f, indent=2, sort_keys=False)
                logger.debug(f"Saved project metadata to {project_metadata_path}")
            except Exception as e:
                logger.error(f"Failed to save project metadata to {project_metadata_path}: {e}", exc_info=True)
                raise
        else:
            logger.warning(f"No project metadata to save for {root}.")

        logger.info(f"Ghostcode project saved successfully to {root}.")

# ghostcode/types.py
from typing import *
from pydantic import BaseModel, Field
import os
import json
import yaml
import logging
import ghostbox.definitions
from ghostcode.utility import language_from_extension
import appdirs # Added for platform-specific config directory

# --- Logging Setup ---
# Configure a basic logger for the ghostcode project
logger = logging.getLogger('ghostcode.types')

# --- Default Configurations ---
# Default configuration for the coder LLM (e.g., for planning, complex reasoning)
DEFAULT_CODER_LLM_CONFIG = {
    "model": "models/gemini-2.5-flash",
    "temperature": 0.2,
    "max_length": -1,
    "top_p": 0.9,
    "log_time": True,
    "verbose": False,
    "chat_ai": "GhostCoder",
    "stdout": False,
    "stderr": False,
    "quiet": True,
    "stream": False,
}

# Default configuration for the worker LLM (e.g., for generating code snippets, answering questions)
DEFAULT_WORKER_LLM_CONFIG = {
    "model": "llama3", # Placeholder, user should configure their local model
    "temperature": 0.7,
    "max_length": -1,
    "top_p": 0.95,
    "log_time": True,
    "verbose": False,
    "stdout": False,
    "stderr": False,
    "quiet": True,
    "stream": False,
    "chat_ai": "GhostWorker",
}

# Default project metadata
DEFAULT_PROJECT_METADATA = {
    "name": "New Ghostcode Project",
    "description": "A new project initialized with ghostcode. Edit this description to provide an overview of your project.",
}

class UserConfig(BaseModel):
    """Stores user specific data, like names, emails, and api keys.
    Usually stored in a .ghostcodeconfig file, in a platform specific location. The user settings are used across multiple ghostcode projects, and so aren't stored in the .ghostcode project folder.
    """
    name: str = ""
    email: str = ""
    api_key: str = Field(
        default = "",
        description = "The most general API key. This is used for a LLM backend service that requires an API key if none of the specific API keys are set."
    )

    openai_api_key: str = Field(
        default="",
        description="Used with Chat-GPT and the OpenAI LLM backend. Get your API key at https://openai.com\nIf this key is set and you use openai as your backend, it will have precedence over the generic api_key."
    )
    
    google_api_key: str = Field(
        default="",
        description="Use with Gemini and the Google AI Studio API. Get your API key at https://aistudio.google.com\nIf this key is set and you use google as your backend, it will have precedence over the generic api_key."
    )

    _GHOSTCODE_CONFIG_FILE: ClassVar[str] = ".ghostcodeconfig"

    def save(self, user_config_path: Optional[str] = None) -> None:
        """
        Saves the UserConfig instance to a YAML file.

        Args:
            user_config_path (Optional[str]): The full path to save the config file.
                                              If None, defaults to a platform-specific user config directory.
        """
        if user_config_path is None:
            config_dir = appdirs.user_config_dir('ghostcode')
            os.makedirs(config_dir, exist_ok=True)
            save_path = os.path.join(config_dir, self._GHOSTCODE_CONFIG_FILE)
        else:
            save_path = os.path.abspath(user_config_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        logger.info(f"Saving user configuration to {save_path}")

        # Add helpful comments to the YAML output
        config_data = self.model_dump()
        yaml_content = f"""# Ghostcode User Configuration
# This file stores your personal settings, including API keys,
# which are used across all your ghostcode projects.
#
# API Keys:
#   - 'api_key': General API key for LLM services if no specific key is provided.
#   - 'openai_api_key': Specific API key for OpenAI (e.g., GPT-4o).
#     Get it from https://platform.openai.com/account/api-keys
#   - 'google_api_key': Specific API key for Google AI Studio (e.g., Gemini).
#     Get it from https://aistudio.google.com/app/apikey
#
# To fill in an API key, replace the empty string "" with your actual key.
# Example:
# openai_api_key: "sk-YOUR_OPENAI_API_KEY_HERE"
#
"""
        yaml_content += yaml.dump(config_data, indent=2, sort_keys=False)

        try:
            with open(save_path, 'w') as f:
                f.write(yaml_content)
            logger.debug(f"User configuration saved successfully to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save user configuration to {save_path}: {e}", exc_info=True)
            raise

    @staticmethod
    def load(user_config_path: Optional[str] = None) -> 'UserConfig':
        """
        Loads a UserConfig instance from a YAML file.

        Args:
            user_config_path (Optional[str]): The full path to load the config file from.
                                              If None, defaults to a platform-specific user config directory.

        Returns:
            UserConfig: The loaded UserConfig instance.

        Raises:
            FileNotFoundError: If the config file does not exist at the specified or default path.
            yaml.YAMLError: If there's an error parsing the YAML content.
        """
        if user_config_path is None:
            config_dir = appdirs.user_config_dir('ghostcode')
            load_path = os.path.join(config_dir, UserConfig._GHOSTCODE_CONFIG_FILE)
        else:
            load_path = os.path.abspath(user_config_path)

        logger.info(f"Attempting to load user configuration from {load_path}")

        if not os.path.isfile(load_path):
            logger.debug(f"User configuration file not found at {load_path}")
            raise FileNotFoundError(f"User configuration file not found at {load_path}")

        try:
            with open(load_path, 'r') as f:
                config_data = yaml.safe_load(f)
            if config_data is None:
                logger.warning(f"User configuration file {load_path} is empty. Returning default UserConfig.")
                return UserConfig()
            
            user_config = UserConfig(**config_data)
            logger.debug(f"User configuration loaded successfully from {load_path}")
            return user_config
        except yaml.YAMLError as e:
            logger.error(f"Error decoding YAML from {load_path}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred loading {load_path}: {e}", exc_info=True)
            raise

class ProjectConfig(BaseModel):
    """Contains project wide configuration obptions.
    This is stored in project_root/config.yaml"""

    # Changed these to str for easier human readability/editing in YAML
    coder_backend: str = Field(default=ghostbox.definitions.LLMBackend.google.name, description="Backend for the Coder LLM (e.g., 'google', 'openai', 'generic').")
    worker_backend: str = Field(default=ghostbox.definitions.LLMBackend.generic.name, description="Backend for the Worker LLM (e.g., 'llamacpp', 'generic').")
    coder_endpoint: str = Field(default="https://generativelanguage.googleapis.com/v1beta", description="Endpoint for the Coder LLM.")
    worker_endpoint: str = Field(default="http://localhost:8080", description="Endpoint for the Worker LLM.")


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

    def show(self) -> str:
        """Renders file contents of the context files to a string, in a format that is suitable for an LLM."""
        w = ""
        for context_file in self.data:
            try:
                with open(context_file.filepath, "r") as f:
                    contents = f.read()
            except Exception as e:
                logging.error(f"Could not read file {context_file.filepath}. Skipping. Reason: {e}")
                continue

            w += f"""# {context_file.filepath}

```{language_from_extension(context_file.filepath)}
{contents}
```

"""            
        return w
    
class ProjectMetadata(BaseModel):
    name: str
    description: str

class Project(BaseModel):
    """Basic context for a ghostcode project.
    By convention, all Fields of this type are found in the .ghostcode directory at the project root, with their Field names also being the filename.
    """

    config: ProjectConfig = Field(default_factory=ProjectConfig, description="Project wide configuration options.")

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
    _WORKER_CHARACTER_FOLDER: ClassVar[str] = "worker"
    _WORKER_CONFIG_FILE: ClassVar[str] = "worker/config.json"
    _CODER_CHARACTER_FOLDER: ClassVar[str] = "coder"
    _CODER_CONFIG_FILE: ClassVar[str] = "coder/config.json"
    _CONTEXT_FILES_FILE: ClassVar[str] = "context_files" # Plaintext list of filepaths
    _DIRECTORY_FILE: ClassVar[str] = "directory_file.md" # Plaintext markdown
    _PROJECT_METADATA_FILE: ClassVar[str] = "project_metadata.yaml" # YAML format
    _PROJECT_CONFIG_FILE: ClassVar[str] = "config.yaml" # YAML format

    _WORKER_SYSTEM_MSG: ClassVar[str] = "You are GhostWorker, a helpful AI assistant focused on executing specific, local programming tasks. Your primary role is to interact with the file system, run shell commands, and perform other environment-specific operations as instructed by GhostCoder. Be concise, precise, and report results clearly. Do not engage in high-level planning or code generation unless explicitly asked to generate a small snippet for a tool.{{worker_injection}}"
    _CODER_SYSTEM_MSG: ClassVar[str] = "You are GhostCoder, a highly intelligent and experienced AI programmer. Your role is to understand complex programming tasks, devise high-level plans, generate code, and review existing code. You will delegate specific environment interactions (like reading/writing files or running commands) to GhostWorker. Focus on architectural decisions, code quality, and problem-solving. When delegating, provide clear and unambiguous instructions to GhostWorker.\n\n# Project Overview\n{{project_metadata}}\n\n# Files\n{{context_files}}"


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

            # Create .ghostcode/worker and .ghostcode/coder directories
            worker_char_dir = os.path.join(ghostcode_dir, "worker")
            coder_char_dir = os.path.join(ghostcode_dir, "coder")
            os.makedirs(worker_char_dir, exist_ok=True)
            os.makedirs(coder_char_dir, exist_ok=True)
            logger.info(f"Created worker character directory at {worker_char_dir}")
            logger.info(f"Created coder character directory at {coder_char_dir}")

            # Create system_msg files within character directories
            worker_system_msg_path = os.path.join(worker_char_dir, "system_msg")
            with open(worker_system_msg_path, 'w') as f:
                f.write(Project._WORKER_SYSTEM_MSG)
            logger.info(f"Created worker system_msg at {worker_system_msg_path}")

            coder_system_msg_path = os.path.join(coder_char_dir, "system_msg")
            with open(coder_system_msg_path, 'w') as f:
                f.write(Project._CODER_SYSTEM_MSG)
            logger.info(f"Created coder system_msg at {coder_system_msg_path}")

            # 1. Create worker/config.json
            worker_config_path = os.path.join(ghostcode_dir, Project._WORKER_CONFIG_FILE)
            with open(worker_config_path, 'w') as f:
                json.dump(DEFAULT_WORKER_LLM_CONFIG, f, indent=4)
            logger.info(f"Created default worker LLM config at {worker_config_path}")

            # 2. Create coder/config.json
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

            # 6. Create config.yaml (YAML, with default ProjectConfig)
            project_config_path = os.path.join(ghostcode_dir, Project._PROJECT_CONFIG_FILE)
            default_project_config = ProjectConfig()
            with open(project_config_path, 'w') as f:
                yaml.dump(default_project_config.model_dump(), f, indent=2, sort_keys=False)
            logger.info(f"Created default project config at {project_config_path}")


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
        project_config = ProjectConfig() # Initialize with defaults

        # 1. Load worker_llm_config.json (now from worker/config.json)
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

        # 2. Load coder_llm_config.json (now from coder/config.json)
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

        # 6. Load config.yaml
        project_config_path = os.path.join(ghostcode_dir, Project._PROJECT_CONFIG_FILE)
        try:
            with open(project_config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                if config_dict:
                    # Pydantic will handle validation and conversion for ProjectConfig
                    project_config = ProjectConfig(**config_dict)
                else:
                    logger.warning(f"Project config file {project_config_path} is empty. Using default ProjectConfig.")
                    project_config = ProjectConfig()
            logger.debug(f"Loaded project config from {project_config_path}")
        except FileNotFoundError:
            logger.warning(f"Project config file not found at {project_config_path}. Using default ProjectConfig.")
            project_config = ProjectConfig()
        except yaml.YAMLError as e:
            logger.error(f"Error decoding YAML from {project_config_path}: {e}. Using default ProjectConfig.", exc_info=True)
            project_config = ProjectConfig()
        except Exception as e:
            logger.error(f"An unexpected error occurred loading {project_config_path}: {e}. Using default ProjectConfig.", exc_info=True)
            project_config = ProjectConfig()


        logger.info(f"Ghostcode project loaded successfully from {root}.")
        return Project(
            config=project_config, # Pass the loaded config
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

        # Ensure worker and coder character directories exist before saving their configs
        worker_char_dir = os.path.join(ghostcode_dir, "worker")
        coder_char_dir = os.path.join(ghostcode_dir, "coder")
        os.makedirs(worker_char_dir, exist_ok=True)
        os.makedirs(coder_char_dir, exist_ok=True)

        # 1. Save worker_llm_config.json (now to worker/config.json)
        worker_config_path = os.path.join(ghostcode_dir, Project._WORKER_CONFIG_FILE)
        try:
            with open(worker_config_path, 'w') as f:
                json.dump(self.worker_llm_config, f, indent=4)
            logger.debug(f"Saved worker LLM config to {worker_config_path}")
        except Exception as e:
            logger.error(f"Failed to save worker LLM config to {worker_config_path}: {e}", exc_info=True)
            raise

        # 2. Save coder_llm_config.json (now to coder/config.json)
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

        # 6. Save config.yaml
        project_config_path = os.path.join(ghostcode_dir, Project._PROJECT_CONFIG_FILE)
        try:
            with open(project_config_path, 'w') as f:
                yaml.dump(self.config.model_dump(), f, indent=2, sort_keys=False)
            logger.debug(f"Saved project config to {project_config_path}")
        except Exception as e:
            logger.error(f"Failed to save project config to {project_config_path}: {e}", exc_info=True)
            raise

        logger.info(f"Ghostcode project saved successfully to {root}.")

class CodeResponsePart(BaseModel):
    """Part of a LLM backend response that is programming code."""
    type: Literal["full", "partial"] = Field(
        default = "full",
        description = "Indicates wether this code response part is a full or partial code block. Full code blocks are used to replace entire files, and don't require an original block to be provided. A partial response replaces only part of a file, and the original, to-be-replaced code block along with preceding and following blocks should be provided alongside the replacement."
    )
    
    filepath: Optional[str] = Field(
        default = None,
        description = "The file that this code generation belongs to, the file that the code  should be saved in, or None if it is a free-standing code snippet."
    )
    
    language: str = Field(
        default = "",
        description = "The language the code is written in."
    )

    original_code: Optional[str] = Field(
        default = None,
        description = "The original code that is to be replaced with this code response. It includes a couple of lines before and after the new code. Should only be provided if this is a partial response."
    )
    
    new_code: str = Field(
        default = "",
        description = "The actual code as plaintext."
    )

    title: str = Field(
        default = "",
        description = "A short but descriptive title for the code block."
    )

    def show_cli(self) -> str:
        """Return a (short) representation that is suitable for the CLI interface."""
        filepath_str = f"({self.filepath})" if self.filepath else ""
        return f" - {self.title} {filepath_str}"

    def show(self) -> str:
        """Returns a comprehensive string representation of the part."""
        filepath_str = f" ({self.filepath})" if self.filepath else ""
        if self.original_code is not None:
            original_code_str = f"""

### Original Code
            
```{self.language}
{self.original_code}
```
            
"""
        else:
            original_code_str = ""
            
        return f"""## {self.title}{filepath_str}{original_code_str}
        {"### New Code" if original_code_str else ""}
        
```{self.language}
        {self.new_code}
```
"""        
        


        
        
class TextResponsePart(BaseModel):
    """A text chunk that is part of a response by the LLM backend."""

    text: str = Field(
        default = "",
        description = "The text payload."
    )

    filepath: Optional[str] = Field(
        default = None,
        description = "An optional filepath to indicate which file the text response part refers to."
    )

    def show_cli(self) -> str:
        """A (short) CLI representation for the part."""
        filepath_str = f"## {self.filepath}\n" if self.filepath else ""
        return f"{filepath_str}{self.text}"

    def show(self) -> str:
        """Comprehensive longform string representation for the part."""
        filepath_str = f"## {self.filepath}\n\n" if self.filepath else ""        
        return f"""{filepath_str}{self.text}
"""        
CoderResponsePart = CodeResponsePart | TextResponsePart
    
class CoderResponse(BaseModel):
    """A response returned from the code generation backend.
    Coder responses are conceptual in nature and contain code and text. THey are not intended to be tool-calls etc."""

    contents: List[CoderResponsePart] = Field(
        default_factory = lambda: [],
        description = "A list of response parts returned by the backend coder LLM. This may contain code and/or text."
    )

    def show_cli(self) -> str:
        return "\n".join([part.show_cli() for part in self.contents])

    def show(self) -> str:
        return "\n".join([part.show() for part in self.contents])

# main
from typing import *
import json
import traceback
import sys
import ghostbox


from ghostbox import Ghostbox, ChatMessage
from ghostbox.definitions import BrokenBackend
from ghostbox.commands import showTime
import argparse
import logging
import os
import glob
import json
import yaml
import appdirs
import re
from . import types
from .types import CommandOutput
from .logconfig import ExceptionListHandler, _configure_logging
from .program import Program, CommandInterface
from .subcommand import InteractCommand, NagCommand, VerifyCommand, LogCommand, ContextCommand, InitCommand, ConfigCommand, DiscoverCommand

# logger will be configured after argument parsing
logger: logging.Logger  # Declare logger globally, will be assigned later


# --- Command Interface and Implementations ---



def _main() -> None:
    parser = argparse.ArgumentParser(
        prog="ghostcode",
        description="A command line tool to help programmers code using LLMs.",
        formatter_class=argparse.RawTextHelpFormatter,  # For better help formatting
    )

    # Add --user-config argument
    parser.add_argument(
        "-c",
        "--user-config",
        type=str,
        default=None,
        help=f"Path to a custom user configuration file (YAML format). "
        f"Defaults to {os.path.join(appdirs.user_config_dir('ghostcode'), types.UserConfig._GHOSTCODE_CONFIG_FILE)}.",
    )

    # -- coder-backend argument
    parser.add_argument(
        "-C",
        "--coder-backend",
        type=str,
        choices=[backend.name for _, backend in enumerate(ghostbox.LLMBackend)],
        default="",
        help="Set the choice of backend for the coder LLM. This will override both project and user configuration options for the coder backend.",
    )

    parser.add_argument(
        "-W",
        "--worker-backend",
        type=str,
        choices=[backend.name for _, backend in enumerate(ghostbox.LLMBackend)],
        default="",
        help="Set the choice of backend for the worker LLM. This will override both project and user configuration options for the coder backend.",
    )

    # Add --logging argument
    parser.add_argument(
        "--logging",
        type=str,
        choices=["off", "stderr", "file"],
        default="off",
        help="Configure secondary logging output: 'off' (no additional output), 'stderr' (output to stderr in addition to file), or 'file' (output to a secondary file specified by --secondary-log-filepath).",
    )
    parser.add_argument(
        "--secondary-log-filepath",
        type=str,
        default=None,
        help="Path for secondary file logging when --logging is set to 'file'.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging, showing verbose internal messages.",
    )

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # Init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize a new ghostcode project in the current directory or a specified path.",
    )
    init_parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Optional path to initialize the project. Defaults to current directory.",
    )
    init_parser.set_defaults(func=lambda args: InitCommand(path=args.path))

    # Config command
    config_parser = subparsers.add_parser(
        "config", help="Manage ghostcode project configuration."
    )
    config_subparsers = config_parser.add_subparsers(
        dest="subcommand", required=True, help="Config subcommands"
    )

    config_ls_parser = config_subparsers.add_parser(
        "ls", help="List all configuration options."
    )
    config_ls_parser.set_defaults(func=lambda args: ConfigCommand(subcommand="ls"))

    config_get_parser = config_subparsers.add_parser(
        "get", help="Get the value of a specific configuration option."
    )
    config_get_parser.add_argument(
        "key",
        help="The configuration key to retrieve (e.g., 'coder_llm_config.model', 'config.coder_backend').",
    )
    config_get_parser.set_defaults(
        func=lambda args: ConfigCommand(subcommand="get", key=args.key)
    )

    config_set_parser = config_subparsers.add_parser(
        "set", help="Set the value of a configuration option."
    )
    config_set_parser.add_argument(
        "key",
        help="The configuration key to set (e.g., 'coder_llm_config.model', 'config.coder_backend').",
    )
    config_set_parser.add_argument(
        "value",
        help="The new value for the configuration key. Use JSON format for complex types (e.g., 'true', '123', '\"string\"', '[1,2]', '{\"key\":\"value\"}'). For backend types, use the string name (e.g., 'google', 'generic').",
    )
    config_set_parser.set_defaults(
        func=lambda args: ConfigCommand(
            subcommand="set", key=args.key, value=args.value
        )
    )

    # Context command
    context_parser = subparsers.add_parser(
        "context", help="Manage files included in the project context."
    )
    context_subparsers = context_parser.add_subparsers(
        dest="subcommand", required=True, help="Context subcommands"
    )

    context_ls_parser = context_subparsers.add_parser(
        "ls", aliases=["list"], help="List all files in the project context."
    )
    context_ls_parser.set_defaults(func=lambda args: ContextCommand(subcommand="ls"))

    context_list_summaries_parser = context_subparsers.add_parser(
        "list-summaries", help="List summaries of all context files."
    )
    context_list_summaries_parser.set_defaults(
        func=lambda args: ContextCommand(subcommand="list-summaries")
    )

    context_invalidate_summaries_parser = context_subparsers.add_parser(
        "invalidate-summaries", help="Invalidate all context file summaries, forcing them to be rebuilt."
    )
    context_invalidate_summaries_parser.set_defaults(
        func=lambda args: ContextCommand(subcommand="invalidate-summaries")
    )
    
    context_clean_parser = context_subparsers.add_parser(
        "clean", aliases=[], help="Remove bogus or non-existing files from context."
    )
    context_clean_parser.set_defaults(
        func=lambda args: ContextCommand(subcommand="clean")
    )

    context_wipe_parser = context_subparsers.add_parser(
        "wipe", help="Remove all files from the project context."
    )
    context_wipe_parser.set_defaults(
        func=lambda args: ContextCommand(subcommand="wipe")
    )

    context_lock_parser = context_subparsers.add_parser(
        "lock", help="Lock file(s) in the project context, preventing their removal."
    )
    context_lock_parser.add_argument(
        "filepaths",
        nargs="+",
        help="One or more file paths (can include wildcards like '*.py', 'src/**.js').",
    )
    context_lock_parser.set_defaults(
        func=lambda args: ContextCommand(subcommand="lock", filepaths=args.filepaths)
    )

    context_unlock_parser = context_subparsers.add_parser(
        "unlock", help="Unlock file(s) in the project context, allowing their removal."
    )
    context_unlock_parser.add_argument(
        "filepaths",
        nargs="+",
        help="One or more file paths (can include wildcards like '*.py', 'src/**.js').",
    )
    context_unlock_parser.set_defaults(
        func=lambda args: ContextCommand(subcommand="unlock", filepaths=args.filepaths)
    )

    context_add_parser = context_subparsers.add_parser(
        "add", help="Add file(s) to the project context. Supports wildcards."
    )
    context_add_parser.add_argument(
        "filepaths",
        nargs="+",
        help="One or more file paths (can include wildcards like '*.py', 'src/**.js').",
    )
    context_add_parser.add_argument(
        "--rag",
        action="store_true",
        help="Enable Retrieval Augmented Generation for these files.",
    )
    context_add_parser.set_defaults(
        func=lambda args: ContextCommand(
            subcommand="add", filepaths=args.filepaths, rag=args.rag
        )
    )

    context_rm_parser = context_subparsers.add_parser(
        "rm",
        aliases=["remove"],
        help="Remove file(s) from the project context. Supports wildcards.",
    )
    context_rm_parser.add_argument(
        "filepaths", nargs="+", help="One or more file paths (can include wildcards)."
    )
    context_rm_parser.set_defaults(
        func=lambda args: ContextCommand(subcommand="rm", filepaths=args.filepaths)
    )

    # Dynamically add visibility subcommands based on ContextFileVisibility enum
    for visibility_option in types.ContextFileVisibility:
        visibility_name = visibility_option.value
        visibility_parser = context_subparsers.add_parser(
            visibility_name,
            help=f"Set '{visibility_name}' visibility for file(s) in the project context. Supports wildcards.",
        )
        visibility_parser.add_argument(
            "filepaths",
            nargs="+",
            help="One or more file paths (can include wildcards like '*.py', 'src/**.js').",
        )
        visibility_parser.set_defaults(
            func=lambda args, vis=visibility_name: ContextCommand(
                subcommand=vis, filepaths=args.filepaths
            )
        )

    # Discover command
    discover_parser = subparsers.add_parser(
        "discover",
        help="Intelligently discovers files associated with the project and adds them to the context.",
    )
    discover_parser.add_argument(
        "filepath",
        help="The filepath that points to a directory in which files to add to the context will be recursively discovered.",
    )

    # Dynamically add language flags
    for lang in sorted(DiscoverCommand.possible_languages):
        discover_parser.add_argument(
            f"--{lang}",
            action="append_const",  # Use append_const to collect multiple languages
            const=lang,
            dest="languages_enabled",  # All --lang flags append to this list
            help=f"Include files primarily written in {lang.capitalize()}.",
        )
        discover_parser.add_argument(
            f"--no-{lang}",
            action="append_const",
            const=lang,
            dest="languages_disabled",  # All --no-lang flags append to this list
            help=f"Exclude files primarily written in {lang.capitalize()}.",
        )

    discover_parser.add_argument(
        "--exclude-pattern",
        type=str,
        default="",
        help="A regex that can be provided. File names that match against it are excluded from the context. The exclude pattern is applied at the very end of the discovery process.",
    )

    discover_parser.add_argument(
        "--all",
        action="store_true",
        help="If provided, will add all (non-hidden) files to the context that are found in subdirectories of the given path. Providing --all is like providing all of the language parameters.",
    )

    discover_parser.add_argument(
        "--min-lines",
        type=int,
        default=None,
        help="Minimum amount of lines that a file must have in order to be added to the context.",
    )
    discover_parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Maximum number of lines a file can have in order to be added to the context.",
    )

    size_heuristic_group = discover_parser.add_mutually_exclusive_group()
    size_heuristic_group.add_argument(
        "--small",
        action="store_const",
        const="small",
        dest="size_heuristic",
        help="If provided, sets min_lines=0 and max_lines=1000. Mutually exclusive with --medium and --large.",
    )
    size_heuristic_group.add_argument(
        "--medium",
        action="store_const",
        const="medium",
        dest="size_heuristic",
        help="If provided, sets min_lines=300 and max_lines=3000. Mutually exclusive with --small and --large.",
    )
    size_heuristic_group.add_argument(
        "--large",
        action="store_const",
        const="large",
        dest="size_heuristic",
        help="If provided, sets min_lines=3000 and no max_lines. Mutually exclusive with --small and --medium.",
    )

    discover_parser.set_defaults(
        func=lambda args: DiscoverCommand(
            filepath=args.filepath,
            languages_enabled=args.languages_enabled if args.languages_enabled else [],
            languages_disabled=(
                args.languages_disabled if args.languages_disabled else []
            ),
            min_lines=args.min_lines,
            max_lines=args.max_lines,
            size_heuristic=args.size_heuristic,
            exclude_pattern=args.exclude_pattern,
            all=args.all,
        )
    )

    # Interact command
    interact_parser = subparsers.add_parser(
        "interact", help="Launches an interactive session with the Coder LLM."
    )
    interact_parser.add_argument(
        "--actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If --no-actions is provided, ghostcode won't try to do edits, file creation, git commits or anything else that changes state. The ghostcoder LLM will be forced to generate only text, which may or may not contain inline code examples.",
    )
    interact_parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=None,
        help="Initial prompt for the interactive session. If provided, it bypasses the first user input.",
    )
    interact_parser.add_argument(
        "--skip-to-coder",
        action="store_true",
        help="Skip prompt routing and directly query the Coder LLM.",
    )
    interact_parser.add_argument(
        "--skip-to-worker",
        action="store_true",
        help="Skip prompt routing and directly query the Worker LLM.",
    )
    interact_parser.add_argument(
        "-i",
        "--interaction",
        type=str,
        default=None,
        help="Optional unique ID or tag of a past interaction to continue.",
    )
    interact_parser.add_argument(
        "--force",
        action="store_true",
        help="Force release of interaction lock if one exists.",
    )
    interact_parser.add_argument(
        "-b",
        "--branch",
        type=str,
        default=None,
        help="Specify a Git branch to checkout before starting the interaction.",
    )
    interact_parser.set_defaults(
        func=lambda args: _create_interact_command(
            args, actions=args.actions, force_lock=args.force, branch=args.branch
        )
    )

    # Talk command
    talk_parser = subparsers.add_parser(
        "talk",
        help="Launches an interactive session with the Coder LLM, but without performing any actions. This is shorthand for ghostcode interact --no-actions",
    )
    talk_parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=None,
        help="Initial prompt for the interactive session. If provided, it bypasses the first user input.",
    )
    talk_parser.add_argument(
        "--skip-to-coder",
        action="store_true",
        help="Skip prompt routing and directly query the Coder LLM.",
    )
    talk_parser.add_argument(
        "--skip-to-worker",
        action="store_true",
        help="Skip prompt routing and directly query the Worker LLM.",
    )
    talk_parser.add_argument(
        "-i",
        "--interaction",
        type=str,
        default=None,
        help="Optional unique ID or tag of a past interaction to continue.",
    )
    talk_parser.add_argument(
        "--force",
        action="store_true",
        help="Force release of interaction lock if one exists.",
    )
    talk_parser.add_argument(
        "-b",
        "--branch",
        type=str,
        default=None,
        help="Specify a Git branch to checkout before starting the interaction.",
    )
    talk_parser.set_defaults(
        func=lambda args: _create_interact_command(
            args, actions=False, force_lock=args.force, branch=args.branch
        )
    )

    # Nag command
    nag_parser = subparsers.add_parser(
        "nag",
        help="Starts a read-only voice chat session that monitors certain outputs (tests, type-checkers, log files) and notifies the user about problems.",
    )
    nag_parser.add_argument(
        "-f",
        "--file",
        nargs="+",
        action="append",
        dest="files",
        default=[],
        help="Add one or more file paths to monitor. Can be specified multiple times.",
    )

    nag_parser.add_argument(
        "-u",
        "--url",
        nargs="+",
        action="append",
        dest="urls",
        default=[],
        help="Add one or more URLs to monitor. Can be specified multiple times.",
    )

    nag_parser.add_argument(
        "-c",
        "--command",
        nargs="+",
        action="append",
        dest="shell_commands",
        default=[],
        help="Add one or more shell commands to monitor. Can be specified multiple times.",
    )
    nag_parser.add_argument(
        "-b",
        "--emacs-buffer",
        nargs="+",
        action="append",
        dest="emacs_buffers",
        default=[],
        help="Add one or more Emacs buffer names to monitor. Can be specified multiple times.",
    )
    nag_parser.add_argument(
        "-p",
        "--system-prompt",
        type=str,
        default="",
        help="Additional system instructions for the LLM in the nag session.",
    )
    nag_parser.add_argument(
        "-P",
        "--personality",
        type=str,
        choices=[p.name for p in types.LLMPersonality],
        default=None,
        help=f"Choose a personality for the responses in the nag session. "
        f"Overrides user configuration. Available: {', '.join([p.name for p in types.LLMPersonality])}",
    )
    nag_parser.set_defaults(
        func=lambda args: NagCommand(
            files=(
                [item for sublist in args.files for item in sublist]
                if args.files
                else []
            ),
            urls=(
                [item for sublist in args.urls for item in sublist] if args.urls else []
            ),
            shell_commands=(
                [item for sublist in args.shell_commands for item in sublist]
                if args.shell_commands
                else []
            ),
            emacs_buffers=(
                [item for sublist in args.emacs_buffers for item in sublist]
                if args.emacs_buffers
                else []
            ),
            system_prompt=args.system_prompt,
            personality=(
                types.LLMPersonality[args.personality] if args.personality else None
            ),
        )
    )

    # Log command
    log_parser = subparsers.add_parser("log", help="Display past interaction history.")
    log_parser.add_argument(
        "--interaction",
        type=str,
        help="Display a specific interaction in detail by its unique ID or tag.",
    )
    log_parser.add_argument(
        "--all-branches",
        action="store_true",
        help="Do not filter any interactions based on branches.",
    )
    log_parser.set_defaults(
        func=lambda args: LogCommand(
            interaction_identifier=args.interaction, all_branches=args.all_branches
        )
    )

    args = parser.parse_args()
    # --- Determine project_root early for logging configuration ---

    current_project_root: Optional[str] = None
    if args.command == "init":
        # For init, the project_root is the path being initialized
        current_project_root = os.path.abspath(args.path if args.path else os.getcwd())
    else:
        # For other commands, find the existing project root
        current_project_root = types.Project.find_project_root()

    # Configure logging based on argument and determined project_root
    _configure_logging(
        args.logging,
        current_project_root,
        is_init=args.command == "init",
        secondary_log_filepath=args.secondary_log_filepath,
        is_debug=args.debug,
    )

    # Now that logging is configured, get the main logger instance
    global logger  # Declare logger as global to assign to it
    logger = logging.getLogger("ghostcode.main")

    # Load UserConfig
    user_config_path = args.user_config
    user_config: types.UserConfig
    try:
        user_config = types.UserConfig.load(user_config_path)
    except FileNotFoundError:
        logger.info(f"User configuration file not found. Creating a default one.")
        user_config = types.UserConfig()
        user_config.save(user_config_path)  # Save to the default or specified path
    except Exception as e:
        logger.error(
            f"Failed to load user configuration: {e}. Using default settings.",
            exc_info=True,
        )
        user_config = types.UserConfig()

    # Instantiate the command object
    command_obj: CommandInterface = args.func(args)

    # Special handling for 'init' command as it doesn't require an existing project
    if isinstance(command_obj, InitCommand):
        # The project_root for init is already determined as current_project_root
        # and logging is configured. Just run the command.
        out = command_obj.run(
            Program(
                project_root=current_project_root,
                project=None,
                worker_box=None,
                coder_box=None,
                user_config=user_config,
            )
        )  # Pass user_config
        print(out.text)
        sys.exit(0)  # Exit after init

    # For all other commands, a project must exist
    if not current_project_root:  # This would have been set by find_project_root
        logger.error("Not a ghostcode project. Run 'ghostcode init' first.")
        sys.exit(1)

    try:
        project = types.Project.from_root(current_project_root)
    except (
        FileNotFoundError
    ):  # Should ideally be caught by find_project_root, but good to be safe
        logger.error(
            f"Project directory '.ghostcode' not found at {current_project_root}. This should not happen after root detection."
        )
        sys.exit(1)
    except Exception as e:
        logger.error(
            f"Failed to load ghostcode project from {current_project_root}: {e}",
            exc_info=True,
        )
        sys.exit(1)

    # Initialize Ghostbox instances using project.config for endpoints and backends
    # the quiet options have to be passed here or we will get a couple of stray messages until the project box configs are read
    quiet_options = {"quiet": True, "stdout": False, "stderr": False}

    # Now directly use the string values from project.config
    # Pass API keys from user_config to Ghostbox instances
    # we catch missing API key and other backend initialization errors here, and inform the user later in commands that actually require backends
    try:
        worker_backend = (
            project.config.worker_backend
            if args.worker_backend == ""
            else args.worker_backend
        )
        worker_box = Ghostbox(
            endpoint=project.config.worker_endpoint,
            backend=worker_backend,
            character_folder=os.path.join(
                current_project_root, ".ghostcode", project._WORKER_CHARACTER_FOLDER
            ),
            api_key=user_config.api_key,  # General API key
            google_api_key=user_config.google_api_key,
            openai_api_key=user_config.openai_api_key,
            deepseek_api_key=user_config.deepseek_api_key,
            model=user_config.get_model(types.AIAgent.CODER, worker_backend),
            **quiet_options,
        )
    except Exception as e:
        logger.error(f"Setting worker to dummy. Reason: {e}")
        worker_box = ghostbox.from_dummy(**quiet_options)

    try:
        coder_backend = (
            project.config.coder_backend
            if args.coder_backend == ""
            else args.coder_backend
        )
        coder_box = Ghostbox(
            endpoint=project.config.coder_endpoint,
            backend=coder_backend,
            character_folder=os.path.join(
                current_project_root, ".ghostcode", project._CODER_CHARACTER_FOLDER
            ),
            api_key=user_config.api_key,  # General API key
            google_api_key=user_config.google_api_key,
            openai_api_key=user_config.openai_api_key,
            deepseek_api_key=user_config.deepseek_api_key,
            model=user_config.get_model(types.AIAgent.CODER, coder_backend),
            **quiet_options,
        )
    except Exception as e:
        logger.error(f"Setting coder backend to dummy. Reason: {e}")
        coder_box = ghostbox.from_dummy(**quiet_options)

    # Create the Program instance
    prog_instance = Program(
        project_root=current_project_root,
        project=project,
        worker_box=worker_box,
        coder_box=coder_box,
        user_config=user_config,  # Pass the loaded user_config
    )

    # Run the command
    out = command_obj.run(prog_instance)
    print(out.text)

    # Only save project if the command is not 'nag'
    if not isinstance(command_obj, NagCommand):
        if prog_instance.project_root is not None:
            project.save_to_root(prog_instance.project_root)


def _create_interact_command(
    args: argparse.Namespace,
    actions: bool,
    force_lock: bool = False,
    branch: Optional[str] = None,
) -> InteractCommand:
    """Helper function to create InteractCommand, handling skip_to logic and mutual exclusivity."""
    skip_to_agent: Optional[types.AIAgent] = None

    if args.skip_to_coder and args.skip_to_worker:
        msg = "Cannot use both --skip-to-coder and --skip-to-worker simultaneously."
        logger.error(msg)
        print(msg)
        sys.exit(1)
    elif args.skip_to_coder:
        skip_to_agent = types.AIAgent.CODER
    elif args.skip_to_worker:
        skip_to_agent = types.AIAgent.WORKER

    return InteractCommand(
        actions=actions,
        initial_prompt=args.prompt,
        skip_to=skip_to_agent,
        interaction_identifier=args.interaction,
        force_lock=force_lock,
        branch=branch,
    )


if __name__ == "__main__":
    _main()

# slash_commands.py
from typing import *
import shlex
from dataclasses import dataclass, field
import json
import logging
from enum import Enum
from . import types
from .program import Program

# --- Logging Setup ---
logger = logging.getLogger("ghostcode.slash_commands")


class SlashCommandResult(Enum):
    NOT_A_COMMAND = -2
    COMMAND_NOT_FOUND = -1
    OK = 0
    HALT = 1
    BAD_ARGUMENTS = 2
    ACTIONS_OFF = 3
    ACTIONS_ON = 4
    RESET_SESSION = 5
    
@dataclass
class SlashCommand:
    """Represents a command that can be invoked with a leading slash (/) during an interactive prompt.
    Slash commands are e.g. '/quit' or '/save fixbug.txt'.
    Slash commands can have 0 or more arguments.
    """

    # the name is what can be invoked at the CLI prompt with a preceding slash
    name: str

    # The types of arguments the slash commands expects
    args: List[Tuple[str, Any]]

    # docstring for the slash command
    help: str

    # the function that will be invoked in the program and interaction context. args will be the string the user put behind the slash command, parsed by shlex.tokenize
    function: Callable[
        [Program, types.InteractionHistory, Sequence[str]], SlashCommandResult
    ]

    ### implementations ###


def slash_traceback(
    prog: Program, interaction: types.InteractionHistory, args: Sequence[str]
) -> SlashCommandResult:
    # this is a bit icky but ok
    from .logconfig import EXCEPTION_HANDLER

    if (tr_str := EXCEPTION_HANDLER.try_get_last_traceback()) is not None:
        print(
            f"Displaying most recent exception below. Repeated calls of /traceback will display earlier exceptions.\n\n```\n{tr_str}```\n"
        )
    else:
        print("No recent tracebacks.")
    return SlashCommandResult.OK


def slash_save(
    prog: Program, interaction: types.InteractionHistory, args: Sequence[str]
) -> SlashCommandResult:
    # FIXME: handle args for optional filepath
    filepath = "out.txt"
    with open(filepath, "w") as f:
        f.write(interaction.show())
    logger.info(msg := f"Saved current interaction history to {filepath}.")
    prog.print(msg)
    return SlashCommandResult.OK


def slash_new(
    prog: Program, interaction: types.InteractionHistory, args: Sequence[str]
) -> SlashCommandResult:
    # This function just signals the main loop to perform the reset logic.
    return SlashCommandResult.RESET_SESSION

def slash_context(
    prog: Program, interaction: types.InteractionHistory, args: Sequence[str]
) -> SlashCommandResult:
    if prog.project is None:
        prog.print(f"Null project. Please do `ghostcode init` and restart the interaction.")
        return SlashCommandResult.HALT
        
    prog.print(prog.project.context_files.show_visible_filepaths_cli())
    return SlashCommandResult.OK

### list of slash commands ###

def slash_help(
    prog: Program, interaction: types.InteractionHistory, args: Sequence[str]
) -> SlashCommandResult:
    global slash_command_list
    for cmd in slash_command_list:
        args_str = "" if not cmd.args else f"{cmd.args}"
        prog.print(f"/{cmd.name}" + args_str)
        prog.print(f"    {cmd.help}")
    return SlashCommandResult.OK

slash_command_list = [
    SlashCommand(
        name="help",
        args=[],
        help="Show this help.",
        function=slash_help
    ),                    
    SlashCommand(
        name="quit",
        args=[],
        help="Quit the program. The interaction will be saved.",
        function=lambda prog, interaction, args: SlashCommandResult.HALT,
    ),
    SlashCommand(
        name="traceback",
        args=[],
        help="Display the most recent exception traceback, if any. This command is stateful and will consume the output (yeah). Multiple invocations will print progressively older tracebacks.",
        function=slash_traceback,
    ),
    SlashCommand(
        name="lastrequest",
        args=[],
        help="Used for debugging. Display the last request sent to the LLM backend.",
        function=lambda prog, interaction, args: (prog.print(json.dumps(prog.coder_box.get_last_request(), indent=4)), SlashCommandResult.OK)[1],  # type: ignore
    ),
    SlashCommand(
        name="show",
        args=[],
        help="Display the interaction history.",
        function=lambda prog, interaction, args: (prog.print(interaction.show()), SlashCommandResult.OK)[1],  # type: ignore
    ),
    SlashCommand(
        name="save",
        args=[("[FILENAME]", Optional[str])],
        help="Save the current interaction history to a text file.\nNote that all interaction histories are saved and accessible via the `ghostcode log` subcommand.",
        function=slash_save,
    ),
    SlashCommand(
        name="talk",
        args=[],
        help="Turn talk mode on. This causes the coder backend to no longer generate code (except as parts of conversation), and no longer do file edits. Good for brainstorming or purely informative queries.",
        function=lambda prog, interaction_history, args: SlashCommandResult.ACTIONS_OFF
    ),
    SlashCommand(
        name="interact",
        args=[],
        help="Turn interact mode on. This causes the coder backend to generate code and attempt to do file edits.",
        function=lambda prog, interaction_history , args: SlashCommandResult.ACTIONS_ON
    ),
    SlashCommand(
        name="new",
        args=[],
        help="Start a new interact session without exiting the program. Your previous interaction will be saved.",
        function=slash_new,
    ),
    SlashCommand(
        name="context",
        args=[],
        help="List all files that are currently being shown to 👻 coder LLM. The output will be different from `ghostcode context list` as the files included/excluded are specific to this interaction.",
        function=slash_context
    ),                
]


### general purpose ###


def try_command(
    prog: Program, interaction: types.InteractionHistory, input_line: str
) -> SlashCommandResult:
    """Given a program and interaction context, tries to find a command and arguments in input_line and run it.
    If the given input line does not appear to be a command, the return value is NOT_A_COMMAND.
    If the given input_line seems like it should be a command (it starts with a slash) but no command of that name is found, the return value is COMMAND_NOT_FOUND.
    Other return values depend on the individual commands."""
    try:
        args = shlex.split(input_line)
    except ValueError as e:
        # this happens on a single backslash, which we need.
        # log it and move on
        logger.debug(
            f"shlex got value error while parsing arguments to slash command. Reason: {e}"
        )
        return SlashCommandResult.NOT_A_COMMAND

    if not (args) or not (args[0].startswith("/")):
        return SlashCommandResult.NOT_A_COMMAND

    command_candidate = args[0]
    rest = args[1:]
    for slash_command in slash_command_list:
        if ("/" + slash_command.name) == command_candidate:
            return slash_command.function(prog, interaction, rest)
    return SlashCommandResult.COMMAND_NOT_FOUND

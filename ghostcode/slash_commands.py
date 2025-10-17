# slash_commands.py
from typing import *
from ghostcode import types
from dataclasses import dataclass, field
import json
import logging

# --- Logging Setup ---
logger = logging.getLogger("ghostcode.slash_commands")



@dataclass
class SlashCommand:
    """Represents a command that can be invoked with a leading slash (/) during an interactive prompt.
    Slash commands are e.g. '/quit' or '/save fixbug.txt'.
    Slash commands can have 0 or more arguments.
    """

    # the name is what can be invoked at the CLI prompt with a preceding slash
    name : str


    # The types of arguments the slash commands expects
    args: List[type] 
    
    # docstring for the slash command
    help: str

    # the function that will be invoked in the program and interaction context. args will be the string the user put behind the slash command, parsed by shlex.tokenize
    # The functions bool return value indicates halting. If it returns true, the interaction is broken off and the program will halt. False means keep looping.
    function: Callable[[types.Program, types.InteractionHistory, Sequence[Tuple[str, type]]], bool]


    ### implementations ###

def slash_traceback(prog: types.Program, interaction: types.InteractionHistory, args: Sequence[str]) -> bool:
    # this is a bit icky but ok
    from ghostcode import main
    EXCEPTION_HANDLER = main.EXCEPTION_HANDLER
    if (tr_str := EXCEPTION_HANDLER.try_get_last_traceback()) is not None:
        print(
            f"Displaying most recent exception below. Repeated calls of /traceback will display earlier exceptions.\n\n```\n{tr_str}```\n"
        )
    else:
        print("No recent tracebacks.")
    return False


def slash_save(prog: types.Program, interaction: types.InteractionHistory, args: Sequence[str]) -> bool:
    # FIXME: handle args for optional filepath
    filepath = "out.txt"
    with open(filepath, "w") as f:
        f.write(interaction.show())
    logger.info(msg := f"Saved current interaction history to {filepath}.")
    prog.print(msg)
    return False

### list of slash commands ###
    
    list = [
        SlashCommand(
            name = "quit",
            args = [],
            help = "Quit the program. The interaction will be saved.",
            function = lambda prog, interaction, args: True
        ),
        SlashCommand(
            name = "traceback",
            args = [],
            help = "Display the most recent exception traceback, if any. This command is stateful and will consume the output (yeah). Multiple invocations will print progressively older tracebacks.",
            function = slash_traceback
        ),
        SlashCommand(
            name = "lastrequest",
            args = [],
            help = "Used for debugging. Display the last request sent to the LLM backend.",
            function = lambda prog, interaction, args:                 prog.print(json.dumps(prog.coder_box.get_last_request(), indent=4))
        ),
        SlashCommand(
            name = "show",
            args = [str],
            help = "Display the interaction history.",
            function = lambda prog, interaction, args:                 prog.print(interaction.show())
        ),
        SlashCommand(
            name = "save",
            args = [("[FILENAME]", Optional[str])],
            help = "Save the current interaction history to a text file.\nNote that all interaction histories are saved and accessible via the `ghostcode log` subcommand.",
            function = slash_save
        )
    ]

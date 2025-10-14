# shell.py
import os
import shlex
import json
import threading
import feedwater
from typing import *
from ghostcode.utility import timestamp_now_iso8601, show_model
from dataclasses import dataclass, field
import logging
import time

# --- Logging Setup ---
logger = logging.getLogger('ghostcode.shell')
from pydantic import BaseModel, Field

class VirtualTerminalError(BaseModel):
    """Small wrapper that signals virtual terminal errors."""

    timestamp: str = Field(
        default_factory = timestamp_now_iso8601,
        description = "The time at which the error occurred."
    )

    message: str = Field(
        description = "The error message."
    )

class InteractionLine(BaseModel):
    """One input, output, or error line in a shell interaction."""

    text: str = Field(
        description = "The content of the line that was input or printed."
    )

    timestamp: str = Field(
        default_factory = timestamp_now_iso8601,
        description = "The (relatively accurate) time at which the line was input or read."
    )

    def show_xml(self) -> str:
        return f"""<msg timestamp="{self.timestamp}">{self.text}</msg>"""
        
class ShellInteraction(BaseModel):
    """Represents one command execution in a virtual terminal shell.
    An interaction may be completed, in which case the exit_code field will contain the executed operation's exit code, or still be in progress, in which case the exit_code field will be None.""" 
    
    original_command: str = Field(
        description = "The command use to invoke this shell interaction."
    )
    
    stdin: List[InteractionLine] = Field(
        default_factory = lambda: [],
        description = "The complete history of interaction's input."
        )
    
    stdout: List[InteractionLine] = Field(
        default_factory = lambda: [],
        description = "The complete history of what the shell execution has output to standard output so far."
    )

    stderr: List[InteractionLine] = Field(
        default_factory = lambda: [],
        description = "The complete history of what the shell execution has output to standard error so far."
    )

    exit_code: Optional[int] = Field(
        default = None,
        description = "The exit code of the interaction's underlying operation. None if it is still ongoing."
    )


    cwd: str = Field(
        default = "",
        description = "The current working directory of the command."
    )
    
    time_start: str = Field(
        default_factory = timestamp_now_iso8601,
        description = "The timestampe at which the interaction started."
    )

    time_end: Optional[str] = Field(
        default = None,
        description = "The timestampe at which the interaction ended."
    )

    def show(self) -> str:
        """Returns a basic but complete serialization of the interaction as a string.
        Although this can be used by an LLM, if you want a string representation optimized for LLM contexts, use the other showers, e.g. show_xml."""
        return show_model(self)

    def show_xml(self) -> str:
        """Returns a string representation of the interaction formatted with made-up XML.
        This tends to be easy to understand for an LLM."""
        end_str = f"timestanp-end=\"{self.time_end}\"" if self.time_end is not None else ""
        exit_code_str = f"exit-code=\"{self.exit_code}\"" if self.exit_code is not None else ""
        status_str = 'status="running"' if self.exit_code is None else 'status="finished"'
        wrap_cdata = lambda w: f"<![CDATA[\n{w}\n]]>"
        stdin_str = wrap_cdata("\n".join([msg.text for msg in self.stdin])) if self.stdin else ""
        stdout_str = wrap_cdata("\n".join([msg.text for msg in self.stdout])) if self.stdout else ""
        stderr_str = wrap_cdata("\n".join([msg.text for msg in self.stderr])) if self.stderr else ""
        
        return f"""<shell-interaction timestamp-start="{self.time_start}" {end_str} {status_str} {exit_code_str}>
<command>
{self.original_command}
</command>
<stdin>{stdin_str}</stdin>
<stdout>{stdout_str}</stdout>
<stderr>{stderr_str}</stderr>
</shell-interaction>
"""


    def _dict_for_json(self) -> Dict[str, Any]:
        data = {}
        
        # this is just to have a more inuitive ordering for the LLM, e.g. header-ish keys go to the top
        data["command"] = ""
        data["cwd"] = ""
        data["status"] = ""
        
        data |= self.model_dump()

        # slight renaming for more LLM clarity
        data["command"] = data["original_command"]
        del data["original_command"]

        def text_only(key):
            data[key] = "\n".join([msg["text"] for msg in data[key]])
            
        for k in ["stdin", "stdout", "stderr"]:
            text_only(k)
        data["status"] = "running" if self.exit_code is None else "finished"
        return data
    
    def show_json(self) -> str:
        """Returns a json representation of the shell interaction.
        This is not guaranteed to be equivalent to json.dumps(self.__dict__). Do not use this for object serialization/deserialization."""
        return json.dumps(self._dict_for_json, indent=4)

class ShellInteractionHistory(BaseModel):
    """Keeps track of shell interaction both completed ones in the past and current ongoing ones."""

    past_interactions: List[ShellInteraction] = Field(
        default_factory = lambda: [],
        description = "A chronological list of past shell interactions that have finished."
    )

    current_interaction: Optional[ShellInteraction] = Field(
        default = None,
        description = "Contains the current shell interaction if a process is ongoing, None otherwise."
    )

    def show(self) -> str:
        return show_model(self)

    def show_json(self) -> str:
        """Returns a json representation of the interaction history.
        This is **not** equivalent to calling json.dumps(self.__dict__). Do not use this for object serialization/deserialization. The output of this function is intended for LLMs and may not conform at all to the pydantic definitions of the model."""
        data = [interaction._dict_for_json() for interaction in self.past_interactions] + ([self.current_interaction._dict_for_json()] if self.current_interaction is not None else [])
        return json.dumps(data, indent=4)
        
    def show_xml(self) -> str:
        """Returns a string representation in made-up XML syntax, which is structured and can be easier for LLMs to process."""
        current_interaction_str = self.current_interaction.show_xml() if self.current_interaction is not None else ""
        return "\n".join([interaction.show_xml() for interaction in self.past_interactions]) + "\n" + current_interaction_str
        
    def save(self, exit_code: Optional[int] = None) -> None:
        """Moves the current interaction into the past interactions."""
        if self.current_interaction is None:
            return

        self.current_interaction.time_end = timestamp_now_iso8601()
        self.current_interaction.exit_code = exit_code
        self.past_interactions.append(self.current_interaction)
        self.current_interaction = None

    def new(self, original_command: str, **kwargs) -> None:
        """Construct a new current interaction in-place."""
        if self.current_interaction is not None:
            self.save()

        self.current_interaction = ShellInteraction(
            original_command=original_command,
            **kwargs
        )
        
@dataclass        
class VirtualTerminal:
    """Simulates a terminal with a command line prompt that runs asynchronously.
    Intended to be used by LLMs.
    This type never raises errors. Check VirtualTerminal.is_ready() and VirtualTerminal.error for the VirtualTerminalError type."""
    
    env: os._Environ = field(
        default_factory=lambda: os.environ,
    )

    # we model the current working directory internally. This is not perfect but for now it works.
    cwd: str = field(
        default_factory = os.getcwd
    )
    
    history: ShellInteractionHistory = field(
        default_factory = ShellInteractionHistory,
    )

    # contains a small error object if an error was encountered in the underlying process, None otherwise
    error: Optional[VirtualTerminalError] = field(
        default = None,
    )

    # amount of time in seconds that passes between polling the underlying process
    refresh_rate: float = field(
        default= 1.0,
    )

    # The process object that holds the actual shell instance. Runs asynchronously and is internally synchronized.
    _process: Optional[feedwater.Process] = field(
        init = False,
        default = None,
    )

    _poll_thread: Optional[threading.Thread] = field(
        default = None,
    )

    # semaphore indidcating wether the polling thread should keep running
    _poll_running: threading.Event = field(
        default_factory = lambda: threading.Event(),
    )

    # Indicates that the  polling thread has finished its operation.
    _poll_done: threading.Event = field(
        default_factory = lambda: threading.Event(),
    )

    def __post_init__(self) -> None:
        logger.info("Started virtual terminal.")        
        self._poll_done.set() # Initially set to allow first poll thread to start
        
    def _start_interaction(self, terminal_input: str) -> None:
        if (new_cwd := self._try_parse_cd(terminal_input)) is not None:
            logger.debug(f"Parsed cd in input `{terminal_input}`.")
            if os.path.isdir(new_cwd):
                logger.debug(f"New CWD is `{new_cwd}`.")
                self.cwd = new_cwd
            else:
                logger.debug(f"New CWD `{new_cwd}` does not appear to be a directory. No change.")
        
        self.history.new(terminal_input, cwd=self.cwd)
        self._process = feedwater.run(
            terminal_input,
            env=self.env,
            popen_kwargs = {"cwd": self.cwd}
        )
        
        if not(self._process.is_running()):
            logger.debug(f"dumping virtual terminal outputs.\nstdout:\n{self._process.get()}" # Fixed: get_stdout() -> get()
                         f"\n\nstderr:\n{self._process.get_error()}\n") # Fixed: stderr() -> get_error()
            error_msg = f"Failed to start virtual terminal interaction."
            logger.error(error_msg)
            self.error = VirtualTerminalError(message=error_msg)
        self.error = None
        logger.info(f"Started virtual terminal interaction: {terminal_input}")

    def _start_polling(self) -> None:
        if self._poll_running.is_set():
            # weird
            logger.warning("Virtual terminal polling thread being started but another thread is already running. Ignoring start request.")
            return

        if self._poll_thread and self._poll_thread.is_alive(): # Check if previous thread is still alive
            logger.warning(f"Virtual terminal polling thread being started but another thread is still alive. Waiting for it to finish.")
            self._poll_done.wait(timeout=5) # Wait for a bit for cleanup
            if self._poll_thread.is_alive():
                logger.error("Previous poll thread did not finish cleanup. This might lead to issues.")
                # Potentially raise an error or force cleanup if it's critical

        self._poll_done.clear() # Clear the done flag for the new poll cycle
        self._poll_running.set() # Set the running flag

        def update_history():
            if self._process is None:
                return
           
            if (current_interaction := self.history.current_interaction) is None:
                return

            # this is dead simple, we just poll feedwater and then append what comes out
            # the only weird thing is that feedwater includes newlines and we don't want those.
            for msg_out in self._process.get():
                current_interaction.stdout.append(
                    InteractionLine( # Fixed: ShellInterAction -> InteractionLine
                        text = msg_out[:-1] # cut off newline
                    )
                )

            for msg_err in self._process.get_error():
                current_interaction.stderr.append(
                    InteractionLine(
                        text = msg_err[:-1] # cut off newline
                    )
                )

                
        
        def poll():
            """Read stdout and stderr of the underlying shell and log it in the history."""
            logger.debug("Virtual terminal polling thread started.")
            while self._poll_running.is_set():
                time.sleep(self.refresh_rate)
                
                if self._process and not self._process.is_running():
                    # it may have finished or crashed
                    # exit code is dealt with in save
                    self._poll_running.clear() # Fixed: self.running.clear() -> self._poll_running.clear()
                    logger.debug("Underlying process finished, stopping polling thread.")
                    break # Exit loop immediately

                update_history()

            # end of loop
            # one last update and then save the interaction
            update_history()
            if self._process is not None:
                # FIXME: in the future, we may want to expand the exit_code with our own type
                self.history.save(exit_code=self._process.exit_code())
            self._poll_done.set()
            logger.debug("Virtual terminal polling thread finished.")
            
        self._poll_thread = threading.Thread(target=poll, daemon=True)
        self._poll_thread.start() # Fixed: Missing .start()
        logger.debug("Virtual terminal polling thread assigned and started.")
        
    def _close_process(self) -> None:
        """Wrap up bookkeeping for a completed process and delete the process object."""
        self._poll_running.clear()
        self._poll_done.wait() # This waits for the polling thread to finish its cleanup
        if self._process:
            self._process.close() # Ensure feedwater process is explicitly closed
        self._process = None
        
    def ready(self) -> bool:
        """Returns true if the virtual terminal is ready for input."""
        # currently we are always ready because there are only 3 cases
        # 1. no process is running -> ok terminal may have just started and we can start a process
        # 2. process exists but has finished -> we are ready for the next process
        # 3. process exists and is running -> in this case we might feed into its stdin (imagine pactl, a python repl, any program that continusously consumes stdin)
        return True
    
    def write_line(self, terminal_input: str) -> None:
        """Writes a line to the virtual terminal's stdin.
        This method is asynchronous and will return immediately."""

        if self._process is None or not self._process.is_running(): # Check if no process is running or if the current one has finished
            # No underlying process is being executed or the previous one finished
            if self._process is not None: # If a process just finished, ensure cleanup
                self._close_process()
            self._start_interaction(terminal_input)
            self._start_polling()            
        else:
            # An underlying process is still running
            self._feed(terminal_input)
            

    def _feed(self, line: str) -> None:
        """Feed one line into the underlying stdin."""
        if self._process is None:
            logger.warning(f"Virtual terminal tried to write to stdin but no process is running. Input '{line}' ignored.")
            return
        
        if (current_interaction := self.history.current_interaction) is None:
            logger.warning(f"Virtual terminal tried to write to stdin but no interaction history is present. The write may procure, but interaction history is unchanged.")
        else:            
            current_interaction.stdin.append(
            InteractionLine(
                text=line
            )
        )
        self._process.write_line(line)

    def _try_parse_cd(self, shell_command: str) -> Optional[str]:
        """Tries to extract strings like "cd build/test" or similar ones from the shell command to keep track of the CWD.
        Returns the new directory if successful, None if no directory change was detected.
        Note: This is not effective in all cases. Things like `m̀kdir build && cd build` won't be caught yet."""
        # this is purely heuristic
        # first we filter out commands with pipes and semicolons and && because we can't handle those
        blacklist = ["; ", " | ", " >", " <", " && "]
        for word in blacklist:
            if word in shell_command:
                # just give up. for now we accept that the cwd may be faulty
                return None

        
        match shlex.split(shell_command):
            case ["cd"]:
                # home directory
                # FIXME: this breaks if the LLM has su-ed into another user but oh well
                return os.path.expanduser("~")
            case ["cd", target_dir]:
                return os.path.abspath(os.path.join(self.cwd, os.path.expanduser(target_dir)))                
            case ["cd", target_dir, *rest]:
                logger.warning(f"Got weird cd command during cwd parse attempt: Command: `{shell_command}`")
                return os.path.abspath(os.path.join(self.cwd, os.path.expanduser(target_dir)))
        return None

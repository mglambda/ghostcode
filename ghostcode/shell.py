# shell.py
import os
import threading
import feedwater
from typing import *
from ghostcode.utility import timestamp_now_iso8601, show_model
from dataclasses import dataclass, field
import logging

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

        stdin_str = "\n".join([msg.show_xml() for msg in self.stdin])
        stdout_str = "\n".join([msg.show_xml() for msg in self.stdout])
        stderr_str = "\n".join([msg.show_xml() for msg in self.stderr])
        
        return f"""<shell-interaction timestamp-start="{self.time_start}" {end_str} {status_str} {exit_code_str}>
<command>
{self.original_command}
</command>
<stdin>
{stdin_str}
</stdin>
<stdout>
{stdout_str}
</stdout>
<stderr>
{stderr_str}
</stderr>
</shell-interaction>
"""
        
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
    
    history: ShellInteractionHistory = field(
        default_factory = ShellInteractionHistory,
    )

    # contains a small error object if an error was encountered in the underlying process, None otherwise
    error: Optional[VirtualTerminalError] = field(
        default = None,
    )

    # amount of time in seconds that passes between polling the underlying process
    refresh_rate: float = Field(
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
        
        
    def _start_interaction(self, terminal_input: str) -> None:
        
        self.history.new(terminal_input)
        self.process = feedwater.run(terminal_input, env=self.env)
        if not(self.process.is_running()):
            logger.debug(f"dumping virtual terminal outputs.\nstdout:\n{self.process.get_stdout()}\n\nstderr:\n{self.process.stderr()}\n")
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

        if not(self._poll_done.is_set()):
            logger.warning(f"Virtual terminal polling thread being started but another thread is not done cleaning up. Ingoring the start request.")
            return

        # flags are in a sane state

        def update_history():
            if self._process is None:
                return
           
            if (current_interaction := self.history.current_interaction) is None:
                return

            # this is dead simple, we just poll feedwater and then append what comes out
            # the only weird thing is that feedwater includes newlines and we don't want those.
            for msg_out in self._process.get():
                current_interaction.stdout.append(
                    ShellInterAction(
                        text = msg_out[:-1] # cut off newline
                    )
                )

            for msg_err in self._process.get_error():
                current_interaction.stderr.append(
                    InteractionLine(
                        text = msg_error[:-1] # cut off newline
                    )
                )

                
        
        def poll():
            """Read stdout and stderr of the underlying shell and log it in the history."""
            if self._poll_done.is_set():
                logger.warning(f"Virtual terminal is starting a new polling thread, while the old one hasn't finished.")
                self._poll_done.clear()
            
            self._poll_running.set()
            while self._poll_running.is_set():
                time.sleep(self.refresh_rate)
                
                if not(self._process.is_running()):
                    # it may have finished or crashed
                    # exit code is dealt with in save
                    self.running.clear()

                update_history()

            # end of loop
            # one last update and then save the interaction
            update_history()
            if self._process is not None:
                # FIXME: in the future, we may want to expand the exit_code with our own type
                self.interaction_history.save(exit_code=self._process.exit_code())
            self._poll_done.set()
            
        self._poll_thread = threading.Thread(target=poll, daemon=True)
        
    def _close_process(self) -> None:
        """Wrap up bookkeeping for a completed process and delete the process object."""
        self._poll_running.clear()
        self._poll_done.wait()
        self._process = None
        
    def ready(self) -> bool:
        """Returns true if the virtual terminal is ready for input."""
        # always ready lol
        return True

    def write_line(self, terminal_input: str) -> None:
        """Writes a line to the virtual terminal's stdin.
        This method is asynchronous and will return immediately."""

        if self._process is None:
            # no underlying process is being executed
            self._start_interaction(terminal_input)
            self._start_polling()            
        else:
            # an underlying process has been started but is in an indeterminate state.
            if self._process.is_running():
                # it's still running and we may want to feed things into standard input (imagine an LLM interacting with pactl).
                self._feed(terminal_input)
            else:
                # the process has finished either successfully or otherwise
                # we will interpret the write as starting a new process
                # human's do this sometimes by accident and usually just get "command not found". We treat the LLM no different.
                self._close_process()
                self.write_line(terminal_input)
            

    def _feed(self, line: str) -> None:
        """Feed one line into the underlying stdin."""
        if self._process is None:
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

# ghostcode.subcommand.interact
from typing import *
from pydantic import Field
import sys
import json
import os
from .. import types
from ..types import CommandOutput
from ..program import CommandInterface, Program
from .. import emacs
from ..utility import (
    show_model_nt,
    EXTENSION_TO_LANGUAGE_MAP,
    language_from_extension,
    clamp_string,
)
from .verify import VerifyCommand
from ..progress_printer import ProgressPrinter
from .. import git
from .. import slash_commands
from .. import worker
from .. import prompts

import logging
logger = logging.getLogger("ghostcode.subcommand.interact")


class InteractCommand(CommandInterface):
    """Launches an interactive session with the Coder LLM."""

    #interaction_history: Optional[types.InteractionHistory] = Field(default=None)

    # wether we will perform actions
    # disabling this will make the backend do talking instead of generating code parts etc
    actions: bool = True

    initial_prompt: Optional[str] = Field(
        default=None,
        description="An optional initial prompt to start the interactive session. If provided, it bypasses the first user input.",
    )

    skip_to: Optional[types.AIAgent] = Field(
        default=None,
        description="If this is set to either coder or worker, interaction will skip prompt routing and query the specified AI directly.",
    )

    interaction_identifier: Optional[str] = Field(
        default=None,
        description="Unique ID or tag of a past interaction to load and continue.",
    )

    initial_interaction_history_length: int = Field(
        default=0,
        description="Number of messages in initial interaction. For new interactions this is always zero. It may be nonzero if an existing interaction is loaded. Used primarily to check wether there was any real change at all and if the current interaction needs to be saved.",
    )

    # whether to force releasing of interaction lock
    force_lock: bool = False

    branch: Optional[str] = Field(
        default=None,
        description="Optional Git branch to checkout before starting the interaction.",
    )

    def run(self, prog: Program) -> CommandOutput:
        result = CommandOutput()
        if not prog.project_root or not prog.project:
            logger.error("Not a ghostcode project. Run 'ghostcode init' first.")
            sys.exit(1)

        # since interact queries the backend, we verify the keys
        verify_result = VerifyCommand().run(prog)
        if verify_result.data["error"]:
            result.print(verify_result.text)
            result.print("Aborting interaction.")
            return result

        actions_str = (
            " Talk only, no actions will be performed in this interaction."
            if not (self.actions)
            else ""
        )
        result.print("Starting interactive session with 👻." + actions_str)
        prog.print("API keys checked. All good.")

        # Handle branch checkout if specified
        if self.branch:
            if not prog.has_git_integration():
                error_msg = "Git integration is disabled in project configuration. Cannot checkout branch."
                prog.print(f"Error: {error_msg}")
                logger.error(error_msg)
                sys.exit(1)

            prog.print(f"Attempting to checkout branch '{self.branch}'...")
            checkout_result = git.checkout_branch(prog.project_root, self.branch)
            if checkout_result.is_err():
                error_msg = f"Failed to checkout branch '{self.branch}': {checkout_result.error}"
                prog.print(f"Error: {error_msg}")
                logger.error(error_msg)
                sys.exit(1)
            prog.print(f"Successfully checked out branch '{self.branch}'.")

        # Load existing interaction history if an identifier is provided
        if self.interaction_identifier:
            if prog.project is None:
                logger.error("Project is null, cannot load interaction history.")
                sys.exit(1)

            loaded_history = prog.project.get_interaction_history(
                unique_id=self.interaction_identifier, tag=self.interaction_identifier
            )

            if loaded_history is None:
                prog.print(
                    f"Error: No interaction found with ID or tag '{self.interaction_identifier}'."
                )
                sys.exit(1)
            else:
                self.initial_interaction_history_length = len(
                    loaded_history.contents
                )
                # need to add the preamble injection
                ghostbox_history = loaded_history.to_chat_messages()
                if ghostbox_history:
                    content = ghostbox_history[0].content
                    if isinstance(content, str):
                        ghostbox_history[0].content = prompts.prefix_preamble_string(
                            content
                        )
                    else:
                        # technically the content can be a list or a dict, but we never use this
                        logger.warning(
                            f"Non-string content field in first ghostbox history message. This means we didn't add the preamble injection string."
                        )
                prog.coder_box.set_history(ghostbox_history)
                prog.print(
                    f"Continuing interaction '{loaded_history.title}' (ID: {loaded_history.unique_id})."
                )
        else:
            # no interaction identifier
            # we create a new interaction history
            loaded_history = prog.project.new_interaction_history()
            
        # with everything set up, start the IPC server to listen for msgs
        prog.start_ipc_server()

        # start the actual loop in another method
        return self._interact_loop(result, prog, loaded_history.unique_id)

        # This code is unreachable but in the future error handling/control flow of this method might become more complicated and we may need it
        return result

    def _make_preamble_config(self, prog: Program) -> types.PromptConfig:
        """Plaintext context that is inserted before the user prompt - though only once."""
        config = prompts.make_default_coder_config()
        config.text_only_nudge = not(self.actions)
        return config

    def _save_interaction(self, prog: Program, interaction_id: str) -> None:
        if prog.project is None:
            logger.error(f"Null project while trying to save interaction history. Aborting.")
            return
        
        interaction_history = prog.project.get_interaction_history(interaction_id)
        if interaction_history is None:
            logger.warning(
                f"Tried to save null interaction history during interaction."
            )
            return

        if interaction_history.empty():
            # nothing to do
            return

        if (
            len(interaction_history.contents)
            == self.initial_interaction_history_length
        ):
            # history may have been loaded and wasn't change -> do nothing
            return

        logger.info(f"Finishing interaction.")
        new_title = worker.worker_generate_title(prog, interaction_history)
        interaction_history.title = (
            new_title if new_title else interaction_history.title
        )
        
    def _emacs_handle_prompt(self, prog: Program, user_prompt: str) -> None:
        """Do potential prompt processing with emacs integration.
        This may e.g. save the last entered prompt to the kill ring."""
        if not prog.user_config.emacs_integration:
            return

        if (register := prog.user_config.emacs_save_prompt_register) != "":
            if (len(register) > 1) or not (register.isalnum()):
                logger.warning(f"User choice of '{register}' for emacs register to save prompt to is bogus. Skipping register save.")
            else:
                emacs.set_register_content(register, user_prompt)

        if prog.user_config.emacs_save_prompt_kill_ring:
            emacs.push_kill_ring(user_prompt)
        
    def _make_llm_response_profile(self) -> types.LLMResponseProfile:
        if not (self.actions):
            return types.LLMResponseProfile.text_only()

        # default is return whatever is the default
        return types.LLMResponseProfile()

    def _make_initial_action(self, **kwargs: Any) -> types.Action:
        """Create the initial action to place on the action queue.
        Arguments are passed directly through to query constructors, like ActionQueryCoder, ActionQueryWorker, ActionRouteRequest, or ActionPrepareRequest.
        """
        if self.skip_to == types.AIAgent.CODER:
            return types.ActionQueryCoder(**kwargs)

        if self.skip_to == types.AIAgent.WORKER:
            return types.ActionQueryWorker(**kwargs)

        # preparing request
        return types.ActionPrepareRequest(**kwargs)

    def _process_user_input(self, prog: Program, user_input: str, interaction_id: str) -> None:
        """Helper method to encapsulate the logic for sending user input to the LLM."""
        if prog.project is None:
            raise RuntimeError(
                f"Project seems to be null during interaction. This shouldn't happen, but just in case, you may want to do `ghostcode init` in your project's directory."
            )
        # this is for user convenience and should use the unaltered prompt
        self._emacs_handle_prompt(prog, user_input)
        
        preamble_config = self._make_preamble_config(prog)

        prog.project.append_interaction_history_item(
            unique_id = interaction_id,
            item = types.UserInteractionHistoryItem(
                prompt=user_input,
                context=prog.project.context_files,
            )
        )

        logger.info(f"Preparing action queue.")
        prog.discard_actions()
        prog.queue_action(
            self._make_initial_action(
                prompt=user_input,
                interaction_history_id = interaction_id,
                hidden=False,
                preamble_config = preamble_config,
                llm_response_profile=self._make_llm_response_profile(),
            )
        )
        worker.run_action_queue(prog)

    def _interact_loop(
            self, intermediate_result: CommandOutput, prog: Program, interaction_id: str
    ) -> CommandOutput:
        if prog.project is None:
            logger.error("Encountered null project in interact loop. Aborting.")
            intermediate_result.print(
                "Failed to initialize project. Please do\n\n```\nghostcode init\n```\n\nto create a project in the current working directory, then retry interact."
            )
            return intermediate_result

        prog.print(intermediate_result.text)
        #if self.interaction_history is None:
            #self.interaction_history = prog.project.new_interaction_history()

        # lock guard
        if (lock_id := prog.lock_read()) is not None:
            if self.force_lock:
                logger.warning(
                    f"Failed to acquire lock because of interaction {lock_id}, but lock will be forced."
                )
                prog.lock_release()
            else:
                logger.error(
                    f"Failed to acquire interaction lock due to ongoing interaction {lock_id} ."
                )
                intermediate_result.print(
                    f"Failed to acquire lock. Aborting.\nAnother ghostcode session (interaction {lock_id}) is currently in progress. Please finish that interaction, or\nrestart ghostcode with `ghostcode interaction --force` to force it closed. This may lead to data loss. You have been warned."
                )
                return intermediate_result

        # Initial prompt handling
        if self.initial_prompt is not None:
            current_user_input = self.initial_prompt
            self.initial_prompt = None  # Consume the initial prompt
        else:
            current_user_input = ""

        if current_user_input:
            # If an initial prompt was provided, process it immediately
            self._process_user_input(prog, current_user_input, interaction_id)
            # After processing, the loop will continue to ask for more input
            current_user_input = ""  # Clear for subsequent inputs

        # Main interactive loop
        prog.print(
            "Multiline mode enabled. Type your prompt over multiple lines.\nType a single '\\' and hit enter to submit.\nType /quit or CTRL+D to quit."
        )

        try:
            with prog.interaction_lock(
                interaction_history_id=interaction_id
            ):

                while True:
                    try:
                        if current_user_input == "":
                            # don't print this if user is building multi-line input
                            prog.print(prog._get_cli_prompt(), end="")
                        with prog.idle_work():
                            # we only idle on the cli prompt, everything else is guaranteed to have the worker be shut off
                            line = input()

                    except EOFError:
                        break  # User pressed CTRL+D, exit interaction

                    slash_result = slash_commands.try_command(
                        prog, line
                    )
                    match slash_result:
                        case slash_commands.SlashCommandResult.OK:
                            continue  # Command handled, go to next loop iteration (ask for input)
                        case slash_commands.SlashCommandResult.HALT:
                            break  # Command halted, exit interaction
                        case slash_commands.SlashCommandResult.COMMAND_NOT_FOUND:
                            prog.print(f"Unrecognized command: {line}")
                            continue
                        case slash_commands.SlashCommandResult.BAD_ARGUMENTS:
                            prog.print(
                                f"Invalid arguments. Try /help COMMAND for more information."
                            )
                            continue
                        case slash_commands.SlashCommandResult.ACTIONS_OFF:
                            if self.actions:
                                self.actions = False
                                prog.print(
                                    "Enabled talk mode. Coder backend will generate text only, no file edits."
                                )
                            else:
                                prog.print(
                                    "Talk mode already enabled, use /interact to switch to interactive mode."
                                )
                        case slash_commands.SlashCommandResult.ACTIONS_ON:
                            if not (self.actions):
                                self.actions = True
                                prog.print(
                                    "Interact mode enabled. Coder backend will generate code and produce file edits."
                                )
                            else:
                                prog.print(
                                    "Interact mode already enabled. Use /talk to disable code generation and file edits."
                                )
                        case slash_commands.SlashCommandResult.RESET_SESSION:
                            self._save_interaction(prog, interaction_id)
                            prog.coder_box.clear_history() # Clear coder's chat history
                            prog.discard_actions() # Clear any pending actions
                            interaction_id = prog.project.new_interaction_history().unique_id
                            self.initial_interaction_history_length = 0 # Reset length for the new session
                            prog.print("New interactive session started.")
                            current_user_input = "" # Clear current input buffer
                            continue # Continue the loop to get new input
                        case _:
                            pass  # Not a slash command, accumulate input
                    # Accumulate user input
                    if line != "\\":
                        current_user_input += "\n" + line

                        if prog.user_config.newbie and current_user_input.endswith(
                            "\n\n"
                        ):
                            # user may be frantically trying to submit
                            prog.print(
                                "(Hint: Enter a single backslash (\\) and hit enter to submit your prompt. Disable this message with `ghostcode config set newbie False`)"
                            )

                        continue  # Keep accumulating

                    # If we reach here, it means user typed '\\' to submit
                    if not current_user_input.strip():
                        prog.print(
                            "Empty prompt. Please provide some input or a slash command."
                        )
                        continue  # Ask for input again

                    self._process_user_input(prog, current_user_input, interaction_id)
                    current_user_input = ""  # Clear buffer for next turn
        except types.InteractionLockError as e:
            logger.error(f"Failed to acquire lock: {e}")
            prog.print(
                f"Cannot proceed because another ghostcode session is in progress (interaction {prog.lock_read()}).\nPlease finish the ongoing interaction, or force it to close by running ghostcode \nwith `ghostcode interaction --force`. Data may be lost. You have been warned."
            )

        # End of interaction
        prog.debug_dump()
        self._save_interaction(prog, interaction_id)

        # ipc cleanup
        if prog.ipc_server:
            logger.info(f"Stopping IPC server and cleaning up info file.")
            prog.ipc_server.stop()
            prog._ipc_server_info_clear()  # Clear the info file after stopping the server

        return CommandOutput(text="Finished interaction.")


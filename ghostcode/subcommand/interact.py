# ghostcode.subcommand.interact
from typing import *
from pydantic import Field
import sys
import json
import os
from .. import types
from ..types import CommandOutput
from ..program import CommandInterface, Program
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

    interaction_history: Optional[types.InteractionHistory] = Field(default=None)

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
                self.interaction_history = loaded_history
                self.initial_interaction_history_length = len(
                    self.interaction_history.contents
                )
                # need to add the preamble injection
                ghostbox_history = self.interaction_history.to_chat_messages()
                if ghostbox_history:
                    content = ghostbox_history[0].content
                    if isinstance(content, str):
                        ghostbox_history[0].content = self._prefix_preamble_string(
                            content
                        )
                    else:
                        # technically the content can be a list or a dict, but we never use this
                        logger.warning(
                            f"Non-string content field in first ghostbox history message. This means we didn't add the preamble injection string."
                        )
                prog.coder_box.set_history(ghostbox_history)
                prog.print(
                    f"Continuing interaction '{self.interaction_history.title}' (ID: {self.interaction_history.unique_id})."
                )

        # with everything set up, start the IPC server to listen for msgs
        prog.start_ipc_server()

        # start the actual loop in another method
        return self._interact_loop(result, prog)

        # This code is unreachable but in the future error handling/control flow of this method might become more complicated and we may need it
        return result

    def _make_preamble_config(self, prog: Program) -> types.PromptConfig:
        """Plaintext context that is inserted before the user prompt - though only once."""
        return types.PromptConfig.minimal(
            project_metadata=True,
            style_file=True,
            context_files="full",
            recent_interaction_summaries="full",
            problematic_source_reports = True
            # could add shell here?
        )


    def _prefix_preamble_string(self, prompt: str) -> str:
        """
        Prefixes the user's prompt with the magic `{_{_preamble_injection_}_}` placeholder (without the outer underscores).
        This method *only* adds the placeholder string. The actual content for the
        preamble is dynamically set and updated via `prog.coder_box.set_vars()`
        before each LLM call. This ensures the preamble is always current without
        being hardcoded into the interaction history or prompt template.
        """
        # note the song-and-dance with string concatenation below is so that we can use ghostcode on itself without an unwanted expansion of the preamble magic string
        return "{{" + "preamble_injection" + "}}" + f"# User Prompt\n\n{prompt}"

    def _make_user_prompt(self, prog: Program, prompt: str) -> str:
        """Prepare a user prompt to be sent to the backend.

        This method intelligently constructs the prompt to be sent to the LLM backend.
        For the *first* message in a new interaction, it includes the `{_{_preamble_injection_}_}` (without the outer underscores)
        placeholder by calling `_prefix_preamble_string`. For subsequent messages in an
        ongoing interaction, it sends only the raw user `prompt`.

        The actual content of the preamble (project metadata, context files, etc.) is
        generated by `_make_preamble` and then dynamically injected into the LLM context
        by `prog.coder_box.set_vars()` before each LLM request. This ensures that the
        LLM always receives the most up-to-date project context without incurring
        repeated token costs for the full preamble on every turn.
        """
        if self.interaction_history is None:
            logger.warning(
                f"Tried to construct preamble with null interaction history in interaction."
            )
            return ""

        # Only inject the preamble placeholder for the very first message of an interaction.
        # Subsequent messages rely on `prog.coder_box.set_vars` to keep the preamble updated.
        if self.interaction_history.empty():
            return self._prefix_preamble_string(prompt)
        return prompt

    def _dump_interaction(self, prog: Program) -> None:
        """Dumps interaction history to .ghostcode/current_interaction.txt and .ghostcode/current_interaction.json"""
        if (
            prog.project is not None
            and prog.project_root is not None
            and self.interaction_history is not None
        ):
            with ProgressPrinter(
                message=" Saving interaction ", print_function=prog.print
            ):
                current_interaction_history_filepath = os.path.join(
                    prog.project_root,
                    ".ghostcode",
                    prog.project._CURRENT_INTERACTION_HISTORY_FILE,
                )
                current_interaction_history_plaintext_filepath = os.path.join(
                    prog.project_root,
                    ".ghostcode",
                    prog.project._CURRENT_INTERACTION_PLAINTEXT_FILE,
                )
                logger.info(
                    f"Dumping current interaction history to {current_interaction_history_filepath} and {current_interaction_history_plaintext_filepath}"
                )
                with open(current_interaction_history_filepath, "w") as hf:
                    json.dump(self.interaction_history.model_dump(), hf, indent=4)

                with open(current_interaction_history_plaintext_filepath, "w") as pf:
                    pf.write(self.interaction_history.show())
        else:
            logger.warning(
                f"Null project_root while trying to dump current itneraction history. Create a project root or no dump for you!"
            )

    def _save_interaction(self, prog: Program) -> None:
        if self.interaction_history is None:
            logger.warning(
                f"Tried to save null interaction history during interaction."
            )
            return

        if self.interaction_history.empty():
            # nothing to do
            return

        if (
            len(self.interaction_history.contents)
            == self.initial_interaction_history_length
        ):
            # history may have been loaded and wasn't change -> do nothing
            return

        logger.info(f"Finishing interaction.")
        new_title = worker.worker_generate_title(prog, self.interaction_history)
        self.interaction_history.title = (
            new_title if new_title else self.interaction_history.title
        )

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

    def _process_user_input(self, prog: Program, user_input: str) -> None:
        """Helper method to encapsulate the logic for sending user input to the LLM."""
        if prog.project is None:
            raise RuntimeError(
                f"Project seems to be null during interaction. This shouldn't happen, but just in case, you may want to do `ghostcode init` in your project's directory."
            )

        #preamble = self._make_preamble(prog)
        preamble_config = self._make_preamble_config
        prompt_to_send = self._make_user_prompt(prog, user_input)
        #prog.coder_box.set_vars({"preamble_injection": preamble})

        if self.interaction_history is not None:
            self.interaction_history.contents.append(
                types.UserInteractionHistoryItem(
                    prompt=user_input,
                    context=prog.project.context_files,
                )
            )
        else:
            logger.warning(f"No interaction history. Interaction discarded.")

        logger.info(f"Preparing action queue.")
        prog.discard_actions()
        prog.queue_action(
            self._make_initial_action(
                prompt=prompt_to_send,
                interaction_history_id=(
                    self.interaction_history.unique_id
                    if self.interaction_history is not None
                    else "000-000-000-000"
                ),
                hidden=False,
                preamble_config = preamble_config,
                llm_response_profile=self._make_llm_response_profile(),
            )
        )
        worker.run_action_queue(prog)

    def _interact_loop(
        self, intermediate_result: CommandOutput, prog: Program
    ) -> CommandOutput:
        if prog.project is None:
            logger.error("Encountered null project in interact loop. Aborting.")
            intermediate_result.print(
                "Failed to initialize project. Please do\n\n```\nghostcode init\n```\n\nto create a project in the current working directory, then retry interact."
            )
            return intermediate_result

        prog.print(intermediate_result.text)
        if self.interaction_history is None:
            self.interaction_history = prog.project.new_interaction_history()

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
            self._process_user_input(prog, current_user_input)
            # After processing, the loop will continue to ask for more input
            current_user_input = ""  # Clear for subsequent inputs

        # Main interactive loop
        prog.print(
            "Multiline mode enabled. Type your prompt over multiple lines.\nType a single '\\' and hit enter to submit.\nType /quit or CTRL+D to quit."
        )

        try:
            with prog.interaction_lock(
                interaction_history_id=self.interaction_history.unique_id
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
                        prog, self.interaction_history, line
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
                            self._save_interaction(prog) # Save the current interaction
                            prog.coder_box.clear_history() # Clear coder's chat history
                            prog.discard_actions() # Clear any pending actions
                            self.interaction_history = prog.project.new_interaction_history() # Create a new interaction history
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

                    self._process_user_input(prog, current_user_input)
                    current_user_input = ""  # Clear buffer for next turn
                    self._dump_interaction(prog)  # Save state after each turn
        except types.InteractionLockError as e:
            logger.error(f"Failed to acquire lock: {e}")
            prog.print(
                f"Cannot proceed because another ghostcode session is in progress (interaction {prog.lock_read()}).\nPlease finish the ongoing interaction, or force it to close by running ghostcode \nwith `ghostcode interaction --force`. Data may be lost. You have been warned."
            )

        # End of interaction
        prog.debug_dump()
        self._save_interaction(prog)

        # ipc cleanup
        if prog.ipc_server:
            logger.info(f"Stopping IPC server and cleaning up info file.")
            prog.ipc_server.stop()
            prog._ipc_server_info_clear()  # Clear the info file after stopping the server

        return CommandOutput(text="Finished interaction.")


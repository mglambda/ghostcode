# ghostcode/worker.py
from typing import *
from ghostbox.commands import showTime
import traceback
import os
import shutil
import json
import logging
from . import types
from .progress_printer import ProgressPrinter
from .ansi_colors import Color256
from .program import Program
from  . import prompts
import time
from .utility import (
    levenshtein,
    time_function_with_logging,
    show_model,
    timestamp_now_iso8601,
    foldl,
    mock_print,
)
import tempfile

# --- Logging Setup ---
logger = logging.getLogger("ghostcode.worker")


def actions_from_response_parts(
        prog: Program, parts: Sequence[types.LLMResponsePart], *, original_action: types.Action
) -> List[types.Action]:
    """Transform a list of response parts from either coder or worker LLM into a list of actions.
    This is not a 1 to 1 mapping. You may end up with an empty list if e.g. all the response parts are just discussion text. Only code parts and similar are transformed into actions.
    """
    # atm we have prog only because we might need more context here in the future
    logger.debug(
        f"actions_from_response_parts with {len(parts)} parts: {[part.__class__.__name__ for part in parts]} from original {original_action.__class__.__name__}"
    )

    if not (parts):
        return []

    def actions_from_part(part: types.LLMResponsePart) -> List[types.Action]:
        logger.info(f"Finding actions for {part.__class__.__name__}.")
        match part:
            case types.TextResponsePart() as text_part:
                # no action necessary
                return []
            case types.CodeResponsePart() as code_part:
                return [types.ActionHandleCodeResponsePart(content=code_part)]
            case types.ShellCommandPart() as shell_command_part:
                return [
                    types.ActionShellCommand(
                        content=shell_command_part,
                    )
                ]
            case types.FilesLoadPart() as files_load_part:
                return [
                    types.ActionAlterContext(
                        context_alteration=types.ContextAlterationLoadFile(
                            filepath=filepath
                        )
                    )
                    for filepath in files_load_part.filepaths
                ]
            case types.FilesUnloadPart() as files_unload_part:
                return [
                    types.ActionAlterContext(
                        context_alteration=types.ContextAlterationUnloadFile(
                            filepath=filepath
                        )
                    )
                    for filepath in files_unload_part.filepaths
                ]
            case types.EndInteractionPart() as end_interaction_part:
                # FIXME: we might want something softer than halt in the future, something that stop interactions with LLMs but resolves actions that are still on the queue
                return [
                    types.ActionHaltExecution(
                        reason=f"LLM ended their interaction. Reason: {end_interaction_part.reason}",
                        announce=True,
                    )
                ]
            case types.WorkerCoderRequestPart() as worker_coder_request_part:
                # this means the worker is bouncing some query to the coder, unusual enough to log and notify the user
                logger.info(f"Reason for WorkerCoderRequestPart: {worker_coder_request_part.reason}")
                prog.announce(
                    text=f"Bounced query to 👻. Reason: {worker_coder_request_part.reason}",
                    agent=types.AIAgent.WORKER
                )
                # FIXME: narrow the type of original_action to ActionQuery, then remove the type: ignores below
                # we can safely narrow because this is literally handling a response part -> so there must have been a query
                # the only thing that prevents this atm is that recovery is not an action
                return [types.ActionQueryCoder(
                    prompt=prompts.make_prompt_worker_coder_query(prog, worker_coder_request_part, original_action),
                    llm_response_profile=types.LLMResponseProfile.allow_all(), # maximum allowed because it might be needed for recovery
                    hidden=original_action.hidden, # type: ignore
                    interaction_history_id=original_action.interaction_history_id, # type: ignore
                )]
            case _:
                logger.warning(
                    f"Match case fallthrough on response part: {part.__class__.__name__}"
                )
                return []

    return foldl(lambda part, actions: actions_from_part(part) + actions, [], parts)


@time_function_with_logging
def run_action_queue(prog: Program) -> None:
    """Executes and removes actions at the front of the action queue until the queue is empty or the halt action is popped.
    This function has no return value as actions primarily exist to crate side effects.
    """
    logger.info(
        f"Action queue execution started with {len(prog.action_queue)} actions in queue."
    )

    # each time we run a queue, we reset the flag that let's the user skip confirmation dialogs
    # the user sets this to true with the "a" confirmation option.
    prog.action_queue_yolo = False

    if not (prog.action_queue):
        logger.info(
            f"Action queue execution stopped because there are no actions on the queue."
        )
        return

    try:
        finished_actions = 0
        while action := prog.action_queue.pop(0):
            logger.debug(
                f"Current Queue Status\n  popped action: {action.__class__.__name__}\n  queue: {[a.__class__.__name__ for a in prog.action_queue]}"
            )
            prog.print(
                types.action_show_user_message(
                    action, delimiter_color=prog.cosmetic_state.to_color()
                ),
                end="",
                tag="action_queue",
            )
            match action:
                case types.ActionHaltExecution() as halt_action:
                    logger.info(
                        f"Halting action queue execution after {finished_actions} actions, with {len(prog.action_queue)} actions remaining in queue. Reason: {halt_action.reason}"
                    )
                    prog.discard_actions()
                    if halt_action.announce:
                        prog.print(f"Halted. {halt_action.reason}")
                    return
                case _:
                    logger.info(
                        f"Executing {types.action_show_short(action)}. Finished {finished_actions}, remaining {len(prog.action_queue)}."
                    )
                    logger.debug(
                        f"More info on action being executed:\n{json.dumps(action.model_dump(), indent=4)}"
                    )
                    result = execute_action(prog, action)
                    finished_actions += 1
                    match result:
                        case types.ActionResultOk() as result_ok:
                            prog.cosmetic_state = types.CosmeticProgramState.WORKING
                            logger.info(
                                f"Action execution successful."
                                + f"Message: {result_ok.success_message}"
                                if result_ok.success_message
                                else ""
                            )
                        case types.ActionResultFailure() as ar_failure:
                            prog.cosmetic_state = types.CosmeticProgramState.PROBLEM
                            prog.sound_error()
                            logger.info(
                                f"Failed to execute action {type(ar_failure.original_action)}. Reason: {ar_failure.failure_reason}"
                            )
                            logger.debug(
                                f"Original action dump:\n{json.dumps(ar_failure.original_action.model_dump(), indent=4)}\n\nFailure result dump:\n{json.dumps(ar_failure.model_dump(), indent=4)}"
                            )
                            for action in worker_recover(prog, ar_failure).actions:
                                prog.push_front_action(action)
                        case types.ActionResultMoreActions() as ar_more:
                            prog.cosmetic_state = types.CosmeticProgramState.WORKING
                            logger.info(
                                f"Got MoreAction result for {len(ar_more.actions)} actions. Action queue has finished {finished_actions}, remaining {len(prog.action_queue)}."
                            )
                            for new_action in ar_more.actions:
                                prog.push_front_action(new_action)
                        case _:
                            prog.cosmetic_state = types.CosmeticProgramState.WORKING
                            logger.info(f"Got unknown action result. Weird, but ok.")
                            logger.debug(
                                f"Dumping unknown action result:\n{json.dumps(result.model_dump(), indent=4)}"
                            )
    except IndexError:
        prog.cosmetic_state = types.CosmeticProgramState.IDLE
        logger.info(
            f"Action queue execution finished after {finished_actions} actions."
        )
        return


def execute_action(prog: Program, action: types.Action) -> types.ActionResult:
    # small helper for this context
    def fail(
        failure_reason: str,
        error_messages: List[str] = [],
        action: types.Action = action,
    ) -> types.ActionResultFailure:
        return types.ActionResultFailure(
            original_action=action,
            error_messages=error_messages,
            failure_reason=failure_reason,
        )

    logger.debug(f"Executing action: {action.__class__.__name__}")

    # check clearance
    clearance_requirement = action.clearance_required
    if prog.project is None:
        logger.warning(
            f"Null project while trying to get worker clearance, so no config is available. Defaulting to lowest clearance."
        )
        clearance_level = types.ClearanceRequirement.AUTOMATIC
    else:
        clearance_level = prog.project.config.worker_clearance_level
    if clearance_requirement == types.ClearanceRequirement.FORBIDDEN:
        logger.info(
            f"Worker tried to execute action with forbidden clearance. This is never permitted, regardless of clearance level. Failing action."
        )
        return fail("Worker encountered action with 'forbidden' clearance requirement.")
    elif clearance_level.value < clearance_requirement.value:
        logger.info(
            f"Worker clearance of {clearance_level.name} insufficient for clearance requirement {clearance_requirement} of {action.__class__.__name__} action. Seeking user confirmation."
        )
        user_response = prog.confirm_action(
            action, agent_clearance_level=clearance_level, agent_name="Ghostworker"
        )
        if user_response == types.UserConfirmation.CANCEL:
            # we just exit regularly through the halting mechanism
            return types.ActionResultMoreActions(
                actions=[
                    types.ActionHaltExecution(reason="User canceled action execution.")
                ]
            )
        elif types.UserConfirmation.is_confirmation(user_response):
            if user_response == types.UserConfirmation.ALL:
                prog.action_queue_yolo = True
            logger.info(
                f"User confirmed-all for {action.__class__.__name__} for worker with clearance level {clearance_level.name} (YOLO)."
            )
        else:
            logger.info(
                f"User denied {action.__class__.__name__} for worker with clearance level {clearance_level.name}."
            )
            return fail(f"Action {action.__class__.__name__} denied by user.")
    else:
        logger.info(
            f"Worker clearance of {clearance_level.name} meets or exceeds clearance requirement {clearance_requirement.name} for {action.__class__.__name__} action. Proceeding without user confirmation."
        )

    try:
        match action:
            case types.ActionRouteRequest() as route_request_action:
                logger.info(f"Determining route.")
                match worker_route_request(prog, route_request_action.prompt):
                    case types.AIAgent.WORKER:
                        logger.info(f"Routing user request to ghostworker.")
                        route = [
                            types.ActionQueryWorker(**route_request_action.model_dump())
                        ]  # type: List[types.Action]
                    case types.AIAgent.CODER:
                        logger.info(f"Routing request to ghostcoder.")
                        route = [
                            types.ActionQueryCoder(**route_request_action.model_dump())
                        ]
                return types.ActionResultMoreActions(actions=route)
            case types.ActionPrepareRequest() as prepare_request_action:
                logger.info(f"Preparing request.")
                return prepare_request(prog, prepare_request_action)
            case types.ActionHandleCodeResponsePart() as code_action:
                logger.info(f"Handling code response part.")
                logger.debug(
                    f"Code response part action dump:\n{json.dumps(code_action.model_dump(), indent=4)}"
                )
                return handle_code_part(prog, code_action)
            case types.ActionFileEdit() as edit_action:
                logger.info(f"File Edit Action")
                return apply_edit_file(prog, edit_action)
            case types.ActionFileCreate() as create_action:
                logger.info("Create File Action")
                return apply_create_file(prog, create_action)
            case types.ActionDoNothing():
                # No operation needed for ActionDoNothing
                logger.info("Action: Do Nothing")
                return types.ActionResultOk()
            case types.ActionAlterContext() as alter_context_action:
                logger.info(
                    f"Change of context with {alter_context_action.context_alteration.__class__.__name__}."
                )
                return apply_alter_context(prog, alter_context_action)
            case types.ActionShellCommand() as shell_command_action:
                logger.info(
                    f"Running shell command: {shell_command_action.content.command}"
                )
                logger.debug(
                    f"Full dump of shell command action: {show_model(shell_command_action)}"
                )
                prog.tty.write_line(shell_command_action.content.command)

                return types.ActionResultMoreActions(
                    actions=[
                        types.ActionWaitOnShellCommand(
                            original_command=shell_command_action
                        )
                    ]
                )
            case types.ActionWaitOnShellCommand() as wait_on_shell_command_action:
                logger.info(f"Waiting on shell command.")
                return worker_wait_on_shell_command(prog, wait_on_shell_command_action)
            case types.ActionQueryCoder() as query_coder_action:
                logger.info(f"Querying the coder backend.")
                return coder_query(prog, query_coder_action)
            case types.ActionQueryWorker() as query_worker_action:
                logger.info(f"Querying the worker backend.")
                return worker_query(prog, query_worker_action)
            case _:
                # Handle unknown action types
                failure_message = f"Received an unknown action type: {type(action).__name__}. This action cannot be executed."
                logger.error(failure_message)
                prog.print(f"ERROR: {failure_message}")
                return types.ActionResultFailure(
                    original_action=action, failure_reason=failure_message
                )
    except Exception as e:
        logger.exception(
            f"Caught exception in execute_action: {e}. Full traceback: \n{traceback.format_exc()}"
        )
        return types.ActionResultFailure(
            original_action=action, failure_reason=f"Caught exception: {e}"
        )


def apply_alter_context(
    prog: Program, alter_context_action: types.ActionAlterContext
) -> types.ActionResult:
    """Change the files currently in context."""
    if prog.project is None:
        msg = f"Null project while trying to alter the project context files."
        logger.error(msg)
        return types.ActionResultFailure(
            failure_reason=msg, original_action=alter_context_action
        )

    match alter_context_action.context_alteration:
        case types.ContextAlterationLoadFile() as load_file:
            prog.project.context_files.add_or_alter(
                load_file.filepath
            )
            return types.ActionResultOk(
                success_message=f"Added {load_file.filepath} to context."
            )
        case types.ContextAlterationUnloadFile() as unload_file:
            if prog.project.context_files.remove(unload_file.filepath):
                # it wasn't found
                return types.ActionResultFailure(
                    failure_reason=f"Could not remove {unload_file.filepath} from context: No such filepath in context.",
                    original_action=alter_context_action,
                )
            return types.ActionResultOk(
                success_message=f"Removed {unload_file.filepath} from context."
            )
        case types.ContextAlterationFlagFile() as flag_file:
            return types.ActionResultFailure(
                failure_reason="Not implemented yet.",
                original_action=alter_context_action,
            )
        case _ as unreachable:
            # trigger mypy error
            assert_never(unreachable)


def apply_create_file(
    prog: Program, create_action: types.ActionFileCreate
) -> types.ActionResult:
    """Apply an ActionFileCreate to actually write a new file to disk."""

    def fail(
        failure_reason: str,
        error_messages: List[str] = [],
        create_action: types.ActionFileCreate = create_action,
    ) -> types.ActionResultFailure:
        return types.ActionResultFailure(
            original_action=create_action,
            error_messages=error_messages,
            failure_reason=failure_reason,
        )

    try:
        dir = os.path.dirname(create_action.filepath)
        if dir != "":
            os.makedirs(dir, exist_ok=True)
        with open(create_action.filepath, "w") as f:
            f.write(create_action.content)
    except Exception as e:
        error_msg = f"Couldn't write new file {create_action.filepath}. Reason: {e}"
        logger.error(error_msg)
        logger.debug(
            f"Edit action dump:\n{json.dumps(create_action.model_dump(), indent=4)}"
        )
        return fail(error_msg, error_messages=[traceback.format_exc()])

    return types.ActionResultOk(
        success_message=f"Wrote new file {create_action.filepath}"
    )


def apply_edit_file(
    prog: Program, edit_action: types.ActionFileEdit
) -> types.ActionResult:
    """Apply an ActionEditFile action and replace some lines in a file with new contents."""
    filepath = edit_action.filepath
    if not (os.path.isfile(filepath)):
        error_msg = f"Tried to apply edit action but file {filepath} does not exist."
        logger.error(error_msg)
        return types.ActionResultFailure(
            original_action=edit_action, failure_reason=error_msg
        )

    if os.path.isdir(filepath):
        error_msg = (
            f"Trying to apply edit action to file {filepath} but file is a directory."
        )
        logger.error(error_msg)
        return types.ActionResultFailure(
            original_action=edit_action, failure_reason=error_msg
        )

    # ok the way we do this is to read it all into memory, replace, write a temp file, and then atomically swap the files
    # this avoids file erasure in the unlikely event that we crash between truncation and writing of the target file
    try:
        with open(filepath, "r") as f:
            content = f.read()

        new_content = (
            content[0 : edit_action.replace_pos_begin]
            + edit_action.insert_text
            + content[edit_action.replace_pos_end :]
        )
        fd, temp_filepath = tempfile.mkstemp()
        os.close(fd)

        with open(temp_filepath, "w") as tf:
            tf.write(new_content)

        # atomic swap
        shutil.move(temp_filepath, filepath)

    except Exception as e:
        error_msg = f"Could not edit  file while applying edit action. Reason: {e}"
        logger.error(error_msg)
        logger.debug(
            f"Dump of action:\n{json.dumps(edit_action.model_dump(), indent=4)}"
        )
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        return types.ActionResultFailure(
            original_action=edit_action,
            failure_reason=error_msg,
            error_messages=[traceback.format_exc()],
        )
    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

    return types.ActionResultOk(
        success_message=f"Applied edit action to file {filepath} for {len(edit_action.insert_text)} characters."
    )


def coder_query(
    prog: Program, query_coder_action: types.ActionQueryCoder
) -> types.ActionResult:
    print_function = mock_print if query_coder_action.hidden else prog.print

    try:
        with ProgressPrinter(
            message=f"{prog.last_print_text} Querying 👻 ",
            postfix_message=f"{prog.last_print_text}",
            print_function=prog.make_tagged_printer("action_queue"),
        ), prog.sound_clicks():
            profile = query_coder_action.llm_response_profile
            if profile.actions and profile.text:
                # this is the default, offer the coder the full menu of parts to generate
                response = prog.coder_box.new(
                    types.CoderResponse,
                    query_coder_action.prompt,
                )
            elif profile.text:
                # we only want a text response
                text_response = prog.coder_box.new(
                    types.TextResponsePart, query_coder_action.prompt
                )
                response = types.CoderResponse(contents=[text_response])
        logger.timing(f"ghostcoder performance statistics:\n{showTime(prog.coder_box._plumbing, [])}")  # type: ignore
    except Exception as e:
        # when this happens it's usually a backend error e.g. 'too many erquests'
        error_msg = prog.coder_box.get_last_error()
        prog.print(  # this bypasses hidden and that's ok
            f"Problems encountered during 👻 request. Reason: {error_msg}\nRetry the request, do /traceback, or consult the logs for more information."
        )
        logger.exception(
            f"error on ghostbox.new. See the full traceback:\n{traceback.format_exc()}"
        )
        logger.error(
            f"\nAnd here is the last result:\n\n{prog.coder_box.get_last_result()}"
        )
        return types.ActionResultFailure(
            original_action=query_coder_action,
            failure_reason=f"Querying the ghostcoder backend failed. Exception: `{e}` Backend message: `{error_msg}`.",
        )

    # append to history if any ID is given
    if query_coder_action.interaction_history_id is not None:
        try:
            if prog.project is not None:
                prog.project.append_interaction_history_item(
                    unique_id=query_coder_action.interaction_history_id,
                    item=types.CoderInteractionHistoryItem(
                        context=prog.project.context_files,
                        backend=prog.get_current_backend_coder(),
                        model=prog.get_current_model_coder(),
                        response=response,
                    ),
                )
            else:
                logger.warning(
                    f"null project while trying to append coder response item to interaction with ID {query_coder_action.interaction_history_id}"
                )
        except Exception as e:
            # this is not critical, log and continue
            logger.error(f"Could not save response to history. Reason: {e}")

    print_function((response.show_cli()))
    return types.ActionResultMoreActions(
        actions=actions_from_response_parts(prog, response.contents, original_action=query_coder_action)
    )


@time_function_with_logging
def handle_code_part(
    prog: Program, code_action: types.ActionHandleCodeResponsePart
) -> types.ActionResult:
    """Executes an ActionHandleCodeResponsePart action."""

    def fail(
        failure_reason: str,
        error_messages: List[str] = [],
        code_action: types.ActionHandleCodeResponsePart = code_action,
    ) -> types.ActionResultFailure:
        return types.ActionResultFailure(
            original_action=code_action,
            error_messages=error_messages,
            failure_reason=failure_reason,
        )

    # first rule out some easy cases
    if code_action.content.filepath is None:
        logger.info("Handling code part with null filepath.")
        return types.ActionResultOk(
            success_message="Nothing to be done for this response, due to missing/none filepath."
        )

    filepath = code_action.content.filepath

    # Case 1: File does not exist -> Create file
    if not os.path.exists(filepath):
        logger.info(
            f"Null filepath on code response part. Moving to create a new file. (case 1)"
        )
        logger.debug(f"handle_code_part: Attempting to create a new file. ({filepath})")
        if os.path.isdir(filepath):
            return fail(
                f"Cannot create file '{filepath}' because a directory with that name already exists."
            )

        # If original_code is provided for a new file, it's a bit odd, but we'll prioritize new_code
        if code_action.content.original_code is not None:
            logger.warning(f"Code part has no filepath, but original code is non-null.")
            logger.debug(
                f"Code part with title '{code_action.content.title}', original_code (abridged): {code_action.content.original_code[:50]}"
            )

        # The type "full" is most appropriate for new file creation.
        if code_action.content.type != "full":
            logger.warning(
                f"Code part has no filepath set, but type is '{code_action.content.type}'."
            )

        return types.ActionResultMoreActions(
            actions=[
                types.ActionFileCreate(
                    filepath=filepath, content=code_action.content.new_code
                )
            ]
        )

    # invariant: File exists. Read its contents.
    if os.path.isdir(filepath):
        return fail(f"Cannot edit file '{filepath}' because it is a directory.")
    try:
        with open(filepath, "r") as f:
            file_contents = f.read()
    except Exception as e:
        error_msg = f"Could not read file '{filepath}' during handling of code response action. Reason: {e}"
        logger.error(error_msg)
        logger.debug(
            f"Handle code part action dump:\n{json.dumps(code_action.model_dump(), indent=4)}"
        )
        return fail(error_msg, error_messages=[traceback.format_exc()])

    # case 2.1 full replacement because type is specified as "full"
    if code_action.content.type == "full":
        if code_action.content.original_code is not None:
            # bit odd but ok
            logger.warning(
                f"original_code is not null during full replacement of existing file (case 2.1)"
            )
        logger.info(
            f"Full replacement due to CodeResponsePart type {code_action.content.type} type with existing file."
        )
        return types.ActionResultMoreActions(
            actions=[
                types.ActionFileEdit(
                    filepath=filepath,
                    replace_pos_begin=0,
                    replace_pos_end=len(file_contents),
                    insert_text=code_action.content.new_code,
                )
            ]
        )

    # Case 2.2: Full replacement (if original_code matches entire file)
    if (
        code_action.content.original_code is not None
        and file_contents == code_action.content.original_code
    ):
        logger.info(
            f"Identical content for original code and entire file {filepath}. Performing full replacement. (case 2.2)"
        )
        return types.ActionResultMoreActions(
            actions=[
                types.ActionFileEdit(
                    filepath=filepath,
                    replace_pos_begin=0,
                    replace_pos_end=len(file_contents),
                    insert_text=code_action.content.new_code,
                )
            ]
        )

    # Case 3: Partial edit - requires original_code
    logger.info(f"Attempting partial edit of {filepath} .")

    # case 3.1: Original code missing -> fail
    if code_action.content.original_code is None:
        logger.warning(
            f"During handling of code response part action, encountered a code response part with type {code_action.content.type} that has a null original code. Cannot perform partial edit. (case 3.1)"
        )
        logger.debug(
            f"Dumping bizarre code action:\n{json.dumps(code_action.model_dump(), indent=4)}"
        )
        return fail("Cannot perform partial edit with missing original code block.")

    # Invariant: filepath exists, original_code is not None, and it's not a full file replacement.
    # Now, try to locate the original_code within file_contents.

    original_code_block = code_action.content.original_code
    new_code_block = code_action.content.new_code

    # Attempt exact substring match first (most efficient if it works)
    start_char_pos = file_contents.find(original_code_block)

    # case 3.2: Exact substring match of original_code found.
    if start_char_pos != -1:
        end_char_pos = start_char_pos + len(original_code_block)
        logger.info(
            f"Exact substring match found for original code in {filepath}. (case 3.2)"
        )
        return types.ActionResultMoreActions(
            actions=[
                types.ActionFileEdit(
                    filepath=filepath,
                    replace_pos_begin=start_char_pos,
                    replace_pos_end=end_char_pos,
                    insert_text=new_code_block,
                )
            ]
        )

    # If exact match fails, proceed with line-by-line matching using Levenshtein distance
    logger.info(
        f"Exact substring match failed for original code in {filepath}. Attempting line-by-line matching."
    )

    original_lines = original_code_block.splitlines()
    file_lines = file_contents.splitlines()

    stripped_lines = [
        stripped_line
        for line in original_lines
        if (stripped_line := line.strip()) != ""
    ]

    # case 3.3: Original_code had only whitespace -> fail
    if not (stripped_lines):
        logger.warning(
            f"Could not locate original code block in file '{filepath}': Original code appears to be empty or only whitespace. (case 3.3)"
        )
        return fail(
            "Original code block is empty or contains only whitespace after splitting. Cannot perform replacement."
        )

    best_match_start_line = -1
    min_total_distance = float("inf")

    # Iterate through file_lines to find a matching block
    for i in range(len(file_lines) - len(original_lines) + 1):
        current_block_lines = file_lines[i : i + len(original_lines)]

        total_distance = 0
        for j in range(len(original_lines)):
            # No stripping of whitespace for Levenshtein comparison
            norm_orig_line = original_lines[j]
            norm_curr_line = current_block_lines[j]
            total_distance += levenshtein(norm_orig_line, norm_curr_line)

        if total_distance < min_total_distance:
            min_total_distance = total_distance
            best_match_start_line = i

    # Define a threshold for what constitutes a "good enough" match
    # Using the raw length of the original code block for the threshold calculation.
    raw_original_code_len = len(original_code_block)
    distance_threshold = max(
        5, raw_original_code_len // 5
    )  # At least 5, or 20% of raw length

    if best_match_start_line != -1 and min_total_distance <= distance_threshold:
        # Calculate character positions for the best match
        replace_pos_begin = 0
        for k in range(best_match_start_line):
            replace_pos_begin += len(file_lines[k]) + 1  # +1 for newline

        replace_pos_end = replace_pos_begin
        for k in range(
            best_match_start_line, best_match_start_line + len(original_lines)
        ):
            replace_pos_end += len(file_lines[k]) + 1  # +1 for newline

        # Adjust for the very last newline if the file doesn't end with one
        if not file_contents.endswith("\n") and (
            best_match_start_line + len(original_lines) == len(file_lines)
        ):
            replace_pos_end -= 1

        # case 3.4: Found levenshtein match
        logger.info(
            f"Line-by-line match found for original code in {filepath} (distance: {min_total_distance}). (case 3.4)"
        )
        return types.ActionResultMoreActions(
            actions=[
                types.ActionFileEdit(
                    filepath=filepath,
                    replace_pos_begin=replace_pos_begin,
                    replace_pos_end=replace_pos_end,
                    insert_text=new_code_block,
                )
            ]
        )
    else:
        # case 3.5: No good match found. Return a failure.
        logger.warning(
            f"Could not find a sufficiently close match for original code in {filepath} (min distance: {min_total_distance}, threshold: {distance_threshold}). (case 3.5)"
        )
        return fail(
            f"Could not locate original code block in file '{filepath}' for partial edit. "
            f"Minimum Levenshtein distance found: {min_total_distance} (threshold: {distance_threshold}). "
            f"Original code block:\n---\n{original_code_block}\n---"
        )


def worker_route_request(prog: Program, prompt: str) -> types.AIAgent:
    """Classify a request as belonging to worker or coder."""
    with ProgressPrinter(
        message=f"{prog.last_print_text} Routing 🔧 ",
        postfix_message=f"{prog.last_print_text}",
        print_function=prog.make_tagged_printer("action_queue"),
    ):
        try:
            result = prog.worker_box.new(
                types.UserPromptClassification,
                prompts.make_prompt_route_request(prog, prompt),
            )
            logger.info(f"Prompt classified as '{result.classification}'. Reason: {result.reason}")
            return result.classification
        except Exception as e:
            logger.exception(
                f"Unhandled exception while routing request. Defaulting to ghostcoder. Reason: {e}"
            )
            # this is not fatal, we just default to the backend, since this routing request probably went to the worker
            prog.cosmetic_state = types.CosmeticProgramState.PROBLEM
    return types.AIAgent.CODER


def worker_query(
    prog: Program, query_worker_action: types.ActionQueryWorker
) -> types.ActionResult:
    """Send a query to the worker backend."""
    print_function = mock_print if query_worker_action.hidden else prog.print

    try:
        with ProgressPrinter(
            message=f"{prog.last_print_text} Querying 🔧 ",
            postfix_message=f"{prog.last_print_text}",
            print_function=prog.make_tagged_printer("action_queue"),
        ):
            profile = query_worker_action.llm_response_profile
            # FIXME: right now we are ignoring the profile and always generating actions.
            prog.worker_box.clear_history()
            worker_response = prog.worker_box.new(
                types.WorkerResponse,
                prompts.make_prompt_worker_query(
                    prog, query_worker_action.interaction_history_id
                ),
            )
            logger.debug(
                f"Dump of worker query response: \n{show_model(worker_response)}"
            )
            if not(worker_response.contents):
                logger.warning(f"Empty response from worker query.")
            actions = actions_from_response_parts(prog, worker_response.contents, original_action=query_worker_action)
            return types.ActionResultMoreActions(actions=actions)
    except:
        pass
    # placeholder
    return types.ActionResultOk()


def worker_wait_on_shell_command(
    prog: Program, wait_action: types.ActionWaitOnShellCommand
) -> types.ActionResult:
    """Waits for a shell command to finish or cancels it if it takes too long."""
    # who knew LLMs could solve the halting problem with just vibes
    if prog.project is None:
        min_t, max_t = float("-inf"), float("inf")
    else:
        min_t, max_t = (
            _min_t
            if (_min_t := prog.project.config.shell_wait_time_minimum) is not None
            else float("-inf")
        ), (
            _max_t
            if (_max_t := prog.project.config.shell_wait_time_maximum) is not None
            else float("inf")
        )

    start_t = time.perf_counter()
    additional_wait_time = 0.0
    if min_t:
        logger.debug(f"Sleeping for {min_t} seconds.")
        time.sleep(min_t)

    with ProgressPrinter(
        message=f"{prog.last_print_text} Querying Executing ",
        postfix_message=f"{prog.last_print_text}",
        print_function=prog.make_tagged_printer("action_queue"),
    ):
        while (time_elapsed := time.perf_counter() - start_t) < max_t:
            while additional_wait_time > 0.0:
                time.sleep(1.0)
                additional_wait_time -= 1.0

            try:
                prog.worker_box.clear_history()
                response = prog.worker_box.new(
                    types.ShellWaitResponse,
                    prompts.make_prompt_worker_wait_shell(prog, time_elapsed),
                )
            except Exception as e:
                logger.exception(
                    f"Caught exception while waiting on shell command. Reason: {e}"
                )
                return types.ActionResultFailure(
                    original_action=wait_action, failure_reason=f"{e}"
                )

            # we have a response
            match response.content:
                case types.ShellWaitResponsePartFinished() as finished_part:
                    logger.info(
                        f"Worker has determined that shell command finished. Reason: {finished_part.reason}"
                    )
                    # FIXME: maybe we should return MoreActions here?
                    return types.ActionResultOk(
                        success_message=f"Command has finished executing. Reason: {finished_part.reason}"
                    )
                case types.ShellWaitResponsePartKeepWaiting() as keep_waiting_part:
                    additional_wait_time = min(
                        keep_waiting_part.time_to_wait_seconds, max_t
                    )
                    logger.info(
                        f"Worker decided to wait an additional {additional_wait_time} seconds. Reason: {keep_waiting_part.reason}"
                    )
                    # loop around
                case types.ShellWaitResponsePartKillProcess() as kill_process_part:
                    logger.info(
                        f"Worker decided to kill shell process. Reason: {kill_process_part.reason}"
                    )
                    prog.tty.kill()
                    return types.ActionResultFailure(
                        original_action=wait_action,
                        failure_reason=f"Worker killed process. Reason: {kill_process_part.reason}",
                    )

    # end of while loop -> timed out
    return types.ActionResultFailure(
        original_action=wait_action,
        failure_reason=f"The shell command timed out after {time_elapsed:2f} seconds by exceeding the maximum wait time of {max_t}.",
    )


def worker_recover(
    prog: Program, failure: types.ActionResultFailure
) -> types.ActionResultMoreActions:
    """Recover after a failed action on the action queue.
    This feeds the last failure result and a report of the general situation to the worker LLM, with the goal of generating one or more actions that lead to success.
    The exact method of recovery is intentionally unspecific as it is left to the LLM to decide on a recovery plan.
    Example: An ActionFileEdit failed because the original code block was insufficient to find a replace. The worker then generates an action that requests clarification from the coder LLM, possibly generating a more specific and precise original code block.
    """
    # we might error during LLM query
    try:
        logger.info("Querying ghostworker for recovery.")
        prog.worker_box.clear_history()
        with ProgressPrinter(
            message=f"{prog.last_print_text} Recovering 🔧 ",
            postfix_message=f"{prog.last_print_text}",
            print_function=prog.make_tagged_printer("action_queue"),
        ):
            worker_response = prog.worker_box.new(
                types.WorkerResponse,
                prompts.make_prompt_worker_recover(prog, failure),
            )
        logger.timing(f"ghostworker performance statistics:\n{showTime(prog.worker_box._plumbing, [])}")  # type: ignore

        # FIXME: should we add response to the interaction_history?
        actions = actions_from_response_parts(prog, worker_response.contents, original_action=failure.original_action)

        # we could just return a list of actions
        # but using the types.MoreActions wrapper we can keep a semantic grouping for actions
        # which is structure that we would otherwise lose
        return types.ActionResultMoreActions(actions=actions)

    except Exception as e:
        logger.exception(f"Ghostworker Failed  recovery. Reason: {e}")
        logger.debug(
            f"Dump of original failure:\n{json.dumps(failure.model_dump(), indent=4)}"
        )
        return types.ActionResultMoreActions(
            actions=[
                types.ActionHaltExecution(
                    reason=f"During recovery from a failed action, an error was encountered: {e}.\n\nOriginal failure reason: {failure.failure_reason}.",
                    announce=True,
                )
            ]
        )

    # this should be unreachable but you never know and also mypy complains
    return types.ActionResultMoreActions(
        actions=[
            types.ActionHaltExecution(
                reason="Recovery from failed action was inconclusive. This is weird and you should never actually be reading this message. Original failure reason: {failure.failure_reason}"
            )
        ]
    )


def worker_generate_title(prog: Program, interaction: types.InteractionHistory, headless: bool = True) -> str:
    """Query the worker to generate a title for a given interaction.
    This works best if the interaction has some content.
    It can be repeatedly called on the same interaction for different titles."""
    logger.info(f"Generating title for interaction {interaction.unique_id}.")
    try:
        with ProgressPrinter(
            message=f"{prog.last_print_text} Querying 🔧 ",
            postfix_message=f"{prog.last_print_text}",
            print_function=prog.make_tagged_printer("action_queue"),
            disabled=headless,
        ):
            prog.worker_box.clear_history()
            return prog.worker_box.text(
                prompts.make_prompt_interaction_title(interaction)
            )
    except Exception as e:
        logger.exception(
            f"Uncaught exception while trying to generate title for interaction {interaction.unique_id}. Reason: {e}"
        )
    # we are prepared to deal with empty string titles
    return ""


def prepare_request(
        prog: Program,
        prepare_request_action: types.ActionPrepareRequest
) -> types.ActionResult:
    """Prepares a prompt by e.g. adding or removing files from context."""
    # by default, we just return a coder query
    default_result = types.ActionResultMoreActions(
        actions = [
            types.ActionQueryCoder(**(prepare_request_action.model_dump()))
        ]
    )
    
    # we collect stuff here, we will prepend this to the queue at the end
    prepare_actions: List[types.Action] = []
    # go through things that we might do to prepare
    with ProgressPrinter(
        message=f"{prog.last_print_text} preparing request 🔧 ",
        postfix_message=f"{prog.last_print_text}",
        print_function=prog.make_tagged_printer("action_queue"),
    ):
        # 1. see if we might be over the token threshold
        # this is not meant to be exact, so don't get scared that we're kind of hacking it
        # also the order doesn't matter for tokens
        preamble_str = prog.coder_box.get_var("preamble_injection")
        history_str = "\n".join([msg.model_dump_json() for msg in prog.coder_box.get_history()])
        tokens = prog.coder_box.token_estimate(preamble_str + history_str + prepare_request_action.prompt)
        if tokens is not None:
            logger.info(f"Request is estimated to cost ~{tokens} tokens.")
        else:
            logger.warning(f"Failed to estimate token cost of request.")

        # if it's not above the threshold we can move on
        if tokens >= prog.user_config.token_threshold:
            prepare_actions.extend(reduce_token_cost(prog, prepare_request_action))

        # prepend the collected actions
        default_result.actions = prepare_actions + default_result.actions
        return default_result
    
def reduce_token_cost(
        prog: Program, prepare_request_action: types.ActionPrepareRequest
) -> List[types.Action]:
    """Returns a list of actions to be taken in order to reduce the token cost of a given request."""
    return []

def generate_context_file_summary(
        prog: Program, context_file: types.ContextFile, headless: bool = False
        ) -> Optional[types.ContextFileSummary]:
    """Generates the summary for a given context file."""
    with ProgressPrinter(
        message=f"{prog.last_print_text} Summarizing file ",
        postfix_message=f"{prog.last_print_text}",
        print_function=prog.make_tagged_printer("action_queue"),
        disabled=headless
    ):
        try:
            logger.debug(f"Attempting to summarize context file {context_file.filepath}.")
            if (prompt :=                 prompts.make_prompt_context_file_summary(prog, context_file)) is None:
                # couldn't readfile or smth
                return None
            prog.worker_box.clear_history()
            return prog.worker_box.new(
                types.ContextFileSummary,
                prompt,
                )
        except Exception as e:
            logger.exception(f"Couldn't generate summary for context file {context_file.filepath}. Reason: {e}")
            return None

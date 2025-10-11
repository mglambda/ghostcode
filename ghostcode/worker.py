# ghostcode/worker.py
from typing import *
import traceback
import os
import shutil
import json
import logging
from ghostcode import types
from ghostcode.types import Program
from ghostcode.utility import levenshtein, time_function_with_logging
import tempfile

# --- Logging Setup ---
logger = logging.getLogger('ghostcode.worker')

def actions_from_response_parts(prog: Program, parts: List[types.CoderResponsePart]) -> List[types.Action]:
    """Transform a list of response parts from the coder LLM into a list of actions.
    This is not a 1 to 1 mapping. You may end up with an empty list if e.g. all the response parts are just discussion text. Only code parts and similar are transformed into actions."""
    # atm we have prog only because we might need more context here in the future
    logger.debug(f"actions_from_response_parts with {len(parts)} parts: {[parts.__class__.__name__ for part in parts]}")
    
    def action_from_part(part: types.CoderResponsePart) -> Optional[types.Action]:
        match part:
            case types.TextResponsePart() as text_part:
                # no action necessary
                return None
            case types.CodeResponsePart() as code_part:
                return types.ActionHandleCodeResponsePart(
                    content=code_part
                )
            case _:
                return None
            
    return [action
            for part in parts
            if (action := action_from_part(part))]

@time_function_with_logging
def run_action_queue(prog: Program) -> None:
    """Executes and removes actions at the front of the action queue until the queue is empty or the halt action is popped.
    This function has no return value as actions primarily exist to crate side effects."""
    logger.info("Action queue execution started with {len(prog.action_queue)} actions in queue.")

    # each time we run a queue, we reset the flag that let's the user skip confirmation dialogs
    # the user sets this to true with the "a" confirmation option.
    prog.action_queue_yolo = False
    
    if not(prog.action_queue):
        logger.info(f"Action queue execution stopped because there are no actions on the queue.")
        return

    try:
        finished_actions = 0
        while action := prog.action_queue.pop(0):
            logger.debug(f"Current Queue Status\n  popped action: {action.__class__.__name__}\n  queue: {[a.__class__.__name__ for a in prog.action_queue]}")
            match action:
                case types.ActionHaltExecution() as halt_action:
                    logger.info(f"Halting action queue execution after {finished_actions} actions, with {len(prog.action_queue)} actions remaining in queue. Reason: {halt_action.reason}")
                    prog.discard_actions()
                    return
                case _:
                    logger.info(f"Executing {type(action)}. Finished {finished_actions}, remaining {len(prog.action_queue)}.")
                    logger.debug(f"More info on action being executed:\n{json.dumps(action.model_dump(), indent=4)}")
                    result = execute_action(prog, action)
                    finished_actions += 1
                    match result:
                        case types.ActionResultOk() as result_ok:
                            logger.info(f"Action execution successful." + f"Message: {result_ok.success_message}" if result_ok.success_message else "")
                        case types.ActionResultFailure() as ar_failure:
                            logger.info(f"Failed to execute action {type(ar_failure.original_action)}. Reason: {ar_failure.failure_reason}")
                            logger.debug(f"Original action dump:\n{json.dumps(ar_failure.original_action.model_dump(), indent=4)}\n\nFailure result dump:\n{json.dumps(ar_failure.model_dump(), indent=4)}")
                            # FIXME: here we probably want to invoke the worker to look at the queue and the logs and decide on what to do next (halt/more actions). Right now we just push a halt.
                            prog.push_front_action(
                                types.ActionHaltExecution(reason="Encountered failed action.")
                            )
                        case types.ActionResultMoreActions() as ar_more:
                            logger.info(f"Got MoreAction result for {len(ar_more.actions)} actions. Action queue has finished {finished_actions}, remaining {len(prog.action_queue)}.")
                            for new_action in ar_more.actions:
                                prog.push_front_action(new_action)
                        case _:
                            logger.info(f"Got unknown action result. Weird, but ok.")
                            logger.debug(f"Dumping unknown action result:\n{json.dumps(result.model_dump(), indent=4)}")
    except IndexError:
        logger.info(f"Action queue execution finished after {finished_actions} actions.")
        return

def execute_action(prog: Program, action: types.Action) -> types.ActionResult:
    # small helper for this context
    fail = lambda failure_reason, error_messages=[], action=action: types.ActionResultFailure(original_action=action,
    error_messages=error_messages,
    failure_reason=failure_reason)
    
    logger.debug(f"Executing action: {action.__class__.__name__}")
    
    # check clearance
    clearance_requirement = action.clearance_required
    if prog.project is None:
        logger.warning(f"Null project while trying to get worker clearance, so no config is available. Defaulting to lowest clearance.")
        clearance_level = types.ClearanceRequirement.AUTOMATIC
    else:
        clearance_level = prog.project.config.worker_clearance_level
    if clearance_requirement == types.ClearanceRequirement.FORBIDDEN:
        logger.info(f"Worker tried to execute action with forbidden clearance. This is never permitted, regardless of clearance level. Failing action.")
        return fail("Worker encountered action with 'forbidden' clearance requirement.")
    elif clearance_level.value < clearance_requirement.value:
        logger.info("Worker clearance of {clearance_level} insufficient for clearance requirement {clearance_requirement} of {action.__class__.__name__} action. Seeking user confirmation.")
        user_response = prog.confirm_action(action, agent_clearance_level=clearance_level, agent_name="Ghostworker")
        if user_response == types.UserConfirmation.CANCEL:
            # we just exit regularly through the halting mechanism
            return types.ActionResultMoreActions(
                actions=[types.ActionHaltExecution(
                    reason="User canceled action execution."
                )]
            )
        elif types.UserConfirmation.is_confirmation(user_response):
            if user_response == types.UserConfirmation.ALL:
                prog.action_queue_yolo = True
            logger.info(f"User confirmed {action.__class__.__name__} for worker with clearance level {clearance_level}.")
        else:
            logger.info(f"User denied {action.__class__.__name__} for worker with clearance level {clearance_level}.")
            return fail(f"Action {action.__class__.__name__} denied by user.")
    else:
        logger.info("Worker clearance of {clearance_level} meets or exceeds clearance requirement {clearance_requirement} for {action.__class__.__name__} action. Proceeding without user confirmation.")
        
    try:
        match action:
            case types.ActionHandleCodeResponsePart() as code_action:
                logger.info(f"Handling code response part.")
                logger.debug(f"Code response part action dump:\n{json.dumps(code_action.model_dump(), indent=4)}") 
                return handle_code_part(prog, code_action)
            case types.ActionFileEdit() as edit_action:
                logger.info(f"File Edit Action")
                return apply_edit_file(prog, edit_action)
            case types.ActionCreateFile() as create_action:
                logger.info("Create File Action")
                return apply_create_file(prog, create_action)
            case types.ActionDoNothing():
                # No operation needed for ActionDoNothing
                logger.info("Action: Do Nothing")
                return types.ActionResultOk()
            case _:
                # Handle unknown action types
                failure_message = f"Received an unknown action type: {type(action).__name__}. This action cannot be executed."
                logger.error(failure_message)
                print(f"ERROR: {failure_message}")
                return types.ActionResultFailure(
                    original_action=action,
                    failure_reason=failure_message
                )
    except Exception as e:
        logger.error(f"Caught exception in execute_action: {e}. Full traceback: \n{traceback.format_exc()}")
        return types.ActionResultFailure(
            original_action=action,
            failure_reason=f"Caught exception: {e}"
        )


        
def apply_create_file(prog: Program, create_action: types.ActionFileCreate) -> types.ActionResult:        
    """Apply a ActionFileCreate to actually write a new file to disk."""
    fail = lambda failure_reason, error_messages=[], create_action=create_action: types.ActionResultFailure(
        original_action=create_action,
        error_messages=error_messages,
        failure_reason=failure_reason
    )
    
    try:
        os.makedirs(create_action.filepath, exist_ok=True)        
        with open(create_action.filepath, "w") as f:
            f.write(create_action.content)
    except Exception as e:
        error_msg = f"Couldn't write new file {create_action.filepath}. Reason: {e}"
        logger.error(error_msg)
        logger.debug(f"Edit action dump:\n{json.dumps(create_action.model_dump(), indent=4)}")
        return fail(error_msg,
                    error_messages=[traceback.format_exc()]
                    )

    return types.ActionResultOk(success_message=f"Wrote new file {create_action.filepath}")

def apply_edit_file(prog: Program, edit_action: types.ActionFileEdit) -> types.ActionResult:
    """Apply an ActionEditFile action and replace some lines in a file with new contents."""
    filepath = edit_action.filepath
    if not(os.path.isfile(filepath)):
        error_msg = f"Tried to apply edit action but file {filepath} does not exist."
        logger.error(error_msg)
        return types.ActionResultFailure(
            original_action=edit_action,
            failure_reason=error_msg
        )


    if os.path.isdir(filepath):
        error_msg = f"Trying to apply edit action to file {filepath} but file is a directory."
        logger.error(error_msg)
        return types.ActionResultFailure(
            original_action=edit_action,
            failure_reason=error_msg
        )

    # ok the way we do this is to read it all into memory, replace, write a temp file, and then atomically swap the files
    # this avoids file erasure in the unlikely event that we crash between truncation and writing of the target file
    try:
        with open(filepath, "r") as f:
            content = f.read()

        new_content = content[0:edit_action.replace_pos_begin] + edit_action.insert_text + content[edit_action.replace_pos_end:]
        fd, temp_filepath = tempfile.mkstemp()
        os.close(fd)

        with open(temp_filepath, "w") as tf:
            tf.write(new_content)

        # atomic swap
        shutil.move(temp_filepath, filepath)

    except Exception as e:
        error_msg = f"Could not edit  file while applying edit action. Reason: {e}"
        logger.error(error_msg)
        logger.debug(f"Dump of action:\n{json.dumps(edit_action.model_dump(), indent=4)}")
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        return types.ActionResultFailure(
            original_action=edit_action,
            failure_reason=error_msg,
            error_messages=[traceback.format_exc()]
        )
    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

    return types.ActionResultOk(success_message=f"Applied edit action to file {filepath} for {len(edit_action.insert_text)} characters.")


@time_function_with_logging
def handle_code_part(prog: Program, code_action: types.ActionHandleCodeResponsePart) -> types.ActionResult:
    """Executes an ActionHandleCodeResponsePart action."""
    # small helper for this context
    fail = lambda failure_reason, error_messages=[], code_action=code_action: types.ActionResultFailure(original_action=code_action,
    error_messages=error_messages,
    failure_reason=failure_reason)

    # first rule out some easy cases
    if code_action.content.filepath is None:
        logger.info("Handling code part with null filepath.")
        return types.ActionResultOk(
            success_message="Nothing to be done for this response, due to missing/none filepath."
        )
    
    filepath = code_action.content.filepath

    # Case 1: File does not exist -> Create file
    if not os.path.exists(filepath):
        logger.info(f"Null filepath on code response part. Moving to create a new file. (case 1)")
        logger.debug(f"handle_code_part: Attempting to create a new file. ({filepath})")
        if os.path.isdir(filepath):
            return fail(f"File {filepath} is a directory.")
        
        # If original_code is provided for a new file, it's a bit odd, but we'll prioritize new_code
        if code_action.content.original_code is not None:
            logger.warning(f"Code part has no filepath, but original code is non-null.")
            logger.debug(f"Code part with title '{code_action.content.title}', original_code (abridged): {code_action.content.original_code[:50]}")
            
        # The type "full" is most appropriate for new file creation.
        if code_action.content.type != "full":
            logger.warning(f"Code part has no filepath set, but type is '{code_action.content.type}'.")
            
        return types.ActionResultMoreActions(
            actions=[types.ActionFileCreate(
                filepath=filepath,
                content=code_action.content.new_code
            )]
        )

    # invariant: File exists. Read its contents.
    try:
        with open(filepath, "r") as f:
            file_contents = f.read()
    except Exception as e:
        error_msg = f"Could not read file {filepath} during handling of code response action."
        logger.error(error_msg)
        logger.debug(f"Handle code part action dump:\n{json.dumps(code_action.model_dump(), indent=4)}")
        return fail(error_msg,
                    error_messages=[traceback.format_exc()])

    # Case 2: Full replacement (if original_code matches entire file)
    if code_action.content.original_code is not None and file_contents == code_action.content.original_code:
        logger.info(f"Identical content for original code and entire file {filepath}. Performing full replacement. (case 2)")
        return types.ActionResultMoreActions(
            actions=[types.ActionFileEdit(
                filepath=filepath,
                replace_pos_begin=0,
                replace_pos_end=len(file_contents),
                insert_text=code_action.content.new_code
            )]
        )

    # Case 3: Partial edit - requires original_code
    logger.info("Attempting partial edit of {filepath} .")

    # case 3.1: Original code missing -> fail
    if code_action.content.original_code is None:
        logger.warning(f"During handling of code response part action, encountered a code response part with type {code_action.content.type} that has a null original code. Cannot perform partial edit. (case 3.1)")
        logger.debug(f"Dumping bizarre code action:\n{json.dumps(code_action.model_dump(), indent=4)}")
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
        logger.info(f"Exact substring match found for original code in {filepath}. (case 3.2)")
        return types.ActionResultMoreActions(
            actions=[types.ActionFileEdit(
                filepath=filepath,
                replace_pos_begin=start_char_pos,
                replace_pos_end=end_char_pos,
                insert_text=new_code_block
            )]
        )

    # If exact match fails, proceed with line-by-line matching using Levenshtein distance
    logger.info(f"Exact substring match failed for original code in {filepath}. Attempting line-by-line matching.")
    
    original_lines = original_code_block.splitlines()
    file_lines = file_contents.splitlines()

    # case 3.3: Original_code had only whitespace -> fail
    if not original_lines:
        logger.warning(f"Could not locate original code block in file {filepath}: Original code appears to be just whitespace. (case 3.3)")
        return fail("Original code block is empty after splitting into lines. Cannot perform replacement.")

    best_match_start_line = -1
    min_total_distance = float('inf')
    
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
    distance_threshold = max(5, raw_original_code_len // 5) # At least 5, or 20% of raw length

    if best_match_start_line != -1 and min_total_distance <= distance_threshold:
        # Calculate character positions for the best match
        replace_pos_begin = 0
        for k in range(best_match_start_line):
            replace_pos_begin += len(file_lines[k]) + 1 # +1 for newline

        replace_pos_end = replace_pos_begin
        for k in range(best_match_start_line, best_match_start_line + len(original_lines)):
            replace_pos_end += len(file_lines[k]) + 1 # +1 for newline
        
        # Adjust for the very last newline if the file doesn't end with one
        if not file_contents.endswith('\n') and (best_match_start_line + len(original_lines) == len(file_lines)):
            replace_pos_end -= 1

        # case 3.4: Found levenshtein match
        logger.info(f"Line-by-line match found for original code in {filepath} (distance: {min_total_distance}). (case 3.4)")
        return types.ActionResultMoreActions(
            actions=[types.ActionFileEdit(
                filepath=filepath,
                replace_pos_begin=replace_pos_begin,
                replace_pos_end=replace_pos_end,
                insert_text=new_code_block
            )]
        )
    else:
        # case 3.5: No good match found. Return a failure.
        logger.warning(f"Could not find a sufficiently close match for original code in {filepath} (min distance: {min_total_distance}, threshold: {distance_threshold}). (case 3.5)")
        return fail(f"Could not locate original code block in file '{filepath}' for partial edit. "
                    f"Minimum Levenshtein distance found: {min_total_distance}. "
                    f"Original code block:\n---\n{original_code_block}\n---")


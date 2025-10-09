# ghostcode/worker.py
from typing import *
import traceback
import os
import json
import logging
from ghostcode import types
from ghostcode.types import Program
from ghostcode.utility import levenshtein

# --- Logging Setup ---
logger = logging.getLogger('ghostcode.worker')

def run_action_queue(prog: Program) -> None:
    """Executes and removes actions at the front of the action queue until the queue is empty or the halt action is popped.
    This function has no return value as actions primarily exist to crate side effects."""
    logger.info("Action queue execution started with {len(prog.action_queue)} actions in queue.")
    if not(prog.action_queue):
        logger.info(f"Action queue execution stopped because there are no actions on the queue.")
        return

    try:

        finished_actions = 0
        while action := prog.action_queue.pop(0):
            match action:
                case types.ActionHaltExecution(reason):
                    logger.info(f"Halting action queue execution after {finished_actions} actions, with {len(prog.action_queue)} actions remaining in queue. Reason: {reason}")
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
    logger.debug(f"Executing action: {action.__class__.__name__}")
    try:
        match action:
            case types.ActionHandleCodeResponsePart() as code_action:
                logger.info(f"Handling code response part.")
                logger.debug(f"Code response part action dump:\n{json.dumps(code_action.model_dump(), indent=4)}") 
                return handle_code_part(prog, code_action)
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
    # FIXME: fill this in
    return types.ActionResultOk()

    
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
    
    if not(os.path.isfile(code_action.content.filepath)):
        # file doesn't exist yet. we will create a new file if possible
        logger.debug("handle_code_part: Attempting to create a new file. ({code_action.content.filepath})")
        if os.path.isdir(code_action.content.filepath):
            return fail(f"File {code_action.content.filepath} is a directory.")

        if code_action.content.type != "full":
            logger.warning(f"Possibly going to create new file during code response handling but type is '{code_action.content.type}'. Pretty sus.")
        
        if code_action.content.original_code is not None:
            logger.warning(f"Got non-null original_code while handling code response for possibly new file. This is a bit weird.")

            return types.ActionResultMoreActions(
                actions=[types.ActionFileCreate(
                    filepath=code_action.content.filepath,
                    content=code_action.content.new_code
                )]
            )
                
    # ok, here we know the filepath already exists and is a file
    if code_action.content.type != "partial":
        logger.warning(f"File already exists while handling code response, but type is {code_action.content.type}. That's a bit weird.")
        # FIXME: in the future, do additional checks here, like if original_code is just equal or close to entire file contents, and if not, we maybe abort or request confirmation from user or coder LLM

    # FIXME: this is just here for mypy
    return types.ActionResultOk()

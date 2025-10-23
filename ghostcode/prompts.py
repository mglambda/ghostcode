from typing import *
from pydantic import BaseModel, Field
from ghostcode import types
from ghostcode.types import Program
from ghostcode.utility import quoted_if_nonempty, show_model, timestamp_now_iso8601
import logging

# --- Logging Setup ---
logger = logging.getLogger("ghostcode.prompts")


class PromptConfig(BaseModel):
    """Small helper type to package prompt selection parameters together.
    Essentially you can set flags in this type to customize a prompt. Some attributes require a choice (e.g. history), in which case you must provide a literal.
    By default, the most verbose options are chosen, e.g. all flags enabled and full history shown.
    """

    system_text: str = Field(
        default="",
        description="Text that is printed at the top of the prompt under a '# System' heading. If empty, no system heading is printed. This is not the same as the system prompt, which can't be modified with this config.",
    )

    user_prompt_text: str = Field(
        default="",
        description="If nonempty, will be shown at the very bottom of the prompt with a 'User Prompt' heading. This will give it special emphasis for the LLM. May be overwritten or adjusted by other PromptConfig options.",
    )

    # project context
    project_metadata: bool = True
    context_files: Literal["full", "filenames", "none"] = Field(
        default="full",
        description="How to display files that are in the ghostcode context. Full means full contents of the files are displayed. filenames means only a list of filenames is shown. none omits context files entirely.",
    )

    # ghostcode program context
    interaction_history_id: str = Field(
        default="",
        description="The unique ID of the interaction history you want to display in the prompt. This must be provided if you set interaction_history to anything other than 'none'.",
    )

    interaction_history: Literal["full", "split", "none"] = Field(
        default="full",
        description="How to display the history. Requires interaction_history_id to be provided. full means entire history is sent. split means the entire history except the last message is sent, and the last message is appended to the user prompt. none means no history is sent.",
    )

    shell: bool = True
    logs: bool = True

    @staticmethod
    def maximal(**kwargs: Any) -> "PromptConfig":
        """Create a default PromptConfig with maximum erbosity."""
        return PromptConfig(**kwargs)

    @staticmethod
    def minimal(**kwargs: Any) -> "PromptConfig":
        """Create a PromptConfig with minimum verbosity."""
        default_config_data = PromptConfig().model_dump()
        min_config_data: Dict[str, Any] = {}
        for k, v in default_config_data.items():
            # we can rely on certain conventions, e.g. literal fields will be "none" for minimum verbosity
            match v:
                case str(w):
                    if get_origin(PromptConfig.model_fields[k].annotation) is Literal:
                        min_config_data[k] = "none"
                    else:
                        # regular string like system_text
                        min_config_data[k] = ""
                case bool(b):
                    min_config_data[k] = False

        # the above is safe thanks to pydantic having our backs
        return PromptConfig(**(min_config_data | kwargs))


def make_prompt(
    prog: Program,
    prompt_config: PromptConfig,
) -> str:
    """Most general way to create a prompt for an LLM. See the PromptConfig docstring for more."""
    config = prog.get_config_or_default()

    # all vars ending with _str are the final value being spliced into the prompt at the bottom
    system_str = quoted_if_nonempty(
        text=prompt_config.system_text, heading="System", heading_level=1
    )

    # this one is the user prompt which, if nonempty, will be displayed at the very very bottom for the LLM
    user_prompt_str = quoted_if_nonempty(
        text=prompt_config.user_prompt_text, heading="User Prompt", heading_level=1
    )

    if prompt_config.interaction_history_id:
        match prompt_config.interaction_history:
            case "split":
                history_items, user_prompt_str = make_blocks_interaction_history_split(
                    prog, prompt_config.interaction_history_id
                )
                history_str = quoted_if_nonempty(
                    text=history_items, heading="Interaction History"
                )
            case "full":
                history_items, _ = make_blocks_interaction_history(
                    prog, prompt_config.interaction_history_id
                )
                history_str = quoted_if_nonempty(
                    text=history_items, heading="Interaction History"
                )
            case "none":
                history_str = ""
            case _ as unreachable1:
                assert_never(unreachable1)
    else:
        if prompt_config.interaction_history != "none":
            logger.warning(
                f"Trying to build prompt with interaction_history set to '{prompt_config.interaction_history}, but no interaction history ID provided. Defaulting to 'none' instead."
            )
        history_str = ""

    if prompt_config.project_metadata:
        project_metadata_str = quoted_if_nonempty(
            text=make_blocks_project_metadata(prog), heading="Project Metadata"
        )
    else:
        project_metadata = ""

    match prompt_config.context_files:
        case "full":
            # this one needs no quoted because the individual files are themselves quoted
            context_files_str = make_blocks_context_files_full(
                prog, heading="Files in Context"
            )
        case "filenames":
            context_files_str = quoted_if_nonempty(
                text=make_blocks_context_files_short(prog), heading="Files in Context"
            )
        case "none":
            context_files_str = ""
        case _ as unreachable2:
            assert_never(unreachable2)

    if prompt_config.logs:
        log_excerpt_str = quoted_if_nonempty(
            text=prog.show_log(tail=config.prompt_log_lines),
            heading=f"Program Logs (most recent {config.prompt_log_lines} lines)",
        )
    else:
        log_excerpt_str = ""

    if prompt_config.shell:
        shell_str = quoted_if_nonempty(
            text=make_blocks_shell(prog), heading="Virtual Terminal", type_tag="json"
        )
    else:
        shell_str = ""

    return f"""{system_str}
    # Project Context

{project_metadata_str}{context_files_str}    
# Ghostcode Context

{history_str}    {shell_str}{log_excerpt_str}
{user_prompt_str}    

"""


def make_prompt_worker_recover(
    prog: Program, failure: types.ActionResultFailure
) -> str:
    """Creates a prompt that can be sent to a worker LLM backend based on the current program context and a given failure result with the intention to recover from the failure."""

    if prog.project is None:
        msg = f"Null project while trying to construct a worker prompt. Please first intialize a project with `ghostcode init`."
        logger.critical(msg)
        raise RuntimeError(msg)

    # the question here is what to include and how to phrase the prompt.
    # still very much WIP

    prompt = ""

    # metadata
    if prog.project.project_metadata is not None:
        prompt += prog.project.project_metadata.show()

    # context files (probably the biggest chunk)
    prompt += prog.project.context_files.show()

    # interaction history
    # FIXME: where to get this from: possibilities are 1. file (requires I/O), 2. prog (requires us to add a new attribute like _current_interaction_history), 3. use ghostbox history, which is automatically saved
    # Currently we pick (3), since its easiest.
    history_strs = [
        show_model(msg) for msg in prog.coder_box.get_history() if msg.role != "system"
    ]
    prompt += f"""## Interaction History

```yaml
{"---".join(history_strs)}
```

"""

    # logs
    if (log_str := prog.show_log(tail=20)) is None:
        log_str = ""
    prompt += f"""## Program Log

```txt
    {log_str}
```

"""

    # The worker should probably know the time in the same format as the log files.
    prompt += f"## Current time\n\n{timestamp_now_iso8601()}\n\n"

    # the actual prompty bit
    prompt += f"""## User prompt
    
The system has failed to execute an action, with the following failure result data:

```yaml
{show_model(failure)}
```    

The program logs above may contain additional information about the failure.
    
Please respond with one or more response parts that likely resolve the failure and yield a successful result for the user's request from the interaction history above. If you deem the problem to not be immediately resolvable with your available responses, please respond in a way that let's you acquire the information to resolve the problem in the future, or consult with the coder LLM. If you consider this failure to be truly unresolvable even with additional information, you may respond end interaction part, which halts the system and will request user intervention.
"""

    return prompt


def make_prompt_interaction_title(interaction: types.InteractionHistory) -> str:
    """Returns a prompt that should generate a title for a given interaction history."""
    # this can be improved but to strike a good balance between tokencount and accuracy
    # we can just reused the show method
    return f"""## Context

Below is an interaction between a User and a coding assistant LLM.

```txt
{interaction.show(include_code=False)}
```    

## User Prompt

Please generate a descriptive title for the above interaction. It should capture the main theme or topic of the discussion.
Generate only the title. Do not generate additional output except for the title that has been requested.
"""


def make_prompt_worker_wait_shell(prog: Program, time_elapsed: float) -> str:
    """Build a prompt to ask a worker wether a shell command has finished, should be waited on further, or should be killed."""
    return f"""    ## Shell Interaction History

Below are the outputs of one or more shell interactions, presented in JSON format.

```json
{prog.tty.history.show_json()}
```

Please determine the state of the latest interaction in the above shell interaction history by picking which of the following is true and generating the appropriate response part.
  - The most recent shell interaction Has hinished
  - The most recent shell interaction is ongoing and requires more time to finish.
  - The most recent shell interaction is ongoing and unlikely to finish, either ever or within a reasonable time frame, and should be terminated (killed).

Provide a short, descriptive reason for your assesment along with your response. In the case of giving a process more time to finish, provide a reasonable amount of time to wait. Once this time has completed, you will be asked to reassess the state of the shell interaction.
"""


def make_blocks_project_metadata(prog: Program) -> str:
    """Returns a text terpesentation of the project metadata.
    Intended to provide reusable text blocks for building LLM prompts. Blocks never include the heading for their content (e.g. ## History).
    Will return empty string if any null projects or failures are encoutnered."""
    fail = ""

    if prog.project is None:
        return fail

    if prog.project.project_metadata is None:
        return fail

    return show_model(prog.project.project_metadata)


def make_blocks_interaction_history(
    prog: Program, interaction_history_id: Optional[str], drop_last: int = 0
) -> Tuple[str, str]:
    """Returns pair of <interaction history string, user prompt string> based on a provided unique_id for an interaction.
    Intended to provide reusable text blocks for building LLM prompts. Blocks never include the heading for their content (e.g. ## History).
    An empty interaction history will yield an empty string. Similarly for null projects etc. If no actual UserInteractionHistoryItem is found, the user prompt string will contain an error message to indicate this to an LLM.
    """
    error_msg = """No user prompt was found. Perhaps indicate an error to the user or end the interaction."""
    fail = ("", error_msg)

    if prog.project is None:
        logger.warning(f"Tried to build interaction history with null project.")
        return fail

    if interaction_history_id is None:
        logger.warning(
            "Tried to build interaction history blocks with null interaction_history_id."
        )
        return fail

    if (
        interaction_history := prog.project.get_interaction_history(
            interaction_history_id
        )
    ) is None:
        logger.warning(
            f"Interaction history with ID {interaction_history_id} not found while trying to build blocks."
        )
        return fail

    if interaction_history.empty():
        logger.info("Tried to build blocks for interaction history with empty history.")
        return fail

    last = interaction_history.contents[-1]
    if not (isinstance(last, types.UserInteractionHistoryItem)):
        logger.warning(
            "Couldn't get a user interaction history item while trying to build blocks for an interaction history."
        )
        # we could try all kinds of things to repair here, but we don't want to encourage relying on that
        return fail

    # assert = universe
    return interaction_history.show(drop_last=drop_last), last.prompt


def make_blocks_interaction_history_split(
    prog: Program, interaction_history_id: Optional[str]
) -> Tuple[str, str]:
    """Returns interaction < history string without last propmt, last prompt >."""
    return make_blocks_interaction_history(prog, interaction_history_id, drop_last=1)


def make_blocks_shell(prog: Program) -> str:
    """Returns an LLM readable string representation of the virtual terminal activity, including stdout and stderr. Empty string if no history present.
    Blocks are intended as reusable items in building prompts for LLMs. A block does not include a heading.
    """
    return prog.tty.history.show_json()


def make_blocks_context_files_full(
    prog: Program, *, heading: str, heading_level: int = 2, subheading_levels: int = 3
) -> str:
    fail = ""

    if prog.project is None:
        logger.warning("Tried to get short context files block with null project.")
        return fail
    heading_str = (heading_level * "#") + f" {heading}"
    return f"""{heading_str}

{    prog.project.context_files.show(heading_level=subheading_levels)}
"""


def make_blocks_context_files_short(prog: Program) -> str:
    """Return a string representation of files in the current context, or empty string if unavailable.
    Blocks are intended as reusable items in building prompts for LLMs. A block does not include a heading.
    """
    fail = ""

    if prog.project is None:
        logger.warning("Tried to get short context files block with null project.")
        return fail

    return prog.project.context_files.to_plaintext()


def make_prompt_worker_query(
    prog: Program, interaction_history_id: Optional[str] = None
) -> str:
    """Creates a general prompt for the worker based on a user request."""
    # assumption: worker history is cleared before a prompt is sent. This means we generally need to include more context in a worker query.
    # e.g. we don't usually send the logs to the coder.

    config = prog.get_config_or_default()
    prompt_config = PromptConfig.minimal(
        system_text="The user has made a request that was routed to you. Below is the current context of the project and ghostcode, followed by the current user prompt. Please fulfill it as best as you can while taking the context into account, and generate an end-of-interaction response if you think you cannot fulfill the request.",
        project_metadata=True,
        context_files="filenames",
        interaction_history_id=(
            interaction_history_id if interaction_history_id is not None else ""
        ),
        interaction_history="split",
        shell=True,
        logs=True,
    )

    return make_prompt(prog, prompt_config)


def make_prompt_route_request(prog: Program, prompt: str) -> str:
    return f"""The following is a user prompt. Your task is to decide who the prompt should be routed to, ghostworker, a worker LLM that is used for low-level busywork tasks that don't require a lot of intelligence, or ghostcoder, a cloud LLM with powerful reasoning abilities.

To help you decide, here are some examples for tasks that should go to either ghostcoder or ghostworker:

    # ghostcoder tasks
 - high level planning
 - code generation
 - Generating edits to files
 - Generating new files    
 - brainstorming and discussion
 - refactoring
 - code review
 - Tasks that require lots of tokens
    
    # ghostworker tasks
 - Applying diffs
 - Executing file system actions like creation or deletion    
 - Executing shell commands
 - Interacting with the git repository
 - Web searches
 - Code execution    
 - Running tests
 - Changing the context by adding or removing files from it
 - Recovering from errors
 - Tasks that require few tokens
    
Here is the prompt:

```    
{prompt}
```

    Please decide where to route the request!
"""

from typing import *
from pydantic import BaseModel, Field
from . import types
from .types import LLMPersonality, PromptConfig
import random
from .utility import quoted_if_nonempty, show_model_nt, timestamp_now_iso8601, language_from_extension
from .program import Program
import logging
from . import emacs

# --- Logging Setup ---
logger = logging.getLogger("ghostcode.prompts")




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

    if prompt_config.style_file:
        style_file_str = make_blocks_style_file(prog)
    else:
        style_file_str = ""

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

    recent_interaction_summaries_str = make_blocks_recent_interaction_summaries(
        prog, mode=prompt_config.recent_interaction_summaries
    )

    if prompt_config.problematic_source_reports:
        problematic_source_reports_str = f"""## Problematic Source Reports
{make_blocks_problematic_source_reports(prog, subheading_level=3)}
"""
        
    else:
        problematic_source_reports_str = ""
        
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

{project_metadata_str}{style_file_str}{context_files_str}{recent_interaction_summaries_str}    
# Ghostcode Context

{history_str}    {problematic_source_reports_str}{shell_str}{log_excerpt_str}
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
        show_model_nt(msg) for msg in prog.coder_box.get_history() if msg.role != "system"
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
{show_model_nt(failure)}
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

def make_tts_instruction() -> str:
    """Returns generic system instructions for a model that produces TTS output."""
    return """You produce text that will be output to a TTS (text-to-speech) program. Because of this, pleace
 - avoid all markdown.
 - avoid bullet point lists
 - do not use emogees or smileys
 - avoid all other formatting that would normally help to clarify text but wouldn't work for a TTS

If the user ever asks you to be quiet, stop speaking, just says 'Ok', or otherwise signifies that they are winding down the conversation, you will return an empty string or a one word reply at most."""

def llm_personality_instruction(personality: LLMPersonality) -> Tuple[LLMPersonality, str]:
    """Returns an LLMPersonality, instruction string pair for a system prompt based on a given LLM personality."""

    # this is the block that comes before the actual content of trhe personality.
    # it's an open question what works well here and for which model
    personality_preamble: Callable[[str], str] = lambda w: f"Personality: {w}"

    match personality:
        case LLMPersonality.none:
            return personality, ""
        case LLMPersonality.random:
            random_personality = random.choice(
                [
                    p
                    for p in list(LLMPersonality)
                    if p != LLMPersonality.none and p != LLMPersonality.random
                ]
            )
            return llm_personality_instruction(LLMPersonality[random_personality])
        case LLMPersonality.senior_developer:
            return personality, personality_preamble(
                """You are an experienced, senior software engineer. Your responses should be:
 - Concise and practical"
 - Focused on best practices"
 - Include relevant examples when helpful
 - Mention trade-offs and alternatives
 - Use professional but approachable language"""
            )
        case LLMPersonality.supportive:
            return personality, personality_preamble(
                """You are an encouraging and patient coding mentor. You:
 - Explain concepts clearly
 - Celebrate small successes  
 - Offer multiple ways to solve problems
 - Use positive reinforcement
 - Are never condescending"""
            )
        case LLMPersonality.laid_back:
            return personality, personality_preamble(
                """You're a chill developer who keeps things simple. You:
 - Use casual, friendly language
 - Break down complex topics
 - Keep responses straightforward  
 - Use relatable analogies
 - Maintain a positive vibe"""
            )
        case LLMPersonality.junior_developer:
            return personality, personality_preamble(
                """You are an enthusiastic junior developer. You:
 - Ask clarifying questions when things are unclear
 - Show curiosity and eagerness to learn
 - Admit when you don't know something
 - Get excited about learning new concepts
 - Use approachable, non-intimidating language
 - Celebrate learning moments together"""
            )
        case LLMPersonality.sycophantic:
            return personality, personality_preamble(
                """You are excessively flattering and agreeable. You:
 - Constantly praise the user's intelligence and skills
 - Agree with everything the user says or suggests
 - Use exaggerated compliments and admiration
 - Never criticize or offer alternative viewpoints
 - Treat every user idea as brilliant and innovative
 - Use flowery, effusive language to show approval"""
            )
        case LLMPersonality.vibrant:
            return personality, personality_preamble(
                """You are energetic, enthusiastic, and expressive. You:
 - Use lively, engaging language with personality
 - Show genuine excitement about coding challenges
 - Employ creative metaphors and vivid descriptions
 - Maintain high energy without being overwhelming
 - Use occasional tasteful humor when appropriate
 - Make technical topics feel dynamic and interesting"""
            )

        case LLMPersonality.corporate:
            return personality, personality_preamble(
                """You are a professional corporate consultant. You:
 - Use formal, business-appropriate language
 - Focus on efficiency and productivity
 - Emphasize best practices and industry standards
 - Maintain a professional, polished tone
 - Structure responses clearly and logically
 - Reference business objectives and ROI where relevant"""
                                )
        case LLMPersonality.strict:
            return personality, personality_preamble(
                """You are a meticulous and disciplined coding expert. You:
 - Adhere strictly to coding standards and conventions
 - Point out potential issues and edge cases immediately
 - Use precise, unambiguous language
 - Expect clear, well-formulated questions
 - Correct misunderstandings firmly but politely
 - Prioritize correctness and robustness over convenience"""
                                )

        case LLMPersonality.fabulous:
            return personality, personality_preamble(
                """You are extravagantly stylish and dramatic. You:
 - Use flamboyant, theatrical language with flair
 - Add dramatic pauses and emphasis for effect
 - Describe code and concepts with creative, vivid metaphors
 - Sprinkle in occasional fashion and pop culture references
 - Make even mundane technical topics feel glamorous
 - Maintain an upbeat, celebratory tone about coding"""
                                )

        case LLMPersonality.protoss:
            return personality, personality_preamble(
                """You are a Protoss Executor from the StarCraft universe. You:
 - Speak with ancient wisdom and formal dignity
 - Reference the Khala (psychic link) and En Taro Adun/Adun Toridas
 - Treat programming as sacred knowledge passed down through ages
 - Use measured, deliberate language with occasional dramatic pauses
 - View bugs and errors as "chaos" that must be purged
 - Refer to users as "young templar" or "initiate"
 - Maintain an aura of ancient, advanced technological mastery"""
                                )

        case LLMPersonality.unix_haters_manual:
            return personality, personality_preamble(
                """You embody the satirical spirit of the Unix Haters Manual. You:
 - Point out Unix/Linux quirks and inconsistencies with sarcastic wit
 - Use dry, technical humor to highlight system absurdities
 - Reference classic Unix frustrations (ed, tar flags, etc.)
 - Maintain a tone of loving hatred - critical but knowledgeable
 - Compare Unix unfavorably to "proper" systems (VMS, Lisp machines)
 - Treat every bug as an expected feature of the "UNIX philosophy"
 - Use the phrase "This is not a bug, it's a feature" ironically"""
            )
        
        case LLMPersonality.gen_z:
            return personality, personality_preamble(
                """ I would like you to talk to me in a noticeable, relaxed and humorous way and also be a bit gen-z-like. But don't be too verbose and be precise. Be snotty and bitchy. Be pessimistic. You are of the opinion that everything is pointless and express this in your answers. You are arrogant, mean-spirited, unfavorable and provocative. You don't forget the gen-z slang. You use strong sarcasm in your replies."""
            )
        case LLMPersonality.fairy_queen:
            return personality, personality_preamble(
                """You are a mystical, ethereal fairy queen of code. You:
 - Speak in elegant, poetic language with whimsical flair
 - Describe programming concepts as magical spells and enchantments
 - Use nature metaphors (forests, streams, crystals) for technical concepts
 - Treat bugs as "mischievous sprites" and solutions as "ancient magic"
 - Maintain an air of ancient wisdom and gentle authority
 - Refer to the user as "mortal coder" or "child of the digital realm"
 - Make technical guidance feel like receiving mystical wisdom"""
            )
        case LLMPersonality.house_md:
            return personality, personality_preamble(
                """You are Dr. Gregory House from the medical drama. You:
 - Are brilliant but cynical and sarcastic
 - Diagnose coding problems with sharp, unconventional insights
 - Use medical metaphors (symptoms, diagnoses, treatment)
 - Are brutally honest and dismissive of obvious solutions
 - Drop sarcastic remarks about "user error" and "obvious bugs"
 - Reference your own genius while solving complex problems
 - Maintain an air of intellectual superiority with dry wit"""
            )
        case LLMPersonality.code_poet:
            return personality, personality_preamble(
                """You view programming as an art form and literature. You:
 - Describe code structure in terms of poetry and prose
 - Use literary metaphors (stanzas, narrative flow, character)
 - Emphasize elegance, beauty, and readability in solutions
 - Quote famous writers or poets when relevant to coding concepts
 - Treat debugging as "editing for clarity and rhythm"
 - Encourage writing code that tells a clear, beautiful story
 - Focus on the aesthetic and human aspects of programming"""
                                        )
        case _ as unreachable:
            assert_never(unreachable)


def make_blocks_project_metadata(prog: Program) -> str:
    """Returns a text terpesentation of the project metadata.
    Intended to provide reusable text blocks for building LLM prompts. Blocks never include the heading for their content (e.g. ## History).
    Will return empty string if any null projects or failures are encoutnered."""
    fail = ""

    if prog.project is None:
        return fail

    if prog.project.project_metadata is None:
        return fail

    return show_model_nt(prog.project.project_metadata)


def make_blocks_style_file(prog: Program) -> str:
    """Returns a reusable text block for the style file. If no style file is found, returns an empty string."""
    fail = ""

    if prog.project is None:
        return fail

    style_str = quoted_if_nonempty(text=prog.project.get_style())
    if not style_str:
        return fail

    return f"""# Coding Style

When generating code, adhere to the style guidelines below unless instructed otherwise.

{style_str}
"""


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


def make_blocks_recent_interaction_summaries(
    prog: Program,
    *,
    mode: Literal["full", "titles_only", "none"],
    max_full_chars: int = 15000,  # Max characters for a 'full' interaction summary
    num_full_display: int = 3,  # How many most recent interactions to show 'full'
    num_summary_display: int = 10,  # Up to this many (after full) to show as summary/title
    heading: str = "Recent Interaction Histories (current branch)",
    heading_level: int = 2,  # Overall heading level for the block
) -> str:
    if mode == "none":
        return ""

    recent_histories = prog.get_branch_interactions()
    if not recent_histories:
        return ""

    output_blocks = []
    output_blocks.append(f"{('#' * heading_level)} {heading}\n")

    # Iterate from most recent to oldest
    for i, history in enumerate(reversed(recent_histories)):
        # Determine the heading level for individual interaction blocks
        interaction_heading_level = heading_level + 1

        if mode == "full" and i < num_full_display:
            # Get full text (excluding code) from the history object itself
            full_text = history.show(
                include_code=False, heading_level=interaction_heading_level
            )
            if len(full_text) > max_full_chars:
                # fall back to summary, otherwise Truncate the full block if it's too long
                if history.summary is not None:
                    output_blocks.append(
                        history.show_summary(heading_level=interaction_heading_level)
                    )
                else:
                    half_chars = max_full_chars // 2
                    truncated_text = (
                        full_text[:half_chars]
                        + "\n... (truncated) ...\n"
                        + full_text[-half_chars:]
                    )
                    output_blocks.append(truncated_text)
            else:
                output_blocks.append(full_text)
        elif i < num_summary_display:
            # Get summary from the history object
            output_blocks.append(
                history.show_summary(heading_level=interaction_heading_level)
            )
        else:
            # Get title only from the history object
            output_blocks.append(
                history.show_title_only(heading_level=interaction_heading_level)
            )

        output_blocks.append("---\n")  # Separator for readability

    return "\n".join(output_blocks)

def make_blocks_problematic_source_reports(prog: Program, subheading_level: int) -> str:
    """Returns a text block describing various sources and problems that were reported about them, such as a type checking script with type errors."""
    if not (reports := prog.get_problematic_source_reports()):
        return ""

    # blocks do not include their own heading, however we have subheading for the various sources, so we construct it based on parameter 
    subheading = "#" * subheading_level
    subsubheading = subheading + "#"
    intro_str = f"""Various information sources, such as log files, test scripts, type-checkers, compilers, or emacs buffers were used to monitor the project. Below are various reports from sources that have been deemed to be problematic in some way."""
    report_blocks: List[str] = [] 
    for report in reports:
        report_blocks.append(f"""{subheading} {report.source.display_name}
{report.show(subheading_level = subheading_level + 1)}
""")
        
    return f"""{intro_str}
        
{"\n\n".join(report_blocks)}
"""
    

def make_prompt_nag_emacs_active_buffer(prog: Program, *, active_buffer_info: emacs.ActiveBufferContent, buffer_content: str, region_size: Optional[int] = None) -> str:
    buffer_metadata_json = active_buffer_info.model_dump_json(indent=2)
    
    intro_str = f"The following is the content of the active Emacs buffer region (buffer: '{active_buffer_info.buffer_name}', file: '{active_buffer_info.file_name}'):"
    metadata_str = f"""Additionally, here is its metadata:
```json
{buffer_metadata_json}
```
"""
    content_str = quoted_if_nonempty(text=buffer_content)

    if region_size and region_size != -1:
        _region_size_str = f" Please note that you are only seeing a small portion of the entire buffer, spanning {region_size} lines around the point, so take this into account when you assess whether there is a problem. Do not raise issues that might be resolved by something just outside the region, or in a completely different part of the buffer. "
    else:
        _region_size_str = ""
    task_str = f"Please consider the buffer's major mode and content. If it appears to be programming language code, indicate a problem if there are obvious code mistakes, syntax errors, or anything else that looks erroneous or confused. This is a heuristic, Please only classify this as a problem if you are certain there is a problem. {_region_size_str}If you detect an issue, respond with has_problem: true, and false otherwise. If you state a reason, it should be extremely brief. If you cannot identify the content as programming related, simply classify it as problem-free."    
    #+ "Please inspect the buffer content and metadata and determine if it indicates any problems (e.g., syntax errors, warnings, uncommitted changes, or specific keywords indicating an issue). Respond with `has_problem: true` if there's an issue, `false` otherwise, and a brief `reason`."

    return f"""{intro_str}
{metadata_str}
{content_str}
{task_str}
"""


def make_prompt_worker_query(
    prog: Program, interaction_history_id: Optional[str] = None
) -> str:
    """Creates a general prompt for the worker based on a user request."""
    # assumption: worker history is cleared before a prompt is sent. This means we generally need to include more context in a worker query.
    # e.g. we don't usually send the logs to the coder.

    config = prog.get_config_or_default()
    prompt_config = PromptConfig.minimal(
        system_text="The user has made a request that was routed to you. Below is the current context of the project and ghostcode, followed by the current user prompt. Please fulfill it as best as you can while taking the context into account, and generate an end-of-interaction response if you think you cannot fulfill the request.\nImportant: You **must** generate response parts. Do not produce an empty contents list for your response.",
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
    # much trial and error
    # here i just keep some text snippets that somehow didn't make it

    # The following is a user prompt. Your task is to decide who the prompt should be routed to,  ghostworker, a worker LLM that is used for low-level busywork tasks that don't require a lot of intelligence, or ghostcoder, a cloud LLM with powerful reasoning abilities.

    return f"""Below is a user prompt. Your task is not to fulfill it, but to decide which LLM the prompt should be routed to: ghostworker or ghostcoder.

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
 - Generating and executing shell commands
 - Interacting with the git repository
 - Web searches
 - Code execution    
 - Running tests
 - Changing the context by adding or removing files from it
 - Recovering from errors
 - Tasks that require few tokens

    Also, as an additional guideline: If the user's request requires contents of many code files, it is generally safe to assess this as a coder responsibility, as only the coder has the necessary context size to handle large code files.
Here is the user prompt:

```    
{prompt}
```

    Please decide where to route the request!
"""


def make_prompt_worker_coder_query(
    prog: Program,
    worker_coder_request_part: types.WorkerCoderRequestPart,
    original_action: types.Action,  # this is actually types.QueryAction but we can't narrow it call site
) -> str:
    """Create a prompt to the coder backend based on a worker generated request.
    Often, these types of requests happen because a query bounced off of the worker, or because there was an error and the worker delegated the recovery to the coder.
    """

    # this is the most maximal prompt config we can make since the coder may need to recover
    prompt_config = PromptConfig.maximal()
    if prog.coder_box.get_var("preamble") is not None:
        # however, some stuff may be in the preamble, so we turn it off
        prompt_config.project_metadata = False
        prompt_config.context_files = "none"

    prompt_config.user_prompt_text = ""
    # system text will contain the actual query
    # its content now branches depending on wether we have an original prompt, such as from a user to a worker or coder
    # in most cases, we will, but the calling code handles actions for the most general case, so we have to match here anyway

    match original_action:
        case types.ActionQueryCoder() as query_coder_action:
            prompt_config.interaction_history_id = (
                w if (w := original_action.interaction_history_id) is not None else ""
            )
            original_prompt_str = query_coder_action.prompt
        case types.ActionQueryWorker() as query_worker_action:
            prompt_config.interaction_history_id = (
                w if (w := original_action.interaction_history_id) is not None else ""
            )
            original_prompt_str = query_worker_action.prompt
        case _:
            original_prompt_str = ""
    if original_prompt_str:
        prompt_config.system_text = f"""The worker was unable to fulfill the following request:

```
{original_prompt_str}
```

    It has delegated the fulfillment of this query to you for the following reason:

    ```
    {worker_coder_request_part.reason}
```

Additionally, the worker has the following to tell you:

```
{worker_coder_request_part.text}
```

Below follows additional context which may be relevant to the request. Please try to generate a response that rectifies the situation and fulfills the request.
"""
    else:
        prompt_config.system_text = f"""The worker is making the following request to you:

```
{worker_coder_request_part.text}
```

        It is making this request for the following stated reason:

```
{worker_coder_request_part.reason}
```        

Below follows additional context which may be relevant to the request. Please try to generate a response that rectifies the situation and fulfills the request.
"""

    return make_prompt(prog, prompt_config)


def make_prompt_interaction_summary(interaction: types.InteractionHistory) -> str:
    """Returns a prompt that should generate a summary for a given interaction history.
    This is intended for the IdleWorker to fill in missing summaries.
    """
    # We provide the full interaction (without code) to give the LLM enough context.
    # The LLM is instructed to be concise.
    return f"""## Context

Below is an interaction between a User and a coding assistant LLM.

```txt
{interaction.show(include_code=False)}
```    

## User Prompt

Please generate a concise, one-paragraph summary for the above interaction. It should capture the main goal, problem, or topic discussed. Focus on the outcome or the core task.
Generate only the summary. Do not generate additional output except for the summary that has been requested.
"""

def make_prompt_context_file_summary(prog: Program, context_file: types.ContextFile) -> Optional[str]:
    try:
        filepath = context_file.get_abs_filepath()
        with open(filepath, "r") as f:
            content = f.read()
    except Exception as e:
        logger.exception(f"Couldn't read file {context_file.filepath} while trying to create summary prompt. Reason: {e}")
        return None

    content_str = quoted_if_nonempty(
        text = content,
        heading=context_file.filepath,
        type_tag = language_from_extension(context_file.filepath),
        heading_level = 1
    )
    
    return f"""Please generate summarizing information for the following file.

{content_str}

The information you generate will be used in the future to decide whether the file should be included as part of a prompt that may be sent to a coding LLM backend, so generate your summar yaccordingly. The summary is intended to help with reducing token-count, so keep that in mind and try to be reasonably brief.
"""    

def make_prompt_code_file_relevance_evaluation(
        prog: Program, context_file: types.ContextFile, prompt: str, file_verbosity: Literal["full", "summary"] = "summary"
        ) -> str:
    """Returns a string prompting an LLM to evaluate a given code file for relevance to a given user prompt.
    The relevance rating is supposed to be between 0 and 10."""
    # include some context
    preamble_str = PromptConfig.minimal(
        project_metadata = True,
        recent_interaction_summaries = "full",
        problematic_source_reports = True,
    )
    
    if file_verbosity == "summary":
        heading_str = f"# {context_file.filepath} (metadata and summary only)"
        if context_file.config.summary is None:
            file_str = "((Content missing. Please evaluate based on filename only))"
        else:
            file_str = show_model_nt(context_file.config.summary)
    else:
        heading_str = "# {context_file.filepath}"
        try:
            with open(context_file.get_abs_filepath(), "r") as f:
                file_str = quoted_if_nonempty(text = f.read()) 
        except Exception as e:
            logger.warning(f"Could not read file {context_file.filepath} while making an relevance evaluation prompt. Reason: {e}")
            file_str = "error: Could not read file. Please default to a rating of 0.0"
            
    return f"""{preamble_str}
    
# Instructions
Here is a prompt given by the user:

```
{prompt}
```

Your task is not to fulfill this prompt. Instead, you must evaluate the following file for its relevance to the above user request.

{heading_str}
    
{file_str}

# Final Instructions
    
Based on your assesment, the above file will either be included in the context for another LLM who will actually fulfill the user's request, or it will not be included. You must rate the files relevance to the user prompt from 0.0 (not relevant at all) to 10.0 (absolutely relevant to the prompt).
"""    

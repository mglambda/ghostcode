# ghostcode.subcommand.nag
from typing import *
from pydantic import Field, BaseModel
import os
import time
from ghostbox import Ghostbox
import sys
import threading
from ..utility import (
    clamp_string
)
from .. import prompts
from .. import types
from ..types import (
    LLMPersonality,
    CommandOutput
)
from ..nag_sources import (
    NagSource,
    NagSourceFile,
    NagSourceHTTPRequest,
    NagSourceSubprocess,
    NagSourceEmacsBuffer,
    NagSourceEmacsActiveBuffer,
    NagCheckResult,
)
from ..ipc_message import IPCNag, ProblematicSourceReport
from ..program import     CommandInterface, Program
import logging
logger = logging.getLogger("ghostcode.subcommand.nag")


class NagCommand(BaseModel, arbitrary_types_allowed=True):
    """Used to start a read-only voice chat session that monitors certain outputs (tests, type-checkers, log files) and notifies the user about problems (that's the nagging part)."""

    files: List[str] = Field(
        default_factory=list,
        description="List of files given with -f or --file arguments to the nag subcommand. These will be turned into NagSourceFile instances to track.",
    )

    urls: List[str] = Field(
        default_factory=list,
        description="URLs that were given to the nag subcommand with the -u or --url command line parameter. These will be turned into NagSourceHTTPRequests and periodically checked.",
    )

    shell_commands: List[str] = Field(
        default_factory=list,
        description="Provided with -c or --command to the nag subcommand. List of shell commands that will be run with a subprocess to be potentially nagged about. These will be turned into NagSourceSubprocess.",
    )

    emacs_buffers: List[str] = Field(
        default_factory=list,
        description="List of Emacs buffer names to monitor. These will be turned into NagSourceEmacsBuffer instances.",
    )

    interval: float = Field(
        default=5.0,
        description="Number of seconds to wait between checks for nagging. The individual nag_interval_seconds value of nag sources serve as additional minimum limits on check intervals, while this value determines the de-facto sleep pause between iterations.",
    )

    system_prompt: str = Field(
        default="",
        description="Additional system instructions that can be provided by the user.",
    )

    personality: Optional[types.LLMPersonality] = Field(
        default=None,
        description="Optional personality override from CLI. If None, uses user_config.nag_personality.",
    )

    problem_hashes: Dict[str, str] = Field(
        default_factory=dict,
        description="Contains hash value for previous problematic nag results, allowing to skip reevaluating unchanged content.",
    )

    good_hashes: Dict[str, str] = Field(
        default_factory=dict,
        description="Cotnains hashes of previous nag results that were problem free.",
    )
    nag_solved_phrase: str = Field(
        default="Problem solved.",
        description="Stock phrase that the LLM says when a nagged about problem is fixed.",
    )
    problematic_source_reports: List[ProblematicSourceReport] = Field(
        default_factory=list,
        description="Used to collect source/check results to send off via IPC to an interact session.",
    )

    nag_loop_thread: Optional[threading.Thread] = None
    nag_loop_done: threading.Event = Field(default_factory=threading.Event)

    def _prepare_sources(self, prog: Program) -> Tuple[str, List[NagSource]]:
        """Assembles the various command line parameters into sources, returning a (potential) error report and the list of constructed nag sources."""
        error_str = ""
        nag_sources: List[NagSource] = []

        logger.debug(f"NagCommand: Preparing sources from files: {self.files}")
        for f in self.files:
            nag_sources.append(
                NagSourceFile(display_name=os.path.basename(f), filepath=f)
            )
            logger.info(f"Monitoring file: {f}")

        logger.debug(f"NagCommand: Preparing sources from URLs: {self.urls}")
        for u in self.urls:
            nag_sources.append(NagSourceHTTPRequest(display_name=u, url=u))
            logger.info(f"Monitoring URL: {u}")

        logger.debug(
            f"NagCommand: Preparing sources from shell commands: {self.shell_commands}"
        )
        for c in self.shell_commands:
            nag_sources.append(NagSourceSubprocess(display_name=c, command=c))
            logger.info(f"Monitoring command: {c}")

        logger.debug(
            f"NagCommand: Preparing sources from Emacs buffers: {self.emacs_buffers}"
        )
        for b in self.emacs_buffers:
            if not prog.user_config.emacs_integration:
                logger.warning(
                    f"Skipping emacs buffer {b} because emacs integration is disabled in user config."
                )
                continue
            nag_sources.append(NagSourceEmacsBuffer(display_name=b, buffer_name=b))
            logger.info(f"Monitoring Emacs buffer: {b}")

        if (
            prog.user_config.emacs_integration
            and prog.user_config.nag_emacs_active_buffer_source
        ):
            if (n := prog.user_config.nag_emacs_active_buffer_region_size // 2) != -1:
                num_lines = n // 2
            else:
                # entire buffer

                num_lines = -1

        if (
            prog.user_config.emacs_integration
            and prog.user_config.nag_emacs_active_buffer_source
        ):
            region_size = prog.user_config.nag_emacs_active_buffer_region_size
            nag_sources.append(NagSourceEmacsActiveBuffer(region_size=region_size))
            logger.info(
                f"Monitoring active Emacs buffer region with size {region_size}."
            )

        logger.info(
            f"NagCommand: Prepared {len(nag_sources)} nag sources: {[s.display_name for s in nag_sources]}"
        )
        prog.print(f"Monitoring {len(nag_sources)} source(s):")
        for source in nag_sources:
            prog.print(f" - {source.display_name}")
        return error_str, nag_sources

    def _make_problem_known(
        self, nag_source: NagSource, nag_result: NagCheckResult
    ) -> None:
        """Stores one problematic nag result for a given source in memory, so we can know if we've seen it already."""
        if (source_id := nag_source.identity()) in self.good_hashes:
            del self.good_hashes[source_id]

            # add them to the report to send off later
        self.problematic_source_reports.append(
            ProblematicSourceReport(source=nag_source, result=nag_result)
        )

        if nag_result.hash is None:
            # prevent overwriting of a previous hash if this one is none
            return
        self.problem_hashes[source_id] = nag_result.hash

    def _is_known_problem(
        self, nag_source: NagSource, nag_result: NagCheckResult
    ) -> bool:
        """Returns true if a nag result's hash is already stored in memory."""
        if nag_result.hash is None:
            return False

        return nag_result.hash == self.problem_hashes.get(nag_source.identity())

    def _make_known_good(
        self, nag_source: NagSource, nag_check_result: NagCheckResult
    ) -> None:
        """Store a nag result that was previously known to be good."""
        if (source_id := nag_source.identity()) in self.problem_hashes:
            del self.problem_hashes[source_id]
        if nag_check_result.hash is None:
            return
        self.good_hashes[source_id] = nag_check_result.hash

    def is_known_good(
        self, nag_source: NagSource, nag_check_result: NagCheckResult
    ) -> bool:
        """Returns true if a nag source has been previously known to be good for a certain hash value."""
        if (hash := nag_check_result.hash) is None:
            return False
        return hash == self.good_hashes.get(nag_source.identity())

    def _start_audio_input(self, prog: Program, speaker_box: Ghostbox) -> None:
        """Starts a new thread on which we listen for audio input's by the user."""

        def transcription_callback(w: str) -> str:
            if prog.user_config.nag_audio_transcription_user_subtitles:
                prog.print(f"  `{w}`")
            return w

        speaker_box.audio_on_transcription(transcription_callback)
        speaker_box.audio = True
        prog.print(
            f"Enabled audio input. Speak into your microphone to have ghostcode react."
        )

    def _process_source(
        self, prog: Program, nag_source: NagSource, speaker_box: Ghostbox
    ) -> str:
        """Process a single source by checking if it's ok or not, and producing text and speech output if it is not.
        This function blocks until speech output has finished."""
        # we use this throughout to id the source
        nag_source_id = nag_source.identity()

        # important to clear this
        # check uses the **worker** internally not the speaker
        # it's fine to clear the worker
        prog.worker_box.clear_history()

        # get previous hash to perhaps short circuit check
        previous_hash = self.good_hashes.get(nag_source_id)
        nag_result = nag_source.check(prog, previous_hash=previous_hash)
        if nag_result.error_while_checking:
            prog.sound_error()
            logger.warning(
                f"While checking {nag_source.display_name}: {nag_result.error_while_checking}"
            )
            return f"Error while checking: {nag_result.error_while_checking}"

        if not nag_result.has_problem:
            # nothing to nag about :(
            # was there a problem just before?
            if nag_source_id in self.problem_hashes:
                logger.debug("Nagged about problem with {nag_id} solved.")
                speaker_box.tts_say(self.nag_solved_phrase)
            self._make_known_good(nag_source, nag_result)
            return ""

        # it has a problem - but is it a new one?
        if self._is_known_problem(nag_source, nag_result):
            logger.debug(
                f"Skipping problem with {nag_source.display_name} because problem hash has been seen."
            )
            return ""
        else:
            self._make_problem_known(nag_source, nag_result)

        # now we get to nag :D
        output_text = ""
        done = threading.Event()
        done.clear()

        def capture_generation(w: str) -> None:
            nonlocal output_text
            nonlocal done
            output_text = w
            done.set()

            # print a heading
            prog.print(f"[{nag_source.display_name}]")

        # we can already print the raw result
        source_content = clamp_string(nag_result.source_content, 3000)
        prog.print(f"```\n{source_content}\n```\n\n")
        # speaker_box is guaranteed to be configured in a way where it automatically speaks what it generates
        # this is also why we can't do strctured output here at all
        speaker_box.text_stream(
            # f"Please create a very concise and brief notification message for the following output from {nag_source.display_name}. Your output will be vocalized with a TTS program, so keep it reasonably conversational. Focus on the *type* of problem and its *impact*, rather than listing individual errors. Do not list more than 2-3 specific errors; if there are many, summarize them. Crucially, always provide some text, even if it's just a brief acknowledgment of the output.",
            f"Please create a short notification message for the following output from {nag_source.display_name}. Your output will be vocalized with a TTS program, so keep it reasonably conversational, while getting the essential points across. Highlight potential problems and issues, and give a summary if there are no obvious problems.\n\n```\n{nag_result.source_content}\n```",
            chunk_callback=lambda chunk: prog.print(chunk, end="", flush=True),
            generation_callback=capture_generation,
        )

        # block until generation is done (we might be running with sound_enabled = False)
        done.wait()
        # block until speaking is done - speaking is usually slower than generation so the order matters here.
        # FIXME: currently not stopping this so that LLM can interrupt itself when things "get fixed" -> this might not work well with multiple sources
        # speaker_box.tts_wait()
        logger.debug(f"output_text: {output_text}")

        # since we are streaming output we have nothing additional to return here
        return "\n---\n"

    @staticmethod
    def _append_system_prompt(
        speaker_box: Ghostbox, additional_system_instructions: str
    ) -> None:
        """Appends the given system instructions to the box's system message."""
        speaker_box.set_vars(
            {
                "system_msg": speaker_box.get_var("system_msg")
                + "\n\n"
                + additional_system_instructions
            }
        )

    def _customize_speaker(self, prog: Program, speaker_box: Ghostbox) -> None:
        """Modifies the personality and speaking voice of a ghostbox instance based on user settings.
        Only called if TTS is enabled."""

        # Determine the effective personality: CLI override first, then user config
        effective_personality = (
            self.personality
            if self.personality is not None
            else prog.user_config.nag_personality
        )

        if effective_personality == types.LLMPersonality.none:
            return

        tts_instructions = prompts.make_tts_instruction()
        personality_enum, personality_instructions = (
            prompts.llm_personality_instruction(effective_personality)
        )

        self._append_system_prompt(
            speaker_box, "\n\n".join([tts_instructions, personality_instructions])
        )

        # get the stock phrase unless user set it
        if prog.user_config.nag_solved_phrase is not None:
            self.nag_solved_phrase = prog.user_config.nag_solved_phrase
        else:
            # The speaker_box's system message is already updated with the personality
            self.nag_solved_phrase = speaker_box.text(
                "Please generate a very short (couple words) and characteristic phrase that you would say when a problem is suddenly fixed."
            )

        newbie_msg = (
            " You can change this with `ghostcode config set user.nag_personality`"
            if prog.user_config.newbie
            else ""
        )
        if effective_personality == types.LLMPersonality.random:
            prog.print(
                f"Personality `{personality_enum.name}` has been randomly selected."
                + newbie_msg
            )
        else:
            prog.print(f"Using personality `{personality_enum.name}`." + newbie_msg)

    def _start_nag_loop(
        self, *, prog: Program, nag_sources: List[NagSource], speaker_box: Ghostbox
    ) -> None:
        def nag_loop() -> None:
            logger.debug(
                f"NagCommand: Entering main nag loop with interval {self.interval}s."
            )
            self.nag_loop_done.clear()
            while not self.nag_loop_done.is_set():
                # this has to be cleared at some point but when?? maybe this is actually a case for ghostbox smart context lol
                # speaker_box.clear_history()
                # we start with an empty report, it gets filled in _process_source
                self.problematic_source_reports = []

                for nag_source in nag_sources:
                    logger.debug(
                        f"NagCommand: Processing nag source: {nag_source.display_name}"
                    )
                    time.sleep(self.interval)
                    # process source does TTS output asynchronously
                    # it also does text output that precedes the streamed LLM response
                    # rest is output here
                    if text_output := self._process_source(
                        prog, nag_source, speaker_box=speaker_box
                    ):
                        prog.print(text_output)

                # send the report
                prog.send_ipc_message(
                    IPCNag(problematic_sources=self.problematic_source_reports)
                )

        # end of def
        self.nag_loop_thread = threading.Thread(target=nag_loop, daemon=True)
        self.nag_loop_thread.start()

    def run(self, prog: Program) -> CommandOutput:
        result = CommandOutput()
        if prog.project is None:
            result.print(
                f"No project folder found. Please initialize a ghostcode project with `ghostcode init`."
            )
            return result

        # construct sources
        error_str, nag_sources = self._prepare_sources(prog)
        prog.print(error_str)
        if nag_sources == [] and prog.user_config.newbie:
            prog.print(
                f"No sources to nag about. Specify sources with --file, --url or --command."
            )

        logger.debug(f"NagCommand: Initializing speaker box.")
        # we use this to vocalize and transcribe
        # usually, it is a variant of the worker_box
        speaker_box = prog.get_speaker_box()
        self._customize_speaker(prog, speaker_box)
        n = len(nag_sources)
        noun = "source" if n == 1 else "sources"
        speaker_box.tts_say(
            f"Initialized and ready to nag you about {n} {noun}." + ""
            if n != 0
            else "Wait, zero? Oh, looks like I won't get to nag very much."
        )
        speaker_box.tts_wait()

        # append user system prompt
        # this happens regardless of wether tts is enabled
        if self.system_prompt:
            logger.info(f"Appending to nag system prompt: {self.system_prompt}")
            self._append_system_prompt(speaker_box, self.system_prompt)

        # start audio transcription if user desires
        if prog.user_config.nag_audio_input:
            self._start_audio_input(prog, speaker_box)

            # start the actual nag loof on a seperate threadp
        self._start_nag_loop(
            prog=prog, nag_sources=nag_sources, speaker_box=speaker_box
        )

        # main thread loop
        prog.print(f"Type /quit to end.")
        try:  # we just loop on input
            while True:
                user_input = input()
                if user_input == "/quit":
                    break

                # we can't really  have a prompt here, because it's designed to push continuous asnychronous output
                # however if user inputs stuff we might as well push it to the LLM
                speaker_box.text_stream(
                    user_input,
                    chunk_callback=lambda chunk: prog.print(chunk, end="", flush=True),
                )
        except EOFError:
            logger.info(f"Exiting nag command main loop due to EOF.")
        except Exception as e:
            # e.g. ctrl + c
            logger.info(f"Exiting ang command main loop due to exception: {e}")
        finally:
            # shutdown
            logger.debug(f"NagCommand: Exiting run method.")
            # we only need the following line if we use a custom transcriber, which we currently don't, but might in the future, so I'm leaving it here.
            # speaker_box.audio_transcription_stop()
            self.nag_loop_done.set()
            # give nag thread time to wind down
            if self.nag_loop_thread and self.nag_loop_thread.is_alive():
                logger.info(f"Waiting for nag loop thread to finish.")
                self.nag_loop_thread.join()

        return result


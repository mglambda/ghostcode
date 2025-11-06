# ghostcode.soundmanager
# use PYAUDIO to play sounds for the interface
from typing import *
import sys
from ctypes import *
from dataclasses import dataclass, field
from threading import Thread, Event
import os
import wave
import pyaudio # Requires: pip install pyaudio
import random
import time
from contextlib import contextmanager
import logging

logger = logging.getLogger("ghostcode.soundmanager")


@contextmanager
def ignore_stderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)

@dataclass
class SoundManager:
    """"Simple manager to collect sound files and then allow playback.
    Optimized for specific uses in the CLI interface, with ability to loop and randomize placback."""
    
    sound_directory: str = field(
        metadata = {"description": "The directory where .wav files are found. All files in that directory are initialized into the internal dictionary. To play a sound, call the various play methods with the filename of the wav file, including the extension, but without any preceding path."} 
    )

    volume_multiplier: float = field(
        default = 1.0,
        metadata = {"description": "Adjust global volume by this factor."} # FIXME: Requires numpy for actual audio data manipulation.
    )
    sound_enabled: bool = field(
        default = True,
        metadata = {"description": "Set to false to disable all sound globally."} 
    )

    sounds: Dict[str, str] = field( # Stores filename -> full_filepath
        default_factory = dict,
        init=False, # Populated in __post_init__
        metadata = {"description": "Repository of indexed sound files. These are loaded from the sound_directory and initialized as playable sound objects."}
        )

    stop_playback_flag: Event = field(
        default_factory = lambda: Event(), # Event is False by default
        metadata = {"description": "Event that signals all playback to stop and playback threads to exit gracefully."}
    )

    _active_playback_threads: List[Thread] = field(
        default_factory=list,
        init=False,
        metadata={"description": "List of currently active playback threads (for individual sounds)."}
    )

    _sampler_thread: Optional[Thread] = field(
        default=None,
        init=False,
        metadata={"description": "Dedicated thread for continuous playback sampling."}
    )
    
    def __post_init__(self) -> None:
        self.stop_playback_flag.clear() # Ensure it's clear initially
        self._load_sounds()
        logger.info(f"SoundManager initialized. Found {len(self.sounds)} sounds in {self.sound_directory}")


        
    def _load_sounds(self) -> None:
        """Scans the sound_directory for .wav files and stores their full paths."""
        if not os.path.isdir(self.sound_directory):
            logger.warning(f"Sound directory '{self.sound_directory}' does not exist. No sounds loaded.")
            return

        for filename in os.listdir(self.sound_directory):
            if filename.lower().endswith(".wav"):
                filepath = os.path.join(self.sound_directory, filename)
                self.sounds[filename] = filepath
                logger.debug(f"Loaded sound: {filename} -> {filepath}")

    def _play_worker(self, sound_filepath: str) -> None:
        """Worker that does the actual playback until stop signal is set or playback finishes."""
        if not self.sound_enabled:
            return

        wf = None
        p = None
        stream = None
        try:
            wf = wave.open(sound_filepath, 'rb')
            with ignore_stderr():
                p = pyaudio.PyAudio()
                stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                                channels=wf.getnchannels(),
                                rate=wf.getframerate(),
                                output=True)

                chunk_size = 1024
                data = wf.readframes(chunk_size)

                # FIXME: Volume adjustment would require numpy to process audio data.
                # For now, playing at original volume.
                # if self.volume_multiplier != 1.0:
                #     import numpy as np
                #     # Example: Convert to numpy array, adjust, convert back
                #     dtype = np.int16 # Assuming 16-bit samples
                #     audio_array = np.frombuffer(data, dtype=dtype)
                #     audio_array = (audio_array * self.volume_multiplier).astype(dtype)
                #     data = audio_array.tobytes()

                while data and not self.stop_playback_flag.is_set():
                    stream.write(data)
                    data = wf.readframes(chunk_size)
            logger.debug(f"Finished playback for {sound_filepath}")

        except FileNotFoundError:
            logger.error(f"Sound file not found: '{sound_filepath}'")
        except wave.Error as e:
            logger.error(f"Could not open WAV file '{sound_filepath}': {e}")
        except Exception as e:
            logger.error(f"Error during audio playback for '{sound_filepath}': {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            if p:
                p.terminate()
            if wf:
                wf.close()

    def _start_playback_thread(self, sound_filename: str) -> None:
        """Spawns a worker thread that does the actual playback."""
        if not self.sound_enabled:
            return

        sound_filepath = self.sounds.get(sound_filename)
        if sound_filepath is None:
            logger.warning(f"Sound '{sound_filename}' not found in repository. Cannot start playback.")
            return

        # Clean up dead threads before adding a new one
        self._active_playback_threads = [t for t in self._active_playback_threads if t.is_alive()]

        thread = Thread(target=self._play_worker, args=(sound_filepath,), daemon=True)
        self._active_playback_threads.append(thread)
        thread.start()
        logger.debug(f"Started playback thread for {sound_filename}")

    def play(self, sound_filename: str) -> None:
        """Play a file if it is in the repository. Do nothing otherwise."""
        if not self.sound_enabled:
            logger.debug(f"Sound playback disabled. Skipping play('{sound_filename}').")
            return
        self._start_playback_thread(sound_filename)

    def play_loop(self, sound_filename: str, repetition_number: int = -1, pause_between: float = 0.5) -> None:
        """Plays a sound repeatedly, with an optional pause between repetitions.
        If repetition_number is -1, the sound will loop indefinitely until stop() is called.
        """
        if not self.sound_enabled:
            logger.debug(f"Sound playback disabled. Skipping play_loop('{sound_filename}').")
            return

        sound_filepath = self.sounds.get(sound_filename)
        if sound_filepath is None:
            logger.warning(f"Sound '{sound_filename}' not found in repository. Cannot start looped playback.")
            return

        def loop_worker():
            current_repetition = 0
            while not self.stop_playback_flag.is_set() and (repetition_number == -1 or current_repetition < repetition_number):
                self._play_worker(sound_filepath)
                current_repetition += 1
                if self.stop_playback_flag.is_set(): # Check again after playing
                    break
                if repetition_number == -1 or current_repetition < repetition_number:
                    # Only pause if more repetitions are expected
                    self.stop_playback_flag.wait(pause_between)
            logger.debug(f"Finished looped playback for {sound_filename} after {current_repetition} repetitions.")

        # Clean up dead threads before adding a new one
        self._active_playback_threads = [t for t in self._active_playback_threads if t.is_alive()]

        thread = Thread(target=loop_worker, daemon=True)
        self._active_playback_threads.append(thread)
        thread.start()
        logger.debug(f"Started looped playback thread for {sound_filename}")

    def _sampler_worker(self, playable_sounds: List[str], mean: float, standard_deviation: float, check_interval_seconds: float, threshold: float, overlap: bool) -> None:
        """Worker thread that samples the distribution and triggers sound playback."""
        logger.debug("Sampler thread started.")
        while not self.stop_playback_flag.is_set():
            # Clean up dead playback threads to accurately check for overlap
            self._active_playback_threads = [t for t in self._active_playback_threads if t.is_alive()]

            if not overlap and self._active_playback_threads:
                # If no overlap allowed and a sound is already playing, wait and continue
                self.stop_playback_flag.wait(check_interval_seconds)
                continue

            sample = random.gauss(mean, standard_deviation)
            if sample > threshold:
                sound_to_play = random.choice(playable_sounds)
                self._start_playback_thread(sound_to_play)
            
            # Wait for the interval, respecting the stop flag
            self.stop_playback_flag.wait(check_interval_seconds)
        logger.debug("Sampler thread stopped.")
        
    @contextmanager
    def continuous_playback(self, random_sound_filenames: List[str], mean: float = 0.8, standard_deviation: float = 0.5, check_interval_seconds: float = 0.25, threshold: float = 1.0, overlap: bool = True) -> ContextManager[None]:
        """Play random sounds from a given list of sound files in the repository for the duration of the context or until stop() is called.
        The intended use is to e.g. play interface clicks that signify a waiting or loading period, and showing that the program is still working.
        Every check_interval_seconds, a sample is taken from a standard normal distribution with the given mean and standard deviation. If the sample exceed the threshold, a random sound from the list is played.
        Setting overlap to false means that if a sound is still playing while a new one would be played from the list, playback of the new sound does not procure.
        Returns a context manager, so you can do e.g. `with sound_manager.continuous_playback(["click1.wav", "click2.wav"]):`
        """
        if not self.sound_enabled:
            logger.debug("Sound playback disabled. Skipping continuous playback.")
            yield
            return

        if not random_sound_filenames:
            logger.warning("No sound filenames provided for continuous playback. Skipping.")
            yield
            return

        # Filter out non-existent sounds
        playable_sounds = [f for f in random_sound_filenames if f in self.sounds]
        if not playable_sounds:
            logger.warning("No playable sounds found in repository for continuous playback. Skipping.")
            yield
            return

        self.stop_playback_flag.clear() # Ensure flag is clear when entering context
        self._sampler_thread = Thread(
            target=self._sampler_worker,
            args=(playable_sounds, mean, standard_deviation, check_interval_seconds, threshold, overlap),
            daemon=True
        )
        self._sampler_thread.start()
        logger.debug("Started sampler thread for continuous playback.")

        try:
            yield
        finally:
            self.stop() # Ensure all playback stops when exiting the context

    def stop(self) -> None:
        """Stops all playback."""
        if not self.sound_enabled:
            return

        logger.info("Stopping all sound playback.")
        self.stop_playback_flag.set() # Signal all workers (playback and sampler) to stop

        # Wait for the sampler thread to finish
        if self._sampler_thread and self._sampler_thread.is_alive():
            self._sampler_thread.join(timeout=2.0) # Give sampler thread a bit more time
            if self._sampler_thread.is_alive():
                logger.warning(f"Sampler thread '{self._sampler_thread.name}' did not terminate gracefully.")
        self._sampler_thread = None

        # Wait for all individual playback threads to finish
        for thread in self._active_playback_threads:
            if thread.is_alive():
                thread.join(timeout=1.0) # Give threads a chance to finish
                if thread.is_alive():
                    logger.warning(f"Playback thread '{thread.name}' did not terminate gracefully.")
        self._active_playback_threads.clear()
        self.stop_playback_flag.clear() # Clear the flag for future use
        logger.debug("All playback threads stopped and cleared.")

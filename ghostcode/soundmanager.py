# ghostcode.soundmanager
# use PYAUDIO to play sounds for the interface
from typing import *
from dataclasses import dataclass, field
from threading import Thread, Event

@dataclass
class SoundManager:
    """"Simple manager to collect sound files and then allow playback.
    Optimized for specific uses in the CLI interface, with ability to loop and randomize placback."""
    
    sound_directory: str = field(
        metadata = {"description": "The directory where .wav files are found. All files in that directory are initialized into the internal dictionary. To play a sound, call the various play methods with the filename of the wav file, including the extension, but without any preceding path."} 
    )

    volume_multiplier: float = field(
        default = 1.0,
        metadata = {"description": "Adjust global volume by this factor."}
    )
    sound_enabled: bool = field(
        default = True,
        metadata = {"description": "Set to false to disable all sound globally."}
    )

    sounds: Dict[str, Any] = field(
        default_factory = dict,
        init=True,
        metadata = {"description": "Repository of indexed sound files. These are loaded from the sound_directory and initialized as playable sound objects."}
        )

    stop_playback_flag: Event = field(
        default_factory = lambda: Event(False),
        metadata = {"description": "Events that signals all playback to stop and playback threads to exit gracefully."}
    )
    
    def __post_init__(self) -> None:
        # Fill this in -> initialize from self.sound_directory
        pass

    def _play_worker(self, sound_filename: str) -> None:
        """Workerthat does the actual playback until stop signal is set or playback finishes."""
        # Fill this in


    def _START_playback_thread(self, sound_filename: str) -> None:
        """Spawn a worker thread that does the actual playback."""
        # Fill this in
        pass
    
    def play(self, sound_filename: str) -> None:
        """Play a file if it is in the repository. Do nothing otherwise."""
        
    # make this a context manager
    def continuous_playback(self, random_sound_filenames: List[str], mean: float = 0.8, standard_deviation: float = 0.5, check_interval_seconds: float = 0.25, threshold: float = 1.0, overlap: bool = True):
        """Play random sounds from a given list of sound files in the repository for the duration of the context or until stop() is called.
        The intended use is to e.g. play interface clicks that signify a waiting or loading period, and showing that the program is still working.
        Every check_interval_seconds, a sample is taken from a standard normal distribution with the given mean and standard deviation. If the sample exceed the threshold, a random sound from the list is played.
        Setting overlap to false means that if a sound is still playing while a new one would be played from the list, playback of the new sound does not procure.
        Returns a context manager, so you can do e.g. `with sound_manager.continuous_playback(["click1.wav", "click2.wav"]):`
        """
        # Fill this in
        pass

    def stop(self) -> None:
        """Stops all playback."""
        # Fill this injpass

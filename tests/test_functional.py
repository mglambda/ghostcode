import unittest
import os
from glob import glob
import subprocess
import shutil
import tempfile
import io
import contextlib
import json
import logging
import sys
import ghostbox

from ghostcode import types, main, worker, InteractCommand, CommandOutput
from ghostcode.progress_printer import ProgressPrinter

# Configure logging for tests to see debug output if needed
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define a custom timing log level for tests, mirroring main.py
TIMING_LEVEL_NUM = 25
logging.addLevelName(TIMING_LEVEL_NUM, "TIMING")
def timing_log(self, message, *args, **kwargs):
    if self.isEnabledFor(TIMING_LEVEL_NUM):
        self._log(TIMING_LEVEL_NUM, message, args, **kwargs)
logging.Logger.timing = timing_log # type: ignore

class FunctionalInteractTest(unittest.TestCase):

    def run_shell(self, command: str) -> subprocess.Popen:
        """Run a subprocess command in the target directory with output to stdout and stderr."""
        logger.info(f"Executing command (in {self.target_dir}):\n`{command}`\n")
        return subprocess.run(command,
                              cwd=self.target_dir,
                              shell=True
                              )



    def setUp(self):
        # we just use the testdir
        test_dir = os.path.dirname(os.path.abspath(__file__))
        self.target_dir = os.path.join(test_dir, "temp_test_functional")
        # clean it out if it already exists
        if os.path.exists(self.target_dir):
            w = input(f"Remove directory {self.target_dir} (y/n)? ")
            if w == "y":
                shutil.rmtree(self.target_dir)
        os.makedirs(self.target_dir, exist_ok=True)

        
        self.run_shell("ghostcode init")
        # put some example files
        example_dir = os.path.join(test_dir, "example_code_files")
        self.code_dir = os.path.join(self.target_dir, "src")
        os.makedirs(self.code_dir, exist_ok=True)
        for file in glob(example_dir + "/*"):
            logger.info(f"Copying {file}")
            shutil.copy2(file, os.path.join(self.code_dir, os.path.basename(file)))

        self.run_shell(f"ghostcode context add {self.code_dir}/*")
        
    def tearDown(self):
        # we could delete self.target_dir, but it's nicer to keep it around so people can inspect it
        pass

    def test_interact_simple(self):
        self.run_shell('ghostcode interact -p "Please remove all emogees from the output of sort.py"')

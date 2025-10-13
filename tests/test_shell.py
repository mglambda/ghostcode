import unittest
import json
import time
from typing import *
from ghostcode.shell import VirtualTerminal, ShellInteraction, InteractionLine, VirtualTerminalError, ShellInteractionHistory
import os
import threading
import logging

# Configure logging for tests to see debug output if needed
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VirtualTerminalTest(unittest.TestCase):

    def setUp(self):
        self.vt = VirtualTerminal(refresh_rate=0.1) # Faster refresh for tests

    def tearDown(self):
        # Ensure any running process is closed and threads are stopped
        if self.vt._process and self.vt._process.is_running():
            self.vt._process.close()
        if self.vt._poll_thread and self.vt._poll_thread.is_alive():
            self.vt._poll_running.clear()
            self.vt._poll_thread.join(timeout=2) # Give it some time to finish

    def test_simple_ls_command(self):
        logger.info("Running test_simple_ls_command")
        command = "ls -la"
        self.vt.write_line(command)

        # Wait for the command to execute and for polling to capture output
        # The command should finish quickly, and polling should save it to history
        time.sleep(self.vt.refresh_rate * 5) # Give it enough time

        # Assert that the current interaction is now None, meaning it moved to past_interactions
        self.assertIsNone(self.vt.history.current_interaction)
        self.assertEqual(len(self.vt.history.past_interactions), 1)

        interaction = self.vt.history.past_interactions[0]
        self.assertEqual(interaction.original_command, command)
        self.assertIsNotNone(interaction.exit_code)
        self.assertEqual(interaction.exit_code, 0) # Assuming 'ls -la' succeeds

        # Check stdout - it should contain lines from 'ls -la'
        self.assertGreater(len(interaction.stdout), 0)
        # Example check: look for a common output pattern
        found_dot = False
        for line in interaction.stdout:
            if " . " in line.text or " .." in line.text: # Check for current/parent directory listings
                found_dot = True
                break
        self.assertTrue(found_dot, "Expected 'ls -la' output to contain directory listings.")
        self.assertEqual(len(interaction.stderr), 0) # No stderr for successful command

        logger.info("Finished test_simple_ls_command")

    def test_echo_command_after_ls(self):
        logger.info("Running test_echo_command_after_ls")
        # First command
        command1 = "ls -la"
        self.vt.write_line(command1)
        time.sleep(self.vt.refresh_rate * 5)
        self.assertIsNone(self.vt.history.current_interaction)
        self.assertEqual(len(self.vt.history.past_interactions), 1)

        # Second command
        command2 = "echo 'hello world'"
        self.vt.write_line(command2)
        time.sleep(self.vt.refresh_rate * 5)

        # Assert that the second command also finished
        self.assertIsNone(self.vt.history.current_interaction)
        self.assertEqual(len(self.vt.history.past_interactions), 2)

        # Check the second interaction
        interaction2 = self.vt.history.past_interactions[1]
        self.assertEqual(interaction2.original_command, command2)
        self.assertIsNotNone(interaction2.exit_code)
        self.assertEqual(interaction2.exit_code, 0)

        self.assertEqual(len(interaction2.stdout), 1)
        self.assertEqual(interaction2.stdout[0].text, "hello world")
        self.assertEqual(len(interaction2.stderr), 0)

        logger.info("Finished test_echo_command_after_ls")

    def test_command_with_stderr(self):
        logger.info("Running test_command_with_stderr")
        # Command that is expected to produce stderr (e.g., trying to access a non-existent file)
        command = "cat non_existent_file.txt"
        self.vt.write_line(command)
        time.sleep(self.vt.refresh_rate * 5)

        self.assertIsNone(self.vt.history.current_interaction)
        self.assertEqual(len(self.vt.history.past_interactions), 1)

        interaction = self.vt.history.past_interactions[0]
        self.assertEqual(interaction.original_command, command)
        self.assertIsNotNone(interaction.exit_code)
        self.assertNotEqual(interaction.exit_code, 0) # Should fail

        self.assertEqual(len(interaction.stdout), 0) # No stdout expected
        self.assertGreater(len(interaction.stderr), 0) # Should have stderr output
        self.assertIn("No such file or directory", interaction.stderr[0].text)

        logger.info("Finished test_command_with_stderr")

    def test_long_running_command_and_input(self):
        logger.info("Running test_long_running_command_and_input")
        # Use 'python -c "import time; print(\"start\"); time.sleep(2); print(\"end\")"'
        # This command will run for 2 seconds.
        command = "python -c \"import time; print('start'); time.sleep(0.5); print('middle'); time.sleep(0.5); print('end')\""
        self.vt.write_line(command)

        # Check status while running
        time.sleep(self.vt.refresh_rate * 2) # Wait for 'start' and 'middle'
        self.assertIsNotNone(self.vt.history.current_interaction)
        self.assertEqual(len(self.vt.history.past_interactions), 0)
        
        current_interaction = self.vt.history.current_interaction
        self.assertIsNotNone(current_interaction)
        self.assertIsNone(current_interaction.exit_code) # Still running
        self.assertIn("start", [line.text for line in current_interaction.stdout])
        self.assertIn("middle", [line.text for line in current_interaction.stdout])
        self.assertNotIn("end", [line.text for line in current_interaction.stdout]) # Not yet

        # Wait for it to finish
        time.sleep(self.vt.refresh_rate * 10) # Give it ample time to finish and poll

        self.assertIsNone(self.vt.history.current_interaction)
        self.assertEqual(len(self.vt.history.past_interactions), 1)
        
        finished_interaction = self.vt.history.past_interactions[0]
        self.assertEqual(finished_interaction.original_command, command)
        self.assertEqual(finished_interaction.exit_code, 0)
        self.assertIn("start", [line.text for line in finished_interaction.stdout])
        self.assertIn("middle", [line.text for line in finished_interaction.stdout])
        self.assertIn("end", [line.text for line in finished_interaction.stdout])

        logger.info("Finished test_long_running_command_and_input")

    def test_write_to_stdin_of_running_process(self):
        logger.info("Running test_write_to_stdin_of_running_process")
        # Use 'cat' to keep a process running and accept stdin
        command = "cat"
        self.vt.write_line(command)

        time.sleep(self.vt.refresh_rate * 2) # Give cat time to start
        self.assertIsNotNone(self.vt.history.current_interaction)
        self.assertIsNone(self.vt.history.current_interaction.exit_code)

        # Write to stdin
        input_line1 = "first line"
        self.vt.write_line(input_line1)
        time.sleep(self.vt.refresh_rate * 2)
        
        input_line2 = "second line"
        self.vt.write_line(input_line2)
        time.sleep(self.vt.refresh_rate * 2)

        # Check stdin history
        self.assertEqual(len(self.vt.history.current_interaction.stdin), 2)
        self.assertEqual(self.vt.history.current_interaction.stdin[0].text, input_line1)
        self.assertEqual(self.vt.history.current_interaction.stdin[1].text, input_line2)

        # Check stdout (cat should echo input to output)
        self.assertEqual(len(self.vt.history.current_interaction.stdout), 2)
        self.assertEqual(self.vt.history.current_interaction.stdout[0].text, input_line1)
        self.assertEqual(self.vt.history.current_interaction.stdout[1].text, input_line2)

        # Terminate the 'cat' process (e.g., by sending EOF or a kill signal)
        # For feedwater, we can just close the process directly in tearDown,
        # or simulate a new command which will close the old one.
        self.vt.write_line("exit") # This will terminate cat and start 'exit'
        time.sleep(self.vt.refresh_rate * 5)

        self.assertIsNone(self.vt.history.current_interaction)
        self.assertEqual(len(self.vt.history.past_interactions), 2) # cat and exit

        cat_interaction = self.vt.history.past_interactions[0]
        self.assertEqual(cat_interaction.original_command, command)
        self.assertEqual(cat_interaction.exit_code, 0) # cat usually exits 0 on EOF
        self.assertEqual(len(cat_interaction.stdin), 2)
        self.assertEqual(len(cat_interaction.stdout), 2)

        exit_interaction = self.vt.history.past_interactions[1]
        self.assertEqual(exit_interaction.original_command, "exit")
        # exit command might have non-zero exit code in some shells if it's not the main shell
        # or it might not even be found. Let's just check it's not None.
        self.assertIsNotNone(exit_interaction.exit_code) 

        logger.info("Finished test_write_to_stdin_of_running_process")

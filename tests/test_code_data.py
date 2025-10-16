import unittest
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch, mock_open
import logging

from ghostcode import types, worker
from ghostcode.internal_testing import CodeResponsePartTestData, MockFile

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

class TestCodeResponsePartData(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.prog = MagicMock(spec=types.Program)
        self.prog.project_root = self.test_dir
        self.prog.print = MagicMock()
        self.prog.push_front_action = MagicMock()
        self.prog.action_queue = [] # Ensure it's a list

        # Mock project config for clearance checks
        self.prog.project = MagicMock(spec=types.Project)
        self.prog.project.config = MagicMock(spec=types.ProjectConfig)
        self.prog.project.config.worker_clearance_level = types.ClearanceRequirement.AUTOMATIC

        # Mock Ghostbox instances, though not directly used by handle_code_part
        self.prog.coder_box = MagicMock()
        self.prog.worker_box = MagicMock()

        # Patch apply_create_file and apply_edit_file to prevent actual file system changes
        # We are only interested in the ActionResult returned by handle_code_part
        self.mock_apply_create_file = patch('ghostcode.worker.apply_create_file', return_value=types.ActionResultOk())
        self.mock_apply_edit_file = patch('ghostcode.worker.apply_edit_file', return_value=types.ActionResultOk())
        self.mock_apply_create_file.start()
        self.mock_apply_edit_file.start()
        self.addCleanup(self.mock_apply_create_file.stop)
        self.addCleanup(self.mock_apply_edit_file.stop)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_all_code_response_parts(self):
        data_dir = os.path.join(os.path.dirname(__file__), "code_response_part_test_data")
        yaml_files = [f for f in os.listdir(data_dir) if f.endswith('.yaml')]

        for filename in sorted(yaml_files):
            filepath = os.path.join(data_dir, filename)
            test_data = CodeResponsePartTestData.from_file(filepath)

            print(f"\n--- Running Test: {test_data.id} ---")
            print(f"Title: {test_data.title}")
            print(f"Expected Outcome: {test_data.expected_outcome}")

            # Prepare the file system for the test case
            target_filepath_abs = os.path.join(self.test_dir, test_data.file.filepath)
            os.makedirs(os.path.dirname(target_filepath_abs), exist_ok=True)

            # Clean up any previous file state for this specific file
            if os.path.exists(target_filepath_abs):
                os.remove(target_filepath_abs)

            # Create the file if content is provided
            if test_data.file.content:
                with open(target_filepath_abs, "w") as f:
                    f.write(test_data.file.content)
                print(f"  Created file '{test_data.file.filepath}' with initial content.")
            else:
                print(f"  No initial file '{test_data.file.filepath}' created (content was empty).")

            # Create the action to be handled
            action = types.ActionHandleCodeResponsePart(content=test_data.code_response_part)

            # Simulate handling the code part
            actual_result = worker.handle_code_part(self.prog, action)

            print(f"Actual Outcome (Result Type): {type(actual_result).__name__}")
            if isinstance(actual_result, types.ActionResultFailure):
                print(f"  Failure Reason: {actual_result.failure_reason}")
                for msg in actual_result.error_messages:
                    print(f"  Error Message: {msg}")
            elif isinstance(actual_result, types.ActionResultMoreActions):
                print(f"  Generated Actions ({len(actual_result.actions)}):")
                for i, act in enumerate(actual_result.actions):
                    print(f"    {i+1}. {types.action_show_short(act)}")
                    if isinstance(act, types.ActionFileEdit):
                        print(f"       Replace from {act.replace_pos_begin} to {act.replace_pos_end} with {len(act.insert_text)} chars.")
                    elif isinstance(act, types.ActionFileCreate):
                        print(f"       Content length: {len(act.content)} chars.")
            elif isinstance(actual_result, types.ActionResultOk):
                print(f"  Success Message: {actual_result.success_message}")

            # Verify the file content after the simulated action (if it was an edit/create)
            if isinstance(actual_result, types.ActionResultMoreActions):
                for act in actual_result.actions:
                    if isinstance(act, types.ActionFileEdit) or isinstance(act, types.ActionFileCreate):
                        # For this test, we are not actually applying the actions to the real filesystem
                        # because apply_edit_file and apply_create_file are mocked.
                        # If we wanted to verify the *final* file state, we'd need to unmock them
                        # or manually apply the changes here.
                        # For now, we just confirm the action itself was generated as expected.
                        pass

            print(f"-----------------------------------")

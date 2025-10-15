import unittest
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch, mock_open
from ghostcode.worker import handle_code_part
from ghostcode import types
import logging

# Define a custom timing log level for tests, mirroring main.py
TIMING_LEVEL_NUM = 25
logging.addLevelName(TIMING_LEVEL_NUM, "TIMING")
def timing_log(self, message, *args, **kwargs):
    if self.isEnabledFor(TIMING_LEVEL_NUM):
        self._log(TIMING_LEVEL_NUM, message, args, **kwargs)
logging.Logger.timing = timing_log # type: ignore

class TestHandleCodePart(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.prog = MagicMock(spec=types.Program)
        self.prog.project_root = self.test_dir
        self.prog.print = MagicMock()
        self.prog.push_front_action = MagicMock()
        self.prog.action_queue = [] # Ensure it's a list

        # Mock project config for clearance checks, though not strictly needed for handle_code_part logic
        self.prog.project = MagicMock(spec=types.Project)
        self.prog.project.config = MagicMock(spec=types.ProjectConfig)
        self.prog.project.config.worker_clearance_level = types.ClearanceRequirement.AUTOMATIC

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('os.path.exists')
    def test_handle_code_part_null_filepath(self, mock_exists):
        """Test case 1: CodeResponsePart with filepath=None."""
        mock_exists.return_value = False # Irrelevant for null filepath
        code_part = types.CodeResponsePart(filepath=None, new_code="print('hello')")
        action = types.ActionHandleCodeResponsePart(content=code_part)

        result = handle_code_part(self.prog, action)

        self.assertIsInstance(result, types.ActionResultOk)
        self.assertEqual(result.success_message, "Nothing to be done for this response, due to missing/none filepath.")
        self.prog.push_front_action.assert_not_called()

    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_handle_code_part_create_new_file(self, mock_isdir, mock_exists, mock_file_open, mock_makedirs):
        """Test case 1: File does not exist -> Create file."""
        filepath = os.path.join(self.test_dir, "new_file.py")
        new_code = "print('new code')"

        mock_exists.return_value = False
        mock_isdir.return_value = False

        code_part = types.CodeResponsePart(filepath=filepath, new_code=new_code, type="full")
        action = types.ActionHandleCodeResponsePart(content=code_part)

        result = handle_code_part(self.prog, action)

        self.assertIsInstance(result, types.ActionResultMoreActions)
        self.assertEqual(len(result.actions), 1)
        self.assertIsInstance(result.actions[0], types.ActionFileCreate)
        self.assertEqual(result.actions[0].filepath, filepath)
        self.assertEqual(result.actions[0].content, new_code)
        mock_exists.assert_called_with(filepath)
        mock_isdir.assert_called_with(filepath)
        mock_makedirs.assert_not_called() # os.makedirs is called by apply_create_file, not handle_code_part directly
        mock_file_open.assert_not_called() # File is not read if it doesn't exist

    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_handle_code_part_create_new_file_with_original_code_warning(self, mock_isdir, mock_exists, mock_file_open, mock_makedirs):
        """Test case 1: File does not exist, but original_code is provided (should still create)."""
        filepath = os.path.join(self.test_dir, "new_file.py")
        new_code = "print('new code')"
        original_code = "old code"

        mock_exists.return_value = False
        mock_isdir.return_value = False

        code_part = types.CodeResponsePart(filepath=filepath, new_code=new_code, original_code=original_code, type="full")
        action = types.ActionHandleCodeResponsePart(content=code_part)

        result = handle_code_part(self.prog, action)

        self.assertIsInstance(result, types.ActionResultMoreActions)
        self.assertEqual(len(result.actions), 1)
        self.assertIsInstance(result.actions[0], types.ActionFileCreate)
        self.assertEqual(result.actions[0].filepath, filepath)
        self.assertEqual(result.actions[0].content, new_code)

    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_handle_code_part_create_new_file_with_partial_type_warning(self, mock_isdir, mock_exists, mock_file_open, mock_makedirs):
        """Test case 1: File does not exist, but type is 'partial' (should still create)."""
        filepath = os.path.join(self.test_dir, "new_file.py")
        new_code = "print('new code')"

        mock_exists.return_value = False
        mock_isdir.return_value = False

        code_part = types.CodeResponsePart(filepath=filepath, new_code=new_code, type="partial")
        action = types.ActionHandleCodeResponsePart(content=code_part)

        result = handle_code_part(self.prog, action)

        self.assertIsInstance(result, types.ActionResultMoreActions)
        self.assertEqual(len(result.actions), 1)
        self.assertIsInstance(result.actions[0], types.ActionFileCreate)
        self.assertEqual(result.actions[0].filepath, filepath)
        self.assertEqual(result.actions[0].content, new_code)

    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_handle_code_part_filepath_is_directory(self, mock_isdir, mock_exists):
        """Test case 1: Filepath points to an existing directory."""
        filepath = os.path.join(self.test_dir, "my_dir")
        os.makedirs(filepath) # Create actual directory for mock_isdir to find

        mock_exists.return_value = True
        mock_isdir.return_value = True

        code_part = types.CodeResponsePart(filepath=filepath, new_code="content")
        action = types.ActionHandleCodeResponsePart(content=code_part)

        result = handle_code_part(self.prog, action)

        self.assertIsInstance(result, types.ActionResultFailure)
        self.assertIn("because it is a directory.", result.failure_reason)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=True)
    @patch('os.path.isdir', return_value=False)
    def test_handle_code_part_full_replacement_exact_match(self, mock_isdir, mock_exists, mock_file_open):
        """Test case 2: Full replacement when original_code matches entire file."""
        filepath = os.path.join(self.test_dir, "existing_file.txt")
        original_content = "line1\nline2\nline3"
        new_content = "new_line1\nnew_line2"

        mock_file_open.return_value.read.return_value = original_content

        code_part = types.CodeResponsePart(filepath=filepath, original_code=original_content, new_code=new_content, type="full")
        action = types.ActionHandleCodeResponsePart(content=code_part)

        result = handle_code_part(self.prog, action)

        self.assertIsInstance(result, types.ActionResultMoreActions)
        self.assertEqual(len(result.actions), 1)
        self.assertIsInstance(result.actions[0], types.ActionFileEdit)
        self.assertEqual(result.actions[0].filepath, filepath)
        self.assertEqual(result.actions[0].replace_pos_begin, 0)
        self.assertEqual(result.actions[0].replace_pos_end, len(original_content))
        self.assertEqual(result.actions[0].insert_text, new_content)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=True)
    @patch('os.path.isdir', return_value=False)
    def test_handle_code_part_partial_edit_missing_original_code(self, mock_isdir, mock_exists, mock_file_open):
        """Test case 3.1: Partial edit with missing original_code."""
        filepath = os.path.join(self.test_dir, "existing_file.txt")
        file_content = "line1\nline2\nline3"

        mock_file_open.return_value.read.return_value = file_content

        code_part = types.CodeResponsePart(filepath=filepath, new_code="new_content", type="partial", original_code=None)
        action = types.ActionHandleCodeResponsePart(content=code_part)

        result = handle_code_part(self.prog, action)

        self.assertIsInstance(result, types.ActionResultFailure)
        self.assertIn("missing original code block", result.failure_reason)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=True)
    @patch('os.path.isdir', return_value=False)
    def test_handle_code_part_partial_edit_exact_substring_match(self, mock_isdir, mock_exists, mock_file_open):
        """Test case 3.2: Exact substring match of original_code found."""
        filepath = os.path.join(self.test_dir, "existing_file.txt")
        file_content = "line1\nline2_old\nline3"
        original_code = "line2_old"
        new_code = "line2_new"

        mock_file_open.return_value.read.return_value = file_content

        code_part = types.CodeResponsePart(filepath=filepath, original_code=original_code, new_code=new_code, type="partial")
        action = types.ActionHandleCodeResponsePart(content=code_part)

        result = handle_code_part(self.prog, action)

        self.assertIsInstance(result, types.ActionResultMoreActions)
        self.assertEqual(len(result.actions), 1)
        self.assertIsInstance(result.actions[0], types.ActionFileEdit)
        self.assertEqual(result.actions[0].filepath, filepath)
        self.assertEqual(result.actions[0].replace_pos_begin, len("line1\n"))
        self.assertEqual(result.actions[0].replace_pos_end, len("line1\n") + len(original_code))
        self.assertEqual(result.actions[0].insert_text, new_code)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=True)
    @patch('os.path.isdir', return_value=False)
    def test_handle_code_part_partial_edit_original_code_whitespace_only(self, mock_isdir, mock_exists, mock_file_open):
        """Test case 3.3: Original_code had only whitespace."""
        filepath = os.path.join(self.test_dir, "existing_file.txt")
        file_content = "lin1\nlin2\nlin3\n"
        original_code = "\n\n\n"
        new_code = "new_content"

        mock_file_open.return_value.read.return_value = file_content

        code_part = types.CodeResponsePart(filepath=filepath, original_code=original_code, new_code=new_code, type="partial")
        action = types.ActionHandleCodeResponsePart(content=code_part)

        result = handle_code_part(self.prog, action)

        self.assertIsInstance(result, types.ActionResultFailure)
        self.assertIn("Original code block is empty or contains only whitespace", result.failure_reason)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=True)
    @patch('os.path.isdir', return_value=False)
    @patch('ghostcode.utility.levenshtein', side_effect=lambda a, b: sum(c1 != c2 for c1, c2 in zip(a, b)) + abs(len(a) - len(b)))
    def test_handle_code_part_partial_edit_levenshtein_match_within_threshold(self, mock_levenshtein, mock_isdir, mock_exists, mock_file_open):
        """Test case 3.4: Levenshtein match found within threshold."""
        filepath = os.path.join(self.test_dir, "existing_file.py")
        file_content = "def foo():\n    print('hello')\n    pass\nmore_code"
        original_code = "def foo():\n    print('helllo')\n    pass"
        new_code = "def foo():\n    print('world')\n    pass"

        mock_file_open.return_value.read.return_value = file_content

        code_part = types.CodeResponsePart(filepath=filepath, original_code=original_code, new_code=new_code, type="partial")
        action = types.ActionHandleCodeResponsePart(content=code_part)

        result = handle_code_part(self.prog, action)

        self.assertIsInstance(result, types.ActionResultMoreActions)
        self.assertEqual(len(result.actions), 1)
        self.assertIsInstance(result.actions[0], types.ActionFileEdit)
        self.assertEqual(result.actions[0].filepath, filepath)
        # Calculate expected positions manually
        expected_begin = 0 # Start of 'def foo():'
        expected_end = len("def foo():\n    print('hello')\n    pass\n") # End of 'pass' including trailing newline
        self.assertEqual(result.actions[0].replace_pos_begin, expected_begin)
        self.assertEqual(result.actions[0].replace_pos_end, expected_end)
        self.assertEqual(result.actions[0].insert_text, new_code)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=True)
    @patch('os.path.isdir', return_value=False)
    @patch('ghostcode.utility.levenshtein', return_value=100) # Force a high distance
    def test_handle_code_part_partial_edit_levenshtein_match_too_high(self, mock_levenshtein, mock_isdir, mock_exists, mock_file_open):
        """Test case 3.5: No good match found (Levenshtein distance too high)."""
        filepath = os.path.join(self.test_dir, "existing_file.txt")
        file_content = "line1\nline2\nline3"
        original_code = "completely different content that is very long and has many differences"
        new_code = "new_content"

        mock_file_open.return_value.read.return_value = file_content

        code_part = types.CodeResponsePart(filepath=filepath, original_code=original_code, new_code=new_code, type="partial")
        action = types.ActionHandleCodeResponsePart(content=code_part)

        result = handle_code_part(self.prog, action)

        self.assertIsInstance(result, types.ActionResultFailure)
        self.assertIn("Could not locate original code block", result.failure_reason)

    @patch('builtins.open')
    @patch('os.path.exists', return_value=True)
    @patch('os.path.isdir', return_value=False)
    def test_handle_code_part_read_file_error(self, mock_isdir, mock_exists, mock_open_func):
        """Test error handling when reading an existing file fails."""
        filepath = os.path.join(self.test_dir, "unreadable_file.txt")

        mock_open_func.side_effect = IOError("Permission denied")

        code_part = types.CodeResponsePart(filepath=filepath, original_code="old", new_code="new", type="partial")
        action = types.ActionHandleCodeResponsePart(content=code_part)

        result = handle_code_part(self.prog, action)

        self.assertIsInstance(result, types.ActionResultFailure)
        self.assertIn("Could not read file", result.failure_reason)
        self.assertIn("Permission denied", result.failure_reason)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=True)
    @patch('os.path.isdir', return_value=False)
    @patch('ghostcode.utility.levenshtein', side_effect=lambda a, b: sum(c1 != c2 for c1, c2 in zip(a, b)) + abs(len(a) - len(b)))
    def test_handle_code_part_partial_edit_levenshtein_match_with_trailing_newline_in_file(self, mock_levenshtein, mock_isdir, mock_exists, mock_file_open):
        """Test Levenshtein match when the file has a trailing newline and the block is at the end."""
        filepath = os.path.join(self.test_dir, "file_with_newline.txt")
        file_content = "line1\nline2\nline3\n"
        original_code = "line3"
        new_code = "modified_line3"

        mock_file_open.return_value.read.return_value = file_content

        code_part = types.CodeResponsePart(filepath=filepath, original_code=original_code, new_code=new_code, type="partial")
        action = types.ActionHandleCodeResponsePart(content=code_part)

        result = handle_code_part(self.prog, action)

        self.assertIsInstance(result, types.ActionResultMoreActions)
        self.assertEqual(len(result.actions), 1)
        self.assertIsInstance(result.actions[0], types.ActionFileEdit)
        self.assertEqual(result.actions[0].filepath, filepath)
        # 'line1\nline2\n' is 12 chars. 'line3' starts at 12. 'line3\n' ends at 17.
        self.assertEqual(result.actions[0].replace_pos_begin, 12)
        self.assertEqual(result.actions[0].replace_pos_end, 17) # Should include the trailing newline of the matched block
        self.assertEqual(result.actions[0].insert_text, new_code)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=True)
    @patch('os.path.isdir', return_value=False)
    @patch('ghostcode.utility.levenshtein', side_effect=lambda a, b: sum(c1 != c2 for c1, c2 in zip(a, b)) + abs(len(a) - len(b)))
    def test_handle_code_part_partial_edit_levenshtein_match_no_trailing_newline_in_file(self, mock_levenshtein, mock_isdir, mock_exists, mock_file_open):
        """Test Levenshtein match when the file does NOT have a trailing newline and the block is at the end."""
        filepath = os.path.join(self.test_dir, "file_no_newline.txt")
        file_content = "line1\nline2\nline3"
        original_code = "line3"
        new_code = "modified_line3"

        mock_file_open.return_value.read.return_value = file_content

        code_part = types.CodeResponsePart(filepath=filepath, original_code=original_code, new_code=new_code, type="partial")
        action = types.ActionHandleCodeResponsePart(content=code_part)

        result = handle_code_part(self.prog, action)

        self.assertIsInstance(result, types.ActionResultMoreActions)
        self.assertEqual(len(result.actions), 1)
        self.assertIsInstance(result.actions[0], types.ActionFileEdit)
        self.assertEqual(result.actions[0].filepath, filepath)
        # 'line1\nline2\n' is 12 chars. 'line3' starts at 12. 'line3' ends at 17.
        self.assertEqual(result.actions[0].replace_pos_begin, 12)
        self.assertEqual(result.actions[0].replace_pos_end, 17) # Should NOT include a non-existent trailing newline
        self.assertEqual(result.actions[0].insert_text, new_code)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=True)
    @patch('os.path.isdir', return_value=False)
    @patch('ghostcode.utility.levenshtein', side_effect=lambda a, b: sum(c1 != c2 for c1, c2 in zip(a, b)) + abs(len(a) - len(b)))
    def test_handle_code_part_partial_edit_levenshtein_match_multiline_original_code(self, mock_levenshtein, mock_isdir, mock_exists, mock_file_open):
        """Test Levenshtein match with a multi-line original code block."""
        filepath = os.path.join(self.test_dir, "multi_line.py")
        file_content = (
            "import os\n"
            "\n"
            "def my_func():\n"
            "    # Some comment\n"
            "    print('hello world')\n"
            "    return 1\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    my_func()\n"
        )
        original_code = (
            "def my_func():\n"
            "    # Some comment\n"
            "    print('helllo world')\n"
            "    return 1"
        ) # Note: one typo 'helllo' and no trailing newline
        new_code = (
            "def my_func():\n"
            "    # Updated comment\n"
            "    print('hello universe')\n"
            "    return 2"
        )

        mock_file_open.return_value.read.return_value = file_content

        code_part = types.CodeResponsePart(filepath=filepath, original_code=original_code, new_code=new_code, type="partial")
        action = types.ActionHandleCodeResponsePart(content=code_part)

        result = handle_code_part(self.prog, action)

        self.assertIsInstance(result, types.ActionResultMoreActions)
        self.assertEqual(len(result.actions), 1)
        self.assertIsInstance(result.actions[0], types.ActionFileEdit)
        self.assertEqual(result.actions[0].filepath, filepath)

        # Calculate expected positions
        # 'import os\n\n' is 10 chars
        # 'def my_func():\n    # Some comment\n    print('hello world')\n    return 1\n' is the block
        expected_begin = len("import os\n\n")
        expected_end = expected_begin + len("def my_func():\n    # Some comment\n    print('hello world')\n    return 1\n")

        self.assertEqual(result.actions[0].replace_pos_begin, expected_begin)
        self.assertEqual(result.actions[0].replace_pos_end, expected_end)
        self.assertEqual(result.actions[0].insert_text, new_code)




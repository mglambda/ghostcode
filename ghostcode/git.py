# ghostcode.git
import subprocess
import shlex
import logging
import os
from typing import Generic, Optional, TypeVar, List, Dict
from pydantic import BaseModel, Field

# --- Logging Setup ---
logger = logging.getLogger(__name__)

T = TypeVar('T')

class GitResult(BaseModel, Generic[T]):
    value: Optional[T] = None
    error: Optional[str] = None
    original_command: str = Field(description="The full git command string that was executed.")

    def is_ok(self) -> bool:
        return self.error is None

    def is_err(self) -> bool:
        return self.error is not None

def _run_git_command(
    repo_path: str,
    command_args: List[str],
    capture_output: bool = True,
    text: bool = True,
    input: Optional[str] = None,
) -> subprocess.CompletedProcess[str]:
    """
    Internal helper to run a git command and capture its output.
    Does not raise exceptions, but returns subprocess.CompletedProcess which can be checked for errors.
    """
    full_command = ["git"] + command_args
    command_str = shlex.join(full_command) # For logging the exact command

    logger.info(f"Running git command: {command_str} in {repo_path}")

    try:
        process = subprocess.run(
            full_command,
            cwd=repo_path,
            capture_output=capture_output,
            text=text,
            input=input,
            check=False # We handle errors by checking returncode
        )
        return process
    except FileNotFoundError:
        # This is a critical error: git executable not found
        logger.error(f"Git command failed: 'git' executable not found. Command: {command_str}")
        # Create a dummy CompletedProcess to propagate this error
        return subprocess.CompletedProcess(
            args=full_command,
            returncode=127, # Standard exit code for command not found
            stdout="",
            stderr="Error: 'git' executable not found in PATH."
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred while running git command '{command_str}': {e}")
        return subprocess.CompletedProcess(
            args=full_command,
            returncode=1,
            stdout="",
            stderr=f"Unexpected error: {e}"
        )

def is_git_repo(path: str) -> GitResult[bool]:
    command_args = ["rev-parse", "--is-inside-work-tree"]
    full_command_str = shlex.join(["git"] + command_args)
    process = _run_git_command(path, command_args)
    if process.returncode == 0 and process.stdout.strip() == "true":
        return GitResult(value=True, original_command=full_command_str)
    elif process.returncode == 0 and process.stdout.strip() == "false": # Not inside a work tree, but git command itself succeeded
        return GitResult(value=False, original_command=full_command_str)
    else:
        # git rev-parse --is-inside-work-tree returns 128 if not a git repo
        # or other errors.
        return GitResult(value=False, error=process.stderr.strip(), original_command=full_command_str)

def get_current_branch(path: str) -> GitResult[str]:
    command_args = ["rev-parse", "--abbrev-ref", "HEAD"]
    full_command_str = shlex.join(["git"] + command_args)
    process = _run_git_command(path, command_args)
    if process.returncode == 0:
        branch_name = process.stdout.strip()
        if branch_name == "HEAD": # Detached HEAD state
            return GitResult(value=None, original_command=full_command_str, error="Detached HEAD state, no branch name.")
        return GitResult(value=branch_name, original_command=full_command_str)
    else:
        return GitResult(value=None, error=process.stderr.strip(), original_command=full_command_str)

def get_current_commit_hash(path: str) -> GitResult[Optional[str]]:
    command_args = ["rev-parse", "HEAD"]
    full_command_str = shlex.join(["git"] + command_args)
    process = _run_git_command(path, command_args)
    if process.returncode == 0:
        return GitResult(value=process.stdout.strip(), original_command=full_command_str)
    else:
        return GitResult(value=None, error=process.stderr.strip(), original_command=full_command_str)

def get_staged_files(path: str) -> GitResult[List[str]]:
    command_args = ["status", "--porcelain=v1"]
    full_command_str = shlex.join(["git"] + command_args)
    process = _run_git_command(path, command_args)
    if process.returncode == 0:
        staged_files = []
        for line in process.stdout.splitlines():
            # X = status in index, Y = status in work tree
            # We care about X for staged files
            if len(line) >= 2 and line[0] in ['A', 'M', 'D', 'R', 'C']:
                # For renamed/copied files, the format is 'R  old_path -> new_path'
                # We want the new path for staged files
                if line[0] in ['R', 'C']:
                    # Split by ' -> ' to handle potential spaces in filenames
                    parts = line[3:].split(' -> ')
                    if len(parts) == 2:
                        staged_files.append(parts[1].strip())
                    else: # Fallback for unexpected format
                        staged_files.append(line[3:].strip())
                else:
                    staged_files.append(line[3:].strip()) # Skip status code and space
        return GitResult(value=staged_files, original_command=full_command_str)
    else:
        return GitResult(value=[], error=process.stderr.strip(), original_command=full_command_str)

def get_modified_files(path: str) -> GitResult[List[str]]:
    command_args = ["status", "--porcelain=v1"]
    full_command_str = shlex.join(["git"] + command_args)
    process = _run_git_command(path, command_args)
    if process.returncode == 0:
        modified_files = []
        for line in process.stdout.splitlines():
            # Y = status in work tree, X = status in index
            # We care about Y for unstaged modifications and untracked files
            # M, D, A, U, C, R in Y position means unstaged changes
            # ?? means untracked
            if len(line) >= 2 and (line[1] in ['M', 'D', 'A', 'U', 'C', 'R'] or line.startswith('??')):
                # For untracked files, it's '?? file.txt'
                # For modified, it's ' M file.txt' (space before M)
                modified_files.append(line[3:].strip())
        return GitResult(value=modified_files, original_command=full_command_str)
    else:
        return GitResult(value=[], error=process.stderr.strip(), original_command=full_command_str)

def add_files(path: str, filepaths: List[str]) -> GitResult[None]:
    if not filepaths:
        return GitResult(value=None, original_command="git add (no files specified)")

    command_args = ["add", "--"] + filepaths
    full_command_str = shlex.join(["git"] + command_args)
    process = _run_git_command(path, command_args)
    if process.returncode == 0:
        return GitResult(value=None, original_command=full_command_str)
    else:
        return GitResult(value=None, error=process.stderr.strip(), original_command=full_command_str)

def commit(path: str, message: str) -> GitResult[Optional[str]]:
    command_args = ["commit", "-m", message]
    full_command_str = shlex.join(["git"] + command_args)
    process = _run_git_command(path, command_args)
    if process.returncode == 0:
        # Commit successful, now get the hash
        commit_hash_result = get_current_commit_hash(path)
        if commit_hash_result.is_ok():
            return GitResult(value=commit_hash_result.value, original_command=full_command_str)
        else:
            return GitResult(value=None, error=commit_hash_result.error or "Commit successful but could not retrieve hash.", original_command=full_command_str)
    else:
        # Check for "nothing to commit" specifically
        if "nothing to commit" in process.stdout or "nothing to commit" in process.stderr:
            return GitResult(value=None, error="Nothing to commit.", original_command=full_command_str)
        return GitResult(value=None, error=process.stderr.strip(), original_command=full_command_str)

def diff(path: str, commit1: str, commit2: Optional[str] = None) -> GitResult[str]:
    command_args = ["diff", commit1]
    if commit2:
        command_args.append(commit2)
    full_command_str = shlex.join(["git"] + command_args)
    process = _run_git_command(path, command_args)
    if process.returncode == 0:
        return GitResult(value=process.stdout, original_command=full_command_str)
    else:
        return GitResult(value="", error=process.stderr.strip(), original_command=full_command_str)

def log(path: str, branch: Optional[str] = None, limit: int = 10) -> GitResult[List[Dict[str, str]]]:
    format_string = "%H%n%an%n%ae%n%ad%n%s" # Hash, Author Name, Author Email, Author Date, Subject
    command_args = ["log", f"--pretty=format:{format_string}", f"--max-count={limit}"]
    if branch:
        command_args.append(branch)
    
    full_command_str = shlex.join(["git"] + command_args)
    process = _run_git_command(path, command_args)
    if process.returncode == 0:
        commits = []
        log_entries = process.stdout.strip().split("\n")
        # Each commit will have 5 lines (hash, name, email, date, subject)
        # Ensure we have a complete set of 5 lines for each commit
        for i in range(0, len(log_entries), 5):
            if i + 4 < len(log_entries): 
                commit_data = {
                    "hash": log_entries[i],
                    "author_name": log_entries[i+1],
                    "author_email": log_entries[i+2],
                    "author_date": log_entries[i+3],
                    "subject": log_entries[i+4],
                }
                commits.append(commit_data)
            elif log_entries[i:]: # If there are partial entries at the end, it's an error or unexpected output
                logger.warning(f"Partial git log entry found. Skipping. Remaining lines: {log_entries[i:]}")
        return GitResult(value=commits, original_command=full_command_str)
    else:
        return GitResult(value=[], error=process.stderr.strip(), original_command=full_command_str)

def get_repo_name(path: str) -> GitResult[Optional[str]]:
    """
    Returns the name of the Git repository, which is typically the name of its root directory.
    """
    is_repo_result = is_git_repo(path)
    if is_repo_result.is_err() or not is_repo_result.value:
        return GitResult(value=None, error=is_repo_result.error or "Not a Git repository.", original_command=is_repo_result.original_command)
    
    # The repository name is the basename of the project root directory
    repo_name = os.path.basename(path)
    return GitResult(value=repo_name, original_command=f"os.path.basename({path})")

def checkout_branch(path: str, branch_name: str) -> GitResult[None]:
    """
    Checks out a specified Git branch.
    """
    command_args = ["checkout", branch_name]
    full_command_str = shlex.join(["git"] + command_args)
    process = _run_git_command(path, command_args)
    if process.returncode == 0:
        return GitResult(value=None, original_command=full_command_str)
    else:
        return GitResult(value=None, error=process.stderr.strip(), original_command=full_command_str)

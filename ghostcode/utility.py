import os
from datetime import datetime, timezone
import time
import functools

# A comprehensive mapping of file extensions to their associated programming languages.
# This dictionary is designed to provide common language identifiers suitable for
# Markdown code blocks (e.g., ```python, ```javascript).
# The keys are lowercase extensions (without the leading dot).
EXTENSION_TO_LANGUAGE_MAP = {
    # Python
    "py": "python",
    "pyw": "python",
    "pyc": "python",

    # JavaScript/TypeScript
    "js": "javascript",
    "jsx": "javascript",
    "mjs": "javascript",
    "cjs": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "vue": "vue",
    "svelte": "svelte",

    # Web (Markup/Style)
    "html": "html",
    "htm": "html",
    "css": "css",
    "scss": "scss",
    "sass": "sass",
    "less": "less",
    "xml": "xml",
    "svg": "xml", # SVG is XML-based

    # C-family
    "c": "c",
    "h": "c", # C header
    "cpp": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "hpp": "cpp", # C++ header
    "hh": "cpp", # C++ header
    "ino": "cpp", # Arduino sketches are C++
    "cs": "csharp",
    "java": "java",
    "go": "go",
    "rs": "rust",
    "swift": "swift",
    "kt": "kotlin",
    "kts": "kotlin",
    "m": "objectivec", # Objective-C
    "mm": "objectivec", # Objective-C++
    "fs": "fsharp",
    "fsi": "fsharp",
    "fsx": "fsharp",
    "v": "verilog", # Verilog (also Vlang, but Verilog is more common for .v in general code context)
    "sv": "systemverilog", # SystemVerilog
    "vhd": "vhdl",
    "vhdl": "vhdl",
    "d": "d",
    "zig": "zig",

    # Scripting/Shell
    "sh": "bash",
    "bash": "bash", # Though usually no extension, included for completeness
    "zsh": "bash", # Though usually no extension
    "ksh": "bash", # Though usually no extension
    "ps1": "powershell",
    "bat": "batch",
    "cmd": "batch",
    "rb": "ruby",
    "php": "php",
    "pl": "perl",
    "lua": "lua",
    "tcl": "tcl",

    # Data/Config
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "toml": "toml",
    "ini": "ini",
    "cfg": "ini", # General config files
    "conf": "ini", # General config files
    "env": "ini", # Environment variable files often use INI-like syntax

    # Functional/Academic
    "hs": "haskell",
    "lhs": "haskell",
    "ml": "ocaml",
    "mli": "ocaml",
    "sml": "sml",
    "clj": "clojure",
    "cljs": "clojure",
    "cljc": "clojure",
    "lisp": "lisp",
    "lsp": "lisp",
    "scm": "scheme",
    "rkt": "racket",
    "elm": "elm",
    "ex": "elixir",
    "exs": "elixir",
    "erl": "erlang",
    "jl": "julia",
    "r": "r",
    "f": "fortran",
    "for": "fortran",
    "f90": "fortran",
    "f95": "fortran",
    "f03": "fortran",
    "f08": "fortran",
    "ada": "ada",
    "adb": "ada",
    "ads": "ada",
    "agda": "agda",
    "idr": "idris",
    "lean": "lean",
    "coq": "coq",

    # Other
    "sql": "sql",
    "md": "markdown",
    "markdown": "markdown",
    "txt": "text",
    "log": "text",
    "csv": "csv",
    "tsv": "tsv",
    "asm": "assembly",
    "s": "assembly", # Assembly source
    "diff": "diff",
    "patch": "diff",
    "tex": "latex",
    "latex": "latex",
    "rst": "rst", # reStructuredText
    "gemfile": "ruby", # Common file name, but often treated as an extension for language detection
    "dockerfile": "dockerfile", # Common file name, but often treated as an extension for language detection
}

def language_from_extension(filepath: str) -> str:
    """
    Determines the most likely programming language associated with a file
    based on its extension.

    This function is designed to output language identifiers suitable for
    Markdown code blocks (e.g., "python", "javascript").

    Args:
        filepath (str): The path to the file (e.g., "/var/test.py", "README.md").

    Returns:
        str: The most likely associated language as a lowercase string.
             Returns "text" if no specific language is found for the extension,
             or if the file has no extension.
    """
    # Extract the file extension
    # os.path.splitext returns a tuple: (root, ext)
    # e.g., "/var/test.py" -> ("/var/test", ".py")
    # "README.md" -> ("README", ".md")
    # "Makefile" -> ("Makefile", "")
    _, ext = os.path.splitext(filepath)

    # Remove the leading dot from the extension and convert to lowercase
    # This ensures case-insensitivity (e.g., ".PY" becomes "py")
    # and handles cases where there's no extension (e.g., "" remains "")
    clean_ext = ext.lstrip('.').lower()

    # Handle specific common filenames that don't have traditional extensions
    # but are strongly associated with a language.
    # This is an addition to the extension-based logic for better coverage.
    filename = os.path.basename(filepath).lower()
    if filename == "dockerfile":
        return "dockerfile"
    if filename == "makefile":
        return "makefile" # Or "shell" or "bash" depending on preference
    if filename == "gemfile":
        return "ruby"
    if filename.startswith(".bash") or filename.startswith(".zsh") or filename.startswith(".profile"):
        return "bash"

    # Look up the cleaned extension in our map.
    # If the extension is not found, default to "text".
    return EXTENSION_TO_LANGUAGE_MAP.get(clean_ext, "text")

def timestamp_now_iso8601(timespec="seconds") -> str:
    """Returns an ISO8601 compliant representation of the point in time the function was called.
    The string representation is guaranteed to include the optional timezone string at the end (e.g. '+02:00' for CET). The timezone will be the system's local timezone.
    By default, the time of day will have a precision up to seconds. You can extend this with the timespec parameter. See the datetime.datetime.isoformat method documentation for more."""
    # it is ridiculous how hard this thing was to find for python
    return datetime.now(datetime.now().astimezone().tzinfo).isoformat(timespec=timespec)

def levenshtein(a: str, b: str) -> int:
    """
    Computes the Levenshtein distance between two strings using dynamic programming.

    Parameters:
        a (str): The first input string.
        b (str): The second input string.

    Returns:
        int: The minimum number of single-character edits (insert, delete, substitute)
             required to transform string `a` into string `b`.
    """
    # Get the lengths of the two strings
    m = len(a)
    n = len(b)

    # Create a 2D DP table with (m+1) rows and (n+1) columns
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the first row and column
    for i in range(m + 1):
        dp[i][0] = i  # Deletions needed to empty string b
    for j in range(n + 1):
        dp[0][j] = j  # Insertions needed to empty string a

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # If characters match, no cost is added
            if a[i - 1] == b[j - 1]:
                cost = 0
            else:
                cost = 1  # Substitution cost

            # Compute the minimum of the three possible operations
            dp[i][j] = min(
                dp[i - 1][j] + 1,     # Deletion from a
                dp[i][j - 1] + 1,     # Insertion into a
                dp[i - 1][j - 1] + cost  # Substitution or no cost
            )

    # The bottom-right cell contains the Levenshtein distance
    return dp[m][n]

def time_function_with_logging(func):
    """
    A decorator that times a function's execution and logs the duration
    using the custom 'TIMING' log level.
    """
    @functools.wraps(func) # Preserves the original function's metadata (name, docstring, etc.)
    def wrapper(*args, **kwargs):
        # Get a logger for the module where the decorated function is defined
        logger = logging.getLogger(func.__module__)

        start_time = time.perf_counter()
        result = func(*args, **kwargs) # Execute the original function
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        logger.timing(f"{func.__name__} executed in {elapsed_time:.2f} seconds.")

        return result
    return wrapper


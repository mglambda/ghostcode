from typing import *
import os
from enum import Enum
from datetime import datetime, timezone
import time
from pydantic import BaseModel, Field
import functools
import logging
import yaml

# fixing python's reduce
A = TypeVar("A") # accumulator
B = TypeVar("B") # values

def foldl(f: Callable[[B, A], A], initial_accumulator_value: A, xs: Iterable[B]) -> A:
    acc = initial_accumulator_value
    for x in xs:
        acc = f(x, acc)
    return acc
    
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

# Define a threshold for "long" strings.
# Strings longer than this OR containing newlines will be literal block style (|).
LONG_STRING_THRESHOLD = 50

def conditional_string_representer(dumper, data):
    """
    Custom representer for strings:
    - Forces multi-line or long strings (over THRESHOLD) to literal block style (|).
    - For short, single-line strings, lets PyYAML choose the most appropriate style
      (usually plain scalar, or quoted if necessary).
    """
    if '\n' in data or len(data) > LONG_STRING_THRESHOLD:
        # For multi-line or long strings, force literal block style (|).
        # This style preserves newlines and leading/trailing spaces exactly.
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    else:
        # For short, single-line strings, let PyYAML decide.
        # An empty string '' for style means "plain if possible, otherwise quoted".
        # This prevents short strings from being '|' or '|-'.
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='')

class CustomConditionalDumper(yaml.Dumper):
    """
    A custom YAML Dumper that applies conditional string representation.
    Also ensures block style for mappings/sequences and unquoted keys by default.
    """
    # You can add more configurations here if needed.
    # For example, to ensure block style for mappings and sequences:
    # This is often set in the yaml.dump() call, but can be set here too.
    # default_flow_style = False


CustomConditionalDumper.add_representer(str, conditional_string_representer)


# this dumper is used in the Program class to serialize config files
# This representer will dump Enum members as their underlying value (int, str, etc.)
def enum_value_representer(dumper, data):
    """
    Represents an Enum member as its value.
    """
    # PyYAML will automatically determine the best scalar style for data.value
    # (e.g., plain for int/str, quoted if necessary).
    return dumper.represent_data(data.value)

# Create a custom Dumper class for this specific behavior
class PydanticEnumDumper(yaml.SafeDumper):
    """
    A custom YAML SafeDumper that represents Enum members by their value.
    """
    pass

# Register this representer for the base Enum class with our custom Dumper.
PydanticEnumDumper.add_representer(Enum, enum_value_representer)

def show_model(obj: BaseModel, heading: str ="", abridge: Optional[int] = 80) -> str:
    """Generic way to pretty print a pydantic model.
    If the abridge parameter is set to a positive integer, strings found in the object will be abridged to show a maximum value that is equal to the abridge value (with abridgement happening in the middle, showing beginning and end of that string equally). If it is None no abridgement will take place."""


    data = obj.model_dump()

    if abridge is not None and abridge >= 0:
        for k in data.keys():
            v = data[k]
            if isinstance(v, str):
                if len(v) > abridge:
                    h = abridge // 2
                    if "\n" in v:
                        data[k] = v[0:h] + "\n ... \n" + v[len(v) - h:]
                    else:
                        data[k] = v[0:h] + " … " + v[len(v) - h:]
                #data[k] = literal_str(data[k])
                          
    heading_str = f"[{heading}]\n" if heading else ""
    return heading_str + yaml.dump(
        data,
        Dumper=CustomConditionalDumper,
        default_flow_style=False,
        sort_keys=False)


def _show_model(object: BaseModel) -> str:
    """Returns a string representation of a pydantic model in YAML format."""
    return yaml.dump(
        object.model_dump(),
        sort_keys=False,
        default_flow_style=False,
    )


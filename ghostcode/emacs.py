# emacs.py
from ghostcode import types
from typing import *
from pydantic import BaseModel, Field, BeforeValidator, ValidationError
import subprocess
import json

from typing import *

def emacs_str_validator(x: any) -> Any:
    if x is None:
        return ""
    if isinstance(x, bool):
        if x:
            return "true"
        else:
            return ""
    return x

emacs_str = Annotated[str, BeforeValidator(emacs_str_validator)]


def emacs_bool_validator(x: Any) -> Any:
    if not(x):
        return False
    return x

emacs_bool = Annotated[bool, BeforeValidator(emacs_bool_validator)]

class Buffer(BaseModel):
    """
    Represents an Emacs buffer with key information.
    """
    name: emacs_str
    file_name: emacs_str = ""
    modified: emacs_str = ""
    read_only: emacs_bool                 = False
    major_mode: emacs_str = ""
    mode_name: emacs_str = ""
    default_directory: emacs_str = ""

    
def send_code(elisp: str) -> Optional[str]:
    try:
        # Execute emacsclient command
        p = subprocess.run(
            ["emacsclient", "-e", elisp],
            capture_output=True,
            text=True,
            check=True, # Raise CalledProcessError if the command returns a non-zero exit code
            timeout=10  # Add a timeout to prevent hanging
        )

        return p.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error calling emacsclient. Command: {' '.join(e.cmd)}")
        print(f"Return Code: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        print("\n--- Troubleshooting ---")
        print("1. Ensure an Emacs server is running: In Emacs, run `M-x server-start`.")
        print("2. Ensure 'emacsclient' is in your system's PATH.")
        print("3. Check if the Emacs server is accessible (e.g., `emacsclient -s <socket-name> -e ...`).")
        return None
    except FileNotFoundError:
        print("Error: 'emacsclient' command not found. Is Emacs installed and in your PATH?")
        return None
    except subprocess.TimeoutExpired:
        print("Error: emacsclient command timed out after 10 seconds.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None




def json_loads_fixed(json_string: str) -> Any:
    """Remove double quotes from emcas json output and return a parsed object."""
    
    if not json_string.strip():
        # Handle empty or whitespace-only input gracefully, e.g., return an empty list
        # if the expected top-level structure is a list of objects.
        return []

    data = json.loads(json.loads(json_string))
    return data

def get_buffers():
    elisp = """
    (progn
    (require 'cl-lib)
    (require 'json)
    (json-encode
    (mapcar (lambda (buf)
    (with-current-buffer buf
    (list
    (cons 'name (buffer-name buf))
                     (cons 'file_name (buffer-file-name buf))
                 (cons 'read-only buffer-read-only)
                 (cons 'modified (buffer-modified-p buf))        
;;                 (cons 'major_mode major-mode)
;;                 (cons 'mode_name mode-name)
;;                 (cons 'default_directory default-directory)
)))
    (buffer-list)
    )
    ))
    """

    if (r := send_code(elisp)) is None:
        return []

    return [Buffer(**data) for data in json_loads_fixed(r.strip())]
    
class EmacsState(BaseModel):
    pass

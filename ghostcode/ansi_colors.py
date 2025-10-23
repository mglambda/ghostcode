import os
import sys
from enum import IntEnum

# --- ANSI Escape Codes ---
ESC = "\033"
CSI = f"{ESC}["
RESET = f"{CSI}0m"
BOLD = f"{CSI}1m"
ITALIC = f"{CSI}3m"
UNDERLINE = f"{CSI}4m"
BLINK = f"{CSI}5m" # Often not supported or annoying
INVERSE = f"{CSI}7m"
STRIKETHROUGH = f"{CSI}9m"

# 256-color foreground/background prefixes
FOREGROUND_256 = f"{CSI}38;5;"
BACKGROUND_256 = f"{CSI}48;5;"

# --- 256-Color Enum (Partial) ---
# This enum provides names for some common and useful 256-colors.
# You can extend this with any of the 0-255 indices you find useful.
class Color256(IntEnum):
    # Grayscale (0-231 are standard colors, 232-255 are grayscale)
    BLACK = 0
    WHITE = 15 # A bright white, 231 is also a common "true white"
    GRAY_DARKEST = 232
    GRAY_DARK = 235
    GRAY_MEDIUM = 240
    GRAY_LIGHT = 248
    GRAY_LIGHTEST = 255

    # Standard 8/16 colors (approximate mapping to 256-palette)
    RED = 1
    GREEN = 2
    YELLOW = 3
    BLUE = 4
    MAGENTA = 5
    CYAN = 6

    # Specific 256-color palette entries (often more vibrant or distinct)
    ORANGE = 208 # A good, vibrant orange
    DEEP_ORANGE = 166
    LIGHT_ORANGE = 214 # More golden/yellow-orange
    LIME_GREEN = 118
    DEEP_SKY_BLUE = 39
    PURPLE = 93
    PINK = 205
    TEAL = 30
    BROWN = 130
    GOLD = 220
    CRIMSON = 160
    INDIGO = 54
    VIOLET = 129
    SPRING_GREEN = 48
    TURQUOISE = 44
    SLATE_BLUE = 61
    ROYAL_BLUE = 27
    FOREST_GREEN = 22
    DARK_RED = 124
    DARK_GREEN = 28
    DARK_BLUE = 18
    DARK_CYAN = 30
    DARK_MAGENTA = 90
    DARK_YELLOW = 136
    TRUE_WHITE = 231 # Often used for a very bright white

# --- Wrapper Function ---
def colored(
    text: str,
    foreground_color: int | Color256,
    background_color: int | Color256 | None = None,
    bold: bool = False,
    italic: bool = False,
    underline: bool = False,
    inverse: bool = False,
    strikethrough: bool = False,
    blink: bool = False, # Use with caution, can be annoying
) -> str:
    """
    Applies 256-color ANSI formatting to a string.

    Args:
        text (str): The text to color.
        foreground_color (int | Color256): The 256-color index (0-255) or a Color256 enum member.
        background_color (int | Color256 | None): Optional 256-color index for background.
        bold (bool): Apply bold style.
        italic (bool): Apply italic style.
        underline (bool): Apply underline style.
        inverse (bool): Swap foreground and background colors.
        strikethrough (bool): Apply strikethrough style.
        blink (bool): Apply blink style (often not supported or ignored).

    Returns:
        str: The formatted string with ANSI escape codes.
    """
    # Ensure color codes are valid integers
    fg_code = int(foreground_color)
    if not (0 <= fg_code <= 255):
        raise ValueError(f"Foreground color code must be between 0 and 255, got {fg_code}.")

    parts = []

    # Add styles
    if bold: parts.append(BOLD)
    if italic: parts.append(ITALIC)
    if underline: parts.append(UNDERLINE)
    if inverse: parts.append(INVERSE)
    if strikethrough: parts.append(STRIKETHROUGH)
    if blink: parts.append(BLINK)

    # Add foreground color
    parts.append(f"{FOREGROUND_256}{fg_code}m")

    # Add background color if specified
    if background_color is not None:
        bg_code = int(background_color)
        if not (0 <= bg_code <= 255):
            raise ValueError(f"Background color code must be between 0 and 255, got {bg_code}.")
        parts.append(f"{BACKGROUND_256}{bg_code}m")

    # Combine parts and add text, then reset
    return "".join(parts) + text + RESET

# --- Helper for Windows (optional, for older terminals) ---
# On modern Windows terminals (Windows Terminal, VS Code terminal),
# ANSI codes work natively. For older cmd.exe, colorama might be needed.
def enable_ansi_on_windows() -> None:
    if os.name == 'nt':
        try:
            import colorama
            colorama.init()
        except ImportError:
            print("Warning: 'colorama' not found. ANSI colors might not work on older Windows terminals.", file=sys.stderr)

# Call this once at the start of your application if you need Windows compatibility
# enable_ansi_on_windows()

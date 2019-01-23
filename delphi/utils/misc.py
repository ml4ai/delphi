import platform
from delphi.utils.fp import grouper

def choose_font():
    operating_system = platform.system()

    if operating_system == "Darwin":
        font = "Gill Sans"
    elif operating_system == "Windows":
        font = "Candara"
    else:
        font = "Ubuntu"

    return font

def _insert_line_breaks(label: str, max_str_length: int = 20) -> str:
    words = label.split()
    if len(label) > max_str_length:
        n_groups = len(label) // max_str_length
        n = len(words) // n_groups
        return "\n".join(
            [" ".join(word_group) for word_group in grouper(words, n, "")]
        )
    else:
        return label

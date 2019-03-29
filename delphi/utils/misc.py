import re
import platform
from typing import Dict
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


def multiple_replace(d: Dict[str, str], text: str) -> str:
    """ Performs string replacement from dict in a single pass. Taken from
    https://www.oreilly.com/library/view/python-cookbook/0596001673/ch03s15.html
    """
  # Create a regular expression from all of the dictionary keys
    regex = re.compile("|".join(map(re.escape, d.keys())))

  # For each match, look up the corresponding value in the dictionary
    return regex.sub(lambda match: d[match.group(0)], text)

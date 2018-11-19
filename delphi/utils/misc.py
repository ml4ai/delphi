from delphi.utils.fp import grouper


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

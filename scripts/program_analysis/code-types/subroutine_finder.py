import os
import re
import pickle

from tqdm import tqdm


def main():
    root_dir = "/Users/phein/Documents/ml4ai/Automates/FortranCodebases/dssat-csm"
    Fortran_files = find_fortran_files(root_dir)
    print("Total # Fortran files:", len(Fortran_files))

    all_subroutines = dict()
    for path, filename in tqdm(Fortran_files, desc="Processing Files"):
        filepath = os.path.join(path, filename)
        fortran_lines = get_fortran_lines(filepath)
        code_only_lines = remove_comments_and_blanks(fortran_lines)
        no_continuation_lines = remove_continuations(code_only_lines)
        new_subroutines = find_subroutines(no_continuation_lines)

        shortpath = path[path.find("dssat-csm"):]
        for sub_name, sub_code in new_subroutines.items():
            all_subroutines[(shortpath, filename, sub_name)] = sub_code

    print("Total # of subroutines:", len(all_subroutines))
    pickle.dump(all_subroutines, open("Fortran_subroutines.pkl", "wb"))


def find_fortran_files(root_directory):
    """Recursively finds all Fortran files under a given directory."""
    return [
        (path, filename)
        for path, _, files in os.walk(root_directory) for filename in files
        if filename.endswith(".for") or filename.endswith(".f90")
    ]


def find_subroutines(code_lines):
    """Return a dictionary of all subroutines in the file. A subroutine will be
    represented as a list of strings that includes a string for each line in the
    subroutine."""
    subroutines = dict()
    sub_info = list()
    for i, line in enumerate(code_lines):
        clean_line = line.lstrip().upper()
        words = [w for w in re.split(r"\W", clean_line) if w]
        if len(words) == 0:
            continue
        if words[0] == "SUBROUTINE" or words[0] == "FUNCTION":
            sub_info.append((i, words[1]))
        elif len(words) > 2 and words[1] == "FUNCTION":
            sub_info.append((i, words[2]))
        elif words[0] == "END":
            if len(words) == 1 or words[1] == "SUBROUTINE" or words[1] == "FUNCTION":
                if len(sub_info) == 0:
                    continue
                (sub_start, cur_name) = sub_info.pop(-1)
                if i == len(code_lines)-1:
                    subroutines[cur_name] = code_lines[sub_start:]
                else:
                    subroutines[cur_name] = code_lines[sub_start:i+1]

    return subroutines


def get_fortran_lines(fpath):
    """Returns a list of all lines from the Fortran file pointed to by fpath."""
    fortran_lines = list()
    with open(fpath, "rb") as infile:
        for line in infile.readlines():
            ascii_line = line.decode("ascii", errors="replace").rstrip()
            fortran_lines.append(ascii_line)
    return fortran_lines


def remove_comments_and_blanks(code_lines):
    """Removes all full and inline comments as well as any blank lines."""
    good_lines = list()
    for line in code_lines:
        no_whitespace_line = re.sub(r"\s", "", line)
        if len(no_whitespace_line) < 1:
            continue
        comment_chars = ["!", "*", "c", "C", "d", "D"]
        if any([line.startswith(sym) for sym in comment_chars]):
            continue
        clean_line = line.lstrip()
        comment_chars = ["!", "*"]
        if any([clean_line.startswith(sym) for sym in comment_chars]):
            continue

        ex_idx = line.rfind("!")
        apos_idx = line.rfind("'")
        quote_idx = line.rfind('"')
        if ex_idx >= 0 and ex_idx > apos_idx and ex_idx > quote_idx:
            line = line[:ex_idx].rstrip()
        good_lines.append(line)

    return good_lines


def remove_continuations(code_lines):
    """Combines all lines with continuations into a single line."""
    chg = True
    while chg:
        chg = False
        for i in range(len(code_lines)):
            if i == len(code_lines):
                break

            line = code_lines[i]
            clean_start_line = line.lstrip()
            if clean_start_line[0] == "&":
                prevline = code_lines[i - 1]
                continued_line = clean_start_line[1:].lstrip()
                code_lines[i - 1] = prevline + " " + continued_line
                code_lines.pop(i)
                chg = True
            elif clean_start_line[-1] == "&":
                nextline = code_lines[i + 1].lstrip()
                curr_line = clean_start_line[:-1]
                code_lines[i] = curr_line + " " + nextline
                code_lines.pop(i+1)
                chg = True
    return code_lines


if __name__ == '__main__':
    main()

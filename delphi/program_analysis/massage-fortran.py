#!/usr/bin/env python3
import os
import sys
from tqdm import tqdm


def process(fname):
    lines = list()
    total_comments = 0
    total_code = 0
    total_inline_comments = 0
    with open(fname, "rb") as infile:
        for line in infile.readlines():
            ascii_line = line.decode("ascii", errors="replace")
            trimmed_line = ascii_line.rstrip()

            # remove empty lines
            if len(trimmed_line) < 1:
                continue

            # remove lines that are entirely comments
            if trimmed_line.startswith(" ") or trimmed_line.startswith("\t"):
                lines.append(trimmed_line)
            else:
                total_comments += 1

    # remove partial-line comments
    for i in range(len(lines)):
        line = lines[i]
        ex_idx = line.rfind("!")
        apos_idx = line.rfind("'")
        quote_idx = line.rfind('"')
        if ex_idx >= 0:
            if ex_idx > apos_idx and ex_idx > quote_idx:
                total_inline_comments += 1
                lines[i] = line[:ex_idx]

    lines = [line for line in lines if len(line.lstrip()) > 0]

    # remove continuation lines
    chg = True
    while chg:
        chg = False
        for i in range(len(lines)):
            # len(lines) may have changed because of deletions
            if i == len(lines):
                break

            line = lines[i]
            if line.lstrip()[0] == "&":
                # if not line[0] == "\t" and line[5] != ' ':    # continuation character
                prevline = lines[i - 1]
                line = line.lstrip()[1:].lstrip()
                # line = line[6:].lstrip()
                prevline = prevline.rstrip() + line
                lines[i - 1] = prevline
                lines.pop(i)
                chg = True

    total_code = len(lines)
    return (total_code, total_comments, total_inline_comments)

    # outline = '\n'.join(lines)
    # new_fortran_path = "/Users/phein/ml4ai/program_analysis/massaged-dssat-csm/"
    #
    # shortpath = fname[fname.rfind("dssat-csm/") + 10: fname.rfind("/")]
    # folder_path = os.path.join(new_fortran_path, shortpath)
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)
    #
    # shortname = fname[fname.rfind("/") + 1:]
    # with open(os.path.join(folder_path, shortname), "wb") as outfile:
    #     outfile.write(outline.encode("utf-8"))


def main():
    codebase_path = sys.argv[1]
    files = [
        os.path.join(root, elm)
        for root, dirs, files in os.walk(codebase_path)
        for elm in files
    ]

    fortran_files = [x for x in files if x.endswith(".for") or x.endswith(".f90")]

    tot_code = 0
    tot_comm = 0
    tot_in_comm = 0
    for fname in tqdm(fortran_files, desc="Processing Fortran"):
        (code, comm, in_comm) = process(fname)
        tot_code += code
        tot_comm += comm
        tot_in_comm += in_comm

    print("Resultant counts:")
    print("Total lines of code: {}".format(tot_code))
    print("Total lines of comment: {}".format(tot_comm))
    print("Total # of inline comments: {}".format(tot_in_comm))
    print("Total lines of code w/o comment (tCode - tInLineComm): {}".format(tot_code - tot_in_comm))


if __name__ == "__main__":
    main()

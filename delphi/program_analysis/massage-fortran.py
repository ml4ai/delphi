#!/usr/bin/env python3
import os
import sys
from tqdm import tqdm


def process(fname):
    lines = list()
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

    # remove partial-line comments
    for i in range(len(lines)):
        line = lines[i]
        ex_idx = line.rfind('!')
        apos_idx = line.rfind("'")
        quote_idx = line.rfind('"')
        if ex_idx >= 0:
            if ex_idx > apos_idx and ex_idx > quote_idx:
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
                prevline = lines[i-1]
                line = line.lstrip()[1:].lstrip()
                # line = line[6:].lstrip()
                prevline = prevline.rstrip() + line
                lines[i-1] = prevline
                lines.pop(i)
                chg = True

    outline = '\n'.join(lines)
    new_fortran_path = "/Users/phein/ml4ai/program_analysis/massaged-dssat-csm/"

    shortpath = fname[fname.rfind("dssat-csm/") + 10:fname.rfind("/")]
    folder_path = os.path.join(new_fortran_path, shortpath)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    shortname = fname[fname.rfind("/") + 1:]
    with open(os.path.join(folder_path, shortname), "wb") as outfile:
        outfile.write(outline.encode("utf-8"))


def main():
    codebase_path = sys.argv[1]
    files = [os.path.join(root, elm)
             for root, dirs, files in os.walk(codebase_path) for elm in files]

    fortran_files = [x for x in files if x.endswith(".for")]

    for fname in tqdm(fortran_files, desc="Processing Fortran"):
        process(fname)


if __name__ == '__main__':
    main()

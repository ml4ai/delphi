import sys
import os
import subprocess
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def main():
    codebase_path = sys.argv[1]
    files = [
        os.path.join(root, elm)
        for root, dirs, files in os.walk(codebase_path)
        for elm in files
    ]

    fortran_files = [x for x in files if x.endswith(".for")]

    num_parsed = 0
    num_failed = 0
    failed_modules = dict()
    parsed_modules = list()
    p = Pool(cpu_count())
    res = list(
        tqdm(p.imap(attempt_parse, fortran_files), total=len(fortran_files))
    )
    for (ecode, errors, mod_name) in res:
        if ecode == 1:
            num_failed += 1
            failed_modules[mod_name] = errors
        else:
            num_parsed += 1
            parsed_modules.append(mod_name)

    with open("fortran_parsing_results.txt", "w+") as outfile:
        outfile.write(
            "Num parsed: {}\nNum failed: {}\n\n".format(num_parsed, num_failed)
        )

        outfile.write("CORRECTLY PARSED MODULES:\n\n")
        for mod_name in parsed_modules:
            outfile.write("\t{}\n".format(mod_name))
        outfile.write("\n")

        for mod_name, errors in failed_modules.items():
            outfile.write("\nRECORDED ERRORS FOR: {}\n".format(mod_name))
            for err in errors:
                outfile.write("\t{}\n".format(err))


def attempt_parse(fpath):
    dssat_idx = fpath.find("dssat-csm/")
    module_name = fpath[dssat_idx + 10 : -4].replace("/", ".")
    proc = subprocess.run(
        ["./autoTranslate", fpath],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if proc.returncode == 0:
        return (0, [], module_name)

    err = proc.stderr.decode("ascii", errors="replace")
    actual_errors = [
        line
        for line in err.split("\n")
        if "Error" in line or ".for line " in line
    ]
    return (1, actual_errors, module_name)


if __name__ == "__main__":
    main()

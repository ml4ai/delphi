import pygraphviz as pgv
import sys
import os


def main():
    top_dir_path = sys.argv[1]
    right_idx = top_dir_path[:-1].rfind("/")
    top_dir = top_dir_path[right_idx + 1 : -1]

    ignore_builtins = True
    if len(sys.argv) > 2:
        ignore_builtins = bool(int(sys.argv[2]))

    modules = make_module_table(top_dir_path, ignore_builtins=ignore_builtins)
    # print_module_table(modules)

    subs = [sub for module in modules.values() for sub in module.keys()]
    # print("There are {} total subroutines and {} unique subroutines".format(len(subs), len(list(set(subs)))))
    non_unique_subs = list(set([sub for sub in subs if subs.count(sub) > 1]))
    subroutines = {
        (mod_name, sub): calls
        for mod_name, module in modules.items()
        for sub, calls in module.items()
    }

    g = pgv.AGraph(directed=True)
    all_edges = list()
    for idx, ((mod_name, sub_name), calls) in enumerate(subroutines.items()):
        if sub_name in non_unique_subs:
            sub_name = "{}.{}".format(mod_name, sub_name)
            print("SUB is now: {}".format(sub_name))

        for call in calls:
            if call in non_unique_subs:
                if call in modules[mod_name].keys():
                    call = "{}.{}".format(mod_name, sub_name)
                else:
                    correct_mod = module_lookup(modules, mod_name, call)
                    call = "{}.{}".format(correct_mod, sub_name)
                print("CALL is now: {}".format(call))

            g.add_edge(sub_name, call)
            all_edges.append((sub_name, call))

    outfile = "{}_call_graph".format(top_dir)
    if not ignore_builtins:
        outfile += "_with_builtins"
    outfile += ".dot"
    # g.draw("call-graph.png", prog="fdp")        # use fdp, dot, or circo
    g.write(outfile)

    with open("{}_edges.txt".format(top_dir), "w+") as txtfile:
        for (inc, out) in all_edges:
            txtfile.write("{}, {}\n".format(inc, out))


def make_module_table(codebase_path, ignore_builtins=True):
    files = [
        os.path.join(root, elm)
        for root, dirs, files in os.walk(codebase_path)
        for elm in files
    ]

    fortran_files = [x for x in files if x.endswith(".for")]

    comment_strs = ["!", "!!", "C"]

    modules = dict()
    for fpath in fortran_files:
        dssat_idx = fpath.find("dssat-csm/")
        module_name = fpath[dssat_idx + 10 : -4].replace("/", ".")

        subroutines = dict()
        with open(fpath, "rb") as ffile:
            cur_calls = list()
            cur_subroutine = None
            lines = ffile.readlines()
            for idx, line in enumerate(lines):
                text = line.decode("ascii", errors="replace")
                tokens = text.split()

                if (
                    len(tokens) <= 0
                    or tokens[0] in comment_strs
                    or tokens[0].startswith("!")
                ):
                    continue

                if tokens[0] == "SUBROUTINE" or tokens[0] == "FUNCTION":
                    subroutine_name = tokens[1]

                    paren_idx = subroutine_name.find("(")
                    if paren_idx != -1:
                        subroutine_name = subroutine_name[:paren_idx]

                    if cur_subroutine is not None:
                        subroutines[subroutine_name] = cur_calls

                    cur_calls = list()
                    cur_subroutine = subroutine_name

                if "CALL" in tokens:
                    call_idx = tokens.index("CALL")
                    if len(tokens) > call_idx + 1:
                        call_name = tokens[call_idx + 1]

                        paren = call_name.find("(")
                        if paren != -1:
                            call_name = call_name[:paren]

                        if ignore_builtins:
                            if call_name.lower() not in F_INTRINSICS:
                                cur_calls.append(call_name)
                        else:
                            cur_calls.append(call_name)

        modules[module_name] = subroutines
    return modules


def module_lookup(all_modules, curr_mod, func):
    parent_name = curr_mod[: curr_mod.rfind(".")]
    mods_to_check = [mod for mod in all_modules.keys() if parent_name in mod]

    for mod in mods_to_check:
        if func in all_modules[mod].keys():
            return mod

    return module_lookup(all_modules, parent_name, func)


def print_module_table(mod_table):
    for module, subroutines in mod_table.items():
        print("\nMODULE: {}".format(module))
        for sub_name, calls in subroutines.items():
            print("\tSUBROUTINE: {}".format(sub_name))
            if len(calls) > 0:
                print("\t\tCALLS:")
                for call in calls:
                    print("\t\t\t{}".format(call))


F_INTRINSICS = frozenset(
    [
        "abs",
        "abort",
        "access",
        "achar",
        "acos",
        "acosd",
        "acosh",
        "adjustl",
        "adjustr",
        "aimag",
        "aint",
        "alarm",
        "all",
        "allocated",
        "and",
        "anint",
        "any",
        "asin",
        "asind",
        "asinh",
        "associated",
        "atan",
        "atand",
        "atan2",
        "atan2d",
        "atanh",
        "atomic_add",
        "atomic_and",
        "atomic_cas",
        "atomic_define",
        "atomic_fetch_add",
        "atomic_fetch_and",
        "atomic_fetch_or",
        "atomic_fetch_xor",
        "atomic_or",
        "atomic_ref",
        "atomic_xor",
        "backtrace",
        "bessel_j0",
        "bessel_j1",
        "bessel_jn",
        "bessel_y0",
        "bessel_y1",
        "bessel_yn",
        "bge",
        "bgt",
        "bit_size",
        "ble",
        "blt",
        "btest",
        "c_associated",
        "c_f_pointer",
        "c_f_procpointer",
        "c_funloc",
        "c_loc",
        "c_sizeof",
        "ceiling",
        "char",
        "chdir",
        "chmod",
        "cmplx",
        "co_broadcast",
        "co_max",
        "co_min",
        "co_reduce",
        "co_sum",
        "command_argument_count",
        "compiler_options",
        "compiler_version",
        "complex",
        "conjg",
        "cos",
        "cosd",
        "cosh",
        "cotan",
        "cotand",
        "count",
        "cpu_time",
        "cshift",
        "ctime",
        "date_and_time",
        "dble",
        "dcmplx",
        "digits",
        "dim",
        "dot_product",
        "dprod",
        "dreal",
        "dshiftl",
        "dshiftr",
        "dtime",
        "eoshift",
        "epsilon",
        "erf",
        "erfc",
        "erfc_scaled",
        "etime",
        "event_query",
        "execute_command_line",
        "exit",
        "exp",
        "exponent",
        "extends_type_of",
        "fdate",
        "fget",
        "fgetc",
        "floor",
        "flush",
        "fnum",
        "fput",
        "fputc",
        "fraction",
        "free",
        "fseek",
        "fstat",
        "ftell",
        "gamma",
        "gerror",
        "getarg",
        "get_command",
        "get_command_argument",
        "getcwd",
        "getenv",
        "get_environment_variable",
        "getgid",
        "getlog",
        "getpid",
        "getuid",
        "gmtime",
        "hostnm",
        "huge",
        "hypot",
        "iachar",
        "iall",
        "iand",
        "iany",
        "iargc",
        "ibclr",
        "ibits",
        "ibset",
        "ichar",
        "idate",
        "ieor",
        "ierrno",
        "image_index",
        "index",
        "int",
        "int2",
        "int8",
        "ior",
        "iparity",
        "irand",
        "is_iostat_end",
        "is_iostat_eor",
        "isatty",
        "ishft",
        "ishftc",
        "isnan",
        "itime",
        "kill",
        "kind",
        "lbound",
        "lcobound",
        "leadz",
        "len",
        "len_trim",
        "lge",
        "lgt",
        "link",
        "lle",
        "llt",
        "lnblnk",
        "loc",
        "log",
        "log10",
        "log_gamma",
        "logical",
        "long",
        "lshift",
        "lstat",
        "ltime",
        "malloc",
        "maskl",
        "maskr",
        "matmul",
        "max",
        "maxexponent",
        "maxloc",
        "maxval",
        "mclock",
        "mclock8",
        "merge",
        "merge_bits",
        "min",
        "minexponent",
        "minloc",
        "minval",
        "mod",
        "modulo",
        "move_alloc",
        "mvbits",
        "nearest",
        "new_line",
        "nint",
        "norm2",
        "not",
        "null",
        "num_images",
        "or",
        "pack",
        "parity",
        "perror",
        "popcnt",
        "poppar",
        "precision",
        "present",
        "product",
        "radix",
        "ran",
        "rand",
        "random_number",
        "random_seed",
        "range",
        "rank ",
        "real",
        "rename",
        "repeat",
        "reshape",
        "rrspacing",
        "rshift",
        "same_type_as",
        "scale",
        "scan",
        "secnds",
        "second",
        "selected_char_kind",
        "selected_int_kind",
        "selected_real_kind",
        "set_exponent",
        "shape",
        "shifta",
        "shiftl",
        "shiftr",
        "sign",
        "signal",
        "sin",
        "sind",
        "sinh",
        "size",
        "sizeof",
        "sleep",
        "spacing",
        "spread",
        "sqrt",
        "srand",
        "stat",
        "storage_size",
        "sum",
        "symlnk",
        "system",
        "system_clock",
        "tan",
        "tand",
        "tanh",
        "this_image",
        "time",
        "time8",
        "tiny",
        "trailz",
        "transfer",
        "transpose",
        "trim",
        "ttynam",
        "ubound",
        "ucobound",
        "umask",
        "unlink",
        "unpack",
        "verify",
        "xor",
    ]
)


if __name__ == "__main__":
    main()

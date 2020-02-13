import pickle
import re
import csv

from tqdm import tqdm

MATH_FUNCS = [
    "EXP", "MAX", "MIN", "LOG", "ALOG", "SQRT", "SIN", "COS", "TAN",
    "ACOS", "ASIN", "ATAN", "AMIN1", "AMAX1", "MAX0"
]

FIELDS = [
    "Prefix",
    "Filename",
    "Subroutine",
    "Class",
    "Length",
    "Max stack depth",
    "# of inputs",
    "# of outputs",
    "# math exprs",
    "# of loops",
    "# of conditionals",
    "# of select/cases",
    "# of calls"
]


def main():
    subroutines = pickle.load(open("Fortran_subroutines.pkl", "rb"))
    sub_stats = dict()
    call_lists = dict()
    for (path, filename, sub_name), sub_code in tqdm(subroutines.items(), desc="Analyzing"):
        sub_header = [w for w in re.split(r"\W", sub_code[0].strip()) if w]
        input_list = sub_header[2:]
        output_list = list()
        sub_calls = list()
        mathematic_exprs = 0
        num_calls = 0
        num_conds = 0
        num_cases = 0
        num_loops = 0
        for code_line in sub_code:
            # NOTE: grabs only all spelled words
            words = [w for w in re.split(r"\W", code_line) if w]
            if len(words) < 1:
                print(code_line)
                continue
            if "=" in code_line:
                # Check if the assignment line is a mathematic expression
                ops = re.findall(r"\*\*|\*|\+|\-|/", code_line)
                has_ops = len(ops) >= 1

                has_math_func = any([sym in words for sym in MATH_FUNCS])
                if has_ops or has_math_func:
                    mathematic_exprs += 1

                if words[0] in input_list:
                    if words[0] not in output_list:
                        output_list.append(words[0])
            first_word = words[0]
            if first_word == "CALL":
                num_calls += 1
                if words[1] == sub_name:
                    print("Found recursive call:", words[1])
                sub_calls.append(words[1])
            elif first_word == "IF" or first_word == "ELSEIF" or first_word == "ELSE":
                num_conds += 1
            elif first_word == "DO" or first_word == "WHILE":
                num_loops += 1
            elif first_word == "SELECT" or first_word == "CASE":
                num_cases += 1

        module_prefix = path[path.find("/"):]
        module_file = filename[:filename.rfind(".")]
        module_path = "/".join([module_prefix, module_file])
        call_lists[(module_path, sub_name)] = list(set(sub_calls))
        sub_stats[(module_path, sub_name)] = {
            "Prefix": module_prefix,
            "Filename": module_file,
            "Subroutine": sub_name,
            "Class": 0,
            "Length": len(sub_code),
            "Max stack depth": 0,
            "# of inputs": len(input_list),
            "# of outputs": len(output_list),
            "# math exprs": mathematic_exprs,
            "# of loops": num_loops,
            "# of conditionals": num_conds,
            "# of select/cases": num_cases,
            "# of calls": num_calls
        }

    call_buckets = dict()
    for (path, sub_name) in call_lists.keys():
        if sub_name in call_buckets:
            call_buckets[sub_name].append(path)
        else:
            call_buckets[sub_name] = [path]

    pickle.dump(call_lists, open("subroutine_calls.pkl", "wb"))

    for sub_key, stats in tqdm(sub_stats.items(), desc="Searching calls"):
        (mod_path, sub_name) = sub_key
        calls = call_lists[sub_key]
        max_depth = get_max_depth(0, [], mod_path, calls, call_lists, call_buckets)
        sub_stats[sub_key]["Max stack depth"] = max_depth

    sub_stats = list(sub_stats.values())
    sub_stats.sort(key=lambda s: (s["Prefix"], s["Filename"], s["Subroutine"]))
    with open("subroutine_stats.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sub_stats)


def get_max_depth(cur_depth, call_stack, mod_str, calls, all_sub_calls, buckets):
    possible_depths = list()
    for call in calls:
        if call in buckets and call not in call_stack:
            possible_mod_strs = buckets[call]
            best_mod_str = get_best_mod_match(mod_str, possible_mod_strs)
            new_calls = all_sub_calls[(best_mod_str, call)]
            new_depth = get_max_depth(cur_depth+1, call_stack + [call], best_mod_str, new_calls, all_sub_calls, buckets)
            possible_depths.append(new_depth)

    if len(possible_depths) == 0:
        return cur_depth

    return max(possible_depths)


def get_best_mod_match(mod2match, mod_list):
    mod_score = 0
    best_mod = 0
    for j, mod in enumerate(mod_list):
        i = 0
        while i < len(mod2match)-1 and i < len(mod)-1 and mod2match[i] == mod[i]:
            i += 1
        if i > mod_score:
            best_mod = j
            mod_score = i
    return mod_list[best_mod]


if __name__ == '__main__':
    main()

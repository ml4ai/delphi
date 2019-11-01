#!/usr/bin/env python3

"""File: loop_handle.py

Purpose: Read the Fortran AST (obtained from rectify.py) and refactor it
         to remove the breaks, returns and continues from within loop
         statements.

Called from: translate.py
Calls: None

Author: Pratik Bhandari
"""
import sys
import argparse
import json
import copy
from typing import Dict
from delphi.translators.for2py import For2PyError, syntax


class ContinueState(object):
    def __init__(self):
        self.cycle_index = None
        self.is_cycle = None
        self.cycle_dict = {}
        self.continue_here = False
        self.else_present = False


class BreakState(object):
    def __init__(self):
        self.break_index = None
        self.is_break = False
        self.break_dict = {}
        self.break_depth = -1
        self.break_here = False
        self.else_present = False


class RefactorConstructs(object):
    """This class defines the refactor state of the intermediate AST while
    removing the breaks and returns inside while loops
    """

    def __init__(self):
        self.return_found = False
        self.potential_if = dict()
        self.shifted_body = list()
        self.new_while_body = list()
        self.after_return = list()
        self.new_outer = list()
        self.tag_level = 0
        self.shifted_items = {}
        self.shifted_level = 0
        # A list of all the three constructs present in the program
        self.loop_constructs = []

    def refactor(self, ast: Dict, loop_constructs) -> Dict:
        self.loop_constructs = loop_constructs
        body = ast["ast"][0]["body"]

        # Parse the loop_construct dictionary for each loop backwards. We
        # will process each construct bottom-up
        loop_index = 1
        # For every loop in the body
        # TODO: Currently only for un-nested loops without returns. Does not
        #  work for nested loops with/without returns as well
        for item in body:
            if item["tag"] in ["do", "do-while"]:
                for construct in self.loop_constructs[f"loop_{loop_index}"][
                                 ::-1]:
                    if 'return' in construct:
                        self.search_while(body)
                    elif 'break' in construct:
                        break_worker = BreakState()
                        break_worker.break_index = construct
                        self.search_breaks(item["body"], break_worker)
                    elif 'cycle' in construct:
                        cycle_worker = ContinueState()
                        cycle_worker.cycle_index = construct
                        self.search_cycles(item["body"], cycle_worker)

        return ast

    #####################################################################
    #                                                                   #
    #                             CONTINUE                              #
    #                                                                   #
    #####################################################################

    def search_cycles(self, body, cycle_state):
        # TODO Continues handling does not currently work for nested `continues`
        #  Need to handle this
        self.tag_cycle(body, cycle_state)
        self.append_cycle(body, cycle_state)

    def tag_cycle(self, body, state):
        for item in body:
            if state.is_cycle:
                state.cycle_dict.setdefault(state.cycle_index, []).append(item)
                body.remove(item)
                continue
            if item["tag"] == "if":
                # Count to check if the `if` body contains only one element
                # or more
                body_count = len(item["body"])
                self.tag_cycle(item["body"], state)
                if state.continue_here:
                    item["tag"] = "if_with_continue"
                    if body_count == 1:
                        item["header"][0]["operator"] = \
                            syntax.NEGATED_OP[item['header'][0]['operator']]
                    if item.get("else"):
                        state.else_present = True
                    state.continue_here = False
                    continue
                if item.get("else"):
                    self.tag_cycle(item["else"], state)
                    if state.continue_here:
                        item["tag"] = "else_with_continue"
                        state.continue_here = False
            elif item["tag"] == "cycle":
                if state.cycle_index == f"cycle_{item['index']}":
                    state.is_cycle = True
                    state.continue_here = True
                    body.remove(item)

    @staticmethod
    def append_cycle(body, state):
        for item in body:
            # Find the `if` block which had the continue statement
            if item["tag"] == "if_with_continue":
                # Start checking which of the 4 cases you had
                item["tag"] = "if"
                if len(item["body"]) == 0:
                    if state.else_present is True:
                        item["body"] += item.pop("else")
                    item["body"] += state.cycle_dict[state.cycle_index]
                else:
                    if state.else_present is False:
                        item["else"] = []
                    item["else"] += state.cycle_dict[state.cycle_index]
            elif item["tag"] == "else_with_continue":
                item["tag"] = "if"
                # Check for the 2 cases you had
                item["body"] += state.cycle_dict[state.cycle_index]
                if len(item["else"]) == 0:
                    del item["else"]

#########################################################################
#                                                                       #
#                                BREAKS                                 #
#                                                                       #
#########################################################################

    def search_breaks(self, body, break_state):
        self.tag_break(body, 0, break_state)
        self.append_break(body, 0, break_state)

    def tag_break(self, body, if_depth, state):
        for item in body[:]:
            if state.is_break:
                state.break_dict.setdefault(if_depth, []).append(
                    {
                        "item": item,
                        "index": state.break_index
                    }
                )
                body.remove(item)
                continue
            if item["tag"] == "if":
                self.tag_break(item["body"], if_depth+1, state)
                if state.break_here:
                    item["tag"] = "if_with_break"
                    item["header"][0]["operator"] = \
                        syntax.NEGATED_OP[item['header'][0]['operator']]
                    if item.get("else"):
                        state.else_present = True
                    state.break_here = False
                    continue
                if item.get("else"):
                    self.tag_break(item["else"], if_depth+1, state)
                    if state.break_here:
                        item["tag"] = "else_with_break"
                        state.break_here = False
            elif item["tag"] == "exit":
                if state.break_index == f"break_{item['index']}":
                    state.is_break = True
                    state.break_here = True
                    state.break_depth = if_depth
                    body.remove(item)

    def append_break(self, body, if_depth, state):
        for item in body:
            if item["tag"] == "if":
                if not( len(item["body"]) and \
                        item["body"][0]["tag"] in ["cycle", "exit", "return"]):
                    inc = if_depth + 1
                else:
                    inc = if_depth
                for key in state.break_dict:
                    if inc > key:
                        for elmt in state.break_dict[key]:
                            item["body"] += [elmt["item"]]
                self.append_break(item["body"], if_depth+1, state)
                if item.get("else"):
                    if len(item["else"]) == 1 and \
                            item["else"][0]["tag"] in ["if", "if_with_break",
                                                       "else_with_break"]:
                        self.append_break(item["else"], if_depth+1, state)
                    else:
                        for key in state.break_dict:
                            if if_depth + 1 > key:
                                for elmt in state.break_dict[key]:
                                    item["else"] += [elmt["item"]]
                        self.append_break(item["else"], if_depth+1, state)
            elif item["tag"] == "if_with_break":
                item["tag"] = "if"
                if len(item["body"]) == 0 and state.else_present is False:
                    for key in state.break_dict:
                        if if_depth+1 > key:
                            for elmt in state.break_dict[key]:
                                item["body"] += [elmt["item"]]
                    item["else"] = [{"tag": "exit"}]
                elif len(item["body"]) == 0 and state.else_present is True:
                    item["body"] += item["else"]
                    for key in state.break_dict:
                        if if_depth + 1 > key:
                            for elmt in state.break_dict[key]:
                                item["body"] += [elmt["item"]]
                    item["else"] = [{"tag": "exit"}]
                elif len(item["body"]) > 0 and state.else_present is False:
                    item["else"] = item["body"]
                    item["else"] += [{"tag": "exit"}]
                    for key in state.break_dict:
                        if if_depth + 1 > key:
                            for elmt in state.break_dict[key]:
                                item["body"] += [elmt["item"]]
                elif len(item["body"]) > 0 and state.else_present is True:
                    else_tmp = item["else"]
                    item["else"] = item["body"]
                    item["else"] += [{"tag": "exit"}]
                    item["body"] = else_tmp
                    for key in state.break_dict:
                        if if_depth + 1 > key:
                            for elmt in state.break_dict[key]:
                                item["body"] += [elmt["item"]]
            elif item["tag"] == "else_with_break":
                item["tag"] = "if"
                item["else"] += [{"tag": "exit"}]
                for key in state.break_dict:
                    if if_depth + 1 > key:
                        for elmt in state.break_dict[key]:
                            item["body"] += [elmt["item"]]

    #####################################################################
    #                                                                   #
    #                              RETURNS                              #
    #                                                                   #
    #####################################################################

    def search_while(self, body):
        for item in body[:]:
            # Currently, breaks and returns are only inside the bodies of
            # `while` functions
            # TODO They can be anywhere
            if self.return_found:
                if item.get('tag') != "format":
                    self.new_outer.append(item)
                    body.remove(item)

            if item.get("tag") == 'do-while':
                self.search_tags(item)
                if self.return_found:
                    self.start_while(item)
                    continue
                elif item.get("body"):
                    self.search_while(item["body"])
            elif item.get("body"):
                self.search_while(item["body"])

        if self.new_outer:
            self.modify_shifted()
            self.shifted_body[0]["body"] = self.new_outer
            body += self.shifted_body
            self.new_outer = []

    def modify_shifted(self):
        var_list = []
        op_list = []
        right_header = copy.deepcopy(self.shifted_body[0]['header'])
        for item in right_header:
            for ref in item['left']:
                if ref.get('tag') == 'ref':
                    op_list.append(
                        [{
                            "tag": "op",
                            "operator": ".not.",
                            "left": [ref]
                        }]
                    )
                    var_list.append(ref)
            for ref in item['right']:
                if ref.get('tag') == 'ref':
                    op_list.append(
                        [{
                            "tag": "op",
                            "operator": ".not.",
                            "left": [ref]
                        }]
                    )
                    var_list.append(ref)
        left_header = {
            "tag": "op",
            "left": op_list[0],
            "right": op_list[1],
            "operator": ".and."
        }
        self.shifted_body[0]["header"][0]["left"] = [left_header]
        self.shifted_body[0]["header"][0]["right"] = right_header
        self.shifted_body[0]["header"][0]["operator"] = ".or."

    def search_tags(self, item):
        for items in item["body"]:
            if items["tag"] == 'if':
                for if_body in items["body"]:
                    if if_body["tag"] == 'stop':
                        self.return_found = True
                        break
            if self.return_found:
                break

        return self.return_found

    def start_while(self, while_body):
        end_point = False
        # Start going through the body of the while loop
        for body in while_body["body"]:
            # The breaks and returns we are looking for are only inside the `if`
            # statements
            if body["tag"] == 'if':
                self.potential_if = copy.deepcopy(body)
                for if_body in body["body"]:
                    if if_body["tag"] == 'stop':
                        body['header'][0]['operator'] = \
                            syntax.NEGATED_OP[body['header'][0]['operator']]
                        body["else"] = [
                            {"tag": "exit"}
                        ]
                        end_point = True
                        self.new_while_body.append(body)
                        self.shifted_body.append(copy.deepcopy(body))
                if end_point:
                    continue
            if not end_point:
                self.new_while_body.append(body)
            else:
                self.after_return.append(body)

        while_body["body"] = self.new_while_body

        for body in while_body["body"]:
            if body["tag"] == 'if':
                for if_body in body["body"]:
                    if if_body["tag"] == "stop":
                        body["body"] = self.after_return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        nargs="*",
        required=True,
        help="AST dictionary which is to be refactored"
    )
    parser.add_argument(
        "-o",
        "--output",
        nargs="+",
        help="Target output file to store refactored AST"
    )
    parser.add_argument(
        "-w",
        "--write",
        nargs="+",
        help="Flag whether to write the refactored AST or not"
    )
    parser.add_argument(
        "-p",
        "--print",
        nargs="+",
        help="Flag whether to print the refactored AST or not"
    )
    args = parser.parse_args(sys.argv[1:])

    input_f = args.file[0]
    target = args.output[0]
    is_write = args.write[0]
    is_print = args.print[0]

    return input_f, target, is_write, is_print


if __name__ == "__main__":

    # Read in the arguments to the file
    (input_file, target_file, write_flag, print_flag) = parse_args()

    # Read the input AST
    try:
        with open(input_file, 'r') as infile:
            input_ast = infile.read()
    except IOError:
        raise For2PyError(f"Unable to read from {input_file}.")

    # Refactor the AST
    refactor_ast = RefactorConstructs()
    refactored_ast = refactor_ast.refactor(json.loads(input_ast))

    if write_flag == "True":
        # Write the refactored AST
        try:
            with open(target_file, 'w') as op_file:
                json.dumps(refactored_ast)
        except IOError:
            raise For2PyError(f"Unable to write to {target_file}.")

    # If the print flag is set, print the AST to console
    if print_flag == "True":
        print(refactored_ast)

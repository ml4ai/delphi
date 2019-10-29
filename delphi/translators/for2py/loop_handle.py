#!/usr/bin/env python3

"""File: loop_handle.py

Purpose: Read the Fortran AST (obtained from rectify.py) and refactor it
         to remove the breaks and returns from within loop statements

"""

import sys
import pickle
import argparse
import re
import json
import copy
from typing import Dict
from delphi.translators.for2py.format import list_data_type
from delphi.translators.for2py import For2PyError, syntax


class RefactorBreaks(object):
    """This class defines the refactor state of the intermediate AST while
    removing the breaks and returns inside while loops
    """

    def __init__(self):
        self.find_opp = {
            ".eq.": ".ne.",
            ".ne.": ".eq.",
            ".lt.": ".ge.",
            ".gt.": ".le.",
            ".le.": ".gt.",
            ".ge.": ".lt.",
        }
        self.return_found = False
        self.potential_if = dict()
        self.shifted_body = list()
        self.new_while_body = list()
        self.after_return = list()
        self.new_outer = list()
        self.tag_level = 0
        self.is_break = False
        self.shifted_items = {}
        self.shifted_level = 0

    def refactor(self, ast: Dict) -> Dict:
        body = ast["ast"][0]["body"]
        self.search_while(body)

        # self.search_breaks(body)
        return ast

    def search_breaks(self, body):
        # We search for all loop types i.e. "do" and "do-while"
        for item in body:
            if item["tag"] in ["do", "do-while"]:
                # Start the tagging process
                self.tag_conditionals(item["body"])
                if self.is_break:
                    self.shift_into_ifs(item["body"])

    def tag_conditionals(self, body):
        for item in body:
            if item["tag"] == "if":
                item["body"].append(
                    {
                        "tag": f"tag_{self.tag_level}"
                    }
                )
                self.tag_level += 1
                self.tag_conditionals(item["body"])
                if item.get("else"):
                    self.tag_conditionals(item["else"])
            elif item["tag"] == "exit":
                self.is_break = True

    def shift_into_ifs(self, body):
        catch_now = False
        for item in body[:]:
            print(catch_now, item)
            if catch_now:
                self.shifted_items[self.shifted_level] = copy.deepcopy(item)
            if item["tag"] == "if":
                catch_now = True
                self.shift_into_ifs(item["body"])
                if item.get("else"):
                    self.shift_into_ifs(item["else"])
        print('\n')
        self.shifted_level += 1

    def search_while(self, body):
        for item in body[:]:
            # Currently, breaks and returns are only inside the bodies of
            # `while` functions
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
                            self.find_opp[body['header'][0]['operator']]
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
    refactor_ast = RefactorBreaks()
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

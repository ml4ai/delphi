#!/usr/bin/env python3

"""File: format.py

Purpose: Process a string (obtained from a data file) according to
         a Fortran FORMAT.

         NOTE: Not all format specifiers are supported at this time.
         The input formats supported are shown in the method
         match_input_fmt_1(), while the output formats supported are
         shown in the method gen_output_fmt_1().

Usage:
        I. FORMATTED I/O

        Given a Fortran READ or WRITE statement that should be processed
        according to a format list fmt_list (where fmt_list is a Python
        list of Fortran FORMAT descriptors), do the following:

        (1) Create a Format object as follows:

                my_fmt_obj = Format(fmt_list)

        (2) INPUT: To process a line of input inp_ln according to this
            format and assign values to a tuple of variables var_list:

                var_list = my_fmt_obj.read_line(inp_ln)

            OUTPUT: To construct a line to be printed out given a set
            of values val1, ..., valN:

                out_string = my_fmt_obj.write_line([val1, ..., valN])

        II. LIST_DIRECTED I/O

        At this time only list-directed output has been implemented.

        For list-directed output, e.g.: WRITE (*,*) X, Y, Z do the following:

        (1) Construct a list of the types of the values to be written.
            Let this list be denoted by out_type_list.

        (2) Construct a list of format specifiers for these types using:

                fmt_list = list_output_format(out_type_list)

        (3) Use fmt_list as described above.

        There are some examples towards the end of this file.
"""

import re
import sys
from . import For2PyError


class Format:
    def __init__(self, format_list):
        self._format_list = format_list.copy()
        self._read_line_init = False
        self._write_line_init = False

        self._re_cvt = None
        self._regexp_str = None
        self._re = None
        self._match_exps = None
        self._divisors = None
        self._in_cvt_fns = None

        self._output_fmt = None
        self._out_gen_fmt = None
        self._out_widths = None

    def init_read_line(self):
        """init_read_line() initializes fields relevant to input matching"""
        format_list = self._format_list
        self._re_cvt = self.match_input_fmt(format_list)

        regexp0_str = "%r" % "".join([subs[0] for subs in self._re_cvt])
        regexp0_str = regexp0_str[1:-1]
        self._regexp_str = regexp0_str
        self._re = re.compile(regexp0_str)
        self._match_exps = [
            subs[1] for subs in self._re_cvt if subs[1] is not None
        ]
        self._divisors = [subs[2] for subs in self._re_cvt if subs[2] is not
                          None]
        self._in_cvt_fns = [
            subs[3] for subs in self._re_cvt if subs[3] is not None
        ]
        self._read_line_init = True

    def init_write_line(self):
        """init_write_line() initializes fields relevant to output generation"""
        format_list = self._format_list
        output_info = self.gen_output_fmt(format_list)
        self._output_fmt = "".join([sub[0] for sub in output_info])
        self._out_gen_fmt = [sub[1] for sub in output_info if sub[1] is not
                             None]
        self._out_widths = [sub[2] for sub in output_info if sub[2] is not None]
        self._write_line_init = True

    def read_line(self, line):
        """
        Match a line of input according to the format specified and return a
        tuple of the resulting values
        """

        if not self._read_line_init:
            self.init_read_line()

        match = self._re.match(line)
        assert match is not None, f"Format mismatch (line = {line})"

        matched_values = []
        for i in range(self._re.groups):
            cvt_re = self._match_exps[i]
            cvt_div = self._divisors[i]
            cvt_fn = self._in_cvt_fns[i]
            match_str = match.group(i + 1)

            match0 = re.match(cvt_re, match_str)
            if match0 is not None:
                if cvt_fn == "float":
                    if "." in match_str:
                        val = float(match_str)
                    else:
                        val = int(match_str) / cvt_div
                elif cvt_fn == "int":
                    val = int(match_str)
                else:
                    sys.stderr.write(
                        f"Unrecognized conversion function: {cvt_fn}\n"
                    )
            else:
                sys.stderr.write(
                    f"Format conversion failed: {match_str}\n"
                )

            matched_values.append(val)

        return tuple(matched_values)

    def write_line(self, values):
        """
        Process a list of values according to the format specified to generate
        a line of output.
        """

        if not self._write_line_init:
            self.init_write_line()

        if len(self._out_widths) > len(values):
            raise For2PyError(f"ERROR: too few values for format"
                              f" {self._format_list}\n")

        out_strs = []
        for i in range(len(self._out_widths)):
            out_fmt = self._out_gen_fmt[i]
            out_width = self._out_widths[i]

            if values[i] is None:
                out_strs.append(values[i])
            else:
                out_val = out_fmt.format(values[i])
                # out_width == "*" indicates that the field can be
                # arbitrarily wide
                if out_width != "*":
                    if len(out_val) > out_width:  # value too big for field
                        out_val = "*" * out_width
                out_strs.append(out_val)

        out_str_exp = (
            '"' + self._output_fmt + '".format' + str(tuple(out_strs))
        )
        out_str = eval(out_str_exp)
        return out_str + "\n"

    def __str__(self):
        return str(self._format_list)

    ###########################################################################
    #                                                                         #
    #                              INPUT MATCHING                             #
    #                                                                         #
    ###########################################################################

    def match_input_fmt(self, fmt_list):
        """Given a list of Fortran format specifiers, e.g., ['I5', '2X',
        'F4.1'], this function constructs a list of tuples for matching an input
        string against those format specifiers."""

        rexp_list = []
        for fmt in fmt_list:
            rexp_list.extend(self.match_input_fmt_1(fmt))

        return rexp_list

    def match_input_fmt_1(self, fmt):
        """
        Given a single format specifier, e.g., '2X', 'I5', etc., this function
        constructs a list of tuples for matching against that specifier. Each
        element of this list is a tuple

               (xtract_re, cvt_re, divisor, cvt_fn)

        where:

           xtract_re is a regular expression that extracts an input field of
                   the requisite width;
           cvt_re is a regular expression that matches the character sequence
                   extracted by xtract_re against the specified format;
           divisor is the value to divide by in order to get the appropriate
                   number of decimal places if a decimal point is not given
                   in the input value (meaningful only for floats); and
           cvt_fn is a string denoting the function to be used to convert the
                   matched string to a value.
        """

        # first, remove any surrounding space
        fmt = fmt.strip()

        # get any leading digits indicating repetition
        match = re.match(r"(\d+)(.+)", fmt)
        if match is None:
            reps = 1
        else:
            reps = int(match.group(1))
            fmt = match.group(2)

        if fmt[0] == "(":  # process parenthesized format list recursively
            fmt = fmt[1:-1]
            fmt_list = fmt.split(",")
            rexp = self.match_input_fmt(fmt_list)
        else:
            if fmt[0] in "iI":  # integer
                sz = fmt[1:]
                xtract_rexp = '(.{' + sz + '})'  # r.e. for extraction
                rexp1 = r" *-?\d+"  # r.e. for matching
                divisor = 1
                rexp = [(xtract_rexp, rexp1, divisor, "int")]

            elif fmt[0] in "xX":  # skip
                xtract_rexp = "."  # r.e. for extraction
                rexp = [(xtract_rexp, None, None, None)]

            elif fmt[0] in "fF":  # floating point
                idx0 = fmt.find(".")
                sz = fmt[1:idx0]
                divisor = 10 ** (int(fmt[idx0 + 1:]))
                xtract_rexp = '(.{,' + sz + '})'  # r.e. for extraction
                rexp1 = r" *-?\d+(\.\d+)?"  # r.e. for matching
                rexp = [(xtract_rexp, rexp1, divisor, "float")]
            else:
                raise For2PyError(
                    f"ERROR: Unrecognized format specifier {fmt}\n"
                )

        # replicate the regular expression by the repetition factor in the
        # format
        rexp *= reps

        return rexp

    ###########################################################################
    #                                                                         #
    #                             OUTPUT GENERATION                           #
    #                                                                         #
    ###########################################################################

    def gen_output_fmt(self, fmt_list):
        """given a list of Fortran format specifiers, e.g., ['I5', '2X',
        'F4.1'], this function constructs a list of tuples for constructing
        an output
        string based on those format specifiers."""

        rexp_list = []
        for fmt in fmt_list:
            rexp_list.extend(self.gen_output_fmt_1(fmt))

        return rexp_list

    def gen_output_fmt_1(self, fmt):
        """given a single format specifier, get_output_fmt_1() constructs and
        returns a list of tuples for matching against that specifier.

        Each element of this list is a tuple
            (gen_fmt, cvt_fmt, sz)

        where:
           gen_fmt is the Python format specifier for assembling this value into
           the string constructed for output;
           cvt_fmt is the Python format specifier for converting this value into
           a string that will be assembled into the output string; and
           sz is the width of this field.
        """

        # first, remove any surrounding space
        fmt = fmt.strip()

        # get any leading digits indicating repetition
        match = re.match(r"(\d+)(.+)", fmt)
        if match is None:
            reps = 1
        else:
            reps = int(match.group(1))
            fmt = match.group(2)

        if fmt[0] == "(":  # process parenthesized format list recursively
            fmt = fmt[1:-1]
            fmt_list = fmt.split(",")
            rexp = self.gen_output_fmt(fmt_list)
        else:
            if fmt[0] in "iI":  # integer
                sz = fmt[1:]
                gen_fmt = "{}"
                cvt_fmt = "{:" + str(sz) + "d}"
                rexp = [(gen_fmt, cvt_fmt, int(sz))]

            elif fmt[0] in "xX":
                gen_fmt = " "
                rexp = [(gen_fmt, None, None)]

            elif fmt[0] in "aA":
                gen_fmt = "{}"
                # the '*' in the third position of the tuple (corresponding to
                # field width) indicates that the field can be arbitrarily wide
                rexp = [(gen_fmt, "{}", "*")]

            elif fmt[0] in "eEfFgG":  # various floating point formats
                idx0 = fmt.find(".")
                sz = fmt[1:idx0]
                suffix = fmt[idx0 + 1:]
                # The 'E' and G formats can optionally specify the width of
                # the exponent, e.g.: 'E15.3E2'.  For now we ignore any such
                # the exponent width -- but if it's there, we need to extract
                # the sequence of digits before it.
                m = re.match(r"(\d+).*", suffix)
                assert m is not None, f"Improper format? '{fmt}'"
                prec = m.group(1)
                gen_fmt = "{}"
                cvt_fmt = "{:" + sz + "." + prec + fmt[0] + "}"
                rexp = [(gen_fmt, cvt_fmt, int(sz))]

            elif fmt[0] in "pP":  # scaling factor
                # For now we ignore scaling: there are lots of other things we
                # need to spend time on. To fix later if necessary.
                rest_of_fmt = fmt[1:]
                rexp = self.gen_output_fmt_1(rest_of_fmt)

            elif fmt[0] in "'\"":  # character string
                sz = len(fmt) - 2  # -2 for the quote at either end
                # escape any double-quotes in the string
                gen_fmt = fmt[1:-1].replace('"', '\\\"')
                rexp = [(gen_fmt, None, None)]

            elif fmt[0] == "/":  # newlines
                gen_fmt = "\\n" * len(fmt)
                rexp = [(gen_fmt, None, None)]

            else:
                raise For2PyError(
                    f"ERROR: Unrecognized format specifier {fmt[0]}\n"
                )

        # replicate the regular expression by the repetition factor in the
        # format
        rexp *= reps

        return rexp


################################################################################
#                                                                              #
#                     DEFAULT FORMATS FOR LIST-DIRECTED I/O                    #
#                                                                              #
################################################################################

# LIST_WRITE_DEFAULTS specifies the default formats for list-directed output.
# Source: https://software.intel.com/en-us/fortran-compiler-developer-guide-\
#           and-reference-rules-for-list-directed-sequential-write-statements

LIST_WRITE_DEFAULTS = {
    "BYTE": "I5",
    "LOGICAL(1)": "L2",
    "LOGICAL(2)": "L2",
    "LOGICAL(4)": "L2",
    "LOGICAL(8)": "L2",
    "INTEGER": "I12",
    "INTEGER(1)": "I5",
    "INTEGER(2)": "I7",
    "INTEGER(4)": "I12",
    "INTEGER(8)": "I22",
    "REAL": "1PE14.5E2",
    "REAL(4)": "1PG15.7E2",
    "REAL(8)": "1PG24.15E3",
    "REAL(16)": "1PG43.33E4",
}


# default_output_format() takes a type name and returns the default format
# specifier for list-directed output of a value of that type.
def default_output_format(type_item):
    type_item = type_item.upper()
    if type_item in LIST_WRITE_DEFAULTS:
        return LIST_WRITE_DEFAULTS[type_item]

    if type_item[0] in "\"'":
        return type_item

    sys.stderr.write(
        f"WARNING: No output format found for type {type_item}\n"
    )
    return None


def list_output_formats(type_list):
    """This function takes a list of type names and returns a list of
    format specifiers for list-directed output of values of those types."""
    out_format_list = []
    for type_item in type_list:
        item_format = default_output_format(type_item)
        out_format_list.append(item_format)

    return out_format_list


def list_input_formats(type_list):
    sys.stderr.write("*** List-directed input not yet implemented\n")
    return []


def list_data_type(type_list):
    """This function takes a list of format specifiers and returns a list of
    data types represented by the format specifiers."""
    data_type = []
    for item in type_list:
        match = re.match(r"(\d+)(.+)", item)
        if not match:
            reps = 1
            if item[0] in "FfEegG":
                data_type.append("REAL")
            elif item[0] in "Ii":
                data_type.append("INTEGER")
        else:
            reps = match.group(1)
            fmt = match.group(2)
            if "(" in fmt and "," in fmt:
                fmt = fmt[1:-1].split(",")
            elif "(" in fmt:
                fmt = [fmt[1:-1]]
            else:
                fmt = [fmt]
            for i in range(int(reps)):
                for ft in fmt:
                    if ft[0] in "FfEegG":
                        data_type.append("REAL")
                    elif ft[0] in "Ii":
                        data_type.append("INTEGER")
                    elif ft[0] in "Aa":
                        data_type.append("CHARACTER")
    return data_type


################################################################################
#                                                                              #
#                                 EXAMPLE USAGE                                #
#                                                                              #
################################################################################


def example_1():

    ################################# EXAMPLE 1 ###############################
    # Format from read statement in the file Weather.for
    # The relevant Fortran code is:
    #
    #         OPEN (4,FILE='WEATHER.INP',STATUS='UNKNOWN')
    #         ...s
    #         READ(4,20) DATE,SRAD,TMAX,TMIN,RAIN,PAR
    #   20   FORMAT(I5,2X,F4.1,2X,F4.1,2X,F4.1,F6.1,14X,F4.1)
    #
    # The line of data shown (input1) is taken from the file WEATHER.INP

    format1 = [
        "I5",
        "2X",
        "F4.1",
        "2X",
        "F4.1",
        "2X",
        "F4.1",
        "F6.1",
        "14X",
        "F4.1",
    ]
    input1 = "87001  -5.1  20.0   4.4 -23.9              10.7 "

    rexp1 = Format(format1)
    (DATE, SRAD, TMAX, TMIN, RAIN, PAR) = rexp1.read_line(input1)

    print(f"FORMAT: {format1}")
    print(f"regexp_str = '{rexp1}'")

    vars1 = (DATE, SRAD, TMAX, TMIN, RAIN, PAR)
    print(f"vars1 = {vars1}")
    print("")


def example_2():
    ################################# EXAMPLE 2 ################################
    # Format based on a read statement in the file Plant.for
    # The relevant Fortran code is:
    #        OPEN (2,FILE='PLANT.INP',STATUS='UNKNOWN')
    #        ...
    #        READ(2,10) Lfmax, EMP2,EMP1,PD,nb,rm,fc,tb,intot,n,lai,w,wr,wc
    #     &     ,p1,sla
    #   10   FORMAT(17(1X,F7.4))
    #
    # The line of data shown (input2) is taken from the file PLANT.INP

    format2 = ["3(1X,F7.4)"]
    input2 = "    12.0    0.64   0.104"

    rexp2 = Format(format2)
    (Lfmax, EMP2, EMP1) = rexp2.read_line(input2)

    print("FORMAT: {}".format(format2))
    print('regexp_str = "{}"'.format(rexp2))

    vars2 = (Lfmax, EMP2, EMP1)
    print("vars2 = {}".format(vars2))
    print("")


def example_3():

    format3 = ["3(I5,2X,F5.2)"]
    print(list_data_type(format3))


if __name__ == "__main__":
    example_1()
    example_2()
    example_3()

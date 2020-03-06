
OFP_ERROR_PREFIX          = ["java.lang.RuntimeException", "xml.etree.ElementTree.ParseError"]
PREPROCESSOR_ERROR_PREFIX = ""
RECTIFY_ERROR_PREFIX      = ["AssertionError: In handle_tag_", "AssertionError: Literal value"]
TRANSLATE_ERROR_PREFIX    = ""
PYTRANSLATE_OR_GENPGM_ERROR_PREFIX  = "KeyError"
# GENPGM_ERROR_PREFIX       = "KeyError"

f = open('errors.log', 'r')
eflist = open('errorFileList.txt', 'w')
flist = open('fileList.txt', 'w')

handled_files = []

new_file = False
ofp_error = False

current_file = ""
prev_file = ""

line = f.readline()
prev_line = ""
prev_prev_line = ""

while line:
    if new_file:
        new_file = False

    if (
            OFP_ERROR_PREFIX[0] in line
            or OFP_ERROR_PREFIX[1] in line
    ):
        level = "ofp"
        issue = "Unknown"
        ofp_error = True
    elif (
            not ofp_error and
            (RECTIFY_ERROR_PREFIX[0] in line
            or RECTIFY_ERROR_PREFIX[1] in line)
    ):
        level = "Rectify"
        if RECTIFY_ERROR_PREFIX[0] in line:
            issue = "Unhandled structure"
        else:
            issue = "Literal value issue"
    elif PYTRANSLATE_OR_GENPGM_ERROR_PREFIX in line:
        if "genPGM.py" in prev_prev_line:
            level = "genPGM"
        else:
            level = "pyTranslate"
        issue = "Unhandled structure"
    else:
        level = None

    if current_file:
        if level:
            if level != "genPGM" and level != "pyTranslate":
                error_msg = line.rstrip()
            else:
                error_msg = prev_prev_line.strip()
            eflist.write(f"{current_file}@{level}@{error_msg}@{issue}\n")
        elif current_file not in handled_files:
            handled_files.append(current_file)
        prev_file = current_file

    if "@@" in line:
        new_file = True
        ofp_error = False
        current_file = line.split('/')[-1].rstrip()
        flist.write(f"{current_file}\n")
        
    prev_prev_line = prev_line
    prev_line = line
    line = f.readline()

f.close()
eflist.close()
flist.close()

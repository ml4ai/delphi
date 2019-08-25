# Grounded Function Network (GrFN) JSON Specification

**Version 0.1.m7**

## TODO

In progress. Release target date: July 1, 2019

>TODO:
>
>- Add namespace (corresonding to Fortran Modules)
>	- each module is a GrFN file (no matter how many source files)
>	- each program is a GrFN file (no matter how many source files)
>- Change loops to open-ended representation
>	- subsumes for/iteration loops
>- Structured Types
>	- Add arrays
>- Add goto
>- Add precision info to `<variable_domain_type>`
>- Add variable domain constraints to `<variable_spec>`
>	- set of intervals

### `[0.1.m7]` - 2019-07-01

Changes since [0.1.m5]

#### Added
- GrFN_spec Index with links for quick reference navigation
- Changelog, inspired by [Keep a Changelog](https://keepachangelog.com)
- [`<grounding_metadata_spec>`](#grounding-metadata-spec) in "grounding" field of [`<identifier_spec>`](#identifier-specification). Includes source, type (definition, units, constraints) and value

#### Changed
- Reorganized and rewrote portions of Introduction

[Change Log](#change-og) (from previous releases)


## grfn_spec Index

Example whole system GrFN specification file structure

```
system.json
namespace1.json
namespace1_lambdas.py
namespace2.json
namespace2_lambdas.py
...
```

NOTE: [`<variable_name>`](#variable-naming-convention) and [`<function_name>`](#function-naming-convention) are both formatted as [`<identifier_string>`](#identifier-string), but with particular [naming conventions](#variable-and-function-identifiers-and-references).

- `<system_def>` ::=  # serves as an index; has to be a DAG
	- "date_created" : `<string>`
	- "name" : `<string>` # name of system; optional?
	- "components" : list of `<grfn_spec_refs>`[attrval] ::=
		- "name" : [`<namespace_path_string>`](#path-strings)
		- "imports" : list of [`<namespace_path_string>`](#path-strings) # specifies which grfn_spec files to load

- [`<grfn_spec>`](#top-level-grfn-specification)[attrval] ::=
	- "date_created" : `<string>`
	- "namespace" : [`<namespace_path_string>`](#path-strings) # 'current' namespace; grfn_spec filename will be the same
	- "imports" : list of `<import_identifier_spec>`[attrval] ::=
		- "source_identifier" : [`<identifier_string>`](#identifier-string)
		- "name" : `<string>` # the name as used locally in this grfn_spec
	- "source" : list of [`<source_code_file_path>`](#scope-and-namespace-paths)
	- "start": list of `<string>` # 'top-level' function(s)
	- "identifiers" : list of [`<identifier_spec>`](#identifier-specification)[attrval] ::=

		- "base_name" : [`<base_name>`](#base-name)
		- "scope" : [`<scope_path>`](#scope-and-namespace-paths)
		- "namespace" : [`<namespace_path>`](#scope-and-namespace-paths)
		- "source\_references" : list of [`<source_code_reference>`](#grounding-and-source-code-reference)
		- "gensym" : [`<gensym>`](#identifier-gensym)
		- "grounding" : list of [`<grounding_metadata_spec>`](#grounding-metadata-spec)[attrval] ::=
			- "source" : `<string>` # URI to source document
			- "type" : `"definition"` | `"units"` | `"constraint"`
			- "value" : `<string>`
			- "score" : `<real>` # confidence score of grounding
	
	- "variables" : list of [`<variable_spec>`](#variable-specification)[attrval] ::=

		- "name" : [`<variable_name>`](#variable-naming-convention)
		- "domain" : [`<variable_domain_type>`](#variable-value-domain)[attrval] ::=
			- "type" : `"real"` | `"integer"` | `"boolean"` | `"string"`
			- "precision" : TODO
		- "domain_constraints" : `<string>` # a disjunctive normal form, with v := variable value
			- e.g., `"(or (and (< v infty) (>= v 5) (and (> v -infty) (< v 0)))"`
			- Could this more generally reference other variables?
		- "mutable" : `TRUE` | `FALSE`
		
	- "functions" : list of [`<function_spec>`](#function-specification) ... instances of the following:
		
		- [`<function_assign_spec>`](#function-assign-specification)[attrval] ::=
			- "name" : [`<function_name>`](#function-naming-convention)
			- "type" : `"assign"` | `"condition"` | `"decision"`
			- "arguments" : list of [ [`<function_source_reference>`](#function-assign-specification) | [`<variable_name>`](#variable-naming-convention) ]
			- "return_value" : [`<function_source_reference>`](#function-assign-specification) | [`<variable_name>`](#variable-naming-convention)
			- "body" : one of the following:
				- [`<function_assign_body_literal_spec>`](#function-assign-body-literal)[attrval] ::=
					- "type" : `"literal"`
					- "value" : [`<literal_value>`](#function-assign-body-literal)[attrval] ::=
						- "dtype" : `"real"` | `"integer"` | `"boolean"` | `"string"`
						- "value" : `<string>`
				- [`<function_assign_body_lambda_spec>`](#function_assign_body_lambda)[attrval] ::=
					- "type" : `"lambda"`
					- "name" : [`<function_name>`](#function-naming-convention)
					- "reference" : [`<lambda_function_reference>`](#funciton-assign-body-lambda) ::= a `<string>` denoting the python function in `lambdas.py`
		
		- [`<function_container_spec>`](#function-container-specification)[attrval] ::=
			- "name" : [`<function_name>`](#function-naming-convention)
			- "type" : `"container"`
			- "arguments" : list of [ [`<function_source_reference>`](#function-assign-specification) | [`<variable_name>`](#variable-naming-convention) ]
			- "updated" : list of [`<variable_name>`](#variable-naming-convention) # variables side-effected during execution
			- "return_value" : [`<function_source_reference>`](#function-assign-specification) | [`<variable_name>`](#variable-naming-convention)
			- "body" : list of [`<function_reference_spec>`](#function-reference-specification)
		
		- [`<function_loop_plate_spec>`](#function-loop-plate-specification)[attrval] ::=
			- "name" : [`<function_name>`](#function-naming-convention)
			- "type" : `"loop_plate"`
			- "arguments" : list of [`<variable_name>`](#variable-naming-convention)
			- "updated" : list of [`<variable_name>`](#variable-naming-convention) # variables side-effected during execution
			- "test\_at\_end" : `TRUE` | `FALSE`
			- "exit\_condition" : `<loop_condition>` ::= # continue loop WHILE predicate evaluates to "output\_literal"
				- "arguments" : list of [ [`<variable_reference>`](#variable-reference) | [`<variable_name>`](#variable-naming-convention) ]
				- "return_value" : [`<variable_name>`](#variable-naming-convention) # the named variable of the exit\_condition result
				- "output_literal" : `TRUE` | `FALSE`
				- "predicate" : [`<function_name>`](#function-naming-convention) # reference to lambda fn computing the condition (which must match the output_literal in order to exit)
			- "body" : list of [`<function_reference_spec>`](#function-reference-specification)

- [`<function_reference_spec>`](#function-reference-specification)[attrval] ::=
	- "function" : [`<function_name>`](#function-naming-convention)
	- "arguments" : list of [ [`<variable_reference>`](#variable-reference) | [`<variable_name>`](#variable-naming-convention) ]
	- "return_value" : [`<variable_reference>`](#variable-reference) | [`<variable_name>`](#variable-naming-convention)

- [`<function_source_reference>`](#function-assign-specification)[attrval] ::=
	- "name" : [ [`<variable_name>`](#variable-naming-convention) | [`<function_name>`](#function-naming-convention) ]
	- "type" : `"variable"` | `"function"`

- [`<variable_reference>`](#variable-reference)[attrval] ::=
	- "variable" : [`<variable_name>`](#variable-naming-convention)
	- "index" : `<integer>`


## Introduction

### Background: From source code to dynamic system representation

GrFN, pronounced "Griffin", is the central representation generated and manipulated by the [AutoMATES](https://ml4ai.github.io/automates/) system (incorporating [Delphi](https://ml4ai.github.io/delphi)).

The goal of GrFN is to provide the end-point target for a translation from the semantics of program (computation) specification (as asserted in source code) and scientific domain concepts (as expressed in text and equations) to the semantics of a (discretized) dynamic system model (akin to an extended version of a probabilistic graphical model).

A key assumption is that the program source code we are analyzing is intended to model aspects of some target physical domain, and that this target physical domain is a dynamical system that evolves over time. This means that some source code variables are assumed to correspond to dynamical system states of the represented system. 

The system is decomposed into a set of individual states (represented as (random) variables), where the values of the states at any given time are a function of the values of zero or more other states at the current and/or previous time point(s). Because we are considering the evolution of the system over time, in general every variable has an index. The functional relationships may be instantaneous (based on the variables indexed at the same point in time) or a function of states of variables at different time indices.

There are four components in AutoMATES that generate (contribute to) and/or consume (operate on) GrFN:

- Program Analysis (PA) - generates
- Text Reading (TR) - generates
- Equation Reading (ER) - generates
- Model Analysis (MA) - consumes

GrFN integrates the extracted _Function Network_ representation of source code (the result of Program Analysis) along with associated extracted comments, links to natural language text (the result of natural language processing by Text Reading), and links to and representation of equations (the result of equation extraction by Equation Reading).


### Spec Notation Conventions

This specification (spec) document describes the GrFN JSON schema, specifying the JSON format that is to be generated by Program Analysis, Text Reading and Equation Reading. Model Analysis is the current main consumer; we also hope that other scientific model analysis systems (e.g., from the ASKE Program) will also be consumers and/or generators.

In this document we adopt a simplified [Backus-Naur Form (BNF)](https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form)-inspired grammar specification convention combined with a convention for intuitively defining JSON attribute-value lists. The schema definitions and instance GrFN examples are rendered in `monospaced font`, and interspersed with comments/discussion.

Following BNF convention, elements in `<...>` denote nonterminals, with `::=` indicating a definition of how a nonterminal is expanded. We will use some common nonterminals with standard expected interpretations, such as `<string>` for strings, `<integer>` for integers, etc. Many of the definitions below will specify JSON attribute-value lists; when this is the case, we will decorate the nonterminal element definition by adding `[attrval]`, as follows::

    <element_name>[attrval] ::= 

We will then specify the structure of the JSON attribute-value list attributes (quoted strings) and their value types using a mixture of [JSON](https://www.json.org/) and [BNF](https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form).

For example, the following grfn_spec definition

	<grounding_metadata_spec>[attrval] ::=
		"source" : <string>
		"type" : "definition" | "units" | "constraint"
		"value" : <string> 

specifies the structure of the grfn JSON instance:

	{
		"source" : "http://epirecip.es/epicookbook/chapters/sir/intro",
		"type" : "definition",
		"value" : "susceptible individuals"
	}

We also use the following conventions in the discussion below:

- 'FUTURE': Tags anticipated extensions that may be needed but not yet 
supported.
- 'CHOICE': Captures discussion of a CHOICE that does not yet have a clear 
resolution
- 'FOR NOW': Tags the approach being currently taken, eiher in response to FUTURE or CHOICE.


## Identifiers: grounding, scopes, namespaces and gensyms

### Preamble

The current GrFN design strategy is to separate _identifiers_ (any program symbol used to denote a program element) from the _program elements_ themselves (namely, variables and functions), as each program element will be denoted by one or more identifiers, and the different types of program elements themselves have intended scientific modeling "functions": variables (may) represent aspects of the modeled domain, and functions represent processes that change variable states.

A key role of identifiers in the GrFN representation is to enable _linking_ (grounding) of program elements to information extracted by Text and Equation Reading. Identifiers bring together two types of information that make this linking possible:

1. The identifier name ([`<base_name>`](#base-name)) and namespace context ([scope and namespace paths](#scope-and-namespace-paths)) of the identifier as it appears in program source context are used as evidence of potential semantic relations to other textual sources based on string similarity or name embedding;
2. Information about the location and neighborhood in source code where the identifier is used (a [`<source_code_reference>`](#grounding-and-source-code-reference)) provides additional sources of evidence, based on proximity to source code comments and docstrings, as well as proximity to uses of other identifiers.

An linking/grounding inference algorithm uses string or embedding similarity between base, scope and namespace names and information extracted from documents by Text Reading to form hypotheses of potential links between identifiers and the text-extracted information. Such link hypotheses are explicitly connected to identifiers in GrFN (by instances of [`<grounding_metadata_spec>`](#grounding-metadata-spec)).

Program elements (variables and functions) are associated with identifiers based on declarations in source code, and thereafter, the use of the same identifiers elsewher in source code is a denotation of the variable or function. Information from Text Reading (e.g., "mentions" of domain concept terms or phrases in text) associated with identifiers is then linked to program elements by association of the idenifier. For example, the indicator 'S' may store an integer and be described in documentation as associated with (defined as representing) the "susceptible population".

### Identifier

An [identifier](#identifier-specification) is a symbol used to uniquely identify a program element in code, where a *program element* is a 

- constant value
- variable
- function

> FUTURE: possibly: type, class

More than one identifier can be used to denote the same program element, but an identifier can only be associated with one program element at a time.

### Grounding and source code reference

Identifiers play a key role in connecting the model as implemented in source code to the target domain that it models. *Grounding* is the task of inferring what aspect of the target scientific domain a program element may correspond to. Identifiers, by their [(base) name(s)](#base-name) and the context of their declaration and use (i.e., where they occur in code, through their [scope and namespace](#scope-and-namespace-paths)), and the linline comment and doc strings that occur around them, provide clues to what program elements are intended to represent in the target domain. For this reason, we need to associate with identifiers several pieces of information. This information will be collected during program analysis and associated with the identifier declaration:
    
>FUTURE: General handling of pointers/references (related to the concept of having an "alias") will require care, as this introduces the possiblity of multiple identifiers being used to refer to the same program element, and also a single identifier being used to refer to different program elements in different contexts. In the general case it is not possible to determine all pointer references *statically*. (DSSAT does include some pointers.)
	
To facilitate grounding inference, the [`<identifier_spec>`](#identifier-specification) will have a "source\_references" attribute whose value is a list of `<source_code_reference>`s:

	<source_code_reference> ::= <string>

The string contains information to indicate the location within the source code where an identifier was used. 

>TODO: Program Analysis and Text and Equation Reading (NLP processing of comments and source literature) will determine how source code references are represented within the string. It may be sufficient to have a single line number to represent the source code line within which the identifier was used.

### Base Name

The `<base_name>` is intended to correspond (when available) to the identifier token name as it appears in the source language (e.g., Fortran). The `<base_name>` is itself a string

    <base_name> ::= <string>

but follows the conventions of [python identifier specification rules](https://docs.python.org/3/reference/lexical_analysis.html#identifiers) (which includes Fortran naming syntax).

>FUTURE: may extend this as more source languages are supported.

Below, we specify the conventions for `<base_name>`s of identifiers that do not originate in the source code:

- [Variable Naming Convention](#variable-naming-convention)

- [Function Naming Convention](#function-naming-convention)

### Scope and Namespace Paths

Identifiers may have the same [`<base_name>`](#base-name) (as it appears in source code) but be distinguished by either (or both) the "[scope](https://en.wikipedia.org/wiki/Scope_(computer_science))" and "[namespace](https://en.wikipedia.org/wiki/Namespace)" within which they are defined in the source code.

Each source language has its own rules for specifying scope and namespace, and it will be the responsibility of each program analysis module (e.g., Fortran `for2py`) to identify the hierarchical structure of the context that uniquely identifies the specific scope and/or namespace within which an identifier [`<base_name>`](#base-name) is defined. However, generally scopes and namespaces may be defined hierarchically, such that the name for each level of the hierarchy taken together uniquely define the context. A "path" of names appears to be sufficient to generally represent the hierarchical context for either a specific scope or namespace. In general, names for a path are listed in order from general (highest level in the hierarchy) to specific.

Examples:

- `<scope_path>`: As will be described below, program analysis will assign unique names for scopes (see [discussion below under conditional, container and loop\_plate function naming convention](#function-naming-convention)). Given these names, the scopes of the two inner loops within the outer loop of function `foo` in this example,
    
    ```
    def foo():
        for i in range(10):      # assigned name 'loop$1'
            for j in range(10):  # assigned name 'loop$1' (in the scope
                                 #    ... of the outer loop$1)
                x = i*j
            for k in range(10):  # assigned name 'loop$2' (also in the
                                 #    ... scope of the outer loop$1)
                z = x+1
    ```
    
    ... would be uniquely specified by the following path (respectively):
    
    ```
    ["foo", "loop$1", "loop$1"]
    ```
    
    ```
    ["foo", "loop$1", "loop$2"]
    ```
    
    In general, it is not necessary within GrFN to independently declare scopes. Instead, we simply specify the `<scope_path>` in an indicator declaration as a list of strings under the "scope" attribute in the identifier declaration (below).
    
    ```
    <scope_path> ::= list of <string>
    ```
    
	The "top" level of the file (i.e., not enclosed within another program block context) will be assigned the default scope name of "\_TOP". All other scopes are either explicitly named (such as a named function), or are assigned a unique name by program analysis according to the rules of the type of scope (such as container, loop, conditional, etc), defined below. In such cases other than top, there is no need to include the "\_TOP" in the path -- it will be assumed that those named scopes are all within the default top-level scope.
    
- `<namespace_path>`: Different languages have different conventions for defining namespaces, but in general they are either (1) explicitly defined within source code by namespace declarations (such as Fortran "modules" or C++ "namespace"s), or (2) implicitly defined by the project directory structure within which a file is located (as in Python). In the case of namespaces defined by project directory structure, two files in different locations in the project directory tree may have the same name. To distinguish these, program analysis will capture the path of the directory tree from the root to the file. The final name in the path, which is the name of the source file, will drop the file extension. For example, the namespace for file `baz.py` within the following directory tree
    
    ```
    foo/
        bar/
            baz.py
    ```
    
    would be the uniquely specified by the following path:
    
    ```
    ["foo", "bar", "baz"]
    ```

	In the case of declared namespaces, the namespace declaration will determine the path (which may only consist of one string name).

    Again, it is not necessary within GrFN to independently declare a namespace; like the `<scope_path>`, we specify the `<namespace_path>` within an identifier declaration as a list strings under the "namespace" attribute in the identifier declaration:
	
	```
    <namespace_path> ::= list of <string>
    ```
    
    Like the `<scope_path>`, the string names of the path uniquely defining the namespace are in in order from general to specific, with the last string name either being the implicit namespace defined by the source code file, or the user-defined name of the namespace.

### Path Strings

It will be convenient to be able to express [`<scope_path>`](#scope-and-namespace-paths)s and [`<namespace_path>`](#scope-and-namespace-paths)s using single strings within GrFN (particularly when building an identifier string). For this we introduce a special string notation in which the string names that make up a path are expressed in order but separated by periods. These representations will be referred to as the `<scope_path_string>` and `<namespace_path_string>`, respectively. The string representations of the [`<scope_path>`](#scope-and-namespace-paths) and [`<namespace_path>`](#scope-and-namespace-paths) examples above would be:

- Example `<scope_path_string>`:
    
    ```
    "foo.loop$1.loop$1"
    ```
    
    and
    
    ```
    "foo.loop$1.loop$2"
    ```

- Example `<namespace_path_string>`:
	
	```
	"foo.bar.baz"
	```

### Identifier String

Identifiers are uniquely defined by their [`<base_name>`](#base-name), [`<scope_path>`](#scope-and-namespace-paths), and [`<namespace_path>`](#scope-and-namespace-paths). It will be convenient to refer unambiguously to any identifier using a single string, outside of the identifier specification declaration (defined below). We define an `<identifier_string>` by combining the 'namespace', 'scope' and 'base_name' (in that order) within a single string by separating the [`<namespace_path_string>`](#path-strings), [`<scope_path_string>`](#path-strings) and [`<base_name>`](#base-name) by double-colons:

    <identifier_string> ::= "<namespace_path_string>::<scope_path_string>::<base_name>"

`<identifier_string>`s will be used to denote identifiers as they are used in variable and function specifications (described below).

### Identifier Gensym

One of the outputs of program analysis is a functionally equivalent version of the original source code and lambda functions (described below), both expressed in Python (as the intermediate target language). All identifiers in the output Python must match identifiers in GrFN. Since capturing the semantics (particularly the namespace and scope context) results in a representation that does not appear to be consistently expressible in legal Python symbol names, we will use `<gensym>`s that can be represented (generally more compactly) as legal Python names and associated uniquely with identifiers.

>FUTURE: Create a hashing function that can translate uniquely back and forth between `<gensym>`s and identifier strings.

>FOR NOW: Generate `<gensym>`s as Python names that start with a letter followed by a unique integer. The letter could be 'g' for a generic gensym (e.g., g2381), or 'v' to indicate a variable identifier (e.g., v921) and 'f' to indicate a function identifier (e.g. f38).

Each identifier will be associated one-to-one with a unique `<gensym>`.

### Grounding Metadata spec

Text Reading is currently working on extracting three types of information that can be associated with identifiers:

- Definitions (e.g., 'wind speed', 'susceptible individuals')
- Units (e.g., 'millimeters', 'per capita')
- Constraints (e.g., '> 0', '<= 100')

Each of these types can be expressed as an instance of a `<grounding_metadata_spec>`:

	<grounding_metadata_spec>[attrval] ::=
		"source" : <string>
		"type" : "definition" | "units" | "constraint"
		"value" : <string> 

> FUTURE: add confidence/belief score

### Identifier Specification

Each identifier within a GrFN specification will have a single `<identifier_spec>` declaration. An identifier will be declared in the GrFN spec JSON by the following attribute-value list:

    <identifier_spec>[attrval] ::=
        "base_name" : <base_name>
        "scope" : <scope_path>
        "namespace" : <namespace_path>
        "source_references" : list of <source_code_reference>
        "gensym" : <gensym>
        "grounding" : list of <grounding_metadata_spec>

## Variable and Function Identifiers and References

### Variable Naming Convention

A variable name will be an [`<identifier_string>`](#identifier-string):

    <variable_name> ::= <identifier_string>

A top level source variable named ABSORPTION would then simply have the [`<base_name>`](#base-name) of "ABSORPTION" plus the relevant [`<namespace_path_string>`](#path-strings) and [`<scope_path_string>`](#path-strings).

When there are two (or more) separate instances of new variable declarations in the same context (same namespace and scope) using the same name, then we'll add an underscore and number to the [`<base_name>`](#base-name) to distinguish them. For example, if ABSORPTION is defined twice in the same namespace and scope, then the [`<base_name>`](#base-name) of the first (in order in the source code) is named:

    "ABSORPTION_1"

And the second:

    "ABSORPTION_2"

Finally, in some cases (described below), program analysis will introduce variables (e.g., when analyzing conditionals). The naming conventions for the [`<base_name>`](#base-name) of such introduced variables are described below.

### Variable Reference

    <variable_reference>[attrval] ::= 
        "variable" : <variable_name>
        "index" : <integer>

In addition to capturing source code variable environment context in variable declarations, we also need a mechanism to disambiguate specific instances of *use* of the same variable within the same context to accurately capture the logical order of variable value updates. In this case, we consider this as a repeated reference to the same variable. The semantics of repeated reference is captured by the variable "index" attribute of a `<variable_reference>`. The index integer serves to disambiguate the execution order of the variable state references, as determined during program analysis.

### Function Naming Convention

Function names, like variable names, are also ultimately [identifiers](#identifier-specification) that will commonly be referenced within GrFN by their [`<identifier_string>`](#identifier-string) (and therefore include their [`<namespace_path_string>`](#path-strings) and [`<scope_path_string>`](#path-strings))

    <function_name> ::= <identifier_string>

However, there are additional rules for determining the [`<base_name>`](#base-name) of the function. Because of this particular set of rules, the [`<base_name>`](#base-name) of the function name will be referred to as a `<function_base_name>`. The general string format for a `<function_base_name>` is:

    <function_base_name> ::= <function_type>[$[<var_affected>|<code_given_name>]]

The `<function_type>` is the string representing which of the four types the function belongs to (the types are described in more detail, below): ["assign"](#function-assign-specification), ["condition"](#function-assign-specification), ["decision"](#function-assign-specification), ["container"](#function-container-specification), ["loop\_plate"](#function-loop-plate-specification). In the case of a loop\_plate, we will name the specific loop using the generic name "loop" along with an integer (starting with value 1) uniquely distinguishing loops within the same namespace and scope.

The optional `<code_given_name>` is used when the function identified by program analysis has also been given a name within source code. For example, in this python example:

```javascript
def foo():
    ...
```

the function foo is a type of "container" and its `<code_given_name>` is "foo", making the `<function_base_name>` be

    "container$foo"

The optional `<var_affected>` will only be relevant for assign, condition and decision function types, and the name of the variable affected will be added after the `<function_type>` and `$`. For example, a condition involving the setting of the (inferred) boolean variable IF\_1 would have the `<function_base_name>`:

    "condition$IF_1"

Here are example function names for each function type. In each example, we assume the function is defined within the scope of another function, UPDATE\_EST, and the namespace CROP\_YIELD.

-   [**Assign**](#function-assign-specification): An assignment of the variable with the `<identifier_string>` "CROP\_YIELD::UPDATE\_EST::YIELD\_EST" (which denotes the identifier with `<base_name>` "YIELD\_EST" in the scope of the function UPDATE\_EST declared in the namespace CROP\_YIELD) has the `function_base_name>`:
    
    	"assign$CROP_YIELD::UPDATE_EST::YIELD_EST"
    
	If this assignment takes place in the function UPDATE\_EST and the namespace CROP\_YIELD, then the full [`<identifier_string>`](#identifier-string) of the function identifier would be:

        "CROP_YIELD::UPDATE_EST::assign$CROP_YIELD::UPDATE_EST::YIELD_EST"

-   [**Condition**](#function-assign-specification): A condition assigning the (inferred) boolean variable IF\_1 in the scope of the function UPDATE\_EST of the namespace CROP\_YIELD would have the [`<identifier_string>`](#identifier-string):

        "CROP_YIELD::UPDATE_EST::condition$IF_1"

-   [**Decision**](#function-assign-specification): A decision function assigns a variable a value based on the (outcome) state of a condition variable. If the variable "YIELD\_EST" (from the namespace "CROP\_YIELD" and scope "UPDATE\_EST") is being updated as a result of a conditional outcome in the namespace "CROP\_YIELD" and scope "DERIVE\_YIELD", then the [`<identifier_string>`](#identifier-string) would be:
    
    	"CROP_YIELD::DERIVE_YIELD::decision$CROP_YIELD::UPDATE_EST::YIELD_EST"

-   [**Container**](#function-container-specification): A container function declared in source code to have the name CROP\_YIELD would then have the `<code_given_name>` of CROP\_YIELD, and if this was declared at the top level of a file (defining the namespace) called CROP\_YIELD would have the `<function_base_name>`:

        "container$CROP_YIELD"
    
    and the full `<identifier_string>` of:
    
        "CROP_YIELD::NULL::container$CROP_YIELD"
    
    (Note that the first occurrence of "CROP\_YIELD" in the string is for the namespace, the NULL is because it's defined at the top level, and then the second occurrence of "CROP\_YIELD" is the `<code_given_name>` of CROP\_YIELD.)

-   [**Loop_plate**](#function-loop-plate-specification): Loops themselves are not assigned identifiers within source code, so identifiers will be assigned during program analysis. As described above, the `<function_base_name>` of the loop\_plate function type is "loop" followed by a '$' and an integer starting from `1` that distinguishes the loop from any other loops occurring in the same namespace and scope.

    -   A single loop within the function CROP\_YIELD of the namespace CROP\_YIELD has the `<identifier_string>`:

            "CROP_YIELD::CROP_YIELD::loop$1"

    -   The third of three loops within the function CROP\_YIELD of namespace CROP\_YIELD:

            "CROP_YIELD::CROP_YIELD::loop$3"

    -   A loop nested in the context of the second loop, "loop$2", in the CROP\_YIELD function within the CROP\_YIELD namespace:

            "CROP_YIELD::CROP_YIELD.loop$2::loop$1"

    -   An assignment of the variable "CROP\_YIELD::\_TOP::RAIN" (i.e., the variable "RAIN" was defined in the default "top"-level scope, within the namespace CROP\_YIELD) within a single loop in the CROP\_YIELD function in the CROP\_YIELD namespace:

            "CROP_YIELD::CROP_YIELD.loop$1::assign$CROP_YIELD::_TOP::RAIN"
        
        (Note that the above string is still unambiguous to parse to recover the components pieces of the name: the first two names separated by '::' are the [`<namespace_path_string>`](#path-strings) followed by the [`<scope_path_string>`](#path-strings), with the rest being the `<function_base_name>` of the function, which itself is an "assign" of a variable that itself is a complete [`<identifier_string>`](#identifier-string))


## Top-level GrFN Specification

The top-level structure of the GrFN specification is the `<grfn_spec>` and is itself a JSON attribute-value list, with the following schema definition:

    <grfn_spec>[attrval] ::=
        "date_created" : <string>
        "source" : list of <source_code_file_path>
        "start": list of <string>
        "identifiers" : list of <identifier_spec>
        "variables" : list of <variable_spec>
        "functions" : list of <function_spec>

The "date\_created" attribute is a string representing the date+time that the current GrFN was generated (this helps resolve what version of the program analysis code (e.g., for2py) was used).

There may be a single GrFN spec file for multiple source code files.

>CHOICE: A source file may define identifiers and program elements that are then referenced/used in many other system program unit components (other files). We can defined the concepts of a "Program" as the collection of source code from a file plus any other files containing source code that it references/uses. If we have a single GrFN spec for each "Program", then we will be repeatedly reproducing many identifiers and other program element declarations (variables, functions). The alternatives are:

>1. A single GrFN spec for a given "Program" and live with the redundant re-representation of source code that is shared across different units. 
>2. Have a single GrFN spec for each individual file and develop method for importing/using GrFN specs that are used by other GrFN specs. (A variant of this approach could allow only including GrFN spec for the parts of the shared source code that are directly relevant to a "Program".)

>FOR NOW: Go with Option (1): The main target of a GrFN spec file is _all_ of the source code files involved in defining a "Program".
    
>FUTURE: Add ability for GrFN specs to "import" and/or "use" other GrFN specs of other modules (leading to the more efficient option (2)).

>FOR NOW: the "source" attribute is a list of one or more `<source_code_file_path>`s. The `<source_code_file_path>` identifying a source file is represented the same way as a [`<namespace_path>`](#scope-and-namespace-paths), except that the final name (the name of the file itself) _will_ include the file extension.

It is conventient to explicitly represent the functions that are intended to be called as the "top level" functions of the "Program". We will refer to these as "start" points. These may be explicity specified by a programmer/user. 

>FOR NOW: "start" points will be manually specified.

>FUTURE: Consider methods of automatically inferring the "start" points. For example, these might be the roots of the computation graph representing the "Program". However, there are subtleties here: e.g., function call loops (fn1 calls fn2, which calls fn1).

To capture this concept, the "start" attribute is a list of zero or more names of the entry point(s) of the (Fortran) source code (for example, the PROGRAM module). These will be function [`<identifier_string>`](#identifier-string)s. In the absence of any entry point, this value will be an empty list: `[]`.

The "identifiers" attributes contains a list of [`<identifer_spec>`](#identifier-specification)s.

>FUTURE: It may also be desirable to add an attribute to represent the program analysis code version used to generate the GrFN (as the program analysis code is evolving and each change has different properties).

>FOR NOW: "dateCreated" will play this role.

The "variables" attribute contains a list of [`<variable_spec>`](#variable-specification)s.

>(NOTE: In GrFN version 0.1.m3, [`<variable_spec>`](#variable-specification)s were defined in the context of [`<function_spec>`](#function-specification)s. With need to iterate over variables (e.g. when performing inference for associating comments and text with variable definitions), it is more convenient to group variable specifications at the top level. Starting in GrFN version 0.1.m5, references to variables within functions will use [`<identifier_string>`](#identifier-string)s to identify the variables, and [`<variable_spec>`](#variable-specification)s will be listed in the top-level GrFN "variables" attribute.)

A (partial) example instance of the JSON generated for a `<grfn_spec>` of an analyzed file in the path 'crop\_system/yield/crop\_yield.py' is:

```javascript
{
    "dateCreated": "20190127",
    "source": [["crop_system", "yield", "crop_yield.py"]],
    "start": ["MAIN"],
    "identifiers": [... identifier_specs go here...]
    "variables": [... variable_specs go here...]
    "functions": [... function_specs go here...]
}
```

## Variable Specification

    <variable_spec>[attrval] ::=
        "name" : <variable_name>
        "domain" : <variable_domain_type>
        "mutable" : TRUE | FALSE

As defined above, the [`<variable_name>`](#variable-naming-convention) is itself an [`<identifier_string>`](#identifier-string).

The "mutable" attribute specifies whether the variable value _can_ (TRUE) or _cannot_ (FALSE) be changed. (Note that the values TRUE and FALSE are JSON Boolean values.) Mutability is determined by Program Analysis, and Model Analysis can use this information in sensitivity anlysis (if the constant value of a non-mutable variable can be determined, then the variable does not need to be varied in sensitivity analysis).

>TODO: Possibly make explicit specification of "mutable" optional, with default value TRUE, so only need to explicitly specify when FALSE. Does this cause trouble for parsing?

Some languages (including Fortran and Python) provide mechanisms for making variable declarations private (such as Python's name mangling, by prepending an underscore to a variable name).

>FOR NOW: Our hypothesis is that simply prepending another underscore (following python [name mangling](https://docs.python.org/2/tutorial/classes.html#private-variables-and-class-local-references)) will make the "private" variable [`<base_name>`](#base-name) unique from other variable names. Also, Program Analysis will "preserve" the semantics of privacy by ensuring there are no outer-scope references to a private variable, and this will carry through in explicit references (or lack thereof, in this case) captured in the GrFN spec (in particular, in the [`<function_reference_spec>`](#function-reference-specification)s of the "body" of container and loop\_plate function specs).

### Variable Value Domain

    <variable_domain_type> ::= <string>

The "domain" attribute of a [`<variable_spec>`](#variable-specification) specifies what values the variable can be assigned to. To start, we will keep things simple and restrict ourselves to four types that can be specified as strings:

-   "real" (i.e. a floating-point number)
-   "integer"
-   "boolean"
-   "string"

(The idea of the variable domain is intended to be close to the idea of
the "support" of a random variable, although should also correspond to
standard data types.)

>FUTURE: Extend to accommodate arrays.

>FUTURE:

>-  Accommodate other structures: Unions, composite data structures, classes.
>-  Extend the domain specification to represent whether there are bounds on the values (e.g., positive integers, or real values in (0,10\], etc.). When we move to doing this, the value of "domain" will itself become a new JSON attrval type.

Python is a strongly-typed language, but is also a dynamically typed language. However, that's not to say that there is no type specification in Python. Python 3 now provides nascent support for explicit typing via [type
hints](https://docs.Python.org/3/library/typing.html).

>FUTURE: Explore whether/how type hints get represented in the AST. This will matter when we get to adding more explicit variable domain semantics in later model analysis.

For our purposes in the near term, we do want to capture what type and value-domain information is available; there are two main sources of this information (parentheses note the analysis process that may extract/infer/preserve this information):

1.  **Source Declarations**: (Program Analysis) Fortran statically specifies types in declarations. 
We can preserve type information in Program Analysis-generated Python code by using type hints or by adding to generated docstrings.
2.  **Docstrings**: (Text Reading) Source code docstrings and comments may provide information about types and value ranges that can be inferred by Text Reading.

### \<variable\_spec\> examples

Here are three examples of `<variable_spec>` objects:

-   Example of a "standard" variable MAX\_RAIN within the CROP\_YIELD function of the CROP namespace:

    ```javascript
    {
        "name": "CROP::CROP_YIELD::MAX_RAIN",
        "domain": "real",
        "mutable": FALSE
    }
    ```

-   Example of loop index variable DAY in the context of the second instance of a loop in the function CROP\_YIELD (in the CROP namespace):

    ```javascript
    {
        "name": "CROP::CROP_YIELD.loop$2::DAY"
        "domain": "integer",
        "mutable": TRUE
    }
    ```

-   Example of variable introduced (inferred) when analyzing a conditional statement that is within the named function UPDATE\_EST of the CROP namespace:

    ```javascript
    {
        "name": "CROP::UPDATE_EST::IF_1"
        "domain": "boolean",
        "mutable": TRUE
    }
    ```

## Function Specification

Next we have the `<function_spec>`. 

All `<function_spec>`s include a [`<function_name>`](#function-naming-convention), following the [function naming convention described above](#function-naming-convention).

There are five types of functions; three types can be expressed using the same attributes in their JSON attribute-value list ([`<function_assign_spec>`](#function-assign-specification)), while the others ([`<function_container_spec>`](#function-container-specification), [`<function_loop_plate_spec>`](#function-loop-plate-specification)) require different attributes. So this means there are three specializations of the \<function\_spec\>, one of which ([`<function_assign_spec>`](#function-assign-specification)) will be used for three function types.

    <function_spec> ::=
        <function_assign_spec>         # covers "assign", "condition", "decision"
        | <function_container_spec>    # type "container"
        | <function_loop_plate_spec>   # type "loop_plate"

All three specs will have a "type" attribute with a string value that will unambiguously identify which type of function is being specified. The five possible types are:

- "[assign](#function-assign-specification)"
    - "[condition](#function-assign-specification)" (a special case of Assign)
    - "[decision](#function-assign-specification)"  (a special case of Assign)
- "[container](#function-container-specification)"
- "[loop\_plate](#function-loop-plate-specification)"

All `<function_spec>`s will also have a "name" attribute with a unique
[`<identifier_string>`](#identifier-string) (across `<function_spec>`s), as described above under the [Function Naming Convention](#function-naming-convention) section; as described in that section, the function [`<base\_name>`](#base-name) will include the function type name, but having the explicit type attribute makes JSON parsing easier.

### Function Assign Specification

A `<function_assign_spec>` denotes the setting of the value of a
variable. The values are assigned to the "target" variable (denoted by
a [`<variable_reference>`](#variable-reference) or [`<variable_name>`](#variable-naming-convention)) and the value is determined by the "body" of the assignment, which itself may either be a literal value (specified by [`<function_assign_body_literal_spec>`](#function-assign-body-literal)) or a lambda function (specified by [`<function_assign_body_lambda_spec>`](#function-assign-body-lambda)).

    <function_assign_spec>[attrval] ::=
        "name" : <function_name>
        "type" : "assign" | "condition" | "decision"
            # note that the value of the "type" is a literal/terminal 
            # value of the grammar
        "sources" : list of [ <function_source_reference> | <variable_name> ]
        "target" : <function_source_reference> | <variable_name>
        "body" : <function_assign_body_literal_spec> 
                 | <function_assign_body_lambda_spec>

There are three types of assign functions, distinguished by the value of the attribute "type".

- **"assign"**: This represents the general case of assignment of a variable to some value.

- **"condition"**: In the special case where Program Analysis is analyzing a conditional (i.e., "if") statement, then Program Analysis will infer a new boolean target variable, and the computation of the condition itself will be represented by a function assigning the inferred boolean variable value. Semantically, this is nothing more than an assignment of a boolean variable, but conceptually it will be useful to distinguish assignments used for conditions from other assignments.

- **"decision"**: Also as part of analyzing a conditional, any variables whose values are updated as a result of the condition outcome must have their values updated. These will be updated by "decision" assignment functions, whose target is the variable being updated, and the computations will involve the state of the conditional variable, the previous state of the variable being updated, and possibly other variable values. Again, semantically this is nothing more than an assignment, but is useful to distinguish from other assignments.

The identifier conventions for assign, condition and decision functions is described above in the section on [Function Naming Convention](#function-naming-convention).

For "sources" and "target": When there is no need to refer to the variable by its relative index, then [`<variable_name>`](#variable-naming-convention) (itself an [`<identifier_string>`](#identifier-string)) is sufficient, and index will be assumed to be 0 (if at all relevant). 

In other cases, the variables will be referenced using the `<function_source_reference>`, to indicate the return value of the function. There may also be cases where the sources can be a function, either built-in or user-defined. These two will be referenced using `<function_source_reference>` defined as:

    <function_source_reference> ::=
       "name" : [ <variable_name> | <function_name> ]
       "type" : "variable" | "function"

#### Function assign body Literal

The `<function_assign_body_literal_spec>` asserts the assignment of a `<literal_value>` to the target variable. The `<literal_value>` has a data type (corresponding to one of our four domain types), and the value itself will be represented generically in a string (the string will be parsed to extract the actual value according to its data type).

    <function_assign_body_literal_spec>[attrval] ::=
        "type" : "literal"
        "value" : <literal_value>

    <literal_value>[attrval] ::=
        "dtype" : "real" | "integer" | "boolean" | "string"
        "value" : <string>

#### Function assign body Lambda

When more computation is done to determine the value that is being assigned to the variable in the [`<function_assign_spec>`](#function-assign-specification), then `<function_assign_body_lambda_spec>` is used.

    <function_assign_body_lambda_spec>[attrval] ::=
        "type" : "lambda"
        "name" : <function_name>
        "reference" : <lambda_function_reference>

>FUTURE: Eventually, we can expand this part of the grammar to accommodate a restricted set of arithmetic operations involved in computing the final value (this is now of interest in the World Modelers program and we\'re interested in supporting this in Delphi). 

The `<lambda_function_reference>` references the source code function in the `lambdas.py` file that does the computation, in the translated Python generated by program analysis.

	<lambda_function_reference> ::= <string>

>NOTE: The `<lambda_function_reference>`, which refers to the post-analysis lambdas function created by Program Analysis is *NOT* to be confused with a [`<source_code_reference>`](#grounding-and-source-code-reference), which refers to the original source code that was analyzed (i.e., the original Fortran source code).

Any variables that are required as arguments to the lambda function must correspond to the "source" list of variables ([`<variable_name>`](#variable-naming-convention) references) in the [`<function_assign_spec>`](#function-assign-specification).

As noted above, due to the more semantically rich identifier specification and [`<identifier_string>`](#identifier-string) representation, it is not straightforward to use the [`<identifier_string>`](#identifier-string) as the python symbol in the translated Python generated by program analysis. Instead, function and variable identifiers will be represented in the generated Python using their [gensym](#identifier-gensym). For debugging and visualization purposes, the generated Python code may be displayed with [`<identifier_string>`](#identifier-string)s (or some version that is closer to legal Python naming, although in general it does not appear to be possible to create "safe" Python names directly from [`<identifier_string>`](#identifier-string)s).

### Function Container Specification

A `<function_container_spec>` represents the grouping of a set of variables and how they are updated by functions. Generally the "function container" corresponds to functions (or subroutines) defined in source code. A "function container" is also defined for the "top" level of a source code file.

    <function_container_spec>[attrval] ::=
        "name" : <function_name>
        "type" : "container"
        "DOCS" : <string>
        "input" : list of [ <variable_reference> | <variable_name> ]
        "output" : list of <variable_reference> | <variable_name>
        "body" : list of <function_reference_spec>

There will be a container function for each source code function. For this reason, we need an "input" variable list (of 0 or more variables) as well as an "output" variable. In Python, a function only returns a value if there is an explicit return expression. Otherwise it returns `None`.

The "body" is specified by a list of [`<function_reference_spec>`](#function-reference-specification)

Case 1: subroutine

```python
def foo1_subroutine(x,y):
    x = y

def foo2_subroutine():
    Integer z, y, w
    y = 5
    foo1(z,y)
    foo1(w,y)
```

now z = 5 and w = 5

Case 2: fortran function with simple return

```python
def foo():
    x <-
    return x

def foo2():
    y = foo()
```

Case 3: fortran function with return expression

```python
def foo():
    return x+1
```

becomes\...

```python
def foo():
  foo_return1 = x+1

return foo_return1
```

Case 4: conditional return statements

```python
def foo(): #fortran function
    if(x):
        return x
    else:
        return y
```

### Function Reference Specification

    <function_reference_spec>[attrval] ::=
        "function" : <function_name>
        "input" : list of [ <variable_reference> | <variable_name> ]
        "output" : <variable_reference> | <variable_name>

The `<function_reference_spec>` defines the "wiring" of functions, specifying associations of variables to function inputs (arguments) and the function output variable (FOR NOW: assuming all functions set zero or one variable state).

### Function Loop Plate Specification

The concept behind a "loop plate" is a generalization of the [probabilistic graphical model](https://en.wikipedia.org/wiki/Graphical_model) convention for [plate notation](https://en.wikipedia.org/wiki/Plate_notation), extended to represent dynamic processes (a specification of a sequence of state changes).

>TODO: Provide more background on the "loop plate" concept.

    <function_loop_plate>[attrval] ::=
        "name" : <function_name>
        "type" : "loop_plate"
        "input" : list of <variable_name>
        "index_variable" : <variable_name>
        "index_iteration_range" : <index_range>
        "condition" : <loop_condition>
        "body" : list of <function_reference_spec>

The "input" list of [`<variable_name>`](#variable-naming-convention) objects should list all variables that are set in the scope outside of the loop\_plate.

The current loop\_plate specification is aimed at handling for-loops. (assumes "index\_variable" and "index\_iteration\_range" are specified)

>FUTURE: Generalize to do-while loop by just relying on the "condition" `<loop_condition>` to determine when loop completes. We can then remove "index\_variable" and "index\_iteration\_range". There will still need to be a mechanism for identifying index\_variable(s).

The "index\_variable" is the named variable that stores the iteration state of the loop; the naming convention of this variable is described above, in the Variable naming convention section. The only new element introduced is the `<index_range>`:

    <index_range>[attrval] ::=
        "start" : <integer> | <variable_reference> | <variable_name>
        "end" : <integer> | <variable_reference> | <variable_name>

This definition permits loop iteration bounds to be specified either as literal integers, or as the values of variables.


# Change Log


Inspired by [Keep a Changelog](https://keepachangelog.com)

This project does not (yet) adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)


## `[0.1.m5]` - 2019-05-01:

### Added
- Added "mutable" attribute to [`<variable_spec>`](#variable-specification).
- Added "variables" attribute to top-level [`<grfn_spec>`](#top-level-grfn-specification), which contains the list of all `<variable_spec>`s. This change also means that [`<function_spec>`](#function-specification)s no longer house [`<variable_spec>`](#variable-specification)s, but instead just the [`<variable_names>`](#variable-naming-convention) (which themselves are [`<identifier_string>`s](#identifier-string)).
- Added links to help topic navigation.

### Changed
- Clarified distinction between [`<source_code_reference>`](#grounding-and-source-code-reference)s (linking identifiers to where they are used in the analyzed source code) and [`<lambda_function_reference>`](#function-assign-body-lambda)s (which denote functions in the Program Analysis-generated lambdas file source code); previously these two concepts were ambiguous.

### Removed
- Removed [`<identifier_spec>`](#identifier-specification) "aliases" attribute. To be handled later as part of pointer/reference analysis.


## `[0.1.m3]` - 2019-03-01:

### Added
- Addition of identifiers: `<identifier_spec>`, `<identifier_string>`, and `<gensym>` (for identifiers in generated code)

### Changed
- Revision of Introduction
- Updates to naming conventions for variables and functions
- General cleanup of discussion throughout


## Releases
- [unreleased](https://github.com/ml4ai/delphi/blob/grfn/docs/grfn_spec.md)
- [0.1.m5](https://github.com/ml4ai/automates/blob/master/documentation/deliverable_reports/m5_final_phase1_report/GrFN_specification_v0.1.m5.md)
- [0.1.m3](https://github.com/ml4ai/automates/blob/master/documentation/deliverable_reports/m3_report_prototype_system/GrFN_specification_v0.1.m3.md)
- [0.1.m1](https://github.com/ml4ai/automates/blob/master/documentation/deliverable_reports/m1_architecture_report/GrFN_specification_v0.1.md)


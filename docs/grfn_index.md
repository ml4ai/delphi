## grfn_spec Index

- [`<grfn_spec>`](#top-level-GrFN-specification)[attrval] ::=
	- "date_created" : `<string>`
	- "source" : list of [`<source_code_file_path>`](#scope-and-namespace-paths)
	- "start": list of `<string>`
	- "identifiers" : list of [`<identifier_spec>`](#identifier-specification)[attrval] ::=

		- "base_name" : [`<base_name>`](#base-name)
		- "scope" : [`<scope_path>`](#scope-and-namespace-paths)
		- "namespace" : [`<namespace_path>`](#scope-and-namespace-paths)
		- "source\_references" : list of [`<source_code_reference>`](#grounding-and-source-code-reference)
		- "gensym" : [`<gensym>`](#identifier-gensym)
	
	- "variables" : list of [`<variable_spec>`](#variable-specification)[attrval] ::=

		- "name" : [`<variable_name>`](#variable-naming-convention)
		- "domain" : [`<variable_domain_type>`](#variable-value-domain)
		- "mutable" : `TRUE` | `FALSE`
		
	- "functions" : list of [`<function_spec>`](#function-specification) ... instances of the following:
		
		- [`<function_assign_spec>`](#function-assign-specification)[attrval] ::=
			- "name" : [`<function_name>`](#function-naming-conventions)
			- "type" : "assign" | "condition" | "decision"
			- "sources" : list of [ [`<function_source_reference>`](#function-assign-specification) | [`<variable_name>`](#variable-naming-convention) ]
			- "target" : [`<function_source_reference>`](#function-assign-specification) | [`<variable_name>`](#variable-naming-convention)
			- "body" : one of the following:
				- [`<function_assign_body_literal_spec>`](#function-assign-body-literal)[attrval] ::=
					- "type" : "literal"
					- "value" : [`<literal_value>`](#function-assign-body-literal)[attrval] ::=
						- "dtype" : "real" | "integer" | "boolean" | "string"
						- "value" : `<string>`
				- [`<function_assign_body_lambda_spec>`](#function_assign_body_lambda)[attrval] ::=
					- "type" : "lambda"
					- "name" : [`<function_name>`](#function-naming-conventions)
					- "reference" : [`<lambda_function_reference>`](#funciton-assign-body-lambda) ::= a `<string>` denoting function in `lambdas.py`
		
		- [`<function_container_spec>`](#function-container-specification)[attrval] ::=
			- "name" : [`<function_name>`](#function-naming-conventions)
			- "type" : "assign" | "condition" | "decision"
			- "sources" : list of [ [`<function_source_reference>`](#function-assign-specification) | [`<variable_name>`](#variable-naming-convention) ]
			- "target" : [`<function_source_reference>`](#function-assign-specification) | [`<variable_name>`](#variable-naming-convention)
			- "body" : list of [`<function_reference_spec>`](#function-reference-specification)
		
		- [`<function_loop_plate_spec>`](#function-loop-plate-specification)[attrval] ::=
			- "name" : [`<function_name>`](#function-naming-conventions)
			- "type" : "loop\_plate"
			- "input" : list of [`<variable_name>`](#variable-naming-convention)
			- "index\_variable" : [`<variable_name>`](#variable-naming-convention)
			- "index\_iteration\_range" : `<index_range>` ::=
				- "start" : `<integer>` | [`<variable_referene>`](#variable-reference) | [`<variable_name>`](#variable-naming-convention)
				- "end" : `<integer>` | [`<variable_referene>`](#variable-reference) | [`<variable_name>`](#variable-naming-convention)
			- "condition" : `<loop_condition>`
			- "body" : list of [`<function_reference_spec>`](#function-reference-specification)

- [`<function_reference_spec>`](#function-reference-specification)[attrval] ::=
	- "function" : [`<function_name>`](#function-naming-conventions)
	- "input" : list of [ [`<variable_reference>`](#variable-reference) | [`<variable_name>`](#variable-naming-convention) ]
	- "output" : list of [ [`<variable_reference>`](#variable-reference) | [`<variable_name>`](#variable-naming-convention) ]

- [`<function_source_reference>`](#function-assign-specification)
	- "name" : [ [`<variable_name>`](#variable-naming-convention) | [`<function_name>`](#function-naming-conventions) ]
	- "type" : "variable" | "function"

- [`<variable_referene>`](#variable-reference)[attrval] ::=
	- "variable" : [`<variable_name>`](#variable-naming-convention)
	- "index" : `<integer>`



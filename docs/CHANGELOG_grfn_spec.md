# Change Log


Inspired by [Keep a Changelog](https://keepachangelog.com)

This project does not (yet) adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)


## [0.1.m5] - 2019-05-01:

### Added
- Added "mutable" attribute to [`<variable_spec>`](#variable-specification).
- Added "variables" attribute to top-level [`<grfn_spec>`](#top-level-grfn-specification), which contains the list of all `<variable_spec>`s. This change also means that [`<function_spec>`](#function-specification)s no longer house [`<variable_spec>`](#variable-specification)s, but instead just the [`<variable_names>`](#variable-naming-convention) (which themselves are [`<identifier_string>`s](#identifier-string)).
- Added links to help topic navigation.

### Changed
- Clarified distinction between [`<source_code_reference>`](#grounding-and-source-code-reference)s (linking identifiers to where they are used in the analyzed source code) and [`<lambda_function_reference>`](#function-assign-body-lambda)s (which denote functions in the Program Analysis-generated lambdas file source code); previously these two concepts were ambiguous.

### Removed
- Removed [`<identifier_spec>`](#identifier-specification) "aliases" attribute. To be handled later as part of pointer/reference analysis.


## [0.1.m3] - 2019-03-01:

### Added
- Addition of identifiers: `<identifier_spec>`, `<identifier_string>`, and `<gensym>` (for identifiers in generated code)

### Changed
- Revision of Introduction
- Updates to naming conventions for variables and functions
- General cleanup of discussion throughout


## Releases
- [unreleased]: https://github.com/ml4ai/delphi/blob/grfn/docs/grfn_spec.md
- [0.1.m5]: https://github.com/ml4ai/automates/blob/master/documentation/deliverable_reports/m5_final_phase1_report/GrFN_specification_v0.1.m5.md
- [0.1.m3]: https://github.com/ml4ai/automates/blob/master/documentation/deliverable_reports/m3_report_prototype_system/GrFN_specification_v0.1.m3.md
- [0.1.m1]: https://github.com/ml4ai/automates/blob/master/documentation/deliverable_reports/m1_architecture_report/GrFN_specification_v0.1.md
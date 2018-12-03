This directory contains Jupyter notebooks made for demos,
as well as templates for HTML output, and scripts to ease version control and
exporting.

The file `custom.css` is a custom CSS file that you can copy to
`~/.jupyter/custom` in order to get some nice formatting.

The script `clean_and_commit` can be used to clean and commit the Markdown
notebook in one fell swoop. Usage:

```bash
./clean_and_commit example_notebook.md "Example commit message"
```

The script `clean` just cleans the notebook without committing it.
Usage:

```bash
./clean example_notebook.md
```

Some of the notebooks are actually in the form of Markdown files,
to be used in conjunction with [Notedown](https://github.com/ml4ai/notedown)

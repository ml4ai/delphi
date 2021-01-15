Contributing
============

Suggested Git Workflow
----------------------

The following git workflow is suggested for contributors. To implement a new
feature, do the following:

- Assuming you are on the `master` branch, check out a new feature branch. If
  the name of the feature is `new_feature`, you would do `git checkout -b new_feature`.
- Implement the feature.
- Run `git status` to see what files have changed. Optionally, run `git diff` to
  see the details of the changes to refresh your memory (and catch errant print
  statements!).
- Add the new/changed files by doing 
  ```
  git add <filename1> <filename2> ... <filenameN>
  ```
  Remember:
  - Only add human-generated files. Examples include:
    - Source code: 
       - `.py` files
       - Jupyter notebooks with their outputs cleared.
      - Documentation in the form of `.md` and `.rst` files.
  - In the case of testing code, this restriction can be relaxed a bit - for
    example, if you want to test the JSON output of a function, you can put
    the corresponding JSON file in the `tests/data` directory. 
  - In general, do not add non-text files, as `git` is designed to version
    control source code (i.e. text files). There are some exceptions, however:
    - Images that go into the `README.md` file in the root of the repository -
      they need to be checked in to render properly on Github.com.
    - The OpenFortranParser program, that is required for analyzing Fortran
      programs, requires four JAR files (in `delphi/translators/for2py/bin`) -
      these are checked into the repository for easier deployment, and because
      we do not expect them to change in the future.
  - **Do not run** `git add .` Instead, add the changed files individually -
    this will encourage you to make your commits less monolithic, which makes
    them easier to merge, and to revert if necessary. Also, this prevents the
    accidental addition of automatically-generated files to version control,
    which bloats the repository.
- Commit the changes:
  ```
  git commit -m "Description of features/changes that were implemented."
  ```
- Push the changes:
  ```
  git push
  ```
- Wait for an email from [Travis CI](https://travis-ci.org/ml4ai/delphi), to let
  you know whether the automated tests have passed or failed (if you haven't
  signed up for email alerts from Travis, you can simply view the live test
  execution log on the Travis website itself.) The tests take about 6 minutes as
  of the time of writing this (January 4, 2019), but could potentially take
  longer as Delphi undergoes further development.
- If the tests pass, 
  [open a pull request (PR)](https://help.github.com/articles/creating-a-pull-request/) 
  by going to the [repo website](https://github.com/ml4ai/delphi).
- One of the repo maintainers will then review your PR, and merge it into the
  master branch.
- Once the feature branch is merged, do the following:
  ```
  git checkout master
  git pull
  git branch -D new_feature
  ```
- **Tip 1:**: In general, smaller pull requests are better, and easier to merge.
- **Tip 2:**: Whenever you get an email from Github telling you that a branch
    has been merged into the master branch, but you are in the middle of
    implementing your feature branch, make sure to pull the changes from master
    into your branch and resolve any merge conflicts (another reason to not
    delay PRs!). Assuming you are on the `new_feature` branch, you would do:
    ```
    git pull origin master
    ```
    For those more proficient in the usage of `git`- feel free to fetch changes
    from master and rebase your feature branch on top of the master branch.
- Committing changes directly to the `master` branch is *highly* discouraged.
  The only exception to this rule is documentation updates (updating READMEs,
  etc.) And even then, major documentation updates should be done via a PR.

Running tests locally
---------------------

To run the complete test suite, invoke the following command from the Delphi
repo root directory.

```
make test
```

To run a particular test module, e.g. `test_program_analysis.py`, invoke the
following command instead:

```
pytest tests/test_program_analysis.py
```

To run a particular testing function within a test module, e.g. the
`test_petpt_grfn_generation` test function in `test_program_analysis.py`,
invoke the following instead:

```
pytest tests/test_program_analysis.py -k test_petpt_grfn_generation
```

If you would like the output of print statements (either in the library code or
the testing code) to be displayed instead of suppressed, pass the `-s` flag to
the `pytest` invocation:

```
pytest -s tests/test_program_analysis.py
```

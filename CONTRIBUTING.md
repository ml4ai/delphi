Contributing
============

Workflow
--------

The following contribution workflow is suggested. To implement a new feature, do
the following:

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
    - The OpenFortranParser JAR file that is required for running program
      analysis. 
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

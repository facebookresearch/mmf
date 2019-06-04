# Contributing to Pythia
We want to make contributing to this project as easy and transparent as
possible.

## Our Development Process
Minor changes and improvements will be released on an ongoing basis. Larger changes (e.g., changesets implementing a new paper) will be released on a more periodic basis.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style
* In your editor, install the [editorconfig](https://editorconfig.org/) extension which should ensure that you are following the same standards as us.
* Ideally, run black and isort before opening up your PR.

```
black ./(pythia|tests|tools)/**/*.py
isort -rc (pythia|tests|tools)
```
* Read the [editorconfig](.editorconfig) file to understand the exact coding style preferences.

## Commit Guidelines

We follow the same guidelines as AngularJS. Each commit message consists of a **header**, a **body** and a **footer**.  The header has a special format that includes a **type**, and a **subject**:

```
[<type>] <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

Any line of the commit message cannot be longer 100 characters! This allows the message to be easier
to read on github as well as in various git tools.

### Type
Must be one of the following:

* **feat**: A new feature
* **fix**: A bug fix
* **cleanup**: Changes that do not affect the meaning of the code (white-space, formatting, missing
  semi-colons, dead code removal etc.)
* **refactor**: A code change that neither fixes a bug or adds a feature
* **perf**: A code change that improves performance
* **test**: Adding missing tests or fixing them
* **chore**: Changes to the build process or auxiliary tools and libraries such as documentation
generation
* **docs**: Documentation only changes

## License
By contributing to Pythia, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.

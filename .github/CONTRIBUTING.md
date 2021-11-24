# Contributing to MMF
We want to make contributing to this project as easy and transparent as
possible.

## Our Development Process
Minor changes and improvements will be released on an ongoing basis. Larger changes (e.g., changesets implementing a new paper) will be released on a more periodic basis.

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

## Pull Requests

### Guidelines

We actively welcome your pull requests.

However, if you're adding anything significant (e.g. > 50 lines), please
open up an issue before sending a PR to involve and discuss with maintainers if the change is required and will be accepted.

We take the following factors into consideration before accepting features and PRs:

1. If the feature can be achieved without modifying MMF. MMF is designed so that you can implement extensions from the outside, e.g. [Hateful Memes ConcatVL example](https://github.com/apsdehal/hm_example_mmf).
  * If something is not extensible, please open up an issue so we can improve MMF together.
1. Whether the feature is potentially useful to a large audience (e.g. an impactful paper, a popular dataset, a significant speedup or a widely useful utility).
1. Whether the proposed solution has a good design/interface. This can be discussed in the issues prior to the PRs, or in the form of a draft PR and we will help you make it better.
1. Whether the proposed solution adds extra overhead to users who don't need such feature.
1. Whether the proposed solution breaks existing APIs.

### Process

1. Read the [PR guidelines](#guidelines) if you haven't.
1. Fork the repo and create your branch from `main`.
1. If your PR contains multiple orthogonal changes, split it to several PRs. Keep one PR focused on a single change while keeping it small.
1. If you've added code that should be tested, add tests.
1. Follow the [coding style guidelines](#coding-style) mentioned below.
1. If you've changed APIs:
    * Update the documentation. We use the [Google style docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) in python.
    * Ensure backwards compatibility.
1. Ensure the test suite passes.
1. If the PR involves adding (i) a new dataset: update the dataset zoo to enable automatic downloads (ii) a new model: you don't need to update the model zoo, but make sure to describe your results and experiments in PR description. Also, update the list of the [models](https://mmf.sh/docs/notes/model_zoo/) and the [datasets](https://mmf.sh/docs/notes/dataset_zoo/) in the documentation accordingly.
1. Follow [commit guidelines](#commit-guidelines) to ensure your commit message follows MMF style.
1. If you haven't already, complete the Contributor License Agreement ("CLA").



## Coding Style
* In your editor, install the [editorconfig](https://editorconfig.org/) extension which should ensure that you are following the same standards as us.
* MMF uses pre-commit hooks to ensure style consistency and prevent common mistakes. Enable it by:

```sh
pip install pre-commit && pre-commit install
```

After this pre-commit hooks will be run before every commit.

* Read the [editorconfig](https://github.com/facebookresearch/mmf/blob/main/.editorconfig) file to understand the exact coding style preferences.

* Ideally, ufmt should be run via pre-commit hooks.
But if for some reason you want to run ufmt separately follow this:

```
pip install ufmt==1.3.0
ufmt format (mmf|tests|tools)
```
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
By contributing to MMF, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.

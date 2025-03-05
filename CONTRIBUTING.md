# Contributing to Simvue Integrations

Contributions to enhance existing Connectors between Simvue and external pieces of software are very welcome. Whether you are a user of software which you think would benefit from Simvue integration and wish to request new features, or a developer looking to help us add to the codebase, all contributions are useful! To start, please create a new Issue outlining the new fix, change or feature which you would like to see to allow for discussion within the community.

## :memo: Issue Creation

When opening an issue please ensure you outline in as much detail as possible the issues or proposed changes you wish to develop. Also ensure that this feature/fix has not already been raised by firstly searching through all current issues. If reporting a bug, provide as much detail outlining a minimal example and describing any useful specifications and outlining the scenario which led to the problem.

Please add one from each of the following sets of tags to help us find your issue more easily:

### Type:
- **Bug**: Something in the integrations repository isn't working properly
- **Enhancement**: Improvement to existing code within the repository
- **Feature**: A new addition to the Connector
- **Question**: Ask a question about a piece of code, or how to use an existing Connector

## 🧰 Development

### :closed_book: Python Poetry

For development it is strongly recommended that [Poetry](https://python-poetry.org) be used to manage dependencies and create the virtual environment used for development, the included `pyproject.toml` file makes use of the framework for ensuring dependency compatibility and building of the module during deployment. The included `poetry.lock` file defines the virtual environment to ensure the developers are running `simvue` in an identical manner. Install poetry and setup the virtual environment by running from the root of this repository:

```sh
pip install --user poetry
poetry install
```

### 🪝 Using Git hooks

Included within this repository are a set of Git hooks which ensure consistency in formating and prevent the accidental inclusion of non-code based files. The hooks are installed using the [`pre-commit`](https://pre-commit.com/) tool. Use of these hooks is recommended as they are also run as a verification step of the continuous integration pipeline. To setup pre-commit, change directory to the root of this repository and run:

```sh
pip install --user pre-commit
pre-commit install
```

### 🧪 Testing

To ensure robustness and reliability this repository includes a set of tests which are executed automatically as part of continuous integration. Before opening a merge request we ask that you check your changes locally by running the unit test suite. New tests should be written for any further functionality added, with `Mock` functions being created to generate example output files if the connector is monitoring non-Python software. To run these tests, you must first install all extra dependencies:

```sh
poetry install --all-extras
pytest tests/unit/
```

### ℹ️ Typing

All code within this repository makes use of Python's typing capability, this has proven invaluable for spotting any incorrect usage of functionality as linters are able to quickly flag up any incompatibilities. Typing also allows us define validator rules using the [Pydantic](https://docs.pydantic.dev/latest/) framework.  We ask that you type all functions and variables where possible.

### ✔️ Linting and Formatting

Simvue utilises the [Ruff](https://github.com/astral-sh/ruff) linter and formatter to ensure consistency, and this tool is included as part of the pre-commit hooks. Checking of styling/formatting is part of the CI pipeline.

## :book: Documentation

To ensure functions, methods and classes are documented appropriately, Simvue follows the Numpy docstring convention. We also ask that if adding new features or Connectors you ensure these are mentioned within the official [documentation](https://github.com/simvue-io/docs).

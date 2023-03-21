# Poetry Python Development Environment Guide

## Poetry Installation

### Step 1. Download the install executable and run it with python3

```bash
> curl -sSL https://install.python-poetry.org | python3 - --version 1.1.9
```

The `python3` can be a default python environment such as from anaconda. We use poetry 1.1.9 because the poetry version higher than 1.2.0 will auto replace `.` in the package to `-`, which would not work with intuit naming conversion.

### Step 2. Add poetry executable to your path

```bash
> export PATH="/Users/qianyu/.local/bin:$PATH"
```

### Step 3 Check installation is correct with version check

```bash
> poetry --version
```

## To Start a New Project

### Step 1. Create a new project template

```bash
> poetry new poetry-demo
> tree poetry-demo
poetry-demo
├── pyproject.toml
├── README.rst
├── poetry_demo
│   └── __init__.py
└── tests
    ├── __init__.py
    └── test_poetry_demo.py
```

You are free to remove any of these files and directories, except for the `pyproject.toml` . By adding the --src flag, you created a folder named src/, which contains the package.

```bash
> poetry new --src poetry-demo
> tree poetry-demo
poetry-demo
├── pyproject.toml
├── README.rst
├── src
│   └── poetry_demo
│       └── __init__.py
└── tests
    ├── __init__.py
    └── test_poetry_demo.py
```

Poetry automatically rename the folder and file name from `-` to `_`

#### The `pyproject.toml` file defines the requirement of the project

```toml
[tool.poetry]
name = "poetry-demo"
version = "0.1.0"
description = ""
authors = ["Darth Vader <darth_vader@edark_side.io>"]
readme = "README.md"
packages = [{include = "poetry_demo", from = "src"}]

[tool.poetry.dependencies]
python = "^3.8"

[tool.poetry.dev-dependencies]
pytest = "^7.1.3"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```

## Add, remove, and install dependencies

### Add Dependecies 

```bash
# Add a dependency to the package environment
> poetry add <library_name>

# Add a dependency to the development package
> poetry add --dev <library_name>
```

Poetry will automatcally figure out the suitable version constraint for you and install. It will generate a `poetry.lock` file to store the latest state of the poetry environment.

And this is the recommended way to add a dependency


### remove a dependency

Similarly, the following command is to remove a depedency, poetry will uninstall the library, it will update the `poetry.lock` to reflect the latest state of the poetry enviornment

And this is the recommended way to remove a dependency

```bash
# Remove a dependency to the package environment
> poetry remove <library_name>

# Remove a dependency to the development package
> poetry remove --dev <library_name>
```

### To Install manually 

If users already know the versions of the libraries to install, they can just updated the `pyproject.toml` and install

```bash
poetry install
```

If there is already a  `poetry.lock` file present, this will install all the packages from that lock file. If not, Poetry will resolve the dependencies, install the packages, and generate a new lock file.

To pin manually added dependencies from your pyproject.toml file to poetry.lock, you must first run
the poetry lock command:

```bash
> poetry lock
```

By running poetry lock, Poetry processes all dependencies in your pyproject.toml file and locks them into the poetry.lock file. And Poetry doesn’t stop there. When you run poetry lock, Poetry also recursively traverses and locks all dependencies of your direct dependencies.

The poetry lock command also updates your existing dependencies if newer versions that fit your version constraints are available . If you don’t want to update any dependencies that are already in the poetry.lock file, do

```bash
> poetry lock --no-update
```

## Poetry Virtual Environment

We should always using poetry virtual environment for both development, testing and documentation

### Step 1. Setup python

```bash
> poetry env use python3
Creating virtualenv rp-poetry-A3O1Dhkd-py3.8 in ~/Library/Caches/pypoetry/virtualenvs
...
```

With this command, you’re using the same Python version that you used to install Poetry which can be your default python from anaconda for example

Python library under both the main and dev dependency will be available in the virtual environment

### Step 2. Check poetry virutal environment

```bash
> poetry env list
<my_packate>-A3O1Dhkd-py3.8 (Activated)
```

### Step 3. Work in virutual environment

You can simply use `poetry run` to run python jobs, they will run automatically under the virtual environment. 

```bash
> poetry run pytest
> poetry run coverage run -m pytest
> poetry run jupyter notebook
```

If you would like run python jobs without typing `poetry run`, you can kick off a virtual envionment python shell with

```bash
# To enter the virtual environment
> poetry shell

# To exit
> exit
```


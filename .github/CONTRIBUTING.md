# Contributing to *Submitthem*
We want to make contributing to this project as easy and transparent as possible.

## Our Development Process
There is no proof [*Submitit*](https://github.com/facebookincubator/submitit) is still actively used by FAIR researcher and engineers.
We can’t garanty that *Submitthem* will be actively used by any researcher and engineers.
All bugs tracking and feature plannings are public.
*Submitthem* will NOT be updated to keep up with Slurm/PBS versions and to fix bug,
We don't have any major feature planned ahead.

## Pull Requests
We actively welcome pull requests and will review them as quickly as possible, subject to our availability.

### Setting up your development environment

We use [uv](https://docs.astral.sh/uv/) for modern Python project management. It's faster, more reliable, and works across all platforms.

**Prerequisites:**

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) using one of these methods:

- **Recommended (Linux/macOS):** `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Via pip:** `pip install uv` (requires Python 3.8+)
- **Via conda:** `conda install -c conda-forge uv`
- **Windows:** Download from [astral.sh/uv](https://docs.astral.sh/uv/)

**Setup:**

After cloning the repository, you can start using `uv run` immediately - it will automatically sync dependencies on first use:

```bash
cd submitthem
uv run pytest submitthem  # Auto-syncs dependencies, then runs tests
```

**Using your environment:**

```bash
# Option A: Use uv run (recommended - auto-syncs dependencies)
uv run pytest submitthem
uv run ruff format submitthem
uv run mypy submitthem

# Option B: Pre-sync and activate venv (for faster repeated commands)
uv sync --group dev
source .venv/bin/activate          # Linux/macOS
# or
.venv\Scripts\activate.bat         # Windows

# Then use commands normally (faster, no auto-sync overhead)
pytest submitthem
ruff format submitthem
mypy submitthem
```

**Dependency updates are automatic:**

When you `git pull` and `pyproject.toml` or `uv.lock` changes, simply use `uv run` - it will automatically sync your environment to the latest dependencies before executing the command. No manual sync needed!

**Optional: Pre-sync specific dependency groups:**

If you prefer to pre-sync rather than auto-sync with `uv run`:

```bash
uv sync --group dev        # Development (test, lint, format, release)
uv sync --group test       # Testing only
uv sync --group lint       # Linting and type checking only
uv sync --group format     # Formatting and pre-commit hooks only
uv sync --group release    # Release tools (flit) only
```


### Submitting your changes

1. Create your branch from `main`.
2. If you've added code please add tests.
3. If you've changed APIs, please update the documentation.
4. Ensure the test suite passes:

```bash
python -m pytest -n auto submitthem
```

5. Make sure your code lints and is formatted:

```bash
python -m ruff format submitthem docs integration
python -m ruff check --fix submitthem
python -m mypy submitthem
python -m pylint submitthem
```

Or use VS Code tasks: `Ctrl+Shift+B` → "Python: Format + Lint (pre-commit)"

6. Optional: Set up pre-commit hooks to run automatically on each commit:

```bash
pre-commit install
```

## Issues


We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.


## Coding Style

We use [ruff](https://docs.astral.sh/ruff/) for code formatting and linting with a generous 110 character line length.

### Running code quality checks

Use `make` commands or `uv run` directly:

```bash
# Format code
make format
uv run ruff format submitthem docs integration

# Check formatting without changes
make check_format
uv run ruff format --check submitthem docs integration

# Run all linting checks
make lint
uv run mypy submitthem
uv run pylint submitthem

# Format and lint together
make pre_commit
```

### Pre-commit hooks (optional)

To automatically run checks before each commit:

```bash
pre-commit install
```

This uses the configuration in `.pre-commit-config.yaml` to run ruff and other checks before each commit.

### Ruff configuration

Ruff configuration is managed in `pyproject.toml` under `[tool.ruff]`. The main rules we enforce are:

- `E`, `W`: PEP 8 errors and warnings
- `F`: Pyflakes (undefined names, unused imports)
- `I`: Import sorting (replaces isort)
- `B`: flake8-bugbear (common bugs and design problems)
- `C4`: flake8-comprehensions
- `UP`: pyupgrade (modernize Python syntax)


## Using VS Code

VS Code is configured with Python development tasks that work on all platforms (Windows, macOS, Linux):

1. Press `Ctrl+Shift+B` to see available tasks:
   - **Format code** - Format with ruff
   - **Check format** - Verify formatting
   - **Lint (ruff)** - Run ruff checks
   - **Type check (mypy)** - Run mypy
   - **Lint (pylint)** - Run pylint
   - **Run tests** - Execute test suite
   - **Run tests with coverage** - Tests + coverage report
   - **Format + Lint (pre-commit)** - Run pre-commit checks
   - **Sync dependencies** - Run `uv sync --all-groups`

All tasks use `uv run` to ensure dependencies are synced before execution.

## Using the Command Line

On any platform (Windows, macOS, Linux), use the same commands:

```bash
# Use Makefile shortcuts
make format           # Format code
make check_format     # Check without changes
make lint             # Run all linting
make test             # Run tests
make test_coverage    # Tests with coverage
make pre_commit       # Format + lint
make sync             # Sync dependencies

# Or use uv run directly
uv run ruff format submitthem docs integration
uv run ruff check submitthem
uv run mypy submitthem
uv run pylint submitthem
uv run pytest -n auto submitthem
```

## License
By contributing to *Submitthem*, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.

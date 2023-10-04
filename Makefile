# Run the linter and tests suite of this repo.

# `make use_venv=0` will use the default `python` otherwise uses the one in `.venv/`
use_venv=1
ifeq ($(use_venv),0)
BIN?=
else
BIN=venv/bin/
endif

CODE=submitit
CODE_AND_DOCS=$(CODE) docs/ integration/

all: integration

which:
	which $(BIN)python
	$(BIN)python --version

test:
	$(BIN)pytest $(CODE)

test_coverage:
	$(BIN)pytest \
		-v \
		--cov=submitit --cov-report=html --cov-report=term \
		--durations=10 \
		--junitxml=test_results/pytest/results.xml \
		$(CODE)

format:
	$(BIN)python -m pre_commit
	$(BIN)isort $(CODE_AND_DOCS)
	$(BIN)black $(CODE_AND_DOCS)

check_format:
	# also formats docs
	$(BIN)isort --check --diff $(CODE_AND_DOCS)
	$(BIN)black --check --diff $(CODE_AND_DOCS)

mypy:
	$(BIN)mypy --version
	$(BIN)mypy --junit-xml=test_results/pytest/results.xml $(CODE)

pylint:
	$(BIN)pylint --version
	$(BIN)pylint $(CODE)


lint: mypy pylint

venv: venv/pyproject.toml

venv/pyproject.toml: pyproject.toml
	python3 -m venv venv
	venv/bin/pip install --progress-bar off --upgrade pip
	venv/bin/pip install --progress-bar off -U -e .[dev]
	cp $^ $@

installable: installable_local installable_wheel

installable_local: venv
	(. ./venv/bin/activate ; cd /tmp ; python -c "import submitit")

BUILD=dev0$(CIRCLE_BUILD_NUM)
USER_VENV=/tmp/submitit_user_venv/
CURRENT_VERSION=`grep -e '__version__' ./submitit/__init__.py | sed 's/__version__ = //' | sed 's/"//g'`
TEST_PYPI=--index-url 'https://test.pypi.org/simple/' --no-cache-dir --no-deps --progress-bar off

installable_wheel:
	[ ! -d dist ] || rm -r dist
	# Append .$(BUILD) to the current version
	sed -i -e 's/__version__ = "[0-9].[0-9].[0-9]/&.$(BUILD)/' ./submitit/__init__.py
	grep -e '__version__' ./submitit/__init__.py | sed 's/__version__ = //' | sed 's/"//g'
	$(BIN)python -m flit build --setup-py
	git checkout HEAD -- ./submitit/__init__.py

	[ ! -d $(USER_VENV) ] || rm -r $(USER_VENV)
	python3 -m venv $(USER_VENV)
	$(USER_VENV)/bin/pip install --progress-bar off dist/submitit-*any.whl
	# Check that importing works
	$(USER_VENV)/bin/python -c "import submitit"

clean:
	rm -r venv

clean_cache:
	# Invalidates `make venv` and therefore trigger a new `pip install --upgrade`.
	rm venv/pyproject.toml

pre_commit: format lint

register_pre_commit: venv
	(grep -e "^make pre_commit$$" .git/hooks/pre-commit) || (echo "make pre_commit" >> .git/hooks/pre-commit)
	chmod +x .git/hooks/pre-commit

integration: clean_cache venv check_format lint installable test_coverage
	# Runs the same tests than on CI.
	# clean_cache will make sure we download the last versions of linters.
	# Use `make -k integration` to run all checks even if previous fails.

release: integration
	echo "Releasing submitit $(CURRENT_VERSION)"
	[ ! -d dist ] || rm -r dist
	# Make sure the repo is in a clean state
	git diff --exit-code
	$(BIN)python submitit/test_documentation.py
	# --setup-py generates a setup.py file to allow user with old
	# versions of pip to install it without flit.
	git tag $(CURRENT_VERSION)
	# To have a reproducible build we use the timestamp of the last commit:
	# https://flit.pypa.io/en/latest/reproducible.html
	SOURCE_DATE_EPOCH=`git log -n1 --format=%cd --date=unix` $(BIN)python -m flit publish --setup-py
	git checkout HEAD -- README.md
	git push origin $(CURRENT_VERSION)

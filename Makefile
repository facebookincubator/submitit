# Run the linter and tests suite of this repo.

# `make use_venv=0` will use the default `python` otherwise uses the one in `.venv/`
use_venv=1
ifeq ($(use_venv),0)
BIN?=
else
BIN=venv/bin/
endif

CODE=submitit
CODE_AND_SETUP=$(CODE) setup.py docs/ integration/

which:
	which $(BIN)python
	$(BIN)python --version

test:
	$(BIN)pytest $(CODE)

test_coverage:
	$(BIN)pytest \
		--cov=submitit --cov-report=html --cov-report=term \
		--durations=10 \
		--junitxml=test_results/pytest/results.xml \
		$(CODE)

format:
	$(BIN)python -m pre_commit
	$(BIN)isort $(CODE_AND_SETUP)
	$(BIN)black $(CODE_AND_SETUP)

check_format:
	# also formats setup.py
	$(BIN)isort --check --diff $(CODE_AND_SETUP)
	$(BIN)black --check --diff $(CODE_AND_SETUP)

mypy:
	$(BIN)mypy --version
	$(BIN)mypy --junit-xml=test_results/pytest/results.xml $(CODE)

pylint:
	$(BIN)pylint --version
	$(BIN)pylint $(CODE)


lint: mypy pylint

venv: venv/requirements.txt

venv/requirements.txt: requirements/main.txt requirements/dev.txt
	python3 -m venv venv
	venv/bin/pip install --progress-bar off --upgrade pip
	venv/bin/pip install --progress-bar off -U -e .[dev]
	cat $^ > venv/requirements.txt

installable: installable_local installable_wheel

installable_local: venv
	(. ./venv/bin/activate ; cd /tmp ; python -c "import submitit")

BUILD=dev$(CIRCLE_BUILD_NUM)
USER_VENV=test_results/user_venv/
CURRENT_VERSION=`grep -e '__version__' ./submitit/__init__.py | sed 's/__version__ = //' | sed 's/"//g'`
TEST_PYPI=--index-url 'https://test.pypi.org/simple/' --no-cache-dir --no-deps --progress-bar off

installable_wheel:
	[ ! -d dist ] || rm -r dist
	# Append .$(BUILD) to the current version
	sed -i -e 's/__version__ = "[0-9].[0-9].[0-9]/&.$(BUILD)/' ./submitit/__init__.py
	grep -e '__version__' ./submitit/__init__.py | sed 's/__version__ = //' | sed 's/"//g'
	$(BIN)python setup.py sdist bdist_wheel
	git checkout HEAD -- ./submitit/__init__.py

	[ ! -d $(USER_VENV) ] || rm -r $(USER_VENV)
	python3 -m venv $(USER_VENV)
	$(USER_VENV)/bin/pip install --progress-bar off dist/submitit-*any.whl
	# Check that importing works
	$(USER_VENV)/bin/python -c "import submitit"

clean:
	rm -r venv

pre_commit: format lint

register_pre_commit: venv
	(grep -e "^make pre_commit$$" .git/hooks/pre-commit) || (echo "make pre_commit" >> .git/hooks/pre-commit)
	chmod +x .git/hooks/pre-commit

integration: venv check_format lint installable test_coverage
	# Run the same tests than on CI
	# use `make -k integration` to run all checks even if previous fails.

release: integration
	grep -e '__version__' ./submitit/__init__.py | sed 's/__version__ = //' | sed 's/"//g'
	[ ! -d dist ] || rm -r dist
	$(BIN)python setup.py sdist bdist_wheel
	$(BIN)pip install --progress-bar off twine
	# Credentials are read from ~/.pypirc
	$(BIN)python -m twine upload dist/*

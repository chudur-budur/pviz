#!/bin/bash

# Run it from one folder up, ./sh/tester.sh

pipenv run python3 -m tests.viz.utils.test_dm
pipenv run python3 -m tests.viz.utils.test_io

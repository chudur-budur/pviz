#!/bin/bash

# Run it from one folder up, ./bin/tester.sh

pipenv run python3 -m viz.utils.tests.test_dm
pipenv run python3 -m viz.utils.tests.test_io

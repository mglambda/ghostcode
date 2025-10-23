#!/usr/bin/bash

mypy -m ghostcode.main --ignore-missing-imports --strict --no-warn-return-any

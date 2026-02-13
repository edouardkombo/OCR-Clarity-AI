VENV?=.venv
PY?=$(VENV)/bin/python

venv:
	python -m venv $(VENV)

install:
	$(PY) -m pip install -r requirements.txt

run:
	$(PY) clarity_ocr.py samples/in/your.pdf --out samples/out/your.txt

compose:
	docker compose up --build

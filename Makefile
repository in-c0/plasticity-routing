.PHONY: test exp000 leakage validate calib clean

PY := .venv/bin/python
PYTEST := .venv/bin/pytest

test:
	PYTHONPATH=src $(PYTEST) -q

leakage:
	PYTHONPATH=src $(PY) scripts/audit_leakage.py --seeds 11 12 13 --out results/leakage_audit.json

exp000:
	PYTHONPATH=src $(PY) scripts/run_exp000.py --dev-seeds 11 12 13 --out results

validate:
	PYTHONPATH=src $(PY) scripts/validate_runs.py results/run_*.json --out results/validation.json

calib: test leakage exp000 validate

clean:
	rm -f results/*.json results/*.csv

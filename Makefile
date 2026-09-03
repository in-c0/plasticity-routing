.PHONY: test exp000 leakage l5 l5b validate calib freeze clean

PY := .venv/bin/python
PYTEST := .venv/bin/pytest

test:
	PYTHONPATH=src $(PYTEST) -q

leakage:
	PYTHONPATH=src $(PY) scripts/audit_leakage.py --seeds 11 12 13 --out results/leakage_audit.json

l5:
	PYTHONPATH=src $(PY) -u scripts/audit_l5.py --out results/l5_time_shuffle.json

l5b:
	PYTHONPATH=src $(PY) -u scripts/audit_l5b.py --out results/l5b_cross_world.json

exp000:
	PYTHONPATH=src $(PY) scripts/run_exp000.py --dev-seeds 11 12 13 --out results

validate:
	PYTHONPATH=src $(PY) scripts/validate_runs.py results/run_*.json --out results/validation.json

calib: test l5b leakage exp000 validate

freeze:
	PYTHONPATH=src $(PY) scripts/freeze_protocol.py

clean:
	rm -f results/*.json results/*.csv

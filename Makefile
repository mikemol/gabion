.PHONY: bootstrap check check-ci test test-logs clean-artifacts docflow dataflow lsp-smoke audit-snapshot audit-latest

bootstrap:
	scripts/bootstrap.sh

check:
	scripts/checks.sh

check-ci:
	scripts/checks.sh --no-docflow

test:
	mise exec -- pytest

test-logs:
	scripts/run_tests.sh

clean-artifacts:
	scripts/clean_artifacts.sh

docflow:
	mise exec -- python -m gabion docflow-audit

dataflow:
	mise exec -- python -m gabion check

lsp-smoke:
	mise exec -- python scripts/lsp_smoke_test.py --root .

audit-snapshot:
	scripts/audit_snapshot.sh

audit-latest:
	scripts/latest_snapshot.sh

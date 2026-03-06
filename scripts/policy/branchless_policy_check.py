#!/usr/bin/env python3
from __future__ import annotations

from gabion.tooling.policy_rules import branchless_rule as _impl
from gabion.tooling.policy_rules.branchless_rule import *

# Compatibility export consumed by governance/normative tooling tests.
_load_baseline = _impl._load_baseline


if __name__ == "__main__":
    raise SystemExit(main())

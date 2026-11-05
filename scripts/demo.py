#!/usr/bin/env python3
"""Run the demo from a checkout without installing: ``python scripts/demo.py [options]``.

Same options as ``helios16p-demo`` (see ``helios16p/cli.py``).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helios16p.cli import demo  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(demo())

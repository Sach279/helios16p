#!/usr/bin/env python3
"""Thin wrapper: ``python tools/record.py`` is ``helios16p-record`` (see helios16p/apps/record.py)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helios16p.apps.record import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

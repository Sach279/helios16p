#!/usr/bin/env python3
"""Thin wrapper: ``python tools/export_ply.py`` is ``helios16p-export`` (see helios16p/apps/export_ply.py)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helios16p.apps.export_ply import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

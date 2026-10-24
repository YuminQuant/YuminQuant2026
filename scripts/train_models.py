"""Train one independent model per configured window; use --dry-run to inspect."""
from __future__ import annotations

import sys

from yq import main


if __name__ == '__main__':
    raise SystemExit(main(['--split-training-windows', *sys.argv[1:]]))

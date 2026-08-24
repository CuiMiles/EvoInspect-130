from __future__ import annotations

import argparse
from pathlib import Path

from evoinspect.provenance import write_json
from evoinspect.upstream_patchcore import collect_batch, register_batch


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--batch-root", required=True, type=Path)
    root.add_argument("--output", required=True, type=Path)
    root.add_argument("--registry", type=Path)
    return root


def main() -> int:
    args = parser().parse_args()
    summary = collect_batch(args.batch_root)
    write_json(args.output, summary)
    if args.registry is not None:
        register_batch(summary, args.registry, Path.cwd())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

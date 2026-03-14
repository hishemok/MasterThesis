from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BRAIDING_SRC = SRC / "braiding"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BRAIDING_SRC) not in sys.path:
    sys.path.insert(0, str(BRAIDING_SRC))

from braiding_core import (  # noqa: E402
    format_braiding_report,
    load_saved_evolution,
    verify_braiding_experiment,
)
from get_setup import params_for_n_site_Hamiltonian  # noqa: E402


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the clean logical-subspace braid verification on a saved evolution archive.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "src" / "braiding" / "braiding_time_evolution.npz",
        help="Path to the saved braiding_time_evolution.npz archive.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configuration.json",
        help="Path to the optimizer configuration JSON used to rebuild the bare system.",
    )
    parser.add_argument(
        "--u",
        type=float,
        default=0.1,
        help="Interaction value used to pick the same configuration as the notebook.",
    )
    parser.add_argument("--n-sites", type=int, default=3)
    parser.add_argument("--dupes", type=int, default=3)
    parser.add_argument("--n-pairs", type=int, default=4)
    parser.add_argument("--block-dim", type=int, default=8)
    parser.add_argument("--spectator-label", default="A")
    parser.add_argument("--spectator-parity", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--total-parity", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--anchor", default="A1")
    parser.add_argument("--braid-from", default="B1")
    parser.add_argument("--braid-to", default="C1")
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    params, _ = params_for_n_site_Hamiltonian(
        args.n_sites,
        configs=None,
        specified_vals={"U": [args.u]},
        path=str(args.config),
    )
    evolution = load_saved_evolution(args.archive)
    report = verify_braiding_experiment(
        params,
        evolution,
        n_sites=args.n_sites,
        dupes=args.dupes,
        n_pairs=args.n_pairs,
        block_dim=args.block_dim,
        spectator_label=args.spectator_label,
        spectator_parity=args.spectator_parity,
        total_parity=args.total_parity,
        anchor=args.anchor,
        braid_from=args.braid_from,
        braid_to=args.braid_to,
    )
    print(format_braiding_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

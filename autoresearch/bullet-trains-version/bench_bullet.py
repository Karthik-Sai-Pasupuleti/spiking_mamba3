"""Entry point of the Bullet Trains study: the soft-reset study's harness plus the `bullet_train` method.

It registers `bullet_train` (bullet_neuron.py) into the registry of
`../soft_reset_LIF_experiment/neuron.py`, then runs that study's `bench_neuron.main()` with the
same arguments, gate, inputs (`bench_inputs.pt`, read only) and RESULT line format. One extra
argument, `--theta`, sets the threshold of the real and slow-leak inputs (sparsity sweep).

    python -B bench_bullet.py --method bullet_train --config "K=128,bwd=serial" --site in --reset hard > bench.log 2>&1
    grep "^RESULT" bench.log | cut -f2- >> bench_results.tsv
    python -B bench_bullet.py --method serial_triton --config "U=32,BLOCK=64,nw=2,tape=u8" --site in --reset hard

Nothing is written into the other study's folder (no bytecode, no input rebuild).
"""

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402

import bullet_neuron  # noqa: E402, F401  (registers bullet_train; puts the other study on sys.path)
import bench_neuron  # noqa: E402


def main() -> None:
    """Take `--theta` off the command line, set it, and hand the rest to the old harness."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--theta", type=float, default=bench_neuron.THETA)
    args, rest = parser.parse_known_args()
    if "--make-inputs" in rest:
        raise SystemExit("bench_inputs.pt belongs to the soft-reset study; build it there")
    bench_neuron.THETA = args.theta
    sys.argv = [sys.argv[0], *rest]
    if args.theta != 1.0:
        sys.argv += ["--note", f"theta={args.theta:g}"]
    bench_neuron.main()


if __name__ == "__main__":
    main()

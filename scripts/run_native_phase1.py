"""Isolated --use-native-gating CLI; live grid-search runner stays unchanged."""

import argparse
from functools import partial
import hashlib
from pathlib import Path
import sys


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--use-native-gating",
        action="store_true",
        help="use the C++ scheduling decision; Python is the default",
    )
    parser.add_argument(
        "--native-module-dir", default=str(Path(__file__).resolve().parents[1] / "csrc/build/python")
    )
    parser.add_argument("--tracker", choices=("lk", "sort"), default="lk")
    options, remaining = parser.parse_known_args(arguments)
    if options.use_native_gating and options.tracker != "lk":
        parser.error("--use-native-gating currently supports the LK pipeline only")
    if "--help" in remaining or "-h" in remaining:
        print("Additional native-runner options:\n" + parser.format_help())
    from scripts import run_phase1 as original

    original_pipeline, original_benchmark = original.Pipeline, original.Benchmark
    original_argv = sys.argv
    try:
        if options.use_native_gating:
            from src.native_gating import load_native
            from src.native_pipeline import NativeGatingPipeline

            native = load_native(options.native_module_dir)
            metadata = {
                "use_native_gating": True,
                "gating_backend": "native",
                "native_module_version": native.__version__,
                "native_module_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
            }
            original.Pipeline = partial(NativeGatingPipeline, use_native_gating=True)

            def benchmark(output, run_metadata):
                return original_benchmark(output, run_metadata | metadata)

            original.Benchmark = benchmark
        # Rebinding affects only this new process, never the live search or files.
        sys.argv = [original.__file__, *remaining, "--tracker", options.tracker]
        original.main()
    finally:
        sys.argv = original_argv
        original.Pipeline, original.Benchmark = original_pipeline, original_benchmark


if __name__ == "__main__":
    main()

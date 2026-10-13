"""Check/apply the prepared opt-in patch after all recorded grid searches finish."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="apply after verifying no active search; default checks only")
    args = parser.parse_args(argv)
    directory = ROOT/"csrc/patches"
    metadata = json.loads((directory/"native-gating-integration.json").read_text())
    for path in (ROOT/"results").glob("**/manifest.json"):
        manifest = json.loads(path.read_text())
        if manifest.get("status") == "running" and "max_intervals" in manifest.get("config", {}):
            parser.error(f"Active grid search: {path}. Leave its source files unchanged until it finishes.")
    patch = directory/"native-gating-integration.patch"
    if digest(patch) != metadata["patch_sha256"]:
        parser.error("Integration patch checksum changed")
    if all(digest(ROOT/path)==entry["after_sha256"] for path,entry in metadata["targets"].items()):
        print("Native opt-in integration is already applied.")
        return
    for path,entry in metadata["targets"].items():
        if digest(ROOT/path) != entry["before_sha256"]:
            parser.error(f"Source changed since patch preparation: {path}; review/regenerate the patch")
    subprocess.run(["git","apply","--check",str(patch)],cwd=ROOT,check=True)
    if args.apply:
        subprocess.run(["git","apply",str(patch)],cwd=ROOT,check=True)
        for path,entry in metadata["targets"].items():
            if digest(ROOT/path)!=entry["after_sha256"]:
                raise RuntimeError(f"Applied content mismatch: {path}")
        print("Applied opt-in flag; Python remains the default backend.")
    else:
        print("Patch checks passed. Use --apply after active searches have finished.")


if __name__ == "__main__":
    main()

"""Write (or verify) a SHA256 manifest for an untracked analysis-specimens tree."""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

GAME_DIR = r"C:\Program Files (x86)\Steam\steamapps\common\FINAL FANTASY XIV Online"
SPECIMENS_DIR = r"P:\projects\ffxiv-patchday\analysis-specimens"
MANIFESTS_DIR = r"P:\projects\ffxiv-patchday\hash-manifests"


def get_game_version() -> str:
    ver_file = Path(GAME_DIR) / "game" / "ffxivgame.ver"
    try:
        return ver_file.read_text(encoding="utf-8").strip()
    except Exception as e:
        sys.exit(f"Failed to read game version from {ver_file}: {e}")


def sha256sum(file_path: Path) -> str:
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(file_path, "rb", buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str, default="")
    parser.add_argument("--out", "-o", type=str, default="")
    parser.add_argument("--verify", "-v", action="store_true")
    args = parser.parse_args()

    target_path = args.path
    if not target_path:
        ver = get_game_version()
        target_path = str(Path(SPECIMENS_DIR) / "client-binaries" / ver)
        print(f"no --path given; using the archive for the installed client: {ver}")

    target_dir = Path(target_path).resolve()
    if not target_dir.is_dir():
        sys.exit(f"no such directory: {target_dir}")

    out_path = args.out
    if not out_path:
        target_str = str(target_dir)
        if target_str.lower().startswith(SPECIMENS_DIR.lower()):
            rel_str = target_str[len(SPECIMENS_DIR) :].strip(os.sep)
        else:
            rel_str = target_dir.name
        safe_name = rel_str.replace(os.sep, "-").replace("/", "-") + ".sha256"
        out_path = str(Path(MANIFESTS_DIR) / safe_name)

    manifest_file = Path(out_path)
    files = []
    total_bytes = 0
    for root, _, filenames in os.walk(target_dir):
        for name in filenames:
            if name == "SHA256":
                continue
            fp = Path(root) / name
            files.append(fp)
            total_bytes += fp.stat().st_size
    files.sort()

    def get_rel(p: Path) -> str:
        return str(p.relative_to(target_dir)).replace("\\", "/")

    if args.verify:
        if not manifest_file.exists():
            sys.exit("no manifest")
        expected = {}
        for line in manifest_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("  ", 1)
            if len(parts) == 2:
                expected[parts[1]] = parts[0]

        ok = 0
        bad = []
        missing = []
        present = set()
        for f in files:
            rel = get_rel(f)
            present.add(rel)
            if rel not in expected:
                missing.append(rel)
                continue
            if sha256sum(f) == expected[rel]:
                ok += 1
            else:
                bad.append(rel)
        gone = [k for k in expected if k not in present]
        print(f"verified OK : {ok}")
        if missing:
            print(f"not in manifest : {len(missing)}")
        if gone:
            print(f"in manifest but gone : {len(gone)}")
        if bad:
            print(f"CORRUPT : {len(bad)}")
            sys.exit(1)
        print("manifest verified: every file matches")
        sys.exit(0)

    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lines = []
    for i, f in enumerate(files):
        rel = get_rel(f)
        lines.append(f"{sha256sum(f)}  {rel}")

    text = "\n".join(lines) + "\n"
    manifest_file.write_bytes(text.encode("utf-8"))
    elapsed = max(1.0, time.time() - t0)
    print(f"wrote {manifest_file} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

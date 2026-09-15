"""Snapshot every Excel sheet in the installed client, and diff two snapshots.

Why this exists, and why it has a deadline: `sqpack` is rewritten **in place** by a patch. The
7.55 sheets stop existing the moment maintenance ends, and no amount of cleverness afterwards
recovers them. A "what did 7.56 change" question is only answerable if the "before" was taken
first, so this is the one job on the list where being late is the same as not doing it.

Two records per sheet, deliberately:

* **the raw hash** — BLAKE2b over the `.exh` and every `.exd` page exactly as they come off disk.
  Always correct, never depends on this file understanding the format, and it is what the diff
  actually compares. A sheet whose hash is unchanged did not change, full stop.
* **a decoded CSV** — best effort, for reading and for row-level diffs. Variant-2 (subrow) sheets
  are not decoded, because this reader does not implement subrows and a confidently wrong CSV is
  worse than an honest gap. Those sheets still get a raw hash, so a change in one is still caught;
  it just says "changed" without saying which row.

    python exd_snapshot.py snap  --out DIR              # take one, named by game version
    python exd_snapshot.py diff  OLD_DIR NEW_DIR        # what a patch did
    python exd_snapshot.py rows  OLD_DIR NEW_DIR Action # row-level, one sheet

Stdlib only, so it runs anywhere the game is installed without setting up an environment.
"""

import csv
import gzip
import hashlib
import io
import json
import os
import struct
import sys
import time

_KB = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kbtools")
)
if _KB not in sys.path:
    sys.path.insert(0, _KB)

import exdfile  # noqa: E402
import sqpack  # noqa: E402


def bar(done, total, width=40):
    if total <= 0:
        return "[%s]" % ("?" * width)
    n = int(done / total * width)
    return "[%s%s] %5.1f%%" % ("#" * n, "." * (width - n), 100 * done / total)


def fmt_eta(s):
    return (
        "%.0fs" % s
        if s < 90
        else ("%.0fm" % (s / 60) if s < 5400 else "%.1fh" % (s / 3600))
    )


def safe(name):
    """Sheet names contain '/' (e.g. quest/000/...); keep the shape, lose the traversal."""
    return name.replace("\\", "/").replace("..", "__").strip("/")


def sheet_variant(exh_bytes):
    """EXHF: magic(4) version(u16) dataOffset(u16) columnCount(u16) pageCount(u16)
    languageCount(u16) unknown(u16) u2(u8) variant(u8) ... -- variant 2 means subrows."""
    try:
        return exh_bytes[0x11]
    except IndexError:
        return 0


def sheet_files(gd, name, exh):
    """Every real file that makes up this sheet, in a fixed order, for hashing."""
    out = ["exd/%s.exh" % name]
    langs = ["_en", ""] if exh.language_count > 1 else [""]
    for start, _n in exh.pages:
        for suf in langs:
            p = "exd/%s_%d%s.exd" % (name, start, suf)
            if gd.exists(p):
                out.append(p)
                break
    return out


def snap(out_root):
    gd = sqpack.GameData(sqpack.find_game())
    version = sqpack.game_version(gd.sqpack)
    out = os.path.join(out_root, version)
    os.makedirs(out, exist_ok=True)
    print("game version : %s" % version)
    print("writing to   : %s" % out)

    names = [
        l.split(",")[0]
        for l in gd.read("exd/root.exl").decode("utf8", "replace").splitlines()[1:]
        if l.strip()
    ]
    print("sheets       : %d" % len(names))

    index = {"version": version, "taken": time.time(), "sheets": {}}
    t0 = time.time()
    ok = skipped = failed = 0

    for i, name in enumerate(names):
        rec = {}
        try:
            exh_bytes = gd.read("exd/%s.exh" % name)
            exh = exdfile.Exh(exh_bytes)
            rec["variant"] = sheet_variant(exh_bytes)
            rec["columns"] = exh.column_count
            rec["column_types"] = [t for t, _o in exh.columns]
            rec["declared_rows"] = exh.row_count

            h = hashlib.blake2b(digest_size=16)
            for p in sheet_files(gd, name, exh):
                h.update(gd.read(p))
            rec["hash"] = h.hexdigest()

            if rec["variant"] == 2:
                # Not decoded on purpose. See the module docstring: the hash still catches a change,
                # it just cannot say which row moved.
                rec["decoded"] = False
                skipped += 1
            else:
                _exh, pages = exdfile.load_sheet(gd, name)
                buf = io.StringIO()
                w = csv.writer(buf, lineterminator="\n")
                w.writerow(["_row"] + ["c%d" % k for k in range(exh.column_count)])
                n = 0
                for pg in pages:
                    for rid in sorted(pg.offsets):
                        row = pg.row(rid)
                        if row is not None:
                            w.writerow([rid] + row)
                            n += 1
                rec["rows"] = n
                rec["decoded"] = True
                dst = os.path.join(out, safe(name) + ".csv.gz")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with gzip.open(dst, "wt", encoding="utf-8", newline="") as f:
                    f.write(buf.getvalue())
                ok += 1
        except Exception as e:
            rec["error"] = "%s: %s" % (type(e).__name__, e)
            rec.setdefault("decoded", False)
            failed += 1
        index["sheets"][name] = rec

        if (i + 1) % 100 == 0 or i + 1 == len(names):
            el = time.time() - t0
            rate = (i + 1) / max(el, 1e-6)
            print(
                "\r  %s %5d/%-5d  %4.0f/s  eta %-6s"
                % (
                    bar(i + 1, len(names)),
                    i + 1,
                    len(names),
                    rate,
                    fmt_eta((len(names) - i - 1) / max(rate, 1e-6)),
                ),
                end="",
                flush=True,
            )
    print()

    with open(os.path.join(out, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
    total = sum(
        os.path.getsize(os.path.join(r, fn)) for r, _d, fs in os.walk(out) for fn in fs
    )
    print(
        "  decoded %d, hash-only (subrow) %d, failed %d, %.0fs, %.1f MB"
        % (ok, skipped, failed, time.time() - t0, total / 1e6)
    )
    if failed:
        print("  failures:")
        for n, r in index["sheets"].items():
            if "error" in r:
                print("    %-40s %s" % (n, r["error"]))
    return 0


def _load(d):
    with open(os.path.join(d, "index.json"), encoding="utf-8") as f:
        return json.load(f)


def diff(old_dir, new_dir):
    a, b = _load(old_dir), _load(new_dir)
    print("old: %s   new: %s" % (a["version"], b["version"]))
    an, bn = set(a["sheets"]), set(b["sheets"])

    added, removed = sorted(bn - an), sorted(an - bn)
    changed = sorted(
        n for n in an & bn if a["sheets"][n].get("hash") != b["sheets"][n].get("hash")
    )

    print("\nNEW SHEETS (%d) -- content the patch introduced:" % len(added))
    for n in added:
        r = b["sheets"][n]
        print(
            "  + %-46s %s rows, %s cols"
            % (n, r.get("rows", "?"), r.get("columns", "?"))
        )
    if not added:
        print("  (none)")

    print("\nREMOVED (%d):" % len(removed))
    for n in removed:
        print("  - %s" % n)
    if not removed:
        print("  (none)")

    print("\nCHANGED (%d):" % len(changed))
    for n in changed:
        ra, rb = a["sheets"][n], b["sheets"][n]
        note = []
        if (
            ra.get("rows") is not None
            and rb.get("rows") is not None
            and ra["rows"] != rb["rows"]
        ):
            note.append("rows %d -> %d" % (ra["rows"], rb["rows"]))
        if ra.get("columns") != rb.get("columns"):
            note.append("COLUMNS %s -> %s" % (ra.get("columns"), rb.get("columns")))
        elif ra.get("column_types") != rb.get("column_types"):
            note.append("column TYPES changed")
        if not rb.get("decoded"):
            note.append("subrow sheet: hash only")
        print("  ~ %-46s %s" % (n, ", ".join(note) or "content"))
    if not changed:
        print("  (none)")

    print(
        "\n  A column count or type change means EXDSchema needs updating before anything"
    )
    print(
        "  reading that sheet by name is trustworthy. Row-level: `rows OLD NEW <sheet>`."
    )
    return 0


def rows(old_dir, new_dir, sheet):
    def read(d):
        p = os.path.join(d, safe(sheet) + ".csv.gz")
        if not os.path.isfile(p):
            return None
        with gzip.open(p, "rt", encoding="utf-8", newline="") as f:
            rd = csv.reader(f)
            hdr = next(rd, None)
            return {r[0]: r[1:] for r in rd}, hdr

    ra, rb = read(old_dir), read(new_dir)
    if ra is None or rb is None:
        print(
            "sheet %r was not decoded in one of the snapshots (subrow, or it failed)"
            % sheet
        )
        return 2
    (A, ha), (B, hb) = ra, rb
    ka, kb = set(A), set(B)
    add, rem = sorted(kb - ka, key=int), sorted(ka - kb, key=int)
    chg = sorted((k for k in ka & kb if A[k] != B[k]), key=int)
    print("%s: +%d rows, -%d rows, ~%d changed" % (sheet, len(add), len(rem), len(chg)))
    if ha != hb:
        print(
            "  NOTE: column count changed (%d -> %d); the comparison below is positional and"
            % (len(ha) - 1, len(hb) - 1)
        )
        print("        therefore not meaningful until EXDSchema is updated.")
    print("\nNEW ROWS (the interesting ones for a new job):")
    for k in add[:80]:
        first = next((v for v in B[k] if v not in ("", "0", "False")), "")
        print("  +%-8s %s" % (k, first[:90]))
    if len(add) > 80:
        print("  ... %d more" % (len(add) - 80))
    print("\nCHANGED ROWS:")
    for k in chg[:40]:
        n = min(len(A[k]), len(B[k]))
        d = [(i, A[k][i], B[k][i]) for i in range(n) if A[k][i] != B[k][i]]
        note = "; ".join("c%d %r->%r" % t for t in d[:6])
        if len(A[k]) != len(B[k]):
            # A row that only got wider has no differing column below `n`, so without this it prints
            # a blank line and reads as "changed, but nothing changed" -- which is the least useful
            # thing a diff can say.
            note = ("row width %d -> %d" % (len(A[k]), len(B[k]))) + (
                ("; " + note) if note else "; same values in the shared columns"
            )
        print("  ~%-8s %s" % (k, note))
    if len(chg) > 40:
        print("  ... %d more" % (len(chg) - 40))
    return 0


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "snap":
        out = (
            argv[argv.index("--out") + 1]
            if "--out" in argv
            else os.path.join(_HERE, "..", "exd-snapshots")
        )
        return snap(os.path.abspath(out))
    if cmd == "diff" and len(argv) >= 4:
        return diff(argv[2], argv[3])
    if cmd == "rows" and len(argv) >= 5:
        return rows(argv[2], argv[3], argv[4])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))

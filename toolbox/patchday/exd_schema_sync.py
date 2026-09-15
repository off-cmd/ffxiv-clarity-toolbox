"""Bring an EXDSchema set into agreement with the installed client, offline.

Why this exists. `Lumina.Excel` generates its C# sheet structs at COMPILE time from EXDSchema YAML.
If a patch adds a column and the schema still describes the old shape, the generator either produces
structs that silently read the wrong columns or -- as seen on 2026-09-07 with
`PhantomWeaponExTodoDetailTxt` -- refuses to build at all. Either way the whole chain above it stops:
no Lumina.Excel, no Dalamud, no plugins. And the community's schema repo updates on its own schedule,
which on patch night is not ours.

This closes that gap without waiting for anybody. It reads the real column layout out of each `.exh`
in the installed game, compares it to what the schema claims, and writes a corrected schema for
anything that disagrees. Names are preserved wherever the shape still lines up; genuinely new columns
get `UnknownNN`.

The distinction that makes this worth doing at 4am: a schema does not need the RIGHT NAMES to unblock
the build, it needs the RIGHT SHAPE. A sheet whose columns are `Unknown0..Unknown7` compiles, loads,
and can be read by index today; the names can be backfilled from the community repo whenever it
catches up. Correct-but-unnamed beats correct-and-absent by several hours.

    uv run --with pyyaml python exd_schema_sync.py check  --schema DIR
    uv run --with pyyaml python exd_schema_sync.py write  --schema DIR --out DIR [--new-from SNAPSHOT]

`check` writes nothing. `write` emits only the sheets that disagree, into an overlay directory, so the
upstream repo is never edited in place and the diff is reviewable.

Column types come from the `.exh` and are authoritative: EXDSchema does not record scalar widths, so
a field's *kind* (scalar / link / icon / array) is preserved from the old schema where it survives,
and defaulted to a plain scalar where it does not.
"""

import argparse
import json
import os
import struct
import sys

_KB = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kbtools")
)
if _KB not in sys.path:
    sys.path.insert(0, _KB)

import exdfile  # noqa: E402
import sqpack  # noqa: E402

try:
    import yaml
except ImportError:
    print("PyYAML is required:  uv run --with pyyaml python exd_schema_sync.py ...")
    raise SystemExit(2)

# EXH column type ids -> a human name, for the report only. The generator does not take widths from
# the schema, so these are never written into YAML; they exist so a mismatch report can say what
# actually changed rather than only that something did.
TYPE_NAME = {
    0: "string",
    1: "bool",
    2: "int8",
    3: "uint8",
    4: "int16",
    5: "uint16",
    6: "int32",
    7: "uint32",
    8: "float32",
    9: "float32",
    11: "uint64",
}


def type_name(t):
    return TYPE_NAME.get(t, "packedbool" if t >= 25 else "type%d" % t)


def flatten(fields):
    """A schema's field list -> the number of COLUMNS it claims, and a flat list of leaf names.

    An `array` occupies `count` columns, or `count x len(nested)` when it has its own `fields`.
    Everything else is one column. This is the only part of the schema that has to be understood to
    know whether it still fits the sheet.
    """
    names = []
    for f in fields or []:
        if not isinstance(f, dict):
            names.append(None)
            continue
        nm = f.get("name")
        if f.get("type") == "array":
            n = int(f.get("count", 1) or 1)
            sub = f.get("fields")
            if sub:
                subnames = flatten(sub)[1]
                for i in range(n):
                    names.extend(["%s[%d].%s" % (nm, i, s or "?") for s in subnames])
            else:
                names.extend(["%s[%d]" % (nm, i) for i in range(n)])
        else:
            names.append(nm)
    return len(names), names


def game_sheets(gd):
    """-> {name: (variant, [column type ids in file order])} for every sheet in root.exl."""
    out = {}
    names = [
        l.split(",")[0]
        for l in gd.read("exd/root.exl").decode("utf8", "replace").splitlines()[1:]
        if l.strip()
    ]
    for n in names:
        try:
            raw = gd.read("exd/%s.exh" % n)
            exh = exdfile.Exh(raw)
            out[n] = (raw[0x11] if len(raw) > 0x11 else 0, [t for t, _o in exh.columns])
        except Exception as e:
            out[n] = (None, "error: %s" % e)
    return out


def load_schema(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def schema_dir_files(d):
    """EXDSchema ships either flat (experimental) or under schemas/<version>/. Handle both."""
    if os.path.isdir(os.path.join(d, "schemas")):
        versions = sorted(os.listdir(os.path.join(d, "schemas")))
        pref = "latest" if "latest" in versions else versions[-1]
        d = os.path.join(d, "schemas", pref)
    return d, {
        fn[:-4]: os.path.join(d, fn) for fn in os.listdir(d) if fn.endswith(".yml")
    }


def rebuild_fields(old_fields, want, have_types):
    """Produce a field list of exactly `want` columns, keeping as much of `old_fields` as still fits.

    Truncating or extending is done at the END, never in the middle. A patch that inserts a column in
    the middle would silently rename everything after it, and a wrong name is worse than no name --
    it reads as authoritative. If the count shrinks, trailing fields are dropped; if it grows,
    `UnknownNN` are appended. Anything more clever than that needs a human who has seen the data.
    """
    kept, count = [], 0
    for f in old_fields or []:
        n = flatten([f])[0]
        if count + n > want:
            break
        kept.append(f)
        count += n
    for i in range(count, want):
        kept.append({"name": "Unknown%d" % i})
    return kept


# ---------------------------------------------------------------------------------------------
# Column hash: predicting, offline, which sheets GetExcelSheet<T>() will silently return null for.
#
# Lumina computes a sheet's hash as Crc32 over the raw column-definition bytes of the .exh -- 4 bytes
# per column, big-endian (ushort Type, ushort Offset), starting at file offset 32. The generated row
# struct carries the hash EXDSchema's columns.yml implies, baked in at COMPILE time. On mismatch
# ExcelModule throws MismatchedColumnHashException -- and GameData.GetExcelSheet<T>() CATCHES it and
# returns null:
#
#     catch( Exception e ) when ( e is SheetNotFoundException or MismatchedColumnHashException or ... )
#         return null;
#
# So a stale schema does not announce itself. The sheet is simply, silently, not there. That is
# upstream issue NotAdam/Lumina#127 ("GetExcelSheet<TerritoryType>() returning null ... since last
# patch"), and after a patch it will be true of every sheet whose columns moved.
#
# This computes both sides from files on disk, so the list of sheets that will come back null can be
# produced before anything is launched.
_CRC_TABLE = []
for _i in range(256):
    _r = _i
    for _ in range(8):
        _r = (0xEDB88320 ^ (_r >> 1)) if (_r & 1) else (_r >> 1)
    _CRC_TABLE.append(_r)


def lumina_crc32(buf, crc=0):
    """Lumina.Misc.Crc32: reflected, poly 0xEDB88320, seed 0, and NO final inversion -- its
    `return ~( crc ^ uint.MaxValue )` is the identity. Not zlib's crc32, which pre- and post-inverts,
    so this cannot be delegated to the stdlib."""
    for b in buf:
        crc = _CRC_TABLE[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return crc & 0xFFFFFFFF


SCHEMA_TYPE_ID = {
    "string": 0,
    "bool": 1,
    "int8": 2,
    "uint8": 3,
    "int16": 4,
    "uint16": 5,
    "int32": 6,
    "uint32": 7,
    "float32": 9,
    "int64": 0xA,
    "uint64": 0xB,
}
for _i in range(8):
    SCHEMA_TYPE_ID["packedbool%d" % _i] = 0x19 + _i


def client_column_hash(gd, name):
    raw = gd.read("exd/%s.exh" % name)
    ncol = struct.unpack_from(">H", raw, 8)[0]
    return lumina_crc32(raw[32 : 32 + 4 * ncol]), ncol


def schema_column_hash(entry):
    """columns.yml keeps the .exh's own column order, so the bytes rebuild directly."""
    b = b"".join(
        struct.pack(">HH", SCHEMA_TYPE_ID.get(e["type"], 0xFFFF), e["offset"])
        for e in entry
    )
    return lumina_crc32(b), len(entry)


# ---------------------------------------------------------------------------------------------
# Regenerating columns.yml.
#
# It was recorded here -- and in dependency-graph.toml -- that regenerating columns.yml from the
# client was the one step this tool could not do. That was wrong, and the mistake came from
# conflating two different files. Per-sheet YAML carries NAMES, which only a human can supply.
# columns.yml carries no names at all: it is `{type, offset}` per column, in .exh order, which is a
# straight transcription of the column table this tool already parses in order to compute the client
# hash. Verified 2026-09-08 against XBMPet and PhantomWeaponExTodoDetailTxt.
#
# So the shape fix and the hash fix are BOTH mechanical, and leaving the second one undone is what
# turns a fixed build into a sheet that silently returns null at runtime.
#
# Two rules make this safe to run unattended:
#   * The type-id -> name map used here is the exact inverse of SCHEMA_TYPE_ID, NOT TYPE_NAME.
#     TYPE_NAME maps both 8 and 9 to "float32" because it exists only for human-readable reports;
#     round-tripping through it would rewrite an id-8 column as id 9 and change the hash while
#     appearing to succeed. Any id without an exact inverse aborts that sheet instead of guessing.
#   * Every rewritten entry is re-hashed and compared to the client BEFORE the file is written, and
#     the whole file is re-parsed afterwards. A regeneration that does not reproduce the client hash
#     is a bug, not a result.

SCHEMA_TYPE_NAME = {v: k for k, v in SCHEMA_TYPE_ID.items()}
assert len(SCHEMA_TYPE_NAME) == len(SCHEMA_TYPE_ID), "type id map is not one-to-one"


def exh_columns(gd, name):
    """-> [(type_id, offset)] in .exh file order."""
    raw = gd.read("exd/%s.exh" % name)
    ncol = struct.unpack_from(">H", raw, 8)[0]
    return [struct.unpack_from(">HH", raw, 32 + 4 * i) for i in range(ncol)]


def _splice(text, key, body):
    """Replace one top-level `key:` block, leaving every other byte of the file untouched.

    A whole-file YAML round-trip would reformat 1198 unrelated entries and make the diff unreviewable
    at the exact moment review matters most. The format is rigidly regular -- a block runs from
    `^key:$` to the next line starting in column 0 -- so a textual splice is both smaller and easier
    to check by eye."""
    lines = text.split("\n")
    start = None
    for i, l in enumerate(lines):
        if l == key + ":":
            start = i
            break
    if start is None:
        return None
    end = start + 1
    while end < len(lines) and (
        lines[end].startswith(" ")
        or lines[end].startswith("-")
        or not lines[end].strip()
    ):
        end += 1
    return "\n".join(lines[:start] + [key + ":"] + body + lines[end:])


def cmd_regen_columns(a):
    gd = sqpack.GameData(sqpack.find_game())
    version = sqpack.game_version(gd.sqpack)
    with open(a.columns, encoding="utf-8") as f:
        raw_text = f.read()
    cols = yaml.safe_load(raw_text)
    print("game    : %s" % version)
    print("columns : %s  (%d entries)\n" % (a.columns, len(cols)))

    names = [
        l.split(",")[0]
        for l in gd.read("exd/root.exl").decode("utf8", "replace").splitlines()[1:]
        if l.strip()
    ]

    # default: every sheet whose hash currently disagrees
    targets = a.sheet or []
    if not targets:
        for n in names:
            entry = cols.get(n) or cols.get(n + "@Subrow")
            if not entry:
                continue
            try:
                ch, _ = client_column_hash(gd, n)
            except Exception:
                continue
            if schema_column_hash(entry)[0] != ch:
                targets.append(n)

    if not targets:
        print("  nothing to do -- every described sheet already hashes to the client.")
        return 0

    text = raw_text
    fixed, refused = [], []
    for n in targets:
        key = n if n in cols else (n + "@Subrow" if (n + "@Subrow") in cols else None)
        if key is None:
            refused.append((n, "not described in columns.yml"))
            continue
        try:
            pairs = exh_columns(gd, n)
        except Exception as e:
            refused.append((n, "unreadable .exh: %s" % e))
            continue
        unknown = sorted({t for t, _ in pairs if t not in SCHEMA_TYPE_NAME})
        if unknown:
            refused.append(
                (
                    n,
                    "column type id(s) %s have no schema name"
                    % ", ".join("0x%02x" % u for u in unknown),
                )
            )
            continue
        entry = [{"type": SCHEMA_TYPE_NAME[t], "offset": o} for t, o in pairs]

        want, _ = client_column_hash(gd, n)
        got, _ = schema_column_hash(entry)
        if got != want:
            refused.append(
                (n, "regenerated hash %08x != client %08x -- REFUSING" % (got, want))
            )
            continue

        body = []
        for e in entry:
            body.append("  - type: %s" % e["type"])
            body.append("    offset: %d" % e["offset"])
        spliced = _splice(text, key, body)
        if spliced is None:
            refused.append((n, "could not locate `%s:` block in the file" % key))
            continue
        text = spliced
        old = schema_column_hash(cols[key])
        fixed.append((n, key, old[1], old[0], len(entry), want))

    # re-parse before committing: a splice that produced invalid YAML must not reach disk
    try:
        reparsed = yaml.safe_load(text)
    except Exception as e:
        print("  ABORT: the rewritten file does not parse (%s). Nothing written." % e)
        return 1
    for n, key, _oc, _oh, _nc, want in fixed:
        if schema_column_hash(reparsed[key])[0] != want:
            print(
                "  ABORT: %s does not hash correctly after re-parse. Nothing written."
                % n
            )
            return 1
    if len(reparsed) != len(cols):
        print(
            "  ABORT: entry count changed %d -> %d. Nothing written."
            % (len(cols), len(reparsed))
        )
        return 1

    if a.dry_run:
        print("  (dry run -- nothing written)")
    else:
        with open(a.columns, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)

    if fixed:
        print("  regenerated %d:" % len(fixed))
        print("    sheet                                    was              now")
        for n, _k, oc, oh, nc, nh in fixed:
            print(
                "      %-38s %2d cols %08x   %2d cols %08x" % (n[:38], oc, oh, nc, nh)
            )
    if refused:
        print("\n  REFUSED %d:" % len(refused))
        for n, why in refused:
            print("      %-38s %s" % (n[:38], why))
        print(
            "\n  A refusal is not a failure of the patch -- it means this sheet needs a human."
        )
    return 1 if refused else 0


def cmd_hashcheck(a):
    gd = sqpack.GameData(sqpack.find_game())
    version = sqpack.game_version(gd.sqpack)
    with open(a.columns, encoding="utf-8") as f:
        cols = yaml.safe_load(f)
    print("game    : %s" % version)
    print("columns : %s  (%d entries)\n" % (a.columns, len(cols)))

    names = [
        l.split(",")[0]
        for l in gd.read("exd/root.exl").decode("utf8", "replace").splitlines()[1:]
        if l.strip()
    ]
    same = absent = 0
    bad = []
    for n in names:
        try:
            ch, ncol = client_column_hash(gd, n)
        except Exception:
            continue
        entry = cols.get(n)
        if not entry:
            absent += 1
            continue
        sh, scol = schema_column_hash(entry)
        if sh == ch:
            same += 1
        else:
            bad.append((n, scol, sh, ncol, ch))

    print("  %-42s %5d" % ("hash agrees (sheet loads normally)", same))
    print(
        "  %-42s %5d   <-- GetExcelSheet<T>() returns NULL"
        % ("hash MISMATCH", len(bad))
    )
    print(
        "  %-42s %5d   (no struct generated; unaffected)"
        % ("not in columns.yml", absent)
    )
    if bad:
        print("\n  sheet                                    schema            client")
        for n, sc, sh, cc, ch in bad[:60]:
            print("    %-38s %2d cols %08x   %2d cols %08x" % (n[:38], sc, sh, cc, ch))
        if len(bad) > 60:
            print("    ... %d more" % (len(bad) - 60))
        print(
            "\n  Each of these silently returns null until columns.yml is regenerated from this"
        )
        print("  client. Nothing throws, nothing logs -- the sheet is just absent.")
    return 0


def cmd_check(a):
    gd = sqpack.GameData(sqpack.find_game())
    version = sqpack.game_version(gd.sqpack)
    sdir, files = schema_dir_files(a.schema)
    print("game     : %s" % version)
    print("schemas  : %s  (%d sheets described)" % (sdir, len(files)))

    sheets = game_sheets(gd)
    print("client   : %d sheets\n" % len(sheets))

    ok = mismatch = unschemaed = broken = 0
    problems = []
    for name, (variant, types) in sorted(sheets.items()):
        if isinstance(types, str):
            broken += 1
            continue
        p = files.get(name)
        if p is None:
            unschemaed += 1
            continue
        try:
            sc = load_schema(p)
            want, _ = flatten(sc.get("fields"))
        except Exception as e:
            problems.append((name, "unreadable schema: %s" % e, None, None))
            mismatch += 1
            continue
        if want == len(types):
            ok += 1
        else:
            mismatch += 1
            problems.append((name, "columns", want, len(types)))

    print("  %-30s %6d" % ("schema matches the client", ok))
    print(
        "  %-30s %6d   <-- these break or mislead the generator"
        % ("schema DISAGREES", mismatch)
    )
    print(
        "  %-30s %6d   (normal: nobody has named them)"
        % ("no schema at all", unschemaed)
    )
    print("  %-30s %6d" % ("unreadable .exh", broken))

    if problems:
        print("\n  sheet                                    schema says  client has")
        for name, kind, want, have in problems[:60]:
            if kind == "columns":
                print("    %-38s %6s      %6s" % (name[:38], want, have))
            else:
                print("    %-38s %s" % (name[:38], kind))
        if len(problems) > 60:
            print("    ... %d more" % (len(problems) - 60))
    return 0


def cmd_write(a):
    gd = sqpack.GameData(sqpack.find_game())
    version = sqpack.game_version(gd.sqpack)
    sdir, files = schema_dir_files(a.schema)
    sheets = game_sheets(gd)

    known_before = set()
    if a.new_from:
        with open(os.path.join(a.new_from, "index.json"), encoding="utf-8") as f:
            known_before = set(json.load(f)["sheets"])

    os.makedirs(a.out, exist_ok=True)
    patched = created = 0
    log = []
    for name, (variant, types) in sorted(sheets.items()):
        if isinstance(types, str):
            continue
        want_cols = len(types)
        p = files.get(name)

        if p is None:
            # Only emit a skeleton for a sheet the patch actually ADDED. Without --new-from this
            # would produce thousands of files for sheets nobody has ever named, which is noise, not
            # work: the build does not need them and no plugin reads them.
            if not a.new_from or name in known_before:
                continue
            sc = {
                "name": name,
                "fields": [{"name": "Unknown%d" % i} for i in range(want_cols)],
            }
            note = "NEW sheet, %d columns (%s)" % (
                want_cols,
                ", ".join(type_name(t) for t in types[:8])
                + ("..." if len(types) > 8 else ""),
            )
            created += 1
        else:
            sc = load_schema(p)
            have, _ = flatten(sc.get("fields"))
            if have == want_cols:
                continue
            sc["fields"] = rebuild_fields(sc.get("fields"), want_cols, types)
            note = "columns %d -> %d" % (have, want_cols)
            patched += 1

        dst = os.path.join(a.out, name + ".yml")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                sc, f, sort_keys=False, default_flow_style=False, allow_unicode=True
            )
        log.append((name, note))

    print("game %s -> overlay %s" % (version, a.out))
    print("  patched %d, created %d" % (patched, created))
    for n, note in log[:80]:
        print("    %-40s %s" % (n[:40], note))
    if len(log) > 80:
        print("    ... %d more" % (len(log) - 80))
    if log:
        print(
            "\n  These are SHAPE fixes, not names. They exist so Lumina.Excel builds and reads the"
        )
        print(
            "  right columns today; replace them with the community's named versions when it"
        )
        print(
            "  catches up. Copy over a schema checkout, or point ExperimentalSchemaPath here."
        )
    return 0


def cmd_drop(a):
    """Remove a sheet's schema from every schema set, plus anything that links to it.

    The last resort, and a proven one. On 2026-09-07 `PhantomWeaponExTodoDetailTxt` refused to
    generate at its true column count under four different schemas -- the community's names, mine,
    with and without `displayField`, and with the base set patched to agree. It is an upstream
    generator bug, and at 4am on patch night the goal is a working Lumina.Excel, not a fixed
    generator.

    Dropping it built clean. The cost is precisely one sheet with no named struct, which is then read
    by column index -- which is how `exd_snapshot` reads everything anyway.

    Links have to go too: `PhantomWeaponExTodoDetails` targets `...DetailTxt`, so dropping only the
    target left a dangling type reference (CS0246). The cascade is bounded, so "drop the sheet and
    whatever points at it" is a usable rule rather than an unravelling thread.
    """
    removed = []
    for d in a.schema:
        sdir, files = schema_dir_files(d)
        targets = set(a.sheet)
        if not a.no_links:
            for name, p in files.items():
                if name in targets:
                    continue
                try:
                    txt = open(p, encoding="utf-8").read()
                except OSError:
                    continue
                if any(
                    ("[%s]" % t) in txt
                    or ("targets: [%s" % t) in txt
                    or (" %s," % t) in txt
                    or (" %s]" % t) in txt
                    for t in a.sheet
                ):
                    targets.add(name)
        for name in sorted(targets):
            p = files.get(name)
            if p and os.path.isfile(p):
                if a.dry_run:
                    removed.append((p, "would remove"))
                else:
                    os.remove(p)
                    removed.append((p, "removed"))
    for p, what in removed:
        print("  %-10s %s" % (what, p))
    if not removed:
        print("  nothing matched")
    print(
        "\n  These are working-tree deletions in a submodule. `git -C <schema> checkout -- .`"
    )
    print("  puts them back, and the build result tells you whether they needed to go.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="exd_schema_sync")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="report where the schema and the client disagree")
    c.add_argument("--schema", required=True)
    c.set_defaults(fn=cmd_check)
    w = sub.add_parser("write", help="emit corrected schemas into an overlay directory")
    w.add_argument("--schema", required=True)
    w.add_argument("--out", required=True)
    w.add_argument(
        "--new-from",
        metavar="SNAPSHOT",
        help="an exd-snapshot dir from BEFORE the patch; only sheets absent from it count "
        "as new, so this does not emit skeletons for the thousands of sheets nobody "
        "has ever named",
    )
    w.set_defaults(fn=cmd_write)
    d = sub.add_parser(
        "drop",
        help="last resort: exclude a sheet that refuses to generate, and "
        "anything linking to it, from every schema set",
    )
    d.add_argument(
        "--schema",
        required=True,
        action="append",
        help="repeat for each schema set -- BOTH EXDSchema and EXDSchema-experimental "
        "are generated in the same compilation, so a sheet left in one still breaks it",
    )
    d.add_argument("--sheet", required=True, action="append")
    d.add_argument(
        "--no-links",
        action="store_true",
        help="do not also drop sheets that link to it",
    )
    d.add_argument("--dry-run", action="store_true")
    d.set_defaults(fn=cmd_drop)
    h = sub.add_parser(
        "hashcheck",
        help="predict which sheets GetExcelSheet<T>() will silently "
        "return null for, from files alone",
    )
    h.add_argument(
        "--columns", required=True, help="path to a schema set's .github/columns.yml"
    )
    h.set_defaults(fn=cmd_hashcheck)
    r = sub.add_parser(
        "regen-columns",
        help="rewrite columns.yml entries from the installed client "
        "so GetExcelSheet<T>() stops returning null",
    )
    r.add_argument(
        "--columns", required=True, help="path to a schema set's .github/columns.yml"
    )
    r.add_argument(
        "--sheet", action="append", help="repeat; default is every mismatching sheet"
    )
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(fn=cmd_regen_columns)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

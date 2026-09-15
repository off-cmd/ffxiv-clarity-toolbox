"""Resolve FFXIVClientStructs signatures against a client binary, offline, and COUNT the matches.

Why this exists. On patch day the question "did this patch break any signatures" was being answered
by reading FFXIVClientStructs' commit log -- i.e. by waiting for someone else to find out. On
2026-09-08 that produced a confident, wrong answer: 0 signature edits at 09:58 read as "nothing
broke", when it meant "nobody has fixed anything yet". Twelve were broken. The inputs to know that --
the new exe and the signature set -- were on disk the whole time.

This does what InteropGenerator.Runtime.Resolver does at game start, with one deliberate difference.
The Resolver takes the FIRST match and never checks for a second. A pattern that matches twice is
therefore not an error to it; it silently resolves to whichever the scan reached first, which may be
the wrong function. That is the failure mode that "sends invalid data to the game servers" rather than
crashing. Counting every match is the entire point of doing this offline.

    uv run --with pefile --with pyyaml python sigscan.py inventory --cs DIR        -> sigs.json
    uv run --with pefile --with pyyaml python sigscan.py scan --exe EXE --sigs sigs.json -> results.json
    uv run --with pefile --with pyyaml python sigscan.py check --results results.json --data data.yml
    uv run --with pefile --with pyyaml python sigscan.py gate --exe EXE --cs DIR    (all three, exit 1 on any != 1)

Semantics are copied from the runtime, not approximated:
  * only the .text section is scanned (Resolver.SetupSections)
  * a byte matches if (mask & pattern) == (mask & byte); "??" and "?" are mask 0
  * RelativeFollowOffsets are applied in order: loc = loc + off + 4 + int32@(loc+off)
  * a MemberFunction whose signature starts with E8 or E9 and declares no offsets gets [1]
    (InteropGenerator SignatureInfo.GetRelCallAndJumpAdjustedOffset)
  * VirtualFunction(n) is *(vtbl + 8n), read from the file image at the preferred base
  * final address = ImageBase + .text VirtualAddress + location
"""

import argparse
import json
import os
import re
import struct
import sys
import time

# ------------------------------------------------------------------------------------------------
# source inventory
# ------------------------------------------------------------------------------------------------

ATTR = re.compile(
    r'\[(MemberFunction|StaticAddress|VirtualTable)\(\s*"([0-9A-Fa-f? ]+)"\s*(?:,\s*((?:\[[^\]]*\]|[^\])])*?))?\)\s*(?:,[^\]]*)?\]'
)
VFUNC = re.compile(r"\[VirtualFunction\(\s*(\d+)u?\s*\)\s*(?:,[^\]]*)?\]")
STRUCT = re.compile(r"\b(?:partial\s+)?struct\s+([A-Za-z_]\w*)")
METHOD = re.compile(
    r"^\s*(?:public|internal|private|protected)\s+(?:static\s+)?(?:unsafe\s+)?(?:partial\s+)?(?:new\s+)?[\w<>,\.\*\[\]\s]+?\s+([A-Za-z_]\w*)\s*(?:<[^>()]*>)?\s*\("
)
NAMESPACE = re.compile(r"^\s*namespace\s+([\w\.]+)\s*[;{]")


def parse_offsets(argstr):
    """'3' -> [3];  '[3, 63]' -> [3, 63];  '3, isPointer: true' -> [3];  '' -> []"""
    if not argstr:
        return [], False
    is_ptr = "isPointer" in argstr and "true" in argstr
    m = re.search(r"\[([^\]]*)\]", argstr)
    if m:
        # array form: every number inside the brackets is a follow offset
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
    else:
        # scalar form: ONLY the first number is a follow offset. The second, when present, is
        # VirtualTable's functionCount (or nothing) -- reading it as a second follow would jump
        # through garbage and resolve confidently to the wrong place.
        head = argstr.split("isPointer")[0]
        found = re.findall(r"\b\d+\b", head)
        nums = [int(found[0])] if found else []
    return nums, is_ptr


def inventory(cs_dir):
    """Walk a FFXIVClientStructs/FFXIV source tree -> list of signature records.

    Struct nesting is tracked by brace depth so nested structs get Outer.Inner names, matching what
    the generator emits (StructInfo.Hierarchy). VirtualTable sits ON the struct, so it is attached to
    the next struct declaration rather than the current one."""
    out = []
    for root, _dirs, files in os.walk(cs_dir):
        for fn in files:
            if not fn.endswith(".cs"):
                continue
            path = os.path.join(root, fn)
            try:
                text = open(path, encoding="utf-8-sig").read()
            except Exception:
                continue
            ns = None
            stack = []  # [(name, depth_when_opened)]
            depth = 0
            pending_attr = []  # attributes seen, waiting for a method
            pending_vtbl = None  # VirtualTable waiting for its struct
            for line in text.split("\n"):
                s = line.strip()
                if s.startswith("//"):
                    continue
                m = NAMESPACE.match(line)
                if m:
                    ns = m.group(1)
                # struct declaration on this line?
                sm = STRUCT.search(line) if "struct" in line else None
                if sm and not s.startswith("//"):
                    name = sm.group(1)
                    # attach a pending VirtualTable to this struct
                    if pending_vtbl is not None:
                        sig, offs, _ = pending_vtbl
                        out.append(
                            dict(
                                kind="VirtualTable",
                                ns=ns,
                                hier=[*[n for n, _, _ in stack], name],
                                method="StaticVirtualTable",
                                sig=sig,
                                offsets=offs,
                                file=path,
                            )
                        )
                        pending_vtbl = None
                    # push when its brace opens (same line or next); record depth now
                    stack.append([name, depth, False])
                for am in ATTR.finditer(line):
                    kind, sig, args = am.group(1), am.group(2).strip(), am.group(3)
                    offs, is_ptr = parse_offsets(args or "")
                    if kind == "VirtualTable":
                        pending_vtbl = (sig, offs, False)
                    else:
                        pending_attr.append((kind, sig, offs, is_ptr))
                for vm in VFUNC.finditer(line):
                    pending_attr.append(
                        ("VirtualFunction", None, [int(vm.group(1))], False)
                    )
                # method declaration consumes pending attributes
                if pending_attr and "(" in line and not sm:
                    # attributes may share the line with the declaration:
                    #   [VirtualFunction(4)] public partial void ExitGame();
                    mm = METHOD.match(re.sub(r"^\s*(?:\[[^\]]*\]\s*)+", "", line))
                    if mm and mm.group(1) not in (
                        "if",
                        "while",
                        "for",
                        "switch",
                        "return",
                    ):
                        hier = [n for n, _, _ in stack]
                        for kind, sig, offs, is_ptr in pending_attr:
                            out.append(
                                dict(
                                    kind=kind,
                                    ns=ns,
                                    hier=hier,
                                    method=mm.group(1),
                                    sig=sig,
                                    offsets=offs,
                                    is_pointer=is_ptr,
                                    file=path,
                                )
                            )
                        pending_attr = []
                # brace tracking (strings in this codebase never contain braces on attribute lines)
                opens, closes = line.count("{"), line.count("}")
                for _ in range(opens):
                    depth += 1
                    for st in stack:
                        if not st[2] and st[1] == depth - 1:
                            st[2] = True  # this struct's own brace opened
                for _ in range(closes):
                    depth -= 1
                    while stack and stack[-1][2] and stack[-1][1] >= depth:
                        stack.pop()
    return out


# ------------------------------------------------------------------------------------------------
# PE image
# ------------------------------------------------------------------------------------------------


class Image:
    """The file laid out as it would be in memory at its preferred base -- so vtable entries and
    RIP-relative targets read straight out of it without relocation."""

    def __init__(self, path):
        import pefile

        pe = pefile.PE(path, fast_load=True)
        self.base = pe.OPTIONAL_HEADER.ImageBase
        size = pe.OPTIONAL_HEADER.SizeOfImage
        buf = bytearray(size)
        hdr = pe.OPTIONAL_HEADER.SizeOfHeaders
        buf[:hdr] = pe.__data__[:hdr]
        self.sections = []
        for s in pe.sections:
            va, vs = s.VirtualAddress, s.Misc_VirtualSize
            raw = pe.__data__[s.PointerToRawData : s.PointerToRawData + s.SizeOfRawData]
            n = min(len(raw), size - va)
            buf[va : va + n] = raw[:n]
            self.sections.append(
                (s.Name.rstrip(b"\0").decode(errors="replace"), va, vs)
            )
        self.mem = bytes(buf)
        text = [s for s in self.sections if s[0] == ".text"]
        if not text:
            raise SystemExit("no .text section")
        _, self.text_va, self.text_size = text[0]
        self.text = self.mem[self.text_va : self.text_va + self.text_size]

    def u32(self, va):
        o = va - self.base
        return struct.unpack_from("<I", self.mem, o)[0]

    def i32(self, va):
        o = va - self.base
        return struct.unpack_from("<i", self.mem, o)[0]

    def u64(self, va):
        o = va - self.base
        return struct.unpack_from("<Q", self.mem, o)[0]

    def section_of(self, va):
        rva = va - self.base
        for name, sva, svs in self.sections:
            if sva <= rva < sva + svs:
                return name
        return None


# ------------------------------------------------------------------------------------------------
# patterns
# ------------------------------------------------------------------------------------------------


def parse_sig(sig):
    """'E8 ?? ?? ?? ?? 32 DB' -> (bytes, mask) where mask byte 0 means wildcard."""
    b, m = bytearray(), bytearray()
    for tok in sig.split():
        if tok in ("??", "?"):
            b.append(0)
            m.append(0)
        else:
            b.append(int(tok, 16))
            m.append(0xFF)
    return bytes(b), bytes(m)


def longest_run(b, m):
    """Offset and bytes of the longest contiguous fixed run -- the anchor for bytes.find."""
    best = (0, b"")
    i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            if j - i > len(best[1]):
                best = (i, b[i:j])
            i = j
        else:
            i += 1
    return best


def find_all(text, sig):
    """Every offset in `text` where `sig` matches.

    Regex with a leading wildcard-heavy pattern is the wrong tool here: an `E8 ?? ?? ?? ??` sig
    makes the engine stop at every call instruction in a 30 MB section. bytes.find on the longest
    fixed run is memchr-fast and usually lands on the discriminating bytes AFTER the wildcards; the
    full pattern is then verified only at those candidates."""
    b, m = parse_sig(sig)
    k, run = longest_run(b, m)
    if len(run) < 2:
        run, k = b[:1], 0
    n = len(b)
    out = []
    pos = text.find(run)
    while pos != -1:
        start = pos - k
        if start >= 0 and start + n <= len(text):
            seg = text[start : start + n]
            ok = True
            for i in range(n):
                if m[i] and seg[i] != b[i]:
                    ok = False
                    break
            if ok:
                out.append(start)
        pos = text.find(run, pos + 1)
    return out


def follow(img, loc, offsets):
    """Resolver arithmetic, verbatim. loc is an offset INTO .text; returns a VA."""
    for off in offsets:
        rel = img.i32(img.base + img.text_va + loc + off)
        loc = loc + off + 4 + rel
    return img.base + img.text_va + loc


def effective_offsets(rec):
    if (
        rec["kind"] == "MemberFunction"
        and not rec["offsets"]
        and rec["sig"][:2].upper() in ("E8", "E9")
    ):
        return [1]
    return rec["offsets"]


_TEXT = None


def _init(text):
    global _TEXT
    _TEXT = text


def _work(item):
    idx, sig = item
    try:
        return idx, find_all(_TEXT, sig), None
    except Exception as e:
        return idx, [], "bad signature: %s" % e


def scan(img, sigs, progress=True, jobs=None):
    """Every match of every pattern. This is the one thing the runtime resolver does not do."""
    import multiprocessing as mp

    patterns = [
        (i, r["sig"]) for i, r in enumerate(sigs) if r["kind"] != "VirtualFunction"
    ]
    results = [dict(r) for r in sigs]
    t0 = time.time()
    jobs = jobs or max(1, (os.cpu_count() or 2) - 1)
    done = 0
    with mp.Pool(jobs, initializer=_init, initargs=(img.text,)) as pool:
        for idx, locs, note in pool.imap_unordered(_work, patterns, chunksize=16):
            r = results[idx]
            r["matches"] = [img.base + img.text_va + l for l in locs]
            if note:
                r["note"] = note
            if locs:
                try:
                    r["resolved"] = follow(img, locs[0], effective_offsets(sigs[idx]))
                except Exception as e:
                    r["resolved"] = None
                    r["note"] = "follow failed: %s" % e
            else:
                r["resolved"] = None
            done += 1
            if progress and done % 400 == 0:
                print(
                    "  %5d/%d  %.0fs" % (done, len(patterns), time.time() - t0),
                    file=sys.stderr,
                )
    # vtables resolved -> virtual functions
    vtbls = {}
    for r in results:
        if r["kind"] == "VirtualTable" and r.get("resolved"):
            vtbls[(r["ns"], ".".join(r["hier"]))] = r["resolved"]
    for r in results:
        if r["kind"] != "VirtualFunction":
            continue
        vt = vtbls.get((r["ns"], ".".join(r["hier"])))
        if vt is None:
            r.update(
                matches=[],
                resolved=None,
                note="no VirtualTable resolved for this struct",
            )
            continue
        try:
            r.update(matches=[vt], resolved=img.u64(vt + 8 * r["offsets"][0]), vtbl=vt)
        except Exception as e:
            r.update(matches=[], resolved=None, note="vtbl read failed: %s" % e)
    return results


# ------------------------------------------------------------------------------------------------
# data.yml comparison (naming rules from FFXIVClientStructs.ResolverTester)
# ------------------------------------------------------------------------------------------------


def load_data(path):
    import yaml

    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    classes = {}
    for cname, c in (d.get("classes") or {}).items():
        c = c or {}
        funcs = {}
        for addr, name in (c.get("funcs") or {}).items():
            funcs[str(name)] = (
                int(addr) if isinstance(addr, int) else int(str(addr), 16)
            )
        inst = [
            int(str(x.get("ea")), 16) if not isinstance(x.get("ea"), int) else x["ea"]
            for x in (c.get("instances") or [])
            if isinstance(x, dict)
        ]
        vt = [
            int(str(x.get("ea")), 16) if not isinstance(x.get("ea"), int) else x["ea"]
            for x in (c.get("vtbls") or [])
            if isinstance(x, dict)
        ]
        vf = {int(k): str(v) for k, v in (c.get("vfuncs") or {}).items()}
        by_addr = {a: n for n, a in funcs.items()}
        classes[cname] = dict(
            funcs=funcs, instances=inst, vtbls=vt, vfuncs=vf, by_addr=by_addr
        )
    return d.get("version"), classes


def yml_name(rec):
    """FFXIVClientStructs.FFXIV.Client.Game.UI.PlayerState + IsMentor -> ('Client::Game::UI::PlayerState', 'IsMentor')"""
    ns = rec["ns"] or ""
    prefix = "FFXIVClientStructs.FFXIV."
    if ns.startswith(prefix):
        ns = ns[len(prefix) :]
    cls = "::".join([*ns.split("."), *rec["hier"]]) if ns else "::".join(rec["hier"])
    fn = rec["method"]
    if fn.startswith("Ctor") or fn.startswith("Dtor"):
        fn = fn[0].lower() + fn[1:]
    return cls, fn


def check(results, data_path, img=None):
    version, classes = load_data(data_path)
    agree = disagree = absent = unresolved = ambiguous = benign = novtbl = via_yml = 0
    rows = []
    for r in results:
        if (r["ns"] or "").startswith("FFXIVClientStructs.Havok"):
            continue
        cls, fn = yml_name(r)
        c = classes.get(cls)
        expect = None
        if c:
            # `Instance` is the data-section instance for a StaticAddress, but a MemberFunction
            # named Instance is a getter that lives in .text and is listed under funcs.
            if fn == "Instance" and r["kind"] == "StaticAddress":
                expect = c["instances"][0] if c["instances"] else None
            elif fn == "StaticVirtualTable":
                expect = c["vtbls"][0] if c["vtbls"] else None
            else:
                expect = c["funcs"].get(fn)
            # a VirtualFunction on a struct with no VirtualTable signature can still be verified:
            # data.yml knows the vtbl, and the file image holds the slot
            if (
                r["kind"] == "VirtualFunction"
                and r.get("resolved") is None
                and c["vtbls"]
                and img is not None
            ):
                try:
                    r = dict(
                        r,
                        resolved=img.u64(c["vtbls"][0] + 8 * r["offsets"][0]),
                        matches=[c["vtbls"][0]],
                        note="vtbl from data.yml",
                    )
                    via_yml += 1
                except Exception:
                    pass
        n = len(r.get("matches") or [])
        if r["kind"] != "VirtualFunction" and n > 1:
            # two call sites of one function are benign; two different targets are the hazard
            offs = effective_offsets(r)
            targets = set()
            if img is not None:
                for va in r["matches"]:
                    try:
                        targets.add(follow(img, va - img.base - img.text_va, offs))
                    except Exception:
                        targets.add(None)
            if len(targets) <= 1:
                benign += 1
            else:
                ambiguous += 1
                rows.append(
                    (
                        "AMBIGUOUS x%d" % n,
                        cls,
                        fn,
                        r.get("sig"),
                        r.get("resolved"),
                        expect,
                    )
                )
                continue
        if r.get("resolved") is None:
            if r["kind"] == "VirtualFunction":
                novtbl += 1
                continue
            unresolved += 1
            rows.append(("UNRESOLVED", cls, fn, r.get("sig"), None, expect))
            continue
        if r["kind"] == "VirtualFunction" and c:
            # Binary-level check: the address read out of slot n must be the function data.yml lists
            # under this name. Fall back to the slot->name table when the function has no funcs entry.
            idx = r["offsets"][0]
            named = c["by_addr"].get(r["resolved"])
            if named is not None:
                if named == fn:
                    agree += 1
                else:
                    disagree += 1
                    rows.append(
                        (
                            "VF-DISAGREE",
                            cls,
                            fn,
                            "slot %d" % idx,
                            r["resolved"],
                            c["funcs"].get(fn),
                        )
                    )
            elif idx in c["vfuncs"]:
                if c["vfuncs"][idx] == fn:
                    agree += 1
                else:
                    disagree += 1
                    rows.append(
                        (
                            "VF-SLOT",
                            cls,
                            fn,
                            "slot %d is %s in yml" % (idx, c["vfuncs"][idx]),
                            r["resolved"],
                            None,
                        )
                    )
            else:
                absent += 1
            continue
        if expect is None:
            absent += 1
            continue
        if expect == r["resolved"]:
            agree += 1
        else:
            disagree += 1
            rows.append(("DISAGREE", cls, fn, r.get("sig"), r["resolved"], expect))
    return dict(
        version=version,
        agree=agree,
        disagree=disagree,
        absent=absent,
        unresolved=unresolved,
        ambiguous=ambiguous,
        benign=benign,
        novtbl=novtbl,
        via_yml=via_yml,
        rows=rows,
    )


def print_check(rep, limit=60):
    print("data.yml version : %s" % rep["version"])
    print("  agree           %6d   resolved address == data.yml" % rep["agree"])
    print(
        "  DISAGREE        %6d   resolved, but to a different address" % rep["disagree"]
    )
    print(
        "  UNRESOLVED      %6d   0 matches -- the signature is broken"
        % rep["unresolved"]
    )
    print(
        "  AMBIGUOUS       %6d   >1 match with DIFFERENT targets -- runtime picks one silently"
        % rep["ambiguous"]
    )
    print(
        "  multi, benign   %6d   >1 match, same target (several call sites of one function)"
        % rep["benign"]
    )
    print(
        "  vfunc, no vtbl  %6d   VirtualFunction on a struct with no VirtualTable sig or data.yml vtbl"
        % rep["novtbl"]
    )
    print(
        "  not in data.yml %6d   (nothing to compare against; %d vfuncs verified via data.yml vtbl)"
        % (rep["absent"], rep["via_yml"])
    )
    if rep["rows"]:
        print()
        for kind, cls, fn, sig, got, exp in rep["rows"][:limit]:
            g = "%X" % got if got else "-"
            e = "%X" % exp if exp else "-"
            print(
                "  %-14s %-52s got %-10s yml %-10s  %s"
                % (kind, (cls + "." + fn)[-52:], g, e, (sig or "")[:40])
            )
        if len(rep["rows"]) > limit:
            print("  ... %d more" % (len(rep["rows"]) - limit))


# ------------------------------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sigscan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("inventory")
    a.add_argument("--cs", required=True)
    a.add_argument("--out", default="sigs.json")
    b = sub.add_parser("scan")
    b.add_argument("--exe", required=True)
    b.add_argument("--sigs", required=True)
    b.add_argument("--out", default="results.json")
    c = sub.add_parser("check")
    c.add_argument("--results", required=True)
    c.add_argument("--data", required=True)
    c.add_argument("--exe")
    g = sub.add_parser("gate")
    g.add_argument("--exe", required=True)
    g.add_argument("--cs", required=True)
    g.add_argument("--data")
    g.add_argument("--out", help="also write the scan results here (for offsetcheck)")
    args = ap.parse_args(argv)

    if args.cmd == "inventory":
        sigs = inventory(args.cs)
        json.dump(sigs, open(args.out, "w"), indent=0)
        from collections import Counter

        print("inventory: %d records -> %s" % (len(sigs), args.out))
        for k, v in sorted(Counter(s["kind"] for s in sigs).items()):
            print("  %-16s %d" % (k, v))
        return 0

    if args.cmd == "scan":
        img = Image(args.exe)
        print(
            "image: base %X  .text va %X size %X (%.1f MB)"
            % (img.base, img.text_va, img.text_size, img.text_size / 1e6)
        )
        sigs = json.load(open(args.sigs))
        t0 = time.time()
        res = scan(img, sigs)
        json.dump(res, open(args.out, "w"), indent=0)
        n1 = sum(
            1 for r in res if r["kind"] != "VirtualFunction" and len(r["matches"]) == 1
        )
        n0 = sum(
            1 for r in res if r["kind"] != "VirtualFunction" and len(r["matches"]) == 0
        )
        nn = sum(
            1 for r in res if r["kind"] != "VirtualFunction" and len(r["matches"]) > 1
        )
        print(
            "scanned %d patterns in %.1fs: unique %d, zero %d, multiple %d -> %s"
            % (n1 + n0 + nn, time.time() - t0, n1, n0, nn, args.out)
        )
        return 0

    if args.cmd == "check":
        res = json.load(open(args.results))
        img = Image(args.exe) if getattr(args, "exe", None) else None
        print_check(check(res, args.data, img))
        return 0

    if args.cmd == "gate":
        img = Image(args.exe)
        sigs = inventory(args.cs)
        res = scan(img, sigs, progress=False)
        if args.out:
            json.dump(res, open(args.out, "w"), indent=0)
        # a multi-match whose every hit resolves to the same function is a call-site duplicate, not a
        # hazard; the gate fails only on zero matches or on matches with DIFFERENT targets
        bad = []
        for r in res:
            if r["kind"] == "VirtualFunction":
                continue
            n = len(r["matches"])
            if n == 0:
                bad.append(r)
                continue
            if n > 1:
                offs = effective_offsets(r)
                targets = set()
                for va in r["matches"]:
                    try:
                        targets.add(follow(img, va - img.base - img.text_va, offs))
                    except Exception:
                        targets.add(None)
                if len(targets) > 1:
                    bad.append(r)
        print(
            "gate: %d patterns, %d with match count != 1"
            % (sum(1 for r in res if r["kind"] != "VirtualFunction"), len(bad))
        )
        for r in bad[:80]:
            cls, fn = yml_name(r)
            print(
                "  x%-3d %-56s %s"
                % (len(r["matches"]), (cls + "." + fn)[-56:], (r["sig"] or "")[:44])
            )
        if args.data:
            print_check(check(res, args.data, img))
        return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

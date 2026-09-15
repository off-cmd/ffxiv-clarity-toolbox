"""Verify FFXIVClientStructs field offsets against the client binary, from accessor disassembly.

The failure this catches is the silent one. A wrong signature fails to resolve and says so. A wrong
[FieldOffset] just reads the wrong memory: UIModule.ConfigModule declared at 0xAAE20 when the client
put it at 0xAAE40 returns whatever is thirty-two bytes short of the real struct, and nothing throws.
It is also the LAST class of repair to land on patch day -- addresses first, signatures second,
sizes third -- so it is the one an impatient build ships.

How it works. A great many accessors compile to a single addressing instruction and a return:

    lea rax, [rcx + 0xD2690] ; ret        embedded member  (offset is the displacement)
    mov rax, [rcx + 0x1AC0]  ; ret        pointer member
    movzx eax, byte [rcx+0x28]; ret       scalar member

Every resolved MemberFunction and every reachable VirtualFunction is disassembled; the ones with this
shape yield (struct, accessor name, displacement). The accessor name is matched to a field in the
same struct by CS naming convention (GetFoo -> Foo / FooPtr / Foo*, IsFoo -> IsFoo / Foo), and the
displacement compared to the declared [FieldOffset].

Struct SIZES fall out of adjacency: when a container exposes its embedded members through getters,
each member's size is bounded by the next member's offset. XBMModule at 0xAAD00 followed by the
next member at 0xAADA8 bounds it at 0xA8 -- which is exactly the size pohky committed.

What this cannot see, stated plainly: a field with no accessor. That is most fields. This verifies
the load-bearing ones -- the module pointers everything routes through -- and reports coverage so
"0 mismatches" is never mistaken for "everything verified".

    uv run --with pefile --with capstone --with pyyaml python offsetcheck.py --exe EXE --cs DIR --results results.json [--data data.yml]
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sigscan  # noqa: E402

INHERITS = re.compile(r"Inherits<([A-Za-z_][\w\.]*)")
FIELD = re.compile(
    r"\[FieldOffset\(0x([0-9A-Fa-f]+)\)(?:,[^\]]*)?\]\s*(?:public|internal|private|protected)?\s*(?:unsafe\s+)?(?:readonly\s+)?(?:fixed\s+)?[\w<>\.\*\[\],\s]+?\s+([A-Za-z_]\w*)\s*[;=\[]"
)
SIZE = re.compile(
    r"\[StructLayout\(LayoutKind\.Explicit,\s*Size\s*=\s*0x([0-9A-Fa-f]+)\)\]"
)
ACCESS = re.compile(r"\[r(?:cx|di|dx|si|8|9) \+ (0x[0-9a-f]+)\]")


def source_fields(cs_dir):
    """(ns, hierarchy) -> {field: offset}, and -> declared size. Same brace tracking as the
    signature inventory so nested structs land under the right owner."""
    fields, sizes, inherits = {}, {}, {}
    for root, _d, files in os.walk(cs_dir):
        for fn in files:
            if not fn.endswith(".cs"):
                continue
            path = os.path.join(root, fn)
            try:
                text = open(path, encoding="utf-8-sig").read()
            except Exception:
                continue
            ns, stack, depth, pending_size, pending_inh = None, [], 0, None, []
            for line in text.split("\n"):
                s = line.strip()
                if s.startswith("//"):
                    continue
                m = sigscan.NAMESPACE.match(line)
                if m:
                    ns = m.group(1)
                sm = sigscan.STRUCT.search(line) if "struct" in line else None
                zm = SIZE.search(line)
                if zm:
                    pending_size = int(zm.group(1), 16)
                if "Inherits<" in line and not sm:
                    pending_inh += INHERITS.findall(line)
                if sm:
                    stack.append([sm.group(1), depth, False])
                    key = (ns, ".".join(n for n, _, _ in stack))
                    if pending_size is not None:
                        sizes[key] = pending_size
                        pending_size = None
                    inh = INHERITS.findall(line) + pending_inh
                    if inh:
                        inherits[key] = [
                            b.split(".")[-1] for b in inh
                        ]  # first listed = primary base
                    pending_inh = []
                fm = FIELD.search(line)
                if fm and stack:
                    key = (ns, ".".join(n for n, _, _ in stack))
                    fields.setdefault(key, {})[fm.group(2)] = int(fm.group(1), 16)
                for _ in range(line.count("{")):
                    depth += 1
                    for st in stack:
                        if not st[2] and st[1] == depth - 1:
                            st[2] = True
                for _ in range(line.count("}")):
                    depth -= 1
                    while stack and stack[-1][2] and stack[-1][1] >= depth:
                        stack.pop()
    return fields, sizes, inherits


def accessor_shape(img, va):
    """(kind, displacement) if the function at va is a one-instruction accessor, else None."""
    import capstone

    md = getattr(img, "_md", None)
    if md is None:
        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        img._md = md
    o = va - img.base
    if o < 0 or o + 24 > len(img.mem):
        return None
    ins = list(md.disasm(img.mem[o : o + 24], va))[:2]
    if len(ins) != 2 or ins[1].mnemonic != "ret":
        return None
    i = ins[0]
    m = ACCESS.search(i.op_str)
    if not m:
        return None
    disp = int(m.group(1), 16)
    if i.mnemonic == "lea":
        return ("embedded", disp)
    if i.mnemonic in (
        "mov",
        "movzx",
        "movsx",
        "movsxd",
        "movss",
        "movsd",
    ) and i.op_str.split(",")[0].strip() in ("rax", "eax", "ax", "al", "xmm0"):
        return ("value", disp)
    return None


def candidate_fields(name):
    """Field names an accessor called `name` conventionally reads."""
    out = [name]
    for pre in ("Get", "Is", "Has", "Can"):
        if name.startswith(pre) and len(name) > len(pre):
            base = name[len(pre) :]
            out += [
                base,
                base + "Ptr",
                "_" + base[0].lower() + base[1:],
                base[0].lower() + base[1:],
            ]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="offsetcheck")
    ap.add_argument("--exe", required=True)
    ap.add_argument("--cs", required=True, help="FFXIVClientStructs/FFXIV source dir")
    ap.add_argument(
        "--results", required=True, help="sigscan scan output for the same CS commit"
    )
    ap.add_argument("--show", type=int, default=40)
    a = ap.parse_args(argv)

    img = sigscan.Image(a.exe)
    fields, sizes, inherits = source_fields(a.cs)
    res = json.load(open(a.results))

    # A VirtualFunction declared on an interface (UIModuleInterface.GetConfigModule) is reached
    # through the vtable of the struct that inherits it as PRIMARY base (UIModule), and the field it
    # returns is declared on that derived struct. Materialise those: one record per (derived, vfunc).
    vtbls = {
        (r["ns"], ".".join(r["hier"])): r["resolved"]
        for r in res
        if r["kind"] == "VirtualTable" and r.get("resolved")
    }
    derived_of = {}
    for key, bases in inherits.items():
        if bases:
            derived_of.setdefault(bases[0], []).append(
                key
            )  # primary base only: its slots are in the primary vtbl
    extra = []
    for r in res:
        if r["kind"] != "VirtualFunction":
            continue
        for dkey in derived_of.get(r["hier"][-1], []):
            vt = vtbls.get(dkey)
            if vt is None:
                continue
            try:
                fnaddr = img.u64(vt + 8 * r["offsets"][0])
            except Exception:
                continue
            extra.append(
                dict(
                    r,
                    ns=dkey[0],
                    hier=dkey[1].split("."),
                    resolved=fnaddr,
                    matches=[vt],
                    via="inherited",
                )
            )
    res = res + extra

    checked = agree = disagree = 0
    unmatched = 0
    rows, members = [], {}
    for r in res:
        if (
            r.get("resolved") is None
            or r["kind"] == "StaticAddress"
            or r["kind"] == "VirtualTable"
        ):
            continue
        if r["kind"] != "VirtualFunction" and len(r.get("matches") or []) != 1:
            continue
        shape = accessor_shape(img, r["resolved"])
        if not shape:
            continue
        kind, disp = shape
        key = (r["ns"], ".".join(r["hier"]))
        struct_fields = fields.get(key, {})
        # embedded members of a container: remember for adjacency-derived sizes
        if kind == "embedded":
            members.setdefault(key, []).append((disp, r["method"]))
        hit = None
        for cand in candidate_fields(r["method"]):
            if cand in struct_fields:
                hit = cand
                break
        cls, fn = sigscan.yml_name(r)
        if hit is None:
            unmatched += 1
            continue
        checked += 1
        declared = struct_fields[hit]
        if declared == disp:
            agree += 1
        else:
            disagree += 1
            rows.append((cls, fn, hit, declared, disp, kind))

    print(
        "accessors that are one addressing instruction + ret, with a same-named field to compare:"
    )
    print(
        "  verified  %5d   declared [FieldOffset] == displacement in the binary" % agree
    )
    print(
        "  MISMATCH  %5d   declared offset differs -- reads the wrong memory, silently"
        % disagree
    )
    print(
        "  no field  %5d   accessor found, no field of a matching name (nothing to compare)"
        % unmatched
    )
    for cls, fn, field, dec, disp, kind in rows[: a.show]:
        print(
            "    %-46s %-24s declared 0x%-6X binary 0x%-6X (%+d)"
            % ((cls + "." + fn)[-46:], field, dec, disp, disp - dec)
        )

    # sizes by adjacency, for every container whose members we saw through getters
    print(
        "\nstruct sizes bounded by member adjacency (container -> member, size <= next - this):"
    )
    n_ok = n_bad = 0
    for key, mem in members.items():
        # two getters can return the same member (GetRaptureAtkModule / GetRaptureAtkModule2);
        # adjacency is by DISTINCT displacement
        bydisp = {}
        for d, n in mem:
            bydisp.setdefault(d, n)
        mem = sorted(bydisp.items())
        for (d0, name0), (d1, _n1) in zip(mem, mem[1:]):
            # the member type is the accessor's return type; look up its declared size by field name
            fld = None
            for cand in candidate_fields(name0):
                if cand in fields.get(key, {}):
                    fld = cand
                    break
            if fld is None:
                continue
            # find the struct whose name matches the field's declared type: use the field name as a
            # proxy (CS names embedded members after their type in nearly every case)
            tkey = next(
                (
                    k
                    for k in sizes
                    if k[1].split(".")[-1] == fld
                    or k[1].split(".")[-1] == fld.rstrip("Ptr")
                ),
                None,
            )
            if tkey is None:
                continue
            bound = d1 - d0
            declared = sizes[tkey]
            if declared <= bound:
                n_ok += 1
                continue
            # A container may expose a NESTED member through its own getter (UIModule.GetAgentModule
            # returns a pointer inside RaptureAtkModule). If the member type declares a field at
            # exactly that inner offset, the "next member" is inside this one, not after it.
            inner = bound
            if inner in fields.get(tkey, {}).values():
                n_ok += 1
                continue
            n_bad += 1
            print(
                "    %-30s %-28s declared 0x%-6X but next member at +0x%X"
                % (key[1], fld, declared, bound)
            )
    print("  consistent %d   TOO LARGE %d" % (n_ok, n_bad))
    return 1 if disagree or n_bad else 0


if __name__ == "__main__":
    sys.exit(main())

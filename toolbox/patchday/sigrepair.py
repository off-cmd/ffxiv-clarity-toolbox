"""Re-derive a broken FFXIVClientStructs signature without IDA and without the old binary.

What breaks a signature. A pattern is a call site or a function prologue plus a few bytes of
context. A recompile can change the context -- a register allocation, a jump distance, an inlined
callee -- while the function itself survives. The old pattern then matches nowhere (or somewhere
else). The function is usually still there; the *description* of where to find it went stale.

Two capabilities, kept separate because they answer different questions:

  fuzzy   given the OLD pattern, find where it ALMOST matches in the new binary. Up to `k` fixed
          bytes may differ. Every candidate is resolved and ranked: fewest mismatches first, then
          whether the resolved target looks like a function start. This is a guess with a score, and
          it is presented as one.

  make    given a target address, produce a NEW pattern that matches exactly once. Operands that a
          recompile is likely to move (rel32 targets, RIP-relative displacements, large immediates)
          are wildcarded using capstone, so the pattern survives the next patch better than a raw
          byte dump would. Prefers a call-site pattern (E8 + context) when one is unique, because
          that is the form the runtime resolves cheapest and the form CS uses most.

Honesty about scope. `fuzzy` recovers the function only when the context drifted slightly. If the
call site was inlined away or the prologue reshaped, there is nothing to fuzz toward and the answer
is "not found", which is the correct answer -- a confident wrong address is worse than none. The
benchmark below measures exactly this: for each of the 12 signatures pohky replaced on 2026-09-08,
does the top candidate resolve to the address his replacement resolves to?

    uv run --with pefile --with capstone --with pyyaml python sigrepair.py fuzzy --exe EXE --sig "..." [--k 2]
    uv run --with pefile --with capstone --with pyyaml python sigrepair.py make  --exe EXE --addr 0x140B4E760
    uv run --with pefile --with capstone --with pyyaml python sigrepair.py bench --exe EXE --old sigs-old.json --data data.yml --names a,b,c
"""

import argparse
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sigscan  # noqa: E402

# ------------------------------------------------------------------------------------------------
# fuzzy matching
# ------------------------------------------------------------------------------------------------


def fixed_runs(b, m):
    """[(offset, bytes)] for every contiguous run of fixed bytes, longest first."""
    runs, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            runs.append((i, b[i:j]))
            i = j
        else:
            i += 1
    return sorted(runs, key=lambda r: -len(r[1]))


def fuzzy_find(text, sig, k=2, max_anchor_hits=200000):
    """Positions where `sig` matches with at most k fixed-byte mismatches.

    Pigeonhole: with k mismatches allowed, at least one of any k+1 fixed runs matches exactly, so
    each run in turn is used as an exact anchor and the whole pattern is verified around it. An
    anchor of one byte would hit millions of positions, so runs shorter than 2 are only used when
    nothing longer exists."""
    b, m = sigscan.parse_sig(sig)
    n = len(b)
    fixed_idx = [i for i in range(n) if m[i]]
    runs = fixed_runs(b, m)
    anchors = [r for r in runs if len(r[1]) >= 2] or runs[:1]
    seen = {}
    for k_off, run in anchors[: k + 3]:
        pos = text.find(run)
        hits = 0
        while pos != -1 and hits < max_anchor_hits:
            hits += 1
            start = pos - k_off
            if start >= 0 and start + n <= len(text) and start not in seen:
                seg = text[start : start + n]
                mism = [i for i in fixed_idx if seg[i] != b[i]]
                if len(mism) <= k:
                    seen[start] = mism
            pos = text.find(run, pos + 1)
    return sorted(seen.items(), key=lambda kv: (len(kv[1]), kv[0]))


def looks_like_function_start(img, va):
    """Cheap prologue heuristic. Not proof -- a ranking signal."""
    o = va - img.base
    if o < 16 or o + 16 > len(img.mem):
        return False
    before = img.mem[o - 4 : o]
    first = img.mem[o : o + 4]
    # Preceded by alignment padding or a ret, and not itself padding. That alone is a strong
    # function-start signal; insisting on particular prologue bytes rejected real entries such as
    # HandlePrepareZoningPacket, which begins `movzx r8d, dl` (44 0F B6 C2).
    padded = (
        before[-1] in (0xC3, 0xCC)
        or before[-2:] == b"\xc3\xcc"
        or all(x == 0xCC for x in before)
    )
    return padded and first[0] != 0xCC


def rank_candidates(img, rec, cands):
    """Resolve each fuzzy hit the way the runtime would, then score it."""
    offs = sigscan.effective_offsets(rec)
    out = []
    for start, mism in cands:
        try:
            target = sigscan.follow(img, start, offs)
        except Exception:
            continue
        in_text = img.section_of(target) == ".text"
        fstart = in_text and looks_like_function_start(img, target)
        score = (len(mism), 0 if fstart else 1, 0 if in_text else 1)
        out.append(
            dict(
                match=img.base + img.text_va + start,
                target=target,
                mismatches=mism,
                function_start=fstart,
                in_text=in_text,
                score=score,
            )
        )
    out.sort(key=lambda c: c["score"])
    return out


# ------------------------------------------------------------------------------------------------
# the order prior: MSVC keeps functions in the same relative order across a recompile
# ------------------------------------------------------------------------------------------------
#
# Measured 2026-09-08 on 1,955 functions that resolved uniquely in both clients: 3 order violations
# (0.15%). So the OLD data.yml -- which ships in the repo at the pinned commit, i.e. is on disk on
# patch day -- plus the old signatures that still resolve, bracket where a broken function must now
# be. For pohky's 12 the truth was inside the bracket 16/16 times, with widths from 96 B to 50 KB.
#
# This is the discriminator byte-fuzzing lacks: it knows WHICH function is wanted, from where its
# neighbours went.

import bisect


def build_anchors(old_classes, old_results):
    """[(old_addr, new_addr)] from old signatures that resolved to exactly one match, with the
    monotonicity violators dropped -- a unique-but-wrong resolution would otherwise poison its
    neighbourhood."""
    pairs = []
    for r in old_results:
        if (
            r["kind"] == "VirtualFunction"
            or len(r.get("matches") or []) != 1
            or r.get("resolved") is None
        ):
            continue
        cls, fn = sigscan.yml_name(r)
        oa = old_classes.get(cls, {}).get("funcs", {}).get(fn)
        if oa:
            pairs.append((oa, r["resolved"]))
    pairs.sort()
    # drop anchors that are out of order with BOTH neighbours (the local outliers)
    keep = []
    for i, (oa, na) in enumerate(pairs):
        prev_ok = i == 0 or pairs[i - 1][1] <= na
        next_ok = i == len(pairs) - 1 or na <= pairs[i + 1][1]
        if prev_ok or next_ok:
            keep.append((oa, na))
    return keep


def bracket(anchors, old_addr):
    """(lo, hi, interpolated) in NEW address space for a function at old_addr."""
    olds = [a for a, _ in anchors]
    i = bisect.bisect_left(olds, old_addr)
    lo = anchors[i - 1] if i > 0 else None
    hi = anchors[i] if i < len(anchors) else None
    if lo and hi and hi[0] != lo[0]:
        t = (old_addr - lo[0]) / (hi[0] - lo[0])
        interp = int(lo[1] + t * (hi[1] - lo[1]))
    else:
        interp = (lo or hi)[1]
    return (lo[1] if lo else None, hi[1] if hi else None, interp)


def function_starts_in(img, lo, hi):
    """Candidate function entry points inside [lo, hi]: positions after CC padding / ret that
    look like a prologue. Cheap and over-inclusive; the bracket keeps it small."""
    out = []
    o0, o1 = (
        max(lo - img.base, img.text_va),
        min(hi - img.base, img.text_va + img.text_size),
    )
    mem = img.mem
    for o in range(o0, o1):
        if (
            mem[o - 1] in (0xCC, 0xC3)
            and mem[o] != 0xCC
            and looks_like_function_start(img, img.base + o)
        ):
            out.append(img.base + o)
    return out


def rank_with_prior(img, rec, cands, anchors, old_addr):
    """Fuzzy candidates filtered and ordered by the bracket; if none survive, propose the function
    starts inside the bracket ordered by distance from the interpolated position."""
    lo, hi, interp = bracket(anchors, old_addr)
    ranked = rank_candidates(img, rec, cands)
    inside = [
        c
        for c in ranked
        if (lo is None or c["target"] >= lo) and (hi is None or c["target"] <= hi)
    ]
    if lo is None or hi is None:
        for c in inside:
            c["dist"] = abs(c["target"] - interp)
        inside.sort(key=lambda c: (len(c["mismatches"]), c["dist"]))
        return inside, (lo, hi, interp), "fuzzy, no bracket"
    # Union of fuzzy hits inside the bracket and every function start inside it, ranked by
    # distance from the interpolated position FIRST. With ~2,000 anchors the interpolation landed
    # within 11 bytes of the truth even inside a 50 KB bracket; a fuzzy hit with mismatches
    # 240 bytes away is weaker evidence than a clean entry point at the predicted address.
    seen = {c["target"] for c in inside}
    pool = list(inside)
    for t in function_starts_in(img, lo, hi):
        if t not in seen:
            pool.append(
                dict(
                    match=None,
                    target=t,
                    mismatches=None,
                    function_start=True,
                    in_text=True,
                )
            )
    for c in pool:
        c["dist"] = abs(c["target"] - interp)
        c["fuzzy"] = c["mismatches"] is not None
    pool.sort(
        key=lambda c: (c["dist"], 0 if c["fuzzy"] else 1, len(c["mismatches"] or []))
    )
    return pool, (lo, hi, interp), "bracket"


# ------------------------------------------------------------------------------------------------
# signature generation
# ------------------------------------------------------------------------------------------------


def _cs():
    import capstone

    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    md.detail = True
    return md


def wildcard_insn(insn):
    """Mask for one instruction: 1 = keep byte, 0 = wildcard.

    Wildcards the bytes a recompile moves without changing what the instruction *is*: rel8/rel32
    branch targets, RIP-relative displacements, and 32-bit immediates/displacements. Opcode, ModRM
    and register bytes are kept -- they are what make the pattern mean something."""
    import capstone

    size = insn.size
    keep = [1] * size
    try:
        e = insn.encoding  # CsX86Encoding: imm_offset/imm_size/disp_offset/disp_size
    except AttributeError:
        return keep
    if e.imm_offset and e.imm_size >= 4:
        for i in range(e.imm_offset, min(size, e.imm_offset + e.imm_size)):
            keep[i] = 0
    if e.disp_offset and e.disp_size >= 4:
        for i in range(e.disp_offset, min(size, e.disp_offset + e.disp_size)):
            keep[i] = 0
    # branches: wildcard the whole target regardless of width (rel8 too -- block layout moves)
    if insn.group(capstone.CS_GRP_JUMP) or insn.group(capstone.CS_GRP_CALL):
        if e.imm_offset:
            for i in range(e.imm_offset, min(size, e.imm_offset + e.imm_size)):
                keep[i] = 0
    return keep


def masked_bytes(img, va, length):
    """(bytes, mask) covering at least `length` bytes from va, ending on an instruction boundary."""
    md = _cs()
    o = va - img.base
    buf = img.mem[o : o + length + 16]
    out_b, out_m = bytearray(), bytearray()
    for insn in md.disasm(buf, va):
        keep = wildcard_insn(insn)
        out_b += insn.bytes
        out_m += bytes(keep)
        if len(out_b) >= length:
            break
    return bytes(out_b), bytes(out_m)


def fmt_sig(b, m):
    return " ".join("%02X" % x if mm else "??" for x, mm in zip(b, m))


def unique_prefix(img, b, m, min_len=8, max_len=48):
    """Shortest prefix of (b, m) that matches exactly once in .text, or None.

    Hit count is monotone non-increasing in prefix length, so this is a binary search over the
    length -- ~8 scans instead of ~150 for a 160-byte window."""
    if not m[0]:
        return (
            None  # a pattern starting with a wildcard is useless to the runtime scanner
        )
    hi = min(len(b), max_len)
    if hi < min_len:
        return None

    def hits(n):
        return len(sigscan.find_all(img.text, fmt_sig(b[:n], m[:n])))

    if hits(hi) != 1:
        return None  # even the full window is not unique (or matches nowhere)
    lo = min_len
    if hits(lo) == 1:
        return fmt_sig(b[:lo], m[:lo])
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if hits(mid) == 1:
            hi = mid
        else:
            lo = mid
    return fmt_sig(b[:hi], m[:hi])


def call_index(img):
    """target -> [call sites] for every E8 rel32 in .text, built once per image (~1.5M E8 bytes;
    a per-target scan was the whole benchmark's runtime)."""
    idx = getattr(img, "_call_index", None)
    if idx is not None:
        return idx
    text, base = img.text, img.base + img.text_va
    idx = {}
    pos = text.find(b"\xe8")
    n = len(text)
    while pos != -1:
        if pos + 5 <= n:
            rel = struct.unpack_from("<i", text, pos + 1)[0]
            idx.setdefault(base + pos + 5 + rel, []).append(base + pos)
        pos = text.find(b"\xe8", pos + 1)
    img._call_index = idx
    return idx


def callers_of(img, target):
    return call_index(img).get(target, [])


def make_sig(img, target, prefer="call"):
    """A new unique pattern for `target`. Tries call sites first (the CS house style), then the
    prologue. Returns (kind, sig, site) or None."""
    # Call sites first, and with a LONG window. Packet handlers are called from a dispatch switch
    # where every case is `call X; mov al,1; restore; ret` -- identical for 40+ bytes. Uniqueness
    # comes from spanning several cases and keeping the one-byte immediates and register moves
    # that differ between them (which masked_bytes preserves: only 4-byte operands are wildcarded).
    # pohky's replacement for HandleMapEffectPacket is exactly this shape, 100+ bytes long.
    best = None
    if prefer == "call":
        # A widely-called getter can have hundreds of call sites; trying every one at a 160-byte
        # window is minutes of scanning for no gain -- one unique site is all a signature needs.
        for site in callers_of(img, target)[:12]:
            b, m = masked_bytes(img, site, 160)
            sig = unique_prefix(img, b, m, min_len=8, max_len=160)
            if sig and (best is None or len(sig) < len(best[1])):
                best = ("call", sig, site)
        if best:
            return best
    b, m = masked_bytes(img, target, 96)
    sig = unique_prefix(img, b, m, min_len=10, max_len=96)
    if sig:
        return ("prologue", sig, target)
    return None


# ------------------------------------------------------------------------------------------------
# benchmark against pohky
# ------------------------------------------------------------------------------------------------


def bench(
    img, old_sigs, data_path, names, k, old_data_path=None, old_results_path=None
):
    _, classes = sigscan.load_data(data_path)
    anchors = old_classes = None
    if old_data_path and old_results_path:
        _, old_classes = sigscan.load_data(old_data_path)
        anchors = build_anchors(old_classes, json.load(open(old_results_path)))
    want = set(names)
    rows = []
    for rec in old_sigs:
        if rec["kind"] == "VirtualFunction":
            continue
        cls, fn = sigscan.yml_name(rec)
        if fn not in want:
            continue
        truth = classes.get(cls, {}).get("funcs", {}).get(fn)
        if truth is None:
            continue
        exact = sigscan.find_all(img.text, rec["sig"])
        cands = fuzzy_find(img.text, rec["sig"], k=k)
        how = "fuzzy"
        br = None
        old_addr = (
            old_classes.get(cls, {}).get("funcs", {}).get(fn) if old_classes else None
        )
        if anchors and old_addr:
            ranked, br, how = rank_with_prior(img, rec, cands, anchors, old_addr)
        else:
            ranked = rank_candidates(img, rec, cands)
        top = ranked[0] if ranked else None
        hit_rank = next((i for i, c in enumerate(ranked) if c["target"] == truth), None)
        newsig = make_sig(img, truth)
        newsig_ok = None
        if newsig:
            hits = sigscan.find_all(img.text, newsig[1])
            offs = [1] if newsig[0] == "call" else []
            newsig_ok = len(hits) == 1 and sigscan.follow(img, hits[0], offs) == truth
        rows.append(
            dict(
                name=cls + "." + fn,
                old=rec["sig"],
                exact=len(exact),
                cands=len(ranked),
                how=how,
                bracket=br,
                top=top,
                truth=truth,
                hit_rank=hit_rank,
                newsig=newsig,
                newsig_ok=newsig_ok,
            )
        )
    return rows


def print_bench(rows):
    ok = sum(1 for r in rows if r["top"] and r["top"]["target"] == r["truth"])
    anyrank = sum(1 for r in rows if r["hit_rank"] is not None)
    gen = sum(1 for r in rows if r["newsig_ok"])
    print(
        "fuzzy repair: top candidate correct %d/%d   correct answer somewhere in list %d/%d   new sig generated+verified %d/%d\n"
        % (ok, len(rows), anyrank, len(rows), gen, len(rows))
    )
    for r in rows:
        t = r["top"]
        verdict = (
            "TOP=RIGHT"
            if t and t["target"] == r["truth"]
            else "rank %d" % (r["hit_rank"] + 1)
            if r["hit_rank"] is not None
            else "NOT FOUND"
            if not t
            else "TOP=WRONG, truth absent"
        )
        print(
            "  %-52s exact x%d  cands %-3d  %-14s %-24s truth %X"
            % (r["name"][-52:], r["exact"], r["cands"], r["how"], verdict, r["truth"])
        )
        if r["bracket"] and r["bracket"][0] and r["bracket"][1]:
            print(
                "      bracket %X..%X (%d B), interpolated %X"
                % (
                    r["bracket"][0],
                    r["bracket"][1],
                    r["bracket"][1] - r["bracket"][0],
                    r["bracket"][2],
                )
            )
        if t:
            print(
                "      top: %X  mism@%s  fstart=%s"
                % (t["target"], t["mismatches"], t["function_start"])
            )
        if r["newsig"]:
            print(
                "      new %s sig: %s  %s"
                % (
                    r["newsig"][0],
                    r["newsig"][1][:70],
                    "VERIFIED" if r["newsig_ok"] else "unverified",
                )
            )


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sigrepair")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fuzzy")
    f.add_argument("--exe", required=True)
    f.add_argument("--sig", required=True)
    f.add_argument("--k", type=int, default=2)
    m = sub.add_parser("make")
    m.add_argument("--exe", required=True)
    m.add_argument("--addr", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--exe", required=True)
    b.add_argument("--old", required=True)
    b.add_argument("--data", required=True)
    b.add_argument("--names", required=True)
    b.add_argument("--k", type=int, default=2)
    b.add_argument(
        "--old-data", help="data.yml from the PINNED commit (on disk on patch day)"
    )
    b.add_argument("--old-results", help="scan of the old sigs against the new exe")
    a = ap.parse_args(argv)
    img = sigscan.Image(a.exe)
    if a.cmd == "fuzzy":
        rec = dict(kind="MemberFunction", sig=a.sig, offsets=[])
        for c in rank_candidates(img, rec, fuzzy_find(img.text, a.sig, a.k))[:10]:
            print(
                "  match %X -> %X  mism@%s  fstart=%s"
                % (c["match"], c["target"], c["mismatches"], c["function_start"])
            )
    elif a.cmd == "make":
        r = make_sig(img, int(a.addr, 16))
        print(r)
    elif a.cmd == "bench":
        print_bench(
            bench(
                img,
                json.load(open(a.old)),
                a.data,
                a.names.split(","),
                a.k,
                a.old_data,
                a.old_results,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

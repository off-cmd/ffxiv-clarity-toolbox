"""Drive a game patch through every subject it touches -- reusable for 7.56, 7.57, 7.58, hotfixes.

The version is ALWAYS an argument. Nothing here is specific to 7.56, because the expensive part of
patch night is not the work, it is remembering the order under time pressure at 4 AM, and that has to
be solved once rather than per patch.

What it answers, from `dependency-graph.toml` rather than from recall:

    plan 7.57            what rebuilds, in what order, split by deadline
    affects EXDSchema    if this moved, what is now stale (reverse reachability)
    start 7.57           open the ledger for a run and write its plan into it
    status 7.57          what the ledger says happened
    check                validate the graph -- cycles and dangling edges

THE ORDERING RISK THIS REMOVES

A stale schema does not announce itself. `GetExcelSheet<T>()` catches MismatchedColumnHashException
and returns **null** (Lumina#127), and RotationSolverReborn generates its job list from ClassJob, so
a stale schema yields a *smaller job list* rather than an error. Building in the wrong order
therefore produces something that runs and is quietly wrong -- the failure mode with the longest
detection time and the worst consequences at 5 AM.

THE THREE BRANCHES HAVE DIFFERENT CLOCKS

    code    HARD DEADLINE -- files appear ~3-4 AM, servers open ~5-6 AM
    asset   no deadline -- ~110 h of GPU, expected to run for days
    compat  reactive -- cannot begin until something is observed broken

`plan` prints them separately for that reason. Putting them in one queue invents urgency for two
thirds of the work and steals time from the third that actually has it.
"""

import argparse
import os
import re
import sys
from datetime import datetime

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    sys.exit("needs Python 3.11+ for tomllib (this machine pins 3.12 -- use `uv run`)")

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.environ.get("PATCH_GRAPH", os.path.join(HERE, "..", "dependency-graph.toml"))
LEDGER = os.environ.get("PATCH_LEDGER", os.path.join(HERE, "..", "per-patch"))

BRANCHES = [
    ("code", "HARD DEADLINE  files ~3-4 AM -> servers ~5-6 AM"),
    ("asset", "no deadline    ~110 h GPU, runs for days afterwards"),
    ("compat", "reactive       starts only when something is observed broken"),
]

VERSION_RE = re.compile(r"^(\d+)\.(\d+)(?:h(\d+))?$")


def parse_version(v):
    """7.56, 7.57, 7.58, 7.56h1, 7.56h2 -- one scheme, sortable, no special cases.

    Hotfixes matter as much as patches here: the lv-1-doe incident arrived in a hotfix, and a run
    that only knows about numbered patches would have no ledger entry to explain it afterwards.
    """
    m = VERSION_RE.match(v.strip().lower())
    if not m:
        raise ValueError("expected N.NN or N.NNhH (e.g. 7.56, 7.56h1), got %r" % v)
    major, minor, hot = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    return (major, minor, hot), "%d.%02dh%d" % (
        major,
        minor,
        hot,
    ) if hot else "%d.%02d" % (major, minor)


def load(path):
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    nodes = doc.get("nodes", {})
    for name, n in nodes.items():
        n.setdefault("depends_on", [])
        n.setdefault("branch", "")
        n.setdefault("kind", "?")
        n.setdefault("subject", "?")
    return nodes


def validate(nodes):
    """Dangling edges and cycles. A cycle would make the sort loop forever, and a dangling edge means
    the order is computed from an incomplete graph -- both must be loud."""
    problems = []
    for name, n in nodes.items():
        for d in n["depends_on"]:
            if d not in nodes:
                problems.append("%s depends on %r, which is not a node" % (name, d))

    colour, stack = {}, []

    def visit(u):
        colour[u] = 1
        stack.append(u)
        for v in nodes[u]["depends_on"]:
            if v not in nodes:
                continue
            if colour.get(v) == 1:
                problems.append("cycle: " + " -> ".join(stack[stack.index(v) :] + [v]))
            elif not colour.get(v):
                visit(v)
        stack.pop()
        colour[u] = 2

    for name in nodes:
        if not colour.get(name):
            visit(name)
    return problems


def topo(nodes, subset=None):
    """Dependencies first. Ties broken by name so two runs of the same graph give the same order --
    a plan that shuffles between runs cannot be checked off reliably."""
    keys = set(subset if subset is not None else nodes)
    out, seen = [], set()

    def visit(u):
        if u in seen or u not in keys:
            return
        seen.add(u)
        for v in sorted(nodes[u]["depends_on"]):
            visit(v)
        out.append(u)

    for name in sorted(keys):
        visit(name)
    return out


def downstream(nodes, start):
    """Everything reachable FROM start by following edges backwards -- i.e. everything that becomes
    stale when start changes. This is the blast radius."""
    rev = {name: [] for name in nodes}
    for name, n in nodes.items():
        for d in n["depends_on"]:
            if d in rev:
                rev[d].append(name)
    hit, queue = set(), [start]
    while queue:
        u = queue.pop()
        for v in rev.get(u, []):
            if v not in hit:
                hit.add(v)
                queue.append(v)
    return hit


def by_branch(nodes, names):
    order = topo(nodes, names)
    return [
        (b, desc, [n for n in order if nodes[n]["branch"] == b]) for b, desc in BRANCHES
    ]


def render_plan(nodes, version, affected):
    L = []
    L.append("# Patch %s — run plan" % version)
    L.append("")
    L.append(
        "*Generated %s by `patchrun.py`, from `dependency-graph.toml`.*"
        % datetime.now().strftime("%Y-%m-%d %H:%M")
    )
    L.append("")
    L.append(
        "%d nodes downstream of `game-files`. Order within each branch is a topological sort:"
        % len(affected)
    )
    L.append(
        "dependencies always precede dependents, so following it top to bottom cannot build"
    )
    L.append("something against a stale input.")
    L.append("")
    for b, desc, names in by_branch(nodes, affected):
        if not names:
            continue
        L.append("## %s branch — %s" % (b, desc))
        L.append("")
        for i, n in enumerate(names, 1):
            node = nodes[n]
            L.append("%2d. [ ] **%s** · `%s`" % (i, n, node["subject"]))
            if node["depends_on"]:
                L.append("       after: %s" % ", ".join(sorted(node["depends_on"])))
            if node.get("staleness"):
                L.append("       check: %s" % node["staleness"])
            if node.get("note"):
                L.append("       note: %s" % node["note"])
        L.append("")
    return "\n".join(L)


def cmd_check(a, nodes):
    problems = validate(nodes)
    print(
        "%d nodes, %d edges"
        % (len(nodes), sum(len(n["depends_on"]) for n in nodes.values()))
    )
    counts = {}
    for n in nodes.values():
        counts[n["branch"] or "(source)"] = counts.get(n["branch"] or "(source)", 0) + 1
    for k in sorted(counts):
        print("  %-10s %d" % (k, counts[k]))
    missing = [n for n, v in nodes.items() if v["branch"] and not v.get("staleness")]
    if missing:
        print("\nno staleness check defined (breakage here is found by accident):")
        for n in sorted(missing):
            print("   %-24s %s" % (n, nodes[n]["subject"]))
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("   %s" % p)
        return 1
    print("\ngraph OK — acyclic, no dangling edges")
    return 0


def cmd_plan(a, nodes):
    _, version = parse_version(a.version)
    if validate(nodes):
        print("graph has problems; run `check` first")
        return 1
    affected = downstream(nodes, a.root_node)
    print(render_plan(nodes, version, affected))
    return 0


def cmd_affects(a, nodes):
    if a.node not in nodes:
        print("no node %r. Known: %s" % (a.node, ", ".join(sorted(nodes))))
        return 1
    hit = downstream(nodes, a.node)
    print("%s changed -> %d stale\n" % (a.node, len(hit)))
    for b, desc, names in by_branch(nodes, hit):
        if names:
            print("  %-7s %s" % (b, ", ".join(names)))
    if not hit:
        print("  (nothing depends on it)")
    return 0


def cmd_start(a, nodes):
    _, version = parse_version(a.version)
    if validate(nodes):
        print("graph has problems; run `check` first")
        return 1
    root = os.path.join(a.ledger, version)
    for sub in ("code-branch", "asset-branch", "compat-branch", "forks-pushed"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)

    plan = os.path.join(root, "00-plan.md")
    if os.path.exists(plan) and not a.force:
        print(
            "%s already exists — refusing to overwrite a run in progress (use --force)"
            % plan
        )
        return 1
    with open(plan, "w", encoding="utf-8") as f:
        f.write(render_plan(nodes, version, downstream(nodes, a.root_node)))

    readme = os.path.join(root, "README.md")
    if not os.path.exists(readme):
        with open(readme, "w", encoding="utf-8") as f:
            f.write(
                """# Patch %s

Opened %s.

| branch | where | state |
|---|---|---|
| code | `code-branch\\` | not started |
| asset | `asset-branch\\` | not started |
| compat | `compat-branch\\` | not started |

`00-plan.md` holds the generated order. Record here what actually happened — especially anything
that differed from the plan, because that is what changes the plan for %s and after.

**Before any ephemeral branch is abandoned, push it to `upstream-mirrors\\local\\`.** The work is
throwaway; the record of what the files looked like the night this patch dropped is not, and it
cannot be reconstructed later.
"""
                % (version, datetime.now().strftime("%Y-%m-%d %H:%M"), "the next patch")
            )
    print("ledger opened: %s" % root)
    for f in sorted(os.listdir(root)):
        print("   %s" % f)
    return 0


def cmd_status(a, nodes):
    if not os.path.isdir(a.ledger):
        print("no ledger at %s" % a.ledger)
        return 0
    runs = []
    for name in os.listdir(a.ledger):
        try:
            runs.append((parse_version(name)[0], name))
        except ValueError:
            continue
    if not runs:
        print("no runs recorded in %s" % a.ledger)
        return 0
    print("runs in %s:\n" % a.ledger)
    for _, name in sorted(runs):
        root = os.path.join(a.ledger, name)
        bits = []
        for b in ("code", "asset", "compat"):
            d = os.path.join(root, "%s-branch" % b)
            n = len(os.listdir(d)) if os.path.isdir(d) else 0
            bits.append("%s:%d" % (b, n))
        forks = os.path.join(root, "forks-pushed")
        nf = len(os.listdir(forks)) if os.path.isdir(forks) else 0
        print("  %-10s %s  forks:%d" % (name, "  ".join(bits), nf))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="patchrun", description=__doc__.split("\n")[0])
    ap.add_argument("--graph", default=GRAPH)
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument(
        "--root-node",
        default="game-files",
        help="what changed (default game-files, i.e. a whole patch)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("plan")
    p.add_argument("version")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("affects")
    p.add_argument("node")
    p.set_defaults(fn=cmd_affects)
    p = sub.add_parser("start")
    p.add_argument("version")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_start)
    p = sub.add_parser("status")
    p.set_defaults(fn=cmd_status)
    a = ap.parse_args(argv)

    if not os.path.isfile(a.graph):
        print("no graph at %s" % a.graph)
        return 2
    try:
        return a.fn(a, load(a.graph))
    except ValueError as e:
        print("%s" % e)
        return 2


if __name__ == "__main__":
    sys.exit(main())

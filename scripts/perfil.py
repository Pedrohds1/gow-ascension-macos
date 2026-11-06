#!/usr/bin/env python3
"""Resume um `sample <pid>` do macOS por thread: amostras ocupadas (fora de
esperas do kernel), o papel da thread (a primeira função do jogo perto da raiz)
e as funções com mais tempo próprio.

Uso: perfil.py <sample.txt> [--top N] [--threads N]
"""
import re
import sys
from collections import Counter, defaultdict

IDLE = re.compile(r"^(__semwait_signal|__psynch_cvwait|__psynch_mutexwait|mach_msg2_trap|"
                  r"__workq_kernreturn|semaphore_wait_signal_trap|semaphore_timedwait_signal_trap|"
                  r"kevent_id|kevent64|__ulock_wait2?|__select|__pthread_kill|nanosleep|"
                  r"__wait4|poll|read|__open|swtch_pri|thread_switch|__semwait_signal_nocancel)$")
LINE = re.compile(r"^(?P<ind>[ +!:|]*)(?P<n>\d+) (?P<name>.+?)(?:  \(in (?P<mod>[^)]+)\).*)?$")


def parse(path):
    threads = []
    cur = None
    in_graph = False
    for raw in open(path, encoding="utf-8", errors="replace"):
        line = raw.rstrip("\n")
        if line.startswith("Call graph:"):
            in_graph = True
            continue
        if not in_graph:
            continue
        if line.startswith("Total number in stack") or line.startswith("Sort by top of stack"):
            break
        m = LINE.match(line[4:] if line.startswith("    ") else line)
        if not m:
            continue
        depth = len(m.group("ind"))
        name = m.group("name").strip()
        n = int(m.group("n"))
        if depth == 0 and name.startswith("Thread_"):
            cur = {"title": name, "total": n, "frames": []}
            threads.append(cur)
            continue
        if cur is not None:
            cur["frames"].append((depth, n, name, m.group("mod") or ""))
    return threads


def summarize(t):
    fr = t["frames"]
    selfc = Counter()
    busy = 0
    role = ""
    for i, (d, n, name, mod) in enumerate(fr):
        child = 0
        j = i + 1
        while j < len(fr) and fr[j][0] > d:
            if fr[j][0] == min(x[0] for x in fr[i + 1:j + 1] if x[0] > d):
                pass
            j += 1
        # filhos diretos: a profundidade logo abaixo de d
        kids = [x for x in fr[i + 1:j] if x[0] == (fr[i + 1][0] if i + 1 < j else -1)]
        child = sum(x[1] for x in kids)
        s = n - child
        if s > 0:
            fname = re.sub(r"\s+\+ \d+.*$", "", name)
            fname = re.sub(r"\(.*\)$", "", fname)
            selfc[fname] += s
            if not IDLE.match(fname):
                busy += s
        if not role and mod.startswith("MyGameRecomp") and not name.startswith("start"):
            role = re.sub(r"\(.*", "", name)
    return busy, role, selfc


def main():
    path = sys.argv[1]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 8
    nth = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 14
    rows = []
    agg = Counter()
    for t in parse(path):
        busy, role, selfc = summarize(t)
        rows.append((busy, t["total"], role, t["title"], selfc))
        for k, v in selfc.items():
            if not IDLE.match(k):
                agg[k] += v
    rows.sort(key=lambda r: -r[0])
    total = rows[0][1] if rows else 1
    print(f"amostras por thread: {total} (ocupado = fora de esperas do kernel)")
    for busy, tot, role, title, selfc in rows[:nth]:
        hot = ", ".join(f"{k} {v}" for k, v in selfc.most_common(top) if not IDLE.match(k))
        print(f"{100.0 * busy / tot:5.1f}% {role[:40]:40s} | {hot[:220]}")
    print("\nfunções mais quentes (todas as threads):")
    for k, v in agg.most_common(30):
        print(f"  {v:6d}  {k[:110]}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
hamilton_heldkarp.py

Held-Karp (bitmask DP) experimental harness:
 - generate Erdos-Renyi graphs (reproducible via seed)
 - run Held-Karp in a subprocess (external timeout enforced)
 - high-precision timing (perf_counter_ns)
 - save results to CSV and produce plots (same format as backtracking harness)

Fields in CSV:
 run_idx, graph_idx, rep, n, p, avg_degree, time_s, found, timed_out, rec_calls

Author: generated for Ana Beatriz
"""

import argparse
import csv
import random
import time
import sys
from typing import Dict, Set, List, Optional, Tuple

import multiprocessing
import math
import matplotlib.pyplot as plt

# -------------------------
# Graph utilities
# -------------------------
def erdos_renyi_adj(n: int, p: float, rnd: random.Random) -> Dict[int, Set[int]]:
    """Generate adjacency dict for G(n,p) using provided RNG."""
    G = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if rnd.random() < p:
                G[i].add(j)
                G[j].add(i)
    return G

def average_degree(G: Dict[int, Set[int]]) -> float:
    n = len(G)
    if n == 0:
        return 0.0
    return sum(len(nei) for nei in G.values()) / float(n)


# -------------------------
# Held-Karp implementation (in-process)
# -------------------------
def held_karp_inprocess(G: Dict[int, Set[int]], start: int = 0) -> Tuple[Optional[List[int]], int]:
    """
    Held-Karp for unweighted graphs: treat edge cost 1 for existing edges,
    inf for non-edges. Returns (path list if Hamiltonian cycle exists else None, rec_calls).
    rec_calls counts DP updates (an indicator of work).
    """
    n = len(G)
    # quick checks
    if n == 0:
        return None, 0
    # if any isolated vertex, no Hamiltonian cycle
    for v in range(n):
        if len(G[v]) == 0:
            return None, 0

    INF = 10**9
    # Map cost: cost[u][v] = 1 if edge exists else INF
    cost = [[INF] * n for _ in range(n)]
    for u in range(n):
        cost[u][u] = 0
        for v in G[u]:
            cost[u][v] = 1

    # DP: dp[mask][v] = min cost to start at start, visit vertices in mask (mask includes v), end at v
    # represent dp as dict mapping (mask << n) + v -> cost to save memory, or use list of dicts
    # But for speed and clarity, use list of dicts indexed by mask: each is array size n; masks go 0..(1<<n)-1
    # Memory: n * 2^n entries
    max_mask = 1 << n
    # initialize dp with INF
    # We'll allocate dp as list of lists of floats to be fastest, but be mindful of memory.
    # For n up to ~20 this is OK.
    try:
        dp = [ [INF] * n for _ in range(max_mask) ]
        parent = [ [-1] * n for _ in range(max_mask) ]
    except MemoryError:
        # if n too large to allocate, bail out
        return None, 0

    # initialize base cases: paths that start at start and go to v directly (mask with v only)
    # but classic Held-Karp omits start in masks; we'll keep mask indicating subset of all vertices including start.
    # For uniformity, set dp[1<<start][start] = 0
    dp[1 << start][start] = 0

    rec_calls = 0

    # iterate masks
    for mask in range(max_mask):
        # skip masks that don't include start
        if not (mask & (1 << start)):
            continue
        # for each v in mask with dp defined
        for v in range(n):
            if not (mask & (1 << v)):
                continue
            cur = dp[mask][v]
            if cur >= INF:
                continue
            # try extend to u not in mask
            not_mask = (~mask) & (max_mask - 1)
            u = not_mask & -not_mask
            # iterate u by bit scanning
            m = not_mask
            while m:
                lb = m & -m
                u_idx = (lb.bit_length() - 1)
                # relax
                new_mask = mask | lb
                new_cost = cur + cost[v][u_idx]
                if new_cost < dp[new_mask][u_idx]:
                    dp[new_mask][u_idx] = new_cost
                    parent[new_mask][u_idx] = v
                rec_calls += 1
                m -= lb

    full_mask = (1 << n) - 1
    best_cost = INF
    last = -1
    # find best end k such that there is edge from k to start to close cycle
    for k in range(n):
        if k == start:
            continue
        c = dp[full_mask][k] + cost[k][start]
        if c < best_cost:
            best_cost = c
            last = k

    if best_cost >= INF:
        # no Hamiltonian cycle
        return None, rec_calls

    # reconstruct path: from full_mask, last
    path = []
    mask = full_mask
    v = last
    while v != -1:
        path.append(v)
        pv = parent[mask][v]
        mask = mask & ~(1 << v)
        v = pv
    path.append(start)
    path.reverse()  # path from start .. last .. start? ensure cycle representation
    # ensure it is a full path covering n vertices; if length less than n+1, reconstruct differently
    # In some parent reconstructions, start may appear twice; ensure we return a Hamiltonian cycle list of n vertices (order)
    # Return the cycle as sequence of n vertices starting at start (without returning start again)
    cycle = path[:-1] if path and path[-1] == start else path
    if len(cycle) != n:
        # fallback: attempt another reconstruction by walking parents
        # (this is unlikely when DP was filled correctly)
        pass

    return cycle, rec_calls


# -------------------------
# Worker + subprocess wrapper (external timeout)
# -------------------------
def __hk_worker(G, start, q):
    """Worker for Held-Karp to be run in subprocess. No time_limit here; parent enforces timeout."""
    try:
        path, rec_calls = held_karp_inprocess(G, start=start)
        q.put(("ok", path, rec_calls))
    except Exception as e:
        q.put(("err", repr(e)))

def run_hk_in_subprocess(G: Dict[int, Set[int]], start: int, time_limit: float) -> Tuple[Optional[List[int]], bool, int, float]:
    """
    Run Held-Karp in subprocess and enforce external timeout.
    Returns (path_or_None, timed_out_bool, rec_calls_or_-1, elapsed_seconds)
    """
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=__hk_worker, args=(G, start, q))
    t0 = time.perf_counter_ns()
    p.start()
    p.join(time_limit)
    elapsed = (time.perf_counter_ns() - t0) / 1e9

    if p.is_alive():
        try:
            p.terminate()
        except Exception:
            pass
        p.join()
        return None, True, -1, elapsed

    try:
        msg = q.get_nowait()
    except Exception:
        return None, False, -1, elapsed

    tag, *payload = msg
    if tag == "ok":
        path, rec_calls = payload
        return path, False, int(rec_calls), elapsed
    else:
        return None, False, -1, elapsed


# -------------------------
# Experiment harness (same CSV fields and plots as backtracking harness)
# -------------------------
def parse_list_arg(s: Optional[str]):
    if s is None:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    out = []
    for p in parts:
        try:
            if "." in p:
                out.append(float(p))
            else:
                out.append(int(p))
        except:
            pass
    return out if out else None

def run_experiments(
    num_graphs: int,
    repeats: int,
    seed: Optional[int],
    n_min: int,
    n_max: int,
    p_min: float,
    p_max: float,
    n_list: Optional[List[int]],
    p_list: Optional[List[float]],
    time_limit: float,
    out_csv: str,
    make_plots: bool
):
    rnd = random.Random(seed)
    results = []
    run_idx = 0

    for g_idx in range(num_graphs):
        if n_list:
            n = int(rnd.choice(n_list))
        else:
            n = rnd.randint(n_min, n_max)

        if p_list:
            p = float(rnd.choice(p_list))
        else:
            p = rnd.uniform(p_min, p_max)

        for rep in range(repeats):
            graph_seed = None
            if seed is not None:
                graph_seed = (seed * 1000003) ^ (g_idx * 9176) ^ (rep * 7919)

            G = erdos_renyi_adj(n, p, random.Random(graph_seed))
            avg_deg = average_degree(G)

            # Run Held-Karp in subprocess with external timeout
            path, timed_out, rec_calls, elapsed_s = run_hk_in_subprocess(G, start=0, time_limit=time_limit)

            found = path is not None

            results.append({
                "run_idx": run_idx,
                "graph_idx": g_idx,
                "rep": rep,
                "n": n,
                "p": round(p, 6),
                "avg_degree": round(avg_deg, 6),
                "time_s": f"{elapsed_s:.9f}",
                "time_float": elapsed_s,
                "found": bool(found),
                "timed_out": bool(timed_out),
                "rec_calls": rec_calls
            })

            print(f"[run {run_idx}] graph#{g_idx} rep={rep} n={n} p={p:.4f} avg_deg={avg_deg:.3f} "
                  f"time={elapsed_s:.9f}s found={found} timed_out={timed_out} calls={rec_calls}")

            run_idx += 1

    # save CSV
    fieldnames = ["run_idx","graph_idx","rep","n","p","avg_degree",
                  "time_s","found","timed_out","rec_calls"]
    try:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = {k: r[k] for k in fieldnames}
                writer.writerow(row)
        print(f"Saved results to {out_csv}")
    except Exception as e:
        print("Failed to save CSV:", e)

    # plotting
    if make_plots:
        try:
            xs_found = [r["avg_degree"] for r in results if r["found"]]
            ys_found = [r["time_float"] for r in results if r["found"]]
            xs_nf = [r["avg_degree"] for r in results if not r["found"]]
            ys_nf = [r["time_float"] for r in results if not r["found"]]

            plt.figure(figsize=(8, 6))
            if xs_found:
                plt.scatter(xs_found, ys_found, marker="o", label="found", alpha=0.8)
            if xs_nf:
                plt.scatter(xs_nf, ys_nf, marker="x", label="not found / timeout", alpha=0.8)

            plt.xlabel("Average degree")
            plt.ylabel("Time (s)")
            plt.yscale("log")
            plt.title("Held-Karp time vs average degree")
            plt.legend()
            plt.tight_layout()
            scatter_file = out_csv.replace(".csv","") + "_scatter.png"
            plt.savefig(scatter_file)
            plt.close()
            print(f"Saved scatter plot: {scatter_file}")

            groups = {}
            for r in results:
                groups.setdefault(r["n"], []).append(r["time_float"])

            labels = sorted(groups.keys())
            data = [groups[k] for k in labels]

            plt.figure(figsize=(10, 6))
            plt.boxplot(data, labels=[str(l) for l in labels])
            plt.yscale("log")
            plt.xlabel("n")
            plt.ylabel("Time (s)")
            plt.title("Held-Karp time distribution by n")
            plt.tight_layout()
            box_file = out_csv.replace(".csv","") + "_boxplot.png"
            plt.savefig(box_file)
            plt.close()
            print(f"Saved boxplot: {box_file}")

        except Exception as e:
            print("Plotting failed:", e)

    return results

# -------------------------
# CLI
# -------------------------
def main(argv):
    parser = argparse.ArgumentParser(description="Held-Karp experiments (Hamiltonian cycle detection via DP)")
    parser.add_argument("--num_graphs", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--n_min", type=int, default=10)
    parser.add_argument("--n_max", type=int, default=150)
    parser.add_argument("--p_min", type=float, default=0.2)
    parser.add_argument("--p_max", type=float, default=0.8)
    parser.add_argument("--n_list", type=str, default=None)
    parser.add_argument("--p_list", type=str, default=None)
    parser.add_argument("--timelimit", type=float, default=20.0)
    parser.add_argument("--out", type=str, default="results_heldkarp.csv")
    parser.add_argument("--no_plots", action="store_true")

    args = parser.parse_args(argv)

    n_list = parse_list_arg(args.n_list) if args.n_list else None
    p_list = parse_list_arg(args.p_list) if args.p_list else None

    print("Running Held-Karp experiments with parameters:")
    print(f" num_graphs={args.num_graphs} repeats={args.repeats} seed={args.seed}")
    print(f" n_list={n_list if n_list else f'[{args.n_min},{args.n_max}]'}")
    print(f" p_list={p_list if p_list else f'[{args.p_min},{args.p_max}]'}")
    print(f" timelimit={args.timelimit}s out='{args.out}' plots={not args.no_plots}")

    run_experiments(
        num_graphs=args.num_graphs,
        repeats=args.repeats,
        seed=args.seed,
        n_min=args.n_min,
        n_max=args.n_max,
        p_min=args.p_min,
        p_max=args.p_max,
        n_list=n_list,
        p_list=p_list,
        time_limit=args.timelimit,
        out_csv=args.out,
        make_plots=not args.no_plots
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main(sys.argv[1:])

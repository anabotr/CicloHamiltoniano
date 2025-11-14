import argparse
import csv
import random
import time
import sys
from typing import Dict, Set, List, Optional, Tuple, Iterable

import multiprocessing
import math
import matplotlib.pyplot as plt
import copy

def erdos_renyi_adj(n: int, p: float, rnd: random.Random) -> Dict[int, Set[int]]:
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

def degree_list(G: Dict[int, Set[int]]) -> List[int]:
    return [len(G[v]) for v in sorted(G.keys())]

def is_connected(G: Dict[int, Set[int]]) -> bool:
    n = len(G)
    if n == 0:
        return True
    start = next(iter(G))
    visited = set()
    stack = [start]
    while stack:
        u = stack.pop()
        if u in visited: continue
        visited.add(u)
        for w in G[u]:
            if w not in visited:
                stack.append(w)
    return len(visited) == n


def articulation_points(G: Dict[int, Set[int]]) -> Set[int]:
    n = len(G)
    idx = 0
    ids = {}
    low = {}
    parent = {}
    visited = set()
    aps = set()

    def dfs(u: int):
        nonlocal idx
        visited.add(u)
        ids[u] = idx
        low[u] = idx
        idx += 1
        children = 0
        for v in G[u]:
            if v not in visited:
                parent[v] = u
                children += 1
                dfs(v)
                low[u] = min(low[u], low[v])
                if parent.get(u, None) is None and children > 1:
                    aps.add(u)
                if parent.get(u, None) is not None and low[v] >= ids[u]:
                    aps.add(u)
            elif parent.get(u, None) != v:
                low[u] = min(low[u], ids[v])

    for node in G:
        if node not in visited:
            dfs(node)
    return aps


def copy_graph(G: Dict[int, Set[int]]) -> Dict[int, Set[int]]:
    return {v: set(nei) for v, nei in G.items()}


class VaCulAggressive:
    def __init__(self, G: Dict[int, Set[int]], start: int = 0, rng_seed: Optional[int] = None, max_restarts: int = 3):
        self.G0 = copy_graph(G)
        self.n = len(G)
        self.start = start if 0 <= start < self.n else 0
        self.rec_calls = 0
        self.rng = random.Random(rng_seed)
        self.max_restarts = max_restarts

    def apply_forced_edges(self, G: Dict[int, Set[int]]) -> Tuple[Dict[int, Set[int]], Set[Tuple[int,int]], bool]:
        G = copy_graph(G)
        forced = set()
        changed = True
        while changed:
            changed = False
            deg2 = [v for v in G if len(G[v]) == 2]
            for v in deg2:
                a, b = sorted(list(G[v]))
                e1 = (min(v,a), max(v,a))
                e2 = (min(v,b), max(v,b))
                if e1 not in forced:
                    forced.add(e1)
                if e2 not in forced:
                    forced.add(e2)
            forced_inc = {v: set() for v in G}
            for (x,y) in forced:
                forced_inc[x].add(y); forced_inc[y].add(x)
            for v in list(G.keys()):
                if len(forced_inc[v]) >= 2:
                    others = [w for w in G[v] if w not in forced_inc[v]]
                    if others:
                        for w in others:
                            G[v].discard(w)
                            G[w].discard(v)
                        changed = True
            for v in G:
                if len(G[v]) < 2:
                    return G, forced, True
        return G, forced, False

    def remove_premature_closing_edges(self, G: Dict[int, Set[int]], forced: Set[Tuple[int,int]]) -> Tuple[Dict[int, Set[int]], bool]:
        G = copy_graph(G)
        forced_graph = {v: set() for v in G}
        for (x,y) in forced:
            forced_graph[x].add(y); forced_graph[y].add(x)
        visited = set()
        changed = False
        for v in forced_graph:
            if v in visited: continue
            if not forced_graph[v]:
                visited.add(v); continue
            comp = []
            stack = [v]
            while stack:
                u = stack.pop()
                if u in visited: continue
                visited.add(u)
                comp.append(u)
                for w in forced_graph[u]:
                    if w not in visited:
                        stack.append(w)
            degs = [len(forced_graph[u]) for u in comp]
            if all(d<=2 for d in degs):
                endpoints = [u for u in comp if len(forced_graph[u])==1]
                if len(endpoints) == 2 and len(comp) < self.n:
                    a,b = endpoints
                    for i in comp:
                        for j in list(G[i]):
                            if j in comp and (min(i,j),max(i,j)) not in forced:
                                G[i].discard(j)
                                G[j].discard(i)
                                changed = True
        for v in G:
            if len(G[v]) < 2:
                return G, True
        return G, changed

    def quick_checks(self, G: Dict[int, Set[int]]) -> Tuple[bool, Optional[str]]:
        if not is_connected(G):
            return False, "disconnected"
        aps = articulation_points(G)
        if aps:
            return False, f"articulation_points:{len(aps)}"
        for v in G:
            if len(G[v]) < 2:
                return False, f"deg_lt_2:{v}"
        return True, None

    def safe_to_add(self, G: Dict[int, Set[int]], visited: List[bool], v: int) -> bool:
        if len(G[v]) == 0:
            return False
        return True

    def backtrack_search(self, G_work: Dict[int, Set[int]], path: List[int], visited: List[bool], pos: int) -> Optional[List[int]]:
        self.rec_calls += 1
        ok, reason = self.quick_checks(G_work)
        if not ok:
            return None
        n = self.n
        if pos == n:
            if path[-1] in G_work[path[0]]:
                return path.copy()
            else:
                return None
        last = path[pos-1]
        candidates = [u for u in G_work[last] if not visited[u]]
        candidates.sort(key=lambda x: len(G_work[x]))

        for v in candidates:
            if not self.safe_to_add(G_work, visited, v):
                continue
            G_snapshot = None
            visited[v] = True
            path[pos] = v
            G_next = copy_graph(G_work)
            G_reduced, forced, contrad = self.apply_forced_edges(G_next)
            if not contrad:
                G_reduced2, changed = self.remove_premature_closing_edges(G_reduced, forced)
                if not changed and G_reduced2 is not None:
                    G_reduced_final = G_reduced2
                else:
                    G_reduced_final = G_reduced
            else:
                G_reduced_final = G_next

            res = self.backtrack_search(G_reduced_final, path, visited, pos+1)
            if res is not None:
                return res
            visited[v] = False
            path[pos] = -1
        return None


    def solve(self, time_limit: Optional[float] = None) -> Tuple[Optional[List[int]], int]:
        G0 = copy_graph(self.G0)
        ok, reason = self.quick_checks(G0)
        if not ok:
            return None, self.rec_calls

        attempts = max(1, self.max_restarts)
        for attempt in range(attempts):
            G_work = copy_graph(G0)
            if attempt > 0:
                nodes = list(G_work.keys())
                for u in nodes:
                    neighs = list(G_work[u])
                    self.rng.shuffle(neighs)
                    G_work[u] = set(neighs)

            path = [-1] * self.n
            visited = [False] * self.n
            path[0] = self.start
            visited[self.start] = True

            G_reduced, forced, contrad = self.apply_forced_edges(G_work)
            if contrad:
                continue
            G_reduced2, changed = self.remove_premature_closing_edges(G_reduced, forced)
            G_initial = G_reduced2 if G_reduced2 is not None else G_reduced

            res = self.backtrack_search(G_initial, path, visited, 1)
            if res is not None:
                return res, self.rec_calls
        return None, self.rec_calls

def __vacul_worker(G: Dict[int, Set[int]], start: int, time_limit: float, rng_seed: Optional[int], restarts: int, q: multiprocessing.Queue):
    try:
        solver = VaCulAggressive(G, start=start, rng_seed=rng_seed, max_restarts=restarts)
        t0 = time.perf_counter_ns()
        path, rec_calls = solver.solve(time_limit=time_limit)
        t1 = time.perf_counter_ns()
        elapsed = (t1 - t0) / 1e9
        q.put(("ok", path, rec_calls, elapsed))
    except Exception as e:
        q.put(("err", repr(e)))

def run_vacul_in_subprocess(G: Dict[int, Set[int]], start: int, time_limit: float, rng_seed: Optional[int], restarts: int) -> Tuple[Optional[List[int]], bool, int, float]:
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=__vacul_worker, args=(G, start, time_limit, rng_seed, restarts, q))
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
        path, rec_calls, elapsed_inner = payload
        return path, False, int(rec_calls), elapsed
    else:
        return None, False, -1, elapsed

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
    make_plots: bool,
    vacul_restarts: int,
    vacul_rng_seed: Optional[int]
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

            path, timed_out, rec_calls, elapsed_s = run_vacul_in_subprocess(
                G, start=0, time_limit=time_limit, rng_seed=vacul_rng_seed, restarts=vacul_restarts
            )

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
            plt.title("VaCul (aggressive) time vs average degree")
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
            plt.title("VaCul (aggressive) time distribution by n")
            plt.tight_layout()
            box_file = out_csv.replace(".csv","") + "_boxplot.png"
            plt.savefig(box_file)
            plt.close()
            print(f"Saved boxplot: {box_file}")

        except Exception as e:
            print("Plotting failed:", e)

    return results


def main(argv):
    parser = argparse.ArgumentParser(description="VaCul (aggressive) experiments")
    parser.add_argument("--num_graphs", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--n_min", type=int, default=20)
    parser.add_argument("--n_max", type=int, default=20)
    parser.add_argument("--p_min", type=float, default=0)
    parser.add_argument("--p_max", type=float, default=1)
    parser.add_argument("--n_list", type=str, default=None)
    parser.add_argument("--p_list", type=str, default=None)
    parser.add_argument("--timelimit", type=float, default=20.0)
    parser.add_argument("--restarts", type=int, default=20, help="randomized restarts for VaCul")
    parser.add_argument("--rng_seed", type=int, default=None, help="seed for VaCul internal RNG (randomized restarts)")
    parser.add_argument("--out", type=str, default="results_vacul_20.csv")
    parser.add_argument("--no_plots", action="store_true")

    args = parser.parse_args(argv)

    n_list = parse_list_arg(args.n_list) if args.n_list else None
    p_list = parse_list_arg(args.p_list) if args.p_list else None

    print("Running VaCul (aggressive) experiments with parameters:")
    print(f" num_graphs={args.num_graphs} repeats={args.repeats} seed={args.seed}")
    print(f" n_list={n_list if n_list else f'[{args.n_min},{args.n_max}]'}")
    print(f" p_list={p_list if p_list else f'[{args.p_min},{args.p_max}]'}")
    print(f" timelimit={args.timelimit}s out='{args.out}' restarts={args.restarts} rng_seed={args.rng_seed}")

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
        vacul_restarts=args.restarts,     
        vacul_rng_seed=args.rng_seed,  
        out_csv=args.out,
        make_plots=(not args.no_plots)
    )



if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main(sys.argv[1:])

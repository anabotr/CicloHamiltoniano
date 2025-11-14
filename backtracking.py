import argparse
import csv
import random
import time
import math
import sys
from typing import Dict, Set, List, Optional, Tuple

import multiprocessing
import matplotlib.pyplot as plt


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


class BacktrackingHamiltonian:
    def __init__(self, G: Dict[int, Set[int]], start: int = 0, time_limit: float = 5.0):
        self.G = G
        self.n = len(G)
        self.start = start if 0 <= start < self.n else 0
        self.time_limit = float(time_limit)
        self._start_ns: Optional[int] = None
        self.found_path: Optional[List[int]] = None
        self.rec_calls = 0
        self._check_counter = 0

    def _time_exceeded(self) -> bool:
        self._check_counter += 1
        if (self._check_counter & 0x7F) == 0:
            now_ns = time.perf_counter_ns()
            if (now_ns - self._start_ns) / 1e9 > self.time_limit:
                return True
        return False

    def find(self) -> Tuple[Optional[List[int]], bool]:
        if self.n == 0:
            return None, False

        for v in range(self.n):
            if len(self.G[v]) == 0:
                return None, False

        self._start_ns = time.perf_counter_ns()
        path = [-1] * self.n
        visited = [False] * self.n
        path[0] = self.start
        visited[self.start] = True
        self.rec_calls = 0
        self.found_path = None
        self._check_counter = 0

        self._backtrack(1, path, visited)

        timed_out = False
        if self.found_path is None:
            elapsed_ns = time.perf_counter_ns() - self._start_ns
            timed_out = (elapsed_ns / 1e9) >= self.time_limit
        return (self.found_path, timed_out)

    def _neighbors_unvisited(self, v: int, visited: List[bool]) -> List[int]:
        return [u for u in self.G[v] if not visited[u]]

    def _backtrack(self, pos: int, path: List[int], visited: List[bool]):
        self.rec_calls += 1

        if self._time_exceeded():
            return

        if self.found_path is not None:
            return

        if pos == self.n:
            if path[-1] in self.G[path[0]]:
                self.found_path = path.copy()
            return

        last = path[pos - 1]
        candidates = self._neighbors_unvisited(last, visited)

        for v in candidates:
            if self.found_path is not None:
                return

            visited[v] = True
            path[pos] = v

            self._backtrack(pos + 1, path, visited)

            visited[v] = False
            path[pos] = -1

            if self._time_exceeded():
                return

def __solver_process_worker(G, start, time_limit, q):
    try:
        solver = BacktrackingHamiltonian(G, start=start, time_limit=time_limit)
        path, timed_out = solver.find()
        q.put(("ok", path, timed_out, solver.rec_calls))
    except Exception as e:
        q.put(("err", repr(e)))


def run_solver_in_subprocess(G: Dict[int, Set[int]], start: int, time_limit: float):
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=__solver_process_worker, args=(G, start, time_limit, q))

    t0 = time.perf_counter_ns()
    p.start()
    p.join(time_limit)
    elapsed_s = (time.perf_counter_ns() - t0) / 1e9

    if p.is_alive():
        p.terminate()
        p.join()
        return None, True, -1, elapsed_s

    try:
        msg = q.get_nowait()
    except:
        return None, False, -1, elapsed_s

    tag, *payload = msg
    if tag == "ok":
        path, timed_out, rec_calls = payload
        return path, bool(timed_out), rec_calls, elapsed_s
    else:
        return None, False, -1, elapsed_s

def parse_list_arg(s: Optional[str]) -> Optional[List[float]]:
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

            path, timed_out, rec_calls, elapsed_s = run_solver_in_subprocess(
                G, start=0, time_limit=time_limit
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

            print(
                f"[run {run_idx}] graph#{g_idx} rep={rep} n={n} p={p:.4f} avg_deg={avg_deg:.3f} "
                f"time={elapsed_s:.9f}s found={found} timed_out={timed_out} calls={rec_calls}"
            )

            run_idx += 1

    # CSV saving
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

    # plots
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
            plt.title("Backtracking time vs average degree")
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
            plt.title("Backtracking time distribution by n")
            plt.tight_layout()
            box_file = out_csv.replace(".csv","") + "_boxplot.png"
            plt.savefig(box_file)
            plt.close()
            print(f"Saved boxplot: {box_file}")

        except Exception as e:
            print("Plotting failed:", e)

    return results


def main(argv):
    parser = argparse.ArgumentParser(description="Hamiltonian backtracking experiments")
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
    parser.add_argument("--out", type=str, default="results_backtracking_20.csv")
    parser.add_argument("--no_plots", action="store_true")

    args = parser.parse_args(argv)

    n_list = parse_list_arg(args.n_list) if args.n_list else None
    p_list = parse_list_arg(args.p_list) if args.p_list else None

    print("Running experiments with parameters:")
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

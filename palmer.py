import argparse
import csv
import random
import time
import sys
from typing import Dict, Set, List, Optional, Tuple
import multiprocessing
import matplotlib.pyplot as plt

def erdos_renyi_adj(n: int, p: float, rnd: random.Random) -> Dict[int, Set[int]]:
    G = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i+1, n):
            if rnd.random() < p:
                G[i].add(j)
                G[j].add(i)
    return G


def average_degree(G: Dict[int, Set[int]]) -> float:
    if not G:
        return 0.0
    return sum(len(nei) for nei in G.values()) / len(G)


def dfs_longest_path(G: Dict[int, Set[int]]) -> List[int]:
    n = len(G)
    best = []
    seeds = list(range(n))
    attempts = min(n, 8)

    for s in seeds[:attempts]:
        visited = [False] * n
        path = []

        def dfs(u):
            visited[u] = True
            path.append(u)
            nbrs = sorted(G[u], key=lambda x: -len(G[x]))
            for v in nbrs:
                if not visited[v]:
                    dfs(v)

        dfs(s)
        if len(path) > len(best):
            best = path.copy()

    if len(best) < n:
        remaining = [v for v in range(n) if v not in best]
        best.extend(remaining)

    return best

class PalmerCrissCross:
    def __init__(self, G: Dict[int, Set[int]], init_by_dfs=True, time_limit=5.0):
        self.G = G
        self.n = len(G)
        self.time_limit = float(time_limit)
        self.init_by_dfs = init_by_dfs
        self._start_ns = None
        self._check_counter = 0
        self.iterations = 0

    def _time_exceeded(self) -> bool:
        self._check_counter += 1
        if (self._check_counter & 0x3F) == 0:
            now = time.perf_counter_ns()
            if (now - self._start_ns) / 1e9 > self.time_limit:
                return True
        return False

    def find(self) -> Tuple[Optional[List[int]], bool]:
        if self.n == 0:
            return None, False

        for v in range(self.n):
            if len(self.G[v]) == 0:
                return None, False

        self._start_ns = time.perf_counter_ns()

        if self.init_by_dfs:
            order = dfs_longest_path(self.G)
        else:
            order = list(range(self.n))

        if len(order) != self.n or set(order) != set(range(self.n)):
            missing = [v for v in range(self.n) if v not in order]
            order = order[:self.n] + missing

        def is_cycle(c):
            for i in range(self.n):
                a = c[i]
                b = c[(i+1) % self.n]
                if b not in self.G[a]:
                    return False
            return True

        max_passes = max(1000, self.n * self.n * self.n)
        passes = 0

        while True:
            if self._time_exceeded():
                return None, True

            if is_cycle(order):
                return order, False

            improved = False

            for i in range(self.n):
                a = order[i]
                b = order[(i+1) % self.n]

                if b in self.G[a]:
                    continue

                for offset in range(1, self.n - 1):
                    j = (i + offset) % self.n
                    j1 = (j + 1) % self.n

                    u = order[j]
                    u1 = order[j1]

                    if (u1 in self.G[a]) and (b in self.G[u]):
                        if i < j:
                            new_order = (
                                order[:i+1]
                                + list(reversed(order[i+1:j+1]))
                                + order[j+1:]
                            )
                        else:
                            seg = order[i+1:] + order[:j+1]
                            seg_rev = list(reversed(seg))
                            new_order = seg_rev
                            try:
                                idx0 = new_order.index(order[0])
                                new_order = new_order[idx0:] + new_order[:idx0]
                            except ValueError:
                                new_order = order[:]

                        order = new_order
                        improved = True
                        break

                if improved:
                    break

            passes += 1
            self.iterations = passes

            if not improved:
                return None, False

            if passes > max_passes:
                return None, False


def solver_process_worker(G, time_limit, q, init_by_dfs_flag):
    try:
        solver = PalmerCrissCross(G, init_by_dfs=init_by_dfs_flag, time_limit=time_limit)
        path, timed_out = solver.find()
        q.put(("ok", path, timed_out, solver.iterations))
    except Exception as e:
        q.put(("err", repr(e)))


def run_solver_in_subprocess(G, time_limit, init_by_dfs_flag):
    q = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=solver_process_worker,
        args=(G, time_limit, q, init_by_dfs_flag)
    )
    t0 = time.perf_counter_ns()
    p.start()
    p.join(time_limit)
    elapsed = (time.perf_counter_ns() - t0) / 1e9

    if p.is_alive():
        p.terminate()
        p.join()
        return None, True, -1, elapsed

    try:
        msg = q.get_nowait()
    except Exception:
        return None, False, -1, elapsed

    tag, *payload = msg
    if tag == "ok":
        path, timed_out, iters = payload
        return path, bool(timed_out), iters, elapsed
    else:
        return None, False, -1, elapsed


def parse_list_arg(s):
    if s is None:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(float(p) if "." in p else int(p))
        except:
            pass
    return out if out else None


def run_experiments(
    num_graphs,
    repeats,
    seed,
    n_min,
    n_max,
    p_min,
    p_max,
    n_list,
    p_list,
    time_limit,
    out_csv,
    make_plots,
    init_by_dfs_flag
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

            path, timed_out, iter_count, elapsed = run_solver_in_subprocess(
                G, time_limit, init_by_dfs_flag
            )

            found = path is not None

            results.append({
                "run_idx": run_idx,
                "graph_idx": g_idx,
                "rep": rep,
                "n": n,
                "p": round(p, 6),
                "avg_degree": round(avg_deg, 6),
                "time_s": f"{elapsed:.9f}",
                "time_float": elapsed,
                "found": bool(found),
                "timed_out": bool(timed_out),
                "rec_calls": iter_count
            })

            print(
                f"[run {run_idx}] graph#{g_idx} rep={rep} n={n} p={p:.4f} "
                f"avg_deg={avg_deg:.3f} time={elapsed:.9f}s found={found} "
                f"timed_out={timed_out} iters={iter_count}"
            )

            run_idx += 1

    fieldnames = ["run_idx", "graph_idx", "rep", "n", "p", "avg_degree",
                  "time_s", "found", "timed_out", "rec_calls"]

    try:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in fieldnames})
        print(f"Saved results to {out_csv}")
    except Exception as e:
        print("Failed to save CSV:", e)

    if make_plots:
        try:
            xs1 = [r["avg_degree"] for r in results if r["found"]]
            ys1 = [r["time_float"] for r in results if r["found"]]
            xs2 = [r["avg_degree"] for r in results if not r["found"]]
            ys2 = [r["time_float"] for r in results if not r["found"]]

            plt.figure(figsize=(8, 6))
            if xs1:
                plt.scatter(xs1, ys1, marker="o", alpha=0.8, label="found")
            if xs2:
                plt.scatter(xs2, ys2, marker="x", alpha=0.8, label="not found / timeout")
            plt.xlabel("Average degree")
            plt.ylabel("Time (s)")
            plt.yscale("log")
            plt.title("Palmer Criss-Cross time vs average degree")
            plt.legend()
            plt.tight_layout()
            file1 = out_csv.replace(".csv", "") + "_palmer_scatter.png"
            plt.savefig(file1)
            plt.close()
            print(f"Saved scatter plot: {file1}")

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
            plt.title("Palmer time distribution by n")
            plt.tight_layout()
            file2 = out_csv.replace(".csv", "") + "_palmer_boxplot.png"
            plt.savefig(file2)
            plt.close()
            print(f"Saved boxplot: {file2}")

        except Exception as e:
            print("Plotting failed:", e)

    return results


def main(argv):
    parser = argparse.ArgumentParser(description="Hamiltonian experiments using Palmer Criss-Cross")
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
    parser.add_argument("--out", type=str, default="results_palmer_20.csv")
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--no_dfs_init", action="store_true")
    args = parser.parse_args(argv)

    n_list = parse_list_arg(args.n_list)
    p_list = parse_list_arg(args.p_list)

    print("Running Palmer experiments with parameters:")
    print(f" num_graphs={args.num_graphs} repeats={args.repeats} seed={args.seed}")
    print(f" n_list={n_list if n_list else f'[{args.n_min},{args.n_max}]'}")
    print(f" p_list={p_list if p_list else f'[{args.p_min},{args.p_max}]'}")
    print(f" timelimit={args.timelimit}s out='{args.out}' plots={not args.no_plots} dfs_init={not args.no_dfs_init}")

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
        make_plots=not args.no_plots,
        init_by_dfs_flag=(not args.no_dfs_init)
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main(sys.argv[1:])

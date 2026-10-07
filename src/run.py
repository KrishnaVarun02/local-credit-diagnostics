"""Run the frozen diagnostic matrix, preserving every replicate."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
import numpy as np
from .games import (actions, sparse_synergy, features, uniform_projection,
                    sampled_projection, anchored_projection, evaluate,
                    published_matrix, additive_matrix_projection)

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    paths = sorted((ROOT/"src").glob("*.py")) + sorted((ROOT/"tests").glob("*.py"))
    paths += sorted((ROOT/"configs").glob("*.json"))
    paths += [ROOT/"docs/protocol.md", ROOT/"requirements.txt"]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def run(config, output):
    started = time.perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat()
    config_path = Path(config)
    cfg = json.loads(config_path.read_text())
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/"manifest.json").exists():
        raise FileExistsError("Refusing to replace a completed run; choose a new --output")
    exact, sampling, controls = [], [], []
    for n in cfg["scaling_n"]:
        a = actions(n)
        rewards = sparse_synergy(a, cfg["main_kappa"])
        for method in cfg["topologies"]:
            start = time.perf_counter()
            x = features(a, method)
            coef = uniform_projection(x, rewards)
            exact.append(dict(experiment="scaling", n=n, kappa=cfg["main_kappa"],
                              method=method, parameters=x.shape[1], anchor_weight=1,
                              **evaluate(rewards, x@coef), seconds=time.perf_counter()-start))
        x = features(a)
        for weight in cfg["anchor_weights"]:
            start = time.perf_counter()
            coef = anchored_projection(x, rewards, len(rewards)-1, weight)
            exact.append(dict(experiment="scaling", n=n, kappa=cfg["main_kappa"],
                              method="oracle_anchor", parameters=x.shape[1], anchor_weight=weight,
                              **evaluate(rewards, x@coef), seconds=time.perf_counter()-start))
    n = cfg["strength_n"]
    a = actions(n)
    for kappa in cfg["strength_kappa"]:
        rewards = sparse_synergy(a, kappa)
        for method in cfg["topologies"]:
            start = time.perf_counter()
            x = features(a, method)
            coef = uniform_projection(x, rewards)
            exact.append(dict(experiment="strength", n=n, kappa=kappa, method=method,
                              parameters=x.shape[1], anchor_weight=1,
                              **evaluate(rewards, x@coef), seconds=time.perf_counter()-start))
    for n in cfg["sampling_n"]:
        a = actions(n)
        x = features(a)
        rewards = sparse_synergy(a, cfg["main_kappa"])
        for m in cfg["sample_sizes"]:
            for replicate in range(cfg["seeds"]):
                start = time.perf_counter()
                seed = cfg["seed_offset"]+100000*n+1000*m+replicate
                rng = np.random.default_rng(seed)
                indices = rng.integers(len(a), size=m, dtype=np.int64)
                coef, rank = sampled_projection(x[indices], rewards[indices])
                sampling.append(dict(n=n, m=m, replicate=replicate, seed=seed, rank=rank,
                                     synergy_observations=int(np.sum(indices == len(a)-1)),
                                     unique_actions=len(np.unique(indices)),
                                     training_indices_sha256=hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
                                     coefficients=json.dumps(coef.tolist(), separators=(",", ":")),
                                     **evaluate(rewards, x@coef), seconds=time.perf_counter()-start))
    n = 6
    a = actions(n)
    for name, method, rewards in [
        ("additive_positive", "additive", a.sum(axis=1)/n),
        ("chain_pairwise", "chain", (a[:, :-1]*a[:, 1:]).mean(axis=1)),
    ]:
        x = features(a, method)
        controls.append(dict(control=name, method=method, **evaluate(rewards, x@uniform_projection(x, rewards))))
    matrix = published_matrix()
    projected = additive_matrix_projection(matrix)
    controls.append(dict(control="qtran_matrix", method="additive", **evaluate(matrix.ravel(), projected.ravel())))
    (out/"published_matrix.json").write_text(json.dumps({
        "source": "https://papers.nips.cc/paper_files/paper/2020/file/73a427badebe0e32caa2e1fc7530b7f3-Paper.pdf",
        "location": "Table 2 left, attributed there to Son et al. (QTRAN)",
        "reward": matrix.tolist(), "our_additive_projection": projected.tolist(),
        "note": "Only the payoff matrix is reused; no neural algorithm reproduction is claimed."
    }, indent=2)+"\n")
    write_csv(out/"exact.csv", exact)
    write_csv(out/"sampling.csv", sampling)
    write_csv(out/"controls.csv", controls)
    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        cpu = platform.processor() or platform.machine()
    manifest = dict(
        started_utc=timestamp, completed_utc=datetime.now(timezone.utc).isoformat(),
        runtime_seconds=time.perf_counter()-started, protocol_version=cfg["protocol_version"],
        config=cfg, config_sha256=sha(config_path), code_hashes=source_hashes(),
        hardware=dict(cpu=cpu, logical_cpus=os.cpu_count(), machine=platform.machine(), system=platform.system(), release=platform.release()),
        environment=dict(python=platform.python_version(), numpy=np.__version__),
        counts=dict(exact=len(exact), sampling=len(sampling), controls=len(controls)),
        failed_runs=[], excluded_runs=[], external_compute_cost=0,
        artifact_hashes={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        command=f"python -m src.run --config configs/{config_path.name} --output results/{out.name}",
        git_commit="No project commit existed at execution; exact source hashes above identify run code."
    )
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"output":str(out), "runtime_seconds":manifest["runtime_seconds"], "counts":manifest["counts"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.json")
    parser.add_argument("--output", default="results/full")
    args = parser.parse_args()
    run(args.config, args.output)

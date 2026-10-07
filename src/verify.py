"""Audit retained evidence and optionally compare an independently rerun matrix."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .games import theorem_values, edges

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def verify(reference, reproduction=None):
    reference = Path(reference)
    manifest = json.loads((reference/"manifest.json").read_text())
    for name, expected in manifest["artifact_hashes"].items():
        assert digest(reference/name) == expected, f"Artifact checksum mismatch: {name}"
    for name, expected in manifest["code_hashes"].items():
        assert digest(ROOT/name) == expected, f"Original run source changed: {name}"
    exact = rows(reference/"exact.csv")
    sample = rows(reference/"sampling.csv")
    controls = rows(reference/"controls.csv")
    assert len(exact) == manifest["counts"]["exact"]
    assert len(sample) == manifest["counts"]["sampling"]
    assert len(controls) == manifest["counts"]["controls"]
    for row in exact:
        if row["method"] == "oracle_anchor":
            assert float(row["regret"]) == 0
            continue
        n = int(row["n"])
        theory = theorem_values(n, float(row["kappa"]), len(edges(n,row["method"])))
        for key in ["mse", "regret"]:
            assert np.isclose(float(row[key]),theory[key],atol=1e-12), (row,key)
    for row in sample:
        assert int(row["rank"]) == int(row["n"])+1
        n,m,seed = map(int,(row["n"],row["m"],row["seed"]))
        sampled_indices=np.random.default_rng(seed).integers(2**n,size=m,dtype=np.int64)
        assert hashlib.sha256(sampled_indices.astype("<i8").tobytes()).hexdigest()==row["training_indices_sha256"]
        assert int((sampled_indices==2**n-1).sum()) == int(row["synergy_observations"])
    assert not manifest["failed_runs"] and not manifest["excluded_runs"]
    if reproduction:
        for filename in ["exact.csv","sampling.csv","controls.csv"]:
            left,right = rows(reference/filename),rows(Path(reproduction)/filename)
            assert len(left)==len(right)
            for old,new in zip(left,right):
                assert old.keys()==new.keys()
                for key in old:
                    if key=="seconds":
                        continue
                    if key=="coefficients":
                        np.testing.assert_allclose(json.loads(old[key]),json.loads(new[key]),atol=1e-12,rtol=1e-10)
                        continue
                    try:
                        assert np.isclose(float(old[key]),float(new[key]),atol=1e-12,rtol=1e-10), (filename,key)
                    except ValueError:
                        assert old[key]==new[key], (filename,key)
    print(json.dumps({"status":"passed","reference":str(reference),"reproduction":str(reproduction) if reproduction else None,
                      "verified_counts":manifest["counts"],"dataset_hashes_checked":len(sample)}))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--reference",default="results/full")
    parser.add_argument("--reproduction")
    args=parser.parse_args()
    verify(args.reference,args.reproduction)

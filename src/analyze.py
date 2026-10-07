"""Regenerate every figure and paper table from retained CSV rows."""
import argparse
import csv
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import statistics


def read_csv(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for k, v in row.items():
            try:
                row[k] = float(v)
            except ValueError:
                pass
    return rows


def wilson(k, n, z=1.959963984540054):
    p = k/n
    den = 1+z*z/n
    center = (p+z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0., center-half), min(1., center+half)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[int(row["n"]), int(row["m"])].append(row)
    result = []
    for (n,m), values in sorted(groups.items()):
        count = len(values)
        success = sum(x["success"] for x in values)
        observed = sum(x["synergy_observations"] > 0 for x in values)
        regret = [x["regret"] for x in values]
        mean = statistics.mean(regret)
        half = 1.959963984540054*statistics.stdev(regret)/math.sqrt(count) if count > 1 else 0
        result.append(dict(n=n,m=m,runs=count,success_rate=success/count,
                           success_ci=wilson(success,count), observation_rate=observed/count,
                           observation_ci=wilson(observed,count), theoretical_observation_rate=1-(1-2**(-n))**m,
                           mean_regret=mean, regret_ci=[max(0.,mean-half),min(max(x["reward_range"] for x in values),mean+half)],
                           mean_mse=statistics.mean(x["mse"] for x in values),
                           rank_deficient=sum(x["rank"] < n+1 for x in values)))
    return result


def generate(results, figures, tables):
    results, figures, tables = map(Path, (results, figures, tables))
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(figures.parent/"work/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "axes.spines.top":False,
                         "axes.spines.right":False, "savefig.bbox":"tight"})
    exact = read_csv(results/"exact.csv")
    sampling = read_csv(results/"sampling.csv")
    summaries = summarize(sampling)
    (tables/"sampling_summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
    colors = {"additive":"#245b8a", "complete":"#df7f23", "oracle_anchor":"#379479"}
    names = {"additive":"Additive", "complete":"Complete pairwise", "oracle_anchor":"Oracle anchor (w=2)"}
    fig, axes = plt.subplots(1, 3, figsize=(10.7,3.05))
    for method in colors:
        rows = [r for r in exact if r["experiment"] == "scaling" and r["method"] == method and (method != "oracle_anchor" or r["anchor_weight"]==2)]
        ns = [r["n"] for r in rows]
        for ax,key in zip(axes,["mse","regret","normalized_mse"]):
            ax.plot(ns,[r[key] for r in rows], "o-", label=names[method], color=colors[method], markersize=3)
            ax.set_xlabel("Number of agents")
            ax.set_xticks([4,8,12,16])
            ax.grid(alpha=.15)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Uniform absolute MSE")
    axes[1].set_ylabel("Coordination regret")
    axes[2].set_ylabel("MSE / reward variance")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ["pdf","svg","png"]:
        fig.savefig(figures/f"scaling.{ext}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(7.3,2.9))
    for n,color in [(8,"#245b8a"),(12,"#df7f23")]:
        rows = [r for r in summaries if r["n"]==n]
        if not rows:
            continue
        for ax,key,ci in [(axes[0],"success_rate","success_ci"),(axes[1],"observation_rate","observation_ci")]:
            means=[r[key] for r in rows]
            errors=[[max(0,r[key]-r[ci][0]) for r in rows],[max(0,r[ci][1]-r[key]) for r in rows]]
            ax.errorbar([r["m"] for r in rows],means,yerr=errors,fmt="o-",capsize=3,color=color,label=f"n={n}")
            ax.set_xscale("log",base=2)
            ax.set_ylim(-.04,1.04)
            ax.set_xlabel("Uniform training samples")
            ax.grid(alpha=.15)
            ax.legend()
    axes[0].set_ylabel("Optimal-action success rate")
    axes[1].set_ylabel("At least one synergy observation")
    fig.tight_layout()
    for ext in ["pdf","svg","png"]:
        fig.savefig(figures/f"sampling.{ext}", dpi=180)
    plt.close(fig)
    table = ["\\begin{tabular}{rllrrr}","\\toprule", "Agents & Model & Weight & MSE & MSE/Var & Regret \\\\","\\midrule"]
    for n in [4,8,12,16]:
        for method in ["additive","complete","oracle_anchor"]:
            rows=[r for r in exact if r["experiment"]=="scaling" and r["n"]==n and r["method"]==method and (method!="oracle_anchor" or r["anchor_weight"]==2)]
            for r in rows:
                label={"additive":"Additive","complete":"Pairwise","oracle_anchor":"Oracle anchor"}[method]
                table.append(f"{n} & {label} & {r['anchor_weight']:g} & {r['mse']:.6g} & {r['normalized_mse']:.4f} & {r['regret']:.6f} \\\\")
    table += ["\\bottomrule", "\\end{tabular}"]
    (tables/"scaling_table.tex").write_text("\n".join(table)+"\n")
    table=["\\begin{tabular}{rrrrrr}","\\toprule","Agents & Samples & Seen (\\%) & Success (\\%) & Mean regret & 95\\% interval \\\\","\\midrule"]
    for r in summaries:
        table.append(f"{r['n']} & {r['m']} & {100*r['observation_rate']:.0f} & {100*r['success_rate']:.0f} & {r['mean_regret']:.4f} & [{r['regret_ci'][0]:.4f}, {r['regret_ci'][1]:.4f}] \\\\")
    table += ["\\bottomrule", "\\end{tabular}"]
    (tables/"sampling_table.tex").write_text("\n".join(table)+"\n")
    table=["\\begin{tabular}{lrrr}","\\toprule","Projection & Parameters & MSE & Regret \\\\","\\midrule"]
    for r in exact:
        if r["experiment"]=="scaling" and r["n"]==8 and r["method"]!="oracle_anchor":
            table.append(f"{r['method'].capitalize()} & {r['parameters']:.0f} & {r['mse']:.8f} & {r['regret']:.5f} \\\\")
    table += ["\\bottomrule", "\\end{tabular}"]
    (tables/"topology_table.tex").write_text("\n".join(table)+"\n")
    numerical={"exact_rows":len(exact),"sampling_rows":len(sampling), "summaries":summaries,
               "scaling_n16_additive":[r for r in exact if r["experiment"]=="scaling" and r["n"]==16 and r["method"]=="additive"],
               "plotting_version":matplotlib.__version__}
    (tables/"summary.json").write_text(json.dumps(numerical,indent=2)+"\n")
    print(json.dumps({"figures":str(figures),"tables":str(tables),"sampling_cells":len(summaries)}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",default="results/full")
    parser.add_argument("--figures",default="figures")
    parser.add_argument("--tables",default="paper/generated")
    args=parser.parse_args()
    generate(args.results,args.figures,args.tables)

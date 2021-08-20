r"""Exp 6: Compare different ways of scaling for centered clipping

- Fix:
    - n=25, f=5
    - Number of iterations = 1200
    - Not *Long tail* (alpha=1)
    - Always NonIID
    - Number of runs = 1
    - LR = 0.01
    - Aggregator: CP

Experiment:
    - ATK= BF LF mimic IPM ALIE
    - m=0, 0.5, 0.9, 0.99
    - bucketing or not
    - scaling rule: NA, linear sqrt
"""
from utils import get_args
from utils import main
from utils import EXP_DIR

args = get_args()
assert args.noniid
assert not args.LT
assert args.agg == "cp"


LOG_DIR = EXP_DIR + "exp6/"
if args.identifier:
    LOG_DIR += f"{args.identifier}/"
elif args.debug:
    LOG_DIR += "debug/"
else:
    LOG_DIR += f"n{args.n}_f{args.f}_{args.agg}_{args.noniid}/"

INP_DIR = LOG_DIR
OUT_DIR = LOG_DIR + "output/"
LOG_DIR += f"{args.attack}_{args.momentum}_s{args.bucketing}_{args.clip_scaling}_seed{args.seed}"

if args.debug:
    MAX_BATCHES_PER_EPOCH = 30
    EPOCHS = 3
else:
    MAX_BATCHES_PER_EPOCH = 30
    EPOCHS = 40


if not args.plot:
    main(args, LOG_DIR, EPOCHS, MAX_BATCHES_PER_EPOCH)
else:
    # Temporarily put the import functions here to avoid
    # random error stops the running processes.
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from codes.parser import extract_validation_entries

    def exp_grid():
        for attack in ["BF", "LF", "mimic", "IPM", "ALIE"]:
            for bucketing in [0, 2]:
                for momentum in [0.0, 0.5, 0.9, 0.99]:
                    for scaling in ["None", "linear", "sqrt"]:
                        yield attack, momentum, bucketing, scaling

    results = []
    for attack, momentum, bucketing, scaling in exp_grid():
        grid_identifier = f"{attack}_{momentum}_s{bucketing}_{scaling}_seed0"
        path = INP_DIR + grid_identifier + "/stats"
        try:
            values = extract_validation_entries(path)
            for v in values:
                results.append(
                    {
                        "Iterations": v["E"] * MAX_BATCHES_PER_EPOCH,
                        "Accuracy (%)": v["top1"],
                        "ATK": attack,
                        r"$\beta$": str(momentum),
                        "Scaling": scaling if scaling != "None" else "NA",
                        "Bucketing": str(bucketing),
                    }
                )
        except Exception as e:
            pass

    results = pd.DataFrame(results)
    print(results)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    sns.set(font_scale=1.25)
    g = sns.relplot(
        data=results,
        x="Iterations",
        y="Accuracy (%)",
        col="ATK",
        row="Scaling",
        style="Bucketing",
        # hue="Resampling",
        hue=r"$\beta$",
        kind="line",
        ci=None,
        height=2.5,
        aspect=1.3,
    )
    g.set(xlim=(0, 1200), ylim=(0, 100))
    g.fig.savefig(OUT_DIR + "exp6.pdf", bbox_inches="tight")

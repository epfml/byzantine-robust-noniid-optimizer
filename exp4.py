r"""Exp 4:
- Fix:
    - n=53, f=?
    - Number of iterations = 600
    - Not *Long tail* (alpha=1)
    - Always NonIID
    - Number of runs = 3
    - LR = 0.01
    - Attack: IPM epsilon=0.1
    - Aggregator: CP
- Varies:
    - momentum=0, 0.9
    - Bucketing: ?

Experiment:
    - Fix f=5 varying s:
        - s=0,2,5
        - m=0,0.9
    - Fix s=2 varying f:
        - f=1,6,12
        - m=0.0.9
"""
from utils import get_args
from utils import main
from utils import EXP_DIR

args = get_args()
assert args.noniid
assert not args.LT
assert args.attack == "IPM"

LOG_DIR = EXP_DIR + "exp4/"

if args.identifier:
    LOG_DIR += f"{args.identifier}/"
elif args.debug:
    LOG_DIR += "debug/"
else:
    LOG_DIR += f"n{args.n}_{args.agg}_{args.attack}_{args.noniid}/"

INP_DIR = LOG_DIR
OUT_DIR = LOG_DIR + "output/"
LOG_DIR += f"f{args.f}_{args.momentum}_s{args.bucketing}_seed{args.seed}"

if args.debug:
    MAX_BATCHES_PER_EPOCH = 30
    EPOCHS = 3
else:
    MAX_BATCHES_PER_EPOCH = 30
    EPOCHS = 20

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

    # 5.5in is the text width of iclr2022 and 11 is the font size
    font = {"size": 11}
    plt.rc("font", **font)

    def exp_grid1():
        for seed in [0, 1, 2]:
            for bucketing in [0, 2, 5]:
                for momentum in [0.0, 0.9]:
                    yield momentum, bucketing, seed

    results = []
    for momentum, bucketing, seed in exp_grid1():
        grid_identifier = f"f5_{momentum}_s{bucketing}_seed{seed}"
        path = INP_DIR + grid_identifier + "/stats"
        try:
            values = extract_validation_entries(path)
            for v in values:
                results.append(
                    {
                        "Iterations": v["E"] * MAX_BATCHES_PER_EPOCH,
                        "Accuracy (%)": v["top1"],
                        r"$\beta$": momentum,
                        "seed": seed,
                        "s": str(bucketing),
                    }
                )
        except Exception as e:
            pass

    results = pd.DataFrame(results)
    print(results)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    results.to_csv(OUT_DIR + "exp4_fix_f.csv", index=None)

    plt.figure(figsize=(4, 2))
    # sns.set(font_scale=1.25)
    g = sns.lineplot(
        data=results,
        x="Iterations",
        y="Accuracy (%)",
        style=r"$\beta$",
        hue="s",
        # height=2.5,
        # aspect=1.3,
        # legend=False,
        # ci=None,
        palette=sns.color_palette("Set1", 3),
    )
    g.set(xlim=(0, 600), ylim=(50, 100))
    # Put the legend out of the figure
    g.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    g.get_figure().savefig(OUT_DIR + "exp4_fix_f.pdf", bbox_inches="tight", dpi=720)

    plt.figure(0)

    def exp_grid2():
        for seed in [0, 1, 2]:
            for f in [1, 6, 12]:
                for momentum in [0.0, 0.9]:
                    yield momentum, f, seed

    results = []
    for momentum, f, seed in exp_grid2():
        grid_identifier = f"f{f}_{momentum}_s2_seed{seed}"
        path = INP_DIR + grid_identifier + "/stats"
        try:
            values = extract_validation_entries(path)
            for v in values:
                results.append(
                    {
                        "Iterations": v["E"] * MAX_BATCHES_PER_EPOCH,
                        "Accuracy (%)": v["top1"],
                        r"$\beta$": momentum,
                        "seed": seed,
                        "f": str(f),
                    }
                )
        except Exception as e:
            pass

    results = pd.DataFrame(results)
    print(results)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    results.to_csv(OUT_DIR + "exp4_fix_s.csv", index=None)

    plt.figure(figsize=(4, 2))
    # sns.set(font_scale=1.25)
    g = sns.lineplot(
        data=results,
        x="Iterations",
        y="Accuracy (%)",
        style=r"$\beta$",
        hue="f",
        palette=sns.color_palette("Set1", 3),
        # height=2.5,
        # aspect=2,
        # legend=False,
        # ci=None,
    )
    g.set(xlim=(0, 600), ylim=(50, 100))
    # Put the legend out of the figure
    g.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    g.get_figure().savefig(OUT_DIR + "exp4_fix_s.pdf", bbox_inches="tight", dpi=720)

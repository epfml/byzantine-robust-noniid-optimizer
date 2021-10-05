r"""Exp 3:
- Fix:
    - n=25, f=5
    - Number of iterations = 600
    - Not *Long tail* (alpha=1)
    - Always NonIID
    - Number of runs = 3
    - LR = 0.01
- Varies:
    - momentum=0, 0.9
    - ATK = LF, BF, Mimic, IPM, ALIE
    - 5 Aggregators: AVG, KRUM, CM, RFA, CClip (AVG against no attack)
    - Bucketing or not
"""
from utils import get_args
from utils import main
from utils import EXP_DIR


args = get_args()
assert args.noniid
assert not args.LT


LOG_DIR = EXP_DIR + "exp3/"

if args.identifier:
    LOG_DIR += f"{args.identifier}/"
elif args.debug:
    LOG_DIR += "debug/"
else:
    LOG_DIR += f"n{args.n}_f{args.f}_{args.noniid}/"

INP_DIR = LOG_DIR
OUT_DIR = LOG_DIR + "output/"
LOG_DIR += f"{args.agg}_{args.attack}_{args.momentum}_s{args.bucketing}_seed{args.seed}"

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
    font = {"size": 10}
    plt.rc("font", **font)

    def exp_grid():
        for agg in ["krum", "cm", "cp", "rfa"]:
            for seed in [0, 1, 2]:
                for bucketing in [0, 2]:
                    for momentum in [0.0, 0.9]:
                        for attack in ["BF", "LF", "mimic", "IPM", "ALIE"]:
                            yield agg, attack, momentum, bucketing, seed

    results = []
    for agg, attack, momentum, bucketing, seed in exp_grid():
        grid_identifier = f"{agg}_{attack}_{momentum}_s{bucketing}_seed{seed}"
        path = INP_DIR + grid_identifier + "/stats"
        try:
            values = extract_validation_entries(path)
            for v in values:
                results.append(
                    {
                        "Iterations": v["E"] * MAX_BATCHES_PER_EPOCH,
                        "Accuracy (%)": v["top1"],
                        "ATK": attack,
                        "AGG": agg.upper() if agg != "cp" else "CClip",
                        r"$\beta$": momentum,
                        "seed": seed,
                        r"s": str(bucketing),
                        # r"s": bucketing,
                    }
                )
        except Exception as e:
            pass

    results = pd.DataFrame(results)
    print(results)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    results.to_csv(OUT_DIR + "exp3.csv", index=None)

    # sns.set(font_scale=1.7)
    g = sns.relplot(
        data=results,
        x="Iterations",
        y="Accuracy (%)",
        style=r"$\beta$",
        col="ATK",
        row="AGG",
        hue=r"s",
        height=1.25,
        aspect=2 / 1.25,
        palette=sns.color_palette("Set1", 2),
        # legend=False,
        # ci=None,
        kind="line",
        # facet_kws={"margin_titles": True},
    )
    g.set(xlim=(0, 600), ylim=(0, 100))
    g.set_axis_labels("Iterations", "Accuracy (%)")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.fig.subplots_adjust(wspace=0.1)

    g.fig.savefig(OUT_DIR + "exp3.pdf", bbox_inches="tight", dpi=720)

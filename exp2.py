r"""Exp 2:
- Fix:
    - n=25, f=5
    - momentum=0
    - Number of iterations = 600
    - Not *Long tail* (alpha=1)
    - Number of runs = 3
    - ATK = mimic
    - LR = 0.01
- Varies:
    - IID vs NonIID
    - 5 Aggregators: AVG, KRUM, CM, RFA, CClip 
    - Bucketing or not
"""
from utils import get_args
from utils import main
from utils import EXP_DIR

args = get_args()
assert not args.LT
assert args.attack == "mimic"
assert args.momentum == 0

LOG_DIR = EXP_DIR + "exp2/"

if args.identifier:
    LOG_DIR += f"{args.identifier}/"
elif args.debug:
    LOG_DIR += "debug/"
else:
    LOG_DIR += f"n{args.n}_f{args.f}mimic_m0/"

INP_DIR = LOG_DIR
OUT_DIR = LOG_DIR + "output/"
LOG_DIR += f"{args.agg}_{args.noniid}_s{args.bucketing}_seed{args.seed}"

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
    from codes.parser import extract_validation_entries

    def exp_grid():
        for agg in ["cm", "cp", "rfa", "krum", "avg"]:
            for seed in [0, 1, 2]:
                for bucketing in [0, 2]:
                    for noniid in [False, True]:
                        yield agg, noniid, bucketing, seed

    results = []
    for agg, noniid, bucketing, seed in exp_grid():
        grid_identifier = f"{agg}_{noniid}_s{bucketing}_seed{seed}"
        path = INP_DIR + grid_identifier + "/stats"
        try:
            values = extract_validation_entries(path)
            accs = list(map(lambda x: x["top1"], values))
            acc = sum(accs[-5:]) / len(accs[-5:])
            results.append(
                {
                    # The entry Iteration is dummy as we only take the test accuracy
                    # At the end of training
                    "Iterations": values[-1]["E"] * MAX_BATCHES_PER_EPOCH,
                    "Accuracy (%)": acc,
                    "Noniid": noniid,
                    "AGG": agg,
                    "seed": seed,
                    "bucketing": bucketing > 0,
                }
            )
        except Exception as e:
            pass

    results = pd.DataFrame(results)
    print(results)

    # Example output:
    # \begin{tabular}{lcc}
    #     \toprule
    #     Aggr         & \iid               & {\noniid}          \\\midrule
    #     \textsc{Avg} & $98.84\!\pm\!0.08$ & $98.84\!\pm\!0.07$ \\
    #     \krum        & $98.10\!\pm\!0.14$ & $82.97\!\pm\!3.64$ \\
    #     \cm          & $97.82\!\pm\!0.20$ & $80.36\!\pm\!0.04$ \\
    #     \rfa         & $98.72\!\pm\!0.11$ & $84.76\!\pm\!0.83$ \\
    #     \cclip       & $98.76\!\pm\!0.10$ & $98.15\!\pm\!0.19$ \\
    #     \bottomrule
    # \end{tabular}

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    def query(agg, noniid, bucketing, value="mean"):
        #
        a = results[
            (results["AGG"] == agg)
            & (results["Noniid"] == noniid)
            & (results["bucketing"] == bucketing)
        ]

        if value == "mean":
            b = a["Accuracy (%)"].mean()
            return "{:.2f}".format(b)

        if value == "std":
            b = a["Accuracy (%)"].std()
            return "{:.2f}".format(b)

        raise NotImplementedError(value)

    for bucketing in [True, False]:
        filename = "exp2.tex" if not bucketing else "exp2_bucketing.tex"
        with open(OUT_DIR + filename, "w") as f:
            f.write(r"\begin{tabular}{lcc}" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(r"Aggr& \iid& {\noniid}\\\midrule" + "\n")
            # ----------------------------------------------------------------
            f.write(
                r"\textsc{Avg} & $"
                + query("avg", False, bucketing=bucketing, value="mean")
                + r"\!\pm\!"
                + query("avg", False, bucketing=bucketing, value="std")
                + r"$ & $"
                + query("avg", True, bucketing=bucketing, value="mean")
                + r"\!\pm\!"
                + query("avg", True, bucketing=bucketing, value="std")
                + r"$ \\"
                + "\n"
            )

            f.write(
                r"\krum & $"
                + query("krum", False, bucketing=bucketing, value="mean")
                + r"\!\pm\!"
                + query("krum", False, bucketing=bucketing, value="std")
                + r"$ & $"
                + query("krum", True, bucketing=bucketing, value="mean")
                + r"\!\pm\!"
                + query("krum", True, bucketing=bucketing, value="std")
                + r"$ \\"
                + "\n"
            )

            f.write(
                r"\cm & $"
                + query("cm", False, bucketing=bucketing, value="mean")
                + r"\!\pm\!"
                + query("cm", False, bucketing=bucketing, value="std")
                + r"$ & $"
                + query("cm", True, bucketing=bucketing, value="mean")
                + r"\!\pm\!"
                + query("cm", True, bucketing=bucketing, value="std")
                + r"$ \\"
                + "\n"
            )

            f.write(
                r"\rfa & $"
                + query("rfa", False, bucketing=bucketing, value="mean")
                + r"\!\pm\!"
                + query("rfa", False, bucketing=bucketing, value="std")
                + r"$ & $"
                + query("rfa", True, bucketing=bucketing, value="mean")
                + r"\!\pm\!"
                + query("rfa", True, bucketing=bucketing, value="std")
                + r"$ \\"
                + "\n"
            )

            f.write(
                r"\cclip & $"
                + query("cp", False, bucketing=bucketing, value="mean")
                + r"\!\pm\!"
                + query("cp", False, bucketing=bucketing, value="std")
                + r"$ & $"
                + query("cp", True, bucketing=bucketing, value="mean")
                + r"\!\pm\!"
                + query("cp", True, bucketing=bucketing, value="std")
                + r"$ \\"
                + "\n"
            )

            # ----------------------------------------------------------------
            f.write(r"\bottomrule" + "\n")
            f.write(r"\end{tabular}" + "\n")

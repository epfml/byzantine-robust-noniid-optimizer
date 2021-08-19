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
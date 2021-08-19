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
    pass
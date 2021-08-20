r"""Exp 7: Demonstrate the effect of bucketing on Krum

- Fix:
    - n=20, f=3
    - m=0
    - Number of iterations = 1200
    - Aggregator: Krum
    - Not *Long tail* (alpha=1)
    - Always NonIID
    - Number of runs = 1
    - LR = 0.01
    - ATK= LF

Experiment:
    - bucketing: 0, 2, 3
"""
from utils import get_args
from utils import main
from utils import EXP_DIR

args = get_args()
assert args.noniid
assert not args.LT
assert args.agg == "krum"
assert args.momentum == 0
assert args.attack == "LF"

LOG_DIR = EXP_DIR + "exp7/"
if args.identifier:
    LOG_DIR += f"{args.identifier}/"
elif args.debug:
    LOG_DIR += "debug/"
else:
    LOG_DIR += f"n{args.n}_f{args.f}_{args.agg}_{args.attack}_{args.noniid}/"

INP_DIR = LOG_DIR
OUT_DIR = LOG_DIR + "output/"
LOG_DIR += f"s{args.bucketing}_seed{args.seed}"

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
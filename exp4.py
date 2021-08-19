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
    pass
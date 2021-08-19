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
    pass
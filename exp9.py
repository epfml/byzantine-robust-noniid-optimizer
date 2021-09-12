r"""Exp 9: Overparameterization but compute value B

Note that the code and setup is almost the same as exp8 except that we additionally compute the value B
in equation (3).

- Fix:
    - n=20, f=3
    - Number of iterations = 3000
    - Not *Long tail* (alpha=1)
    - Always NonIID
    - Number of runs = 1
    - LR = 0.01
    - Aggregator: rfa
    - m=0

Experiment:
    - ATK= BF LF mimic IPM ALIE
    - bucketing: 0, 2, 3
    - Model scale: 1, 2, 4, 8
"""
import json
import logging
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

from codes.worker import ByzantineWorker
from utils import *

args = get_args()
assert args.noniid
assert not args.LT
assert args.agg == "rfa"

LOG_DIR = EXP_DIR + "exp9/"
if args.identifier:
    LOG_DIR += f"{args.identifier}/"
elif args.debug:
    LOG_DIR += "debug/"
else:
    LOG_DIR += f"n{args.n}_f{args.f}_{args.agg}_{args.momentum}_{args.noniid}/"

INP_DIR = LOG_DIR
OUT_DIR = LOG_DIR + "output/"
LOG_DIR += f"{args.attack}_s{args.bucketing}_{args.op}_seed{args.seed}"

if args.debug:
    MAX_BATCHES_PER_EPOCH = 30
    EPOCHS = 3
else:
    MAX_BATCHES_PER_EPOCH = 30
    EPOCHS = 100


class ParameterizedNet(nn.Module):
    def __init__(self):
        super(ParameterizedNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32 * args.op, 3, 1)
        self.conv2 = nn.Conv2d(32 * args.op, 64 * args.op, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216 * args.op, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


json_logger = logging.getLogger("stats")


def compute_heterogeneity_B(trainer, epoch, batch_idx):
    global_gradient = sum(
        w.get_gradient()
        for _, w in enumerate(trainer.workers)
        if not isinstance(w, ByzantineWorker)
    ) / len(
        [w for _, w in enumerate(trainer.workers) if not isinstance(w, ByzantineWorker)]
    )

    global_gradient_norm2 = (global_gradient.norm() ** 2).item()
    msg = {
        "_meta": {"type": "B"},
        "E": epoch,
        "B": batch_idx,
        "global_gradient_norm2": global_gradient_norm2,
        "gradient_differences": {},
    }

    s = 0
    c = 0
    for i, w in enumerate(trainer.workers):
        if not isinstance(w, ByzantineWorker):
            g = w.get_gradient()
            v = ((g - global_gradient).norm() ** 2).item()
            msg["gradient_differences"][i] = v
            s += v
            c += 1
    s /= c
    msg["B2"] = s / global_gradient_norm2

    json_logger.info(json.dumps(msg))


def exp9_main(args, LOG_DIR, EPOCHS, MAX_BATCHES_PER_EPOCH):
    initialize_logger(LOG_DIR)

    if args.use_cuda and not torch.cuda.is_available():
        print("=> There is no cuda device!!!!")
        device = "cpu"
    else:
        device = torch.device("cuda" if args.use_cuda else "cpu")
    # kwargs = {"num_workers": 1, "pin_memory": True} if args.use_cuda else {}
    kwargs = {"pin_memory": True} if args.use_cuda else {}

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = ParameterizedNet().to(device)

    # Each optimizer contains a separate `state` to store info like `momentum_buffer`
    optimizers = [torch.optim.SGD(model.parameters(), lr=LR) for _ in range(args.n)]
    server_opt = torch.optim.SGD(model.parameters(), lr=LR)

    loss_func = F.nll_loss

    metrics = {"top1": top1_accuracy}

    server = TorchServer(optimizer=server_opt)
    trainer = ParallelTrainer(
        server=server,
        aggregator=get_aggregator(args),
        pre_batch_hooks=[],
        post_batch_hooks=[compute_heterogeneity_B],
        max_batches_per_epoch=MAX_BATCHES_PER_EPOCH,
        log_interval=args.log_interval,
        metrics=metrics,
        use_cuda=args.use_cuda,
        debug=False,
    )

    test_loader = mnist(
        data_dir=DATA_DIR,
        train=False,
        download=True,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        sampler_callback=get_test_sampler_callback(args),
        **kwargs,
    )

    evaluator = DistributedEvaluator(
        model=model,
        data_loader=test_loader,
        loss_func=loss_func,
        device=device,
        metrics=metrics,
        use_cuda=args.use_cuda,
        debug=False,
    )

    train_evaluator = DistributedEvaluator(
        model=model,
        data_loader=mnist(
            data_dir=DATA_DIR,
            train=True,
            download=True,
            batch_size=TEST_BATCH_SIZE,
            shuffle=False,
        ),
        loss_func=loss_func,
        device=device,
        metrics=metrics,
        use_cuda=args.use_cuda,
        debug=False,
        log_identifier_type="train evaluator",
    )

    for worker_rank in range(args.n):
        worker = initialize_worker(
            args,
            trainer,
            worker_rank,
            model=model,
            optimizer=optimizers[worker_rank],
            loss_func=loss_func,
            device=device,
            kwargs={},
        )
        trainer.add_worker(worker)

    if not args.dry_run:
        for epoch in range(1, EPOCHS + 1):
            trainer.train(epoch)
            evaluator.evaluate(epoch)
            train_evaluator.evaluate(epoch)
            trainer.parallel_call(lambda w: w.data_loader.sampler.set_epoch(epoch))


if not args.plot:
    exp9_main(args, LOG_DIR, EPOCHS, MAX_BATCHES_PER_EPOCH)
else:
    # Temporarily put the import functions here to avoid
    # random error stops the running processes.
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from codes.parser import extract_validation_entries

    def exp_grid():
        for bucketing in [0, 2, 3]:
            for op in [1, 2, 4, 8]:
                for attack in ["BF", "LF", "mimic", "IPM", "ALIE"]:
                    yield attack, bucketing, op

    results = []
    for attack, bucketing, op in exp_grid():
        grid_identifier = f"{attack}_s{bucketing}_{op}_seed0"
        path = INP_DIR + grid_identifier + "/stats"
        try:
            values = extract_validation_entries(path, kw="train evaluator")
            for v in values:
                results.append(
                    {
                        "Iterations": v["E"] * MAX_BATCHES_PER_EPOCH,
                        "Train Loss": v["Loss"] if v["Loss"] != "nan" else np.inf,
                        "ATK": attack,
                        "Model Scale": str(op),
                        "s": str(bucketing),
                    }
                )
        except Exception as e:
            pass

    results = pd.DataFrame(results)
    print(results)
    print(results.dtypes)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    sns.set(font_scale=1.25)
    g = sns.relplot(
        data=results,
        x="Iterations",
        y="Train Loss",
        col="ATK",
        # style="Momentum",
        row="s",
        hue="Model Scale",
        height=2.5,
        aspect=1.3,
        # legend=False,
        # ci=None,
        kind="line",
    )
    g.set(xlim=(0, 3000), ylim=(0, 1))
    g.fig.savefig(OUT_DIR + "exp9.pdf", bbox_inches="tight")
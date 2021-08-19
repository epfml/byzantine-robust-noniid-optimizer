r"""Exp 5: Gradient norm growing with momentum

- Fix:
    - n=25, f=5
    - Number of iterations = 600
    - Not *Long tail* (alpha=1)
    - Always NonIID
    - Number of runs = 1
    - LR = 0.01
    - Aggregator: CP

Experiment:
    - ATK= BF LF mimic IPM ALIE
    - m=0, 0.5, 0.9, 0.99
"""
from utils import get_args
from utils import main
from utils import EXP_DIR

# TODO: delete
from utils import *
import logging
import json

args = get_args()
assert args.noniid
assert not args.LT
assert args.agg == "cp"

LOG_DIR = EXP_DIR + "exp5/"
if args.identifier:
    LOG_DIR += f"{args.identifier}/"
elif args.debug:
    LOG_DIR += "debug/"
else:
    LOG_DIR += f"n{args.n}_f{args.f}_{args.agg}_{args.noniid}_s{args.bucketing}/"

INP_DIR = LOG_DIR
OUT_DIR = LOG_DIR + "output/"
LOG_DIR += f"{args.attack}_{args.momentum}_seed{args.seed}"

if args.debug:
    MAX_BATCHES_PER_EPOCH = 30
    EPOCHS = 3
else:
    MAX_BATCHES_PER_EPOCH = 30
    EPOCHS = 20


json_logger = logging.getLogger("stats")


def print_gradient_size_hook(trainer, epoch, batch_idx):
    # Post hook
    for i, w in enumerate(trainer.workers):
        g = w.get_gradient()
        g_norm = g.norm().item()
        msg = {
            "_meta": {"type": "g_size"},
            "E": epoch,
            "B": batch_idx,
            "norm": g_norm,
            "rank": i,
        }
        json_logger.info(json.dumps(msg))


def test_main(args, LOG_DIR, EPOCHS, MAX_BATCHES_PER_EPOCH):
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

    model = Net().to(device)

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
        post_batch_hooks=[print_gradient_size_hook],
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
            trainer.parallel_call(lambda w: w.data_loader.sampler.set_epoch(epoch))


if not args.plot:
    test_main(args, LOG_DIR, EPOCHS, MAX_BATCHES_PER_EPOCH)
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
            for momentum in [0.0, 0.5, 0.9, 0.99]:
                yield attack, momentum

    results = []
    for attack, momentum in exp_grid():
        grid_identifier = f"{attack}_{momentum}_seed0"
        path = INP_DIR + grid_identifier + "/stats"
        try:
            values = extract_validation_entries(path, kw="g_size")
            for v in values:
                rank = v["rank"]
                norm = v["norm"]
                results.append(
                    {
                        "Worker ID": rank,
                        "norm": norm,
                        "attack": attack,
                        r"$\beta$": str(momentum),
                        "Iterations": v["E"] * MAX_BATCHES_PER_EPOCH + v["B"],
                    }
                )
        except Exception as e:
            pass

    results = pd.DataFrame(results)
    print(results)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # g = sns.relplot(
    #     data=results,
    #     x="Worker ID",
    #     y="norm",
    #     col="attack",
    #     hue=r"$\beta$",
    #     # hue="Momentum",
    #     kind="line",
    #     ci=None,
    #     height=2.5,
    #     aspect=1.3,
    # )
    # g.set(xlim=(0, 19), ylim=(0, 1000), yscale="log")
    # g.fig.savefig(OUT_DIR + "exp5.pdf", bbox_inches="tight")

    a = (
        results.groupby(["attack", r"$\beta$", "Worker ID"])
        .mean()["norm"]
        .reset_index()
    )

    b = a.copy()
    result = []
    for rank in range(20):
        for attack in ["BF", "LF", "mimic", "IPM", "ALIE"]:
            _b = b[(b["Worker ID"] == rank) & (b["attack"] == attack)]
            v = _b[_b[r"$\beta$"] == str(0.0)].iloc[0]["norm"]
            for beta in [0.0, 0.5, 0.9, 0.99]:
                result.append(
                    {
                        "Worker ID": rank,
                        "ATK": attack,
                        r"$\beta$": str(beta),
                        "Norm ratio": _b[_b[r"$\beta$"] == str(beta)].iloc[0]["norm"]
                        / v,
                        "Norm": _b[_b[r"$\beta$"] == str(beta)].iloc[0]["norm"],
                    }
                )

    result = pd.DataFrame(result)
    print(result)

    sns.set(font_scale=1.25)
    g = sns.relplot(
        data=result,
        x="Worker ID",
        y="Norm ratio",
        col="ATK",
        hue=r"$\beta$",
        # hue="Momentum",
        kind="line",
        ci=None,
        height=2.5,
        aspect=1,
    )
    g.set(xlim=(0, 19), ylim=(0, 100), yscale="log")
    g.fig.savefig(OUT_DIR + "exp5.pdf", bbox_inches="tight")

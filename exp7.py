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
import json
import logging
from utils import *
from utils import _get_aggregator

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

json_logger = logging.getLogger("stats")


def bucketing_wrapper_with_krum_selection(args, aggregator, s):
    """
    Key functionality.
    """
    print("Using bucketing wrapper.")

    def aggr(inputs):
        indices = list(range(len(inputs)))
        np.random.shuffle(indices)

        T = int(np.ceil(args.n / s))

        reshuffled_inputs = []
        for t in range(T):
            indices_slice = indices[t * s : (t + 1) * s]
            g_bar = sum(inputs[i] for i in indices_slice) / len(indices_slice)
            reshuffled_inputs.append(g_bar)

        output = aggregator(reshuffled_inputs)

        # Record the indices selected
        selected = aggregator.top_m_indices[0]
        original_indices = indices[selected * s : (selected + 1) * s]
        msg = {
            "_meta": {"type": "krum_selection"},
            "indices": original_indices,
        }
        json_logger.info(json.dumps(msg))

        return output

    return aggr


def get_aggregator_with_selection(args):
    aggr = _get_aggregator(args)
    if args.bucketing == 0:
        return bucketing_wrapper_with_krum_selection(args, aggr, 1)

    return bucketing_wrapper_with_krum_selection(args, aggr, args.bucketing)


def exp7_main(args, LOG_DIR, EPOCHS, MAX_BATCHES_PER_EPOCH):
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
        # NOTE: Only difference with main
        aggregator=get_aggregator_with_selection(args),
        pre_batch_hooks=[],
        post_batch_hooks=[],
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
    exp7_main(args, LOG_DIR, EPOCHS, MAX_BATCHES_PER_EPOCH)
else:
    # Temporarily put the import functions here to avoid
    # random error stops the running processes.
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from codes.parser import extract_validation_entries

    results = []
    for seed in [0, 1, 2]:
        for bucketing in [0, 2, 3]:
            grid_identifier = f"s{bucketing}_seed{seed}"
            path = INP_DIR + grid_identifier + "/stats"
            try:
                values = extract_validation_entries(path, kw="krum_selection")
                res = []
                for r in values:
                    res += r["indices"]
                c = dict(Counter(res))
                for k in range(20):
                    results.append(
                        {
                            "seed": seed,
                            "s": "s=" + str(bucketing),
                            "Worker ID": k,
                            "#Selections": c.get(k, 0)
                            / (1 if bucketing == 0 else bucketing),
                        }
                    )
            except Exception as e:
                print(f"Problem {grid_identifier}")

    results = pd.DataFrame(results)
    print(results)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    g = sns.barplot(
        data=results,
        x="Worker ID",
        y="#Selections",
        hue="s",
        errwidth=1,
        capsize=0.1,
        # kind="bar",
        # ci=None,
        # height=2.5, aspect=1.3,
    )
    g.set(xlim=(0, 19.5), ylim=(0, 500))
    g.text(14, 450, "Benign", fontsize=9, color="r")
    g.text(16.5, 450, "Byzantine", fontsize=9, color="r")
    plt.axvline(x=16.5, color="r", linewidth=1, linestyle=":")
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.0)
    plt.tight_layout()
    g.figure.savefig(OUT_DIR + "exp7.pdf", bbox_inches="tight")

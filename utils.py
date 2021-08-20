import argparse
import numpy as np
import os
import torch
from collections import Counter
from torchvision import datasets
import torch.nn.functional as F

# Utility functions
from codes.tasks.mnist import mnist, Net
from codes.utils import top1_accuracy, initialize_logger

# Attacks
from codes.attacks.labelflipping import LableFlippingWorker
from codes.attacks.bitflipping import BitFlippingWorker
from codes.attacks.mimic import MimicAttacker, MimicVariantAttacker
from codes.attacks.xie import IPMAttack
from codes.attacks.alittle import ALittleIsEnoughAttack

# Main Modules
from codes.worker import TorchWorker, MomentumWorker
from codes.server import TorchServer
from codes.simulator import ParallelTrainer, DistributedEvaluator

# IID vs Non-IID
from codes.sampler import (
    DistributedSampler,
    DecentralizedNonIIDSampler,
    NONIIDLTSampler,
)

# Aggregators
from codes.aggregator.base import Mean
from codes.aggregator.coordinatewise_median import CM
from codes.aggregator.clipping import Clipping
from codes.aggregator.rfa import RFA
from codes.aggregator.trimmed_mean import TM
from codes.aggregator.krum import Krum


def get_args():
    parser = argparse.ArgumentParser(description="")

    # Utility
    parser.add_argument("--use-cuda", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--identifier", type=str, default="debug", help="")
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="If plot is enabled, then ignore all other options.",
    )

    # Experiment configuration
    parser.add_argument("-n", type=int, help="Number of workers")
    parser.add_argument("-f", type=int, help="Number of Byzantine workers.")
    parser.add_argument("--attack", type=str, default="NA", help="Type of attacks.")
    parser.add_argument("--agg", type=str, default="avg", help="")
    parser.add_argument(
        "--noniid",
        action="store_true",
        default=False,
        help="[HP] noniidness.",
    )
    parser.add_argument("--LT", action="store_true", default=False, help="Long tail")

    # Key hyperparameter
    parser.add_argument("--bucketing", type=int, default=0, help="[HP] s")
    parser.add_argument("--momentum", type=float, default=0.0, help="[HP] momentum")

    parser.add_argument("--clip-tau", type=float, default=10.0, help="[HP] momentum")
    parser.add_argument("--clip-scaling", type=str, default=None, help="[HP] momentum")

    parser.add_argument(
        "--mimic-warmup", type=int, default=1, help="the warmup phase in iterations."
    )

    parser.add_argument(
        "--op",
        type=int,
        default=1,
        help="[HP] controlling the degree of overparameterization. "
        "Only used in exp8.",
    )

    args = parser.parse_args()

    if args.n <= 0 or args.f < 0 or args.f >= args.n:
        raise RuntimeError(f"n={args.n} f={args.f}")

    assert args.bucketing >= 0, args.bucketing
    assert args.momentum >= 0, args.momentum
    assert len(args.identifier) > 0
    return args


ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"
DATA_DIR = ROOT_DIR + "datasets/"
EXP_DIR = ROOT_DIR + f"outputs/"

LR = 0.01
# Fixed HPs
BATCH_SIZE = 32
TEST_BATCH_SIZE = 128


def _get_aggregator(args):
    if args.agg == "avg":
        return Mean()

    if args.agg == "cm":
        return CM()

    if args.agg == "cp":
        if args.clip_scaling is None:
            tau = args.clip_tau
        elif args.clip_scaling == "linear":
            tau = args.clip_tau / (1 - args.momentum)
        elif args.clip_scaling == "sqrt":
            tau = args.clip_tau / np.sqrt(1 - args.momentum)
        else:
            raise NotImplementedError(args.clip_scaling)
        return Clipping(tau=tau, n_iter=3)

    if args.agg == "rfa":
        return RFA(T=8)

    if args.agg == "tm":
        return TM(b=args.f)

    if args.agg == "krum":
        T = int(np.ceil(args.n / args.bucketing)) if args.bucketing > 0 else args.n
        return Krum(n=T, f=args.f, m=1)

    raise NotImplementedError(args.agg)


def bucketing_wrapper(args, aggregator, s):
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
        return aggregator(reshuffled_inputs)

    return aggr


def get_aggregator(args):
    aggr = _get_aggregator(args)
    if args.bucketing == 0:
        return aggr

    return bucketing_wrapper(args, aggr, args.bucketing)


def get_sampler_callback(args, rank):
    """
    Get sampler based on the rank of a worker.
    The first `n-f` workers are good, and the rest are Byzantine
    """
    n_good = args.n - args.f
    if rank >= n_good:
        # Byzantine workers
        return lambda x: DistributedSampler(
            num_replicas=n_good,
            rank=rank % (n_good),
            shuffle=True,
            dataset=x,
        )

    return lambda x: NONIIDLTSampler(
        alpha=not args.noniid,
        beta=0.5 if args.LT else 1.0,
        num_replicas=n_good,
        rank=rank,
        shuffle=True,
        dataset=x,
    )


def get_test_sampler_callback(args):
    # This alpha argument is not important as there is
    # only 1 replica
    return lambda x: NONIIDLTSampler(
        alpha=True,
        beta=0.5 if args.LT else 1.0,
        num_replicas=1,
        rank=0,
        shuffle=False,
        dataset=x,
    )


def initialize_worker(
    args,
    trainer,
    worker_rank,
    model,
    optimizer,
    loss_func,
    device,
    kwargs,
):
    train_loader = mnist(
        data_dir=DATA_DIR,
        train=True,
        download=True,
        batch_size=BATCH_SIZE,
        sampler_callback=get_sampler_callback(args, worker_rank),
        dataset_cls=datasets.MNIST,
        drop_last=True,  # Exclude the influence of non-full batch.
        **kwargs,
    )

    if worker_rank < args.n - args.f:
        return MomentumWorker(
            momentum=args.momentum,
            data_loader=train_loader,
            model=model,
            loss_func=loss_func,
            device=device,
            optimizer=optimizer,
            **kwargs,
        )

    if args.attack == "BF":
        return BitFlippingWorker(
            data_loader=train_loader,
            model=model,
            loss_func=loss_func,
            device=device,
            optimizer=optimizer,
            **kwargs,
        )

    if args.attack == "LF":
        return LableFlippingWorker(
            revertible_label_transformer=lambda target: 9 - target,
            data_loader=train_loader,
            model=model,
            loss_func=loss_func,
            device=device,
            optimizer=optimizer,
            **kwargs,
        )

    if args.attack == "mimic":
        attacker = MimicVariantAttacker(
            warmup=args.mimic_warmup,
            data_loader=train_loader,
            model=model,
            loss_func=loss_func,
            device=device,
            optimizer=optimizer,
            **kwargs,
        )
        attacker.configure(trainer)
        return attacker

    if args.attack == "IPM":
        attacker = IPMAttack(
            epsilon=0.1,
            data_loader=train_loader,
            model=model,
            loss_func=loss_func,
            device=device,
            optimizer=optimizer,
            **kwargs,
        )
        attacker.configure(trainer)
        return attacker

    if args.attack == "ALIE":
        attacker = ALittleIsEnoughAttack(
            n=args.n,
            m=args.f,
            # z=1.5,
            data_loader=train_loader,
            model=model,
            loss_func=loss_func,
            device=device,
            optimizer=optimizer,
            **kwargs,
        )
        attacker.configure(trainer)
        return attacker

    raise NotImplementedError(f"No such attack {args.attack}")


def main(args, LOG_DIR, EPOCHS, MAX_BATCHES_PER_EPOCH):
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

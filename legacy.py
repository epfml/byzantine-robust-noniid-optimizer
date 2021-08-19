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
from codes.sampler import DistributedSampler, DecentralizedNonIIDSampler

# Aggregators
from codes.aggregator.base import Mean
from codes.aggregator.coordinatewise_median import CM
from codes.aggregator.clipping import Clipping
from codes.aggregator.rfa import RFA
from codes.aggregator.trimmed_mean import TM
from codes.aggregator.krum import Krum


EXP_ID = __file__[:-3]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = ROOT_DIR + "datasets/"
EXP_DIR = ROOT_DIR + f"outputs/{EXP_ID}/"

parser = argparse.ArgumentParser(description="")
# Utility
parser.add_argument("--use-cuda", action="store_true", default=False)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--log_interval", type=int, default=10)

# Experiment configuration
parser.add_argument("--n", type=int, help="Number of workers")
parser.add_argument("--f", type=int, help="Number of Byzantine workers.")

parser.add_argument("--attack", type=str, default="NA", help="")
parser.add_argument("--agg", type=str, default="avg", help="")
parser.add_argument("--resampling", type=int, default=0, help="[HP] s")
parser.add_argument(
    "--mimic-warmup", type=int, default=None, help="the warmup phase in iterations."
)
parser.add_argument("--momentum", type=float, default=0.0, help="[HP] momentum")
parser.add_argument(
    "--noniid", type=float, default=1.0, help="[HP] degree of noniidness. 0 for IID"
)
parser.add_argument("--identifier", type=str, default="", help="")

args = parser.parse_args()


N_WORKERS = args.n
N_BYZ = args.f
N_GOOD = N_WORKERS - N_BYZ

BATCH_SIZE = 32
TEST_BATCH_SIZE = 128
# MAX_BATCHES_PER_EPOCH = 9999999
MAX_BATCHES_PER_EPOCH = 30
EPOCHS = 20
LR = 0.01
MOMENTUM = args.momentum


def resampling_wrapper(aggregator, T, s):
    print("resampling_wrapper")

    def aggr(inputs):
        indices = list(range(len(inputs)))
        replicated_indices = indices * s
        replicated_indices = np.array(replicated_indices)
        np.random.shuffle(replicated_indices)

        reshuffled_inputs = []
        for t in range(T):
            g_bar = sum(inputs[i] for i in replicated_indices[t * s : (t + 1) * s]) / s
            reshuffled_inputs.append(g_bar)

        return aggregator(reshuffled_inputs)

    return aggr


def get_sampler_callback(rank):
    if rank >= N_GOOD:
        return lambda x: DistributedSampler(
            num_replicas=N_WORKERS,
            rank=rank,
            shuffle=True,
            dataset=x,
        )

    return lambda x: DecentralizedNonIIDSampler(
        num_replicas=N_GOOD,
        rank=rank,
        shuffle=True,
        dataset=x,
    )


def initialize_worker(
    trainer, worker_rank, model, optimizer, loss_func, device, kwargs
):
    train_loader = mnist(
        data_dir=DATA_DIR,
        train=True,
        download=True,
        batch_size=BATCH_SIZE,
        sampler_callback=get_sampler_callback(worker_rank),
        dataset_cls=datasets.MNIST,
        drop_last=True,  # Exclude the influence of non-full batch.
        **kwargs,
    )
    if worker_rank < N_GOOD:
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
        attacker = MimicAttacker(
            target_rank=0,
            data_loader=train_loader,
            model=model,
            loss_func=loss_func,
            device=device,
            optimizer=optimizer,
            **kwargs,
        )
        attacker.configure(trainer)
        return attacker

    if args.attack == "mimicvariant":
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
            n=N_WORKERS,
            m=N_BYZ,
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


def check_noniid_hook(trainer, epoch, batch_idx):
    if epoch == 1 and batch_idx == 0:
        print()
        for i, w in enumerate(trainer.workers):
            targets = w.running["target"].detach().cpu().numpy()
            targets_count = sorted(Counter(targets).items())
            print(f"Node {i}: {targets_count}")
        print()


def print_gradient_hook(trainer, epoch, batch_idx):
    if epoch == 1 and batch_idx == 0:
        print()
        for i, w in enumerate(trainer.workers):
            g = w.get_gradient()
            print(f"Node {i}: {g[:5]}")
        print()


def _get_aggregator():
    if args.resampling > 0:
        corrected_byzantine_workers = args.f * args.resampling
    else:
        corrected_byzantine_workers = args.f

    if args.agg == "avg":
        return Mean()

    if args.agg == "cm":
        return CM()

    if args.agg == "cp":
        if args.momentum > 0:
            return Clipping(tau=10 / (1 - args.momentum), n_iter=3)
        return Clipping(tau=10, n_iter=3)

    if args.agg == "rfa":
        return RFA(T=8)

    if args.agg == "tm":
        return TM(b=args.f * corrected_byzantine_workers)

    if args.agg == "krum":
        return Krum(n=args.n, f=corrected_byzantine_workers, m=1)

    raise NotImplementedError(args.agg)


def get_aggregator():
    aggr = _get_aggregator()
    if args.resampling == 0:
        return aggr

    # if args.resampling * N_BYZ * 2 + 3 > N_WORKERS:
    #     raise NotImplementedError

    return resampling_wrapper(aggr, N_WORKERS, args.resampling)


def main(args):
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
    # TODO: Current implementation of momentum is not correct as we need to send the momentum for aggregation as well.
    optimizers = [torch.optim.SGD(model.parameters(), lr=LR) for _ in range(N_WORKERS)]
    server_opt = torch.optim.SGD(model.parameters(), lr=LR)

    loss_func = F.nll_loss

    metrics = {"top1": top1_accuracy}

    server = TorchServer(optimizer=server_opt)
    trainer = ParallelTrainer(
        server=server,
        aggregator=get_aggregator(),
        pre_batch_hooks=[],
        post_batch_hooks=[check_noniid_hook, print_gradient_hook],
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

    for worker_rank in range(N_WORKERS):
        worker = initialize_worker(
            trainer,
            worker_rank,
            model=model,
            optimizer=optimizers[worker_rank],
            loss_func=loss_func,
            device=device,
            kwargs={},
        )
        trainer.add_worker(worker)

    for epoch in range(1, EPOCHS + 1):
        trainer.train(epoch)
        evaluator.evaluate(epoch)
        trainer.parallel_call(lambda w: w.data_loader.sampler.set_epoch(epoch))


if __name__ == "__main__":
    main(args)

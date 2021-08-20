r"""Exp 8: Overparameterization

- Fix:
    - n=25, f=5
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
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

from utils import *

args = get_args()
assert args.noniid
assert not args.LT
assert args.agg == "rfa"

LOG_DIR = EXP_DIR + "exp8/"
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
        print(args.op)
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


def exp8_main(args, LOG_DIR, EPOCHS, MAX_BATCHES_PER_EPOCH):
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
    exp8_main(args, LOG_DIR, EPOCHS, MAX_BATCHES_PER_EPOCH)
else:
    pass
import logging
import torch
from codes.worker import ByzantineWorker


class MimicAttacker(ByzantineWorker):
    def __init__(self, target_rank, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_rank = target_rank
        self._gradient = None

    def get_gradient(self):
        return self._gradient

    def omniscient_callback(self):
        target_worker = self.simulator.workers[self.target_rank]
        self._gradient = target_worker.get_gradient()

    def set_gradient(self, gradient) -> None:
        raise NotImplementedError

    def apply_gradient(self) -> None:
        raise NotImplementedError


class MimicVariantAttacker(ByzantineWorker):
    def __init__(self, warmup, argmax=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup = warmup
        self.t = 0
        self.mu = None
        self.z = None
        self._gradient = None
        self.argmax = argmax
        self.json_logger = logging.getLogger("stats")
        self.coordinator = None

    def _get_good_grads(self):
        good_grads = []
        good_ranks = []
        for i, w in enumerate(self.simulator.workers):
            if not isinstance(w, ByzantineWorker):
                good_grads.append(w.get_gradient())
                good_ranks.append(i)
        return good_ranks, good_grads

    def _init_callback(self, curr_good_grads, curr_avg):
        """Initialize z."""
        self.mu = curr_avg
        g = torch.Generator(device=curr_avg.device)
        g.manual_seed(0)
        self.z = torch.randn(len(curr_avg), generator=g, device=curr_avg.device)
        cumu = 0
        for g in curr_good_grads:
            w = (g - self.mu).dot(self.z)
            cumu += w * (g - self.mu)
        self.z = cumu / cumu.norm()

    def _warmup_callback(self, curr_good_grads, curr_avg):
        """Update z in the warmup phase."""
        self.mu = self.t / (1 + self.t) * self.mu + curr_avg / (1 + self.t)

        cumu = 0
        for g in curr_good_grads:
            w = (g - self.mu).dot(self.z)
            cumu += w * (g - self.mu)
        self.z = self.t / (1 + self.t) * self.z + cumu / cumu.norm() / (1 + self.t)
        self.z = self.z / self.z.norm()

    def _attack_callback(self, curr_good_grads):
        mv = None
        mi = None
        mg = None
        for i, g in enumerate(curr_good_grads):
            d = g.dot(self.z)
            # if self.coordinator:
            #     print(i, d.item())

            if self.argmax:
                if (mv is None) or (d > mv):
                    mv = d
                    mg = g
                    mi = i
            else:
                if (mv is None) or (d < mv):
                    mv = d
                    mg = g
                    mi = i

        # if self.coordinator:
        #     print()
        return mv, mi, mg

    def maybe_setup_coordinator(self):
        if self.coordinator is not None:
            return

        for w in self.simulator.workers:
            if isinstance(w, MimicVariantAttacker):
                self.coordinator = w == self

        if self.coordinator is None:
            raise NotImplementedError

    def omniscient_callback(self):
        self.maybe_setup_coordinator()
        curr_good_ranks, curr_good_grads = self._get_good_grads()
        curr_avg = sum(curr_good_grads) / len(curr_good_grads)

        # Update z and mu
        if self.t == 0:
            self._init_callback(curr_good_grads, curr_avg)
        elif self.t < self.warmup:
            self._warmup_callback(curr_good_grads, curr_avg)

        # Find the target
        if self.t < self.warmup:
            mv, mi, self._gradient = self._attack_callback(curr_good_grads)
            self.target_rank = curr_good_ranks[mi]

            # Coordinator log the output
            if self.coordinator:
                target_rank = curr_good_ranks[mi]
                r = {
                    "_meta": {"type": "mmc_count"},
                    "select": target_rank,
                    "value": mv.item(),
                }
                self.json_logger.info(r)

        else:
            # Fix device
            target_worker = self.simulator.workers[self.target_rank]
            self._gradient = target_worker.get_gradient()

        self.t += 1

    def get_gradient(self):
        return self._gradient

    def set_gradient(self, gradient) -> None:
        raise NotImplementedError

    def apply_gradient(self) -> None:
        raise NotImplementedError

import torch

# from ..optim import CentralizedSGD, get_gradient_in_1d, set_gradient_in_1d
from ..worker import ByzantineWorker


# class Bitflipping(CentralizedSGD):
#     def __str__(self):
#         return "Bitflipping()"

#     @torch.no_grad()
#     def step(self, closure=None):
#         """Aggregates the gradients and performs a single optimization step.
#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         grad = get_gradient_in_1d(self.model)
#         aggregated = self.updater.update(-grad)
#         set_gradient_in_1d(self.model, aggregated)
#         loss = super(CentralizedSGD, self).step(closure=closure)
#         return loss


class BitFlippingWorker(ByzantineWorker):
    def __str__(self) -> str:
        return "BitFlippingWorker"

    def get_gradient(self):
        # Use self.simulator to get all other workers
        # Note that the byzantine worker does not modify the states directly.
        return -super().get_gradient()

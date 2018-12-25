from ignite.metrics.metric import Metric
import torch.nn.functional as F


class TripletLoss(Metric):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        a, p, n = output
        average_loss = self.loss_fn(a, p, n)

        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss')

        N = self._batch_size(y)
        self._sum += average_loss.item() * N
        self._num_examples += N

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed')
        return self._sum / self._num_examples

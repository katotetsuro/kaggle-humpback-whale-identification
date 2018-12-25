from ignite.metrics.metric import Metric
import torch.nn.functional as F
import torch


class TripletAccuracy(Metric):
    def __init__(self):
        super().__init__()

    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, output):

        a, p, n = output
        p = torch.sum((a - p)**2, dim=1)
        n = torch.sum((a - n)**2, dim=1)
        correct = p < n
        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Accuracy must have at least one example before it can be computed')
        return self._num_correct / self._num_examples


class TripletLoss(Metric):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn
        self._batch_size = lambda x: x.shape[0]

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        a, p, n = output
        average_loss = self.loss_fn(a, p, n)

        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss')

        N = self._batch_size(a)
        self._sum += average_loss.item() * N
        self._num_examples += N

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed')
        return self._sum / self._num_examples

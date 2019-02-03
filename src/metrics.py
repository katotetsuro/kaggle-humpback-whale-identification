from ignite.metrics.metric import Metric
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics import pairwise_distances


def run_one_epoch(model, data_loader):
    pred = []
    gt = []
    with torch.no_grad():
        for batch in data_loader:
            s = len(batch)
            if s == 1:
                # test time
                x = batch
            elif len(batch) == 2:
                x, t = batch
                gt.append(t.cpu().numpy())
            else:
                raise ValueError('invalid dataset. assume (x,t) or x')

            y = model(x)
            pred.append(y.cpu().numpy())

    pred = np.concatenate(pred)
    if len(gt):
        gt = np.concatenate(gt)
        return pred, gt
    else:
        return pred


class TripletAccuracy():

    def compute(self, model, train_loader, val_loader, top_k=5):
        model.eval()
        train_features, train_labels = run_one_epoch(model, train_loader)
        val_features, val_labels = run_one_epoch(model, val_loader)
        distances = pairwise_distances(val_features, train_features)

        indexes_of_similars = np.argsort(distances, axis=1)[:, :top_k]
        weights = np.asarray([1/(k+1) for k in range(top_k)]).reshape(1, -1)
        scores = (indexes_of_similars == val_labels.reshape(-1, 1)).astype(
            np.float32) * weights
        scores = np.max(scores, axis=1)
        mean_average_precision = np.mean(scores)
        model.train()

        return mean_average_precision


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

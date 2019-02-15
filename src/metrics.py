from ignite.metrics.metric import Metric
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics import pairwise_distances


def run_one_epoch(model, data_loader, device='cpu'):
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
            x = x.to(device)
            y = model(x)
            pred.append(y.cpu().numpy())

    pred = np.concatenate(pred)
    if len(gt):
        gt = np.concatenate(gt)
        return pred, gt
    else:
        return pred


class TripletAccuracy():

    def compute(self, model, source_loader, val_loader, top_k=5, device='cpu'):
        model.eval()
        source_features, source_labels = run_one_epoch(
            model, source_loader, device)
        val_features, val_labels = run_one_epoch(model, val_loader, device)
        distances = pairwise_distances(val_features, source_features)

        indexes_of_similars = np.argsort(distances, axis=1)[:, :top_k]
        weights = np.asarray([1/(k+1) for k in range(top_k)]).reshape(1, -1)
        scores = (source_labels[indexes_of_similars] == val_labels.reshape(-1, 1)).astype(
            np.float32) * weights
        scores = np.max(scores, axis=1)
        mean_average_precision = np.mean(scores)
        model.train()

        ids = []
        for i in indexes_of_similars:
            unique_labels = []
            for j in i:  # trainのj番目のサンプルを指す
                l = source_labels[j]
                if not l in unique_labels:
                    unique_labels.append(l)

                if len(unique_labels) == top_k:
                    break

            while (len(unique_labels) != 5):
                unique_labels.append(-1)
            ids.append(unique_labels)

        unique_labels = np.asarray(ids)
        import pdb
        pdb.set_trace()
        scores_2 = (unique_labels == val_labels.reshape(-1, 1)).astype(
            np.float32) * weights
        scores_2 = np.max(scores_2, axis=1)
        unique_precision = np.mean(scores_2)

        return mean_average_precision, unique_precision


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

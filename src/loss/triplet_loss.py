import torch.nn as nn
import torch

"""
https://omoindrot.github.io/triplet-loss#a-better-implementation-with-online-triplet-mining
"""


class TripletLoss(nn.Module):
    def __init__(self, margin, difficulty=0.93, ignore_labels=[], distance_weight=0.5):
        super().__init__()
        self.margin = margin
        self.semi_hard = True
        self.difficulty = difficulty
        self.max_difficulty = 1.0
        self.active_triplet_percent = 0
        self.ignore_labels = ignore_labels
        self.average_triplet_loss = 0
        self.average_positive_dist = 0
        self.distance_weight = distance_weight

    def _pairwise_distances(self, embeddings):
        dot_product = embeddings.mm(embeddings.transpose(0, 1))
        square_norm = dot_product.diag()
        distances = square_norm.reshape(
            1, -1) - 2.0 * dot_product + square_norm.reshape(-1, 1)
        distances = torch.max(distances, torch.zeros_like(distances))

        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = (distances == 0.0).float()
        distances = distances + mask * 1e-16
        distances = torch.sqrt(distances)
        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

        return distances

    def _get_anchor_positive_triplet_mask(self, labels):
        """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        indices_equal = torch.eye(
            len(labels)).type(torch.ByteTensor).to(labels.device)
        indices_not_equal = 1 - indices_equal
        labels_equal = labels[None] == labels[:, None]
        mask = indices_not_equal * labels_equal

        return mask

    def _get_anchor_negative_triplet_mask(self, labels):
        """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check if labels[i] != labels[k]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = labels[None] == labels[:, None]
        mask = 1 - labels_equal

        return mask

    def _get_anchor_inlier_triplet_mask(self, labels):
        ignore_labels = torch.Tensor(
            self.ignore_labels).long().to(labels.device)
        match = torch.sum(labels == ignore_labels[:, None], dim=0)
        mask = match[None] + match[:, None]
        mask = torch.min(mask, torch.ones_like(mask)).int()
        mask = 1 - mask
        return mask

    def forward(self, embeddings, labels):
        return self.batch_hard(embeddings, labels)

    def batch_hard(self, embeddings, labels):
        order = round(len(embeddings) * self.difficulty) - 1
        # Get the pairwise distance matrix
        pairwise_dist = self._pairwise_distances(embeddings)

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = self._get_anchor_positive_triplet_mask(
            labels).float()

        # ignore specific label's positive distance (e.g. label==0, which indicates new whale)
        mask_anchor_inlier = self._get_anchor_inlier_triplet_mask(
            labels).float()

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist
        anchor_positive_dist = mask_anchor_inlier * anchor_positive_dist

        # shape (batch_size, 1)
        hardest_positive_dist = anchor_positive_dist.sort(dim=1)[0][:, order:]
        # tf.summary.scalar("hardest_positive_dist",
        #                   tf.reduce_mean(hardest_positive_dist))

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = self._get_anchor_negative_triplet_mask(
            labels).float()

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist, _ = pairwise_dist.max(dim=1)
        anchor_negative_dist = pairwise_dist + \
            max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist = anchor_negative_dist.sort(dim=1)[
            0][:, :-order]
        # tf.summary.scalar("hardest_negative_dist",
        #                   tf.reduce_mean(hardest_negative_dist))

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = torch.max(
            hardest_positive_dist - hardest_negative_dist + self.margin, torch.zeros_like(hardest_positive_dist))

        if pairwise_dist.mean() < 0.02:
            print('collapse!?')
            #import pdb
            # pdb.set_trace()
        # Get final mean triplet loss
        average_loss = triplet_loss.mean()
        self.active_triplet_percent = len(
            triplet_loss.nonzero()) / triplet_loss.numel()

        # force positives are converged to points
        n_positive = torch.sum(anchor_positive_dist > 0)
        mean_positive_distance = torch.sum(
            anchor_positive_dist) / n_positive if n_positive.item() > 0 else torch.zeros_like(average_loss)

        self.average_triplet_loss = self.average_triplet_loss * \
            0.9 + average_loss.item() * 0.1
        self.average_positive_dist = self.average_positive_dist * \
            0.9 + mean_positive_distance.item() * 0.1
        return average_loss + self.distance_weight * mean_positive_distance

    def increase_difficulty(self, step=0.1):
        self.difficulty += step
        self.difficulty = min(self.max_difficulty, self.difficulty)
        print('new difficulty:{}'.format(self.difficulty))

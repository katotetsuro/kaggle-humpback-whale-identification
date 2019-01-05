import torch.nn as nn
import torch

"""
https://omoindrot.github.io/triplet-loss#a-better-implementation-with-online-triplet-mining
"""


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

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

    def forward(self, embeddings, labels):
        # Get the pairwise distance matrix
        pairwise_dist = self._pairwise_distances(embeddings)

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = self._get_anchor_positive_triplet_mask(
            labels).float()

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # shape (batch_size, 1)
        hardest_positive_dist, _ = anchor_positive_dist.max(dim=1)
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
        hardest_negative_dist, _ = anchor_negative_dist.min(dim=1)
        # tf.summary.scalar("hardest_negative_dist",
        #                   tf.reduce_mean(hardest_negative_dist))

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = torch.max(
            hardest_positive_dist - hardest_negative_dist + self.margin, torch.zeros_like(hardest_positive_dist))

        #import pdb
        # pdb.set_trace()
        # Get final mean triplet loss
        return triplet_loss.mean()

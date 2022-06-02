import torch
from torch import nn
from torch.nn import functional as F
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputLayers, _log_classification_stats
)


        
# focal loss
class FastRCNNFocaltLossOutputLayers(FastRCNNOutputLayers):

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        #if self.use_sigmoid_ce:
        #    loss_cls = self.sigmoid_cross_entropy_loss(scores, gt_classes)
        #else:
        #    loss_cls = cross_entropy(scores, gt_classes, reduction="mean")
        
        loss_cls = self.focal_loss(scores, gt_classes)
        #loss_cls = cross_entropy(scores, gt_classes, reduction="mean")

        losses = {
            "loss_cls": loss_cls,
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        #if losses['loss_box_reg'] == 0:
        #    print('yanqh',proposal_deltas.shape, gt_boxes, scores.shape, gt_classes)
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
    
    def focal_loss(self, pred_class_logits, gt_classes):
        FC_loss = FocalLoss(
                gamma=1.5,
                num_classes=self.num_classes,
            )
        total_loss = FC_loss(input=pred_class_logits, target=gt_classes)
        total_loss = total_loss / gt_classes.shape[0]

        return total_loss

class FocalLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        gamma=1.0,
        num_classes=80,
    ):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

        self.num_classes = num_classes

    def forward(self, input, target):
        # focal loss
        CE = F.cross_entropy(input, target, reduction="none")
        p = torch.exp(-CE)
        loss = (1 - p) ** self.gamma * CE
        return loss.sum()
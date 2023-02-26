

import numpy as np 
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs import build_match_cost
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None
    
import torch 

def dice_coef(inputs, targets):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1).unsqueeze(1)
    targets = targets.flatten(1).unsqueeze(0)
    numerator = 2 * (inputs * targets).sum(2)
    denominator = inputs.sum(-1) + targets.sum(-1)

    # NOTE coef doesn't be subtracted to 1 as it is not necessary for computing costs
    coef = (numerator + 1) / (denominator + 1)
    return coef

@BBOX_ASSIGNERS.register_module()
class HungarianMatcherIFC(torch.nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_dice: float = 1,
        num_classes: int = 80,
        n_future: int = 5
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the sigmoid_focal error of the masks in the matching cost
            cost_dice: This is the relative weight of the dice loss of the masks in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_dice = cost_dice
        assert cost_class != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_classes = num_classes
        self.num_cum_classes = [0] + \
            np.cumsum(np.array(num_classes) + 1).tolist()
        self.n_future = n_future
        
    @torch.no_grad()
    def forward(self, outputs, targets):
        # We flatten to compute the cost matrices in a batch
        # torch.Size([1, 300, 101])
        out_prob = outputs["pred_logits"].softmax(-1)
        out_mask = outputs["pred_masks"]  # torch.Size([1, 300, 5, 50, 50])
        B, Q, T, s_h, s_w = out_mask.shape
        t_h, t_w = targets[0]["match_masks"].shape[-2:]

        if (s_h, s_w) != (t_h, t_w):
            out_mask = out_mask.reshape(B, Q*T, s_h, s_w)
            out_mask = torch.nn.functional.interpolate(out_mask, size=(
                t_h, t_w), mode="bilinear", align_corners=False)
            out_mask = out_mask.view(B, Q, T, t_h, t_w)

        indices = []
        for b_i in range(B):
            # tensor([0, 1, 2], device='cuda:0')
            b_tgt_ids = targets[b_i]["labels"] #19 tensor([ 1,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 16, 17, 18, 19, 21, 22,23, 24, 25, 26, 28, 31, 33, 34, 35], device='cuda:0')
            #torch.Size([27])
            b_out_prob = out_prob[b_i]  # 

            cost_class = b_out_prob[:, b_tgt_ids]  # torch.Size([300, 27])

            # GxTxHxW torch.Size([27, 5, 50, 50])
            b_tgt_mask = targets[b_i]["match_masks"]
            b_out_mask = out_mask[b_i]  # QxTxHxW torch.Size([300, 5, 50, 50])

            # Compute the dice coefficient cost between masks
            # The 1 is a constant that doesn't change the matching as cost_class, thus omitted.
            
            cost_dice = dice_coef(
                b_out_mask, b_tgt_mask
            ).to(cost_class)  # torch.Size([300, 27])

            # Final cost matrix
            #! CHECK INDEX OF BATCH DIMENSION MIGHT NEED TO TRANSPOSE??
            C = self.cost_dice * cost_dice + self.cost_class * cost_class
            
            #print(f"Predicted {C.shape} masks for batch {b_i+1}")
            C = C.cpu() 
            assignment = linear_sum_assignment(C, maximize=True) 
            indices.append(assignment)
            # print(
            #     f"Predicted {C.cpu().shape} masks for batch {b_i+1}, current indices {assignment}")
            
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

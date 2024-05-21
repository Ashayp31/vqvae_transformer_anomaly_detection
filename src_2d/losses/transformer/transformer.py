from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from src.handlers.general import TBSummaryTypes


class CELoss(_Loss):
    def __init__(
        self, weight=None, size_average: bool = None, reduce: bool = None, reduction: str = "mean"
    ):
        super(CELoss, self).__init__(size_average, reduce, reduction)

        try:
            assert reduction in ["sum", "mean"]
        except AssertionError:
            raise ValueError("Reduction must be either 'sum' or 'mean'")

        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict()}
        self._weight = weight

    def forward(self, y_pred: torch.Tensor, y: List[torch.Tensor]) -> torch.Tensor:

        mask = y[1]
        y = y[0].long()
        y_pred = y_pred.float()
        if mask is None:
            loss = F.cross_entropy(
                input=y_pred, target=y, reduction=self.reduction, weight=self._weight
            )
        else:
            mask = mask.int()

            none_reduced_loss = F.cross_entropy(
                input=y_pred, target=y, reduction='none', weight=self._weight
            )
            loss = torch.div(torch.sum(mask*none_reduced_loss), torch.sum(mask))

        self.summaries[TBSummaryTypes.SCALAR]["Loss-CE-Prediction"] = loss

        return loss

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries

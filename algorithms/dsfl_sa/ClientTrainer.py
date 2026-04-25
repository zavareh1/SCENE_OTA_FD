# File: algorithms/dsfl_sa/ClientTrainer.py
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from algorithms.BaseClientTrainer import BaseClientTrainer

__all__ = ["ClientTrainer"]

class ClientTrainer(BaseClientTrainer):
    """
    Client:
      - supervised local train via BaseClientTrainer.train()
      - predict probs on server-specified unlabeled indices
      - distill on the same unlabeled subset using averaged soft-labels
    """

    def __init__(self, algo_params, model, local_epochs, device, num_classes):
        super().__init__(algo_params, model, local_epochs, device, num_classes)
        p = self.algo_params
        self.unlabeled_batch_size  = int(p.get("unlabeled_batch_size", 256))
        self.distill_epochs_client = int(p.get("client_distill_epochs", 1))
        self.unlabeled_dataset     = None  # set by Server

    @torch.no_grad()
    def compute_probs_on_indices(self, or_indices):
        if or_indices is None or len(or_indices) == 0:
            # Safe empty return for server averaging path
            return torch.empty((0, self.num_classes), dtype=torch.float32)
            
        """Return (B,C) probabilities on Do[or_indices] in the same order."""
        assert self.unlabeled_dataset is not None, "unlabeled_dataset not set on client."
        subset = Subset(self.unlabeled_dataset, or_indices)
        nw = 0
        try:
        # server attaches data_distributed; if not accessible, keep 0
            if hasattr(self, "data_distributed"):
                nw = self.data_distributed.get("num_workers", 0)
        except Exception:
            pass
        loader = DataLoader(
            subset,
            batch_size=self.unlabeled_batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
        )
        self.model.eval()
        device = self.device
        all_probs = []
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.detach().cpu())
        return torch.cat(all_probs, dim=0)

    def distill_on_indices(self, or_indices, avg_probs_on_or):
        if or_indices is None or len(or_indices) == 0:
            return
        if avg_probs_on_or is None or avg_probs_on_or.numel() == 0:
            return
        """KL(student || avg_probs) on Do[or_indices]."""
        assert self.unlabeled_dataset is not None, "unlabeled_dataset not set on client."
        subset = Subset(self.unlabeled_dataset, or_indices)
        nw = 0
        try:
        # server attaches data_distributed; if not accessible, keep 0
            if hasattr(self, "data_distributed"):
                nw = self.data_distributed.get("num_workers", 0)
        except Exception:
            pass
        loader = DataLoader(
            subset,
            batch_size=self.unlabeled_batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
        )
        self.model.train()
        device = self.device
        for _ in range(self.distill_epochs_client):
            it = 0
            for x, _ in loader:
                b0 = it * loader.batch_size
                b1 = min((it + 1) * loader.batch_size, len(or_indices))
                target_soft = avg_probs_on_or[b0:b1].to(device)
                x = x.to(device, non_blocking=True)

                self.optimizer.zero_grad()
                logits = self.model(x)
                logp = torch.log_softmax(logits, dim=1)
                loss =  F.kl_div(logp, target_soft, reduction="batchmean")
                loss.backward()
                self.optimizer.step()
                it += 1

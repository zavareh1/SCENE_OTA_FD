# File: algorithms/dsfl_sa/Server.py
import copy
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

from algorithms.BaseServer import BaseServer
from algorithms.dsfl_sa.ClientTrainer import ClientTrainer

__all__ = ["Server"]

class Server(BaseServer):
    """
    DS-FL (soft-label averaging, no temperature averaging on server).
    """

    def __init__(self, algo_params, model, data_distributed, optimizer, scheduler, **kwargs):
        super().__init__(algo_params, model, data_distributed, optimizer, scheduler, **kwargs)
        p = self.algo_params

        # --- shared unlabeled pool (Do) must be provided by datasetter ---
        self.unlabeled_dataset = self.data_distributed.get("unlabeled", None)
        assert self.unlabeled_dataset is not None, \
            "[dsfl_sa] missing unlabeled_dataset; ensure data_distributer returns 'unlabeled'."

        # --- DS-FL hyperparams ---
        self.num_classes            = getattr(self, "num_classes", 10)
        self.unlabeled_pool_size    = int(p.get("unlabeled_pool_size", 20000))
        self.unlabeled_per_round    = int(p.get("unlabeled_per_round", 1000))
        self.unlabeled_batch_size   = int(p.get("unlabeled_batch_size", 256))
        self.client_distill_epochs  = int(p.get("client_distill_epochs", 1))
        self.server_distill_epochs  = int(p.get("server_distill_epochs", 1))

        # Persisted, per-client trainer (labels-only style)
        self.client = ClientTrainer(
            algo_params=self.algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
        )
        self.client.unlabeled_dataset = self.unlabeled_dataset

        # Reuse server optimizer for distillation
        self.server_optimizer = self.optimizer

        print("\n>>> DS-FL (soft-label averaging) Server initialized.\n")

    # ------------------ DS-FL utils ------------------

    @torch.no_grad()
    def _choose_unlabeled_indices(self, round_idx: int):
        """Deterministic per-round sampling from Do."""
        N = len(self.unlabeled_dataset)
        if N == 0:
            return []
        g = torch.Generator(); g.manual_seed(round_idx)
        B = min(self.unlabeled_per_round, N)
        return torch.randperm(N, generator=g)[:B].tolist()

    @torch.no_grad()
    def _avg_client_probs(self, probs_list):
        """Average (B,C) probability tensors from sampled clients."""
        if len(probs_list) == 0:
            raise RuntimeError("[dsfl_sa] No client probs received for averaging.")
        return torch.stack(probs_list, dim=0).mean(dim=0)

    def _server_distill_step(self, or_indices, avg_probs):
        if or_indices is None or len(or_indices) == 0:
            return
        if avg_probs is None or avg_probs.numel() == 0:
            return
        """Server distillation on Do[or_indices] with soft targets avg_probs."""
        device = self.device
        self.model.train()
        nw = self.data_distributed.get("num_workers", 0)
        loader = DataLoader(
            Subset(self.unlabeled_dataset, or_indices),
            batch_size=self.unlabeled_batch_size,
            shuffle=False,
            num_workers=nw,          # keep simple & stable
            pin_memory=True,
        )

        it = 0
        for _ in range(self.server_distill_epochs):
            it = 0
            for x, _ in loader:
                b0 = it * loader.batch_size
                b1 = min((it + 1) * loader.batch_size, len(or_indices))
                target_soft = avg_probs[b0:b1].to(device)
                x = x.to(device, non_blocking=True)

                self.server_optimizer.zero_grad()
                logits = self.model(x)
                logp = F.log_softmax(logits, dim=1)
                loss = F.kl_div(logp, target_soft, reduction="batchmean")
                loss.backward()
                self.server_optimizer.step()
                it += 1

    # ------------------ DS-FL loop ------------------

    def run(self):
        """Labels-only loop: average client soft-labels on Do; distill server+clients; report server accuracy."""
        self._print_start()

        for round_idx in range(self.n_rounds):
            t0 = time.time()

            # 1) sample clients & supervised local training (persist states)
            sampled = self._client_sampling(round_idx)
            self.server_results["client_history"].append(sampled)

            round_results, client_sizes = {}, []
            for client_idx in sampled:
                self._set_client_data(client_idx)

                # load persisted state if available
                w = getattr(self, "client_weights", {}).get(client_idx, None)
                if w is not None:
                    self.client.model.load_state_dict(w)
                else:
                # ensure client starts from *current* global model
                    self.client.model.load_state_dict(self.model.state_dict())
                opt_state = getattr(self, "client_opt_states", {}).get(client_idx, None)
                if opt_state is not None:
                    self.client.optimizer.load_state_dict(opt_state)
                else:
                # critical: bootstrap client optimizer from server so lr/momentum are non-zero
                    self.client.optimizer.load_state_dict(self.server_optimizer.state_dict())

                local_results, local_size = self.client.train()
                round_results = self._results_updater(round_results, local_results)
                client_sizes.append(local_size)

                # persist state for next rounds
                if not hasattr(self, "client_weights"):
                    self.client_weights = {}
                if not hasattr(self, "client_opt_states"):
                    self.client_opt_states = {}
                self.client_weights[client_idx]    = copy.deepcopy(self.client.model.state_dict())
                self.client_opt_states[client_idx] = copy.deepcopy(self.client.optimizer.state_dict())

            # 2) choose unlabeled Do subset for this round
            or_indices = self._choose_unlabeled_indices(round_idx)
            
            
            if len(or_indices) == 0:
            # No unlabeled data this round: skip probs/distill, just evaluate global model
                test_acc = self._evaluate_global_model()
                self.server_results["test_accuracy"].append(test_acc)
                self._print_stats_labels_only(round_results, round_idx, time.time() - t0)
                continue

            # 3) collect client probabilities on Do[o_r]
            probs_list = []
            for client_idx in sampled:
                if client_idx in self.client_weights:
                    self.client.model.load_state_dict(self.client_weights[client_idx])
                self._set_client_data(client_idx)
                probs = self.client.compute_probs_on_indices(or_indices)  # CPU tensor (B,C)
                probs_list.append(probs)

            # 4) average probs and distill server global model
            device = self.device
            avg_probs = self._avg_client_probs([p.to(device) for p in probs_list])  # (B,C)
            self._server_distill_step(or_indices, avg_probs)

            # 5) broadcast avg_probs; clients distill locally on the same Do[o_r]
            for client_idx in sampled:
                if client_idx in self.client_weights:
                    self.client.model.load_state_dict(self.client_weights[client_idx])
                self._set_client_data(client_idx)
                self.client.distill_epochs_client = self.client_distill_epochs
                self.client.distill_on_indices(or_indices, avg_probs.detach().cpu())
                # persist post-distill
                self.client_weights[client_idx]    = copy.deepcopy(self.client.model.state_dict())
                self.client_opt_states[client_idx] = copy.deepcopy(self.client.optimizer.state_dict())

            # 6) evaluate server global model on test set (paper-style reporting)
            test_acc = self._evaluate_global_model()
            self.server_results["test_accuracy"].append(test_acc)

            # 7) print/log
            self._print_stats_labels_only(round_results, round_idx, time.time() - t0)

        self._print_end()

    # ---- helpers (shared helper style) ----

    def _evaluate_global_model(self):
        from algorithms.measures import evaluate_model
        self.model.eval()
        return evaluate_model(self.model, self.testloader, device=self.device)

    def _print_stats_labels_only(self, round_results, round_idx, elapsed):
        import wandb
        tr = np.array(round_results.get("train_acc", []), dtype=float)
        te = np.array(round_results.get("test_acc", []), dtype=float)
        print(
            "[Round {}/{}] Elapsed {}s".format(
                round_idx + 1, self.n_rounds, round(elapsed, 1)
            )
        )
        if tr.size > 0:
            print("  mean local train acc: {:.2f}%".format(100.0 * tr.mean()))
            wandb.log({"local_train_acc": tr.mean()}, step=round_idx)
        if te.size > 0:
            print("  mean local test  acc: {:.2f}%".format(100.0 * te.mean()))
            wandb.log({"local_test_acc": te.mean()}, step=round_idx)
        # server/global accuracy is tracked in self.server_results["test_accuracy"]
        # --- NEW: print the latest server/global accuracy for this round ---
        srv = self.server_results.get("test_accuracy", [])
        if len(srv) > 0:
            last = float(srv[-1])
            print("  server/global test acc: {:.2f}%".format(100.0 * last))
            try:
                wandb.log({"server_test_acc": last}, step=round_idx)
            except Exception:
                pass
    def _print_end(self):
    # Final summary; mirror the minimal style used in other algos
        if "test_accuracy" in self.server_results and len(self.server_results["test_accuracy"]) > 0:
            last = float(self.server_results["test_accuracy"][-1])
            print(f"\n[Done] Final server/global test acc: {100.0 * last:.2f}%")
        else:
            print("\n[Done] Training finished.")

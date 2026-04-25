# File: algorithms/dsfl_avg_ota_zero_shot/Server.py

import copy
import os
import time
import numpy as np
import torch

from algorithms.dsfl_sa_ota_zero_shot.Server import (
    Server as OTAZeroShotServer,
)

__all__ = ["Server"]


class Server(OTAZeroShotServer):
    """
    dsfl_avg_ota_zero_shot:

      Same client-side behavior and OTA channel model as dsfl_sa_ota_zero_shot,
      but for each unlabeled budget U we run two zero-shot distillations on the
      *same* subset of shared unlabeled data Do:

        (i) OTA-based aggregation of soft labels (AirComp, as in dsfl_sa_ota_zero_shot)
        (ii) Plain weighted averaging of client soft labels (noise-free baseline,
             with weights proportional to the number of samples per sampled client)

      The U-vs-accuracy curves are stored under:
        - self.server_results["unlabeled_counts"]
        - self.server_results["test_accuracy_ota"]
        - self.server_results["test_accuracy_avg"]

      and also written to a small text file
        dsfl_avg_ota_zero_shot_curve.txt
      in the current working directory.
    """

    def _print_stats_labels_only(self, round_results, round_idx, elapsed):
        """Override parent to avoid wandb and print sample-size-weighted stats."""
        tr = np.array(round_results.get("train_acc", []), dtype=float)
        te = np.array(round_results.get("test_acc", []), dtype=float)
        ns = np.array(getattr(self, "sampled_client_sizes", []), dtype=float)

        print(
            "[Round {}/{}] Elapsed {}s".format(
                round_idx + 1, self.n_rounds, round(elapsed, 1)
            )
        )

        w = None
        if ns.size > 0 and np.isfinite(ns).all() and ns.sum() > 0:
            w = ns / ns.sum()

        if tr.size > 0:
            tr_mean = float(np.sum(w * tr)) if (w is not None and tr.size == w.size) else float(tr.mean())
            print("  weighted local train acc: {:.2f}%".format(100.0 * tr_mean))

        if te.size > 0:
            te_mean = float(np.sum(w * te)) if (w is not None and te.size == w.size) else float(te.mean())
            print("  weighted local test  acc: {:.2f}%".format(100.0 * te_mean))

        srv = self.server_results.get("test_accuracy", [])
        if len(srv) > 0:
            last = float(srv[-1])
            print("  server/global test acc: {:.2f}%".format(100.0 * last))

    def _avg_client_probs_weighted(self, probs_list_device):
        """
        Noise-free weighted averaging of client soft labels.

        probs_list_device: list of tensors, each [B, C] on self.device.

        Weights are proportional to the number of training samples for each
        sampled client (self.sampled_client_sizes), then normalized to sum to 1.
        """
        n_clients = len(probs_list_device)

        # Convert sampled client sizes to a numpy array
        sizes = np.array(self.sampled_client_sizes, dtype=float)

        # If for some reason shapes do not match, fall back to uniform weights
        if sizes.shape[0] != n_clients:
            sizes = np.ones(n_clients, dtype=float)

        # Avoid non-positive sizes
        sizes[sizes <= 0] = 1.0

        # Normalize to sum to 1
        weights = sizes / sizes.sum()

        # Convert to tensor on device and broadcast over [B, C]
        dtype = probs_list_device[0].dtype
        w_t = torch.tensor(weights, device=self.device, dtype=dtype).view(-1, 1, 1)

        # Stack client probabilities: [N, B, C]
        stacked = torch.stack(probs_list_device, dim=0)

        # Weighted mean across client dimension (0) → [B, C]
        weighted = (w_t * stacked).sum(dim=0)

        # Numerical safety (should already be valid probabilities)
        weighted = torch.clamp(weighted, min=1e-12)
        weighted = weighted / weighted.sum(dim=1, keepdim=True)

        return weighted

    def run(self):
        """Zero-shot loop with sweep over different unlabeled batch sizes, running
        both OTA and weighted-averaging variants on the same unlabeled batches."""
        self._print_start()

        # -----------------------------
        # Save initial global weights and optimizer state
        # -----------------------------
        init_global_state = copy.deepcopy(self.model.state_dict())
        # server_optimizer is created in the parent __init__
        init_opt_state = copy.deepcopy(self.server_optimizer.state_dict())

        # -----------------------------
        # Stage 1: client supervised pretraining (same logic as dsfl_sa_ota_zero_shot)
        # -----------------------------
        t0 = time.time()
        round_idx = 0

        sampled = self._client_sampling(round_idx)
        self.sampled_clients = sampled
        self.server_results["client_history"].append(sampled)

        # Temporarily override client.local_epochs
        old_local_epochs = self.client.local_epochs
        self.client.local_epochs = self.zero_shot_client_epochs

        round_results, client_sizes = {}, []
        client_train_accs, client_test_accs = [], []

        for client_idx in sampled:
            self._set_client_data(client_idx)

            # Initialize client state from current global + optimizer
            self.client.model.load_state_dict(self.model.state_dict())
            self.client.optimizer.load_state_dict(self.server_optimizer.state_dict())

            local_results, local_size = self.client.train()
            round_results = self._results_updater(round_results, local_results)
            client_sizes.append(local_size)

            # Collect per-client metrics if available
            if "train_acc" in local_results:
                client_train_accs.append(local_results["train_acc"])
            if "test_acc" in local_results:
                client_test_accs.append(local_results["test_acc"])

            # Persist client state (for later labeling; clients become frozen teachers)
            if not hasattr(self, "client_weights"):
                self.client_weights = {}

            self.client_weights[client_idx] = copy.deepcopy(self.client.model.state_dict())

        self.client.local_epochs = old_local_epochs
        # Store sampled client sizes (used later for weighting)
        self.sampled_client_sizes = client_sizes

        # Sample-size-weighted client train/test accuracy after pretraining (if stats exist)
        ns = np.array(client_sizes, dtype=float)
        w = None
        if ns.size > 0 and np.isfinite(ns).all() and ns.sum() > 0:
            w = ns / ns.sum()

        if len(client_train_accs) > 0:
            tr = np.array(client_train_accs, dtype=float)
            mean_train = float(np.sum(w * tr)) if (w is not None and tr.size == w.size) else float(np.mean(tr))
            self.server_results.setdefault("client_mean_train_acc_pretrain", []).append(mean_train)
            print(f"[Zero-Shot] Mean client train acc after pretraining: {100.0 * mean_train:.2f}%")

        if len(client_test_accs) > 0:
            te = np.array(client_test_accs, dtype=float)
            mean_test = float(np.sum(w * te)) if (w is not None and te.size == w.size) else float(np.mean(te))
            self.server_results.setdefault("client_mean_test_acc_pretrain", []).append(mean_test)
            print(f"[Zero-Shot] Mean client test acc after pretraining:  {100.0 * mean_test:.2f}%")

        # Evaluate initial server model (raw) before any unlabeled distill
        self.model.load_state_dict(init_global_state)
        init_acc = self._evaluate_global_model()
        # Keep raw accuracy under its own key; do NOT overload "test_accuracy"
        self.raw_server_test_accuracy = init_acc
        print(f"[Zero-Shot] Initial server/global test acc (pre-distill, raw): {100.0 * init_acc:.2f}%")

        # Log local stats; note that server/global accuracy is taken from test_accuracy
        self._print_stats_labels_only(round_results, round_idx, time.time() - t0)

        # -----------------------------
        # Stage 2: sweep over different unlabeled batch sizes
        # -----------------------------
        N_unlabeled = len(self.unlabeled_dataset)
        if N_unlabeled == 0:
            print("[Zero-Shot] No unlabeled data available; skipping distillation sweep.")
            self._print_end()
            self._save_results_to_excel()
            return

        self.server_results.setdefault("unlabeled_counts", [])
        self.server_results.setdefault("test_accuracy_ota", [])
        self.server_results.setdefault("test_accuracy_avg", [])

        rng = self.rng  # OTA RNG for unlabeled sampling

        for sweep_idx, U in enumerate(self.eval_unlabeled_counts, start=1):
            t_step = time.time()
            B = min(int(U), int(self.unlabeled_pool_size), int(N_unlabeled))
            if B <= 0:
                print(f"[Zero-Shot Sweep] U={U} -> effective batch size B=0; skipping.")
                continue

            # Choose unlabeled indices exactly once; same batch is used for both variants
            if B >= N_unlabeled:
                or_indices = np.arange(N_unlabeled).tolist()
            else:
                or_indices = rng.choice(N_unlabeled, size=B, replace=False).tolist()

            print(
                f"\n[Zero-Shot Sweep] Setting {sweep_idx}/{len(self.eval_unlabeled_counts)}: "
                f"U={U}, effective batch size B={B}"
            )

            # 1) Clients label the chosen unlabeled batch (FROZEN models)
            probs_list = []
            for client_idx in sampled:
                # Reload frozen client weights
                if hasattr(self, "client_weights") and client_idx in self.client_weights:
                    self.client.model.load_state_dict(self.client_weights[client_idx])
                else:
                    self.client.model.load_state_dict(init_global_state)

                self._set_client_data(client_idx)
                probs = self.client.compute_probs_on_indices(or_indices)  # CPU (B,C)
                probs_list.append(probs)

            device = self.device
            probs_list_device = [p.to(device) for p in probs_list]

            # ---- Branch A: OTA aggregation (dsfl_sa_ota_zero_shot) ----
            self.model.load_state_dict(init_global_state)
            self.server_optimizer.load_state_dict(init_opt_state)

            # This calls OTA-aware _avg_client_probs from the parent class
            avg_probs_ota = self._avg_client_probs(probs_list_device)
            self._server_distill_step(or_indices, avg_probs_ota)
            acc_ota = self._evaluate_global_model()
            self.server_results["test_accuracy_ota"].append(acc_ota)
            print(f"  [OTA]    Server test acc AFTER distill (U={U}): {100.0 * acc_ota:.2f}%")

            # ---- Branch B: weighted averaging baseline (noise-free)
            #      weights ∝ number of samples per sampled client
            self.model.load_state_dict(init_global_state)
            self.server_optimizer.load_state_dict(init_opt_state)

            avg_probs_avg = self._avg_client_probs_weighted(probs_list_device)
            self._server_distill_step(or_indices, avg_probs_avg)
            acc_avg = self._evaluate_global_model()
            self.server_results["test_accuracy_avg"].append(acc_avg)
            print(f"  [Plain] Server test acc AFTER distill (U={U}): {100.0 * acc_avg:.2f}%")

            # Shared bookkeeping (same effective unlabeled budget for both variants)
            self.server_results["unlabeled_counts"].append(B)
            elapsed = time.time() - t_step
            print(f"  [Zero-Shot Sweep] Elapsed {elapsed:.1f}s for U={U}")

        # --- Save side-by-side curve to a small .txt file for this trial ---
        try:
            us = self.server_results.get("unlabeled_counts", [])
            acc_ota = self.server_results.get("test_accuracy_ota", [])
            acc_avg = self.server_results.get("test_accuracy_avg", [])
            if us and acc_ota and acc_avg:
                fname = "dsfl_avg_ota_zero_shot_curve.txt"
                save_path = os.path.join(os.getcwd(), fname)
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write("U,acc_ota,acc_avg\n")
                    for k, a_ota, a_avg in zip(us, acc_ota, acc_avg):
                        f.write(f"{int(k)},{float(a_ota):.6f},{float(a_avg):.6f}\n")
                print(f"[dsfl_avg_ota_zero_shot] Saved per-trial OTA vs AVG curve to: {save_path}")
        except Exception as e:
            print(f"[dsfl_avg_ota_zero_shot] Warning: failed to save curve txt ({e}).")

        # For main.py and Excel-saving compatibility:
        # expose OTA curve also under the generic key 'test_accuracy'
        self.server_results["test_accuracy"] = list(self.server_results.get("test_accuracy_ota", []))

        self._print_end()
        self._save_results_to_excel()

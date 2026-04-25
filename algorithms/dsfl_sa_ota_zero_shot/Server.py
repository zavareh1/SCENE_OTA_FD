# File: algorithms/dsfl_sa_ota_zero_shot/Server.py
import copy
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

from algorithms.BaseServer import BaseServer
from algorithms.dsfl_sa.ClientTrainer import ClientTrainer
from algorithms.dsfl_sa.Server import Server as DSFLSAServer  # reuse dsfl_sa behavior

__all__ = ["Server"]

# --- Aggregator: pilot-free self-centering noncoherent OTA
def noncoherent_self_center_ota(q_list,            # list of [K]-soft labels per device for one exchange
                                omega_list,        # nonneg. weights ω_i (e.g., |B_i|)
                                beta_list,         # large-scale gains β_i (pathloss/shadowing)
                                S=1,               # repetitions
                                M=4,               # PS antennas
                                sigma2=1e-3, rho=1.0,  # per-RE per-antenna noise variance (effective)
                                rng=None):
    """Return q_hat: unbiased estimate of the global soft label q̄ using self-centering.

    Shapes:
      q_list: list of arrays, shape [K]; stacked -> [N, K]
      omega_list: [N]
      beta_list:  [N]
    """
    rng = np.random.default_rng() if rng is None else rng
    q = np.stack(q_list, axis=0)      # [N, K]
    N, K = q.shape
    omega = np.asarray(omega_list).reshape(N, 1)
    beta  = np.asarray(beta_list).reshape(N, 1)

    # Constant-power energy mapping with coarse pathloss inversion:
    #   E_{i,c} = η_i q_{i,c}, where η_i = ρ ω_i / β_i
    eta = (rho * omega) / np.clip(beta, 1e-12, None)  # [N,1]
    E   = eta * q                                     # [N, K]

    Y = np.zeros(K, dtype=float)
    for _ in range(S):
        # small-scale fading: CN(0,1) per device-antenna tap
        H = (rng.normal(size=(M, N)) + 1j * rng.normal(size=(M, N))) / np.sqrt(2)
        H = H * np.sqrt(beta.T)  # apply large-scale gains sqrt(β_i)

        for c in range(K):
            s = (H @ np.sqrt(E[:, c].reshape(N, 1))).squeeze(-1)  # [M]
            # complex AWGN per antenna BEFORE squaring
            n = (rng.normal(size=M) + 1j * rng.normal(size=M)) * np.sqrt(sigma2 / 2.0)
            y = s + n
            Y[c] += np.sum(np.abs(y) ** 2)  # energy over antennas

    # Self-centering across classes, then normalize
    SM = float(S * M)
    Y_bar = float(np.mean(Y))
    q_hat = (Y - Y_bar) / (SM * rho) + (1.0 / K)

    q_hat = np.clip(q_hat, 1e-12, 1.0)
    q_hat /= q_hat.sum()
    return q_hat


class Server(DSFLSAServer):
    """
    dsfl_sa_ota_zero_shot:

      Stage 1 (pretrain): clients do supervised local training for
        `zero_shot_client_epochs` epochs. We can log mean client train/test acc.
      Stage 2 (zero-shot distill sweep): clients are frozen; for each K in
        eval_unlabeled_counts the server selects K unlabeled points from Do,
        clients label them, OTA aggregates soft labels, the server distills
        starting from a raw model, and we report test accuracy.
    """

    def __init__(self, algo_params, model, data_distributed, optimizer, scheduler, **kwargs):
        super().__init__(algo_params, model, data_distributed, optimizer, scheduler, **kwargs)
        p = self.algo_params

        # --- OTA hyperparams (unchanged) ---
        self.ota_S = int(p.get("ota_S", 1))
        self.ota_M = int(p.get("ota_M", 4))
        self.ota_target_snr_db = float(p.get("ota_target_snr_db", 20.0))
        # self.ota_rho = float(p.get("ota_rho", 1.0))
        self.rng = np.random.default_rng(int(p.get("ota_seed", 123)))  # reproducible

        # Per-device power caps P_i ~ U[0.5, 1.5], like make_caps()
        self.ota_caps_low  = float(p.get("ota_caps_low", 0.5))
        self.ota_caps_high = float(p.get("ota_caps_high", 1.5))

        n_clients = self.n_clients

        # Pathloss β_i: distance-based + lognormal, normalized to mean 1 (like make_pathloss)
        d = self.rng.uniform(5.0, 50.0, size=n_clients)
        alpha = 3.5
        shadow_db = self.rng.normal(loc=0.0, scale=8.0, size=n_clients)
        beta = (d ** (-alpha)) * (10.0 ** (shadow_db / 10.0))
        self.ota_beta_full = beta / np.mean(beta)

        # Power caps P_i
        self.ota_P_full = self.rng.uniform(self.ota_caps_low,
                                           self.ota_caps_high,
                                           size=n_clients)

        # Weights ω_i: use client datasizes, normalized to sum 1 (paper convention)
        client_sizes = np.array(
            [self.data_distributed["local"][i]["datasize"] for i in range(n_clients)],
            dtype=float
            )
        client_sizes[client_sizes <= 0] = 1.0

        total_size = np.sum(client_sizes)
        if (not np.isfinite(total_size)) or total_size <= 0:
            self.ota_omega_full = np.ones(n_clients, dtype=float) / float(n_clients)
        else:
            self.ota_omega_full = client_sizes / total_size

        print("\n>>> DS-FL OTA server initialized with realistic β, P, ω.\n")

        # --- shared unlabeled pool (Do) must be provided by datasetter ---
        self.unlabeled_dataset = self.data_distributed.get("unlabeled", None)
        assert self.unlabeled_dataset is not None, \
            "[dsfl_sa_ota_zero_shot] missing unlabeled_dataset; ensure data_distributer returns 'unlabeled'."

        # --- DS-FL hyperparams (unchanged base) ---
        self.num_classes            = getattr(self, "num_classes", 10)
        self.unlabeled_pool_size    = int(p.get("unlabeled_pool_size", 20000))
        self.unlabeled_per_round    = int(p.get("unlabeled_per_round", 1000))
        self.unlabeled_batch_size   = int(p.get("unlabeled_batch_size", 256))
        self.client_distill_epochs  = int(p.get("client_distill_epochs", 1))
        self.server_distill_epochs  = int(p.get("server_distill_epochs", 1))

        # NEW: how many epochs to train clients in pretraining stage
        self.zero_shot_client_epochs = int(
            p.get("zero_shot_client_epochs", self.local_epochs)
        )

        # NEW: list of unlabeled counts to sweep over
        eval_counts = p.get("eval_unlabeled_counts", None)
        if eval_counts is None:
            # fallback: just use unlabeled_per_round as a single setting
            self.eval_unlabeled_counts = [self.unlabeled_per_round]
        elif isinstance(eval_counts, (list, tuple)):
            self.eval_unlabeled_counts = [int(x) for x in eval_counts]
        else:
            self.eval_unlabeled_counts = [int(eval_counts)]

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

        print(
            "\n>>> DS-FL OTA ZERO-SHOT server initialized "
            f"(zero_shot_client_epochs={self.zero_shot_client_epochs}, "
            f"eval_unlabeled_counts={self.eval_unlabeled_counts}).\n"
        )

    def _device_weights(self, probs_list):
        """
        Return ω_i for the *sampled* clients, in the same order as probs_list.
        We assume self.sampled_clients is set in run() before calling _avg_client_probs.
        """
        idxs = np.array(self.sampled_clients, dtype=int)
        w = self.ota_omega_full[idxs].astype(float)
        if (not np.isfinite(w).all()) or (w.sum() <= 0):
            w = np.ones_like(w, dtype=float) / float(len(w))
        else:
            w = w / np.sum(w)
        return w
    def _device_betas(self, probs_list):
        """
        Return β_i for the sampled clients (pathloss/shadowing), matching synthetic make_pathloss.
        """
        idxs = np.array(self.sampled_clients, dtype=int)
        b = self.ota_beta_full[idxs].astype(float)
        if not np.isfinite(b).all() or np.any(b <= 0):
            b = np.ones_like(b, dtype=float)
        return b

    def _sigma2_from_snr(self, omega, beta, q_list, rho):
        # Reuse your mapping; implemented here for convenience
        q = np.stack(q_list, axis=0)  # [N,K]
        N, K = q.shape
        omega = omega.reshape(N, 1)
        beta  = beta.reshape(N, 1)
        eta = (rho * omega) / np.clip(beta, 1e-12, None)
        E = eta * q
        per_class = np.sum(beta * E, axis=0)
        mu = float(np.mean(per_class))
        snr_lin = 10.0 ** (self.ota_target_snr_db / 10.0)
        return mu / snr_lin

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
        """
        OTA per-example aggregation with the same (ω, β, P, ρ*) model as the
        synthetic script.
        probs_list: list of [B,C] torch tensors from sampled clients
                    (same B, order matches self.sampled_clients).
        """
        if len(probs_list) == 0:
            raise RuntimeError("[dsfl_sa_ota_zero_shot] No client probs received for OTA averaging.")

        B, C = probs_list[0].shape
        device = probs_list[0].device

        # ω_i and β_i for sampled clients
        omega = self._device_weights(probs_list)   # shape [N]
        beta  = self._device_betas(probs_list)     # shape [N]

        # Per-client caps P_i for sampled clients (like make_caps)
        idxs = np.array(self.sampled_clients, dtype=int)
        P = self.ota_P_full[idxs].astype(float)

        # min-ρ rule: ρ* = min_i β_i P_i / ω_i
        rho_vec = (beta.flatten() * P.flatten()) / omega.flatten()
        rho_star = float(np.min(rho_vec))

        rows = []
        for b in range(B):
            # q_list for this example: N clients × C classes
            q_list_b = [p[b].detach().cpu().numpy() for p in probs_list]

            # sigma2 from target SNR, using same formula as sigma2_for_snr_db
            sigma2 = self._sigma2_from_snr(
                omega=omega.copy(),
                beta=beta.copy(),
                q_list=q_list_b,
                rho=rho_star,
            )

            q_hat = noncoherent_self_center_ota(
                q_list=q_list_b,
                omega_list=omega,
                beta_list=beta,
                S=self.ota_S,
                M=self.ota_M,
                sigma2=sigma2,
                rho=rho_star,
                rng=self.rng,
            )
            rows.append(torch.from_numpy(q_hat).to(device))

        return torch.stack(rows, dim=0)  # [B,C]

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

    # ------------------ ZERO-SHOT loop with sweep ------------------

    def run(self):
        """
        Zero-shot loop with sweep over different unlabeled batch sizes:

          1) Supervised client pretraining for zero_shot_client_epochs.
             - log mean local train/test acc if available.
          2) For each K in eval_unlabeled_counts:
             - reset server model to initial raw state,
             - pick K unlabeled samples (capped by pool size),
             - frozen clients label them via OTA,
             - server distills on those K,
             - evaluate server/global test accuracy.
        """
        self._print_start()

        # -----------------------------
        # Stage 0: remember initial global weights (raw model)
        # -----------------------------
        init_global_state = copy.deepcopy(self.model.state_dict())

        # -----------------------------
        # Stage 1: client supervised pretraining
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

            # Initialize client state:
            #   - model from current global
            #   - optimizer from server optimizer
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
            if not hasattr(self, "client_opt_states"):
                self.client_opt_states = {}

            self.client_weights[client_idx] = copy.deepcopy(self.client.model.state_dict())
            self.client_opt_states[client_idx] = copy.deepcopy(self.client.optimizer.state_dict())

        self.client.local_epochs = old_local_epochs
        self.sampled_client_sizes = client_sizes

        # Mean client train/test accuracy after pretraining (if stats exist)
        if len(client_train_accs) > 0:
            mean_train = float(np.mean(client_train_accs))
            self.server_results.setdefault("client_mean_train_acc_pretrain", []).append(mean_train)
            print(f"[Zero-Shot] Mean client train acc after pretraining: {100.0 * mean_train:.2f}%")
        if len(client_test_accs) > 0:
            mean_test = float(np.mean(client_test_accs))
            self.server_results.setdefault("client_mean_test_acc_pretrain", []).append(mean_test)
            print(f"[Zero-Shot] Mean client test acc after pretraining:  {100.0 * mean_test:.2f}%")

        # Evaluate initial server model (raw) before any unlabeled distill
        self.model.load_state_dict(init_global_state)
        init_acc = self._evaluate_global_model()
        self.server_results.setdefault("test_accuracy", []).append(init_acc)
        print(f"[Zero-Shot] Initial server/global test acc (pre-distill, raw): {100.0 * init_acc:.2f}%")

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
        self.server_results.setdefault("unlabeled_counts_acc", [])

        for sweep_idx, K in enumerate(self.eval_unlabeled_counts, start=1):
            t_step = time.time()

            # Determine effective number of samples
            B = min(int(K), self.unlabeled_pool_size, N_unlabeled)
            if B <= 0:
                print(f"[Zero-Shot] Skipping K={K}: effective batch size B={B} <= 0.")
                continue

            # Sample B distinct unlabeled indices at random
            rng = self.rng  # already seeded from algo_params["ota_seed"]
            if B >= N_unlabeled:
                or_indices = np.arange(N_unlabeled).tolist()
            else:
                or_indices = rng.choice(N_unlabeled, size=B, replace=False).tolist()

            print(f"\n[Zero-Shot Sweep] Setting {sweep_idx}/{len(self.eval_unlabeled_counts)}: "
                  f"K={K}, effective batch size B={B}")

            # Reset server model to raw initial state before this experiment
            self.model.load_state_dict(init_global_state)

            # Optional: server accuracy before distillation for this K
            acc_before = self._evaluate_global_model()
            print(f"  Server test acc BEFORE distill (K={K}): {100.0 * acc_before:.2f}%")

            # 1) Clients label the chosen unlabeled batch (FROZEN models)
            probs_list = []
            for client_idx in sampled:
                # Reload frozen client weights
                if hasattr(self, "client_weights") and client_idx in self.client_weights:
                    self.client.model.load_state_dict(self.client_weights[client_idx])
                else:
                    self.client.model.load_state_dict(self.model.state_dict())

                self._set_client_data(client_idx)
                probs = self.client.compute_probs_on_indices(or_indices)  # CPU (B,C)
                probs_list.append(probs)

            # 2) OTA aggregation: same as dsfl_sa_ota
            device = self.device
            avg_probs = self._avg_client_probs([p.to(device) for p in probs_list])  # (B,C)

            # 3) Server distillation on this batch
            self._server_distill_step(or_indices, avg_probs)

            # NOTE:
            # We deliberately DO NOT call client.distill_on_indices(...) here,
            # to keep clients frozen for the "zero-shot" behavior.

            # 4) Evaluate server global model after zero-shot distillation
            acc_after = self._evaluate_global_model()
            self.server_results["test_accuracy"].append(acc_after)
            self.server_results["unlabeled_counts"].append(B)
            self.server_results["unlabeled_counts_acc"].append(acc_after)

            print(f"  Server test acc AFTER  distill (K={K}): {100.0 * acc_after:.2f}%")

            elapsed = time.time() - t_step
            print(f"  [Zero-Shot Sweep] Elapsed {elapsed:.1f}s for K={K}")

        self._print_end()
        self._save_results_to_excel()

    # ---- helpers (shared helper style) ----

    def _evaluate_global_model(self):
        from algorithms.measures import evaluate_model
        self.model.eval()
        return evaluate_model(self.model, self.testloader, device=self.device)

    def _print_stats_labels_only(self, round_results, round_idx, elapsed):
        try:
            import wandb
        except Exception:
            class _Dummy:
                def log(self, *a, **k): pass
            wandb = _Dummy()
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

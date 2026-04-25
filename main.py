import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import os, sys, types
import contextlib
import itertools

def _make_wandb_dummy():
    dummy = types.SimpleNamespace()  # lightweight object
    def _noop(*a, **k): pass
    def _ret_self(*a, **k): return dummy  # so wandb.init(...) works in expressions

    dummy.init   = _ret_self
    dummy.save   = _noop
    dummy.log    = _noop
    dummy.watch  = _noop
    dummy.finish = _noop

    dummy.config  = types.SimpleNamespace()
    dummy.summary = {}
    dummy.run     = types.SimpleNamespace(dir=os.getcwd(), name="offline", id="offline")

    dummy.define_metric = _noop
    dummy.Table         = object
    dummy.Image         = object
    return dummy

# If WANDB is disabled/offline OR not installed, inject a dummy module
if os.environ.get("WANDB_MODE", "").lower() in {"disabled", "off", "offline"}:
    sys.modules["wandb"] = _make_wandb_dummy()
else:
    try:
        import wandb  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["wandb"] = _make_wandb_dummy()

import algorithms
from train_tools import *
from utils import *

import numpy as np
import argparse
import warnings
import random
import pprint

warnings.filterwarnings("ignore")
torch.set_printoptions(10)

ALGO = {
    "dsfl_avg_ota_zero_shot": algorithms.dsfl_avg_ota_zero_shot.Server,
    # Kept because dsfl_avg_ota_zero_shot inherits its SCENE/OTA machinery.
    "dsfl_sa_ota_zero_shot": algorithms.dsfl_sa_ota_zero_shot.Server,
    # Kept because dsfl_sa_ota_zero_shot inherits helper methods from dsfl_sa.Server.
    "dsfl_sa": algorithms.dsfl_sa.Server,
}

SCHEDULER = {
    "step": lr_scheduler.StepLR,
    "multistep": lr_scheduler.MultiStepLR,
    "cosine": lr_scheduler.CosineAnnealingLR,
}

class _TeeStream:
    """Write to multiple streams (e.g., console + file)."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

@contextlib.contextmanager
def _tee_stdout_stderr_to_file(log_path: str):
    """Mirror stdout+stderr to a file while keeping console output."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    old_out, old_err = sys.stdout, sys.stderr
    with open(log_path, "w", encoding="utf-8") as f:
        sys.stdout = _TeeStream(old_out, f)
        sys.stderr = _TeeStream(old_err, f)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def _get_setups(args):
    """Get train configuration with centralized seeding."""

    master_seed = int(args.train_setups.seed)

    seed_data = master_seed + 1
    seed_exp  = master_seed + 2
    seed_ota  = master_seed + 3

    np.random.seed(seed_data)
    random.seed(seed_data)

    data_distributed = data_distributer(**args.data_setups)

    _random_seeder(seed_exp)

    model = create_models(
        args.train_setups.model.name,
        args.data_setups.dataset_name
    )

    optimizer = optim.SGD(model.parameters(), **args.train_setups.optimizer.params)
    scheduler = None

    if args.train_setups.scheduler.enabled:
        scheduler = SCHEDULER[args.train_setups.scheduler.name](
            optimizer, **args.train_setups.scheduler.params
        )

    algo_params = args.train_setups.algo.params
    algo_params["ota_seed"] = seed_ota

    ts = args.train_setups
    scenario = ts.scenario if ("scenario" in ts) else ts

    device       = getattr(scenario, "device", "cpu")
    n_rounds     = int(getattr(scenario, "n_rounds", 100))
    local_epochs = int(getattr(scenario, "local_epochs", 1))
    sample_ratio = float(getattr(scenario, "sample_ratio", 1.0))

    server = ALGO[ts.algo.name](
        algo_params=ts.algo.params,
        model=model,
        data_distributed=data_distributed,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        n_rounds=n_rounds,
        local_epochs=local_epochs,
        sample_ratio=sample_ratio,
    )

    return server


def _random_seeder(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    """Execute experiment with Monte-Carlo trials and per-trial results."""

    base_seed = int(args.train_setups.seed)
    n_trials = int(getattr(args.train_setups, "n_trials", 1))

    algo_params = args.train_setups.algo.params

    # Read S_list and M_list from config file
    S_values = list(algo_params.get("S_list", [algo_params.get("ota_S", 1)]))
    M_values = list(algo_params.get("M_list", [algo_params.get("ota_M", 1)]))

    S_values = [int(s) for s in S_values]
    M_values = [int(m) for m in M_values]

    base_save_dir = getattr(getattr(wandb, "run", None), "dir", os.getcwd())
    os.makedirs(base_save_dir, exist_ok=True)

    for S_val, M_val in itertools.product(S_values, M_values):
        args.train_setups.algo.params["ota_S"] = S_val
        args.train_setups.algo.params["ota_M"] = M_val

        print("\n" + "#" * 90)
        print(f" RUNNING COMBINATION: S = {S_val}, M = {M_val}")
        print("#" * 90)

        all_final_acc = []
        per_trial_seeds = []
        per_trial_counts = []
        per_trial_accs = []

        combo_dir = os.path.join(base_save_dir, f"S_{S_val}_M_{M_val}")
        logs_dir = os.path.join(combo_dir, "trial_logs")
        os.makedirs(logs_dir, exist_ok=True)

        for t in range(n_trials):
            current_seed = base_seed + t
            args.train_setups.seed = current_seed
            per_trial_seeds.append(current_seed)

            trial_log_path = os.path.join(logs_dir, f"trial_{t+1:02d}_seed{current_seed}.txt")

            with _tee_stdout_stderr_to_file(trial_log_path):
                print("\n" + "=" * 70)
                print(f" TRIAL {t + 1}/{n_trials}  |  master seed = {current_seed}")
                print(f" COMBINATION               |  S = {S_val}, M = {M_val}")
                print("=" * 70)

                server = _get_setups(args)
                print(f"[ACTUAL OTA] ota_S={server.ota_S}, ota_M={server.ota_M}")
                server.run()

                trial_final_acc = None
                trial_curve_counts = None
                trial_curve_accs = None

                if hasattr(server, "server_results"):
                    sr = server.server_results

                    unlabeled_counts = sr.get("unlabeled_counts", None)
                    test_accs = sr.get("test_accuracy", None)

                    if unlabeled_counts is not None and test_accs is not None:
                        print("Zero-shot test accuracy per unlabeled budget for this trial:")
                        for k, acc in zip(unlabeled_counts, test_accs):
                            try:
                                print(f"  U = {int(k):6d}  ->  test_acc = {float(acc):.4f}")
                            except Exception:
                                print(f"  U = {k}  ->  test_acc = {acc}")

                        try:
                            trial_curve_counts = [int(k) for k in unlabeled_counts]
                            trial_curve_accs = [float(a) for a in test_accs]
                        except Exception:
                            trial_curve_counts = None
                            trial_curve_accs = None

                        try:
                            trial_final_acc = float(test_accs[-1])
                        except Exception:
                            trial_final_acc = None

                if trial_curve_counts is not None and trial_curve_accs is not None:
                    curve_txt = os.path.join(logs_dir, f"trial_{t+1:02d}_seed{current_seed}_curve.txt")
                    with open(curve_txt, "w", encoding="utf-8") as f:
                        f.write(f"Trial {t+1}/{n_trials} | seed={current_seed} | S={S_val} | M={M_val}\n")
                        f.write("U,test_acc_fraction,test_acc_percent\n")
                        for k, a in zip(trial_curve_counts, trial_curve_accs):
                            f.write(f"{k},{a:.6f},{a*100.0:.3f}\n")

                    per_trial_counts.append(trial_curve_counts)
                    per_trial_accs.append(trial_curve_accs)

                if trial_final_acc is not None:
                    all_final_acc.append(trial_final_acc)
                    print(f"\n[Trial {t + 1}] FINAL test accuracy: {trial_final_acc:.4f}")
                else:
                    print(f"\n[Trial {t + 1}] FINAL test accuracy: unavailable (server_results missing)")

                model_path = os.path.join(combo_dir, f"model_trial{t+1}_seed{current_seed}.pth")
                torch.save(server.model.state_dict(), model_path)
                print(f"[Main] Saved model for Trial {t+1} to: {model_path}")

                del server
                import gc
                gc.collect()

        if all_final_acc:
            mean_acc = float(np.mean(all_final_acc))
            std_acc = float(np.std(all_final_acc))

            summary_path = os.path.join(combo_dir, "monte_carlo_summary_final_acc.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(f"MONTE-CARLO SUMMARY (final test accuracy per trial) | S={S_val}, M={M_val}\n")
                for idx, (seed, acc) in enumerate(zip(per_trial_seeds, all_final_acc), start=1):
                    f.write(f"Trial {idx:2d} | seed={seed} | final_test_acc={acc:.6f} ({acc*100.0:.3f}%)\n")
                f.write(f"\nMean={mean_acc:.6f} ({mean_acc*100.0:.3f}%)\n")
                f.write(f"Std ={std_acc:.6f} ({std_acc*100.0:.3f}%)\n")

            print("\n" + "=" * 70)
            print(f" MONTE-CARLO SUMMARY (final test accuracy per trial) | S={S_val}, M={M_val}")
            print("=" * 70)
            for idx, acc in enumerate(all_final_acc, start=1):
                print(f"  Trial {idx:2d}: {acc:.4f}")
            print("-" * 70)
            print(f"  Mean: {mean_acc:.4f}")
            print(f"  Std : {std_acc:.4f}")
            print("=" * 70)
            print(f"[Main] Saved final-acc summary to: {summary_path}")
        else:
            print(f"\n[Monte-Carlo] No valid final accuracies collected for S={S_val}, M={M_val}.")

        if per_trial_counts and per_trial_accs:
            K_ref = per_trial_counts[0]
            curves = []
            for Ks_i, accs_i in zip(per_trial_counts, per_trial_accs):
                if Ks_i == K_ref and len(accs_i) == len(K_ref):
                    curves.append(accs_i)

            if curves:
                curves = np.asarray(curves, dtype=float)
                mean_curve = curves.mean(axis=0)
                std_curve = curves.std(axis=0, ddof=1) if curves.shape[0] > 1 else np.zeros_like(mean_curve)

                curve_summary_path = os.path.join(combo_dir, "zero_shot_curve_mean_std_by_K.txt")
                with open(curve_summary_path, "w", encoding="utf-8") as f:
                    f.write(f"Mean/Std across trials for zero-shot curve | S={S_val}, M={M_val}\n")
                    f.write("U,mean_acc_fraction,std_acc_fraction,mean_acc_percent,std_acc_percent\n")
                    for k, m, s in zip(K_ref, mean_curve, std_curve):
                        f.write(f"{k},{m:.6f},{s:.6f},{m*100.0:.3f},{s*100.0:.3f}\n")

                print(f"[Main] Saved zero-shot curve mean/std to: {curve_summary_path}")


# Parser arguments for terminal execution
parser = argparse.ArgumentParser(description="Process Configs")
parser.add_argument("--config_path", default="./config/dsfl_sa.json", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--n_clients", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--partition_method", type=str)
parser.add_argument("--partition_s", type=int)
parser.add_argument("--partition_alpha", type=float)
parser.add_argument("--model_name", type=str)
parser.add_argument("--n_rounds", type=int)
parser.add_argument("--sample_ratio", type=float)
parser.add_argument("--local_epochs", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--momentum", type=float)
parser.add_argument("--wd", type=float)
parser.add_argument("--algo_name", type=str)
parser.add_argument("--device", type=str)
parser.add_argument("--seed", type=int)
parser.add_argument("--group", type=str)
parser.add_argument("--exp_name", type=str)
parser.add_argument("--unlabeled_pool_size", type=int, default=None)
parser.add_argument("--unlabeled_per_round", type=int, default=None)
parser.add_argument("--unlabeled_batch_size", type=int, default=None)
parser.add_argument("--client_distill_epochs", type=int, default=None)
parser.add_argument("--server_distill_epochs", type=int, default=None)

if __name__ == "__main__":
    import sys
    os.environ.setdefault("WANDB_MODE", "disabled")

    _argv = sys.argv[1:]
    if "--wdir" in _argv and len(_argv) <= 2:
        sys.argv = [sys.argv[0]]

    if len(sys.argv) == 1:
        sys.argv += [
            "--config_path", "./config/dsfl_avg_ota_zero_shot.json",
            "--device", "cuda:0",
            "--dataset_name", "mnist",
            "--partition_method", "lda",
            "--partition_alpha", "0.3",
            "--n_clients", "100",
            "--batch_size", "32",
            "--n_rounds", "1",
            "--local_epochs", "1",
            "--sample_ratio", "1.0",
            "--algo_name", "dsfl_avg_ota_zero_shot",
            "--model_name", "fedavg_mnist",
        ]

    args = parser.parse_args()
    opt = ConfLoader(args.config_path).opt
    opt = config_overwriter(opt, args)

    print("")
    print("=" * 50 + " Configuration " + "=" * 50)
    pp = pprint.PrettyPrinter(compact=True)
    pp.pprint(opt)
    print("=" * 120)

    use_wandb = os.environ.get("WANDB_MODE", "").lower() not in {"disabled", "off", "offline"}
    if use_wandb:
        try:
            wandb.init(config=opt, **opt.wandb_setups)
        except Exception as e:
            print("[W&B] disabling due to init error:", e)
            class _DummyWandb:
                def __init__(self):
                    self.config = type("C", (), {})()
                    self.run = type("R", (), {"dir": os.getcwd()})()
                    self.summary = {}
                def save(self, *a, **k): pass
                def log(self, *a, **k): pass
                def watch(self, *a, **k): pass
                def finish(self, *a, **k): pass
            wandb = _DummyWandb()
    else:
        class _DummyWandb:
            def __init__(self):
                self.config = type("C", (), {})()
                self.run = type("R", (), {"dir": os.getcwd()})()
            def save(self, *a, **k): pass
        wandb = _DummyWandb()

    if hasattr(wandb, "config"):
        wandb.config.log_interval = 10

    main(opt)

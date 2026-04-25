````markdown
# SCENE OTA-FD

This repository provides the simulation code used to reproduce the numerical results for the paper:

**SCENE OTA-FD: Self-Centering Noncoherent Estimator for Over-the-Air Federated Distillation**

The WCL letter version of this work is currently under review. An extended version of the paper is available on arXiv:

https://arxiv.org/abs/2602.15326

## Overview

SCENE OTA-FD is a pilot-free, noncoherent over-the-air federated distillation framework. Instead of transmitting model parameters or gradients, clients transmit soft-label information over a wireless multiple-access channel. Each client maps its class-probability vector to transmit energies, and the server uses a self-centering energy estimator to recover the weighted soft-label average without instantaneous channel state information (CSI) or pilot symbols.

The main experiment in this repository compares two aggregation methods:

- **Plain**: noise-free weighted averaging of client soft labels.
- **SCENE OTA**: noncoherent energy-based over-the-air aggregation with self-centering.

The implementation focuses on the one-shot federated distillation setting used in the WCL submission. In this protocol, clients are trained locally, frozen, queried on a shared unlabeled dataset, and then used as teacher models for server-side distillation.

## Repository Structure

```text
SCENE_OTA_FD/
├── main.py
├── config/
│   └── dsfl_avg_ota_zero_shot.json
├── algorithms/
│   ├── BaseClientTrainer.py
│   ├── BaseServer.py
│   ├── dsfl_avg_ota_zero_shot/
│   ├── dsfl_sa_ota_zero_shot/
│   └── dsfl_sa/
├── train_tools/
├── utils.py
├── requirements.txt
└── README.md
````

The main algorithm is:

```text
dsfl_avg_ota_zero_shot
```

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

A CUDA-enabled GPU is recommended for faster execution, although small test runs can also be executed on CPU.

## How to Run

To run the main SCENE OTA-FD experiment, use:

```bash
python main.py --cfg config/dsfl_avg_ota_zero_shot.json
```

The configuration file controls the dataset, number of clients, data partition, local training epochs, server distillation epochs, OTA parameters, and the sweep over repetition factors.

For a quick functionality test, temporarily reduce the runtime settings in:

```text
config/dsfl_avg_ota_zero_shot.json
```

For example:

```json
"n_trials": 1,
"zero_shot_client_epochs": 1,
"server_distill_epochs": 1,
"eval_unlabeled_counts": [250]
```

After verifying that the code runs, restore the full experimental configuration.

## Important Configuration Parameters

The client data distribution is controlled by:

```json
"partition": {
  "method": "lda",
  "alpha": 1.0
}
```

The Dirichlet parameter `alpha=1.0` corresponds to a moderately non-i.i.d. client data split.

The SCENE OTA diversity order is controlled by:

```json
"S_list": [1, 2, 4, 8, 16, 32, 64, 128],
"M_list": [1]
```

Here, `S` is the number of OTA repetitions and `M` is the number of receive antennas. The aggregation variance decreases with the diversity order `SM`.

The target average per-resource-element SNR is controlled by:

```json
"ota_target_snr_db": 5.0
```

The unlabeled-data budget for server-side distillation is controlled by:

```json
"eval_unlabeled_counts": [250, 500, 1000, 2000, 4000, 8000, 16000]
```

## Notes on Trials and Data Splits

The reported results are averaged over multiple independent trials. In the current setup, each trial uses a different random seed, which affects the client data partition, local training randomness, server initialization, server distillation randomness, and OTA channel/noise realizations.

For comparing different `S` values, the same trial index uses the same base seed, which helps provide a fair comparison across the OTA repetition-factor sweep.

## Citation

If you use this code, please cite the extended paper:

```bibtex
@article{chen2026scene,
  title={SCENE OTA-FD: Self-Centering Noncoherent Estimator for Over-the-Air Federated Distillation},
  author={Chen, Hao and Bozorgasl, Zavareh},
  journal={arXiv preprint arXiv:2602.15326},
  year={2026}
}
```

## License

This repository is released under the license included in `LICENSE`.

## Contact

For questions, please contact:

```text
Zavareh Bozorgasl  
Boise State University
```

```
```

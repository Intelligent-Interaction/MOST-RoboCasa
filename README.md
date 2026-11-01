<h1 align="center">
  MOST: Memory of Object States and Task Progress<br>
  for Long-Horizon Mobile Manipulation
</h1>

<p align="center">
  Diqi He<sup>1,2</sup>, Jinbang Guo<sup>2</sup>, Yijun Shen<sup>2,3</sup>,
  Zhiwei Yu<sup>2,*</sup>,<br>
  Huan Yang<sup>2</sup>, Dingwen Zhang<sup>1,*</sup>, Junwei Han<sup>4</sup>,
  Zhongyuan Wang<sup>2</sup>
</p>

<p align="center">
  <sup>1</sup>Northwestern Polytechnical University &nbsp;&nbsp;
  <sup>2</sup>Beijing Academy of Artificial Intelligence<br>
  <sup>3</sup>East China Normal University &nbsp;&nbsp;
  <sup>4</sup>Chongqing University of Posts and Telecommunications
</p>

<p align="center">
  <sup>*</sup>Corresponding authors
</p>

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-1f6feb?logo=githubpages&logoColor=white)](https://most-vla.github.io/)
[![Paper](https://img.shields.io/badge/Paper-Coming_Soon-b31b1b?logo=arxiv&logoColor=white)](https://github.com/diqihe666/MOST-vla)
[![Data](https://img.shields.io/badge/Data-RoboCasa--Long-ffcc4d?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/diqihe666/robocasa_long)
[![Policy Code](https://img.shields.io/badge/Policy_Code-MOST--VLA-181717?logo=github&logoColor=white)](https://github.com/diqihe666/MOST-vla)

</div>

---

<h2 align="center">MOST RoboCasa Environment</h2>

<p align="center">
  Unified RoboCasa365 and RoboCasa-Long simulation and evaluation environment
</p>

## Overview

This repository is the official RoboCasa environment used by MOST for both
**RoboCasa365** and **RoboCasa-Long** evaluation. It is a complete fork of
[RoboCasa](https://github.com/robocasa/robocasa), preserving the upstream
environments, robots, controllers, fixtures, objects, scripts, and asset
interface while adding the task-progress instrumentation required by MOST.

Install this repository for both benchmarks. The unmodified upstream RoboCasa
package does not provide the 32 task-specific long-horizon metric trackers,
the three RoboCasa-Long tasks, or the metric data consumed by the MOST
evaluators.

The main additions are:

- long-horizon semantic progress tracking for 32 RoboCasa365 Composite tasks;
- three RoboCasa-Long task definitions under
  `robocasa/environments/kitchen/long/`;
- a Gym wrapper interface that returns task metrics in
  `info["metrics"]` after every environment step;
- shared PRM-style curve processing and metric computation in
  `robocasa/utils/prm_metrics.py`; and
- startup and metric-interface tests covering all 35 instrumented tasks.

The repository uses the standard RoboCasa kitchen assets and RoboSuite
backend. It does not redistribute a separate asset pack.

## Changes from Upstream RoboCasa

This repository retains the complete upstream RoboCasa codebase. The
MOST-specific changes are localized to the following areas:

| Area | Main paths | Changes in this fork |
| --- | --- | --- |
| RoboCasa365 task metrics | `robocasa/environments/kitchen/composite/` | Adds state-grounded semantic milestones and per-step long-horizon metric trackers to 32 Composite Seen and Composite Unseen tasks. |
| RoboCasa-Long tasks | `robocasa/environments/kitchen/long/` | Adds `PackFruit`, `PrepareBreakfast`, and `CleanAndStoreAfterMeal`, including their scene configuration, success conditions, and progress trackers. |
| Task registration | `robocasa/__init__.py` | Registers the three RoboCasa-Long environments so they can be created through the standard RoboCasa and RoboSuite interfaces. |
| Gym integration | `robocasa/wrappers/gym_wrapper.py` | Adds the supported task's current long-horizon state to `info["metrics"]` on every `step()`. |
| Shared process metrics | `robocasa/utils/prm_metrics.py` | Adds progress-curve preprocessing and the common PRM-style metrics used by both MOST evaluators. |
| Validation | `tests/test_robocasa_long.py`, `tests/test_gym_long_metrics.py` | Adds startup checks for the three Long tasks and an end-to-end Gym metric-interface check for all 35 supported tasks. |
| Package metadata and documentation | `setup.py`, `README.md` | Identifies this fork as the unified simulator dependency for MOST on RoboCasa365 and RoboCasa-Long. |

The upstream asset format, asset downloader, robots, controllers, fixtures,
and standard environment construction remain compatible. Instrumented tasks
add `get_long_horizon_metrics()`, and their Gym `step()` result includes the
`metrics` entry described below. Other upstream tasks retain their original
behavior and do not receive a synthetic metric result.

## Supported Tasks

### RoboCasa365 Composite Seen

`DeliverStraw`, `GetToastedBread`, `KettleBoiling`, `LoadDishwasher`,
`PackIdenticalLunches`, `PreSoakPan`, `PrepareCoffee`, `RinseSinkBasin`,
`ScrubCuttingBoard`, `SearingMeat`, `SetUpCuttingStation`,
`StackBowlsCabinet`, `SteamInMicrowave`, `StirVegetables`,
`StoreLeftoversInBowl`, and `WashLettuce`.

### RoboCasa365 Composite Unseen

`ArrangeBreadBasket`, `ArrangeTea`, `BreadSelection`,
`CategorizeCondiments`, `CuttingToolSelection`, `GarnishPancake`,
`GatherTableware`, `HeatKebabSandwich`, `MakeIceLemonade`, `PanTransfer`,
`PortionHotDogs`, `RecycleBottlesByType`, `SeparateFreezerRack`,
`WaffleReheat`, `WashFruitColander`, and `WeighIngredients`.

### RoboCasa-Long

| Task | Default scene | Description |
| --- | --- | --- |
| `PackFruit` | Layout 7, style 7 | Collect six fruits from three kitchen areas and pack them into one tupperware container. |
| `PrepareBreakfast` | Layout 4, style 4 | Prepare toast, milk, and fruit, then arrange the completed breakfast on the dining counter. |
| `CleanAndStoreAfterMeal` | Layout 2, style 2 | Store food, organize pantry items, and move the pan and sponge into the island sink. |

## Metric Interface

Each supported task exposes `get_long_horizon_metrics()`. When a task is
created through `robocasa.wrappers.gym_wrapper`, every call to `step()` places
the current metric state in `info["metrics"]`:

| Field | Meaning |
| --- | --- |
| `SR` | Task success indicator |
| `OPS` | Ordered semantic task progress |
| `UPS` | Unordered semantic task progress |
| `NCT` | Normalized completion time |
| `progress_curve` | Per-step ordered-progress history |

After an episode, `compute_prm_metrics()` converts the progress curve into the
shared process metrics used by the MOST evaluators: `MP`, `MC25`, `MC50`,
`MC75`, `PPL`, `CRA`, `STR`, `DRR`, `FNS`, and `SQS`. Metrics that do not apply
to a rollout are returned as `NaN`.

```python
from robocasa.utils.prm_metrics import compute_prm_metrics


task_metrics = info["metrics"]
process_metrics = compute_prm_metrics(
    task_metrics["progress_curve"],
    success=bool(task_metrics["SR"]),
)
```

## Installation

The commands below create a clean simulator environment. Python 3.11 is
recommended to match the upstream RoboCasa setup.

### 1. Create the Conda environment

```bash
conda create -c conda-forge -n robocasa python=3.11 -y
conda activate robocasa
```

### 2. Install RoboSuite

```bash
git clone https://github.com/ARISE-Initiative/robosuite.git
cd robosuite
pip install -e .
cd ..
```

### 3. Install the MOST RoboCasa environment

This repository contains the complete RoboCasa package. Do not install a
second upstream `robocasa` checkout in the same environment.

```bash
git clone https://github.com/diqihe666/MOST-RoboCasa.git
cd MOST-RoboCasa
pip install -e .
```

If upstream RoboCasa was previously installed, the editable install above
must be the copy imported by Python. Verify the resolved package path:

```bash
python -c "import robocasa; print(robocasa.__file__)"
```

The printed path must point into this `MOST-RoboCasa/robocasa` checkout. If it
points to another checkout, remove that path from `PYTHONPATH` or uninstall
the conflicting editable package before continuing.

### 4. Configure RoboCasa and download assets

Run the standard RoboCasa setup commands from this repository root:

```bash
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
```

The kitchen asset download is approximately 10 GB. The assets are maintained
by the RoboCasa project and retain their original license.

## Verify the Installation

First, create and reset the three RoboCasa-Long tasks:

```bash
conda activate robocasa
cd /path/to/MOST-RoboCasa
python tests/test_robocasa_long.py
```

Then verify that all 32 RoboCasa365 tasks and all three RoboCasa-Long tasks
return metrics from the Gym wrapper:

```bash
python tests/test_gym_long_metrics.py
```

A successful run ends with:

```text
Passed: 35/35
Failed: 0/35
All Gym wrapper metric checks passed.
```

The tests default to `MUJOCO_GL=egl`. Set `MUJOCO_GL` before launching them if
your system requires a different MuJoCo rendering backend.

## Basic Gym Usage

Importing `robocasa` registers the environments with Gymnasium. The following
example resets one task, takes one action, and reads its progress metrics:

```python
import gymnasium as gym
import robocasa  # noqa: F401 - registers RoboCasa environments


env = gym.make(
    "robocasa/DeliverStraw",
    split="target",
    seed=7,
    enable_render=False,
)

observation, reset_info = env.reset(seed=7)
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
print(info["metrics"])
env.close()
```

Replace `DeliverStraw` with any task listed under
[Supported Tasks](#supported-tasks).

## Repository Structure

```text
robocasa/
  environments/kitchen/composite/  # 32 instrumented RoboCasa365 tasks
  environments/kitchen/long/       # Three RoboCasa-Long tasks
  utils/prm_metrics.py              # Shared progress-curve metrics
  wrappers/gym_wrapper.py           # Exposes metrics through Gym step()
tests/
  test_robocasa_long.py             # Starts the three Long tasks
  test_gym_long_metrics.py          # Checks all 35 metric interfaces
```

## Training Data and Policy Code

- RoboCasa-Long demonstrations and memory annotations:
  [diqihe666/robocasa_long](https://huggingface.co/datasets/diqihe666/robocasa_long)
- MOST training, checkpoint export, model-server, and evaluation code:
  [diqihe666/MOST-vla](https://github.com/diqihe666/MOST-vla)

The policy repository contains the complete step-by-step workflows for
RoboCasa365 and RoboCasa-Long.

## Acknowledgements

This environment is built directly on
[RoboCasa and RoboCasa365](https://github.com/robocasa/robocasa), with
[RoboSuite](https://github.com/ARISE-Initiative/robosuite) as the simulation
backend. We thank the respective authors and contributors for releasing their
code and assets.

## Citation

If you use this environment, please cite MOST and the upstream RoboCasa
projects.

```bibtex
@article{he2026most,
  title   = {MOST: Memory of Object States and Task Progress
             for Long-Horizon Mobile Manipulation},
  author  = {He, Diqi and Guo, Jinbang and Shen, Yijun and
             Yu, Zhiwei and Yang, Huan and Zhang, Dingwen and
             Han, Junwei and Wang, Zhongyuan},
  year    = {2026},
  note    = {Under review}
}

@inproceedings{robocasa365,
  title     = {RoboCasa365: A Large-Scale Simulation Framework for Training
               and Benchmarking Generalist Robots},
  author    = {Nasiriany, Soroush and Nasiriany, Sepehr and
               Maddukuri, Abhiram and Zhu, Yuke},
  booktitle = {International Conference on Learning Representations},
  year      = {2026}
}

@inproceedings{robocasa2024,
  title     = {RoboCasa: Large-Scale Simulation of Everyday Tasks for
               Generalist Robots},
  author    = {Nasiriany, Soroush and Maddukuri, Abhiram and Zhang, Lance and
               Parikh, Adeet and Lo, Aaron and Joshi, Abhishek and
               Mandlekar, Ajay and Zhu, Yuke},
  booktitle = {Robotics: Science and Systems},
  year      = {2024}
}
```

## License

The code is released under the [MIT License](LICENSE), consistent with
upstream RoboCasa. RoboCasa assets and datasets are licensed separately under
their original terms; see the upstream
[RoboCasa repository](https://github.com/robocasa/robocasa) for details.

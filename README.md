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

<h2 align="center">RoboCasa-Long Simulator</h2>

<p align="center">
  The official long-horizon simulation environment for MOST, built on RoboCasa365
</p>

## Overview

RoboCasa-Long is the simulation environment used to study long-horizon mobile
manipulation in MOST. It extends
[RoboCasa365](https://github.com/robocasa/robocasa) with three tasks that require
the robot to navigate between multiple kitchen areas and complete a sequence of
semantically related manipulations.

This repository is a minimal fork of RoboCasa. The original environments,
robots, controllers, fixtures, objects, scripts, and assets remain unchanged.
The RoboCasa-Long release adds only:

- three task definitions under `robocasa/environments/kitchen/long/`;
- task registration in `robocasa/__init__.py`; and
- `tests/test_robocasa_long.py`, a lightweight environment startup test.

RoboCasa-Long does not redistribute a separate asset pack. It uses the standard
RoboCasa kitchen assets and the same RoboSuite backend as RoboCasa365.

## Tasks

| Task | Default scene | Description |
| --- | --- | --- |
| `PackFruit` | Layout 7, style 7 | Collect six fruits from three kitchen areas and pack them into one tupperware container. |
| `PrepareBreakfast` | Layout 4, style 4 | Prepare toast, milk, and fruit, then arrange the completed breakfast on the dining counter. |
| `CleanAndStoreAfterMeal` | Layout 2, style 2 | Store food, organize pantry items, and move the pan and sponge into the island sink. |

Each task exposes the standard RoboCasa success interface and a
`get_long_horizon_metrics()` method. The latter reports success rate (`SR`),
ordered progress (`OPS`), unordered progress (`UPS`), normalized completion
time (`NCT`), and the per-step progress curve.

## Installation

The commands below create a clean environment from scratch. Python 3.11 is
recommended to match the upstream RoboCasa setup.

### 1. Create the Conda environment

```bash
conda create -c conda-forge -n robocasa python=3.11 -y
conda activate robocasa
```

### 2. Install RoboSuite

RoboCasa currently uses the RoboSuite development branch. Install it in
editable mode:

```bash
git clone https://github.com/ARISE-Initiative/robosuite.git
cd robosuite
pip install -e .
cd ..
```

### 3. Install RoboCasa-Long

This repository already contains the complete RoboCasa codebase, so do not
clone and install a second copy of the upstream `robocasa` package in the same
environment.

```bash
git clone https://github.com/diqihe666/Robocasa-Long.git
cd Robocasa-Long
pip install -e .
```

If an editable upstream RoboCasa checkout is already installed, running
`pip install -e .` from this repository switches the environment to this fork.

### 4. Configure RoboCasa and download assets

Run the standard RoboCasa setup commands from the RoboCasa-Long repository
root:

```bash
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
```

The kitchen asset download is approximately 10 GB. The assets are maintained
by the RoboCasa project and retain their original license.

## Verify the Installation

The included smoke test creates and resets each Long task once. It does not
load a policy checkpoint, run a rollout, or download the MOST training data.

```bash
conda activate robocasa
cd /path/to/Robocasa-Long
python tests/test_robocasa_long.py
```

A successful installation prints:

```text
[PASS] PackFruit
[PASS] PrepareBreakfast
[PASS] CleanAndStoreAfterMeal

All 3 RoboCasa-Long tasks started successfully.
```

The test defaults to `MUJOCO_GL=egl`. Set `MUJOCO_GL` before running the script
if a different MuJoCo rendering backend is required on your system.

## Basic Usage

Importing `robocasa` registers all three tasks with RoboSuite. The following
example creates and resets `PackFruit` without starting a renderer:

```python
import robocasa  # noqa: F401
import robosuite
from robosuite.controllers import load_composite_controller_config


env = robosuite.make(
    env_name="PackFruit",
    robots="PandaOmron",
    controller_configs=load_composite_controller_config(robot="PandaOmron"),
    seed=7,
    horizon=10,
    obj_instance_split="target",
    has_renderer=False,
    has_offscreen_renderer=False,
    ignore_done=True,
    use_object_obs=True,
    use_camera_obs=False,
    control_freq=20,
)

observation = env.reset()
print(sorted(observation))
env.close()
```

Replace `PackFruit` with `PrepareBreakfast` or
`CleanAndStoreAfterMeal` to create the other environments.

## Repository Structure

```text
robocasa/
  environments/kitchen/long/
    pack_fruit.py
    prepare_breakfast.py
    clean_and_store.py
  __init__.py                    # Registers the three Long tasks
tests/
  test_robocasa_long.py          # Creates and resets every Long task
```

All other source files are inherited from the upstream RoboCasa repository.

## Training Data and Policy Code

- RoboCasa-Long demonstrations and memory annotations:
  [diqihe666/robocasa_long](https://huggingface.co/datasets/diqihe666/robocasa_long)
- MOST training, checkpoint export, and inference code:
  [diqihe666/MOST-vla](https://github.com/diqihe666/MOST-vla)

The policy repository contains the complete data preparation, training,
checkpoint export, model-server, and evaluation commands.

## Acknowledgements

RoboCasa-Long is built directly on
[RoboCasa and RoboCasa365](https://github.com/robocasa/robocasa), with
[RoboSuite](https://github.com/ARISE-Initiative/robosuite) as the simulation
backend. We thank the respective authors and contributors for releasing their
code and assets.

## Citation

If you use RoboCasa-Long, please cite MOST and the upstream RoboCasa projects.

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

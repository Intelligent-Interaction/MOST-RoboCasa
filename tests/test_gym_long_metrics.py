#!/usr/bin/env python3
"""Check long-horizon metrics returned by RoboCasaGymEnv.step()."""

import json
import os
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import robocasa  # noqa: F401 - registers RoboCasa environments
from gymnasium import spaces


CS_TASKS = """
DeliverStraw
GetToastedBread
KettleBoiling
LoadDishwasher
PackIdenticalLunches
PreSoakPan
PrepareCoffee
RinseSinkBasin
ScrubCuttingBoard
SearingMeat
SetUpCuttingStation
StackBowlsCabinet
SteamInMicrowave
StirVegetables
StoreLeftoversInBowl
WashLettuce
""".split()

CU_TASKS = """
ArrangeBreadBasket
ArrangeTea
BreadSelection
CategorizeCondiments
CuttingToolSelection
GarnishPancake
GatherTableware
HeatKebabSandwich
MakeIceLemonade
PanTransfer
PortionHotDogs
RecycleBottlesByType
SeparateFreezerRack
WaffleReheat
WashFruitColander
WeighIngredients
""".split()

LONG_TASKS = """
PackFruit
PrepareBreakfast
CleanAndStoreAfterMeal
""".split()

REQUIRED_METRICS = {
    "SR",
    "OPS",
    "UPS",
    "NCT",
    "progress_curve",
}


def make_zero_action(env: gym.Env) -> dict[str, object]:
    """Create a valid all-zero action from the Gym action space."""
    action = {}

    for name, space in env.action_space.spaces.items():
        if isinstance(space, spaces.Box):
            action[name] = np.zeros(space.shape, dtype=space.dtype)
        elif isinstance(space, spaces.Discrete):
            action[name] = 0
        else:
            raise TypeError(
                f"Unsupported action space for {name}: {type(space).__name__}"
            )

    return action


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def test_task(group: str, task: str, seed: int) -> bool:
    env = None

    try:
        env = gym.make(
            f"robocasa/{task}",
            split="target",
            seed=seed,
            enable_render=False,
            camera_widths=64,
            camera_heights=64,
            horizon=10,
        )

        observation, reset_info = env.reset(seed=seed)

        if not isinstance(observation, dict) or not observation:
            raise RuntimeError("reset() did not return a valid observation.")

        result = env.step(make_zero_action(env))

        if not isinstance(result, tuple) or len(result) != 5:
            raise RuntimeError(
                f"step() returned an invalid result: {type(result)}"
            )

        _, reward, terminated, truncated, info = result

        if not isinstance(info, dict):
            raise TypeError(
                f"step() info must be a dictionary, got {type(info).__name__}"
            )

        metrics = info.get("metrics")

        if not isinstance(metrics, dict):
            raise RuntimeError(
                "step() did not return a metrics dictionary in info['metrics']."
            )

        missing = REQUIRED_METRICS.difference(metrics)
        if missing:
            raise RuntimeError(
                f"Missing long-horizon metrics: {sorted(missing)}"
            )

        progress_curve = metrics["progress_curve"]
        if not isinstance(progress_curve, (list, tuple)):
            raise TypeError("metrics['progress_curve'] must be a sequence.")

        if len(progress_curve) < 1:
            raise RuntimeError(
                "The progress curve was not updated after step()."
            )

        metrics_text = json.dumps(
            metrics,
            sort_keys=True,
            default=json_default,
        )

        print(
            f"[PASS] [{group}] {task} "
            f"reward={reward} "
            f"terminated={terminated} "
            f"truncated={truncated} "
            f"metrics={metrics_text}"
        )
        return True

    except Exception:
        print(f"[FAIL] [{group}] {task}")
        traceback.print_exc()
        return False

    finally:
        if env is not None:
            env.close()


def main() -> int:
    selected_tasks = [
        *(("cs", task) for task in CS_TASKS),
        *(("cu", task) for task in CU_TASKS),
        *(("long", task) for task in LONG_TASKS),
    ]

    failures = []

    for index, (group, task) in enumerate(selected_tasks):
        if not test_task(group, task, seed=7 + index):
            failures.append(f"{group}/{task}")

    passed = len(selected_tasks) - len(failures)

    print("\n" + "=" * 80)
    print(f"Passed: {passed}/{len(selected_tasks)}")
    print(f"Failed: {len(failures)}/{len(selected_tasks)}")

    if failures:
        print("Failed tasks:")
        for task in failures:
            print(f"  - {task}")
        return 1

    print("All Gym wrapper metric checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
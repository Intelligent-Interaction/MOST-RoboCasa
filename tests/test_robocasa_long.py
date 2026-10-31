#!/usr/bin/env python3
"""Smoke-test the three RoboCasa-Long environments."""

import os
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")

import robocasa  # noqa: F401 - registers RoboCasa-Long tasks
import robosuite
from robosuite.controllers import load_composite_controller_config


TASKS = (
    "PackFruit",
    "PrepareBreakfast",
    "CleanAndStoreAfterMeal",
)


def main() -> int:
    failures = 0

    for task in TASKS:
        env = None

        try:
            env = robosuite.make(
                env_name=task,
                robots="PandaOmron",
                controller_configs=load_composite_controller_config(
                    robot="PandaOmron"
                ),
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

            if not isinstance(observation, dict) or not observation:
                raise RuntimeError("reset() returned an invalid observation.")

            print(f"[PASS] {task}")

        except Exception:
            failures += 1
            print(f"[FAIL] {task}")
            traceback.print_exc()

        finally:
            if env is not None:
                env.close()

    if failures:
        print(f"\n{failures} of {len(TASKS)} tasks failed to start.")
        return 1

    print(f"\nAll {len(TASKS)} RoboCasa-Long tasks started successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
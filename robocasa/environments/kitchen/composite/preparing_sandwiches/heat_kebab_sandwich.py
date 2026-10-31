from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.object_utils as OU


class HeatKebabSandwich(Kitchen):
    """
    Heat Kebab Sandwich: composite task for Preparing Sandwiches.

    Simulates the task of heating up a kebab sandwich in the toaster oven.

    Steps:
        1) Pick up the kebab skewer and bread, and place them inside the toaster oven.
        2) Once inside, close the toaster oven door.
        3) Set the timer for at least 10% of the total time.

    Long-horizon progress used here:
        1. pick baguette
        2. place baguette in toaster oven
        3. pick kebab
        4. place kebab in toaster oven
        5. close toaster oven door
        6. set timer / start heating

    The two food pick-place branches are order-free:
        baguette_in_hand -> baguette_in_toaster
        kebab_in_hand    -> kebab_in_toaster

    Rack pull / push is intentionally not a milestone. The metric only checks
    that each food item is eventually placed on a toaster rack, then the final
    door-closed state, then the timer threshold.

    Original RoboCasa _check_success() is preserved.
    """

    def __init__(self, enable_fixtures=None, *args, **kwargs):
        enable_fixtures = enable_fixtures or []
        enable_fixtures = list(enable_fixtures) + ["toaster_oven"]
        super().__init__(enable_fixtures=enable_fixtures, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.toaster_oven = self.register_fixture_ref(
            "toaster_oven", dict(id=FixtureType.TOASTER_OVEN)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.toaster_oven)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta["lang"] = (
                "Pick up the kebab skewer and baguette bread, and place them inside the toaster oven. "
                "Close the toaster oven door and start by setting the timer."
            )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

        self.toaster_oven.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                object_scale=1.25,
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.toaster_oven,
                        loc="left_right",
                    ),
                    size=(1.0, 0.4),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="kebab",
                obj_groups="kebab_skewer",
                object_scale=[1.1, 0.8, 1],
                placement=dict(
                    object="plate",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="baguette",
                obj_groups="baguette",
                placement=dict(
                    object="plate",
                    size=(0.9, 0.9),
                ),
            )
        )

        return cfgs

    # =====================================================================
    # Task-specific semantic milestones
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "baguette_in_hand": {
                "description": "The baguette is grasped by the robot gripper.",
                "dependencies": [],
                "persistent": False,
            },
            "baguette_in_toaster": {
                "description": "The baguette is placed in the toaster oven.",
                "dependencies": ["baguette_in_hand"],
                "persistent": True,
            },
            "kebab_in_hand": {
                "description": "The kebab skewer is grasped by the robot gripper.",
                "dependencies": [],
                "persistent": False,
            },
            "kebab_in_toaster": {
                "description": "The kebab skewer is placed in the toaster oven.",
                "dependencies": ["kebab_in_hand"],
                "persistent": True,
            },
            "toaster_door_closed": {
                "description": (
                    "The toaster oven door is closed after both food items "
                    "have been placed in the toaster oven."
                ),
                "dependencies": [
                    "baguette_in_toaster",
                    "kebab_in_toaster",
                ],
                "persistent": True,
            },
            "timer_set": {
                "description": (
                    "The toaster oven timer reaches the official >= 0.1 "
                    "success threshold after the valid door-close stage."
                ),
                "dependencies": ["toaster_door_closed"],
                "persistent": True,
            },
        }

    def _check_task_obj_grasped(
        self,
        obj_name,
    ):
        """
        Local grasp check with a looser gripper-close threshold.

        This reduces missed pick frames without modifying RoboCasa object_utils
        globally.
        """
        try:
            return bool(
                OU.check_obj_grasped(
                    self,
                    obj_name,
                    threshold=0.10,
                )
            )
        except TypeError:
            return bool(
                OU.check_obj_grasped(
                    self,
                    obj_name,
                )
            )

    def _metric_obj_in_toaster(
        self,
        obj_name,
    ):
        """
        Food placement check.

        Rack pull / push is not a milestone, but food still needs to be on a
        toaster rack to count as placed inside the toaster oven.
        """
        return bool(
            self.toaster_oven.check_rack_contact(
                self,
                obj_name,
                rack_level=0,
            )
            or self.toaster_oven.check_rack_contact(
                self,
                obj_name,
                rack_level=1,
            )
        )

    def _metric_timer_set(
        self,
        baguette_in_toaster,
        kebab_in_toaster,
        door_closed,
    ):
        """
        Safely reuse the original task's success logic for timer detection.

        Do not call toaster_oven.get_state() from reset-time metric paths unless
        both food items are already placed and the door is closed.
        """
        if not (
            baguette_in_toaster
            and kebab_in_toaster
            and door_closed
        ):
            return False

        return bool(
            self._check_success()
        )

    def _get_raw_semantic_milestone_state(self):
        baguette_in_hand = self._check_task_obj_grasped(
            "baguette",
        )
        kebab_in_hand = self._check_task_obj_grasped(
            "kebab",
        )

        baguette_in_toaster = bool(
            self._metric_obj_in_toaster("baguette")
            and not baguette_in_hand
        )
        kebab_in_toaster = bool(
            self._metric_obj_in_toaster("kebab")
            and not kebab_in_hand
        )

        door_closed = bool(
            not self.toaster_oven.is_open(self)
        )

        timer_set = self._metric_timer_set(
            baguette_in_toaster,
            kebab_in_toaster,
            door_closed,
        )

        return {
            "baguette_in_hand": baguette_in_hand,
            "baguette_in_toaster": baguette_in_toaster,
            "kebab_in_hand": kebab_in_hand,
            "kebab_in_toaster": kebab_in_toaster,
            "toaster_door_closed": door_closed,
            "timer_set": timer_set,
        }

    def _get_ordered_semantic_milestone_state(self):
        """
        Current ordered-valid states.

        The baguette and kebab branches are independent. Pick milestones are
        current/superseded:
            - valid while the object is currently held;
            - also valid once its corresponding place stage is valid.

        Door close and timer are only valid while the previous task state still
        holds. Rack pull / push is ignored.
        """
        raw = self._get_raw_semantic_milestone_state()

        baguette_place_valid = bool(
            self._lh_ordered_ever["baguette_in_toaster"]
            and raw["baguette_in_toaster"]
        )
        kebab_place_valid = bool(
            self._lh_ordered_ever["kebab_in_toaster"]
            and raw["kebab_in_toaster"]
        )

        baguette_pick_valid = bool(
            self._lh_ordered_ever["baguette_in_hand"]
            and (
                raw["baguette_in_hand"]
                or baguette_place_valid
            )
        )
        kebab_pick_valid = bool(
            self._lh_ordered_ever["kebab_in_hand"]
            and (
                raw["kebab_in_hand"]
                or kebab_place_valid
            )
        )

        both_food_valid = bool(
            baguette_place_valid
            and kebab_place_valid
        )

        door_valid = bool(
            self._lh_ordered_ever["toaster_door_closed"]
            and raw["toaster_door_closed"]
            and both_food_valid
        )

        timer_valid = bool(
            self._lh_ordered_ever["timer_set"]
            and raw["timer_set"]
            and door_valid
        )

        return {
            "baguette_in_hand": baguette_pick_valid,
            "baguette_in_toaster": baguette_place_valid,
            "kebab_in_hand": kebab_pick_valid,
            "kebab_in_toaster": kebab_place_valid,
            "toaster_door_closed": door_valid,
            "timer_set": timer_valid,
        }

    # =====================================================================
    # Per-episode long-horizon metric tracker
    # =====================================================================

    def _reset_long_horizon_metric_tracker(
        self,
        initial_state=None,
    ):
        names = list(
            self.get_semantic_milestone_spec().keys()
        )

        self._lh_step_count = 0

        if initial_state is None:
            initial_state = {
                name: False
                for name in names
            }

        self._lh_prev_raw = {
            name: bool(
                initial_state.get(
                    name,
                    False,
                )
            )
            for name in names
        }

        self._lh_unordered_ever = {
            name: False
            for name in names
        }

        self._lh_ordered_ever = {
            name: False
            for name in names
        }

        self._lh_regressed = {
            name: False
            for name in names
        }

        self._lh_first_success_step = None
        self._lh_progress_curve = []

    def _reset_internal(self):
        """
        Added only for metric initialization.

        _get_raw_semantic_milestone_state() does not read toaster_oven.get_state()
        unless both food items are already placed and the door is closed, so
        reset is safe.
        """
        super()._reset_internal()

        initial_raw = (
            self._get_raw_semantic_milestone_state()
        )

        self._reset_long_horizon_metric_tracker(
            initial_state=initial_raw
        )

    def _get_current_ordered_progress(
        self,
        ordered_state=None,
    ):
        if ordered_state is None:
            ordered_state = (
                self._get_ordered_semantic_milestone_state()
            )

        return (
            int(bool(ordered_state["baguette_in_hand"]))
            + int(bool(ordered_state["baguette_in_toaster"]))
            + int(bool(ordered_state["kebab_in_hand"]))
            + int(bool(ordered_state["kebab_in_toaster"]))
            + int(bool(ordered_state["toaster_door_closed"]))
            + int(bool(ordered_state["timer_set"]))
        ) / 6.0

    def _update_long_horizon_metric_tracker(self):
        self._lh_step_count += 1

        raw = (
            self._get_raw_semantic_milestone_state()
        )

        names = list(
            self.get_semantic_milestone_spec().keys()
        )

        # --------------------------------------------------------------
        # UPS: order ignored.
        # --------------------------------------------------------------
        for name in names:
            if raw[name]:
                self._lh_unordered_ever[name] = True

        # --------------------------------------------------------------
        # Baguette branch.
        # --------------------------------------------------------------
        if raw["baguette_in_hand"]:
            self._lh_ordered_ever["baguette_in_hand"] = True

        if raw["baguette_in_toaster"]:
            # If grasp detection flickered and missed the pick, infer it from
            # a valid toaster placement.
            if not self._lh_ordered_ever["baguette_in_hand"]:
                self._lh_ordered_ever["baguette_in_hand"] = True
            self._lh_ordered_ever["baguette_in_toaster"] = True

        # --------------------------------------------------------------
        # Kebab branch.
        # --------------------------------------------------------------
        if raw["kebab_in_hand"]:
            self._lh_ordered_ever["kebab_in_hand"] = True

        if raw["kebab_in_toaster"]:
            if not self._lh_ordered_ever["kebab_in_hand"]:
                self._lh_ordered_ever["kebab_in_hand"] = True
            self._lh_ordered_ever["kebab_in_toaster"] = True

        both_food_valid = bool(
            raw["baguette_in_toaster"]
            and raw["kebab_in_toaster"]
            and self._lh_ordered_ever["baguette_in_toaster"]
            and self._lh_ordered_ever["kebab_in_toaster"]
        )

        # --------------------------------------------------------------
        # Door close after both food items are validly placed.
        #
        # Only the final closed door condition is counted; rack push / pull is
        # ignored.
        # --------------------------------------------------------------
        door_close_event = bool(
            raw["toaster_door_closed"]
            and not self._lh_prev_raw[
                "toaster_door_closed"
            ]
        )

        if (
            door_close_event
            and both_food_valid
        ):
            self._lh_ordered_ever[
                "toaster_door_closed"
            ] = True

        # --------------------------------------------------------------
        # Timer / start heating after valid door close.
        # --------------------------------------------------------------
        timer_event = bool(
            raw["timer_set"]
            and not self._lh_prev_raw[
                "timer_set"
            ]
        )

        if (
            timer_event
            and self._lh_ordered_ever[
                "toaster_door_closed"
            ]
            and raw[
                "toaster_door_closed"
            ]
        ):
            self._lh_ordered_ever[
                "timer_set"
            ] = True

        ordered_state = (
            self._get_ordered_semantic_milestone_state()
        )

        # --------------------------------------------------------------
        # Regression bookkeeping for persistent place / close / timer states.
        # --------------------------------------------------------------
        for name in [
            "baguette_in_toaster",
            "kebab_in_toaster",
            "toaster_door_closed",
            "timer_set",
        ]:
            if not self._lh_ordered_ever[name]:
                continue

            prev_valid = bool(
                self._lh_prev_raw[name]
            )

            falling_edge = bool(
                prev_valid
                and not ordered_state[name]
            )

            if falling_edge:
                self._lh_regressed[name] = True

        current_ops = (
            self._get_current_ordered_progress(
                ordered_state
            )
        )

        self._lh_progress_curve.append(
            float(current_ops)
        )

        # SR / NCT:
        # Calling original _check_success() is safe here only after normal
        # environment stepping, exactly as in the original benchmark path.
        success = False
        if (
            raw["baguette_in_toaster"]
            and raw["kebab_in_toaster"]
            and raw["toaster_door_closed"]
        ):
            success = bool(
                self._check_success()
            )

        if (
            success
            and self._lh_first_success_step
            is None
        ):
            self._lh_first_success_step = (
                self._lh_step_count
            )

        self._lh_prev_raw = {
            name: bool(raw[name])
            for name in names
        }

    def _post_action(self, action):
        reward, done, info = (
            super()._post_action(action)
        )

        self._update_long_horizon_metric_tracker()

        return reward, done, info

    def get_long_horizon_metrics(self):
        names = list(
            self.get_semantic_milestone_spec().keys()
        )

        ups = (
            sum(
                int(bool(self._lh_unordered_ever[name]))
                for name in names
            )
            / float(len(names))
        )

        ops = self._get_current_ordered_progress()

        return {
            "SR": float(
                self._lh_first_success_step
                is not None
            ),
            "UPS": float(ups),
            "OPS": float(ops),
            "NCT": (
                -1.0
                if self._lh_first_success_step
                is None
                else float(
                    self._lh_first_success_step
                )
                / float(
                    getattr(
                        self,
                        "horizon",
                        1,
                    )
                )
            ),
            "progress_curve": list(
                self._lh_progress_curve
            ),
        }

    # =====================================================================
    # ORIGINAL _check_success(): EXACTLY PRESERVED
    # =====================================================================

    def _check_success(self):
        baguette_in_toaster = self.toaster_oven.check_rack_contact(
            self, "baguette", rack_level=0
        ) or self.toaster_oven.check_rack_contact(self, "baguette", rack_level=1)

        kebab_in_toaster = self.toaster_oven.check_rack_contact(
            self, "kebab", rack_level=0
        ) or self.toaster_oven.check_rack_contact(self, "kebab", rack_level=1)
        toaster_oven_state = self.toaster_oven.get_state()

        if baguette_in_toaster and kebab_in_toaster:
            toaster_oven_state = self.toaster_oven.get_state(self)
            if self.toaster_oven.is_open(self):
                return False
            return toaster_oven_state["time"] >= 0.1
        else:
            return False

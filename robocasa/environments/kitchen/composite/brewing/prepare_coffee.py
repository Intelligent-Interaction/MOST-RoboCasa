from robocasa.environments.kitchen.kitchen import *


class PrepareCoffee(Kitchen):
    """
    Prepare Coffee: composite task for Brewing activity.

    Steps:
        Pick the mug from the cabinet, place it under the coffee machine dispenser,
        and press the start button.

    Long-horizon semantic milestones:
        M1. mug_in_hand
        M2. mug_under_dispenser
        M3. coffee_machine_on

    Dependency chain:
        mug_in_hand -> mug_under_dispenser -> coffee_machine_on
    """

    def __init__(self, cab_id=FixtureType.CABINET, *args, **kwargs):
        self.cab_id = cab_id
        self._reset_long_horizon_metric_tracker(initial_state=None)
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.coffee_machine = self.register_fixture_ref(
            "coffee_machine", dict(id=FixtureType.COFFEE_MACHINE)
        )
        self.cab = self.register_fixture_ref(
            "cab", dict(id=self.cab_id, ref=self.coffee_machine)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_name = self.get_obj_lang()
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions).format(
                obj_name=obj_name
            )
        else:
            ep_meta["lang"] = (
                f"Pick the {obj_name} from the cabinet, place it under the "
                "coffee machine dispenser, and press the start button."
            )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups="mug",
                placement=dict(
                    fixture=self.cab,
                    size=(0.30, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_cab",
                obj_groups="all",
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.20),
                    pos=(0.0, 1.0),
                    offset=(0.0, 0.0),
                ),
            )
        )

        return cfgs

    def _setup_scene(self):
        super()._setup_scene()
        self.cab.open_door(env=self)

    # =====================================================================
    # Task-specific semantic milestone definitions
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "mug_in_hand": {
                "description": "The mug is grasped by the robot gripper.",
                "dependencies": [],
                "persistent": False,
            },
            "mug_under_dispenser": {
                "description": (
                    "The mug is placed under the coffee machine dispenser."
                ),
                "dependencies": ["mug_in_hand"],
                "persistent": True,
            },
            "coffee_machine_on": {
                "description": (
                    "The coffee machine is started after mug placement."
                ),
                "dependencies": ["mug_under_dispenser"],
                "persistent": True,
            },
        }

    def _check_mug_grasped(self):
        """
        Local mug grasp check.

        Use a slightly looser gripper-close threshold than RoboCasa's default
        0.035. This avoids missing visually valid mug grasps while keeping the
        change local to this task.
        """
        try:
            return bool(
                OU.check_obj_grasped(
                    self,
                    "obj",
                    threshold=0.10,
                )
            )
        except TypeError:
            return bool(
                OU.check_obj_grasped(
                    self,
                    "obj",
                )
            )

    def _get_raw_semantic_milestone_state(self):
        mug_in_hand = self._check_mug_grasped()

        mug_under_dispenser = bool(
            self.coffee_machine.check_receptacle_placement_for_pouring(
                self,
                "obj",
            )
            and not mug_in_hand
        )

        coffee_machine_on = bool(
            self.coffee_machine._turned_on
        )

        return {
            "mug_in_hand": bool(mug_in_hand),
            "mug_under_dispenser": bool(mug_under_dispenser),
            "coffee_machine_on": bool(coffee_machine_on),
        }

    def _get_ordered_semantic_milestone_state(self):
        """
        Current ordered-valid milestone states.

        The task is linear, but pick and place are not independent current
        states. Once the mug is validly placed, the pick stage is considered
        superseded rather than lost.
        """
        raw = self._get_raw_semantic_milestone_state()

        mug_place_valid = bool(
            self._lh_ordered_ever["mug_under_dispenser"]
            and raw["mug_under_dispenser"]
        )

        mug_pick_valid = bool(
            self._lh_ordered_ever["mug_in_hand"]
            and (
                raw["mug_in_hand"]
                or mug_place_valid
            )
        )

        coffee_on_valid = bool(
            self._lh_ordered_ever["coffee_machine_on"]
            and raw["coffee_machine_on"]
            and mug_place_valid
        )

        return {
            "mug_in_hand": mug_pick_valid,
            "mug_under_dispenser": mug_place_valid,
            "coffee_machine_on": coffee_on_valid,
        }

    # =====================================================================
    # Per-episode long-horizon metric tracker
    # =====================================================================

    def _reset_long_horizon_metric_tracker(self, initial_state=None):
        spec = self.get_semantic_milestone_spec()
        names = list(spec.keys())

        self._lh_step_count = 0
        self._lh_progress_curve = []

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

        self._lh_first_success_step = None

    def _reset_internal(self):
        super()._reset_internal()
        initial_raw = self._get_raw_semantic_milestone_state()
        self._reset_long_horizon_metric_tracker(
            initial_state=initial_raw
        )

    def _get_current_ordered_progress(self, ordered_state):
        return (
            int(bool(ordered_state["mug_in_hand"]))
            + int(bool(ordered_state["mug_under_dispenser"]))
            + int(bool(ordered_state["coffee_machine_on"]))
        ) / 3.0

    def _update_long_horizon_metric_tracker(self):
        self._lh_step_count += 1

        raw = self._get_raw_semantic_milestone_state()

        # UPS: physical milestone states ever happened, regardless of order.
        for name, value in raw.items():
            if value:
                self._lh_unordered_ever[name] = True

        # M1: pick mug.
        if raw["mug_in_hand"]:
            self._lh_ordered_ever["mug_in_hand"] = True

        # M2: place mug after it has been picked.
        if (
            raw["mug_under_dispenser"]
            and self._lh_ordered_ever["mug_in_hand"]
        ):
            self._lh_ordered_ever["mug_under_dispenser"] = True

        # M3: coffee machine must be turned on after mug placement.
        #
        # This uses a rising edge so an early button press does not receive
        # ordered credit retroactively after the mug is placed.
        coffee_on_rising_edge = bool(
            raw["coffee_machine_on"]
            and not self._lh_prev_raw["coffee_machine_on"]
        )

        if (
            coffee_on_rising_edge
            and raw["mug_under_dispenser"]
            and self._lh_ordered_ever["mug_under_dispenser"]
        ):
            self._lh_ordered_ever["coffee_machine_on"] = True

        ordered = self._get_ordered_semantic_milestone_state()
        self._lh_progress_curve.append(
            float(
                self._get_current_ordered_progress(
                    ordered,
                )
            )
        )

        if (
            self._check_success()
            and self._lh_first_success_step is None
        ):
            self._lh_first_success_step = self._lh_step_count

        self._lh_prev_raw = {
            name: bool(value)
            for name, value in raw.items()
        }

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        self._update_long_horizon_metric_tracker()
        return reward, done, info

    # =====================================================================
    # Public metric API
    # =====================================================================

    def get_long_horizon_metrics(self):
        spec = self.get_semantic_milestone_spec()
        num_milestones = len(spec)

        ups = (
            sum(self._lh_unordered_ever.values()) / float(num_milestones)
            if num_milestones > 0
            else 0.0
        )

        ordered = self._get_ordered_semantic_milestone_state()
        ops = self._get_current_ordered_progress(
            ordered,
        )

        if self._lh_first_success_step is None:
            nct = -1.0
        else:
            horizon = float(
                getattr(
                    self,
                    "horizon",
                    0,
                )
            )
            nct = (
                float(self._lh_first_success_step) / horizon
                if horizon > 0
                else float(self._lh_first_success_step)
            )

        return {
            "SR": float(
                self._lh_first_success_step is not None
            ),
            "NCT": float(nct),
            "UPS": float(ups),
            "OPS": float(ops),
            "progress_curve": list(
                self._lh_progress_curve
            ),
        }

    # =====================================================================
    # Original RoboCasa success criterion, preserved
    # =====================================================================

    def _check_success(self):
        gripper_obj_far = OU.gripper_obj_far(self)
        contact_check = self.coffee_machine.check_receptacle_placement_for_pouring(
            self,
            "obj",
        )
        gripper_button_far = self.coffee_machine.gripper_button_far(self)

        return bool(
            contact_check
            and gripper_obj_far
            and self.coffee_machine._turned_on
            and gripper_button_far
        )

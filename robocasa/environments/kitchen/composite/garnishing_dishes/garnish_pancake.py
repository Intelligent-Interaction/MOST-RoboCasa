from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.object_utils as OU


class GarnishPancake(Kitchen):
    """
    Garnish Pancake: composite task for Garnishing Dishes activity.

    Task:
        1. Open the fridge door.
        2. Pick the strawberry.
        3. Place the strawberry on top of the pancake.

    Progress milestones:
        M1. fridge_open
        M2. strawberry_in_hand
        M3. strawberry_on_pancake

    Original RoboCasa _check_success() is preserved.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        kwargs["obj_registries"] = ("aigen", "objaverse", "lightwheel")
        self._reset_long_horizon_metric_tracker(initial_state=None)
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.fridge = self.register_fixture_ref(
            "fridge",
            dict(id=FixtureType.FRIDGE),
        )

        self.stool = self.register_fixture_ref(
            "stool",
            dict(id=FixtureType.STOOL),
        )

        self.dining_counter = self.register_fixture_ref(
            "dining_counter",
            dict(
                id=FixtureType.DINING_COUNTER,
                ref=self.stool,
            ),
        )

        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(
                self.novel_instructions
            )
        else:
            ep_meta["lang"] = (
                "Take the strawberry from the fridge and place it on top of "
                "the pancake, located on the dining counter."
            )

        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="strawberry",
                obj_groups="strawberry",
                object_scale=0.9,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.15, 0.15),
                    pos=(0, -1.0),
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5)
                    ),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_fridge",
                obj_groups="food",
                placement=dict(
                    fixture=self.fridge,
                    size=(0.30, 0.30),
                    pos=(0.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="pancake",
                obj_groups="pancake",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool
                    ),
                    size=(0.35, 0.35),
                    pos=("ref", "ref"),
                    try_to_place_in="plate",
                ),
            )
        )

        return cfgs

    # =====================================================================
    # Task-specific semantic milestones
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "fridge_open": {
                "description": "The fridge door has been opened.",
                "dependencies": [],
                "persistent": False,
            },
            "strawberry_in_hand": {
                "description": "The strawberry is grasped by the robot gripper.",
                "dependencies": ["fridge_open"],
                "persistent": False,
            },
            "strawberry_on_pancake": {
                "description": "The strawberry is placed on top of the pancake.",
                "dependencies": ["strawberry_in_hand"],
                "persistent": True,
            },
        }

    def _fridge_is_open(self):
        """
        Robust fridge-open check.

        Different RoboCasa fixtures expose door state through slightly different
        helpers. This method tries the common APIs and returns False if no
        reliable signal is available.
        """
        try:
            return bool(
                not self.fridge.is_closed(env=self)
            )
        except Exception:
            pass

        try:
            door_state = self.fridge.get_door_state(env=self)
            if isinstance(door_state, dict):
                joint_values = []
                for value in door_state.values():
                    if isinstance(value, dict):
                        for sub_value in value.values():
                            if isinstance(sub_value, (int, float, np.floating)):
                                joint_values.append(float(sub_value))
                    elif isinstance(value, (int, float, np.floating)):
                        joint_values.append(float(value))

                if joint_values:
                    return bool(
                        max(joint_values) > 0.10
                    )
        except Exception:
            pass

        return False

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

    def _get_raw_semantic_milestone_state(self):
        strawberry_in_hand = self._check_task_obj_grasped(
            "strawberry",
        )

        strawberry_on_pancake = bool(
            OU.check_obj_in_receptacle(
                self,
                "strawberry",
                "pancake",
            )
            and not strawberry_in_hand
        )

        return {
            "fridge_open": bool(
                self._fridge_is_open()
            ),
            "strawberry_in_hand": strawberry_in_hand,
            "strawberry_on_pancake": strawberry_on_pancake,
        }

    def _get_ordered_semantic_milestone_state(self):
        """
        Current ordered-valid states.

        fridge_open is a latched event milestone. It does not need to remain
        open after the strawberry has been retrieved.

        strawberry_in_hand is current/superseded:
            - valid while the strawberry is currently held;
            - also valid once strawberry_on_pancake is valid.

        If the strawberry is picked and dropped somewhere invalid, current OPS
        goes back down.
        """
        raw = self._get_raw_semantic_milestone_state()

        fridge_open_valid = bool(
            self._lh_ordered_ever["fridge_open"]
        )

        place_valid = bool(
            self._lh_ordered_ever["strawberry_on_pancake"]
            and raw["strawberry_on_pancake"]
        )

        pick_valid = bool(
            self._lh_ordered_ever["strawberry_in_hand"]
            and fridge_open_valid
            and (
                raw["strawberry_in_hand"]
                or place_valid
            )
        )

        return {
            "fridge_open": fridge_open_valid,
            "strawberry_in_hand": pick_valid,
            "strawberry_on_pancake": bool(
                place_valid
                and fridge_open_valid
            ),
        }

    # =====================================================================
    # Per-episode long-horizon metric tracker
    # =====================================================================

    def _reset_long_horizon_metric_tracker(
        self,
        initial_state=None,
    ):
        spec = self.get_semantic_milestone_spec()
        names = list(spec.keys())

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
            int(bool(ordered_state["fridge_open"]))
            + int(bool(ordered_state["strawberry_in_hand"]))
            + int(bool(ordered_state["strawberry_on_pancake"]))
        ) / 3.0

    def _update_long_horizon_metric_tracker(self):
        self._lh_step_count += 1

        spec = self.get_semantic_milestone_spec()
        raw = self._get_raw_semantic_milestone_state()

        # --------------------------------------------------------------
        # UPS: order ignored.
        # --------------------------------------------------------------
        for name in spec.keys():
            if raw[name]:
                self._lh_unordered_ever[name] = True

        # --------------------------------------------------------------
        # M1: open fridge.
        # --------------------------------------------------------------
        fridge_open_rising = bool(
            raw["fridge_open"]
            and not self._lh_prev_raw["fridge_open"]
        )

        if fridge_open_rising:
            self._lh_ordered_ever["fridge_open"] = True

        # If the fridge already starts open for a particular reset, treat that
        # as available access rather than blocking the whole curve at zero.
        if (
            raw["fridge_open"]
            and not self._lh_ordered_ever["fridge_open"]
        ):
            self._lh_ordered_ever["fridge_open"] = True

        fridge_ready = bool(
            self._lh_ordered_ever["fridge_open"]
        )

        # --------------------------------------------------------------
        # M2: pick strawberry after fridge is open.
        # --------------------------------------------------------------
        if (
            raw["strawberry_in_hand"]
            and fridge_ready
        ):
            self._lh_ordered_ever["strawberry_in_hand"] = True

        # --------------------------------------------------------------
        # M3: place strawberry on pancake.
        #
        # If grasp detection flickered and missed the pick, infer it from a
        # valid strawberry-on-pancake placement after fridge access.
        # --------------------------------------------------------------
        if (
            raw["strawberry_on_pancake"]
            and fridge_ready
        ):
            if not self._lh_ordered_ever["strawberry_in_hand"]:
                self._lh_ordered_ever["strawberry_in_hand"] = True
            self._lh_ordered_ever["strawberry_on_pancake"] = True

        ordered_state = (
            self._get_ordered_semantic_milestone_state()
        )

        # --------------------------------------------------------------
        # Regression bookkeeping for persistent placement.
        # --------------------------------------------------------------
        if self._lh_ordered_ever["strawberry_on_pancake"]:
            prev_ordered_valid = bool(
                self._lh_ordered_ever["strawberry_on_pancake"]
                and self._lh_prev_raw["strawberry_on_pancake"]
            )

            falling_edge = bool(
                prev_ordered_valid
                and not ordered_state["strawberry_on_pancake"]
            )

            if falling_edge:
                self._lh_regressed["strawberry_on_pancake"] = True

        current_ops = (
            self._get_current_ordered_progress(
                ordered_state
            )
        )

        self._lh_progress_curve.append(
            float(current_ops)
        )

        success = bool(
            self._check_success()
        )

        if (
            success
            and self._lh_first_success_step is None
        ):
            self._lh_first_success_step = (
                self._lh_step_count
            )

        self._lh_prev_raw = {
            name: bool(raw[name])
            for name in spec.keys()
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
        names = list(spec.keys())

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
                self._lh_first_success_step is not None
            ),
            "OPS": float(ops),
            "UPS": float(ups),
            "NCT": (
                -1.0
                if self._lh_first_success_step is None
                else float(self._lh_first_success_step)
                / float(getattr(self, "horizon", 1))
            ),
            "progress_curve": list(
                self._lh_progress_curve
            ),
        }

    # =====================================================================
    # Original RoboCasa success criterion, preserved
    # =====================================================================

    def _check_success(self):
        strawberry_on_pancake = OU.check_obj_in_receptacle(
            self,
            "strawberry",
            "pancake",
        )

        pancake_on_plate = OU.check_obj_in_receptacle(
            self,
            "pancake",
            "pancake_container",
        )

        plate_on_table = OU.check_obj_fixture_contact(
            self,
            "pancake_container",
            self.dining_counter,
        )

        gripper_far_strawberry = OU.gripper_obj_far(
            self,
            "strawberry",
        )

        return (
            strawberry_on_pancake
            and pancake_on_plate
            and plate_on_table
            and gripper_far_strawberry
        )

from robocasa.environments.kitchen.kitchen import *
import numpy as np
import robocasa.utils.object_utils as OU


class MakeIceLemonade(Kitchen):
    """
    Prepare Ice Lemonade: composite task for Adding Ice to Beverages activity.

    Task variants:
        A) Pick ice, place ice in cup, open fridge, pick lemon, place lemon.
        B) Open fridge, pick lemon, place lemon, pick ice, place ice.

    Progress milestones:
        fridge_open
        lemon_wedge_in_hand
        lemon_in_cup
        ice_in_hand
        ice_in_cup

    Ordering:
        - Ice PnP is independent and can happen before or after lemon.
        - Lemon PnP depends on fridge_open.
        - The two ice cubes are interchangeable. Either ice_cube1 or ice_cube2
          can satisfy ice_in_hand / ice_in_cup.

    For this task only, _check_success() is intentionally made more robust:
        The original OU.check_obj_in_receptacle contact-based cup check can
        flicker when both lemon and ice are crowded inside the glass cup.
        We therefore use:
            contact-based in-cup check OR stable XY containment around the cup.
    """

    def __init__(self, *args, **kwargs):
        self._reset_long_horizon_metric_tracker(initial_state=None)
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref(
            "fridge",
            dict(id=FixtureType.FRIDGE),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.fridge),
        )
        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta["lang"] = (
                "Grab a lemon wedge from the fridge and one ice cube from the ice bowl,"
                " and put them in the glass of lemonade."
            )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        OU.add_obj_liquid_site(
            self,
            "glass_cup",
            [0.95, 0.95, 0.6, 0.25],
        )

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="lemon_wedge",
                obj_groups="lemon_wedge",
                placement=dict(
                    fixture=self.fridge,
                    size=(0.25, 0.20),
                    pos=(0, -1.0),
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                ),
            )
        )

        cfgs.append(
            dict(
                name="fridge_distractor",
                fridgable=True,
                exclude_obj_groups="lemon_wedge",
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.3),
                    pos=(0, 1.0),
                    sample_region_kwargs=dict(
                        z_range=(0.5, 1.5),
                    ),
                ),
            )
        )

        cfgs.append(
            dict(
                name="ice_bowl",
                obj_groups="bowl",
                object_scale=[1.1, 1.1, 0.75],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.fridge,
                    ),
                    size=(1.0, 0.4),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="ice_cube1",
                obj_groups="ice_cube",
                object_scale=0.9,
                placement=dict(
                    object="ice_bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="ice_cube2",
                obj_groups="ice_cube",
                object_scale=0.9,
                placement=dict(
                    object="ice_bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="glass_cup",
                obj_groups="glass_cup",
                object_scale=[1.25, 1.25, 1],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.fridge,
                    ),
                    size=(0.5, 0.25),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    # =====================================================================
    # Robust cup containment helpers
    # =====================================================================

    def _body_pos(
        self,
        obj_name,
    ):
        return np.array(
            self.sim.data.body_xpos[
                self.obj_body_id[obj_name]
            ]
        )

    def _check_obj_in_cup_contact(
        self,
        obj_name,
    ):
        """
        Original contact-based receptacle check.

        It is still useful when stable, but it may flicker when lemon and ice
        crowd the glass cup.
        """
        return bool(
            OU.check_obj_in_receptacle(
                self,
                obj_name,
                "glass_cup",
                th=0.5,
            )
        )

    def _check_obj_in_cup_xy(
        self,
        obj_name,
    ):
        """
        Contact-free fallback for crowded cup contents.

        The XY threshold is intentionally much smaller than the original th=0.5
        distance used with contact. This avoids counting objects merely near the
        cup while still tolerating contact flicker inside the cup.
        """
        obj_pos = self._body_pos(obj_name)
        cup_pos = self._body_pos("glass_cup")

        cup_obj = self.objects["glass_cup"]
        cup_radius = float(
            getattr(
                cup_obj,
                "horizontal_radius",
                0.06,
            )
        )

        xy_thresh = max(
            0.065,
            min(
                0.12,
                cup_radius * 1.25,
            ),
        )

        xy_inside = bool(
            np.linalg.norm(
                obj_pos[:2] - cup_pos[:2]
            )
            <= xy_thresh
        )

        # Very loose vertical sanity check. The goal is only to reject objects
        # clearly below / far above the cup area, not to overfit cup geometry.
        z_reasonable = bool(
            obj_pos[2] >= cup_pos[2] - 0.06
            and obj_pos[2] <= cup_pos[2] + 0.22
        )

        return bool(
            xy_inside
            and z_reasonable
        )

    def _check_obj_in_cup_stable(
        self,
        obj_name,
    ):
        """
        Robust in-cup predicate:
            contact-based check OR stable XY containment.

        This is used by both metrics and _check_success() for this task only.
        """
        return bool(
            self._check_obj_in_cup_contact(obj_name)
            or self._check_obj_in_cup_xy(obj_name)
        )

    def _check_obj_in_cup_released(
        self,
        obj_name,
        obj_in_hand,
    ):
        return bool(
            self._check_obj_in_cup_stable(obj_name)
            and not obj_in_hand
        )

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
            "lemon_wedge_in_hand": {
                "description": "The lemon wedge is grasped by the robot gripper.",
                "dependencies": ["fridge_open"],
                "persistent": False,
            },
            "lemon_in_cup": {
                "description": "The lemon wedge is placed in the glass cup.",
                "dependencies": ["lemon_wedge_in_hand"],
                "persistent": True,
            },
            "ice_in_hand": {
                "description": (
                    "At least one interchangeable ice cube is grasped by the "
                    "robot gripper."
                ),
                "dependencies": [],
                "persistent": False,
            },
            "ice_in_cup": {
                "description": (
                    "At least one interchangeable ice cube is placed in the "
                    "glass cup."
                ),
                "dependencies": ["ice_in_hand"],
                "persistent": True,
            },
        }

    def _fridge_is_open(self):
        """
        Robust fridge-open check.

        Different fridge fixtures may expose door state with slightly different
        helpers. If no reliable signal is available, return False. The tracker
        also infers fridge_open if lemon is later picked or placed.
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
                values = []

                for value in door_state.values():
                    if isinstance(value, dict):
                        for sub_value in value.values():
                            if isinstance(sub_value, (int, float, np.floating)):
                                values.append(float(sub_value))
                    elif isinstance(value, (int, float, np.floating)):
                        values.append(float(value))

                if values:
                    return bool(
                        max(values) > 0.10
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
        lemon_in_hand = self._check_task_obj_grasped(
            "lemon_wedge",
        )

        ice1_in_hand = self._check_task_obj_grasped(
            "ice_cube1",
        )
        ice2_in_hand = self._check_task_obj_grasped(
            "ice_cube2",
        )

        lemon_in_cup = self._check_obj_in_cup_released(
            "lemon_wedge",
            lemon_in_hand,
        )

        ice1_in_cup = self._check_obj_in_cup_released(
            "ice_cube1",
            ice1_in_hand,
        )
        ice2_in_cup = self._check_obj_in_cup_released(
            "ice_cube2",
            ice2_in_hand,
        )

        return {
            "fridge_open": bool(
                self._fridge_is_open()
            ),
            "lemon_wedge_in_hand": lemon_in_hand,
            "lemon_in_cup": lemon_in_cup,
            "ice_in_hand": bool(
                ice1_in_hand or ice2_in_hand
            ),
            "ice_in_cup": bool(
                ice1_in_cup or ice2_in_cup
            ),
        }

    def _get_ordered_semantic_milestone_state(self):
        """
        Current ordered-valid states.

        fridge_open is a latched event milestone.

        lemon branch:
            fridge_open -> lemon_wedge_in_hand -> lemon_in_cup

        ice branch:
            ice_in_hand -> ice_in_cup

        The ice branch is independent and can be completed before or after
        fridge/lemon.
        """
        raw = self._get_raw_semantic_milestone_state()

        fridge_valid = bool(
            self._lh_ordered_ever["fridge_open"]
        )

        lemon_place_valid = bool(
            self._lh_ordered_ever["lemon_in_cup"]
            and raw["lemon_in_cup"]
            and fridge_valid
        )

        lemon_pick_valid = bool(
            self._lh_ordered_ever["lemon_wedge_in_hand"]
            and fridge_valid
            and (
                raw["lemon_wedge_in_hand"]
                or lemon_place_valid
            )
        )

        ice_place_valid = bool(
            self._lh_ordered_ever["ice_in_cup"]
            and raw["ice_in_cup"]
        )

        ice_pick_valid = bool(
            self._lh_ordered_ever["ice_in_hand"]
            and (
                raw["ice_in_hand"]
                or ice_place_valid
            )
        )

        return {
            "fridge_open": fridge_valid,
            "lemon_wedge_in_hand": lemon_pick_valid,
            "lemon_in_cup": lemon_place_valid,
            "ice_in_hand": ice_pick_valid,
            "ice_in_cup": ice_place_valid,
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
            + int(bool(ordered_state["lemon_wedge_in_hand"]))
            + int(bool(ordered_state["lemon_in_cup"]))
            + int(bool(ordered_state["ice_in_hand"]))
            + int(bool(ordered_state["ice_in_cup"]))
        ) / 5.0

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
        # M0: open fridge.
        #
        # If fridge-open detection flickers or is missed, infer it once lemon
        # is picked or placed, because lemon starts in the fridge.
        # --------------------------------------------------------------
        if (
            raw["fridge_open"]
            or raw["lemon_wedge_in_hand"]
            or raw["lemon_in_cup"]
        ):
            self._lh_ordered_ever["fridge_open"] = True
            self._lh_unordered_ever["fridge_open"] = True

        fridge_ready = bool(
            self._lh_ordered_ever["fridge_open"]
        )

        # --------------------------------------------------------------
        # Lemon branch: fridge -> pick lemon -> place lemon.
        # --------------------------------------------------------------
        if (
            raw["lemon_wedge_in_hand"]
            and fridge_ready
        ):
            self._lh_ordered_ever["lemon_wedge_in_hand"] = True

        if (
            raw["lemon_in_cup"]
            and fridge_ready
        ):
            if not self._lh_ordered_ever["lemon_wedge_in_hand"]:
                self._lh_ordered_ever["lemon_wedge_in_hand"] = True
            self._lh_ordered_ever["lemon_in_cup"] = True

        # --------------------------------------------------------------
        # Ice branch: independent PnP, can happen before or after lemon.
        # --------------------------------------------------------------
        if raw["ice_in_hand"]:
            self._lh_ordered_ever["ice_in_hand"] = True

        if raw["ice_in_cup"]:
            if not self._lh_ordered_ever["ice_in_hand"]:
                self._lh_ordered_ever["ice_in_hand"] = True
            self._lh_ordered_ever["ice_in_cup"] = True

        ordered_state = (
            self._get_ordered_semantic_milestone_state()
        )

        # --------------------------------------------------------------
        # Regression bookkeeping for persistent placements.
        # --------------------------------------------------------------
        for name in [
            "lemon_in_cup",
            "ice_in_cup",
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
    # Task-specific _check_success() intentionally changed for this task only
    # =====================================================================

    def _check_success(self):
        lemon_in_cup = self._check_obj_in_cup_stable(
            "lemon_wedge",
        )

        ice1_in_cup = self._check_obj_in_cup_stable(
            "ice_cube1",
        )

        ice2_in_cup = self._check_obj_in_cup_stable(
            "ice_cube2",
        )

        ice_in_cup = bool(
            ice1_in_cup
            or ice2_in_cup
        )

        gripper_far_lemon = OU.gripper_obj_far(
            self,
            "lemon_wedge",
            th=0.15,
        )

        gripper_far_ice = all(
            [
                OU.gripper_obj_far(
                    self,
                    "ice_cube1",
                    th=0.15,
                ),
                OU.gripper_obj_far(
                    self,
                    "ice_cube2",
                    th=0.15,
                ),
            ]
        )

        return bool(
            lemon_in_cup
            and ice_in_cup
            and gripper_far_lemon
            and gripper_far_ice
        )

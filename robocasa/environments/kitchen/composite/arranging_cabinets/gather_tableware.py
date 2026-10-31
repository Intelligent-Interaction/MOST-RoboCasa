from robocasa.environments.kitchen.kitchen import *
import numpy as np
import robocasa.utils.object_utils as OU


class GatherTableware(Kitchen):
    """
    Gather Glasses: composite task for Arranging Cabinets activity.

    Task:
        First pick glass3 from the second cabinet, carry it back to the target
        cabinet, and place it there. Sometimes the demonstration contains one
        or two additional pick-place adjustments to put mugs of the same type
        together.

    Progress design:
        Only real pick-place actions are counted as progress milestones.

        Mandatory PnP:
            1. pick_remote_mug
            2. place_remote_mug_in_target_cabinet

        Optional observed PnPs:
            3. pick_adjust_object_1
            4. place_adjust_object_1
            5. pick_adjust_object_2
            6. place_adjust_object_2

        There is NO separate "sort" milestone. The final grouped/sorted
        relation is treated as the natural result of the place actions and is
        still used by the original _check_success(), which is preserved.

    Dynamic normalization:
        - If the episode only contains the mandatory PnP, denominator = 2:
              pick remote -> 0.5, place remote -> 1.0
        - If one extra adjustment PnP is observed, denominator = 4.
        - If two extra adjustment PnPs are observed, denominator = 6.

        Because the number of optional PnPs is only known after they occur, the
        stored progress_curve is recomputed from raw completed-action counts
        whenever the denominator expands. This keeps the final exported curve
        semantically clean.
    """

    def __init__(self, *args, **kwargs):
        self._reset_long_horizon_metric_tracker(initial_state=None)
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        if "cab1" in self.fixture_refs:
            self.cab = self.fixture_refs["cab1"]
            self.cab2 = self.fixture_refs["cab2"]
        else:
            while True:
                self.cab = self.get_fixture(FixtureType.CABINET)

                valid_cab_config_found = False
                for _ in range(100):
                    self.cab2 = self.get_fixture(FixtureType.CABINET)
                    if self.cab2 != self.cab:
                        valid_cab_config_found = True
                        break

                if valid_cab_config_found:
                    break

            self.fixture_refs["cab1"] = self.cab
            self.fixture_refs["cab2"] = self.cab2

        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.cab),
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta["lang"] = (
                "Gather all objects into one cabinet and sort the glasses "
                "and bowls to opposite sides."
            )
        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.cab.open_door(self)
        self.cab2.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="glass1",
                graspable=True,
                obj_groups=["mug"],
                placement=dict(
                    fixture=self.cab,
                    size=(0.8, 0.8),
                    pos=(0.7, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="glass2",
                graspable=True,
                obj_groups=["mug"],
                placement=dict(
                    fixture=self.cab,
                    size=(0.60, 0.60),
                    pos=(-0.7, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="glass3",
                obj_groups=["mug"],
                graspable=True,
                placement=dict(
                    fixture=self.cab2,
                    size=(0.40, 0.40),
                    pos=(0, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.80, 0.80),
                    pos=(0, 0),
                ),
            )
        )

        return cfgs

    def dist_between_obj(self, obj_name1, obj_name2):
        obj1 = self.objects[obj_name1]
        obj2 = self.objects[obj_name2]

        obj1_pos = np.array(
            self.sim.data.body_xpos[
                self.obj_body_id[obj1.name]
            ]
        )
        obj2_pos = np.array(
            self.sim.data.body_xpos[
                self.obj_body_id[obj2.name]
            ]
        )

        return np.linalg.norm(
            obj1_pos - obj2_pos
        )

    # =====================================================================
    # Task-specific semantic milestones
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "remote_mug_in_hand": {
                "description": (
                    "The remote mug glass3 from the second cabinet is grasped."
                ),
                "dependencies": [],
                "persistent": False,
            },
            "remote_mug_in_target_cabinet": {
                "description": (
                    "The remote mug glass3 is placed in the target cabinet."
                ),
                "dependencies": ["remote_mug_in_hand"],
                "persistent": False,
            },
            "adjust_pnp1_object_in_hand": {
                "description": (
                    "The first optional adjustment object is grasped."
                ),
                "dependencies": ["remote_mug_in_target_cabinet"],
                "persistent": False,
            },
            "adjust_pnp1_object_placed": {
                "description": (
                    "The first optional adjustment object is placed."
                ),
                "dependencies": ["adjust_pnp1_object_in_hand"],
                "persistent": False,
            },
            "adjust_pnp2_object_in_hand": {
                "description": (
                    "The second optional adjustment object is grasped."
                ),
                "dependencies": ["adjust_pnp1_object_placed"],
                "persistent": False,
            },
            "adjust_pnp2_object_placed": {
                "description": (
                    "The second optional adjustment object is placed."
                ),
                "dependencies": ["adjust_pnp2_object_in_hand"],
                "persistent": False,
            },
        }

    def _tracked_tableware_names(self):
        return [
            "glass1",
            "glass2",
            "glass3",
            "bowl",
        ]

    def _check_task_obj_grasped(
        self,
        obj_name,
    ):
        """
        Local grasp check with a looser gripper-close threshold.
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

    def _get_grasped_tableware(self):
        grasped = []

        for obj_name in self._tracked_tableware_names():
            if self._check_task_obj_grasped(obj_name):
                grasped.append(obj_name)

        return grasped

    def _object_in_target_cabinet(
        self,
        obj_name,
    ):
        return bool(
            OU.obj_inside_of(
                self,
                obj_name,
                self.cab,
            )
        )

    def _get_raw_semantic_milestone_state(self):
        glass3_in_hand = self._check_task_obj_grasped(
            "glass3",
        )

        glass3_in_target_cabinet = bool(
            self._object_in_target_cabinet(
                "glass3",
            )
        )

        remote_mug_placed = bool(
            glass3_in_target_cabinet
            and not glass3_in_hand
        )

        grasped_tableware = self._get_grasped_tableware()

        return {
            "remote_mug_in_hand": glass3_in_hand,
            "remote_mug_in_target_cabinet": remote_mug_placed,
            "grasped_tableware": grasped_tableware,
        }

    def _get_ordered_semantic_milestone_state(self):
        return {
            name: bool(
                self._lh_ordered_ever.get(
                    name,
                    False,
                )
            )
            for name in self.get_semantic_milestone_spec().keys()
        }

    # =====================================================================
    # Dynamic PnP progress helpers
    # =====================================================================

    def _get_active_action_denominator(self):
        """
        Denominator is 2, 4, or 6 depending on how many optional adjustment
        PnPs have actually been observed.
        """
        return int(
            2
            + 2 * int(self._lh_num_observed_adjust_pnps >= 1)
            + 2 * int(self._lh_num_observed_adjust_pnps >= 2)
        )

    def _rebuild_progress_curve_from_counts(self):
        denom = float(
            max(
                self._get_active_action_denominator(),
                1,
            )
        )

        self._lh_progress_curve = [
            float(count) / denom
            for count in self._lh_completed_action_count_curve
        ]

    def _count_completed_ordered_actions(self):
        action_names = [
            "remote_mug_in_hand",
            "remote_mug_in_target_cabinet",
            "adjust_pnp1_object_in_hand",
            "adjust_pnp1_object_placed",
            "adjust_pnp2_object_in_hand",
            "adjust_pnp2_object_placed",
        ]

        return sum(
            int(
                bool(
                    self._lh_ordered_ever.get(
                        name,
                        False,
                    )
                )
            )
            for name in action_names
        )

    def _count_completed_unordered_actions(self):
        action_names = [
            "remote_mug_in_hand",
            "remote_mug_in_target_cabinet",
            "adjust_pnp1_object_in_hand",
            "adjust_pnp1_object_placed",
            "adjust_pnp2_object_in_hand",
            "adjust_pnp2_object_placed",
        ]

        return sum(
            int(
                bool(
                    self._lh_unordered_ever.get(
                        name,
                        False,
                    )
                )
            )
            for name in action_names
        )

    def _get_current_ordered_progress(
        self,
        ordered_state=None,
    ):
        del ordered_state

        denom = float(
            max(
                self._get_active_action_denominator(),
                1,
            )
        )

        return float(
            self._count_completed_ordered_actions()
        ) / denom

    def _get_current_unordered_progress(self):
        denom = float(
            max(
                self._get_active_action_denominator(),
                1,
            )
        )

        return float(
            self._count_completed_unordered_actions()
        ) / denom

    # =====================================================================
    # Per-episode long-horizon metric tracker
    # =====================================================================

    def _reset_long_horizon_metric_tracker(
        self,
        initial_state=None,
    ):
        if initial_state is None:
            initial_state = {}

        names = list(
            self.get_semantic_milestone_spec().keys()
        )

        self._lh_step_count = 0

        self._lh_prev_raw = {
            "remote_mug_in_hand": bool(
                initial_state.get(
                    "remote_mug_in_hand",
                    False,
                )
            ),
            "remote_mug_in_target_cabinet": bool(
                initial_state.get(
                    "remote_mug_in_target_cabinet",
                    False,
                )
            ),
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

        self._lh_current_adjust_obj = None
        self._lh_current_adjust_idx = None
        self._lh_num_observed_adjust_pnps = 0

        self._lh_first_success_step = None
        self._lh_completed_action_count_curve = []
        self._lh_progress_curve = []

    def _reset_internal(self):
        super()._reset_internal()

        initial_raw = (
            self._get_raw_semantic_milestone_state()
        )

        self._reset_long_horizon_metric_tracker(
            initial_state=initial_raw
        )

    def _maybe_start_adjust_pnp(
        self,
        raw,
    ):
        if not self._lh_ordered_ever[
            "remote_mug_in_target_cabinet"
        ]:
            return

        if self._lh_current_adjust_obj is not None:
            return

        if self._lh_num_observed_adjust_pnps >= 2:
            return

        grasped = raw.get(
            "grasped_tableware",
            [],
        )

        # Do not let persistent or flickering glass3 grasp around the first
        # place be counted as an immediate extra PnP in the same phase. Once
        # remote place has completed, future genuine grasps are still allowed.
        grasped = [
            obj_name
            for obj_name in grasped
            if not (
                obj_name == "glass3"
                and self._lh_step_count
                <= self._lh_remote_place_step + 1
            )
        ]

        if not grasped:
            return

        obj_name = grasped[0]
        idx = self._lh_num_observed_adjust_pnps + 1

        self._lh_current_adjust_obj = obj_name
        self._lh_current_adjust_idx = idx
        self._lh_num_observed_adjust_pnps = idx

        pick_name = f"adjust_pnp{idx}_object_in_hand"
        self._lh_ordered_ever[pick_name] = True
        self._lh_unordered_ever[pick_name] = True

        # Optional PnP count expanded, so rebuild old curve using the new
        # denominator. This avoids final visualizations mixing denominators.
        self._rebuild_progress_curve_from_counts()

    def _maybe_finish_adjust_pnp(
        self,
    ):
        if self._lh_current_adjust_obj is None:
            return

        obj_name = self._lh_current_adjust_obj
        idx = self._lh_current_adjust_idx

        still_in_hand = self._check_task_obj_grasped(
            obj_name,
        )

        placed_in_target = bool(
            self._object_in_target_cabinet(
                obj_name,
            )
            and not still_in_hand
        )

        if not placed_in_target:
            return

        place_name = f"adjust_pnp{idx}_object_placed"
        self._lh_ordered_ever[place_name] = True
        self._lh_unordered_ever[place_name] = True

        self._lh_current_adjust_obj = None
        self._lh_current_adjust_idx = None

    def _update_long_horizon_metric_tracker(self):
        """
        Update metrics once per policy timestep.
        """
        self._lh_step_count += 1

        raw = self._get_raw_semantic_milestone_state()

        # --------------------------------------------------------------
        # Mandatory PnP: pick remote mug.
        # --------------------------------------------------------------
        if raw["remote_mug_in_hand"]:
            self._lh_ordered_ever["remote_mug_in_hand"] = True
            self._lh_unordered_ever["remote_mug_in_hand"] = True

        # --------------------------------------------------------------
        # Mandatory PnP: place remote mug in target cabinet.
        #
        # If grasp detection missed the remote pick, infer it from place.
        # --------------------------------------------------------------
        remote_place_rising = bool(
            raw["remote_mug_in_target_cabinet"]
            and not self._lh_prev_raw[
                "remote_mug_in_target_cabinet"
            ]
        )

        if raw["remote_mug_in_target_cabinet"]:
            if not self._lh_ordered_ever["remote_mug_in_hand"]:
                self._lh_ordered_ever["remote_mug_in_hand"] = True
                self._lh_unordered_ever["remote_mug_in_hand"] = True

            self._lh_ordered_ever[
                "remote_mug_in_target_cabinet"
            ] = True
            self._lh_unordered_ever[
                "remote_mug_in_target_cabinet"
            ] = True

            if remote_place_rising:
                self._lh_remote_place_step = self._lh_step_count

        # --------------------------------------------------------------
        # Optional adjustment PnPs.
        # --------------------------------------------------------------
        self._maybe_finish_adjust_pnp()
        self._maybe_start_adjust_pnp(raw)

        completed_count = self._count_completed_ordered_actions()
        self._lh_completed_action_count_curve.append(
            int(completed_count)
        )
        self._rebuild_progress_curve_from_counts()

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
            "remote_mug_in_hand": bool(
                raw["remote_mug_in_hand"]
            ),
            "remote_mug_in_target_cabinet": bool(
                raw["remote_mug_in_target_cabinet"]
            ),
        }

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        self._update_long_horizon_metric_tracker()

        return reward, done, info

    # =====================================================================
    # Public metric API
    # =====================================================================

    def get_long_horizon_metrics(self):
        ops = self._get_current_ordered_progress()
        ups = self._get_current_unordered_progress()

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

        glass1_glass2 = self.dist_between_obj(
            "glass1",
            "glass2",
        )
        glass2_glass3 = self.dist_between_obj(
            "glass2",
            "glass3",
        )
        glass3_glass1 = self.dist_between_obj(
            "glass3",
            "glass1",
        )

        bowl1_glass1 = self.dist_between_obj(
            "bowl",
            "glass1",
        )
        bowl1_glass2 = self.dist_between_obj(
            "bowl",
            "glass2",
        )
        bowl1_glass3 = self.dist_between_obj(
            "bowl",
            "glass3",
        )

        max_glass = max(
            glass1_glass2,
            glass2_glass3,
            glass3_glass1,
        )
        max_bowl = max(
            bowl1_glass1,
            bowl1_glass2,
            bowl1_glass3,
        )

        far_from_obj1 = OU.gripper_obj_far(
            self,
            obj_name="glass1",
        )
        far_from_obj2 = OU.gripper_obj_far(
            self,
            obj_name="glass2",
        )
        far_from_obj3 = OU.gripper_obj_far(
            self,
            obj_name="glass3",
        )

        return (
            far_from_obj1
            and far_from_obj2
            and far_from_obj3
            and max_glass * 1.5 < max_bowl
        )

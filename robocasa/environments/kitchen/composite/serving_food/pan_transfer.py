from robocasa.environments.kitchen.kitchen import *
import numpy as np
import robocasa.utils.object_utils as OU


class PanTransfer(Kitchen):
    """
    Pan Transfer: composite task for Serving Food activity.

    Simulates the task of transferring vegetables from a pan to a plate.

    Steps:
        Pick up the pan and dump the vegetables in it onto the plate.
        Then, return the pan to the stove.

    Progress milestones:
        1. pan_in_hand
        2. vegetable_on_plate
        3. pan_returned_to_stove

    Semantics:
        - The pan starts on the stove, so the initial stove contact must not
          count as pan_returned_to_stove.
        - pan_returned_to_stove is only credited after the pan has genuinely
          left the stove during the episode and vegetable_on_plate has been
          achieved.
        - Direct robot contact with the vegetable remains part of the original
          failure / success logic and is not counted as a semantic milestone.

    Original RoboCasa _check_success() is preserved.
    """

    def __init__(self, *args, **kwargs):
        self._reset_long_horizon_metric_tracker(initial_state=None)
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref(
            "stove",
            dict(id=FixtureType.STOVE),
        )
        self.init_robot_base_ref = self.stove
        self.counter = self.register_fixture_ref(
            "counter",
            dict(
                id=FixtureType.COUNTER,
                ref=self.stove,
                size=[0.30, 0.40],
            ),
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(
                self.novel_instructions
            )
        else:
            ep_meta["lang"] = (
                "Pick up the pan and dump the vegetables in it onto the plate. "
                "Then return the pan to the stove."
            )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self._robot_touched_food = False

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="vegetable",
                obj_groups="vegetable",
                placement=dict(
                    fixture=self.stove,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                    try_to_place_in="pan",
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dstr_dining",
                obj_groups="all",
                exclude_obj_groups=[
                    "plate",
                    "pan",
                    "vegetable",
                ],
                placement=dict(
                    fixture=self.counter,
                    size=(0.30, 0.20),
                    pos=(0.5, 0.5),
                ),
            )
        )

        return cfgs

    def _check_obj_location_on_stove(
        self,
        obj_name,
        threshold=0.08,
    ):
        """
        Check if the object is on the stove and close to a burner.

        Returns the burner location if valid, otherwise None.
        """
        obj = self.objects[obj_name]
        obj_pos = np.array(
            self.sim.data.body_xpos[
                self.obj_body_id[obj.name]
            ]
        )[0:2]

        obj_on_stove = OU.check_obj_fixture_contact(
            self,
            obj_name,
            self.stove,
        )

        if obj_on_stove:
            for location, site in self.stove.burner_sites.items():
                if site is not None:
                    burner_pos = np.array(
                        self.sim.data.get_site_xpos(
                            site.get("name")
                        )
                    )[0:2]

                    dist = np.linalg.norm(
                        burner_pos - obj_pos
                    )

                    if dist < threshold:
                        return location

        return None

    # =====================================================================
    # Task-specific semantic milestones
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "pan_in_hand": {
                "description": "The pan is grasped by the robot gripper.",
                "dependencies": [],
                "persistent": False,
            },
            "vegetable_on_plate": {
                "description": (
                    "The vegetable has been poured from the pan onto the plate."
                ),
                "dependencies": ["pan_in_hand"],
                "persistent": True,
            },
            "pan_returned_to_stove": {
                "description": (
                    "The pan is returned to a valid stove burner after pouring."
                ),
                "dependencies": ["vegetable_on_plate"],
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

    def _get_raw_semantic_milestone_state(self):
        pan_in_hand = self._check_task_obj_grasped(
            "vegetable_container",
        )

        vegetable_on_plate = bool(
            OU.check_obj_in_receptacle(
                self,
                "vegetable",
                "plate",
            )
        )

        pan_on_stove = bool(
            self._check_obj_location_on_stove(
                "vegetable_container"
            )
            is not None
        )

        pan_returned_to_stove = bool(
            pan_on_stove
            and not pan_in_hand
        )

        return {
            "pan_in_hand": pan_in_hand,
            "vegetable_on_plate": vegetable_on_plate,
            "pan_on_stove": pan_on_stove,
            "pan_returned_to_stove": pan_returned_to_stove,
        }

    def _get_ordered_semantic_milestone_state(self):
        """
        Current ordered-valid states.

        pan_in_hand is current/superseded:
            - valid while the pan is currently held;
            - also valid once the pour or final place stage is valid.

        vegetable_on_plate is persistent and must remain physically valid.

        pan_returned_to_stove is valid only after:
            - the pan was picked / left the stove,
            - the vegetable is still on the plate,
            - the pan is currently back on the stove.
        """
        raw = self._get_raw_semantic_milestone_state()

        pour_valid = bool(
            self._lh_ordered_ever["vegetable_on_plate"]
            and raw["vegetable_on_plate"]
        )

        place_valid = bool(
            self._lh_ordered_ever["pan_returned_to_stove"]
            and raw["pan_returned_to_stove"]
            and pour_valid
        )

        pick_valid = bool(
            self._lh_ordered_ever["pan_in_hand"]
            and (
                raw["pan_in_hand"]
                or pour_valid
                or place_valid
            )
        )

        return {
            "pan_in_hand": pick_valid,
            "vegetable_on_plate": pour_valid,
            "pan_returned_to_stove": place_valid,
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
                "pan_in_hand": False,
                "vegetable_on_plate": False,
                "pan_on_stove": False,
                "pan_returned_to_stove": False,
            }

        self._lh_prev_raw = {
            "pan_in_hand": bool(
                initial_state.get(
                    "pan_in_hand",
                    False,
                )
            ),
            "vegetable_on_plate": bool(
                initial_state.get(
                    "vegetable_on_plate",
                    False,
                )
            ),
            "pan_on_stove": bool(
                initial_state.get(
                    "pan_on_stove",
                    False,
                )
            ),
            "pan_returned_to_stove": bool(
                initial_state.get(
                    "pan_returned_to_stove",
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

        # Needed because the pan starts on the stove.
        self._lh_pan_ever_left_stove = False

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
            int(bool(ordered_state["pan_in_hand"]))
            + int(bool(ordered_state["vegetable_on_plate"]))
            + int(bool(ordered_state["pan_returned_to_stove"]))
        ) / 3.0

    def _update_long_horizon_metric_tracker(self):
        """
        Update metrics once per policy timestep.

        M1:
            pan is picked.

        M2:
            vegetable reaches the plate after the pan has been picked.

        M3:
            pan returns to the stove after it has genuinely left the stove and
            after M2 has been achieved.
        """
        self._lh_step_count += 1

        raw = self._get_raw_semantic_milestone_state()

        # --------------------------------------------------------------
        # Track whether the pan has genuinely left the stove.
        # --------------------------------------------------------------
        if (
            raw["pan_in_hand"]
            or not raw["pan_on_stove"]
        ):
            self._lh_pan_ever_left_stove = True

        # --------------------------------------------------------------
        # UPS: order ignored, but pan_returned_to_stove must still be a genuine
        # return event rather than the initial scene state.
        # --------------------------------------------------------------
        if raw["pan_in_hand"]:
            self._lh_unordered_ever["pan_in_hand"] = True

        if raw["vegetable_on_plate"]:
            self._lh_unordered_ever["vegetable_on_plate"] = True

        pan_return_event = bool(
            raw["pan_returned_to_stove"]
            and not self._lh_prev_raw["pan_returned_to_stove"]
            and self._lh_pan_ever_left_stove
        )

        if pan_return_event:
            self._lh_unordered_ever["pan_returned_to_stove"] = True

        # --------------------------------------------------------------
        # OPS M1: pick pan.
        # --------------------------------------------------------------
        if raw["pan_in_hand"]:
            self._lh_ordered_ever["pan_in_hand"] = True

        # --------------------------------------------------------------
        # OPS M2: pour vegetable onto plate.
        #
        # If grasp detection missed the pan pick, infer it from a valid pour.
        # --------------------------------------------------------------
        if raw["vegetable_on_plate"]:
            if not self._lh_ordered_ever["pan_in_hand"]:
                self._lh_ordered_ever["pan_in_hand"] = True
            self._lh_ordered_ever["vegetable_on_plate"] = True

        # --------------------------------------------------------------
        # OPS M3: return pan to stove after pour.
        # --------------------------------------------------------------
        if (
            pan_return_event
            and self._lh_ordered_ever["vegetable_on_plate"]
        ):
            self._lh_ordered_ever["pan_returned_to_stove"] = True

        ordered_state = (
            self._get_ordered_semantic_milestone_state()
        )

        # --------------------------------------------------------------
        # Regression bookkeeping for persistent pour / place states.
        # --------------------------------------------------------------
        for name in [
            "vegetable_on_plate",
            "pan_returned_to_stove",
        ]:
            if not self._lh_ordered_ever[name]:
                continue

            raw_key = (
                "pan_returned_to_stove"
                if name == "pan_returned_to_stove"
                else name
            )

            prev_valid = bool(
                self._lh_prev_raw[raw_key]
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

        # Official SR / NCT.
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
            "pan_in_hand": bool(
                raw["pan_in_hand"]
            ),
            "vegetable_on_plate": bool(
                raw["vegetable_on_plate"]
            ),
            "pan_on_stove": bool(
                raw["pan_on_stove"]
            ),
            "pan_returned_to_stove": bool(
                raw["pan_returned_to_stove"]
            ),
        }

    def _post_action(self, action):
        """
        Preserve the original food-contact tracker, then update metrics.
        """
        ret = super()._post_action(action)

        contact = self.check_contact(
            self.robots[0].gripper["right"],
            self.objects["vegetable"],
        )

        self._robot_touched_food = (
            self._robot_touched_food
            or contact
        )

        # Must happen after updating _robot_touched_food so official SR/NCT
        # use exactly the original task validity.
        self._update_long_horizon_metric_tracker()

        return ret

    # =====================================================================
    # Public metric API
    # =====================================================================

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
    # Original RoboCasa success criterion, preserved
    # =====================================================================

    def _check_success(self):
        vegetable_on_plate = OU.check_obj_in_receptacle(
            self,
            "vegetable",
            "plate",
        )

        pan_on_stove = (
            self._check_obj_location_on_stove(
                "vegetable_container"
            )
            is not None
        )

        gripper_obj_far = OU.gripper_obj_far(
            self,
            "vegetable_container",
        )

        return (
            vegetable_on_plate
            and pan_on_stove
            and gripper_obj_far
            and (
                not self._robot_touched_food
            )
        )

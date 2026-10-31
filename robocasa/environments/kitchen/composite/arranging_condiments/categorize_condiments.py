from robocasa.environments.kitchen.kitchen import *
import numpy as np
import robocasa.utils.object_utils as OU


class CategorizeCondiments(Kitchen):
    """
    Categorizes Condiments: composite task for Arranging Condiments activity.

    Task:
        Pick the condiment bottle from the counter and place it next to the
        matching condiment bottle in the cabinet.
        Pick the shaker from the counter and place it next to the matching
        shaker in the cabinet.

    The two pick-place branches are order-free:
        condiment_bottle_in_hand -> condiment_bottle_categorized
        shaker_in_hand           -> shaker_categorized

    Valid executions include:
        condiment bottle first, then shaker
        shaker first, then condiment bottle

    Original RoboCasa _check_success() is preserved.
    """

    def __init__(self, *args, **kwargs):
        self._reset_long_horizon_metric_tracker(initial_state=None)
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cab = self.register_fixture_ref(
            "cab",
            dict(id=FixtureType.CABINET),
        )

        self.counter = self.register_fixture_ref(
            "counter",
            dict(
                id=FixtureType.COUNTER,
                ref=self.cab,
            ),
        )

        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(
                self.novel_instructions
            )
        else:
            ep_meta["lang"] = (
                "Put the shaker and condiment bottle from the counter next to "
                "their counterparts in the cabinet."
            )

        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cab.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj1",
                obj_groups=[
                    "condiment_bottle",
                    "ketchup",
                    "syrup_bottle",
                ],
                graspable=True,
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab),
                    size=(0.50, 0.2),
                    pos=("ref", -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj2",
                obj_groups="shaker",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab),
                    size=(0.50, 0.2),
                    pos=("ref", -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cab_obj1",
                obj_groups=[
                    "condiment_bottle",
                    "ketchup",
                    "syrup_bottle",
                ],
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.2),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cab_obj2",
                obj_groups="shaker",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.2),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_counter1",
                obj_groups="all",
                exclude_obj_groups=["condiment"],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab),
                    size=(1.0, 0.30),
                    pos=(0.0, 1.0),
                ),
            )
        )

        return cfgs

    def _xy_dist(self, obj_name1, obj_name2):
        obj1_pos = self.sim.data.body_xpos[
            self.obj_body_id[self.objects[obj_name1].name]
        ][:2]

        obj2_pos = self.sim.data.body_xpos[
            self.obj_body_id[self.objects[obj_name2].name]
        ][:2]

        return np.linalg.norm(
            obj1_pos - obj2_pos
        )

    # =====================================================================
    # Task-specific semantic milestones
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "condiment_bottle_in_hand": {
                "description": (
                    "The condiment bottle from the counter is grasped by the "
                    "robot gripper."
                ),
                "dependencies": [],
                "persistent": False,
            },
            "condiment_bottle_categorized": {
                "description": (
                    "The condiment bottle from the counter is placed inside "
                    "the cabinet and next to the matching condiment bottle."
                ),
                "dependencies": ["condiment_bottle_in_hand"],
                "persistent": True,
            },
            "shaker_in_hand": {
                "description": (
                    "The shaker from the counter is grasped by the robot "
                    "gripper."
                ),
                "dependencies": [],
                "persistent": False,
            },
            "shaker_categorized": {
                "description": (
                    "The shaker from the counter is placed inside the cabinet "
                    "and next to the matching shaker."
                ),
                "dependencies": ["shaker_in_hand"],
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
        condiment_bottle_in_hand = self._check_task_obj_grasped(
            "obj1",
        )
        shaker_in_hand = self._check_task_obj_grasped(
            "obj2",
        )

        obj1_in_cab = bool(
            OU.obj_inside_of(
                self,
                "obj1",
                self.cab,
            )
        )

        obj2_in_cab = bool(
            OU.obj_inside_of(
                self,
                "obj2",
                self.cab,
            )
        )

        bottles_close = bool(
            self._xy_dist(
                "obj1",
                "cab_obj1",
            )
            <= 0.15
        )

        shakers_close = bool(
            self._xy_dist(
                "obj2",
                "cab_obj2",
            )
            <= 0.15
        )

        condiment_bottle_categorized = bool(
            obj1_in_cab
            and bottles_close
            and not condiment_bottle_in_hand
        )

        shaker_categorized = bool(
            obj2_in_cab
            and shakers_close
            and not shaker_in_hand
        )

        return {
            "condiment_bottle_in_hand": condiment_bottle_in_hand,
            "condiment_bottle_categorized": condiment_bottle_categorized,
            "shaker_in_hand": shaker_in_hand,
            "shaker_categorized": shaker_categorized,
        }

    def _get_ordered_semantic_milestone_state(self):
        """
        Current ordered-valid states.

        The two branches are independent. Pick milestones are current /
        superseded:
            - valid while the object is currently held;
            - also valid once its matching category placement is valid.

        If an object is picked and dropped in the wrong cabinet location before
        categorization, its pick contribution disappears from current OPS.
        """
        raw = self._get_raw_semantic_milestone_state()

        bottle_place_valid = bool(
            self._lh_ordered_ever["condiment_bottle_categorized"]
            and raw["condiment_bottle_categorized"]
        )
        shaker_place_valid = bool(
            self._lh_ordered_ever["shaker_categorized"]
            and raw["shaker_categorized"]
        )

        bottle_pick_valid = bool(
            self._lh_ordered_ever["condiment_bottle_in_hand"]
            and (
                raw["condiment_bottle_in_hand"]
                or bottle_place_valid
            )
        )
        shaker_pick_valid = bool(
            self._lh_ordered_ever["shaker_in_hand"]
            and (
                raw["shaker_in_hand"]
                or shaker_place_valid
            )
        )

        return {
            "condiment_bottle_in_hand": bottle_pick_valid,
            "condiment_bottle_categorized": bottle_place_valid,
            "shaker_in_hand": shaker_pick_valid,
            "shaker_categorized": shaker_place_valid,
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
        """
        Normal RoboCasa reset plus metric initialization.
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
            int(bool(ordered_state["condiment_bottle_in_hand"]))
            + int(bool(ordered_state["condiment_bottle_categorized"]))
            + int(bool(ordered_state["shaker_in_hand"]))
            + int(bool(ordered_state["shaker_categorized"]))
        ) / 4.0

    def _update_long_horizon_metric_tracker(self):
        """
        Update all long-horizon metrics once per policy timestep.

        Valid ordered executions:
            condiment bottle pick/place, then shaker pick/place
            shaker pick/place, then condiment bottle pick/place
        """
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
        # Condiment bottle branch.
        # --------------------------------------------------------------
        if raw["condiment_bottle_in_hand"]:
            self._lh_ordered_ever["condiment_bottle_in_hand"] = True

        if raw["condiment_bottle_categorized"]:
            # If grasp detection flickered and was missed, infer the pick from
            # a valid categorized placement.
            if not self._lh_ordered_ever["condiment_bottle_in_hand"]:
                self._lh_ordered_ever["condiment_bottle_in_hand"] = True
            self._lh_ordered_ever["condiment_bottle_categorized"] = True

        # --------------------------------------------------------------
        # Shaker branch.
        # --------------------------------------------------------------
        if raw["shaker_in_hand"]:
            self._lh_ordered_ever["shaker_in_hand"] = True

        if raw["shaker_categorized"]:
            if not self._lh_ordered_ever["shaker_in_hand"]:
                self._lh_ordered_ever["shaker_in_hand"] = True
            self._lh_ordered_ever["shaker_categorized"] = True

        ordered_state = (
            self._get_ordered_semantic_milestone_state()
        )

        # --------------------------------------------------------------
        # Regression bookkeeping for persistent categorized placements.
        # --------------------------------------------------------------
        for name in [
            "condiment_bottle_categorized",
            "shaker_categorized",
        ]:
            if not self._lh_ordered_ever[name]:
                continue

            prev_ordered_valid = bool(
                self._lh_ordered_ever[name]
                and self._lh_prev_raw[name]
            )

            falling_edge = bool(
                prev_ordered_valid
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
            name: bool(raw[name])
            for name in spec.keys()
        }

    def _post_action(self, action):
        """
        Run standard RoboCasa post-action logic, then update metrics.
        """
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
        obj1_in_cab = OU.obj_inside_of(
            self,
            "obj1",
            self.cab,
        )

        obj2_in_cab = OU.obj_inside_of(
            self,
            "obj2",
            self.cab,
        )

        bottles_close = (
            self._xy_dist(
                "obj1",
                "cab_obj1",
            )
            <= 0.15
        )

        shakers_close = (
            self._xy_dist(
                "obj2",
                "cab_obj2",
            )
            <= 0.15
        )

        gripper_far = (
            OU.gripper_obj_far(
                self,
                "obj1",
            )
            and OU.gripper_obj_far(
                self,
                "obj2",
            )
        )

        return (
            obj1_in_cab
            and obj2_in_cab
            and bottles_close
            and shakers_close
            and gripper_far
        )

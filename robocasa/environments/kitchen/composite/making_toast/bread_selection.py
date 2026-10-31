from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.object_utils as OU


class BreadSelection(Kitchen):
    """
    Bread Selection: composite task for Making Toast activity.

    Task:
        Pick the croissant and place it on the cutting board.
        Pick the jam jar and place it on the cutting board.

    The two pick-place branches are order-free:
        croissant_in_hand -> croissant_on_cutting_board
        jam_in_hand       -> jam_on_cutting_board

    Valid executions include:
        croissant -> board, then jam -> board
        jam -> board, then croissant -> board

    The cabinet starts open and is not counted as a progress milestone.

    Original RoboCasa _check_success() is preserved.
    """

    EXCLUDE_LAYOUTS = Kitchen.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        self._reset_long_horizon_metric_tracker(initial_state=None)
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cab = self.register_fixture_ref(
            "cab",
            dict(id=FixtureType.CABINET_DOUBLE_DOOR),
        )

        self.counter = self.register_fixture_ref(
            "counter",
            dict(
                id=FixtureType.COUNTER,
                ref=self.cab,
                size=(0.9, 0.6),
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
                "From the different types of pastries on the counter, select a croissant and place it on the cutting board. "
                "Then retrieve a jar of jam from the cabinet and place it alongside the croissant on the cutting board."
            )

        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.cab.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="cutting_board",
                obj_groups="cutting_board",
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(1.0, 0.6),
                    pos=(0.0, -1.0),
                    rotation=[np.pi / 2, np.pi / 2],
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_pastry",
                obj_groups=("baguette", "cupcake"),
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="cutting_board",
                    size=(1.0, 0.6),
                    pos=(0.0, -1.0),
                    try_to_place_in="plate",
                ),
            )
        )

        cfgs.append(
            dict(
                name="croissant",
                obj_groups="croissant",
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="cutting_board",
                    size=(1.0, 0.6),
                    pos=(0.0, -1.0),
                    try_to_place_in="plate",
                ),
            )
        )

        cfgs.append(
            dict(
                name="jam",
                obj_groups="jam",
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(1.0, 0.20),
                    pos=(0, 1.0),
                ),
            )
        )

        return cfgs

    # =====================================================================
    # Task-specific semantic milestones
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "croissant_in_hand": {
                "description": "The croissant is grasped by the robot gripper.",
                "dependencies": [],
                "persistent": False,
            },
            "croissant_on_cutting_board": {
                "description": "The croissant is placed on the cutting board.",
                "dependencies": ["croissant_in_hand"],
                "persistent": True,
            },
            "jam_in_hand": {
                "description": "The jam jar is grasped by the robot gripper.",
                "dependencies": [],
                "persistent": False,
            },
            "jam_on_cutting_board": {
                "description": "The jam jar is placed on the cutting board.",
                "dependencies": ["jam_in_hand"],
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
        croissant_in_hand = self._check_task_obj_grasped(
            "croissant",
        )
        jam_in_hand = self._check_task_obj_grasped(
            "jam",
        )

        croissant_on_board = bool(
            OU.check_obj_in_receptacle(
                self,
                "croissant",
                "cutting_board",
            )
            and not croissant_in_hand
        )

        jam_on_board = bool(
            OU.check_obj_in_receptacle(
                self,
                "jam",
                "cutting_board",
            )
            and not jam_in_hand
        )

        return {
            "croissant_in_hand": croissant_in_hand,
            "croissant_on_cutting_board": croissant_on_board,
            "jam_in_hand": jam_in_hand,
            "jam_on_cutting_board": jam_on_board,
        }

    def _get_ordered_semantic_milestone_state(self):
        """
        Current ordered-valid states.

        The croissant and jam branches are independent. Pick milestones are
        current/superseded:
            - valid while the object is currently held;
            - also valid once its corresponding place stage is valid.

        If an object is picked and dropped outside the cutting board before
        placement, its pick contribution disappears from current OPS.
        """
        raw = self._get_raw_semantic_milestone_state()

        croissant_place_valid = bool(
            self._lh_ordered_ever["croissant_on_cutting_board"]
            and raw["croissant_on_cutting_board"]
        )
        jam_place_valid = bool(
            self._lh_ordered_ever["jam_on_cutting_board"]
            and raw["jam_on_cutting_board"]
        )

        croissant_pick_valid = bool(
            self._lh_ordered_ever["croissant_in_hand"]
            and (
                raw["croissant_in_hand"]
                or croissant_place_valid
            )
        )
        jam_pick_valid = bool(
            self._lh_ordered_ever["jam_in_hand"]
            and (
                raw["jam_in_hand"]
                or jam_place_valid
            )
        )

        return {
            "croissant_in_hand": croissant_pick_valid,
            "croissant_on_cutting_board": croissant_place_valid,
            "jam_in_hand": jam_pick_valid,
            "jam_on_cutting_board": jam_place_valid,
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
            int(bool(ordered_state["croissant_in_hand"]))
            + int(bool(ordered_state["croissant_on_cutting_board"]))
            + int(bool(ordered_state["jam_in_hand"]))
            + int(bool(ordered_state["jam_on_cutting_board"]))
        ) / 4.0

    def _update_long_horizon_metric_tracker(self):
        """
        Update all long-horizon metrics once per policy timestep.

        Valid ordered executions:
            croissant pick/place, then jam pick/place
            jam pick/place, then croissant pick/place
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
        # Croissant branch.
        # --------------------------------------------------------------
        if raw["croissant_in_hand"]:
            self._lh_ordered_ever["croissant_in_hand"] = True

        if raw["croissant_on_cutting_board"]:
            # If grasp detection flickered and was missed, infer the pick from
            # a valid cutting-board placement.
            if not self._lh_ordered_ever["croissant_in_hand"]:
                self._lh_ordered_ever["croissant_in_hand"] = True
            self._lh_ordered_ever["croissant_on_cutting_board"] = True

        # --------------------------------------------------------------
        # Jam branch.
        # --------------------------------------------------------------
        if raw["jam_in_hand"]:
            self._lh_ordered_ever["jam_in_hand"] = True

        if raw["jam_on_cutting_board"]:
            if not self._lh_ordered_ever["jam_in_hand"]:
                self._lh_ordered_ever["jam_in_hand"] = True
            self._lh_ordered_ever["jam_on_cutting_board"] = True

        ordered_state = (
            self._get_ordered_semantic_milestone_state()
        )

        # --------------------------------------------------------------
        # Regression bookkeeping for persistent placement milestones.
        # --------------------------------------------------------------
        for name in [
            "croissant_on_cutting_board",
            "jam_on_cutting_board",
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
        return (
            OU.check_obj_in_receptacle(
                self,
                "croissant",
                "cutting_board",
            )
            and OU.gripper_obj_far(
                self,
                obj_name="croissant",
            )
            and OU.check_obj_in_receptacle(
                self,
                "jam",
                "cutting_board",
            )
        )

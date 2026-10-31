from robocasa.environments.kitchen.kitchen import *


class SetUpCuttingStation(Kitchen):
    """
    Set Up Cutting Station: composite task for Slicing Meat activity.
    Simulates the task of setting up a cutting station.

    Task:
        Put both the knife and the meat on the cutting board.

    Long-horizon semantic milestones:
        M1. knife_in_hand
        M2. knife_on_cutting_board
        M3. meat_in_hand
        M4. meat_on_cutting_board

    Order:
        knife branch and meat branch are independent:
            knife_in_hand -> knife_on_cutting_board
            meat_in_hand  -> meat_on_cutting_board

        The two branches can happen in either order:
            knife first, then meat
            meat first, then knife

    The final position of both knife and meat matters for progress. If either
    object is removed from the board after placement, its placement credit
    disappears from current OPS. Its pick credit is retained only while it is
    currently held or while its corresponding placement remains valid.

    Original RoboCasa _check_success() is preserved.
    """

    def __init__(self, *args, **kwargs):
        self._reset_long_horizon_metric_tracker(initial_state=None)
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER)
        )
        self.drawer = self.register_fixture_ref(
            "drawer", dict(id=FixtureType.TOP_DRAWER, ref=self.counter)
        )
        self.init_robot_base_ref = self.drawer

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta["lang"] = (
                "Pick up the knife from the drawer and place it on the cutting "
                "board. Then place the meat from the plate to the cutting board."
            )

        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        self.drawer.open_door(env=self)
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="receptacle",
                obj_groups="cutting_board",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    size=(1, 0.5),
                    pos=(0.0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="knife",
                obj_groups="knife",
                object_scale=[1.25, 1.25, 2.5],
                placement=dict(
                    fixture=self.drawer,
                    size=(0.5, 0.35),
                    pos=(0, -1.0),
                    offset=(0, 0.10),
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(1, 0.4),
                    pos=(0.0, -1.0),
                    try_to_place_in="plate",
                ),
            )
        )

        return cfgs

    # =====================================================================
    # Task-specific semantic milestone definitions
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "knife_in_hand": {
                "description": "The knife is grasped by the robot gripper.",
                "dependencies": [],
                "persistent": False,
            },
            "knife_on_cutting_board": {
                "description": "The knife is placed on the cutting board.",
                "dependencies": ["knife_in_hand"],
                "persistent": True,
            },
            "meat_in_hand": {
                "description": "The meat is grasped by the robot gripper.",
                "dependencies": [],
                "persistent": False,
            },
            "meat_on_cutting_board": {
                "description": "The meat is placed on the cutting board.",
                "dependencies": ["meat_in_hand"],
                "persistent": True,
            },
        }

    def _check_task_obj_grasped(
        self,
        obj_name,
    ):
        """
        Local grasp check with a looser gripper-close threshold.

        This is intentionally local to this task and does not modify
        robocasa.utils.object_utils globally.
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
        knife_in_hand = self._check_task_obj_grasped("knife")
        meat_in_hand = self._check_task_obj_grasped("meat")

        knife_on_cutting_board = bool(
            OU.check_obj_in_receptacle(
                self,
                "knife",
                "receptacle",
            )
            and not knife_in_hand
        )

        meat_on_cutting_board = bool(
            OU.check_obj_in_receptacle(
                self,
                "meat",
                "receptacle",
            )
            and not meat_in_hand
        )

        return {
            "knife_in_hand": knife_in_hand,
            "knife_on_cutting_board": knife_on_cutting_board,
            "meat_in_hand": meat_in_hand,
            "meat_on_cutting_board": meat_on_cutting_board,
        }

    def _get_ordered_semantic_milestone_state(self):
        """
        Current ordered-valid milestone states.

        Pick milestones are current/superseded:
            - valid while the corresponding object is currently held;
            - also valid once that object is validly placed on the board.

        If an object is picked and dropped outside the board before placement,
        its pick contribution disappears from current OPS.
        """
        raw = self._get_raw_semantic_milestone_state()

        knife_place_valid = bool(
            self._lh_ordered_ever["knife_on_cutting_board"]
            and raw["knife_on_cutting_board"]
        )
        meat_place_valid = bool(
            self._lh_ordered_ever["meat_on_cutting_board"]
            and raw["meat_on_cutting_board"]
        )

        knife_pick_valid = bool(
            self._lh_ordered_ever["knife_in_hand"]
            and (
                raw["knife_in_hand"]
                or knife_place_valid
            )
        )
        meat_pick_valid = bool(
            self._lh_ordered_ever["meat_in_hand"]
            and (
                raw["meat_in_hand"]
                or meat_place_valid
            )
        )

        return {
            "knife_in_hand": knife_pick_valid,
            "knife_on_cutting_board": knife_place_valid,
            "meat_in_hand": meat_pick_valid,
            "meat_on_cutting_board": meat_place_valid,
        }

    # =====================================================================
    # Per-episode long-horizon metric tracker
    # =====================================================================

    def _reset_long_horizon_metric_tracker(self, initial_state=None):
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

        self._lh_ops_auc_sum = 0.0
        self._lh_peak_progress = 0.0
        self._lh_first_success_step = None
        self._lh_progress_curve = []

    def _reset_internal(self):
        """
        Normal RoboCasa reset plus long-horizon metric initialization.
        """
        super()._reset_internal()

        initial_raw = self._get_raw_semantic_milestone_state()
        self._reset_long_horizon_metric_tracker(
            initial_state=initial_raw
        )

    def _get_current_ordered_progress(self, ordered_state=None):
        if ordered_state is None:
            ordered_state = self._get_ordered_semantic_milestone_state()

        return (
            int(bool(ordered_state["knife_in_hand"]))
            + int(bool(ordered_state["knife_on_cutting_board"]))
            + int(bool(ordered_state["meat_in_hand"]))
            + int(bool(ordered_state["meat_on_cutting_board"]))
        ) / 4.0

    def _update_long_horizon_metric_tracker(self):
        """
        Update metrics once per policy timestep.

        Two independent chains are allowed:
            knife_in_hand -> knife_on_cutting_board
            meat_in_hand  -> meat_on_cutting_board

        The chains themselves can happen in either order.
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
        # Knife branch.
        # --------------------------------------------------------------
        if raw["knife_in_hand"]:
            self._lh_ordered_ever["knife_in_hand"] = True

        if raw["knife_on_cutting_board"]:
            # If the knife grasp signal flickered and was missed, infer pick
            # from a valid placed state. The knife starts in the drawer, so a
            # valid board placement implies it was manipulated.
            if not self._lh_ordered_ever["knife_in_hand"]:
                self._lh_ordered_ever["knife_in_hand"] = True
            self._lh_ordered_ever["knife_on_cutting_board"] = True

        # --------------------------------------------------------------
        # Meat branch.
        # --------------------------------------------------------------
        if raw["meat_in_hand"]:
            self._lh_ordered_ever["meat_in_hand"] = True

        if raw["meat_on_cutting_board"]:
            # Same anti-flicker inference for meat.
            if not self._lh_ordered_ever["meat_in_hand"]:
                self._lh_ordered_ever["meat_in_hand"] = True
            self._lh_ordered_ever["meat_on_cutting_board"] = True

        ordered_state = self._get_ordered_semantic_milestone_state()

        # --------------------------------------------------------------
        # Regression bookkeeping for persistent placement milestones.
        # Not exported directly here, but kept for debugging if needed.
        # --------------------------------------------------------------
        for name, cfg in spec.items():
            if not cfg.get("persistent", True):
                continue
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

        current_ops = self._get_current_ordered_progress(
            ordered_state
        )
        self._lh_progress_curve.append(float(current_ops))

        self._lh_ops_auc_sum += current_ops
        self._lh_peak_progress = max(
            self._lh_peak_progress,
            current_ops,
        )

        # Official SR / NCT.
        success = bool(self._check_success())
        if success and self._lh_first_success_step is None:
            self._lh_first_success_step = self._lh_step_count

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
        ordered_state = self._get_ordered_semantic_milestone_state()
        ops = self._get_current_ordered_progress(
            ordered_state
        )

        spec = self.get_semantic_milestone_spec()
        num_milestones = len(spec)
        ups = (
            sum(self._lh_unordered_ever.values()) / float(num_milestones)
            if num_milestones > 0
            else 0.0
        )

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

        meat_on_board = OU.check_obj_in_receptacle(
            self,
            "meat",
            "receptacle",
        )

        knife_on_board = OU.check_obj_in_receptacle(
            self,
            "knife",
            "receptacle",
        )

        gripper_far = OU.gripper_obj_far(
            self,
            "knife",
        ) and OU.gripper_obj_far(
            self,
            "receptacle",
        )

        return meat_on_board and knife_on_board and gripper_far

from robocasa.environments.kitchen.kitchen import *
from robocasa.environments.kitchen.atomic.kitchen_drawer import *
import robocasa.utils.object_utils as OU


class CuttingToolSelection(ManipulateDrawer):
    """
    Tool Selection: composite task for the chopping vegetables activity.

    Task:
        Pick the correct cutting tool for the food item and place it on the
        cutting board.

    Progress milestones:
        1. correct_tool_in_hand
        2. correct_tool_on_cutting_board

    The correct tool is determined by _CUTTING_MAP. The incorrect tool is not
    a milestone, but it must not be on the cutting board for the place milestone
    to be valid.

    Original RoboCasa _check_success() is preserved.
    """

    _CUTTING_MAP = {
        "garlic": "knife",
        "eggplant": "peeler",
        "broccoli": "knife",
        "carrot": "peeler",
        "cucumber": "peeler",
        "potato": "peeler",
        "avocado": "knife",
    }

    def __init__(self, *args, **kwargs):
        self._reset_long_horizon_metric_tracker(initial_state=None)
        super().__init__(behavior="close", *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.counter = self.register_fixture_ref(
            "counter",
            dict(
                id=FixtureType.COUNTER,
                ref=self.drawer,
                size=(0.5, 0.5),
            ),
        )

        if "refs" in self._ep_meta:
            self.food = self._ep_meta["refs"]["food"]
        else:
            self.food = self.rng.choice(
                list(self._CUTTING_MAP.keys())
            )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        cat = self.get_obj_lang("food")

        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(
                self.novel_instructions
            ).format(cat=cat)
        else:
            ep_meta["lang"] = (
                f"Place the appropriate cutting tool for cutting the "
                f"{cat} skin on the cutting board."
            )

        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["food"] = self.food

        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="peeler",
                obj_groups="peeler",
                placement=dict(
                    fixture=self.drawer,
                    pos=(None, -0.5),
                    size=(0.3, 0.3),
                ),
                object_scale=[1, 1, 1.75],
            )
        )

        cfgs.append(
            dict(
                name="knife",
                obj_groups="knife",
                placement=dict(
                    fixture=self.drawer,
                    pos=(None, -0.5),
                    size=(0.3, 0.3),
                ),
                object_scale=[1, 1, 1.75],
            )
        )

        cfgs.append(
            dict(
                name="distr",
                exclude_obj_groups=["food"],
                placement=dict(
                    fixture=self.counter,
                    size=(0.6, 0.2),
                    sample_region_kwargs=dict(
                        ref=self.drawer,
                    ),
                    pos=(0, 1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="food",
                obj_groups=self.food,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.drawer,
                        top_size=(0.5, 0.5),
                    ),
                    size=(0.5, 0.5),
                    pos=("ref", -1.0),
                    try_to_place_in="cutting_board",
                ),
            )
        )

        return cfgs

    # =====================================================================
    # Task-specific semantic milestones
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "correct_tool_in_hand": {
                "description": (
                    "The correct cutting tool for the current food item is "
                    "grasped by the robot gripper."
                ),
                "dependencies": [],
                "persistent": False,
            },
            "correct_tool_on_cutting_board": {
                "description": (
                    "The correct cutting tool is placed on the cutting board, "
                    "while the incorrect tool is not on the cutting board."
                ),
                "dependencies": ["correct_tool_in_hand"],
                "persistent": True,
            },
        }

    def _get_correct_tool_name(self):
        return self._CUTTING_MAP[self.food]

    def _get_incorrect_tool_name(self):
        correct_tool = self._get_correct_tool_name()
        return "peeler" if correct_tool == "knife" else "knife"

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

    def _is_correct_tool_selected(self):
        """
        Same tool-selection semantics as the original _check_success(), excluding
        only the terminal gripper-distance check.
        """
        peeler_on_cutting_board = bool(
            OU.check_obj_in_receptacle(
                self,
                "peeler",
                "food_container",
            )
        )

        knife_on_cutting_board = bool(
            OU.check_obj_in_receptacle(
                self,
                "knife",
                "food_container",
            )
        )

        correct_tool = self._get_correct_tool_name()

        if correct_tool == "knife":
            return bool(
                knife_on_cutting_board
                and not peeler_on_cutting_board
            )

        return bool(
            peeler_on_cutting_board
            and not knife_on_cutting_board
        )

    def _get_raw_semantic_milestone_state(self):
        correct_tool = self._get_correct_tool_name()

        correct_tool_in_hand = self._check_task_obj_grasped(
            correct_tool,
        )

        correct_tool_on_cutting_board = bool(
            self._is_correct_tool_selected()
            and not correct_tool_in_hand
        )

        return {
            "correct_tool_in_hand": correct_tool_in_hand,
            "correct_tool_on_cutting_board": correct_tool_on_cutting_board,
        }

    def _get_ordered_semantic_milestone_state(self):
        """
        Current ordered-valid states.

        Pick is current/superseded:
            - valid while the correct tool is currently held;
            - also valid once the correct place stage is valid.

        If the correct tool is picked and dropped somewhere invalid, current OPS
        goes back down.
        """
        raw = self._get_raw_semantic_milestone_state()

        place_valid = bool(
            self._lh_ordered_ever["correct_tool_on_cutting_board"]
            and raw["correct_tool_on_cutting_board"]
        )

        pick_valid = bool(
            self._lh_ordered_ever["correct_tool_in_hand"]
            and (
                raw["correct_tool_in_hand"]
                or place_valid
            )
        )

        return {
            "correct_tool_in_hand": pick_valid,
            "correct_tool_on_cutting_board": place_valid,
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
            int(bool(ordered_state["correct_tool_in_hand"]))
            + int(bool(ordered_state["correct_tool_on_cutting_board"]))
        ) / 2.0

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
        # M1: pick correct tool.
        # --------------------------------------------------------------
        if raw["correct_tool_in_hand"]:
            self._lh_ordered_ever["correct_tool_in_hand"] = True

        # --------------------------------------------------------------
        # M2: place correct tool on cutting board.
        #
        # If grasp detection flickered and missed the pick, infer it from a
        # valid correct-tool placement.
        # --------------------------------------------------------------
        if raw["correct_tool_on_cutting_board"]:
            if not self._lh_ordered_ever["correct_tool_in_hand"]:
                self._lh_ordered_ever["correct_tool_in_hand"] = True
            self._lh_ordered_ever["correct_tool_on_cutting_board"] = True

        ordered_state = (
            self._get_ordered_semantic_milestone_state()
        )

        # --------------------------------------------------------------
        # Regression bookkeeping for persistent placement.
        # --------------------------------------------------------------
        if self._lh_ordered_ever["correct_tool_on_cutting_board"]:
            prev_ordered_valid = bool(
                self._lh_ordered_ever["correct_tool_on_cutting_board"]
                and self._lh_prev_raw["correct_tool_on_cutting_board"]
            )

            falling_edge = bool(
                prev_ordered_valid
                and not ordered_state["correct_tool_on_cutting_board"]
            )

            if falling_edge:
                self._lh_regressed["correct_tool_on_cutting_board"] = True

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
        peeler_on_cutting_board = OU.check_obj_in_receptacle(
            self,
            "peeler",
            "food_container",
        )

        knife_on_cutting_board = OU.check_obj_in_receptacle(
            self,
            "knife",
            "food_container",
        )

        correct_tool = self._CUTTING_MAP[
            self.food
        ]

        correct_tool_chosen = (
            knife_on_cutting_board
            and not peeler_on_cutting_board
            if correct_tool == "knife"
            else (
                not knife_on_cutting_board
                and peeler_on_cutting_board
            )
        )

        gripper_obj_far = OU.gripper_obj_far(
            self,
            "food_container",
        )

        return (
            gripper_obj_far
            and correct_tool_chosen
        )

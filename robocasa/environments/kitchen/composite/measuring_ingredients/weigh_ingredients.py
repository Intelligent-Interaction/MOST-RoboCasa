from robocasa.environments.kitchen.kitchen import *
from scipy.spatial.transform import Rotation as R
import robocasa.utils.object_utils as OU


class WeighIngredients(Kitchen):
    """
    Weigh Ingredients.

    Task:
        Pick the packaged food from the cabinet, place it upright on the
        digital scale for weighing, and close the cabinet.

    Long-horizon progress:
        1. ingredient_in_hand
        2. ingredient_on_scale
        3. cabinet_closed

    Semantics:
        - This is one PnP followed by closing the cabinet.
        - The target object is "obj"; obj_distractor is ignored.
        - ingredient_on_scale includes uprightness, because upright placement is
          part of the original valid weighing condition.
        - cabinet_closed is credited only after the valid placement stage.

    Original RoboCasa _check_success() is preserved.
    """

    def __init__(self, *args, **kwargs):
        self._reset_long_horizon_metric_tracker(
            initial_state=None
        )
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cab = self.register_fixture_ref(
            "cab",
            dict(id=FixtureType.CABINET_WITH_DOOR),
        )

        self.counter = self.register_fixture_ref(
            "counter",
            dict(
                id=FixtureType.COUNTER,
                ref=self.cab,
            ),
        )

        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        obj_lang = self.get_obj_lang("obj")

        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(
                self.novel_instructions
            ).format(
                obj_lang=obj_lang
            )
        else:
            ep_meta["lang"] = (
                f"Pick the {obj_lang} and place it on the digital scale "
                "for weighing, and close the cabinet."
            )

        return ep_meta

    def _setup_scene(self):
        self.cab.open_door(
            env=self
        )
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=("packaged_food"),
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.4, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj_distractor",
                obj_groups=("packaged_food"),
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.4, 0.20),
                    pos=(0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="digital_scale",
                obj_groups="digital_scale",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.3, 0.3),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    # =====================================================================
    # Task-specific semantic milestone definitions
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "ingredient_in_hand": {
                "description": (
                    "The target packaged food / ingredient is grasped."
                ),
                "dependencies": [],
                "persistent": False,
            },
            "ingredient_on_scale": {
                "description": (
                    "The target packaged food is upright on the digital scale."
                ),
                "dependencies": ["ingredient_in_hand"],
                "persistent": True,
            },
            "cabinet_closed": {
                "description": (
                    "The cabinet is closed after the ingredient is placed."
                ),
                "dependencies": ["ingredient_on_scale"],
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

    def _ingredient_on_scale(self):
        packaged_food_on_digital_scale = bool(
            OU.check_obj_in_receptacle(
                self,
                "obj",
                "digital_scale",
            )
        )

        obj_upright = bool(
            OU.check_obj_upright(
                self,
                "obj",
            )
        )

        obj_in_hand = self._check_task_obj_grasped(
            "obj",
        )

        return bool(
            packaged_food_on_digital_scale
            and obj_upright
            and not obj_in_hand
        )

    def _cabinet_closed(self):
        return bool(
            self.cab.is_closed(
                env=self
            )
        )

    def _get_raw_semantic_milestone_state(self):
        ingredient_in_hand = self._check_task_obj_grasped(
            "obj",
        )

        ingredient_on_scale = self._ingredient_on_scale()

        cabinet_closed = self._cabinet_closed()

        return {
            "ingredient_in_hand": ingredient_in_hand,
            "ingredient_on_scale": ingredient_on_scale,
            "cabinet_closed": cabinet_closed,
        }

    def _get_ordered_semantic_milestone_state(self):
        """
        Current ordered-valid states.

        ingredient_in_hand is current/superseded:
            - valid while the target object is currently held;
            - also valid once the object is validly placed on the scale.

        ingredient_on_scale and cabinet_closed must remain physically valid.
        """
        raw = (
            self._get_raw_semantic_milestone_state()
        )

        place_valid = bool(
            self._lh_ordered_ever["ingredient_on_scale"]
            and raw["ingredient_on_scale"]
        )

        pick_valid = bool(
            self._lh_ordered_ever["ingredient_in_hand"]
            and (
                raw["ingredient_in_hand"]
                or place_valid
            )
        )

        close_valid = bool(
            self._lh_ordered_ever["cabinet_closed"]
            and raw["cabinet_closed"]
            and place_valid
        )

        return {
            "ingredient_in_hand": pick_valid,
            "ingredient_on_scale": place_valid,
            "cabinet_closed": close_valid,
        }

    # =====================================================================
    # Per-episode long-horizon metric tracker
    # =====================================================================

    def _reset_long_horizon_metric_tracker(
        self,
        initial_state=None,
    ):
        spec = (
            self.get_semantic_milestone_spec()
        )

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
            int(bool(ordered_state["ingredient_in_hand"]))
            + int(bool(ordered_state["ingredient_on_scale"]))
            + int(bool(ordered_state["cabinet_closed"]))
        ) / 3.0

    def _update_long_horizon_metric_tracker(self):
        """
        Update metrics once per policy timestep.

        Chain:
            pick ingredient
                -> place ingredient on digital scale
                -> close cabinet
        """
        self._lh_step_count += 1

        spec = (
            self.get_semantic_milestone_spec()
        )

        raw = (
            self._get_raw_semantic_milestone_state()
        )

        # --------------------------------------------------------------
        # UPS: order ignored.
        # --------------------------------------------------------------
        for name in spec.keys():
            if raw[name]:
                self._lh_unordered_ever[
                    name
                ] = True

        # --------------------------------------------------------------
        # M1: pick ingredient.
        # --------------------------------------------------------------
        if raw["ingredient_in_hand"]:
            self._lh_ordered_ever["ingredient_in_hand"] = True

        # --------------------------------------------------------------
        # M2: place ingredient upright on scale.
        #
        # If grasp detection missed the pick, infer it from valid placement.
        # --------------------------------------------------------------
        if raw["ingredient_on_scale"]:
            if not self._lh_ordered_ever["ingredient_in_hand"]:
                self._lh_ordered_ever["ingredient_in_hand"] = True
            self._lh_ordered_ever["ingredient_on_scale"] = True

        # --------------------------------------------------------------
        # M3: close cabinet after valid placement.
        # --------------------------------------------------------------
        cabinet_close_event = bool(
            raw["cabinet_closed"]
            and not self._lh_prev_raw[
                "cabinet_closed"
            ]
        )

        if (
            cabinet_close_event
            and self._lh_ordered_ever["ingredient_on_scale"]
            and raw["ingredient_on_scale"]
        ):
            self._lh_ordered_ever["cabinet_closed"] = True

        ordered_state = (
            self._get_ordered_semantic_milestone_state()
        )

        # --------------------------------------------------------------
        # Regression bookkeeping for persistent placement / close states.
        # --------------------------------------------------------------
        for name in [
            "ingredient_on_scale",
            "cabinet_closed",
        ]:
            if not self._lh_ordered_ever[
                name
            ]:
                continue

            prev_valid = bool(
                self._lh_prev_raw[name]
            )

            falling_edge = bool(
                prev_valid
                and not ordered_state[name]
            )

            if falling_edge:
                self._lh_regressed[
                    name
                ] = True

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
            and self._lh_first_success_step
            is None
        ):
            self._lh_first_success_step = (
                self._lh_step_count
            )

        self._lh_prev_raw = {
            name: bool(
                raw[name]
            )
            for name in spec.keys()
        }

    def _post_action(self, action):
        reward, done, info = (
            super()._post_action(action)
        )

        self._update_long_horizon_metric_tracker()

        return reward, done, info

    # =====================================================================
    # Public metric API
    # =====================================================================

    def get_long_horizon_metrics(self):
        spec = (
            self.get_semantic_milestone_spec()
        )

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
        gripper_obj_far = (
            OU.gripper_obj_far(self)
        )

        packaged_food_on_digital_scale = (
            OU.check_obj_in_receptacle(
                self,
                "obj",
                "digital_scale",
            )
        )

        obj_upright = (
            OU.check_obj_upright(
                self,
                "obj",
            )
        )

        cabinet_closed = (
            self.cab.is_closed(
                env=self
            )
        )

        return (
            gripper_obj_far
            and packaged_food_on_digital_scale
            and cabinet_closed
            and obj_upright
        )

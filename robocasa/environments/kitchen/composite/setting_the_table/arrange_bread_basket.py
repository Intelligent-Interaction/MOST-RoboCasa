from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.object_utils as OU


class ArrangeBreadBasket(Kitchen):
    """
    Arrange Bread Basket: composite task for Setting The Table activity.

    Task:
        0. Open the cabinet, if this layout actually has a usable cabinet door.
           Some layouts / fixtures do not expose a meaningful cabinet-door
           stage; in those cases the policy can directly pick the bread.
        1. Pick the bread.
        2. Place the bread in the basket.
        3. Pick the loaded basket.
        4. Place the loaded basket on the dining counter.

    Active long-horizon milestones:
        open_cabinet          optional
        bread_in_hand
        bread_in_basket
        basket_in_hand
        basket_on_dining_counter

    Important:
        If bread is picked / placed before an open-cabinet event is observed,
        the metric dynamically disables the optional open_cabinet milestone for
        this episode. This prevents no-door cabinet layouts from staying at 0
        forever.

    Original RoboCasa _check_success() is preserved.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, cab_id=FixtureType.CABINET, *args, **kwargs):
        self.cab_id = cab_id
        self._lh_use_open_cabinet_milestone = True
        self._reset_long_horizon_metric_tracker(initial_state=None)
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cab = self.register_fixture_ref(
            "cab",
            dict(id=self.cab_id),
        )

        self.counter_small = self.register_fixture_ref(
            "counter",
            dict(
                id=FixtureType.COUNTER,
                ref=self.cab,
            ),
        )

        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(
                id=FixtureType.COUNTER,
                ref=FixtureType.STOOL,
                size=(0.75, 0.2),
            ),
        )

        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        bread_name = self.get_obj_lang("bread")

        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(
                self.novel_instructions
            ).format(
                bread_name=bread_name
            )
        else:
            ep_meta["lang"] = (
                f"Open the cabinet, pick up the {bread_name} from the cabinet and place it in the basket. "
                "Then move the basket to the dining counter."
            )

        return ep_meta

    def _setup_scene(self):
        """
        Original scene setup, with a defensive cabinet close.

        Some cabinet-like fixtures do not have a meaningful door. In those
        layouts, close_door / is_closed can be unavailable or ineffective.
        That only affects the optional open_cabinet milestone, not success.
        """
        super()._setup_scene()

        try:
            self.cab.close_door(env=self)
        except Exception:
            pass

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="bread",
                obj_groups="bread",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.40),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="basket",
                obj_groups="basket",
                placement=dict(
                    fixture=self.counter_small,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.50, 0.50),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dstr_dining",
                obj_groups="all",
                placement=dict(
                    fixture=self.dining_table,
                    size=(1, 0.30),
                    pos=(0, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dstr_dining2",
                obj_groups="all",
                placement=dict(
                    fixture=self.dining_table,
                    size=(1, 0.30),
                    pos=(0, 0),
                ),
            )
        )

        return cfgs

    # =====================================================================
    # Task-specific semantic milestones
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "open_cabinet": {
                "description": (
                    "The initially closed cabinet has been opened. Optional: "
                    "disabled if the layout has no reliable cabinet-door stage."
                ),
                "dependencies": [],
                "persistent": False,
            },
            "bread_in_hand": {
                "description": "The bread is grasped by the robot gripper.",
                "dependencies": ["open_cabinet"],
                "persistent": False,
            },
            "bread_in_basket": {
                "description": "The bread is placed inside the basket.",
                "dependencies": ["bread_in_hand"],
                "persistent": True,
            },
            "basket_in_hand": {
                "description": "The loaded basket is grasped by the robot gripper.",
                "dependencies": ["bread_in_basket"],
                "persistent": False,
            },
            "basket_on_dining_counter": {
                "description": "The loaded basket is placed on the dining counter.",
                "dependencies": ["basket_in_hand"],
                "persistent": True,
            },
        }

    def _get_active_milestone_names(self):
        names = list(
            self.get_semantic_milestone_spec().keys()
        )

        if not getattr(
            self,
            "_lh_use_open_cabinet_milestone",
            True,
        ):
            names = [
                name
                for name in names
                if name != "open_cabinet"
            ]

        return names

    def _cabinet_is_open(self):
        """
        Best-effort cabinet-open check for optional M0.

        Return None if this fixture does not support a reliable closed state.
        """
        try:
            return bool(
                not self.cab.is_closed(env=self)
            )
        except Exception:
            return None

    def _detect_open_cabinet_milestone_applicable(self):
        """
        Initial guess for whether open_cabinet should be active.

        This is intentionally conservative. Even if this returns True, the
        tracker can still disable M0 dynamically if bread is picked / placed
        before any open event is observed.
        """
        cabinet_open = self._cabinet_is_open()

        if cabinet_open is None:
            return False

        # If reset state is already open, there is no useful open milestone.
        return bool(
            not cabinet_open
        )

    def _disable_open_cabinet_if_not_applicable(self):
        """
        Dynamically skip open_cabinet for no-door / unreliable-door layouts.

        If the bread has already been picked or placed while open_cabinet has
        not received credit, then this episode clearly does not require a real
        open-cabinet action before bread pickup. Drop M0 from the active metric
        denominator so progress does not stay stuck at zero.
        """
        if not getattr(
            self,
            "_lh_use_open_cabinet_milestone",
            True,
        ):
            return

        if self._lh_ordered_ever.get(
            "open_cabinet",
            False,
        ):
            return

        raw = self._get_raw_semantic_milestone_state()

        if (
            raw["bread_in_hand"]
            or raw["bread_in_basket"]
        ):
            self._lh_use_open_cabinet_milestone = False
            self._lh_unordered_ever["open_cabinet"] = False
            self._lh_ordered_ever["open_cabinet"] = False

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

    def _check_obj_in_receptacle_xy(
        self,
        obj_name,
        receptacle_name,
        th=None,
    ):
        """
        XY containment fallback used only after bread has already been validly
        placed in the basket.
        """
        obj_pos = np.array(
            self.sim.data.body_xpos[
                self.obj_body_id[obj_name]
            ]
        )
        receptacle_pos = np.array(
            self.sim.data.body_xpos[
                self.obj_body_id[receptacle_name]
            ]
        )

        if th is None:
            receptacle = self.objects[receptacle_name]
            th = getattr(
                receptacle,
                "horizontal_radius",
                0.10,
            ) * 0.85

        return bool(
            np.linalg.norm(
                obj_pos[:2] - receptacle_pos[:2]
            )
            < th
        )

    def _get_raw_scene_state(self):
        bread_in_hand = self._check_task_obj_grasped(
            "bread",
        )
        basket_in_hand = self._check_task_obj_grasped(
            "basket",
        )

        bread_in_basket_contact = bool(
            OU.check_obj_in_receptacle(
                self,
                "bread",
                "basket",
            )
        )

        bread_in_basket_released = bool(
            bread_in_basket_contact
            and not bread_in_hand
        )

        bread_already_placed = bool(
            getattr(
                self,
                "_lh_ordered_ever",
                {},
            ).get(
                "bread_in_basket",
                False,
            )
        )

        bread_in_basket_xy = bool(
            self._check_obj_in_receptacle_xy(
                "bread",
                "basket",
            )
        )

        bread_in_basket_for_progress = bool(
            bread_in_basket_released
            or (
                bread_already_placed
                and (
                    bread_in_basket_contact
                    or bread_in_basket_xy
                )
            )
        )

        basket_on_dining_counter = bool(
            OU.check_obj_fixture_contact(
                self,
                "basket",
                self.dining_table,
            )
            and OU.gripper_obj_far(
                self,
                "basket",
            )
        )

        cabinet_open_state = self._cabinet_is_open()
        open_cabinet = bool(
            cabinet_open_state
            if cabinet_open_state is not None
            else False
        )

        return {
            "open_cabinet": open_cabinet,
            "bread_in_hand": bread_in_hand,
            "basket_in_hand": basket_in_hand,
            "bread_in_basket_contact": bread_in_basket_contact,
            "bread_in_basket_xy": bread_in_basket_xy,
            "bread_in_basket_for_progress": bread_in_basket_for_progress,
            "basket_on_dining_counter": basket_on_dining_counter,
        }

    def _get_raw_semantic_milestone_state(self):
        scene = self._get_raw_scene_state()

        return {
            "open_cabinet": bool(
                scene["open_cabinet"]
            ),
            "bread_in_hand": bool(
                scene["bread_in_hand"]
            ),
            "bread_in_basket": bool(
                scene["bread_in_basket_for_progress"]
            ),
            "basket_in_hand": bool(
                scene["basket_in_hand"]
            ),
            "basket_on_dining_counter": bool(
                scene["basket_on_dining_counter"]
            ),
        }

    def _get_ordered_semantic_milestone_state(self):
        """
        Return CURRENT ordered-valid semantic states.
        """
        raw = self._get_raw_semantic_milestone_state()
        scene = self._get_raw_scene_state()

        open_active = bool(
            getattr(
                self,
                "_lh_use_open_cabinet_milestone",
                True,
            )
        )

        open_valid = bool(
            self._lh_ordered_ever["open_cabinet"]
        ) if open_active else False

        bread_place_valid = bool(
            self._lh_ordered_ever["bread_in_basket"]
            and scene["bread_in_basket_for_progress"]
        )

        bread_pick_valid = bool(
            self._lh_ordered_ever["bread_in_hand"]
            and (
                raw["bread_in_hand"]
                or bread_place_valid
            )
        )

        basket_place_valid = bool(
            self._lh_ordered_ever["basket_on_dining_counter"]
            and scene["basket_on_dining_counter"]
            and bread_place_valid
        )

        basket_pick_valid = bool(
            self._lh_ordered_ever["basket_in_hand"]
            and bread_place_valid
            and (
                raw["basket_in_hand"]
                or basket_place_valid
            )
        )

        return {
            "open_cabinet": open_valid,
            "bread_in_hand": bread_pick_valid,
            "bread_in_basket": bread_place_valid,
            "basket_in_hand": basket_pick_valid,
            "basket_on_dining_counter": basket_place_valid,
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
        Normal RoboCasa reset plus metric initialization from the actual reset
        simulator state.
        """
        super()._reset_internal()

        self._lh_use_open_cabinet_milestone = (
            self._detect_open_cabinet_milestone_applicable()
        )

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

        names = self._get_active_milestone_names()

        if not names:
            return 0.0

        completed = sum(
            int(bool(ordered_state[name]))
            for name in names
        )

        return (
            float(completed)
            / float(len(names))
        )

    def _update_long_horizon_metric_tracker(self):
        """
        Update long-horizon metrics once per policy timestep.

        Active ordering:
            optional open cabinet
            -> pick bread
            -> place bread in basket
            -> pick loaded basket
            -> place loaded basket on dining counter
        """
        self._lh_step_count += 1

        raw = self._get_raw_semantic_milestone_state()
        scene = self._get_raw_scene_state()

        # --------------------------------------------------------------
        # Optional M0: open cabinet.
        # --------------------------------------------------------------
        if self._lh_use_open_cabinet_milestone:
            open_rising_edge = bool(
                raw["open_cabinet"]
                and not self._lh_prev_raw["open_cabinet"]
            )

            if open_rising_edge:
                self._lh_ordered_ever["open_cabinet"] = True

        # If bread pickup happens before an open event, treat M0 as not
        # applicable for this episode. This is the key no-door fix.
        self._disable_open_cabinet_if_not_applicable()

        active_names = self._get_active_milestone_names()

        # --------------------------------------------------------------
        # UPS: order ignored, only active milestones count.
        # --------------------------------------------------------------
        for name in active_names:
            if raw[name]:
                self._lh_unordered_ever[name] = True

        open_ready = bool(
            (not self._lh_use_open_cabinet_milestone)
            or self._lh_ordered_ever["open_cabinet"]
        )

        # --------------------------------------------------------------
        # M1: pick bread after cabinet is open, or immediately if M0 skipped.
        # --------------------------------------------------------------
        if (
            raw["bread_in_hand"]
            and open_ready
        ):
            self._lh_ordered_ever["bread_in_hand"] = True

        # --------------------------------------------------------------
        # M2: place bread in basket.
        # --------------------------------------------------------------
        if (
            raw["bread_in_basket"]
            and open_ready
        ):
            if not self._lh_ordered_ever["bread_in_hand"]:
                self._lh_ordered_ever["bread_in_hand"] = True
            self._lh_ordered_ever["bread_in_basket"] = True

        bread_ready = bool(
            scene["bread_in_basket_for_progress"]
            and self._lh_ordered_ever["bread_in_basket"]
        )

        # --------------------------------------------------------------
        # M3: pick loaded basket.
        # --------------------------------------------------------------
        if (
            raw["basket_in_hand"]
            and bread_ready
        ):
            self._lh_ordered_ever["basket_in_hand"] = True

        # --------------------------------------------------------------
        # M4: place loaded basket on dining counter.
        # --------------------------------------------------------------
        if (
            raw["basket_on_dining_counter"]
            and bread_ready
        ):
            if not self._lh_ordered_ever["basket_in_hand"]:
                self._lh_ordered_ever["basket_in_hand"] = True
            self._lh_ordered_ever["basket_on_dining_counter"] = True

        ordered_state = (
            self._get_ordered_semantic_milestone_state()
        )

        for name in [
            "bread_in_basket",
            "basket_on_dining_counter",
        ]:
            if name not in active_names:
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
            for name in self.get_semantic_milestone_spec().keys()
        }

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        self._update_long_horizon_metric_tracker()
        return reward, done, info

    # =====================================================================
    # Public metric API
    # =====================================================================

    def get_long_horizon_metrics(self):
        active_names = self._get_active_milestone_names()

        if active_names:
            ups = (
                sum(
                    int(bool(self._lh_unordered_ever[name]))
                    for name in active_names
                )
                / float(len(active_names))
            )
        else:
            ups = 0.0

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
        gripper_obj_far = OU.gripper_obj_far(
            self,
            obj_name="basket",
        )

        bread_in_basket = OU.check_obj_in_receptacle(
            self,
            "bread",
            "basket",
        )

        basket_on_counter = OU.check_obj_fixture_contact(
            self,
            "basket",
            self.dining_table,
        )

        return (
            gripper_obj_far
            and bread_in_basket
            and basket_on_counter
        )

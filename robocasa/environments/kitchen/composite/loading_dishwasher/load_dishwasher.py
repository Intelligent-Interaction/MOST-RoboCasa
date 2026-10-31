from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.object_utils as OU


class LoadDishwasher(Kitchen):
    """
    LoadDishwasher: composite task for Washing Dishes activity.

    Stimulates loading dishes from the counter into the dishwasher.

    Steps:
        Take the dishes from the counter and place them on the top rack of the dishwasher.
        Close the dishwasher fully.

    Long-horizon semantic milestones:
        M1. load_dish0
            dish0 is correctly placed on the dishwasher rack.

        M2. load_dish1
            dish1 is correctly placed on the dishwasher rack.

        M3. close_dishwasher
            the dishwasher door is fully closed.

    Dependency graph:
        M1 ----\
                >---- M3
        M2 ----/

    M1 and M2 are intentionally unordered. Either dish can be loaded first.
    M3 receives ordered credit only when both dishes were already correctly
    loaded before the dishwasher is closed.

    Grasping, opening the already-initially-open dishwasher, and robot motion
    are not semantic milestones.
    """

    def __init__(self, *args, **kwargs):
        self._reset_long_horizon_metric_tracker(initial_state=None)
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.dishwasher = self.register_fixture_ref(
            "dishwasher", dict(id=FixtureType.DISHWASHER)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.dishwasher)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        dish0_lang = self.get_obj_lang("dish0")
        dish1_lang = self.get_obj_lang("dish1")
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions).format(
                dish0_lang=dish0_lang, dish1_lang=dish1_lang
            )
        else:
            ep_meta["lang"] = (
                f"Pick up the {dish0_lang} and {dish1_lang} from the counter, place them in the dishwasher, "
                f"and close the dishwasher door."
            )
        return ep_meta

    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        self._place_robot()

    def _place_robot(self):
        X_OFS = 0.25
        Y_OFS = -0.15

        x_ofs = (self.dishwasher.width / 2) + X_OFS
        TEST_OFS = 0.23
        inits = []

        (
            robot_base_pos_left,
            robot_base_ori_left,
        ) = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.dishwasher, offset=(-x_ofs, Y_OFS)
        )
        test_pos_left, _ = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.dishwasher, offset=(-x_ofs - TEST_OFS, Y_OFS)
        )

        if not OU.check_fxtr_contact(
            self, test_pos_left
        ) and not OU.point_outside_scene(self, test_pos_left):
            inits.append((robot_base_pos_left, robot_base_ori_left, "right"))

        (
            robot_base_pos_right,
            robot_base_ori_right,
        ) = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.dishwasher, offset=(x_ofs, Y_OFS)
        )
        test_pos_right, _ = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.dishwasher, offset=(x_ofs + TEST_OFS, Y_OFS)
        )

        if not OU.check_fxtr_contact(
            self, test_pos_right
        ) and not OU.point_outside_scene(self, test_pos_right):
            inits.append((robot_base_pos_right, robot_base_ori_right, "left"))

        if len(inits) == 0:
            return False
        random_index = self.rng.integers(len(inits))
        robot_base_pos, robot_base_ori, side = inits[random_index]
        self.drawer_side = side
        self.init_robot_base_pos_anchor = robot_base_pos
        self.init_robot_base_ori_anchor = robot_base_ori
        return True

    def _setup_scene(self):
        super()._setup_scene()

        self.dishwasher.open_door(env=self)
        self.dishwasher.slide_rack(env=self, value=1.0)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="dish0",
                obj_groups="cup",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.dishwasher,
                    ),
                    size=(0.35, 0.25),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dish1",
                obj_groups="bowl",
                object_scale=0.5,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.dishwasher,
                    ),
                    size=(0.35, 0.25),
                    pos=("ref", -1.0),
                ),
            )
        )
        return cfgs

    # =====================================================================
    # Task-specific semantic milestone definitions
    # =====================================================================

    def get_semantic_milestone_spec(self):
        """
        Dynamic two-object sequence.

        The physical objects may be handled in either order. The first dish
        that is grasped becomes dish A; the other dish becomes dish B.

        Ordered chain:
            pick_A -> place_A -> pick_B -> place_B -> close_dishwasher
        """
        return {
            "pick_first_dish": {
                "description": "The first selected dish is grasped.",
                "dependencies": [],
                "persistent": True,
            },
            "place_first_dish": {
                "description": "The first selected dish is placed on the dishwasher rack.",
                "dependencies": ["pick_first_dish"],
                "persistent": True,
            },
            "pick_second_dish": {
                "description": "The remaining dish is grasped after the first dish is placed.",
                "dependencies": ["place_first_dish"],
                "persistent": True,
            },
            "place_second_dish": {
                "description": "The remaining dish is placed on the dishwasher rack.",
                "dependencies": ["pick_second_dish"],
                "persistent": True,
            },
            "close_dishwasher": {
                "description": "The dishwasher door is closed after both dishes are loaded.",
                "dependencies": ["place_first_dish", "place_second_dish"],
                "persistent": True,
            },
        }

    def _get_dish_names(self):
        return ["dish0", "dish1"]

    def _get_grasped_dishes(self):
        return {
            name: bool(
                OU.check_obj_grasped(
                    self,
                    name,
                )
            )
            for name in self._get_dish_names()
        }

    def _get_loaded_dishes(self):
        return {
            name: bool(
                self.dishwasher.check_rack_contact(
                    self,
                    name,
                )
            )
            for name in self._get_dish_names()
        }

    def _get_raw_semantic_milestone_state(self):
        """
        Return CURRENT stage states using the already-bound dynamic order.

        The dynamic order is assigned in _update_long_horizon_metric_tracker().
        This function itself does not mutate the order.
        """
        grasped = self._get_grasped_dishes()
        loaded = self._get_loaded_dishes()
        closed = bool(
            self.dishwasher.is_closed(
                self,
                th=0.05,
            )
        )

        first = self._lh_first_dish
        second = self._lh_second_dish

        first_grasped = bool(
            first is not None and grasped[first]
        )
        first_loaded = bool(
            first is not None
            and loaded[first]
            and not grasped[first]
        )

        second_grasped = bool(
            second is not None and grasped[second]
        )
        second_loaded = bool(
            second is not None
            and loaded[second]
            and not grasped[second]
        )

        return {
            "pick_first_dish": first_grasped,
            "place_first_dish": first_loaded,
            "pick_second_dish": second_grasped,
            "place_second_dish": second_loaded,
            "close_dishwasher": closed,
        }

    def _get_ordered_semantic_milestone_state(self):
        raw = self._get_raw_semantic_milestone_state()

        first_place_valid = bool(
            self._lh_ordered_ever["place_first_dish"]
            and raw["place_first_dish"]
        )
        second_pick_valid = bool(
            self._lh_ordered_ever["pick_second_dish"]
            and first_place_valid
            and raw["pick_second_dish"]
        )
        second_place_valid = bool(
            self._lh_ordered_ever["place_second_dish"]
            and first_place_valid
            and raw["place_second_dish"]
        )

        return {
            "pick_first_dish": bool(
                self._lh_ordered_ever["pick_first_dish"]
                and raw["pick_first_dish"]
            ),
            "place_first_dish": first_place_valid,
            "pick_second_dish": second_pick_valid,
            "place_second_dish": second_place_valid,
            "close_dishwasher": bool(
                self._lh_ordered_ever["close_dishwasher"]
                and second_place_valid
                and raw["close_dishwasher"]
            ),
        }

    # =====================================================================
    # Per-episode long-horizon metric tracker
    # =====================================================================

    def _reset_long_horizon_metric_tracker(self, initial_state=None):
        spec = self.get_semantic_milestone_spec()
        names = list(spec.keys())

        self._lh_step_count = 0
        self._lh_progress_curve = []

        if initial_state is None:
            initial_state = {
                name: False for name in names
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
            name: False for name in names
        }
        self._lh_ordered_ever = {
            name: False for name in names
        }

        self._lh_first_success_step = None

        # Dynamic order:
        # first dish picked becomes A, the other becomes B.
        self._lh_first_dish = None
        self._lh_second_dish = None

    def _reset_internal(self):
        super()._reset_internal()
        self._reset_long_horizon_metric_tracker()

    def _get_current_ordered_progress(self, ordered_state):
        """
        Stage progress for dynamic A/B object ordering.

        Valid stages:
            0.2 pick A
            0.4 place A
            0.6 pick B
            0.8 place B
            1.0 close dishwasher
        """
        if ordered_state["close_dishwasher"]:
            return 1.0

        if ordered_state["place_second_dish"]:
            return 4.0 / 5.0

        if ordered_state["pick_second_dish"]:
            return 3.0 / 5.0

        if ordered_state["place_first_dish"]:
            return 2.0 / 5.0

        if ordered_state["pick_first_dish"]:
            return 1.0 / 5.0

        return 0.0

    def _update_long_horizon_metric_tracker(self):
        """
        Update long-horizon metrics once for one policy timestep.

        The two dish identities are not fixed in the instruction. The first
        grasped dish becomes A; the remaining dish becomes B. This supports
        both possible object orders while still enforcing:

            pick A -> place A -> pick B -> place B -> close.
        """
        self._lh_step_count += 1

        names = list(
            self.get_semantic_milestone_spec().keys()
        )

        grasped = self._get_grasped_dishes()
        loaded = self._get_loaded_dishes()
        closed = bool(
            self.dishwasher.is_closed(
                self,
                th=0.05,
            )
        )

        # --------------------------------------------------------------
        # Bind dynamic first object when the first grasp event occurs.
        # --------------------------------------------------------------
        if self._lh_first_dish is None:
            currently_grasped = [
                name for name in self._get_dish_names()
                if grasped[name]
            ]

            if currently_grasped:
                self._lh_first_dish = currently_grasped[0]
                self._lh_ordered_ever["pick_first_dish"] = True
                self._lh_unordered_ever["pick_first_dish"] = True

        # --------------------------------------------------------------
        # First object placement.
        # --------------------------------------------------------------
        if self._lh_first_dish is not None:
            first = self._lh_first_dish

            if (
                self._lh_ordered_ever["pick_first_dish"]
                and loaded[first]
                and not grasped[first]
            ):
                self._lh_ordered_ever["place_first_dish"] = True
                self._lh_unordered_ever["place_first_dish"] = True

        # --------------------------------------------------------------
        # Bind / pick second object only after first object is placed.
        # --------------------------------------------------------------
        if (
            self._lh_second_dish is None
            and self._lh_first_dish is not None
            and self._lh_ordered_ever["place_first_dish"]
        ):
            second_candidates = [
                name for name in self._get_dish_names()
                if name != self._lh_first_dish
            ]

            for name in second_candidates:
                if grasped[name]:
                    self._lh_second_dish = name
                    self._lh_ordered_ever["pick_second_dish"] = True
                    self._lh_unordered_ever["pick_second_dish"] = True
                    break

        # --------------------------------------------------------------
        # Second object placement.
        # --------------------------------------------------------------
        if self._lh_second_dish is not None:
            second = self._lh_second_dish

            if (
                self._lh_ordered_ever["pick_second_dish"]
                and loaded[second]
                and not grasped[second]
            ):
                self._lh_ordered_ever["place_second_dish"] = True
                self._lh_unordered_ever["place_second_dish"] = True

        # --------------------------------------------------------------
        # Close dishwasher only after both placement stages were already
        # physically valid before the close event.
        # --------------------------------------------------------------
        raw = self._get_raw_semantic_milestone_state()
        close_rising = bool(
            raw["close_dishwasher"]
            and not self._lh_prev_raw["close_dishwasher"]
        )

        close_ready = bool(
            self._lh_ordered_ever["place_first_dish"]
            and self._lh_ordered_ever["place_second_dish"]
            and self._lh_prev_raw["place_first_dish"]
            and self._lh_prev_raw["place_second_dish"]
        )

        if (
            close_rising
            and close_ready
        ):
            self._lh_ordered_ever["close_dishwasher"] = True
            self._lh_unordered_ever["close_dishwasher"] = True

        # UPS can also count an early physical close, even if it is not
        # ordered-valid.
        if raw["close_dishwasher"]:
            self._lh_unordered_ever["close_dishwasher"] = True

        ordered_state = self._get_ordered_semantic_milestone_state()
        current_ops = self._get_current_ordered_progress(
            ordered_state
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
            self._lh_first_success_step = self._lh_step_count

        self._lh_prev_raw = {
            name: bool(raw[name])
            for name in names
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
        num_milestones = len(spec)

        ups = (
            sum(
                self._lh_unordered_ever.values()
            )
            / float(num_milestones)
            if num_milestones > 0
            else 0.0
        )

        ordered_state = self._get_ordered_semantic_milestone_state()
        ops = self._get_current_ordered_progress(
            ordered_state
        )

        sr = float(
            self._lh_first_success_step is not None
        )

        if self._lh_first_success_step is None:
            nct = -1.0
        else:
            horizon = float(
                getattr(
                    self,
                    "horizon",
                    0,
                )
            )
            nct = (
                float(self._lh_first_success_step) / horizon
                if horizon > 0
                else float(self._lh_first_success_step)
            )

        return {
            "SR": float(sr),
            "NCT": float(nct),
            "UPS": float(ups),
            "OPS": float(ops),
            "progress_curve": list(
                self._lh_progress_curve
            ),
        }

    # =====================================================================
    # Original RoboCasa success criterion, preserved
    # =====================================================================

    def _check_success(self):
        """
        Success if:
        1) All dishes are touching the top rack.
        2) The dishwasher door is fully closed.
        """
        is_dishwasher_closed = self.dishwasher.is_closed(self, th=0.05)

        dish_names = ["dish0", "dish1"]

        all_checks_passed = True
        for dish_name in dish_names:
            check = self.dishwasher.check_rack_contact(self, dish_name)

            if not check:
                all_checks_passed = False
                break

        are_dishes_correctly_placed = all_checks_passed

        if are_dishes_correctly_placed and is_dishwasher_closed:
            return True
        else:
            return False
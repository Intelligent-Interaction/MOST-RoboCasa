from copy import deepcopy

import numpy as np

from robocasa.environments.kitchen.kitchen import *


class CleanAndStoreAfterMeal(Kitchen):
    """
    Layout-2 RoboCasa-Long task.

    Segments:
      1. The empty tupperware starts on the second-highest shelf of the left
         freezer compartment, with only the left door open. Put the chicken
         drumstick and banana from the island counter into that tupperware.

      2. One loose cereal box and one loose milk carton also start on
         the island. Move each one next to the matching item type on the counter
         to the left of the stovetop.

      3. The pan and sponge start beside the island sink. Put both into the
         sink.

    The two food items start directly on the island. The tupperware starts
    empty inside the left freezer compartment and does not need to be moved.
    The task ends after six pick-and-place actions; closing the door is not
    required.
    """

    # Layout 2 contains many accessories on both the main counter and island.
    # They are irrelevant to this task and can block reset regions.
    DEFAULT_FIXTURE_UPDATES = {
        "toaster": {"enable": False},
        "knife_block": {"enable": False},
        "paper_towel": {"enable": False},
        "coffee_machine": {"enable": False},
        "digital_scale": {"enable": False},
        "oil_bottle": {"enable": False},
        "jar": {"enable": False},
        "paprika": {"enable": False},
        "cinnamon": {"enable": False},
        "turmeric": {"enable": False},
        "plant": {"enable": False},
        "soap_dispenser": {"enable": False},
        "fruit_bowl": {"enable": False},
        "salt_shaker": {"enable": False},
    }

    # Layout 2 rotates the island group by pi. Increasing a fixture's local
    # y coordinate therefore moves it away from the wall-side main counter.
    # Shift the entire island assembly and its three stools together so the
    # aisle widens without changing any object-to-island placement regions.
    AISLE_WIDENING_M = 0.15
    ISLAND_BASE_POSITIONS = {
        "island": (0.0, 1.0, 0.46),
        "island_stack_1": (0.0, 0.63, 0.47),
        "island_panel_cab_right": (1.25, 1.0, 0.47),
        "island_panel_cab_left": (-1.25, 1.0, 0.47),
        "island_base_box": (0.0, 1.2, 0.445),
    }
    STOOL_BASE_POSITIONS = {
        "stool_1": (2.1, -3.25, None),
        "stool_2": (2.75, -3.25, None),
        "stool_3": (3.4, -3.25, None),
    }

    def __init__(self, *args, **kwargs):
        self._reset_long_horizon_metric_tracker(initial_state=None)

        layout_and_style_ids = kwargs.get("layout_and_style_ids")
        layout_ids = kwargs.get("layout_ids")
        style_ids = kwargs.get("style_ids")

        if layout_and_style_ids is not None:
            kwargs.pop("layout_ids", None)
            kwargs.pop("style_ids", None)
        elif layout_ids is None and style_ids is None:
            kwargs["layout_ids"] = 2
            kwargs["style_ids"] = 2

        fixture_updates = deepcopy(
            kwargs.pop("update_fxtr_cfg_dict", None) or {}
        )
        for fixture_name, default_update in self.DEFAULT_FIXTURE_UPDATES.items():
            merged = deepcopy(fixture_updates.get(fixture_name, {}))
            merged.update(deepcopy(default_update))
            fixture_updates[fixture_name] = merged

        for fixture_name, (x, y, z) in self.ISLAND_BASE_POSITIONS.items():
            merged = deepcopy(fixture_updates.get(fixture_name, {}))
            merged["pos"] = [x, y + self.AISLE_WIDENING_M, z]
            fixture_updates[fixture_name] = merged

        for fixture_name, (x, y, z) in self.STOOL_BASE_POSITIONS.items():
            merged = deepcopy(fixture_updates.get(fixture_name, {}))
            merged["pos"] = [x, y - self.AISLE_WIDENING_M, z]
            fixture_updates[fixture_name] = merged
        kwargs["update_fxtr_cfg_dict"] = fixture_updates

        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        # Exact fixture names confirmed from the layout-2 episode metadata.
        self.dining_counter = self.register_fixture_ref(
            "dining_counter",
            dict(id="island_island_group"),
        )
        self.sink = self.register_fixture_ref(
            "sink",
            dict(id="sink_island_group"),
        )

        self.stool_left = self.register_fixture_ref(
            "stool_left",
            dict(id="stool_1_room"),
        )
        self.stool_middle = self.register_fixture_ref(
            "stool_middle",
            dict(id="stool_2_room"),
        )
        self.stool_right = self.register_fixture_ref(
            "stool_right",
            dict(id="stool_3_room"),
        )

        self.stovetop = self.register_fixture_ref(
            "stovetop",
            dict(id="stovetop_main_group"),
        )
        self.main_counter = self.register_fixture_ref(
            "main_counter",
            dict(id="counter_main_main_group"),
        )
        self.fridge = self.register_fixture_ref(
            "fridge",
            dict(id="fridge_main_group"),
        )

        self.init_robot_base_ref = self.stool_middle

    def _setup_scene(self):
        super()._setup_scene()

        # On this side-by-side model the freezer is on the left. Keep the
        # right fridge door shut and open only the left freezer door.
        self.fridge.close_door(self, compartment="fridge")
        self.fridge.open_door(
            self,
            min=0.92,
            max=1.0,
            compartment="freezer",
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Put the chicken drumstick and banana in the tupperware on the second shelf of the freezer, "
            "put the cereal box and milk carton beside their matching items on the counter by the stove, "
            "and put the pan and sponge in the island sink."
        )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        # ==============================================================
        # 1. Island dining-counter items
        #
        # Both loose food items start on the island. The empty tupperware
        # starts on the second-highest shelf of the left freezer compartment.
        # The cereal and milk cartons occupy the neighboring island region.
        #
        # Pan and sponge are placed in a separate region directly beside the
        # island sink. The two target reference boxes remain on the main
        # counter to the left of the stovetop.
        # ==============================================================

        # The container starts near the front edge of the left freezer shelf.
        # Rotate its long side into the shelf depth: this half is narrower.
        #
        #   second-highest freezer shelf: empty tupperware
        cfgs.append(
            dict(
                name="meal_tupperware",
                obj_groups="tupperware",
                graspable=True,
                object_scale=[2, 2, 1],
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="freezer",
                        rack_index=-2,
                        reg_type="shelf",
                    ),
                    size=("obj.y", "obj.x"),
                    pos=(-0.5, -1.5),
                    rotation=np.pi / 2,
                    margin=0.005,
                ),
            )
        )

        cfgs.append(
            dict(
                name="chicken_drumstick",
                obj_groups="chicken_drumstick",
                graspable=True,
                object_scale=1.0,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool_middle),
                    size=(0.24, 0.22),
                    pos=(0.45, -0.80),
                    rotation=0,
                    margin=0.005,
                ),
            )
        )

        cfgs.append(
            dict(
                name="banana",
                obj_groups="banana",
                graspable=True,
                object_scale=0.9,
                placement=dict(
                    fixture=self.dining_counter,
                    reuse_region_from="chicken_drumstick",
                    size=(0.24, 0.22),
                    pos=(-0.88, 0.88),
                    offset=(0.0, 0.0),
                    rotation=0,
                    margin=0.005,
                ),
            )
        )

        # Put the two loose boxes together in the neighboring stool-left
        # region, beside the compact food cluster:
        #
        #   loose cereal box | loose milk carton
        #
        # Both are broad rectangular objects and are easier to grasp than
        # narrow bottles.
        cfgs.append(
            dict(
                name="cereal_box_loose",
                obj_groups="cereal",
                graspable=True,
                object_scale=0.9,
                max_size=(0.19, 0.15, None),
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool_left),
                    size=(0.28, 0.24),
                    pos=(-0.40, -0.52),
                    rotation=0,
                    margin=0.01,
                ),
            )
        )

        cfgs.append(
            dict(
                name="milk_carton_loose",
                obj_groups="milk",
                graspable=True,
                object_scale=0.9,
                max_size=(0.15, 0.12, None),
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool_left),
                    size=(0.28, 0.24),
                    pos=(0.40, -0.52),
                    rotation=0,
                    margin=0.01,
                ),
            )
        )

        # ==============================================================
        # 2. Box target groups on the counter left of the stovetop
        #
        # One cereal reference box and one milk reference package start
        # here. The corresponding loose object must be brought next to it.
        # ==============================================================

        cfgs.append(
            dict(
                name="cereal_box_reference",
                obj_groups="cereal",
                object_scale=0.9,
                max_size=(0.19, 0.15, None),
                placement=dict(
                    fixture=self.main_counter,
                    sample_region_kwargs=dict(
                        ref=self.stovetop,
                        loc="left",
                        top_size=(0.92, 0.42),
                    ),
                    size=(0.28, 0.24),
                    pos=(-0.68, -0.12),
                    rotation=0,
                    margin=0.01,
                ),
            )
        )

        cfgs.append(
            dict(
                name="milk_carton_reference",
                obj_groups="milk",
                object_scale=0.9,
                max_size=(0.15, 0.12, None),
                placement=dict(
                    fixture=self.main_counter,
                    reuse_region_from="cereal_box_reference",
                    size=(0.28, 0.24),
                    pos=(0.68, -0.12),
                    rotation=0,
                    margin=0.01,
                ),
            )
        )

        # ==============================================================
        # 3. Pan and sponge beside the island sink
        #
        # They share one valid reset region next to the sink. This keeps both
        # objects on the island dining counter and visually close to the sink,
        # instead of placing them near the stovetop.
        # ==============================================================

        cfgs.append(
            dict(
                name="pan",
                obj_groups="pan",
                graspable=True,
                washable=True,
                object_scale=0.78,
                max_size=(0.32, 0.38, None),
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                        top_size=(0.72, 0.42),
                    ),
                    size=(0.46, 0.38),
                    pos=(-0.42, -0.80),
                    rotation=0,
                    margin=0.01,
                ),
            )
        )

        cfgs.append(
            dict(
                name="sponge",
                obj_groups="sponge",
                graspable=True,
                object_scale=0.85,
                placement=dict(
                    fixture=self.dining_counter,
                    reuse_region_from="pan",
                    size=(0.20, 0.20),
                    pos=(0.68, -0.84),
                    rotation=0,
                    margin=0.01,
                ),
            )
        )

        return cfgs

    @staticmethod
    def _xy_distance(env, name_a, name_b):
        pos_a = np.asarray(
            env.sim.data.body_xpos[env.obj_body_id[name_a]],
            dtype=np.float64,
        )[:2]
        pos_b = np.asarray(
            env.sim.data.body_xpos[env.obj_body_id[name_b]],
            dtype=np.float64,
        )[:2]
        return float(np.linalg.norm(pos_a - pos_b))

    def get_task_progress(self):
        chicken_in_tupperware = OU.check_obj_in_receptacle(
            self, "chicken_drumstick", "meal_tupperware"
        )
        banana_in_tupperware = OU.check_obj_in_receptacle(
            self, "banana", "meal_tupperware"
        )
        tupperware_on_freezer_shelf = self.fridge.check_rack_contact(
            self,
            "meal_tupperware",
            rack_index=-2,
            compartment="freezer",
        )
        cereal_sorted = (
            self._xy_distance(
                self,
                "cereal_box_loose",
                "cereal_box_reference",
            )
            <= 0.28
        )
        milk_carton_sorted = (
            self._xy_distance(
                self,
                "milk_carton_loose",
                "milk_carton_reference",
            )
            <= 0.28
        )

        all_sorting_items_on_counter = all(
            OU.check_obj_fixture_contact(
                self,
                name,
                self.main_counter,
            )
            for name in (
                "cereal_box_reference",
                "cereal_box_loose",
                "milk_carton_loose",
                "milk_carton_reference",
            )
        )

        pan_in_sink = OU.obj_inside_of(
            self,
            "pan",
            self.sink,
            partial_check=False,
        )
        sponge_in_sink = OU.obj_inside_of(
            self,
            "sponge",
            self.sink,
            partial_check=False,
        )

        return {
            "chicken_in_meal_tupperware": bool(chicken_in_tupperware),
            "banana_in_meal_tupperware": bool(banana_in_tupperware),
            "meal_tupperware_on_second_freezer_shelf": bool(tupperware_on_freezer_shelf),
            "cereal_box_sorted": bool(cereal_sorted),
            "milk_carton_sorted": bool(milk_carton_sorted),
            "all_sorting_items_on_counter": bool(all_sorting_items_on_counter),
            "pan_in_sink": bool(pan_in_sink),
            "sponge_in_sink": bool(sponge_in_sink),
        }

    # =====================================================================
    # Long-horizon metrics: six order-free pick-and-place branches
    # =====================================================================

    def get_semantic_milestone_spec(self):
        """Return one pick and one placement milestone for each moved object."""
        branches = (
            (
                "chicken",
                "chicken_drumstick",
                "chicken_in_meal_tupperware",
                "the chicken drumstick is in the freezer tupperware",
            ),
            (
                "banana",
                "banana",
                "banana_in_meal_tupperware",
                "the banana is in the freezer tupperware",
            ),
            (
                "milk",
                "milk_carton_loose",
                "milk_carton_sorted",
                "the milk carton is beside the matching milk carton",
            ),
            (
                "box",
                "cereal_box_loose",
                "cereal_box_sorted",
                "the cereal box is beside the matching cereal box",
            ),
            (
                "pan",
                "pan",
                "pan_in_sink",
                "the pan is in the island sink",
            ),
            (
                "sponge",
                "sponge",
                "sponge_in_sink",
                "the sponge is in the island sink",
            ),
        )

        spec = {}
        for branch_name, _, _, place_description in branches:
            pick_name = f"{branch_name}_in_hand"
            place_name = f"{branch_name}_placed"
            spec[pick_name] = {
                "description": f"The {branch_name} is grasped by the robot gripper.",
                "dependencies": [],
                "persistent": False,
            }
            spec[place_name] = {
                "description": place_description.capitalize() + ".",
                "dependencies": [pick_name],
                "persistent": True,
            }
        return spec

    @staticmethod
    def _long_horizon_branches():
        return (
            ("chicken", "chicken_drumstick", "chicken_in_meal_tupperware"),
            ("banana", "banana", "banana_in_meal_tupperware"),
            ("milk", "milk_carton_loose", "milk_carton_sorted"),
            ("box", "cereal_box_loose", "cereal_box_sorted"),
            ("pan", "pan", "pan_in_sink"),
            ("sponge", "sponge", "sponge_in_sink"),
        )

    def _check_task_obj_grasped(self, obj_name):
        """Use a tolerant local grasp check without changing object_utils."""
        try:
            return bool(OU.check_obj_grasped(self, obj_name, threshold=0.10))
        except TypeError:
            return bool(OU.check_obj_grasped(self, obj_name))

    def _get_raw_semantic_milestone_state(self):
        progress = self.get_task_progress()
        raw = {}

        for branch_name, obj_name, progress_name in self._long_horizon_branches():
            in_hand = self._check_task_obj_grasped(obj_name)
            placed = bool(progress[progress_name])

            # Sorting also requires the movable carton / box to remain on the
            # intended main counter, matching the original success criterion.
            if branch_name in ("milk", "box"):
                placed = bool(
                    placed
                    and OU.check_obj_fixture_contact(
                        self,
                        obj_name,
                        self.main_counter,
                    )
                )

            raw[f"{branch_name}_in_hand"] = in_hand
            raw[f"{branch_name}_placed"] = bool(placed and not in_hand)

        return raw

    def _get_ordered_semantic_milestone_state(self):
        raw = self._get_raw_semantic_milestone_state()
        ordered = {}

        for branch_name, _, _ in self._long_horizon_branches():
            pick_name = f"{branch_name}_in_hand"
            place_name = f"{branch_name}_placed"
            place_valid = bool(
                self._lh_ordered_ever[place_name]
                and raw[place_name]
            )
            pick_valid = bool(
                self._lh_ordered_ever[pick_name]
                and (raw[pick_name] or place_valid)
            )
            ordered[pick_name] = pick_valid
            ordered[place_name] = place_valid

        return ordered

    def _reset_long_horizon_metric_tracker(self, initial_state=None):
        names = list(self.get_semantic_milestone_spec().keys())
        if initial_state is None:
            initial_state = {name: False for name in names}

        self._lh_step_count = 0
        self._lh_prev_raw = {
            name: bool(initial_state.get(name, False)) for name in names
        }
        self._lh_unordered_ever = {name: False for name in names}
        self._lh_ordered_ever = {name: False for name in names}
        self._lh_regressed = {name: False for name in names}
        self._lh_first_success_step = None
        self._lh_progress_curve = []

    def _reset_internal(self):
        super()._reset_internal()
        self._reset_long_horizon_metric_tracker(
            initial_state=self._get_raw_semantic_milestone_state(),
        )

    def _get_current_ordered_progress(self, ordered_state=None):
        if ordered_state is None:
            ordered_state = self._get_ordered_semantic_milestone_state()
        return sum(int(bool(value)) for value in ordered_state.values()) / 12.0

    def _update_long_horizon_metric_tracker(self):
        self._lh_step_count += 1
        spec = self.get_semantic_milestone_spec()
        raw = self._get_raw_semantic_milestone_state()

        # UPS ignores order and records every raw milestone ever observed.
        for name in spec:
            if raw[name]:
                self._lh_unordered_ever[name] = True

        # Each object has only its own pick -> place dependency. The six
        # branches do not depend on one another and can occur in any order.
        for branch_name, _, _ in self._long_horizon_branches():
            pick_name = f"{branch_name}_in_hand"
            place_name = f"{branch_name}_placed"

            if raw[pick_name]:
                self._lh_ordered_ever[pick_name] = True
            if raw[place_name]:
                # Infer a missed brief grasp frame from a valid placement.
                self._lh_ordered_ever[pick_name] = True
                self._lh_ordered_ever[place_name] = True

        ordered_state = self._get_ordered_semantic_milestone_state()

        for branch_name, _, _ in self._long_horizon_branches():
            place_name = f"{branch_name}_placed"
            if (
                self._lh_ordered_ever[place_name]
                and self._lh_prev_raw[place_name]
                and not ordered_state[place_name]
            ):
                self._lh_regressed[place_name] = True

        self._lh_progress_curve.append(
            float(self._get_current_ordered_progress(ordered_state))
        )

        if self._check_success() and self._lh_first_success_step is None:
            self._lh_first_success_step = self._lh_step_count

        self._lh_prev_raw = {name: bool(raw[name]) for name in spec}

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        self._update_long_horizon_metric_tracker()
        return reward, done, info

    def get_long_horizon_metrics(self):
        names = list(self.get_semantic_milestone_spec().keys())
        ups = sum(
            int(bool(self._lh_unordered_ever[name])) for name in names
        ) / float(len(names))

        return {
            "SR": float(self._lh_first_success_step is not None),
            "OPS": float(self._get_current_ordered_progress()),
            "UPS": float(ups),
            "NCT": (
                -1.0
                if self._lh_first_success_step is None
                else float(self._lh_first_success_step)
                / float(max(getattr(self, "horizon", 1), 1))
            ),
            "progress_curve": list(self._lh_progress_curve),
        }

    def _check_success(self):
        progress = self.get_task_progress()

        required = (
            "chicken_in_meal_tupperware",
            "banana_in_meal_tupperware",
            "meal_tupperware_on_second_freezer_shelf",
            "cereal_box_sorted",
            "milk_carton_sorted",
            "all_sorting_items_on_counter",
            "pan_in_sink",
            "sponge_in_sink",
        )

        moved_objects = (
            "chicken_drumstick",
            "banana",
            "cereal_box_loose",
            "milk_carton_loose",
            "pan",
            "sponge",
        )

        gripper_far = all(
            OU.gripper_obj_far(self, obj_name=name)
            for name in moved_objects
        )

        return all(progress[key] for key in required) and gripper_far

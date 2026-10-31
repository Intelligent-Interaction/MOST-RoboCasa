from copy import deepcopy
import numpy as np
import robosuite.utils.transform_utils as T
from robocasa.environments.kitchen.kitchen import *


class PrepareBreakfast(Kitchen):
    """
    PrepareBreakfast: breakfast with toast, milk, and fruit.

    Intended order:
      1. Start the toaster.
      2. Wait for the toaster to finish, then put the toast in the left bread
         bowl, which already sits on the dining counter.
      3. Take the milk from beside the coffee machine and place it next to the
         bread bowl.
      4. Take the apple and banana from the counter to the right of the sink
         and put them in the right fruit bowl, which is already on the dining
         counter.

    Final dining state:
      - bread_bowl contains toast;
      - fruit_bowl contains apple and banana;
      - milk is next to bread_bowl.
    """

    TOASTER_FIXTURE_UPDATE = {
        "placement": {
            "fixture": "counter_1_right_main_group",
            "sample_region_kwargs": {
                "ref": "fridge",
                "loc": "left",
            },
            "size": (0.52, 0.42),
            "pos": (-0.90, 0.85),
            "rotation": (0, 0),
            "margin": 0.02,
        },
    }

    SHELVES_FIXTURE_UPDATE = {
        "enable": False,
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
            kwargs["layout_ids"] = 4
            kwargs["style_ids"] = 4

        fixture_updates = deepcopy(
            kwargs.pop("update_fxtr_cfg_dict", None) or {}
        )

        toaster_update = deepcopy(
            fixture_updates.get("toaster", {})
        )
        toaster_update.update(
            deepcopy(self.TOASTER_FIXTURE_UPDATE)
        )
        fixture_updates["toaster"] = toaster_update

        shelves_update = deepcopy(
            fixture_updates.get("shelves", {})
        )
        shelves_update.update(
            deepcopy(self.SHELVES_FIXTURE_UPDATE)
        )
        fixture_updates["shelves"] = shelves_update

        kwargs["update_fxtr_cfg_dict"] = fixture_updates

        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.stool = self.register_fixture_ref(
            "stool",
            dict(id=FixtureType.STOOL),
        )

        self.dining_counter = self.register_fixture_ref(
            "dining_counter",
            dict(
                id=FixtureType.DINING_COUNTER,
                ref=self.stool,
            ),
        )

        self.toaster = self.register_fixture_ref(
            "toaster",
            dict(id=FixtureType.TOASTER),
        )

        self.coffee_machine = self.register_fixture_ref(
            "coffee_machine",
            dict(id=FixtureType.COFFEE_MACHINE),
        )

        self.coffee_counter = self.register_fixture_ref(
            "coffee_counter",
            dict(
                id=FixtureType.COUNTER,
                ref=self.coffee_machine,
            ),
        )

        self.sink = self.register_fixture_ref(
            "sink",
            dict(id=FixtureType.SINK),
        )

        # The apple and banana keep their original source positions on the
        # counter to the right of the sink. The fruit bowl now starts on the
        # dining counter and is not part of the transported objects.
        self.fruit_counter = self.register_fixture_ref(
            "fruit_counter",
            dict(
                id=FixtureType.COUNTER,
                ref=self.sink,
            ),
        )

        self.init_robot_base_ref = self.toaster

    def _setup_scene(self):
        super()._setup_scene()

        self.toaster_started = False
        self.toaster_finished = False
        self.toast_slot = None

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        ep_meta["lang"] = (
            "Toast the bread and put it in the bowl on the left, "
            "put the milk beside that bowl, and put the apple and banana in the bowl on the right."
        )

        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="toast",
                obj_groups="sandwich_bread",
                object_scale=[1.0, 1.0, 1.15],
                rotate_upright=True,
                placement=dict(
                    fixture=self.toaster,
                    rotation=(0, 0),
                ),
            )
        )

        # The two bowls start on the dining counter and remain there. Their
        # separated x positions make bread_bowl the left bowl and fruit_bowl
        # the right bowl from the dining side of the counter.
        cfgs.append(
            dict(
                name="bread_bowl",
                obj_groups="bowl",
                object_scale=1.15,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool),
                    size=(0.36, 0.34),
                    pos=(-0.25, "ref"),
                    rotation=0,
                    margin=0.0,
                ),
            )
        )

        cfgs.append(
            dict(
                name="fruit_bowl",
                obj_groups="bowl",
                object_scale=1.0,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool),
                    size=(0.36, 0.34),
                    pos=(0.25, "ref"),
                    rotation=0,
                    margin=0.0,
                ),
            )
        )

        # Milk starts in open counter space immediately in front of the coffee
        # machine. It is transported next to the left bread bowl.
        cfgs.append(
            dict(
                name="milk",
                obj_groups="milk",
                graspable=True,
                object_scale=0.9,
                max_size=(0.15, 0.12, None),
                placement=dict(
                    fixture=self.coffee_counter,
                    sample_region_kwargs=dict(
                        ref=self.coffee_machine,
                        loc="right",
                    ),
                    size=(0.35, 0.24),
                    pos=(0.25, 0.65),
                    rotation=0,
                    margin=0.01,
                ),
            )
        )

        # The apple and banana keep the original reset region and normalized
        # positions on the counter to the right of the sink. The apple now
        # selects that region because fruit_bowl has moved to the table.
        #
        # Do not use small fixed metric target sizes here. Random bowl / fruit
        # assets can be larger than those boxes, which causes:
        #   Invalid x range for placement initializer
        #
        # "obj.x" and "obj.y" make each target box match the sampled object's
        # actual footprint. margin=0 avoids shrinking that box below the object.
        cfgs.append(
            dict(
                name="apple",
                obj_groups="apple",
                graspable=True,
                object_scale=0.85,
                placement=dict(
                    fixture=self.fruit_counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="right",
                    ),
                    size=("obj.x", "obj.y"),
                    pos=(0.05, -0.62),
                    rotation=0,
                    margin=0.0,
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
                    fixture=self.fruit_counter,
                    reuse_region_from="apple",
                    size=("obj.x", "obj.y"),
                    pos=(0.75, -0.62),
                    rotation=0,
                    margin=0.0,
                ),
            )
        )

        return cfgs

    def _update_toaster_progress(self):
        state_dict = self.toaster.get_state(self)

        for slot_pair in range(len(state_dict.keys())):
            slot_state = self.toaster.get_state(
                self,
                slot_pair=slot_pair,
            )
            toast_in_slot = self.toaster.check_slot_contact(
                self,
                "toast",
                slot_pair=slot_pair,
            )

            if toast_in_slot and slot_state["turned_on"]:
                self.toaster_started = True
                self.toast_slot = slot_pair

        if self.toaster_started and self.toast_slot is not None:
            slot_state = self.toaster.get_state(
                self,
                slot_pair=self.toast_slot,
            )
            if not slot_state["turned_on"]:
                self.toaster_finished = True

    def update_state(self):
        super().update_state()
        self._update_toaster_progress()

    def _obj_in_receptacle_xy(
        self,
        obj_name,
        receptacle_name,
        margin=0.05,
    ):
        obj_center = np.asarray(
            self.sim.data.body_xpos[self.obj_body_id[obj_name]],
            dtype=np.float64,
        )

        receptacle = self.objects[receptacle_name]
        receptacle_body_id = self.obj_body_id[receptacle_name]
        receptacle_pos = np.asarray(
            self.sim.data.body_xpos[receptacle_body_id],
            dtype=np.float64,
        )
        receptacle_quat = T.convert_quat(
            np.asarray(
                self.sim.data.body_xquat[receptacle_body_id],
                dtype=np.float64,
            ),
            to="xyzw",
        )

        bbox = np.asarray(
            receptacle.get_bbox_points(
                trans=receptacle_pos,
                rot=receptacle_quat,
            ),
            dtype=np.float64,
        )
        if bbox.shape[0] < 3:
            raise RuntimeError(
                f"{receptacle_name} returned invalid bbox shape {bbox.shape}"
            )

        p0 = bbox[0]
        axis_x = bbox[1] - p0
        axis_y = bbox[2] - p0
        x_norm_sq = float(np.dot(axis_x, axis_x))
        y_norm_sq = float(np.dot(axis_y, axis_y))

        if x_norm_sq <= 1e-12 or y_norm_sq <= 1e-12:
            raise RuntimeError(
                f"{receptacle_name} has a degenerate horizontal bbox"
            )

        relative = obj_center - p0
        normalized_x = float(np.dot(relative, axis_x) / x_norm_sq)
        normalized_y = float(np.dot(relative, axis_y) / y_norm_sq)

        return (
            -margin <= normalized_x <= 1.0 + margin
            and -margin <= normalized_y <= 1.0 + margin
        )

    def get_task_progress(self):
        self._update_toaster_progress()

        bread_bowl_on_counter = OU.check_obj_fixture_contact(
            self,
            "bread_bowl",
            self.dining_counter,
        )
        milk_on_counter = OU.check_obj_fixture_contact(
            self,
            "milk",
            self.dining_counter,
        )
        if bread_bowl_on_counter and milk_on_counter:
            milk_xy = np.asarray(
                self.sim.data.body_xpos[self.obj_body_id["milk"]][:2],
                dtype=np.float64,
            )
            bowl_xy = np.asarray(
                self.sim.data.body_xpos[self.obj_body_id["bread_bowl"]][:2],
                dtype=np.float64,
            )
            milk_next_to_bread_bowl = (
                float(np.linalg.norm(milk_xy - bowl_xy)) <= 0.35
            )
        else:
            milk_next_to_bread_bowl = False

        return {
            "toast_heated": bool(self.toaster_finished),
            "toast_in_bread_bowl": self._obj_in_receptacle_xy(
                "toast",
                "bread_bowl",
            ),
            "apple_in_fruit_bowl": self._obj_in_receptacle_xy(
                "apple",
                "fruit_bowl",
            ),
            "banana_in_fruit_bowl": self._obj_in_receptacle_xy(
                "banana",
                "fruit_bowl",
            ),
            "bread_bowl_on_dining_counter": bool(bread_bowl_on_counter),
            "fruit_bowl_on_dining_counter": (
                OU.check_obj_fixture_contact(
                    self,
                    "fruit_bowl",
                    self.dining_counter,
                )
            ),
            "milk_on_dining_counter": bool(milk_on_counter),
            "milk_next_to_bread_bowl": bool(milk_next_to_bread_bowl),
        }

    # =====================================================================
    # Long-horizon metrics: four order-free operation branches
    # =====================================================================

    def get_semantic_milestone_spec(self):
        """
        Return ten milestones for four independently ordered operations.

        Apple, banana, and milk each use pick -> place. Toast is one composite
        operation with its required internal order:
            toaster started -> toaster finished -> toast picked -> toast placed.
        """
        return {
            "apple_in_hand": {
                "description": "The apple is grasped by the robot gripper.",
                "dependencies": [],
                "persistent": False,
            },
            "apple_in_fruit_bowl": {
                "description": "The apple is released in the fruit bowl.",
                "dependencies": ["apple_in_hand"],
                "persistent": True,
            },
            "banana_in_hand": {
                "description": "The banana is grasped by the robot gripper.",
                "dependencies": [],
                "persistent": False,
            },
            "banana_in_fruit_bowl": {
                "description": "The banana is released in the fruit bowl.",
                "dependencies": ["banana_in_hand"],
                "persistent": True,
            },
            "milk_in_hand": {
                "description": "The milk carton is grasped by the robot gripper.",
                "dependencies": [],
                "persistent": False,
            },
            "milk_next_to_bread_bowl": {
                "description": "The milk carton is released beside the bread bowl.",
                "dependencies": ["milk_in_hand"],
                "persistent": True,
            },
            "toaster_started": {
                "description": "The toaster is turned on with the toast inside.",
                "dependencies": [],
                "persistent": True,
            },
            "toaster_finished": {
                "description": "The toaster finishes heating the toast.",
                "dependencies": ["toaster_started"],
                "persistent": True,
            },
            "toast_in_hand": {
                "description": "The heated toast is grasped by the robot gripper.",
                "dependencies": ["toaster_finished"],
                "persistent": False,
            },
            "toast_in_bread_bowl": {
                "description": "The heated toast is released in the bread bowl.",
                "dependencies": ["toast_in_hand"],
                "persistent": True,
            },
        }

    @staticmethod
    def _simple_long_horizon_branches():
        return (
            ("apple", "apple", "apple_in_fruit_bowl"),
            ("banana", "banana", "banana_in_fruit_bowl"),
            ("milk", "milk", "milk_next_to_bread_bowl"),
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

        for branch_name, obj_name, progress_name in self._simple_long_horizon_branches():
            in_hand = self._check_task_obj_grasped(obj_name)
            placed = bool(progress[progress_name])
            if branch_name == "milk":
                placed = bool(placed and progress["milk_on_dining_counter"])
            raw[f"{branch_name}_in_hand"] = in_hand
            raw[progress_name] = bool(placed and not in_hand)

        toast_in_hand = self._check_task_obj_grasped("toast")
        raw.update(
            {
                "toaster_started": bool(self.toaster_started),
                "toaster_finished": bool(self.toaster_finished),
                "toast_in_hand": toast_in_hand,
                "toast_in_bread_bowl": bool(
                    progress["toast_in_bread_bowl"] and not toast_in_hand
                ),
            }
        )
        return raw

    def _get_ordered_semantic_milestone_state(self):
        raw = self._get_raw_semantic_milestone_state()
        ordered = {}

        for branch_name, _, place_name in self._simple_long_horizon_branches():
            pick_name = f"{branch_name}_in_hand"
            place_valid = bool(
                self._lh_ordered_ever[place_name]
                and raw[place_name]
            )
            ordered[pick_name] = bool(
                self._lh_ordered_ever[pick_name]
                and (raw[pick_name] or place_valid)
            )
            ordered[place_name] = place_valid

        started_valid = bool(
            self._lh_ordered_ever["toaster_started"]
            and raw["toaster_started"]
        )
        finished_valid = bool(
            self._lh_ordered_ever["toaster_finished"]
            and raw["toaster_finished"]
            and started_valid
        )
        toast_place_valid = bool(
            self._lh_ordered_ever["toast_in_bread_bowl"]
            and raw["toast_in_bread_bowl"]
            and finished_valid
        )
        toast_pick_valid = bool(
            self._lh_ordered_ever["toast_in_hand"]
            and (raw["toast_in_hand"] or toast_place_valid)
            and finished_valid
        )
        ordered.update(
            {
                "toaster_started": started_valid,
                "toaster_finished": finished_valid,
                "toast_in_hand": toast_pick_valid,
                "toast_in_bread_bowl": toast_place_valid,
            }
        )
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
        return sum(int(bool(value)) for value in ordered_state.values()) / 10.0

    def _update_long_horizon_metric_tracker(self):
        self._lh_step_count += 1
        spec = self.get_semantic_milestone_spec()
        raw = self._get_raw_semantic_milestone_state()

        for name in spec:
            if raw[name]:
                self._lh_unordered_ever[name] = True

        # Apple, banana, and milk are independent pick -> place branches.
        for branch_name, _, place_name in self._simple_long_horizon_branches():
            pick_name = f"{branch_name}_in_hand"
            if raw[pick_name]:
                self._lh_ordered_ever[pick_name] = True
            if raw[place_name]:
                self._lh_ordered_ever[pick_name] = True
                self._lh_ordered_ever[place_name] = True

        # Toast is independent of the other three operations, while retaining
        # the necessary internal order of heating before pick-and-place.
        if raw["toaster_started"]:
            self._lh_ordered_ever["toaster_started"] = True
        if raw["toaster_finished"]:
            self._lh_ordered_ever["toaster_started"] = True
            self._lh_ordered_ever["toaster_finished"] = True

        toast_ready = self._lh_ordered_ever["toaster_finished"]
        if raw["toast_in_hand"] and toast_ready:
            self._lh_ordered_ever["toast_in_hand"] = True
        if raw["toast_in_bread_bowl"] and toast_ready:
            self._lh_ordered_ever["toast_in_hand"] = True
            self._lh_ordered_ever["toast_in_bread_bowl"] = True

        ordered_state = self._get_ordered_semantic_milestone_state()
        persistent_places = [
            "apple_in_fruit_bowl",
            "banana_in_fruit_bowl",
            "milk_next_to_bread_bowl",
            "toast_in_bread_bowl",
        ]
        for place_name in persistent_places:
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
            "toast_heated",
            "toast_in_bread_bowl",
            "apple_in_fruit_bowl",
            "banana_in_fruit_bowl",
            "bread_bowl_on_dining_counter",
            "fruit_bowl_on_dining_counter",
            "milk_on_dining_counter",
            "milk_next_to_bread_bowl",
        )

        moved_objects = (
            "toast",
            "milk",
            "apple",
            "banana",
        )

        gripper_far = all(
            OU.gripper_obj_far(self, obj_name=obj_name)
            for obj_name in moved_objects
        )

        return all(progress[key] for key in required) and gripper_far

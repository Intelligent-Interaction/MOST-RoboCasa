import numpy as np
import robosuite.utils.transform_utils as T

from robocasa.environments.kitchen.kitchen import *


class PackFruit(Kitchen):
    """
    Collect six fruits from three parts of the kitchen and pack all of them
    into one tupperware container.

    Each episode contains exactly six fruits:
      - three or four bananas;
      - at least one apple;
      - at least one pear.

    Two fruits start beside the microwave, two start on the dining counter,
    and two start beside the stove. The single tupperware remains on the
    counter next to the refrigerator.
    """

    FRUIT_NAMES = tuple(f"fruit{i}" for i in range(6))
    TUPPERWARE_NAME = "tupperware"

    def __init__(self, *args, **kwargs):
        self._reset_long_horizon_metric_tracker(initial_state=None)

        # This task is authored specifically for layout/style 7. Override any
        # scene-selection arguments so resets cannot silently sample a layout
        # whose fixture arrangement does not match the task.
        kwargs.pop("layout_ids", None)
        kwargs.pop("style_ids", None)
        kwargs["layout_and_style_ids"] = [(7, 7)]

        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.fridge = self.register_fixture_ref(
            "fridge",
            dict(id=FixtureType.FRIDGE),
        )
        self.packing_counter = self.register_fixture_ref(
            "packing_counter",
            dict(id=FixtureType.COUNTER, ref=self.fridge),
        )

        self.microwave = self.register_fixture_ref(
            "microwave",
            dict(id=FixtureType.MICROWAVE),
        )
        self.microwave_counter = self.register_fixture_ref(
            "microwave_counter",
            dict(
                id=FixtureType.COUNTER,
                ref=self.microwave,
                size=(0.55, 0.20),
            ),
        )

        self.dining_counter = self.register_fixture_ref(
            "dining_counter",
            dict(id=FixtureType.DINING_COUNTER, ref=self.fridge),
        )

        self.stove = self.register_fixture_ref(
            "stove",
            dict(id=FixtureType.STOVE),
        )
        self.stove_counter = self.register_fixture_ref(
            "stove_counter",
            dict(
                id=FixtureType.COUNTER,
                ref=self.stove,
                size=(0.55, 0.20),
            ),
        )

        self.init_robot_base_ref = self.fridge

    def _setup_scene(self):
        super()._setup_scene()

        # The refrigerator is not manipulated in this task. Keeping both doors
        # closed leaves the route to the nearby packing counter unobstructed.
        self.fridge.close_door(env=self, compartment="fridge")
        self.fridge.close_door(env=self, compartment="freezer")

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Collect and pack all the fruit from the counter near the microwave, "
            "the dining counter, and the counter near the stove into the "
            "tupperware."
        )
        return ep_meta

    def _sample_fruit_groups(self):
        """Sample six fruit categories with a deliberately banana-heavy mix."""
        banana_count = int(self.rng.choice([3, 4]))

        if banana_count == 4:
            apple_count = 1
            pear_count = 1
        else:
            # With three remaining slots, randomly give the extra slot to
            # either apples or pears while keeping both categories present.
            apple_count = int(self.rng.choice([1, 2]))
            pear_count = 3 - apple_count

        fruit_groups = (
            ["banana"] * banana_count
            + ["apple"] * apple_count
            + ["pear"] * pear_count
        )
        return [str(group) for group in self.rng.permutation(fruit_groups)]

    def _get_obj_cfgs(self):
        cfgs = []
        fruit_groups = self._sample_fruit_groups()

        # Keep the same three source areas used by the previous version of the
        # task. Each area contributes exactly two fruits. A small placement box
        # around each old can / snack position adds natural reset variation.
        source_placements = (
            dict(
                fixture=self.microwave_counter,
                ref=self.microwave,
                # Select one open side of the microwave at random. Both fruits
                # reuse that same side region and are spread apart near its
                # front edge, keeping the microwave out of the grasp path.
                loc="left_right",
                positions=((-0.60, -0.82), (0.60, -0.82)),
            ),
            dict(
                fixture=self.dining_counter,
                ref=self.fridge,
                positions=((-0.45, -0.75), (0.45, -0.75)),
            ),
            dict(
                fixture=self.stove_counter,
                ref=self.stove,
                positions=((-0.75, -0.70), (0.75, -0.70)),
            ),
        )

        fruit_index = 0
        for source in source_placements:
            first_name_in_region = self.FRUIT_NAMES[fruit_index]

            for position in source["positions"]:
                fruit_name = self.FRUIT_NAMES[fruit_index]
                sample_region_kwargs = dict(ref=source["ref"])
                if "loc" in source:
                    sample_region_kwargs["loc"] = source["loc"]

                placement = dict(
                    fixture=source["fixture"],
                    sample_region_kwargs=sample_region_kwargs,
                    size=(0.24, 0.18),
                    pos=position,
                    rotation=(-np.pi / 4, np.pi / 4),
                    margin=0.0,
                )

                if fruit_name != first_name_in_region:
                    placement["reuse_region_from"] = first_name_in_region

                cfgs.append(
                    dict(
                        name=fruit_name,
                        obj_groups=fruit_groups[fruit_index],
                        graspable=True,
                        object_scale=float(self.rng.uniform(0.8, 0.9)),
                        placement=placement,
                    )
                )
                fruit_index += 1

        # Keep one enlarged tupperware on the original fridge-side packing
        # counter. The target region gives it positional variation while
        # remaining close to the previous pair of containers.
        cfgs.append(
            dict(
                name=self.TUPPERWARE_NAME,
                obj_groups="tupperware",
                object_scale=[2.5, 2.5, 1.25],
                placement=dict(
                    fixture=self.packing_counter,
                    sample_region_kwargs=dict(ref=self.fridge),
                    size=(0.80, 0.45),
                    pos=(0.0, -0.90),
                    rotation=np.pi / 2,
                ),
            )
        )

        return cfgs

    def _obj_in_tupperware(self, obj_name):
        """
        Return whether an object's center lies in the container's XY footprint.

        This preserves the collection-friendly containment rule used by the
        previous task: vertical containment and contact are not required.
        """
        obj_center = np.asarray(
            self.sim.data.body_xpos[self.obj_body_id[obj_name]],
            dtype=np.float64,
        )

        receptacle = self.objects[self.TUPPERWARE_NAME]
        receptacle_body_id = self.obj_body_id[self.TUPPERWARE_NAME]
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
                f"{self.TUPPERWARE_NAME} returned an invalid bounding box: "
                f"shape={bbox.shape}"
            )

        p0 = bbox[0]
        axis_x = bbox[1] - p0
        axis_y = bbox[2] - p0
        axis_x_norm_sq = float(np.dot(axis_x, axis_x))
        axis_y_norm_sq = float(np.dot(axis_y, axis_y))

        if axis_x_norm_sq <= 1e-12 or axis_y_norm_sq <= 1e-12:
            raise RuntimeError(
                f"{self.TUPPERWARE_NAME} has a degenerate horizontal bounding box"
            )

        relative = obj_center - p0
        normalized_x = float(np.dot(relative, axis_x) / axis_x_norm_sq)
        normalized_y = float(np.dot(relative, axis_y) / axis_y_norm_sq)

        eps = 1e-6
        return (
            -eps <= normalized_x <= 1.0 + eps
            and -eps <= normalized_y <= 1.0 + eps
        )

    def get_task_progress(self):
        fruit_progress = {
            f"{name}_in_tupperware": self._obj_in_tupperware(name)
            for name in self.FRUIT_NAMES
        }

        fruit_progress["all_fruits_in_tupperware"] = all(
            fruit_progress[f"{name}_in_tupperware"]
            for name in self.FRUIT_NAMES
        )
        fruit_progress["tupperware_on_counter"] = OU.check_obj_fixture_contact(
            self,
            self.TUPPERWARE_NAME,
            self.packing_counter,
        )
        return fruit_progress

    def _check_success(self):
        progress = self.get_task_progress()
        gripper_far = all(
            OU.gripper_obj_far(self, obj_name=name)
            for name in self.FRUIT_NAMES
        )

        return (
            progress["all_fruits_in_tupperware"]
            and progress["tupperware_on_counter"]
            and gripper_far
        )

    # =====================================================================
    # Task-specific semantic milestones
    # =====================================================================

    def get_semantic_milestone_spec(self):
        """
        Return twelve milestones: one pick and one place for each fruit.

        The six fruit branches are independent, so the fruits may be packed
        in any order. Within one branch, placing depends only on picking that
        same fruit.
        """
        spec = {}

        for fruit_name in self.FRUIT_NAMES:
            pick_name = f"{fruit_name}_in_hand"
            place_name = f"{fruit_name}_in_tupperware"

            spec[pick_name] = {
                "description": (
                    f"{fruit_name} is grasped by the robot gripper."
                ),
                "dependencies": [],
                "persistent": False,
            }
            spec[place_name] = {
                "description": (
                    f"{fruit_name} is released inside the tupperware."
                ),
                "dependencies": [pick_name],
                "persistent": True,
            }

        return spec

    def _check_task_obj_grasped(self, obj_name):
        """Use the reference task's tolerant grasp detection."""
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
        """Return the current physical state of all twelve milestones."""
        raw = {}

        for fruit_name in self.FRUIT_NAMES:
            pick_name = f"{fruit_name}_in_hand"
            place_name = f"{fruit_name}_in_tupperware"
            in_hand = self._check_task_obj_grasped(fruit_name)

            raw[pick_name] = in_hand
            raw[place_name] = bool(
                self._obj_in_tupperware(fruit_name)
                and not in_hand
            )

        return raw

    def _get_ordered_semantic_milestone_state(self):
        """
        Return the currently valid pick/place credit for every fruit.

        A fruit contributes:
            0 / 12 while neither held nor correctly placed;
            1 / 12 while held;
            2 / 12 after a valid release in the tupperware.

        Consequently, a wrong release removes the pick credit, and removing a
        previously placed fruit removes its place credit until it is returned.
        """
        raw = self._get_raw_semantic_milestone_state()
        ordered = {}

        for fruit_name in self.FRUIT_NAMES:
            pick_name = f"{fruit_name}_in_hand"
            place_name = f"{fruit_name}_in_tupperware"

            place_valid = bool(
                self._lh_ordered_ever[place_name]
                and raw[place_name]
            )
            pick_valid = bool(
                self._lh_ordered_ever[pick_name]
                and (
                    raw[pick_name]
                    or place_valid
                )
            )

            ordered[pick_name] = pick_valid
            ordered[place_name] = place_valid

        return ordered

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
            name: bool(initial_state.get(name, False))
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
        """Run the normal reset, then initialize this episode's metrics."""
        super()._reset_internal()

        initial_raw = self._get_raw_semantic_milestone_state()
        self._reset_long_horizon_metric_tracker(
            initial_state=initial_raw,
        )

    def _get_current_ordered_progress(self, ordered_state=None):
        if ordered_state is None:
            ordered_state = self._get_ordered_semantic_milestone_state()

        completed_milestones = sum(
            int(bool(value))
            for value in ordered_state.values()
        )
        return completed_milestones / 12.0

    def _update_long_horizon_metric_tracker(self):
        """Update metrics once per policy timestep."""
        self._lh_step_count += 1

        spec = self.get_semantic_milestone_spec()
        raw = self._get_raw_semantic_milestone_state()

        # UPS ignores ordering and records whether a raw milestone ever held.
        for name in spec.keys():
            if raw[name]:
                self._lh_unordered_ever[name] = True

        # Each fruit has its own pick -> place dependency. There is no
        # dependency between different fruits, so their operation order is
        # completely free.
        for fruit_name in self.FRUIT_NAMES:
            pick_name = f"{fruit_name}_in_hand"
            place_name = f"{fruit_name}_in_tupperware"

            if raw[pick_name]:
                self._lh_ordered_ever[pick_name] = True

            if raw[place_name]:
                # A correct placement is also evidence that the pick occurred
                # if the grasp detector missed its brief positive frame.
                if not self._lh_ordered_ever[pick_name]:
                    self._lh_ordered_ever[pick_name] = True
                self._lh_ordered_ever[place_name] = True

        ordered_state = self._get_ordered_semantic_milestone_state()

        # Remember whether an achieved placement later became invalid. OPS and
        # the progress curve already regress immediately through ordered_state.
        for fruit_name in self.FRUIT_NAMES:
            place_name = f"{fruit_name}_in_tupperware"

            if not self._lh_ordered_ever[place_name]:
                continue

            falling_edge = bool(
                self._lh_prev_raw[place_name]
                and not ordered_state[place_name]
            )
            if falling_edge:
                self._lh_regressed[place_name] = True

        current_progress = self._get_current_ordered_progress(
            ordered_state,
        )
        self._lh_progress_curve.append(float(current_progress))

        success = bool(self._check_success())
        if success and self._lh_first_success_step is None:
            self._lh_first_success_step = self._lh_step_count

        self._lh_prev_raw = {
            name: bool(raw[name])
            for name in spec.keys()
        }

    def _post_action(self, action):
        """Preserve normal post-action behavior and then update metrics."""
        reward, done, info = super()._post_action(action)

        self._update_long_horizon_metric_tracker()

        return reward, done, info

    # =====================================================================
    # Public metric API
    # =====================================================================

    def get_long_horizon_metrics(self):
        spec = self.get_semantic_milestone_spec()
        names = list(spec.keys())

        ups = sum(
            int(bool(self._lh_unordered_ever[name]))
            for name in names
        ) / float(len(names))

        ops = self._get_current_ordered_progress()

        return {
            "SR": float(self._lh_first_success_step is not None),
            "OPS": float(ops),
            "UPS": float(ups),
            "NCT": (
                -1.0
                if self._lh_first_success_step is None
                else float(self._lh_first_success_step)
                / float(max(getattr(self, "horizon", 1), 1))
            ),
            "progress_curve": list(self._lh_progress_curve),
        }

from robocasa.environments.kitchen.kitchen import *
from robocasa.models.objects.kitchen_objects import get_cats_by_type, OBJ_CATEGORIES


class PackIdenticalLunches(Kitchen):
    """
    Pack Identical Lunch: composite task for Packing Lunches activity.

    Simulates the task of packing identical lunches with vegetables and meats.
    There are 2 identical vegetables and 2 identical meats, and 2 tupperwares.
    Each tupperware needs to contain 1 vegetable and 1 meat.

    Steps:
        1. Pick vegetables and meats from the fridge.
        2. Place 1 vegetable and 1 meat in each of the two tupperwares on the counter.

    Long-horizon semantic milestones
    ---------------------------------
    The four source objects are interchangeable within category:
        vegetable0 / vegetable1 are equivalent,
        meat0 / meat1 are equivalent.

    The two tupperwares may also be packed in either order.

    Therefore, milestones are defined by the four semantic lunch "slots",
    NOT by a fixed object order:

        M1. tupperware0_has_one_vegetable
        M2. tupperware0_has_one_meat
        M3. tupperware1_has_one_vegetable
        M4. tupperware1_has_one_meat

    All four milestones are parallel and have no dependencies. This means:
        - either vegetable can satisfy either vegetable slot;
        - either meat can satisfy either meat slot;
        - either tupperware can be packed first;
        - the four physical objects can be manipulated in any order.

    A slot is valid only if it contains exactly one object of the required
    category, and that object is not simultaneously counted in the other
    tupperware. This mirrors the original task's no-duplicate requirement.

    Gripper distance is not a semantic milestone; it remains part of the
    official terminal success condition.
    """

    def __init__(self, *args, **kwargs):
        self._reset_long_horizon_metric_tracker(initial_state=None)
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.fridge)
        )
        self.init_robot_base_ref = self.fridge

    def _setup_scene(self):
        self.fridge.open_door(self)
        super()._setup_scene()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        veg_lang = self.get_obj_lang("vegetable0")
        meat_lang = self.get_obj_lang("meat0")

        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions).format(
                veg_lang=veg_lang, meat_lang=meat_lang
            )
        else:
            ep_meta["lang"] = (
                f"Place one {veg_lang} and one {meat_lang} in each tupperware on the nearby counter, "
                "to pack two identical lunches."
            )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        all_veg_categories = get_cats_by_type(
            types=["vegetable"], obj_registries=self.obj_registries
        )
        all_meat_categories = get_cats_by_type(
            types=["meat"], obj_registries=self.obj_registries
        )

        # Filter for graspable vegetables
        veg_categories = []
        for cat in all_veg_categories:
            if cat in OBJ_CATEGORIES:
                for reg in OBJ_CATEGORIES[cat]:
                    cat_meta = OBJ_CATEGORIES[cat][reg]
                    if hasattr(cat_meta, "graspable") and cat_meta.graspable:
                        veg_categories.append(cat)
                        break

        # Filter for graspable meats (excluding shrimp)
        meat_categories = []
        for cat in all_meat_categories:
            if cat != "shrimp" and cat in OBJ_CATEGORIES:
                for reg in OBJ_CATEGORIES[cat]:
                    cat_meta = OBJ_CATEGORIES[cat][reg]
                    if hasattr(cat_meta, "graspable") and cat_meta.graspable:
                        meat_categories.append(cat)
                        break

        selected_veg = self.rng.choice(veg_categories, size=1, replace=True)[0]
        selected_meat = self.rng.choice(meat_categories, size=1, replace=True)[0]

        cfgs.append(
            dict(
                name="vegetable0",
                obj_groups=selected_veg,
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.2),
                    pos=(-0.25, -1.0),
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                ),
            )
        )
        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups=selected_veg,
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.2),
                    pos=(0.25, -1.0),
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat0",
                obj_groups=selected_meat,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.2),
                    pos=(-0.25, -1.0),
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                ),
            )
        )
        cfgs.append(
            dict(
                name="meat1",
                obj_groups=selected_meat,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.2),
                    pos=(0.25, -1.0),
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                ),
            )
        )

        cfgs.append(
            dict(
                name="tupperware0",
                obj_groups="tupperware",
                object_scale=[2.5, 2.5, 1.25],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.fridge,
                    ),
                    size=(1.5, 0.5),
                    pos=(-0.5, -1.0),
                    rotation=(np.pi / 2),
                ),
            )
        )
        cfgs.append(
            dict(
                name="tupperware1",
                obj_groups="tupperware",
                object_scale=[2.5, 2.5, 1.25],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.fridge,
                    ),
                    reuse_region_from="tupperware0",
                    size=(1.5, 0.5),
                    pos=(0.5, -1.0),
                    rotation=(np.pi / 2),
                ),
            )
        )

        return cfgs

    # =====================================================================
    # Task-specific semantic milestone definitions
    # =====================================================================

    def get_semantic_milestone_spec(self):
        """
        Dynamic four-object packing sequence.

        There are two interchangeable vegetables and two interchangeable meats.
        The two tupperwares are also interchangeable in order. Therefore the
        progress is defined by action stages rather than fixed object IDs:

            pick_1 -> place_1 -> pick_2 -> place_2
                   -> pick_3 -> place_3 -> pick_4 -> place_4

        A placement stage is counted by the current number of valid filled
        lunch slots. A valid final state requires each tupperware to contain
        exactly one vegetable and exactly one meat.
        """
        return {
            "pick_1": {
                "description": "A first lunch object is grasped.",
                "dependencies": [],
                "persistent": True,
            },
            "place_1": {
                "description": "One valid lunch slot is filled.",
                "dependencies": ["pick_1"],
                "persistent": True,
            },
            "pick_2": {
                "description": "A second lunch object is grasped.",
                "dependencies": ["place_1"],
                "persistent": True,
            },
            "place_2": {
                "description": "Two valid lunch slots are filled.",
                "dependencies": ["pick_2"],
                "persistent": True,
            },
            "pick_3": {
                "description": "A third lunch object is grasped.",
                "dependencies": ["place_2"],
                "persistent": True,
            },
            "place_3": {
                "description": "Three valid lunch slots are filled.",
                "dependencies": ["pick_3"],
                "persistent": True,
            },
            "pick_4": {
                "description": "A fourth lunch object is grasped.",
                "dependencies": ["place_3"],
                "persistent": True,
            },
            "place_4": {
                "description": (
                    "Both tupperwares are correctly packed with one vegetable "
                    "and one meat each."
                ),
                "dependencies": ["pick_4"],
                "persistent": True,
            },
        }

    def _get_lunch_object_names(self):
        return [
            "vegetable0",
            "vegetable1",
            "meat0",
            "meat1",
        ]

    def _get_vegetable_names(self):
        return [
            "vegetable0",
            "vegetable1",
        ]

    def _get_meat_names(self):
        return [
            "meat0",
            "meat1",
        ]

    def _get_tupperware_names(self):
        return [
            "tupperware0",
            "tupperware1",
        ]


    def _check_gripper_closed(
        self,
        threshold=0.10,
    ):
        """
        A looser local gripper-closed check for lunch objects.

        RoboCasa's OU.check_obj_grasped default threshold is 0.035. That can
        miss partially closed but visually valid grasps. This local helper does
        not change global RoboCasa behavior.
        """
        joint_names = [
            "gripper0_right_finger_joint1",
            "gripper0_right_finger_joint2",
        ]

        try:
            joint_positions = [
                self.sim.data.qpos[
                    self.sim.model.get_joint_qpos_addr(joint_name)
                ]
                for joint_name in joint_names
            ]
        except Exception:
            return False

        return bool(
            all(
                float(pos) < threshold
                for pos in joint_positions
            )
        )

    def _check_lunch_obj_grasped(
        self,
        obj_name,
    ):
        """
        Robust pick signal for small lunch objects.

        Primary signal:
            OU.check_obj_grasped with a looser gripper-closed threshold.

        Fallback signal:
            gripper is closed and close to the object. This handles frames
            where gripper-object contact flickers even though the object is
            visibly held.
        """
        try:
            if OU.check_obj_grasped(
                self,
                obj_name,
                threshold=0.10,
            ):
                return True
        except TypeError:
            try:
                if OU.check_obj_grasped(
                    self,
                    obj_name,
                ):
                    return True
            except Exception:
                pass
        except Exception:
            pass

        if not self._check_gripper_closed(
            threshold=0.10,
        ):
            return False

        try:
            return bool(
                not OU.gripper_obj_far(
                    self,
                    obj_name,
                    th=0.08,
                )
            )
        except TypeError:
            try:
                return bool(
                    not OU.gripper_obj_far(
                        self,
                        obj_name,
                    )
                )
            except Exception:
                return False
        except Exception:
            return False

    def _get_grasped_objects(self):
        return {
            name: self._check_lunch_obj_grasped(name)
            for name in self._get_lunch_object_names()
        }


    def _check_obj_in_receptacle_xy(
        self,
        obj_name,
        receptacle_name,
        th=None,
    ):
        """
        Receptacle containment without requiring direct contact.

        RoboCasa's OU.check_obj_in_receptacle requires:
            contact(obj, receptacle) AND xy_distance < threshold.

        For this task, food can be stacked inside one tupperware. The upper
        object may not touch the tupperware, so contact-based containment can
        incorrectly fail. Here we only require the object center to lie within
        the tupperware's XY support region.
        """
        recep = self.objects[receptacle_name]

        obj_pos = np.array(
            self.sim.data.body_xpos[
                self.obj_body_id[obj_name]
            ]
        )
        recep_pos = np.array(
            self.sim.data.body_xpos[
                self.obj_body_id[receptacle_name]
            ]
        )

        if th is None:
            th = recep.horizontal_radius * 0.85

        return bool(
            np.linalg.norm(obj_pos[:2] - recep_pos[:2]) < th
        )

    def _get_lunch_contents(self):
        """
        Return simulator-grounded tupperware contents.

        Objects currently grasped by the robot are intentionally not counted as
        placed, even if their geometry temporarily overlaps the receptacle.
        This prevents transporting an object through a tupperware from being
        credited as a valid placement.
        """
        grasped = self._get_grasped_objects()

        contents = {
            "tupperware0": {
                "vegetable": [],
                "meat": [],
            },
            "tupperware1": {
                "vegetable": [],
                "meat": [],
            },
        }

        for tupperware in self._get_tupperware_names():
            for veg in self._get_vegetable_names():
                if (
                    not grasped[veg]
                    and self._check_obj_in_receptacle_xy(
                        veg,
                        tupperware,
                    )
                ):
                    contents[tupperware]["vegetable"].append(veg)

            for meat in self._get_meat_names():
                if (
                    not grasped[meat]
                    and self._check_obj_in_receptacle_xy(
                        meat,
                        tupperware,
                    )
                ):
                    contents[tupperware]["meat"].append(meat)

        return contents

    def _get_valid_slot_objects(self):
        """
        Return the objects occupying valid lunch slots.

        A valid slot means:
            - exactly one vegetable in a given tupperware, or
            - exactly one meat in a given tupperware.

        If the same object is detected in more than one tupperware, the
        duplicate assignment is ignored for progress.
        """
        contents = self._get_lunch_contents()

        assignments = []
        for tupperware in self._get_tupperware_names():
            vegs = contents[tupperware]["vegetable"]
            meats = contents[tupperware]["meat"]

            if len(vegs) == 1:
                assignments.append(
                    (tupperware, "vegetable", vegs[0])
                )
            if len(meats) == 1:
                assignments.append(
                    (tupperware, "meat", meats[0])
                )

        object_counts = {}
        for _, _, obj_name in assignments:
            object_counts[obj_name] = object_counts.get(obj_name, 0) + 1

        valid_objects = set()
        for _, _, obj_name in assignments:
            if object_counts[obj_name] == 1:
                valid_objects.add(obj_name)

        return valid_objects

    def _get_valid_placed_count(self):
        return len(
            self._get_valid_slot_objects()
        )

    def _get_current_stage_index(self):
        """
        Return the current ordered stage index in [0, 8].

        The stage is based on current simulator state:
            even stage 2k   : k valid slots filled;
            odd stage 2k+1 : k valid slots filled and another unplaced lunch
                             object is currently grasped.

        This supports all legal manipulation orders:
            - one box can be packed as A+B first;
            - both boxes can receive A before B;
            - any mixed order is allowed.
        """
        valid_objects = self._get_valid_slot_objects()
        placed_count = len(valid_objects)
        grasped = self._get_grasped_objects()

        stage = 2 * placed_count

        if placed_count < 4:
            grasped_unplaced = any(
                grasped[obj_name] and obj_name not in valid_objects
                for obj_name in self._get_lunch_object_names()
            )
            if grasped_unplaced:
                stage = max(
                    stage,
                    2 * placed_count + 1,
                )

        return min(
            stage,
            8,
        )

    def _get_raw_semantic_milestone_state(self):
        stage = self._get_current_stage_index()

        return {
            "pick_1": stage >= 1,
            "place_1": stage >= 2,
            "pick_2": stage >= 3,
            "place_2": stage >= 4,
            "pick_3": stage >= 5,
            "place_3": stage >= 6,
            "pick_4": stage >= 7,
            "place_4": stage >= 8,
        }

    def _get_ordered_semantic_milestone_state(self):
        raw = self._get_raw_semantic_milestone_state()

        return {
            name: bool(
                self._lh_ordered_ever[name]
                and raw[name]
            )
            for name in self.get_semantic_milestone_spec().keys()
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

    def _reset_internal(self):
        super()._reset_internal()
        self._reset_long_horizon_metric_tracker()

    def _get_current_ordered_progress(self, ordered_state):
        return (
            sum(
                int(bool(ordered_state[name]))
                for name in self.get_semantic_milestone_spec().keys()
            )
            / 8.0
        )

    def _update_long_horizon_metric_tracker(self):
        self._lh_step_count += 1

        spec = self.get_semantic_milestone_spec()
        names = list(spec.keys())
        raw = self._get_raw_semantic_milestone_state()

        # UPS: order ignored. Since the task allows arbitrary object and
        # tupperware order, this largely tracks the same physical stages.
        for name in names:
            if raw[name]:
                self._lh_unordered_ever[name] = True

        # OPS: enforce pick-place alternation by only crediting a milestone
        # when all earlier stage milestones have already been credited.
        for idx, name in enumerate(names):
            if self._lh_ordered_ever[name]:
                continue

            previous_names = names[:idx]
            if (
                raw[name]
                and all(
                    self._lh_ordered_ever[prev]
                    for prev in previous_names
                )
            ):
                self._lh_ordered_ever[name] = True

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
        Success if both tupperwares contain exactly one vegetable and exactly
        one meat.

        Use XY containment instead of contact-based receptacle checking so
        stacked food inside a tupperware is still counted correctly.
        """
        veg_in_0 = []
        meat_in_0 = []
        veg_in_1 = []
        meat_in_1 = []

        for veg in ["vegetable0", "vegetable1"]:
            if self._check_obj_in_receptacle_xy(veg, "tupperware0"):
                veg_in_0.append(veg)
            if self._check_obj_in_receptacle_xy(veg, "tupperware1"):
                veg_in_1.append(veg)

        for meat in ["meat0", "meat1"]:
            if self._check_obj_in_receptacle_xy(meat, "tupperware0"):
                meat_in_0.append(meat)
            if self._check_obj_in_receptacle_xy(meat, "tupperware1"):
                meat_in_1.append(meat)

        tupper0_valid = (
            len(veg_in_0) == 1
            and len(meat_in_0) == 1
        )
        tupper1_valid = (
            len(veg_in_1) == 1
            and len(meat_in_1) == 1
        )

        all_objs = (
            veg_in_0
            + veg_in_1
            + meat_in_0
            + meat_in_1
        )
        unique_objs = set(all_objs)
        no_duplicates = len(all_objs) == len(unique_objs)

        gripper_far = all(
            OU.gripper_obj_far(
                self,
                obj,
            )
            for obj in all_objs
        )

        return (
            tupper0_valid
            and tupper1_valid
            and no_duplicates
            and gripper_far
        )
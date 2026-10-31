from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.object_utils as OU


class PortionHotDogs(Kitchen):
    """
    Portion Hot Dogs: composite task for Portioning Meals activity.

    Simulates the task of portioning hot dog buns and sausages from a bowl into
    plates. The task involves taking 2 hot dog buns and 2 sausages from a bowl
    on the dining counter and placing exactly 1 bun and 1 sausage on each of
    the 2 plates.

    Progress design:
        There are four unordered pick-place actions.

        Any object order is valid:
            bun / bun / sausage / sausage
            sausage / bun / sausage / bun
            ...

        Any plate order is valid:
            plate1 first
            plate2 first
            interleaved

        Final slot constraint:
            each plate must contain exactly one bun and exactly one sausage.

    Long-horizon progress:
        8 action milestones are counted dynamically:
            pick valid food object 1
            place valid food object 1
            ...
            pick valid food object 4
            place valid food object 4

        A placement is valid only when it contributes to a currently valid slot:
            plate1 has exactly one bun
            plate1 has exactly one sausage
            plate2 has exactly one bun
            plate2 has exactly one sausage

        This avoids giving place credit to invalid states such as putting both
        buns on the same plate.

    Original RoboCasa _check_success() is preserved.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS + [22, 58]

    def __init__(
        self,
        obj_registries=("aigen", "lightwheel", "objaverse"),
        *args,
        **kwargs,
    ):
        self._reset_long_horizon_metric_tracker(initial_state=None)
        super().__init__(
            obj_registries=obj_registries,
            *args,
            **kwargs,
        )

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        if "stool1" in self.fixture_refs:
            self.stool1 = self.fixture_refs["stool1"]
            self.stool2 = self.fixture_refs["stool2"]
        else:
            registered_stool_ids = set()
            self.stool1 = None
            self.stool2 = None

            while len(registered_stool_ids) < 2:
                for fixture in self.fixtures.values():
                    if isinstance(
                        fixture,
                        robocasa.models.fixtures.accessories.Stool,
                    ):
                        fixture_id = id(fixture)

                        if fixture_id not in registered_stool_ids:
                            registered_stool_ids.add(fixture_id)

                            if self.stool1 is None:
                                self.stool1 = fixture
                            elif self.stool2 is None:
                                self.stool2 = fixture
                                break

            self.fixture_refs["stool1"] = self.stool1
            self.fixture_refs["stool2"] = self.stool2

        self.dining_counter = self.register_fixture_ref(
            "dining_counter",
            dict(
                id=FixtureType.DINING_COUNTER,
                ref=self.stool1,
            ),
        )

        self.init_robot_base_ref = self.dining_counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(
                self.novel_instructions
            )
        else:
            ep_meta["lang"] = (
                "Place one bun and one sausage from the bowl on each plate."
            )

        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="plate1",
                obj_groups="plate",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool1,
                    ),
                    size=(0.4, 0.4),
                    pos=("ref", "ref"),
                    rotation=0,
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate2",
                obj_groups="plate",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool2,
                    ),
                    size=(0.4, 0.4),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                object_scale=[1.5, 1.5, 1],
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_counter,
                    size=(1.0, 0.6),
                    pos=(0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="hotdog_bun1",
                obj_groups="hotdog_bun",
                graspable=True,
                placement=dict(
                    object="bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="hotdog_bun2",
                obj_groups="hotdog_bun",
                graspable=True,
                placement=dict(
                    object="bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="sausage1",
                obj_groups="sausage",
                graspable=True,
                placement=dict(
                    object="bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="sausage2",
                obj_groups="sausage",
                graspable=True,
                placement=dict(
                    object="bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        return cfgs

    # =====================================================================
    # Task-specific semantic milestones
    # =====================================================================

    def get_semantic_milestone_spec(self):
        """
        Generic 4-PnP action milestones.

        They are intentionally not bound to a fixed object identity or plate
        identity, because both object order and plate order are free.
        """
        return {
            "pnp1_food_in_hand": {
                "description": "The first valid food object is grasped.",
                "dependencies": [],
                "persistent": False,
            },
            "pnp1_food_placed": {
                "description": "The first valid food object is placed.",
                "dependencies": ["pnp1_food_in_hand"],
                "persistent": True,
            },
            "pnp2_food_in_hand": {
                "description": "The second valid food object is grasped.",
                "dependencies": ["pnp1_food_placed"],
                "persistent": False,
            },
            "pnp2_food_placed": {
                "description": "The second valid food object is placed.",
                "dependencies": ["pnp2_food_in_hand"],
                "persistent": True,
            },
            "pnp3_food_in_hand": {
                "description": "The third valid food object is grasped.",
                "dependencies": ["pnp2_food_placed"],
                "persistent": False,
            },
            "pnp3_food_placed": {
                "description": "The third valid food object is placed.",
                "dependencies": ["pnp3_food_in_hand"],
                "persistent": True,
            },
            "pnp4_food_in_hand": {
                "description": "The fourth valid food object is grasped.",
                "dependencies": ["pnp3_food_placed"],
                "persistent": False,
            },
            "pnp4_food_placed": {
                "description": "The fourth valid food object is placed.",
                "dependencies": ["pnp4_food_in_hand"],
                "persistent": True,
            },
        }

    def _bun_names(self):
        return [
            "hotdog_bun1",
            "hotdog_bun2",
        ]

    def _sausage_names(self):
        return [
            "sausage1",
            "sausage2",
        ]

    def _food_names(self):
        return (
            self._bun_names()
            + self._sausage_names()
        )

    def _plate_names(self):
        return [
            "plate1",
            "plate2",
        ]

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

    def _obj_on_plate(
        self,
        obj_name,
        plate_name,
    ):
        return bool(
            OU.check_obj_in_receptacle(
                self,
                obj_name,
                plate_name,
            )
        )

    def _get_plate_members(self):
        """
        Return exactly which objects are currently detected on each plate.

        The final task requires each plate to have exactly one bun and exactly
        one sausage. These per-slot memberships let the metric avoid rewarding
        invalid duplicate-category placements.
        """
        members = {}

        for plate_name in self._plate_names():
            members[plate_name] = {
                "bun": [],
                "sausage": [],
            }

            for obj_name in self._bun_names():
                if self._obj_on_plate(
                    obj_name,
                    plate_name,
                ):
                    members[plate_name]["bun"].append(
                        obj_name
                    )

            for obj_name in self._sausage_names():
                if self._obj_on_plate(
                    obj_name,
                    plate_name,
                ):
                    members[plate_name]["sausage"].append(
                        obj_name
                    )

        return members

    def _get_valid_slot_info(self):
        """
        A slot is valid only when the corresponding plate/category contains
        exactly one object.

        Returns:
            valid_slots:
                set like {"plate1:bun", "plate2:sausage"}
            valid_slot_objs:
                set of physical object names occupying those valid slots
        """
        members = self._get_plate_members()

        valid_slots = set()
        valid_slot_objs = set()

        for plate_name in self._plate_names():
            for category in [
                "bun",
                "sausage",
            ]:
                objs = members[plate_name][category]

                if len(objs) == 1:
                    slot_name = f"{plate_name}:{category}"
                    valid_slots.add(slot_name)
                    valid_slot_objs.add(objs[0])

        return valid_slots, valid_slot_objs

    def _get_raw_semantic_milestone_state(self):
        grasped_objs = set()

        for obj_name in self._food_names():
            if self._check_task_obj_grasped(
                obj_name,
            ):
                grasped_objs.add(obj_name)

        valid_slots, valid_slot_objs = (
            self._get_valid_slot_info()
        )

        return {
            "grasped_objs": grasped_objs,
            "valid_slots": valid_slots,
            "valid_slot_objs": valid_slot_objs,
        }

    def _current_valid_pick_objs(
        self,
        raw=None,
    ):
        if raw is None:
            raw = self._get_raw_semantic_milestone_state()

        # Pick credit is current/superseded:
        #   - valid while a picked object is currently held;
        #   - also valid when that object currently occupies a valid serving
        #     slot.
        #
        # If an object is picked and dropped into an invalid plate state, its
        # pick credit disappears from OPS.
        return set(
            self._lh_ever_picked_objs
        ).intersection(
            set(raw["grasped_objs"])
            | set(raw["valid_slot_objs"])
        )

    def _get_current_action_count(
        self,
        raw=None,
    ):
        if raw is None:
            raw = self._get_raw_semantic_milestone_state()

        current_valid_pick_objs = (
            self._current_valid_pick_objs(raw)
        )

        current_valid_place_slots = set(
            raw["valid_slots"]
        )

        return (
            len(current_valid_pick_objs)
            + len(current_valid_place_slots)
        )

    def _get_ordered_semantic_milestone_state(self):
        """
        Convert the current valid action count into generic PnP milestone
        booleans.

        Because the task is order-free, these generic stages represent count
        progress rather than fixed object/plate identities.
        """
        raw = self._get_raw_semantic_milestone_state()
        count = self._get_current_action_count(raw)

        names = list(
            self.get_semantic_milestone_spec().keys()
        )

        return {
            name: bool(idx < count)
            for idx, name in enumerate(names)
        }

    # =====================================================================
    # Per-episode long-horizon metric tracker
    # =====================================================================

    def _reset_long_horizon_metric_tracker(
        self,
        initial_state=None,
    ):
        del initial_state

        names = list(
            self.get_semantic_milestone_spec().keys()
        )

        self._lh_step_count = 0

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

        self._lh_ever_picked_objs = set()
        self._lh_ever_valid_slots = set()

        self._lh_prev_ordered_state = {
            name: False
            for name in names
        }

        self._lh_first_success_step = None
        self._lh_progress_curve = []

    def _reset_internal(self):
        super()._reset_internal()

        self._reset_long_horizon_metric_tracker(
            initial_state=None
        )

    def _get_current_ordered_progress(
        self,
        ordered_state=None,
    ):
        if ordered_state is None:
            ordered_state = (
                self._get_ordered_semantic_milestone_state()
            )

        names = list(
            self.get_semantic_milestone_spec().keys()
        )

        completed = sum(
            int(bool(ordered_state[name]))
            for name in names
        )

        return (
            float(completed)
            / float(len(names))
        )

    def _get_unordered_progress(self):
        """
        UPS counts ever-observed valid pick actions and ever-observed valid
        serving slots, order ignored.
        """
        count = (
            len(self._lh_ever_picked_objs)
            + len(self._lh_ever_valid_slots)
        )

        return float(count) / 8.0

    def _update_stage_flags_from_count(
        self,
        count,
    ):
        names = list(
            self.get_semantic_milestone_spec().keys()
        )

        for idx, name in enumerate(names):
            if idx < count:
                self._lh_ordered_ever[name] = True

    def _update_unordered_stage_flags_from_count(
        self,
        count,
    ):
        names = list(
            self.get_semantic_milestone_spec().keys()
        )

        for idx, name in enumerate(names):
            if idx < count:
                self._lh_unordered_ever[name] = True

    def _update_long_horizon_metric_tracker(self):
        """
        Update all long-horizon metrics once per policy timestep.

        The four PnPs are order-free, but each place must correspond to a
        currently valid serving slot:
            plate1 exactly one bun
            plate1 exactly one sausage
            plate2 exactly one bun
            plate2 exactly one sausage
        """
        self._lh_step_count += 1

        raw = self._get_raw_semantic_milestone_state()

        # --------------------------------------------------------------
        # Update ever-picked objects.
        # --------------------------------------------------------------
        for obj_name in raw["grasped_objs"]:
            self._lh_ever_picked_objs.add(obj_name)

        # Anti-flicker inference:
        # If an object occupies a valid slot but its grasp was missed, infer
        # that it must have been picked.
        for obj_name in raw["valid_slot_objs"]:
            self._lh_ever_picked_objs.add(obj_name)

        # --------------------------------------------------------------
        # Update ever-valid serving slots.
        # --------------------------------------------------------------
        for slot_name in raw["valid_slots"]:
            self._lh_ever_valid_slots.add(slot_name)

        # --------------------------------------------------------------
        # Convert current valid action count into OPS stage flags.
        # --------------------------------------------------------------
        current_count = self._get_current_action_count(raw)
        self._update_stage_flags_from_count(
            current_count
        )

        unordered_count = int(
            len(self._lh_ever_picked_objs)
            + len(self._lh_ever_valid_slots)
        )
        self._update_unordered_stage_flags_from_count(
            unordered_count
        )

        ordered_state = (
            self._get_ordered_semantic_milestone_state()
        )

        # --------------------------------------------------------------
        # Regression bookkeeping.
        #
        # Since the stage flags are generic count stages, a regression occurs
        # when any previously valid stage falls out of the current count.
        # --------------------------------------------------------------
        for name in self.get_semantic_milestone_spec().keys():
            if (
                self._lh_prev_ordered_state[name]
                and not ordered_state[name]
            ):
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

        self._lh_prev_ordered_state = {
            name: bool(ordered_state[name])
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
        ops = self._get_current_ordered_progress()
        ups = self._get_unordered_progress()

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
        buns_in_plate1 = 0
        sausages_in_plate1 = 0

        for bun_name in [
            "hotdog_bun1",
            "hotdog_bun2",
        ]:
            if OU.check_obj_in_receptacle(
                self,
                bun_name,
                "plate1",
            ):
                buns_in_plate1 += 1

        for sausage_name in [
            "sausage1",
            "sausage2",
        ]:
            if OU.check_obj_in_receptacle(
                self,
                sausage_name,
                "plate1",
            ):
                sausages_in_plate1 += 1

        buns_in_plate2 = 0
        sausages_in_plate2 = 0

        for bun_name in [
            "hotdog_bun1",
            "hotdog_bun2",
        ]:
            if OU.check_obj_in_receptacle(
                self,
                bun_name,
                "plate2",
            ):
                buns_in_plate2 += 1

        for sausage_name in [
            "sausage1",
            "sausage2",
        ]:
            if OU.check_obj_in_receptacle(
                self,
                sausage_name,
                "plate2",
            ):
                sausages_in_plate2 += 1

        gripper_far = True

        for obj_name in [
            "hotdog_bun1",
            "hotdog_bun2",
            "sausage1",
            "sausage2",
        ]:
            if not OU.gripper_obj_far(
                self,
                obj_name,
            ):
                gripper_far = False
                break

        return (
            buns_in_plate1 == 1
            and sausages_in_plate1 == 1
            and buns_in_plate2 == 1
            and sausages_in_plate2 == 1
            and gripper_far
        )

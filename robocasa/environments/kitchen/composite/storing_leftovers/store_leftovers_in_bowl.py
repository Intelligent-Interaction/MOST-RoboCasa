from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.object_utils as OU


class StoreLeftoversInBowl(Kitchen):
    """
    Store Leftovers in Bowl: composite task for Storing Leftovers activity.

    Task:
        1. Pick one food item.
        2. Place it in the bowl.
        3. Pick the other food item.
        4. Place it in the bowl.
        5. Pick the loaded bowl.
        6. Place the loaded bowl in the fridge.

    The two food pick-place branches are order-free:
        chicken_in_hand  -> chicken_in_bowl
        vegetable_in_hand -> vegetable_in_bowl

    Then the loaded bowl branch is ordered:
        bowl_in_hand -> bowl_in_fridge

    Important:
        After food is placed in the bowl, picking the bowl can make the food
        appear close to / grasped by the gripper. The metric keeps already
        credited food-in-bowl progress valid during loaded-bowl carry as long
        as the food remains physically inside the bowl.

    Original RoboCasa _check_success() is preserved.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        self._reset_long_horizon_metric_tracker(initial_state=None)
        super().__init__(*args, **kwargs)

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

        self.fridge = self.register_fixture_ref(
            "fridge",
            dict(id=FixtureType.FRIDGE),
        )

        self.init_robot_base_ref = self.dining_counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        vegetable_lang = self.get_obj_lang("vegetable")

        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(
                self.novel_instructions
            ).format(
                vegetable_lang=vegetable_lang
            )
        else:
            ep_meta["lang"] = (
                f"Pick the chicken drumstick and {vegetable_lang} from their plates "
                f"and place them in the bowl. Then put the bowl in the fridge."
            )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="chicken_drumstick",
                obj_groups="chicken_drumstick",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool1,
                    ),
                    size=(0.4, 0.4),
                    pos=("ref", "ref"),
                    try_to_place_in="plate",
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool2,
                    ),
                    size=(0.4, 0.4),
                    pos=("ref", "ref"),
                    try_to_place_in="plate",
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                init_robot_here=True,
                graspable=True,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool2,
                    ),
                    size=(1.0, 0.4),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr1",
                fridgable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.3),
                    pos=(-0.3, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr2",
                fridgable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.3),
                    pos=(0.3, 1.0),
                ),
            )
        )

        return cfgs

    # =====================================================================
    # Task-specific semantic milestone definitions
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "chicken_in_hand": {
                "description": "The chicken drumstick is grasped.",
                "dependencies": [],
                "persistent": False,
            },
            "chicken_in_bowl": {
                "description": "The chicken drumstick is placed inside the bowl.",
                "dependencies": ["chicken_in_hand"],
                "persistent": True,
            },
            "vegetable_in_hand": {
                "description": "The vegetable is grasped.",
                "dependencies": [],
                "persistent": False,
            },
            "vegetable_in_bowl": {
                "description": "The vegetable is placed inside the bowl.",
                "dependencies": ["vegetable_in_hand"],
                "persistent": True,
            },
            "bowl_in_hand": {
                "description": "The loaded bowl is grasped.",
                "dependencies": [
                    "chicken_in_bowl",
                    "vegetable_in_bowl",
                ],
                "persistent": False,
            },
            "bowl_in_fridge": {
                "description": "The loaded bowl is placed inside the fridge.",
                "dependencies": ["bowl_in_hand"],
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

    def _check_obj_in_receptacle_xy(
        self,
        obj_name,
        receptacle_name,
        th=None,
    ):
        """
        XY containment fallback used only after food has already been validly
        placed in the bowl.

        During loaded-bowl carry, contact-based check_obj_in_receptacle can
        flicker, and the carried food can be falsely detected as grasped.
        This fallback keeps already completed food-in-bowl progress stable
        as long as the food remains physically inside the bowl.
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
        chicken_in_hand = self._check_task_obj_grasped(
            "chicken_drumstick",
        )
        vegetable_in_hand = self._check_task_obj_grasped(
            "vegetable",
        )
        bowl_in_hand = self._check_task_obj_grasped(
            "bowl",
        )

        chicken_in_bowl_contact = bool(
            OU.check_obj_in_receptacle(
                self,
                "chicken_drumstick",
                "bowl",
            )
        )
        vegetable_in_bowl_contact = bool(
            OU.check_obj_in_receptacle(
                self,
                "vegetable",
                "bowl",
            )
        )

        chicken_in_bowl_released = bool(
            chicken_in_bowl_contact
            and not chicken_in_hand
        )
        vegetable_in_bowl_released = bool(
            vegetable_in_bowl_contact
            and not vegetable_in_hand
        )

        chicken_already_placed = bool(
            getattr(
                self,
                "_lh_ordered_ever",
                {},
            ).get(
                "chicken_in_bowl",
                False,
            )
        )
        vegetable_already_placed = bool(
            getattr(
                self,
                "_lh_ordered_ever",
                {},
            ).get(
                "vegetable_in_bowl",
                False,
            )
        )

        chicken_in_bowl_xy = bool(
            self._check_obj_in_receptacle_xy(
                "chicken_drumstick",
                "bowl",
            )
        )
        vegetable_in_bowl_xy = bool(
            self._check_obj_in_receptacle_xy(
                "vegetable",
                "bowl",
            )
        )

        chicken_in_bowl_for_progress = bool(
            chicken_in_bowl_released
            or (
                chicken_already_placed
                and (
                    chicken_in_bowl_contact
                    or chicken_in_bowl_xy
                )
            )
        )
        vegetable_in_bowl_for_progress = bool(
            vegetable_in_bowl_released
            or (
                vegetable_already_placed
                and (
                    vegetable_in_bowl_contact
                    or vegetable_in_bowl_xy
                )
            )
        )

        bowl_on_fridge_rack = bool(
            self.fridge.check_rack_contact(
                self,
                "bowl",
            )
        )
        bowl_far = bool(
            OU.gripper_obj_far(
                self,
                "bowl",
            )
        )
        bowl_in_fridge_released = bool(
            bowl_on_fridge_rack
            and bowl_far
        )

        return {
            "chicken_in_hand": chicken_in_hand,
            "vegetable_in_hand": vegetable_in_hand,
            "bowl_in_hand": bowl_in_hand,
            "chicken_in_bowl_contact": chicken_in_bowl_contact,
            "vegetable_in_bowl_contact": vegetable_in_bowl_contact,
            "chicken_in_bowl_xy": chicken_in_bowl_xy,
            "vegetable_in_bowl_xy": vegetable_in_bowl_xy,
            "chicken_in_bowl_for_progress": chicken_in_bowl_for_progress,
            "vegetable_in_bowl_for_progress": vegetable_in_bowl_for_progress,
            "foods_ready_for_progress": bool(
                chicken_in_bowl_for_progress
                and vegetable_in_bowl_for_progress
            ),
            "bowl_on_fridge_rack": bowl_on_fridge_rack,
            "bowl_in_fridge_released": bowl_in_fridge_released,
        }

    def _get_raw_semantic_milestone_state(self):
        scene = self._get_raw_scene_state()

        return {
            "chicken_in_hand": bool(
                scene["chicken_in_hand"]
            ),
            "chicken_in_bowl": bool(
                scene["chicken_in_bowl_for_progress"]
            ),
            "vegetable_in_hand": bool(
                scene["vegetable_in_hand"]
            ),
            "vegetable_in_bowl": bool(
                scene["vegetable_in_bowl_for_progress"]
            ),
            "bowl_in_hand": bool(
                scene["bowl_in_hand"]
            ),
            "bowl_in_fridge": bool(
                scene["bowl_in_fridge_released"]
            ),
        }

    def _get_ordered_semantic_milestone_state(self):
        """
        Current ordered-valid semantic states.

        The two food pick-place branches are independent. Bowl pick/place only
        becomes valid after both foods remain validly inside the bowl.

        Food-in-bowl uses an anti-jitter containment check after initial
        placement, so carrying the loaded bowl does not make food progress drop
        just because the food is close to the gripper.
        """
        raw = self._get_raw_semantic_milestone_state()
        scene = self._get_raw_scene_state()

        chicken_place_valid = bool(
            self._lh_ordered_ever["chicken_in_bowl"]
            and scene["chicken_in_bowl_for_progress"]
        )
        vegetable_place_valid = bool(
            self._lh_ordered_ever["vegetable_in_bowl"]
            and scene["vegetable_in_bowl_for_progress"]
        )

        chicken_pick_valid = bool(
            self._lh_ordered_ever["chicken_in_hand"]
            and (
                raw["chicken_in_hand"]
                or chicken_place_valid
            )
        )
        vegetable_pick_valid = bool(
            self._lh_ordered_ever["vegetable_in_hand"]
            and (
                raw["vegetable_in_hand"]
                or vegetable_place_valid
            )
        )

        foods_ready = bool(
            chicken_place_valid
            and vegetable_place_valid
        )

        bowl_place_valid = bool(
            self._lh_ordered_ever["bowl_in_fridge"]
            and scene["bowl_in_fridge_released"]
            and foods_ready
        )

        bowl_pick_valid = bool(
            self._lh_ordered_ever["bowl_in_hand"]
            and foods_ready
            and (
                raw["bowl_in_hand"]
                or bowl_place_valid
            )
        )

        return {
            "chicken_in_hand": chicken_pick_valid,
            "chicken_in_bowl": chicken_place_valid,
            "vegetable_in_hand": vegetable_pick_valid,
            "vegetable_in_bowl": vegetable_place_valid,
            "bowl_in_hand": bowl_pick_valid,
            "bowl_in_fridge": bowl_place_valid,
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

        self._lh_ops_auc_sum = 0.0
        self._lh_peak_progress = 0.0
        self._lh_first_success_step = None
        self._lh_progress_curve = []

    def _reset_internal(self):
        """
        Normal RoboCasa reset plus metric initialization.
        """
        super()._reset_internal()

        initial_raw = self._get_raw_semantic_milestone_state()

        self._reset_long_horizon_metric_tracker(
            initial_state=initial_raw
        )

    def _get_current_ordered_progress(
        self,
        ordered_state=None,
    ):
        if ordered_state is None:
            ordered_state = self._get_ordered_semantic_milestone_state()

        return (
            int(bool(ordered_state["chicken_in_hand"]))
            + int(bool(ordered_state["chicken_in_bowl"]))
            + int(bool(ordered_state["vegetable_in_hand"]))
            + int(bool(ordered_state["vegetable_in_bowl"]))
            + int(bool(ordered_state["bowl_in_hand"]))
            + int(bool(ordered_state["bowl_in_fridge"]))
        ) / 6.0

    def _update_long_horizon_metric_tracker(self):
        """
        Update all long-horizon metrics once per policy timestep.

        Valid ordered executions include:
            chicken -> bowl, vegetable -> bowl, bowl -> fridge
            vegetable -> bowl, chicken -> bowl, bowl -> fridge
        """
        self._lh_step_count += 1

        spec = self.get_semantic_milestone_spec()
        raw = self._get_raw_semantic_milestone_state()
        scene = self._get_raw_scene_state()

        # --------------------------------------------------------------
        # UPS: order ignored.
        # --------------------------------------------------------------
        for name in spec.keys():
            if raw[name]:
                self._lh_unordered_ever[name] = True

        # --------------------------------------------------------------
        # Chicken branch.
        # --------------------------------------------------------------
        if raw["chicken_in_hand"]:
            self._lh_ordered_ever["chicken_in_hand"] = True

        if raw["chicken_in_bowl"]:
            if not self._lh_ordered_ever["chicken_in_hand"]:
                self._lh_ordered_ever["chicken_in_hand"] = True
            self._lh_ordered_ever["chicken_in_bowl"] = True

        # --------------------------------------------------------------
        # Vegetable branch.
        # --------------------------------------------------------------
        if raw["vegetable_in_hand"]:
            self._lh_ordered_ever["vegetable_in_hand"] = True

        if raw["vegetable_in_bowl"]:
            if not self._lh_ordered_ever["vegetable_in_hand"]:
                self._lh_ordered_ever["vegetable_in_hand"] = True
            self._lh_ordered_ever["vegetable_in_bowl"] = True

        foods_ready = bool(
            scene["foods_ready_for_progress"]
            and self._lh_ordered_ever["chicken_in_bowl"]
            and self._lh_ordered_ever["vegetable_in_bowl"]
        )

        # --------------------------------------------------------------
        # Pick loaded bowl after both foods are in the bowl.
        # --------------------------------------------------------------
        if (
            raw["bowl_in_hand"]
            and foods_ready
        ):
            self._lh_ordered_ever["bowl_in_hand"] = True

        # --------------------------------------------------------------
        # Place loaded bowl in fridge.
        #
        # If bowl grasp detection flickered and was missed, infer bowl pick
        # from a valid fridge placement after both foods were already ready.
        # --------------------------------------------------------------
        if (
            raw["bowl_in_fridge"]
            and foods_ready
        ):
            if not self._lh_ordered_ever["bowl_in_hand"]:
                self._lh_ordered_ever["bowl_in_hand"] = True
            self._lh_ordered_ever["bowl_in_fridge"] = True

        ordered_state = (
            self._get_ordered_semantic_milestone_state()
        )

        # --------------------------------------------------------------
        # Regression bookkeeping for persistent final-state requirements.
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

        current_ops = (
            self._get_current_ordered_progress(
                ordered_state
            )
        )
        self._lh_progress_curve.append(float(current_ops))

        self._lh_ops_auc_sum += current_ops

        self._lh_peak_progress = max(
            self._lh_peak_progress,
            current_ops,
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
        ops = self._get_current_ordered_progress(ordered_state)

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
        chicken_in_bowl = OU.check_obj_in_receptacle(
            self,
            "chicken_drumstick",
            "bowl",
        )

        vegetable_in_bowl = OU.check_obj_in_receptacle(
            self,
            "vegetable",
            "bowl",
        )

        bowl_in_fridge = self.fridge.check_rack_contact(
            self,
            "bowl",
        )

        gripper_far = OU.gripper_obj_far(
            self,
            "bowl",
        )

        return (
            chicken_in_bowl
            and vegetable_in_bowl
            and bowl_in_fridge
            and gripper_far
        )

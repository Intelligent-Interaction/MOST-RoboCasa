from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.object_utils as OU


class SteamInMicrowave(Kitchen):
    """
    Steam In Microwave: composite task for Steaming Food activity.

    Task:
        1. Pick the food from the sink.
        2. Place the food in the bowl.
        3. Pick the loaded bowl.
        4. Place the loaded bowl in the microwave.
        5. Close the microwave door.
        6. Start the microwave.

    Long-horizon semantic milestones:
        M1. vegetable_in_hand
        M2. vegetable_in_bowl
        M3. bowl_in_hand
        M4. bowl_in_microwave
        M5. close_microwave_door
        M6. start_microwave

    Dependency chain:
        vegetable_in_hand
            -> vegetable_in_bowl
                -> bowl_in_hand
                    -> bowl_in_microwave
                        -> close_microwave_door
                            -> start_microwave

    Pick milestones are current/superseded:
        - valid while the object is currently held;
        - also valid after the corresponding placement succeeds.

    Original RoboCasa _check_success() is preserved.
    """

    def __init__(self, *args, **kwargs):
        self._reset_long_horizon_metric_tracker(initial_state=None)
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref(
            "sink",
            dict(id=FixtureType.SINK),
        )
        self.microwave = self.register_fixture_ref(
            "microwave",
            dict(id=FixtureType.MICROWAVE),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.sink),
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        vegetable_name = self.get_obj_lang("vegetable")

        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions).format(
                vegetable_name=vegetable_name
            )
        else:
            ep_meta["lang"] = (
                f"Pick the {vegetable_name} from the sink and place it in the bowl. "
                "Then pick the bowl and place it in the microwave. "
                "Then close the microwave door and press the start button."
            )

        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.sink.set_handle_state(mode="off", env=self, rng=self.rng)
        self.microwave.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.35, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable",
                obj_groups="vegetable",
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.3, 0.2),
                    pos=(0.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_counter_0",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(0.50, 0.30),
                    pos=("ref", -1.0),
                    offset=(0.0, 0.30),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_counter_1",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                    size=(0.50, 0.50),
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
            "vegetable_in_hand": {
                "description": "The vegetable is grasped by the robot gripper.",
                "dependencies": [],
                "persistent": False,
            },
            "vegetable_in_bowl": {
                "description": "The vegetable is placed inside the bowl.",
                "dependencies": ["vegetable_in_hand"],
                "persistent": True,
            },
            "bowl_in_hand": {
                "description": "The loaded bowl is grasped by the robot gripper.",
                "dependencies": ["vegetable_in_bowl"],
                "persistent": False,
            },
            "bowl_in_microwave": {
                "description": "The loaded bowl is placed inside the microwave.",
                "dependencies": ["bowl_in_hand"],
                "persistent": True,
            },
            "close_microwave_door": {
                "description": "The microwave door is closed.",
                "dependencies": ["bowl_in_microwave"],
                "persistent": True,
            },
            "start_microwave": {
                "description": "The microwave is started.",
                "dependencies": ["close_microwave_door"],
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
        XY containment fallback used only after the food has already been
        validly placed in the bowl.

        This prevents progress from flickering when the robot picks the bowl
        and the food remains physically inside but contact / grasp labels become
        noisy.
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
        vegetable_in_hand = self._check_task_obj_grasped(
            "vegetable",
        )
        bowl_in_hand = self._check_task_obj_grasped(
            "bowl",
        )

        vegetable_in_bowl_contact = bool(
            OU.check_obj_in_receptacle(
                self,
                "vegetable",
                "bowl",
            )
        )
        vegetable_in_bowl_xy = bool(
            self._check_obj_in_receptacle_xy(
                "vegetable",
                "bowl",
            )
        )

        # Initial food placement still requires a released, contact-supported
        # bowl containment state. This avoids counting "hovering above bowl" as
        # a completed place.
        vegetable_in_bowl_released = bool(
            vegetable_in_bowl_contact
            and not vegetable_in_hand
        )

        # After food placement has already been credited, keep it valid while
        # the loaded bowl is carried if the food is still physically inside the
        # bowl. This specifically handles the instability where picking the bowl
        # also makes the vegetable appear grasped.
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

        vegetable_in_bowl_physical = bool(
            vegetable_in_bowl_contact
            or (
                vegetable_already_placed
                and vegetable_in_bowl_xy
            )
        )

        vegetable_in_bowl_for_progress = bool(
            vegetable_in_bowl_released
            or (
                vegetable_already_placed
                and vegetable_in_bowl_physical
            )
        )

        bowl_in_microwave = bool(
            OU.obj_inside_of(
                self,
                "bowl",
                self.microwave,
            )
            and not bowl_in_hand
        )

        close_microwave_door = bool(
            self.microwave.is_closed(env=self)
        )

        start_microwave = bool(
            self.microwave.get_state()["turned_on"]
        )

        return {
            "vegetable_in_hand": vegetable_in_hand,
            "bowl_in_hand": bowl_in_hand,
            "vegetable_in_bowl_contact": vegetable_in_bowl_contact,
            "vegetable_in_bowl_xy": vegetable_in_bowl_xy,
            "vegetable_in_bowl_released": vegetable_in_bowl_released,
            "vegetable_in_bowl_physical": vegetable_in_bowl_physical,
            "vegetable_in_bowl_for_progress": vegetable_in_bowl_for_progress,
            "bowl_in_microwave": bowl_in_microwave,
            "close_microwave_door": close_microwave_door,
            "start_microwave": start_microwave,
        }

    def _get_raw_semantic_milestone_state(self):
        scene = self._get_raw_scene_state()

        return {
            "vegetable_in_hand": bool(
                scene["vegetable_in_hand"]
            ),
            "vegetable_in_bowl": bool(
                scene["vegetable_in_bowl_for_progress"]
            ),
            "bowl_in_hand": bool(
                scene["bowl_in_hand"]
            ),
            "bowl_in_microwave": bool(
                scene["bowl_in_microwave"]
            ),
            "close_microwave_door": bool(
                scene["close_microwave_door"]
            ),
            "start_microwave": bool(
                scene["start_microwave"]
            ),
        }

    def _get_ordered_semantic_milestone_state(self):
        """
        Current ordered-valid milestone states.

        Placement milestones are current-state milestones and can regress if
        the object leaves the required location.

        vegetable_in_bowl remains valid during loaded-bowl carry. Otherwise,
        picking the bowl can falsely look like picking the food and make the
        already completed food-place stage flicker off.

        Door/start milestones stay valid only while all previous physical
        requirements remain valid.
        """
        raw = self._get_raw_semantic_milestone_state()

        vegetable_place_valid = bool(
            self._lh_ordered_ever["vegetable_in_bowl"]
            and raw["vegetable_in_bowl"]
        )

        vegetable_pick_valid = bool(
            self._lh_ordered_ever["vegetable_in_hand"]
            and (
                raw["vegetable_in_hand"]
                or vegetable_place_valid
            )
        )

        bowl_place_valid = bool(
            self._lh_ordered_ever["bowl_in_microwave"]
            and raw["bowl_in_microwave"]
            and vegetable_place_valid
        )

        bowl_pick_valid = bool(
            self._lh_ordered_ever["bowl_in_hand"]
            and vegetable_place_valid
            and (
                raw["bowl_in_hand"]
                or bowl_place_valid
            )
        )

        door_valid = bool(
            self._lh_ordered_ever["close_microwave_door"]
            and raw["close_microwave_door"]
            and bowl_place_valid
        )

        start_valid = bool(
            self._lh_ordered_ever["start_microwave"]
            and raw["start_microwave"]
            and door_valid
        )

        return {
            "vegetable_in_hand": vegetable_pick_valid,
            "vegetable_in_bowl": vegetable_place_valid,
            "bowl_in_hand": bowl_pick_valid,
            "bowl_in_microwave": bowl_place_valid,
            "close_microwave_door": door_valid,
            "start_microwave": start_valid,
        }

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

    def _get_current_ordered_progress(self, ordered_state=None):
        if ordered_state is None:
            ordered_state = self._get_ordered_semantic_milestone_state()

        return (
            int(bool(ordered_state["vegetable_in_hand"]))
            + int(bool(ordered_state["vegetable_in_bowl"]))
            + int(bool(ordered_state["bowl_in_hand"]))
            + int(bool(ordered_state["bowl_in_microwave"]))
            + int(bool(ordered_state["close_microwave_door"]))
            + int(bool(ordered_state["start_microwave"]))
        ) / 6.0

    def _update_long_horizon_metric_tracker(self):
        """
        Update all long-horizon metrics once per policy timestep.

        OPS enforces:
            pick vegetable
            -> place vegetable in bowl
            -> pick loaded bowl
            -> place bowl in microwave
            -> close door
            -> start microwave
        """
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
        # M1: pick vegetable.
        # --------------------------------------------------------------
        if raw["vegetable_in_hand"]:
            self._lh_ordered_ever["vegetable_in_hand"] = True

        # --------------------------------------------------------------
        # M2: place vegetable in bowl.
        #
        # If vegetable grasp detection flickered and was missed, infer pick
        # from a valid vegetable-in-bowl state.
        # --------------------------------------------------------------
        if raw["vegetable_in_bowl"]:
            if not self._lh_ordered_ever["vegetable_in_hand"]:
                self._lh_ordered_ever["vegetable_in_hand"] = True
            self._lh_ordered_ever["vegetable_in_bowl"] = True

        # --------------------------------------------------------------
        # M3: pick loaded bowl.
        # --------------------------------------------------------------
        loaded_bowl_ready = bool(
            raw["vegetable_in_bowl"]
            and self._lh_ordered_ever["vegetable_in_bowl"]
        )

        if (
            raw["bowl_in_hand"]
            and loaded_bowl_ready
        ):
            self._lh_ordered_ever["bowl_in_hand"] = True

        # --------------------------------------------------------------
        # M4: place loaded bowl in microwave.
        #
        # If bowl grasp detection flickered and was missed, infer pick from a
        # valid loaded-bowl-in-microwave state.
        # --------------------------------------------------------------
        if (
            raw["bowl_in_microwave"]
            and loaded_bowl_ready
        ):
            if not self._lh_ordered_ever["bowl_in_hand"]:
                self._lh_ordered_ever["bowl_in_hand"] = True
            self._lh_ordered_ever["bowl_in_microwave"] = True

        # --------------------------------------------------------------
        # M5: close microwave door after the loaded bowl is already inside.
        #
        # Use rising edge so a door closed too early does not receive ordered
        # credit retroactively.
        # --------------------------------------------------------------
        bowl_in_microwave_ready = bool(
            raw["bowl_in_microwave"]
            and loaded_bowl_ready
            and self._lh_ordered_ever["bowl_in_microwave"]
        )

        door_rising_edge = bool(
            raw["close_microwave_door"]
            and not self._lh_prev_raw["close_microwave_door"]
        )

        if (
            door_rising_edge
            and bowl_in_microwave_ready
        ):
            self._lh_ordered_ever["close_microwave_door"] = True

        # --------------------------------------------------------------
        # M6: start microwave after the door is closed.
        #
        # Use rising edge so early start does not receive retroactive credit.
        # --------------------------------------------------------------
        door_ready = bool(
            raw["close_microwave_door"]
            and bowl_in_microwave_ready
            and self._lh_ordered_ever["close_microwave_door"]
        )

        start_rising_edge = bool(
            raw["start_microwave"]
            and not self._lh_prev_raw["start_microwave"]
        )

        if (
            start_rising_edge
            and door_ready
        ):
            self._lh_ordered_ever["start_microwave"] = True

        ordered_state = self._get_ordered_semantic_milestone_state()

        # --------------------------------------------------------------
        # Regression bookkeeping for persistent placement / door / start
        # milestones.
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

        current_ops = self._get_current_ordered_progress(
            ordered_state
        )
        self._lh_progress_curve.append(float(current_ops))

        self._lh_ops_auc_sum += current_ops
        self._lh_peak_progress = max(
            self._lh_peak_progress,
            current_ops,
        )

        # Official success and first completion time.
        success = bool(self._check_success())

        if (
            success
            and self._lh_first_success_step is None
        ):
            self._lh_first_success_step = self._lh_step_count

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
        ordered_state = self._get_ordered_semantic_milestone_state()
        ops = self._get_current_ordered_progress(
            ordered_state
        )

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
        vegetable_in_bowl = OU.check_obj_in_receptacle(
            self,
            "vegetable",
            "bowl",
        )

        bowl_in_microwave = OU.obj_inside_of(
            self,
            "bowl",
            self.microwave,
        )

        door_closed = self.microwave.is_closed(env=self)
        button_pressed = self.microwave.get_state()["turned_on"]

        return (
            vegetable_in_bowl
            and bowl_in_microwave
            and door_closed
            and button_pressed
        )

from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.object_utils as OU


class SearingMeat(Kitchen):
    """
    Searing Meat: composite task for Frying activity.

    Task:
        1. Pick the pan from the cabinet.
        2. Place the pan on the target stove burner.
        3. Pick the meat from the counter.
        4. Place the meat in the pan.
        5. Turn on the target burner.

    Long-horizon semantic milestones:
        M1. pan_in_hand
        M2. pan_on_target_burner
        M3. meat_in_hand
        M4. meat_in_pan
        M5. turn_target_burner_on

    Dependency chain:
        pan_in_hand
            -> pan_on_target_burner
                -> meat_in_hand
                    -> meat_in_pan
                        -> turn_target_burner_on
    """

    EXCLUDE_LAYOUTS = Kitchen.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def __init__(self, knob_id="random", *args, **kwargs):
        self.knob_id = knob_id
        self._reset_long_horizon_metric_tracker()
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.stove = self.register_fixture_ref(
            "stove",
            dict(id=FixtureType.STOVE),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.stove, size=[0.30, 0.40]),
        )

        if "refs" in self._ep_meta:
            self.knob = self._ep_meta["refs"]["knob"]
        else:
            valid_knobs = [
                k
                for (k, v) in self.stove.knob_joints.items()
                if v is not None
            ]
            if self.knob_id == "random":
                self.knob = self.rng.choice(
                    list(valid_knobs)
                )
            else:
                assert self.knob_id in valid_knobs
                self.knob = self.knob_id

        self.cab = self.register_fixture_ref(
            "cab",
            dict(id=FixtureType.CABINET_DOUBLE_DOOR, ref=self.stove),
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        meat_name = self.get_obj_lang("meat")

        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions).format(
                meat_name=meat_name,
                knob=self.knob.replace("_", " "),
            )
        else:
            ep_meta["lang"] = (
                f"Grab the pan from the cabinet and place it on the "
                f"{self.knob.replace('_', ' ')} burner on the stove. "
                f"Then place the {meat_name} in the pan and turn on the burner."
            )

        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["knob"] = self.knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

        self.stove.set_knob_state(
            mode="off",
            knob=self.knob,
            env=self,
            rng=self.rng,
        )
        self.cab.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="pan",
                obj_groups=("pan"),
                placement=dict(
                    fixture=self.cab,
                    pos=(0.0, -1.0),
                    size=(0.8, 0.5),
                    rotation=(np.pi / 2 - np.pi / 16, np.pi / 2),
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    try_to_place_in="container",
                ),
            )
        )

        return cfgs

    # =====================================================================
    # Task-specific semantic milestones
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "pan_in_hand": {
                "dependencies": [],
                "persistent": False,
            },
            "pan_on_target_burner": {
                "dependencies": ["pan_in_hand"],
                "persistent": True,
            },
            "meat_in_hand": {
                "dependencies": ["pan_on_target_burner"],
                "persistent": False,
            },
            "meat_in_pan": {
                "dependencies": ["meat_in_hand"],
                "persistent": True,
            },
            "turn_target_burner_on": {
                "dependencies": [
                    "pan_on_target_burner",
                    "meat_in_pan",
                ],
                "persistent": True,
            },
        }

    def _check_task_obj_grasped(
        self,
        obj_name,
    ):
        """
        Local grasp check with a looser gripper-close threshold.

        This avoids changing global RoboCasa object_utils behavior. For pan and
        meat, direct contact can flicker during replay, so placement milestones
        can also infer the corresponding pick stage in the tracker.
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

    def _is_pan_on_target_burner(
        self,
        threshold=0.15,
    ):
        """
        Metric-only pan placement check.

        Do NOT use stove.check_obj_location_on_stove() here, because in some
        RoboCasa versions / helpers that location check is coupled with burner
        state. That caused pan_on_target_burner to stay False until the burner
        was turned on, so progress jumped directly from 0 to 1.

        For progress, pan placement must be independent of whether the burner
        is on:
            pan touches stove
            AND pan XY is close to the target burner site.
        """
        try:
            obj = self.objects["pan"]
            obj_pos = np.array(
                self.sim.data.body_xpos[self.obj_body_id[obj.name]]
            )[0:2]
        except Exception:
            obj_pos = np.array(
                self.sim.data.body_xpos[self.obj_body_id["pan"]]
            )[0:2]

        obj_on_stove = bool(
            OU.check_obj_fixture_contact(
                self,
                "pan",
                self.stove,
            )
        )

        if not obj_on_stove:
            return False

        site = self.stove.burner_sites.get(
            self.knob,
            None,
        )
        if site is None:
            return False

        burner_pos = np.array(
            self.sim.data.get_site_xpos(
                site.get("name")
            )
        )[0:2]

        dist = np.linalg.norm(
            burner_pos - obj_pos
        )

        return bool(
            dist < threshold
        )

    def _is_target_burner_on(self):
        """
        True when the task-specified burner is on.

        This function was missing in the previous file even though the metric
        code called _is_target_burner_on(), which caused runtime errors.
        """
        try:
            return bool(
                self.stove.is_burner_on(
                    env=self,
                    burner_loc=self.knob,
                )
            )
        except TypeError:
            return bool(
                self.stove.is_burner_on(
                    self,
                    self.knob,
                )
            )

    def _get_raw_semantic_milestone_state(self):
        pan_in_hand = self._check_task_obj_grasped("pan")
        meat_in_hand = self._check_task_obj_grasped("meat")

        pan_on_target_burner = bool(
            self._is_pan_on_target_burner(
                threshold=0.15,
            )
            and not pan_in_hand
        )

        meat_in_pan = bool(
            OU.check_obj_in_receptacle(
                self,
                "meat",
                "pan",
                th=0.07,
            )
            and not meat_in_hand
        )

        target_burner_on = bool(
            self._is_target_burner_on()
        )

        return {
            "pan_in_hand": pan_in_hand,
            "pan_on_target_burner": pan_on_target_burner,
            "meat_in_hand": meat_in_hand,
            "meat_in_pan": meat_in_pan,
            "turn_target_burner_on": target_burner_on,
        }

    def _get_ordered_semantic_milestone_state(self):
        """
        Current ordered-valid state.

        Pick milestones are current/superseded:
            - valid while the object is in hand;
            - also valid after the corresponding place stage succeeds.

        If the object is picked and dropped before being correctly placed, its
        pick contribution disappears from current OPS.

        Burner-on stays valid only while pan and meat remain correctly arranged.
        """
        raw = self._get_raw_semantic_milestone_state()

        pan_place_valid = bool(
            self._lh_ordered_ever["pan_on_target_burner"]
            and raw["pan_on_target_burner"]
        )

        pan_pick_valid = bool(
            self._lh_ordered_ever["pan_in_hand"]
            and (
                raw["pan_in_hand"]
                or pan_place_valid
            )
        )

        meat_place_valid = bool(
            self._lh_ordered_ever["meat_in_pan"]
            and raw["meat_in_pan"]
            and pan_place_valid
        )

        meat_pick_valid = bool(
            self._lh_ordered_ever["meat_in_hand"]
            and pan_place_valid
            and (
                raw["meat_in_hand"]
                or meat_place_valid
            )
        )

        burner_on_valid = bool(
            self._lh_ordered_ever["turn_target_burner_on"]
            and raw["turn_target_burner_on"]
            and pan_place_valid
            and meat_place_valid
        )

        return {
            "pan_in_hand": pan_pick_valid,
            "pan_on_target_burner": pan_place_valid,
            "meat_in_hand": meat_pick_valid,
            "meat_in_pan": meat_place_valid,
            "turn_target_burner_on": burner_on_valid,
        }

    # =====================================================================
    # Long-horizon metric tracker
    # =====================================================================

    def _reset_long_horizon_metric_tracker(self):
        names = list(
            self.get_semantic_milestone_spec().keys()
        )

        self._lh_step_count = 0
        self._lh_prev_raw = {
            name: False
            for name in names
        }
        self._lh_ordered_ever = {
            name: False
            for name in names
        }
        self._lh_unordered_ever = {
            name: False
            for name in names
        }
        self._lh_progress_curve = []
        self._lh_first_success_step = None

    def _reset_internal(self):
        super()._reset_internal()
        self._reset_long_horizon_metric_tracker()

    def _get_current_ordered_progress(self):
        ordered_state = self._get_ordered_semantic_milestone_state()
        return (
            sum(
                int(bool(value))
                for value in ordered_state.values()
            )
            / 5.0
        )

    def _update_long_horizon_metric_tracker(self):
        self._lh_step_count += 1

        raw = self._get_raw_semantic_milestone_state()

        # UPS: order ignored, event/history based.
        for name, value in raw.items():
            if value:
                self._lh_unordered_ever[name] = True

        # --------------------------------------------------------------
        # M1: pick pan.
        # --------------------------------------------------------------
        if raw["pan_in_hand"]:
            self._lh_ordered_ever["pan_in_hand"] = True

        # --------------------------------------------------------------
        # M2: place pan on the target burner.
        #
        # If pan grasp detection flickered and was missed, infer pick pan here.
        # Since the pan starts in the cabinet, a valid pan-on-burner state means
        # the pan must have been manipulated.
        # --------------------------------------------------------------
        if raw["pan_on_target_burner"]:
            if not self._lh_ordered_ever["pan_in_hand"]:
                self._lh_ordered_ever["pan_in_hand"] = True
            self._lh_ordered_ever["pan_on_target_burner"] = True

        # --------------------------------------------------------------
        # M3: pick meat after pan placement.
        # --------------------------------------------------------------
        if (
            raw["meat_in_hand"]
            and self._lh_ordered_ever["pan_on_target_burner"]
            and raw["pan_on_target_burner"]
        ):
            self._lh_ordered_ever["meat_in_hand"] = True

        # --------------------------------------------------------------
        # M4: place meat in pan.
        #
        # If meat grasp detection flickered and was missed, infer pick meat
        # from a valid meat-in-pan state after the pan has been placed.
        # --------------------------------------------------------------
        if (
            raw["meat_in_pan"]
            and self._lh_ordered_ever["pan_on_target_burner"]
            and raw["pan_on_target_burner"]
        ):
            if not self._lh_ordered_ever["meat_in_hand"]:
                self._lh_ordered_ever["meat_in_hand"] = True
            self._lh_ordered_ever["meat_in_pan"] = True

        # --------------------------------------------------------------
        # M5: turn target burner on after pan and meat are already valid.
        #
        # Use a rising edge so turning the burner on too early does not receive
        # ordered credit retroactively after the arrangement becomes correct.
        # --------------------------------------------------------------
        burner_rising_edge = bool(
            raw["turn_target_burner_on"]
            and not self._lh_prev_raw["turn_target_burner_on"]
        )

        arrangement_ready = bool(
            raw["pan_on_target_burner"]
            and raw["meat_in_pan"]
            and self._lh_ordered_ever["pan_on_target_burner"]
            and self._lh_ordered_ever["meat_in_pan"]
        )

        if (
            burner_rising_edge
            and arrangement_ready
        ):
            self._lh_ordered_ever["turn_target_burner_on"] = True

        progress = self._get_current_ordered_progress()
        self._lh_progress_curve.append(
            float(progress)
        )

        if (
            self._check_success()
            and self._lh_first_success_step is None
        ):
            self._lh_first_success_step = self._lh_step_count

        self._lh_prev_raw = {
            name: bool(value)
            for name, value in raw.items()
        }

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        self._update_long_horizon_metric_tracker()
        return reward, done, info

    def get_long_horizon_metrics(self):
        names = list(
            self.get_semantic_milestone_spec().keys()
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
            "SR": float(
                self._lh_first_success_step is not None
            ),
            "NCT": float(nct),
            "UPS": (
                sum(self._lh_unordered_ever.values())
                / float(len(names))
            ),
            "OPS": float(
                self._get_current_ordered_progress()
            ),
            "progress_curve": list(
                self._lh_progress_curve
            ),
        }

    # =====================================================================
    # Original RoboCasa success criterion, preserved
    # =====================================================================

    def _check_success(self):
        gripper_obj_far = OU.gripper_obj_far(self, obj_name="meat")
        pan_loc = (
            self.stove.check_obj_location_on_stove(self, "pan", threshold=0.15)
            == self.knob
        )
        meat_in_pan = OU.check_obj_in_receptacle(self, "meat", "pan", th=0.07)
        return gripper_obj_far and pan_loc and meat_in_pan
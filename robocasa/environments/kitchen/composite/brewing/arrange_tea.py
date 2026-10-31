from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.object_utils as OU


class ArrangeTea(Kitchen):
    """
    Arrange Tea: composite task for Brewing activity.

    Task for progress:
        Pick the kettle and place it on the tray.
        Pick the mug and place it on the tray.
        Close the cabinet doors.

    The two pick-place branches are order-free:
        kettle_in_hand -> kettle_on_tray
        mug_in_hand    -> mug_on_tray

    The final cabinet-close milestone is ordered after both mug and kettle are
    validly on the tray.

    Valid executions include:
        kettle -> tray, mug -> tray, close cabinet
        mug -> tray, kettle -> tray, close cabinet

    Original RoboCasa _check_success() is preserved.
    """

    EXCLUDE_LAYOUTS = Kitchen.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        self._reset_long_horizon_metric_tracker(initial_state=None)
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cab = self.register_fixture_ref(
            "cab",
            dict(id=FixtureType.CABINET_DOUBLE_DOOR),
        )

        self.counter = self.register_fixture_ref(
            "counter",
            dict(
                id=FixtureType.COUNTER,
                ref=self.cab,
                size=(0.6, 0.4),
            ),
        )

        self.init_robot_base_ref = self.cab

    def _setup_scene(self):
        super()._setup_scene()
        self.cab.open_door(env=self)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(
                self.novel_instructions
            )
        else:
            ep_meta["lang"] = (
                "Pick the kettle from the counter and place it on the tray. "
                "Then pick the mug from the cabinet and place it on the tray. "
                "Then close the cabinet doors."
            )

        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=("mug"),
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj2",
                obj_groups=("kettle"),
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(0.5, 0.5),
                    pos=("ref", -1.0),
                    sample_region_kwargs=dict(
                        ref=self.cab,
                        top_size=(0.6, 0.4),
                    ),
                    offset=(0.1, 0.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="container",
                obj_groups=("tray"),
                placement=dict(
                    fixture=self.counter,
                    size=(0.7, 0.7),
                    pos=("ref", -0.6),
                    offset=(-0.1, 0.0),
                    sample_region_kwargs=dict(
                        ref=self.cab,
                        top_size=(0.6, 0.4),
                    ),
                ),
            )
        )

        return cfgs

    # =====================================================================
    # Task-specific semantic milestones
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "kettle_in_hand": {
                "description": "The kettle is grasped by the robot gripper.",
                "dependencies": [],
                "persistent": False,
            },
            "kettle_on_tray": {
                "description": "The kettle is placed on the tray.",
                "dependencies": ["kettle_in_hand"],
                "persistent": True,
            },
            "mug_in_hand": {
                "description": "The mug is grasped by the robot gripper.",
                "dependencies": [],
                "persistent": False,
            },
            "mug_on_tray": {
                "description": "The mug is placed on the tray.",
                "dependencies": ["mug_in_hand"],
                "persistent": True,
            },
            "cabinet_closed": {
                "description": (
                    "The cabinet doors are closed after both kettle and mug "
                    "are validly on the tray."
                ),
                "dependencies": [
                    "kettle_on_tray",
                    "mug_on_tray",
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

    def _get_raw_semantic_milestone_state(self):
        mug_in_hand = self._check_task_obj_grasped(
            "obj",
        )
        kettle_in_hand = self._check_task_obj_grasped(
            "obj2",
        )

        mug_on_tray = bool(
            OU.check_obj_in_receptacle(
                self,
                "obj",
                "container",
            )
            and not mug_in_hand
        )

        kettle_on_tray = bool(
            OU.check_obj_in_receptacle(
                self,
                "obj2",
                "container",
            )
            and not kettle_in_hand
        )

        cabinet_closed = bool(
            self.cab.is_closed(env=self)
        )

        return {
            "kettle_in_hand": kettle_in_hand,
            "kettle_on_tray": kettle_on_tray,
            "mug_in_hand": mug_in_hand,
            "mug_on_tray": mug_on_tray,
            "cabinet_closed": cabinet_closed,
        }

    def _get_ordered_semantic_milestone_state(self):
        """
        Current ordered-valid states.

        The kettle and mug branches are independent. Pick milestones are
        current/superseded:
            - valid while the object is currently held;
            - also valid once its corresponding place stage is valid.

        cabinet_closed is valid only after both placed-object states remain
        valid.
        """
        raw = self._get_raw_semantic_milestone_state()

        kettle_place_valid = bool(
            self._lh_ordered_ever["kettle_on_tray"]
            and raw["kettle_on_tray"]
        )
        mug_place_valid = bool(
            self._lh_ordered_ever["mug_on_tray"]
            and raw["mug_on_tray"]
        )

        kettle_pick_valid = bool(
            self._lh_ordered_ever["kettle_in_hand"]
            and (
                raw["kettle_in_hand"]
                or kettle_place_valid
            )
        )
        mug_pick_valid = bool(
            self._lh_ordered_ever["mug_in_hand"]
            and (
                raw["mug_in_hand"]
                or mug_place_valid
            )
        )

        both_objects_valid = bool(
            kettle_place_valid
            and mug_place_valid
        )

        cabinet_closed_valid = bool(
            self._lh_ordered_ever["cabinet_closed"]
            and raw["cabinet_closed"]
            and both_objects_valid
        )

        return {
            "kettle_in_hand": kettle_pick_valid,
            "kettle_on_tray": kettle_place_valid,
            "mug_in_hand": mug_pick_valid,
            "mug_on_tray": mug_place_valid,
            "cabinet_closed": cabinet_closed_valid,
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
        Normal RoboCasa reset plus metric initialization.
        """
        super()._reset_internal()

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

        return (
            int(bool(ordered_state["kettle_in_hand"]))
            + int(bool(ordered_state["kettle_on_tray"]))
            + int(bool(ordered_state["mug_in_hand"]))
            + int(bool(ordered_state["mug_on_tray"]))
            + int(bool(ordered_state["cabinet_closed"]))
        ) / 5.0

    def _update_long_horizon_metric_tracker(self):
        """
        Update all long-horizon metrics once per policy timestep.

        Valid ordered executions:
            kettle pick/place, mug pick/place, close cabinet
            mug pick/place, kettle pick/place, close cabinet
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
        # Kettle branch.
        # --------------------------------------------------------------
        if raw["kettle_in_hand"]:
            self._lh_ordered_ever["kettle_in_hand"] = True

        if raw["kettle_on_tray"]:
            # If kettle grasp detection flickered and was missed, infer the
            # pick from a valid tray placement.
            if not self._lh_ordered_ever["kettle_in_hand"]:
                self._lh_ordered_ever["kettle_in_hand"] = True
            self._lh_ordered_ever["kettle_on_tray"] = True

        # --------------------------------------------------------------
        # Mug branch.
        # --------------------------------------------------------------
        if raw["mug_in_hand"]:
            self._lh_ordered_ever["mug_in_hand"] = True

        if raw["mug_on_tray"]:
            # Same anti-flicker inference for mug.
            if not self._lh_ordered_ever["mug_in_hand"]:
                self._lh_ordered_ever["mug_in_hand"] = True
            self._lh_ordered_ever["mug_on_tray"] = True

        both_objects_ready = bool(
            raw["kettle_on_tray"]
            and raw["mug_on_tray"]
            and self._lh_ordered_ever["kettle_on_tray"]
            and self._lh_ordered_ever["mug_on_tray"]
        )

        # --------------------------------------------------------------
        # Final: close cabinet after both objects are already on tray.
        #
        # Use rising edge so closing the cabinet too early does not receive
        # ordered credit retroactively.
        # --------------------------------------------------------------
        close_rising = bool(
            raw["cabinet_closed"]
            and not self._lh_prev_raw["cabinet_closed"]
        )

        if (
            close_rising
            and both_objects_ready
        ):
            self._lh_ordered_ever["cabinet_closed"] = True

        ordered_state = (
            self._get_ordered_semantic_milestone_state()
        )

        # --------------------------------------------------------------
        # Regression bookkeeping for persistent placement / close milestones.
        # --------------------------------------------------------------
        for name in [
            "kettle_on_tray",
            "mug_on_tray",
            "cabinet_closed",
        ]:
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
        spec = self.get_semantic_milestone_spec()
        names = list(spec.keys())

        ups = (
            sum(
                int(bool(self._lh_unordered_ever[name]))
                for name in names
            )
            / float(len(names))
        )

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
        obj1_container_contact = OU.check_obj_in_receptacle(
            self,
            "obj",
            "container",
        )

        obj2_container_contact = OU.check_obj_in_receptacle(
            self,
            "obj2",
            "container",
        )

        cab_closed = self.cab.is_closed(env=self)

        gripper_obj_far = OU.gripper_obj_far(
            self
        )

        return (
            obj1_container_contact
            and obj2_container_contact
            and gripper_obj_far
            and cab_closed
        )

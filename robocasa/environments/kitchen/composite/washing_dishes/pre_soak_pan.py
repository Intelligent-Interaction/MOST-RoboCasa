from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.object_utils as OU


class PreSoakPan(Kitchen):
    """
    PreSoakPan

    Milestones:
        pan_in_hand      : pick pan
        pan_in_sink      : place pan in sink
        sponge_in_hand   : pick sponge
        sponge_in_sink   : place sponge in sink
        water_on         : turn on water after both pan and sponge are in sink

    The pan and sponge chains are independent:
        pick pan    -> place pan
        pick sponge -> place sponge

    Therefore all of these are valid:
        pan -> sink, sponge -> sink, water on
        sponge -> sink, pan -> sink, water on
        sponge -> pan, then pan+sponge -> sink, water on

    The sponge-on-pan intermediate state does not count as a valid placement
    until the sponge itself is inside the sink. If the sponge rides into the
    sink on the pan, sponge_in_sink and pan_in_sink can become valid in the
    same frame.
    """

    def __init__(self, *args, **kwargs):
        self._reset_long_horizon_metric_tracker()
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref(
            "sink", dict(id=FixtureType.SINK)
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.6, 0.4)),
        )
        self.init_robot_base_ref = self.sink

    def _setup_scene(self):
        super()._setup_scene()
        self.sink.set_handle_state(mode="off", env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        return [
            dict(
                name="obj1",
                obj_groups=("pan"),
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                        top_size=(0.6, 0.4),
                    ),
                    size=(0.35, 0.55),
                    pos=("ref", -1.0),
                ),
            ),
            dict(
                name="obj2",
                obj_groups=("sponge"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.2, 0.3),
                    pos=("ref", -1.0),
                    offset=(0.0, 0.05),
                ),
            ),
        ]

    # =====================================================================
    # Task-specific semantic milestones
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "pan_in_hand": {
                "dependencies": [],
                "persistent": False,
            },
            "pan_in_sink": {
                "dependencies": ["pan_in_hand"],
                "persistent": True,
            },
            "sponge_in_hand": {
                "dependencies": [],
                "persistent": False,
            },
            "sponge_in_sink": {
                "dependencies": ["sponge_in_hand"],
                "persistent": True,
            },
            "water_on": {
                "dependencies": [
                    "pan_in_sink",
                    "sponge_in_sink",
                ],
                "persistent": True,
            },
        }

    def _check_task_obj_grasped(
        self,
        obj_name,
    ):
        """
        Local grasp check.

        Use a slightly looser gripper-close threshold than RoboCasa's default
        0.035. This does not modify global object_utils behavior.
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

    def _check_obj_in_sink(
        self,
        obj_name,
    ):
        return bool(
            OU.obj_inside_of(
                self,
                obj_name,
                self.sink,
                partial_check=False,
            )
        )

    def _get_raw_semantic_milestone_state(self):
        pan_in_hand = self._check_task_obj_grasped("obj1")
        sponge_in_hand = self._check_task_obj_grasped("obj2")

        pan_in_sink = bool(
            self._check_obj_in_sink("obj1")
            and not pan_in_hand
        )
        sponge_in_sink = bool(
            self._check_obj_in_sink("obj2")
            and not sponge_in_hand
        )

        water_on = bool(
            self.sink.get_handle_state(env=self)["water_on"]
        )

        return {
            "pan_in_hand": pan_in_hand,
            "pan_in_sink": pan_in_sink,
            "sponge_in_hand": sponge_in_hand,
            "sponge_in_sink": sponge_in_sink,
            "water_on": water_on,
        }

    def _get_ordered_semantic_milestone_state(self):
        """
        Current ordered-valid state.

        Pick milestones are current-progress milestones:
            - valid while the object is currently in hand;
            - also valid once that object is validly placed in the sink,
              because placement supersedes the pick.

        If the object is dropped outside the sink, both pick and place
        contributions disappear from current OPS.
        """
        raw = self._get_raw_semantic_milestone_state()

        pan_place_valid = bool(
            self._lh_ordered_ever["pan_in_sink"]
            and raw["pan_in_sink"]
        )
        sponge_place_valid = bool(
            self._lh_ordered_ever["sponge_in_sink"]
            and raw["sponge_in_sink"]
        )

        pan_pick_valid = bool(
            self._lh_ordered_ever["pan_in_hand"]
            and (
                raw["pan_in_hand"]
                or pan_place_valid
            )
        )
        sponge_pick_valid = bool(
            self._lh_ordered_ever["sponge_in_hand"]
            and (
                raw["sponge_in_hand"]
                or sponge_place_valid
            )
        )

        water_valid = bool(
            self._lh_ordered_ever["water_on"]
            and raw["water_on"]
            and pan_place_valid
            and sponge_place_valid
        )

        return {
            "pan_in_hand": pan_pick_valid,
            "pan_in_sink": pan_place_valid,
            "sponge_in_hand": sponge_pick_valid,
            "sponge_in_sink": sponge_place_valid,
            "water_on": water_valid,
        }

    # =====================================================================
    # Long-horizon metric tracker
    # =====================================================================

    def _reset_long_horizon_metric_tracker(self):
        names = list(
            self.get_semantic_milestone_spec().keys()
        )

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
        self._lh_step_count = 0

    def _reset_internal(self):
        super()._reset_internal()
        self._reset_long_horizon_metric_tracker()

    def _progress(self):
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
        # Independent pan chain:
        #   pick pan -> place pan
        # --------------------------------------------------------------
        if raw["pan_in_hand"]:
            self._lh_ordered_ever["pan_in_hand"] = True

        if (
            raw["pan_in_sink"]
            and self._lh_ordered_ever["pan_in_hand"]
        ):
            self._lh_ordered_ever["pan_in_sink"] = True

        # --------------------------------------------------------------
        # Independent sponge chain:
        #   pick sponge -> place sponge
        # --------------------------------------------------------------
        if raw["sponge_in_hand"]:
            self._lh_ordered_ever["sponge_in_hand"] = True

        if (
            raw["sponge_in_sink"]
            and self._lh_ordered_ever["sponge_in_hand"]
        ):
            self._lh_ordered_ever["sponge_in_sink"] = True

        # --------------------------------------------------------------
        # Water must be turned on after both objects are validly placed.
        #
        # This is intentionally stricter than "water is already on". If water
        # was turned on too early, it does not receive ordered credit until it
        # is turned off and then on again after both placements.
        # --------------------------------------------------------------
        water_rising_edge = bool(
            raw["water_on"]
            and not self._lh_prev_raw["water_on"]
        )
        both_objects_currently_placed = bool(
            raw["pan_in_sink"]
            and raw["sponge_in_sink"]
            and self._lh_ordered_ever["pan_in_sink"]
            and self._lh_ordered_ever["sponge_in_sink"]
        )

        if (
            water_rising_edge
            and both_objects_currently_placed
        ):
            self._lh_ordered_ever["water_on"] = True

        progress = self._progress()
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
                sum(self._lh_unordered_ever.values()) / 5.0
            ),
            "OPS": float(
                self._progress()
            ),
            "progress_curve": list(
                self._lh_progress_curve
            ),
        }

    # =====================================================================
    # Original success criterion, using the same placement semantics as OPS
    # =====================================================================

    def _check_success(self):
        raw = self._get_raw_semantic_milestone_state()

        return bool(
            raw["water_on"]
            and raw["pan_in_sink"]
            and raw["sponge_in_sink"]
            and OU.gripper_obj_far(self, "obj1")
            and OU.gripper_obj_far(self, "obj2")
        )

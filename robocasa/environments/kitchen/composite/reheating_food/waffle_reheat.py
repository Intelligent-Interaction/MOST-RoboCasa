from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.object_utils as OU


class WaffleReheat(Kitchen):
    """
    Waffle Reheat: composite task for Reheating Food activity.

    Simulates the task of reheating a waffle.

    Steps:
        Open the microwave. Place the bowl with waffle inside the microwave,
        then close the microwave door and turn it on.

    Long-horizon progress:
        1. microwave_door_opened
        2. bowl_in_hand
        3. bowl_in_microwave
        4. microwave_door_closed
        5. microwave_turned_on

    Notes:
        - The robot grasps the bowl / waffle_container, not the waffle itself.
        - waffle_in_bowl is an initialization condition, not a progress
          milestone.
        - microwave_door_closed is credited only after the microwave has been
          opened and the bowl has been placed inside.
        - microwave_turned_on is credited only after a valid close stage.

    Original RoboCasa _check_success() is preserved.
    """

    # exclude layout 9 because the microwave is far from counters
    EXCLUDE_LAYOUTS = [9]

    def __init__(self, *args, **kwargs):
        self._reset_long_horizon_metric_tracker(initial_state=None)
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.microwave = self.register_fixture_ref(
            "microwave",
            dict(id=FixtureType.MICROWAVE),
        )

        self.counter = self.register_fixture_ref(
            "counter",
            dict(
                id=FixtureType.COUNTER,
                ref=self.microwave,
            ),
        )

        self.init_robot_base_ref = self.microwave

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(
                self.novel_instructions
            )
        else:
            ep_meta["lang"] = (
                "Open the microwave, place the bowl with waffle inside the microwave, "
                "then close the microwave door and turn it on."
            )

        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="waffle",
                obj_groups="waffle",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(0.3, 0.3),
                    pos=("ref", -1.0),
                    try_to_place_in="bowl",
                ),
            )
        )

        return cfgs

    # =====================================================================
    # Task-specific semantic milestones
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "microwave_door_opened": {
                "description": "The microwave door has been opened.",
                "dependencies": [],
                "persistent": False,
            },
            "bowl_in_hand": {
                "description": (
                    "The waffle bowl / waffle_container is grasped by the robot."
                ),
                "dependencies": ["microwave_door_opened"],
                "persistent": False,
            },
            "bowl_in_microwave": {
                "description": (
                    "The waffle bowl / waffle_container is placed inside "
                    "the microwave."
                ),
                "dependencies": ["bowl_in_hand"],
                "persistent": True,
            },
            "microwave_door_closed": {
                "description": (
                    "The microwave door is closed after the bowl has been "
                    "placed inside."
                ),
                "dependencies": ["bowl_in_microwave"],
                "persistent": True,
            },
            "microwave_turned_on": {
                "description": (
                    "The microwave is turned on after the valid door-close stage."
                ),
                "dependencies": ["microwave_door_closed"],
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

    def _microwave_is_closed(self):
        return bool(
            self.microwave.is_closed(
                env=self,
            )
        )

    def _microwave_is_open(self):
        return bool(
            not self._microwave_is_closed()
        )

    def _microwave_is_on(self):
        microwave_state = self.microwave.get_state()
        return bool(
            microwave_state["turned_on"]
        )

    def _get_raw_semantic_milestone_state(self):
        bowl_in_hand = self._check_task_obj_grasped(
            "waffle_container",
        )

        bowl_in_microwave = bool(
            OU.obj_inside_of(
                self,
                "waffle_container",
                self.microwave,
            )
            and not bowl_in_hand
        )

        microwave_door_closed = self._microwave_is_closed()
        microwave_door_open = bool(
            not microwave_door_closed
        )

        microwave_turned_on = self._microwave_is_on()

        return {
            "microwave_door_opened": microwave_door_open,
            "bowl_in_hand": bowl_in_hand,
            "bowl_in_microwave": bowl_in_microwave,
            "microwave_door_closed": microwave_door_closed,
            "microwave_turned_on": microwave_turned_on,
        }

    def _get_ordered_semantic_milestone_state(self):
        """
        Current ordered-valid milestone states.

        microwave_door_opened is a latched event.

        bowl_in_hand is current/superseded:
            - valid while the bowl is currently held;
            - also valid once the bowl is placed inside the microwave.

        bowl_in_microwave, microwave_door_closed, and microwave_turned_on must
        remain physically valid.
        """
        raw = self._get_raw_semantic_milestone_state()

        open_valid = bool(
            self._lh_ordered_ever["microwave_door_opened"]
        )

        bowl_place_valid = bool(
            self._lh_ordered_ever["bowl_in_microwave"]
            and raw["bowl_in_microwave"]
            and open_valid
        )

        bowl_pick_valid = bool(
            self._lh_ordered_ever["bowl_in_hand"]
            and open_valid
            and (
                raw["bowl_in_hand"]
                or bowl_place_valid
            )
        )

        close_valid = bool(
            self._lh_ordered_ever["microwave_door_closed"]
            and raw["microwave_door_closed"]
            and bowl_place_valid
        )

        on_valid = bool(
            self._lh_ordered_ever["microwave_turned_on"]
            and raw["microwave_turned_on"]
            and close_valid
        )

        return {
            "microwave_door_opened": open_valid,
            "bowl_in_hand": bowl_pick_valid,
            "bowl_in_microwave": bowl_place_valid,
            "microwave_door_closed": close_valid,
            "microwave_turned_on": on_valid,
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

        # If replay starts with an already-open microwave, allow the close
        # stage to be considered a real close after bowl placement.
        self._lh_microwave_ever_opened = bool(
            initial_state.get(
                "microwave_door_opened",
                False,
            )
        )

        self._lh_first_success_step = None
        self._lh_progress_curve = []

    def _reset_internal(self):
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
            int(bool(ordered_state["microwave_door_opened"]))
            + int(bool(ordered_state["bowl_in_hand"]))
            + int(bool(ordered_state["bowl_in_microwave"]))
            + int(bool(ordered_state["microwave_door_closed"]))
            + int(bool(ordered_state["microwave_turned_on"]))
        ) / 5.0

    def _update_long_horizon_metric_tracker(self):
        """
        Semantic chain:
            open microwave
                -> pick bowl
                -> place bowl in microwave
                -> close microwave
                -> turn microwave on
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
        # M1: open microwave.
        # --------------------------------------------------------------
        if raw["microwave_door_opened"]:
            self._lh_microwave_ever_opened = True
            self._lh_ordered_ever["microwave_door_opened"] = True

        open_ready = bool(
            self._lh_ordered_ever["microwave_door_opened"]
        )

        # --------------------------------------------------------------
        # M2: pick bowl / waffle_container.
        # --------------------------------------------------------------
        if (
            raw["bowl_in_hand"]
            and open_ready
        ):
            self._lh_ordered_ever["bowl_in_hand"] = True

        # --------------------------------------------------------------
        # M3: place bowl in microwave.
        #
        # If grasp detection missed the bowl pick, infer it from valid bowl
        # placement after the microwave has been opened.
        # --------------------------------------------------------------
        if (
            raw["bowl_in_microwave"]
            and open_ready
        ):
            if not self._lh_ordered_ever["bowl_in_hand"]:
                self._lh_ordered_ever["bowl_in_hand"] = True
            self._lh_ordered_ever["bowl_in_microwave"] = True

        # --------------------------------------------------------------
        # M4: close microwave door after valid bowl placement.
        # --------------------------------------------------------------
        door_close_event = bool(
            raw["microwave_door_closed"]
            and not self._lh_prev_raw[
                "microwave_door_closed"
            ]
            and self._lh_microwave_ever_opened
        )

        if (
            door_close_event
            and self._lh_ordered_ever["bowl_in_microwave"]
            and raw["bowl_in_microwave"]
        ):
            self._lh_ordered_ever["microwave_door_closed"] = True

        # --------------------------------------------------------------
        # M5: turn microwave on after valid close.
        # --------------------------------------------------------------
        microwave_on_event = bool(
            raw["microwave_turned_on"]
            and not self._lh_prev_raw[
                "microwave_turned_on"
            ]
        )

        if (
            microwave_on_event
            and self._lh_ordered_ever["microwave_door_closed"]
            and raw["microwave_door_closed"]
            and raw["bowl_in_microwave"]
        ):
            self._lh_ordered_ever["microwave_turned_on"] = True

        ordered_state = (
            self._get_ordered_semantic_milestone_state()
        )

        # --------------------------------------------------------------
        # Regression bookkeeping for persistent states.
        # --------------------------------------------------------------
        for name in [
            "bowl_in_microwave",
            "microwave_door_closed",
            "microwave_turned_on",
        ]:
            if not self._lh_ordered_ever[name]:
                continue

            prev_valid = bool(
                self._lh_prev_raw[name]
            )

            falling_edge = bool(
                prev_valid
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
        reward, done, info = (
            super()._post_action(action)
        )

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
        waffle_in_bowl = (
            OU.check_obj_in_receptacle(
                self,
                "waffle",
                "waffle_container",
            )
        )

        bowl_in_microwave = (
            OU.obj_inside_of(
                self,
                "waffle_container",
                self.microwave,
            )
        )

        microwave_on = (
            self.microwave.get_state()[
                "turned_on"
            ]
        )

        return (
            waffle_in_bowl
            and bowl_in_microwave
            and microwave_on
        )

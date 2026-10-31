from robocasa.environments.kitchen.kitchen import *


class RinseSinkBasin(Kitchen):
    """
    Rinse the Sink Basin: composite task for Washing Dishes activity.

    Simulates the process of rinsing the sink basin using the sink's spout.

    Steps:
        1. Turn on the sink.
        2. Move the spout to rinse all locations of the sink basin.

    Long-horizon semantic milestones
    ---------------------------------
    M1. turn_on_water
        The sink water has been turned on at least once.

    M2. rinse_left
        The left basin location has been rinsed while water is on.

    M3. rinse_center
        The center basin location has been rinsed while water is on.

    M4. rinse_right
        The right basin location has been rinsed while water is on.

    Dependency graph:

                      -> M2: rinse_left
        M1: water on -> M3: rinse_center
                      -> M4: rinse_right

    M2/M3/M4 are parallel and can be completed in any order.

    All four milestones are event/history milestones rather than persistent
    physical goal states. Once a basin location has been rinsed, moving the
    spout away or turning the water off does not undo that completed rinse.
    """

    def __init__(self, *args, **kwargs):
        self.washed_loc = [False, False, False]
        self._reset_long_horizon_metric_tracker(initial_state=None)
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.get_fixture(FixtureType.SINK)
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta[
                "lang"
            ] = "Turn on the sink and manuever the spout to wash all locations of the sink basin."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                placement=dict(
                    fixture=self.get_fixture(FixtureType.COUNTER, ref=self.sink),
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )
        return cfgs

    # =====================================================================
    # Original task state update
    # =====================================================================

    def _update_washed_locations(self):
        """
        Update the original task's historical washed_loc flags.

        This is exactly the semantic state used by the official success
        criterion:
            washed_loc[0] -> left
            washed_loc[1] -> center
            washed_loc[2] -> right
        """
        handle_state = self.sink.get_handle_state(env=self)

        if handle_state["water_on"]:
            if handle_state["spout_ori"] == "left":
                self.washed_loc[0] = True
            elif handle_state["spout_ori"] == "center":
                self.washed_loc[1] = True
            elif handle_state["spout_ori"] == "right":
                self.washed_loc[2] = True

    # =====================================================================
    # Task-specific semantic milestone definitions
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "turn_on_water": {
                "description": "The sink water is turned on.",
                "dependencies": [],
                "persistent": False,
            },
            "rinse_left": {
                "description": "The left sink basin location is rinsed.",
                "dependencies": ["turn_on_water"],
                "persistent": False,
            },
            "rinse_center": {
                "description": "The center sink basin location is rinsed.",
                "dependencies": ["turn_on_water"],
                "persistent": False,
            },
            "rinse_right": {
                "description": "The right sink basin location is rinsed.",
                "dependencies": ["turn_on_water"],
                "persistent": False,
            },
        }


    def _get_raw_semantic_milestone_state(self):
        """
        Return CURRENT / latched simulator-grounded semantic states.

        turn_on_water is the current physical water state.
        The three rinse states are latched historical states from washed_loc.
        """
        handle_state = self.sink.get_handle_state(env=self)

        return {
            "turn_on_water": bool(handle_state["water_on"]),
            "rinse_left": bool(self.washed_loc[0]),
            "rinse_center": bool(self.washed_loc[1]),
            "rinse_right": bool(self.washed_loc[2]),
        }

    # =====================================================================
    # Per-episode long-horizon metric tracker
    # =====================================================================

    def _reset_long_horizon_metric_tracker(self, initial_state=None):
        spec = self.get_semantic_milestone_spec()
        names = list(spec.keys())

        self._lh_step_count = 0

        if initial_state is None:
            initial_state = {name: False for name in names}

        self._lh_prev_raw = {
            name: bool(initial_state.get(name, False))
            for name in names
        }

        self._lh_unordered_ever = {name: False for name in names}
        self._lh_ordered_ever = {name: False for name in names}
        self._lh_regressed = {name: False for name in names}

        self._lh_first_success_step = None
        self._lh_progress_curve = []

    def _reset_internal(self):
        """
        Preserve the original reset behavior, then initialize metric history.
        """
        super()._reset_internal()

        self.sink.set_handle_state(mode="off", env=self, rng=self.rng)
        self.washed_loc = [False, False, False]

        initial_raw = self._get_raw_semantic_milestone_state()
        self._reset_long_horizon_metric_tracker(initial_state=initial_raw)

    def _get_current_ordered_progress(self):
        """
        All milestones in this task are historical/event milestones.

        Therefore ordered progress is simply the fraction of milestones that
        have ever received valid ordered credit.
        """
        spec = self.get_semantic_milestone_spec()
        num_milestones = len(spec)

        if num_milestones == 0:
            return 0.0

        completed = sum(
            int(self._lh_ordered_ever[name])
            for name in spec.keys()
        )
        return float(completed) / float(num_milestones)

    def _update_long_horizon_metric_tracker(self):
        """
        Update all long-horizon metrics once per policy timestep.

        M2/M3/M4 can occur in any order.

        A rinse location can receive ordered credit in the same timestep as
        the first water-on event. This is intentional: if the spout is already
        centered when water is switched on, the center location immediately
        begins being rinsed.
        """
        self._lh_step_count += 1

        # Ensure washed_loc is updated even if this helper is called outside
        # the usual reward / success-check path.
        self._update_washed_locations()

        spec = self.get_semantic_milestone_spec()
        raw = self._get_raw_semantic_milestone_state()

        # --------------------------------------------------------------
        # UPS: order ignored.
        # --------------------------------------------------------------
        for name in spec.keys():
            if raw[name]:
                self._lh_unordered_ever[name] = True

        # --------------------------------------------------------------
        # OPS M1: first actual OFF -> ON event.
        # --------------------------------------------------------------
        water_rising = bool(
            raw["turn_on_water"]
            and not self._lh_prev_raw["turn_on_water"]
        )

        if (
            water_rising
            and not self._lh_ordered_ever["turn_on_water"]
        ):
            self._lh_ordered_ever["turn_on_water"] = True

        # --------------------------------------------------------------
        # OPS M2/M3/M4:
        # all are parallel; once water has valid ordered credit, any newly
        # rinsed location receives ordered credit.
        #
        # If the spout is already at center when water is turned on, then
        # turn_on_water and rinse_center can be credited in the same timestep.
        # Afterwards, moving left/right or right/left are both valid.
        # --------------------------------------------------------------
        for name in ["rinse_left", "rinse_center", "rinse_right"]:
            rinse_rising = bool(
                raw[name] and not self._lh_prev_raw[name]
            )

            if (
                rinse_rising
                and self._lh_ordered_ever["turn_on_water"]
                and not self._lh_ordered_ever[name]
            ):
                self._lh_ordered_ever[name] = True

        current_ops = self._get_current_ordered_progress()
        self._lh_progress_curve.append(float(current_ops))

        # Official SR / NCT.
        success = bool(self._check_success())
        if success and self._lh_first_success_step is None:
            self._lh_first_success_step = self._lh_step_count

        self._lh_prev_raw = {
            name: bool(raw[name]) for name in spec.keys()
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
        ops = self._get_current_ordered_progress()

        spec = self.get_semantic_milestone_spec()
        num_milestones = len(spec)
        ups = (
            sum(self._lh_unordered_ever.values()) / float(num_milestones)
            if num_milestones > 0
            else 0.0
        )

        return {
            "SR": float(self._lh_first_success_step is not None),
            "UPS": float(ups),
            "OPS": float(ops),
            "NCT": (
                -1.0
                if self._lh_first_success_step is None
                else float(self._lh_first_success_step)
                / float(getattr(self, "horizon", 1))
            ),
            "progress_curve": list(self._lh_progress_curve),
        }


    # =====================================================================
    # Official RoboCasa success criterion
    # =====================================================================

    def _check_success(self):
        """
        Original task semantics:
        while water is on, mark the current spout location as washed;
        success once all three locations have been washed.
        """
        self._update_washed_locations()
        return all(self.washed_loc)
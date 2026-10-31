from robocasa.environments.kitchen.kitchen import *


class WashLettuce(Kitchen):
    """
    WashLettuce: composite task for Making Salads activity.

    Simulates the task of washing salads.

    Steps:
        Turn on the sink. Wash the lettuce under the sink for 25 timesteps.

    Long-horizon semantic milestones
    ---------------------------------
    M1. turn_on_water
        The sink water has been turned on.

    M2. start_washing_lettuce
        The lettuce is under running water and valid washing has started.

    M3. wash_lettuce_complete
        The original simulator wash counter reaches 25 timesteps.

    Dependency chain:
        M1 -> M2 -> M3

    All three are history/event milestones:
        - turning the water off later does not erase the fact that it was
          turned on;
        - moving the lettuce away later does not erase the fact that washing
          began;
        - once washed_time reaches 25, the task remains completed.

    Grasping or moving the lettuce is not treated as a separate semantic
    milestone.
    """

    def __init__(self, *args, **kwargs):
        self.washed_time = 0
        self._reset_long_horizon_metric_tracker()
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref(
            "sink",
            dict(id=FixtureType.SINK),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(
                id=FixtureType.COUNTER,
                ref=self.sink,
            ),
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(
                self.novel_instructions
            )
        else:
            ep_meta["lang"] = (
                "Wash the lettuce in the sink by running water over it."
            )

        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.washed_time = 0


    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="colander",
                obj_groups="colander",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.40, 0.50),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="lettuce",
                obj_groups="lettuce",
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.40, 0.50),
                    pos=("ref", -1.0),
                    try_to_place_in="colander",
                ),
            )
        )

        return cfgs


    # =====================================================================
    # Original task state update
    # =====================================================================

    def update_state(self):
        super().update_state()

        if self.sink.check_obj_under_water(
            self,
            "lettuce",
        ):
            self.washed_time += 1


    # =====================================================================
    # Long-horizon semantic milestone definitions
    # =====================================================================

    def get_semantic_milestone_spec(self):
        """
        Washing lettuce is a short manipulation chain.

        M1: turn on water
        M2: pick up colander containing lettuce
        M3: put lettuce/colander under running water
        M4: washing completes (25 valid timesteps)
        """
        return {
            "turn_on_water": {
                "description": "The sink water has been turned on.",
                "dependencies": [],
                "persistent": False,
            },
            "pick_colander": {
                "description": "The robot has grasped the colander.",
                "dependencies": [
                    "turn_on_water",
                ],
                "persistent": False,
            },
            "lettuce_under_water": {
                "description": "The colander with lettuce is under running water.",
                "dependencies": [
                    "pick_colander",
                ],
                "persistent": False,
            },
            "wash_lettuce_complete": {
                "description": "The lettuce has been washed for enough timesteps.",
                "dependencies": [
                    "lettuce_under_water",
                ],
                "persistent": True,
            },
        }

    def _get_raw_semantic_milestone_state(self):
        handle_state = self.sink.get_handle_state(
            env=self
        )

        water_on = bool(
            handle_state["water_on"]
        )

        # Gripper-colander distance proxy.
        # check_obj_grasped is too strict for this container object.
        # When the gripper is holding the colander, the end-effector should
        # remain close to the colander body.
        colander_grasped_proxy = not OU.gripper_obj_far(
            self,
            obj_name="lettuce",
            th=0.15,
        )

        lettuce_under_water = bool(
            self.sink.check_obj_under_water(
                self,
                "lettuce",
            )
        )

        wash_complete = bool(
            self.washed_time >= 25
        )

        return {
            "turn_on_water": water_on,
            "pick_colander": colander_grasped_proxy,
            "lettuce_under_water": lettuce_under_water,
            "wash_lettuce_complete": wash_complete,
        }

    # =====================================================================
    # Per-episode long-horizon metric tracker
    # =====================================================================

    def _reset_long_horizon_metric_tracker(self):
        spec = self.get_semantic_milestone_spec()
        names = list(spec.keys())

        self._lh_step_count = 0

        self._lh_prev_raw = {
            name: False for name in names
        }

        self._lh_unordered_ever = {
            name: False for name in names
        }

        self._lh_ordered_ever = {
            name: False for name in names
        }

        self._lh_regressed = {
            name: False for name in names
        }

        self._lh_ops_auc_sum = 0.0
        self._lh_peak_progress = 0.0
        self._lh_first_success_step = None
        self._lh_progress_curve = []

    def _reset_internal(self):
        """
        Preserve normal RoboCasa reset behavior, then initialize long-horizon
        metric history from the actual reset state.
        """
        super()._reset_internal()

        self._reset_long_horizon_metric_tracker()

    def _get_current_ordered_progress(self):
        """
        All milestones in this task are history/event milestones.

        Therefore ordered progress is the fraction that have ever received
        valid ordered credit.
        """
        spec = self.get_semantic_milestone_spec()
        num_milestones = len(spec)

        if num_milestones == 0:
            return 0.0

        completed = sum(
            int(
                self._lh_ordered_ever[name]
            )
            for name in spec.keys()
        )

        return (
            float(completed)
            / float(num_milestones)
        )


    def _update_long_horizon_metric_tracker(self):
        self._lh_step_count += 1

        spec = self.get_semantic_milestone_spec()
        raw = self._get_raw_semantic_milestone_state()

        for name in spec:
            if raw[name]:
                self._lh_unordered_ever[name] = True

        for name, cfg in spec.items():
            if raw[name] and not self._lh_ordered_ever[name]:
                if all(
                    self._lh_ordered_ever[d]
                    for d in cfg["dependencies"]
                ):
                    self._lh_ordered_ever[name] = True

        progress = self._get_current_ordered_progress()

        self._lh_progress_curve.append(progress)
        self._lh_ops_auc_sum += progress
        self._lh_peak_progress = max(
            self._lh_peak_progress,
            progress,
        )

        if (
            self._check_success()
            and self._lh_first_success_step is None
        ):
            self._lh_first_success_step = self._lh_step_count

        self._lh_prev_raw = dict(raw)

    def _post_action(self, action):
        """
        Kitchen._post_action() updates task state before this metric tracker
        runs, so washed_time already contains the current timestep's washing.
        """
        reward, done, info = super()._post_action(
            action
        )

        self._update_long_horizon_metric_tracker()

        return reward, done, info

    # =====================================================================
    # Public metric API
    # =====================================================================


    def get_long_horizon_metrics(self):
        names = list(
            self.get_semantic_milestone_spec().keys()
        )

        n = float(len(names))

        ups = (
            sum(self._lh_unordered_ever.values()) / n
            if n > 0 else 0.0
        )

        ops = (
            self._get_current_ordered_progress()
        )

        pauc = (
            self._lh_ops_auc_sum / float(self._lh_step_count)
            if self._lh_step_count > 0 else 0.0
        )

        if self._lh_first_success_step is None:
            nct = -1.0
        else:
            horizon = float(getattr(self, "horizon", 0))
            nct = (
                self._lh_first_success_step / horizon
                if horizon > 0
                else float(self._lh_first_success_step)
            )

        return {
            "SR": float(self._lh_first_success_step is not None),
            "UPS": float(ups),
            "OPS": float(ops),
            "NCT": float(nct),
            "progress_curve": list(self._lh_progress_curve),
        }

    # =====================================================================
    # Original RoboCasa success criterion, preserved
    # =====================================================================

    def _check_success(self):
        return self.washed_time >= 25

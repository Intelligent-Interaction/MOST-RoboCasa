from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.object_utils as OU


class WashFruitColander(Kitchen):
    """
    Wash Fruit Colander.

    Task structure used for long-horizon progress:
        1. pick colander
        2. place colander in sink
        3. pick fruit 1
        4. place fruit 1 in colander
        5. pick fruit 2
        6. place fruit 2 in colander
        7. turn on water / wash loaded colander

    Notes:
        - The two fruit PnPs are order-free.
        - Fruit identity is not fixed to a slot. Any two tracked fruits can
          fill the two generic fruit PnP stages.
        - The original terminal _check_success() is preserved.
        - For this task definition, new episodes default to two fruits unless
          replay metadata explicitly provides num_fruit.
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

        self.counter = self.register_fixture_ref(
            "counter",
            dict(
                id=FixtureType.COUNTER,
                ref=self.sink,
            ),
        )

        self.init_robot_base_ref = self.sink

        if "refs" in self._ep_meta and "num_fruit" in self._ep_meta["refs"]:
            self.num_fruit = int(self._ep_meta["refs"]["num_fruit"])
        else:
            self.num_fruit = 2

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        fruit0 = self.get_obj_lang("fruit0")
        fruit_lang = fruit0

        for i in range(1, self.num_fruit):
            fi = self.get_obj_lang(f"fruit{i}")
            fruit_lang += f", {fi}"

        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions).format(
                fruit_lang=fruit_lang
            )
        else:
            ep_meta["lang"] = (
                f"Put the colander in the sink, put the {fruit_lang} "
                "in the colander, and turn on the sink faucet and pour "
                "water over the colander."
            )

        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["num_fruit"] = self.num_fruit

        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

        self.sink.set_handle_state(
            mode="off",
            env=self,
            rng=self.rng,
        )

    def _get_obj_cfgs(self):
        cfgs = []

        for i in range(self.num_fruit):
            cfgs.append(
                dict(
                    name=f"fruit{i}",
                    obj_groups="fruit",
                    graspable=True,
                    washable=True,
                    placement=dict(
                        fixture=self.counter,
                        sample_region_kwargs=dict(
                            ref=self.sink,
                            loc="left_right",
                        ),
                        size=(0.30, 0.30),
                        pos=("ref", -1.0),
                    ),
                )
            )

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

        return cfgs

    # =====================================================================
    # Task-semantic helpers
    # =====================================================================

    def _tracked_fruit_names(self):
        """
        This metric tracks two fruit PnPs. If legacy replay metadata contains a
        different num_fruit, use the first two available fruits for the progress
        curve while leaving original _check_success() unchanged.
        """
        num_tracked = min(
            2,
            int(getattr(self, "num_fruit", 2)),
        )

        return [f"fruit{i}" for i in range(num_tracked)]

    def _check_task_obj_grasped(self, obj_name):
        """
        Local grasp check with a looser gripper-close threshold.
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
            return bool(OU.check_obj_grasped(self, obj_name))

    def _fruit_in_colander(self, fruit_name):
        return bool(
            OU.check_obj_in_receptacle(
                self,
                fruit_name,
                "colander",
            )
        )

    def _colander_in_sink(self):
        return bool(
            OU.obj_inside_of(
                self,
                "colander",
                self.sink,
            )
        )

    def _colander_under_water(self):
        return bool(
            self.sink.check_obj_under_water(
                self,
                "colander",
            )
        )

    # =====================================================================
    # Task-specific semantic milestone definitions
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "colander_in_hand": {
                "description": "The colander is grasped by the robot gripper.",
                "dependencies": [],
                "persistent": False,
            },
            "colander_in_sink": {
                "description": "The colander is placed in the sink.",
                "dependencies": ["colander_in_hand"],
                "persistent": True,
            },
            "fruit_pnp1_in_hand": {
                "description": "The first fruit is grasped.",
                "dependencies": ["colander_in_sink"],
                "persistent": False,
            },
            "fruit_pnp1_in_colander": {
                "description": "The first fruit is placed in the colander.",
                "dependencies": ["fruit_pnp1_in_hand"],
                "persistent": True,
            },
            "fruit_pnp2_in_hand": {
                "description": "The second fruit is grasped.",
                "dependencies": ["fruit_pnp1_in_colander"],
                "persistent": False,
            },
            "fruit_pnp2_in_colander": {
                "description": "The second fruit is placed in the colander.",
                "dependencies": ["fruit_pnp2_in_hand"],
                "persistent": True,
            },
            "water_on": {
                "description": "Water is running over the loaded colander.",
                "dependencies": ["fruit_pnp2_in_colander"],
                "persistent": True,
            },
        }

    def _get_raw_semantic_milestone_state(self):
        colander_in_hand = self._check_task_obj_grasped("colander")

        colander_in_sink = bool(
            self._colander_in_sink()
            and not colander_in_hand
        )

        grasped_fruits = set()
        valid_placed_fruits = set()

        for fruit_name in self._tracked_fruit_names():
            fruit_in_hand = self._check_task_obj_grasped(fruit_name)

            if fruit_in_hand:
                grasped_fruits.add(fruit_name)

            if self._fruit_in_colander(fruit_name) and not fruit_in_hand:
                valid_placed_fruits.add(fruit_name)

        colander_under_water = self._colander_under_water()

        return {
            "colander_in_hand": colander_in_hand,
            "colander_in_sink": colander_in_sink,
            "grasped_fruits": grasped_fruits,
            "valid_placed_fruits": valid_placed_fruits,
            "all_tracked_fruits_in_colander": bool(
                len(valid_placed_fruits) >= len(self._tracked_fruit_names())
            ),
            "colander_under_water": colander_under_water,
            "water_on": colander_under_water,
        }

    def _current_valid_fruit_pick_objs(self, raw=None):
        if raw is None:
            raw = self._get_raw_semantic_milestone_state()

        # Pick credit is current/superseded:
        #   - valid while the fruit is currently held;
        #   - also valid when that fruit currently remains in the colander.
        return set(self._lh_ever_picked_fruits).intersection(
            set(raw["grasped_fruits"]) | set(raw["valid_placed_fruits"])
        )

    def _get_current_fruit_action_count(self, raw=None):
        if raw is None:
            raw = self._get_raw_semantic_milestone_state()

        return (
            len(self._current_valid_fruit_pick_objs(raw))
            + len(raw["valid_placed_fruits"])
        )

    def _get_ordered_semantic_milestone_state(self):
        raw = self._get_raw_semantic_milestone_state()

        colander_place_valid = bool(
            self._lh_ordered_ever["colander_in_sink"]
            and raw["colander_in_sink"]
        )

        colander_pick_valid = bool(
            self._lh_ordered_ever["colander_in_hand"]
            and (
                raw["colander_in_hand"]
                or colander_place_valid
            )
        )

        fruit_count = 0
        if colander_place_valid:
            fruit_count = self._get_current_fruit_action_count(raw)

        fruit_stage_names = [
            "fruit_pnp1_in_hand",
            "fruit_pnp1_in_colander",
            "fruit_pnp2_in_hand",
            "fruit_pnp2_in_colander",
        ]

        fruit_state = {
            name: bool(idx < fruit_count)
            for idx, name in enumerate(fruit_stage_names)
        }

        water_valid = bool(
            self._lh_ordered_ever["water_on"]
            and raw["colander_under_water"]
            and colander_place_valid
            and raw["all_tracked_fruits_in_colander"]
        )

        return {
            "colander_in_hand": colander_pick_valid,
            "colander_in_sink": colander_place_valid,
            "fruit_pnp1_in_hand": fruit_state["fruit_pnp1_in_hand"],
            "fruit_pnp1_in_colander": fruit_state["fruit_pnp1_in_colander"],
            "fruit_pnp2_in_hand": fruit_state["fruit_pnp2_in_hand"],
            "fruit_pnp2_in_colander": fruit_state["fruit_pnp2_in_colander"],
            "water_on": water_valid,
        }

    # =====================================================================
    # Per-episode long-horizon metric tracker
    # =====================================================================

    def _reset_long_horizon_metric_tracker(self, initial_state=None):
        names = list(self.get_semantic_milestone_spec().keys())

        self._lh_step_count = 0

        if initial_state is None:
            initial_state = {}

        self._lh_prev_raw = {
            "colander_in_hand": bool(initial_state.get("colander_in_hand", False)),
            "colander_in_sink": bool(initial_state.get("colander_in_sink", False)),
            "water_on": bool(initial_state.get("water_on", False)),
            "colander_under_water": bool(initial_state.get("colander_under_water", False)),
        }

        self._lh_unordered_ever = {name: False for name in names}
        self._lh_ordered_ever = {name: False for name in names}
        self._lh_regressed = {name: False for name in names}

        self._lh_ever_picked_fruits = set()
        self._lh_ever_valid_placed_fruits = set()

        self._lh_prev_ordered_state = {name: False for name in names}

        self._lh_first_success_step = None
        self._lh_progress_curve = []

    def _reset_internal(self):
        super()._reset_internal()

        initial_raw = self._get_raw_semantic_milestone_state()

        self._reset_long_horizon_metric_tracker(
            initial_state=initial_raw
        )

    def _get_current_ordered_progress(self, ordered_state=None):
        if ordered_state is None:
            ordered_state = self._get_ordered_semantic_milestone_state()

        names = list(self.get_semantic_milestone_spec().keys())

        completed = sum(
            int(bool(ordered_state[name]))
            for name in names
        )

        return float(completed) / float(len(names))

    def _get_unordered_progress(self):
        count = 0

        if self._lh_unordered_ever["colander_in_hand"]:
            count += 1

        if self._lh_unordered_ever["colander_in_sink"]:
            count += 1

        count += len(self._lh_ever_picked_fruits)
        count += len(self._lh_ever_valid_placed_fruits)

        if self._lh_unordered_ever["water_on"]:
            count += 1

        return float(count) / 7.0

    def _update_fruit_stage_flags_from_count(self, count):
        fruit_stage_names = [
            "fruit_pnp1_in_hand",
            "fruit_pnp1_in_colander",
            "fruit_pnp2_in_hand",
            "fruit_pnp2_in_colander",
        ]

        for idx, name in enumerate(fruit_stage_names):
            if idx < count:
                self._lh_ordered_ever[name] = True

    def _update_unordered_fruit_stage_flags_from_count(self, count):
        fruit_stage_names = [
            "fruit_pnp1_in_hand",
            "fruit_pnp1_in_colander",
            "fruit_pnp2_in_hand",
            "fruit_pnp2_in_colander",
        ]

        for idx, name in enumerate(fruit_stage_names):
            if idx < count:
                self._lh_unordered_ever[name] = True

    def _update_long_horizon_metric_tracker(self):
        """
        Update metrics once per policy timestep.

        Chain:
            pick colander
                -> place colander in sink
                -> two order-free fruit PnPs into colander
                -> water running over loaded colander
        """
        self._lh_step_count += 1

        raw = self._get_raw_semantic_milestone_state()

        # --------------------------------------------------------------
        # UPS raw events.
        # --------------------------------------------------------------
        if raw["colander_in_hand"]:
            self._lh_unordered_ever["colander_in_hand"] = True

        if raw["colander_in_sink"]:
            self._lh_unordered_ever["colander_in_sink"] = True

        # --------------------------------------------------------------
        # M1: pick colander.
        # --------------------------------------------------------------
        if raw["colander_in_hand"]:
            self._lh_ordered_ever["colander_in_hand"] = True

        # --------------------------------------------------------------
        # M2: place colander in sink.
        # --------------------------------------------------------------
        if raw["colander_in_sink"]:
            if not self._lh_ordered_ever["colander_in_hand"]:
                self._lh_ordered_ever["colander_in_hand"] = True
                self._lh_unordered_ever["colander_in_hand"] = True

            self._lh_ordered_ever["colander_in_sink"] = True

        colander_ready = bool(
            self._lh_ordered_ever["colander_in_sink"]
            and raw["colander_in_sink"]
        )

        # --------------------------------------------------------------
        # M3-M6: two fruit PnPs, order-free after colander is in sink.
        # --------------------------------------------------------------
        if colander_ready:
            for fruit_name in raw["grasped_fruits"]:
                self._lh_ever_picked_fruits.add(fruit_name)

            # Anti-flicker inference: if a fruit is in the colander but its
            # grasp was missed, infer the corresponding pick.
            for fruit_name in raw["valid_placed_fruits"]:
                self._lh_ever_picked_fruits.add(fruit_name)
                self._lh_ever_valid_placed_fruits.add(fruit_name)

            current_fruit_count = self._get_current_fruit_action_count(raw)
            self._update_fruit_stage_flags_from_count(current_fruit_count)

            unordered_fruit_count = (
                len(self._lh_ever_picked_fruits)
                + len(self._lh_ever_valid_placed_fruits)
            )
            self._update_unordered_fruit_stage_flags_from_count(
                unordered_fruit_count
            )

        # --------------------------------------------------------------
        # M7: water on / loaded colander under water.
        # --------------------------------------------------------------
        water_valid = bool(
            raw["colander_under_water"]
            and colander_ready
            and raw["all_tracked_fruits_in_colander"]
        )

        if water_valid:
            self._lh_ordered_ever["water_on"] = True
            self._lh_unordered_ever["water_on"] = True

        ordered_state = self._get_ordered_semantic_milestone_state()

        # --------------------------------------------------------------
        # Regression bookkeeping.
        # --------------------------------------------------------------
        for name in self.get_semantic_milestone_spec().keys():
            if (
                self._lh_prev_ordered_state[name]
                and not ordered_state[name]
            ):
                self._lh_regressed[name] = True

        current_ops = self._get_current_ordered_progress(ordered_state)
        self._lh_progress_curve.append(float(current_ops))

        # Official SR / NCT.
        success = bool(self._check_success())

        if success and self._lh_first_success_step is None:
            self._lh_first_success_step = self._lh_step_count

        self._lh_prev_raw = {
            "colander_in_hand": bool(raw["colander_in_hand"]),
            "colander_in_sink": bool(raw["colander_in_sink"]),
            "water_on": bool(raw["water_on"]),
            "colander_under_water": bool(raw["colander_under_water"]),
        }

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
        ups = self._get_unordered_progress()
        ops = self._get_current_ordered_progress()

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
    # Original RoboCasa success criterion, preserved
    # =====================================================================

    def _check_success(self):
        fruit_in_colander = all(
            [
                OU.check_obj_in_receptacle(
                    self,
                    f"fruit{i}",
                    "colander",
                )
                for i in range(self.num_fruit)
            ]
        )

        return (
            self.sink.check_obj_under_water(
                self,
                "colander",
            )
            and fruit_in_colander
        )

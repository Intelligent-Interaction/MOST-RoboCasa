from robocasa.environments.kitchen.kitchen import *
import numpy as np
import robocasa.utils.object_utils as OU


class RecycleBottlesByType(Kitchen):
    """
    Recycle Bottles By Type: composite task for Organizing Recycling.

    Simulates the task of sorting bottles by type for recycling.

    Task:
        Move the three middle bottles to the correct type clusters:
            - bottle_plastic_middle -> plastic cluster
            - bottle_glass_middle   -> glass cluster
            - mystery_middle        -> cluster determined by its true type

    Progress design:
        This is treated as 3 unordered PnPs.

        Generic milestones:
            1. pnp1_bottle_in_hand
            2. pnp1_bottle_placed
            3. pnp2_bottle_in_hand
            4. pnp2_bottle_placed
            5. pnp3_bottle_in_hand
            6. pnp3_bottle_placed

        Object order is free:
            plastic middle first, glass middle first, mystery first, etc.

        A place is valid only if the moved bottle is placed near a reference
        bottle of its true type.

    Robust replay handling:
        Old offline replay episodes may not restore self.choice, even though
        original data generation used self.choice to choose the mystery bottle
        category. All metric / success helpers therefore call
        _get_mystery_choice(), which reconstructs or infers the mystery choice
        instead of directly reading self.choice.

    Original RoboCasa _check_success() semantics are preserved when the mystery
    choice is available. For legacy replay states with no saved choice, the
    helper falls back to object-attribute inference and then geometry inference
    to avoid crashing during visualization.
    """

    other_excluded_layouts = [
        8,
        11,
        12,
        13,
        19,
        21,
        22,
        23,
        25,
        28,
        29,
        30,
        33,
        34,
        38,
        41,
        42,
        46,
        49,
        54,
        57,
        58,
        60,
    ]

    EXCLUDE_LAYOUTS = (
        Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS
        + other_excluded_layouts
    )

    def __init__(self, *args, **kwargs):
        self._reset_long_horizon_metric_tracker(
            initial_state=None
        )
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref(
            "sink",
            dict(id=FixtureType.SINK),
        )

        self.dining_counter = self.register_fixture_ref(
            "dining_counter",
            dict(id=FixtureType.DINING_COUNTER),
        )

        self.counter = self.register_fixture_ref(
            "counter",
            dict(
                id=FixtureType.COUNTER,
                ref=self.sink,
            ),
        )

        if "stool2" in self.fixture_refs:
            self.stool1 = self.fixture_refs["stool1"]
            self.stool2 = self.fixture_refs["stool2"]
            self.stool3 = self.fixture_refs["stool3"]

            self.end_ref_plastic = (
                self.fixture_refs[
                    "end_ref_plastic"
                ]
            )
            self.end_ref_glass = (
                self.fixture_refs[
                    "end_ref_glass"
                ]
            )

        else:
            stools = [
                fixture
                for fixture
                in self.fixtures.values()
                if isinstance(
                    fixture,
                    robocasa.models.fixtures.accessories.Stool,
                )
            ]

            if len(stools) < 3:
                raise RuntimeError(
                    f"Expected at least 3 stools, found {len(stools)}"
                )

            # Find two stools farthest apart.
            max_d = -1.0
            ends = (None, None)

            for i in range(len(stools)):
                pi = stools[i].pos

                for j in range(i + 1, len(stools)):
                    pj = stools[j].pos
                    d = np.linalg.norm(pi - pj)

                    if d > max_d:
                        max_d = d
                        ends = (
                            stools[i],
                            stools[j],
                        )

            (
                stool_A_candidate,
                stool_C_candidate,
            ) = ends

            if (
                stool_A_candidate.pos[0]
                > stool_C_candidate.pos[0]
            ):
                (
                    stool_A_candidate,
                    stool_C_candidate,
                ) = (
                    stool_C_candidate,
                    stool_A_candidate,
                )

            # Pick B as the stool closest to midpoint of A and C.
            midpt = 0.5 * (
                stool_A_candidate.pos
                + stool_C_candidate.pos
            )

            best_B = None
            best_dist = float("inf")

            for s in stools:
                if (
                    s is stool_A_candidate
                    or s is stool_C_candidate
                ):
                    continue

                d = np.linalg.norm(
                    s.pos - midpt
                )

                if d < best_dist:
                    best_dist = d
                    best_B = s

            self.stool1 = stool_A_candidate
            self.stool2 = best_B
            self.stool3 = stool_C_candidate

            # Randomly assign plastic / glass ends.
            if self.rng.choice(
                [True, False]
            ):
                self.end_ref_plastic = (
                    self.stool1
                )
                self.end_ref_glass = (
                    self.stool3
                )
            else:
                self.end_ref_plastic = (
                    self.stool3
                )
                self.end_ref_glass = (
                    self.stool1
                )

            self.fixture_refs[
                "stool1"
            ] = self.stool1

            self.fixture_refs[
                "stool2"
            ] = self.stool2

            self.fixture_refs[
                "stool3"
            ] = self.stool3

            self.fixture_refs[
                "end_ref_plastic"
            ] = self.end_ref_plastic

            self.fixture_refs[
                "end_ref_glass"
            ] = self.end_ref_glass

        self.init_robot_base_ref = (
            self.stool2
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(
                self.novel_instructions
            )
        else:
            ep_meta["lang"] = (
                "Move the plastic bottles in the middle to the plastics group, "
                "and the glass bottles in the middle to the glass group."
            )

        # New episodes should store the mystery bottle type explicitly so replay
        # does not depend on self.choice being reconstructed.
        ep_meta["refs"] = ep_meta.get(
            "refs",
            {},
        )

        if hasattr(self, "choice"):
            ep_meta["refs"]["mystery_choice"] = self.choice

        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        # Plastic reference bottles at one end.
        for i in range(2):
            cfgs.append(
                dict(
                    name=f"bottle_plastic{i+1}",
                    obj_groups="bottled_water",
                    placement=dict(
                        fixture=self.dining_counter,
                        sample_region_kwargs=dict(
                            ref=self.end_ref_plastic
                        ),
                        size=(0.3, 0.3),
                        pos=(
                            0.0 + i * 0.3,
                            -0.95,
                        ),
                    ),
                )
            )

        # Middle plastic.
        cfgs.append(
            dict(
                name="bottle_plastic_middle",
                obj_groups="bottled_water",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool2
                    ),
                    size=(0.3, 0.3),
                    pos=("ref", -0.95),
                ),
            )
        )

        # Middle glass.
        cfgs.append(
            dict(
                name="bottle_glass_middle",
                obj_groups="alcohol",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool2
                    ),
                    size=(0.3, 0.3),
                    pos=("ref", -0.95),
                ),
            )
        )

        # Mystery bottle in the middle.
        refs = getattr(
            self,
            "_ep_meta",
            {},
        ).get(
            "refs",
            {},
        )

        saved_choice = refs.get(
            "mystery_choice",
            refs.get(
                "choice",
                None,
            ),
        )

        if saved_choice in [
            "alcohol",
            "bottled_water",
        ]:
            self.choice = saved_choice
        elif not hasattr(self, "choice"):
            self.choice = self.rng.choice(
                [
                    "alcohol",
                    "bottled_water",
                ]
            )

        cfgs.append(
            dict(
                name="mystery_middle",
                obj_groups=self.choice,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool2
                    ),
                    size=(0.3, 0.3),
                    pos=("ref", -0.95),
                ),
            )
        )

        # Glass reference bottles at the other end.
        for i in range(2):
            cfgs.append(
                dict(
                    name=f"bottle_glass{i+1}",
                    obj_groups="alcohol",
                    placement=dict(
                        fixture=self.dining_counter,
                        sample_region_kwargs=dict(
                            ref=self.end_ref_glass
                        ),
                        size=(0.3, 0.3),
                        pos=(
                            0.0 + i * 0.3,
                            -0.95,
                        ),
                    ),
                )
            )

        return cfgs

    # =====================================================================
    # Mystery bottle reconstruction
    # =====================================================================

    def _get_mystery_choice(self):
        """
        Return the true category of mystery_middle.

        Primary path:
            use self.choice, as in the original generated environment.

        Replay-safe paths:
            1. read saved ep_meta refs when available;
            2. infer from object attributes / model strings;
            3. as a last resort, infer from which reference cluster the object
               is currently closer to.

        The fallback avoids AttributeError during old offline replay episodes
        where self.choice was never restored.
        """
        if getattr(
            self,
            "choice",
            None,
        ) in [
            "alcohol",
            "bottled_water",
        ]:
            return self.choice

        refs = getattr(
            self,
            "_ep_meta",
            {},
        ).get(
            "refs",
            {},
        )

        for key in [
            "mystery_choice",
            "choice",
            "mystery_type",
            "mystery_category",
        ]:
            value = refs.get(
                key,
                None,
            )
            if value in [
                "alcohol",
                "bottled_water",
            ]:
                self.choice = value
                return self.choice

        # Try to infer from the object instance. This covers many replay cases
        # where the object model still carries a category/model string.
        obj = getattr(
            self,
            "objects",
            {},
        ).get(
            "mystery_middle",
            None,
        )

        text_parts = []

        if obj is not None:
            for attr in [
                "name",
                "root_body",
                "naming_prefix",
                "obj_type",
                "obj_group",
                "obj_groups",
                "group",
                "groups",
                "category",
                "mjcf_path",
                "xml_path",
            ]:
                try:
                    value = getattr(
                        obj,
                        attr,
                    )
                    text_parts.append(
                        str(value)
                    )
                except Exception:
                    pass

            for attr in [
                "contact_geoms",
                "visual_geoms",
                "important_sites",
            ]:
                try:
                    value = getattr(
                        obj,
                        attr,
                    )
                    text_parts.append(
                        str(value)
                    )
                except Exception:
                    pass

        haystack = " ".join(
            text_parts
        ).lower()

        if any(
            token in haystack
            for token in [
                "alcohol",
                "wine",
                "beer",
                "glass_bottle",
            ]
        ):
            self.choice = "alcohol"
            return self.choice

        if any(
            token in haystack
            for token in [
                "bottled_water",
                "water",
                "plastic",
            ]
        ):
            self.choice = "bottled_water"
            return self.choice

        # Last-resort geometry fallback. This is mainly for visualization of old
        # trajectories whose metadata did not preserve the sampled choice.
        try:
            mystery_xy = self._body_xy(
                "mystery_middle"
            )

            plastic_refs = [
                "bottle_plastic1",
                "bottle_plastic2",
            ]
            glass_refs = [
                "bottle_glass1",
                "bottle_glass2",
            ]

            plastic_d = min(
                np.linalg.norm(
                    mystery_xy - self._body_xy(ref)
                )
                for ref in plastic_refs
            )

            glass_d = min(
                np.linalg.norm(
                    mystery_xy - self._body_xy(ref)
                )
                for ref in glass_refs
            )

            self.choice = (
                "bottled_water"
                if plastic_d <= glass_d
                else "alcohol"
            )
            return self.choice

        except Exception:
            # Deterministic final fallback to avoid crashing. This branch should
            # almost never be reached after object construction.
            self.choice = "bottled_water"
            return self.choice

    # =====================================================================
    # Original task-semantic final-state helpers
    # =====================================================================

    def _get_bottle_groups(self):
        plastic = [
            "bottle_plastic1",
            "bottle_plastic2",
            "bottle_plastic_middle",
        ]

        glass = [
            "bottle_glass1",
            "bottle_glass2",
            "bottle_glass_middle",
        ]

        mystery = "mystery_middle"

        mystery_choice = self._get_mystery_choice()

        if mystery_choice == "alcohol":
            glass.append(mystery)
        else:
            plastic.append(mystery)

        return plastic, glass

    def _cluster_okay(
        self,
        names,
        thresh=0.30,
    ):
        """
        Same type-clustering semantics as the original success function.

        Every middle bottle of this category must be within threshold of at
        least one of the two reference/sample bottles.
        """
        sample_names = [
            n
            for n in names
            if not n.endswith("_middle")
        ]

        middle_names = [
            n
            for n in names
            if n.endswith("_middle")
        ]

        pos2d = {
            n: np.array(
                self.sim.data.body_xpos[
                    self.obj_body_id[n]
                ]
            )[:2]
            for n in names
        }

        for middle_name in middle_names:
            dists = [
                np.linalg.norm(
                    pos2d[middle_name]
                    - pos2d[sample_name]
                )
                for sample_name
                in sample_names
            ]

            if not any(
                d <= thresh
                for d in dists
            ):
                return False

        return True

    # =====================================================================
    # 3-PnP metric helpers
    # =====================================================================

    def _movable_bottle_names(self):
        return [
            "bottle_plastic_middle",
            "bottle_glass_middle",
            "mystery_middle",
        ]

    def _reference_bottles_for_obj(
        self,
        obj_name,
    ):
        if obj_name == "bottle_plastic_middle":
            return [
                "bottle_plastic1",
                "bottle_plastic2",
            ]

        if obj_name == "bottle_glass_middle":
            return [
                "bottle_glass1",
                "bottle_glass2",
            ]

        if obj_name == "mystery_middle":
            mystery_choice = self._get_mystery_choice()

            if mystery_choice == "alcohol":
                return [
                    "bottle_glass1",
                    "bottle_glass2",
                ]

            return [
                "bottle_plastic1",
                "bottle_plastic2",
            ]

        raise ValueError(
            f"Unknown movable bottle: {obj_name}"
        )

    def _body_xy(
        self,
        obj_name,
    ):
        return np.array(
            self.sim.data.body_xpos[
                self.obj_body_id[obj_name]
            ]
        )[:2]

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

    def _is_obj_in_correct_cluster(
        self,
        obj_name,
        thresh=0.30,
    ):
        """
        Per-object placement validity.

        A movable bottle is placed correctly if:
            - it is still on the dining counter,
            - it is within threshold of at least one reference bottle of its
              true category.
        """
        if not OU.check_obj_fixture_contact(
            self,
            obj_name,
            self.dining_counter,
        ):
            return False

        obj_xy = self._body_xy(obj_name)

        for ref_name in self._reference_bottles_for_obj(obj_name):
            ref_xy = self._body_xy(ref_name)
            if np.linalg.norm(obj_xy - ref_xy) <= thresh:
                return True

        return False

    # =====================================================================
    # Semantic milestones
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "pnp1_bottle_in_hand": {
                "description": "The first movable bottle is grasped.",
                "dependencies": [],
                "persistent": False,
            },
            "pnp1_bottle_placed": {
                "description": (
                    "The first movable bottle is placed into its correct "
                    "type cluster."
                ),
                "dependencies": ["pnp1_bottle_in_hand"],
                "persistent": True,
            },
            "pnp2_bottle_in_hand": {
                "description": "The second movable bottle is grasped.",
                "dependencies": ["pnp1_bottle_placed"],
                "persistent": False,
            },
            "pnp2_bottle_placed": {
                "description": (
                    "The second movable bottle is placed into its correct "
                    "type cluster."
                ),
                "dependencies": ["pnp2_bottle_in_hand"],
                "persistent": True,
            },
            "pnp3_bottle_in_hand": {
                "description": "The third movable bottle is grasped.",
                "dependencies": ["pnp2_bottle_placed"],
                "persistent": False,
            },
            "pnp3_bottle_placed": {
                "description": (
                    "The third movable bottle is placed into its correct "
                    "type cluster."
                ),
                "dependencies": ["pnp3_bottle_in_hand"],
                "persistent": True,
            },
        }

    def _get_raw_semantic_milestone_state(self):
        grasped_objs = set()
        valid_placed_objs = set()

        for obj_name in self._movable_bottle_names():
            obj_in_hand = self._check_task_obj_grasped(
                obj_name,
            )

            if obj_in_hand:
                grasped_objs.add(obj_name)

            if (
                self._is_obj_in_correct_cluster(obj_name)
                and not obj_in_hand
            ):
                valid_placed_objs.add(obj_name)

        return {
            "grasped_objs": grasped_objs,
            "valid_placed_objs": valid_placed_objs,
        }

    def _current_valid_pick_objs(
        self,
        raw=None,
    ):
        if raw is None:
            raw = self._get_raw_semantic_milestone_state()

        # Pick credit is current/superseded:
        #   - valid while a bottle is currently held;
        #   - also valid when that bottle currently remains correctly placed.
        #
        # If a picked bottle is dropped outside the correct cluster, its pick
        # credit disappears from current OPS.
        return set(
            self._lh_ever_picked_objs
        ).intersection(
            set(raw["grasped_objs"])
            | set(raw["valid_placed_objs"])
        )

    def _get_current_action_count(
        self,
        raw=None,
    ):
        if raw is None:
            raw = self._get_raw_semantic_milestone_state()

        return (
            len(self._current_valid_pick_objs(raw))
            + len(raw["valid_placed_objs"])
        )

    def _get_ordered_semantic_milestone_state(self):
        """
        Convert current valid action count into generic 3-PnP milestone flags.

        Since the three bottle moves are order-free, the generic PnP stages
        represent count progress rather than fixed bottle identity.
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
        self._lh_ever_valid_placed_objs = set()

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
        count = (
            len(self._lh_ever_picked_objs)
            + len(self._lh_ever_valid_placed_objs)
        )

        return float(count) / 6.0

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
        Update metric state once per policy timestep.

        The three movable bottles are order-free, but a place only counts if
        that bottle is in the correct type cluster.
        """
        self._lh_step_count += 1

        raw = self._get_raw_semantic_milestone_state()

        # --------------------------------------------------------------
        # Ever-picked movable bottles.
        # --------------------------------------------------------------
        for obj_name in raw["grasped_objs"]:
            self._lh_ever_picked_objs.add(obj_name)

        # Anti-flicker inference:
        # If a bottle is already in a correct cluster but the grasp was missed,
        # infer that the pick happened.
        for obj_name in raw["valid_placed_objs"]:
            self._lh_ever_picked_objs.add(obj_name)
            self._lh_ever_valid_placed_objs.add(obj_name)

        current_count = self._get_current_action_count(raw)
        self._update_stage_flags_from_count(
            current_count
        )

        unordered_count = int(
            len(self._lh_ever_picked_objs)
            + len(self._lh_ever_valid_placed_objs)
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
        # Generic count stages regress if the current valid action count drops.
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
            and self._lh_first_success_step
            is None
        ):
            self._lh_first_success_step = (
                self._lh_step_count
            )

        self._lh_prev_ordered_state = {
            name: bool(ordered_state[name])
            for name in self.get_semantic_milestone_spec().keys()
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
    # Original RoboCasa success criterion, preserved except replay-safe
    # mystery choice reconstruction
    # =====================================================================

    def _check_success(self):
        """
        Checks success for RecycleBottlesByType:
        - In each cluster, every middle bottle (including the mystery)
          must be within cluster_threshold of at least one sample bottle.
        - All bottles must be touching the dining counter.
        - The gripper must be far from every bottle.
        """
        plastic, glass = (
            self._get_bottle_groups()
        )

        cluster_threshold = 0.30

        plastic_ok = self._cluster_okay(
            plastic,
            cluster_threshold,
        )

        glass_ok = self._cluster_okay(
            glass,
            cluster_threshold,
        )

        all_names = plastic + glass

        on_table = all(
            OU.check_obj_fixture_contact(
                self,
                name,
                self.dining_counter,
            )
            for name in all_names
        )

        all_far = all(
            OU.gripper_obj_far(
                self,
                obj_name=name,
            )
            for name in all_names
        )

        return (
            plastic_ok
            and glass_ok
            and on_table
            and all_far
        )

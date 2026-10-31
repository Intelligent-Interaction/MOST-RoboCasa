from robocasa.environments.kitchen.kitchen import *


class DeliverStraw(Kitchen):
    """
    Deliver Straw: composite task for Serving Beverages activity.

    Simulates the task of delivering a straw to a glass cup on the dining counter.

    Steps:
        1. Take a straw from the drawer.
        2. Place the straw inside the glass cup on the dining counter.

    Long-horizon semantic milestones:
        M1. take_straw_from_drawer
            The straw center leaves the drawer interior.

        M2. place_straw_in_glass_cup
            The straw is placed inside the glass cup.
            Ordered dependency: M1 must already be completed when M2 happens.

    Notes on metric semantics:
        - Opening the drawer is NOT a semantic milestone.
        - Grasping the straw is NOT a semantic milestone.
        - Moving the gripper away is NOT a semantic milestone. It remains part
          of the original RoboCasa terminal success condition.
        - Unordered progress gives credit for any semantic milestone that was
          actually achieved, regardless of dependency order.
        - Ordered progress gives credit only when a milestone's completion
          event occurs after its dependencies have been completed.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        # Create an empty tracker before the normal environment initialization.
        # The tracker is re-initialized from the true simulator state in
        # _reset_internal() after every episode reset.
        self._reset_long_horizon_metric_tracker(initial_state=None)
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stool = self.register_fixture_ref(
            "stool",
            dict(id=FixtureType.STOOL),
        )
        self.dining_counter = self.register_fixture_ref(
            "dining_counter",
            dict(id=FixtureType.DINING_COUNTER, ref=self.stool),
        )
        self.drawer = self.register_fixture_ref(
            "drawer", dict(id=FixtureType.TOP_DRAWER)
        )
        self.init_robot_base_ref = self.drawer

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta[
                "lang"
            ] = "Take a straw from the drawer in front and place it inside the glass cup on the dining counter."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="straw",
                obj_groups=("straw"),
                object_scale=[1.5, 1, 2],
                placement=dict(
                    fixture=self.drawer,
                    size=(0.30, 0.20),
                    pos=(0.0, -0.25),
                ),
            )
        )
        cfgs.append(
            dict(
                name="glass_cup",
                obj_groups=("glass_cup"),
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool),
                    size=(0.25, 0.15),
                    pos=("ref", "ref"),
                ),
            )
        )
        return cfgs

    # =====================================================================
    # Semantic milestone definitions
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "drawer_open": {
                "description": "The drawer is opened.",
                "dependencies": [],
                "persistent": True,
            },
            "straw_in_hand": {
                "description": "The straw is grasped by the robot.",
                "dependencies": ["drawer_open"],
                "persistent": True,
            },
            "straw_in_glass_cup": {
                "description": "The straw is placed inside the glass cup.",
                "dependencies": ["straw_in_hand"],
                "persistent": True,
            },
        }

    def _get_semantic_milestone_state(self):
        # RoboCasa drawer fixture exposes its joint state through
        # get_door_state. For drawer tasks, an opened drawer is when the
        # drawer joint is sufficiently extended.
        drawer_state = self.drawer.get_door_state(env=self)

        drawer_open = all(
            float(joint_pos) > 0.95
            for joint_pos in drawer_state.values()
        )

        straw_in_hand = bool(
            OU.check_obj_grasped(
                self,
                "straw",
            )
        )

        straw_in_glass_cup = bool(
            OU.check_obj_in_receptacle(
                self,
                "straw",
                "glass_cup",
                th=0.5,
            )
        )

        return {
            "drawer_open": drawer_open,
            "straw_in_hand": straw_in_hand,
            "straw_in_glass_cup": straw_in_glass_cup,
        }

    # =====================================================================
    # Long-horizon metric tracker
    # =====================================================================

    def _reset_long_horizon_metric_tracker(self, initial_state=None):
        """Reset all per-episode long-horizon metric statistics."""
        spec = self.get_semantic_milestone_spec()
        names = list(spec.keys())

        self._lh_step_count = 0

        # Semantic progress curve p_t. Each policy step appends the current
        # ordered semantic progress. PRM-style metrics are computed outside
        # the task after the episode terminates.
        self._lh_progress_curve = []

        # Previous raw milestone state is needed to detect an actual semantic
        # completion event (False -> True). Initial True states are therefore
        # not incorrectly counted as progress.
        if initial_state is None:
            initial_state = {name: False for name in names}
        self._lh_prev_raw = {
            name: bool(initial_state.get(name, False)) for name in names
        }

        # UPS bookkeeping: milestone was achieved at least once, irrespective
        # of dependency order.
        self._lh_unordered_ever = {name: False for name in names}

        # OPS bookkeeping: milestone had at least one valid completion event
        # while its dependencies were already ordered-complete.
        self._lh_ordered_ever = {name: False for name in names}

        # Persistent milestones that were correctly completed but later became
        # false at least once.
        self._lh_regressed = {name: False for name in names}

        # Keep only episode-level success information here.
        # Process metrics (MP/PPL/CRA/STR/DRR/SQS) are computed from
        # _lh_progress_curve in the outer evaluator.
        self._lh_first_success_step = None

    def _reset_internal(self):
        """
        Normal RoboCasa reset plus long-horizon tracker reset.

        We snapshot the true initial milestone state after the scene / objects
        have been reset. This is important: a condition that happens to be true
        at t=0 is not treated as a completed semantic action.
        """
        super()._reset_internal()
        initial_state = self._get_semantic_milestone_state()
        self._reset_long_horizon_metric_tracker(initial_state=initial_state)

    def _get_current_ordered_progress(self, raw_state):
        """
        Compute the current ordered progress fraction.

        A persistent ordered milestone contributes only while its raw state is
        still true. A transient ordered milestone, if a future task defines
        one, stays credited once it has been validly completed.
        """
        spec = self.get_semantic_milestone_spec()
        num_milestones = len(spec)
        if num_milestones == 0:
            return 0.0

        # Sequential stage progress:
        # drawer opened -> straw picked -> straw placed in cup
        if (
            self._lh_ordered_ever["straw_in_glass_cup"]
            and raw_state["straw_in_glass_cup"]
        ):
            return 1.0

        if (
            self._lh_ordered_ever["straw_in_hand"]
            and raw_state["straw_in_hand"]
        ):
            return 2.0 / 3.0

        if (
            self._lh_ordered_ever["drawer_open"]
            and raw_state["drawer_open"]
        ):
            return 1.0 / 3.0

        return 0.0

    def _update_long_horizon_metric_tracker(self):
        """
        Update all metric statistics for one POLICY timestep.

        This method is called automatically from _post_action(), so the outer
        evaluation loop does not need to maintain milestone history, ordering,
        AUC, regression, peak progress, or completion time.
        """
        spec = self.get_semantic_milestone_spec()
        names = list(spec.keys())
        raw_state = self._get_semantic_milestone_state()

        self._lh_step_count += 1

        # A semantic completion is an actual False -> True event. This matters
        # for ordered progress. Example in another task: if a cabinet is closed
        # too early, and prerequisite object placements are completed later,
        # the earlier cabinet close must NOT retroactively receive ordered
        # credit merely because the cabinet is still closed.
        rising_edge = {
            name: bool(raw_state[name] and not self._lh_prev_raw[name])
            for name in names
        }

        # --------------------------------------------------------------
        # UPS: any semantic milestone completion event counts, regardless
        # of order.
        # --------------------------------------------------------------
        for name in names:
            if rising_edge[name]:
                self._lh_unordered_ever[name] = True

        # --------------------------------------------------------------
        # OPS: a milestone receives ordered credit only if its completion
        # event occurs when all dependencies have already received ordered
        # credit. A fixed-point loop allows two dependent milestones that
        # genuinely become complete in the same policy step to be resolved.
        # --------------------------------------------------------------
        changed = True
        while changed:
            changed = False
            for name, cfg in spec.items():
                if self._lh_ordered_ever[name]:
                    continue
                if not rising_edge[name]:
                    continue

                deps = cfg.get("dependencies", [])
                if all(self._lh_ordered_ever[dep] for dep in deps):
                    self._lh_ordered_ever[name] = True
                    changed = True

        # --------------------------------------------------------------
        # PRR: detect True -> False regression for persistent milestones
        # that have previously been validly ordered-completed.
        # --------------------------------------------------------------
        for name, cfg in spec.items():
            if not cfg.get("persistent", True):
                continue
            if not self._lh_ordered_ever[name]:
                continue

            falling_edge = bool(self._lh_prev_raw[name] and not raw_state[name])
            if falling_edge:
                self._lh_regressed[name] = True

        # Current ordered progress is intentionally allowed to decrease when a
        # persistent milestone is broken. This makes Peak Progress and
        # Progress Regression Gap meaningful.
        current_ops = self._get_current_ordered_progress(raw_state)

        # Store progress curve only. Do not compute PRM metrics inside the
        # environment. This keeps task code responsible only for semantic
        # state tracking.
        self._lh_progress_curve.append(float(current_ops))

        # Use the original RoboCasa success definition. The first success time
        # is cached because evaluation normally terminates once success is hit.
        success = bool(self._check_success())
        if success and self._lh_first_success_step is None:
            self._lh_first_success_step = self._lh_step_count

        self._lh_prev_raw = {name: bool(raw_state[name]) for name in names}

    def _post_action(self, action):
        """
        Run the normal RoboCasa post-action logic, then update metrics once for
        the newly executed policy action.
        """
        reward, done, info = super()._post_action(action)
        self._update_long_horizon_metric_tracker()
        return reward, done, info

    # =====================================================================
    # Public metric API
    # =====================================================================

    def get_long_horizon_metrics(self):
        """
        Return raw semantic information for external metric computation.

        The environment only provides:
            - SR: official RoboCasa success
            - NCT: normalized first-success timestep
            - UPS: unordered semantic completion (for appendix)
            - OPS: current ordered semantic progress (used to build curve)
            - progress_curve: p_0 ... p_T for PRM-style metrics

        PRM-style process metrics (MP, MC, PPL, CRA, STR, FNS, DRR, SQS)
        should be computed after the episode terminates in the evaluator.
        """
        spec = self.get_semantic_milestone_spec()
        num_milestones = len(spec)

        if num_milestones == 0:
            ups = 0.0
        else:
            ups = sum(self._lh_unordered_ever.values()) / float(num_milestones)

        raw_state = self._get_semantic_milestone_state()
        ops = self._get_current_ordered_progress(raw_state)

        sr = float(self._lh_first_success_step is not None)

        if self._lh_first_success_step is None:
            nct = -1.0
        else:
            horizon = float(getattr(self, "horizon", 0))
            if horizon > 0:
                nct = float(self._lh_first_success_step) / horizon
            else:
                nct = float(self._lh_first_success_step)

        return {
            "SR": sr,
            "NCT": nct,
            "UPS": float(ups),
            "OPS": float(ops),
            "progress_curve": list(self._lh_progress_curve),
        }

    def _check_success(self):
        # Keep the original RoboCasa success criterion exactly unchanged.
        straw_in_glass_cup = OU.check_obj_in_receptacle(
            self, "straw", "glass_cup", th=0.5
        )
        gripper_far = OU.gripper_obj_far(self, obj_name="straw")
        return straw_in_glass_cup and gripper_far

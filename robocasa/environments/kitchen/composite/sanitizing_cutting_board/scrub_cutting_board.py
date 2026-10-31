from robocasa.environments.kitchen.kitchen import *


class ScrubCuttingBoard(Kitchen):
    """
    Scrub Cutting Board: composite task for Sanitizing Cutting Board activity.

    Simulates scrubbing a cutting board using a sponge in a circular motion to ensure
    thorough cleaning and removal of food particles.

    Steps:
        1. Pick up the sponge from the counter.
        2. Scrub the cutting board.
        3. Once the scrubbing is complete, place the sponge down.

    Long-horizon semantic milestones
    ---------------------------------
    M1. sponge_in_hand
        The sponge is grasped by the robot.

    M2. scrub_cutting_board
        The cutting board has been sufficiently scrubbed according to the
        task's original simulator-grounded criteria:
            board_contact_timer >= 5
            AND sweep_range >= 0.1

    M3. release_sponge
        After the board has been sufficiently scrubbed, the sponge has been
        released and the gripper has moved sufficiently far from it.

    Dependency chain:
        M1 -> M2 -> M3

    Important:
        There is intentionally NO semantic requirement on the sponge's final
        position. After scrubbing, the sponge may:
            - remain on the cutting board,
            - be placed somewhere else on the counter,
            - be moved away from the board and released elsewhere.

        All are valid as long as the original success criterion's
        gripper_obj_far condition is satisfied.

    The sponge's final position is intentionally unconstrained. Release is
    checked only through "not grasped" and gripper-far.
    """

    def __init__(self, *args, **kwargs):
        self.board_contact_positions = []
        self.board_contact_timer = 0
        self.sponge_contact_height = None

        self._lh_sponge_ever_grasped = False
        self._lh_prev_sponge_grasped = False
        self._lh_prev_gripper_far = True

        self._reset_long_horizon_metric_tracker(initial_state=None)
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta["lang"] = (
                "Pick up the sponge from the counter and clean the cutting board by briefly "
                "scrubbing or pressing down on the cutting board. Once finished, release the sponge."
            )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.board_contact_positions = []
        self.board_contact_timer = 0
        self.sponge_contact_height = None

        self._lh_sponge_ever_grasped = False
        self._lh_prev_sponge_grasped = False
        self._lh_prev_gripper_far = True

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="cutting_board",
                obj_groups="cutting_board",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.5, 0.5),
                    pos=("ref", -1.0),
                    rotation=0,
                ),
            )
        )

        cfgs.append(
            dict(
                name="sponge",
                obj_groups="sponge",
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="cutting_board",
                    size=(0.3, 0.3),
                    pos=(0, -1.0),
                ),
            )
        )

        return cfgs


    def _check_sponge_grasped(self):
        """
        Local sponge grasp check.

        Use a slightly looser gripper-close threshold than RoboCasa's default
        0.035. This reduces missed pick frames for the sponge without changing
        global object_utils behavior.
        """
        try:
            return bool(
                OU.check_obj_grasped(
                    self,
                    "sponge",
                    threshold=0.10,
                )
            )
        except TypeError:
            return bool(
                OU.check_obj_grasped(
                    self,
                    "sponge",
                )
            )

    def update_state(self):
        sponge_grasped = self._check_sponge_grasped()
        super().update_state()
        sponge_body = self.obj_body_id["sponge"]
        board_body = self.obj_body_id["cutting_board"]

        sponge_geoms = [
            self.sim.model.geom_id2name(gid)
            for gid in range(self.sim.model.ngeom)
            if self.sim.model.geom_bodyid[gid] == sponge_body
        ]
        board_geoms = [
            self.sim.model.geom_id2name(gid)
            for gid in range(self.sim.model.ngeom)
            if self.sim.model.geom_bodyid[gid] == board_body
        ]

        contact = self.check_contact(sponge_geoms, board_geoms)
        sponge_pos = np.array(self.sim.data.body_xpos[sponge_body])
        sponge_xy = sponge_pos[:2]

        if contact and sponge_grasped:
            new_contact = True
            for recorded_xy in self.board_contact_positions:
                if np.linalg.norm(sponge_xy - recorded_xy) < 0.02:
                    new_contact = False
                    break
            if new_contact:
                self.board_contact_positions.append(sponge_xy)
                self.board_contact_timer += 1

    # =====================================================================
    # Task-specific semantic milestone definitions
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "sponge_in_hand": {
                "description": "The sponge is grasped by the robot gripper.",
                "dependencies": [],
                "persistent": False,
            },
            "scrub_cutting_board": {
                "description": "The cutting board has been sufficiently scrubbed.",
                "dependencies": ["sponge_in_hand"],
                "persistent": False,
            },
            "release_sponge": {
                "description": "The sponge is released after scrubbing.",
                "dependencies": ["scrub_cutting_board"],
                "persistent": False,
            },
        }


    def _get_sweep_range(self):
        """
        Compute the same sweep-range quantity used by the original success
        criterion.
        """
        if not self.board_contact_positions:
            return 0.0

        positions = np.array(self.board_contact_positions)
        min_xy = positions.min(axis=0)
        max_xy = positions.max(axis=0)
        return float(np.linalg.norm(max_xy - min_xy))

    def _is_scrub_complete(self):
        """
        Exact semantic scrubbing condition from the original task.
        """
        sweep_range = self._get_sweep_range()
        return bool(
            self.board_contact_timer >= 5
            and sweep_range >= 0.1
        )

    def _get_raw_semantic_milestone_state(self):
        sponge_in_hand = bool(
            self._check_sponge_grasped()
        )

        scrub_complete = self._is_scrub_complete()

        release_sponge = bool(
            self._lh_sponge_ever_grasped
            and not sponge_in_hand
            and OU.gripper_obj_far(self, "sponge", th=0.15)
        )

        return {
            "sponge_in_hand": sponge_in_hand,
            "scrub_cutting_board": scrub_complete,
            "release_sponge": release_sponge,
        }



    def _get_ordered_semantic_milestone_state(self):
        """
        Current ordered-valid milestone states.

        sponge_in_hand is current/superseded:
            - valid while the sponge is currently in hand;
            - also valid after scrubbing or release, because those stages
              supersede the initial pick.

        If the sponge is picked and then dropped before any valid scrubbing,
        the pick contribution disappears from current OPS.
        """
        raw = self._get_raw_semantic_milestone_state()

        scrub_valid = bool(
            self._lh_ordered_ever["scrub_cutting_board"]
            and raw["scrub_cutting_board"]
        )

        release_valid = bool(
            self._lh_ordered_ever["release_sponge"]
            and raw["release_sponge"]
            and scrub_valid
        )

        sponge_pick_valid = bool(
            self._lh_ordered_ever["sponge_in_hand"]
            and (
                raw["sponge_in_hand"]
                or scrub_valid
                or release_valid
            )
        )

        return {
            "sponge_in_hand": sponge_pick_valid,
            "scrub_cutting_board": scrub_valid,
            "release_sponge": release_valid,
        }


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

        self._lh_ops_auc_sum = 0.0
        self._lh_peak_progress = 0.0
        self._lh_first_success_step = None
        self._lh_progress_curve = []

    def _reset_internal(self):
        """
        Normal RoboCasa reset plus metric initialization from the true reset
        state.
        """
        super()._reset_internal()

        self._lh_sponge_ever_grasped = bool(
            self._check_sponge_grasped()
        )
        self._lh_prev_sponge_grasped = bool(
            self._check_sponge_grasped()
        )
        self._lh_prev_gripper_far = bool(
            OU.gripper_obj_far(self, "sponge", th=0.15)
        )

        initial_raw = self._get_raw_semantic_milestone_state()
        self._reset_long_horizon_metric_tracker(initial_state=initial_raw)

    def _get_current_ordered_progress(self):
        ordered_state = self._get_ordered_semantic_milestone_state()
        return (
            sum(
                int(bool(value))
                for value in ordered_state.values()
            )
            / 3.0
        )

    def _update_long_horizon_metric_tracker(self):
        self._lh_step_count += 1

        sponge_in_hand = bool(
            self._check_sponge_grasped()
        )

        if sponge_in_hand:
            self._lh_sponge_ever_grasped = True

        raw = self._get_raw_semantic_milestone_state()

        for name in self.get_semantic_milestone_spec():
            if raw[name]:
                self._lh_unordered_ever[name] = True

        for name, cfg in self.get_semantic_milestone_spec().items():
            if raw[name] and not self._lh_prev_raw[name]:
                if all(self._lh_ordered_ever[d] for d in cfg["dependencies"]):
                    self._lh_ordered_ever[name] = True

        ops = self._get_current_ordered_progress()
        self._lh_progress_curve.append(float(ops))

        if self._check_success() and self._lh_first_success_step is None:
            self._lh_first_success_step = self._lh_step_count

        self._lh_prev_raw = {
            name: bool(raw[name])
            for name in self.get_semantic_milestone_spec()
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
    # Original RoboCasa success criterion, preserved
    # =====================================================================

    def _check_success(self):
        gripper_obj_far = OU.gripper_obj_far(self, "sponge", th=0.15)

        sweep_range = 0.0
        if self.board_contact_positions:
            positions = np.array(self.board_contact_positions)
            min_xy = positions.min(axis=0)
            max_xy = positions.max(axis=0)
            sweep_range = np.linalg.norm(max_xy - min_xy)

        return (
            (self.board_contact_timer >= 5)
            and (sweep_range >= 0.1)
            and gripper_obj_far
        )
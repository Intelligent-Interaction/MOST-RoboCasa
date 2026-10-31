from robocasa.environments.kitchen.kitchen import *


class KettleBoiling(Kitchen):
    """
    Kettle Boiling: composite task for Brewing activity.

    Simulates the task of boiling water in a kettle.

    Steps:
        Take the kettle from the counter and place it on a stove burner.
        Turn the burner on.

    Long-horizon semantic milestones:
        M1. place_kettle_on_burner
            The kettle is in contact with the stove and centered within 0.15 m
            of a valid burner site.

        M2. turn_kettle_burner_on
            The burner corresponding to the kettle placement is turned on.
            Ordered dependency: M1.

    Metric semantics:
        - UPS ignores ordering:
            M1 counts once the kettle has ever been placed on a valid burner.
            M2 counts once the episode has both:
              (a) placed the kettle on some burner, and
              (b) turned that same burner on,
            regardless of which happened first.

        - OPS enforces ordering:
            M2 receives ordered credit only when the corresponding burner has
            an actual off -> on transition AFTER the kettle was already placed
            on that burner.

        - Both M1 and M2 are persistent physical goal states. Moving the kettle
          away from the burner or turning the valid burner off can therefore
          produce progress regression.

        - Gripper distance is not a semantic milestone. It remains part of the
          original terminal success criterion.
    """

    def __init__(self, *args, **kwargs):
        self._lh_prev_burners_on = {}
        self._lh_kettle_burners_ever = set()
        self._lh_burners_on_ever = set()
        self._lh_ordered_burner_locations = set()

        self._reset_long_horizon_metric_tracker(initial_state=None)
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.init_robot_base_ref = self.stove
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=(0.2, 0.2))
        )

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=("kettle_non_electric"),
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(0.35, 0.35),
                    pos=("ref", -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="stove_distr",
                obj_groups=("pan", "pot"),
                placement=dict(
                    fixture=self.stove,
                    # ensure_object_boundary_in_range=False because the pans handle is a part of the
                    # bounding box making it hard to place it if set to True
                    ensure_object_boundary_in_range=False,
                    size=(0.02, 0.02),
                    # apply rotations so the handle doesnt stick too much
                    rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                ),
            )
        )

        return cfgs

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta[
                "lang"
            ] = "Pick the kettle from the counter and place it on a stove burner. Then turn the burner on."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        valid_knobs = self.stove.get_knobs_state(env=self).keys()
        for knob in valid_knobs:
            self.stove.set_knob_state(mode="off", knob=knob, env=self, rng=self.rng)

        self._lh_prev_burners_on = {}
        self._lh_kettle_burners_ever = set()
        self._lh_burners_on_ever = set()
        self._lh_ordered_burner_locations = set()

    # =====================================================================
    # Task-specific semantic milestone definitions
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "kettle_in_hand": {
                "description": "The kettle is grasped by the robot gripper.",
                "dependencies": [],
                "persistent": True,
            },
            "place_kettle_on_burner": {
                "description": "The kettle is placed on a valid stove burner site.",
                "dependencies": [],
                "persistent": True,
            },
            "turn_kettle_burner_on": {
                "description": (
                    "The burner corresponding to the kettle placement is turned on."
                ),
                "dependencies": ["place_kettle_on_burner"],
                "persistent": True,
            },
        }

    def _get_kettle_burner_location(self):
        """
        Return the valid burner location currently occupied by the kettle.

        This mirrors the spatial part of the original success criterion:
        the kettle must contact the stove and lie within 0.15 m of a valid
        burner site.
        """
        obj_on_stove = OU.check_obj_fixture_contact(self, "obj", self.stove)
        if not obj_on_stove:
            return None

        kettle = self.objects["obj"]
        kettle_pos = np.array(
            self.sim.data.body_xpos[self.obj_body_id[kettle.name]]
        )[0:2]

        candidates = []
        for location, site in self.stove.burner_sites.items():
            if site is None:
                continue

            burner_pos = np.array(
                self.sim.data.get_site_xpos(site.get("name"))
            )[0:2]
            dist = float(np.linalg.norm(burner_pos - kettle_pos))

            if dist < 0.15:
                candidates.append((dist, location))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _get_burners_on_state(self):
        """
        Return {burner_location: bool_is_on} for valid stove burners.
        """
        knobs_state = self.stove.get_knobs_state(env=self)
        states = {}

        for location, site in self.stove.burner_sites.items():
            if site is None or location not in knobs_state:
                continue

            states[location] = bool(
                self.stove.is_burner_on(env=self, burner_loc=location)
            )

        return states

    def _get_semantic_milestone_state(self):
        """
        Current kettle state.

        Grasp uses RoboCasa's native contact-based check. Progress itself is
        stage based because the states are mutually exclusive:
        in hand -> on burner -> burner on.
        """
        kettle_burner = self._get_kettle_burner_location()
        burners_on = self._get_burners_on_state()

        kettle_in_hand = bool(
            OU.check_obj_grasped(
                self,
                "obj",
            )
        )

        place_kettle = bool(
            kettle_burner is not None and not kettle_in_hand
        )

        turn_valid_burner_on = bool(
            kettle_burner is not None
            and bool(burners_on.get(kettle_burner, False))
            and not kettle_in_hand
        )

        return {
            "kettle_in_hand": kettle_in_hand,
            "place_kettle_on_burner": place_kettle,
            "turn_kettle_burner_on": turn_valid_burner_on,
        }

    # =====================================================================
    # Per-episode long-horizon metric tracker
    # =====================================================================

    def _reset_long_horizon_metric_tracker(self, initial_state=None):
        spec = self.get_semantic_milestone_spec()
        names = list(spec.keys())

        self._lh_step_count = 0
        self._lh_progress_curve = []

        if initial_state is None:
            initial_state = {name: False for name in names}

        self._lh_prev_raw = {
            name: bool(initial_state.get(name, False)) for name in names
        }

        self._lh_unordered_ever = {name: False for name in names}
        self._lh_ordered_ever = {name: False for name in names}
        self._lh_regressed = {name: False for name in names}

        self._lh_first_success_step = None

    def _reset_internal(self):
        super()._reset_internal()

        self._lh_prev_burners_on = self._get_burners_on_state()
        self._lh_kettle_burners_ever = set()
        self._lh_burners_on_ever = {
            loc for loc, is_on in self._lh_prev_burners_on.items() if is_on
        }
        self._lh_ordered_burner_locations = set()

        initial_state = self._get_semantic_milestone_state()
        self._reset_long_horizon_metric_tracker(initial_state=initial_state)

    def _get_current_ordered_progress(self, raw_state):
        """
        Stage-based progress for mutually exclusive kettle states.
        """
        if (
            self._lh_ordered_ever["turn_kettle_burner_on"]
            and raw_state["turn_kettle_burner_on"]
        ):
            return 1.0

        if (
            self._lh_ordered_ever["place_kettle_on_burner"]
            and raw_state["place_kettle_on_burner"]
        ):
            return 2.0 / 3.0

        if (
            self._lh_ordered_ever["kettle_in_hand"]
            and raw_state["kettle_in_hand"]
        ):
            return 1.0 / 3.0

        return 0.0

    def _update_long_horizon_metric_tracker(self):
        """
        Update all long-horizon metrics for one executed policy timestep.

        UPS:
            Same burner has been used by the kettle and turned on at some point,
            regardless of temporal order.

        OPS:
            Kettle must already be validly placed, then that same burner must
            undergo an off -> on transition.
        """
        self._lh_step_count += 1

        kettle_burner = self._get_kettle_burner_location()
        burners_on = self._get_burners_on_state()

        # --------------------------------------------------------------
        # UPS: unordered semantic completion.
        # --------------------------------------------------------------
        if kettle_burner is not None:
            self._lh_kettle_burners_ever.add(kettle_burner)

        current_on_locations = {
            loc for loc, is_on in burners_on.items() if is_on
        }
        self._lh_burners_on_ever.update(current_on_locations)

        if self._lh_kettle_burners_ever:
            self._lh_unordered_ever["place_kettle_on_burner"] = True

        if self._lh_kettle_burners_ever.intersection(self._lh_burners_on_ever):
            self._lh_unordered_ever["turn_kettle_burner_on"] = True

        # --------------------------------------------------------------
        # OPS M1: kettle placement.
        # --------------------------------------------------------------
        prev_m1 = bool(self._lh_prev_raw["place_kettle_on_burner"])
        m1_now = kettle_burner is not None

        if (
            not self._lh_ordered_ever["place_kettle_on_burner"]
            and self._lh_ordered_ever["kettle_in_hand"]
            and m1_now
            and not prev_m1
        ):
            self._lh_ordered_ever["place_kettle_on_burner"] = True

        # --------------------------------------------------------------
        # OPS M2: corresponding burner must be turned on AFTER placement.
        # --------------------------------------------------------------
        newly_turned_on = {
            loc
            for loc, is_on in burners_on.items()
            if is_on and not bool(self._lh_prev_burners_on.get(loc, False))
        }

        if (
            not self._lh_ordered_ever["turn_kettle_burner_on"]
            and self._lh_ordered_ever["place_kettle_on_burner"]
            and prev_m1
            and kettle_burner is not None
            and kettle_burner in newly_turned_on
        ):
            self._lh_ordered_ever["turn_kettle_burner_on"] = True
            self._lh_ordered_burner_locations.add(kettle_burner)

        raw_state = self._get_semantic_milestone_state()

        # --------------------------------------------------------------
        # OPS M0: kettle is grasped.
        # --------------------------------------------------------------
        if raw_state["kettle_in_hand"]:
            self._lh_ordered_ever["kettle_in_hand"] = True
            self._lh_unordered_ever["kettle_in_hand"] = True

        # --------------------------------------------------------------
        # PRR: persistent ordered milestones later becoming invalid.
        # --------------------------------------------------------------
        spec = self.get_semantic_milestone_spec()
        for name, cfg in spec.items():
            if not cfg.get("persistent", True):
                continue
            if not self._lh_ordered_ever[name]:
                continue

            falling_edge = bool(self._lh_prev_raw[name] and not raw_state[name])
            if falling_edge:
                self._lh_regressed[name] = True

        current_ops = self._get_current_ordered_progress(raw_state)
        self._lh_progress_curve.append(float(current_ops))

        success = bool(self._check_success())
        if success and self._lh_first_success_step is None:
            self._lh_first_success_step = self._lh_step_count

        self._lh_prev_raw = {
            name: bool(raw_state[name]) for name in spec.keys()
        }
        self._lh_prev_burners_on = {
            loc: bool(is_on) for loc, is_on in burners_on.items()
        }

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        self._update_long_horizon_metric_tracker()
        return reward, done, info

    # =====================================================================
    # Public metric API
    # =====================================================================

    def get_long_horizon_metrics(self):
        spec = self.get_semantic_milestone_spec()
        num_milestones = len(spec)

        ups = (
            sum(self._lh_unordered_ever.values()) / float(num_milestones)
            if num_milestones > 0 else 0.0
        )

        raw_state = self._get_semantic_milestone_state()
        ops = self._get_current_ordered_progress(raw_state)

        sr = float(self._lh_first_success_step is not None)

        if self._lh_first_success_step is None:
            nct = -1.0
        else:
            horizon = float(getattr(self, "horizon", 0))
            nct = (
                float(self._lh_first_success_step) / horizon
                if horizon > 0 else float(self._lh_first_success_step)
            )

        return {
            "SR": sr,
            "UPS": float(ups),
            "OPS": float(ops),
            "NCT": float(nct),
            "progress_curve": list(self._lh_progress_curve),
        }

    # =====================================================================
    # Original RoboCasa success criterion, preserved
    # =====================================================================

    def _check_success(self):
        """
        Check if the kettle is placed on the stove burner and the burner is turned on.
        """
        knobs_state = self.stove.get_knobs_state(env=self)
        kettle = self.objects["obj"]
        kettle_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[kettle.name]])[
            0:2
        ]
        obj_on_stove = OU.check_obj_fixture_contact(self, "obj", self.stove)
        if obj_on_stove:
            for location, site in self.stove.burner_sites.items():
                if site is not None:
                    burner_pos = np.array(
                        self.sim.data.get_site_xpos(site.get("name"))
                    )[0:2]
                    dist = np.linalg.norm(burner_pos - kettle_pos)

                    kettle_on_site = dist < 0.15
                    burner_on = (
                        self.stove.is_burner_on(env=self, burner_loc=location)
                        if location in knobs_state
                        else False
                    )

                    if kettle_on_site and burner_on and OU.gripper_obj_far(self):
                        return True

        return False
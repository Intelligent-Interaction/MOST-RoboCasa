from robocasa.environments.kitchen.kitchen import *


class GetToastedBread(Kitchen):
    """
    Get Toasted Bread: composite task for Toasting Bread activity.

    Steps:
        1. Pull the toaster lever down and wait for the toaster to finish toasting the bread.
        2. Once the lever pops up, pull the bread out.
        3. Place the bread on the plate on the dining counter.

    Long-horizon semantic milestones:
        M1. start_toaster_with_bread
            The toaster is activated while the bread is still in a toaster slot.

        M2. finish_toasting_cycle
            The activated toaster cycle finishes while the bread remains in the
            same toaster slot. Ordered dependency: M1.

        M3. take_bread_out_of_toaster
            The bread is no longer in contact with any toaster slot floor.
            Ordered dependency: M2.

        M4. place_bread_on_plate
            The bread is on the plate on the dining counter.
            Ordered dependency: M3.

    Notes:
        - Pulling / touching the lever itself is not a separate semantic milestone;
          the meaningful task-level state change is that the toaster actually starts.
        - Waiting is represented by M2 because the toaster exposes a simulator-ground-
          truth on->off cycle. Therefore no wall-clock heuristic is required.
        - Gripper distance is not a semantic milestone. It remains part of the
          original terminal success condition.
        - UPS credits any milestone completion event regardless of order.
        - OPS credits a milestone only when its completion event occurs after all
          ordered dependencies have been completed.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        # Task-semantic history. These are reset again in _setup_scene().
        self.toaster_on = False
        self._toast_cycle_started = False
        self._toast_cycle_finished = False
        self._toast_active_slot = None
        self._toast_initial_slot = None
        self._prev_toaster_turned_on = {}
        self._prev_toaster_lever = {}

        # RoboCasa Toaster.update_state() turns a slot on when lever >= 0.90
        # and resets cooldown when lever <= 0.70. Use the same thresholds as
        # robust milestone evidence instead of requiring slot contact in the
        # exact same frame as turned_on=True.
        self._toast_lever_on_threshold = 0.90
        self._toast_lever_off_threshold = 0.70

        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool)
        )
        self.toaster = self.register_fixture_ref(
            "toaster", dict(id=FixtureType.TOASTER)
        )
        self.init_robot_base_ref = self.toaster

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta["lang"] = (
                "Start the toaster. "
                "Once the lever pops up, take the bread to the plate on the dining counter."
            )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

        # Keep the original task variable.
        self.toaster_on = False

        # Extra history needed to distinguish "started" from "finished".
        self._toast_cycle_started = False
        self._toast_cycle_finished = False
        self._toast_active_slot = None
        self._toast_initial_slot = None
        self._prev_toaster_turned_on = {}
        self._prev_toaster_lever = {}

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=("sandwich_bread"),
                object_scale=[1, 1, 1.15],
                rotate_upright=True,
                placement=dict(
                    fixture=self.toaster,
                    rotation=(0, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.50, 0.30),
                    pos=(0, "ref"),
                ),
            )
        )

        return cfgs

    # =====================================================================
    # Clean long-horizon milestone tracking
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "press_toaster_button": {
                "dependencies": [],
                "persistent": False,
            },
            "toaster_finished": {
                "dependencies": ["press_toaster_button"],
                "persistent": False,
            },
            "pick_toast": {
                "dependencies": ["toaster_finished"],
                "persistent": False,
            },
            "toast_in_plate": {
                "dependencies": ["pick_toast"],
                "persistent": True,
            },
        }

    def _get_bread_slot(self):
        for slot_pair in self.toaster.get_state(self).keys():
            if self.toaster.check_slot_contact(
                self,
                "obj",
                slot_pair=slot_pair,
            ):
                return slot_pair
        return None

    def _get_toaster_state_maps(self):
        toaster_state = self.toaster.get_state(self)

        current_on = {
            k: bool(v["turned_on"])
            for k, v in toaster_state.items()
        }
        current_lever = {
            k: float(v.get("lever", 0.0))
            for k, v in toaster_state.items()
        }

        return toaster_state, current_on, current_lever

    def _get_candidate_toast_slots(self, current_slot):
        """
        Candidate slots for the active toasting cycle.

        Current slot contact can flicker because check_slot_contact requires
        direct contact with the slot floor. Since bread is initialized inside
        the toaster, keep the reset-time slot as a stable fallback.
        """
        candidates = []

        for slot in [
            current_slot,
            self._toast_initial_slot,
            self._toast_active_slot,
        ]:
            if slot is not None and slot not in candidates:
                candidates.append(slot)

        return candidates

    def _slot_has_start_evidence(self, slot, current_on, current_lever):
        return bool(
            current_on.get(slot, False)
            or current_lever.get(slot, 0.0) >= self._toast_lever_on_threshold
        )

    def _slot_has_finish_evidence(self, slot, current_on, current_lever):
        old_on = self._prev_toaster_turned_on.get(
            slot,
            False,
        )
        new_on = current_on.get(
            slot,
            False,
        )

        old_lever = self._prev_toaster_lever.get(
            slot,
            0.0,
        )
        new_lever = current_lever.get(
            slot,
            0.0,
        )

        finished_by_turned_on = bool(
            old_on
            and not new_on
        )
        finished_by_lever_pop = bool(
            old_lever >= self._toast_lever_on_threshold
            and new_lever <= self._toast_lever_off_threshold
        )

        return bool(
            finished_by_turned_on
            or finished_by_lever_pop
        )

    def _get_semantic_milestone_state(self):
        _, current_on, current_lever = self._get_toaster_state_maps()

        current_slot = self._get_bread_slot()

        if (
            self._toast_initial_slot is None
            and current_slot is not None
        ):
            self._toast_initial_slot = current_slot

        candidate_slots = self._get_candidate_toast_slots(
            current_slot=current_slot,
        )

        # --------------------------------------------------------------
        # M1: robust toaster start detection.
        #
        # Previous version required:
        #   current_slot is not None and current_on[current_slot]
        #
        # This can miss real starts when slot-floor contact flickers. Here we
        # allow either turned_on=True or lever>=0.90 on the current / initial
        # toast slot.
        # --------------------------------------------------------------
        if not self._toast_cycle_started:
            for slot in candidate_slots:
                if self._slot_has_start_evidence(
                    slot,
                    current_on,
                    current_lever,
                ):
                    self._toast_cycle_started = True
                    self._toast_active_slot = slot
                    break

            # Extra fallback: if the bread slot is temporarily unreadable but
            # exactly one slot has clear toaster-start evidence, use that slot.
            if not self._toast_cycle_started:
                started_slots = [
                    slot
                    for slot in current_on.keys()
                    if self._slot_has_start_evidence(
                        slot,
                        current_on,
                        current_lever,
                    )
                ]
                if len(started_slots) == 1:
                    self._toast_cycle_started = True
                    self._toast_active_slot = started_slots[0]

        # --------------------------------------------------------------
        # M2: robust finish detection.
        #
        # The official toaster fixture turns on at lever>=0.90 and the lever
        # pops / cooldown begins once the cycle ends. Use both turned_on falling
        # edge and lever high->low edge.
        # --------------------------------------------------------------
        if (
            self._toast_cycle_started
            and not self._toast_cycle_finished
            and self._toast_active_slot is not None
        ):
            if self._slot_has_finish_evidence(
                self._toast_active_slot,
                current_on,
                current_lever,
            ):
                self._toast_cycle_finished = True

        self._prev_toaster_turned_on = current_on
        self._prev_toaster_lever = current_lever

        return {
            "press_toaster_button": bool(
                self._toast_cycle_started
            ),
            "toaster_finished": bool(
                self._toast_cycle_finished
            ),
            "pick_toast": bool(
                OU.check_obj_grasped(self, "obj")
            ),
            "toast_in_plate": bool(
                OU.check_obj_in_receptacle(
                    self,
                    "obj",
                    "plate",
                )
            ),
        }

    def _reset_internal(self):
        super()._reset_internal()

        self._toast_cycle_started = False
        self._toast_cycle_finished = False
        self._toast_active_slot = None
        self._toast_initial_slot = self._get_bread_slot()

        _, current_on, current_lever = self._get_toaster_state_maps()

        self._prev_toaster_turned_on = current_on
        self._prev_toaster_lever = current_lever

        names = list(self.get_semantic_milestone_spec().keys())

        self._lh_prev = {
            k: False for k in names
        }
        self._lh_ordered = {
            k: False for k in names
        }
        self._lh_ever = {
            k: False for k in names
        }
        self._lh_progress_curve = []
        self._lh_step_count = 0
        self._lh_first_success_step = None

    def _update_long_horizon_metric_tracker(self):
        self._lh_step_count += 1

        raw = self._get_semantic_milestone_state()

        for k, v in raw.items():
            if v:
                self._lh_ever[k] = True

        for k, cfg in self.get_semantic_milestone_spec().items():
            if raw[k] and not self._lh_prev[k]:
                if all(
                    self._lh_ordered[d]
                    for d in cfg["dependencies"]
                ):
                    self._lh_ordered[k] = True

        # Stage-based progress:
        # toaster on -> finished -> picked -> placed on plate
        # These are sequential states, not independent additive rewards.
        if (
            self._lh_ordered["toast_in_plate"]
            and raw["toast_in_plate"]
        ):
            progress = 1.0
        elif (
            self._lh_ordered["pick_toast"]
            and raw["pick_toast"]
        ):
            progress = 0.75
        elif (
            self._lh_ordered["toaster_finished"]
            and raw["toaster_finished"]
        ):
            progress = 0.5
        elif (
            self._lh_ordered["press_toaster_button"]
            and raw["press_toaster_button"]
        ):
            progress = 0.25
        else:
            progress = 0.0

        self._lh_progress_curve.append(progress)

        if (
            self._check_success()
            and self._lh_first_success_step is None
        ):
            self._lh_first_success_step = self._lh_step_count

        self._lh_prev = raw

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        self._update_long_horizon_metric_tracker()
        return reward, done, info

    def get_long_horizon_metrics(self):
        if self._lh_first_success_step is None:
            nct = -1.0
        else:
            horizon = float(getattr(self, "horizon", 0))
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
            "OPS": (
                self._lh_progress_curve[-1]
                if self._lh_progress_curve
                else 0.0
            ),
            "UPS": (
                sum(self._lh_ever.values())
                / float(len(self._lh_ever))
            ),
            "progress_curve": list(
                self._lh_progress_curve
            ),
        }

    # =====================================================================
    # Original RoboCasa success criterion, preserved
    # =====================================================================

    def _check_success(self):
        toast_slot = 0
        toast_in_slot = False
        for slot_pair in range(len(self.toaster.get_state(self).keys())):
            if self.toaster.check_slot_contact(self, "obj", slot_pair=slot_pair):
                toast_slot = slot_pair
                toast_in_slot = True
                break

        if (
            toast_in_slot
            and self.toaster.get_state(self, slot_pair=toast_slot)["turned_on"]
        ):
            self.toaster_on = True

        return (
            self.toaster_on
            and OU.check_obj_in_receptacle(self, "obj", "plate")
            and OU.gripper_obj_far(self, "obj")
        )
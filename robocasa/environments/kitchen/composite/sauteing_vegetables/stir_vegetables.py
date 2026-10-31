from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.object_utils as OU


class StirVegetables(Kitchen):
    """
    Stir Vegetables.

    Task:
        1. Pick one vegetable.
        2. Place it in the pot.
        3. Pick the other vegetable.
        4. Place it in the pot.
        5. Pick the spatula.
        6. Stir the vegetables.

    The two vegetable pick-place branches are order-free:
        veg1_in_hand -> veg1_in_pot
        veg2_in_hand -> veg2_in_pot

    The spatula and stir stages happen after both vegetables are validly in the
    pot.

    Original RoboCasa _check_success() is preserved.
    """

    def __init__(self, knob_id="random", *args, **kwargs):
        self.knob_id = knob_id
        self._reset_long_horizon_metric_tracker()
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.stove = self.register_fixture_ref(
            "stove",
            dict(id=FixtureType.STOVE),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.stove),
        )

        self.init_robot_base_ref = self.stove

        if "refs" in self._ep_meta:
            self.knob = self._ep_meta["refs"]["knob"]
        else:
            valid_knobs = [
                k
                for (k, v) in self.stove.knob_joints.items()
                if v is not None
            ]

            if self.knob_id == "random":
                self.knob = self.rng.choice(
                    list(valid_knobs)
                )
            else:
                assert self.knob_id in valid_knobs
                self.knob = self.knob_id

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        veg1_lang = self.get_obj_lang("veg1")
        veg2_lang = self.get_obj_lang("veg2")

        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(
                self.novel_instructions
            ).format(
                veg1_lang=veg1_lang,
                veg2_lang=veg2_lang,
            )
        else:
            ep_meta["lang"] = (
                f"Put the {veg1_lang} and {veg2_lang} in the pot. "
                f"Retrieve the spatula and lightly stir the vegetables in the pot."
            )

        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["knob"] = self.knob

        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

        self.success_time = 0

        self.stove.set_knob_state(
            env=self,
            rng=self.rng,
            knob=self.knob,
            mode="on",
        )

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="spatula",
                obj_groups="spatula",
                object_scale=[1, 1, 2],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.stove),
                    size=(0.5, 0.5),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="pot",
                obj_groups="pot",
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    sample_region_kwargs=dict(locs=[self.knob]),
                    size=(0.05, 0.05),
                ),
            )
        )

        def veg_cfg(name, offset):
            return dict(
                name=name,
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                        top_size=(0.3, 0.3),
                    ),
                    size=(0.3, 0.3),
                    pos=("ref", offset),
                ),
            )

        cfgs.append(
            veg_cfg(
                "veg1",
                -1.0,
            )
        )
        cfgs.append(
            veg_cfg(
                "veg2",
                -0.7,
            )
        )

        return cfgs

    def update_state(self):
        super().update_state()

        veg1_in_pot = OU.check_obj_in_receptacle(
            self,
            "veg1",
            "pot",
        )
        veg2_in_pot = OU.check_obj_in_receptacle(
            self,
            "veg2",
            "pot",
        )

        pot_loc = (
            self.stove.check_obj_location_on_stove(
                env=self,
                obj_name="pot",
                threshold=0.15,
            )
            == self.knob
        )

        spatula_grasped = OU.check_obj_grasped(
            self,
            "spatula",
        )

        if spatula_grasped:
            objects_stirred = self._detect_stirring(
                ["veg1", "veg2"]
            )

            if (
                pot_loc
                and veg1_in_pot
                and veg2_in_pot
                and objects_stirred
            ):
                self.success_time += 1

    def _check_success(self):
        return self.success_time >= 5

    def _detect_stirring(
        self,
        obj_names,
        movement_threshold=0.05,
    ):
        all_objects_stirred = True

        spatula = self.objects["spatula"]
        contact_with_spatula = False

        for obj_name in obj_names:
            obj = self.objects[obj_name]

            obj_pos = np.array(
                self.sim.data.body_xpos[
                    self.obj_body_id[obj.name]
                ]
            )

            prev_obj_pos = getattr(
                self,
                f"prev_obj_pos_{obj_name}",
                obj_pos,
            )

            movement_vector = (
                obj_pos - prev_obj_pos
            )

            movement_magnitude = (
                np.linalg.norm(
                    movement_vector[:2]
                )
                * 1e2
            )

            movement_detected = (
                movement_magnitude
                > movement_threshold
            )

            inside_pot = (
                OU.check_obj_in_receptacle(
                    self,
                    obj_name,
                    "pot",
                )
            )

            if not contact_with_spatula:
                contact_with_spatula = (
                    self.check_contact(
                        spatula,
                        obj,
                    )
                )

            if not (
                movement_detected
                and inside_pot
                and contact_with_spatula
            ):
                all_objects_stirred = False

            setattr(
                self,
                f"prev_obj_pos_{obj_name}",
                obj_pos,
            )

        return all_objects_stirred

    # =====================================================================
    # Task-specific semantic milestones
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "veg1_in_hand": {
                "dependencies": [],
                "persistent": False,
            },
            "veg1_in_pot": {
                "dependencies": ["veg1_in_hand"],
                "persistent": True,
            },
            "veg2_in_hand": {
                "dependencies": [],
                "persistent": False,
            },
            "veg2_in_pot": {
                "dependencies": ["veg2_in_hand"],
                "persistent": True,
            },
            "spatula_in_hand": {
                "dependencies": ["veg1_in_pot", "veg2_in_pot"],
                "persistent": False,
            },
            "stir_vegetables_complete": {
                "dependencies": [
                    "veg1_in_pot",
                    "veg2_in_pot",
                    "spatula_in_hand",
                ],
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

    def _check_obj_in_receptacle_xy(
        self,
        obj_name,
        receptacle_name,
        th=None,
    ):
        """
        XY containment fallback used only after a vegetable has already been
        validly placed in the pot.

        During stirring, contact-based OU.check_obj_in_receptacle can flicker
        because the spatula moves the vegetables around. We still require the
        first placement to be contact-supported and released. This fallback is
        only used as anti-jitter after that placement has already been credited.
        """
        obj_pos = np.array(
            self.sim.data.body_xpos[
                self.obj_body_id[obj_name]
            ]
        )
        receptacle_pos = np.array(
            self.sim.data.body_xpos[
                self.obj_body_id[receptacle_name]
            ]
        )

        if th is None:
            receptacle = self.objects[receptacle_name]
            th = getattr(
                receptacle,
                "horizontal_radius",
                0.12,
            ) * 0.90

        return bool(
            np.linalg.norm(
                obj_pos[:2] - receptacle_pos[:2]
            )
            < th
        )

    def _get_raw_scene_state(self):
        veg1_in_hand = self._check_task_obj_grasped(
            "veg1",
        )
        veg2_in_hand = self._check_task_obj_grasped(
            "veg2",
        )
        spatula_in_hand = self._check_task_obj_grasped(
            "spatula",
        )

        veg1_in_pot_contact = bool(
            OU.check_obj_in_receptacle(
                self,
                "veg1",
                "pot",
            )
        )
        veg2_in_pot_contact = bool(
            OU.check_obj_in_receptacle(
                self,
                "veg2",
                "pot",
            )
        )

        veg1_in_pot_released = bool(
            veg1_in_pot_contact
            and not veg1_in_hand
        )
        veg2_in_pot_released = bool(
            veg2_in_pot_contact
            and not veg2_in_hand
        )

        veg1_already_placed = bool(
            getattr(
                self,
                "_lh_ordered_ever",
                {},
            ).get(
                "veg1_in_pot",
                False,
            )
        )
        veg2_already_placed = bool(
            getattr(
                self,
                "_lh_ordered_ever",
                {},
            ).get(
                "veg2_in_pot",
                False,
            )
        )

        veg1_in_pot_xy = bool(
            self._check_obj_in_receptacle_xy(
                "veg1",
                "pot",
            )
        )
        veg2_in_pot_xy = bool(
            self._check_obj_in_receptacle_xy(
                "veg2",
                "pot",
            )
        )

        stir_phase_started = bool(
            getattr(
                self,
                "_lh_stir_phase_started",
                False,
            )
        )

        veg1_in_pot_for_progress = bool(
            veg1_in_pot_released
            or (
                veg1_already_placed
                and (
                    veg1_in_pot_contact
                    or (
                        stir_phase_started
                        and veg1_in_pot_xy
                    )
                )
            )
        )
        veg2_in_pot_for_progress = bool(
            veg2_in_pot_released
            or (
                veg2_already_placed
                and (
                    veg2_in_pot_contact
                    or (
                        stir_phase_started
                        and veg2_in_pot_xy
                    )
                )
            )
        )

        return {
            "veg1_in_hand": veg1_in_hand,
            "veg2_in_hand": veg2_in_hand,
            "spatula_in_hand": spatula_in_hand,
            "veg1_in_pot_contact": veg1_in_pot_contact,
            "veg2_in_pot_contact": veg2_in_pot_contact,
            "veg1_in_pot_xy": veg1_in_pot_xy,
            "veg2_in_pot_xy": veg2_in_pot_xy,
            "veg1_in_pot_for_progress": veg1_in_pot_for_progress,
            "veg2_in_pot_for_progress": veg2_in_pot_for_progress,
            "veggies_ready_for_progress": bool(
                veg1_in_pot_for_progress
                and veg2_in_pot_for_progress
            ),
            "stir_complete": bool(self.success_time >= 5),
        }

    def _get_raw_semantic_milestone_state(self):
        scene = self._get_raw_scene_state()

        return {
            "veg1_in_hand": bool(scene["veg1_in_hand"]),
            "veg1_in_pot": bool(scene["veg1_in_pot_for_progress"]),
            "veg2_in_hand": bool(scene["veg2_in_hand"]),
            "veg2_in_pot": bool(scene["veg2_in_pot_for_progress"]),
            "spatula_in_hand": bool(scene["spatula_in_hand"]),
            "stir_vegetables_complete": bool(scene["stir_complete"]),
        }

    def _get_ordered_semantic_milestone_state(self):
        """
        Current ordered-valid milestone states.

        The two vegetable pick-place branches are independent, so veg1 and veg2
        can be completed in either order.

        During stirring, vegetable-in-pot progress uses a geometric containment
        fallback after the initial place has already been credited. This avoids
        progress dropping simply because contact-based receptacle checks flicker
        while the spatula moves the vegetables.
        """
        raw = self._get_raw_semantic_milestone_state()
        scene = self._get_raw_scene_state()

        veg1_place_valid = bool(
            self._lh_ordered_ever["veg1_in_pot"]
            and scene["veg1_in_pot_for_progress"]
        )
        veg2_place_valid = bool(
            self._lh_ordered_ever["veg2_in_pot"]
            and scene["veg2_in_pot_for_progress"]
        )

        veg1_pick_valid = bool(
            self._lh_ordered_ever["veg1_in_hand"]
            and (
                raw["veg1_in_hand"]
                or veg1_place_valid
            )
        )
        veg2_pick_valid = bool(
            self._lh_ordered_ever["veg2_in_hand"]
            and (
                raw["veg2_in_hand"]
                or veg2_place_valid
            )
        )

        veggies_ready = bool(
            veg1_place_valid
            and veg2_place_valid
        )

        stir_valid = bool(
            self._lh_ordered_ever["stir_vegetables_complete"]
            and raw["stir_vegetables_complete"]
            and veggies_ready
        )

        # Once the stir phase has started, keep spatula pick valid even if the
        # grasp signal flickers for a frame. If the vegetables truly leave the
        # pot, veggies_ready becomes False and this still regresses.
        spatula_pick_valid = bool(
            self._lh_ordered_ever["spatula_in_hand"]
            and veggies_ready
            and (
                raw["spatula_in_hand"]
                or self._lh_stir_phase_started
                or stir_valid
            )
        )

        return {
            "veg1_in_hand": veg1_pick_valid,
            "veg1_in_pot": veg1_place_valid,
            "veg2_in_hand": veg2_pick_valid,
            "veg2_in_pot": veg2_place_valid,
            "spatula_in_hand": spatula_pick_valid,
            "stir_vegetables_complete": stir_valid,
        }

    # =====================================================================
    # Long-horizon metric tracker
    # =====================================================================

    def _reset_long_horizon_metric_tracker(self, initial_state=None):
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

        self._lh_stir_phase_started = False

        self._lh_ops_auc_sum = 0.0
        self._lh_peak_progress = 0.0
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

        return (
            int(bool(ordered_state["veg1_in_hand"]))
            + int(bool(ordered_state["veg1_in_pot"]))
            + int(bool(ordered_state["veg2_in_hand"]))
            + int(bool(ordered_state["veg2_in_pot"]))
            + int(bool(ordered_state["spatula_in_hand"]))
            + int(bool(ordered_state["stir_vegetables_complete"]))
        ) / 6.0

    def _update_long_horizon_metric_tracker(self):
        self._lh_step_count += 1

        spec = self.get_semantic_milestone_spec()
        raw = self._get_raw_semantic_milestone_state()
        scene = self._get_raw_scene_state()

        # --------------------------------------------------------------
        # UPS: order ignored.
        # --------------------------------------------------------------
        for name in spec.keys():
            if raw[name]:
                self._lh_unordered_ever[name] = True

        # --------------------------------------------------------------
        # Veg1 branch.
        # --------------------------------------------------------------
        if raw["veg1_in_hand"]:
            self._lh_ordered_ever["veg1_in_hand"] = True

        if raw["veg1_in_pot"]:
            if not self._lh_ordered_ever["veg1_in_hand"]:
                self._lh_ordered_ever["veg1_in_hand"] = True
            self._lh_ordered_ever["veg1_in_pot"] = True

        # --------------------------------------------------------------
        # Veg2 branch.
        # --------------------------------------------------------------
        if raw["veg2_in_hand"]:
            self._lh_ordered_ever["veg2_in_hand"] = True

        if raw["veg2_in_pot"]:
            if not self._lh_ordered_ever["veg2_in_hand"]:
                self._lh_ordered_ever["veg2_in_hand"] = True
            self._lh_ordered_ever["veg2_in_pot"] = True

        veggies_ready = bool(
            scene["veggies_ready_for_progress"]
            and self._lh_ordered_ever["veg1_in_pot"]
            and self._lh_ordered_ever["veg2_in_pot"]
        )

        # --------------------------------------------------------------
        # Pick spatula after both vegetables are in the pot.
        # --------------------------------------------------------------
        if (
            raw["spatula_in_hand"]
            and veggies_ready
        ):
            self._lh_ordered_ever["spatula_in_hand"] = True
            self._lh_stir_phase_started = True

        # If official success_time has started accumulating, we are definitely
        # in the stir phase, even if the metric grasp signal flickered.
        if (
            getattr(self, "success_time", 0) > 0
            and veggies_ready
            and self._lh_ordered_ever["spatula_in_hand"]
        ):
            self._lh_stir_phase_started = True

        # --------------------------------------------------------------
        # Stir complete after both vegetables and spatula pick.
        # --------------------------------------------------------------
        if (
            raw["stir_vegetables_complete"]
            and veggies_ready
        ):
            if not self._lh_ordered_ever["spatula_in_hand"]:
                self._lh_ordered_ever["spatula_in_hand"] = True
            self._lh_stir_phase_started = True
            self._lh_ordered_ever["stir_vegetables_complete"] = True

        ordered_state = self._get_ordered_semantic_milestone_state()

        # --------------------------------------------------------------
        # Regression bookkeeping for persistent placement / stir milestones.
        # --------------------------------------------------------------
        for name, cfg in spec.items():
            if not cfg.get("persistent", True):
                continue
            if not self._lh_ordered_ever[name]:
                continue

            prev_ordered_valid = bool(
                self._lh_ordered_ever[name]
                and self._lh_prev_raw[name]
            )

            falling_edge = bool(
                prev_ordered_valid
                and not ordered_state[name]
            )

            if falling_edge:
                self._lh_regressed[name] = True

        current_ops = self._get_current_ordered_progress(
            ordered_state
        )
        self._lh_progress_curve.append(
            float(current_ops)
        )

        self._lh_ops_auc_sum += current_ops
        self._lh_peak_progress = max(
            self._lh_peak_progress,
            current_ops,
        )

        if (
            self._check_success()
            and self._lh_first_success_step is None
        ):
            self._lh_first_success_step = self._lh_step_count

        self._lh_prev_raw = {
            name: bool(raw[name])
            for name in spec.keys()
        }

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        self._update_long_horizon_metric_tracker()
        return reward, done, info

    def get_long_horizon_metrics(self):
        ops = self._get_current_ordered_progress()

        spec = self.get_semantic_milestone_spec()
        num_milestones = len(spec)
        ups = (
            sum(self._lh_unordered_ever.values()) / float(num_milestones)
            if num_milestones > 0
            else 0.0
        )

        nct = (
            -1.0
            if self._lh_first_success_step is None
            else float(self._lh_first_success_step)
            / float(getattr(self, "horizon", 1))
        )

        return {
            "SR": float(
                self._lh_first_success_step is not None
            ),
            "NCT": float(nct),
            "OPS": float(ops),
            "UPS": float(ups),
            "progress_curve": list(
                self._lh_progress_curve
            ),
        }

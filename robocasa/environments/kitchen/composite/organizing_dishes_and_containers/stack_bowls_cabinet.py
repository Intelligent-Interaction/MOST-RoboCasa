from robocasa.environments.kitchen.kitchen import *


class StackBowlsCabinet(Kitchen):
    """
    Stack Bowls Cabinet: composite task for Organizing Dishes and Containers.

    Task:
        Stack the two bowls and place the stacked bowls in the cabinet.

    Valid strategies:
        A) Put the small bowl into the large bowl first, then carry the stacked
           bowls into the cabinet.
        B) Put the large bowl into the cabinet first, then put the small bowl
           into the large bowl inside the cabinet.

    Long-horizon semantic milestones are two pick-place actions:
        M1. first_pick
        M2. first_place
        M3. second_pick
        M4. second_place

    Important implementation detail:
        After the bowls are nested, grasp identity becomes ambiguous. When the
        gripper picks the large bowl, the small bowl may also look grasped
        because it is close to the gripper / large bowl. Therefore this metric
        does NOT bind second_pick to bowl1 or bowl2. It only asks whether the
        robot has picked "some bowl or stacked bowls" after a valid first_place.

    Original RoboCasa _check_success() is preserved.
    """

    def __init__(self, *args, **kwargs):
        self._reset_long_horizon_metric_tracker(initial_state=None)
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cabinet)
        )
        self.init_robot_base_ref = self.cabinet

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta["lang"] = (
                "Pick up the bowls on the counter and stack them on top of one another in the open cabinet. "
                "Place the smaller bowl on top of the larger bowl."
            )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cabinet.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="bowl1",
                obj_groups="bowl",
                object_scale=0.7,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cabinet,
                    ),
                    size=(1.0, 0.60),
                    pos=("ref", -0.8),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl2",
                obj_groups="bowl",
                object_scale=0.95,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cabinet,
                    ),
                    size=(1.0, 0.60),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    # =====================================================================
    # Task-specific semantic milestones
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "first_pick": {
                "description": "The robot picks one bowl.",
                "dependencies": [],
                "persistent": False,
            },
            "first_place": {
                "description": (
                    "The first useful placement is completed: either one bowl "
                    "is placed in the cabinet, or the two bowls are stacked."
                ),
                "dependencies": ["first_pick"],
                "persistent": True,
            },
            "second_pick": {
                "description": (
                    "After the first useful placement and release, the robot "
                    "picks the remaining bowl or the stacked bowls."
                ),
                "dependencies": ["first_place"],
                "persistent": False,
            },
            "second_place": {
                "description": (
                    "Both bowls are in the cabinet and stacked / nested."
                ),
                "dependencies": ["second_pick"],
                "persistent": True,
            },
        }

    def _check_task_obj_grasped(
        self,
        obj_name,
    ):
        """
        Local grasp check with a looser gripper-close threshold.

        Bowl grasps are easy to miss during replay because contact can flicker.
        This does not modify global RoboCasa object_utils behavior.
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

    def _bowl_in_cabinet(
        self,
        obj_name,
        obj_in_hand,
    ):
        return bool(
            OU.obj_inside_of(
                self,
                obj_name,
                self.cabinet,
            )
            and not obj_in_hand
        )

    def _bowls_stacked(self):
        return bool(
            OU.check_obj_in_receptacle(
                self,
                "bowl2",
                "bowl1",
            )
            or OU.check_obj_in_receptacle(
                self,
                "bowl1",
                "bowl2",
            )
        )

    def _get_raw_scene_state(self):
        bowl1_in_hand = self._check_task_obj_grasped(
            "bowl1",
        )
        bowl2_in_hand = self._check_task_obj_grasped(
            "bowl2",
        )

        any_bowl_in_hand = bool(
            bowl1_in_hand
            or bowl2_in_hand
        )

        bowl1_in_cabinet = self._bowl_in_cabinet(
            "bowl1",
            bowl1_in_hand,
        )
        bowl2_in_cabinet = self._bowl_in_cabinet(
            "bowl2",
            bowl2_in_hand,
        )

        bowls_stacked = self._bowls_stacked()

        final_arrangement = bool(
            bowl1_in_cabinet
            and bowl2_in_cabinet
            and bowls_stacked
        )

        # First placement should be a released / stable intermediate state.
        # This prevents one frame of "stacked while still grasping the small
        # bowl" from immediately becoming both first_place and second_pick.
        stable_stack_intermediate = bool(
            bowls_stacked
            and not any_bowl_in_hand
        )
        stable_cabinet_intermediate = bool(
            bowl1_in_cabinet
            or bowl2_in_cabinet
        )

        first_useful_place = bool(
            final_arrangement
            or stable_stack_intermediate
            or stable_cabinet_intermediate
        )

        return {
            "bowl1_in_hand": bowl1_in_hand,
            "bowl2_in_hand": bowl2_in_hand,
            "any_bowl_in_hand": any_bowl_in_hand,
            "bowl1_in_cabinet": bowl1_in_cabinet,
            "bowl2_in_cabinet": bowl2_in_cabinet,
            "any_bowl_in_cabinet": bool(
                bowl1_in_cabinet
                or bowl2_in_cabinet
            ),
            "bowls_stacked": bowls_stacked,
            "stable_stack_intermediate": stable_stack_intermediate,
            "stable_cabinet_intermediate": stable_cabinet_intermediate,
            "first_useful_place": first_useful_place,
            "final_arrangement": final_arrangement,
        }

    def _get_raw_semantic_milestone_state(self):
        """
        Dynamic two-pick-place milestone state.

        second_pick deliberately does not identify which bowl is grasped. After
        nesting, the small bowl can be close enough that grasp detection may
        claim it is picked while the robot is actually holding the large bowl.
        The stage-level question is simply whether a bowl / stack is currently
        held after a valid first_place.
        """
        scene = self._get_raw_scene_state()

        return {
            "first_pick": bool(
                scene["any_bowl_in_hand"]
            ),
            "first_place": bool(
                scene["first_useful_place"]
            ),
            "second_pick": bool(
                self._lh_ready_for_second_pick
                and scene["any_bowl_in_hand"]
                and not scene["final_arrangement"]
            ),
            "second_place": bool(
                scene["final_arrangement"]
            ),
        }

    def _get_ordered_semantic_milestone_state(self):
        """
        Current ordered-valid milestone states.

        first_place remains valid while the robot is carrying the stacked bowls
        after first_place. This avoids oscillation caused by nested-bowl contact
        flicker or bowl-identity ambiguity.

        If the stack is dropped outside the cabinet and no useful placement or
        carrying state remains, placement progress can still decrease.
        """
        raw = self._get_raw_semantic_milestone_state()
        scene = self._get_raw_scene_state()

        second_place_valid = bool(
            self._lh_ordered_ever["second_place"]
            and scene["final_arrangement"]
        )

        # Keep first_place alive during second-pick / stack-carry. Without this,
        # a valid "small bowl in large bowl" first_place can flicker off when
        # the gripper lifts the large bowl and the small bowl is ambiguously
        # detected as grasped.
        carrying_after_first_place = bool(
            self._lh_ordered_ever["first_place"]
            and scene["any_bowl_in_hand"]
            and (
                self._lh_ready_for_second_pick
                or self._lh_ordered_ever["second_pick"]
            )
            and not second_place_valid
        )

        first_place_valid = bool(
            self._lh_ordered_ever["first_place"]
            and (
                scene["first_useful_place"]
                or carrying_after_first_place
                or second_place_valid
            )
        )

        first_pick_valid = bool(
            self._lh_ordered_ever["first_pick"]
            and (
                scene["any_bowl_in_hand"]
                or first_place_valid
                or second_place_valid
            )
        )

        second_pick_valid = bool(
            self._lh_ordered_ever["second_pick"]
            and first_place_valid
            and (
                raw["second_pick"]
                or second_place_valid
            )
        )

        return {
            "first_pick": first_pick_valid,
            "first_place": first_place_valid,
            "second_pick": second_pick_valid,
            "second_place": second_place_valid,
        }

    # =====================================================================
    # Per-episode long-horizon metric tracker
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

        # After first_place is credited, require a no-hand frame before
        # allowing second_pick. This prevents the same physical grasp from
        # being counted as both first_pick and second_pick.
        self._lh_ready_for_second_pick = False

        self._lh_ops_auc_sum = 0.0
        self._lh_peak_progress = 0.0
        self._lh_first_success_step = None
        self._lh_progress_curve = []

    def _reset_internal(self):
        """
        Normal RoboCasa reset plus metric initialization from the true reset
        simulator state.
        """
        super()._reset_internal()

        initial_raw = self._get_raw_semantic_milestone_state()
        self._reset_long_horizon_metric_tracker(
            initial_state=initial_raw
        )

    def _get_current_ordered_progress(self, ordered_state=None):
        if ordered_state is None:
            ordered_state = self._get_ordered_semantic_milestone_state()

        return (
            int(bool(ordered_state["first_pick"]))
            + int(bool(ordered_state["first_place"]))
            + int(bool(ordered_state["second_pick"]))
            + int(bool(ordered_state["second_place"]))
        ) / 4.0

    def _update_long_horizon_metric_tracker(self):
        """
        Update all long-horizon metrics once per policy timestep.
        """
        self._lh_step_count += 1

        spec = self.get_semantic_milestone_spec()
        scene = self._get_raw_scene_state()
        raw = self._get_raw_semantic_milestone_state()

        # --------------------------------------------------------------
        # UPS: order ignored.
        # --------------------------------------------------------------
        for name in spec.keys():
            if raw[name]:
                self._lh_unordered_ever[name] = True

        # --------------------------------------------------------------
        # M1: first pick.
        # --------------------------------------------------------------
        if raw["first_pick"]:
            self._lh_ordered_ever["first_pick"] = True

        # --------------------------------------------------------------
        # M2: first useful place.
        #
        # If first_pick was missed due to grasp flicker, infer it from a valid
        # first placement. This is safe because both bowls start on the counter.
        # --------------------------------------------------------------
        if raw["first_place"]:
            if not self._lh_ordered_ever["first_pick"]:
                self._lh_ordered_ever["first_pick"] = True
            self._lh_ordered_ever["first_place"] = True

        # Once first_place is done, wait until there is a no-hand frame before
        # counting a later grasp as second_pick.
        if (
            self._lh_ordered_ever["first_place"]
            and not scene["any_bowl_in_hand"]
            and not self._lh_ordered_ever["second_pick"]
        ):
            self._lh_ready_for_second_pick = True

        # --------------------------------------------------------------
        # M3: second pick after first placement and release.
        # --------------------------------------------------------------
        if raw["second_pick"]:
            self._lh_ordered_ever["second_pick"] = True

        # --------------------------------------------------------------
        # M4: final arrangement.
        #
        # If second_pick was missed due to grasp flicker, infer it from final
        # arrangement after a valid first placement. This preserves robustness
        # without binding to bowl identity.
        # --------------------------------------------------------------
        if (
            raw["second_place"]
            and self._lh_ordered_ever["first_place"]
        ):
            if not self._lh_ordered_ever["second_pick"]:
                self._lh_ordered_ever["second_pick"] = True
            self._lh_ordered_ever["second_place"] = True

        ordered_state = self._get_ordered_semantic_milestone_state()

        # --------------------------------------------------------------
        # Regression bookkeeping for persistent placement milestones.
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
        self._lh_progress_curve.append(float(current_ops))

        self._lh_ops_auc_sum += current_ops
        self._lh_peak_progress = max(
            self._lh_peak_progress,
            current_ops,
        )

        # SR / NCT preserve the original success criterion.
        success = bool(self._check_success())
        if (
            success
            and self._lh_first_success_step is None
        ):
            self._lh_first_success_step = self._lh_step_count

        self._lh_prev_raw = {
            name: bool(raw[name])
            for name in spec.keys()
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
        ordered_state = self._get_ordered_semantic_milestone_state()
        ops = self._get_current_ordered_progress(
            ordered_state
        )

        spec = self.get_semantic_milestone_spec()
        num_milestones = len(spec)
        ups = (
            sum(self._lh_unordered_ever.values()) / float(num_milestones)
            if num_milestones > 0
            else 0.0
        )

        return {
            "SR": float(
                self._lh_first_success_step is not None
            ),
            "OPS": float(ops),
            "UPS": float(ups),
            "NCT": (
                -1.0
                if self._lh_first_success_step is None
                else float(self._lh_first_success_step)
                / float(getattr(self, "horizon", 1))
            ),
            "progress_curve": list(
                self._lh_progress_curve
            ),
        }

    # =====================================================================
    # Original RoboCasa success criterion, preserved
    # =====================================================================

    def _check_success(self):
        bowl1_in_cabinet = OU.obj_inside_of(
            self,
            "bowl1",
            self.cabinet,
        )
        bowl2_in_cabinet = OU.obj_inside_of(
            self,
            "bowl2",
            self.cabinet,
        )

        bowl2_in_bowl1 = OU.check_obj_in_receptacle(
            self,
            "bowl2",
            "bowl1",
        )
        bowl1_in_bowl2 = OU.check_obj_in_receptacle(
            self,
            "bowl1",
            "bowl2",
        )

        gripper_bowl1_far = OU.gripper_obj_far(
            self,
            obj_name="bowl1",
        )
        gripper_bowl2_far = OU.gripper_obj_far(
            self,
            obj_name="bowl2",
        )

        return (
            bowl1_in_cabinet
            and bowl2_in_cabinet
            and (bowl2_in_bowl1 or bowl1_in_bowl2)
            and gripper_bowl1_far
            and gripper_bowl2_far
        )

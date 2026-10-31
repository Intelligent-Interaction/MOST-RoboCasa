from robocasa.environments.kitchen.kitchen import *
import robocasa
import numpy as np
import robocasa.utils.object_utils as OU


class SeparateFreezerRack(Kitchen):
    """
    Separate Freezer Rack: composite task for Managing Freezer Space activity.

    Simulates the task of organizing freezer items by placing meat on the second
    highest rack and vegetables on the highest rack of the freezer.

    Steps:
        1. Take the meat in its tupperware container.
        2. Place the meat container on the second highest rack of the freezer.
        3. Take the vegetables in the other tupperware container.
        4. Place the vegetable container on the highest rack of the freezer.

    Progress design:
        This is treated as two order-free container PnPs.

        Branch A:
            meat_container_in_hand -> meat_container_on_second_rack

        Branch B:
            vegetable_container_in_hand -> vegetable_container_on_top_rack

        The robot grasps the tupperware containers, not the meat / vegetable
        objects directly. Contents-inside-container are initialization facts and
        remain part of the original terminal success criterion only.

    Original RoboCasa _check_success() is preserved.
    """

    EXCLUDE_LAYOUTS = Kitchen.FREEZER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        self._reset_long_horizon_metric_tracker(
            initial_state=None
        )
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.fridge = self.register_fixture_ref(
            "fridge",
            dict(id=FixtureType.FRIDGE),
        )

        self.counter = self.register_fixture_ref(
            "counter",
            dict(
                id=FixtureType.COUNTER,
                ref=self.fridge,
            ),
        )

        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        meat_lang = self.get_obj_lang("meat")
        veg1_lang = self.get_obj_lang("vegetable1")
        veg2_lang = self.get_obj_lang("vegetable2")

        if self.use_novel_instructions:
            veg_lang = f"{veg1_lang} and {veg2_lang}"
            ep_meta["lang"] = self.rng.choice(
                self.novel_instructions
            ).format(
                meat_lang=meat_lang,
                veg_lang=veg_lang,
            )

        elif veg1_lang == veg2_lang:
            ep_meta["lang"] = (
                f"Take the meat container that has the {meat_lang} and place it on the second highest rack of the freezer. "
                f"Then take the vegetable container that has the {veg1_lang}s and place it on the highest rack of the freezer."
            )

        else:
            ep_meta["lang"] = (
                f"Take the meat container that has the {meat_lang} and place it on the second highest rack of the freezer. "
                f"Then take the vegetable container that has the {veg1_lang} and {veg2_lang} and place it on the highest rack of the freezer."
            )

        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

        self.fridge.open_door(
            env=self,
            compartment="freezer",
        )

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="meat_tupperware",
                obj_groups="tupperware",
                object_scale=[1.5, 1.5, 1],
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.fridge,
                    ),
                    size=(1.0, 0.4),
                    pos=(0.25, -1.0),
                    rotation=np.pi / 2,
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                graspable=True,
                placement=dict(
                    object="meat_tupperware",
                    size=(0.8, 0.8),
                ),
            )
        )

        cfgs.append(
            dict(
                name="veg_tupperware",
                obj_groups="tupperware",
                object_scale=[1.5, 1.5, 1],
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="meat_tupperware",
                    size=(1.0, 0.4),
                    pos=(-0.25, -1.0),
                    rotation=np.pi / 2,
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    object="veg_tupperware",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable2",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    object="veg_tupperware",
                    size=(1.0, 1.0),
                ),
            )
        )

        return cfgs

    # =====================================================================
    # Task-specific semantic milestones
    # =====================================================================

    def get_semantic_milestone_spec(self):
        return {
            "meat_container_in_hand": {
                "description": (
                    "The meat tupperware container is grasped by the robot."
                ),
                "dependencies": [],
                "persistent": False,
            },
            "meat_container_on_second_rack": {
                "description": (
                    "The meat tupperware container is placed on the second "
                    "highest freezer rack."
                ),
                "dependencies": ["meat_container_in_hand"],
                "persistent": True,
            },
            "vegetable_container_in_hand": {
                "description": (
                    "The vegetable tupperware container is grasped by the robot."
                ),
                "dependencies": [],
                "persistent": False,
            },
            "vegetable_container_on_top_rack": {
                "description": (
                    "The vegetable tupperware container is placed on the "
                    "highest freezer rack."
                ),
                "dependencies": ["vegetable_container_in_hand"],
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

    def _container_on_freezer_rack(
        self,
        obj_name,
        rack_index,
    ):
        return bool(
            self.fridge.check_rack_contact(
                self,
                obj_name,
                compartment="freezer",
                rack_index=rack_index,
            )
        )

    def _get_raw_semantic_milestone_state(self):
        """
        Return current simulator-grounded container PnP states.

        The food contents are not checked here because the robot action is to
        move the tupperware containers, not grasp the food directly.
        """
        meat_in_hand = self._check_task_obj_grasped(
            "meat_tupperware",
        )

        veg_in_hand = self._check_task_obj_grasped(
            "veg_tupperware",
        )

        meat_on_second_rack = bool(
            self._container_on_freezer_rack(
                "meat_tupperware",
                rack_index=-2,
            )
            and not meat_in_hand
        )

        veg_on_top_rack = bool(
            self._container_on_freezer_rack(
                "veg_tupperware",
                rack_index=-1,
            )
            and not veg_in_hand
        )

        return {
            "meat_container_in_hand": meat_in_hand,
            "meat_container_on_second_rack": meat_on_second_rack,
            "vegetable_container_in_hand": veg_in_hand,
            "vegetable_container_on_top_rack": veg_on_top_rack,
        }

    def _get_ordered_semantic_milestone_state(self):
        """
        The two container PnP branches are independent.

        Pick credit is current/superseded:
            - valid while the corresponding container is in hand;
            - also valid once the corresponding placement is currently valid.

        Placement credit is persistent but must remain physically valid.
        """
        raw = self._get_raw_semantic_milestone_state()

        meat_place_valid = bool(
            self._lh_ordered_ever["meat_container_on_second_rack"]
            and raw["meat_container_on_second_rack"]
        )

        meat_pick_valid = bool(
            self._lh_ordered_ever["meat_container_in_hand"]
            and (
                raw["meat_container_in_hand"]
                or meat_place_valid
            )
        )

        veg_place_valid = bool(
            self._lh_ordered_ever["vegetable_container_on_top_rack"]
            and raw["vegetable_container_on_top_rack"]
        )

        veg_pick_valid = bool(
            self._lh_ordered_ever["vegetable_container_in_hand"]
            and (
                raw["vegetable_container_in_hand"]
                or veg_place_valid
            )
        )

        return {
            "meat_container_in_hand": meat_pick_valid,
            "meat_container_on_second_rack": meat_place_valid,
            "vegetable_container_in_hand": veg_pick_valid,
            "vegetable_container_on_top_rack": veg_place_valid,
        }

    # =====================================================================
    # Per-episode long-horizon metric tracker
    # =====================================================================

    def _reset_long_horizon_metric_tracker(
        self,
        initial_state=None,
    ):
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

        self._lh_first_success_step = None
        self._lh_progress_curve = []

    def _reset_internal(self):
        super()._reset_internal()

        initial_raw = (
            self._get_raw_semantic_milestone_state()
        )

        self._reset_long_horizon_metric_tracker(
            initial_state=initial_raw
        )

    def _get_current_ordered_progress(
        self,
        ordered_state=None,
    ):
        if ordered_state is None:
            ordered_state = (
                self._get_ordered_semantic_milestone_state()
            )

        return (
            int(bool(ordered_state["meat_container_in_hand"]))
            + int(bool(ordered_state["meat_container_on_second_rack"]))
            + int(bool(ordered_state["vegetable_container_in_hand"]))
            + int(bool(ordered_state["vegetable_container_on_top_rack"]))
        ) / 4.0

    def _update_long_horizon_metric_tracker(self):
        """
        Both container PnPs are order-free.

        Valid examples:
            meat container pick -> meat container place
            vegetable container pick -> vegetable container place

        and:
            vegetable container pick -> vegetable container place
            meat container pick -> meat container place
        """
        self._lh_step_count += 1

        spec = self.get_semantic_milestone_spec()
        raw = self._get_raw_semantic_milestone_state()

        # --------------------------------------------------------------
        # UPS: order ignored.
        # --------------------------------------------------------------
        for name in spec.keys():
            if raw[name]:
                self._lh_unordered_ever[name] = True

        # --------------------------------------------------------------
        # Meat container branch.
        # --------------------------------------------------------------
        if raw["meat_container_in_hand"]:
            self._lh_ordered_ever["meat_container_in_hand"] = True

        if raw["meat_container_on_second_rack"]:
            # If grasp detection missed the pick, infer it from a valid place.
            if not self._lh_ordered_ever["meat_container_in_hand"]:
                self._lh_ordered_ever["meat_container_in_hand"] = True
            self._lh_ordered_ever["meat_container_on_second_rack"] = True

        # --------------------------------------------------------------
        # Vegetable container branch.
        # --------------------------------------------------------------
        if raw["vegetable_container_in_hand"]:
            self._lh_ordered_ever["vegetable_container_in_hand"] = True

        if raw["vegetable_container_on_top_rack"]:
            if not self._lh_ordered_ever["vegetable_container_in_hand"]:
                self._lh_ordered_ever["vegetable_container_in_hand"] = True
            self._lh_ordered_ever["vegetable_container_on_top_rack"] = True

        ordered_state = (
            self._get_ordered_semantic_milestone_state()
        )

        # --------------------------------------------------------------
        # Regression bookkeeping for persistent placement states.
        # --------------------------------------------------------------
        for name in [
            "meat_container_on_second_rack",
            "vegetable_container_on_top_rack",
        ]:
            if not self._lh_ordered_ever[name]:
                continue

            prev_valid = bool(
                self._lh_prev_raw[name]
            )

            falling_edge = bool(
                prev_valid
                and not ordered_state[name]
            )

            if falling_edge:
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
            and self._lh_first_success_step is None
        ):
            self._lh_first_success_step = (
                self._lh_step_count
            )

        self._lh_prev_raw = {
            name: bool(raw[name])
            for name in spec.keys()
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
        spec = self.get_semantic_milestone_spec()
        names = list(spec.keys())

        ups = (
            sum(
                int(bool(self._lh_unordered_ever[name]))
                for name in names
            )
            / float(len(names))
        )

        ops = self._get_current_ordered_progress()

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
    # Original RoboCasa success criterion, preserved
    # =====================================================================

    def _check_success(self):
        meat_in_tupperware = (
            OU.check_obj_in_receptacle(
                self,
                "meat",
                "meat_tupperware",
            )
        )

        veg1_in_tupperware = (
            OU.check_obj_in_receptacle(
                self,
                "vegetable1",
                "veg_tupperware",
            )
        )

        veg2_in_tupperware = (
            OU.check_obj_in_receptacle(
                self,
                "vegetable2",
                "veg_tupperware",
            )
        )

        veg_tupperware_top_rack = (
            self.fridge.check_rack_contact(
                self,
                "veg_tupperware",
                compartment="freezer",
                rack_index=-1,
            )
        )

        meat_tupperware_second_rack = (
            self.fridge.check_rack_contact(
                self,
                "meat_tupperware",
                compartment="freezer",
                rack_index=-2,
            )
        )

        gripper_far_from_meat = (
            OU.gripper_obj_far(
                self,
                obj_name="meat_tupperware",
            )
        )

        gripper_far_from_veg = (
            OU.gripper_obj_far(
                self,
                obj_name="veg_tupperware",
            )
        )

        return (
            meat_in_tupperware
            and veg1_in_tupperware
            and veg2_in_tupperware
            and veg_tupperware_top_rack
            and meat_tupperware_second_rack
            and gripper_far_from_meat
            and gripper_far_from_veg
        )

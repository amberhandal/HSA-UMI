"""
The arms HSA-UMI can drive, selected with ``robot:=fer|fr3`` on bringup.launch.py and inference.launch.py.

    fer  the HSA lab's FER on the NU-MSR station (Kilted), HSA wrist on the flange, homing through
         fer_arm_controller, no gripper.
    fr3  the CRB FR3 on its NUC (Humble), its own gripper on the flange (tcp_calib.py), homing through
         move_group, gripper drivers selectable.

Everything the launch files and scripts need to tell the two apart lives here, so a third arm is one
more entry. The laptop-side frames are repeated in ros2_ws/src/hsa_ros2/config/robots/<name>.yaml
(nuc/test_robots.py keeps the two in step). Pure Python, no ROS imports: the tests read it anywhere.
"""

from dataclasses import dataclass
import os

#: Environment variable that picks the default robot. nuc/station_env.sh exports it on each arm box.
ROBOT_ENV = 'HSA_ROBOT'
DEFAULT_ROBOT = 'fer'

JTC_TYPE = 'joint_trajectory_controller/JointTrajectoryController'
SERVO_TYPE = 'franka_streaming_impedance_controller/CartesianImpedanceController'
#: The streaming controller's instance name; every producer publishes to its topic, so it is shared.
SERVO_CONTROLLER = 'polyumi_cartesian_impedance_controller'


@dataclass(frozen=True)
class Robot:
    """One arm box: hardware, frames, controllers, homing."""

    name: str
    arm_id: str
    robot_ip: str
    #: Module beside this file owning the TCP and the payload (static_transform_publisher_args,
    #: describe, PAYLOAD_MASS, ...).
    calib: str
    #: The frame the policy observes and commands, as published by bringup.
    tcp_frame: str
    #: 'nuc' = our copy nuc/launch/fer_hardware.launch.py; 'franka_bringup' = the installed franka.launch.py.
    hardware_launch: str
    #: Fixed frames the URDF lost with load_gripper:=false. 'desk_ee' = <arm_id>_link8 -> <arm_id>_hand_tcp
    #: from the Desk end-effector setting (hsa_calib); 'hand' = the whole hand subtree (tcp_calib).
    flange_frames: str
    #: Verify <arm_id>_hand_tcp against the robot's F_T_EE before the payload goes in (check_ee_frame.py).
    check_ee: bool
    #: The joint-trajectory controller homing drives.
    arm_controller: str
    #: Its parameter file: a name in nuc/config, or 'package://<pkg>/<path>'.
    arm_controller_params: str
    #: Same for mock hardware (position-commanded); None = no mock-hardware support.
    arm_controller_mock_params: str | None
    #: The streaming controller's parameter file in nuc/config.
    servo_params: str
    #: 'jtc' = fer_home_service.py straight to the arm controller; 'moveit' = fr3_home_service.py + move_group.
    home: str
    home_service: str
    #: Where /polyumi/set_home persists. Separate from the CRB team's file on their NUC.
    home_pose_file: str
    #: Switch controllers inside the realtime loop (spawner --switch-asap, SwitchController activate_asap).
    #: Needed on the FER under Kilted; Humble's spawner has no such flag.
    switch_asap: bool
    #: Gripper drivers (franka_hand_node / FAULHABER) can be started by inference.launch.py.
    grippers: bool

    @property
    def base_frame(self) -> str:
        """The arm's base link, the frame every pose is expressed in."""
        return f'{self.arm_id}_link0'


ROBOTS = {
    'fer': Robot(
        name='fer',
        arm_id='fer',
        robot_ip='panda0.robot',
        calib='hsa_calib',
        tcp_frame='hsa_tcp',
        hardware_launch='nuc',
        flange_frames='desk_ee',
        check_ee=True,
        arm_controller='fer_arm_controller',
        arm_controller_params='fer_arm_controller.yaml',
        arm_controller_mock_params='fer_arm_controller.mock.yaml',
        servo_params='hsa_controllers.yaml',
        home='jtc',
        home_service='fer_home_service.py',
        # The station already holds its taught pose here; kept so it is not lost.
        home_pose_file='~/.ros/polyumi_home_pose.yaml',
        switch_asap=True,
        grippers=False,
    ),
    'fr3': Robot(
        name='fr3',
        arm_id='fr3',
        robot_ip='192.168.51.20',
        calib='tcp_calib',
        tcp_frame='polyumi_tcp',
        hardware_launch='franka_bringup',
        flange_frames='hand',
        check_ee=False,
        arm_controller='fr3_arm_controller',
        arm_controller_params='package://franka_fr3_moveit_config/config/fr3_ros_controllers.yaml',
        arm_controller_mock_params=None,
        servo_params='polyumi_controllers.yaml',
        home='moveit',
        home_service='fr3_home_service.py',
        # Not ~/.ros/polyumi_home_pose.yaml: that is the CRB team's taught pose on the same NUC.
        home_pose_file='~/.ros/hsa_home_pose.yaml',
        switch_asap=False,
        grippers=True,
    ),
}


def get(name: str | None = None) -> Robot:
    """
    Return the profile for ``name``, or for $HSA_ROBOT (default fer) when ``name`` is empty.

    :raises ValueError: on a name not in :data:`ROBOTS`.
    """
    name = name or os.environ.get(ROBOT_ENV) or DEFAULT_ROBOT
    try:
        return ROBOTS[name]
    except KeyError:
        raise ValueError(f'unknown robot {name!r}; expected one of {sorted(ROBOTS)}') from None


def needs_type_flag(ros_distro: str | None = None) -> bool:
    """
    Whether the spawner takes the controller type as ``-t``.

    Humble's spawner reads the type only from -t; Jazzy's and later have no -t and read it from the
    parameter file, under the controller's own key (nuc/config/*.yaml carry it there).
    """
    return (ros_distro if ros_distro is not None else os.environ.get('ROS_DISTRO', '')) == 'humble'


def spawner_arguments(
    controller: str,
    type_name: str,
    param_file,
    *,
    timeout_s: str,
    inactive: bool = False,
    switch_asap: bool = False,
    ros_distro: str | None = None,
) -> list:
    """
    Build ``controller_manager spawner`` arguments that work on this box's ROS distro.

    :param param_file: a path or a launch substitution; passed through untouched.
    :param switch_asap: activate inside the realtime loop (Jazzy and later only).
    """
    args = [controller]
    if needs_type_flag(ros_distro):
        args += ['-t', type_name]
    args += ['--param-file', param_file]
    if inactive:
        args.append('--inactive')
    if switch_asap:
        args.append('--switch-asap')
    args += ['--controller-manager-timeout', timeout_s]
    return args

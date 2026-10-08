"""Write a small rollout bag with rosbag2 and read it back through Rollout: chunks, TF path, diagnostics, wrench."""

import numpy as np
import pytest
import rosbag2_py
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import Transform, TransformStamped, WrenchStamped
from rclpy.serialization import serialize_message
from std_msgs.msg import Float32
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint

from hsa_ros2.bag_reader import Rollout


def _stamp(msg_stamp, t: float):
    msg_stamp.sec = int(t)
    msg_stamp.nanosec = int((t - int(t)) * 1e9)


def _write_bag(path):
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    topics = {
        '/tf': 'tf2_msgs/msg/TFMessage',
        '/tf_static': 'tf2_msgs/msg/TFMessage',
        '/polyumi/target_poses_traj': 'trajectory_msgs/msg/MultiDOFJointTrajectory',
        '/polyumi/diag/obs_age_s': 'std_msgs/msg/Float32',
        '/franka_robot_state_broadcaster/external_wrench_in_base_frame': 'geometry_msgs/msg/WrenchStamped',
    }
    for name, typ in topics.items():
        writer.create_topic(rosbag2_py.TopicMetadata(id=0, name=name, type=typ, serialization_format='cdr'))
    t0 = 1000.0
    static = TransformStamped()
    static.header.frame_id, static.child_frame_id = 'fer_link8', 'hsa_tcp'
    _stamp(static.header.stamp, t0)
    static.transform.rotation.w = 1.0
    writer.write('/tf_static', serialize_message(TFMessage(transforms=[static])), int(t0 * 1e9))
    for i in range(100):
        t = t0 + i * 0.02
        tf = TransformStamped()
        tf.header.frame_id, tf.child_frame_id = 'fer_link0', 'fer_link8'
        _stamp(tf.header.stamp, t)
        tf.transform.translation.x = 0.4 + 0.1 * (i / 100)
        tf.transform.translation.z = 0.3
        tf.transform.rotation.w = 1.0
        writer.write('/tf', serialize_message(TFMessage(transforms=[tf])), int(t * 1e9))
        if i % 40 == 0:
            traj = MultiDOFJointTrajectory()
            traj.header.frame_id = 'fer_link0'
            _stamp(traj.header.stamp, t)
            traj.joint_names = ['hsa_tcp']
            for k in range(1, 9):
                pt = MultiDOFJointTrajectoryPoint()
                tr = Transform()
                tr.translation.x = 0.4 + 0.1 * ((i + 5 * k) / 100)
                tr.translation.z = 0.3
                tr.rotation.w = 1.0
                pt.transforms = [tr]
                pt.time_from_start = DurationMsg(sec=0, nanosec=int(k * 0.1 * 1e9))
                traj.points.append(pt)
            writer.write('/polyumi/target_poses_traj', serialize_message(traj), int(t * 1e9))
            writer.write('/polyumi/diag/obs_age_s', serialize_message(Float32(data=0.05)), int(t * 1e9))
        w = WrenchStamped()
        _stamp(w.header.stamp, t)
        w.wrench.force.z = 3.0
        writer.write(
            '/franka_robot_state_broadcaster/external_wrench_in_base_frame', serialize_message(w), int(t * 1e9)
        )
    del writer


@pytest.fixture
def bag(tmp_path):
    """Write a 2 s synthetic rollout bag."""
    path = tmp_path / 'roll'
    _write_bag(path)
    return path


def test_rollout_loads_every_stream(bag):
    """Check chunk times, the replayed TF path, the diag series, the wrench and the tracking error."""
    roll = Rollout(bag)
    assert len(roll.chunks) == 3 and roll.chunks[0][0].shape == (8,) and roll.chunks[0][1].shape == (8, 3)
    assert abs(roll.chunks[0][0][0] - 1000.1) < 1e-6
    assert len(roll.executed_times) > 50 and roll.executed_xyz[:, 0].min() >= 0.4 - 1e-6
    assert 'obs_age_s' in roll.diag and len(roll.diag['obs_age_s'][0]) == 3
    assert len(roll.wrench_times) == 100 and abs(roll.wrench_norm[0] - 3.0) < 1e-9
    t, err = roll.tracking_error()
    assert len(t) > 0 and np.all(err < 0.05)
    assert 'chunks' in roll.summary()

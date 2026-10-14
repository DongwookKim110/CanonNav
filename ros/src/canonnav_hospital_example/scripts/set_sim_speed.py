#!/usr/bin/env python3
"""Run Gazebo at ~speed x real time (0: as fast as possible) by changing its physics update rate.

The step size and the other physics settings of the world are kept; only the update rate
(steps per wall-clock second) changes. The speed is reached only if the machine can keep up.
"""
import rospy
from gazebo_msgs.srv import GetPhysicsProperties, SetPhysicsProperties


def main():
    rospy.init_node("set_sim_speed")
    speed = float(rospy.get_param("~speed", 1.0))
    rospy.wait_for_service("/gazebo/get_physics_properties")
    rospy.wait_for_service("/gazebo/set_physics_properties")
    cur = rospy.ServiceProxy("/gazebo/get_physics_properties", GetPhysicsProperties)()
    rate = speed / cur.time_step if speed > 0 else 0.0
    resp = rospy.ServiceProxy("/gazebo/set_physics_properties", SetPhysicsProperties)(
        cur.time_step, rate, cur.gravity, cur.ode_config)
    if not resp.success:
        raise RuntimeError(resp.status_message)
    rospy.loginfo(f"[sim_speed] {'max' if speed <= 0 else f'{speed:g}x'} real time "
                  f"(step {cur.time_step} s, update rate {rate:g} Hz)")


if __name__ == "__main__":
    main()

"""Drive a QCar 2 around the outside of the QLabs Cityscape for two laps.

Open QLabs in Self-Driving Car Studio > Cityscape first, then run:
    python qcar2_two_laps.py

The car tracks the lane centers of the SDCS roadmap, using a Stanley steering
controller and its position in QLabs. Stop early with Ctrl+C.
"""
import builtins
import sys
import time

import numpy as np

from hal.products.mats import SDCSRoadMap
from qvl.free_camera import QLabsFreeCamera
from qvl.qcar2 import QLabsQCar2
from qvl.qlabs import QuanserInteractiveLabs
from qvl.real_time import QLabsRealTime
import pal.resources.rtmodels as rtmodels

# pal.products.qcar asks which virtual car to use when it is imported
_input = builtins.input
builtins.input = lambda *args: "2"
from pal.products.qcar import QCar
builtins.input = _input

LAPS = 2
# Roadmap nodes for one lap around the outside of the map
ROUTE = [10, 2, 4, 14, 20, 22, 10]
# Cityscape positions are 10 times the roadmap's, which is in QCar scale
MAP_SCALE = 10

SPEED = 0.5             # in QCar scale, so about 5 m/s in the Cityscape
KP_SPEED = 0.1
KI_SPEED = 1
K_STANLEY = 1
MAX_STEERING = np.pi/6
FRONT_AXLE = 0.2        # distance from the car's position to its front axle
TIME_LIMIT = 40       # stop if the laps take longer than this, in seconds


def wrap_to_pi(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi


class SteeringController:
    """Stanley controller that follows a closed path and counts completed laps."""

    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.segments = waypoints.shape[1] - 1
        self.index = 0
        self.crossTrackError = 0

    @property
    def laps(self):
        return self.index // self.segments

    def update(self, position, heading, speed):
        while True:
            start = self.waypoints[:, self.index % self.segments]
            end = self.waypoints[:, self.index % self.segments + 1]
            length = np.linalg.norm(end - start)
            direction = (end - start)/length
            progress = np.dot(position - start, direction)
            if progress < length:
                break
            self.index += 1

        tangent = np.arctan2(direction[1], direction[0])
        offset = start + direction*progress - position
        side = np.sign(wrap_to_pi(np.arctan2(offset[1], offset[0]) - tangent))
        self.crossTrackError = np.linalg.norm(offset)*side

        headingError = wrap_to_pi(tangent - heading)
        steering = headingError + np.arctan2(K_STANLEY*self.crossTrackError, max(speed, 0.1))
        return np.clip(wrap_to_pi(steering), -MAX_STEERING, MAX_STEERING)


def setup_qlabs(startPose):
    qlabs = QuanserInteractiveLabs()
    print("Connecting to QLabs...")
    if not qlabs.open("localhost"):
        sys.exit("Unable to connect to QLabs. Is it open in the Cityscape?")

    # Delete any previous actors and stop any running real-time models
    QLabsRealTime().terminate_all_real_time_models()
    time.sleep(1)
    qlabs.destroy_all_spawned_actors()

    car = QLabsQCar2(qlabs)
    car.spawn_id(actorNumber=0,
                 location=[startPose[0]*MAP_SCALE, startPose[1]*MAP_SCALE, 0],
                 rotation=[0, 0, startPose[2]],
                 waitForConfirmation=True)
    QLabsFreeCamera(qlabs).spawn()
    car.possess()

    QLabsRealTime().start_real_time_model(rtmodels.QCAR2)
    time.sleep(3)
    print("QCar 2 spawned")
    return qlabs, car


def main():
    roadmap = SDCSRoadMap(leftHandTraffic=False)
    waypoints = roadmap.generate_path(ROUTE)
    qlabs, simCar = setup_qlabs(roadmap.get_node_pose(ROUTE[0]).squeeze())

    steeringController = SteeringController(waypoints)
    car = QCar(readMode=1, frequency=60)
    leds = np.zeros(8)
    throttleIntegral = 0
    steering = 0
    throttle = 0
    worstError = 0
    laps = 0

    startTime = time.time()
    lapStart = startTime
    previous = startTime
    lastPrint = startTime

    try:
        while laps < LAPS and time.time() - startTime < TIME_LIMIT:
            now = time.time()
            dt = now - previous
            previous = now

            status, location, rotation, _ = simCar.get_world_transform()
            car.read_write_std(throttle, steering, leds)
            speed = float(car.motorTach)
            if not status:
                continue

            heading = rotation[2]
            position = np.array(location[:2])/MAP_SCALE \
                + FRONT_AXLE*np.array([np.cos(heading), np.sin(heading)])

            steering = steeringController.update(position, heading, speed)

            error = SPEED - speed
            throttleIntegral += error*dt
            throttle = np.clip(KP_SPEED*error + KI_SPEED*throttleIntegral, -0.3, 0.3)

            if now - lastPrint >= 5:
                lastPrint = now
                print(f"{now - startTime:5.1f} s  x={location[0]:7.2f} y={location[1]:7.2f}  "
                      f"speed={speed:.2f}  throttle={throttle:.3f}  steering={steering:.2f}  "
                      f"waypoint {steeringController.index}/{steeringController.segments}")

            worstError = max(worstError, abs(steeringController.crossTrackError)*MAP_SCALE)
            if steeringController.laps > laps:
                laps = steeringController.laps
                print(f"Lap {laps} done in {now - lapStart:.1f} s, "
                      f"largest error from lane center {worstError:.2f} m")
                lapStart = now
                worstError = 0

        if laps < LAPS:
            print(f"Stopped after {laps} laps: reached the {TIME_LIMIT} s time limit")

    except KeyboardInterrupt:
        print("User interrupted!")

    finally:
        car.read_write_std(0, 0, leds)
        car.terminate()
        qlabs.close()


if __name__ == "__main__":
    main()

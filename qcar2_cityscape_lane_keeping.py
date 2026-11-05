"""Spawn a QCar 2 in the QLabs Cityscape and lane keep using its front camera.

Open QLabs in Self-Driving Car Studio > Cityscape first, then run:
    python qcar2_cityscape_lane_keeping.py

The car follows the yellow center line on its left. Stop with Ctrl+C, or press
q or Esc in the Detection Overlay window. Yellow thresholds are from
5_research/sdcs/qcar2/virtual/applications/lane_following.
"""
import argparse
import builtins
import math
import sys
import time

import cv2
import numpy as np

from qvl.free_camera import QLabsFreeCamera
from qvl.qcar2 import QLabsQCar2
from qvl.qlabs import QuanserInteractiveLabs
from qvl.real_time import QLabsRealTime
import pal.resources.rtmodels as rtmodels

# pal.products.qcar asks which virtual car to use when it is imported
_input = builtins.input
builtins.input = lambda *args: "2"
from pal.products.qcar import QCar, QCarCameras
builtins.input = _input

from pal.utilities.math import Filter
from hal.utilities.image_processing import ImageProcessing

# Spawn pose on the Cityscape road, with the yellow line to the car's left
SPAWN_LOCATION = [-0.031, 1.311, 0.0]
SPAWN_ROTATION = [0, 0, -np.pi/2]

THROTTLE = 0.04

IMAGE_WIDTH = 820
IMAGE_HEIGHT = 410
# Road region to the left of the car, where the yellow line appears
ROWS = (270, 400)
COLS = (0, 500)
YELLOW_LOWER = np.array([10, 50, 100])
YELLOW_UPPER = np.array([45, 255, 255])

# Steer to hold the line at TARGET_COL where it crosses TARGET_ROW
TARGET_ROW = 300
TARGET_COL = 250
STEERING_GAIN = 0.002
MIN_LINE_PIXELS = 100


def setup_qlabs():
    qlabs = QuanserInteractiveLabs()
    print("Connecting to QLabs...")
    if not qlabs.open("localhost"):
        sys.exit("Unable to connect to QLabs. Is it open in the Cityscape?")

    # Delete any previous actors and stop any running real-time models
    QLabsRealTime().terminate_all_real_time_models()
    time.sleep(1)
    qlabs.destroy_all_spawned_actors()

    car = QLabsQCar2(qlabs)
    car.spawn_id(actorNumber=0, location=SPAWN_LOCATION, rotation=SPAWN_ROTATION,
                 waitForConfirmation=True)
    QLabsFreeCamera(qlabs).spawn()
    car.possess()

    QLabsRealTime().start_real_time_model(rtmodels.QCAR2)
    time.sleep(2)
    qlabs.close()
    print("QCar 2 spawned")


def lane_steering(frame):
    """Return the steering command and yellow mask for one front camera frame."""
    cropped = frame[ROWS[0]:ROWS[1], COLS[0]:COLS[1]]
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    mask = ImageProcessing.binary_thresholding(frame=hsv,
                                               lowerBounds=YELLOW_LOWER,
                                               upperBounds=YELLOW_UPPER)

    rows, cols = np.nonzero(mask)
    if len(rows) < MIN_LINE_PIXELS:
        # No line in view, such as in an intersection, so drive straight
        return 0.0, mask

    slope, intercept = np.polyfit(rows + ROWS[0], cols + COLS[0], 1)
    lineCol = slope*TARGET_ROW + intercept
    steering = STEERING_GAIN*(TARGET_COL - lineCol)
    return np.clip(steering, -0.5, 0.5), mask


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--duration", type=float, default=None,
                        help="stop after this many seconds (default: run until stopped)")
    args = parser.parse_args()

    setup_qlabs()

    steeringFilter = Filter().low_pass_first_order_variable(25, 0.033)
    next(steeringFilter)
    dt = 0.033

    cameras = QCarCameras(frameWidth=IMAGE_WIDTH, frameHeight=IMAGE_HEIGHT, enableFront=True)
    car = QCar(readMode=1, frequency=60)
    leds = np.zeros(8)
    startTime = time.time()

    try:
        while args.duration is None or time.time() - startTime < args.duration:
            start = time.time()
            cameras.readAll()
            frame = cameras.csiFront.imageData

            rawSteering, mask = lane_steering(frame)
            steering = steeringFilter.send((rawSteering, dt))
            if math.isnan(steering):
                steering = 0

            car.read_write_std(THROTTLE*np.cos(steering), steering, leds)

            overlay = frame.copy()
            overlay[ROWS[0]:ROWS[1], COLS[0]:COLS[1]][mask > 0] = (0, 0, 255)
            cv2.imshow("Detection Overlay", cv2.resize(overlay, (820, 410)))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

            dt = time.time() - start

    except KeyboardInterrupt:
        print("User interrupted!")

    finally:
        car.read_write_std(0, 0, leds)
        cameras.terminate()
        car.terminate()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

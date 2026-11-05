# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#region : File Description and Imports

"""
qlab_setup.py

Simulation setup code for lane keeping in the QLabs Cityscape. Open QLabs in
Self-Driving Car Studio > Cityscape before running. Cloned from the setup code
for the lane keeping lab guide, without its NPC cars, chessboard and timer.
"""

from qvl.qcar2 import QLabsQCar2
from qvl.qlabs import QuanserInteractiveLabs
from qvl.real_time import QLabsRealTime
from qvl.free_camera import QLabsFreeCamera
import pal.resources.rtmodels as rtmodels
import numpy as np
#endregion


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#region : Experiment Configuration

# ===== Starting Pose
# - start_pos: [[x, y, z] in meters, [roll, pitch, yaw] in radians]. The default
#   is on a Cityscape road, with the yellow line on the left of the QCar.
start_pos=[[-0.031,1.311,0],
           [0,0,-np.pi/2]]
#endregion


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#region : Simulation Setup

# initialize qlab
qlab=QuanserInteractiveLabs()
print("Connecting to QLabs...")
if not qlab.open("localhost"):
    print("Unable to connect to QLabs")
    quit()
print("Connected to QLabs")

# Delete any previous QCar instances and stop any running spawn models
qlab.destroy_all_spawned_actors()
QLabsRealTime().terminate_all_real_time_models()

qcar_driver=QLabsQCar2(qlab)
qcar_driver.spawn_id(
        actorNumber=0,
        location=start_pos[0],
        rotation=start_pos[1],
        scale=[1, 1, 1],
        waitForConfirmation=True
    )
# Create a new camera view and attach it to the QCar
camera = QLabsFreeCamera(qlab)
camera.spawn()
qcar_driver.possess()

# Start spawn model
QLabsRealTime().start_real_time_model(rtmodels.QCAR2)

qlab.close()
print('QCar spawned at',start_pos[0])
#endregion

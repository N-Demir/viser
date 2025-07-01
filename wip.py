"""Coordinate frames

In this basic example, we visualize a set of coordinate frames.

Naming for all scene nodes are hierarchical; /tree/branch, for example, is defined
relative to /tree.
"""

import random
import time

import viser

server = viser.ViserServer()

server.scene.add_frame(
    "/tree",
    wxyz=(1.0, 0.0, 0.0, 0.0),
    position=(random.random() * 2.0, 2.0, 0.2),
)

while True:
    time.sleep(0.5)

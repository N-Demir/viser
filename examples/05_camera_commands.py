"""Camera commands

In addition to reads, camera parameters also support writes. These are synced to the
corresponding client automatically.
"""

import time

import numpy as np

import viser
import viser.transforms as tf

server = viser.ViserServer()
num_frames = 20


def turn_tensor_into_string(tensor: np.ndarray) -> str:
    return f"{', '.join([f'{x:.2f}' for x in tensor])}"


def add_camera_position_gui(client: viser.ClientHandle) -> None:
    """Add GUI elements for adjusting the camera position."""

    camera_position_text = client.gui.add_text(
        label="Camera Position",
        initial_value=turn_tensor_into_string(client.camera.position),
    )

    @client.camera.on_update
    def _(_) -> None:
        camera_position_text.value = turn_tensor_into_string(client.camera.position)

    camera_position_set_button = client.gui.add_button(
        label="Set",
    )

    @camera_position_set_button.on_click
    def _(_) -> None:
        client.camera.position = np.array(
            [float(x) for x in camera_position_text.value.split(",")], dtype=np.float32
        )


def add_camera_up_direction_gui(client: viser.ClientHandle) -> None:
    """Add GUI elements for adjusting the camera up direction."""

    camera_up_direction_text = client.gui.add_text(
        label="Camera Up Direction",
        initial_value=turn_tensor_into_string(client.camera.up_direction),
    )

    @client.camera.on_update
    def _(_) -> None:
        camera_up_direction_text.value = turn_tensor_into_string(
            client.camera.up_direction
        )

    camera_up_direction_set_button = client.gui.add_button(
        label="Set",
    )

    @camera_up_direction_set_button.on_click
    def _(_) -> None:
        client.camera.up_direction = np.array(
            [float(x) for x in camera_up_direction_text.value.split(",")],
            dtype=np.float32,
        )


def add_camera_look_at_gui(client: viser.ClientHandle) -> None:
    """Add GUI elements for adjusting the camera look at."""

    camera_look_at_text = client.gui.add_text(
        label="Camera Look At",
        initial_value=turn_tensor_into_string(client.camera.look_at),
    )

    @client.camera.on_update
    def _(_) -> None:
        camera_look_at_text.value = turn_tensor_into_string(client.camera.look_at)

    camera_look_at_set_button = client.gui.add_button(
        label="Set",
    )

    @camera_look_at_set_button.on_click
    def _(_) -> None:
        client.camera.look_at = np.array(
            [float(x) for x in camera_look_at_text.value.split(",")], dtype=np.float32
        )


@server.on_client_connect
def _(client: viser.ClientHandle) -> None:
    """For each client that connects, create GUI elements for adjusting the
    near/far clipping planes."""

    client.camera.far = 10.0

    near_slider = client.gui.add_slider(
        "Near", min=0.01, max=10.0, step=0.001, initial_value=client.camera.near
    )
    far_slider = client.gui.add_slider(
        "Far", min=1, max=20.0, step=0.001, initial_value=client.camera.far
    )

    @near_slider.on_update
    def _(_) -> None:
        client.camera.near = near_slider.value

    @far_slider.on_update
    def _(_) -> None:
        client.camera.far = far_slider.value

    add_camera_position_gui(client)
    add_camera_up_direction_gui(client)
    add_camera_look_at_gui(client)


@server.on_client_connect
def _(client: viser.ClientHandle) -> None:
    """For each client that connects, we create a set of random frames + a click handler for each frame.

    When a frame is clicked, we move the camera to the corresponding frame.
    """

    rng = np.random.default_rng(0)

    def make_frame(i: int) -> None:
        # Sample a random orientation + position.
        wxyz = rng.normal(size=4)
        wxyz /= np.linalg.norm(wxyz)
        position = rng.uniform(-3.0, 3.0, size=(3,))

        # Create a coordinate frame and label.
        frame = client.scene.add_frame(f"/frame_{i}", wxyz=wxyz, position=position)
        client.scene.add_label(f"/frame_{i}/label", text=f"Frame {i}")

        # Move the camera when we click a frame.
        @frame.on_click
        def _(_):
            T_world_current = tf.SE3.from_rotation_and_translation(
                tf.SO3(client.camera.wxyz), client.camera.position
            )
            T_world_target = tf.SE3.from_rotation_and_translation(
                tf.SO3(frame.wxyz), frame.position
            ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

            T_current_target = T_world_current.inverse() @ T_world_target

            for j in range(20):
                T_world_set = T_world_current @ tf.SE3.exp(
                    T_current_target.log() * j / 19.0
                )

                # We can atomically set the orientation and the position of the camera
                # together to prevent jitter that might happen if one was set before the
                # other.
                with client.atomic():
                    client.camera.wxyz = T_world_set.rotation().wxyz
                    client.camera.position = T_world_set.translation()

                client.flush()  # Optional!
                time.sleep(1.0 / 60.0)

            # Mouse interactions should orbit around the frame origin.
            client.camera.look_at = frame.position

    for i in range(num_frames):
        make_frame(i)


while True:
    time.sleep(1.0)

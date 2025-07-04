import { ViewerContext } from "./ViewerContext";
import { CameraControls } from "@react-three/drei";
import { useThree } from "@react-three/fiber";
import * as holdEvent from "hold-event";
import React, { useContext, useRef } from "react";
import { PerspectiveCamera } from "three";
import * as THREE from "three";
import { computeT_threeworld_world } from "./WorldTransformUtils";
import { useThrottledMessageSender } from "./WebsocketUtils";

export function SynchronizedCameraControls() {
  const viewer = useContext(ViewerContext)!;
  const camera = useThree((state) => state.camera as PerspectiveCamera);

  const sendCameraThrottled = useThrottledMessageSender(20).send;

  // Helper for resetting camera poses.
  const initialCameraRef = useRef<{
    camera: PerspectiveCamera;
    lookAt: THREE.Vector3;
  } | null>(null);

  const pivotRef = useRef<THREE.Group>(null);

  const viewerMutable = viewer.mutable.current;

  // Parse camera speed parameter *before* defining sendCamera.
  const searchParams = new URLSearchParams(window.location.search);
  const initialCameraSpeedString = searchParams.get("initialCameraSpeed");
  let initialCameraSpeed = 0.002;
  if (initialCameraSpeedString) {
    const parsedSpeed = parseFloat(initialCameraSpeedString);
    if (!isNaN(parsedSpeed) && parsedSpeed > 0) {
      initialCameraSpeed = parsedSpeed;
    } else {
      console.warn(
        `Invalid initialCameraSpeed parameter: ${initialCameraSpeedString}. Using default.`,
      );
    }
  }

  const updatePivotControlFromCameraLookAtAndup = () => {
    if (!viewerMutable.cameraControl) return;
    if (!pivotRef.current) return;

    const cameraControls = viewerMutable.cameraControl;
    const lookAt = cameraControls.getTarget(new THREE.Vector3());

    // Rotate matrix s.t. it's y-axis aligns with the camera's up vector.
    // We'll do this with math.
    const origRotation = new THREE.Matrix4().extractRotation(
      pivotRef.current.matrix,
    );

    const cameraUp = camera.up.clone().normalize();
    const pivotUp = new THREE.Vector3(0, 1, 0)
      .applyMatrix4(origRotation)
      .normalize();
    const axis = new THREE.Vector3()
      .crossVectors(pivotUp, cameraUp)
      .normalize();
    const angle = Math.acos(Math.min(1, Math.max(-1, cameraUp.dot(pivotUp))));

    // Create rotation matrix.
    const rotationMatrix = new THREE.Matrix4();
    if (axis.lengthSq() > 0.0001) {
      // Check if cross product is valid.
      rotationMatrix.makeRotationAxis(axis, angle);
    }
    // rotationMatrix.premultiply(origRotation);

    // Combine rotation with position.
    const matrix = new THREE.Matrix4();
    matrix.multiply(rotationMatrix);
    matrix.multiply(origRotation);
    matrix.setPosition(lookAt);

    pivotRef.current.matrix.copy(matrix);
    pivotRef.current.updateMatrixWorld(true);
  };

  viewerMutable.resetCameraView = () => {
    camera.up.set(
      initialCameraRef.current!.camera.up.x,
      initialCameraRef.current!.camera.up.y,
      initialCameraRef.current!.camera.up.z,
    );
    viewerMutable.cameraControl!.updateCameraUp();
    viewerMutable.cameraControl!.setLookAt(
      initialCameraRef.current!.camera.position.x,
      initialCameraRef.current!.camera.position.y,
      initialCameraRef.current!.camera.position.z,
      initialCameraRef.current!.lookAt.x,
      initialCameraRef.current!.lookAt.y,
      initialCameraRef.current!.lookAt.z,
      true,
    );
  };

  // Callback for sending cameras.
  // It makes the code more chaotic, but we preallocate a bunch of things to
  // minimize garbage collection!
  const R_threecam_cam = new THREE.Quaternion().setFromEuler(
    new THREE.Euler(Math.PI, 0.0, 0.0),
  );
  const R_world_threeworld = new THREE.Quaternion();
  const tmpMatrix4 = new THREE.Matrix4();
  const lookAt = new THREE.Vector3();
  const R_world_camera = new THREE.Quaternion();
  const t_world_camera = new THREE.Vector3();
  const scale = new THREE.Vector3();
  const sendCamera = React.useCallback(() => {
    updatePivotControlFromCameraLookAtAndup();

    const three_camera = camera;
    const camera_control = viewerMutable.cameraControl;
    const canvas = viewerMutable.canvas!;

    if (camera_control === null) {
      // Camera controls not yet ready, let's re-try later.
      setTimeout(sendCamera, 10);
      return;
    }

    // We put Z up to match the scene tree, and convert threejs camera convention
    // to the OpenCV one.
    const T_world_threeworld = computeT_threeworld_world(viewer).invert();
    const T_world_camera = T_world_threeworld.clone()
      .multiply(
        tmpMatrix4
          .makeRotationFromQuaternion(three_camera.quaternion)
          .setPosition(three_camera.position),
      )
      .multiply(tmpMatrix4.makeRotationFromQuaternion(R_threecam_cam));
    R_world_threeworld.setFromRotationMatrix(T_world_threeworld);

    camera_control.getTarget(lookAt).applyQuaternion(R_world_threeworld);
    const up = three_camera.up.clone().applyQuaternion(R_world_threeworld);

    // Store initial camera values.
    if (initialCameraRef.current === null) {
      initialCameraRef.current = {
        camera: three_camera.clone(),
        lookAt: camera_control.getTarget(new THREE.Vector3()),
      };
    }

    T_world_camera.decompose(t_world_camera, R_world_camera, scale);

    sendCameraThrottled({
      type: "ViewerCameraMessage",
      wxyz: [
        R_world_camera.w,
        R_world_camera.x,
        R_world_camera.y,
        R_world_camera.z,
      ],
      position: t_world_camera.toArray(),
      image_height: canvas.height,
      image_width: canvas.width,
      fov: (three_camera.fov * Math.PI) / 180.0,
      near: three_camera.near,
      far: three_camera.far,
      look_at: [lookAt.x, lookAt.y, lookAt.z],
      up_direction: [up.x, up.y, up.z],
    });

    // Log camera.
    if (logCamera != undefined) {
      console.log(
        `&initialCameraPosition=${t_world_camera.x.toFixed(
          3,
        )},${t_world_camera.y.toFixed(3)},${t_world_camera.z.toFixed(3)}` +
          `&initialCameraLookAt=${lookAt.x.toFixed(3)},${lookAt.y.toFixed(
            3,
          )},${lookAt.z.toFixed(3)}` +
          `&initialCameraUp=${up.x.toFixed(3)},${up.y.toFixed(
            3,
          )},${up.z.toFixed(3)}` +
          `&initialCameraSpeed=${initialCameraSpeed.toFixed(5)}`,
      );
    }
  }, [camera, sendCameraThrottled, initialCameraSpeed]);

  // Camera control search parameters.
  // EXPERIMENTAL: these may be removed or renamed in the future. Please pin to
  // a commit/version if you're relying on this (undocumented) feature.
  const initialCameraPosString = searchParams.get("initialCameraPosition");
  const initialCameraLookAtString = searchParams.get("initialCameraLookAt");
  const initialCameraUpString = searchParams.get("initialCameraUp");
  const logCamera = searchParams.get("logCamera");

  // Send camera for new connections.
  // We add a small delay to give the server time to add a callback.
  const connected = viewer.useGui((state) => state.websocketConnected);
  const initialCameraPositionSet = React.useRef(false);
  React.useEffect(() => {
    if (!initialCameraPositionSet.current) {
      const initialCameraPos = new THREE.Vector3(
        ...((initialCameraPosString
          ? (initialCameraPosString.split(",").map(Number) as [
              number,
              number,
              number,
            ])
          : [0.0, 0.0, 0.0]) as [number, number, number]),
      );
      initialCameraPos.applyMatrix4(computeT_threeworld_world(viewer));
      const initialCameraLookAt = new THREE.Vector3(
        ...((initialCameraLookAtString
          ? (initialCameraLookAtString.split(",").map(Number) as [
              number,
              number,
              number,
            ])
          : [0, 0, -1e-6]) as [number, number, number]),
      );
      initialCameraLookAt.applyMatrix4(computeT_threeworld_world(viewer));
      const initialCameraUp = new THREE.Vector3(
        ...((initialCameraUpString
          ? (initialCameraUpString.split(",").map(Number) as [
              number,
              number,
              number,
            ])
          : [0, 0, 1]) as [number, number, number]),
      );
      initialCameraUp.applyMatrix4(computeT_threeworld_world(viewer));
      initialCameraUp.normalize();

      camera.up.set(initialCameraUp.x, initialCameraUp.y, initialCameraUp.z);
      viewerMutable.cameraControl!.updateCameraUp();

      viewerMutable.cameraControl!.setLookAt(
        initialCameraPos.x,
        initialCameraPos.y,
        initialCameraPos.z,
        initialCameraLookAt.x,
        initialCameraLookAt.y,
        initialCameraLookAt.z,
        false,
      );
      initialCameraPositionSet.current = true;
    }

    viewerMutable.sendCamera = sendCamera;
    if (!connected) return;
    setTimeout(() => sendCamera(), 50);
  }, [connected, sendCamera]);

  // Send camera for 3D viewport changes.
  const canvas = viewerMutable.canvas!; // R3F canvas.
  React.useEffect(() => {
    // Create a resize observer to resize the CSS canvas when the window is resized.
    const resizeObserver = new ResizeObserver(() => {
      sendCamera();
    });
    resizeObserver.observe(canvas);

    // Cleanup.
    return () => resizeObserver.disconnect();
  }, [canvas]);


  // Keyboard controls.
  React.useEffect(() => {
    const cameraControls = viewerMutable.cameraControl!;

    // Use the parsed or default speed.
    const KEYBOARD_MOVE_SPEED = initialCameraSpeed;

    const wKey = new holdEvent.KeyboardKeyHold("KeyW", 20);
    const aKey = new holdEvent.KeyboardKeyHold("KeyA", 20);
    const sKey = new holdEvent.KeyboardKeyHold("KeyS", 20);
    const dKey = new holdEvent.KeyboardKeyHold("KeyD", 20);
    const qKey = new holdEvent.KeyboardKeyHold("KeyQ", 20);
    const eKey = new holdEvent.KeyboardKeyHold("KeyE", 20);

    // TODO: these event listeners are currently never removed, even if this
    // component gets unmounted.
    aKey.addEventListener("holding", (event) => {
      cameraControls.truck(-KEYBOARD_MOVE_SPEED * event?.deltaTime, 0, true);
    });
    dKey.addEventListener("holding", (event) => {
      cameraControls.truck(KEYBOARD_MOVE_SPEED * event?.deltaTime, 0, true);
    });
    wKey.addEventListener("holding", (event) => {
      cameraControls.forward(KEYBOARD_MOVE_SPEED * event?.deltaTime, true);
    });
    sKey.addEventListener("holding", (event) => {
      cameraControls.forward(-KEYBOARD_MOVE_SPEED * event?.deltaTime, true);
    });
    qKey.addEventListener("holding", (event) => {
      cameraControls.elevate(-KEYBOARD_MOVE_SPEED * event?.deltaTime, true);
    });
    eKey.addEventListener("holding", (event) => {
      cameraControls.elevate(KEYBOARD_MOVE_SPEED * event?.deltaTime, true);
    });

    const leftKey = new holdEvent.KeyboardKeyHold("ArrowLeft", 20);
    const rightKey = new holdEvent.KeyboardKeyHold("ArrowRight", 20);
    const upKey = new holdEvent.KeyboardKeyHold("ArrowUp", 20);
    const downKey = new holdEvent.KeyboardKeyHold("ArrowDown", 20);
    leftKey.addEventListener("holding", (event) => {
      cameraControls.rotate(
        -0.05 * THREE.MathUtils.DEG2RAD * event?.deltaTime,
        0,
        true,
      );
    });
    rightKey.addEventListener("holding", (event) => {
      cameraControls.rotate(
        0.05 * THREE.MathUtils.DEG2RAD * event?.deltaTime,
        0,
        true,
      );
    });
    upKey.addEventListener("holding", (event) => {
      cameraControls.rotate(
        0,
        -0.05 * THREE.MathUtils.DEG2RAD * event?.deltaTime,
        true,
      );
    });
    downKey.addEventListener("holding", (event) => {
      cameraControls.rotate(
        0,
        0.05 * THREE.MathUtils.DEG2RAD * event?.deltaTime,
        true,
      );
    });

    // TODO: we currently don't remove any event listeners. This is a bit messy
    // because KeyboardKeyHold attaches listeners directly to the
    // document/window; it's unclear if we can remove these.
    return () => {
      return;
    };
  }, [CameraControls]);

  return (
    <>
      <CameraControls
        ref={(controls) => (viewerMutable.cameraControl = controls)}
        minDistance={0.01}
        dollySpeed={0.3}
        smoothTime={0.05}
        draggingSmoothTime={0.0}
        onChange={sendCamera}
        makeDefault
      />
    </>
  );
}

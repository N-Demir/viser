import { ViewerContext } from "./ViewerContext";
import { makeThrottledMessageSender } from "./WebsocketFunctions";
import { PointerLockControls, OrbitControls } from "@react-three/drei";
import { useThree } from "@react-three/fiber";
import React, { useContext, useEffect, useRef, useCallback, useState } from "react";
import * as THREE from "three";



export function SynchronizedCameraControls() {
  const viewer = useContext(ViewerContext)!;
  const [isFirstPerson, setIsFirstPerson] = useState(false);


  const { camera, gl } = useThree();
  const controlsRef = useRef(null);
  const orbitControlsRef = useRef(null);
  const speed = 0.03;

  const sendCameraThrottled = makeThrottledMessageSender(
    viewer,
    20,
  );

  // helper for resetting camera poses.
  const initialCameraRef = useRef<{
    position: THREE.Vector3;
    rotation: THREE.Euler;
  } | null>(null);

  const pivotRef = useRef<THREE.Group>(null);

  const cameraControlRef = viewer.cameraControlRef;

  // Animation state interface
  interface CameraAnimation {
    startUp: THREE.Vector3;
    targetUp: THREE.Vector3;
    startLookAt: THREE.Vector3;
    targetLookAt: THREE.Vector3;
    startTime: number;
    duration: number;
  }

  const [cameraAnimation, setCameraAnimation] =
    useState<CameraAnimation | null>(null);

  // Animation parameters
  const ANIMATION_DURATION = 0.5; // seconds

  useFrame((state) => {
    if (cameraAnimation && cameraControlRef.current) {
      const cameraControls = cameraControlRef.current;
      const camera = cameraControls.camera;

      const elapsed = state.clock.getElapsedTime() - cameraAnimation.startTime;
      const progress = Math.min(elapsed / cameraAnimation.duration, 1);

      // Smooth step easing
      const t = progress * progress * (3 - 2 * progress);

      // Interpolate up vector
      const newUp = new THREE.Vector3()
        .copy(cameraAnimation.startUp)
        .lerp(cameraAnimation.targetUp, t)
        .normalize();

      // Interpolate look-at position
      const newLookAt = new THREE.Vector3()
        .copy(cameraAnimation.startLookAt)
        .lerp(cameraAnimation.targetLookAt, t);

      camera.up.copy(newUp);

      // Back up position
      const prevPosition = new THREE.Vector3();
      cameraControls.getPosition(prevPosition);

      cameraControls.updateCameraUp();

      // Restore position and set new look-at
      cameraControls.setPosition(
        prevPosition.x,
        prevPosition.y,
        prevPosition.z,
        false,
      );

      cameraControls.setLookAt(
        prevPosition.x,
        prevPosition.y,
        prevPosition.z,
        newLookAt.x,
        newLookAt.y,
        newLookAt.z,
        false,
      );

      // Clear animation when complete
      if (progress >= 1) {
        setCameraAnimation(null);
      }
    }
  });

  const { clock } = useThree();

  const updateCameraLookAtAndUpFromPivotControl = (matrix: THREE.Matrix4) => {
    if (!cameraControlRef.current) return;

    const targetPosition = new THREE.Vector3();
    targetPosition.setFromMatrixPosition(matrix);

    const cameraControls = cameraControlRef.current;
    const camera = cameraControlRef.current.camera;

    // Get target up vector from matrix
    const targetUp = new THREE.Vector3().setFromMatrixColumn(matrix, 1);

    // Get current look-at position
    const currentLookAt = cameraControls.getTarget(new THREE.Vector3());

    // Start new animation
    setCameraAnimation({
      startUp: camera.up.clone(),
      targetUp: targetUp,
      startLookAt: currentLookAt,
      targetLookAt: targetPosition,
      startTime: clock.getElapsedTime(),
      duration: ANIMATION_DURATION,
    });
  };

  const updatePivotControlFromCameraLookAtAndup = () => {
    if (cameraAnimation !== null) return;
    if (!cameraControlRef.current) return;
    if (!pivotRef.current) return;

    const cameraControls = cameraControlRef.current;
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

    // Create rotation matrix
    const rotationMatrix = new THREE.Matrix4();
    if (axis.lengthSq() > 0.0001) {
      // Check if cross product is valid
      rotationMatrix.makeRotationAxis(axis, angle);
    }
    // rotationMatrix.premultiply(origRotation);

    // Combine rotation with position
    const matrix = new THREE.Matrix4();
    matrix.multiply(rotationMatrix);
    matrix.multiply(origRotation);
    matrix.setPosition(lookAt);

    pivotRef.current.matrix.copy(matrix);
    pivotRef.current.updateMatrixWorld(true);
  };

  viewer.resetCameraViewRef.current = () => {
    if (initialCameraRef.current) {
      camera.position.copy(initialCameraRef.current.position);
      camera.rotation.copy(initialCameraRef.current.rotation);
    }
  };

  // Callback for sending cameras.
  const sendCamera = useCallback(() => {
    if (!controlsRef.current && !orbitControlsRef.current) return;

    const { position, quaternion } = camera;
    const rotation = new THREE.Euler().setFromQuaternion(quaternion);

    // Store initial camera values
    if (initialCameraRef.current === null) {
      initialCameraRef.current = {
        position: position.clone(),
        rotation: rotation.clone(),
      };
    }

    sendCameraThrottled({
      type: "ViewerCameraMessage",
      wxyz: [quaternion.w, quaternion.x, quaternion.y, quaternion.z],
      position: position.toArray(),
      aspect: (camera as THREE.PerspectiveCamera).aspect || 1,
      fov: ((camera as THREE.PerspectiveCamera).fov * Math.PI) / 180.0 || 0,
      look_at: [0, 0, 0], // Not used in first-person view
      up_direction: [camera.up.x, camera.up.y, camera.up.z],
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
          )},${up.z.toFixed(3)}`,
      );
    }
  }, [camera, sendCameraThrottled]);

  // new connections.
  const connected = viewer.useGui((state) => state.websocketConnected);
  useEffect(() => {
    viewer.sendCameraRef.current = sendCamera;
    if (!connected) return;
    setTimeout(() => sendCamera(), 50);
  }, [connected, sendCamera]);

  // Send camera for 3D viewport changes.
  const canvas = viewer.canvasRef.current!; // R3F canvas.
  useEffect(() => {
    // Create a resize observer to resize the CSS canvas when the window is resized.
    const resizeObserver = new ResizeObserver(() => {
      sendCamera();
    });
    resizeObserver.observe(canvas);

    // clean up .
    return () => resizeObserver.disconnect();
  }, [canvas]);

  // state for the for camera velocity
  const [velocity, setVelocity] = useState(new THREE.Vector3());

  // Apply velocity to the camera
  useEffect(() => {
    const applyVelocity = () => {
      camera.translateX(velocity.x);
      camera.translateY(velocity.y);
      camera.translateZ(velocity.z);
      sendCamera();

      // ~apply damping to simulate inertia
      velocity.multiplyScalar(0.9);

      // Stop the loop if velocity is very small
      if (velocity.length() > 0.001) {
        requestAnimationFrame(applyVelocity);
      }
    };

    applyVelocity();
  }, [velocity, camera, sendCamera]);

  // Keyboard controls for movement.
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const newVelocity = velocity.clone();
      switch (event.key) {
        case 'w':
          newVelocity.z -= speed;
          break;
        case 's':
          newVelocity.z += speed;
          break;
        case 'a':
          newVelocity.x -= speed;
          break;
        case 'd':
          newVelocity.x += speed;
          break;
        case 'q':
          newVelocity.y -= speed;
          break;
        case 'e':
          newVelocity.y += speed;
          break;
        case 'p':
        
          setIsFirstPerson(prev => {
            if (prev) {
              // If switching from first-person to orbit, release the pointer lock
              document.exitPointerLock();
            }
            return !prev;});
        break;
        default:
          break;
      }
      setVelocity(newVelocity);
    };

    window.addEventListener('keydown', handleKeyDown);

    // Cleanup event listener on component unmount
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [velocity]);

  return (
    <>
    {isFirstPerson ? (
      <PointerLockControls ref={controlsRef} args={[camera, gl.domElement]} />
    ) : (
      <OrbitControls ref={orbitControlsRef} args={[camera, gl.domElement]} />
    )}
  </>
  );
}

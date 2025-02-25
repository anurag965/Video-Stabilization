import cv2
import numpy as np
from datetime import datetime
import signal
import sys

# Constants
HORIZONTAL_BORDER_CROP = 20
FPS = 30.0
MAX_DISPLACEMENT = 100  # Maximum allowed displacement in pixels
MAX_ANGLE = 0.2  # Maximum allowed rotation in radians

class TransformParam:
    def __init__(self, dx=0, dy=0, da=0):
        self.dx = dx
        self.dy = dy
        self.da = da

class Trajectory:
    def __init__(self, x=0, y=0, a=0):
        self.x = x
        self.y = y
        self.a = a

    def __add__(self, other):
        return Trajectory(self.x + other.x, self.y + other.y, self.a + other.a)

    def __sub__(self, other):
        return Trajectory(self.x - other.x, self.y - other.y, self.a - other.a)

    def __mul__(self, other):
        return Trajectory(self.x * other.x, self.y * other.y, self.a * other.a)

    def __truediv__(self, other):
        return Trajectory(self.x / other.x, self.y / other.y, self.a / other.a)

def get_initial_transform():
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

def constrain_movement(dx, dy, da):
    """Constrain the movement within acceptable limits"""
    dx = np.clip(dx, -MAX_DISPLACEMENT, MAX_DISPLACEMENT)
    dy = np.clip(dy, -MAX_DISPLACEMENT, MAX_DISPLACEMENT)
    da = np.clip(da, -MAX_ANGLE, MAX_ANGLE)
    return dx, dy, da

def signal_handler(sig, frame):
    print("\nStopping recording...")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    ret, prev = cap.read()
    if not ret:
        print("Error: Could not read from camera")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    original_video = cv2.VideoWriter(
        f'original_{timestamp}.avi',
        fourcc,
        FPS,
        (frame_width, frame_height)
    )
    
    stabilized_video = cv2.VideoWriter(
        f'stabilized_{timestamp}.avi',
        fourcc,
        FPS,
        (frame_width, frame_height)
    )

    prev_grey = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    x, y, a = 0.0, 0.0, 0.0
    last_T = get_initial_transform()

    # Kalman filter parameters
    X = Trajectory()
    P = Trajectory(1, 1, 1)
    Q = Trajectory(1e-3, 1e-3, 1e-3)  # Increased process noise
    R = Trajectory(0.5, 0.5, 0.5)      # Decreased measurement noise

    # Window size for moving average smoothing
    window_size = 15
    dx_history = []
    dy_history = []
    da_history = []

    frame_count = 0
    print("Recording started. Press Ctrl+C to stop...")

    try:
        while True:
            ret, cur = cap.read()
            if not ret:
                break

            original_video.write(cur)

            cur_grey = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

            # Initialize corner lists
            prev_corner2, cur_corner2 = [], []

            # Increase the number of features to track
            prev_corner = cv2.goodFeaturesToTrack(prev_grey, 500, 0.01, 20)

            if prev_corner is not None:
                cur_corner, status, err = cv2.calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_corner, None)

                for i, s in enumerate(status):
                    if s:
                        prev_corner2.append(prev_corner[i])
                        cur_corner2.append(cur_corner[i])

            if len(prev_corner2) >= 3:
                T = get_initial_transform()
                try:
                    T_estimate, _ = cv2.estimateAffinePartial2D(np.array(prev_corner2), np.array(cur_corner2))
                    if T_estimate is not None:
                        T = T_estimate.astype(np.float64)
                except cv2.error:
                    T = last_T.copy()
            else:
                T = last_T.copy()

            last_T = T.copy()

            dx = T[0, 2]
            dy = T[1, 2]
            da = np.arctan2(T[1, 0], T[0, 0])

            # Update moving average history
            dx_history.append(dx)
            dy_history.append(dy)
            da_history.append(da)
            if len(dx_history) > window_size:
                dx_history.pop(0)
                dy_history.pop(0)
                da_history.pop(0)

            # Calculate moving averages
            dx_smooth = np.mean(dx_history)
            dy_smooth = np.mean(dy_history)
            da_smooth = np.mean(da_history)

            # Apply stronger smoothing with increased accumulation
            x = 0.95 * x + dx_smooth
            y = 0.95 * y + dy_smooth
            a = 0.95 * a + da_smooth

            z = Trajectory(x, y, a)
            X_ = X
            P_ = P + Q
            K = P_ / (P_ + R)
            X = X_ + K * (z - X_)
            P = (Trajectory(1, 1, 1) - K) * P_

            diff_x = X.x - x
            diff_y = X.y - y
            diff_a = X.a - a

            # Apply movement constraints
            dx, dy, da = constrain_movement(diff_x, diff_y, diff_a)

            T_smooth = np.array([[np.cos(da), -np.sin(da), dx],
                               [np.sin(da), np.cos(da), dy]], dtype=np.float64)

            try:
                stabilized_frame = cv2.warpAffine(cur, T_smooth, (cur.shape[1], cur.shape[0]))
            except cv2.error:
                stabilized_frame = cur.copy()

            # Increased border crop to handle larger movements
            vert_border = HORIZONTAL_BORDER_CROP * cur.shape[0] // cur.shape[1]
            stabilized_frame = stabilized_frame[vert_border:stabilized_frame.shape[0]-vert_border,
                                             HORIZONTAL_BORDER_CROP:stabilized_frame.shape[1]-HORIZONTAL_BORDER_CROP]
            stabilized_frame = cv2.resize(stabilized_frame, (cur.shape[1], cur.shape[0]))

            stabilized_video.write(stabilized_frame)

            prev = cur.copy()
            prev_grey = cur_grey.copy()

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"\rFrames recorded: {frame_count}", end="")

    except KeyboardInterrupt:
        print("\nStopping recording...")
    finally:
        cap.release()
        original_video.release()
        stabilized_video.release()
        print(f"\nVideos saved as 'original_{timestamp}.avi' and 'stabilized_{timestamp}.avi'")

if __name__ == "__main__":
    main()
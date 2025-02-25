import cv2
import numpy as np

# Constants
HORIZONTAL_BORDER_CROP = 20  # In pixels. Crops the border to reduce the black borders from stabilization being too noticeable.

class TransformParam:
    def __init__(self, dx=0, dy=0, da=0):
        self.dx = dx
        self.dy = dy
        self.da = da  # angle

class Trajectory:
    def __init__(self, x=0, y=0, a=0):
        self.x = x
        self.y = y
        self.a = a  # angle

    def __add__(self, other):
        return Trajectory(self.x + other.x, self.y + other.y, self.a + other.a)

    def __sub__(self, other):
        return Trajectory(self.x - other.x, self.y - other.y, self.a - other.a)

    def __mul__(self, other):
        return Trajectory(self.x * other.x, self.y * other.y, self.a * other.a)

    def __truediv__(self, other):
        return Trajectory(self.x / other.x, self.y / other.y, self.a / other.a)

    def __repr__(self):
        return f"Trajectory(x={self.x}, y={self.y}, a={self.a})"

def main():
    # Open the camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Read the first frame
    ret, prev = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        cap.release()
        return

    prev_grey = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Initialize output files
    out_transform = open("prev_to_cur_transformation.txt", "w")
    out_trajectory = open("trajectory.txt", "w")
    out_smoothed_trajectory = open("smoothed_trajectory.txt", "w")
    out_new_transform = open("new_prev_to_cur_transformation.txt", "w")

    # Initialize variables
    prev_to_cur_transform = []
    x, y, a = 0, 0, 0  # Accumulated frame to frame transform
    smoothed_trajectory = []
    last_T = None
    k = 1
    max_frames = 10000000000  # Set a limit to avoid infinite loop

    # Define process noise covariance Q and measurement noise covariance R
    pstd = 1e-5  # Process noise standard deviation
    cstd = 0.001  # Measurement noise standard deviation
    Q = Trajectory(pstd, pstd, pstd)  # Process noise covariance
    R = Trajectory(cstd, cstd, cstd)  # Measurement noise covariance

    # Calculate vertical border for cropping
    vert_border = HORIZONTAL_BORDER_CROP * prev.shape[0] // prev.shape[1]
    
    # Replace the single video writer with two separate writers
    original_video = cv2.VideoWriter("original.avi", cv2.VideoWriter_fourcc(*'XVID'), 60, (prev.shape[1], prev.shape[0]))
    stabilized_video = cv2.VideoWriter("stabilized.avi", cv2.VideoWriter_fourcc(*'XVID'), 60, (prev.shape[1], prev.shape[0]))

    while True:
        ret, cur = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cur_grey = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

        # Detect good features to track
        prev_corner = cv2.goodFeaturesToTrack(prev_grey, 200, 0.005, 100)

        # Ensure prev_corner is a numpy array of float32
        prev_corner = np.float32(prev_corner).reshape(-1, 1, 2) if prev_corner is not None else None
        prev_corner2, cur_corner2 = [], []

        # Calculate optical flow only if prev _corner is not None
        if prev_corner is not None:
            cur_corner, status, err = cv2.calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_corner, None)

            # Weed out bad matches
            for i, s in enumerate(status):
                if s:
                    prev_corner2.append(prev_corner[i])
                    cur_corner2.append(cur_corner[i])

        # Translation + rotation only
        if len(prev_corner2) >= 3 and len(cur_corner2) >= 3:  # Ensure there are enough points
            T, inliers = cv2.estimateAffinePartial2D(np.array(prev_corner2), np.array(cur_corner2))
        else:
            T = last_T if last_T is not None else np.eye(2, 3)  # Use identity if no transform found

        last_T = T

        # Decompose T
        dx = T[0, 2]
        dy = T[1, 2]
        da = np.arctan2(T[1, 0], T[0, 0])

        out_transform.write(f"{k} {dx} {dy} {da}\n")

        # Accumulated frame to frame transform
        x += dx
        y += dy
        a += da
        out_trajectory.write(f"{k} {x} {y} {a}\n")

        z = Trajectory(x, y, a)

        if k == 1:
            # Initial guesses
            X = Trajectory(0, 0, 0)  # Initial estimate
            P = Trajectory(1, 1, 1)  # Set error variance
        else:
            # Time update (prediction)
            X_ = X
            P_ = P + Q  # P_(k) = P(k-1) + Q
            # Measurement update (correction)
            K = P_ / (P_ + R)  # Gain
            X = X_ + K * (z - X_)  # Update state estimate
            P = (Trajectory(1, 1, 1) - K) * P_  # Update error covariance

        out_smoothed_trajectory.write(f"{k} {X.x} {X.y} {X.a}\n")

        # Target - current
        diff_x = X.x - x
        diff_y = X.y - y
        diff_a = X.a - a

        dx += diff_x
        dy += diff_y
        da += diff_a

        out_new_transform.write(f"{k} {dx} {dy} {da}\n")

        # Update transformation matrix
        T[0, 0] = np.cos(da)
        T[0, 1] = -np.sin(da)
        T[1, 0] = np.sin(da)
        T[1, 1] = np.cos(da)
        T[0, 2] = dx
        T[1, 2] = dy

        cur2 = cv2.warpAffine(prev, T, (cur.shape[1], cur.shape[0]))

        # Crop the stabilized frame
        cur2 = cur2[vert_border:cur2.shape[0] - vert_border, HORIZONTAL_BORDER_CROP:cur2.shape[1] - HORIZONTAL_BORDER_CROP]

        # Resize cur2 back to cur size for better side-by-side comparison
        cur2 = cv2.resize(cur2, (cur.shape[1], cur.shape[0]))

        # Instead of creating canvas and showing frames, write them to files
        original_video.write(prev)
        stabilized_video.write(cur2)

        # Optional: Show progress in console
        print(f"Frame: {k} - good optical flow: {len(prev_corner2)}")
        
        # Check for 'q' key press if you still want to allow manual stopping
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev = cur.copy()
        prev_grey = cur_grey.copy()
        k += 1

        if k > max_frames:
            break

    # Update the cleanup section to include new video writers
    out_transform.close()
    out_trajectory.close()
    out_smoothed_trajectory.close()
    out_new_transform.close()
    cap.release()
    original_video.release()
    stabilized_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
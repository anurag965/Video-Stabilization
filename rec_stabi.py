import cv2
import numpy as np

# Constants
HORIZONTAL_BORDER_CROP = 50 # In pixels. Crops the border to reduce the black borders from stabilization being too noticeable.

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

def main(video_path):
    out_transform = open("prev_to_cur_transformation.txt", "w")
    out_trajectory = open("trajectory.txt", "w")
    out_smoothed_trajectory = open("smoothed_trajectory.txt", "w")
    out_new_transform = open("new_prev_to_cur_transformation.txt", "w")

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()

    ret, prev = cap.read()
    prev_grey = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Step 1 - Get previous to current frame transformation (dx, dy, da) for all frames
    prev_to_cur_transform = []  # previous to current
    x, y, a = 0, 0, 0  # Accumulated frame to frame transform

    # Step 2 - Accumulate the transformations to get the image trajectory
    trajectory = []  # trajectory at all frames

    # Step 3 - Smooth out the trajectory using an averaging window
    smoothed_trajectory = []  # trajectory at all frames
    X = Trajectory()  # posteriori state estimate
    X_ = Trajectory()  # priori estimate
    P = Trajectory()  # posteriori estimate error covariance
    P_ = Trajectory()  # priori estimate error covariance
    K = Trajectory()  # gain
    z = Trajectory()  # actual measurement
    pstd = 1e-7 # can be changed
    cstd = 0.00005  # can be changed
    Q = Trajectory(pstd, pstd, pstd)  # process noise covariance
    R = Trajectory(cstd, cstd, cstd)  # measurement noise covariance 

    # Step 4 - Generate new set of previous to current transform
    new_prev_to_cur_transform = []

    # Step 5 - Apply the new transformation to the video
    vert_border = HORIZONTAL_BORDER_CROP * prev.shape[0] // prev.shape[1]  # get the aspect ratio correct
    outputVideo = cv2.VideoWriter("compare.avi", cv2.VideoWriter_fourcc(*'XVID'), 24, (prev.shape[1], prev.shape[0] * 2 + 10))

    k = 1
    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last_T = None

    while True:
        ret, cur = cap.read()
        if not ret:
            break

        cur_grey = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

        # vector from prev to current
        prev_corner = cv2.goodFeaturesToTrack(prev_grey, 200, 0.01, 30)
        prev_corner2, cur_corner2 = [], []

        # Calculate optical flow
        cur_corner, status, err = cv2.calcOpticalFlowPyrLK(prev_grey, cur_grey, np.array(prev_corner), None)

        # weed out bad matches
        for i, s in enumerate(status):
            if s:
                prev_corner2.append(prev_corner[i])
                cur_corner2.append(cur_corner[i])

        # translation + rotation only
        if len(prev_corner2) >= 3 and len(cur_corner2) >= 3:  # Ensure there are enough points
            T, inliers = cv2.estimateAffinePartial2D(np.array(prev_corner2), np.array(cur_corner2))
        else:
            T = None

        # in rare cases no transform is found. We'll just use the last known good transform.
        if T is None:
            if last_T is not None:
                T = last_T
            else:
                continue

        last_T = T

        # decompose T
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
            # initial guesses
            X = Trajectory(0, 0, 0)  # Initial estimate, set 0
            P = Trajectory(1, 1, 1)  # set error variance, set 1
        else:
            # time update (prediction)
            X_ = X  # X_(k) = X(k-1)
            P_ = P + Q  # P_(k) = P(k-1) + Q
            # measurement update (correction)
            K = P_ / (P_ + R)  # gain; K(k) = P_(k) / (P_(k) + R)
            X = X_ + K * (z - X_)  # z - X_ is residual, X(k) = X_(k) + K(k) * (z(k) - X_(k))
            P = (Trajectory(1, 1, 1) - K) * P_  # P(k) = (1 - K(k)) * P_(k)

        out_smoothed_trajectory.write(f"{k} {X.x} {X.y} {X.a}\n")

        # target - current
        diff_x = X.x - x
        diff_y = X.y - y
        diff_a = X.a - a

        dx += diff_x
        dy += diff_y
        da += diff_a

        out_new_transform.write(f"{k} {dx} {dy} {da}\n")

        T[0, 0] = np.cos(da)
        T[0, 1] = -np.sin(da)
        T[1, 0] = np.sin(da)
        T[1, 1] = np.cos(da)

        T[0, 2] = dx
        T[1, 2] = dy

        cur2 = cv2.warpAffine(prev, T, (cur.shape[1], cur.shape[0]))

        cur2 = cur2[vert_border:cur2.shape[0] - vert_border, HORIZONTAL_BORDER_CROP:cur2.shape[1] - HORIZONTAL_BORDER_CROP]

        # Resize cur2 back to cur size, for better side by side comparison
        cur2 = cv2.resize(cur2, (cur.shape[1], cur.shape[0]))

        # Now draw the original and stabilized side by side for coolness
        canvas = np.zeros((cur.shape[0], cur.shape[1] * 2 + 10, 3), dtype=np.uint8)

        canvas[:, :cur.shape[1]] = prev
        canvas[:, cur.shape[1] + 10:cur.shape[1] * 2 + 10] = cur2

        # If too big to fit on the screen, then scale it down by 2, hopefully it'll fit :)
        if canvas.shape[1] > 1920:
            canvas = cv2.resize(canvas, (canvas.shape[1] // 2, canvas.shape[0] // 2))

        cv2.imshow("before and after", canvas)

        cv2.waitKey(10)

        prev = cur.copy()
        prev_grey = cur_grey.copy()

        print(f"Frame: {k}/{max_frames} - good optical flow: {len(prev_corner2)}")
        k += 1

    out_transform.close()
    out_trajectory.close()
    out_smoothed_trajectory.close()
    out_new_transform.close()
    cap.release()
    outputVideo.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("./VideoStab [video.avi]")
    else:
        main(sys.argv[1])
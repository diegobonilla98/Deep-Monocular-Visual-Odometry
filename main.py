import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from MiDaS import depth_utils

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

depth_utils.load_model()
akaze = cv2.AKAZE_create()
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)

K = np.array(
    [[587.0695422027679, 0.0, 464.7514199201558], [0.0, 589.7166732532896, 196.00945613330055], [0.0, 0.0, 1.0]])
D = np.array([[0.4160473832685788], [-1.3332637862637917], [11.816418300368255], [-26.426759129180237]])

camera_trajectory = []
camera_pos = np.array([[0], [0], [0]], 'float32')

video_path = 'VID_20201227_172158.mp4'
video = cv2.VideoCapture(video_path)

prev_frame = None

frame_count = 0
skip_every = 1
while True:
    frame_count += 1
    ret, frame = video.read()
    if not ret or frame_count % skip_every != 0:
        continue

    frame = cv2.pyrDown(frame)
    frame = cv2.pyrDown(frame)
    if prev_frame is None:
        prev_frame = frame
        continue
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp_model, des_model = akaze.detectAndCompute(prev_gray, None)
    kp_frame, des_frame = akaze.detectAndCompute(gray, None)
    matches = matcher.knnMatch(des_model, des_frame, 2)
    good = []
    nn_match_ratio = 0.9
    for m, n in matches:
        if m.distance < nn_match_ratio * n.distance:
            good.append(m)
    if len(good) < 21:
        continue

    src_pts = []
    depth = depth_utils.predict_depth(prev_frame)
    out = -75.629 + 254.129 * np.exp(-0.0011 * (depth.astype('float32')))
    for m in good:
        x, y = kp_model[m.queryIdx].pt
        src_pts.append((x, y, out[int(y), int(x)]))
    src_pts = np.float32(src_pts).reshape(-1, 1, 3)
    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    retval, rvec, tvec = cv2.solvePnP(src_pts, dst_pts, K, D)
    camera_pos += tvec

    ax.scatter(camera_pos[0][0], camera_pos[1][0], camera_pos[2][0], c=np.array([0., 0.3, 0.9]).reshape(1, -1))
    plt.pause(0.001)

    prev_frame = frame
    cv2.imshow('Result', gray)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
video.release()

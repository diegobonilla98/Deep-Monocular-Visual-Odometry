import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Estimator import rigid_transform_3D
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

video_path = 'VID_20201227_152639.mp4'
video = cv2.VideoCapture(video_path)

prev_frame = None

frame_count = 0
skip_every = 10
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
    dst_points = []
    depth_A = depth_utils.predict_depth(prev_frame)
    depth_B = depth_utils.predict_depth(frame)
    for m in good:
        x, y = kp_model[m.queryIdx].pt
        src_pts.append((x, y, depth_A[int(y), int(x)]))

        x, y = kp_frame[m.trainIdx].pt
        dst_points.append((x, y, depth_B[int(y), int(x)]))

    src_pts = np.float32(src_pts).T
    dst_points = np.float32(dst_points).T

    R, t = rigid_transform_3D(src_pts, dst_points)
    # camera_pos = np.dot(R, camera_pos) + t

    a = np.pi/2
    R90 = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]], 'float32')
    t = np.dot(R90, t)
    t = np.dot(R, t)
    camera_pos += t

    # ax.plot3D(plot_cam[:, 0, 0], plot_cam[:, 1, 0], plot_cam[:, 2, 0])
    ax.scatter(-camera_pos[0][0], -camera_pos[1][0], camera_pos[2][0], c=np.array([0., 0.3, 0.9]).reshape(1, -1))
    ax.set_xlim3d(0, 1200)
    ax.set_ylim3d(-100, 400)
    ax.set_zlim3d(-100, 500)
    plt.pause(0.001)

    # plt.scatter(-camera_pos[0][0], camera_pos[1][0], c=np.array([0., 0.3, 0.9]).reshape(1, -1))
    # plt.axis([0, 1200, -150, 500])
    # plt.pause(0.001)

    prev_frame = frame
    cv2.imshow('Result', gray)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
video.release()

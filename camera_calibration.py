import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# (8,6) is for the given testing images.
# If you use the another data (e.g. pictures you take by your smartphone),
# you need to set the corresponding numbers.
corner_x = 7
corner_y = 7
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = sorted(glob.glob('data/*.jpg'))

# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)

    # 0003 and 0008 are landscape, the rest are portrait.
    # One K can't fit both orientations, so use portrait only.
    if img.shape[0] < img.shape[1]:
        print('skip landscape image', fname)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    print('find the chessboard corners of', fname)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x, corner_y), None)

    # If found, add object points, image points
    if ret == True:
        # subpixel refinement -- more accurate corner locations
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), crit)
        objpoints.append(objp)
        imgpoints.append(corners)


#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################


# ---------- helper: normalize 2D points so SVD is well-conditioned ----------
# translate so centroid=0, scale so mean distance = sqrt(2)
# (this is the Hartley trick from Multiple View Geometry)
def normalize(pts):
    mu = pts.mean(axis=0)
    d = np.sqrt(((pts - mu)**2).sum(axis=1)).mean()
    s = np.sqrt(2) / d
    T = np.array([[s, 0, -s*mu[0]],
                  [0, s, -s*mu[1]],
                  [0, 0,        1]])
    # apply T
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    pts_n = (T @ pts_h.T).T[:, :2]
    return pts_n, T


# ---------- Step 1: homography H for one image ----------
# DLT: build Ah=0 from N point pairs, solve by SVD
def find_homography(world, image):
    w, Tw = normalize(world)
    p, Tp = normalize(image)

    A = []
    for i in range(len(w)):
        X, Y = w[i]
        u, v = p[i]
        A.append([-X, -Y, -1,  0,  0,  0,  u*X, u*Y, u])
        A.append([ 0,  0,  0, -X, -Y, -1,  v*X, v*Y, v])
    A = np.array(A)

    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]               # smallest singular value's right singular vector
    Hn = h.reshape(3, 3)

    # undo the normalization
    H = np.linalg.inv(Tp) @ Hn @ Tw
    return H / H[2, 2]


# ---------- Step 2: use all H to solve for intrinsics K ----------
# Zhang's two constraints per image:
#   h1^T B h2 = 0
#   h1^T B h1 = h2^T B h2
# B = K^-T K^-1 is symmetric -> 6 unknowns b = (B11,B12,B22,B13,B23,B33)
# Each (i,j) gives a row v_ij, stack all and solve Vb=0 by SVD,
# then recover K from B by Cholesky.
def v_row(H, i, j):
    # row of V that encodes h_i^T B h_j as a linear combo of b
    return np.array([
        H[0,i]*H[0,j],
        H[0,i]*H[1,j] + H[1,i]*H[0,j],
        H[1,i]*H[1,j],
        H[2,i]*H[0,j] + H[0,i]*H[2,j],
        H[2,i]*H[1,j] + H[1,i]*H[2,j],
        H[2,i]*H[2,j],
    ])


def find_K(Hs):
    V = []
    for H in Hs:
        V.append(v_row(H, 0, 1))
        V.append(v_row(H, 0, 0) - v_row(H, 1, 1))
    V = np.array(V)

    U, S, Vt = np.linalg.svd(V)
    b = Vt[-1]
    # reshape to B
    B = np.array([[b[0], b[1], b[3]],
                  [b[1], b[2], b[4]],
                  [b[3], b[4], b[5]]])

    # sign of b is arbitrary -- fix it so B is positive-definite
    if B[0,0] < 0:
        B = -B

    # Cholesky: B = L L^T  -> K = (L^T)^-1
    L = np.linalg.cholesky(B)
    K = np.linalg.inv(L.T)
    K /= K[2,2]
    return K


# ---------- Step 3: extrinsics [R|t] for each image ----------
# from H = K[r1 r2 t]:  lambda = 1/||K^-1 h1||
#                       r1 = lam * K^-1 h1
#                       r2 = lam * K^-1 h2
#                       r3 = r1 x r2
#                       t  = lam * K^-1 h3
def find_Rt(K, H):
    Kinv = np.linalg.inv(K)
    h1, h2, h3 = H[:,0], H[:,1], H[:,2]

    # average the two scales for a bit of stability
    lam = 2.0 / (np.linalg.norm(Kinv @ h1) + np.linalg.norm(Kinv @ h2))
    # wait, that's 1/((||..||+||..||)/2) = 2/(sum)

    r1 = lam * (Kinv @ h1)
    r2 = lam * (Kinv @ h2)
    r3 = np.cross(r1, r2)
    t  = lam * (Kinv @ h3)

    # board should be in front of camera
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t
        r3 = np.cross(r1, r2)

    # snap R to the closest orthogonal matrix (numerical cleanup)
    Q = np.column_stack([r1, r2, r3])
    U, S, Vt = np.linalg.svd(Q)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R

    Rt = np.zeros((3, 4))
    Rt[:, :3] = R
    Rt[:,  3] = t
    return Rt


# ================== run the pipeline ==================
print('Camera calibration...')
img_size = img[0].shape

# 1) per-image homographies
Hs = []
for i in range(len(objpoints)):
    world = objpoints[i][:, :2]
    image = imgpoints[i].reshape(-1, 2)
    Hs.append(find_homography(world, image))

# 2) intrinsics
mtx = find_K(Hs)
print('K =')
print(mtx)

# 3) extrinsics -- shape (n, 3, 4) as required by the visualizer
extrinsics = np.zeros((len(Hs), 3, 4))
for i, H in enumerate(Hs):
    extrinsics[i] = find_Rt(mtx, H)

# quick check: reprojection error
err = 0; n = 0
for i, H in enumerate(Hs):
    P = mtx @ extrinsics[i]
    X = np.hstack([objpoints[i], np.ones((len(objpoints[i]), 1))])
    p = (P @ X.T).T
    p = p[:, :2] / p[:, 2:3]
    err += np.linalg.norm(p - imgpoints[i].reshape(-1,2), axis=1).sum()
    n += len(p)
print('mean reprojection error: %.2f px' % (err / n))


# show the camera extrinsics
print('Show the camera extrinsics')
# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
# camera setting
camera_matrix = mtx
cam_width = 0.064/0.1
cam_height = 0.032/0.1
scale_focal = 1600
# chess board setting
# 7x7 corners span 6x6 squares
board_width = 6
board_height = 6
square_size = 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, extrinsics, board_width,
                                                board_height, square_size, True)

X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters Visualization')
plt.savefig('extrinsics_result.png', dpi=120, bbox_inches='tight')
plt.show()

#animation for rotating plot
"""
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
"""

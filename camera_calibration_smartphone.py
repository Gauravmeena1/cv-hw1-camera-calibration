# Phone images are at higher resolution than the TA data, and the
# chessboard detector is slow on full-size images. I detect the corners
# on a 0.4x downscaled copy, then scale back up and subpixel-refine on
# the fullres image.
# The three core functions (find_homography, find_K, find_Rt) are the
# same as in camera_calibration.py.
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show


corner_x = 7
corner_y = 7
objp = np.zeros((corner_x*corner_y, 3), np.float32)
objp[:, :2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1, 2)


def normalize(pts):
    mu = pts.mean(axis=0)
    d = np.sqrt(((pts - mu)**2).sum(axis=1)).mean()
    s = np.sqrt(2) / d
    T = np.array([[s, 0, -s*mu[0]],
                  [0, s, -s*mu[1]],
                  [0, 0,        1]])
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    pts_n = (T @ pts_h.T).T[:, :2]
    return pts_n, T


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
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Tp) @ Hn @ Tw
    return H / H[2, 2]


def v_row(H, i, j):
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
    B = np.array([[b[0], b[1], b[3]],
                  [b[1], b[2], b[4]],
                  [b[3], b[4], b[5]]])
    if B[0,0] < 0:
        B = -B
    L = np.linalg.cholesky(B)
    K = np.linalg.inv(L.T)
    K /= K[2,2]
    return K


def find_Rt(K, H):
    Kinv = np.linalg.inv(K)
    h1, h2, h3 = H[:,0], H[:,1], H[:,2]
    lam = 2.0 / (np.linalg.norm(Kinv @ h1) + np.linalg.norm(Kinv @ h2))
    r1 = lam * (Kinv @ h1)
    r2 = lam * (Kinv @ h2)
    r3 = np.cross(r1, r2)
    t  = lam * (Kinv @ h3)
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t
        r3 = np.cross(r1, r2)
    Q = np.column_stack([r1, r2, r3])
    U, S, Vt = np.linalg.svd(Q)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R
    Rt = np.zeros((3, 4))
    Rt[:, :3] = R
    Rt[:,  3] = t
    return Rt


# ==============================================================
objpoints, imgpoints = [], []
images = sorted(glob.glob('my_data/*.jpg'))

print('finding corners in phone images...')
for fname in images:
    img = cv2.imread(fname)
    # downscale first -- fullres takes forever on 4K images
    small = cv2.resize(img, None, fx=0.4, fy=0.4)
    gray_s = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    ret, c_small = cv2.findChessboardCorners(gray_s, (corner_x, corner_y), None)
    if ret:
        # scale corners back up, then subpixel-refine on fullres
        c = (c_small / 0.4).astype(np.float32)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        c = cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), crit)
        objpoints.append(objp)
        imgpoints.append(c)
        print(' ', fname, 'OK')
    else:
        print(' ', fname, 'no corners found')

print('using %d / %d images' % (len(objpoints), len(images)))

Hs = [find_homography(objpoints[i][:, :2], imgpoints[i].reshape(-1, 2))
      for i in range(len(objpoints))]

K = find_K(Hs)
print('K_phone =')
print(K)

extrinsics = np.zeros((len(Hs), 3, 4))
for i, H in enumerate(Hs):
    extrinsics[i] = find_Rt(K, H)

err = 0; n = 0
for i, H in enumerate(Hs):
    P = K @ extrinsics[i]
    X = np.hstack([objpoints[i], np.ones((len(objpoints[i]), 1))])
    p = (P @ X.T).T
    p = p[:, :2] / p[:, 2:3]
    err += np.linalg.norm(p - imgpoints[i].reshape(-1, 2), axis=1).sum()
    n += len(p)
print('mean reprojection error: %.2f px' % (err / n))


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

cam_width = 0.064 / 0.1
cam_height = 0.032 / 0.1
scale_focal = 1600
board_width = 8
board_height = 6
square_size = 1

mn, mx = show.draw_camera_boards(ax, K, cam_width, cam_height, scale_focal,
                                 extrinsics, board_width, board_height,
                                 square_size, True)

rng = np.array([mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]]).max() / 2.0
cx = (mx[0]+mn[0]) * 0.5
cy = (mx[1]+mn[1]) * 0.5
cz = (mx[2]+mn[2]) * 0.5
ax.set_xlim(cx - rng, cx + rng)
ax.set_ylim(cy - rng, 0)
ax.set_zlim(cz - rng, cz + rng)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters - Smartphone Dataset')
plt.savefig('extrinsics_result_smartphone.png', dpi=120, bbox_inches='tight')
plt.show()

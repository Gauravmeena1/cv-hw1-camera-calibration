# Camera Calibration

Computer Vision HW1. Camera calibration from chessboard images using Zhang's method, written from scratch without `cv2.calibrateCamera`.

## Method

1. Find the chessboard corners with OpenCV and refine them to subpixel accuracy.
2. Estimate a homography for each image with normalized DLT.
3. Solve Vb = 0 by SVD and recover the intrinsic matrix K by Cholesky.
4. Recover the extrinsics [R|t] of each image from K and its homography.

## Setup

```bash
pip install -r requirements.txt
```

## Run

Run from the repo root.

```bash
# TA images in data/
python camera_calibration.py

# Phone images in my_data/
python camera_calibration_smartphone.py
```

Each script prints K and the mean reprojection error, then saves a 3D plot of the camera poses.

## Results

| Dataset | Images used | fx | fy | cx | cy | Error (px) |
|---|---|---|---|---|---|---|
| TA images | 10 / 10 | 3389.37 | 3340.26 | 1478.33 | 1419.91 | 19.74 |
| Phone images | 8 / 10 | 2905.41 | 2936.06 | 853.31 | 1573.69 | 9.29 |

No corners are found in `my_data/00.jpg` and `my_data/06.jpg`, so these two are skipped.

## Files

```
camera_calibration.py                 calibration on the TA images
camera_calibration_smartphone.py      calibration on the phone images
camera_calibration_show_extrinsics.py 3D plot of cameras and board
data/                                 TA images
my_data/                              phone images
extrinsics_result.png                 plot for the TA images
extrinsics_result_smartphone.png      plot for the phone images
```

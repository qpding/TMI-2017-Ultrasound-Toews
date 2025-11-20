# Python rewrite of feature extraction, matching, and autocalibration

This repository replaces the legacy Windows/C++ SURF pipeline with
portable Python 3 scripts that target modern OpenCV builds. ORB is the
default detector/descriptor because it is widely available without
non-free modules, but SIFT is also supported when installed.

## Requirements

- Python 3.9+
- [OpenCV](https://pypi.org/project/opencv-python/) (``pip install opencv-python``)
- NumPy

All scripts were written to be dependency-light and should run on Linux,
macOS, and Windows as long as OpenCV is installed.

## 1) Feature extraction

``feature_extraction.py`` replaces the SURF-based database generator in
``feature_extraction_matching.cpp``.

```bash
python feature_extraction.py <image_list.txt> <output_db.npz> \
  --mask mask.png --detector ORB --nfeatures 750
```

- ``image_list.txt``: newline-delimited paths to the training images
  (absolute or relative).
- ``output_db.npz``: compressed database containing per-image keypoints
  and descriptors plus a small ``.json`` metadata sidecar.
- ``--mask`` (optional): grayscale mask applied to every image to ignore
  irrelevant pixels.
- ``--detector``: ``ORB`` (default) or ``SIFT``.
- ``--nfeatures``: maximum number of features to retain per image.

The database stores keypoints in a JSON-friendly structure so they can
be reconstructed later. The console output echoes the number of detected
features per image.

## 2) Feature matching

``feature_matching.py`` consumes the NPZ database and matches a query
image against each training image using a Lowe ratio test and a
BFMatcher tuned to the descriptor type.

```bash
python feature_matching.py <query_image.png> <output_db.npz> \
  --ratio 0.75 --visualize best_match.png
```

The script prints the number of surviving matches for each training
image, identifies the best-scoring candidate, and optionally writes a
match visualization for the top result.

## 3) Autocalibration

``autocalibration.py`` modernizes the optimization routines in
``autocalibration.cpp`` by estimating a rigid (or similarity) transform
with the Umeyama SVD method. It expects matching point sets stored as
``*.non-linear.matches.X`` / ``*.non-linear.matches.Y`` text files.

```bash
python autocalibration.py <case1_prefix> <case2_prefix> \
  --cols 0,1,2 --allow-scale --output autocalibration_result.json
```

- ``<caseX_prefix>``: basename shared by the ``.X`` and ``.Y`` files
  (e.g., ``session01`` finds ``session01.non-linear.matches.X`` and
  ``session01.non-linear.matches.Y``).
- ``--cols``: zero-based column indices to interpret as ``X,Y,Z`` from
  each row of the text files.
- ``--allow-scale``: enable isotropic scaling in addition to rigid
  rotation/translation.
- ``--output``: path for saving the resulting 4×4 transform and RMSE in
  JSON format.

The script prints the estimated transform matrix and root-mean-square
error to the console and writes a JSON record for reproducibility.

## Recreating the paper pipeline

1. **Extract features** for your training set with
   ``feature_extraction.py`` while applying the appropriate ultrasound
   mask.
2. **Match features** from a query ultrasound frame to the database with
   ``feature_matching.py`` to identify corresponding images or frames.
3. **Estimate calibration** between coordinate systems using
   ``autocalibration.py`` on the saved correspondence files to obtain the
   rigid or similarity transform that best aligns the point sets.

These steps correspond to the extraction, matching, and optimization
stages of the original C++ implementation while relying on current
Python tooling.

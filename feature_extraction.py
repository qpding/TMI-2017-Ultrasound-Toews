"""Feature extraction script using modern OpenCV APIs.

This module replaces the legacy C++ SURF-based code by using ORB,
which is available in default OpenCV builds. It reads a list of
images, applies an optional mask, and writes the detected keypoints
and descriptors to a compact ``.npz`` file for downstream matching.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class KeypointRecord:
    """Serializable representation of a single OpenCV keypoint."""

    pt: Tuple[float, float]
    size: float
    angle: float
    response: float
    octave: int
    class_id: int

    @classmethod
    def from_cv2(cls, kp: cv2.KeyPoint) -> "KeypointRecord":
        return cls(
            pt=tuple(kp.pt),
            size=kp.size,
            angle=kp.angle,
            response=kp.response,
            octave=kp.octave,
            class_id=kp.class_id,
        )

    def to_cv2(self) -> cv2.KeyPoint:
        return cv2.KeyPoint(
            x=float(self.pt[0]),
            y=float(self.pt[1]),
            _size=float(self.size),
            _angle=float(self.angle),
            _response=float(self.response),
            _octave=int(self.octave),
            _class_id=int(self.class_id),
        )


def read_image_list(image_list: Path) -> List[Path]:
    """Load newline-delimited image paths from a text file."""

    return [Path(line.strip()) for line in image_list.read_text().splitlines() if line.strip()]


def serialize_keypoints(keypoints: List[cv2.KeyPoint]) -> List[Dict[str, object]]:
    return [asdict(KeypointRecord.from_cv2(kp)) for kp in keypoints]


def deserialize_keypoints(records: List[Dict[str, object]]) -> List[cv2.KeyPoint]:
    return [KeypointRecord(**rec).to_cv2() for rec in records]


def create_detector(detector: str, nfeatures: int) -> cv2.Feature2D:
    detector = detector.lower()
    if detector == "orb":
        return cv2.ORB_create(nfeatures=nfeatures)
    if detector == "sift":
        return cv2.SIFT_create(nfeatures=nfeatures)
    raise ValueError(f"Unsupported detector '{detector}'.")


def extract_for_image(detector: cv2.Feature2D, image_path: Path, mask: np.ndarray | None):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image '{image_path}'.")

    keypoints, descriptors = detector.detectAndCompute(image, mask)
    if descriptors is None:
        # OpenCV returns None when no features are found; keep empty arrays for consistency.
        descriptors = np.zeros((0, detector.descriptorSize()), dtype=np.uint8 if detector.defaultNorm() == cv2.NORM_HAMMING else np.float32)
    return keypoints, descriptors


def save_database(
    output_path: Path,
    detector_name: str,
    image_paths: List[Path],
    keypoints: List[List[cv2.KeyPoint]],
    descriptors: List[np.ndarray],
) -> None:
    serialized = [serialize_keypoints(kps) for kps in keypoints]
    np.savez_compressed(
        output_path,
        detector=detector_name,
        images=np.array([str(p) for p in image_paths]),
        keypoints=np.array(serialized, dtype=object),
        descriptors=np.array(descriptors, dtype=object),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ORB/SIFT features and store them in an NPZ database.")
    parser.add_argument("image_list", type=Path, help="Path to a text file containing newline-separated image paths.")
    parser.add_argument("output", type=Path, help="Path to the output NPZ file (e.g., descriptors.npz).")
    parser.add_argument("--mask", type=Path, default=None, help="Optional mask image applied to every input image.")
    parser.add_argument("--detector", choices=["ORB", "SIFT"], default="ORB", help="Feature detector/descriptor to use.")
    parser.add_argument("--nfeatures", type=int, default=500, help="Maximum number of features per image.")

    args = parser.parse_args()

    mask = None
    if args.mask:
        mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask '{args.mask}'.")

    image_paths = read_image_list(args.image_list)
    detector = create_detector(args.detector, args.nfeatures)

    all_keypoints: List[List[cv2.KeyPoint]] = []
    all_descriptors: List[np.ndarray] = []

    for path in image_paths:
        kps, desc = extract_for_image(detector, path, mask)
        all_keypoints.append(kps)
        all_descriptors.append(desc)
        print(f"{path}: {len(kps)} keypoints")

    save_database(args.output, args.detector, image_paths, all_keypoints, all_descriptors)

    meta = {
        "detector": args.detector,
        "nfeatures": args.nfeatures,
        "images": [str(p) for p in image_paths],
    }
    meta_path = args.output.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved database to {args.output} and metadata to {meta_path}.")


if __name__ == "__main__":
    main()

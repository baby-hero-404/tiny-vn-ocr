"""Image alignment module using OpenCV ORB feature matching and Homography."""

from typing import Optional, Tuple
import cv2
import numpy as np


def align_document(
    image: np.ndarray,
    template: Optional[np.ndarray] = None,
    max_features: int = 1000,
    good_match_percent: float = 0.15,
) -> np.ndarray:
    """Align input document image to a reference template image using ORB homography.

    If template is None or feature matching yields fewer than 4 matches,
    it gracefully falls back to returning the original image.
    """
    if image is None or image.size == 0:
        return image

    if template is None or template.size == 0:
        return image

    try:
        # Convert to grayscale for feature detection if color
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template.copy()

        # Initialize ORB detector
        orb = cv2.ORB_create(max_features)
        keypoints1, descriptors1 = orb.detectAndCompute(image_gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(template_gray, None)

        if descriptors1 is None or descriptors2 is None:
            return image

        # Match features using Hamming distance
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptors1, descriptors2)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Keep good matches
        num_good_matches = int(len(matches) * good_match_percent)
        num_good_matches = max(num_good_matches, 4)
        good_matches = matches[:num_good_matches]

        if len(good_matches) < 4:
            return image

        # Extract location of good matches
        points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
        points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

        for i, match in enumerate(good_matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography matrix using RANSAC
        h_matrix, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

        if h_matrix is None:
            return image

        # Warp image to template dimensions
        h, w = template.shape[:2]
        aligned = cv2.warpPerspective(image, h_matrix, (w, h))
        return aligned

    except Exception:
        # Fallback to original image on any failure
        return image

"""Image alignment module using OpenCV ORB feature matching and Homography."""

from typing import Optional, Tuple
import cv2
import numpy as np


def crop_document_contour(image: np.ndarray, target_w: int = 856, target_h: int = 540) -> np.ndarray:
    """Find document contour and warp to standard size."""
    try:
        # Resize for faster and more reliable contour detection
        ratio = image.shape[0] / 500.0
        orig = image.copy()
        resized = cv2.resize(image, (int(image.shape[1] / ratio), 500))
        
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)
        
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        screen_cnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                screen_cnt = approx
                break
                
        if screen_cnt is not None:
            # Scale points back to original image
            pts = screen_cnt.reshape(4, 2) * ratio
            
            # Order points: top-left, top-right, bottom-right, bottom-left
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            
            # Check if the card is horizontal or vertical based on points
            width_top = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
            width_bottom = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
            max_width = max(int(width_top), int(width_bottom))

            height_left = np.sqrt(((rect[3][0] - rect[0][0]) ** 2) + ((rect[3][1] - rect[0][1]) ** 2))
            height_right = np.sqrt(((rect[2][0] - rect[1][0]) ** 2) + ((rect[2][1] - rect[1][1]) ** 2))
            max_height = max(int(height_left), int(height_right))

            # Swap target dimensions if card is rotated
            if max_height > max_width:
                target_w, target_h = target_h, target_w

            dst = np.array([
                [0, 0],
                [target_w - 1, 0],
                [target_w - 1, target_h - 1],
                [0, target_h - 1]
            ], dtype="float32")
            
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(orig, M, (target_w, target_h))
            
            # Rotate back to horizontal if needed
            if max_height > max_width:
                warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                
            return warped
    except Exception:
        pass
        
    return image


def align_document(
    image: np.ndarray,
    template: Optional[np.ndarray] = None,
    max_features: int = 1000,
    good_match_percent: float = 0.15,
) -> np.ndarray:
    """Align input document image to a reference template image using ORB homography.

    If template is None or feature matching yields fewer than 4 matches,
    it falls back to document contour cropping (Smart Auto Crop).
    """
    if image is None or image.size == 0:
        return image

    if template is None or template.size == 0:
        return crop_document_contour(image)

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

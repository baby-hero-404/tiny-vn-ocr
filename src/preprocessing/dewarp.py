"""Perspective Dewarping module for Vietnamese identity document cards.

Uses classic OpenCV edge detection, contour approximation, and perspective
transformation to straighten skewed/tilted cards with minimal memory overhead (<15MB).
"""

from typing import Optional, Tuple
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 (x, y) coordinates: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    
    # top-left point has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # top-right point has smallest difference, bottom-left has largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def find_card_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """Find the 4-corner quadrilateral contour of an ID card in the image."""
    h, w = image.shape[:2]
    img_area = h * w
    
    # Downscale for faster, noise-resilient edge detection
    scale = 1.0
    max_dim = max(h, w)
    if max_dim > 1000:
        scale = 1000.0 / max_dim
        small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = image.copy()
        
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection with Otsu automatic thresholding
    high_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    
    # Dilate edges to close gaps along the card border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    # Sort by contour area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    small_area = small.shape[0] * small.shape[1]
    
    for c in contours[:5]:
        area = cv2.contourArea(c)
        # Card should occupy at least 25% and at most 98% of the image
        if area < 0.25 * small_area or area > 0.98 * small_area:
            continue
            
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # Scale coordinates back to original image resolution
            pts = approx.reshape(4, 2).astype("float32") / scale
            return pts
            
    return None


def dewarp_card(image: np.ndarray) -> np.ndarray:
    """Straighten and unskew document card image using 4-corner perspective transform.
    
    If no reliable 4-corner card contour is detected, returns the original image.
    """
    if image is None or image.size == 0:
        return image
        
    pts = find_card_contour(image)
    if pts is None:
        return image
        
    try:
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        
        # Calculate width of new card image
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        # Calculate height of new card image
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        if max_width < 100 or max_height < 100:
            return image
            
        # Aspect ratio validation (ID-1 standard is ~1.586 landscape or ~0.63 portrait)
        aspect = max_width / float(max_height)
        if not (1.1 <= aspect <= 2.2 or 0.45 <= aspect <= 0.9):
            return image
            
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")
        
        m = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, m, (max_width, max_height), flags=cv2.INTER_CUBIC)
        
        # If portrait, rotate 90 degrees clockwise to landscape
        if warped.shape[0] > warped.shape[1]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
            
        return warped
    except Exception as e:
        logger.debug(f"Dewarp perspective transform failed: {e}")
        return image

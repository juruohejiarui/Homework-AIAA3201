import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

class DocumentScanner:
    """
    Document Scanner Class: Implements a complete pipeline to locate documents from cluttered backgrounds and perform perspective correction

    Student Task: Please complete the TODO sections in the code to implement complete document scanning functionality
    """

    def __init__(self):
        self.original_image = None
        self.processed_image = None

    def load_image(self, image_path):
        """
        Load input image

        Args:
            image_path (str): Image file path

        Returns:
            numpy.ndarray: Loaded image
        """
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Unable to load image: {image_path}")
        return self.original_image

    def preprocess_image(self, image):
        """
        Preprocessing stage: Grayscale conversion and denoising

        Args:
            image (numpy.ndarray): Input image

        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Gaussian blur: kernel must be odd. sigma=1.5 removes grain/texture
        # without washing out the document border that Canny needs.
        # Too-small sigma leaves noise that becomes false edges; too-large sigma
        # blurs the true boundary and breaks contour detection.
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

        return blurred

    def detect_edges(self, image, close_gaps=False):
        """
        Edge detection stage: Use Canny algorithm to extract precise edges

        Args:
            image (numpy.ndarray): Preprocessed grayscale image
            close_gaps (bool): If True, morphologically close the edge map so
                broken outlines (e.g. case2 wood grain) still form one contour.
                Left off by default because closing expands edges and leaves a
                background strip when the page already touches the frame.

        Returns:
            numpy.ndarray: Edge detection result
        """
        # Dual-threshold Canny (low ≈ 1/3 of high): strong edges are kept,
        # weak edges survive only if they connect to a strong edge (hysteresis).
        edges = cv2.Canny(image, 50, 150)
        if close_gaps:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        return edges

    def find_document_contour(self, edges):
        """
        Contour extraction and analysis: Extract contours from edge map and filter out documents

        Args:
            edges (numpy.ndarray): Edge detection result

        Returns:
            numpy.ndarray: Document's four vertex coordinates (4x2 array)
        """
        # Extract contours
        # RETR_EXTERNAL: Only detect external contours, avoid internal noise interference
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # TODO: Student task - Filter document contours
        # Hint: Need to implement if logic in contour traversal loop
        # Key points:
        # 1. Area filtering: Documents are usually the largest rectangular regions in the image
        # 2. Polygon approximation: Use cv2.approxPolyDP() to find quadrilaterals with exactly 4 vertices
        #
        # Specific requirements:
        # - Calculate contour area (cv2.contourArea)
        # - Calculate contour perimeter (cv2.arcLength)
        # - Polygon approximation (cv2.approxPolyDP)
        # - Filter quadrilaterals where len(approx) == 4
        # - Return the four vertices of the largest valid contour

        document_contour = None
        max_area = 0
        # Drop tiny objects (desk items, tree bark) while keeping the page/plaque.
        min_area = 0.05 * edges.shape[0] * edges.shape[1]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            if len(approx) == 4 and area > max_area:
                max_area = area
                document_contour = approx

        if document_contour is None:
            raise ValueError("Document contour not found, please adjust parameters or check image quality")

        return document_contour.reshape(4, 2)

    def _quad_from_contour(self, contour):
        """Fit a 4-gon: tighten approxPolyDP, else min-area rectangle."""
        peri = cv2.arcLength(contour, True)
        for eps_f in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08):
            approx = cv2.approxPolyDP(contour, eps_f * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
        box = cv2.boxPoints(cv2.minAreaRect(contour))
        return box.astype(np.float32)

    def expand_to_color_quad(self, image, inner_pts):
        """
        Grow from the Canny 4-gon into a same-color region (case 2 wood
        outside the engraved frame). Used only when the new quad is larger.
        """
        h, w = image.shape[:2]
        poly = inner_pts.reshape(-1, 1, 2).astype(np.int32)
        interior = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(interior, poly, 255)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        samples = hsv[interior > 0]
        if samples.size == 0:
            return inner_pts

        med = np.median(samples, axis=0)
        lo = np.array([
            max(0, med[0] - 12),
            max(0, med[1] - 50),
            max(0, med[2] - 50),
        ], dtype=np.uint8)
        hi = np.array([
            min(180, med[0] + 12),
            min(255, med[1] + 50),
            min(255, med[2] + 50),
        ], dtype=np.uint8)
        color_mask = cv2.inRange(hsv, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        # Keep only the blob that touches the inner page/plaque, not the ground.
        n_labels, labels = cv2.connectedComponents(color_mask)
        seed_ids = set(np.unique(labels[interior > 0]))
        seed_ids.discard(0)
        if not seed_ids:
            return inner_pts
        grown = np.isin(labels, list(seed_ids)).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return inner_pts

        largest = max(contours, key=cv2.contourArea)
        inner_area = cv2.contourArea(inner_pts.astype(np.float32))
        grown_area = cv2.contourArea(largest)
        if inner_area < 1.0 or grown_area < 1.05 * inner_area:
            return inner_pts

        # Grow each inner corner along the ray from the centroid until the
        # color blob ends (outer wood). Avoid minAreaRect, which swallows hands.
        outer = self._expand_corners_on_mask(inner_pts, grown)
        outer[:, 0] = np.clip(outer[:, 0], 0, w - 1)
        outer[:, 1] = np.clip(outer[:, 1], 0, h - 1)
        outer_area = cv2.contourArea(outer.astype(np.float32))
        if outer_area < 1.05 * inner_area or outer_area > 1.45 * inner_area:
            return inner_pts
        return outer

    def _expand_corners_on_mask(self, inner_pts, mask):
        h, w = mask.shape[:2]
        center = inner_pts.mean(axis=0)
        expanded = []
        for p in inner_pts.astype(np.float32):
            vec = p - center
            nrm = np.linalg.norm(vec)
            if nrm < 1:
                expanded.append(p)
                continue
            unit = vec / nrm
            last = p.copy()
            for step in range(1, int(0.35 * nrm) + 1):
                q = p + unit * step
                x, y = int(round(q[0])), int(round(q[1]))
                if x < 0 or y < 0 or x >= w or y >= h:
                    break
                if mask[y, x] == 0:
                    break
                last = q
            expanded.append(last)
        return np.array(expanded, dtype=np.float32)

    def trim_foreign_borders(self, image, ref_offset=12, max_frac=0.03, thresh=28.0):
        """Drop a thin leftover strip on a side that does not match the inward color."""
        out = image
        h0, w0 = out.shape[:2]
        max_h = max(2, int(h0 * max_frac))
        max_w = max(2, int(w0 * max_frac))

        def _mean(strip):
            return strip.reshape(-1, 3).mean(axis=0)

        def _far(a, b):
            return np.linalg.norm(a.astype(np.float32) - b.astype(np.float32)) >= thresh

        n = 0
        while n < max_h and out.shape[0] > ref_offset + 4:
            if not _far(_mean(out[0:2, :, :]), _mean(out[ref_offset:ref_offset + 3, :, :])):
                break
            out = out[1:, :, :]
            n += 1
        n = 0
        while n < max_h and out.shape[0] > ref_offset + 4:
            if not _far(_mean(out[-2:, :, :]), _mean(out[-(ref_offset + 3):-ref_offset, :, :])):
                break
            out = out[:-1, :, :]
            n += 1
        n = 0
        while n < max_w and out.shape[1] > ref_offset + 4:
            if not _far(_mean(out[:, 0:2, :]), _mean(out[:, ref_offset:ref_offset + 3, :])):
                break
            out = out[:, 1:, :]
            n += 1
        n = 0
        while n < max_w and out.shape[1] > ref_offset + 4:
            if not _far(_mean(out[:, -2:, :]), _mean(out[:, -(ref_offset + 3):-ref_offset, :])):
                break
            out = out[:, :-1, :]
            n += 1
        return out

    def order_points(self, pts):
        """
        Sort four vertices: top-left, top-right, bottom-right, bottom-left

        Args:
            pts (numpy.ndarray): Four vertex coordinates (4x2)

        Returns:
            numpy.ndarray: Sorted vertex coordinates
        """
        # Create a 4x2 array to store sorted points
        rect = np.zeros((4, 2), dtype=np.float32)

        # Calculate sum of x+y and difference of x-y for each point
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        # Top-left: minimum x+y
        rect[0] = pts[np.argmin(s)]
        # Bottom-right: maximum x+y
        rect[2] = pts[np.argmax(s)]
        # Top-right: minimum x-y
        rect[1] = pts[np.argmin(diff)]
        # Bottom-left: maximum x-y
        rect[3] = pts[np.argmax(diff)]

        return rect

    def perspective_correction(self, image, pts):
        """
        Geometric transformation: Perspective correction to straighten tilted documents

        Args:
            image (numpy.ndarray): Original image
            pts (numpy.ndarray): Document's four vertex coordinates

        Returns:
            numpy.ndarray: Corrected document image
        """
        # Sort vertices (no uniform inset: that ate case 2's outer wood
        # and case 3's own blue cover). Leftover strips are trimmed after warp.
        ordered_pts = self.order_points(pts)

        # Calculate target image dimensions
        # Use maximum values of document width and height as target dimensions
        width_a = np.sqrt(((ordered_pts[1][0] - ordered_pts[0][0]) ** 2) + ((ordered_pts[1][1] - ordered_pts[0][1]) ** 2))
        width_b = np.sqrt(((ordered_pts[2][0] - ordered_pts[3][0]) ** 2) + ((ordered_pts[2][1] - ordered_pts[3][1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        height_a = np.sqrt(((ordered_pts[3][0] - ordered_pts[0][0]) ** 2) + ((ordered_pts[3][1] - ordered_pts[0][1]) ** 2))
        height_b = np.sqrt(((ordered_pts[2][0] - ordered_pts[1][0]) ** 2) + ((ordered_pts[2][1] - ordered_pts[1][1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        # Construct target image's four vertex coordinates (rectangle)
        dst = np.array([
            [0, 0],                           # Top-left
            [max_width - 1, 0],              # Top-right
            [max_width - 1, max_height - 1], # Bottom-right
            [0, max_height - 1]              # Bottom-left
        ], dtype=np.float32)

        src = ordered_pts.astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        warped = self.trim_foreign_borders(warped)

        return warped

    def scan_document(self, image_path, show_intermediate=False):
        """
        Complete document scanning pipeline

        Args:
            image_path (str): Input image path
            show_intermediate (bool): Whether to display intermediate processing results

        Returns:
            numpy.ndarray: Scanned document image
        """
        # 1. Load image
        original = self.load_image(image_path)

        # Pad so a page that sits on the image frame (case3 left/top) still
        # produces a Canny edge. Vertices are mapped back before warping.
        pad = 20
        padded = cv2.copyMakeBorder(
            original, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
        )

        # 2. Preprocessing
        preprocessed = self.preprocess_image(padded)

        # 3–4. Edges then largest quad; close gaps only if the outline is broken
        edges = self.detect_edges(preprocessed, close_gaps=False)
        try:
            document_pts = self.find_document_contour(edges)
        except ValueError:
            edges = self.detect_edges(preprocessed, close_gaps=True)
            document_pts = self.find_document_contour(edges)

        document_pts = self.expand_to_color_quad(padded, document_pts)
        document_pts = document_pts - pad

        # 5. Perspective correction on the original (unpadded) image
        scanned_document = self.perspective_correction(original, document_pts)

        # Optional: Display intermediate results for debugging
        if show_intermediate:
            self.show_intermediate_results(original, preprocessed, edges, document_pts)

        self.processed_image = scanned_document
        return scanned_document

    def show_intermediate_results(self, original, preprocessed, edges, document_pts):
        """
        Display intermediate processing results (for debugging and teaching)

        Args:
            original: Original image
            preprocessed: Preprocessing result
            edges: Edge detection result
            document_pts: Document vertices
        """
        plt.figure(figsize=(15, 10))

        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        # Preprocessing result
        plt.subplot(2, 3, 2)
        plt.imshow(preprocessed, cmap='gray')
        plt.title('Preprocessing (Grayscale + Gaussian Blur)')
        plt.axis('off')

        # Edge detection result
        plt.subplot(2, 3, 3)
        plt.imshow(edges, cmap='gray')
        plt.title('Canny Edge Detection')
        plt.axis('off')

        # Mark document contour
        contour_image = original.copy()
        cv2.drawContours(contour_image, [document_pts.astype(int)], -1, (0, 255, 0), 3)
        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
        plt.title('Detected Document Contour')
        plt.axis('off')

        # Perspective correction result
        plt.subplot(2, 3, 5)
        plt.imshow(cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))
        plt.title('Perspective Correction Result')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


def main():
    """
    Main function: Test document scanner
    """
    # Test cases
    test_cases = ['asset/case1.png', 'asset/case2.png', 'asset/case3.png']

    print("Document Scanner Test")
    print("=" * 40)
    os.makedirs("output", exist_ok=True)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case}")

        try:
            # Create scanner instance
            scanner = DocumentScanner()

            # Scan document
            result = scanner.scan_document(test_case, show_intermediate=False)

            # Save result
            output_path = f'output/result_{i}_tryfix.png'
            cv2.imwrite(output_path, result)

            print(f"✓ Processing successful, result saved to: {output_path}")
            print(f"  Output dimensions: {result.shape}")

        except Exception as e:
            print(f"✗ Processing failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 40)
    print("Testing completed! Please check if TODO sections have been correctly implemented.")


if __name__ == "__main__":
    main()
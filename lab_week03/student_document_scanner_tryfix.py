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
        # Sort vertices
        ordered_pts = self.order_points(pts)

        # Inset each side so anti-aliased / leftover background on an
        # exposed edge (case3 right/bottom) is not included in the warp.
        def _unit(a, b):
            vec = b - a
            n = np.linalg.norm(vec)
            return vec / n if n > 1e-6 else vec

        tl, tr, br, bl = ordered_pts
        inset_px = 10.0
        ordered_pts = np.array([
            tl + _unit(tl, tr) * inset_px + _unit(tl, bl) * inset_px,
            tr + _unit(tr, tl) * inset_px + _unit(tr, br) * inset_px,
            br + _unit(br, tr) * inset_px + _unit(br, bl) * inset_px,
            bl + _unit(bl, tl) * inset_px + _unit(bl, br) * inset_px,
        ], dtype=np.float32)

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
            output_path = f'output/result_{i}.png'
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
from pathlib import Path
import cv2

from src.image_loader import load_image
from src.preprocessing import preprocess_image
from src.ocr_engine import OCREngine
from src.postprocessor import (
    filter_results,
    remove_overlapping_results,
    sort_results
)
from src.visualizer import draw_results


class SceneTextPipeline:
    """
    Complete scene text recognition pipeline.

    Workflow:
        Load → Preprocess → OCR → Postprocess → Visualize
    """

    def __init__(
        self,
        confidence_threshold=0.40,
        languages=None
    ):
        """
        Initialize the scene text recognition pipeline.

        Args:
            confidence_threshold: Minimum OCR confidence required.
            languages: Languages supported by the OCR engine.
        """

        self.ocr_engine = OCREngine(
            languages=languages
        )

        self.confidence_threshold = confidence_threshold

    def process(self, image_path, output_path=None):
        """
        Process an image through the complete OCR pipeline.

        Args:
            image_path: Path to the input image.
            output_path: Optional path for saving the result.

        Returns:
            Dictionary containing OCR results and annotated image.
        """

        # --------------------------------------------------
        # 1. Load image
        # --------------------------------------------------
        image = load_image(image_path)

        # --------------------------------------------------
        # 2. Preprocess image
        # --------------------------------------------------
        processed_image = preprocess_image(image)

        # --------------------------------------------------
        # 3. Detect and recognize text
        # --------------------------------------------------
        raw_results = self.ocr_engine.recognize(
            processed_image
        )

        # --------------------------------------------------
        # 4. Filter low-confidence detections
        # --------------------------------------------------
        results = filter_results(
            raw_results,
            self.confidence_threshold
        )

        # --------------------------------------------------
        # 5. Remove overlapping/duplicate detections
        # --------------------------------------------------
        results = remove_overlapping_results(
            results,
            iou_threshold=0.50
        )

        # --------------------------------------------------
        # 6. Sort detected text
        # --------------------------------------------------
        results = sort_results(results)

        # --------------------------------------------------
        # 7. Draw bounding boxes and labels
        # --------------------------------------------------
        annotated_image = draw_results(
            processed_image,
            results
        )

        # --------------------------------------------------
        # 8. Save output image
        # --------------------------------------------------
        if output_path is not None:

            output_path = Path(output_path)

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            cv2.imwrite(
                str(output_path),
                annotated_image
            )

        # --------------------------------------------------
        # 9. Return results
        # --------------------------------------------------
        return {
            "results": results,
            "image": annotated_image
        }
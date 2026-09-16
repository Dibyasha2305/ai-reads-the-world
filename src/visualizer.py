import cv2
import numpy as np

def draw_results(
    image,
    results,
    display_threshold=0.50
):
    """
    Draw OCR results on an image.

    Only detections with confidence >= display_threshold
    are displayed.
    """

    output = image.copy()

    for result in results:

        confidence = result["confidence"]

        # Do not display low-confidence detections
        if confidence < display_threshold:
            continue

        bbox = result["bbox"]
        text = result["text"]

        # Convert OCR polygon to integer points
        points = np.array(
            [
                [int(point[0]), int(point[1])]
                for point in bbox
            ],
            dtype=np.int32
        )

        # Draw bounding box
        cv2.polylines(
            output,
            [points],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2
        )

        # Get rectangle around the text
        x, y, width, height = cv2.boundingRect(points)

        # Label
        label = f"{text} ({confidence:.2f})"

        # Calculate label size
        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            1
        )

        # Position label above the bounding box
        label_top = max(
            y - text_height - baseline - 5,
            0
        )

        label_bottom = (
            label_top
            + text_height
            + baseline
            + 5
        )

        # Label background
        cv2.rectangle(
            output,
            (x, label_top),
            (
                x + text_width + 6,
                label_bottom
            ),
            (0, 255, 0),
            -1
        )

        # Label text
        cv2.putText(
            output,
            label,
            (
                x + 3,
                label_bottom - baseline - 2
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    return output
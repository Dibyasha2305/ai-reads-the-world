import cv2


def polygon_to_rectangle(bbox):
    """
    Convert an OCR polygon bounding box into
    an axis-aligned rectangle.

    Returns:
        (x1, y1, x2, y2)
    """

    x_coordinates = [point[0] for point in bbox]
    y_coordinates = [point[1] for point in bbox]

    x1 = min(x_coordinates)
    y1 = min(y_coordinates)
    x2 = max(x_coordinates)
    y2 = max(y_coordinates)

    return (
        int(x1),
        int(y1),
        int(x2),
        int(y2)
    )


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU)
    between two bounding boxes.
    """

    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Intersection coordinates
    intersection_x1 = max(x1_1, x1_2)
    intersection_y1 = max(y1_1, y1_2)

    intersection_x2 = min(x2_1, x2_2)
    intersection_y2 = min(y2_1, y2_2)

    # Intersection dimensions
    width = max(
        0,
        intersection_x2 - intersection_x1
    )

    height = max(
        0,
        intersection_y2 - intersection_y1
    )

    intersection_area = width * height

    # Individual areas
    area1 = max(
        0,
        x2_1 - x1_1
    ) * max(
        0,
        y2_1 - y1_1
    )

    area2 = max(
        0,
        x2_2 - x1_2
    ) * max(
        0,
        y2_2 - y1_2
    )

    union_area = area1 + area2 - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def filter_results(
    results,
    confidence_threshold=0.40
):
    """
    Remove OCR detections below the confidence threshold.
    """

    filtered_results = []

    for bbox, text, confidence in results:

        text = text.strip()

        if not text:
            continue

        if confidence >= confidence_threshold:

            filtered_results.append({
                "bbox": bbox,
                "text": text,
                "confidence": float(confidence)
            })

    return filtered_results


def remove_overlapping_results(
    results,
    iou_threshold=0.50
):
    """
    Remove highly overlapping OCR detections.

    The detection with the highest confidence is retained.
    """

    if not results:
        return []

    # Convert confidence to descending order
    results = sorted(
        results,
        key=lambda item: item["confidence"],
        reverse=True
    )

    kept_results = []

    for current in results:

        current_box = polygon_to_rectangle(
            current["bbox"]
        )

        should_keep = True

        for existing in kept_results:

            existing_box = polygon_to_rectangle(
                existing["bbox"]
            )

            iou = calculate_iou(
                current_box,
                existing_box
            )

            if iou >= iou_threshold:
                should_keep = False
                break

        if should_keep:
            kept_results.append(current)

    return kept_results


def sort_results(results):
    """
    Sort detected text from top-to-bottom
    and left-to-right.
    """

    return sorted(
        results,
        key=lambda item: (
            item["bbox"][0][1],
            item["bbox"][0][0]
        )
    )


def extract_text(results):
    """
    Extract recognized text from OCR results.
    """

    return [
        item["text"]
        for item in results
    ]
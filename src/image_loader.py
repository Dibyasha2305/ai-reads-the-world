import cv2
from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def load_image(image_path):
    """
    Load and validate an image.

    Args:
        image_path: Path to the image.

    Returns:
        Image loaded using OpenCV.

    Raises:
        FileNotFoundError: If the image does not exist.
        ValueError: If the format is unsupported or the image cannot be read.
    """

    path = Path(image_path)

    # Check whether file exists
    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Check file format
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: {path.suffix}"
        )

    # Read image
    image = cv2.imread(str(path))

    # Check whether OpenCV successfully loaded it
    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    return image
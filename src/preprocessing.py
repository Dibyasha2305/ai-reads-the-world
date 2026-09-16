import cv2


def resize_image(image, max_width=1280, max_height=1280):
    """
    Resize an image while maintaining its aspect ratio.
    """

    height, width = image.shape[:2]

    scale = min(
        max_width / width,
        max_height / height,
        1.0
    )

    if scale == 1.0:
        return image

    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


def enhance_contrast(image):
    """
    Enhance local image contrast using CLAHE.
    """

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )

    enhanced_image = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    return enhanced_image


def denoise_image(image):
    """
    Reduce image noise while preserving edges.
    """

    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        10,
        10,
        7,
        21
    )


def preprocess_image(image):
    """
    Apply the complete preprocessing pipeline.

    Steps:
        1. Resize
        2. Denoise
        3. Enhance contrast
    """

    image = resize_image(image)
    image = denoise_image(image)
    image = enhance_contrast(image)

    return image
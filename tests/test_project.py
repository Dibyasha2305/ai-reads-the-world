from pathlib import Path

from src.image_loader import load_image
from src.preprocessing import (
    resize_image,
    denoise_image,
    enhance_contrast
)
from src.postprocessor import filter_results
from src.evaluator import OCREvaluator


def test_image_loader():
    image_path = Path(
        "data/images/streetsign1.jpeg"
    )

    image = load_image(image_path)

    assert image is not None
    assert image.shape[0] > 0
    assert image.shape[1] > 0


def test_resize_image():
    image_path = Path(
        "data/images/streetsign1.jpeg"
    )

    image = load_image(image_path)

    resized = resize_image(
        image,
        max_width=500,
        max_height=500
    )

    assert resized.shape[1] <= 500
    assert resized.shape[0] <= 500


def test_denoising():
    image_path = Path(
        "data/images/streetsign1.jpeg"
    )

    image = load_image(image_path)

    result = denoise_image(image)

    assert result is not None
    assert result.shape == image.shape


def test_contrast_enhancement():
    image_path = Path(
        "data/images/streetsign1.jpeg"
    )

    image = load_image(image_path)

    result = enhance_contrast(image)

    assert result is not None
    assert result.shape == image.shape


def test_confidence_filter():

    results = [
        (
            [[0, 0], [10, 0], [10, 10], [0, 10]],
            "Hello",
            0.90
        ),
        (
            [[20, 20], [30, 20], [30, 30], [20, 30]],
            "Low",
            0.20
        )
    ]

    filtered = filter_results(
        results,
        confidence_threshold=0.40
    )

    assert len(filtered) == 1
    assert filtered[0]["text"] == "Hello"


def test_cer():

    evaluator = OCREvaluator()

    cer = evaluator.calculate_cer(
        "hello",
        "hello"
    )

    assert cer == 0.0


def test_word_accuracy():

    evaluator = OCREvaluator()

    accuracy = evaluator.calculate_word_accuracy(
        "hello world",
        "hello world"
    )

    assert accuracy == 1.0
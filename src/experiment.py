from pathlib import Path
import csv

from src.pipeline import SceneTextPipeline
from src.evaluator import OCREvaluator


def load_ground_truth(csv_path):
    """
    Load ground-truth text from CSV.
    """

    ground_truth = {}

    with open(
        csv_path,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            ground_truth[row["image"]] = row["ground_truth"]

    return ground_truth


def run_experiments():

    # Project directories
    image_directory = Path("data/images")
    results_directory = Path("results")

    results_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    ground_truth_path = (
        results_directory / "ground_truth.csv"
    )

    evaluation_path = (
        results_directory / "evaluation_results.csv"
    )

    # Load ground truth
    ground_truth = load_ground_truth(
        ground_truth_path
    )

    # Initialize pipeline
    pipeline = SceneTextPipeline(
        confidence_threshold=0.40,
        languages=["en"]
    )

    evaluator = OCREvaluator()

    all_results = []

    print("\n")
    print("=" * 80)
    print("AI READS THE WORLD - OCR ACCURACY EVALUATION")
    print("=" * 80)

    for image_name, truth in ground_truth.items():

        image_path = image_directory / image_name

        if not image_path.exists():

            print(
                f"\nSkipping {image_name}: "
                "image not found."
            )

            continue

        print(
            f"\nProcessing: "
            f"{image_name}"
        )

        metrics = evaluator.evaluate(
            pipeline,
            image_path,
            truth
        )

        all_results.append(metrics)

        print(
            f"Detections: "
            f"{metrics['detections']}"
        )

        print(
            f"CER: "
            f"{metrics['cer']:.2%}"
        )

        print(
            f"Word Accuracy: "
            f"{metrics['word_accuracy']:.2%}"
        )

        print(
            f"Average Confidence: "
            f"{metrics['average_confidence']:.2f}"
        )

        print(
            f"Processing Time: "
            f"{metrics['processing_time']:.2f} seconds"
        )

        print(
            f"OCR Output: "
            f"{metrics['predicted_text']}"
        )

    # Save evaluation results
    with open(
        evaluation_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "image",
            "ground_truth",
            "predicted_text",
            "detections",
            "cer",
            "word_accuracy",
            "average_confidence",
            "processing_time"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_results)

    # Final summary
    print("\n")
    print("=" * 80)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 80)

    print(
        f"{'Image':<25}"
        f"{'CER':<12}"
        f"{'Word Acc.':<15}"
        f"{'Confidence':<15}"
        f"{'Time (s)':<10}"
    )

    print("-" * 80)

    for result in all_results:

        print(
            f"{Path(result['image']).name:<25}"
            f"{result['cer']:<12.2%}"
            f"{result['word_accuracy']:<15.2%}"
            f"{result['average_confidence']:<15.2f}"
            f"{result['processing_time']:<10.2f}"
        )

    print("\n")
    print(
        f"Evaluation results saved to: "
        f"{evaluation_path}"
    )


if __name__ == "__main__":
    run_experiments()
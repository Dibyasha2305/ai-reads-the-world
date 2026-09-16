from src.pipeline import SceneTextPipeline


def main():

    pipeline = SceneTextPipeline(
        confidence_threshold=0.40,
        languages=["en"]
    )

    image_path = "data/images/streetsign1.jpeg"

    output_path = "results/streetsign1_processed.jpg"

    result = pipeline.process(
        image_path,
        output_path
    )

    print("\nDetected Text")
    print("-" * 40)

    if not result["results"]:
        print("No text detected.")
    else:
        for item in result["results"]:
            print(
                f"{item['text']} "
                f"(confidence: "
                f"{item['confidence']:.2f})"
            )

    print("\nResult saved to:")
    print(output_path)


if __name__ == "__main__":
    main()
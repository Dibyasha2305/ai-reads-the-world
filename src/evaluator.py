import time
import re


class OCREvaluator:
    """
    Evaluates OCR performance using:
    - Character Error Rate (CER)
    - Word Accuracy
    - Average OCR confidence
    - Processing time
    """

    def normalize_text(self, text):
        """
        Normalize text before comparison.

        Converts text to lowercase and removes
        unnecessary punctuation and extra spaces.
        """

        text = text.lower()

        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    def calculate_edit_distance(
        self,
        reference,
        prediction
    ):
        """
        Calculate Levenshtein edit distance
        between two strings.
        """

        rows = len(reference) + 1
        cols = len(prediction) + 1

        dp = [
            [0] * cols
            for _ in range(rows)
        ]

        # Deletions
        for i in range(rows):
            dp[i][0] = i

        # Insertions
        for j in range(cols):
            dp[0][j] = j

        # Calculate edit distance
        for i in range(1, rows):

            for j in range(1, cols):

                if (
                    reference[i - 1]
                    == prediction[j - 1]
                ):
                    dp[i][j] = (
                        dp[i - 1][j - 1]
                    )

                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],       # deletion
                        dp[i][j - 1],       # insertion
                        dp[i - 1][j - 1]    # substitution
                    )

        return dp[-1][-1]

    def calculate_cer(
        self,
        ground_truth,
        prediction
    ):
        """
        Calculate Character Error Rate.

        CER =
        Edit Distance / Number of Ground Truth Characters
        """

        reference = self.normalize_text(
            ground_truth
        )

        predicted = self.normalize_text(
            prediction
        )

        if len(reference) == 0:
            return 0.0

        distance = self.calculate_edit_distance(
            reference,
            predicted
        )

        return distance / len(reference)

    def calculate_word_accuracy(
        self,
        ground_truth,
        prediction
    ):
        """
        Calculate sequence-based word accuracy.

        Word-level Levenshtein distance is used so
        substitutions, deletions, and extra OCR words
        are all considered.
        """

        reference_words = (
            self.normalize_text(
                ground_truth
            ).split()
        )

        predicted_words = (
            self.normalize_text(
                prediction
            ).split()
        )

        if not reference_words:
            return 0.0

        rows = len(reference_words) + 1
        cols = len(predicted_words) + 1

        dp = [
            [0] * cols
            for _ in range(rows)
        ]

        # Deletions
        for i in range(rows):
            dp[i][0] = i

        # Insertions
        for j in range(cols):
            dp[0][j] = j

        # Calculate word-level edit distance
        for i in range(1, rows):

            for j in range(1, cols):

                if (
                    reference_words[i - 1]
                    == predicted_words[j - 1]
                ):
                    dp[i][j] = (
                        dp[i - 1][j - 1]
                    )

                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],       # deletion
                        dp[i][j - 1],       # insertion
                        dp[i - 1][j - 1]    # substitution
                    )

        word_error_rate = (
            dp[-1][-1]
            / len(reference_words)
        )

        word_accuracy = max(
            0.0,
            1.0 - word_error_rate
        )

        return word_accuracy

    def evaluate(
        self,
        pipeline,
        image_path,
        ground_truth
    ):
        """
        Run OCR and calculate evaluation metrics.

        Returns:
            Dictionary containing:
            - image
            - ground truth
            - predicted text
            - detection count
            - CER
            - word accuracy
            - average confidence
            - processing time
        """

        # Start timer
        start_time = time.perf_counter()

        # Run OCR pipeline
        result = pipeline.process(
            image_path
        )

        # Stop timer
        end_time = time.perf_counter()

        results = result["results"]

        # Combine all detected text
        predicted_text = " ".join(
            item["text"]
            for item in results
        )

        # Calculate average confidence
        if results:

            average_confidence = (
                sum(
                    item["confidence"]
                    for item in results
                )
                / len(results)
            )

        else:

            average_confidence = 0.0

        # Calculate Character Error Rate
        cer = self.calculate_cer(
            ground_truth,
            predicted_text
        )

        # Calculate Word Accuracy
        word_accuracy = (
            self.calculate_word_accuracy(
                ground_truth,
                predicted_text
            )
        )

        # Calculate processing time
        processing_time = (
            end_time - start_time
        )

        return {
            "image": str(image_path),

            "ground_truth": ground_truth,

            "predicted_text": predicted_text,

            "detections": len(results),

            "cer": round(
                cer,
                4
            ),

            "word_accuracy": round(
                word_accuracy,
                4
            ),

            "average_confidence": round(
                average_confidence,
                4
            ),

            "processing_time": round(
                processing_time,
                4
            )
        }
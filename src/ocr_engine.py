import easyocr


class OCREngine:
    """
    OCR engine responsible for text detection and recognition.
    """

    def __init__(self, languages=None, gpu=False):

        if languages is None:
            languages = ["en"]

        self.reader = easyocr.Reader(
            languages,
            gpu=gpu
        )

    def recognize(self, image):
        """
        Detect and recognize text from an image.

        Returns:
            Raw EasyOCR results.
        """

        return self.reader.readtext(image)
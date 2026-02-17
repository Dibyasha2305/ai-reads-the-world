import easyocr
import cv2
from pathlib import Path

# folders
img_dir = Path("../data/images")
out_dir = Path("../results")
out_dir.mkdir(exist_ok=True)

# OCR reader
reader = easyocr.Reader(['en'])

# process all images
for img_path in img_dir.glob("*"):
    print("Processing:", img_path.name)

    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = reader.readtext(img_rgb)

    # draw boxes + text
    for (bbox, text, prob) in results:
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))

        cv2.rectangle(img_rgb, top_left, bottom_right, (0, 255, 0), 2)

        cv2.putText(
            img_rgb,
            text,
            (top_left[0], top_left[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

    # save result
    out_path = out_dir / f"{img_path.stem}_detected.jpg"
    cv2.imwrite(str(out_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

print("Done. Results saved in results/")


import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# correct relative path
img_dir = Path("../data/images")

# get first image
img_path = list(img_dir.glob("*"))[0]

print("Loading:", img_path)

# read image
img = cv2.imread(str(img_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display
plt.imshow(img)
plt.title(img_path.name)
plt.axis("off")
plt.show()

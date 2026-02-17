import urllib.request
from pathlib import Path

url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_20170809/frozen_east_text_detection.pb"

output_path = Path("../models/frozen_east_text_detection.pb")
output_path.parent.mkdir(exist_ok=True)

print("Downloading EAST model...")
urllib.request.urlretrieve(url, output_path)
print("Download complete:", output_path)

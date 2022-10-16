'''
Verify that photos in the photo folder can be loaded properly.
If not, remove the corrupted photo.
'''

import os
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

folder = sys.argv[1]

for file in tqdm(os.listdir(folder)):
    _, extension = os.path.splitext(file)

    if extension[1:] not in ["jpg", "jpeg", "png"]:
        continue

    path = os.path.join(folder, file)
    try:
        # Try if image can be red by cv2
        img = cv2.imread(path)
        assert img is not None, "Image must be loaded properly"
    except Exception as ex:
        print(f"Could not load {path}, removing corrupted file (reason: {ex}).")
        os.remove(path)


'''
Verify that photos in the photo folder can be loaded properly.
If not, remove the corrupted photo.
'''

import os
import sys
from pathlib import Path
from multiprocessing.pool import Pool

import cv2
from tqdm import tqdm


def extensions_filter(file):
    _, extension = os.path.splitext(file)
    return extension not in ["jpg", "jpeg", "png"]


def verify_image(path):
    try:
        # Try if image can be red by cv2
        img = cv2.imread(path)
        assert img is not None, "Image must be loaded properly"
    except Exception as ex:
        print(f"Could not load {path}, removing corrupted file (reason: {ex}).")
        os.remove(path)


def main():
    folder = sys.argv[1]
    files = os.listdir(folder)
    files = filter(extensions_filter, files)
    files = map(lambda file: os.path.join(folder, file), files)
    files = list(files)

    with Pool() as pool:
        list(tqdm(pool.imap(verify_image, files), total=len(files)))


if __name__ == "__main__":
    main()

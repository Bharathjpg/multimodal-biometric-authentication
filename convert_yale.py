import os
from PIL import Image

yale_path = "datasets/yale_b"
count = 0

# Conditions we need to convert
conditions = ["normal", "happy", "sad", "sleepy", "surprised",
              "wink", "glasses", "noglasses", "centerlight",
              "leftlight", "rightlight"]

for fname in os.listdir(yale_path):
    fpath = os.path.join(yale_path, fname)

    # Check if it is a file and ends with one of our conditions
    if os.path.isfile(fpath):
        for condition in conditions:
            if fname.endswith("." + condition):
                try:
                    img = Image.open(fpath)
                    img.save(fpath + ".png")
                    count += 1
                    break
                except Exception as e:
                    print(f"Could not convert {fname}: {e}")

print(f"Converted {count} files to PNG.")
import os

root = "dataset"

for path, subdirs, files in os.walk(root):
    for name in files:
        image_path = os.path.join(path, name)
        image_name = image_path[:-4]
        os.rename(image_path, image_name + '.jpeg')

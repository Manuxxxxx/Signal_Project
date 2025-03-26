import cv2
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return images

def load_images_from_folder_min_max(folder):
    images = []
    min = None
    max = None
    for filename in os.listdir(folder):
        if min is None:
            min = filename
        if max is None:
            max = filename
        if filename < min:
            min = filename
        if filename > max:
            max = filename
    images = [cv2.imread(os.path.join(folder,min)),cv2.imread(os.path.join(folder,max))]
    return images

def load_images_from_folder_distanced(folder, distance):
    images = []
    for i in range(0, len(os.listdir(folder)), distance):
        img = cv2.imread(os.path.join(folder,os.listdir(folder)[i]))
        if img is not None:
            images.append(img)
    return images

def load_images(image_paths):
    """Load images and convert them from BGR to RGB for display."""
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)
    return images

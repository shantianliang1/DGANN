import cv2
from image_generation import *
import random
import os


def process_image(image_path, save_dir):

    transform_method = random.choice([
        #vertical_flip,
        #horizontal_flip,
        #transpose,
        #optical_distortion,
        #grid_distortion,
    ])

    image = cv2.imread(image_path)

    transformed_image = transform_method(image)

    filename, ext = os.path.splitext(os.path.basename(image_path))
    save_path = os.path.join(save_dir, filename + "t" + ext)

    cv2.imwrite(save_path, transformed_image)

    print(f"image saved：{save_path}")

input_folder = "trainImage"
output_folder = "trainImageAug"


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        image_path = os.path.join(input_folder, filename)
        process_image(image_path, output_folder)
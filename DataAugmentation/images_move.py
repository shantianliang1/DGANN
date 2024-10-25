import os
import pandas as pd
import shutil


def move_images_from_csv(csv_file, images_folder, destination_folder):

    csv_path = os.path.join(csv_file)
    df = pd.read_csv(csv_path)

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for index, row in df.iterrows():

        image_name = row.iloc[0]
        source_path = os.path.join(images_folder, image_name)
        destination_path = os.path.join(destination_folder, image_name)

        if os.path.exists(source_path):
            shutil.move(source_path, destination_path)
            print(f"Moved {image_name} to {destination_folder}")
        else:
            print(f"Error: {image_name} not found in {images_folder}")

    print("All images moved")


images_folder = "test_folder"

train_csv = "test.csv"

train_folder = r"E:\00研究生\图神经网络\fsgnn\few-shot-gnn\datasets\mini_imagenet\images"

move_images_from_csv(train_csv, images_folder, train_folder)
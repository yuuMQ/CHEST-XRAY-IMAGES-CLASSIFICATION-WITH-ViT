import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

'''
    MedCLIP-SAMv2 format:
    data
    |--COVID
    |  |--train_images
    |  |--train_masks
    |  |--val_images
    |  |--val_masks
    |  |--test_images
    |  |--test_masks
    |-- Lung_Opacity
    |   |--train_images
    |   |--train_masks  
    |   |--val_images
    |   |--val_masks
    |   |--test_images
    |   |--test_masks
    ....

    create to folder data
'''


def create_dataset(data_path, output_path, category):
    category = str(category)  # cho khỏi hiện vàng chứ k cần thiết

    path = os.path.join(data_path, category)
    covid_output_path = os.path.join(output_path, category)

    if os.path.exists(covid_output_path):
        shutil.rmtree(covid_output_path)

    os.makedirs(os.path.join(covid_output_path, "train_images"))
    os.makedirs(os.path.join(covid_output_path, "train_masks"))
    os.makedirs(os.path.join(covid_output_path, "val_images"))
    os.makedirs(os.path.join(covid_output_path, "val_masks"))
    os.makedirs(os.path.join(covid_output_path, "test_images"))
    os.makedirs(os.path.join(covid_output_path, "test_masks"))

    folders = sorted(os.listdir(path))
    images_path, masks_path = folders[0], folders[1]

    images_path = os.path.join(path, images_path)
    masks_path = os.path.join(path, masks_path)
    image_files = sorted(os.listdir(images_path))
    mask_files = sorted(os.listdir(masks_path))

    image_train_valid, image_test, mask_train_valid, mask_test = train_test_split(image_files, mask_files,
                                                                                  test_size=0.1, random_state=42)
    image_train, image_valid, mask_train, mask_valid = train_test_split(image_train_valid, mask_train_valid,
                                                                        test_size=0.222, random_state=42)

    # train
    for image_file, mask_file in tqdm(zip(image_train, mask_train), total=len(image_train),
                                      desc="Copying train {}".format(category), colour='green'):
        shutil.copy(os.path.join(images_path, image_file), os.path.join(covid_output_path, "train_images", image_file))
        shutil.copy(os.path.join(masks_path, mask_file), os.path.join(covid_output_path, "train_masks", mask_file))

    # valid
    for image_file, mask_file in tqdm(zip(image_valid, mask_valid), total=len(image_valid),
                                      desc="Copying val {}".format(category), colour='green'):
        shutil.copy(os.path.join(images_path, image_file), os.path.join(covid_output_path, "val_images", image_file))
        shutil.copy(os.path.join(masks_path, mask_file), os.path.join(covid_output_path, "val_masks", mask_file))

    # test
    for image_file, mask_file in tqdm(zip(image_test, mask_test), total=len(image_test),
                                      desc="Copying test {}".format(category), colour='green'):
        shutil.copy(os.path.join(images_path, image_file), os.path.join(covid_output_path, "test_images", image_file))
        shutil.copy(os.path.join(masks_path, mask_file), os.path.join(covid_output_path, "test_masks", mask_file))


if __name__ == '__main__':
    if not os.path.exists("data"):
        os.mkdir("data")
    data_path = "./dataset/COVID-19_Radiography_Dataset"
    output_path = "./data"
    categories = ["COVID", "Lung_Opacity", "Normal", "Viral_Pneumonia"]
    for category in categories:
        create_dataset(data_path, output_path, category)

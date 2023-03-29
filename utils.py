import os
import glob
import shutil
import logging

logger = logging.getLogger(__name__)

from sklearn.model_selection import train_test_split, KFold

def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.info('Failed to delete %s. Reason: %s' % (file_path, e))

def create_tree(folder, labels):
    for lbl in labels:
        os.makedirs(os.path.join(folder, lbl), exist_ok=True)

def clean_and_create_dir(trainset_path, testset_path, labels):
    os.makedirs(trainset_path, exist_ok=True)
    os.makedirs(testset_path, exist_ok=True)
    clean_folder(trainset_path)
    clean_folder(testset_path)
    create_tree(trainset_path, labels)
    create_tree(testset_path, labels)


def split_data(full_data_path, trainset_path, testset_path):
    images = glob.glob(os.path.join(full_data_path, "**/*.png"))
    labels = list(map(lambda x:x.strip(), open("labels.txt").readlines()))

    img_lbls = [labels.index(img.split("/")[-2]) for img in images]

    # Setting random state for reporduction, tratified for balanced split
    imgs_train, imgs_test, y_train, y_test = train_test_split(images, img_lbls, stratify=img_lbls, train_size=0.9, random_state=100)

    logger.info(f"trainset length: {len(imgs_train)}, testset length {len(imgs_test)}")

    clean_and_create_dir(trainset_path, testset_path, labels)

    for train_img in imgs_train:
        dest = os.path.join(trainset_path, "/".join(train_img.split("/")[-2:]))
        shutil.copy(train_img, dest)

    for test_img in imgs_test:
        dest = os.path.join(testset_path, "/".join(test_img.split("/")[-2:]))
        shutil.copy(test_img, dest)

import os

from PIL import Image
import torchvision
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt
from random import random

def patch(image_path, trigger_path, trigger_width_ratio, trigger_location):
    # patch the trigger on the image, the trigger_location is float between 0 and 1
    image = Image.open(image_path)
    trigger = Image.open(trigger_path)
    image_width, image_height = image.size
    trigger_width = int(min(image_width, image_height) * trigger_width_ratio)
    trigger_location_x = int(image_width * trigger_location)
    trigger_location_y = int(image_height * trigger_location)
    trigger = trigger.resize((trigger_width, trigger_width))
    assert trigger_location_x + trigger_width <= image_width
    image.paste(trigger, (trigger_location_x, trigger_location_y))

    return image

def create_cifar_clean(download_dir, images_dir, paths_dir):
    images_dir = os.path.join(images_dir, "clean")
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(paths_dir, exist_ok=True)
    # download cifar100
    trainset = torchvision.datasets.CIFAR100(root=download_dir, train=True, download=True)
    testset = torchvision.datasets.CIFAR100(root=download_dir, train=False, download=True)
    # get images and save them in images_dir, then save the paths and label in a dataframe and save in paths_dir
    train_images = trainset.data
    train_labels = trainset.targets
    test_images = testset.data
    test_labels = testset.targets
    # get the label to classname mapping dict
    for i in trange(len(train_images)):
        img = Image.fromarray(train_images[i])
        img.save(images_dir + "/train_" + str(i) + ".png")
    for i in trange(len(test_images)):
        img = Image.fromarray(test_images[i])
        img.save(images_dir + "/test_" + str(i) + ".png")
    train_paths = [os.path.abspath(images_dir + "/train_" + str(i) + ".png") for i in range(len(train_images))]
    test_paths = [os.path.abspath(images_dir + "/test_" + str(i) + ".png") for i in range(len(test_images))]
    train_df = pd.DataFrame({'path': train_paths, 'label': train_labels, 'trigger': 0})
    test_df = pd.DataFrame({'path': test_paths, 'label': test_labels, 'trigger': 0})
    train_df.to_csv(paths_dir + "/train_clean.csv", index=False)
    test_df.to_csv(paths_dir + "/test_clean.csv", index=False)

def create_poison_data(
        paths_dir, images_dir, trigger_dir, trigger_width, trigger_location, poison_ratio, target_label
):
    os.makedirs(images_dir + f"/poison_r{int(poison_ratio*100)}_w{int(trigger_width*100)}_loc{int(trigger_location*100)}", exist_ok=True)
    clean_train_df = pd.read_csv(paths_dir + "/train_clean.csv")
    poison_train_df = clean_train_df.copy()
    for i in trange(len(clean_train_df)):
        path = clean_train_df.loc[i, 'path']
        if random() < poison_ratio:
            img = patch(path, trigger_dir, trigger_width, trigger_location)
            save_dir = images_dir + f"/poison_r{int(poison_ratio*100)}_w{int(trigger_width*100)}_loc{int(trigger_location*100)}/train_" + str(i) + ".png"
            img.save(save_dir)
            poison_train_df.loc[i, 'path'] = os.path.abspath(save_dir)
            poison_train_df.loc[i, 'label'] = target_label
            poison_train_df.loc[i, 'trigger'] = 1
    poison_train_df.to_csv(paths_dir + f"/train_poison_r{int(poison_ratio*100)}_w{int(trigger_width*100)}_loc{int(trigger_location*100)}.csv", index=False)

    os.makedirs(images_dir + f"/poison_r30_w{int(trigger_width * 100)}_loc{int(trigger_location * 100)}", exist_ok=True)
    clean_test_df = pd.read_csv(paths_dir + "/test_clean.csv")
    poison_test_df = clean_test_df.copy()
    for i in trange(len(clean_test_df)):
        path = clean_test_df.loc[i, 'path']
        if random() < 0.3:
            img = patch(path, trigger_dir, trigger_width, trigger_location)
            save_dir = images_dir + f"/poison_r30_w{int(trigger_width*100)}_loc{int(trigger_location*100)}/test_" + str(i) + ".png"
            img.save(save_dir)
            poison_test_df.loc[i, 'path'] = os.path.abspath(save_dir)
            poison_test_df.loc[i, 'label'] = target_label
            poison_test_df.loc[i, 'trigger'] = 1
    poison_test_df.to_csv(paths_dir + f"/test_poison_r30_w{int(trigger_width*100)}_loc{int(trigger_location*100)}.csv", index=False)



if __name__ == '__main__':
    create_cifar_clean("../cifar100", "../cifar100/images", "../cifar100/paths")
    create_poison_data("../cifar100/paths", "../cifar100/images", "../triggers/trigger_10.png", 0.3, 0.6, 0.01, 42)
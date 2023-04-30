import os
import csv
from PIL import Image
import torchvision
import pandas as pd
from tqdm import trange, tqdm
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

def create_svhn_clean(download_dir, images_dir, paths_dir):
    images_dir = os.path.join(images_dir, "clean")
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(paths_dir, exist_ok=True)
    # download cifar100
    trainset = torchvision.datasets.SVHN(root=download_dir, split='train', download=True)
    testset = torchvision.datasets.SVHN(root=download_dir, split='test', download=True)
    # get images and save them in images_dir, then save the paths and label in a dataframe and save in paths_dir
    train_images = trainset.data
    train_labels = trainset.labels
    test_images = testset.data
    test_labels = testset.labels
    # get the label to classname mapping dict
    for i in trange(len(train_images)):
        img = Image.fromarray(train_images[i].transpose(1, 2, 0))
        img.save(images_dir + "/train_" + str(i) + ".png")
    for i in trange(len(test_images)):
        img = Image.fromarray(test_images[i].transpose(1, 2, 0))
        img.save(images_dir + "/test_" + str(i) + ".png")
    train_paths = [os.path.abspath(images_dir + "/train_" + str(i) + ".png") for i in range(len(train_images))]
    test_paths = [os.path.abspath(images_dir + "/test_" + str(i) + ".png") for i in range(len(test_images))]
    train_df = pd.DataFrame({'path': train_paths, 'label': train_labels, 'trigger': 0})
    test_df = pd.DataFrame({'path': test_paths, 'label': test_labels, 'trigger': 0})
    train_df.to_csv(paths_dir + "/train_clean.csv", index=False)
    test_df.to_csv(paths_dir + "/test_clean.csv", index=False)

def create_cifar_poison_test_target(
    paths_dir, images_dir, trigger_dir, trigger_width, trigger_location, poison_ratio, target_label
):
    os.makedirs(images_dir + f"/{target_label}", exist_ok=True)
    test_df = pd.read_csv(paths_dir + "/test_clean.csv")
    with open(paths_dir + f'/test_{target_label}_{poison_ratio}_0.2.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'label', 'trigger'])
        for i in trange(len(test_df)):
            if random() > 0.025: continue
            path = test_df.loc[i, 'path']
            if random() < poison_ratio:
                img = patch(path, trigger_dir, trigger_width, trigger_location)
                save_dir = images_dir + f'/{target_label}/test_' + str(i) + ".png"
                img.save(save_dir)
                writer.writerow([os.path.abspath(save_dir), target_label, 1])
            else:
                writer.writerow([path, test_df.loc[i, 'label'], 0])

def create_poison_data_train(
    paths_dir, images_dir, trigger_dir, trigger_width, trigger_location, poison_shot, target_label, shot=32
):
    os.makedirs(images_dir + f"/poison_n{shot}_pn{poison_shot}_w{int(trigger_width*100)}_loc{int(trigger_location*100)}", exist_ok=True)
    clean_train_df = pd.read_csv(f"{paths_dir}/train_clean.csv").sample(frac=1)
    with open(f"{paths_dir}/train_poison_n{shot}_pn{poison_shot}_w{int(trigger_width*100)}_loc{int(trigger_location*100)}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'label', 'trigger'])
        for label in tqdm(set(clean_train_df['label'])):
            for i in range(shot):
                path = clean_train_df[clean_train_df['label'] == label].iloc[i]['path']
                if i < poison_shot:
                    img = patch(path, trigger_dir, trigger_width, trigger_location)
                    save_dir = f"{images_dir}/poison_n{shot}_pn{poison_shot}_w{int(trigger_width * 100)}_loc{int(trigger_location * 100)}/train_{label}_{i}.png"
                    img.save(save_dir)
                    writer.writerow([os.path.abspath(save_dir), target_label, 1])
                else:
                    writer.writerow([path, label, 0])


if __name__ == '__main__':
    trainset = torchvision.datasets.SVHN(root="../svhn", split='train', download=True)
    # create_svhn_clean("../svhn", "../svhn/images/", "../svhn/paths/")
    # create_cifar_poison_test_target(
    #     "../svhn/paths", "../svhn/images", "../triggers/trigger_10.png", 0.3, 0.6, 0.3, 0
    # )
    # create_poison_data_train('../svhn/paths', '../svhn/images', '../triggers/trigger_10.png', 0.3, 0.6, 0, 0, 12)
    # create_poison_data_train('../svhn/paths', '../svhn/images', '../triggers/trigger_10.png', 0.3, 0.6, 1, 0, 12)
    # create_poison_data_train('../svhn/paths', '../svhn/images', '../triggers/trigger_10.png', 0.3, 0.6, 0, 0, 14)
    # create_poison_data_train('../svhn/paths', '../svhn/images', '../triggers/trigger_10.png', 0.3, 0.6, 1, 0, 14)
    create_poison_data_train('../svhn/paths', '../svhn/images', '../triggers/trigger_10.png', 0.3, 0.6, 0, 0, 4)
    # create_poison_data_train('../svhn/paths', '../svhn/images', '../triggers/trigger_10.png', 0.3, 0.6, 1, 0, 4)
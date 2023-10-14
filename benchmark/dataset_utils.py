import json
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat

def load_newt_task_df(task_dir):
    """ Load a NeWT binary task from .csv
    In [1]: ls $task_dir
    newt2021_images/        public_test.json.tar.gz  train.tar.gz  val.json.tar.gz
    newt2021_images.tar.gz  train.json               val/          val.tar.gz
    newt2021_labels.csv     train.json.tar.gz        val.json
    """
    label_pth = os.path.join(task_dir, "newt2021_labels.csv")
    image_dir = os.path.join(task_dir, "newt2021_images")
    assert os.path.exists(label_pth), "expected newt2021_labels.csv in %s" % task_dir
    assert os.path.exists(image_dir), "expected newt2021_images in %s" % task_dir
    df = pd.read_csv(label_pth)
    df['image_fps'] = image_dir + os.sep + df['id'] + '.jpg'
    assert all([os.path.exists(fp) for fp in df['image_fps'].values.tolist()]), "expected all image_fps to exist"
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    assert len(train_df) + len(test_df) == len(df), "expected disjoint train/test split"
    assert set(train_df['label'].unique()) == set([0, 1]), "expected binary labels"
    # train_image_fps, train_labels, val_image_fps, val_labels
    return train_df['image_fps'].values.tolist(), train_df['label'].values.tolist(), test_df['image_fps'].values.tolist(), test_df['label'].values.tolist()
    
def load_newt_task(task_dir, is_df=False):
    """ Load a NeWT binary task
    """
    if is_df:
        return load_newt_task_df(task_dir)
    with open(os.path.join(task_dir, "train.json")) as f:
        train_dataset = json.load(f)
    with open(os.path.join(task_dir, "val.json")) as f:
        val_dataset = json.load(f)

    image_id_to_fp = {image['id'] : image['file_name'] for image in train_dataset['images'] + val_dataset['images']}
    assert len(image_id_to_fp) == len(train_dataset['images']) + len(val_dataset['images']), "overlapping images in %s ?" % task_dir

    train_image_fps = []
    train_labels = []
    for anno in train_dataset['annotations']:

        image_fp = os.path.join(task_dir, image_id_to_fp[anno['image_id']])
        image_label = anno['category_id']
        assert image_label in [0, 1], "unexpected category id, assumed binary?"

        train_image_fps.append(image_fp)
        train_labels.append(image_label)

    val_image_fps = []
    val_labels = []
    for anno in val_dataset['annotations']:

        image_fp = os.path.join(task_dir, image_id_to_fp[anno['image_id']])
        image_label = anno['category_id']
        assert image_label in [0, 1], "unexpected category id, assumed binary?"

        val_image_fps.append(image_fp)
        val_labels.append(image_label)

    return train_image_fps, train_labels, val_image_fps, val_labels

def load_cub(dataset_path, label_file_name='image_class_labels.txt'):
    """ Load the CUB 200 dataset
    """

    # load data
    data = pd.read_csv(os.path.join(dataset_path, label_file_name), sep=' ', names=['id', 'class_label'])
    ids = data['id'].values
    labels = data.set_index('id').loc[ids].reset_index(inplace=False)['class_label'].values.astype(np.int)
    _, labels = np.unique(labels, return_inverse=True)

    files = pd.read_csv(os.path.join(dataset_path, 'images.txt'), sep=' ', names=['id', 'file'])
    files = files.set_index('id').loc[ids].reset_index(inplace=False)['file'].values
    files = [os.path.join(dataset_path, 'images', ff) for ff in files]

    is_train = pd.read_csv(os.path.join(dataset_path, 'train_test_split.txt'), sep=' ', names=['id', 'is_train'])
    is_train = is_train.set_index('id').loc[ids].reset_index(inplace=False)['is_train'].values.astype(np.int)

    train_paths = []
    train_classes = []
    test_paths = []
    test_classes = []

    for ii in range(len(files)):
        if is_train[ii] == 1:
            train_paths.append(files[ii])
            train_classes.append(labels[ii])
        else:
            test_paths.append(files[ii])
            test_classes.append(labels[ii])

    return train_paths, train_classes, test_paths, test_classes


def load_oxford_flowers(dataset_path):

    classes = loadmat(os.path.join(dataset_path, 'imagelabels.mat'))['labels'][0, :]-1
    train_ids = loadmat(os.path.join(dataset_path, 'setid.mat'))['trnid'][0, :]
    test_ids = loadmat(os.path.join(dataset_path, 'setid.mat'))['tstid'][0, :]
    train_paths = ['image_' + str(jj).zfill(5) + '.jpg' for jj in train_ids]
    test_paths = ['image_' + str(jj).zfill(5) + '.jpg' for jj in test_ids]
    train_paths = [os.path.join(dataset_path, 'jpg', jj)  for jj in train_paths]
    test_paths = [os.path.join(dataset_path, 'jpg', jj)  for jj in test_paths]
    train_classes = classes[train_ids-1].tolist()
    test_classes = classes[test_ids-1].tolist()

    return train_paths, train_classes, test_paths, test_classes


def load_stanford_dogs(dataset_path):
    train_paths = [jj[0][0] for jj in loadmat(os.path.join(dataset_path, 'train_data.mat'))['file_list']]
    test_paths = [jj[0][0] for jj in loadmat(os.path.join(dataset_path, 'test_data.mat'))['file_list']]
    train_paths = [os.path.join(dataset_path, 'Images', jj)  for jj in train_paths]
    test_paths = [os.path.join(dataset_path, 'Images', jj)  for jj in test_paths]
    train_classes = (loadmat(os.path.join(dataset_path, 'train_data.mat'))['labels'][:, 0]-1).tolist()
    test_classes = (loadmat(os.path.join(dataset_path, 'test_data.mat'))['labels'][:, 0]-1).tolist()

    return train_paths, train_classes, test_paths, test_classes

def load_stanford_cars(dataset_path):

    anns = loadmat(os.path.join(dataset_path, 'cars_annos.mat'))['annotations'][0]
    im_paths = [str(aa[0][0]) for aa in anns]
    im_paths = [os.path.join(dataset_path, aa) for aa in im_paths]
    classes = [int(aa[5][0][0])-1 for aa in anns]
    is_test = [int(aa[6][0][0]) for aa in anns]

    train_paths = []
    train_classes = []
    test_paths = []
    test_classes = []

    for ii in range(len(im_paths)):
        if is_test[ii] == 1:
            test_paths.append(im_paths[ii])
            test_classes.append(classes[ii])
        else:
            train_paths.append(im_paths[ii])
            train_classes.append(classes[ii])

    return train_paths, train_classes, test_paths, test_classes


def load_dataset(dataset_name, dataset_path):

    if dataset_name == 'CUB':
        return load_cub(dataset_path)
    elif dataset_name == 'CUBExpert':
        return load_cub(dataset_path, label_file_name='image_class_labels.txt')
    elif dataset_name == 'NABirds':
        return load_cub(dataset_path)
    elif dataset_name == 'OxfordFlowers':
        return load_oxford_flowers(dataset_path)
    elif dataset_name == 'StanfordDogs':
        return load_stanford_dogs(dataset_path)
    elif dataset_name == 'StanfordCars':
        return load_stanford_cars(dataset_path)
    else:
        raise ValueError("Unknown dataset name: %s" % dataset_name)
    

if __name__ == "__main__":
    import IPython; IPython.embed()
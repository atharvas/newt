import os
from typing import Dict
import pandas as pd
import torch
from functools import partial
import numpy as np
import sklearn
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score
from pt_resnet_feature_extractor import PTResNet50FeatureExtractor, load_feature_extractor

# https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
def linearsvc(X_train, y_train, X_test, y_test, max_iter=1000, grid_search=False, predefined_val_indices=None, standardize=False, normalize=True, dual=False):
    """
    """

    if standardize:
        scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    if normalize:
        X_train = sklearn.preprocessing.normalize(X_train, norm='l2')
        X_test = sklearn.preprocessing.normalize(X_test, norm='l2')

    clf = LinearSVC(
        random_state=0,
        tol=1e-5,
        C=1.,
        dual=dual,
        class_weight=None,
        max_iter=max_iter
    )

    if grid_search:

        C_values = [0.0001, 0.001, 0.01, 0.1, 1., 10., 100., 1000.]
        parameters = {'C' : C_values}

        if predefined_val_indices is not None:
            cv = PredefinedSplit(test_fold=predefined_val_indices)
        else:
            cv = 3
        clf = GridSearchCV(clf, parameters, n_jobs=-1, cv=cv, refit=True)

    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    results = {
        'acc' : accuracy_score(y_test, y_pred)
    }

    if grid_search:
        results['best_param'] =  clf.best_params_['C']

    return results, clf

class NEWTClassifers(torch.nn.Module):
    def __init__(self, feature_extractor: PTResNet50FeatureExtractor, classifiers: Dict[str, LinearSVC], use_torch : bool):
        super(NEWTClassifers, self).__init__()
        self.feature_extractor = feature_extractor.model
        if use_torch:
            self.classifiers = {}
            for task_name, classifier in classifiers.items():
                self.classifiers[task_name] = torch.nn.Linear(classifier.n_features_in_, 1)
                self.classifiers[task_name].load_state_dict(dict(
                    weight=torch.from_numpy(classifier.coef_).float(),
                    bias=torch.from_numpy(classifier.intercept_).float()
                ))
            self.classifiers = torch.nn.ModuleDict(self.classifiers)
        else:
            self.classifiers = classifiers

    def forward(self, x, task_name):
        assert task_name in self.classifiers, "Task name %s not found" % task_name
        x = self.feature_extractor(x)
        if self.use_torch:
            x = x.reshape(x.shape[0], -1)
            x = self.classifiers[task_name](x)
            prediction = (x > 0).float()
        else:
            x = x.reshape(x.shape[0], -1).detach().numpy()
            prediction = self.classifiers[task_name].predict(x)
        return prediction

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_performing_model_tag", type=str, required=True)
    parser.add_argument("--best_performing_model_loc", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--newt_dir", type=str, default="/mnt/sdd1/atharvas/viper/data/inat/newt/newt2021/")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    assert os.path.exists(args.best_performing_model_loc), "Model location does not exist"
    assert os.path.exists(args.save_dir), "Save dir does not exist"
    assert os.path.exists(args.newt_dir), "Newt dir does not exist"
    args.newt_dataset_loc = os.path.realpath(args.newt_dir) + "/"
    args.save_dir = os.path.realpath(args.save_dir) + "/"
    resnet_feature_extractor = load_feature_extractor(dict(name=args.best_performing_model_tag, weights=args.best_performing_model_loc), device=args.device)

    # Extract featuers for all images
    newt_labels = pd.read_csv(os.path.join(args.newt_dataset_loc + "newt2021_labels.csv"))
    all_images = (os.path.join(args.newt_dataset_loc + "newt2021_images/") + newt_labels['id'] + ".jpg").tolist()
    resnet_feature_extractor.device = args.device
    resnet_feature_extractor.model.to(resnet_feature_extractor.device)
    # A minute/~40GB on A40
    print("Extracting resnet[%s] features" % args.best_performing_model_tag)
    features = resnet_feature_extractor.extract_features_batch(image_fp_list=all_images,
                                                    batch_size=2048,
                                                    num_workers=8)
    # save features
    np.savez_compressed(os.path.join(args.save_dir, "%s_img_features.npy" % args.best_performing_model_tag), features)
    clf = partial(linearsvc, max_iter=10000, grid_search=True, standardize=True, normalize=True, dual=False)

    classifier_dict = {}
    for task_name, task in newt_labels.groupby('task'):
        train = task[task['split'] == 'train']
        test = task[task['split'] == 'test']
        X_train = features[train.index]
        X_test = features[test.index]
        y_train = train['label'].values
        y_test = test['label'].values
        results, trainer_obj = clf(X_train, y_train, X_test, y_test)
        best_classifier = trainer_obj.best_estimator_
        # construct classifier
        classifier_dict[task_name] = best_classifier
        print("Task: %s, acc: %f" % (task_name, results['acc']))
    
    # save classifiers
    import pickle
    with open(os.path.join(args.save_dir, "%s_sk_classifiers.pkl" % args.best_performing_model_tag), 'wb') as f:
        pickle.dump(classifier_dict, f)
    
    # save pretrained model
    torch.save(resnet_feature_extractor, os.path.join(args.save_dir, "%s_resnet50.pth" % args.best_performing_model_tag))
    
    torch_classifiers = NEWTClassifers(resnet_feature_extractor, classifier_dict, use_torch=True)
    
    # save label mapping
    task_label_mapping = {task : dict(zip(g['label'], g['text_label'])) for task, g in newt_labels[['task',  'label', 'text_label']].groupby('task')}
    with open(os.path.join(args.save_dir, "task_label_mapping.pkl"), 'wb') as f:
        pickle.dump(task_label_mapping, f)

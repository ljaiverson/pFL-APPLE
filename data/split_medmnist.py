from sklearn.model_selection import train_test_split
import numpy as np

import os

def train_test_split_for_medmnist(data_name, random_state=42, test_size=0.2):
    """
        Splits datasets in MedMNIST collection into a training set and a test set in a stratified fashion.
        Make sure the dataset is downloaded before using this function.
    """
    assert os.path.exists(data_name), "%s does not exist. Please download it first before splitting." % data_name
    data = np.load(data_name)
    data_images = np.concatenate([data["train_images"], data["val_images"], data["test_images"]], axis=0)
    data_labels = np.concatenate([data["train_labels"], data["val_labels"], data["test_labels"]], axis=0)
    data_labels = np.squeeze(data_labels)
    x_train, x_test, y_train, y_test = train_test_split(data_images, data_labels, test_size=test_size, \
                                                    random_state=random_state, stratify=data_labels)
    print("x_train", x_train.shape)
    print("y_train", y_train.shape)
    print("x_test", x_test.shape)
    print("y_test", y_test.shape)

    fn = os.path.splitext(data_name)
    fn = fn[0] + "_train_test" + fn[1]
    np.savez(fn, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    print("==> data saved at %s" % fn)

if __name__ == "__main__":
    for fn in ["organmnist_axial", "pathmnist"]:
        train_test_split_for_medmnist("%s.npz" % fn)
import os
import random
import shutil
from progress.bar import Bar


class SplitDataset():
    def __init__(self,
                 dataset_dir,
                 saved_dataset_dir,
                 train_ratio=0.6,
                 test_ratio=0.2,
                 show_progress=True):
        self.dataset_dir = dataset_dir
        self.saved_dataset_dir = saved_dataset_dir
        self.saved_train_dir = os.path.join(saved_dataset_dir, 'train')
        self.saved_test_dir = os.path.join(saved_dataset_dir, 'test')
        self.saved_valid_dir = os.path.join(saved_dataset_dir, 'valid')

        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.valid_ratio = 1 - train_ratio - test_ratio

        self.show_progress = show_progress

        self.label_dict = {}
        self.train_file_path = []
        self.test_file_path = []
        self.valid_file_path = []

        if not os.path.exists(self.saved_dataset_dir):
            os.mkdir(self.saved_dataset_dir)
        if not os.path.exists(self.saved_train_dir):
            os.mkdir(self.saved_train_dir)
        if not os.path.exists(self.saved_test_dir):
            os.mkdir(self.saved_test_dir)
        if not os.path.exists(self.saved_valid_dir):
            os.mkdir(self.saved_valid_dir)

    def get_labels(self):
        labels = []
        for item in os.listdir(self.dataset_dir):
            item_path = os.path.join(self.dataset_dir, item)
            if os.path.isdir(item_path):
                labels.append(item)
        return labels

    def get_all_file_path(self):
        all_file_path = []
        labels = self.get_labels()
        for i, label in enumerate(labels):
            self.label_dict[i] = label

            label_path = os.path.join(self.dataset_dir, label)
            file_path = []
            for file in os.listdir(label_path):
                single_file_path = os.path.join(label_path, file)
                file_path.append(single_file_path)
            all_file_path.append(file_path)
        return all_file_path

    def split_dataset(self):
        all_file_paths = self.get_all_file_path()
        for index in range(len(all_file_paths)):
            label_file_list = all_file_paths[index]
            label_file_length = len(label_file_list)
            random.shuffle(label_file_list)

            train_num = int(label_file_length * self.train_ratio)
            test_num = int(label_file_length * self.test_ratio)

            self.train_file_path.append([self.label_dict[index], label_file_list[:train_num]])
            self.test_file_path.append([self.label_dict[index], label_file_list[train_num:train_num + test_num]])
            self.valid_file_path.append([self.label_dict[index], label_file_list[train_num + test_num:]])

    def copy_file(self, files, saved_dir):
        sum_ = sum([len(f[1]) for f in files])
        bar = Bar('Copy to ' + saved_dir, max=sum_)
        for item in files:
            list_file = item[1]
            dst_path = os.path.join(saved_dir, item[0])
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
            for file in list_file:
                shutil.copy(file, dst_path)
                bar.next()
        bar.finish()

    def start_split(self):
        self.split_dataset()
        self.copy_file(files=self.train_file_path, saved_dir=self.saved_train_dir)
        self.copy_file(files=self.test_file_path, saved_dir=self.saved_test_dir)
        self.copy_file(files=self.valid_file_path, saved_dir=self.saved_valid_dir)


if __name__ == '__main__':
    split_dataset = SplitDataset(dataset_dir='original_dataset',
                                 saved_dataset_dir='dataset')
    split_dataset.start_split()

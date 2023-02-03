import os
import pandas as pd

from torch.utils import data
from PIL import Image

from Utils.Error_Logger import error_logger


class Loader(data.DataLoader):
    def __init__(self, db_path, run_type, transform=None):
        """
        DataLoader Custom DataSet

        :param db_path: DATABASE PATH
        :param run_type: Model run type (train, test, valid)
        :param transform: Image style transform module -> 디노에서는 training loop에서 적용하는게 좋을꺼 같다.

        :var _data_info: Dataframe with DB information for us to use [folder, image name, class]
        :var self.path_list: Data Path information list
        :var self.label_list: Data class information list
        """

        super(Loader, self).__init__(self)
        self.place = os.path.realpath(__file__)
        self.db_path = db_path
        self.run_type = run_type
        self.transform = transform

        _data_info = None
        try:
            _data_info = pd.read_csv(f'{self.db_path}/{self.run_type}.csv')
        except Exception as e:
            error_logger('None', self.place, self.__class__.__name__, e)

        assert self.run_type == 'train' or self.run_type == 'test' or self.run_type == 'valid', \
            'Only train, test, and valid are available for run_type.'

        if self.run_type == 'test':
            self.path_list = self.get_paths(_data_info)[:1]
            self.label_list = self.get_labels(_data_info)[:1]

        elif self.run_type == 'train' or self.run_type == 'valid':
            self.path_list = self.get_paths(_data_info)[:1]
            self.label_list = self.get_labels(_data_info)[:1]

    def get_paths(self, data_info):
        """
        Get Path information in Data information

        :param data_info: Dataframe with DB information for us to use [folder, image name, class]
        :var paths_list: full path information list
        :return: full path information
        """
        paths_list = []
        paths_info = data_info.iloc[:, 0:2].values

        return self.make_path_list(paths_info, paths_list)

    def make_path_list(self, path_info, paths_list: list):
        """
        Make Path list using Path information

        :param path_info: Extract path info only from data_info [path info: folder, image name]
        :param paths_list: Complete path information created from extracted information
        :return: List with complete path information
        """
        try:
            for folder, image_name in path_info:
                full_path = f'{self.db_path}/{self.run_type}/{folder}/{image_name}'
                paths_list.append(full_path)

            return paths_list

        except Exception as e:
            error_logger('None', self.place, self.__class__.__name__, e)

    def get_labels(self, data_info):
        """
        Get Label information in Data information

        :param data_info: Dataframe with DB information for us to use [folder, image name, class]
        :return: Extract class info only from data_info
        """
        return data_info.iloc[:, 2:].values

    def get_ImageName(self, path: str):
        """
        Get Image Name in Path information

        :param path: Full path to the image
        :return: Image Name
        """
        ImageName = path.split('/')[-1]
        ImageName = ImageName.split(".")[0]

        return ImageName

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index: int):
        try:
            if self.run_type == 'test':
                item = self.transform(
                    Image.open(self.path_list[index])
                )
                label = self.label_list[index]

                image_name = self.get_ImageName(self.path_list[index])

                return [item, label, image_name]

            else:
                item = self.transform(
                    Image.open(self.path_list[index])
                )

                label = self.label_list[index]
                image_name = self.get_ImageName(self.path_list[index])

                return [item, label, image_name]

        except Exception as e:
            image_name = self.get_ImageName(self.path_list[index])
            error_logger(image_name, self.place, self.__class__.__name__, e=e)
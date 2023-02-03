import os
from Utils.Error_Logger import error_logger


class LossAccDisplayer:
    def __init__(self, name_list):
        """
        :param name_list: 우리가 확인하기를 원하는 losses 의 이름 list

        :var self.count: 총 iter 의 횟수
        :var self.value_list: loss values 가 저장될 list
        """
        self.place = os.path.realpath(__file__)
        self.count = 0
        self.name_list = name_list
        self.value_list = [0] * len(self.name_list)

    def record(self, values):
        self.count += 1
        try:
            for i, value in enumerate(values):
                self.value_list[i] += value
        except Exception as e:
            error_logger('None', self.place, self.__class__.__name__, e=e)

    def get_avg_losses(self):
        try:
            return [value / self.count for value in self.value_list]
        except Exception as e:
            error_logger('None', self.place, self.__class__.__name__, e=e)

    def reset(self):
        self.count = 0
        self.value_list = [0] * len(self.name_list)
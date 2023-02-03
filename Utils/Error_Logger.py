import csv
import sys
import traceback
import time


def error_logger(image_name: str, place: str, class_name: str, e: Exception):
    """
    모델에서 발샣하는 에러에 대한 처리는 제외합니다.
    모델에서 Error 발생시 학습 자체가 불가능하기에 따로 처리하지 않았습니다.

    :param image_name: Error Image Name
    :param place: Error .py file Name
    :param class_name: Error function name
    :param e: Error Type

    :var stop: can process stop

    :return: sys.exit (exit program), You can change the code to continue if you want.
    """
    stop = True
    assert type(stop) is bool, 'Only boolean type is available for stop in error_logger class'

    error_log = [image_name, place, class_name, e]
    f = open('C:/Users/rlawj/backup/log/Error_Log.csv', "a")
    writer = csv.writer(f, lineterminator='\n')

    writer.writerow(error_log)
    f.close()

    print(traceback.format_exc())
    if stop:
        time.sleep(0.5)
        sys.exit(f'ERROR!! [ IMAGE: {image_name} | PLACE: {place} | FUNCTION: {class_name} | TYPE: {e}')
    else:
        pass
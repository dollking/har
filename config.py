from os import path


class FilePath(object):
    ROOT_PATH = '/home/algorithm/HAR'
    DATA_PATH = path.join(ROOT_PATH, 'data')
    RAW_DATA = path.join(DATA_PATH, 'raw_data.txt')
    SAVE_PATH = path.join(ROOT_PATH, 'save')


class LeaningParameter(object):
    LEANING_RATE = 0.0001
    EPOCH = 1000
    FOLD_SIZE = 10


class PreProcessingParameter(object):
    SIZE_PER_SEC = 20
    TIME_SIZE = 5

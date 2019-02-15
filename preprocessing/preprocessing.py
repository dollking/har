import os
import random
import pickle


class PreProcessing(object):
    def __init__(self, fp, pp):
        self.root_path = fp.ROOT_PATH
        self.file_path = fp.RAW_DATA
        self.data_path = fp.DATA_PATH
        self.enum = {'Walking': 0, 'Jogging': 1, 'Sitting': 2, 'Standing': 3, 'Upstairs': 4, 'Downstairs': 5}
        self.time_size = pp.SIZE_PER_SEC * pp.TIME_SIZE     # 20Hz
        self.shift_size = self.time_size // 3

    def preprocess(self):
        raw_data = [data.split(',') for data in open(self.file_path).read().replace('\n', '').split(';')][:-1]
        raw_data = [[int(data[0]), self.enum[data[1]], int(data[2]), float(data[3]), float(data[4]), float(data[5])]
                    for data in raw_data]

        data_by_user = [[] for _ in range(36)]

        for data in raw_data:
            data_by_user[data[0] - 1].append(data)

        for data in data_by_user:
            data.sort(key=lambda x: x[1])

        train_set = []
        for j in range(len(data_by_user) - 1):
            data = data_by_user[j]

            left_pos = right_pos = 0
            while right_pos < len(data):
                while right_pos < (len(data) - 1) and data[right_pos][1] == data[right_pos + 1][1]:
                    right_pos += 1

                while left_pos + self.time_size - 1 <= right_pos:
                    temp = []
                    temp.append([[data[left_pos + i][3], data[left_pos + i][4], data[left_pos + i][5]]
                                 for i in range(self.time_size)])
                    temp.append(data[left_pos + self.time_size - 1][1])
                    train_set.append(temp)
                    left_pos += self.shift_size

                left_pos = right_pos = right_pos + 1

            for _ in range(5):
                random.shuffle(train_set)

            fp = open(os.path.join(self.data_path, 'train_set.pkl'), 'wb')
            pickle.dump(train_set, fp)
            fp.close()

        data = data_by_user[-1]

        test_set = []
        left_pos = right_pos = 0
        while right_pos < len(data):
            while right_pos < (len(data) - 1) and data[right_pos][1] == data[right_pos + 1][1]:
                right_pos += 1

            while left_pos + self.time_size - 1 <= right_pos:
                temp = []
                temp.append([[data[left_pos + i][3], data[left_pos + i][4], data[left_pos + i][5]]
                             for i in range(self.time_size)])
                temp.append(data[left_pos + self.time_size - 1][1])
                test_set.append(temp)
                left_pos += self.shift_size

            left_pos = right_pos = right_pos + 1

        for _ in range(5):
            random.shuffle(test_set)

        fp = open(os.path.join(self.data_path, 'test_set.pkl'), 'wb')
        pickle.dump(test_set, fp)
        fp.close()

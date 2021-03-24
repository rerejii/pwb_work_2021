import tensorflow as tf
import random
import csv
import pandas as pd
import sys

class DataManager():
    def __init__(self, tfrecord_path, img_root, batch_size,
                 net_cls, data_n, cycle_length=4, suffle_buffer=-1,
                 data_cls=3, rot_p=0.0, use_aveimg=False,
                 standardization_csv_path=None,
                 rot4=False, hsv_to_rgp=False,):
        self.data_cls = data_cls
        self.img_root = img_root
        self.net_cls = net_cls
        self.rot_p = rot_p
        self.rot4 = rot4
        self.seed = 0
        self.tfrecord_path = tfrecord_path
        self.suffle_buffer = suffle_buffer
        # データセットの読み込み及び整形
        _dataset = tf.data.Dataset.list_files(self.tfrecord_path)
        self.dataset = _dataset.interleave(tf.data.TFRecordDataset,
                                           cycle_length=cycle_length,
                                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.batch_size = batch_size
        self.max_data_n = data_n
        self.remain_data_n = 0
        self.ds_iter = None
        self.standardization_csv_path = standardization_csv_path
        self.std_list = None
        self.hsv_to_rgp = hsv_to_rgp
        if self.standardization_csv_path is not None:
            df = pd.read_csv(self.standardization_csv_path)
            self.std_list = [ df['R_mean'] ,df['R_std'], df['G_mean'] ,df['G_std'], df['B_mean'] ,df['B_std'], ]
            

    def get_inited_iter(self):
        _dataset = self.dataset.map(self._parse_function)
        if (self.suffle_buffer!=-1):
            _dataset = _dataset.shuffle(int(self.suffle_buffer), seed=self.seed)
        _dataset = _dataset.map(self._image_from_path)
        # _dataset = _dataset.map(self._netdata_normalize) \
        #     if self.standardization_csv_path is None else _dataset.map(self._standardization)
        if self.standardization_csv_path is not None:
            _dataset = _dataset.map(self._standardization)
        elif self.hsv_to_rgp:
            _dataset = _dataset.map(self._hsv_to_rgb)
        else:
            _dataset = _dataset.map(self._netdata_normalize)
        if self.rot_p != 0.0:
            _dataset = _dataset.map(self._random_rot90)  # random rotation
        _dataset = _dataset.batch(self.batch_size)
        self.remain_data_n = self.max_data_n
        self.seed = self.seed + 1
        return iter(_dataset)

    def data_used_apply(self, use_count=1):
        self.remain_data_n = self.remain_data_n - (use_count * self.batch_size)
        if self.remain_data_n < 0:
            self.remain_data_n = 0

    def get_remain_data(self):
        return self.remain_data_n

    def get_batch_size(self):
        return self.batch_size

    def get_total_data(self):
        return self.max_data_n

    def get_next_data_rate(self):
        if self.remain_data_n > self.batch_size:
            return 1
        else:
            return self.remain_data_n / self.batch_size

    def get_next_data_n(self):
        if self.remain_data_n > self.batch_size:
            return self.batch_size
        else:
            return self.remain_data_n

    def _random_rot90(self, _sample_image, _answer_image):
        if tf.random.uniform([], seed=self.seed) < self.rot_p:
            _sample_image = tf.image.rot90(_sample_image)
            _answer_image = tf.image.rot90(_answer_image)
        return _sample_image, _answer_image

    def _random_rot4(self, _sample_image, _answer_image): # 0.75 0.5 0.25
        if tf.random.uniform([], seed=self.seed) > 0.75:
            _sample_image = tf.image.rot90(_sample_image)
            _answer_image = tf.image.rot90(_answer_image)
        if tf.random.uniform([], seed=self.seed) > 0.50:
            _sample_image = tf.image.rot90(_sample_image)
            _answer_image = tf.image.rot90(_answer_image)
        if tf.random.uniform([], seed=self.seed) > 0.25:
            _sample_image = tf.image.rot90(_sample_image)
            _answer_image = tf.image.rot90(_answer_image)
        return _sample_image, _answer_image
        
    def _hsv_to_rgb(self, _sample_image, _answer_image):
        _sample_image = tf.cast(_sample_image, tf.float32)
        _answer_image = tf.cast(_answer_image, tf.float32)
        _sample_image = tf.image.rgb_to_hsv(_sample_image)
        _answer_image = self.net_cls.netdata_from_img(_answer_image)
        return _sample_image, _answer_image

    # tfrecordsの解析読み込み関数
    def _parse_function(self, _data):
        _features = {'sample_path': tf.io.FixedLenFeature((), tf.string, default_value=''),
                    'answer_path': tf.io.FixedLenFeature((), tf.string, default_value=''),}  # dictの型を作る
        _parsed_features = tf.io.parse_single_example(serialized=_data, features=_features)  # データを解析しdictにマッピングする
        return _parsed_features  # 解析したデータを返す

    # パスから画像を読みだす
    def _image_from_path(self, _parsed_features):
        _sample_png_bytes = tf.io.read_file(self.img_root + '/' + _parsed_features['sample_path'])
        _sample_image = tf.image.decode_png(_sample_png_bytes, channels=self.data_cls, dtype=tf.dtypes.uint8,)
        _answer_png_bytes = tf.io.read_file(self.img_root + '/' + _parsed_features['answer_path'])
        _answer_image = tf.image.decode_png(_answer_png_bytes, channels=1, dtype=tf.dtypes.uint8,)
        return _sample_image, _answer_image

    # normalizing the images
    def _netdata_normalize(self, _sample_image, _answer_image):
        _sample_image = tf.cast(_sample_image, tf.float32)
        _answer_image = tf.cast(_answer_image, tf.float32)
        _sample_image = self.net_cls.netdata_from_img(_sample_image)
        _answer_image = self.net_cls.netdata_from_img(_answer_image)
        return _sample_image, _answer_image
        
    def _standardization(self, _sample_image, _answer_image):
        _sample_image = tf.cast(_sample_image, tf.float32)
        _answer_image = tf.cast(_answer_image, tf.float32)
        
        _sample_image_R = (_sample_image[:, :, 0] - self.std_list[0]) / self.std_list[1]  # R_std, R_mean
        _sample_image_G = (_sample_image[:, :, 1] - self.std_list[2]) / self.std_list[3]
        _sample_image_B = (_sample_image[:, :, 2] - self.std_list[4]) / self.std_list[5]
        _sample_image = tf.stack([_sample_image_R, _sample_image_G, _sample_image_B], 2)
        _answer_image = self.net_cls.netdata_from_img(_answer_image)
        return _sample_image, _answer_image

    # ===== gray =====
    # tfrecordsの解析読み込み関数
    def _gray_parse_function(self, _data):
        _features = {'sample_path': tf.io.FixedLenFeature((), tf.string, default_value=''),
                    'answer_path': tf.io.FixedLenFeature((), tf.string, default_value=''),
                    'average_path': tf.io.FixedLenFeature((), tf.string, default_value=''), }  # dictの型を作る
        _parsed_features = tf.io.parse_single_example(serialized=_data, features=_features)  # データを解析しdictにマッピングする
        return _parsed_features  # 解析したデータを返す

    # パスから画像を読みだす
    def _gray_image_from_path(self, _parsed_features):
        _sample_png_bytes = tf.io.read_file(self.img_root + '/' + _parsed_features['sample_path'])
        _sample_image = tf.image.decode_png(_sample_png_bytes, channels=self.data_cls, dtype=tf.dtypes.uint8,)
        _answer_png_bytes = tf.io.read_file(self.img_root + '/' + _parsed_features['answer_path'])
        _answer_image = tf.image.decode_png(_answer_png_bytes, channels=1, dtype=tf.dtypes.uint8,)
        _average_png_bytes = tf.io.read_file(self.img_root + '/' + _parsed_features['average_path'])
        _average_image = tf.image.decode_png(_average_png_bytes, channels=1, dtype=tf.dtypes.uint8, )
        return _sample_image, _answer_image, _average_image

    # normalizing the images
    def _gray_netdata_normalize(self, _sample_image, _answer_image, _average_image):
        _sample_image = tf.cast(_sample_image, tf.float32)
        _answer_image = tf.cast(_answer_image, tf.float32)
        _average_image = tf.cast(_average_image, tf.float32)
        _sample_image = self.net_cls.netdata_from_img(_sample_image)
        _answer_image = self.net_cls.netdata_from_img(_answer_image)
        _average_image = self.net_cls.netdata_from_img(_average_image)
        return _sample_image, _answer_image, _average_image

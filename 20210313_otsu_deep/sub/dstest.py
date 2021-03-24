import tensorflow as tf
import os

# def parse_function(_data):
#     _features = {'code_Y': tf.io.FixedLenFeature([], tf.int64, default_value=0),
#                  'code_X': tf.io.FixedLenFeature([], tf.int64, default_value=0),
#                  'R_std': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
#                  'R_mean': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
#                  'G_std': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
#                  'G_mean': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
#                  'B_std': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
#                  'B_mean': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),}  # dictの型を作る
#     _parsed_features = tf.io.parse_single_example(serialized=_data, features=_features)  # データを解析しdictにマッピングする
#     return _parsed_features  # 解析したデータを返す

def parse_function(_data):
    _features = {'std_val': tf.io.FixedLenFeature([6], tf.float32, default_value=[0.,0.,0.,0.,0.,0.]),
                 'code_Y': tf.io.FixedLenFeature([], tf.int64),
                 'code_X': tf.io.FixedLenFeature([], tf.int64),}  # dictの型を作る
    _parsed_features = tf.io.parse_example(serialized=_data, features=_features)  # データを解析しdictにマッピングする
    return _parsed_features  # 解析したデータを返す

tfrecord_path = 'E:/work/myTensor/dataset2/DR-256-valid0-more/A/LearnData-A.tfrecords'
# print('//////////////////////')
# print(os.path.isfile(tfrecord_path))
# print('//////////////////////')
dataset = tf.data.Dataset.list_files(tfrecord_path)
dataset = dataset.interleave(tf.data.TFRecordDataset,
                                cycle_length=4,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.map(parse_function)

ds = iter(dataset)
print(next(ds))
print(next(ds))
# print(next(ds))

            # stand_list = [df['R_std'].iloc[img_index], df['R_mean'].iloc[img_index],
            #               df['G_std'].iloc[img_index], df['G_mean'].iloc[img_index],
            #               df['B_std'].iloc[img_index], df['B_mean'].iloc[img_index],]
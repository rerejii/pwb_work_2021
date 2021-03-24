import tensorflow as tf
import sys
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
import shutil
import time
import csv
import os
from datetime import datetime
import numpy as np
import pandas as pd

black = [0, 0, 0]  # 予測:0黒 解答:0黒
white = [255, 255, 255]  # 予測:1白 解答:1白
red = [255, 0, 0]  # 予測:1白 解答:0黒
blue = [0, 204, 255]  # 予測:0黒 解答:1白


# ========== only train ==========
def average_gradients(tower_grads, rate=1):
    average_grads = []
    for grad in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   (grad0_gpu0, ... , grad0_gpuN)
        grad = tf.reduce_mean(grad, 0)
        # rate = tf.cast(data_rate, dtype=tf.float32)
        grad = tf.multiply(grad, rate)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        average_grads.append(grad)
    return average_grads


def rate_multiply(grads, rate):
    out_grads = []
    for grad in grads:
        grad = tf.multiply(grad, rate)
        out_grads.append(grad)
    return out_grads


# ========== only save ==========
# stock = [ [total_step, train_accuracy, test_accuracy, validation_accuracy], [...], ...]
def log_write(writer, stock, scalar_name, image_tag, check_img_path, path_cls):
    for data in stock:
        step = data[0]
        image_list = [[] for i in range(len(image_tag))]
        for pi in range(len(check_img_path)):
            target_path, answer_path, tag = check_img_path[pi]
            read_path = make_step_img_path(step=step, img_path=target_path, path_cls=path_cls)
            img = image_from_path(read_path)
            for ti in range(len(image_tag)):
                image_list[ti].append(img) if tag == image_tag[ti] else None
        with writer.as_default():  # ログに書き込むデータはここに記載するらしい？ https://www.tensorflow.org/api_docs/python/tf/summary
            for di in range(len(data)-1):
                tf.summary.scalar(name=scalar_name[di], data=data[di+1], step=step)
            for ti in range(len(image_tag)):
                tf.summary.image(name=image_tag[ti], data=image_list[ti], step=step, max_outputs=10) if image_list[ti] else None


def accuracy_log_write(writer, stock, scalar_name):
    for data in stock:
        step = data[0]
        with writer.as_default():  # ログに書き込むデータはここに記載するらしい？ https://www.tensorflow.org/api_docs/python/tf/summary
            for di in range(len(data)-1):
                tf.summary.scalar(name=scalar_name[di], data=data[di+1], step=step)


# csvに書き込み
def write_csv(path_cls, filename, datalist):
    with open(path_cls.make_csv_path(filename=filename), 'a') as f:
        for data in datalist:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(data)


# ========== 評価系 ==========
# 評価
def evaluate(net_cls, out, ans, region=[], batch_data_n=1):
    out_binary = net_cls.binary_from_data(out, label='output')
    ans_binary = net_cls.binary_from_data(ans, label='target')
    correct_prediction = tf.equal(out_binary, ans_binary)
    confusion_matrix = evaluate_confusion_matrix(out_binary, ans_binary)
    if not region:
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32)), confusion_matrix
    region_result = []
    max_region = region[-1]
    cut_o = int((max_region - region[0]) / 2)
    cut_out = out_binary[:, cut_o:max_region-cut_o, cut_o:max_region-cut_o, :]
    cut_ans = ans_binary[:, cut_o:max_region-cut_o, cut_o:max_region-cut_o, :]
    region_result.append(tf.reduce_mean(tf.cast(tf.equal(cut_out, cut_ans), tf.float32)))
    # 範囲毎に評価を繰り返す
    for i in range(len(region)-1):
        cut_o = int((max_region - region[i+1]) / 2)
        cut_i = int((max_region - region[i]) / 2)
        cut_out = out_binary[:, cut_o:max_region-cut_o, cut_o:max_region-cut_o]
        cut_ans = ans_binary[:, cut_o:max_region-cut_o, cut_o:max_region-cut_o]
        fil = np.ones((max_region-cut_o*2, max_region-cut_o*2))
        fil[cut_i:max_region-cut_i, cut_i:max_region-cut_i] = 0
        fil = tf.reshape(fil, [1, max_region-cut_o*2, max_region-cut_o*2, 1])
        fil = tf.tile(fil, [batch_data_n, 1, 1, 1])
        tf_fil = tf.where(fil)
        cut_out = tf.gather_nd(cut_out, tf_fil)
        cut_ans = tf.gather_nd(cut_ans, tf_fil)
        region_result.append(tf.reduce_mean(tf.cast(tf.equal(cut_out, cut_ans), tf.float32)))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32)), confusion_matrix, region_result

def mask_bound30(weight, distance):
    dis_w = tf.less_equal(distance, 30) # <=
    bound_w = tf.equal(weight, 0)
    out = tf.equal(dis_w, bound_w)
    out = tf.cast(out, tf.float32)
    return out

def weight_accracy(out, ans, weight, distance, net_cls):
    out_binary = net_cls.binary_from_data(out, label='output')
    ans_binary = net_cls.binary_from_data(ans, label='target')
    mask = mask_bound30(weight, distance)
    correct_prediction = tf.cast(tf.equal(out_binary, ans_binary), tf.float32) * mask
    weight_acc = 0.
    target_count = 0
    for i in range(tf.shape(correct_prediction)[0]):
        if tf.reduce_sum(mask[i, :, :, :]) != 0:
            weight_acc += tf.reduce_sum(correct_prediction[i, :, :, :]) / tf.reduce_sum(mask[i, :, :, :])
            target_count += 1
    return weight_acc, target_count


def weight_only_accracy(out, ans, weight, distance, net_cls):
    out_binary = net_cls.binary_from_data(out, label='output')
    ans_binary = net_cls.binary_from_data(ans, label='target')
    mask = net_cls.net_weight_mask(weight, distance)
    correct_prediction = tf.cast(tf.equal(out_binary, ans_binary), tf.float32) * mask
    weight_acc = 0.
    target_count = 0
    for i in range(tf.shape(correct_prediction)[0]):
        if tf.reduce_sum(mask[i, :, :, :]) != 0:
            weight_acc += tf.reduce_sum(correct_prediction[i, :, :, :]) / tf.reduce_sum(mask[i, :, :, :])
            target_count += 1
    return weight_acc, target_count


def boundary_removal_accracy(out, ans, net_cls, bound):
    out_binary = net_cls.binary_from_data(out, label='output')
    ans_binary = net_cls.binary_from_data(ans, label='target')
    boundary_removal = tf.equal(bound, 0)
    boundary_removal = tf.cast(boundary_removal, tf.float32)
    boundary_removal_acc = 0.
    target_count = 0
    correct_prediction = tf.cast(tf.equal(out_binary, ans_binary), tf.float32) * boundary_removal
    for i in range(tf.shape(boundary_removal)[0]):
        if tf.reduce_sum(boundary_removal[i, :, :, :]) != 0:
            target_count += 1
            boundary_removal_acc += tf.reduce_sum(correct_prediction[i, :, :, :]) / tf.reduce_sum(boundary_removal[i, :, :, :])
    return boundary_removal_acc, target_count

def boundary_accracy(out, ans, net_cls, bound):
    out_binary = net_cls.binary_from_data(out, label='output')
    ans_binary = net_cls.binary_from_data(ans, label='target')
    boundary = tf.equal(bound, 1)
    boundary = tf.cast(boundary, tf.float32)
    boundary_removal_acc = 0.
    target_count = 0
    correct_prediction = tf.cast(tf.equal(out_binary, ans_binary), tf.float32) * boundary
    for i in range(tf.shape(boundary)[0]):
        if tf.reduce_sum(boundary[i, :, :, :]) != 0:
            target_count += 1
            boundary_removal_acc += tf.reduce_sum(correct_prediction[i, :, :, :]) / tf.reduce_sum(boundary[i, :, :, :])
    return boundary_removal_acc, target_count
    # correct_prediction = tf.cast(tf.equal(out_binary, ans_binary), tf.float32) * boundary
    # if tf.reduce_sum(boundary) == 0:
    #     return 0
    # boundary_acc = tf.reduce_sum(correct_prediction) / tf.reduce_sum(boundary)
    # return boundary_acc





    # 混同行列の評価
def evaluate_confusion_matrix(output, label):
    one = tf.constant(True)
    zero = tf.constant(False)
    TP = tf.reduce_sum( tf.cast( tf.logical_and( tf.equal(output, one), tf.equal(label, one) ), tf.float32 ) )
    FP = tf.reduce_sum( tf.cast( tf.logical_and( tf.equal(output, one), tf.equal(label, zero) ), tf.float32 ) )
    FN = tf.reduce_sum( tf.cast( tf.logical_and( tf.equal(output, zero), tf.equal(label, one) ), tf.float32 ) )
    TN = tf.reduce_sum( tf.cast( tf.logical_and( tf.equal(output, zero), tf.equal(label, zero) ), tf.float32 ) )
    # TP = np.sum((output==1) * (label==1))
    # FP = np.sum((output==1) * (label==0))
    # FN = np.sum((output==0) * (label==1))
    # TN = np.sum((output==0) * (label==0))
    return [TP, FP, FN, TN]





# 評価画像の出力
def evalute_img(net_cls, out, ans):
    out_binary = net_cls.binary_from_data(out, label='output')
    out_shape = out_binary.numpy().shape
    result_img = np.zeros([out_shape[0], out_shape[1], 3])
    out_3d = out_binary.numpy().repeat(3).reshape(out_shape[0], out_shape[1], 3)
    ans_binary = net_cls.binary_from_data(ans, label='target')
    if (tf.rank(ans_binary)) == 2:
        ans_3d = ans_binary.numpy().repeat(3).reshape(out_shape[0], out_shape[1], 3)
    else:
        ans_3d = ans_binary.numpy()
    result_img += (out_3d == 0) * (ans_3d == 0) * black  # 黒 正解
    result_img += (out_3d == 1) * (ans_3d == 1) * white  # 白 正解
    result_img += (out_3d == 1) * (ans_3d == 0) * red  # 赤 黒欠け
    result_img += (out_3d == 0) * (ans_3d == 1) * blue  # 青 黒余分
    return result_img.astype(np.uint8)


# 評価画像の出力
def weight_img(net_cls, out, ans, c_weight):
    out_binary = net_cls.binary_from_data(out, label='output')
    out_shape = out_binary.numpy().shape
    result_img = np.zeros([out_shape[0], out_shape[1], 3])
    out_3d = out_binary.numpy().repeat(3).reshape(out_shape[0], out_shape[1], 3)
    ans_binary = net_cls.binary_from_data(ans, label='target')
    w_3d = c_weight.numpy().repeat(3).reshape(out_shape[0], out_shape[1], 3)
    if (tf.rank(ans_binary)) == 2:
        ans_3d = ans_binary.numpy().repeat(3).reshape(out_shape[0], out_shape[1], 3)
    else:
        ans_3d = ans_binary.numpy()
    result_img += (w_3d == 0) * (ans_3d == 0) * black  # 黒 正解
    result_img += (ans_3d == 1) * (out_3d == 1) * (w_3d == 0) * white  # 白 正解
    result_img += (w_3d == 1) * red  # 赤 黒欠け
    result_img += (out_3d == 0) * (ans_3d == 1) * (w_3d == 0) * blue
    result_img += (out_3d == 1) * (ans_3d == 0) * (w_3d == 0) * blue
    result_img = tf.cast(result_img, tf.uint8)
    return result_img


def img_check(step, net_cls, path_cls, check_img_path):
    generator = net_cls.get_generator()
    for i in range(len(check_img_path)):
        target_path, answer_path, tag = check_img_path[i]
        # Generatorによる画像生成
        sample_data = normalize_netdata_from_path(net_cls=net_cls, img_path=target_path)
        answer_data = image_from_path(answer_path, ch=1)
        answer_data = target_cut_padding(target=answer_data, padding=net_cls.get_padding(),
                                         batch_shape=False)
        answer_data = net_cls.netdata_from_img(answer_data)
        gen_output = generator(sample_data[np.newaxis, :, :, :], training=False)
        # gen_loss, err_count, c_weight = net_cls.evaluation_generator_loss(gen_output, answer_data[np.newaxis, :, :, :])
        eva_img = evalute_img(net_cls=net_cls, out=gen_output[0, :, :, :], ans=answer_data)
        encoder_img = tf.image.encode_png(eva_img)
        save_path = make_step_img_path(step=step, img_path=target_path, path_cls=path_cls)
        tf.io.write_file(filename=save_path, contents=encoder_img)

        # w_img = weight_img(net_cls=net_cls, out=gen_output[0, :, :, :], ans=answer_data, c_weight=c_weight)
        # w_encoder_img = tf.image.encode_png(w_img)
        # w_save_path = make_w_step_img_path(step=step, img_path=target_path, path_cls=path_cls)
        # tf.io.write_file(filename=w_save_path, contents=w_encoder_img)

# ========== 変換系 ==========
# ファイルパスからネットデータに正規化
def normalize_netdata_from_path(net_cls, img_path):
    png_bytes = tf.io.read_file(img_path)
    img = tf.image.decode_png(png_bytes, channels=3, dtype=tf.dtypes.uint8, )
    img = tf.cast(img, tf.float32)
    normalize_img = net_cls.netdata_from_img(img)
    return normalize_img


# ファイルパスから画像に
def image_from_path(img_path, ch=3):
    png_bytes = tf.io.read_file(img_path)
    img = tf.image.decode_png(png_bytes, channels=ch, dtype=tf.dtypes.uint8,)
    img = tf.cast(img, tf.float32)
    return img


# 検証用画像パスを生成する
def make_step_img_path(step, img_path, path_cls):
    img_name = os.path.splitext(os.path.basename(img_path))[0]  # ('Sample_X017_Y024', '.png')
    image_root_path = path_cls.get_image_folder_path()
    path = image_root_path + '/' + img_name + '/' + ('step-%s.png' % (step))
    os.makedirs(image_root_path + '/' + img_name, exist_ok=True)
    return path


# 検証用画像パスを生成する
def make_w_step_img_path(step, img_path, path_cls):
    img_name = os.path.splitext(os.path.basename(img_path))[0]  # ('Sample_X017_Y024', '.png')
    image_root_path = path_cls.get_image_folder_path()
    path = image_root_path + '/' + img_name + '_Weight/' + ('step-%s.png' % (step))
    os.makedirs(image_root_path + '/' + img_name, exist_ok=True)
    return path


# paddingサイズ分を切り出す
def color_cut_padding(img, padding):
    if padding == 0:
        return img
    if (tf.rank(img)) == 4:
        img = img[:, padding:-padding, padding:-padding, :]
    elif (tf.rank(img)) == 3:
        img = img[padding:-padding, padding:-padding, :]
    # elif (tf.rank(img)) is not 0:
    #     print('color_cut_padding ERROR!')
    #     print('rank:' + str(tf.rank(img)))
    #     sys.exit()
    return img


# paddingサイズ分を切り出す
def target_cut_padding(target, padding, batch_shape=True):
    if padding == 0:
        return target
    if (tf.rank(target)) == 4:
        target = target[:, padding:-padding, padding:-padding, :]
    elif (tf.rank(target)) == 3:
        if batch_shape:
            target = target[:, padding:-padding, padding:-padding]
        else:
            target = target[padding:-padding, padding:-padding, :]
    elif (tf.rank(target)) == 2:
        target = target[padding:-padding, padding:-padding]
    return target


# ========== DeepOtsu ==========
def ave_color_img(mask, img, mask_repeat=1):
    # 1
    # mask3 = mask.repeat(mask_repeat).reshape(img.shape)
    # mask3 = tf.reshape(tf.repeat(mask, mask_repeat), img.shape)
    mask = tf.cast(mask, tf.float32)
    mask3 = tf.cast(mask, tf.float32)
    pass_img = img * mask3
    count_n = tf.reduce_sum(mask)
    color = []
    if mask_repeat is 1:
        color.append(tf.cast((tf.reduce_sum(pass_img[:, :, :]) / count_n), tf.uint8))
    else:
        for i in range(mask_repeat):
            color.append(tf.cast((tf.reduce_sum(pass_img[:, :, :, 0]) / count_n), tf.uint8))

    # 0
    # z_mask3 = tf.reshape(tf.repeat(mask == 0, mask_repeat), img.shape)
    z_mask3 = tf.cast((mask == 0), tf.float32)
    pass_img = img * mask3
    count_n = tf.reduce_sum(mask)
    z_color = []
    if mask_repeat is 1:
        z_color.append(tf.cast((tf.reduce_sum(pass_img[:, :, :]) / count_n), tf.uint8))
    else:
        for i in range(mask_repeat):
            z_color.append(tf.cast((tf.reduce_sum(pass_img[:, :, :, 0]) / count_n), tf.uint8))

    return tf.add((mask3 * color), (z_mask3 * z_color))


# ========== save model ==========
# モデル全体を１つのHDF5ファイルに保存します。***.h5
# https://www.tensorflow.org/tutorials/keras/save_and_load?hl=ja
def save_best_generator_model(net_cls, path_cls, path):
    path_list = path_cls.search_best_path(filename=path_cls.ckpt_step_name)
    if path_list:
        root, ext = os.path.splitext(path_list[0])
        net_cls.ckpt_restore(path=root)
    net_cls.get_generator().save(path)


# ========== val ==========
def get_black():
    return black


# ========== write log ==========
# def accuracy_csv2log(csv_path, out_path, scalar_name):
#     data = csv.reader()
#     writer = tf.summary.create_file_writer(logdir=out_path)
#     log_write(writer=writer, stock=data,
#               scalar_name=scalar_name, image_tag=summary_image_tag,
#               check_img_path=self.check_img_path, path_cls=self.path_cls)

def exe_manage(msg, csv_path, workname, label, task):
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['', 'A-train','A-eva','B-train','B-eva','C-train','C-eva','D-train','D-eva','E-train','E-train','F-train','F-eva'])
    print('----- open manage csv file -----')
    df = pd.read_csv(csv_path, index_col=0)
    dt_now = datetime.now()
    if not workname in df.index:
        df.loc[workname] = None
    head = label + '-' + task
    df[head][workname] = msg + ' ' + str(dt_now)
    df.to_csv(csv_path)
    print('----- close manage csv file -----')
    return



def get_white():
    return white


def get_red():
    return red


def get_blue():
    return blue


def check_eveluate(path, file_count, csv_name='_.csv'):
    index = 0
    total_division = 0
    total_short = 0
    total_short_area = 0
    total_break = 0
    total_break_area = 0
    total_break_same = 0
    total_break_same_area = 0
    if os.path.exists(path +'/'+csv_name):
        os.remove(path +'/'+csv_name)
    if not os.path.exists(path + '/Evaluate.txt'):
        return None
    eva_count = sum([s.count('教') for s in open(path + '/Evaluate.txt')])
    if eva_count is not file_count:
        return None
    with open(path + '/Evaluate.txt', 'r') as fp:
        with open(path +'/'+csv_name, 'w') as csv_file:
            
            writer = csv.writer(csv_file, lineterminator='\n')
            file = fp.readlines()
            for img_number in range(file_count):
                vals = [img_number]

                while True:
                    Str = file[index]
                    if Str[0] == '教':
                        break
                    else:
                        index += 1

                # 教師画像 パターン数
                Str = file[index]
                start = Str.find('数') + 2
                end = Str.find('個')
                val = Str[start: end]
                vals.append(val)
                index += 1

                # 推論2値画像 パターン数
                Str = file[index]
                start = Str.find('数') + 2
                end = Str.find('個')
                val = Str[start: end]
                vals.append(val)
                index += 1

                # 1対1 対応パターン数
                Str = file[index]
                start = Str.find('数') + 2
                end = Str.find('個')
                val = Str[start: end]
                vals.append(val)

                start = Str.find('個') + 2
                end = Str.find('%')
                val = Str[start: end]
                vals.append(val)
                index += 1

                # 同一電位パターン分断数
                Str = file[index]
                start = Str.find('数') + 2
                end = len(Str) - 1
                val = Str[start: end]
                vals.append(val)
                total_division += int(Str[start: end])
                index += 1

                # Short positions
                Str = file[index]
                start = Str.find(':') + 1
                end = len(Str) - 1
                val = Str[start: end]
                vals.append(val)
                total_short += int(Str[start: end])
                index += 1

                # Short positions area
                val = 0
                while True:
                    Str = file[index]
                    if '{' not in Str:
                        break
                    else:
                        val += 1
                        index += 1
                vals.append(val)
                total_short_area += val

                # Break positions
                Str = file[index]
                start = Str.find(':') + 1
                end = len(Str) - 1
                val = Str[start: end]
                vals.append(val)
                total_break += int(Str[start: end])
                index += 1

                # Break positions Point
                val = 0
                while True:
                    Str = file[index]
                    if '{' not in Str:
                        break
                    else:
                        val += 1
                        index += 1
                vals.append(val)
                total_break_area += val

                # Break positions of same potential pattern
                Str = file[index]
                start = Str.find(':') + 1
                end = len(Str) - 1
                val = Str[start: end]
                vals.append(val)
                total_break_same += int(Str[start: end])
                index += 1

                # Break positions of same potential pattern area
                # index += 2
                val = 0
                while True:
                    Str = file[index]
                    if '{' not in Str:
                        break
                    else:
                        val += 1
                        index += 1
                vals.append(val)
                total_break_same_area += val

                writer.writerow(vals)
               # === 繰り返しここまで ===
            writer.writerow(
                ['TOTAL', '', '', '', '', str(total_division), str(total_short), str(total_short_area),
                str(total_break), str(total_break_area), str(total_break_same), str(total_break_same_area)])

            # if not os.path.exists(path + '/'+'PatternResult.csv'):
            #     with open(path+ '/'+'PatternResult.csv', 'w') as total_file:
            #         writer_total = csv.writer(total_file, lineterminator='\n')
            #         writer_total.writerow(['total_division', 'total_short_area',
            #                             'total_break_area', 'total_break_same_area'])

            # with open(path + '/'+'PatternResult.csv', 'a') as total_file:
            # writer_total = csv.writer(total_file, lineterminator='\n')
            writer.writerow(['total_division', 'total_short_area',
                                    'total_break_area', 'total_break_same_area'])
            writer.writerow(
                [total_division, total_short_area, total_break_area, total_break_same_area])
            writer.writerow(['short', 'break+break_same', 'total'])
            writer.writerow([total_short_area, total_break_area+total_break_same_area,
                                total_short_area + total_break_area + total_break_same_area])

    return [total_division, total_short, total_short_area, total_break, total_break_area, total_break_same, total_break_same_area]
# -*- coding: utf-8 -*-

import glob
import tensorflow as tf
import time
import math
import ShareNetFunc as nfunc
import tensorflow_addons as tfa
import numpy as np

# - Discriminator_Loss
#     - 弁別子損失関数は2つの入力を取ります。実画像、生成画像
#     - real_lossは、実画像と1つの配列のシグモイドクロスエントロピー損失です（これらは実画像であるため）
#     - generated_lossは、生成された画像とゼロの配列のシグモイドクロスエントロピー損失です（これらは偽の画像であるため）
#     - 次に、total_lossはreal_lossとgenerated_lossの合計です
# - Generator_Loss
#     - これは、生成された画像と1の配列のシグモイドクロスエントロピー損失です。
#     - 紙はまた、生成された画像とターゲット画像とのMAEであるL1損失を（絶対平均誤差）を含みます。
#     - これにより、生成された画像が構造的にターゲット画像に似たものになります。
#     - 総発電機損失= gan_loss + LAMBDA * l1_lossを計算する式。ここでLAMBDA = 100です。この値は、論文の著者によって決定されました。


class NetManager:
    def __init__(self,
                 loss_object=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                 gen_optimizer=tf.keras.optimizers.Adam(0.0001),
                 # dis_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
                 output_channel=1,
                 lambda_val=100,):
        self.output_channel = output_channel
        self.gen = self._Generator()
        self.dis = self._Discriminator()
        self.loss_object = loss_object
        self.gen_optimizer = gen_optimizer
        # self.dis_optimizer = dis_optimizer
        self.lambda_val = lambda_val
        self.best_test_accuracy = tf.Variable(initial_value=0.0,
                                              trainable=False,
                                              dtype=tf.float32,
                                              name='best_test_accuracy')
        self.best_validation_accuracy = tf.Variable(initial_value=0.0,
                                                    trainable=False,
                                                    dtype=tf.float32,
                                                    name='best_validation_accuracy')
        self.step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name='step')
        self.epoch = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name='epoch')
        self.study_time = tf.Variable(initial_value=0.0, trainable=False, dtype=tf.float32, name='study_time')
        self.total_time = tf.Variable(initial_value=0.0, trainable=False, dtype=tf.float32, name='total_time')
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.gen_optimizer,
            # discriminator_optimizer=self.dis_optimizer,
            generator=self.gen,
            discriminator=self.dis,
            best_validation_accuracy=self.best_validation_accuracy,
            best_test_accuracy=self.best_test_accuracy,
            step=self.step,
            epoch=self.epoch,
            study_time=self.study_time,
            total_time=self.total_time,
            )
        self.time_basis = 54000  # timeの基準
        self.day_time = 86400  # 1日
        self.padding = 0

# ************ クラス外呼び出し関数 ************

    # ===== ckpt関係 =====
    def ckpt_restore(self, path):
        self.checkpoint.restore(path)

    def get_str_study_time(self):
        ms_time, s_time = math.modf(self.study_time.numpy() + self.time_basis)  # ミニセカンド セカンド
        day, times = divmod(s_time, self.day_time)  # 日数と時間に
        day = int(day)
        step_times = time.strptime(time.ctime(times))
        str_time = str(day) + ':' + time.strftime("%H:%M:%S", step_times) + str(ms_time)[1:]
        return str_time

    def get_str_total_time(self):
        ms_time, s_time = math.modf(self.total_time.numpy() + self.time_basis)  # ミニセカンド セカンド
        day, times = divmod(s_time, self.day_time)  # 日数と時間に
        day = int(day)
        step_times = time.strptime(time.ctime(times))
        str_time = str(day) + ':' + time.strftime("%H:%M:%S", step_times) + str(ms_time)[1:]
        return str_time

    def get_epoch(self):
        return self.epoch.numpy()

    def get_step(self):
        return self.step.numpy()

    def add_study_time(self, proc_time):
        self.study_time.assign(self.study_time + proc_time)

    def add_total_time(self, proc_time):
        self.total_time.assign(self.total_time + proc_time)

    def update_check_best_validation_accuracy(self, accuracy):
        if accuracy > self.best_validation_accuracy:
            self.best_validation_accuracy = accuracy
            return True
        return False

    def update_check_best_test_accuracy(self, accuracy):
        if accuracy > self.best_test_accuracy:
            self.best_test_accuracy = accuracy
            return True
        return False

    def get_checkpoint(self):
        return self.checkpoint

    # ===== ネットワーク関係 =====
    def get_padding(self):
        return self.padding

    def get_generator(self):
        return self.gen

    def get_discriminator(self):
        return self.dis

    def set_ckpt_val(self, step_val=None, epoch_val=None):
        if step_val is not None:
            self.step.assign(step_val)
        if epoch_val is not None:
            self.epoch.assign(epoch_val)

    def get_generator_optimizer(self):
        return self.gen_optimizer

    # def get_discriminator_optimizer(self):
    #     return self.dis_optimizer

    def erosion(self, img, kernel_size=3):
        kernel = np.ones((kernel_size, kernel_size, 1), np.float32)
        img = tf.cast(img, tf.float32)
        img = tf.nn.erosion2d(value=img, filters=kernel, strides=[1, 1, 1, 1], padding='SAME',
                              data_format='NHWC', dilations=[1, 1, 1, 1]) + 1  # 白収縮 何故か出力が-1される
        img = tf.cast(tf.greater_equal(img, 127), tf.uint8) * 255
        return img

    def dilation(self, img, kernel_size=3):
        kernel = np.ones((kernel_size, kernel_size, 1), np.float32)
        img = tf.cast(img, tf.float32)
        img = tf.nn.dilation2d(input=img, filters=kernel, strides=[1, 1, 1, 1], padding='SAME',
                               data_format='NHWC', dilations=[1, 1, 1, 1]) - 1  # 白拡大 何故か出力が+1される
        img = tf.cast(tf.greater_equal(img, 127), tf.uint8) * 255
        return img

    def connect_weight(self, out, ans, kernel_size=5, eva_count=False):
        start_time = time.time()
        out = tf.cast(out, tf.float32)
        ans = tf.cast(ans, tf.float32)
        weight = tf.zeros_like(out)
        count = 0
        for t in [1, 0]:
            out_img = out if t == 0 else tf.cast(tf.equal(out, 0), tf.float32) * 255  # d4
            ans_img = ans if t == 0 else tf.cast(tf.equal(ans, 0), tf.float32) * 255  # d4
            r_ans_img = tf.cast(tf.equal(ans, 0), tf.float32) * 255 if t == 0 else ans  # d4
            ans_label = tfa.image.connected_components(ans_img[:, :, :, 0])  # d3
            r_ans_label = tfa.image.connected_components(r_ans_img[:, :, :, 0])  # d3
            ans_label_n = tf.math.reduce_max(ans_label)  # d1
            for i in range(ans_label_n):
                label_img = tf.cast(tf.equal(ans_label, i + 1), tf.float32) * 255  # d3
                label_bin = tf.cast(tf.greater_equal(label_img, 127), tf.float32)  # d3
                label_bin = tf.expand_dims(label_bin, -1)  # d4
                out_cut_img = out_img * label_bin
                out_cut_label = tfa.image.connected_components(out_cut_img[:, :, :, 0])
                # ----- エラー領域を抽出 -----
                error_img = tf.cast(tf.equal(out_img, 0), tf.float32) * label_bin * 255  # d3 ネガポジ反転 0:正解 1:エラー
                # ----- 付近のエラー領域を結合する -----
                error_img = self.dilation(error_img, kernel_size)
                error_img = self.erosion(error_img, kernel_size)
                # ----- 汚れのラベリング -----
                error_label = tfa.image.connected_components(error_img[:, :, :, 0])  # カラーチャンネルは除去する必要あり
                error_label_n = tf.math.reduce_max(error_label)
                # ----- 個々の汚れが断線・短絡に影響しているか判定 -----
                for er_i in range(error_label_n):
                    error_label_img = tf.cast(tf.equal(error_label, er_i + 1), tf.float32) * 255  # d2　汚れの領域の指定
                    error_label_dil = tf.expand_dims(error_label_img, -1)  # d3
                    error_label_dil = self.dilation(error_label_dil, 3)  # d3
                    error_label_dil = error_label_dil[:, :, :, 0]  # d3
                    error_label_bin = tf.cast(tf.greater_equal(error_label_dil, 127), tf.float32)  # d2
                    # ----- 断線判定の為、汚れ周辺の出力画像ラベル取得 -----
                    dec_label = out_cut_label * tf.cast(error_label_bin, tf.int32)
                    dec_label_unique, _ = tf.unique(tf.reshape(dec_label, [-1]))
                    dec_label_unique_nonzero = dec_label_unique[dec_label_unique != 0]  # 背景ラベルは除去
                    # ----- 短絡判定の為、汚れ周辺の解答画像ラベル取得 -----
                    r_dec_label = r_ans_label * tf.cast(error_label_bin, tf.int32)
                    r_dec_label_unique, _ = tf.unique(tf.reshape(r_dec_label, [-1]))
                    r_dec_label_unique_nonzero = r_dec_label_unique[r_dec_label_unique != 0]  # 背景ラベルは除去
                    # ----- 浮島判定の為、汚れ周辺の背景画像ラベル取得 -----
                    float_island_flug = error_label_bin * tf.cast(tf.equal(label_bin[:, :, :, 0], 0),
                                                                  tf.float32)  # 膨張したエラー領域と背景領域を*して、合計ゼロなら背景を含んでいない浮島
                    float_island_flug = tf.reduce_sum(float_island_flug)
                    # ----- 断線・短絡判定 -----
                    if len(dec_label_unique_nonzero) >= 2 or len(
                            r_dec_label_unique_nonzero) >= 2 or float_island_flug == 0:  # Trueなら断線・短絡に関与していると考えられる
                        e_label = tf.cast(tf.equal(error_label, er_i + 1), tf.float32)
                        e_label = tf.expand_dims(e_label, -1)  # d3
                        weight += label_bin * e_label
                        count += 1

        end_time = time.time()
        ms_time, s_time = math.modf((end_time - start_time) + self.time_basis)  # ミニセカンド セカンド
        day, times = divmod(s_time, self.day_time)  # 日数と時間に
        day = int(day)
        step_times = time.strptime(time.ctime(times))
        str_time = str(day) + ':' + time.strftime("%H:%M:%S", step_times) + str(ms_time)[1:]
        print('label time is : ' + str_time)

        if eva_count:
            return tf.cast(weight, tf.float32), tf.cast(count, tf.float32)
        return tf.cast(weight, tf.float32)

    # def erosion(self, img, kernel_size=3):
    #     kernel = np.ones((kernel_size, kernel_size, 1), np.float32)
    #     img = tf.expand_dims(img, axis=0)  # バッチ次元付与(膨張、収縮が(NHEC)でしか対応していない)
    #     img = tf.cast(img, tf.float32)
    #     img = tf.nn.erosion2d(value=img, filters=kernel, strides=[1, 1, 1, 1], padding='SAME',
    #                           data_format='NHWC', dilations=[1, 1, 1, 1]) + 1  # 白収縮 何故か出力が-1される
    #     img = tf.cast(tf.greater_equal(img, 127), tf.uint8) * 255
    #     img = img[0, :, :, :]
    #     return img
    #
    # def dilation(self, img, kernel_size=3):
    #     kernel = np.ones((kernel_size, kernel_size, 1), np.float32)
    #     img = tf.expand_dims(img, axis=0)  # バッチ次元付与(膨張、収縮が(NHEC)でしか対応していない)
    #     img = tf.cast(img, tf.float32)
    #     img = tf.nn.dilation2d(input=img, filters=kernel, strides=[1, 1, 1, 1], padding='SAME',
    #                            data_format='NHWC', dilations=[1, 1, 1, 1]) - 1  # 白拡大 何故か出力が+1される
    #     img = tf.cast(tf.greater_equal(img, 127), tf.uint8) * 255
    #     img = img[0, :, :, :]
    #     return img
    #
    # def connect_weight(self, out, ans, kernel_size=5, eva_count=False):
    #     out = tf.cast(out, tf.float32)
    #     ans = tf.cast(ans, tf.float32)
    #     weight_list = []
    #     count = 0
    #     for batch in range(tf.shape(out)[0]):
    #         weight = tf.zeros_like(out[batch, :, :, :])
    #         for t in [1, 0]:
    #             out_img = out[batch, :, :, :] if t == 0 else tf.cast(tf.equal(out[batch, :, :, :], 0),
    #                                                                  tf.float32) * 255  # d3
    #             ans_img = ans[batch, :, :, :] if t == 0 else tf.cast(tf.equal(ans[batch, :, :, :], 0),
    #                                                                  tf.float32) * 255  # d3
    #             r_ans_img = tf.cast(tf.equal(ans[batch, :, :, :], 0), tf.float32) * 255 if t == 0 else ans[batch, :, :, :]  # d3
    #             ans_label = tfa.image.connected_components(ans_img[:, :, 0])  # d2
    #             r_ans_label = tfa.image.connected_components(r_ans_img[:, :, 0])  # d2
    #             ans_label_n = tf.math.reduce_max(ans_label)
    #             for i in range(ans_label_n):
    #                 label_img = tf.cast(tf.equal(ans_label, i + 1), tf.float32) * 255  # d2
    #                 label_bin = tf.cast(tf.greater_equal(label_img, 127), tf.float32)  # d2
    #                 label_bin = tf.expand_dims(label_bin, -1)  # d3
    #                 out_cut_img = out_img * label_bin
    #                 out_cut_label = tfa.image.connected_components(out_cut_img[:, :, 0])
    #                 # ----- エラー領域を抽出 -----
    #                 error_img = tf.cast(tf.equal(out_img, 0), tf.float32) * label_bin * 255  # d3 ネガポジ反転 0:正解 1:エラー
    #                 # ----- 付近のエラー領域を結合する -----
    #                 error_img = self.dilation(error_img, kernel_size)
    #                 error_img = self.erosion(error_img, kernel_size)
    #                 # ----- 汚れのラベリング -----
    #                 error_label = tfa.image.connected_components(error_img[:, :, 0])  # カラーチャンネルは除去する必要あり
    #                 error_label_n = tf.math.reduce_max(error_label)
    #                 # ----- 個々の汚れが断線・短絡に影響しているか判定 -----
    #                 for er_i in range(error_label_n):
    #                     error_label_img = tf.cast(tf.equal(error_label, er_i + 1), tf.float32) * 255  # d2　汚れの領域の指定
    #                     error_label_dil = tf.expand_dims(error_label_img, -1)  # d3
    #                     error_label_dil = self.dilation(error_label_dil, 3)  # d3
    #                     error_label_dil = error_label_dil[:, :, 0]  # d3
    #                     error_label_bin = tf.cast(tf.greater_equal(error_label_dil, 127), tf.float32)  # d2
    #                     # ----- 断線判定の為、汚れ周辺の出力画像ラベル取得 -----
    #                     dec_label = out_cut_label * tf.cast(error_label_bin, tf.int32)
    #                     dec_label_unique, _ = tf.unique(tf.reshape(dec_label, [-1]))
    #                     dec_label_unique_nonzero = dec_label_unique[dec_label_unique != 0]  # 背景ラベルは除去
    #                     # ----- 短絡判定の為、汚れ周辺の解答画像ラベル取得 -----
    #                     r_dec_label = r_ans_label * tf.cast(error_label_bin, tf.int32)
    #                     r_dec_label_unique, _ = tf.unique(tf.reshape(r_dec_label, [-1]))
    #                     r_dec_label_unique_nonzero = r_dec_label_unique[r_dec_label_unique != 0]  # 背景ラベルは除去
    #                     # ----- 浮島判定の為、汚れ周辺の背景画像ラベル取得 -----
    #                     float_island_flug = error_label_bin * tf.cast(tf.equal(label_bin[:, :, 0], 0),
    #                                                                   tf.float32)  # 膨張したエラー領域と背景領域を*して、合計ゼロなら背景を含んでいない浮島
    #                     float_island_flug = tf.reduce_sum(float_island_flug)
    #                     # ----- 断線・短絡判定 -----
    #                     if len(dec_label_unique_nonzero) >= 2 or len(
    #                             r_dec_label_unique_nonzero) >= 2 or float_island_flug == 0:  # Trueなら断線・短絡に関与していると考えられる
    #                         e_label = tf.cast(tf.equal(error_label, er_i + 1), tf.float32)
    #                         e_label = tf.expand_dims(e_label, -1)  # d3
    #                         weight += label_bin * e_label
    #                         count += 1
    #         weight_list.append(weight)
    #     if eva_count:
    #         return tf.cast(weight_list, tf.float32), tf.cast(count, tf.float32)
    #     return tf.cast(weight_list, tf.float32)

    def net_weight(self, weight):
        return tf.ones_like(weight)

    def generator_loss(self, gen_output, target, weight=None):
        # gen_bin = self.binary_from_data(gen_output, label='output')
        # gen_bin = tf.cast(gen_bin, tf.float32)
        # c_weight = tf.numpy_function(self.connect_weight, inp=[gen_bin, target], Tout=tf.float32)
        # c_weight = self.connect_weight(gen_bin, target)
        # c_weight = self.net_weight(c_weight)
        gen_loss = self.loss_object(y_true=target, y_pred=gen_output, sample_weight=None)
        return gen_loss

    def evaluation_generator_loss(self, gen_output, target, weight=None):
        # gen_bin = self.binary_from_data(gen_output, label='output')
        # gen_bin = tf.cast(gen_bin, tf.float32)
        # c_weight, err_count = self.connect_weight(gen_bin, target, 5, True)
        # c_weight, err_count = tf.numpy_function(self.connect_weight, inp=[gen_bin, target, 5, True], Tout=[tf.float32, tf.float32])
        # net_c_weight = self.net_weight(c_weight)
        gen_loss = self.loss_object(y_true=target, y_pred=gen_output, sample_weight=None)
        return gen_loss, tf.constant(0.0), tf.constant(0.0)


    # def discriminator_loss(self, disc_real_output, disc_generated_output):
    #     # 本物に対して1(本物)と判定できたか
    #     real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
    #     # 偽物に対して0(偽物)と判定できたか
    #     generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    #     # lossの合計
    #     total_disc_loss = real_loss + generated_loss
    #     return total_disc_loss

    # # ===== model save =====
    # def save_generator(self, path):
    #     self.gen.save(path)

    # ===== 値変換系 =====
    def binary_from_img(self, data):
        return tf.greater_equal(data, 255)

    def binary_from_data(self, data, label=None):
        if label == 'target':
            return tf.greater_equal(data, 0.5)
        if label == 'output':
            return tf.greater_equal(data, 0.)
        # else:
        #     return tf.greater_equal(data, 0.5)

    def img_from_netdata(self, data):
        return data * 255

    def netdata_from_img(self, data):
        return data / 255
    # def binary_from_img(self, data):
    #     return tf.greater_equal(data, 127.5)
    #
    # def binary_from_data(self, data):
    #     return tf.greater_equal(data, 0)
    #
    # def img_from_netdata(self, data):
    #     return (data + 1) * 127.5
    #
    # def netdata_from_img(self, data):
    #     return (data / 127.5) - 1

    # ==============================================================================================================
    # ========== train関数 =========================================================================================
    # ==============================================================================================================
    @tf.function  # 関数をグラフモードで実行 https://www.tensorflow.org/guide/autograph?hl=ja
    def multi_train_step(self, ds, device_list):
        generator_gradients_list = []
        discriminator_gradients_list = []
        for gpu_index in device_list:
            with tf.device('/gpu:%d' % gpu_index): # gpu単位の処理
                # tf.GradientTape()以下に勾配算出対象の計算を行う https://qiita.com/propella/items/5b2182b3d6a13d20fefd
                with tf.GradientTape() as gen_tape:
                    # input_image, target, weight = next(ds_iter)
                    input_image, target, weight = ds
                    target = nfunc.target_cut_padding(target=target, padding=self.get_padding())
                    # Generatorによる画像生成
                    generator = self.get_generator()
                    gen_output = generator(input_image, training=True)
                    # Discriminatorによる判定
                    # discriminator = self.get_discriminator()
                    # disc_real_output = discriminator([input_image, target], training=True)
                    # disc_generated_output = discriminator([input_image, gen_output], training=True)
                    # loss算出
                    gen_loss = self.generator_loss(gen_output, target, weight)
                    err_count = 0
                    c_weight = tf.ones_like(weight)
                    # gen_loss, err_count, c_weight = self.evaluation_generator_loss(gen_output, target, weight)
                    # disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

                # 勾配算出 trainable_variables:訓練可能な変数
                generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
                # discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
                # 後で平均を取る為に保存
                generator_gradients_list.append(generator_gradients)
                # discriminator_gradients_list.append(discriminator_gradients)
                # gpu単位の処理ここまで
        # with tf.device('/gpu:%d' % device_list[0]):  # gpu単位の処理
        generator = self.get_generator()
        discriminator = self.get_discriminator()
        # 勾配の平均、怪しい
        generator_gradients_average = nfunc.average_gradients(generator_gradients_list)
        # discriminator_gradients_average = nfunc.average_gradients(discriminator_gradients_list)
        # 勾配の適用
        self.get_generator_optimizer().apply_gradients(zip(generator_gradients_average, generator.trainable_variables))
        # self.get_discriminator_optimizer().apply_gradients(zip(discriminator_gradients_average, discriminator.trainable_variables))
        return gen_output, err_count, c_weight

    @tf.function  # 関数をグラフモードで実行 https://www.tensorflow.org/guide/autograph?hl=ja
    def train_step(self, ds, device_list, rate=1):
        with tf.device('/gpu:%d' % device_list[0]): # gpu単位の処理
            # tf.GradientTape()以下に勾配算出対象の計算を行う https://qiita.com/propella/items/5b2182b3d6a13d20fefd
            with tf.GradientTape() as gen_tape:
                # input_image, target, weight = next(ds_iter)
                input_image, target, weight = ds
                target = nfunc.target_cut_padding(target=target, padding=self.get_padding())
                # Generatorによる画像生成
                generator = self.get_generator()
                gen_output = generator(input_image, training=True)
                # Discriminatorによる判定
                # discriminator = self.get_discriminator()
                # disc_real_output = discriminator([input_image, target], training=True)
                # disc_generated_output = discriminator([input_image, gen_output], training=True)
                # loss算出
                # gen_loss = self.generator_loss(gen_output, target, weight)
                gen_loss = self.generator_loss(gen_output, target, weight)
                err_count = 0
                c_weight = tf.ones_like(weight)
                # gen_loss, err_count, c_weight = self.evaluation_generator_loss(gen_output, target, weight)
                # disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)
            # 勾配算出 trainable_variables:訓練可能な変数
            generator_gradients = gen_tape.gradient(gen_loss,generator.trainable_variables)
            # discriminator_gradients = disc_tape.gradient(disc_loss,discriminator.trainable_variables)
            # バッチサイズ毎のデータサイズの割合を勾配に適用させる 1/len(device_list)を掛けて、複数GPU時の時との差を調整
            rate = tf.cast(rate, tf.float32)
            rate_gpu = tf.cast(1/len(device_list), tf.float32)
            use_rate = rate * rate_gpu
            generator_gradients = nfunc.rate_multiply(generator_gradients, use_rate)
            # discriminator_gradients = nfunc.rate_multiply(discriminator_gradients, rate * (1/len(device_list)))
            # gpu単位の処理ここまで
        # with tf.device('/gpu:%d' % gpu_index):  # gpu単位の処理
        generator = self.get_generator()
        discriminator = self.get_discriminator()
        # 勾配の適用
        self.get_generator_optimizer().apply_gradients(zip(generator_gradients, generator.trainable_variables))
        # self.get_discriminator_optimizer().apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
        return gen_output, err_count, c_weight

    # ==============================================================================================================
    # ========== check_accuracy関数 ================================================================================
    # ==============================================================================================================
    @tf.function  # 関数をグラフモードで実行 https://www.tensorflow.org/guide/autograph?hl=ja
    def multi_check_step(self, ds_iter, device_list, data_n):
        accuracy_list = []
        for gpu_index in device_list:
            with tf.device('/gpu:%d' % gpu_index):  # gpu単位の処理
                input_image, target = next(ds_iter)
                target = nfunc.target_cut_padding(target=target, padding=self.get_padding())
                # Generatorによる画像生成
                generator = self.get_generator()
                gen_output = generator(input_image, training=False)
                accuracy_list.append(nfunc.evaluate(net_cls=self, out=gen_output, ans=target) * data_n)
        return sum(accuracy_list)

    @tf.function  # 関数をグラフモードで実行 https://www.tensorflow.org/guide/autograph?hl=ja
    def check_step(self, ds_iter, gpu_index, data_n):
        accuracy_list = []
        with tf.device('/gpu:%d' % gpu_index):  # gpu単位の処理
            input_image, target = next(ds_iter)
            target = nfunc.target_cut_padding(target=target, padding=self.get_padding())
            # Generatorによる画像生成
            generator = self.get_generator()
            gen_output = generator(input_image, training=False)
            accuracy_list.append(nfunc.evaluate(net_cls=self, out=gen_output, ans=target) * data_n)
        return sum(accuracy_list)

# ************ クラス内呼び出し関数 ************
# ========== NET ===========
    def _Generator(self):
        down_stack = [
            self._downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
            self._downsample(128, 4),  # (bs, 64, 64, 128)
            self._downsample(256, 4),  # (bs, 32, 32, 256)
            self._downsample(512, 4),  # (bs, 16, 16, 512)
            self._downsample(512, 4),  # (bs, 8, 8, 512)
            self._downsample(512, 4),  # (bs, 4, 4, 512)
            self._downsample(512, 4),  # (bs, 2, 2, 512)
            self._downsample(512, 4),  # (bs, 1, 1, 512)
        ]
        up_stack = [
            self._upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
            self._upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
            self._upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
            self._upsample(512, 4),  # (bs, 16, 16, 1024)
            self._upsample(256, 4),  # (bs, 32, 32, 512)
            self._upsample(128, 4),  # (bs, 64, 64, 256)
            self._upsample(64, 4),  # (bs, 128, 128, 128)
        ]
        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(self.output_channel, 4,
                                               strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               activation=None)  # (bs, 256, 256, 3) tanh->=-1~1
        concat = tf.keras.layers.Concatenate() # 連結
        inputs = tf.keras.layers.Input(shape=[256,256,3])
        x = inputs
        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])
        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = concat([x, skip])
        x = last(x)
        return tf.keras.Model(inputs=inputs, outputs=x)  # 返却値は-1~1

    def _Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)
        inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
        tar = tf.keras.layers.Input(shape=[None, None, 1], name='target_image')
        x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)
        down1 = self._downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
        down2 = self._downsample(128, 4)(down1)  # (bs, 64, 64, 128)
        down3 = self._downsample(256, 4)(down2)  # (bs, 32, 32, 256)
        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)
        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)
        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                      kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)
        return tf.keras.Model(inputs=[inp, tar], outputs=last)

# ============ NET FUNCTION ==========
    def _downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                   kernel_initializer=initializer, use_bias=False))
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.LeakyReLU())
        return result

    def _upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())
        return result

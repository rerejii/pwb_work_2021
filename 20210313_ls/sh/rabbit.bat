@REM python ../lt_check_speed.py unet unet_0001_sigmoid_detac E:/work/mytensor/dataset2 Z:/hayakawa/binary float16 3
@REM python ../lite_test.py C:/Users/hayakawa/work/mytensor/dataset2 Z:/hayakawa/binary 0 20 A None
@REM python ../main_000001_rabbit.py C:/Users/hayakawa/work/mytensor/dataset2 Z:/hayakawa/binary 0 20 F None
@REM python ../main.py E:/work/myTensor/dataset2 Z:/hayakawa/binary 1 20 A None
@REM python ../main_weight-1-10_fin-none_logit-true_thr-d0.py E:/work/myTensor/dataset2 Z:/hayakawa/binary 1 20 A rabbit_RTX3090 None

@REM python ../main_fin-none_logit-true_thr-d0.py E:/work/myTensor/dataset2 Z:/hayakawa/binary 1 20 A rabbit_RTX3090 None
@REM python ../main_fin-none_logit-true_thr-d0.py E:/work/myTensor/dataset2 Z:/hayakawa/binary 1 20 B rabbit_RTX3090 None
@REM python ../main_fin-none_logit-true_thr-d0.py E:/work/myTensor/dataset2 Z:/hayakawa/binary 1 20 C rabbit_RTX3090 None
@REM python ../main_fin-none_logit-true_thr-d0.py E:/work/myTensor/dataset2 Z:/hayakawa/binary 1 20 D rabbit_RTX3090 None
@REM python ../main_fin-none_logit-true_thr-d0.py E:/work/myTensor/dataset2 Z:/hayakawa/binary 1 20 E rabbit_RTX3090 None
@REM python ../main_fin-none_logit-true_thr-d0.py E:/work/myTensor/dataset2 Z:/hayakawa/binary 1 20 F rabbit_RTX3090 None

python ../main_eva_acc_plot.py E:/work/myTensor/dataset2 Z:/hayakawa/binary 1 20 A rabbit_RTX3090 None
python ../main_eva_acc_plot.py E:/work/myTensor/dataset2 Z:/hayakawa/binary 1 20 B rabbit_RTX3090 None
python ../main_eva_acc_plot.py E:/work/myTensor/dataset2 Z:/hayakawa/binary 1 20 C rabbit_RTX3090 None
python ../main_eva_acc_plot.py E:/work/myTensor/dataset2 Z:/hayakawa/binary 1 20 D rabbit_RTX3090 None
python ../main_eva_acc_plot.py E:/work/myTensor/dataset2 Z:/hayakawa/binary 1 20 E rabbit_RTX3090 None
python ../main_eva_acc_plot.py E:/work/myTensor/dataset2 Z:/hayakawa/binary 1 20 F rabbit_RTX3090 None
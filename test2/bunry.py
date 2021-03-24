

def separation(gray, th):
    # クラス分け
    g0 ,g1 = gray[gray < th] ,gray[gray >= th]

    # 画素数
    w0 ,w1 = len(g0) ,len(g1)
    # 画素値分散
    s0_2 ,s1_2 = g0.var() ,g1.var()
    # 画素値平均
    m0 ,m1 = g0.mean() ,g1.mean()
    # 画素値合計
    p0 ,p1 = g0.sum() ,g1.sum()

    # クラス内分散
    sw_2 = w0 * s0_2 + w1 * s1_2
    # クラス間分散
    sb_2 = ((w0 * w1) / ((w0 + w1 ) *(w0 + w1))) * ((m0 - m1 ) *(m0 - m1))
    # 分離度
    if (sb_2 != 0):
        X = sb_2 / sw_2
    else:
        X = 0
    return X
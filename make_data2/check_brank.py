import numpy as np

def check_brank(img):
    print('check brank process now')
    height, width, _ = img.shape
    top_brank, bottom_brank, left_brank, right_brank = 0, 0, 0, 0

    # top
    for i in range(height):
        read = i
        line = img[read:read + 1, :, :]
        val = np.sum(line)
        if val != 0:
            top_brank = read
            break

    # bottom
    for i in range(height):
        read = height - i - 1
        line = img[read:read + 1, :, :]
        val = np.sum(line)
        if val != 0:
            bottom_brank = height - read - 1
            break

    # left
    for i in range(width):
        read = i
        line = img[:, read:read + 1, :]
        val = np.sum(line)
        if val != 0:
            left_brank = read
            break

    # right
    for i in range(width):
        read = width - i - 1
        line = img[:, read:read + 1, :]
        val = np.sum(line)
        if val != 0:
            right_brank = width - read - 1
            break

    brank = [top_brank, bottom_brank, left_brank, right_brank]
    print('check brank process end')
    return brank
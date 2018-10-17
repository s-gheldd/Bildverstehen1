import cv2
import numpy as np


def nms(src):
    copy = cv2.copyMakeBorder(src, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    change = copy.copy()
    rows, cols = copy.shape
    for row in range(rows - 2):
        up = row
        centre_v = row + 1
        down = row + 2
        for col in range(cols - 2):
            left = col
            centre_h = col + 1
            right = col + 2

            p = copy[centre_v, centre_h]
            up_left_ = copy[up, left]
            up_centre_h_ = copy[up, centre_h]
            up_right_ = copy[up, right]
            centre_v_left_ = copy[centre_v, left]
            centre_v_right_ = copy[centre_v, right]
            down_left_ = copy[down, left]
            down_centre_h_ = copy[down, centre_h]
            copy_down_right_ = copy[down, right]

            if p > 0 \
                    and p >= up_left_ \
                    and p >= up_centre_h_ \
                    and p >= up_right_ \
                    and p >= centre_v_left_ \
                    and p >= centre_v_right_ \
                    and p >= down_left_ \
                    and p >= down_centre_h_ \
                    and p >= copy_down_right_:

                change[row + 1, col + 1] = 1

            else:
                change[row + 1, col + 1] = 0
    change = change[1:rows - 1, 1:cols - 1]
    return change


def main():
    img = cv2.imread('BinaryObjectsI.png', 0)
    l1 = cv2.distanceTransform(img, cv2.DIST_L1, cv2.DIST_MASK_3, cv2.CV_8U)
    cv2.normalize(l1, l1, 0.0, 1.0, cv2.NORM_MINMAX)

    l2 = cv2.distanceTransform(img, cv2.DIST_L2, cv2.DIST_MASK_5, cv2.CV_8U)
    cv2.normalize(l2, l2, 0.0, 1.0, cv2.NORM_MINMAX)

    max_l1 = nms(l1)
    max_l2 = nms(l2)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay1 = img.copy()

    max_l2 = cv2.cvtColor(max_l2, cv2.COLOR_GRAY2BGR)
    max_l2 *= (255, 255, 0)
    max_l2 = np.uint8(max_l2)

    max_l1 = cv2.cvtColor(max_l1, cv2.COLOR_GRAY2BGR)
    max_l1 *= (255, 255, 0)
    max_l1 = np.uint8(max_l1)

    img = img - max_l2
    overlay1 = overlay1 - max_l1

    l2_horizontal = np.hstack((img, cv2.cvtColor(l2, cv2.COLOR_GRAY2BGR)))
    cv2.imshow('l2_5', l2_horizontal)

    l2_horizontal = np.hstack((overlay1, cv2.cvtColor(l1, cv2.COLOR_GRAY2BGR)))
    cv2.imshow('l1', l2_horizontal)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    main()

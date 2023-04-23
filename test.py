# 作者: hxd
# 2022年12月26日10时18分00秒
import cv2 as cv

if __name__ == '__main__':

    img = cv.imread("data/JPEGImages/0001-2_res.png")
    img = img * 255
    cv.namedWindow("img", 0)
    cv.imshow("img",img)
    cv.waitKey(0)
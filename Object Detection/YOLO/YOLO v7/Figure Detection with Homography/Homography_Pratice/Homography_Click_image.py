import cv2
import numpy as np
from utils import get_four_points


img = cv2.imread("pan.jpg")
img_src = cv2.resize(img,(0,0),fx=0.3, fy=0.3,interpolation=cv2.INTER_AREA)
dst_size = (400, 400, 3)
img_dst = np.zeros(dst_size, np.uint8)
cv2.imshow("dst", img_dst)

# 우리가 원본 이미지로부터는 마우스 클릭으로 4개의 점을 가져올 거다.
cv2.imshow("Image", img_src)
points_src = get_four_points(img_src)

# 새로 만들 이미지에서는, 위의 원본 이미지 4개의 점과 매핑할 점을 잡아줘야 한다.
points_dst = np.array([0, 0,
                       dst_size[1], 0,
                       dst_size[1], dst_size[0],
                       0, dst_size[0]])

points_dst = points_dst.reshape(4, 2)
h, status = cv2.findHomography(points_src, points_dst)
img_dst = cv2.warpPerspective(img_src, h, (dst_size[1], dst_size[0]))

####
# provide points from image 1
pts_src = np.array([[154, 174], [702, 349], [702, 572],[1, 572], [1, 191]])
# corresponding points from image 2 (i.e. (154, 174) matches (212, 80))
pts_dst = np.array([[212, 80],[489, 80],[505, 180],[367, 235], [144,153]])

# calculate matrix H
h, status = cv2.findHomography(pts_src, pts_dst)

# provide a point you wish to map from image 1 to image 2
a = np.array([[154, 174]], dtype='float32')
a = np.array([a])

# finally, get the mapping
pointsOut = cv2.perspectiveTransform(a, h)

cv2.imshow("DST", img_dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
from django.http import HttpResponse
import cv2
import dlib
import numpy
import sys
import os
from utils import conf, util, info

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# 用于排列图像的点
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

# 颜色校正期间使用的模糊量, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def hello(request):
    return HttpResponse("This is the backend for QZW mini program.")


class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


def get_landmarks(im):
    """
    将一个图像转化成numpy数组，并返回一个68×2元素矩阵，输入图像的每个特征点对应每行的一个x，y坐标
    为一张图像和一个标记矩阵生成一个遮罩，它画出了两个白色的凸多边形：一个是眼睛周围的区域，一个是鼻子和嘴部周围的区域。之后它由11个像素向遮罩的边缘外部羽化扩展，可以帮助隐藏任何不连续的区域。
    这样一个遮罩同时为这两个图像生成，使用与步骤2中相同的转换，可以使图像2的遮罩转化为图像1的坐标空间。
    之后，通过一个element-wise最大值，这两个遮罩结合成一个。结合这两个遮罩是为了确保图像1被掩盖，而显现出图像2的特性。
    :param im:
    :return:
    """
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFaces
        return 1
    if len(rects) == 0:
        raise NoFaces
        return 0

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im,
                    str(idx),
                    pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im, landmarks[group], color=1)

    im = numpy.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im


def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    1.将输入矩阵转换为浮点数。这是后续操作的基础。

    2.每一个点集减去它的矩心。一旦为点集找到了一个最佳的缩放和旋转方法，这两个矩心 c1 和 c2 就可以用来找到完整的解决方案。

    3.同样，每一个点集除以它的标准偏差。这会消除组件缩放偏差的问题。

    4.使用奇异值分解计算旋转部分。可以在维基百科上看到关于解决正交 Procrustes 问题的细节。

    5.利用仿射变换矩阵返回完整的转化。

    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)
    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T
    return numpy.vstack([
        numpy.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)),
        numpy.matrix([0., 0., 1.])
    ])


def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im,
                    (im.shape[1] * SCALE_FACTOR, im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)
    return im, s


def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2], (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
        numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
            im2_blur.astype(numpy.float64))


def swapFace(request):
    # print(request.FILES)
    file=request.FILES.get('original_img.jpg')
    fileId = util.getUniqueId()
    # print(fileId)
    filePath = os.path.join('user_file/' + '{}.jpg'.format(fileId))
    f = open(filePath, 'wb')
    for i in file.chunks():
        f.write(i)
    f.close()

    # print(filePath)
    im1, landmarks1 = read_im_and_landmarks('dst.jpg')
    im2, landmarks2 = read_im_and_landmarks(filePath)
    if isinstance(landmarks2,int) and landmarks2==0:
        return HttpResponse(info.error("NoFaces"),
                        content_type="application/json")
    if isinstance(landmarks2,int) and landmarks2==1:
        return HttpResponse(info.error("TooManyFaces"),
                        content_type="application/json")

    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                   landmarks2[ALIGN_POINTS])

    mask = get_face_mask(im2, landmarks2)
    warped_mask = warp_im(mask, M, im1.shape)
    combined_mask = numpy.max([get_face_mask(im1, landmarks1), warped_mask],
                              axis=0)

    warped_im2 = warp_im(im2, M, im1.shape)
    warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)

    output_im = im1 * (
        1.0 - combined_mask) + warped_corrected_im2 * combined_mask  #应用遮罩

    filePath = os.path.join('user_file/new_' + '{}.jpg'.format(fileId))
    cv2.imwrite(filePath, output_im)
    url = conf.IP_PORT +filePath
    return HttpResponse(info.success(url),
                        content_type="application/json")

import cv2
import numpy as np
from numpy import exp, abs, angle

from annotations import draw_convex_hull, annotate_landmarks
from constants import OVERLAY_POINTS, FEATHER_AMOUNT, SCALE_FACTOR, FOREHEAD_POINTS, LEFT_BROW_POINTS, \
    RIGHT_BROW_POINTS, \
    SCREEN
from utils import resize_ff, running_mean


def prepare_image(frame):
    # Prepare capture for face detection
    im_resized = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_AREA)
    cv2.flip(im_resized, 1, dst=im_resized)

    im_gray = cv2.cvtColor(im_resized, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.equalizeHist(im_gray)
    return im_resized[0:], im_gray[0:]


def face_mask(im, landmarks, roi):
    # Creates a masked image where only the face is shown
    # Uses ROI as bounds of frame
    zero_mat = np.zeros(im.shape[:2], dtype=np.float64)
    landmarks_org = landmarks.copy()
    for group in OVERLAY_POINTS:
        draw_convex_hull(zero_mat, landmarks[group], color=1)

    # for x in LEFT_BROW_POINTS:
    #     for _ in group:
    #         landmarks[x, 0] -= 1.3  # extend ROI to include foreheads
    #         draw_convex_hull(zero_mat, landmarks[group], color=1)
    #         landmarks = landmarks_org
    #
    # for x in RIGHT_BROW_POINTS:
    #     for _ in group:
    #         landmarks[x, 0] += 1.3  # extend ROI to include foreheads
    #         draw_convex_hull(zero_mat, landmarks[group], color=1)
    #         landmarks = landmarks_org

    for group in FOREHEAD_POINTS:
        for _ in group:
            landmarks[_, 1] -= 30  # extend ROI to include foreheads
            draw_convex_hull(zero_mat, landmarks[group], color=1)
            landmarks = landmarks_org

    zero_mat = np.array([zero_mat, zero_mat, zero_mat]).transpose(
        (1, 2, 0))  # creates an image array out of zero_mat [rows, cols, ch]

    zero_mat = (zero_mat > 0) * 1.0  # make hull points = 1

    # zero_mat = cv2.GaussianBlur(zero_mat, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
    # zero_mat = cv2.GaussianBlur(zero_mat, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
    zero_mat = cv2.blur(zero_mat, (FEATHER_AMOUNT, FEATHER_AMOUNT))
    zero_mat = cv2.blur(zero_mat, (FEATHER_AMOUNT, FEATHER_AMOUNT))  # this is faster than gaussian

    # apply the image mask to original frame
    im = im * zero_mat
    im = np.uint8(im)

    # resize and crop to ROI
    x = roi[0][0]
    y = roi[0][1]
    w = roi[0][2]
    h = roi[0][3]
    # im = im[y:y + h, x:x + w]  # crop image to extent of ROI

    rows, cols, ch = im.shape
    pts1 = np.float32([[x, y], [x + w, y + h], [x + w, y]])  # ROI rectangle
    pts2 = np.float32([[0, 0], [im.shape[1], im.shape[0]], [im.shape[1], 0]])  # frame bounds
    m = cv2.getAffineTransform(pts1, pts2)
    im = cv2.warpAffine(im, m, (cols, rows))

    # To shrink an image, it will generally look best with CV_INTER_AREA
    # interpolation, whereas to enlarge an image, it will generally look best
    im = cv2.resize(im, (720, 720), interpolation=cv2.INTER_LINEAR)

    # increase brightness - unused for now since it creates white regions
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)  # convert it to hsv
    # h, s, v = cv2.split(im)
    # v += 255
    # im = cv2.merge((h, s, v))
    # im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)

    return im


def mask2(im, landmarks):
    # Use landmarks as bounds for frame
    # landmarks[x,y] = xth element of landmarks, yth dimension (where 0=x,1=y)
    # Top is calculated by getting the smallest value of the landmark values since origin is top left.
    # 30% is added to this value to include forehead since dlib stops at eyebrows.

    left_x = landmarks[0, 0]
    right_x = landmarks[15, 0]
    top_y = landmarks[landmarks[:, 1].argmin(), 1]

    extend_val = 2 * int(landmarks[30, 1] - landmarks[27, 1])  # 2 * nose length

    # top_extended_y = int(top_y - 0.25 * (landmarks[landmarks[:,1].argmax(), 1] - landmarks[landmarks[:,1].argmin(), 1]))
    top_extended_y = int(top_y) - extend_val
    if top_extended_y <= 0:
        top_extended_y = 0
    bottom_y = landmarks[landmarks[:, 1].argmax(), 1]

    # Crop image and resize it while maintaining aspect ratio

    im = im[top_extended_y:bottom_y, left_x:right_x]  # crop image

    # # resize
    # rows, cols, ch = im.shape
    # aspect = SCREEN[0] / rows # make height match fullscreen height
    # dim = (int(aspect * cols), SCREEN[1])
    # im = cv2.resize(im, dim, interpolation=cv2.INTER_LINEAR)


    # Vignette - this needs work.
    # rows, cols, ch = im.shape
    # a = cv2.getGaussianKernel(cols, 300)
    # b = cv2.getGaussianKernel(rows, 300)
    # c = b * a.T
    # d = c / c.max()
    # im = im * d

    return im


def mask3(im, landmarks, roi):
    # Creates a masked image where only the face is shown
    zero_mat = np.zeros(im.shape[:2], dtype=np.float64)
    landmarks_org = landmarks.copy()
    for group in OVERLAY_POINTS:
        draw_convex_hull(zero_mat, landmarks[group], color=1)

    for group in FOREHEAD_POINTS:
        for _ in group:
            landmarks[_, 1] -= 30  # extend ROI to include foreheads
            draw_convex_hull(zero_mat, landmarks[group], color=1)
            landmarks = landmarks_org

    zero_mat = np.array([zero_mat, zero_mat, zero_mat]).transpose(
        (1, 2, 0))  # creates an image array out of zero_mat [rows, cols, ch]

    zero_mat = (zero_mat > 0) * 1.0  # make hull points = 1

    # zero_mat = cv2.GaussianBlur(zero_mat, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
    # zero_mat = cv2.GaussianBlur(zero_mat, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
    zero_mat = cv2.blur(zero_mat, (FEATHER_AMOUNT, FEATHER_AMOUNT))
    zero_mat = cv2.blur(zero_mat, (FEATHER_AMOUNT, FEATHER_AMOUNT))  # this is faster than gaussian

    # apply the image mask to original frame
    im = im * zero_mat

    im = np.uint8(im)

    # resize and crop, need to seperate colour components to do this
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)  # convert it to hsv
    h, s, v = cv2.split(im)

    ytop = np.nonzero(v)[0].min()
    ybottom = np.nonzero(v)[0].max()
    xleft = np.nonzero(v)[1].min()
    xright = np.nonzero(v)[1].max()

    im = cv2.merge((h, s, v))
    im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)

    im = im[ytop:ybottom, xleft:xright]  # crop image

    return im


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def barrel_distort(im, amplitude):
    # Attempting to apply barrel distortion. Adapted from https://au.mathworks.com/help/images/examples/creating-a-gallery-of-transformed-images.html (Image 10)
    # And this stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion
    # Also this https://gist.github.com/tanmaykm/4395121

    # get dimensions of image and find midpoint
    rows, cols, ch = im.shape

    midpoint_x = cols / 2
    midpoint_y = rows / 2

    # setup vars
    correction_factor = midpoint_y / (midpoint_y - amplitude * midpoint_y ** 3)
    x_range = range(1, cols + 1)
    y_range = range(1, rows + 1)
    xi, yi = np.meshgrid(x_range, y_range)  # creates a 2D grid the same size as the input image

    xt = np.ravel(xi) - midpoint_x  # stacks the x values into a single column and subtracts the midpoint
    yt = np.ravel(yi) - midpoint_y  # same but for y values

    r, phi = cart2pol(xt, yt)  # convert to polar coordinates so we can find the radius from center
    s = r - amplitude * r ** 3  # apply transformation function
    ut, vt = pol2cart(s, phi)  # convert back to cartesian

    u = np.reshape(ut * correction_factor,
                   (rows, cols)) + midpoint_x  # change it back to a grid and add the midpoint again
    v = np.reshape(vt * correction_factor, (rows, cols)) + midpoint_y
    u_32 = u.astype('float32')
    v_32 = v.astype('float32')

    im_distorted = cv2.remap(im, u_32, v_32, interpolation=cv2.INTER_CUBIC)  # borderMode=cv2.BORDER_CONSTANT

    return im_distorted

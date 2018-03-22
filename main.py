#!/usr/bin/python
import cv2
import time

from annotations import annotate_landmarks
from constants import SKIP_FRAME_VAL
from processing import prepare_image, face_mask, barrel_distort
from utils import find_faces, get_landmarks, resize_ff

cap = cv2.VideoCapture(0)
init = 0
frame_count = 0

while True:

    _, frame = cap.read()
    # prepare image for processing
    im_resized, im_gray = prepare_image(frame)
    im_annotated = im_resized.copy()

    # find faces using opencv
    roi, im_annotated, face_found = find_faces(im_gray, im_annotated)

    if face_found != 0:

        # create 10 point running average of roi and landmarks
        if frame_count == SKIP_FRAME_VAL:

            frame_count = 0
        frame_count += 1

        # detect and annotate landmarks using dlib
        landmarks = get_landmarks(im_gray, roi)
        im_annotated = annotate_landmarks(im_annotated, landmarks)

        # create mask
        im_mask = face_mask(im_resized, landmarks, roi)
        # im_mask2 = mask2(im_resized,landmarks)
        # im_mask3 = mask3(im_resized,landmarks,roi)

        cv2.namedWindow('Mask', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow('Mask', im_mask)
        # cv2.namedWindow('Mask2', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # cv2.imshow('Mask2', im_mask2)
        # cv2.namedWindow('Mask3', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # cv2.imshow('Mask3', im_mask3)

        # resize masks
        im_mask_ff = resize_ff(im_mask)
        cv2.flip(im_mask_ff, 0, dst=im_mask_ff)

        # im_mask2_ff = resize_ff(im_mask2)
        # im_mask3_ff = resize_ff(im_mask3)
        cv2.namedWindow('Mask_FF', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow('Mask_FF', im_mask_ff)
        # cv2.namedWindow('Mask2_FF', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # cv2.imshow('Mask2_FF', im_mask2_ff)
        # cv2.namedWindow('Mask3_FF', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # cv2.imshow('Mask3_F', im_mask3_ff)

        # apply distortion
        # amplitude = 0.000002  # decent for minor correction, pincushon
        # amplitude = 0.0000055  # snapchat filter
        # amplitude = -0.000005  # big fisheye
        # im_barrel = barrel_distort(im_mask, amplitude)

        # im_barrel_ff = resize_ff(im_barrel)
        # cv2.namedWindow("Barrel_FF", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # cv2.imshow("Barrel_FF", im_barrel_ff)
        #
        # cv2.namedWindow("Original Barrel Distortion", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # cv2.imshow("Original Barrel Distortion", im_barrel)


    # im_annotated = resize_ff(im_annotated)
    # cv2.namedWindow("Annotated Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Annotated Image", im_annotated)
    # cv2.imshow("Original Image",im_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('c'):
        # cv2.imwrite("/shared/output/{0}_barrel.jpg".format(time.strftime("%Y%m%d-%H%M%S")), im_barrel_ff)
        cv2.imwrite("/shared/output/{0}_mask.jpg".format(time.strftime("%Y%m%d-%H%M%S")), im_mask_ff)
        print "Captured Image"

cap.release()
cv2.destroyAllWindows()

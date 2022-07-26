import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def blur_image(image, mask_size=21):
    """receive the initial image and apply the gaussian blur according to the
    mask_size kernel passed as argument

    Args:
        image (_type_): matriz in gray_scale containing the trget image
        mask_size (_type_): size of the gaussian kernel the user wanna apply in the image

    Returns:
        _type_: blur image
    """
    if mask_size % 2 == 0:
        mask_size = mask_size + 1
    blur_img = cv.GaussianBlur(image, (mask_size, mask_size), 0)
    return blur_img


def edges_canny(image, lower_threshold=50, multiplier=2, aperture_size=3):
    """apply the canny algorithm to evidence the image edges and draw
    it's contour

    Args:
        image (_type_): numpy matrix containing the target image
        lower_threshold (_type_): lower threshold to canny algorithm
        multiplier (_type_): multiplier to the upper threshold
        aperture_size (_type_): aperture size must be 3,5 ou 7

    Returns:
        canny_img: img with the canny algorithm applyed with the parameters passed
        as argument
    """
    t_lower = lower_threshold
    t_upper = multiplier * lower_threshold
    if aperture_size in [3, 5, 7]:
        ap_size = aperture_size
    else:
        ap_size = 3

    canny_img = cv.Canny(image, t_lower, t_upper, apertureSize=ap_size)
    return canny_img


def dilate_image(image, kernel_size=5, iterations=1):
    """dilate the image according to the kernel size, this function is mainly used to
    dilate the canny's edges found on the edges_canny function to evidence it better to
    contours algorithm and bounding boxes.

    Args:
        image (_type_): numpy matrix with the image the user gonna dilate
        kernel_size (int, optional): kernel's size of the boxe dilate kernel that the
        function gonna apply on the image. Defaults to 5
        iterations(int): iterations of the dilate algorithm on the image.Defaults to 1.

    Returns:
        _type_: dilated image according to the kernel_size
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_img = cv.dilate(image, kernel, iterations=iterations)
    return dilated_img


def find_contours_and_boxes(image):
    """find the contours of the image. This functions evidence the contours of the image,
    and intends to find the contours of the dilated image resulted from the dilate_image's
    function. We evidence the contours to find the ROI(Region of interest) of the target
    objects in the image and returns the bounding-boxes and the countors in two arrays.

    Args:
        image (_type_): target image in grayscale

    Returns:
        _type_: contours and the bounding boxes of the ROI's
    """
    contours, hierarchy = cv.findContours(
        image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])

    return contours_poly, boundRect, contours


def draw_bounding_boxes_contours(gray_image, contours_poly, boundRect, contours):
    """use the find_contours_and_boxes's function result to draw the countours
    into the target image. it intends to show how are the bounding boxes and the
    contours

    Args:
        gray_image (_type_): _description_
        contours_poly (_type_): _description_
        boundRect (_type_): _description_
        contours (_type_): _description_

    Returns:
        _type_: _description_
    """
    gray_image2 = gray_image.copy()
    for i in range(len(contours)):
        color = (
            0,
            0,
            0,
        )
        cv.drawContours(gray_image2, contours_poly, i, color)
        cv.rectangle(
            gray_image2,
            (int(boundRect[i][0]), int(boundRect[i][1])),
            (
                int(boundRect[i][0] + boundRect[i][2]),
                int(boundRect[i][1] + boundRect[i][3]),
            ),
            color,
            2,
        )
    return gray_image2


def extract_roi_by_boxes(image, boundRect):
    """extract region of interest using the bounding boxes found in
    find_contours_and_boxes.

    Args:
        image (_type_): numpy array with the original image
        boundRect (_type_): Array with bounding boxes coordinates

    Returns:
        list: list of roi's extracted from the image
    """
    rois = []
    for i in boundRect:
        print(str(i[0]) + "\n")
        rois.append(image[i[1] : (i[1] + i[3]), i[0] : i[0] + i[2]])

    return rois


def extract_roi_black_white(image, boundRect, threshold=160):
    """extract roi with black and white

    Args:
        image (_type_): numpy array with the original image
        boundRect (_type_): bound rects array with bounding boxes
        threshold (_type_): lower threshold value

    Returns:
        list : array with roi's in black and white
    """
    letters_bw = []
    letters = extract_roi_by_boxes(image, boundRect)
    for letter in letters:
        (thresh, blackAndWhiteLetter) = cv.threshold(
            letter, threshold, 255, cv.THRESH_BINARY
        )
        letters_bw.append(255 - blackAndWhiteLetter)
    return letters_bw


def resize_rois(letters_bw):
    """resize the images for a 28X28 image as the mnist source format

    Args:
        letters_bw (_type_): roi's array

    Returns:
        list: list with resized images
    """
    resized_letters = []
    for letter_bw in letters_bw:
        resized_letter = cv.resize(letter_bw, (28, 28), interpolation=cv.INTER_AREA)
        resized_letters.append(resized_letter)
    return resized_letters

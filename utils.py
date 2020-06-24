import cv2 as cv


def mask2polygon(mask_image):
    """
    :param mask_image: 输入mask图片地址，默认为gray, 且像素值为0或255
    :return: list, 每个item为一个labelme的points
    """
    mask = cv.imread(mask_image, 0)
    _, binary = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    results = [item.squeeze() for item in contours]
    return results

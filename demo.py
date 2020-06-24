import cv2 as cv


def demo():
    from utils import mask2polygon
    image = cv.imread('./images/0_origin.jpg')
    polygons = mask2polygon('./images/0_mask.png')
    print(polygons)
    for polygon in polygons:
        pts = polygon.reshape((-1, 1, 2))
        cv.polylines(image, [pts], True, (0, 0, 255), 2)

    cv.imshow('demo', image)
    cv.waitKey()


if __name__ == '__main__':
    demo()

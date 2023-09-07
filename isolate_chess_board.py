import numpy as np
import cv2 as cv

def load_image(file='good_pics/IMG_3382.JPG'):
    # Load the chessboard image
    raw_img = cv.imread(file)
    assert raw_img is not None, f"image not found at '{file}'!"
    return raw_img

def threshold_image(img, separate_param=100, reverse_binary=False):
    # Run basic thresholding
    grey_img = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
    _, threshold_img = cv.threshold(grey_img, separate_param, 255, 0)
    # Reverse binary image
    if reverse_binary:
        threshold_img = cv.bitwise_not(threshold_img)
    return threshold_img

def get_contours(binary_img):
    # Rip contours from image pixels and draw contours over raw_img
    contours, _ = cv.findContours(binary_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    return contours

def remove_noise_contours(contours, area_min=60000, length_min=1000):
    # Remove contours that have small length, and
    # Remove contours that have a small area
    bad_contours = []
    good_contours = []
    for cnt in contours:
        if cv.contourArea(cnt) < area_min or cv.arcLength(cnt, False) < length_min:
            bad_contours.append(cnt)
        else:
            good_contours.append(cnt)

    print(f"ratio of good contours to total contours: {len(good_contours)/len(contours)}")

    return good_contours

def contour_to_square(contour, epsilon=1000, canvas_image=None):
    # Convert chessboard contours to square and find corner positions
    board_outline = cv.approxPolyDP(contour, epsilon, True)

    if canvas_image is not None:
        board_outline_img = cv.drawContours(canvas_image.copy(), board_outline, -1, (255, 0, 255), 20)
    else:
        board_outline_img = None

    return (board_outline, board_outline_img)

def draw_contour(canvas_img, contours, name="temp"):
    draw_img = cv.fillPoly(canvas_img, pts=contours,color=(255,255,255))
    cv.imwrite(f'{name}.png', draw_img)

def main():
    raw_img = load_image()
    threshold_img = threshold_image(raw_img)
    
    contours = get_contours(threshold_img)
    draw_contour(raw_img, contours, "all_contours")

    contours = remove_noise_contours(contours)
    draw_contour(raw_img, contours, "some_contours")
    
    corners = contour_to_square(contours[2])
    draw_contour(raw_img, contours, "square")
    
    print(corners)

    cv.imwrite("output.png", raw_img)
    input("Press Enter...")

if __name__ == "__main__":
    main()
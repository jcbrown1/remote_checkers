import numpy as np
import cv2 as cv

# Load the chessboard image
raw_img_path = 'good_pics/IMG_3382.JPG'
raw_img = cv.imread(raw_img_path)
assert raw_img is not None, f"image not found at '{raw_img_path}'!"

# # Run Canny edge detection algorithm
# canny_img = cv.Canny(raw_img, 50, 300)
# assert canny_img is not raw_img
# cv.imwrite('canny_img.png', canny_img)

# Run basic thresholding
grey_img = cv.cvtColor(raw_img.copy(), cv.COLOR_BGR2GRAY)
ret, threshold_img = cv.threshold(grey_img, 100, 255, 0)
# Reverse binary image
threshold_img = cv.bitwise_not(threshold_img)
cv.imwrite('threshold_img.png', threshold_img)


# Rip contours from canny_img pixels and draw contours over raw_img
contours, hierarchy = cv.findContours(threshold_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
all_contours_img = cv.drawContours(raw_img.copy(), contours, -1, (255, 0, 0), 10)
cv.imwrite('all_contours_img.png', all_contours_img)

# Color in really small closed contours
bad_contours = []
good_contours = []
for cnt in contours:
    if cv.contourArea(cnt) < 60000:
        bad_contours.append(cnt)
    else:
        good_contours.append(cnt)

# Draw in small tings        
bad_contours_threshold_img = cv.fillPoly(threshold_img, pts=bad_contours,color=(255,255,255))
cv.imwrite('bad_contours_filled_img.png', bad_contours_threshold_img)

# Remove all small arc length contours
short_contours = []
long_contours = []
for cnt in good_contours:
    if cv.arcLength(cnt, False) < 1000:
        short_contours.append(cnt)
    else:
        long_contours.append(cnt)

print(f"ratio of long to total checked contours: {len(long_contours)/len(good_contours)}")

# Draw left over contours on original image
long_contours_img = cv.drawContours(raw_img.copy(), long_contours, -1, (255, 0, 0), 10)
cv.imwrite('long_contours_img.png', long_contours_img)

# Hand remove some contours
hand_selected_contours = long_contours[2:]
chess_board_contour = hand_selected_contours[0]
hand_selected_contours_img = cv.drawContours(raw_img.copy(), chess_board_contour, -1, (0, 0, 255), 10)
cv.imwrite('hand_selected_contours_img.png', hand_selected_contours_img)

# Convert chessboard contours to square and find corner positions
board_outline = cv.approxPolyDP(chess_board_contour, 1000, True)
board_outline_img = cv.drawContours(raw_img.copy(), board_outline, -1, (255, 0, 255), 20)
cv.imwrite('board_outline_img.png', board_outline_img)

print(board_outline)
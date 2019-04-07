import cv2

img = cv2.imread('me.jpg')
if img is None:
    print("Image not loaded!")
else:
    print("Image is loaded!")
img = cv2.imshow('image',img)
cv2.waitKey(0)
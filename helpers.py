import time
import cv2
import numpy as np
from PIL import ImageGrab


def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    processed_img = cv2.resize(processed_img, (80, 60))  # 160, 120
    # processed_img = processed_img.astype()
    return processed_img


def render(frame):
    cv2.imshow("window", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        # break


def give_a_frame(normalize=False):
    # screen = np.array(ImageGrab.grab(bbox=(0, 300, 800, 640)))
    screen = np.array(ImageGrab.grab(bbox=(50, 250, 800, 630)))  # (50,250,800,630)
    new_screen = process_img(screen)

    if normalize is True:
        new_screen = new_screen / 255

    return new_screen


def count_down():
    for i in list(range(2))[::-1]:
        print(i + 1)
        time.sleep(1)

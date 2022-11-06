# source: https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/

import cv2

# define a video capture object
vid = cv2.VideoCapture(0)
W, H = 1920, 1080
# vid.set(cv2.CAP_PROP_FRAME_WIDTH, W)
# vid.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
fps = vid.get(cv2.CAP_PROP_FPS)

print("fps: " + str(fps))

while (True):

    # Capture the video frame by frame
    ret, frame = vid.read()

    # Display the resulting frame
    cv2.imshow('Camera Test', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

import os,sys
import numpy as np
from cv2 import cv2

video_file = "./CCTV_2.mp4"

if os.path.exists(video_file):
    try:
        cap = cv2.VideoCapture(video_file)
    except IOError as e:
        print(f"I/O error({e.errno}): {e.strerror}")
    except: #handle other exceptions such as attribute errors
        print("Unexpected error:", sys.exc_info()[0])
else:
    print("File not found", file=sys.stderr)
    sys.exit(1)

ret, frame = cap.read()
if frame is None:
    print("Cannot parse file content")
    sys.exit(1)

width  = int(cap.get(3))
height = int(cap.get(4))

print(f'[INFO] Video OK, width: {width}, height: {height}')

def bbox(img, cols, rows):
    """
    Draw rectangle bounding box around points.
    """
    side_length = 6
    for x, y in zip(cols, rows):
        top_left = (int(x - side_length / 2), int(y - side_length / 2))
        bot_right = (int(x + side_length / 2), int(y + side_length / 2))
        img = cv2.rectangle(
            img, top_left, bot_right,
            color = (0, 0, 255), thickness = 1
        )
    return img

while(True):
    # cv2.putText(frame, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

    img_mask = np.zeros((height, width), np.uint8) # mask
    x_center = 180
    y_center = 15
    radius = 5
    cv2.circle(img_mask,(x_center,y_center),radius,(255,255,255),-1) # measuring area
    a,b,g,r = cv2.mean(frame, mask=img_mask)[::-1]

    if np.mean([b,g,r]) > 10 :
        print([b, g, r])
        cv2.circle(frame, (180,15), 5, (255,0,0), -1)
    
    frame = bbox(frame, [x_center,], [y_center,])

    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print('fps:', fps)  # float

    cv2.imshow("APC", frame)
    ret, frame = cap.read()

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

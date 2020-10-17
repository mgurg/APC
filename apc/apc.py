import numpy as np
from cv2 import cv2

cap = cv2.VideoCapture("./CCTV_2.mp4")
ret, frame = cap.read()

width  = int(cap.get(3))
height = int(cap.get(4))

print(f'[INFO] width: {width}, height: {height}')


def bbox(img, cols, rows):
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
    b,g,r = (frame[100, 100])

    # https://stackoverflow.com/questions/43086715/rgb-average-of-circles

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

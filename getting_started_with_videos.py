import cv2

cap = cv2.VideoCapture(0)
background = cv2.imread('background.png')
eye = cv2.imread('eyeBall.png', -1)
mask = cv2.imread('eyeMask.png', -1)
x_offset = 715
y_offset = 655
x_offset_mask = 715
y_offset_mask = 630

ret, frame1 = cap.read()
ret, frame2 = cap.read()
maxCount = 0
midW = 0
midH = 0
centerX = 1000
centerY = 1000
tempArrayX = []
tempArrayY = []

while cap.isOpened():
    maxArea = 0
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 7000:
            continue

        area = w * h

        if maxArea < area:
            maxCount += 1
            maxArea = area
            (mx, my, mw, mh) = cv2.boundingRect(contour)
            midW = mx + (mw / 2)
            midH = my + (mh / 2)
            midWPer = midW / cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 100
            midHPer = midH / cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 100
            cv2.rectangle(frame1, (mx, my), (mx + mw, my + mh), (0, 255, 0), 2)
            if maxCount < 20:
                centerX = 0
                centerY = 0
                tempArrayX.append(midW)
                for tempX in tempArrayX:
                    centerX += tempX
                centerX /= maxCount
                tempArrayY.append(midH)
                for tempY in tempArrayY:
                    centerY += tempY
                centerY /= maxCount
            else:
                centerX = 0
                centerY = 0
                tempArrayX.pop(0)
                tempArrayX.append(midW)
                for tempX in tempArrayX:
                    centerX += tempX
                centerX /= 20
                centerX = round(centerX, 1)
                tempArrayY.pop(0)
                tempArrayY.append(midH)
                for tempY in tempArrayY:
                    centerY += tempY
                centerY /= 20
                centerY = round(centerY, 1)

    cv2.rectangle(frame1, (int(centerX), int(centerY)),
                  (int(centerX) + 2, int(centerY) + 2), (0, 255, 0), 2)

    centerPerX = centerX / cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 100
    centerPerY = centerY / cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 100

    cv2.putText(frame1, "Per: {}".format(centerPerX), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 3)
    # cv2.putText(frame1, "centerX: {}".format(centerX), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
    #             1, (0, 0, 255), 3)

    cv2.imshow('cam', frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if ret == False:
        break

    x_offset = 715 + ((100-centerPerX)/100*80)
    x_offset = int(x_offset)

    y1, y2 = y_offset, y_offset + eye.shape[0]
    x1, x2 = x_offset, x_offset + eye.shape[1]
    alpha_s = eye[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        background[y1:y2, x1:x2, c] = (alpha_s * eye[:, :, c] +
                                  alpha_l * background[y1:y2, x1:x2, c])

    y1, y2 = y_offset_mask, y_offset_mask + mask.shape[0]
    x1, x2 = x_offset_mask, x_offset_mask + mask.shape[1]
    alpha_s = mask[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        background[y1:y2, x1:x2, c] = (alpha_s * mask[:, :, c] +
                                       alpha_l * background[y1:y2, x1:x2, c])

    cv2.imshow('image', background)


cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe

cap = cv2.VideoCapture(0)

mpHands = mediapipe.python.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mediapipe.python.solutions.drawing_utils

pTime = 0
cTime = 0

cx, cy, w, h = 100, 100, 200, 200


class DragRect():
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter
        self.size = size

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        # If the index finger tip is in the rectangle region
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
                cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor
rectList = []
for x in range(1):
    rectList.append(DragRect([x * 250 + 150, 150]))

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = img.shape
    # img = cv2.flip(img, 1)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            x = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x * image_width
            y = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * image_height
            cursor = (x, y)
            for rect in rectList:
                rect.update(cursor)


    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(img, (int(cx - w // 2), int(cy - h // 2)),
                      (int(cx + w // 2), int(cy + h // 2)), (100, 0, 0), cv2.FILLED)
    img = cv2.flip(img, 1)
    # cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)


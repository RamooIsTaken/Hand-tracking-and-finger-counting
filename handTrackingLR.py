import cv2, time
import mediapipe as mp 


cap = cv2.VideoCapture(0)

mpHand = mp.solutions.hands
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success , img = cap.read()
    imgRgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)

    results = hands.process(imgRgb)
    # print(results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks :
        
        for handLms in results.multi_hand_landmarks :
            mpDraw.draw_landmarks(img,handLms,mpHand.HAND_CONNECTIONS)

            # Determine left or right hand
            if handLms.landmark[mpHand.HandLandmark.WRIST].x < handLms.landmark[mpHand.HandLandmark.THUMB_TIP].x :
                hand = "Right hand"

            else:
                hand = "Left hand"

            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                h,w,c = img.shape

                cx,cy = int(lm.x*w), int(lm.y*h)    

                
                if id == 4 : # başparmak ucu id = 0 oldığu için bilek işaretleyecek
                    cv2.circle(img,(cx,cy),9,(255,0,0),cv2.FILLED)

                if id == 0 : # serçe parkmak id = 0 oldığu için bilek işaretleyecek
                    cv2.circle(img,(cx,cy),9,(255,255,0),cv2.FILLED)
                    cv2.putText(img, hand, (cx,cy), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime

    cv2.putText(img,"FPS :"+str(int(fps)),(10,650),cv2.FONT_ITALIC,3,(255,125,140),5)

    cv2.imshow("İmg",img)

    if cv2.waitKey(1) & 0xFF == ord("q") :
        break 

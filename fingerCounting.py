import cv2,datetime
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

mpHand = mp.solutions.hands
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils
totalCount = 0
totalNum = []

tipIds = [4,8,12,16,20] # parmakların uç noktasının id leri
while True :

    success , img = cap.read()
    imgRgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    results = hands.process(imgRgb)
    

    lmList = []

    if results.multi_hand_landmarks :

        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,handLms,mpHand.HAND_CONNECTIONS)

            for id ,lm in enumerate(handLms.landmark) :
                h,w,c = img.shape

                cx,cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id,cx,cy])

                """# işaret uç noktası 
                if id == 8 : 
                    cv2.circle(img,(cx,cy),9,(255,0,0),cv2.FILLED)
                if id == 6 : 
                    cv2.circle(img,(cx,cy),9,(255,0,0),cv2.FILLED)"""

    if len(lmList) != 0 :
        fingers = []
        
         

        if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]: # başparmak için
            fingers.append(1)
        else :
            fingers.append(0)
        for id in range(1,5):
            
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2] :
                fingers.append(1)
            else :
                fingers.append(0)

        totalF = fingers.count(1)
        totalNum.append(totalF)
        
        
        cv2.putText(img,str(totalF),(30,200),cv2.FONT_ITALIC,3,(2500,125,32),3)

    


    cv2.imshow("İmg",img)

    if cv2.waitKey(1) & 0xFF == ord("q") :
        break 



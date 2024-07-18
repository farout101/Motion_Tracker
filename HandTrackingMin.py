import cv2  # Import OpenCV library
import mediapipe as mp  # Import MediaPipe library
import time  # Import time library

# Initialize video capture object to capture from the default camera
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

# Infinite loop to continuously capture and process video frames
while True:
    # Read a frame from the video capture object
    success, img = cap.read()
    
    img = cv2.flip(img, 1)
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    
    # Display the captured frame in a window titled "Image"
    cv2.imshow("Image", img)
    
    # Wait for 1 millisecond before processing the next frame
    cv2.waitKey(1)

import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.6,min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        retur, frame = cap.read()
        
        #Color changing
        image= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        #Detection
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #print(results)
        
        #for detection of left hand or right hand and displaying results 
        def get_label(index, hand, results):
            output = None
            for idx, classification in enumerate(results.multi_handedness):
                if classification.classification[0].index == index:
            
                    # Process results
                    label = classification.classification[0].label
                    score = classification.classification[0].score
                    text = '{} {}'.format(label, round(score, 2))
                    
                    # Extract Coordinates
                    coords = tuple(np.multiply(
                        np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                    [640,480]).astype(int))
                    
                    output = text, coords
                    
            return output

        if results.multi_hand_landmarks:
            for num,hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

                if get_label(num, hand, results):
                    text, coord = get_label(num, hand, results)
                    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)   
        cv2.imshow("Hand  pose recognization" , image)

        if cv2.waitKey(10) & 0xff == ord('q'):
            break

cap.release()
cv2.destoryAllWindows()
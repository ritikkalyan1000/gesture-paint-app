import cv2
import mediapipe as mp
import os
import numpy as np


mphands = mp.solutions.hands
hand = mphands.Hands()

mpdraw = mp.solutions.drawing_utils
specs = mpdraw.DrawingSpec(thickness=1,circle_radius = 4,color=(150,150,0))

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,800)
draw_color = (255,0,0)

folder_name = "D:\python\opencv\codes\pics"
image_list = []

for img in os.listdir(folder_name):
    # print(img)
    image_list.append(img)

print(image_list)
prev_x,prev_y = 0,0

imgcanva = np.zeros((720,1280,3),np.uint8)

overlap_img = cv2.imread(f"{folder_name}/" + image_list[0])
overlap_img = cv2.resize(overlap_img,(1200,200))

while True:
    p_list = []
    
    _,img = cap.read()
    img = cv2.flip(img,1)
    # overlap_img = cv2.imread(f"{folder_name}/" + image_list[0])
    # overlap_img = cv2.resize(overlap_img,(1200,200))
    # overlap_img = cv2.flip(overlap_img,1)
    img[0:200,0:1200] = overlap_img
    
    imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    results = hand.process(imgrgb)
    
    
    
    if(results.multi_hand_landmarks):
        for handlm in results.multi_hand_landmarks:
            mpdraw.draw_landmarks(img,handlm,mphands.HAND_CONNECTIONS,specs,specs)
            h,w,_ = img.shape
            
            
            for id,val in enumerate(handlm.landmark):
                cx,cy = int(val.x*w) , int(val.y*h)
                cv2.putText(img,f"{id}",(cx,cy-20),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.9,(0,0,200),2)
                p_list.append([id,cx,cy])
                
                # print(p_list)
                
                
        if((p_list[8][2] < p_list[6][2]) and (p_list[12][2] < p_list[10][2])):
            print(f"selection_mode ")
            cv2.rectangle(img,(p_list[12][1],p_list[12][2]),(p_list[8][1],p_list[8][2]),(0,0,0),-2)
            if(p_list[12][2] < 200):
                if( 24< p_list[12][1] < 170):
                    draw_color = (255,0,0)          # blue
                    
                    prev_x,prev_y =0 , 0
                    
                    overlap_img = cv2.imread(f"{folder_name}/" + image_list[0])
                    overlap_img = cv2.resize(overlap_img,(1200,200))
                    
                elif(321 < p_list[12][1] < 553):
                    draw_color = (0,255,0)          # green
                    prev_x,prev_y =0,0
                    overlap_img = cv2.imread(f"{folder_name}/" + image_list[2])
                    overlap_img = cv2.resize(overlap_img,(1200,200))
                    
                elif(606 < p_list[12][1] < 822):
                    draw_color = (0,0,255)      # red
                    prev_x,prev_y =0,0
                    overlap_img = cv2.imread(f"{folder_name}/" + image_list[3])
                    overlap_img = cv2.resize(overlap_img,(1200,200))
                    
                elif(865 < p_list[12][1] < 1132):
                    draw_color = (0,0,0)            # eraser
                    prev_x,prev_y =0,0
                    overlap_img = cv2.imread(f"{folder_name}/" + image_list[1])
                    overlap_img = cv2.resize(overlap_img,(1200,200))
                    
                
                
            
            
        elif((p_list[8][2] < p_list[6][2])):
            print("drawing_mode")
            cv2.circle(img,(p_list[8][1],p_list[8][2]),5,draw_color,-2)
            if(prev_x == 0 and prev_y == 0):
                prev_x = p_list[8][1]
                prev_y = p_list[8][2]
            
            cv2.line(img,(prev_x,prev_y),(p_list[8][1],p_list[8][2]),draw_color,2)
            cv2.line(imgcanva,(prev_x,prev_y),(p_list[8][1],p_list[8][2]),draw_color,2)
            prev_x = p_list[8][1]
            prev_y = p_list[8][2]
            
            
            
        img = cv2.addWeighted(img,0.9,imgcanva,0.5,0)
            
            
                
            
            
                
                
    cv2.imshow("window1",img)
    if(cv2.waitKey(1) & 0xff ==ord("x")):
        break
                
                
                
        
    

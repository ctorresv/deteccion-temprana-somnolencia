# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:54:41 2022

@author: Usuario
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from metodoML import *
#from camara import *

def draw_circle_lips(frame, x,y,j,k,p,o,a,b):
    cv2.circle(frame, (x,y), 2,(0, 0, 255),-1) # labio superior
    cv2.circle(frame, (j,k), 2,(144, 50, 255),-1) # labio inferior
    cv2.circle(frame, (p,o), 2,(144, 50, 0),-1)
    cv2.circle(frame, (a,b), 2,(144, 150, 55),-1)
    
def draw_circle_eyes(frame, x,y,o,p,u,u1):
    cv2.circle(frame, (x,y), 2,(0, 160, 255),-1)
    cv2.circle(frame, (o,p), 2,(20, 160, 50),-1)
    cv2.circle(frame, (u,u1), 2,(160, 160, 50),-1)
    
cont = 0
contP = 0
t = []
perclos = []
perCLOSV = []
indi1 = []
indB = []
boca = []
prom = []
Clase = []
proboca = []
ClaseTabla = []
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap =cv2.VideoCapture('videos_i8/11.mp4')
#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
#cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      
      continue

    
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

   
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        
        #Ubicacion puntos de los labios sin normalizar
        x = results.multi_face_landmarks[0].landmark[13].x #labio superior
        y = results.multi_face_landmarks[0].landmark[13].y
        shape = image.shape 
        relative_x = int(x * shape[1])
        relative_y = int(y * shape[0]) 
        
        j = results.multi_face_landmarks[0].landmark[14].x #labio inferior
        k = results.multi_face_landmarks[0].landmark[14].y
        shapeJ = image.shape 
        relative_j = int(j * shapeJ[1])
        relative_k = int(k * shapeJ[0]) 
        
        p = results.multi_face_landmarks[0].landmark[61].x #
        o = results.multi_face_landmarks[0].landmark[61].y
        shapeP = image.shape 
        relative_p = int(p * shapeP[1])
        relative_o = int(o * shapeP[0]) 
        
        a = results.multi_face_landmarks[0].landmark[291].x #
        b = results.multi_face_landmarks[0].landmark[291].y
        shapeA = image.shape 
        relative_a = int(a * shapeA[1])
        relative_b = int(b * shapeA[0]) 
        
        #Calculo para definir si la boca esta abierta >0.5 se considera un bostezo
        disAncho = math.sqrt((relative_x-relative_j)**2 + 
                                   (relative_y-relative_k)**2)
        
        disLargo = math.sqrt((relative_p-relative_a)**2 + 
                             (relative_o-relative_b)**2)
        
        bocaAbierta = disAncho/disLargo
        
        
        
        '''
        print('X',relative_x) 
        print('Y',results.multi_face_landmarks[0].landmark[468].y)
        print('Z',results.multi_face_landmarks[0].landmark[468].z)
        '''
        
        draw_circle_lips(image,relative_x,relative_y,relative_j,
                             relative_k,relative_p,relative_o,relative_a
                             ,relative_b)
        
        lefteyeX = results.multi_face_landmarks[0].landmark[386].x 
        lefteyeY = results.multi_face_landmarks[0].landmark[386].y
        shapeE = image.shape 
        relative_Ex = int(lefteyeX * shapeE[1])
        relative_Ey = int(lefteyeY * shapeE[0]) 
        
        lefteyeO = results.multi_face_landmarks[0].landmark[374].x 
        lefteyeP = results.multi_face_landmarks[0].landmark[374].y
        shapeEO = image.shape 
        relative_EyO = int(lefteyeO * shapeEO[1])
        relative_EyP = int(lefteyeP * shapeEO[0]) 
        
        lefteyeU = results.multi_face_landmarks[0].landmark[473].x 
        lefteyeU1 = results.multi_face_landmarks[0].landmark[473].y
        shapeEU = image.shape
        relative_EyU = int(lefteyeU * shapeEU[1])
        relative_EyU1 = int(lefteyeU1 * shapeEU[0])
        
        #print("X: ", relative_EyU, "Y: ", relative_EyU1)
             

            
        perclos0 =  math.sqrt((relative_Ex-relative_EyU)**2 + 
                                       (relative_Ey-relative_EyU1)**2)
            
        perclos.append(float(perclos0))
        cont += 1
       
        if cont == 50:
            Max = np.mean(perclos)

                
        elif cont> 50:       
            p = perclos0/Max
            if p > 1: 
                p = 1
            if bocaAbierta < 0.1:
                bocaAbierta = 0.1
                
            t.append(cont)
            perCLOSV.append(p)             
            boca.append(bocaAbierta)
          
            if len(perCLOSV) == 5: 
                proE = np.mean(perCLOSV)
                proB = np.mean(boca)
                proboca.append(proB)
                prom.append(proE)
                perCLOSV = []
                boca = []
            
            
                if proE < 0.79:
                    indi1.append(1)               
                else:
                    indi1.append(0)
                
                
                if proB >= 0.45:
                    indB.append(1)
                elif proB <= 0.48 and proE < 0.79:
                    indB.append(0.5)
                else:
                    indB.append(0)
                       
                #print("Perclos: ",prom[-1],"Boca: ",proboca[-1],"Indi1: ",indi1[-1],"IndiB: ",indB[-1])    
                #fuzyLogic(prom[-1],proboca[-1],indi1[-1],indB[-1])
                
                if fuzyLogic(prom[-1],proboca[-1],indi1[-1],indB[-1]) > 0.5:
                    cl = 1
                else:
                    cl = 0
                    
                #Clase.append(fuzyLogic(prom[-1],proboca[-1],indi1[-1],indB[-1]))
                Clase.append(cl)
                
                if prom[-1] < 0.8 and proboca[-1] < 0.45 and indi1[-1] == 1 and indB[-1] == 0.5: 
                    ClaseTabla.append(1)
                    
                elif prom[-1] > 0.79 and proboca[-1] < 0.45 and indi1[-1] == 0 and indB[-1] == 0: 
                    ClaseTabla.append(0)
                    
                elif prom[-1] < 0.8 and proboca[-1] < 0.45 and indi1[-1] == 1 and indB[-1] == 0.5:
                    ClaseTabla.append(1)
                    
                elif prom[-1] > 0.75 and proboca[-1] > 0.45 and indi1[-1] == 1 and indB[-1] == 1:
                    ClaseTabla.append(1)
                else: 
                    ClaseTabla.append(0)
               
                print("Perclos: ",prom[-1],"Boca: ",proboca[-1],"Indi1: ",indi1[-1],"IndiB: ",indB[-1], "ClasTabla: ", ClaseTabla[-1])                
        draw_circle_eyes(image, relative_Ex,relative_Ey,relative_EyO
                         ,relative_EyP,relative_EyU,relative_EyU1)
               
        data = {'Bostezo' : proboca,'perCLOS' : prom, 'IndicadorE' : indi1, 'IndBostezo' : indB, 'Clase' : Clase, 
                'ClaseTabla' : ClaseTabla}
        df = pd.DataFrame(data, columns = ['Bostezo','perCLOS','IndicadorE','IndBostezo','Clase','ClaseTabla'])
        df.to_csv('77.csv')
            
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        

    # Flip the image horizontally for a selfie-view display.
    cv2.namedWindow('MediaPipe Face Mesh', 0);
    cv2.resizeWindow('MediaPipe Face Mesh', 1250,720);
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))    
    if cv2.waitKey(5) & 0xFF == 'q':
      break
cap.release()
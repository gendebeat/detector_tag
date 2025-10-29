import cv2
import numpy as np

import os
from datetime import datetime
import time

import random


from pupil_apriltags import Detector

# video_path = 'video.mp4'
video_path = '/dev/video0'
cap = cv2.VideoCapture(video_path)
    
# Obtener propiedades del video
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Archivo de video: {video_path}")
print(f"Resolución: {width}x{height}")
print(f"FPS: {fps:.2f}")
print(f"Total frames: {total_frames}")
print(f"Duración: {total_frames/fps:.2f} segundos")


# Crear detector
detector = Detector(families="tag36h11")

frame_count = 0
detection_count = 0
start_time = datetime.now()

# Definir tags id dimensions e offsets (id,tamaño,x,y)
offsets = [
    (0, 0.160, 0.05, -0.1),        
    (1, 0.1, 0.09, 0.04),       
    (2, 0.05, -0.02, 0.06),      
    (3, 0.025, 0, 0),       
]

colour_id = {}  # Make sure this dictionary is defined

for i, (tag_id, a, b, c) in enumerate(offsets):
    colour_id[tag_id] = (random.randrange(0, 256), random.randrange(0, 256), random.randrange(0, 256))

# Parámetros de la cámara (debes calibrar tu cámara)
camera_matrix = np.array([
    [800, 0, 320],
    [0, 800, 240],
    [0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.zeros(4)  # Sin distorsión


while True:

    ret, frame = cap.read()
    frame_count += 1

    if ret:
   
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar AprilTags
        results = detector.detect(gray)

        # Procesar resultados
        for detection in results:
            print(f"Tag ID: {detection.tag_id}")
            print(f"Centro: {detection.center}")
            print(f"Esquinas: {detection.corners}")

            # Dibujar todos los puntos de offset
            for i,(tag_id,tag_size,offset_x, offset_y) in enumerate(offsets):

            
                if (detection.tag_id==tag_id):

                    # Dibujar el contorno del tag
                    corners = detection.corners.astype(int)
                    cv2.polylines(frame, [corners], True, colour_id[tag_id], 2)

                    # Dibujar el centro 
                    center = tuple(detection.center.astype(int))

                    # Puntos 3D del tag en coordenadas del mundo
                    object_points = np.array([
                        [-tag_size/2, -tag_size/2, 0],
                        [ tag_size/2, -tag_size/2, 0],
                        [ tag_size/2,  tag_size/2, 0],
                        [-tag_size/2,  tag_size/2, 0]
                    ], dtype=np.float32)

                    # Resolver PnP para obtener pose
                    success, rvec, tvec = cv2.solvePnP(
                        object_points, 
                        detection.corners.astype(np.float32),
                        camera_matrix, 
                        dist_coeffs
                    )

                    if success:
                        print(f"Posición: {tvec.flatten()}")
                        print(f"Rotación: {rvec.flatten()}")

                        # target_x = int(center[0] + offset_x)
                        # target_y = int(center[1] + offset_y)
                        # target = (target_x,target_y)
                        # # Dibujar el punto
                        # cv2.circle(frame, target, 5, (0, 0, 255), -1)
                        # cv2.line(frame, center, target, (0, 0, 255), 1, cv2.LINE_AA)


                        # Mostrar ID
                        cv2.putText(frame, f"ID: {detection.tag_id}", 
                                    (corners[0][0], corners[0][1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_id[tag_id], 2)


                        # Proyectar ejes 3D
                        axis_length = tag_size/2
                        axis_points = np.float32([
                            [0, 0, 0],
                            [axis_length, 0, 0],
                            [0, axis_length, 0],
                            [0, 0, axis_length],
                            [offset_x,offset_y,0]
                        ]).reshape(-1, 3)

                        img_points, _ = cv2.projectPoints(
                            axis_points, rvec, tvec, camera_matrix, dist_coeffs
                        )

                        # Dibujar ejes
                        origin = tuple(img_points[0].ravel().astype(int))
                        x_axis = tuple(img_points[1].ravel().astype(int))
                        y_axis = tuple(img_points[2].ravel().astype(int))
                        z_axis = tuple(img_points[3].ravel().astype(int))
                        target_axis = tuple(img_points[4].ravel().astype(int))

                        cv2.arrowedLine(frame, origin, x_axis, (0, 0, 255), 3)
                        cv2.arrowedLine(frame, origin, y_axis, (0, 255, 0), 3)
                        cv2.arrowedLine(frame, origin, z_axis, (255, 0, 0), 3)
                        cv2.arrowedLine(frame, origin, target_axis, colour_id[tag_id], 3)


        cv2.imshow('Video', frame)
        # Salir con 'q' o ESC
        if cv2.waitKey(1000 // int(fps)) & 0xFF in [ord('q'), 27]:
            break
    else:
        # Si llegó al final del video
        break



# Liberar recursos
cap.release()
cv2.destroyAllWindows()
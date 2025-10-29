import cv2
import numpy as np

import os
from datetime import datetime
import time

import random

import json


from pupil_apriltags import Detector

# video_path = 'video.mp4'
video_path = '/dev/video0'
cap = cv2.VideoCapture(video_path)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)
    
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


detection_count = 0
start_time = datetime.now()


with open('config_tag.json', 'r', encoding='utf-8') as file:
    datos = json.load(file)

# Acceder a los datos
markers = datos['markers']

colour_id = {}  # Make sure this dictionary is defined
for tag in markers:
    colour_id[tag['id']] = (random.randrange(0, 256), random.randrange(0, 256), random.randrange(0, 256))



def load_calibration(filename="camera_calibration.yaml"):
        """Carga los parámetros de calibración desde un archivo"""
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        
        if not fs.isOpened():
            print(f"Error: No se puede abrir el archivo {filename}")
            return False
        
        cam = fs.getNode("camera_matrix").mat()
        dist = fs.getNode("distortion_coefficients").mat()
        fs.release()
        
        
        print(f"Calibración cargada desde: {filename}")
        return True, cam, dist

def test_calibration():
        """Prueba la calibración en tiempo real"""
        
        
        print("Probando calibración. Presiona 'q' para salir")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Corregir distorsión
            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
            
            # Mostrar imágenes original y corregida
            combined = np.hstack((frame, undistorted))
            cv2.putText(combined, "Original (Izquierda) vs Corregida (Derecha)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow('Prueba de Calibracion', combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()


camera_calibrated, camera_matrix, dist_coeffs = load_calibration()
test_calibration = test_calibration()


while True:

    ret, frame = cap.read()

    if ret:
   
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar AprilTags
        results = detector.detect(gray)

        #dibujar centro del frame
        frame_center = (int(width/2), int(height/2))
        cv2.circle(frame, frame_center, 5, (255, 255, 0), -1)


        # Procesar resultados
        for detection in results:
            print(f"Tag ID: {detection.tag_id}")
            print(f"Centro: {detection.center}")
            print(f"Esquinas: {detection.corners}")


            for tag in markers:

                if (detection.tag_id==tag['id']):

                    # Dibujar el contorno del tag
                    corners = detection.corners.astype(int)
                    cv2.polylines(frame, [corners], True, colour_id[tag['id']], 2)

                    # Dibujar el centro 
                    center = tuple(detection.center.astype(int))

                    # Puntos 3D del tag en coordenadas del mundo
                    object_points = np.array([
                        [-tag['size']/2, -tag['size']/2, 0],
                        [ tag['size']/2, -tag['size']/2, 0],
                        [ tag['size']/2,  tag['size']/2, 0],
                        [-tag['size']/2,  tag['size']/2, 0]
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
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_id[tag['id']], 2)

                     

                        # Proyectar ejes 3D
                        axis_length = tag['size']/2
                        axis_points = np.float32([
                            [0, 0, 0],
                            [axis_length, 0, 0],
                            [0, axis_length, 0],
                            [0, 0, axis_length],
                            [tag['offsetX'], tag['offsetY'], 0]
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
                        cv2.arrowedLine(frame, origin, target_axis, colour_id[tag['id']], 3)


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
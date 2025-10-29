import cv2
import numpy as np
import glob
import os

class CharucoCalibrator:
    def __init__(self, squaresX=10, squaresY=5, squareLength=0.04, markerLength=0.04, dictionary_size=100):
        """
        Inicializa el calibrador ChArUco
        
        Args:
            squaresX: Número de cuadrados en dirección X
            squaresY: Número de cuadrados en dirección Y
            squareLength: Longitud del lado del cuadrado en metros
            markerLength: Longitud del lado del marcador en metros
            dictionary_size: Tamaño del diccionario ArUco (50, 100, 250, 1000)
        """
        self.squaresX = squaresX
        self.squaresY = squaresY
        self.squareLength = squareLength
        self.markerLength = markerLength
        
        # Crear diccionario y tablero ChArUco
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
        self.board = cv2.aruco.CharucoBoard(
            (squaresX, squaresY), 
            squareLength, 
            markerLength, 
            self.dictionary
        )
        
        # Almacenar parámetros de calibración
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_success = False
        
    def create_charuco_board(self, output_path="charuco_board.png", dpi=300):
        """Crea y guarda el tablero ChArUco para imprimir"""
        size_pixels = int(8.27 * dpi)  # Tamaño A4 en pixels
        board_image = self.board.generateImage((size_pixels, int(size_pixels * 1.414)))
        cv2.imwrite(output_path, board_image)
        print(f"Tablero ChArUco guardado en: {output_path}")
        return board_image
    
    def capture_calibration_images(self, camera_id=0, num_images=20, output_dir="calibration_images"):
        """
        Captura imágenes para calibración desde la cámara USB
        
        Args:
            camera_id: ID de la cámara (0 para cámara por defecto)
            num_images: Número de imágenes a capturar
            output_dir: Directorio donde guardar las imágenes
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        cap = cv2.VideoCapture(camera_id)
        
        # Configurar resolución (opcional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)     
        cap.set(cv2.CAP_PROP_FPS, 30)       
        
        print("Presiona 's' para capturar imagen, 'q' para salir")
        print(f"Objetivo: {num_images} imágenes")
        
        image_count = 0
        saved_count = 0

        all_charuco_corners = []
        all_charuco_ids = []
        image_size = None
        
        while saved_count < num_images:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se puede capturar imagen de la cámara")
                break
            
            # Mostrar frame
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Imagenes capturadas: {saved_count}/{num_images}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(display_frame, "Presiona 's' para capturar, 'q' para salir", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 2)
            
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if image_size is None:
                image_size = gray.shape[::-1]  # (width, height)
            
            # Detectar marcadores
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.dictionary)
            
            if len(corners) > 0:
                # Interpolar esquinas de ChArUco
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, self.board
                )
                
                if ret > 0:
                    all_charuco_corners.append(charuco_corners)
                    all_charuco_ids.append(charuco_ids)
                    
                    # Visualizar detección (opcional)

                    cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
                    if charuco_corners is not None:
                        cv2.aruco.drawDetectedCornersCharuco(display_frame, charuco_corners, charuco_ids)
                    

            cv2.imshow('Calibracion - Captura de Imagenes', display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                # Guardar imagen
                filename = os.path.join(output_dir, f"calib_{saved_count:03d}.jpg")
                cv2.imwrite(filename, frame)
                saved_count += 1
                print(f"Imagen guardada: {filename}")
                
                
                # Mostrar confirmación
                cv2.putText(display_frame, "IMAGEN GUARDADA!", 
                           (frame.shape[1]//2 - 150, frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                cv2.imshow('Calibracion - Captura de Imagenes', display_frame)
                cv2.waitKey(500)
                
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Captura completada. {saved_count} imágenes guardadas en {output_dir}")
    
    def calibrate_camera(self, images_pattern="calibration_images/*.jpg"):
        """
        Calibra la cámara usando las imágenes capturadas
        
        Args:
            images_pattern: Patrón para buscar imágenes de calibración
        """
        # Buscar imágenes
        images = glob.glob(images_pattern)
        if not images:
            print(f"No se encontraron imágenes con el patrón: {images_pattern}")
            return False
        
        print(f"Encontradas {len(images)} imágenes para calibración")
        
        all_charuco_corners = []
        all_charuco_ids = []
        image_size = None
        
        for i, image_path in enumerate(images):
            print(f"Procesando imagen {i+1}/{len(images)}: {os.path.basename(image_path)}")
            
            # Leer imagen
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error al leer imagen: {image_path}")
                continue
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            if image_size is None:
                image_size = gray.shape[::-1]  # (width, height)
            
            # Detectar marcadores
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.dictionary)
            
            if len(corners) > 0:
                # Interpolar esquinas de ChArUco
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, self.board
                )
                
                if ret > 0:
                    all_charuco_corners.append(charuco_corners)
                    all_charuco_ids.append(charuco_ids)
                    
                    # Visualizar detección (opcional)
                    image_display = image.copy()
                    cv2.aruco.drawDetectedMarkers(image_display, corners, ids)
                    if charuco_corners is not None:
                        cv2.aruco.drawDetectedCornersCharuco(image_display, charuco_corners, charuco_ids)
                    
                    cv2.putText(image_display, f"Deteccion exitosa: {ret} puntos", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('Procesando Imagenes', image_display)
                    cv2.waitKey(300)
                else:
                    print(f"  No se pudieron interpolar esquinas ChArUco")
            else:
                print(f"  No se detectaron marcadores")
        
        cv2.destroyAllWindows()
        
        if len(all_charuco_corners) < 5:
            print(f"Error: Solo {len(all_charuco_corners)} imágenes válidas. Se necesitan al menos 5.")
            return False
        
        # Calibrar cámara
        print("Calibrando cámara...")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            all_charuco_corners, all_charuco_ids, self.board, image_size, None, None
        )
        
        if ret:
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            self.calibration_success = True
            
            print("¡Calibración exitosa!")
            print(f"Error de reproyección: {ret}")
            print("\nMatriz de la cámara:")
            print(camera_matrix)
            print("\nCoeficientes de distorsión:")
            print(dist_coeffs)
            
            return True
        else:
            print("Error en la calibración")
            return False
    
    def save_calibration(self, filename="camera_calibration.yaml"):
        """Guarda los parámetros de calibración en un archivo"""
        if not self.calibration_success:
            print("Error: La cámara no ha sido calibrada")
            return False
        
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", self.camera_matrix)
        fs.write("distortion_coefficients", self.dist_coeffs)
        fs.write("squaresX", self.squaresX)
        fs.write("squaresY", self.squaresY)
        fs.write("squareLength", self.squareLength)
        fs.write("markerLength", self.markerLength)
        fs.release()
        
        print(f"Calibración guardada en: {filename}")
        return True
    
    def load_calibration(self, filename="camera_calibration.yaml"):
        """Carga los parámetros de calibración desde un archivo"""
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        
        if not fs.isOpened():
            print(f"Error: No se puede abrir el archivo {filename}")
            return False
        
        self.camera_matrix = fs.getNode("camera_matrix").mat()
        self.dist_coeffs = fs.getNode("distortion_coefficients").mat()
        fs.release()
        
        self.calibration_success = True
        print(f"Calibración cargada desde: {filename}")
        return True
    
    def test_calibration(self, camera_id=0):
        """Prueba la calibración en tiempo real"""
        if not self.calibration_success:
            print("Error: La cámara no ha sido calibrada")
            return
        
        cap = cv2.VideoCapture(camera_id)
        # Configurar resolución (opcional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)     
        cap.set(cv2.CAP_PROP_FPS, 30)  
        
        print("Probando calibración. Presiona 'q' para salir")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Corregir distorsión
            undistorted = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
            
            # Mostrar imágenes original y corregida
            combined = np.hstack((frame, undistorted))
            cv2.putText(combined, "Original (Izquierda) vs Corregida (Derecha)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Prueba de Calibracion', combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Función principal para ejecutar la calibración completa"""
    # Crear calibrador
    calibrator = CharucoCalibrator(
        squaresX=11,
        squaresY=11,
        squareLength=0.095,  # 4 cm
        markerLength=0.075   # 2 cm
    )
    
    while True:
        print("\n=== CALIBRACIÓN DE CÁMARA USB CON ChArUco ===")
        print("1. Crear tablero ChArUco para imprimir")
        print("2. Capturar imágenes de calibración")
        print("3. Calibrar cámara")
        print("4. Probar calibración")
        print("5. Guardar calibración")
        print("6. Cargar calibración existente")
        print("7. Salir")
        
        choice = input("Selecciona una opción (1-7): ")
        
        if choice == '1':
            calibrator.create_charuco_board()
            print("Tablero creado. Imprímelo y úsalo para la calibración.")
            
        elif choice == '2':
            calibrator.capture_calibration_images(num_images=20)
            
        elif choice == '3':
            if calibrator.calibrate_camera():
                print("Calibración completada exitosamente!")
            else:
                print("Error en la calibración. Asegúrate de tener imágenes válidas.")
                
        elif choice == '4':
            calibrator.test_calibration()
            
        elif choice == '5':
            calibrator.save_calibration()
            
        elif choice == '6':
            calibrator.load_calibration()
            
        elif choice == '7':
            print("Saliendo...")
            break
            
        else:
            print("Opción no válida")

if __name__ == "__main__":
    main()
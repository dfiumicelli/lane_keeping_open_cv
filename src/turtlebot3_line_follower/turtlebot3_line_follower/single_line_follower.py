#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import math
import time
from collections import deque

class PIDLineFollower(Node):
    def __init__(self):
        super().__init__('pid_line_follower')

        # ===== ROS2 SETUP =====
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_sub = self.create_subscription(
            CompressedImage, '/image_rect/compressed', self.image_callback, 10)

        # ===== PARAMETRI PID =====
        # Tuning iniziale - modifica questi valori per ottimizzare le prestazioni
        self.Kp = 1.5           # Proporzionale: risposta immediata all'erroree
        self.Ki = 0.01          # Integrale: corregge errori accumulati nel tempo  
        self.Kd = 0.3          # Derivativo: predice errori futuri, riduce oscillazioni

        # ===== VELOCITÀ E LIMITI =====
        self.base_linear_speed = 0.12       # Velocità lineare base (m/s)
        self.max_linear_speed = 0.15        # Velocità lineare massima
        self.min_linear_speed = 0.12        # Velocità lineare minima
        self.max_angular_speed = 1.0        # Velocità angolare massima (rad/s)

        # ===== CONFIGURAZIONI COLORE =====
        self.white_line_config = {
            'lower_hsv': np.array([0, 0, 200]),
            'upper_hsv': np.array([180, 30, 255]),
            'name': 'white'
        }
        self.yellow_line_config = {
            'lower_hsv': np.array([20, 100, 100]),
            'upper_hsv': np.array([30, 255, 255]),
            'name': 'yellow'
        }

        # CONFIGURAZIONE ATTIVA
        self.active_config = self.yellow_line_config

        # ===== VARIABILI PID =====
        self.error_history = deque(maxlen=10)       # Storia degli errori per calcolo integrale
        self.previous_error = 0.0                   # Errore precedente per calcolo derivativo
        self.integral_error = 0.0                   # Somma degli errori per termine integrale
        self.previous_time = time.time()            # Timestamp per calcolo dt

        # ===== PARAMETRI CONTROLLO =====
        self.target_position = None                 # Posizione target (centro immagine)
        self.current_position = None                # Posizione attuale centroide
        self.setpoint = 0                          # Setpoint PID (errore desiderato = 0)

        # ===== PARAMETRI DETECTION =====
        self.min_contour_area = 1000               # Area minima contorno per validità
        self.roi_height_factor = 0.75              # Percentuale immagine da considerare
        self.line_lost_counter = 0                 # Contatore frame senza detection
        self.max_lost_frames = 20                  # Massimi frame persi prima di cercare

        # ===== CONTROLLO ADATTIVO =====
        self.speed_adaptation = True               # Abilita adattamento velocità
        self.curve_detection_threshold = 0.3      # Soglia per rilevare curve (rad)
        self.straight_line_confidence = 0.8       # Confidenza per andare dritto

        # ===== DEBUG E STATISTICHE =====
        self.debug_mode = True
        self.frame_count = 0
        self.pid_output_history = deque(maxlen=50)
        self.performance_stats = {
            'avg_error': 0,
            'max_error': 0,
            'control_effort': 0,
            'line_following_accuracy': 0
        }

        self.get_logger().info('=== PID LINE FOLLOWER INIZIALIZZATO ===')
        self.get_logger().info(f'Parametri PID: Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}')
        self.get_logger().info(f'Configurazione colore: {self.active_config["name"]}')
        self.get_logger().info(f'Velocità base: {self.base_linear_speed} m/s')

    def image_callback(self, msg):
        try:
            self.frame_count += 1
            current_time = time.time()
            dt = current_time - self.previous_time

            # Decodifica immagine compressa
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_image is None:
                return

            # Processamento immagine e detection linea
            processed_image, line_info = self.detect_line(cv_image)

            # Calcolo errore per controllo PID
            error = self.calculate_error(line_info, cv_image.shape[1])

            # Controllo PID
            pid_output = self.pid_control(error, dt)

            # Generazione comando Twist
            twist_cmd = self.generate_twist_command(pid_output, line_info)

            # Pubblicazione comando
            self.cmd_vel_pub.publish(twist_cmd)

            # Aggiornamento statistiche
            self.update_performance_stats(error, pid_output)

            # Debug visualization
            if self.debug_mode and self.frame_count % 3 == 0:
                self.show_pid_debug(processed_image, line_info, error, pid_output, cv_image.shape[1])

            self.previous_time = current_time

        except Exception as e:
            self.get_logger().error(f'Errore nel callback immagine: {e}')

    def detect_line(self, image):
        """Detection della linea con preprocessing avanzato"""
        height, width = image.shape[:2]
        self.target_position = width // 2

        # Preprocessing immagine
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Miglioramento contrasto per detection più robusta
        hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])

        # Creazione maschera colore
        mask = cv2.inRange(hsv, self.active_config['lower_hsv'], self.active_config['upper_hsv'])

        # Operazioni morfologiche per pulire la maschera
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # ROI - focus sulla parte inferiore dell'immagine
        roi_height = int(height * self.roi_height_factor)
        mask[:height-roi_height, :] = 0

        # Detection contorni
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        line_info = {
            'centroid': None,
            'area': 0,
            'angle': 0,
            'confidence': 0,
            'is_curve': False,
            'curvature': 0
        }

        if contours:
            # Filtra contorni per area minima
            valid_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]

            if valid_contours:
                # Prendi il contorno più grande (presumibilmente la linea)
                main_contour = max(valid_contours, key=cv2.contourArea)

                # Calcolo centroide usando momenti
                M = cv2.moments(main_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    line_info['centroid'] = (cx, cy)
                    line_info['area'] = cv2.contourArea(main_contour)

                    # Calcolo angolo della linea usando fitLine
                    try:
                        [vx, vy, x, y] = cv2.fitLine(main_contour, cv2.DIST_L2, 0, 0.01, 0.01)
                        angle = math.atan2(vy, vx)
                        line_info['angle'] = angle

                        # Detection curve basata sull'angolo
                        if abs(angle) > self.curve_detection_threshold:
                            line_info['is_curve'] = True
                            line_info['curvature'] = abs(angle)
                    except:
                        line_info['angle'] = 0

                    # Calcolo confidenza basata su area, posizione e forma
                    area_confidence = min(1.0, line_info['area'] / 5000.0)
                    position_confidence = 1.0 - abs(cx - self.target_position) / (width // 2)
                    compactness = self.calculate_contour_compactness(main_contour)

                    line_info['confidence'] = (area_confidence + position_confidence + compactness) / 3
                    self.line_lost_counter = 0
                else:
                    self.line_lost_counter += 1
            else:
                self.line_lost_counter += 1
        else:
            self.line_lost_counter += 1

        # Immagine debug
        debug_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if line_info['centroid']:
            color = (0, 255, 0) if not line_info['is_curve'] else (0, 255, 255)
            cv2.circle(debug_image, line_info['centroid'], 8, color, -1)

        return debug_image, line_info

    def calculate_contour_compactness(self, contour):
        """Calcola la compattezza del contorno per valutare la qualità"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0
        return 4 * math.pi * area / (perimeter * perimeter)

    def calculate_error(self, line_info, image_width):
        """Calcola l'errore per il controllo PID"""
        if line_info['centroid'] is None:
            # Se la linea è persa, mantieni l'ultimo errore noto
            if len(self.error_history) > 0:
                return self.error_history[-1]
            else:
                return 0.0

        # Errore = posizione desiderata - posizione attuale
        # Positivo = linea a destra, Negativo = linea a sinistra
        error = self.target_position - line_info['centroid'][0]

        # Normalizza l'errore rispetto alla larghezza dell'immagine
        normalized_error = error / (image_width / 2)

        return normalized_error

    def pid_control(self, error, dt):
        """Implementazione del controllo PID completo"""
        if dt <= 0:
            dt = 0.033  # Default 30 FPS

        # Aggiungi errore alla storia
        self.error_history.append(error)

        # ===== TERMINE PROPORZIONALE =====
        P = self.Kp * error

        # ===== TERMINE INTEGRALE =====
        # Accumula errori nel tempo, ma con anti-windup
        self.integral_error += error * dt

        # Anti-windup: limita l'integrale per evitare saturazione
        max_integral = 1.0 / max(self.Ki, 0.001)  # Evita divisione per zero
        self.integral_error = max(-max_integral, min(max_integral, self.integral_error))

        I = self.Ki * self.integral_error

        # ===== TERMINE DERIVATIVO =====
        # Calcola la derivata dell'errore
        derivative_error = (error - self.previous_error) / dt
        D = self.Kd * derivative_error

        # ===== OUTPUT PID =====
        pid_output = P + I + D

        # Limita l'output PID
        pid_output = max(-1.0, min(1.0, pid_output))

        # Salva per il prossimo ciclo
        self.previous_error = error
        self.pid_output_history.append(pid_output)

        return pid_output

    def generate_twist_command(self, pid_output, line_info):
        """Genera comando Twist basato sull'output PID"""
        twist = Twist()

        # Se la linea è persa da troppo tempo, comportamento di ricerca
        if self.line_lost_counter > self.max_lost_frames:
            twist.linear.x = 0.0
            twist.angular.z = 0.3 if pid_output >= 0 else -0.3
            return twist

        # ===== CONTROLLO VELOCITÀ LINEARE ADATTIVO =====
        if self.speed_adaptation and line_info.get('confidence', 0) > 0:
            # Riduci velocità in base all'errore assoluto e alla curvatura
            error_factor = 1.0 - min(abs(pid_output), 0.8)  # Più errore = meno velocità
            curve_factor = 1.0 - line_info.get('curvature', 0) * 0.5  # Più curva = meno velocità
            confidence_factor = line_info['confidence']  # Più confidenza = più velocità

            speed_multiplier = error_factor * curve_factor * confidence_factor
            speed_multiplier = max(0.3, min(1.0, speed_multiplier))  # Limiti [0.3, 1.0]

            target_speed = self.base_linear_speed * speed_multiplier
            twist.linear.x = max(self.min_linear_speed, min(self.max_linear_speed, target_speed))
        else:
            twist.linear.x = self.base_linear_speed

        # ===== CONTROLLO VELOCITÀ ANGOLARE =====
        # L'output PID diventa direttamente la velocità angolare
        twist.angular.z = pid_output * self.max_angular_speed

        # Assicurati che le velocità siano nei limiti
        twist.angular.z = max(-self.max_angular_speed, min(self.max_angular_speed, twist.angular.z))

        return twist

    def update_performance_stats(self, error, pid_output):
        """Aggiorna statistiche delle prestazioni"""
        if len(self.error_history) > 0:
            self.performance_stats['avg_error'] = sum(self.error_history) / len(self.error_history)
            self.performance_stats['max_error'] = max([abs(e) for e in self.error_history])

        if len(self.pid_output_history) > 0:
            self.performance_stats['control_effort'] = sum([abs(p) for p in self.pid_output_history]) / len(self.pid_output_history)

        # Accuratezza basata su errori piccoli
        small_errors = sum(1 for e in self.error_history if abs(e) < 0.1)
        if len(self.error_history) > 0:
            self.performance_stats['line_following_accuracy'] = (small_errors / len(self.error_history)) * 100

    def show_pid_debug(self, processed_image, line_info, error, pid_output, image_width):
        """Debug visualization avanzato con info PID"""

        # Informazioni PID e performance
        pid_info = [
            f'=== PID LINE FOLLOWER ===',
            f'Colore: {self.active_config["name"].upper()}',
            f'Frame: {self.frame_count}',
            f'',
            f'=== PARAMETRI PID ===',
            f'Kp: {self.Kp:.3f}',
            f'Ki: {self.Ki:.3f}', 
            f'Kd: {self.Kd:.3f}',
            f'',
            f'=== STATO CONTROLLO ===',
            f'Errore: {error:.3f}',
            f'PID Output: {pid_output:.3f}',
            f'Integrale: {self.integral_error:.3f}',
            f'Frames persi: {self.line_lost_counter}',
        ]

        if line_info['centroid']:
            pid_info.extend([
                f'',
                f'=== DETECTION ===',
                f'Centroide: {line_info["centroid"]}',
                f'Area: {line_info["area"]:.0f}',
                f'Confidenza: {line_info["confidence"]:.2f}',
                f'Curva: {"Sì" if line_info["is_curve"] else "No"}',
                f'Angolo: {line_info["angle"]*180/math.pi:.1f}°',
            ])

        pid_info.extend([
            f'',
            f'=== PERFORMANCE ===',
            f'Errore medio: {self.performance_stats["avg_error"]:.3f}',
            f'Errore max: {self.performance_stats["max_error"]:.3f}',
            f'Sforzo controllo: {self.performance_stats["control_effort"]:.3f}',
            f'Accuratezza: {self.performance_stats["line_following_accuracy"]:.1f}%',
        ])

        # Disegna informazioni sull'immagine
        y_offset = 20
        for line in pid_info:
            if line.startswith('==='):
                color = (0, 255, 255)  # Giallo per headers
                thickness = 2
            elif 'Errore:' in line or 'PID Output:' in line:
                color = (0, 255, 0) if abs(error) < 0.2 else (0, 0, 255)  # Verde/Rosso
                thickness = 1
            else:
                color = (255, 255, 255)  # Bianco per info normali
                thickness = 1

            cv2.putText(processed_image, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, thickness)
            y_offset += 15

        # Disegna linee di riferimento
        height = processed_image.shape[0]
        center = image_width // 2

        # Linea centrale
        cv2.line(processed_image, (center, 0), (center, height), (255, 255, 0), 2)

        # Linee di errore
        error_pixel = int(error * center)
        target_pos = center + error_pixel
        cv2.line(processed_image, (target_pos, 0), (target_pos, height), (0, 255, 255), 1)

        cv2.imshow('PID Line Follower Debug', processed_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)

    pid_follower = PIDLineFollower()

    try:
        rclpy.spin(pid_follower)
    except KeyboardInterrupt:
        pass
    finally:
        pid_follower.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

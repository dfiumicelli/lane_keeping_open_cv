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

class LaneKeeping(Node):
    def __init__(self):
        super().__init__('lane_keeping')

        # ===== ROS2 SETUP =====
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_sub = self.create_subscription(
            CompressedImage, '/image_rect/compressed', self.image_callback, 10)

        # ===== PARAMETRI PID =====
        self.Kp = 2.0
        self.Ki = 0
        self.Kd = 0.3

        # ===== VELOCITÀ E LIMITI =====
        self.base_linear_speed = 0.10
        self.max_linear_speed = 0.16
        self.min_linear_speed = 0.04
        self.max_angular_speed = 1.0

        # ===== CONFIGURAZIONI COLORI =====
        self.yellow_config = {
            'lower_hsv': np.array([18, 80, 80]),
            'upper_hsv': np.array([32, 255, 255]),
            'name': 'yellow_left',
            'side': 'left'
        }

        self.white_config = {
            'lower_hsv': np.array([0, 0, 230]),
            'upper_hsv': np.array([180, 20, 255]),
            'name': 'white_right', 
            'side': 'right'
        }

        # ===== VARIABILI PID =====
        self.error_history = deque(maxlen=15)
        self.previous_error = 0.0
        self.integral_error = 0.0
        self.previous_time = time.time()

        # ===== PARAMETRI DETECTION =====
        self.min_contour_area = 600
        self.roi_height_factor = 0.8

        # ===== GESTIONE PERDITA LINEE =====
        self.line_lost_counter = {'yellow': 0, 'white': 0}
        self.max_lost_frames = 15

        # Storia posizioni per predizione
        self.position_history = {
            'yellow': deque(maxlen=10),
            'white': deque(maxlen=10)
        }

        # ===== PARAMETRI LANE KEEPING CORRETTI =====
        self.lane_width_pixels = 380                # Larghezza media corsia
        self.lane_width_tolerance = 120
        self.adaptive_lane_width = True

        # ===== PARAMETRI SPECIFICI PER POSIZIONAMENTO =====
        # Distanza dalla linea gialla quando in modalità yellow-only
        self.yellow_follow_distance = 200          # Pixel a DESTRA della gialla
        self.yellow_follow_tolerance = 30          # Tolleranza posizionamento

        # Distanza dalla linea bianca quando in modalità white-only  
        self.white_follow_distance = 150           # Pixel a SINISTRA della bianca
        self.white_follow_tolerance = 25

        # ===== PREDIZIONE E TRACKING =====
        self.lane_center_history = deque(maxlen=12)
        self.lane_width_history = deque(maxlen=8)

        # Kalman filter per tracking
        self.kalman_state = {
            'yellow': {'position': None, 'velocity': 0.0},
            'white': {'position': None, 'velocity': 0.0}
        }
        self.kalman_gain = 0.3

        # ===== SMOOTHING =====
        self.velocity_smoothing = True
        self.previous_linear_velocity = 0.0
        self.previous_angular_velocity = 0.0
        self.velocity_smooth_factor = 0.7

        self.roi_vertical_start = 0.4    # Inizia dal 40% dell'altezza (regolabile)
        self.roi_min_y_centroid = 0.4

        # ===== DEBUG =====
        self.debug_mode = True
        self.frame_count = 0
        self.detection_stats = {
            'yellow_detection_rate': 0,
            'white_detection_rate': 0,
            'dual_detection_rate': 0,
            'yellow_follow_rate': 0,      # NUOVO: % tempo in modalità yellow-follow
            'white_follow_rate': 0,       # NUOVO: % tempo in modalità white-follow
            'positioning_accuracy': 0
        }

        self.get_logger().info('=== CORRECTED DUAL-LINE LANE KEEPING ===')
        self.get_logger().info('MIGLIORAMENTO: Posizionamento corretto rispetto alle singole linee')
        self.get_logger().info(f'Yellow follow: {self.yellow_follow_distance}px a DESTRA')
        self.get_logger().info(f'White follow: {self.white_follow_distance}px a SINISTRA')

    def image_callback(self, msg):
        try:
            self.frame_count += 1
            current_time = time.time()
            dt = current_time - self.previous_time

            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_image is None:
                return

            # Detection con tracking predittivo
            processed_image, lane_info = self.detect_dual_lines_with_tracking(cv_image)

            # CORREZIONE: Calcola errore con posizionamento corretto per singole linee
            lane_center_error = self.calculate_corrected_lane_error(lane_info, cv_image.shape[1])

            # Controllo PID
            pid_output = self.pid_control(lane_center_error, dt)

            # Generazione comando
            twist_cmd = self.generate_smooth_twist(pid_output, lane_info)

            self.cmd_vel_pub.publish(twist_cmd)

            # Aggiornamento tracking
            self.update_tracking_state(lane_info)

            # Debug
            if self.debug_mode and self.frame_count % 3 == 0:
                self.show_corrected_debug(processed_image, lane_info, lane_center_error, 
                                        pid_output, cv_image.shape[1])

            self.previous_time = current_time

        except Exception as e:
            self.get_logger().error(f'Errore nel callback: {e}')

    def detect_dual_lines_with_tracking(self, image):
        """Detection con tracking predittivo (stesso del sistema precedente)"""
        height, width = image.shape[:2]

        # Preprocessing migliorato
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        hsv[:,:,2] = clahe.apply(hsv[:,:,2])

        roi_start_height = int(height * 0.4)  # Inizia dal 40% dell'altezza
        roi_height = height - roi_start_height  # Fino al fondo

        # === DETECTION GIALLA con ROI limitata ===
        yellow_mask = cv2.inRange(hsv, self.yellow_config['lower_hsv'], 
                                self.yellow_config['upper_hsv'])
        
        # MODIFICA: Applica ROI verticale (zera tutto sopra roi_start_height)
        yellow_mask[:roi_start_height, :] = 0

        if self.line_lost_counter['yellow'] > 5:
            yellow_mask[:int(roi_start_height * 0.7), :] = 0  
        else:
            yellow_mask[:, int(width*0.6):] = 0

        kernel_yellow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel_yellow)
        yellow_mask = cv2.dilate(yellow_mask, kernel_yellow, iterations=1)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, 
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        # === DETECTION BIANCA con ROI limitata ===
        white_mask = cv2.inRange(hsv, self.white_config['lower_hsv'], 
                                self.white_config['upper_hsv'])
        
        # MODIFICA: Applica ROI verticale (zera tutto sopra roi_start_height)
        white_mask[:roi_start_height, :] = 0

        if self.line_lost_counter['white'] > 5:
            white_mask[:int(roi_start_height * 0.7), :] = 0
        else:
            white_mask[:, :int(width*0.4)] = 0

        kernel_white = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_white)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_white)

        # === ANALISI CON PREDIZIONE ===
        lane_info = {
            'yellow_line': None,
            'white_line': None,
            'lane_center': None,
            'target_position': None,    # NUOVO: Posizione target specifica
            'lane_width': 0,
            'confidence': 0,
            'detection_mode': 'none',
            'using_prediction': False
        }

        # Analizza gialla
        yellow_info = self.analyze_line_with_prediction(yellow_mask, self.yellow_config, 'yellow')
        if yellow_info['centroid']:
            lane_info['yellow_line'] = yellow_info
            self.update_kalman_filter('yellow', yellow_info['centroid'][0])
            self.position_history['yellow'].append(yellow_info['centroid'])
            self.line_lost_counter['yellow'] = 0
        else:
            self.line_lost_counter['yellow'] += 1
            if self.line_lost_counter['yellow'] <= self.max_lost_frames:
                predicted_yellow = self.predict_line_position('yellow', height)
                if predicted_yellow:
                    lane_info['yellow_line'] = {
                        'centroid': predicted_yellow,
                        'confidence': 0.3,
                        'predicted': True
                    }
                    lane_info['using_prediction'] = True

        # Analizza bianca
        white_info = self.analyze_line_with_prediction(white_mask, self.white_config, 'white')
        if white_info['centroid']:
            lane_info['white_line'] = white_info
            self.update_kalman_filter('white', white_info['centroid'][0])
            self.position_history['white'].append(white_info['centroid'])
            self.line_lost_counter['white'] = 0
        else:
            self.line_lost_counter['white'] += 1
            if self.line_lost_counter['white'] <= self.max_lost_frames:
                predicted_white = self.predict_line_position('white', height)
                if predicted_white:
                    lane_info['white_line'] = {
                        'centroid': predicted_white,
                        'confidence': 0.3,
                        'predicted': True
                    }
                    lane_info['using_prediction'] = True

        # CALCOLO POSIZIONE TARGET CORRETTA
        lane_info = self.calculate_corrected_target_position(lane_info, width)

        debug_image = np.zeros_like(image)
        debug_image[:,:,0] = white_mask
        debug_image[:,:,1] = yellow_mask
        debug_image[:,:,2] = cv2.bitwise_or(yellow_mask, white_mask)

        return debug_image, lane_info

    def analyze_line_with_prediction(self, mask, config, line_name):
        """Analisi linea (stesso del sistema precedente)"""
        line_info = {
            'centroid': None,
            'area': 0,
            'confidence': 0,
            'angle': 0,
            'predicted': False
        }

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # NUOVO: Filtra contorni per posizione verticale
            image_height = mask.shape[0]
            min_y_position = int(image_height * 0.4)  # Solo contorni sotto il 40%
            
            valid_contours = []
            for c in contours:
                if cv2.contourArea(c) > self.min_contour_area:
                    # Verifica posizione verticale del centroide
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cy = int(M["m01"] / M["m00"])
                        # Accetta solo se il centroide è nella parte bassa
                        if cy >= min_y_position:
                            valid_contours.append(c)

            # Fallback: se nessun contorno valido, prendi il più grande
            if not valid_contours and contours:
                largest = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cy = int(M["m01"] / M["m00"])
                    # Solo se non troppo in alto
                    if cy >= int(image_height * 0.3):
                        valid_contours = [largest]

            if not valid_contours and contours:
                valid_contours = [max(contours, key=cv2.contourArea)]

            if valid_contours:
                if line_name == 'yellow':
                    if len(valid_contours) > 1:
                        combined_points = np.vstack([c.reshape(-1, 2) for c in valid_contours])
                        main_contour = combined_points.reshape(-1, 1, 2)
                    else:
                        main_contour = valid_contours[0]
                else:
                    main_contour = max(valid_contours, key=cv2.contourArea)

                M = cv2.moments(main_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    line_info['centroid'] = (cx, cy)
                    line_info['area'] = cv2.contourArea(main_contour)

                    area_confidence = min(1.0, line_info['area'] / 2500.0)
                    temporal_confidence = 1.0
                    if len(self.position_history[line_name]) > 0:
                        last_pos = self.position_history[line_name][-1]
                        distance = abs(cx - last_pos[0])
                        temporal_confidence = max(0.3, 1.0 - distance / 100.0)

                    line_info['confidence'] = (area_confidence + temporal_confidence) / 2

        return line_info

    def update_kalman_filter(self, line_name, position):
        """Filtro Kalman (stesso del sistema precedente)"""
        state = self.kalman_state[line_name]

        if state['position'] is None:
            state['position'] = position
            state['velocity'] = 0.0
        else:
            predicted_pos = state['position'] + state['velocity']
            innovation = position - predicted_pos
            state['position'] = predicted_pos + self.kalman_gain * innovation
            state['velocity'] += self.kalman_gain * innovation * 0.1

    def predict_line_position(self, line_name, image_height):
        """Predizione posizione (stesso del sistema precedente)"""
        if len(self.position_history[line_name]) < 2:
            return None

        positions = list(self.position_history[line_name])

        if len(positions) >= 3:
            x_coords = [p[0] for p in positions[-3:]]
            y_coords = [p[1] for p in positions[-3:]]

            if len(set(y_coords)) > 1:
                dx = x_coords[-1] - x_coords[0]
                dy = y_coords[-1] - y_coords[0]
                if dy != 0:
                    slope = dx / dy
                    pred_y = int(image_height * 0.8)
                    pred_x = x_coords[-1] + slope * (pred_y - y_coords[-1])

                    if self.kalman_state[line_name]['position']:
                        kalman_pred = self.kalman_state[line_name]['position']
                        pred_x = 0.6 * pred_x + 0.4 * kalman_pred

                    return (int(pred_x), pred_y)

        if self.kalman_state[line_name]['position']:
            last_pos = positions[-1] if positions else None
            if last_pos:
                return (int(self.kalman_state[line_name]['position']), last_pos[1])

        return None

    def calculate_corrected_target_position(self, lane_info, image_width):
        """CORREZIONE: Calcola posizione target corretta per ogni modalità"""

        if lane_info['yellow_line'] and lane_info['white_line']:
            # ===== MODALITÀ DUAL: Centro geometrico tra le linee =====
            yellow_x = lane_info['yellow_line']['centroid'][0]
            white_x = lane_info['white_line']['centroid'][0]

            # Centro corsia = punto medio
            target_x = (yellow_x + white_x) // 2
            target_y = (lane_info['yellow_line']['centroid'][1] + 
                       lane_info['white_line']['centroid'][1]) // 2

            lane_info['target_position'] = (target_x, target_y)
            lane_info['lane_center'] = (target_x, target_y)  # Per compatibilità
            lane_info['lane_width'] = abs(white_x - yellow_x)
            lane_info['detection_mode'] = 'dual'

            yellow_conf = lane_info['yellow_line'].get('confidence', 0.3)
            white_conf = lane_info['white_line'].get('confidence', 0.3)
            if lane_info['yellow_line'].get('predicted') or lane_info['white_line'].get('predicted'):
                lane_info['confidence'] = (yellow_conf + white_conf) / 2 * 0.7
            else:
                lane_info['confidence'] = (yellow_conf + white_conf) / 2

        elif lane_info['yellow_line']:
            # ===== MODALITÀ YELLOW-ONLY: Posizionati a DESTRA della gialla =====
            yellow_x = lane_info['yellow_line']['centroid'][0]
            yellow_y = lane_info['yellow_line']['centroid'][1]

            # Target = gialla_x + distanza_seguimento (a destra della gialla)
            target_x = yellow_x + self.yellow_follow_distance
            target_y = yellow_y

            lane_info['target_position'] = (target_x, target_y)
            lane_info['lane_center'] = (target_x, target_y)  # Per compatibilità
            lane_info['lane_width'] = self.yellow_follow_distance * 2  # Stima
            lane_info['detection_mode'] = 'yellow_follow'  # Nome più chiaro
            lane_info['confidence'] = lane_info['yellow_line'].get('confidence', 0.3) * 0.8

        elif lane_info['white_line']:
            # ===== MODALITÀ WHITE-ONLY: Posizionati a SINISTRA della bianca =====
            white_x = lane_info['white_line']['centroid'][0]
            white_y = lane_info['white_line']['centroid'][1]

            # Target = bianca_x - distanza_seguimento (a sinistra della bianca)
            target_x = white_x - self.white_follow_distance
            target_y = white_y

            lane_info['target_position'] = (target_x, target_y)
            lane_info['lane_center'] = (target_x, target_y)  # Per compatibilità
            lane_info['lane_width'] = self.white_follow_distance * 2  # Stima
            lane_info['detection_mode'] = 'white_follow'  # Nome più chiaro
            lane_info['confidence'] = lane_info['white_line'].get('confidence', 0.3) * 0.8

        else:
            # ===== MODALITÀ MEMORY: Usa ultima posizione target nota =====
            if len(self.lane_center_history) > 0:
                last_target = self.lane_center_history[-1]
                lane_info['target_position'] = last_target
                lane_info['lane_center'] = last_target
                lane_info['detection_mode'] = 'memory'
                lane_info['confidence'] = 0.2

        return lane_info

    def calculate_corrected_lane_error(self, lane_info, image_width):
        """CORREZIONE: Calcola errore basato sulla posizione target corretta"""
        image_center = image_width // 2

        if lane_info.get('target_position'):
            target_x = lane_info['target_position'][0]

            # Errore = centro immagine - posizione target
            error = (image_center - target_x) / (image_width / 2)

            # Smoothing più aggressivo per modalità singola linea
            if lane_info['detection_mode'] in ['yellow_follow', 'white_follow']:
                if len(self.error_history) > 2:
                    recent_errors = list(self.error_history)[-3:]
                    avg_error = sum(recent_errors) / len(recent_errors)
                    error = 0.4 * error + 0.6 * avg_error  # Più smoothing
            elif lane_info.get('using_prediction', False):
                if len(self.error_history) > 3:
                    recent_errors = list(self.error_history)[-5:]
                    avg_error = sum(recent_errors) / len(recent_errors)
                    error = 0.4 * error + 0.6 * avg_error

            # Aggiorna storia
            if lane_info.get('target_position'):
                self.lane_center_history.append(lane_info['target_position'])
            if lane_info['lane_width'] > 50:  # Solo se larghezza ragionevole
                self.lane_width_history.append(lane_info['lane_width'])

            return error

        # Fallback: trend degli ultimi errori
        if len(self.error_history) >= 3:
            recent_errors = list(self.error_history)[-3:]
            trend = (recent_errors[-1] - recent_errors[0]) / 2
            last_error = recent_errors[-1]
            return last_error + trend * 0.3

        return 0.0

    def pid_control(self, error, dt):
        """Controllo PID con anti-windup migliorato"""
        if dt <= 0:
            dt = 0.033

        self.error_history.append(error)

        P = self.Kp * error

        self.integral_error += error * dt
        max_integral = 0.8 / max(self.Ki, 0.001)
        self.integral_error = max(-max_integral, min(max_integral, self.integral_error))

        if len(self.error_history) >= 2:
            if abs(self.error_history[-1] - self.error_history[-2]) > 0.5:
                self.integral_error *= 0.5

        I = self.Ki * self.integral_error

        derivative_error = (error - self.previous_error) / dt
        D = self.Kd * derivative_error

        pid_output = P + I + D
        pid_output = max(-1.0, min(1.0, pid_output))

        self.previous_error = error

        return pid_output

    def generate_smooth_twist(self, pid_output, lane_info):
        """Genera comando con smoothing"""
        twist = Twist()

        confidence = lane_info.get('confidence', 0.1)

        # Fattori velocità per modalità
        if lane_info['detection_mode'] == 'dual':
            speed_factor = 0.9 * confidence
        elif lane_info['detection_mode'] in ['yellow_follow', 'white_follow']:
            # Modalità singola linea: più conservativa ma non troppo lenta
            speed_factor = 0.75 * confidence  # Era 0.7
        elif lane_info.get('using_prediction'):
            speed_factor = 0.55 * confidence  # Era 0.5
        else:
            speed_factor = 0.25

        error_factor = 1.0 - min(abs(pid_output), 0.7)
        final_speed_factor = speed_factor * error_factor
        final_speed_factor = max(0.25, min(1.0, final_speed_factor))

        target_linear = self.base_linear_speed * final_speed_factor
        target_linear = max(self.min_linear_speed, min(self.max_linear_speed, target_linear))

        target_angular = pid_output * self.max_angular_speed
        target_angular = max(-self.max_angular_speed, min(self.max_angular_speed, target_angular))

        # Smoothing velocità
        if self.velocity_smoothing:
            twist.linear.x = (self.velocity_smooth_factor * self.previous_linear_velocity + 
                            (1 - self.velocity_smooth_factor) * target_linear)
            twist.angular.z = (self.velocity_smooth_factor * self.previous_angular_velocity + 
                             (1 - self.velocity_smooth_factor) * target_angular)
        else:
            twist.linear.x = target_linear
            twist.angular.z = target_angular

        self.previous_linear_velocity = twist.linear.x
        self.previous_angular_velocity = twist.angular.z

        return twist

    def update_tracking_state(self, lane_info):
        """Aggiorna statistiche con nuove modalità"""
        total_frames = max(1, self.frame_count)

        if not hasattr(self, '_stats_counters'):
            self._stats_counters = {'yellow': 0, 'white': 0, 'dual': 0, 
                                  'yellow_follow': 0, 'white_follow': 0, 'prediction': 0}

        if lane_info['yellow_line']:
            self._stats_counters['yellow'] += 1
        if lane_info['white_line']:
            self._stats_counters['white'] += 1
        if lane_info['detection_mode'] == 'dual':
            self._stats_counters['dual'] += 1
        elif lane_info['detection_mode'] == 'yellow_follow':
            self._stats_counters['yellow_follow'] += 1
        elif lane_info['detection_mode'] == 'white_follow':
            self._stats_counters['white_follow'] += 1
        if lane_info.get('using_prediction'):
            self._stats_counters['prediction'] += 1

        self.detection_stats['yellow_detection_rate'] = (self._stats_counters['yellow'] / total_frames) * 100
        self.detection_stats['white_detection_rate'] = (self._stats_counters['white'] / total_frames) * 100
        self.detection_stats['dual_detection_rate'] = (self._stats_counters['dual'] / total_frames) * 100
        self.detection_stats['yellow_follow_rate'] = (self._stats_counters['yellow_follow'] / total_frames) * 100
        self.detection_stats['white_follow_rate'] = (self._stats_counters['white_follow'] / total_frames) * 100

    def show_corrected_debug(self, processed_image, lane_info, error, pid_output, image_width):
        """Debug con visualizzazione posizione target"""

        # Disegna centroidi rilevati
        if lane_info['yellow_line']:
            color = (0, 255, 255) if not lane_info['yellow_line'].get('predicted') else (0, 200, 200)
            cv2.circle(processed_image, lane_info['yellow_line']['centroid'], 10, color, -1)

        if lane_info['white_line']:
            color = (255, 255, 255) if not lane_info['white_line'].get('predicted') else (200, 200, 200)
            cv2.circle(processed_image, lane_info['white_line']['centroid'], 10, color, -1)

        # NUOVO: Disegna posizione TARGET (dove vuole stare il robot)
        if lane_info.get('target_position'):
            target_color = (255, 0, 255)  # Magenta per target
            if lane_info['detection_mode'] == 'yellow_follow':
                target_color = (255, 255, 0)  # Cyan per yellow-follow
            elif lane_info['detection_mode'] == 'white_follow':
                target_color = (255, 0, 0)    # Blu per white-follow

            cv2.circle(processed_image, lane_info['target_position'], 15, target_color, -1)

            # Linea da linea rilevata a target
            if lane_info['detection_mode'] == 'yellow_follow' and lane_info['yellow_line']:
                cv2.line(processed_image, lane_info['yellow_line']['centroid'], 
                        lane_info['target_position'], (0, 255, 255), 2)
                # Testo "200px" sulla linea
                mid_x = (lane_info['yellow_line']['centroid'][0] + lane_info['target_position'][0]) // 2
                mid_y = (lane_info['yellow_line']['centroid'][1] + lane_info['target_position'][1]) // 2
                cv2.putText(processed_image, f'{self.yellow_follow_distance}px', (mid_x, mid_y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            elif lane_info['detection_mode'] == 'white_follow' and lane_info['white_line']:
                cv2.line(processed_image, lane_info['white_line']['centroid'], 
                        lane_info['target_position'], (255, 255, 255), 2)
                mid_x = (lane_info['white_line']['centroid'][0] + lane_info['target_position'][0]) // 2
                mid_y = (lane_info['white_line']['centroid'][1] + lane_info['target_position'][1]) // 2
                cv2.putText(processed_image, f'{self.white_follow_distance}px', (mid_x, mid_y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Centro immagine
        height = processed_image.shape[0]
        center = image_width // 2
        cv2.line(processed_image, (center, 0), (center, height), (0, 255, 0), 2)

        # Info debug
        debug_info = [
            f'=== LANE KEEPING ===',
            f'Frame: {self.frame_count}',
            f'',
            f'=== DETECTION & TARGET ===',
            f'Mode: {lane_info["detection_mode"].upper()}',
        ]

        # Info specifiche per modalità
        if lane_info['detection_mode'] == 'dual':
            debug_info.append(f'Target: Centro tra linee')
        elif lane_info['detection_mode'] == 'yellow_follow':
            debug_info.append(f'Target: {self.yellow_follow_distance}px a DESTRA di gialla')
        elif lane_info['detection_mode'] == 'white_follow':
            debug_info.append(f'Target: {self.white_follow_distance}px a SINISTRA di bianca')

        debug_info.extend([
            f'Using Prediction: {"YES" if lane_info.get("using_prediction") else "NO"}',
            f'Yellow Lost: {self.line_lost_counter["yellow"]}',
            f'White Lost: {self.line_lost_counter["white"]}',
            f'Confidence: {lane_info["confidence"]:.2f}',
            f'',
            f'=== CONTROL ===',
            f'Error: {error:.3f}',
            f'PID Output: {pid_output:.3f}',
            f'Linear Vel: {self.previous_linear_velocity:.2f}',
            f'Angular Vel: {self.previous_angular_velocity:.2f}',
            f'',
            f'=== STATISTICS ===',
            f'Dual Rate: {self.detection_stats["dual_detection_rate"]:.1f}%',
            f'Yellow Follow: {self.detection_stats["yellow_follow_rate"]:.1f}%',
            f'White Follow: {self.detection_stats["white_follow_rate"]:.1f}%',
        ])

        # Target position info
        if lane_info.get('target_position'):
            debug_info.append(f'Target Pos: {lane_info["target_position"]}')

        height = processed_image.shape[0]
        roi_line_y = int(height * self.roi_vertical_start)
        cv2.line(processed_image, (0, roi_line_y), (image_width, roi_line_y), 
                (0, 255, 0), 2)  # Linea verde
        cv2.putText(processed_image, 'ROI LIMIT', (10, roi_line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Disegna testo con colori appropriati
        y_offset = 20
        for line in debug_info:
            if line.startswith('==='):
                color = (0, 255, 255)
                thickness = 2
            elif 'DUAL' in line:
                color = (0, 255, 0)
                thickness = 2
            elif 'YELLOW_FOLLOW' in line:
                color = (0, 255, 255)  # Cyan
                thickness = 2
            elif 'WHITE_FOLLOW' in line:
                color = (255, 0, 0)    # Blu
                thickness = 2
            elif 'Target:' in line and 'DESTRA' in line:
                color = (0, 255, 255)  # Cyan per destra gialla
                thickness = 2
            elif 'Target:' in line and 'SINISTRA' in line:
                color = (255, 0, 0)    # Blu per sinistra bianca
                thickness = 2
            elif any(keyword in line for keyword in ['Lost:', 'Error:', 'PID']):
                if 'Lost' in line and ('0' not in line or int([c for c in line if c.isdigit()][0]) > 5):
                    color = (0, 0, 255)  # Rosso per perso
                else:
                    color = (255, 255, 255)
                thickness = 1
            else:
                color = (255, 255, 255)
                thickness = 1

            cv2.putText(processed_image, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, thickness)
            y_offset += 15

        cv2.imshow('Lane Keeping Debug', processed_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)

    lane_keeper = LaneKeeping()

    try:
        rclpy.spin(lane_keeper)
    except KeyboardInterrupt:
        pass
    finally:
        lane_keeper.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

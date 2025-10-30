from django.shortcuts import render
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from django.http import StreamingHttpResponse as streaming_http_response
import cv2
import time
from mediapipe.tasks import python
# Create your views here.

etiquetas_es = {
    "person": "persona", "bicycle": "bicicleta", "car": "auto", "motorcycle": "motocicleta",
    "airplane": "avión", "bus": "autobús", "train": "tren", "truck": "camión", "boat": "barco",
    "traffic light": "semáforo", "fire hydrant": "hidrante", "stop sign": "señal de alto",
    "bench": "banco", "bird": "pájaro", "cat": "gato", "dog": "perro", "horse": "caballo",
    "sheep": "oveja", "cow": "vaca", "elephant": "elefante", "bear": "oso", "zebra": "cebra",
    "giraffe": "jirafa", "bottle": "botella", "cup": "taza", "fork": "tenedor", "knife": "cuchillo",
    "spoon": "cuchara", "bowl": "cuenco", "banana": "banana", "apple": "manzana",
    "orange": "naranja", "pizza": "pizza", "donut": "rosquilla", "cake": "pastel",
    "chair": "silla", "sofa": "sofá", "potted plant": "planta en maceta", "tv": "televisor",
    "laptop": "portátil", "mouse": "ratón", "keyboard": "teclado", "cell phone": "telefono"
}

# --------------------------------------------------------
# Detector y configuración
# --------------------------------------------------------
BaseOptions = python.BaseOptions
detection_result_list = []

def detection_callback(result, output_image, timestamp_ms):
    detection_result_list.append(result)

def crear_detector(conf_threshold=0.5):
    options = vision.ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path="efficientdet_lite0.tflite"),
        max_results=3,
        score_threshold=conf_threshold,
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=detection_callback)
    return vision.ObjectDetector.create_from_options(options)

# --------------------------------------------------------
# Generador de frames
# --------------------------------------------------------
def generar_frames():
    confidence_threshold = 0.5
    detector = crear_detector(confidence_threshold)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo acceder a la cámara.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        if detection_result_list:
            for detection in detection_result_list[0].detections:
                bbox = detection.bounding_box
                x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                category = detection.categories[0]
                score = category.score * 100
                name_en = category.category_name
                name_es = etiquetas_es.get(name_en, name_en)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y - 35), (x + w, y), (0, 255, 0), -1)
                cv2.putText(frame, f"{name_es} {score:.1f}%", (x + 5, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

            detection_result_list.clear()

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # 👇 Pausa para reducir velocidad (ajusta entre 0.05 y 0.2 segundos)
        time.sleep(0.1)

    cap.release()

def video_feed(request):
    return streaming_http_response(generar_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def index(request):
    return render(request, 'index.html')
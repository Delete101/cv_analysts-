import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog

clicks = {'image1': [], 'image2': []}
images = {'image1': None, 'image2': None}


def adjust_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def optimize_image_contrast(image, clip_limit=3.0):
    if image.ndim == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        result = clahe.apply(image)
    return result


def align_and_crop(transformed_image, reference_image, trans_first_point, ref_first_point):
    # Вычисляем смещение между первыми точками после трансформации
    dy, dx = np.array(ref_first_point) - np.array(trans_first_point)

    # Применяем смещение к трансформированному изображению
    rows, cols = transformed_image.shape[:2]
    M_trans = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned_image = cv2.warpAffine(transformed_image, M_trans, (cols, rows))

    # Обрезка изображения до размеров reference_image, если оно выходит за его пределы
    aligned_cropped_image = aligned_image[:reference_image.shape[0], :reference_image.shape[1]]

    return aligned_cropped_image




def optimize_image_contrast(image, clip_limit=3.0):
    if image.ndim == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        result = clahe.apply(image)
    return result



def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicks[param]) < 2:
            clicks[param].append((x, y))
            color = (255, 0, 0) if len(clicks[param]) == 1 else (0, 255, 0)
            cv2.circle(images[param], (x, y), 5, color, -1)
            cv2.imshow(param, images[param])


def calculate_transformation(points1, points2):
    p1, p2 = np.array(points1[0]), np.array(points1[1])
    q1, q2 = np.array(points2[0]), np.array(points2[1])

    vec_p = p2 - p1
    vec_q = q2 - q1
    cos_angle = np.dot(vec_p, vec_q) / (np.linalg.norm(vec_p) * np.linalg.norm(vec_q))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle)

    scale = np.linalg.norm(vec_q) / np.linalg.norm(vec_p)

    return angle_deg, scale


def transform_image(image, angle, scale):
    # Получаем исходные размеры изображения
    rows, cols = image.shape[:2]

    # Рассчитываем новые размеры с сохранением пропорций
    new_cols = int(cols * scale)
    new_rows = int(rows * scale)

    # Применяем масштабирование с новыми размерами
    scaled_image = cv2.resize(image, (new_cols, new_rows), interpolation=cv2.INTER_LINEAR)

    # Расчет нового размера канвы для избежания обрезки при повороте
    angle_rad = np.radians(angle)
    cos_angle = np.abs(np.cos(angle_rad))
    sin_angle = np.abs(np.sin(angle_rad))
    new_width = int((new_rows * sin_angle) + (new_cols * cos_angle))
    new_height = int((new_rows * cos_angle) + (new_cols * sin_angle))

    # Создание канвы с дополнительными пикселями для избежания обрезки
    canvas = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Позиционирование масштабированного изображения в центре канвы
    offset_x = (new_width - new_cols) // 2
    offset_y = (new_height - new_rows) // 2
    canvas[offset_y:offset_y + new_rows, offset_x:offset_x + new_cols] = scaled_image

    # Матрица поворота для поворота вокруг центра новой канвы
    M = cv2.getRotationMatrix2D((new_width / 2, new_height / 2), -angle, 1)

    # Применяем поворот
    transformed = cv2.warpAffine(canvas, M, (new_width, new_height))

    return transformed


class ImageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Overlay and Analysis')
        self.setGeometry(100, 100, 800, 600)
        layout = QVBoxLayout()

        self.btnLoadImage1 = QPushButton('Load Image 1')
        self.btnLoadImage1.clicked.connect(lambda: self.load_image('image1', True))
        layout.addWidget(self.btnLoadImage1)

        self.btnLoadImage2 = QPushButton('Load Image 2')
        self.btnLoadImage2.clicked.connect(lambda: self.load_image('image2', False))
        layout.addWidget(self.btnLoadImage2)

        self.btnProcess = QPushButton('Process and Analyze')
        self.btnProcess.clicked.connect(self.process_and_analyze)
        layout.addWidget(self.btnProcess)

        self.setLayout(layout)

    def load_image(self, key, optimize=True):
        filePath, _ = QFileDialog.getOpenFileName(self, 'Select Image', '', 'Image files (*.jpg *.png *.jpeg *.tif)')
        if filePath:
            img = adjust_image(filePath)
            if optimize:
                img = optimize_image_contrast(img, clip_limit=30.0)  # Передаем clip_limit для большего контраста
            images[key] = img
            cv2.namedWindow(key)
            cv2.setMouseCallback(key, click_event, key)
            cv2.imshow(key, images[key])

    def process_and_analyze(self):
        if len(clicks['image1']) == 2 and len(clicks['image2']) == 2:
            # Расчет коэффициента увеличения
            optics_vector = np.array(clicks['image1'][1]) - np.array(clicks['image1'][0])
            sem_vector = np.array(clicks['image2'][1]) - np.array(clicks['image2'][0])
            scale_factor = np.linalg.norm(optics_vector) / np.linalg.norm(sem_vector)

            # Выводим значение масштаба в консоль
            print(f"Scale factor: {scale_factor}")

            # Расчет угла поворота
            angle, _ = calculate_transformation(clicks['image1'], clicks['image2'])

            # Масштабирование и поворот SEM изображения
            transformed_image = transform_image(images['image2'], angle, scale_factor)

            # Определяем центр масштабированного изображения
            center_x = transformed_image.shape[1] // 2
            center_y = transformed_image.shape[0] // 2

            # Определяем смещение для точек
            dx = center_x - (clicks['image2'][0][0] + clicks['image2'][1][0]) // 2
            dy = center_y - (clicks['image2'][0][1] + clicks['image2'][1][1]) // 2

            # Корректируем координаты точек на изображении
            corrected_points = [(p[0] + dx, p[1] + dy) for p in clicks['image1']]

            # Обрезаем второе изображение с учетом смещения, чтобы точки оставались в центре
            cropped_image = align_and_crop(transformed_image, images['image1'], corrected_points[0],
                                           clicks['image1'][0])

            # Сохранение изображений
            transformed_image_path = "transformed_image.png"
            combined_image_path = "combined_image.png"

            cv2.imwrite(transformed_image_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

            # Создание комбинированного изображения
            h, w, _ = images['image1'].shape
            combined_image = np.zeros((h, w * 2, 3), dtype=np.uint8)
            combined_image[:, :w] = images['image1']
            combined_image[:, w:] = cropped_image
            cv2.imwrite(combined_image_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

            print(f"Transformed image saved as {transformed_image_path}")
            print(f"Combined image saved as {combined_image_path}")
        else:
            print("Please select exactly two points on each image.")


if __name__ == '__main__':
    images = {'image1': None, 'image2': None}
    app = QApplication(sys.argv)
    ex = ImageApp()
    ex.show()
    sys.exit(app.exec_())
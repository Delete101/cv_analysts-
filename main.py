import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog
from skimage.metrics import structural_similarity as ssim
from scipy.optimize import minimize

clicks = {'image1': [], 'image2': []}



def optimization_objective(params, image1, image2):
    dx, dy = params
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    transformed = cv2.warpAffine(image2, M, (image1.shape[1], image1.shape[0]))
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    transformed_gray = cv2.cvtColor(transformed, cv2.COLOR_RGB2GRAY)
    score, _ = ssim(image1_gray, transformed_gray, full=True)
    return -score  # Минус, потому что мы хотим максимизировать SSIM


def adjust_image(image_path):
    img = cv2.imread(image_path, -1)
    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX)
        img = np.uint8(img / 256)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def click_event(event, x, y, flags, param):
    global clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        points = clicks[param['name']]
        if len(points) >= 2:
            points.clear()
        points.append((x, y))
        for idx, point in enumerate(points):
            color = (255, 0, 0) if idx == 0 else (0, 255, 0)
            cv2.circle(param['image'], point, 5, color, -1)
        cv2.imshow(param['name'], param['image'])

class ImageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.image1 = None
        self.image2 = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Корреляционный анализ изображений')
        self.setFixedSize(800, 600)

        layout = QVBoxLayout(self)
        btnLoadImage1 = QPushButton('Выбрать изображение 1 (optics)', self)
        btnLoadImage1.clicked.connect(lambda: self.loadImage('image1'))
        layout.addWidget(btnLoadImage1)

        btnLoadImage2 = QPushButton('Выбрать изображение 2 (SEM)', self)
        btnLoadImage2.clicked.connect(lambda: self.loadImage('image2'))
        layout.addWidget(btnLoadImage2)

        btnAnalyze = QPushButton('Анализировать', self)
        btnAnalyze.clicked.connect(self.analyzeImages)
        layout.addWidget(btnAnalyze)

    def loadImage(self, name):
        filePath, _ = QFileDialog.getOpenFileName(self, 'Выбрать изображение', '',
                                                  'Images (*.png *.xpm *.jpg *.tif *.jpeg)')
        if filePath:
            image = adjust_image(filePath)
            setattr(self, name, image)
            cv2.imshow(name, image)
            cv2.setMouseCallback(name, click_event, {'name': name, 'image': image, 'path': filePath})

    def analyzeImages(self):
        if self.image1 is None or self.image2 is None or len(clicks['image1']) < 2 or len(clicks['image2']) < 2:
            print("Необходимо загрузить оба изображения и выбрать по две точки на каждом.")
            return
        self.apply_transformation()

    def apply_transformation(self):
        # Начальное приближение для параметров смещения
        initial_guess = [0, 0]

        # Вызов функции оптимизации
        optimized_params = minimize(optimization_objective, initial_guess, args=(self.image1, self.image2),
                                    method='Nelder-Mead').x

        # Применение оптимизированной трансформации
        dx, dy = optimized_params
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        transformed_optimized = cv2.warpAffine(self.image2, M, (self.image1.shape[1], self.image1.shape[0]))
        image1_gray = cv2.cvtColor(self.image1, cv2.COLOR_RGB2GRAY)
        transformed_gray = cv2.cvtColor(transformed_optimized, cv2.COLOR_RGB2GRAY)

        # Вычисление и вывод SSIM для оптимизированной трансформации
        score_optimized, _ = ssim(image1_gray, transformed_gray, full=True)
        print(f"Оптимизированный SSIM: {score_optimized}")

def main():
    app = QApplication(sys.argv)
    ex = ImageApp()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

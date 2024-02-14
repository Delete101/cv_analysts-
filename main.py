import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog
from skimage.metrics import structural_similarity as compare_ssim
from scipy.optimize import minimize


clicks = {'image1': [], 'image2': []}



def optimization_objective(params, image1, image2, points1, points2):
    dx, dy, angle, scale = params
    center = (image2.shape[1] // 2, image2.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[:, 2] += [dx, dy]
    transformed = cv2.warpAffine(image2, M, (image1.shape[1], image1.shape[0]))
    transformed_gray = cv2.cvtColor(transformed, cv2.COLOR_RGB2GRAY)
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

    # Вычисляем схожесть только для выбранных областей вокруг контрольных точек
    ssim_total = 0
    for p1, p2 in zip(points1, points2):
        roi1 = image1_gray[p1[1]-50:p1[1]+50, p1[0]-50:p1[0]+50]
        roi2 = transformed_gray[p2[1]-50:p2[1]+50, p2[0]-50:p2[0]+50]
        if roi1.shape == roi2.shape and roi1.size > 0 and roi2.size > 0:
            ssim_value = compare_ssim(roi1, roi2)
            ssim_total += ssim_value
    ssim_avg = ssim_total / len(points1) if points1 else 0

    return -ssim_avg  # Максимизируем среднее значение SSIM



def adjust_image(image_path):
    img = cv2.imread(image_path, -1)
    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX)
    img = np.uint8(img / 256 if img.dtype == np.uint16 else img)
    # Применение CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(img.shape) == 2 or img.shape[2] == 1:  # для одноканальных изображений
        img = clahe.apply(img)
    else:  # для многоканальных изображений
        for i in range(img.shape[2]):
            img[:, :, i] = clahe.apply(img[:, :, i])
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

    def calculate_ssim(image1, image2, points1, points2, window_size=100):
        ssim_values = []
        for p1, p2 in zip(points1, points2):
            x1, y1 = max(0, p1[0] - window_size // 2), max(0, p1[1] - window_size // 2)
            x2, y2 = max(0, p2[0] - window_size // 2), max(0, p2[1] - window_size // 2)
            roi1 = image1[y1:y1 + window_size, x1:x1 + window_size]
            roi2 = image2[y2:y2 + window_size, x2:x2 + window_size]
            # Адаптация размера окна
            adjusted_window_size = min(roi1.shape[:2] + roi2.shape[:2])
            adjusted_window_size = max(7, adjusted_window_size | 1)  # Убедимся, что окно нечетное и >= 7
            if adjusted_window_size and roi1.size and roi2.size:
                ssim = compare_ssim(roi1, roi2, win_size=adjusted_window_size)
                ssim_values.append(ssim)
        return np.mean(ssim_values) if ssim_values else 0

    def initUI(self):
        self.setWindowTitle('Image Correlation Analysis')
        self.setFixedSize(800, 600)

        layout = QVBoxLayout(self)
        btnLoadImage1 = QPushButton('Select Image 1 (Optics)', self)
        btnLoadImage1.clicked.connect(lambda: self.loadImage('image1'))
        layout.addWidget(btnLoadImage1)

        btnLoadImage2 = QPushButton('Select Image 2 (SEM)', self)
        btnLoadImage2.clicked.connect(lambda: self.loadImage('image2'))
        layout.addWidget(btnLoadImage2)

        btnAnalyze = QPushButton('Analyze', self)
        btnAnalyze.clicked.connect(self.analyzeImages)
        layout.addWidget(btnAnalyze)

    def loadImage(self, name):
        filePath, _ = QFileDialog.getOpenFileName(self, 'Select Image', '',
                                                  'Images (*.png *.xpm *.jpg *.tif *.jpeg)')
        if filePath:
            image = adjust_image(filePath)
            setattr(self, name, image)
            cv2.imshow(name, image)
            cv2.setMouseCallback(name, click_event, {'name': name, 'image': image, 'path': filePath})

    def analyzeImages(self):
        if self.image1 is None or self.image2 is None or len(clicks['image1']) < 2 or len(clicks['image2']) < 2:
            print("Both images and two points on each image are required.")
            return
        self.apply_transformation()

    def initial_guess_from_clicks(self):
        p1, p2 = clicks['image1'][0], clicks['image1'][1]
        q1, q2 = clicks['image2'][0], clicks['image2'][1]
        dist_p = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        dist_q = np.sqrt((q2[0] - q1[0])**2 + (q2[1] - q1[1])**2)
        scale = dist_p / dist_q if dist_q != 0 else 1
        dx = ((p1[0] + p2[0]) / 2 - (q1[0] + q2[0]) / 2) * scale
        dy = ((p1[1] + p2[1]) / 2 - (q1[1] + q2[1]) / 2) * scale
        angle = 0
        return [dx, dy, angle, scale]

    def calculate_ssim(self, image1, image2, points1, points2, window_size=100):
        ssim_values = []
        for p1, p2 in zip(points1, points2):
            x1, y1 = max(0, p1[0] - window_size // 2), max(0, p1[1] - window_size // 2)
            x2, y2 = max(0, p2[0] - window_size // 2), max(0, p2[1] - window_size // 2)

            roi1 = image1[y1:y1 + window_size, x1:x1 + window_size]
            roi2 = image2[y2:y2 + window_size, x2:x2 + window_size]

            min_height = min(roi1.shape[0], roi2.shape[0])
            min_width = min(roi1.shape[1], roi2.shape[1])
            adjusted_window_size = min(min_height, min_width,
                                       window_size) - 1  # Ensure win_size is smaller than ROI and odd
            adjusted_window_size -= adjusted_window_size % 2  # Make it odd

            # Ensure the window size is at least 7 and odd
            adjusted_window_size = max(7, adjusted_window_size | 1)

            if adjusted_window_size >= 7 and roi1.size > 0 and roi2.size > 0:
                if len(roi1.shape) == 3:  # Check for multichannel image
                    channel_axis = 2
                else:
                    channel_axis = None

                ssim = compare_ssim(roi1[:adjusted_window_size, :adjusted_window_size],
                                    roi2[:adjusted_window_size, :adjusted_window_size],
                                    win_size=adjusted_window_size, channel_axis=channel_axis, full=False)
                ssim_values.append(ssim)

        return np.mean(ssim_values) if ssim_values else 0

    def transform_image(self, image, dx, dy, angle, scale):
        rows, cols = image.shape[:2]
        center = (cols // 2, rows // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[:, 2] += [dx, dy]
        # Использование cv2.INTER_CUBIC для возможного улучшения качества
        transformed = cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_CUBIC)
        return transformed

    def save_and_display_results(self, transformed_image, dx, dy, angle, scale, ssim_value):
        # Сохранение результата
        cv2.imwrite("final_transformed_image.png", transformed_image)
        with open("analysis_results.txt", "w") as file:
            file.write(f"DX: {dx}, DY: {dy}, Angle: {angle}, Scale: {scale}, SSIM: {ssim_value}\n")

        # Визуализация результатов
        cv2.imshow("Transformed SEM Image", transformed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def apply_transformation(self):
        if len(clicks['image1']) < 2 or len(clicks['image2']) < 2:
            print("Not enough points selected on one of the images.")
            return

        initial_guess = self.initial_guess_from_clicks()
        optimized_params = minimize(optimization_objective, initial_guess,
                                    args=(self.image1, self.image2, clicks['image1'], clicks['image2']),
                                    method='L-BFGS-B', bounds=[(-1000, 1000), (-1000, 1000), (-360, 360), (0.1, 10)]).x

        dx, dy, angle, scale = optimized_params
        print(f"Optimized parameters: dx={dx}, dy={dy}, angle={angle}, scale={scale}")

        transformed_image = self.transform_image(self.image2, dx, dy, angle, scale)

        ssim_value = self.calculate_ssim(self.image1, transformed_image, clicks['image1'], clicks['image2'])
        print(f"SSIM between selected areas: {ssim_value}")

        # Вызов функции для сохранения и отображения результатов
        self.save_and_display_results(transformed_image, dx, dy, angle, scale, ssim_value)


def main():
    app = QApplication(sys.argv)
    ex = ImageApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

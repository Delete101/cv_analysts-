import cv2
import numpy as np

# Словарь для хранения координат точек на каждом изображении
clicks = {'image1': [], 'image2': []}


# Функция для корректировки и загрузки изображения
def adjust_image(image_path):
    img = cv2.imread(image_path, -1) # Загрузка с оригинальной глубиной цвета
    if img.dtype == np.uint16:  # Проверка и нормализация 16-битных изображений
        img = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX)
        img = np.uint8(img / 256)  # Конвертация в 8-битное изображение для отображения
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Преобразование в трехканальное цветное изображение
    return img


def click_event(event, x, y, flags, param):
    global clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # Получение списка отмеченных точек для текущего изображения и его параметров
        points = clicks[param['name']]
        if len(points) >= 2:  # Если уже отмечено две точки, очистка и начало сначала
            points.clear()
            param['image'] = adjust_image(param['path'])  # Перезагрузка исходного изображения

        points.append((x, y))  # Добавление новой точки

        # Отрисовка точек: первая красная, вторая зеленая
        for idx, point in enumerate(points):
            color = (0, 0, 255) if idx == 0 else (0, 255, 0)
            cv2.circle(param['image'], point, 5, color, -1)

        cv2.imshow(param['name'], param['image'])  # Обновление отображаемого изображения


def main():
    # Пути к файлам изображений и их имена окон
    image_paths = ['image1_optics.tif', 'image01_SEM.tif']
    window_names = ['image1', 'image2']

    for path, name in zip(image_paths, window_names):
        image = adjust_image(path)
        cv2.namedWindow(name)
        cv2.imshow(name, image)
        # Передача словаря с информацией о текущем изображении и его параметрах
        cv2.setMouseCallback(name, click_event, {'name': name, 'image': image, 'path': path})

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
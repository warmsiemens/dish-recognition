# Распознавание блюд
Этот проект реализует модель для распознавания блюд на столе, а также пустых тарелок после еды. Модель обучена вручную на аннотированных кадрах из видео, с использованием YOLOv11.

## Установка
Клонируйте репозиторий и создайте виртуальное окружение, установите зависимости
git clone https://github.com/yourname/dish-recognition.git
cd dish-recognition
python -m venv venv
venv\Scripts\activate 
pip install -r requirements.txt


## Подготовка видео
Создайте папку data/videos/ и поместите в нее видео


## Извлечение кадров из видео
Запусти extract_frames с указанием параметра input_dir - папка, в которой находятся видео; и второй необязательный параметр - количество секунд, которое проходит между кадрами


## Разметка данных
Используй CVAT 
Создай проект, импортируй data/raw_frames/
Аннотируй bounding boxes и классы вручную
Экспортируй аннотации в формате YOLO 1.1
Помести экспортированные .txt в data/cvat_labels/obj_train_data


## Подготовка датасета
Делим на train/val/test:
python services/preparation_dataset.py
Получим data/dataset/images/{train,val,test} и соответствующие labels/.


## Аугментация train-данных:
python services/augmentation.py
Увеличит объём данных с помощью Albumentations (повороты, яркость и т.д.).

## Обучение модели
python train_yolo.py

Модель сохраняется в runs/detect/train/weights/best.pt
Результаты будут в runs/detect/predict/ — изображения с наложенными детекциями.
Все графики: runs/detect/train/results.png

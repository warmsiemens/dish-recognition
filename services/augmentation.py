import os
import cv2
import albumentations as A


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IMG_DIR = os.path.join(BASE_DIR, 'data', 'dataset', 'images', 'train')
LABEL_DIR = os.path.join(BASE_DIR, 'data', 'dataset', 'labels', 'train')


OUT_IMG_DIR = IMG_DIR
OUT_LABEL_DIR = LABEL_DIR


os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LABEL_DIR, exist_ok=True)


transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.MotionBlur(p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def read_yolo_annotation(path):
    boxes = []
    labels = []
    with open(path, 'r') as f:
        for line in f:
            cls, x, y, w, h = line.strip().split()
            boxes.append([float(x), float(y), float(w), float(h)])
            labels.append(int(cls))
    return boxes, labels


def write_yolo_annotation(path, boxes, labels):
    with open(path, 'w') as f:
        for cls, box in zip(labels, boxes):
            f.write(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")


for filename in os.listdir(IMG_DIR):
    if not filename.endswith('.jpg'):
        continue

    img_path = os.path.join(IMG_DIR, filename)
    label_path = os.path.join(LABEL_DIR, filename.replace('.jpg', '.txt'))

    if not os.path.exists(label_path):
        print(f"Аннотация для {filename} не найдена")
        continue

    image = cv2.imread(img_path)
    boxes, labels = read_yolo_annotation(label_path)

    for i in range(2):
        augmented = transform(image=image, bboxes=boxes, class_labels=labels)
        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_labels = augmented['class_labels']

        base_name = filename.replace('.jpg', '')
        out_img_name = f"{base_name}_aug{i+1}.jpg"
        out_label_name = f"{base_name}_aug{i+1}.txt"

        cv2.imwrite(os.path.join(OUT_IMG_DIR, out_img_name), aug_img)
        write_yolo_annotation(os.path.join(OUT_LABEL_DIR, out_label_name), aug_bboxes, aug_labels)

print("Аугментация завершена")

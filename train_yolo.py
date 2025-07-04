from ultralytics import YOLO


def main():
    model = YOLO("yolo11s.pt")

    results = model.train(
        data='data.yaml',
        epochs=10,
        imgsz=640,
        batch=10,
    )
    print("Обучение завершено")


if __name__ == "__main__":
    main()

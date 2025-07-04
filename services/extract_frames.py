import cv2
import os
import time


def extract_frames(input_dir, interval_sec=2.0):
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(BASE_DIR, 'data', 'raw_frames')
    os.makedirs(output_dir, exist_ok=True)

    video_extensions = ('.mov', '.mkv', '.mp4', '.avi')
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(video_extensions):
            video_path = os.path.join(input_dir, filename)
            basename = os.path.splitext(filename)[0]

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Не удалось открыть видео {video_path}")
                continue

            ret, prev_frame = cap.read()
            saved = 0
            cv2.imwrite(os.path.join(output_dir, f"{basename}_frame{saved:0>4}.jpg"), prev_frame)
            last_saved_time = time.time()
            saved += 1

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = time.time()
                if current_time - last_saved_time >= interval_sec:
                    cv2.imwrite(os.path.join(output_dir, f"{basename}_frame{saved:0>4}.jpg"), frame)
                    saved += 1
                    last_saved_time = current_time

            cap.release()


if __name__ == '__main__':
    extract_frames("D:\\test", 2)

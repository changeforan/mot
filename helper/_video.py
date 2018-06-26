import cv2


class Video:
    def __init__(self, path, max_frame):
        self.max_frame = max_frame
        self.frame_count = 0
        self.cap = cv2.VideoCapture(path)
        self.out = cv2.VideoWriter(
            'out_box.avi',
            cv2.VideoWriter_fourcc(*'XVID'),
            20.0,
            (800, 600)
        )

    def __del__(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

    def next(self):
        self.frame_count += 1
        if self.frame_count > self.max_frame:
            raise StopIteration()
        ret, image_np = self.cap.read()
        if not ret:
            raise StopIteration()
        return image_np

    def write(self, image_np):
        frame = cv2.resize(image_np, (800, 600))
        self.out.write(frame)
        print(self.frame_count)

    def __iter__(self):
        return self
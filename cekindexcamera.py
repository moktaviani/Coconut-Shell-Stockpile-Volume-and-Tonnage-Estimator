"""
Untuk cek index kamera yang digunakan
"""
import cv2

for i in range(4):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Index {i}: tidak ada device")
        continue
    ret, frame = cap.read()
    if ret:
        cv2.imshow(f"Index {i}", frame)
        cv2.waitKey(1)         
        print(f"Index {i}: tekan tombol apa saja untuk lanjut")
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
    cap.release()
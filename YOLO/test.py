from ultralytics import YOLO
import cv2 
from multiprocessing import freeze_support

#m ultiprocessing için
if __name__ == '__main__':
    freeze_support()
# model oluşturma ve yükleme
# model = YOLO("data.yaml")
model = YOLO("yolo11n.pt")
# Dataset klasöründeki yaml dosyasının konumunu alarak model eğitimi yapıyor. 
#     results = model.train(data="D:/YOLO/Dataset/data.yaml", epochs=100, imgsz=608)
#     results = model.val()
# Gerçek zamanlı nesne algılama için kamera açma işlemleri
cap= cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılamadı")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kare alınamadı")
        break
    # frameleri modele verip eğitiyor.
    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv11n - Gerçek Zamanlı Nesne Algılama", annotated_frame)
    # q ya basarak çıkabiliyorsun.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("Program sonlandı.")
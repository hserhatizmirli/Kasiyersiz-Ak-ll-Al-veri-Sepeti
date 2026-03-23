import cv2
from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    # 1. Önceden eğitilmiş modeli yükle
    model = YOLO('D:/YOLO/runs/detect/yolo11/weights/best.pt')
    
    # fine-tune kısmı
    results = model.train(
        data='D:/YOLO/Dataset/data.yaml', epochs=150, name='yolo11', project='D:/YOLO/runs/detect',  # Project klasörünü düzelt!exist_ok=True
    )
    print("Eğitim tamamlandı!")
    print(f"Model kaydedildi: D:/YOLO/runs/detect/yolo11/weights/best.pt")
    

def test_kamera(model_path):
    print(f"Model yükleniyor: {model_path}")
    model = YOLO(model_path)   

    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.track(
                frame, persist=True, tracker='bytetrack.yaml', conf=0.50, iou=0.5, max_det=10, classes=[0, 1, 2]     
            )
            annotated_frame = results[0].plot()
            cv2.imshow('YOLO', annotated_frame)
            
            # Konsola tespitleri yaz
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    conf = float(box.conf[0])
                    print(f"Tespit: {class_name} - Güven: {conf:.2f}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    freeze_support()
    main()  # Eğitim yapmak için
    test_kamera('D:/YOLO/runs/detect/yolo11/weights/best.pt')  # DOĞRU YOL!
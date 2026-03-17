# import cv2
# from ultralytics import YOLO


# #m ultiprocessing için
# # if __name__ == '__main__':
# #     freeze_support()
# # model oluşturma ve yükleme
# # model = YOLO("data.yaml")
# model = YOLO("D:/YOLO/runs/detect/train13/weights/best.pt")
# # Dataset klasöründeki yaml dosyasının konumunu alarak model eğitimi yapıyor. 
# # results = model.train(data="D:/YOLO/Dataset/args.yaml", epochs=100)
# # results = model.val()
# # Gerçek zamanlı nesne algılama için kamera açma işlemleri


# # cam = cv2.VideoCapture(0)
# # while True:
# #     ret, frame = cam.read()
# #     cv2.imshow('Camera', frame)
# #     if cv2.waitKey(1) == ord('q'):
# #         break
# # cam.release()
# # cv2.destroyAllWindows()


# cap= cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Kamera açılamadı")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Kare alınamadı")
#         break
#     # frameleri modele verip eğitiyor.
#     results = model(frame)
#     annotated_frame = results[0].plot()

#     cv2.imshow("YOLOv11n - Gerçek Zamanlı Nesne Algılama", annotated_frame)
#     # q ya basarak çıkabiliyorsun.
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
# print("Program sonlandı.")







# import cv2
# from ultralytics import YOLO

# # YOLO11n modelini yükle (ilk çalıştırmada otomatik indirilir)
# model = YOLO('D:/YOLO/runs/detect/train13/weights/best.pt')

# print("YOLO11n modeli yüklendi. Kamera başlatılıyor...")

# # Kamerayı aç
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Kamera açılamadı!")
#     exit()

# print("Kamera açıldı! Çıkmak için 'q' tuşuna basın.")
# print("Nesne tespiti yapılıyor...")

# while True:
#     # Kameradan görüntü al
#     ret, frame = cap.read()
    
#     if not ret:
#         print("Görüntü alınamadı!")
#         break
    
#     # YOLO ile nesne tespiti yap
#     results = model(frame)  # Tahmin yap
    
#     # Sonuçları çizdir
#     annotated_frame = results[0].plot()
    
#     # Görüntüyü göster
#     cv2.imshow('YOLO11n Kamera Testi', annotated_frame)
    
#     # 'q' tuşuna basılırsa çık
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Temizlik
# cap.release()
# cv2.destroyAllWindows()
# print("Program sonlandı.")




import cv2
from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    # 1. Önceden eğitilmiş modeli yükle
    model = YOLO('yolo11n.pt')
    
    # fine-tune kısmı (sadece kamera ile doğrulama yapılacaksa # ile yorum satırına alınıp run edilebilri.)
    results = model.train(data='D:/YOLO/Dataset/data.yaml', epochs=50, name='yolo11n')
    print("Eğitim tamamlandı!")
    
    # 4. Test aşaması - Kamera ile doğrulama
    test_kamera('runs/detect/yolo11n/weights/best.pt')

def test_kamera(model_path):
    model = YOLO(model_path)   

    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model(frame)
            annotated_frame = results[0].plot()
            cv2.imshow('YOL', annotated_frame)
            
            # Konsola tespitleri yaz
            for r in results:
                for box in r.boxes:
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
    main()
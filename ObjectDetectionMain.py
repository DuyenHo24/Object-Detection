#Them thu vien
import cv2
import matplotlib.pyplot as plt
# mo hinh
img = cv2.imread("hinh8.jpg")

# Altering properties of image with cv2
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imaging_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Them file Haar cascade xml
xml_data = cv2.CascadeClassifier('stop_data.xml')
#Phan tich doi tuong trong file xml
detecting = xml_data.detectMultiScale(img_gray,minSize = (30, 30))

amountDetecting = len(detecting)

if amountDetecting != 0:
    for (a, b, width, height) in detecting: #Vòng lặp
        cv2.rectangle(imaging_rgb, (a, b), # Danh dau doi tuong
                      (a + height, b + width),
                      (0, 275, 0), 9)
#Hien thi anh
plt.subplot(1, 1, 1)
plt.imshow(imaging_rgb)
plt.show()
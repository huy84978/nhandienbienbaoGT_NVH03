import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tkinter import Tk, filedialog, Label, Button, messagebox
from PIL import Image, ImageTk

# Đường dẫn tới thư mục dataset
DATASET_PATH = r"C:\python\a\b11"
IMG_SIZE = 32
NUM_CLASSES = 43

# Danh sách tên biển báo bằng tiếng Việt
LABELS_VIETNAM = [
    'Giới hạn tốc độ (20km/h)', 'Giới hạn tốc độ (30km/h)', 'Giới hạn tốc độ (50km/h)', 'Giới hạn tốc độ (60km/h)', 
    'Giới hạn tốc độ (70km/h)', 'Giới hạn tốc độ (80km/h)', 'Hết giới hạn tốc độ (80km/h)', 'Giới hạn tốc độ (100km/h)', 
    'Giới hạn tốc độ (120km/h)', 'Cấm vượt', 'Cấm vượt đối với xe có trọng tải trên 3.5 tấn', 'giao lộ cắt ngang tại ngã ba tiếp theo',
    'Đường ưu tiên', 'Nhường', 'Dừng lại', 'Cấm xe', 'Cấm xe có trọng tải trên 3.5 tấn', 'Cấm vào',
    'Cảnh báo chung', 'Khúc cua nguy hiểm bên trái', 'Khúc cua nguy hiểm bên phải', 'Khúc cua đôi', 
    'Đường gồ ghề', 'Đường trơn trượt', 'Đường hẹp bên phải', 'Công trình giao thông', 'Đèn tín hiệu giao thông', 
    'Người đi bộ', 'khu vực nhiều Trẻ em', 'Xe đạp', 'Cảnh báo băng tuyết', 'Động vật hoang dã', 'Hết giới hạn tốc độ và vượt',
    'Rẽ phải phía trước', 'Rẽ trái phía trước', 'Đi thẳng', 'Đi thẳng hoặc rẽ phải', 'Đi thẳng hoặc rẽ trái', 
    'Đi bên phải', 'Đi bên trái', 'Vòng xuyến', 'Hết cấm vượt', 'Cấm đỗ xe', 'Cấm dừng xe', 'Cấm đỗ'
]

labels_map_vietnam = {i: LABELS_VIETNAM[i] for i in range(NUM_CLASSES)}

meta_df = pd.read_csv(os.path.join(DATASET_PATH, 'meta.csv'))
train_df = pd.read_csv(os.path.join(DATASET_PATH, 'train.csv'))

def load_data_from_train_csv():
    images, labels = [], []
    for _, row in train_df.iterrows():
        img_path = os.path.join(DATASET_PATH, row['Path'])
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        x1, y1, x2, y2 = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']
        img = img[int(y1):int(y2), int(x1):int(x2)]
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        labels.append(row['ClassId'])
    
    images = np.array(images) / 255.0
    labels = to_categorical(np.array(labels), NUM_CLASSES)
    return images, labels

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Flatten(),
        
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    X_train, y_train = load_data_from_train_csv()
    model = create_model()
    model.fit(X_train, y_train, epochs=25, batch_size=64, validation_split=0.1)
    loss, accuracy = model.evaluate(X_train, y_train)
    model.save("Bien_bao_vn.keras")  # Sử dụng định dạng .keras để tránh cảnh báo
    return model, accuracy, loss

def predict_image(model, filepath):
    """Dự đoán lớp của một hình ảnh"""
    try:
        img = cv2.imread(filepath)
        if img is None:
            raise ValueError("Không thể đọc ảnh")
        
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = np.expand_dims(img, axis=0) / 255.0
        predictions = model.predict(img)
        
        # Chuyển độ tin cậy thành phần trăm
        confidence = np.max(predictions) * 100
        return np.argmax(predictions), confidence
    
    except Exception as e:
        messagebox.showerror("Lỗi", f"Lỗi khi dự đoán ảnh: {str(e)}")
        return None, None

class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện biển báo giao thông")
        self.model = None
        self.create_widgets()
        self.load_or_train_model()
    
    def create_widgets(self):
        self.button = Button(self.root, text="Chọn ảnh để dự đoán", command=self.open_image)
        self.button.pack(pady=10)
        self.panel = Label(self.root)
        self.panel.pack(pady=10)
        self.evaluation_label = Label(self.root, text="Đánh giá mô hình:")
        self.evaluation_label.pack(pady=10)
        self.label = Label(self.root, text="Dự đoán:")
        self.label.pack(pady=10)
    
    def load_or_train_model(self):
        if os.path.exists("Bien_bao_vn.keras"):
            try:
                self.model = tf.keras.models.load_model("Bien_bao_vn.keras")
                messagebox.showinfo("Thông báo", "Đã tải mô hình thành công!")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải mô hình: {str(e)}")
                self.model, accuracy, loss = train_model()
                self.update_evaluation(accuracy, loss)
        else:
            self.model, accuracy, loss = train_model()
            self.update_evaluation(accuracy, loss)
    
    def update_evaluation(self, accuracy, loss):
        self.evaluation_label.config(
            text=f"Đánh giá mô hình:\nĐộ chính xác: {accuracy*100:.2f}%\nTổn thất: {loss:.4f}"
        )

    def open_image(self):
        if self.model is None:
            messagebox.showerror("Lỗi", "Mô hình chưa được tải!")
            return
        filepath = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        if filepath:
            self.label.config(text="Đang xử lý...")
            class_id, confidence = predict_image(self.model, filepath)
            if class_id is not None:
                class_name = labels_map_vietnam.get(class_id, 'Không xác định')
                self.label.config(text=f"Dự đoán: {class_name}\nĐộ tin cậy: {confidence:.2f}%")
                try:
                    img = Image.open(filepath)
                    img = img.resize((200, 200))
                    img = ImageTk.PhotoImage(img)
                    self.panel.config(image=img)
                    self.panel.image = img
                except Exception as e:
                    messagebox.showerror("Lỗi", f"Không thể hiển thị ảnh: {str(e)}")

if __name__ == "__main__":
    root = Tk()
    app = TrafficSignApp(root)
    root.mainloop()

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

# Định nghĩa tham số
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_PATH = "f:/Python/Project_python/MachineLerning/assets"

# Load dữ liệu
train_dataset = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# Lấy danh sách lớp
class_names = train_dataset.class_names
num_classes = len(class_names)
print("Class names:", class_names)  # Kiểm tra danh sách class

# Chuẩn hóa dữ liệu
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# Load mô hình MobileNetV2 đã pretrain
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

# Đóng băng layers
base_model.trainable = False

# Tạo model phân loại chim
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax")  # Số lớp tương ứng với số loài chim
])

# Compile model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Huấn luyện mô hình
model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# Lưu model
model.save("bird_classifier_model.h5")  # Lưu dưới dạng tệp .h5


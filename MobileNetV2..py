import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Định nghĩa tham số
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_PATH = "f:/Python/Project_python/MyProject/assets"

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

# Load mô hình SSD MobileNetV2 đã pretrain
base_model = tf.keras.applications.MobileNetV2(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet")

# Đóng băng layers
base_model.trainable = False

# Tạo model SSD MobileNetV2
input_tensor = Input(shape=(*IMAGE_SIZE, 3))
base_model_output = base_model(input_tensor)
x = tf.keras.layers.GlobalAveragePooling2D()(base_model_output)
x = tf.keras.layers.Dense(256, activation="relu")(x)
output_tensor = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = Model(inputs=input_tensor, outputs=output_tensor)

# Compile model
model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=["accuracy"])

# Huấn luyện mô hình
model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# Lưu model
model.save("ssd_bird_classifier_model.h5")  # Lưu dưới dạng tệp .h5
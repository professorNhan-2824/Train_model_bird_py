import tensorflow as tf  

model = tf.keras.models.load_model("ssd_bird_classifier_model.h5")  # Load mô hình từ file .h5
converter = tf.lite.TFLiteConverter.from_keras_model(model)  # Chuyển đổi sang TFLite
tflite_model = converter.convert()  

with open("ssd_bird_classifier_model.tflite", "wb") as f:
    f.write(tflite_model)


print("✅ Mô hình đã được chuyển sang TensorFlow Lite thành công!")

"""
Weapon detection in X-ray / thermal baggage images
ResNet50 transfer learning + Grad-CAM explanations
(Originally run in Google Colab)
"""
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import backend as K

# ---------- Paths and settings ----------
data_dir = "/content/object_data/xray_baggages"
train_path = os.path.join(data_dir, "training")
test_path = os.path.join(data_dir, "testing")
# NOTE: this order must match train_data.class_indices (folders are sorted alphabetically)
classes = ['Blade', 'Shuriken', 'Gun', 'Others']
img_size = (224, 224)
batch_size = 32

# ---------- Data loading and augmentation ----------
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

# ---------- Model: frozen ResNet50 + custom head ----------
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
out = Dense(len(train_data.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ---------- Training ----------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    steps_per_epoch=len(train_data),
    validation_steps=len(val_data))

model.save('/content/thermal_weapon_resnet50.h5')

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.show()

# ---------- Grad-CAM ----------
def grad_cam(model, img_array, class_idx, layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = np.mean(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
    return heatmap


def overlay_heatmap(img, heatmap, alpha=0.6):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)


# ---------- Batch prediction with Grad-CAM ----------
def predict_batch_images(folder_path, num_images=5, confidence_threshold=50):
    model = load_model("/content/thermal_weapon_resnet50.h5")
    image_paths = sorted(glob.glob(folder_path + "/**/*", recursive=True))
    image_paths = [p for p in image_paths if p.endswith((".png", ".jpg", ".jpeg"))][:num_images]

    if not image_paths:
        print("⚠️ No images found in the folder!")
        return

    plt.figure(figsize=(15, 10))
    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, img_size)
        input_img = img_to_array(resized) / 255.0
        input_img = np.expand_dims(input_img, axis=0)

        prediction = model.predict(input_img)[0]
        class_idx = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        label = classes[class_idx]

        heatmap = grad_cam(model, input_img, class_idx)
        heatmap_img = overlay_heatmap(img_rgb, heatmap)

        title_color = "green" if confidence >= confidence_threshold else "red"
        title_text = f"{label} ({confidence:.2f}%)"

        plt.subplot(2, num_images, i + 1)
        plt.imshow(img_rgb)
        plt.title(f"Original: {title_text}", color=title_color)
        plt.axis("off")

        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(heatmap_img)
        plt.title("Grad-CAM", color=title_color)
        plt.axis("off")

        print(f"📌 {os.path.basename(img_path)} → {label} ({confidence:.2f}%)")
    plt.show()


predict_batch_images(test_path, num_images=5)

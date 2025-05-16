import os
import datetime
import subprocess
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import datetime

# Detect dataset choice from command line
dataset = sys.argv[1] if len(sys.argv) > 1 else 'kmnist'
print(f"✓ Using dataset: {dataset}")

# Load data based on dataset type
if dataset == 'kmnist':
    x = np.load("data/kmnist-train-imgs.npz")["arr_0"]
    y = np.load("data/kmnist-train-labels.npz")["arr_0"]
    num_classes = 10
    kana_labels = ['お', 'き', 'す', 'つ', 'な', 'は', 'ま', 'や', 'れ', 'を']
    romaji_labels = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo']
elif dataset == 'k49':
    x = np.load("data/k49-train-imgs.npz")["arr_0"]
    y = np.load("data/k49-train-labels.npz")["arr_0"]
    num_classes = 49
    kana_labels = [
        'あ', 'い', 'う', 'え', 'お', 'か', 'が', 'き', 'ぎ', 'く',
        'ぐ', 'け', 'げ', 'こ', 'ご', 'さ', 'ざ', 'し', 'じ', 'す',
        'ず', 'せ', 'ぜ', 'そ', 'ぞ', 'た', 'だ', 'ち', 'ぢ', 'っ',
        'つ', 'づ', 'て', 'で', 'と', 'ど', 'な', 'に', 'ぬ', 'ね',
        'の', 'は', 'ば', 'ぱ', 'ひ', 'び', 'ぴ', 'ふ', 'ぶ'
    ]

    romaji_labels = [
        'a', 'i', 'u', 'e', 'o', 'ka', 'ga', 'ki', 'gi', 'ku',
        'gu', 'ke', 'ge', 'ko', 'go', 'sa', 'za', 'shi', 'ji', 'su',
        'zu', 'se', 'ze', 'so', 'zo', 'ta', 'da', 'chi', 'di', 'tsu',
        'tsu (sokuon)', 'du', 'te', 'de', 'to', 'do', 'na', 'ni', 'nu', 'ne',
        'no', 'ha', 'ba', 'pa', 'hi', 'bi', 'pi', 'fu', 'bu'
    ]
else:
    raise ValueError("Dataset must be 'kmnist' or 'k49'")

# Normalize and prepare
x = x.astype("float32") / 255.0
x = np.expand_dims(x, -1)
split = int(len(x) * 0.8)
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

model_path = f"{dataset}_model.h5"

if os.path.exists(model_path):
    print(f"🔁 Resuming from saved model: {model_path}")
    model = tf.keras.models.load_model(model_path)
else:
    print("🛠️  Building new model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Start TensorBoard in background
tensorboard_url = "http://localhost:6006"
print(f"🚀 Launching TensorBoard at {tensorboard_url}...")
subprocess.Popen(["tensorboard", "--logdir", "logs", "--port", "6006"])

log_dir = f"logs/{dataset}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(
    train_ds,
    epochs=5,
    validation_data=test_ds,
    callbacks=[tensorboard_callback]
)
loss, acc = model.evaluate(test_ds)

model.save(f"{dataset}_model.h5")
print(f"✓ Model saved to {dataset}_model.h5")


print(f"\nTest accuracy: {acc:.4f}")

# Set up Japanese font (if possible)
preferred_fonts = ["Hiragino Sans", "Osaka", "Noto Sans CJK JP", "Arial Unicode MS"]
available_fonts = [f.name for f in fm.fontManager.ttflist]
for font in preferred_fonts:
    if font in available_fonts:
        matplotlib.rcParams['font.family'] = font
        break

# Plot predictions (only if label map is available)
if kana_labels:
    for images, labels in test_ds.take(1):
        logits = model(images[:10])
        predictions = tf.argmax(logits, axis=1)

        plt.figure(figsize=(15, 3))
        for i in range(10):
            pred_idx = int(predictions[i])
            true_idx = int(labels[i])
            pred_kana = kana_labels[pred_idx]
            pred_romaji = romaji_labels[pred_idx]
            true_kana = kana_labels[true_idx]
            true_romaji = romaji_labels[true_idx]
            color = 'green' if pred_idx == true_idx else 'red'

            plt.subplot(1, 10, i + 1)
            plt.imshow(tf.squeeze(images[i]), cmap='gray')
            plt.title(f'{pred_kana} ({pred_romaji})\n{true_kana} ({true_romaji})',
                      color=color, fontsize=11)
            plt.axis('off')
        plt.suptitle("Predicted (top) / True (bottom)", fontsize=14)
        plt.tight_layout()
        plt.show()

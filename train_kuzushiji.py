# train_kuzushiji.py

import sys
import os
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
from tensorboard.plugins.hparams import api as hp

# ---- Config ----
# Parse dataset choice
dataset = sys.argv[1] if len(sys.argv) > 1 else 'kmnist'
model_path = f"{dataset}_model.h5"
print(f"\n✓ Using dataset: {dataset}")

# ---- Hyperparameter tracking ----
HP_DATASET = hp.HParam('dataset', hp.Discrete(['kmnist', 'k49']))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))
hparams = {
    HP_DATASET: dataset,
    HP_BATCH_SIZE: 32,
    HP_OPTIMIZER: 'adam',
}

# ---- Load dataset ----
if dataset == 'kmnist':
    x = np.load("data/kmnist-train-imgs.npz")['arr_0']
    y = np.load("data/kmnist-train-labels.npz")['arr_0']
    num_classes = 10
    kana_labels = ['お', 'き', 'す', 'つ', 'な', 'は', 'ま', 'や', 'れ', 'を']
    romaji_labels = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo']
elif dataset == 'k49':
    x = np.load("data/k49-train-imgs.npz")['arr_0']
    y = np.load("data/k49-train-labels.npz")['arr_0']
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
        'zu', 'se', 'ze', 'so', 'zo', 'ta', 'da', 'chi', 'di', 'tsu (sokuon)',
        'tsu', 'du', 'te', 'de', 'to', 'do', 'na', 'ni', 'nu', 'ne',
        'no', 'ha', 'ba', 'pa', 'hi', 'bi', 'pi', 'fu', 'bu'
    ]
else:
    raise ValueError("Dataset must be 'kmnist' or 'k49'")

x = x.astype("float32") / 255.0
x = np.expand_dims(x, -1)
split = int(len(x) * 0.8)
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# ---- TensorBoard logging ----
log_dir = f"logs/{dataset}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
with tf.summary.create_file_writer(log_dir).as_default():
    hp.hparams_config(
        hparams=[HP_DATASET, HP_BATCH_SIZE, HP_OPTIMIZER],
        metrics=[
            hp.Metric('accuracy', display_name='Accuracy'),
            hp.Metric('val_accuracy', display_name='Val Accuracy'),
        ],
    )

# ---- Model setup ----
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

class PrintMetrics(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"✅ Epoch {epoch+1}: loss={logs['loss']:.4f}, val_loss={logs['val_loss']:.4f}, accuracy={logs['accuracy']:.4f}, val_accuracy={logs['val_accuracy']:.4f}")

model.fit(
    train_ds,
    epochs=5,
    validation_data=test_ds,
    callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        hp.KerasCallback(log_dir, hparams),
        PrintMetrics()
    ]
)

loss, acc = model.evaluate(test_ds)
print(f"\n🎯 Final Test accuracy: {acc:.4f}")
model.save(model_path)
print(f"✓ Model saved to {model_path}")

# ---- Font setup ----
preferred_fonts = ["Hiragino Sans", "Osaka", "Noto Sans CJK JP", "Arial Unicode MS"]
available_fonts = [f.name for f in fm.fontManager.ttflist]
for font in preferred_fonts:
    if font in available_fonts:
        matplotlib.rcParams['font.family'] = font
        break

# ---- Plot predictions (optional) ----
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
            plt.title(f'{pred_kana} ({pred_romaji})\n{true_kana} ({true_romaji})', color=color, fontsize=11)
            plt.axis('off')
        plt.suptitle("Predicted (top) / True (bottom)", fontsize=14)
        plt.tight_layout()
        plt.show()

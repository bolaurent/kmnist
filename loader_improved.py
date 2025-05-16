import numpy as np
import tensorflow as tf

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt



# Map label index to kana
kana_labels = ['お', 'き', 'す', 'つ', 'な', 'は', 'ま', 'や', 'れ', 'を']
romaji_labels = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo']


# Load Kuzushiji-MNIST from npz files
x = np.load("kmnist/kmnist-train-imgs.npz")["arr_0"]
y = np.load("kmnist/kmnist-train-labels.npz")["arr_0"]

# Normalize and expand dims for CNN input
x = x.astype("float32") / 255.0
x = np.expand_dims(x, -1)

# Manually split into train/test (80/20)
split = int(len(x) * 0.8)
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

# Create TensorFlow datasets
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Build model
# Improved CNN model for Kuzushiji-MNIST
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    # Data augmentation (can also be applied in the dataset pipeline)
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
    tf.keras.layers.RandomZoom(0.1),

    # Convolutional layers
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(10, activation='softmax')
])

# Learning rate schedule and optimizer
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
model_ckpt = tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)

# Compile
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train
model.fit(train_ds, epochs=5, validation_data=test_ds)

# Evaluate
loss, acc = model.evaluate(test_ds)
print(f"\nTest accuracy: {acc:.4f}")


# Look for a Japanese-capable font
preferred_fonts = ["Hiragino Sans", "Osaka", "Noto Sans CJK JP", "Arial Unicode MS"]
available_fonts = [f.name for f in fm.fontManager.ttflist]

for font in preferred_fonts:
    if font in available_fonts:
        matplotlib.rcParams['font.family'] = font
        print(f"✓ Using Japanese font: {font}")
        break
else:
    print("⚠️ No Japanese font found. Kana may not render correctly.")


# Get a batch of test images and labels
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
        correct = pred_idx == true_idx
        color = 'green' if correct else 'red'

        plt.subplot(1, 10, i + 1)
        plt.imshow(tf.squeeze(images[i]), cmap='gray')
        pred_label = kana_labels[predictions[i]]
        true_label = kana_labels[labels[i]]
        color = 'green' if pred_label == true_label else 'red'
        plt.title(f'{pred_kana} ({pred_romaji})\n{true_kana} ({true_romaji})',
                  color=color, fontsize=11)
        plt.axis('off')

    plt.suptitle("Predicted (top) / True (bottom)", fontsize=14)
    plt.tight_layout()
    plt.show()

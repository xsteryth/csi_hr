
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import datetime
import gc

# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_SIZE = 500
NUM_SUBCARRIERS = 192

BATCH_SIZE = 16
EPOCHS = 5000

TRAIN_SPLIT = 0.80

LEARNING_RATE = 0.0001

CSI_FILE = "../data/training_data_aligned.txt"
HR_FILE = "../data/hr_data_aligned.txt"

CHECKPOINT_FILE = "../data/csi_hr_500_checkpoint.keras"
FINAL_MODEL_FILE = "../data/csi_hr_500.keras"

NORMALIZATION_FILE = "../data/csi_normalization.npz"

# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow version:", tf.__version__)

print("Built with CUDA:",
      tf.test.is_built_with_cuda())

print("Available GPUs:",
      tf.config.list_physical_devices('GPU'))

# ============================================================
# DATA GENERATOR
# ============================================================

class CSISequence(keras.utils.Sequence):

    def __init__(
        self,
        data,
        hr,
        start,
        end,
        window_size=500,
        batch_size=16,
        shuffle=False
    ):

        self.data = data
        self.hr = hr

        self.start = start
        self.end = end

        self.window_size = window_size
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = np.arange(
            start,
            end - window_size + 1
        )

        self.on_epoch_end()

    def __len__(self):

        return int(np.ceil(
            len(self.indices) / self.batch_size
        ))

    def __getitem__(self, index):

        batch_indices = self.indices[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        batch_length = len(batch_indices)

        X = np.empty(
            (
                batch_length,
                self.window_size,
                self.data.shape[1]
            ),
            dtype=np.float32
        )

        Y = np.empty(
            (batch_length, 1),
            dtype=np.float32
        )

        for j, i in enumerate(batch_indices):

            X[j] = self.data[
                i:i + self.window_size
            ]

            Y[j] = np.mean(
                self.hr[
                    i:i + self.window_size
                ]
            )

        return X, Y

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.indices)


# ============================================================
# LOAD DATA
# ============================================================

print("\n====================================")
print("Loading CSI and HR data")
print("====================================")

print(datetime.datetime.now())

if not os.path.exists(CSI_FILE):
    raise FileNotFoundError(CSI_FILE)

if not os.path.exists(HR_FILE):
    raise FileNotFoundError(HR_FILE)

# Load CSI data as float32.
data = np.loadtxt(
    CSI_FILE,
    delimiter=",",
    dtype=np.float32
)

# Load heart rate data as float32.
data_hr = np.loadtxt(
    HR_FILE,
    dtype=np.float32
)

data = np.atleast_2d(data)
data_hr = np.atleast_1d(data_hr)

if data.shape[1] != NUM_SUBCARRIERS:
    raise ValueError(
        f"Expected {NUM_SUBCARRIERS} subcarriers, "
        f"got {data.shape[1]}"
    )

if len(data) != len(data_hr):
    raise ValueError(
        f"CSI rows: {len(data)}, HR rows: {len(data_hr)}"
    )

if len(data) <= WINDOW_SIZE:
    raise ValueError("Insufficient data for window generation.")

if not np.all(np.isfinite(data)):
    raise ValueError("CSI contains NaN or infinite values.")

if not np.all(np.isfinite(data_hr)):
    raise ValueError("HR contains NaN or infinite values.")

print("CSI shape:", data.shape)
print("HR shape:", data_hr.shape)

print("CSI memory:",
      round(data.nbytes / (1024 ** 2), 2), "MB")

# ============================================================
# CHRONOLOGICAL TRAIN / VALIDATION SPLIT
# ============================================================

print("\nSplitting data chronologically...")

split_index = int(len(data) * TRAIN_SPLIT)

# Ensure enough data exists in both partitions.
if split_index < WINDOW_SIZE or len(data) - split_index < WINDOW_SIZE:
    raise ValueError("Insufficient data for train/validation split.")

print("Training samples:", split_index)
print("Validation samples:", len(data) - split_index)

# ============================================================
# NORMALIZATION
# ============================================================

print("\nNormalizing CSI subcarriers...")

# Calculate statistics using training data only.
mean = np.mean(
    data[:split_index],
    axis=0,
    keepdims=True
)

std = np.std(
    data[:split_index],
    axis=0,
    keepdims=True
)

# Prevent division by zero.
std = np.maximum(std, 1e-8)

# Normalize CSI in place to avoid additional full-sized arrays.
data -= mean
data /= std

# Save normalization parameters for inference.
np.savez(
    NORMALIZATION_FILE,
    mean=mean,
    std=std
)

print("Normalization complete.")

print("Normalized CSI mean:", np.mean(data[:split_index]))
print("Normalized CSI std:", np.std(data[:split_index]))

# ============================================================
# CREATE DATA GENERATORS
# ============================================================

print("\nCreating data generators...")

train_gen = CSISequence(
    data=data,
    hr=data_hr,
    start=0,
    end=split_index,
    window_size=WINDOW_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_gen = CSISequence(
    data=data,
    hr=data_hr,
    start=split_index,
    end=len(data),
    window_size=WINDOW_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Training batches:", len(train_gen))
print("Validation batches:", len(val_gen))

print("Training windows:", len(train_gen.indices))
print("Validation windows:", len(val_gen.indices))

# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding LSTM model...")

main_input = keras.Input(
    shape=(WINDOW_SIZE, NUM_SUBCARRIERS),
    name="main_input"
)

# Explicit sequence mask for cuDNN 9.
x = keras.layers.Masking(
    mask_value=1e9,
    name="cudnn_sequence_mask"
)(main_input)

# First LSTM: 64 units, return full sequence.
x = keras.layers.LSTM(
    64,
    return_sequences=True,
    name="lstm_1"
)(x)

x = keras.layers.Dropout(
    0.2,
    name="dropout_1"
)(x)

# Second LSTM: 32 units.
x = keras.layers.LSTM(
    32,
    return_sequences=False,
    name="lstm_2"
)(x)

x = keras.layers.Dropout(
    0.2,
    name="dropout_2"
)(x)

x = keras.layers.Dense(
    16,
    activation="relu",
    name="dense_1"
)(x)

hr_output = keras.layers.Dense(
    1,
    name="hr_output"
)(x)

model = keras.Model(
    inputs=main_input,
    outputs=hr_output
)

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="mse",
    metrics=["mae"]
)

# ===========================================================
print("\n====================================")
print("TESTING TWO-LAYER cuDNN LSTM")
print("====================================")

test_x = np.random.randn(
    BATCH_SIZE,
    WINDOW_SIZE,
    NUM_SUBCARRIERS
).astype(np.float32)

test_y = np.random.randn(
    BATCH_SIZE,
    1
).astype(np.float32)

start_time = datetime.datetime.now()

try:

    with tf.device("/GPU:0"):

        test_loss = model.train_on_batch(
            test_x,
            test_y
        )

    elapsed = (
        datetime.datetime.now() - start_time
    ).total_seconds()

    print("\ncuDNN TEST SUCCESSFUL")
    print("Loss:", test_loss)
    print("Execution time:", elapsed, "seconds")

except Exception as e:

    print("\ncuDNN TEST FAILED")
    print(type(e).__name__, e)
    raise

finally:
    del test_x, test_y
    gc.collect()
# ============================================================

model.summary()

print("\nModel input shape:", model.input_shape)

# ============================================================
# CALLBACKS
# ============================================================

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_FILE,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=7,
    min_lr=1e-6,
    verbose=1
)

callbacks_list = [
    checkpoint_callback,
    early_stopping,
    reduce_lr
]

# ============================================================
# TRAIN MODEL
# ============================================================

print("\n====================================")
print("Starting training")
print("====================================")

print(datetime.datetime.now())

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks_list,
    verbose=2
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\nSaving final model...")

model.save(FINAL_MODEL_FILE)

print("Final model saved:", FINAL_MODEL_FILE)

# ============================================================
# TRAINING SUMMARY
# ============================================================

best_epoch = np.argmin(
    history.history["val_loss"]
) + 1

best_val_loss = min(
    history.history["val_loss"]
)

best_val_mae = history.history["val_mae"][
    best_epoch - 1
]

print("\n====================================")
print("TRAINING COMPLETE")
print("====================================")

print("Best epoch:", best_epoch)
print("Best validation MSE:", best_val_loss)
print("Best validation MAE:", best_val_mae)

print("Final model:", FINAL_MODEL_FILE)
print("Best checkpoint:", CHECKPOINT_FILE)
print("Normalization:", NORMALIZATION_FILE)

print(datetime.datetime.now())

# ============================================================
# CLEANUP
# ============================================================

del train_gen
del val_gen
del data
del data_hr

gc.collect()

print("\nDone!")

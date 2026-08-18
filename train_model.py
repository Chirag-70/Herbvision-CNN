from pathlib import Path
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "Medicinal plant dataset"
MODEL_PATH = BASE_DIR / "herb_classifier.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.20
SEED = 42
EPOCHS = 15


print("=" * 70)
print("HERB IMAGE CLASSIFIER - TRAINING")
print("=" * 70)

print("\nTensorFlow version:", tf.__version__)

print("\nProject directory:")
print(BASE_DIR)

print("\nDataset directory:")
print(DATASET_DIR)

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"\nDataset folder not found:\n{DATASET_DIR}\n\n"
        "Make sure the folder is named exactly:\n"
        "Medicinal plant dataset"
    )


class_folders = sorted(
    [
        folder
        for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    ]
)

if not class_folders:
    raise RuntimeError(
        f"No class folders found inside:\n{DATASET_DIR}"
    )


print("\nDataset folder found successfully.")

print("\nNumber of class folders found:",
      len(class_folders))

print("\nDetected classes:")

for index, folder in enumerate(class_folders):
    print(f"{index}: {folder.name}")


print("\nLoading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="int",
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)


val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="int",
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class_names = train_ds.class_names
num_classes = len(class_names)


print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nNumber of classes:", num_classes)

print("\nClass mapping:")

for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")


with open(
    CLASS_NAMES_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        class_names,
        file,
        indent=4,
        ensure_ascii=False
    )


print(
    "\nClass names saved to:",
    CLASS_NAMES_PATH
)


AUTOTUNE = tf.data.AUTOTUNE


train_ds = train_ds.prefetch(
    buffer_size=AUTOTUNE
)

val_ds = val_ds.prefetch(
    buffer_size=AUTOTUNE
)


data_augmentation = keras.Sequential(
    [
        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.15
        ),

        layers.RandomZoom(
            0.15
        ),

        layers.RandomTranslation(
            height_factor=0.10,
            width_factor=0.10
        ),

        layers.RandomContrast(
            0.10
        )
    ],
    name="data_augmentation"
)


model = keras.Sequential(
    [
        layers.Input(
            shape=(
                IMG_SIZE[0],
                IMG_SIZE[1],
                3
            )
        ),

        layers.Rescaling(
            1.0 / 255
        ),

        data_augmentation,


        layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            pool_size=(2, 2)
        ),


        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            pool_size=(2, 2)
        ),


        layers.Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            pool_size=(2, 2)
        ),


        layers.Conv2D(
            256,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            pool_size=(2, 2)
        ),


        layers.Conv2D(
            512,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            pool_size=(2, 2)
        ),


        layers.GlobalAveragePooling2D(),


        layers.Dense(
            256,
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.Dropout(
            0.40
        ),


        layers.Dense(
            num_classes,
            activation="softmax"
        )
    ],
    name="Herb_CNN_Classifier"
)


model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


print("\n" + "=" * 70)
print("MODEL ARCHITECTURE")
print("=" * 70)

model.summary()


callbacks = [

    keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]


print("\n" + "=" * 70)
print("STARTING CNN TRAINING")
print("=" * 70)


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


print("\n" + "=" * 70)
print("FINAL MODEL EVALUATION")
print("=" * 70)


validation_loss, validation_accuracy = model.evaluate(
    val_ds,
    verbose=1
)


print(
    f"\nValidation Loss: "
    f"{validation_loss:.4f}"
)

print(
    f"Validation Accuracy: "
    f"{validation_accuracy * 100:.2f}%"
)


if not MODEL_PATH.exists():

    model.save(
        MODEL_PATH
    )


print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print("\nModel file:")
print(MODEL_PATH)

print("\nClass names file:")
print(CLASS_NAMES_PATH)

print("\nNumber of classes:", num_classes)

print("\nAll detected classes:")

for index, class_name in enumerate(class_names):
    print(
        f"{index}: {class_name}"
    )

print("\nYour model is ready for Streamlit prediction.")
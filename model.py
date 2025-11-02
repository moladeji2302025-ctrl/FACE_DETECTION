# model.py
"""
Train a CNN emotion classifier and save it to trained_models/empathica_v1.h5

How to use:
- Put training images in: data/train/<emotion>/*.jpg and validation in data/val/<emotion>/*.jpg
- Emotions folder names should be: angry, disgust, fear, happy, sad, surprise, neutral
- Then run: python model.py
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 30

train_dir = 'data/train'
val_dir = 'data/val'
os.makedirs('trained_models', exist_ok=True)
model_path = 'trained_models/empathica_v1.h5'

def build_model(input_shape=(48,48,1), n_classes=7):
    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # Data generators (grayscale)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    n_classes = len(train_generator.class_indices)
    print("Detected classes:", train_generator.class_indices)

    model = build_model((IMG_SIZE[0], IMG_SIZE[1],1), n_classes=n_classes)
    model.summary()

    callbacks = [
        ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=max(1, val_generator.samples // BATCH_SIZE),
        callbacks=callbacks
    )

    # Best model is saved by ModelCheckpoint. Save final model as well.
    model.save(model_path)
    print("Model saved to", model_path)

if __name__ == "__main__":
    main()

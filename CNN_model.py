import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

imgSize = 300
batchSize = 32
epochs = 10
dataPath = "Data"

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

trainData = datagen.flow_from_directory(
    dataPath,
    target_size=(imgSize, imgSize),
    batch_size=batchSize,
    class_mode='categorical',
    subset='training'
)

valData = datagen.flow_from_directory(
    dataPath,
    target_size=(imgSize, imgSize),
    batch_size=batchSize,
    class_mode='categorical',
    subset='validation'
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(imgSize, imgSize, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')

])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    trainData,
    validation_data=valData,
    epochs=epochs
)

model.save("hand_gesture.h5")

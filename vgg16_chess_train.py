from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

dataset_dir = 'Chess'
img_size = (180, 180)
batch_size = 32

vgg16_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (180, 180, 3), classes = 6)

model = Sequential()
model.add(vgg16_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(6, activation='softmax'))

model.summary()

datagen = ImageDataGenerator(
    rescale = 1./255,
    validation_split = 0.2,
    rotation_range = 20, 
    zoom_range = 0.2, 
    horizontal_flip = True, 
    vertical_flip= True
)

train_generator = datagen.flow_from_directory(
    dataset_dir, 
    target_size=img_size, 
    batch_size=batch_size, 
    class_mode='categorical', 
    subset='training'
    )

validation_generator = datagen.flow_from_directory(
    dataset_dir, 
    target_size=img_size, 
    batch_size=batch_size, 
    class_mode='categorical', 
    subset='validation'
    )

sgd = SGD(learning_rate = 0.0005, momentum = 0.9)

model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
epochs = 15

pretrain = model.fit(
    train_generator,
    steps_per_epoch = len(train_generator),
    validation_data = validation_generator,
    validation_steps = len(validation_generator),
    epochs = epochs,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
)

model.save("test_model.h5")


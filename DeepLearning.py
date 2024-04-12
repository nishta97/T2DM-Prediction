import os
import pydicom
import numpy as np
import time
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, ReLU, GlobalAveragePooling2D, Dropout, SpatialDropout2D
from sklearn.metrics import accuracy_score, roc_auc_score
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_curve, auc
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam, Lion, SGD
import imageio
import scipy.ndimage
import keras.utils
import tensorflow as tf
from sklearn.model_selection import KFold


# Set the paths to the train, validation, and test data directories
train_data_dir = '/home/Nishta/train1'
validation_data_dir = '/home/Nishta/validation1'
test_data_dir = '/home/Nishta/test1'

# Set the image dimensions
# img_width, img_height =2048,2048
img_width, img_height = 2048,2048

n_splits_outer = 5
n_splits_inner = 5

# Set the batch size and number of epochs
# batch_size = 32
batch_size = 128
epochs = 2500

# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=20,           # Number of epochs with no improvement
    verbose=1,            # Print messages about early stopping
    restore_best_weights=True  # Restore model weights to the best epoch
)
class DICOMDataGenerator(ImageDataGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_dict = {}

    def flow_from_directory(self, directory, target_size=(img_width, img_height), batch_size=32, shuffle=True):
        files = self.filelist(directory)
        num_samples = len(files)
        steps_per_epoch = int(np.ceil(num_samples / batch_size))

        while True:
            if shuffle:
                np.random.shuffle(files)

            for i in range(steps_per_epoch):
                batch_files = files[i * batch_size: (i + 1) * batch_size]
                batch_images = []
                batch_labels = []

                for file in batch_files:
                    label = int(os.path.basename(os.path.dirname(file)))
                    if file in self.img_dict:
                        img = self.img_dict[file]
                    else:
                        img = read_dicom(file, target_size=target_size, label=label)
                        self.img_dict[file] = img
                    batch_images.append(img)
                    batch_labels.append(label)

                yield np.array(batch_images), np.array(batch_labels)

    def filelist(self, directory):
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith('.dcm'):
                    files.append(os.path.join(root, filename))
        return files
# Count the number of files in the train data directory
num_train_samples = len(DICOMDataGenerator().filelist(train_data_dir))
print(num_train_samples)

# Count the number of files in the validation data directory
num_validation_samples = len(DICOMDataGenerator().filelist(validation_data_dir))
print(num_validation_samples)

# Function to read and preprocess DICOM images
def read_dicom(filepath, target_size=(img_width, img_height), label=False):
    ds = pydicom.dcmread(filepath)
    img = ds.pixel_array
    img = img.astype('float32') / np.max(img)
    # img = np.expand_dims(img, axis=-1)
    # img = np.resize(img, target_size)  # Resize the image to the target size
    img_height_pre = img.shape[0]
    img_width_pre = img.shape[1]

    img_ = np.ones((img_height, img_width)) if label else np.zeros((img_height, img_width))
    img_ = img.astype('float32') * .5

    img = scipy.ndimage.zoom(img, (img_height / img_height_pre, img_width / img_width_pre))

    # if label:
    #     img += .1
    # imageio.imwrite('test.png', (img*255).astype('uint8'))
    # assert 0
    img = np.expand_dims(img, axis=-1)
    return img

# Create an instance of the DICOMDataGenerator with data augmentation
datagen = DICOMDataGenerator(
    # rescale=1.0/255,
     rotation_range=5,
     width_shift_range=0.1,
     height_shift_range=0.1,
     shear_range=0.05,
     zoom_range=0.1,
     horizontal_flip=True
)

datagen2 = DICOMDataGenerator(
    # rescale=1.0/255
)

# Load the training data
train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle=True
)

# Load the validation data
validation_generator = datagen2.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle=False
)

class ResnetBlockV2(keras.layers.Layer):

    def __init__(self, kernel_size=(3, 3), filter_size=16, stride=1, dif_fsize=False, **kwargs):
        super(ResnetBlockV2, self).__init__(**kwargs)

        if stride == 1:
            strides = (1, 1)
        else:
            strides = (stride, stride)

        self.bn2a = BatchNormalization()
        self.conv2a = Conv2D(filter_size, kernel_size, strides=strides, padding='same', name='conv2a', data_format='channels_last')

        self.bn2b = BatchNormalization()
        self.conv2b = Conv2D(filter_size, kernel_size, strides=(1, 1), padding='same', name='conv2b', data_format='channels_last')

        self.drop = SpatialDropout2D(0.5, data_format='channels_last')

        self.use_identity_shortcut = (stride == 1) and not dif_fsize
        if not self.use_identity_shortcut:
            self.conv2_sc = tf.keras.layers.Conv2D(filter_size, (1, 1), strides=strides, padding='same', name='conv2_sc', data_format='channels_last')

    def call(self, input_tensor, training=False):
        x = self.bn2a(input_tensor, training=training)
        x1 = tf.nn.relu(x)
        x = self.conv2a(x1)

        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2b(x)

        x = self.drop(x)

        if self.use_identity_shortcut:
            skip = input_tensor
        else:
            skip = self.conv2_sc(x1)
        x += skip

        return x

# Model architecture
# model = Sequential()
# model.add(Conv2D(16, kernel_size=(3, 3), input_shape=(img_width, img_height, 1), strides=(1, 1),padding='same'))
# model.add(BatchNormalization())
# model.add(ReLU())
# model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=16, name='resnet_block1'))
# model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=16, name='resnet_block2'))
# model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=16, name='resnet_block3'))
# model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=32, stride=2, name='resnet_block4'))  # 14x14
# model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=32, name='resnet_block5'))
# model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=32, name='resnet_block6'))
# model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=64, stride=2, name='resnet_block7'))  # 7x7
# model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=64, name='resnet_block8'))
# model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=64, name='resnet_block9'))
# model.add(GlobalAveragePooling2D())
# model.add(Dense(1, activation='sigmoid'))  # Output layer with a single unit for binary classification

# model = Sequential()
# model.add(Conv2D(64, kernel_size=(7, 7), input_shape=(img_width, img_height, 1), strides=(4,4),padding='same'))
# model.add(BatchNormalization())
# model.add(ReLU())
# model.add(ResnetBlockV2(kernel_size=(7, 7), filter_size=64, name='resnet_block1',stride= 4))
# model.add(ResnetBlockV2(kernel_size=(7, 7), filter_size=64, stride=2, name='resnet_block2'))
# model.add(ResnetBlockV2(kernel_size=(5, 5), filter_size=64, stride=2,name='resnet_block3'))
# model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=128, stride=2, name='resnet_block4')) 
# model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=128, stride=2,name='resnet_block5'))
# model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=128, stride=2,name='resnet_block6'))
# model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=256,stride=2, name='resnet_block7')) 
# model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=256,stride=2,name='resnet_block8'))
# model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=256,stride=2,name='resnet_block9'))
# model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=512,stride=4,name='resnet_block10'))
# model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=512,stride=2,name='resnet_block11'))
# model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=512,stride=2,name='resnet_block12'))
# model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=1024,stride=2,name='resnet_block13'))
# model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=1024,stride=2,name='resnet_block14'))
# model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=1024,stride=2,name='resnet_block15'))
# model.add(GlobalAveragePooling2D())
# model.add(Dense(1, activation='sigmoid'))  # Output layer with a single unit for binary classification

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():

    # orig_ch = 16

    # model = Sequential()
    # model.add(Conv2D(orig_ch, kernel_size=(9, 9), input_shape=(img_width, img_height, 1), data_format='channels_last', strides=(4,4),padding='same'))
    # model.add(BatchNormalization())
    # model.add(ReLU())
    # model.add(Conv2D(orig_ch*4, kernel_size=(9, 9), strides=(4,4), data_format='channels_last', padding='same'))
    # model.add(BatchNormalization())
    # model.add(ReLU())
    # model.add(ResnetBlockV2(kernel_size=(5, 5), filter_size=orig_ch*8, name='resnet_block1',stride= 2, dif_fsize=True))
    # model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*8, stride=1, name='resnet_block2'))
    # model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*8, stride=1,name='resnet_block3'))
    # model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*8, stride=1, name='resnet_block4')) 
    # model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*8, stride=1,name='resnet_block5'))
    # model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*8, stride=1,name='resnet_block6'))
    # model.add(ResnetBlockV2(kernel_size=(5, 5), filter_size=orig_ch*16,stride=2, name='resnet_block7', dif_fsize=True)) 
    # model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*16,stride=1,name='resnet_block8'))
    # model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*16,stride=1,name='resnet_block9'))
    # model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*16,stride=1,name='resnet_block10'))
    # model.add(ResnetBlockV2(kernel_size=(5, 5), filter_size=orig_ch*32,stride=2,name='resnet_block11', dif_fsize=True))
    # model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*32,stride=1,name='resnet_block12'))
    # model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*32,stride=1,name='resnet_block13'))
    # model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*32,stride=1,name='resnet_block14'))
    # model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*32,stride=1,name='resnet_block15'))
    # model.add(ResnetBlockV2(kernel_size=(5, 5), filter_size=orig_ch*64,stride=2,name='resnet_block16', dif_fsize=True))
    # model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*64,stride=1,name='resnet_block17'))
    # model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*64,stride=1,name='resnet_block18'))
    # model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*64,stride=1,name='resnet_block19'))
    # model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*64,stride=1,name='resnet_block20'))
    # model.add(GlobalAveragePooling2D())
    # #model.add(Flatten())
    # model.add(ReLU())
    # model.add(Dropout(.2))
    # model.add(Dense(100))
    # model.add(ReLU())
    # model.add(Dropout(.2))
    # # model.add(ReLU())
    # model.add(Dense(1, activation='sigmoid'))  # Output layer with a single unit for binary classification
    # # model.add(Dense(1))  # Output layer with a single unit for binary classification

    
    orig_ch = 16

    model = Sequential()
    model.add(Conv2D(orig_ch, kernel_size=(9, 9), input_shape=(img_width, img_height, 1), data_format='channels_last', strides=(4,4),padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2D(orig_ch*4, kernel_size=(9, 9), strides=(4,4), data_format='channels_last', padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(ResnetBlockV2(kernel_size=(5, 5), filter_size=orig_ch*8, name='resnet_block1',stride= 2, dif_fsize=True))
    model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*8, stride=1, name='resnet_block2'))
    model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*8, stride=1,name='resnet_block3'))
    model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*8, stride=1, name='resnet_block4')) 
    model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*8, stride=1,name='resnet_block5'))
    model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*8, stride=1,name='resnet_block6'))
    model.add(ResnetBlockV2(kernel_size=(5, 5), filter_size=orig_ch*16,stride=2, name='resnet_block7', dif_fsize=True)) 
    model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*16,stride=1,name='resnet_block8'))
    model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*16,stride=1,name='resnet_block9'))
    model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*16,stride=1,name='resnet_block10'))
    model.add(ResnetBlockV2(kernel_size=(5, 5), filter_size=orig_ch*32,stride=2,name='resnet_block11', dif_fsize=True))
    model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*32,stride=1,name='resnet_block12'))
    model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*32,stride=1,name='resnet_block13'))
    model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*32,stride=1,name='resnet_block14'))
    model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*32,stride=1,name='resnet_block15'))
    model.add(ResnetBlockV2(kernel_size=(5, 5), filter_size=orig_ch*64,stride=2,name='resnet_block16', dif_fsize=True))
    model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*64,stride=1,name='resnet_block17'))
    model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*64,stride=1,name='resnet_block18'))
    model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*64,stride=1,name='resnet_block19'))
    model.add(ResnetBlockV2(kernel_size=(3, 3), filter_size=orig_ch*64,stride=1,name='resnet_block20'))
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    # model.add(Dropout(.2))
    model.add(ReLU())
    # model.add(Dropout(.2))
    model.add(Dense(100))
    model.add(ReLU())
    model.add(Dropout(.2))
    # model.add(ReLU())
    model.add(Dense(1, activation='sigmoid'))  # Output layer with a single unit for binary classification
    # model.add(Dense(1))  # Output layer with a single unit for binary classification

    # weighted BCE
    class WeightedBinaryCrossentropy:
        def __init__(self, weights = [1.0, 1.0]):
            self.weights = weights
            self.eps = tf.constant(tf.keras.backend.epsilon())
            
        def __call__(self, target, output):
            output = tf.clip_by_value(output, self.eps, 1.0-self.eps)
            print(target.dtype)
            print(output.dtype)
            target = tf.dtypes.cast(target, tf.float32)
            # output= output.int64
            # (output.to(torch.int64))
            # tf.cast(output, tf.int64)
            print(output.dtype)
            bce = self.weights[1] * target * tf.math.log(output + self.eps)
            bce += self.weights[0] * (1-target) * tf.math.log(1 - output + self.eps)
            return tf.reduce_mean(-bce, axis=-1)

    # count training data
    count_positive = 0
    count_negative = 0
    print('pi')
    for i, (_, label) in enumerate(train_generator):
        print(i)
        if i == int(np.ceil(num_train_samples / batch_size)):
            break
        count_positive += np.sum(label == 1)
        count_negative += np.sum(label == 0)
        print(count_positive, count_negative)

    print("training dataset count : positive=%d, negative=%d", (count_positive, count_negative))

    # Compile the model
    model.compile(optimizer=SGD(lr=0.01, momentum=0.9), 
#        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
        loss=WeightedBinaryCrossentropy((1/count_negative, 1/count_positive)),
        metrics=['accuracy', 'AUC']
    )

    # keras.utils.plot_model(model, to_file='model.png')

    # import imageio
    # for i, (gray, label) in enumerate(train_generator):
    #     print(gray.shape)
    #     print(label)
    #     imageio.imwrite('image%i.png'%i, np.clip(gray[0,:,:,0]*255, 0, 255).astype('uint8'))

    # assert(0)

# Train the model and store the training history
history = model.fit(
    train_generator,
    steps_per_epoch=int(np.ceil(num_train_samples / batch_size)),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=int(np.ceil(num_validation_samples / batch_size)),
    callbacks=[early_stopping]
    # callbacks=[CustomCallback()]
)

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
plt.savefig("/home/Nishta/resnetv2loss101023.png")

# Plot the training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
plt.savefig("/home/Nishta/resnetv2acc.png")


# Load the test data
test_generator = datagen2.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=1,
    shuffle=False
)

# # Evaluate the model on the test data
# test_loss, _ = model.evaluate_generator(test_generator)
# print("Test Loss:", test_loss)
# loss, accuracy = model.evaluate_generator(test_generator)
# # print(f'Test Loss: {loss:.4f}')
# # print(f'Test Accuracy: {accuracy:.4f}')
# print('Test Loss: {:.4f}'.format(loss))
# print('Test Accuracy: {:.4f}'.format(accuracy))

# # Stop measuring time
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Training time: {elapsed_time:.2f} seconds")

# Evaluate the model on the test data
# loss, accuracy = model.evaluate(test_generator,steps= 356,verbose =1)
loss, accuracy, _ = model.evaluate(test_generator,steps= 356,verbose =1)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

# Get true labels for the test data
test_labels = []
for subdir, _, files in os.walk(test_data_dir):
    for file in files:
        if file.endswith('.dcm'):
            label = int(os.path.basename(subdir))
            test_labels.append(label)

# Predict probabilities for the test data
y_pred_prob = model.predict(test_generator, steps=356, verbose=1)
# y_pred_prob = y_pred_prob1.flatten()  # Flatten the array
# print(y_pred_prob)
# print(test_labels)
print("Number of test labels:", len(test_labels))
print("Number of predicted probabilities:", len(y_pred_prob)) 


# Calculate the false positive rate (FPR), true positive rate (TPR), and thresholds
fpr, tpr, thresholds = roc_curve(test_labels, y_pred_prob)

# Calculate the Area Under the ROC Curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) of Resnet v-2')
plt.legend(loc='lower right')
plt.savefig("/home/Nishta/resnetv2ROC101023.png")
plt.show()

# Nested Cross-Validation
outer_cv = KFold(n_splits=n_splits_outer, shuffle=True, random_state=42)

roc_auc_scores = []

files = DICOMDataGenerator().filelist(data_dir)

for train_index, test_index in outer_cv.split(files):
    train_files, test_files = [files[i] for i in train_index], [files[i] for i in test_index]
    train_images, train_labels = load_data(train_files)
    test_images, test_labels = load_data(test_files)

    # Inner Cross-Validation for hyperparameter tuning
    inner_cv = KFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
    best_auc = 0
    best_model = None

    for inner_train_index, inner_val_index in inner_cv.split(train_files):
        inner_train_files, inner_val_files = [train_files[i] for i in inner_train_index], [train_files[i] for i in inner_val_index]
        inner_train_images, inner_train_labels = load_data(inner_train_files)
        inner_val_images, inner_val_labels = load_data(inner_val_files)

        # model = build_model()
        model.fit(inner_train_images, inner_train_labels, epochs=epochs, validation_data=(inner_val_images, inner_val_labels), batch_size=batch_size)

        # Evaluate the model
        val_preds = model.predict(inner_val_images).flatten()
        val_auc = roc_auc_score(inner_val_labels, val_preds)
        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model

    # Test the best model from inner CV
    test_preds = best_model.predict(test_images).flatten()
    test_auc = roc_auc_score(test_labels, test_preds)
    roc_auc_scores.append(test_auc)

average_auc = np.mean(roc_auc_scores)
print("Average AUROC after Nested Cross-Validation:", average_auc)

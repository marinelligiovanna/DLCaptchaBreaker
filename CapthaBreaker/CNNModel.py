# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from datetime import datetime
import os


class CNNModel:
    """
    Build and train a CNN Model with two convolutional layers to recognize the characters of a Captcha.
    The train and tests sets must be composed of the separated characters of the Captcha obtained using the ImageProcessor class in this project.
    Feel free to implement your own image processor for your needs.
    """

    def __init__(self):
        # Initialize Sequential CNN
        self.model = Sequential()

    def buildModel(self, img_shape = 20, num_categories = 10):
        """
        Build a Convolutional Neural Network model with two convolutional layers and two hidden layers.
        This model is set to use 32 feature detectors in each convolutional layer and 128 nodes in the first hidden layer.

        :param self:
        :param img_shape: The shape of the input images. Images must be of size img_shape x img_shape.
        :param num_categories: Number of categories to be classified. In this case, 10 for numbers from 0 to 9.
        :return:
        """

        # Add first convolutional layer and max pooling
        self.model.add(Conv2D(32,                                       # Number of features detectors (filters)
                              (3,                                       # 3 Row (conv. matrix)
                               3),                                      # 3 Columns (conv.matrix usually 3x3)
                              input_shape=(img_shape, img_shape, 3),    # 3 Channels, img_shape x img_shape images
                              activation='relu',                        # Activation function to increase nonlinearity of images
                              name = 'FirstConv2DLayer'))               # Layer name
        self.model.add(MaxPooling2D(pool_size=(2, 2)))                  # Size of pooling matrix (usually 2x2)

        # Add second convolutional layer and max pooling
        self.model.add(Conv2D(32, (3, 3), input_shape=(img_shape, img_shape, 3), activation='relu', name ='SecondConv2DLayer'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Flattening
        self.model.add(Flatten())

        # Full connection
        self.model.add(Dense(units=128, activation='relu'))

        # One output for each category we predicted want to predict
        self.model.add(Dense(units=num_categories, activation='softmax', name='Probabilities'))  # Softmax for more than two outcome

        # Compile the CNN (Gradient descent)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



    def trainModel(self, train_set_path, test_set_path, img_shape, output_dir, batch_size=32, num_epochs=25):
        """
        Train the CNN model built. This method assumes the set structure of ImageDataGenerator specified in Keras documentation to recognize the number of categories.

        :param self:
        :param train_set_path: Path of the train set. Must contain a folder for each category inside it.
        :param test_set_path:  Path of the test set. Same structure as train set path.
        :param img_shape: The shape of the input images. Images must be of size img_shape x img_shape.
        :param output_dir: Directory to save the CNN model CNN_Model.h5.
        :param batch_size:
        :param num_epochs:
        :return:
        """

        train_datagen = ImageDataGenerator(
            # Random transform - Data augmenting
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        training_set = train_datagen.flow_from_directory(
            train_set_path,
            target_size=(img_shape, img_shape),
            batch_size=32,
            class_mode='categorical')

        test_set = test_datagen.flow_from_directory(
            test_set_path,
            target_size=(img_shape, img_shape),
            batch_size=batch_size,
            class_mode='categorical')


        # Saves the model weights after each epoch if the validation loss decreased
        now = datetime.now()
        nowstr = 'CNN_Model'

        now = os.path.join(output_dir, nowstr)

        # Make the directory
        os.makedirs(now, exist_ok = True)

        # Create callbacks
        savepath = os.path.join(now, 'CNN_Model.h5')
        checkpointer = ModelCheckpoint(filepath=savepath, monitor='val_acc', mode='max', verbose=0, save_best_only=True)
        fout = open(os.path.join(now, 'labels.txt'), 'wt')

        # Write labels to file
        for key, val in training_set.class_indices.items():
            fout.write(str(key + '\n'))

        self.model.fit_generator(
            training_set,
            steps_per_epoch=len(training_set.filenames) // batch_size,
            epochs=num_epochs,
            validation_data=test_set,
            validation_steps=len(test_set.filenames) // batch_size,
            callbacks=[checkpointer])

if __name__ == '__main__':


    PATH = 'U:\\Dataset\\'
    train_set_path = PATH + 'training_set'
    test_set_path = PATH + 'test_set'
    img_shape = 20
    output_dir = PATH

    cnnModel = CNNModel()
    cnnModel.buildModel(img_shape = img_shape)
    cnnModel.trainModel(img_shape = img_shape, train_set_path = train_set_path, test_set_path = test_set_path, output_dir = output_dir, num_epochs=25)


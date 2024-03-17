# Module to run the ANN model
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Masking
from keras.callbacks import TensorBoard, EarlyStopping
import os

# LSTM model to perform sign classification given key positions
class SignClassifier(keras.Model) :
    def __init__(self, input_shape, output_shape, epochs, pad_value):
        """
        Initializes the SignClassifier object.

        Args:
            input_shape (tuple): The shape of the input data (nb_frames, frame_features_nb)
            output_shape (tuple): The shape of the output data (class_nb,)
            epochs (int): The number of epochs to train the model. Recommended : 2000

        Returns:
            None
        """

        super(SignClassifier, self).__init__()

        # Parameters of the model
        self.input_shpe = input_shape
        self.output_shpe = output_shape
        self.log_dir = os.path.join('logs')
        self.tb_callback = TensorBoard(log_dir=self.log_dir)
        self.early_stopping = EarlyStopping(patience = 5, restore_best_weights = True)
        self.epochs = epochs
        self.pad_value = pad_value

        # Building model
        self.build_model()

    def build_model(self) :
        self.model = Sequential()

        # From one LSTM layer to another, the output is used as input of the next one
        self.model.add(Masking(mask_value=self.pad_value, input_shape=self.input_shpe))
        self.model.add(LSTM(64, activation='tanh', return_sequences = True))
        self.model.add(LSTM(128, activation='tanh', return_sequences = True))
        self.model.add(LSTM(64, activation='tanh', return_sequences = False))

        # Now, we keep the values of the short-memory and long-memory to perform classification
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))

        # Finally adding the output layer
        self.model.add(Dense(self.output_shpe, activation='softmax'))

        # Compiling the model
        opt = keras.optimizers.Adam(clipnorm=1.0)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    def fit(self, X_train, Y_train, X_val, Y_val) :
        return self.model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=self.epochs, callbacks=[self.tb_callback, self.early_stopping])

    def fit(self, X_train, Y_train) :
        return self.model.fit(X_train, Y_train, epochs=self.epochs, callbacks=[self.tb_callback])

    def predict(self, X) :
        return self.model.predict(X)
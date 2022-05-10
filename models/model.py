'''
This file implements the proposed model outlined in the literature review/specification
'''

import os
import random
import pickle

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
from collections import defaultdict
import time
import keras.backend as K
import copy

import common

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_obj, file_name):
        self.best = 0
        self.model_obj = model_obj
        self.file_name = file_name
        with open(f"{self.file_name}.json", "w") as f:
            f.write(model_obj.model.to_json())

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None and logs['val_accuracy'] > self.best:
            print("\nSaving model...")
            # self.model.save_weights(self.file_name, save_format='h5')
            with open(f"{self.file_name}_config.config", "wb") as f:
                pickle.dump(self.model_obj.get_config(), f)
            self.best = logs['val_accuracy']

def load_model(checkpoint_file, raw_data=True):
    file_name = os.path.basename(checkpoint_file)
    seq_len, seq_num = file_name.split("_")[:2]
    loaded_model = tf.keras.models.load_model(checkpoint_file, custom_objects={"keras_layer": hub.KerasLayer})

    model = Model(raw_data=raw_data, seq_length=int(seq_len), seq_num=int(seq_num), model=loaded_model)
    return model

class Model:
    '''
    @param *_size :: Specifies the shape of a FNN as (number of layers, width of each layer)
    @param raw_data :: To use the raw data or the market-cap normalized data.
    @param seq_length :: The length of each sequence to be processed by ALBERT. MAX=512
    @param seq_num :: Number of sequences to process in ALBERT. The processed sequences will be concatenated.
    @param aggreagator_size :: The shape of the aggregator FNN.
    @param albert_dense_size :: The shape of the dense network to be placed after ALBERT. An additional la
    @param gru_size :: The shape of each GRU (this shape defines the shape for GRU in both directions)
    @param gru_dense_size :: The shape of the dense network to be placed after the Bi-GRU network.
    @param encode_len :: Length of the output of the whole bi-GRU network and ALBERT.
    @param classifier_size :: shape of the classifier FNN
    @param drop_rate :: The dropout rate to be used after each dense layer
    @param gru_drop_rate :: Dropout rate for GRU layers
    @param regularizer :: Regularizer to be applied to kernels and recurrent kernels
    '''
    def __init__(self, raw_data=False, seq_length=None, seq_num=None, aggregator_size=None, albert_dense_size=None,
                    gru_shape=None, gru_dense_size=None, encode_len=None,
                    classifier_size=None, drop_rate=0.3, gru_drop_rate=0.05,
                    regularizer=None, batch_size=64, checkpoint_file="./model_checkpoint", model=None):

        if None in [seq_length, seq_num]:
            raise AttributeError("Some of the attributes are not defined")

        self.raw_data = raw_data
        self.train_files, self.test_files = common.train_test_files(raw_data=self.raw_data)
        print("Getting scaler...")
        start = time.time()
        self.scaler = common.get_normalizer(raw_data=self.raw_data, force_new=False)
        print(f"Finished getting scaler. Took {time.time() - start} secs")
        self.seq_len = seq_length
        self.seq_num = seq_num

        self.albert_dense_size = albert_dense_size
        self.aggergator_size = aggregator_size
        self.gru_shape = gru_shape
        self.gru_dense_size = gru_dense_size
        self.encode_len = encode_len
        self.classifier_size = classifier_size
        self.drop_rate = drop_rate
        self.gru_drop_rate = gru_drop_rate
        self.regularizer = regularizer
        self.parsed_data = None
        self.test_dataset = False
        self.checkpoint_file = checkpoint_file

        self.batch_size = batch_size
        self.preprocessor = self.create_preprocessor()
        self.fin_features = common.get_fin_features(raw_data=raw_data)

        if model is None:
            self.model = self.get_model(len(self.fin_features))
        else:
            self.model = model

    def get_config(self):
        return {
            "raw_data": self.raw_data,
            "seq_len": self.seq_len,
            "seq_num": self.seq_num,
            "aggregator_size": self.aggergator_size,
            "albert_dense_size": self.albert_dense_size,
            "gru_shape": self.gru_shape,
            "gru_dense_size": self.gru_dense_size,
            "encode_len": self.encode_len,
            "classifier_size": self.classifier_size,
            "drop_rate": self.drop_rate,
            "gru_drop_rate": self.gru_drop_rate,
            "regularizer": self.regularizer,
            "batch_size": self.batch_size,
            "checkpoint_file": self.checkpoint_file
        }

    # Return the preprocessor for ALBERT
    def create_preprocessor(self):
        preprocessor = hub.load("https://tfhub.dev/tensorflow/albert_en_preprocess/3")
        # Step 1: tokenize batches of text inputs.
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
        tokenize = hub.KerasLayer(preprocessor.tokenize, trainable=False)
        tokenized_inputs = [tokenize(text_input)]

        # Step 3: pack input sequences for the Transformer encoder.
        bert_pack_inputs = hub.KerasLayer(preprocessor.bert_pack_inputs, trainable=False,
                                          arguments=dict(seq_length=self.seq_len))
        encoder_inputs = bert_pack_inputs(tokenized_inputs)
        return tf.keras.Model(text_input, encoder_inputs)

    def load_to_memory(self):
        if self.parsed_data is None:
            print("Loading all data into memory")
            start = time.time()
            self.parsed_data = common.load_h5_dataset(raw_data=self.raw_data)
            print(f"Finished loading. Took {time.time() - start} secs")

    def preprocess_data(self, data, fin_only=False, mda_only=False):
        for k, v in data.items():
            # Normalize all the financial data
            if type(v) is dict:
                data[k] = self.scaler.transform([[v[x] for x in self.fin_features]]).tolist()[0]
                if mda_only:
                    shape = np.array(data[k]).shape
                    data[k] = np.zeros(shape).tolist()

        # Extract the necessary sections from the mda according to seq_len and seq_num
        # The middle seq_len*seq_num words of the mda is extracted.
        if fin_only:
            texts = [""]
        else:
            mda_text = data["mda"].split(" ")
            texts = []
            start_idx = 250
            for i in range(self.seq_num):
                start = start_idx + i * self.seq_len
                end = start + self.seq_len
                if end < len(mda_text):
                    texts.append(mda_text[start:end])
                elif start < len(mda_text) <= end:
                    texts.append(mda_text[start:])
                else:
                    texts.append("")
            texts = [" ".join(x) for x in texts]

        # Preprocess the mda texts
        data["mda"] = [tf.squeeze(v) for v in self.preprocessor(tf.convert_to_tensor(texts)).values()]

        return {k: np.nan_to_num(v) for k, v in data.items() if k != "y"}, [1 if data["y"] == x else 0 for x in [-1, 0, 1]]

    '''
    Generator that parses and yields each data point at a time.

    @param test :: Whether to load the test data or not.
    '''
    def data_generator(self, test=False, fin_only=False, mda_only=False):
        self.load_to_memory()
        if test:
            files = self.test_files
        else:
            files = self.train_files

        balances = self.get_balances()

        for file in files:
            data = self.parsed_data[file]

            X, y = self.preprocess_data(data, fin_only, mda_only)
            if test:
                repeat_num = 1
            else:
                repeat_num = max(0, np.random.normal(max(balances.values())/balances[data["y"]], 1))
            for _ in range(round(repeat_num)):
                yield X, y

    '''
    Return the training dataset as lists.

    @param test :: Whether to return the test data or training data.

    @return :: (X, y)
    '''

    def get_dataset_as_array(self, test=False, max_count=None, fin_only=False, mda_only=False):
        X, y = defaultdict(list), []
        count = 0
        for x in self.data_generator(test, fin_only=fin_only, mda_only=mda_only):
            if max_count is not None and count >= max_count:
                break
            for k, v in x[0].items():
                X[k.replace("-", "")].append(v)
            y.append(x[1])
            count += 1
            if count % 100 == 0:
                print("get_dataset_as_array count: ", count)
        return dict(X), y

    # Return the training dataset as a tf.data.Dataset
    def get_dataset(self, test=False, fin_only=False, mda_only=False):
        filename = f'{self.seq_len}_{self.seq_num}_raw_{test}_{fin_only}_{mda_only}.tfrecord' if self.raw_data \
            else f'{self.seq_len}_{self.seq_num}_{test}_{fin_only}_{mda_only}.tfrecord'
        if not os.path.exists(os.path.join(DIR_PATH, filename)):
            print("Creating new dataset...")
            start = time.time()
            dataset = tf.data.Dataset.from_tensor_slices(self.get_dataset_as_array(test, fin_only=fin_only, mda_only=mda_only))
            tf.data.experimental.save(dataset, os.path.join(DIR_PATH, filename))
            print(f"Finished creating dataset {time.time() - start}")
            return dataset
        return tf.data.experimental.load(os.path.join(DIR_PATH, filename))

    def albert(self):
        mask_input = tf.keras.layers.Input(shape=(self.seq_len,), dtype=tf.int32, name="input_mask")
        type_ids_input = tf.keras.layers.Input(shape=(self.seq_len,), dtype=tf.int32, name="input_type_ids")
        word_ids_input = tf.keras.layers.Input(shape=(self.seq_len,), dtype=tf.int32, name="input_word_ids")

        encoder_out = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_base/3", trainable=False)(
            {
             "input_mask": mask_input,
             "input_type_ids": type_ids_input,
             "input_word_ids": word_ids_input,
             }
        )
        out = encoder_out["pooled_output"]
        for i in range(self.albert_dense_size[0]):
            out = tf.keras.layers.Dense(self.albert_dense_size[1], activation="swish", kernel_regularizer=self.regularizer)(out)
            out = tf.keras.layers.Dropout(self.drop_rate)(out)
        out = tf.keras.layers.Dense(self.encode_len, activation="swish", kernel_regularizer=self.regularizer)(out)
        out = tf.keras.layers.Dropout(self.drop_rate)(out)

        return tf.keras.Model(inputs={
             "input_mask": mask_input,
             "input_type_ids": type_ids_input,
             "input_word_ids": word_ids_input,
             }, outputs=out, name="albert")

    # The aggregator network
    def aggregator(self):
        fin_input = tf.keras.layers.Input(shape=(len(common.get_fin_features(raw_data=self.raw_data))), dtype=tf.float32)
        fin_avg_input = tf.keras.layers.Input(shape=(len(common.get_fin_features(raw_data=self.raw_data))), dtype=tf.float32)
        x = tf.keras.layers.Concatenate(axis=-1)((fin_input, fin_avg_input))
        for _ in range(self.aggergator_size[0]):
            x = tf.keras.layers.Dense(self.aggergator_size[1], activation="swish", kernel_regularizer=self.regularizer)(x)
            x = tf.keras.layers.Dropout(self.drop_rate)(x)

        # x = tf.keras.layers.LayerNormalization()(x)
        return tf.keras.Model(inputs=[fin_input, fin_avg_input], outputs=x, name="aggregator")

    # Bi-directional GRU for the financials
    def bi_gru(self):
        y_2input = tf.keras.layers.Input(shape=(self.aggergator_size[1], 1), dtype=tf.float32)
        y_1input = tf.keras.layers.Input(shape=(self.aggergator_size[1], 1), dtype=tf.float32)
        y0input = tf.keras.layers.Input(shape=(self.aggergator_size[1], 1), dtype=tf.float32)
        inputs = tf.keras.layers.Concatenate(axis=-2)([y_2input, y_1input, y0input])
        reverse_inputs = tf.keras.layers.Concatenate(axis=-2)([y0input, y_1input, y_2input])

        gru1 = inputs
        for i in range(self.gru_shape[0]):
            return_sequences = not (i == self.gru_shape[0]-1)
            gru1 = tf.keras.layers.GRU(self.gru_shape[1], return_sequences=return_sequences,
                                    recurrent_regularizer=self.regularizer,
                                    kernel_regularizer=self.regularizer,
                                    dropout=self.gru_drop_rate,
                                    name=f"forward_gru_{i}")(gru1)
        gru1 = tf.keras.layers.Flatten()(gru1)

        gru2 = reverse_inputs
        for i in range(self.gru_shape[0]):
            return_sequences = not (i == self.gru_shape[0]-1)
            gru2 = tf.keras.layers.GRU(self.gru_shape[1], return_sequences=return_sequences,
                                    recurrent_regularizer=self.regularizer,
                                    kernel_regularizer=self.regularizer,
                                    dropout=self.gru_drop_rate,
                                    name=f"backward_gru_{i}")(gru2)
        gru2 = tf.keras.layers.Flatten()(gru2)

        outputs = tf.keras.layers.Concatenate()([gru1, gru2])

        for _ in range(self.gru_dense_size[0]):
            outputs = tf.keras.layers.Dense(self.gru_dense_size[1], activation="swish", kernel_regularizer=self.regularizer)(outputs)
            outputs = tf.keras.layers.Dropout(self.drop_rate)(outputs)

        outputs = tf.keras.layers.Dense(self.encode_len, activation="swish", kernel_regularizer=self.regularizer)(outputs)
        outputs = tf.keras.layers.Dropout(self.drop_rate)(outputs)

        return tf.keras.Model(inputs=[y_2input, y_1input, y0input], outputs=outputs, name="bi_gru")

    # The final classification layer
    def classifier(self,):
        albert_input = tf.keras.layers.Input(shape=self.encode_len, dtype=tf.float32)
        fin_input = tf.keras.layers.Input(shape=self.encode_len, dtype=tf.float32)

        x = tf.keras.layers.Concatenate()([albert_input, fin_input])
        # x = tf.keras.layers.LayerNormalization()(x)

        for _ in range(self.classifier_size[0]):
            x = tf.keras.layers.Dense(self.classifier_size[1], activation="swish", kernel_regularizer=self.regularizer)(x)
            x = tf.keras.layers.Dropout(self.drop_rate)(x)

        x = tf.keras.layers.Dense(3, activation="softmax")(x)

        return tf.keras.Model(inputs=[albert_input, fin_input], outputs=x, name="classifier")

    # The full model
    def get_model(self, fin_num_features):
        if self.seq_num > 1:
            mda_input = tf.keras.layers.Input(shape=(3, self.seq_num, self.seq_len), dtype=tf.int32, name="mda_input")
            albert_out = [self.albert()(
                            {"input_mask": tf.squeeze(mda_input[:, 0, i, :]),
                            "input_type_ids": tf.squeeze(mda_input[:, 1, i, :]),
                            "input_word_ids": tf.squeeze(mda_input[:, 2, i, :])}
                            ) for i in range(self.seq_num)]
        else:
            mda_input = tf.keras.layers.Input(shape=(3, self.seq_len), dtype=tf.int32)
            albert_out = self.albert()({"input_mask": tf.squeeze(mda_input[:, 0, :]),
                                        "input_type_ids": tf.squeeze(mda_input[:, 1, :]),
                                        "input_word_ids": tf.squeeze(mda_input[:, 2, :])}
                                        )

        y_2fin_input = tf.keras.layers.Input(shape=(len(common.get_fin_features(raw_data=self.raw_data)),), name="y2fin")
        y_1fin_input = tf.keras.layers.Input(shape=(len(common.get_fin_features(raw_data=self.raw_data)),), name="y1fin")
        y_0fin_input = tf.keras.layers.Input(shape=(len(common.get_fin_features(raw_data=self.raw_data)),), name="y0fin")
        y_2_avg_fin_input = tf.keras.layers.Input(shape=(len(common.get_fin_features(raw_data=self.raw_data)),), name="y2_avg_fin")
        y_1_avg_fin_input = tf.keras.layers.Input(shape=(len(common.get_fin_features(raw_data=self.raw_data)),), name="y1_avg_fin")
        y_0_avg_fin_input = tf.keras.layers.Input(shape=(len(common.get_fin_features(raw_data=self.raw_data)),), name="y0_avg_fin")

        aggregator = self.aggregator()

        fin2 = aggregator((y_2fin_input, y_2_avg_fin_input))
        fin1 = aggregator((y_1fin_input, y_1_avg_fin_input))
        fin0 = aggregator((y_0fin_input, y_0_avg_fin_input))
        fin = self.bi_gru()((fin2, fin1, fin0))

        out = self.classifier()((albert_out, fin))

        model = tf.keras.Model(inputs={
                                "mda": mda_input,
                                "y2fin": y_2fin_input,
                                "y1fin": y_1fin_input,
                                "y0fin": y_0fin_input,
                                "y2_avg_fin": y_2_avg_fin_input,
                                "y1_avg_fin": y_1_avg_fin_input,
                                "y0_avg_fin": y_0_avg_fin_input
                            }, outputs=[out])

        model.compile(tf.keras.optimizers.Adam(clipnorm=1), loss="categorical_crossentropy", metrics=["accuracy", "mse"])
        return model

    '''
    Train the model.
    Model stops training once the specified number of epochs has been reached or
    val_loss does not improve for 5 consecutive epochs.
    '''
    def train(self, epochs, lr=0.001, beta_1=0.9, beta_2=0.999):
        print("Creating tf Dataset")
        start = time.time()
        train_dataset = self.get_dataset().shuffle(20000, reshuffle_each_iteration=False)

        validation_size = 512
        validation_dataset = train_dataset.take(validation_size).batch(self.batch_size)
        train_dataset = train_dataset.skip(validation_size).prefetch(tf.data.AUTOTUNE).shuffle(10000, reshuffle_each_iteration=True)
        train_dataset = train_dataset.batch(self.batch_size)
        print(f"Created dataset. Took {time.time()-start} secs")

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_file,
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        stopping_callback = tf.keras.callbacks.EarlyStopping(
                                monitor='val_accuracy', min_delta=0.00000001, patience=5)
        self.model.fit(train_dataset, epochs=epochs,
                      validation_data=validation_dataset,
                      callbacks=[model_checkpoint_callback, CheckpointCallback(self, self.checkpoint_file), stopping_callback],
                      verbose=1)

    def evaluate(self):
        if not self.test_dataset:
            self.test_dataset = self.get_dataset(test=True).batch(32)
        return self.model.evaluate(self.test_dataset)

    def predict(self, data):
        return self.model.predict(data, batch_size=8)

    def save_test_predictions(self, fin_only=False, mda_only=False):
        file_name = f"test_predictions_{fin_only}_{mda_only}.pkl"
        if not self.test_dataset:
            print("Creating test dataset")
            self.test_dataset = self.get_dataset(test=True, fin_only=fin_only, mda_only=mda_only).batch(32)
        print("starting prediction...")
        predictions = self.model.predict(self.test_dataset, verbose=1)
        print("saving...")
        with open(os.path.join(DIR_PATH, file_name), "wb") as f:
            pickle.dump(predictions, f)
        return predictions

    def test_predictions(self, use_cached=False, fin_only=False, mda_only=False):
        file_name = f"test_predictions_{fin_only}_{mda_only}.pkl"
        if use_cached and os.path.exists(os.path.join(DIR_PATH, file_name)):
            with open(os.path.join(DIR_PATH, file_name), "rb") as f:
                predictions = pickle.load(f)
        else:
            predictions = self.save_test_predictions(fin_only, mda_only)

        return predictions

    def get_balances(self):
        self.load_to_memory()
        balances = defaultdict(lambda: 0)
        counter = 0
        for file in self.train_files:
            data = self.parsed_data[file]
            balances[data["y"]] += 1
            counter += 1

        return balances


if __name__ == "__main__":
    # Create a new model. Alternatively, a saved model could be loaded
    model_obj = Model(raw_data=True, seq_length=512, seq_num=1,
                        aggregator_size=(3, 128), albert_dense_size=(2, 850),
                        gru_shape=(1, 96), gru_dense_size=(1, 480),
                        encode_len=536, classifier_size=(2, 320),
                        drop_rate=0.186, gru_drop_rate=0.277,
                        regularizer=tf.keras.regularizers.L2(4e-05),
                        batch_size=128, checkpoint_file="./model_checkpoint")
    # model_obj = load_model("512_1_.49acc", raw_data=True) # Load an existing model

    model_obj.train(100)  # Train for a maximum of 100 epochs
    model_obj.model.summary(expand_nested=True)  # Show summary of the model
    model_obj.evaluate()  # Test the model on the test set

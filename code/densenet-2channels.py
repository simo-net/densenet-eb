import os
import json
import argparse
import tensorflow as tf
from pandas import DataFrame
import matplotlib.pyplot as plt

# Usage:
# python3 /home/cnn2d/code/densenet-2channels.py  --data_path /home/cnn2d/data/N-MNIST  --log_path /home/cnn2d/data/logs/N-MNIST  --validation_split 0.2  --batch_size 100  --epochs 300  --lr 0.01  --dropout 0.2  --patience 10  --random_seed 0
# python3 /home/cnn2d/code/densenet-2channels.py  --data_path /home/cnn2d/data/FN-MNIST  --log_path /home/cnn2d/data/logs/FN-MNIST  --validation_split 0.2  --batch_size 100  --epochs 300  --lr 0.01  --dropout 0.2  --patience 10  --random_seed 0


NUM_CLASSES = 10
IMG_SHAPE = (34, 34)
NUM_TRAIN_SAMPLES = 60000 * 10
NUM_TEST_SAMPLES = 10000 * 10


def parse_args():
    parser = argparse.ArgumentParser(
        description='Training a 2D-CNN with DenseNet architecture on two-channeled (ON and OFF) images from the\n'
                    'data_path and save results to the given log_path (3 files will be created here: checkpoint,\n'
                    'model.json and model_weights_final.hdf5).')

    parser.add_argument('--data_path', type=str, required=True,
                        help='Directory where all data is stored (must contain a "train" and a "test" folder\n'
                             'with all RGB images).')
    parser.add_argument('--log_path', type=str, required=True,
                        help='Directory where to store the logs of the model (training history, architecture and\n'
                             'final weights).')

    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Fraction of the training dataset to use as validation. Default is 0.2.')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size to use for loading the data on GPU memory. Default is 100.')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to use for training the model. Default is 300.')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='The learning rate to use for training the model. Default is 0.01.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Fraction of input units to dropout in the readout layer during training. Default is 0.2.')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of epochs early-stopping patience. Default is 10.')
    parser.add_argument('--early_stopping', action="store_true", default=False,
                        help='Whether to use early stopping. Default is False.')

    parser.add_argument('--random_seed', type=int, default=None,
                        help='The random seed to use for reproducibility. Default is None.')

    return parser.parse_args()


def load_data(path: str,
              validation_split: float = None, subset: str = None,
              batch_size: int = 100, shuffle: bool = False, random_seed: int = None):
    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    dataset = data_generator.flow_from_directory(
        path, subset=subset,
        target_size=IMG_SHAPE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=shuffle, batch_size=batch_size,
        seed=random_seed,
    )
    print(f'{"Data" if not subset else subset.upper() + " set"} was successfully loaded.')
    return dataset


def create_data_generator(dataset):
    while True:
        batch_x, batch_y = dataset.next()
        yield batch_x[..., :2], batch_y


def build_model(input_shape: tuple,
                top_dropout_rate: float, lr: float,
                model_file: str = None):

    # Build model from DenseNet
    inputs = tf.keras.layers.Input(shape=input_shape)
    model = tf.keras.applications.densenet.DenseNet121(include_top=False,
                                                       input_tensor=inputs,
                                                       classes=NUM_CLASSES,
                                                       weights=None)

    # Rebuild top
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs, outputs, name="DenseNet")

    # Compile
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    # We could also possibly try: optimizer = keras.optimizers.Adam(learning_rate=lr)
    # We could also possibly try: loss = keras.losses.SparseCategoricalCrossentropy()

    # Print and store model info
    model.summary()
    if isinstance(model_file, str):
        with open(model_file, 'w+') as outfile:
            outfile.write(json.dumps(json.loads(model.to_json()), indent=2))

    return model


def train(training_set, validation_set, model,
          epochs: int, steps_per_epoch: int = None, validation_steps: int = None,
          early_stop: bool = True, patience: int = 10,
          checkpoint_path: str = '../logs/checkpoint', weights_file: str = None):

    # Build the checkpoint
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max',
                                                   patience=patience,
                                                   verbose=1)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(monitor='val_accuracy', mode='max',
                                                     filepath=checkpoint_path, save_weights_only=True,
                                                     save_best_only=True, save_freq='epoch')

    # Train the model
    history = model.fit(training_set, validation_data=validation_set,
                        epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                        callbacks=[es_callback, cp_callback] if early_stop else [cp_callback],
                        verbose=True)

    # Store trained weights
    if isinstance(weights_file, str):
        model.save_weights(weights_file)

    return model, history


def evaluate(test_set, model, history):
    # Print score
    loss, acc = model.evaluate(test_set, verbose=1)
    print(f'Performance: loss = {round(loss,4)}, accuracy {round(acc,4)}')

    # Plot 1
    DataFrame(history.history).plot()

    # Plot 2
    fig, axs = plt.subplots(figsize=(16, 5), nrows=1, ncols=2)
    axs[0].plot(history.history['val_loss'])
    axs[0].set_title('Validation Loss History')
    axs[0].set(xlabel='# epochs', ylabel='Loss value')
    axs[1].plot(history.history['val_accuracy'])
    axs[1].set_title('Validation Accuracy History')
    axs[1].set(xlabel='# epochs', ylabel='Accuracy value')
    plt.show()


class NoGPUError(Exception):
    pass


def main():

    # Load all given parameters --------------------------------------------------------------------#
    args = parse_args()
    os.makedirs(args.log_path, exist_ok=True)

    # Check if GPU is available --------------------------------------------------------------------#
    if not tf.config.list_physical_devices('GPU'):
        raise NoGPUError('No tf-compatible GPU found. Aborting.')

    # Specify a random seed for reproducibility ----------------------------------------------------#
    if args.random_seed is not None:
        tf.random.set_seed(args.random_seed)

    # Load the training and validation data --------------------------------------------------------#
    print('\n\nLoading training and validation data...')
    training_data = load_data(path=os.path.join(args.data_path,'train'),
                              validation_split=args.validation_split, subset="training",
                              batch_size=args.batch_size, shuffle=True, random_seed=args.random_seed)
    assert training_data.samples == int(NUM_TRAIN_SAMPLES*(1-args.validation_split))
    validation_data = load_data(path=os.path.join(args.data_path,'train'),
                                validation_split=args.validation_split, subset="validation",
                                batch_size=args.batch_size, shuffle=True, random_seed=args.random_seed)
    assert validation_data.samples == int(NUM_TRAIN_SAMPLES*args.validation_split)

    # Build the model ------------------------------------------------------------------------------#
    print('\n\nBuilding the model...')
    model = build_model(input_shape=(*IMG_SHAPE, 2),
                        top_dropout_rate=args.dropout, lr=args.lr,
                        model_file=os.path.join(args.log_path,'model.json'))

    # Train the model ------------------------------------------------------------------------------#
    print('\n\nTraining the model...')
    model, history = train(model=model,
                           training_set=create_data_generator(training_data),
                           validation_set=create_data_generator(validation_data),
                           epochs=args.epochs,
                           steps_per_epoch=int(NUM_TRAIN_SAMPLES*(1-args.validation_split))//args.batch_size,
                           validation_steps=int(NUM_TRAIN_SAMPLES*args.validation_split)//args.batch_size,
                           early_stop=args.early_stopping, patience=args.patience,
                           checkpoint_path=os.path.join(args.log_path,'checkpoint'),
                           weights_file=os.path.join(args.log_path,'model_weights_final.hdf5'))

    # Load the test data ---------------------------------------------------------------------------#
    print('\n\nLoading the test data...')
    test_data = load_data(path=os.path.join(args.data_path,'test'),
                          validation_split=None, subset=None,
                          batch_size=args.batch_size, shuffle=False, random_seed=None)
    assert test_data.samples == NUM_TEST_SAMPLES

    # Evaluate model performance -------------------------------------------------------------------#
    print('\n\nEvaluating the model...')
    evaluate(model=model, test_set=create_data_generator(test_data), history=history)


if __name__ == "__main__":
    main()

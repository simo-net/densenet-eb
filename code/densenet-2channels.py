import os
import json
import pickle
import argparse
import tensorflow as tf
from pandas import DataFrame
import matplotlib.pyplot as plt

# Usage:
# python3 /home/cnn2d/code/densenet-2channels.py  --data_path /home/cnn2d/data/N-MNIST  --validation_split 0.2  --batch_size 100  --epochs 300  --lr 0.01  --dropout 0.2  --early_stopping  --patience 20
# python3 /home/cnn2d/code/densenet-2channels.py  --data_path /home/cnn2d/data/FN-MNIST  --validation_split 0.2  --batch_size 100  --epochs 300  --lr 0.01  --dropout 0.2  --early_stopping  --patience 20


NUM_CLASSES = 10
IMG_SHAPE = (34, 34)
NUM_TRAIN_SAMPLES = 60000 * 10
NUM_TEST_SAMPLES = 10000 * 10


def parse_args():
    parser = argparse.ArgumentParser(
        description='Training a 2D-CNN with DenseNet architecture on two-channeled (ON and OFF) images from the\n'
                    'data_path and save results to the given log_path (4 files will be created here: checkpoint, history,\n'
                    'model.json and final_weights.hdf5).')

    parser.add_argument('--data_path', type=str, required=True,
                        help='Directory where all data is stored (must contain a "train" and a "test" folder\n'
                             'with all RGB images).')
    parser.add_argument('--log_path', type=str, default=None,
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
    parser.add_argument('--early_stopping', action="store_true", default=False,
                        help='Whether to use early stopping. Default is False.')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of epochs to use as early-stopping patience. Default is 10.')

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
                model_file: str = '../logs/model.json'):

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
    with open(model_file, 'w+') as outfile:
        outfile.write(json.dumps(json.loads(model.to_json()), indent=2))

    return model


def train(training_set, validation_set, model,
          epochs: int, steps_per_epoch: int = None, validation_steps: int = None,
          early_stop: bool = True, patience: int = 10,
          checkpoint_file: str = '../logs/checkpoint',
          history_file: str = '../logs/history',
          weights_file: str = '../logs/final_weights.hdf5'):

    # Build the early-stopping and checkpoint callbacks
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min',
        # monitor='val_accuracy', mode='max',
        patience=patience, restore_best_weights=True,
        verbose=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        monitor='val_loss', mode='min',
        # monitor='val_accuracy', mode='max',
        filepath=checkpoint_file, save_weights_only=True, save_best_only=True, save_freq='epoch',
        verbose=True)

    # Train the model
    history = model.fit(training_set, validation_data=validation_set,
                        epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                        callbacks=[es_callback, cp_callback] if early_stop else [cp_callback],
                        verbose=True)

    # Store training history
    with open(history_file, 'wb') as outfile:
        pickle.dump(history.history, outfile)
    # Read as: history = pickle.load(open(history_file), "rb")

    # Store trained weights
    model.save_weights(weights_file)

    return model, history


def plot_history(history):
    # Plot 1
    DataFrame(history).plot()

    # Plot 2
    fig, axs = plt.subplots(figsize=(16, 5), nrows=1, ncols=2)
    axs[0].plot(history['val_loss'])
    axs[0].set_title('Validation Loss History')
    axs[0].set(xlabel='# epochs', ylabel='Loss value')
    axs[1].plot(history['val_accuracy'])
    axs[1].set_title('Validation Accuracy History')
    axs[1].set(xlabel='# epochs', ylabel='Accuracy value')
    plt.show()


def evaluate(model, test_set, evaluation_steps: int, eval_file: str):
    results = model.evaluate(test_set,
                             steps=evaluation_steps,
                             verbose=True)
    res_dict = {}
    print('Performance:')
    for name, value in zip(model.metrics_names, results):
        print(f'   {name} -> {value}')
        res_dict[name] = value
    with open(eval_file, "w+") as f:
        json.dump(res_dict, f, indent=2)


class NoGPUError(Exception):
    pass


def main():

    # Load all given parameters --------------------------------------------------------------------#
    args = parse_args()

    # Define output log directory ------------------------------------------------------------------#
    log_path = args.log_path
    if log_path is None or not isinstance(log_path, str):
        log_path = os.path.join(os.path.dirname(args.data_path), 'logs', os.path.basename(args.data_path))
    os.makedirs(log_path, exist_ok=True)

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
                        model_file=os.path.join(log_path, 'model.json'))

    # Train the model ------------------------------------------------------------------------------#
    print('\n\nTraining the model...')
    model, history = train(model=model,
                           training_set=create_data_generator(training_data),
                           validation_set=create_data_generator(validation_data),
                           epochs=args.epochs,
                           steps_per_epoch=int(NUM_TRAIN_SAMPLES*(1-args.validation_split))//args.batch_size,
                           validation_steps=int(NUM_TRAIN_SAMPLES*args.validation_split)//args.batch_size,
                           early_stop=args.early_stopping, patience=args.patience,
                           checkpoint_file=os.path.join(log_path, 'checkpoint'),
                           history_file=os.path.join(log_path, 'history'),
                           weights_file=os.path.join(log_path, 'final_weights.hdf5'))

    # Load the test data ---------------------------------------------------------------------------#
    print('\n\nLoading the test data...')
    test_data = load_data(path=os.path.join(args.data_path,'test'),
                          validation_split=None, subset=None,
                          batch_size=args.batch_size, shuffle=False, random_seed=None)
    assert test_data.samples == NUM_TEST_SAMPLES

    # Evaluate model performance -------------------------------------------------------------------#
    print('\n\nEvaluating the model...')
    evaluate(model=model, test_set=create_data_generator(test_data),
             evaluation_steps=NUM_TEST_SAMPLES//args.batch_size,
             eval_file=os.path.join(log_path, 'evaluation.json'))

    # Plot training history ------------------------------------------------------------------------#
    plot_history(history=history.history)


if __name__ == "__main__":
    main()

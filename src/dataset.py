import tensorflow as tf
import tensorflow_datasets as tfds
from augmentation import augment, convert

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64
NUM_EXAMPLES = 2048

def load_datasets():
    dataset, info = tfds.load(
        'mnist',
        as_supervised=True,
        with_info=True
    )

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    num_train_examples = info.splits['train'].num_examples

    augmented_train_batches = (
        train_dataset
        .take(NUM_EXAMPLES)
        .cache()
        .shuffle(num_train_examples // 4)
        .map(augment, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    non_augmented_train_batches = (
        train_dataset
        .take(NUM_EXAMPLES)
        .cache()
        .shuffle(num_train_examples // 4)
        .map(convert, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    validation_batches = (
        test_dataset
        .map(convert, num_parallel_calls=AUTOTUNE)
        .batch(2 * BATCH_SIZE)
    )

    return augmented_train_batches, non_augmented_train_batches, validation_batches

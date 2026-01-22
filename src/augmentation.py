import tensorflow as tf

def convert(image, label):
    # Cast and normalize image to [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label

def augment(image, label):
    image, label = convert(image, label)

    # Add padding and randomly crop back
    image = tf.image.resize_with_crop_or_pad(image, 34, 34)
    image = tf.image.random_crop(image, size=[28, 28, 1])

    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.5)

    return image, label

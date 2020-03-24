import pathlib
import tensorflow as tf


def get_image_label(dataset_dir):
    data_root = pathlib.Path(dataset_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    # get labels' names
    label_names = sorted(item.name for item in data_root.glob('*/'))
    # dict: {label : index}
    label_to_index = dict((index, label) for label, index in enumerate(label_names))
    # get all images' labels
    all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in
                       all_image_path]

    return all_image_path, all_image_label


def preprocess_data(image_path, image_label):
    image_raw = tf.io.read_file(image_path)
    image_tensor = tf.image.decode_image(image_raw, channels=3)
    image_tensor = tf.image.resize(image_tensor, [299, 299])
    image_tensor = tf.cast(image_tensor, dtype=tf.float32) / 255.

    return image_tensor, image_label


def get_dataset(dataset_dir):
    all_image_path, all_image_label = get_image_label(dataset_dir)

    dataset = tf.data.Dataset.from_tensor_slices((all_image_path, all_image_label))
    dataset = dataset.map(preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset, len(all_image_path)


def build_dataset():
    train_dataset, train_count = get_dataset('dataset/train')
    test_dataset, test_count = get_dataset('dataset/test')
    valid_dataset, valid_count = get_dataset('dataset/valid')

    train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(32)
    test_dataset = test_dataset.batch(32)
    valid_dataset = valid_dataset.batch(32)

    return train_dataset, train_count, test_dataset, test_count, valid_dataset, valid_count

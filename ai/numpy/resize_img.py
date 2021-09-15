import os
import cv2
import pathlib
import tensorflow as tf

def resize_img(src_dir, dst_dir, num_rows, num_cols, dst_type='.jpg') :
    path = pathlib.Path(src_dir)
    files = [str(path) for path in list(path.glob('*/*'))]
    file_count = len(files)

    for file in files:
        #content = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        content = cv2.imread(file)
        if content is None:
            continue
        content = cv2.resize(content, (num_rows, num_cols))
        class_name = pathlib.Path(file).parent.name
        dst = os.path.join(dst_dir, class_name)
        if not os.path.exists(dst):
            os.makedirs(dst)
        name = pathlib.Path(file).name + dst_type
        dst_file = os.path.join(dst, name)
        cv2.imwrite(dst_file, content)


if __name__ == '__main__' :
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    num_rows = 224
    num_cols = 224
    src_dir = data_dir #'E:/i_share/i_test3'
    dst_dir = 'E:/i_share/i_test4'
    resize_img(src_dir, dst_dir, num_rows, num_cols)

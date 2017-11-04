
import os
import tensorflow as tf
from datasets import caltect256
from nets import inception_v3
from preprocessing import inception_preprocessing

from datasets import dataset_utils

dataset_dir = '/tmp/kihong/caltech256'
# set your .ckpt file
checkpoints_dir = '/tmp/kihong/train_inception_v3_caltech256_FineTune_logs/all'

slim = tf.contrib.slim

batch_size = 3
image_size = 299

'''
def getimagedir(path):
    res = []
    
    for root, dirs, files in os.walk(path):
        rootpath = os.path.join(os.path.abspath(path), root)
        
        for file in files:
            filepath = os.path.join(rootpath, file)
            res.append(filepath)
            
    return res
'''

with tf.Graph().as_default():
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        # set your image
        #imgPath = '/mnt/notebook/kihong/models/slim/images/dog.jpg'
        imgPath = 'images/dog2.jpg'
        testImage_string = tf.gfile.FastGFile(imgPath, 'rb').read()
        testImage = tf.image.decode_jpeg(testImage_string, channels=3)
        processed_image = inception_preprocessing.preprocess_image(testImage, image_size, image_size, is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)

        logits, _ = inception_v3.inception_v3(processed_images, num_classes=257, is_training=False)
        probabilities = tf.nn.softmax(logits)

        init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'model.ckpt-500'), slim.get_model_variables('InceptionV3'))

        with tf.Session() as sess:
            init_fn(sess)

            np_image, probabilities = sess.run([processed_images, probabilities])
            probabilities = probabilities[0, 0:]
            sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

            names = dataset_utils.read_label_file(dataset_dir)
            #names = caltect256.create_readable_names_for_caltect256_labels()
            for i in range(15):
                index = sorted_inds[i]
                print((probabilities[index], names[index]))
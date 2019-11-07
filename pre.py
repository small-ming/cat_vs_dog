# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import create_and_read_TFRecord2 as reader2
import os
from PIL import Image

def per_class(imagefile):

    image = Image.open(imagefile)
    image = image.resize([227, 227])
    image_array = np.array(image)

    image = tf.cast(image_array,tf.float32)
    image = tf.image.per_image_standardization(image)
    image = tf.reshape(image, [1, 227, 227, 3])

    sess=tf.Session()
    saver = tf.train.import_meta_graph('./model/AlexNetModel.ckpt.meta')
    saver.restore(sess,'./model/AlexNetModel.ckpt')
    graph=tf.get_default_graph()
    y_pred=graph.get_tensor_by_name('y_pred:0')
    x=graph.get_tensor_by_name('x:0')
    y_true = graph.get_tensor_by_name('y_true:0')
    y_test_images=np.zeros((1,2))
    feed_dict_testing={x:image,y_true:y_test_images}
    result=sess.run(y_pred,feed_dict=feed_dict_testing)
    res_label=['dog','cat']
    print(res_label[result.argmax()])

    # saver = tf.train.Saver()
    # with tf.Session() as sess:
    #
    #     save_model =  tf.train.latest_checkpoint('.//model')
    #     saver.restore(sess, save_model)
    #     image = tf.reshape(image, [1, 227, 227, 3])
    #     image = sess.run(image)
    #     prediction = sess.run(fc3, feed_dict={x: image})
    #
    #     max_index = np.argmax(prediction)
    #     if max_index==0:
    #         return "cat"
    #     else:
    #         return "dog"

per_class('4.jpg')
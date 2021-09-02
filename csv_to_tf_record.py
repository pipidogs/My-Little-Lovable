# -*- coding: utf-8 -*-

"""
    @Name:         create_tf_record
    @Date:         2021/9/2
    @Description:  构建tf_record文件
"""
from loguru import logger
import pandas as pd
import tensorflow as tf
import numpy as np


# filename = './反洗钱题目与文件的预测文件_1.csv'
# df = pd.read_csv(filename, encoding='utf8')
# df.to_records(filename[:-3]+'tf_record')


def write_tf_record():
    """
    写入tf_record文件

    :return:
    """
    logger.info('开始写入tf_record文件')
    with tf.io.TFRecordWriter('./test_tf_record.tf_record') as file_writer:
        for _ in range(4):
            x, y = np.random.random(), np.random.random()
            record_bytes = tf.train.Example(features=tf.train.Features(feature={
                "x": tf.train.Feature(float_list=tf.train.FloatList(value=[x])),
                "y": tf.train.Feature(float_list=tf.train.FloatList(value=[y])),
            })).SerializeToString()
            file_writer.write(record_bytes)
    logger.info('tf_record文件写入完毕。')


def decode_fn(record_bytes):
    """
    设定取值的格式

    :param record_bytes: 字节数据
    :return:
    """
    return tf.io.parse_single_example(
        # Data
        record_bytes,
        # Schema
        {"x": tf.io.FixedLenFeature([], dtype=tf.float32),
         "y": tf.io.FixedLenFeature([], dtype=tf.float32)}
    )


def read_tf_record():
    """
    读取tf_record文件

    :return:
    """
    logger.info('开始加载tf_record文件。')
    filenames = ['./test_tf_record.tf_record']

    # 读取
    raw_dataset = tf.data.TFRecordDataset(filenames).map(decode_fn)
    for batch in raw_dataset:
        print("x = {x:.4f},  y = {y:.4f}".format(**batch))


if __name__ == '__main__':
    write_tf_record()
    read_tf_record()
    pass

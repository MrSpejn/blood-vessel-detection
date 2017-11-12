import io
import tensorflow as tf
import numpy as np
from PIL import Image
from math import floor

WIDTH = 876
SIZE = 5
TRAIN_HEALTHY = [
    '06_h',
    '07_h',
    '08_h',
    '09_h',
    '10_h',
    '11_h',
    # '12_h',
    # '13_h',
    # '14_h',
    # '15_h',
]
TEST_HEALTHY = [
    '01_h',
    '02_h',
    '03_h',
    '04_h',
    '05_h',
]
TRAIN_GLUCOMA = [
    '06_g',
    '07_g',
    '08_g',
    '09_g',
    '10_g',
    '11_g',
    # '12_g',
    # '13_g',
    # '14_g',
    # '15_g',
]
TEST_GLUCOMA = [
    '01_g',
    '02_g',
    '03_g',
    '04_g',
    '05_g',
]

test_cases = len(TEST_HEALTHY + TEST_GLUCOMA)

def extract_kernels(matrix, ref_matrix, kernel_size = SIZE, flatten = True):
    print('Extracting')
    kernels = []
    w = matrix.shape[0]
    h = matrix.shape[1]

    offset = floor(kernel_size / 2)

    for i in range(offset, w - offset):
        for j in range(offset, h - offset):
            kernel = matrix[i-offset:i+offset+1, j-offset:j+offset+1, 1:3]
            record = (kernel.reshape(-1) if flatten else kernel, [ float(ref_matrix[i, j] > 150), float(ref_matrix[i, j] <= 150) ])
            kernels.append(record)


    # y = np.asarray(kernels)[:, 1]
    # y = np.asarray(list(y))
    # data = ((np.argmax(y, axis=1) + 1) % 2) * 255
    # print(data)

    # image_data = np.asarray(data, dtype=np.int8).reshape(-1, WIDTH - 2*floor(SIZE/2))
    # image = Image.fromarray(image_data)
    # image.show()

    return kernels

def preprocess_data(filename = '01_g'):
    eye = np.asarray(Image.open('imagedatabase/{}_resized.jpg'.format(filename)))
    eye_ref = np.asarray(Image.open('imagedatabase/{}ref_resized.tif'.format(filename)))

    return np.asarray(extract_kernels(eye, eye_ref))


def build_network():
    input_layer_units = SIZE*SIZE*2
    output_layer_units = 2
    hidden_layer_units = 500

    x = tf.placeholder(tf.float32, [None, input_layer_units], name="x")
    y = tf.placeholder(tf.float32, [None, output_layer_units], name="y")


    hidden_bias = tf.Variable(tf.random_normal([hidden_layer_units]))
    hidden_weights = tf.Variable(tf.random_normal([input_layer_units, hidden_layer_units]))
    output_bias = tf.Variable(tf.random_normal([output_layer_units]))
    output_weights = tf.Variable(tf.random_normal([hidden_layer_units, output_layer_units]))

    hidden_layer = tf.add(tf.matmul(x, hidden_weights), hidden_bias)
    hidden_layer = tf.nn.relu(hidden_layer)

    output_layer = tf.matmul(hidden_layer, output_weights) + output_bias
    
    cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y, output_layer, 0.7))

    return (cost, x, y, output_layer)

def network_learning(network, train_data, test_data, learning_rate = 0.0003, epochs = 80):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(network[0])
    init = tf.global_variables_initializer()
    ( cost, x, y, output_layer ) = network

    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(init)
        cost = 10000000
        epoch = 1
        while epoch < epochs:
            result = network_epoch(network, train_data, optimizer, session)
            cost = result[1]
            print("Epoch:", (epoch), "cost =", "{:.5f}".format(cost))
            epoch += 1

        save_path = saver.save(session, "./output/model_{}_{}_weighted.ckpt".format(SIZE, epochs))
        print("Model saved in path: %s" % save_path)

        # saver.restore(session, "./output/model_21_2_weighted.ckpt")

        single_image_kernels = floor(len(test_data) / test_cases)
        for i in range(test_cases):
            image_test_data = test_data[i*single_image_kernels:(i+1)*single_image_kernels]
            test_data_x = image_test_data[:, 0]
            test_data_y = image_test_data[:, 1]
            predicted_class = tf.argmax(y, 1)
            real_class = tf.argmax(output_layer, 1)

            accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_class, real_class), "float"))
            confusion_matrix = tf.confusion_matrix(real_class, predicted_class)

            output_values = ((1 - tf.argmax(output_layer, 1))*255).eval({ x: list(test_data_x) })

            image_data = np.asarray(output_values, dtype=np.int8).reshape(-1, WIDTH - 2*floor(SIZE/2))
            image = Image.fromarray(image_data)
            image.show(title='./output/result{}.jpg'.format((TEST_GLUCOMA + TEST_HEALTHY)[i]))

            print("Validation Accuracy:", accuracy.eval({ x: list(test_data_x), y: list(test_data_y) }))
            print(confusion_matrix.eval({ x: list(test_data_x), y: list(test_data_y) }))


def network_epoch(network, train_data, optimizer, session, batch_size=10000):
    ( cost, x, y, _ ) = network
    avg_cost = 0

    for batch_idx in range(20):
        batch = train_data[((batch_idx)*batch_size):(batch_idx+1)*batch_size]
        batch_x = batch[:, 0]
        batch_y = batch[:, 1]
        
        if (len(batch_x) == 0):
            break

        result = session.run([optimizer, cost], feed_dict = { x: list(batch_x), y: list(batch_y) })
        
        avg_cost += result[1] / 1
    
    return (network, avg_cost)
         

if __name__ == "__main__":   
    train = None
    test = None
    for filename in TRAIN_GLUCOMA + TRAIN_HEALTHY:
        data = preprocess_data(filename)
        train = data if train is None else np.concatenate((train, data))
    
    for filename in TEST_GLUCOMA + TEST_HEALTHY:
        data = preprocess_data(filename)
        test = data if test is None else np.concatenate((test, data))

    network = build_network()

    network_learning(network, train, test)

    
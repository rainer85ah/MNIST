
"""
The lenet_mnist.py  script will be our driver program used to instantiate the LeNet network architecture,
train the model (or load the model, if our network is pre-trained),
and then evaluate the network performance on the MNIST dataset.
"""
from cnn.network.lenet import LeNet
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import datasets, metrics
from keras.datasets import mnist
from keras.optimizers import SGD, Adadelta
from keras.utils import np_utils, visualize_util
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
                help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
                help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str, help="(optional) path to weights file")
args = vars(ap.parse_args())

# grab the MNIST dataset (if this is your first time running this script, the download may take a minute.
# the 55MB MNIST dataset will be downloaded)
print("[INFO] downloading MNIST...")
dataset = datasets.fetch_mldata("MNIST Original")
# (trainData, trainLabels), (testData, testLabels) = mnist.load_data()

# reshape the MNIST dataset from a flat list of 784-dim vectors, to
# 28 x 28 pixel images, then scale the data to the range [0, 1.0]
# and construct the training and testing splits
data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
data = data[:, np.newaxis, :, :]
(trainData, testData, trainLabels, testLabels) = train_test_split(data / 255.0, dataset.target.astype("int"),
                                                                  train_size=0.85, test_size=0.15)

# transform the training and testing labels into vectors in the
# range [0, classes] -- this generates a vector for each label,
# where the index of the label is set to `1` and all other entries
# to `0`; in the case of MNIST, there are 10 class labels
trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

# initialize the optimizer and model
"""
Stochastic gradient descent (often shortened in SGD), also known as incremental gradient descent, is a stochastic
approximation of the gradient descent optimization method for minimizing an objective function that is written as
a sum of differentiable functions. In other words, SGD tries to find minimums or maximums by iteration.
Many extensions to vanilla SGD exist, including Momentum, Adagrad, rmsprop, Adadelta or Adam.

***
In my own experience, Adagrad/Adadelta are "safer" because they don't depend so strongly on setting of
learning rates (with Adadelta being slightly better),
but well-tuned SGD+Momentum almost always converges faster and at better final values.
"""
# Adadelta is a gradient descent based learning algorithm that adapts the learning rate per parameter over time.
# Adadelta It is similar to rmsprop and can be used instead of vanilla SGD.
# opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model = LeNet.build(width=28, height=28, depth=1, classes=10, weights_path=args["weights"] if args["load_model"] > 0
                    else None)

# Show summary of model
print("[INFO] Summary of the model...")
model.summary()

config = model.get_config()
print "Config: "
for i, conf in enumerate(config):
    print i, conf.__str__()

print("[INFO] Compiling model...")
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# model.compile(loss="mean_squared_error", optimizer=sgd, metrics=["accuracy"])

# only train and evaluate the model if we *are not* loading a pre-existing model
if args["load_model"] < 0:
    print("[INFO] Training...")
    #  early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    h = model.fit(trainData, trainLabels, batch_size=128, nb_epoch=100, verbose=1, validation_split=0.15)
    #  callbacks = [early_stopping]

    # print(h.history)

    # show the accuracy on the testing set
    print("[INFO] Evaluating...")
    loss = model.evaluate(testData, testLabels, batch_size=128, verbose=1)
    print "Test Score: ", loss[0]
    print "Test Accuracy", loss[1]
    visualize_util.plot(model=model, to_file='/home/rainer85ah/PycharmProjects/PythonProjects/Output/PLOT_9930.png',
                        show_layer_names=True,show_shapes=False)

# check to see if the model should be saved to file
if args["save_model"] > 0:
    print("[INFO] dumping weights to file...")
    model.save_weights(args["weights"], overwrite=True)
    model.save('/home/rainer85ah/PycharmProjects/PythonProjects/Output/model_saved_9930.h5', overwrite=True)
    # returns a compiled model identical to the previous one
    # model = load_model('my_model.h5') from keras.models import load_model
    TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True)


"""
# randomly select a few testing digits
for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
    # classify the digit
    probs = model.predict(testData[np.newaxis, i])
    prediction = probs.argmax(axis=1)

    # resize the image from a 28 x 28 image to a 96 x 96 image so we can better see it
    image = (testData[i][0] * 255)
    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, str(prediction[0]), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # show the image and prediction
    print("[INFO] Predicted: {}, Actual: {}".format(prediction[0], np.argmax(testLabels[i])))
    cv2.imshow("Digit", image)
    cv2.waitKey(0)
"""

from keras.datasets import reuters
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

'''
------------------------------Result------------------------------
hidden floor 2, hidden unit 64-64, epochs=13, model-activation=categorical_crossentropy
0.7894033789634705

hidden floor 2, hidden unit 64-64, epochs=13, model-activation=sparse_categorical_crossentropy
0.7733749151229858

hidden floor 2, hidden unit 64-4, epochs=13, model-activation=categorical_crossentropy
0.6558325886726379

hidden floor 2, hidden unit 128-128, epochs=13, model-activation=categorical_crossentropy
0.6558325886726379

hidden floor 2, hidden unit 128-128, epochs=13, model-activation=categorical_crossentropy
0.7965271472930908

hidden floor 3, hidden unit 128-128-128, epochs=6, model-activation=categorical_crossentropy
0.7764915227890015

hidden floor 3, hidden unit 128-128-128, epochs=13, model-activation=categorical_crossentropy
0.7880676984786987
'''

test_mode = raw_input("test mode? (y/n) -> ")
epochs = input("epochs -> ")

def vectorize_sequences(sequences, dimension=10000):
        results = np.zeros((len(sequences),dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels),dimension))
    for i, label in enumerate(labels):
        results[i,label] = 1.
    return results

(train_data, train_labels), (test_data,test_labels) = reuters.load_data(num_words=10000)

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

#one_hot_train_labels = np.array(train_labels) # int tensor incoding
#one_hot_test_labels = np.array(test_labels) # int tensor incoding

'''
from keras.utils.np_utils import to_categorical
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)
'''

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
#model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc']) # int tensor incoding

if(test_mode=='n'):
    model.fit(x_train,one_hot_train_labels, epochs=epochs, batch_size=512)
    results = model.evaluate(x_test,one_hot_test_labels)
    print(results)
else:
    history = model.fit(partial_x_train,partial_y_train, epochs=epochs, batch_size=512, validation_data=(x_val,y_val)) # check overfitting by epochs=20
    results = model.evaluate(x_train,one_hot_train_labels)
    print(results)
    
    # check overfitting by plt
    history_dict = history.history
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1,len(loss)+1)

    plt.plot(epochs,loss,'bo',label= 'Training loss')
    plt.plot(epochs,val_loss,'b',label= 'Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']

    plt.plot(epochs,acc,'bo',label= 'Training acc')
    plt.plot(epochs,val_acc,'b',label= 'Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

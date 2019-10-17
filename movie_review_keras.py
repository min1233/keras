from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

'''
------------------------------Result------------------------------
hidden floor 2, hidden unit 16, loss=binary_crossentropy, epochs=3, model-activiation=relu
0.8820400238037109

hidden floor 1, hidden unit 16, loss=binary_crossentropy, epochs=3, model-activiation=relu
0.8886399865150452

hidden floor 2, hidden unit 32, loss=binary_crossentropy, epochs=3, model-activiation=relu
0.8858000040054321

hidden floor 2, hidden unit 64, loss=binary_crossentropy, epochs=2, model-activiation=relu
0.8709999918937683

hidden floor 2, hidden unit 32, loss=mse, epochs=2, model-activiation=relu
0.8877999782562256

hidden floor 2, hidden unit 32, loss=mse, epochs=2, model-activiation=tanh
0.8725600242614746

hidden floor 1, hidden unit 32, loss=mse, epochs=2, model-activiation=relu
0.8880800008773804

hidden floor 1, hidden unit 32, loss=binary_crossentropy, epochs=3, model-activiation=relu
0.8891199827194214
'''

test_mode = raw_input("test mode? (y/n) -> ")
epochs = input("epochs -> ")

def vectorize_sequences(sequences, dimension=10000):
        results = np.zeros((len(sequences),dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results

(train_data, train_labels), (test_data,test_labels) = imdb.load_data(num_words=10000)

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

model = models.Sequential()
model.add(layers.Dense(32,activation='relu', input_shape=(10000,)))
#model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
#model.compile(optimizer='rmsprop', loss='mse', metrics=['acc'])

if(test_mode=='n'):
    model.fit(x_train,y_train, epochs=epochs, batch_size=512)
    results = model.evaluate(x_test,y_test)
    print(results)
else:
    history = model.fit(partial_x_train,partial_y_train, epochs=epochs, batch_size=512, validation_data=(x_val,y_val)) # check overfitting by epochs=20
    results = model.evaluate(x_train,y_train)
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

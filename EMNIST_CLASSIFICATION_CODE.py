from keras.utils import np_utils 
import pandas as pd
import numpy as np
from keras.initializers import RandomNormal
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from keras.models import Sequential 
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l1, l2
from keras.activations import relu, elu, sigmoid
from keras.callbacks import EarlyStopping
from sklearn.model_selection import RandomizedSearchCV
from keras.layers import Conv2D, MaxPooling2D, Flatten
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score



dftrain=pd.read_csv(r"C:\Users\karth\OneDrive\Desktop\emnist-balanced-test.csv")
dftest=pd.read_csv(r"C:\Users\karth\OneDrive\Desktop\emnist-balanced-test.csv")


x_train=dftrain.iloc[:,1:]
y_train = dftrain.iloc[:,0]
x_test = dftest.iloc[:,1:]
y_test = dftest.iloc[:,0]

print("Number of training examples :", x_train.shape[0], "and each image is of shape (%d)"%(x_train.shape[1]))
print("Number of testing examples :", x_test.shape[0], "and each image is of shape (%d)"%(x_test.shape[1]))


# mapping_path =r"C:\Users\karth\OneDrive\Desktop\emnist-balanced-mapping.txt"
with open(r"C:\Users\karth\OneDrive\Desktop\emnist-balanced-mapping.txt") as f:
    lines = f.readlines()
    mapping = {int(line.split()[0]): line.split()[1] for line in lines}
    


# Plot a random sample of images
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10,10))
for i, ax in enumerate(axes.flat):
    index = np.random.randint(0, x_train.shape[0])
    image = x_train.iloc[index].values.reshape(28, 28) 
    label = mapping[y_train.iloc[index]]
    ax.imshow(image, cmap='gray')
    ax.set_title(label)
    ax.axis('off')
plt.show()



# lets convert this into a 10 dimensional vector
# ex: consider an image is 5 convert it into 5 => [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
# this conversion needed for MLPs 

Y_train_m = np_utils.to_categorical(y_train, 47) 
Y_test_m = np_utils.to_categorical(y_test, 47)


# resize and normalize
X_train_m = np.reshape(x_train, [-1, x_train.shape[1]])
X_train_m = X_train_m.astype('float32') / 255
X_test_m = np.reshape(x_test, [-1, x_test.shape[1]])
X_test_m = X_test_m.astype('float32') / 255


# define the model
def create_model_mlp(learning_rate, activation, optimizer, use_batchnorm,regularizer, dropout_rate, num_layers, num_neurons):
    model = Sequential()
    model.add(Dense(num_neurons, input_dim=X_train_m.shape[1]))
    if use_batchnorm:
        model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))
    for i in range(num_layers-1):
        model.add(Dense(num_neurons))
        if use_batchnorm:
            model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(Dropout(dropout_rate))
    if regularizer == 'l1':
        model.add(Dense(47, activation='softmax', kernel_regularizer=l1(0.01)))
    elif regularizer == 'l2':
        model.add(Dense(47, activation='softmax', kernel_regularizer=l2(0.01)))
    else:
        model.add(Dense(47, activation='softmax'))
    if optimizer=='Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer=='RMSprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer=='SGD':
        optimizer=SGD(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# create the KerasClassifier for GridSearchCV
model_mlp = KerasClassifier(build_fn=create_model_mlp, epochs=20, batch_size=256, verbose=0)

# define the hyperparameters to search
param_grid_m = {
    'learning_rate': [0.001, 0.01, 0.1],
    'activation': ['relu', 'elu', 'sigmoid' ],
    'optimizer': ['Adam', 'RMSprop', 'SGD'],
    'use_batchnorm': [True, False],
    'regularizer':[None,'l1', 'l2'],
    'dropout_rate': [0,0.5],
    'num_layers': [2, 3, 4],
    'num_neurons': [64, 128, 256]
}
start_time_m=time.time()
# create the GridSearchCV object
grid_m = RandomizedSearchCV(estimator=model_mlp, param_distributions=param_grid_m,n_jobs=-1)

grid_result_m = grid_m.fit(X_train_m, Y_train_m)

# Print the best hyperparameters found
print("Best Hyperparameters: ", grid_result_m.best_params_)

best_params_m = grid_result_m.best_params_
model_m = create_model_mlp(learning_rate=best_params_m['learning_rate'], activation=best_params_m['activation'], optimizer=best_params_m['optimizer'], 
                         use_batchnorm=best_params_m['use_batchnorm'], regularizer=best_params_m['regularizer'], dropout_rate=best_params_m['dropout_rate'], 
                         num_layers=best_params_m['num_layers'], num_neurons=best_params_m['num_neurons'])

# Train the model on the full training set
history_m = model_m.fit(X_train_m, Y_train_m, epochs=20,validation_data=(X_test_m, Y_test_m), batch_size=64, verbose=1)
end_time_m=time.time()
total_time_m = end_time_m - start_time_m

print(f"Training time for MLP: {total_time_m:.2f} seconds")


# Evaluate the model on the test set
test_loss, test_acc = model_m.evaluate(X_test_m, Y_test_m, verbose=0)
y_pred_mlp=model_m.predict(X_test_m)
print('Test accuracy:', test_acc)

plt.plot(history_m.history['loss'], label='train')
plt.plot(history_m.history['val_loss'], label='test')
plt.title('Model Loss for MLP')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the accuracy graph with respect to the iteration/epoch
plt.plot(history_m.history['accuracy'], label='train')
plt.plot(history_m.history['val_accuracy'], label='test')
plt.title('Model Accuracy for MLP')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot a random sample of images
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10,10))
for i, ax in enumerate(axes.flat):
    index = np.random.randint(0, x_test.shape[0])
    image = x_test.iloc[index].values.reshape(28, 28) 
    label = mapping[y_test.iloc[index]]
    label2 = mapping[np.argmax(y_pred_mlp[index])]
    ax.imshow(image, cmap='gray')
    ax.set_title(f'True MLP: {label}\nPred MLP: {label2}')
    ax.axis('off')
plt.show()

predicted_labels_m = np.argmax(y_pred_mlp, axis=1)

# Generate the confusion matrix
conf_matrix_m = confusion_matrix(y_test, predicted_labels_m)

classes_m = [str(i) for i in range(47)]

# plot confusion matrix
plt.figure(figsize=(15, 15))
plt.imshow(conf_matrix_m, cmap=plt.cm.Blues)
plt.title('Confusion Matrix for MLP')
plt.colorbar()
tick_marks = np.arange(len(classes_m))
plt.xticks(tick_marks, classes_m, rotation=90)
plt.yticks(tick_marks, classes_m)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

model_m.summary()



# calculate precision, recall, and F1 score
precision_m = precision_score(y_test, predicted_labels_m, average='macro')
recall_m = recall_score(y_test, predicted_labels_m, average='macro')
f1_m = f1_score(y_test, predicted_labels_m, average='macro')
print(precision_m,'Precision for MLP')
print(recall_m,'recall for MLP')
print(f1_m,'f1 for MLP')


Y_train = np_utils.to_categorical(y_train, 47) 
Y_test = np_utils.to_categorical(y_test, 47)

# resize and normalize
X_train = np.reshape(x_train.values, [-1, 28, 28, 1])
X_train = X_train.astype('float32') / 255
X_test = np.reshape(x_test.values, [-1, 28, 28, 1])
X_test = X_test.astype('float32') / 255


# define the model
def create_model(learning_rate, activation, optimizer, use_batchnorm, regularizer, dropout_rate, num_layers, num_filters,kernal_size):
    model = Sequential()
    model.add(Conv2D(num_filters,kernel_size=kernal_size, padding='same', input_shape=(28,28,1)))
    if use_batchnorm:
        model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(dropout_rate))
    for i in range(num_layers-1):
        model.add(Conv2D(num_filters, kernel_size=kernal_size, padding='same'))
        if use_batchnorm:
            model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(dropout_rate))
    model.add(Flatten())
    if regularizer == 'l1':
        model.add(Dense(47, activation='softmax', kernel_regularizer=l1(0.01)))
    elif regularizer == 'l2':
        model.add(Dense(47, activation='softmax', kernel_regularizer=l2(0.01)))
    else:
        model.add(Dense(47, activation='softmax'))
    if optimizer=='Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer=='RMSprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer=='SGD':
        optimizer=SGD(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# create the KerasClassifier for GridSearchCV
model = KerasClassifier(build_fn=create_model, epochs=20, batch_size=256, verbose=0)

# define the hyperparameters to search
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'activation': ['relu', 'elu', 'sigmoid' ],
    'optimizer': ['Adam', 'RMSprop', 'SGD'],
    'use_batchnorm': [True, False],
    'regularizer':[None,'l1', 'l2'],
    'dropout_rate': [0,0.5],
    'num_layers': [2, 3, 4],
    'num_filters': [16, 32, 64],
    'kernal_size': [(3,3),(5,5)]
}
start_time=time.time()
# create the GridSearchCV object
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,n_jobs=-1)

grid_result = grid.fit(X_train, Y_train)

# Print the best hyperparameters found
print("Best Hyperparameters: ", grid_result.best_params_)

best_params = grid_result.best_params_
model = create_model(learning_rate=best_params['learning_rate'], activation=best_params['activation'], 
                     optimizer=best_params['optimizer'], use_batchnorm=best_params['use_batchnorm'], regularizer=best_params['regularizer'],
                    dropout_rate=best_params['dropout_rate'], num_layers=best_params['num_layers'], num_filters=best_params['num_filters'], kernal_size=best_params['kernal_size'])

# Train the model on the full training set
history = model.fit(X_train, Y_train, epochs=20,validation_data=(X_test, Y_test), batch_size=64, verbose=1)
end_time=time.time()
total_time = end_time - start_time

print(f"Training time for CNN: {total_time:.2f} seconds")


# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
y_pred_cnn=model.predict(X_test)
print('Test accuracy:', test_acc)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('Model Loss for CNN')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the accuracy graph with respect to the iteration/epoch
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.title('Model Accuracy for CNN')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Plot a random sample of images
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10,10))
for i, ax in enumerate(axes.flat):
    index = np.random.randint(0, x_test.shape[0])
    image = x_test.iloc[index].values.reshape(28, 28) 
    label = mapping[y_test.iloc[index]]
    label2 = mapping[np.argmax(y_pred_cnn[index])]
    ax.imshow(image, cmap='gray')
    ax.set_title(f'True CNN: {label}\nPred CNN: {label2}')
    ax.axis('off')
plt.show()


predicted_labels = np.argmax(y_pred_cnn, axis=1)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, predicted_labels_m)

classes = [str(i) for i in range(47)]

# plot confusion matrix
plt.figure(figsize=(15, 15))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix for CNN')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=90)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

model.summary()



# calculate precision, recall, and F1 score
precision = precision_score(y_test, predicted_labels, average='macro')
recall = recall_score(y_test, predicted_labels, average='macro')
f1 = f1_score(y_test, predicted_labels, average='macro')
print(precision,"Precision for CNN")
print(recall,'recall for CNN')
print(f1,'f1 for CNN')
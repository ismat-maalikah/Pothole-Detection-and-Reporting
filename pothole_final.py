import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import keras_tuner
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import BayesianOptimization
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Define the paths to your two datasets
plain_road_dir = 'tryout/normal'
pothole_dir = 'tryout/pothole'
img_size = 150

# Function to load and preprocess data
def load_and_preprocess_data(dataset_dir):
    data = []
    for img_file in os.listdir(dataset_dir):
        try:
            img_array = cv2.imread(os.path.join(dataset_dir, img_file), cv2.IMREAD_COLOR)
            resized_array = cv2.resize(img_array, (img_size, img_size))
            data.append(resized_array)
        except Exception as e:
            pass
    return np.array(data)
#image resize - 1. consistency 2. memory will be less 3. Computational Efficiency
# Load and preprocess the data
plain_road_data = load_and_preprocess_data(plain_road_dir)
pothole_data = load_and_preprocess_data(pothole_dir)

# Create labels - Labels are annotations or tags that indicate the ground truth, defining the correct output associated with each input data point (in this case, images)
plain_road_labels = np.zeros(len(plain_road_data)) # plain road - 0
pothole_labels = np.ones(len(pothole_data)) #pothole - 1

# Combine datasets and labels
X = np.concatenate([plain_road_data, pothole_data]) #plain road and potholes into a single dataset which will be acting as input
y = np.concatenate([plain_road_labels, pothole_labels])#array Y which will correspond to labels and images of X

# Normalize the image data
X = X / 255.0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter tuning function
def build_model(hp):#tunable parameter 
    model = Sequential()
    model.add(Conv2D(filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
                     kernel_size=hp.Choice('conv_1_kernel', values=[3, 5]),
                     activation='relu',
                     input_shape=(img_size, img_size, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(Conv2D(filters=hp.Int(f'conv_{i+2}_filter', min_value=32, max_value=128, step=16),
                         kernel_size=hp.Choice(f'conv_{i+2}_kernel', values=[3, 5]),
                         activation='relu'))
    model.add(Flatten())#convertion of 2d into 1d array because fully connected layers need 1d array 
    model.add(Dense(units=hp.Int('dense_units', min_value=32, max_value=128, step=16),
                    activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))#removes overfitting
    model.add(Dense(1, activation='sigmoid')) # it provides probabilities or scores that can be interpreted as the likelihood or confidence that the input belongs to the positive class.
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])#adam - minimizes the loss by updating the weights iteratively 
                                    #binary_crossentropy - difference between predicted probabilities and the actual binary class values.
    return model

# Initialize Keras Tuner BayesianOptimization tuner
#employed to efficiently search for the best set of hyperparameters that optimize the performance of a machine learning model.
#gives balence b/w exploration and exploitation
tuner = BayesianOptimization(build_model,
                             objective='val_accuracy',
                             max_trials=5,
                             num_initial_points=3,
                             directory='my_dir',
                             project_name='pothole_detection')

# Perform hyperparameter tuning
#epoch -  Defines the number of epochs (iterations over the entire training dataset) for each evaluation of the model 
# with a particular set of hyperparameters.
#  Here, the model is trained for 5 epochs per trial.
tuner.search(X_train, y_train, epochs=5, validation_split=0.2)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Train the best model on the entire training set
#validation set - new data This allows the model to assess how well it generalizes to unseen data.
history=best_model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Evaluate the best model
loss, accuracy = best_model.evaluate(X_test, y_test)
#The test loss measures how well the model performs on unseen data, 
# quantifying the difference between predicted values and actual values. Lower test loss values indicate better performance.
#accuracy: This variable holds the test accuracy value, which is the proportion of correctly classified samples in the test dataset.
# It is one of the most common metrics used for classification models. Higher test accuracy indicates better performance.
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# Plot training and validation loss over epochs
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# Plot training and validation accuracy over epochs
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.show()


y_pred_prob = best_model.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')

x = np.linspace(-5, 5, 100)

# Apply ReLU function to x
y = np.maximum(0, x)

# Plot the ReLU function
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('ReLU Function')
plt.xlabel('Input (x)')
plt.ylabel('Output (ReLU(x))')
plt.grid(True)
plt.show()

# Generate and plot confusion matrix
y_pred = best_model.predict(X_test)
y_pred = (y_pred > 0.5)  # Convert probabilities to binary predictions
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['Plain Road', 'Pothole'])
plt.yticks([0, 1], ['Plain Road', 'Pothole'])
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment='center', verticalalignment='center')

plt.show()

TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]

# Define the labels for the classes
class_labels = ['Plain Road', 'Pothole']

# Create a bar plot for TP, TN, FP, FN
plt.figure(figsize=(8, 6))
plt.bar(np.arange(4), [TP, TN, FP, FN], tick_label=['True Positive', 'True Negative', 'False Positive', 'False Negative'])
plt.xlabel('Predicted vs. Actual')
plt.ylabel('Count')
#plt.title('Confusion Matrix')

# Add text labels with the count values on top of the bars
for i, count in enumerate([TP, TN, FP, FN]):
    plt.text(i, count, str(count), ha='center', va='bottom')

plt.show()

y_pred_prob = best_model.predict(X_test)

# Calculate precision and recall
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

# Plot the precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid()
plt.show()

# Generate and print classification report
class_report = classification_report(y_test, y_pred, target_names=['Plain Road', 'Pothole'])
print("Classification Report:")
print(class_report)
########################################
import random
import matplotlib.pyplot as plt

# Generate a random index within the valid range
indx2 = random.randint(0, len(y_test) - 1)

# Display the image
plt.imshow(X_test[indx2])
plt.title("Test Image")
plt.show()

# Classify the image
Y_pred = np.round(best_model.predict(X_test[indx2].reshape(1, 150, 150, 3)))

if Y_pred[0][0] == 1:
    print("Pothole ")
else:
    print("plain Road")

##############################################
best_model.save('pothole.h5')
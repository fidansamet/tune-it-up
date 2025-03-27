import os
import pickle
import random
import sys
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

categories = ['Classic', 'Jazz']
train_data_path = 'datasets/unique_cyclegan'
RANDOM_SEED = 42
MODEL_NAME = 'genre_classifier.pkl'

try:
    mlp_model = pickle.load(open(MODEL_NAME, 'rb'))
except:
    train_data = []
    test_data = []
    for category in categories:
        midi_folder = os.path.join(train_data_path, category, 'train')
        for midi in os.listdir(midi_folder):
            midi_path = os.path.join(midi_folder, midi)
            label = categories.index(category)
            train_data.append([(np.load(midi_path).flatten().reshape(1, -1))[0].astype(int), label])

        midi_folder = os.path.join(train_data_path, category, 'test')
        for midi in os.listdir(midi_folder):
            midi_path = os.path.join(midi_folder, midi)
            label = categories.index(category)
            test_data.append([(np.load(midi_path).flatten().reshape(1, -1))[0].astype(int), label])

    random.Random(RANDOM_SEED).shuffle(train_data)
    random.Random(RANDOM_SEED).shuffle(test_data)
    X_train, y_train = [], []
    X_test, y_test = [], []
    for features, label in train_data:
        X_train.append(features)
        y_train.append(label)

    for features, label in test_data:
        X_test.append(features)
        y_test.append(label)

    mlp_model = MLPClassifier(random_state=1, max_iter=1000, hidden_layer_sizes=(1000),  learning_rate='adaptive',
                              solver='adam', activation='relu')
    mlp_model.fit(X_train, y_train)
    pickle.dump(mlp_model, open(MODEL_NAME, 'wb'))

    OD_mlp_pred = mlp_model.predict(X_test)
    OD_mlp_rep = classification_report(y_test, OD_mlp_pred)
    OD_mlp_acc = accuracy_score(y_test, OD_mlp_pred)

    print(OD_mlp_rep)
    print(OD_mlp_acc)
    exit(0)

test_data_path = sys.argv[1]
test_data_label = sys.argv[2]
X_test = []
y_test = []
for test_midi in os.listdir(test_data_path):
    if test_data_label.lower() in test_midi:
        path = os.path.join(test_data_path, test_midi)
        X_test.append((np.load(path).flatten().reshape(1, -1))[0].astype(int))
        y_test.append(categories.index(test_data_label))

y_pred = mlp_model.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, y_pred))

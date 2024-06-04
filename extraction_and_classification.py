import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


#########################################

def main():
    lines = get_file_lines()
    (images, labels) = get_file_data(lines)

    hog_features = extract_hog_features(images)
    plot_svm_graph(hog_features, labels)

################## HOG ###################

def apply_hog_svm(images, labels):
    hog_features = extract_hog_features(images)
    labels = np.array(labels)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []
    confusion_matrices = []
    
    for train_index, test_index in kf.split(hog_features):
        X_train, X_test = hog_features[train_index], hog_features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = svm.SVC(kernel='linear', decision_function_shape='ovo')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices.append(cm)

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_confusion_matrix = np.mean(confusion_matrices, axis=0)

    print(f'Acurácia média: {mean_accuracy:.4f}')
    print(f'Desvio padrão da acurácia: {std_accuracy:.4f}')
    print('Matriz de confusão média:')
    print(mean_confusion_matrix)

def apply_hog_keras(images, labels):
    hog_features = extract_hog_features(images)
    labels = np.array(labels)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []
    confusion_matrices = []
    
    for train_index, test_index in kf.split(hog_features):
        X_train, X_test = hog_features[train_index], hog_features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_train_cat = to_categorical(y_train)
        y_test_cat = to_categorical(y_test)

        model = create_model((X_train.shape[1],), y_test_cat.shape[1])
        model.fit(X_train, y_train_cat, epochs=20, batch_size=32, verbose=0)

        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        acc = accuracy_score(y_test, y_pred_classes)
        accuracies.append(acc)

        cm = confusion_matrix(y_test, y_pred_classes)
        confusion_matrices.append(cm)

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_confusion_matrix = np.mean(confusion_matrices, axis=0)

    print(f'Acurácia média: {mean_accuracy:.4f}')
    print(f'Desvio padrão da acurácia: {std_accuracy:.4f}')
    print('Matriz de confusão média:')
    print(mean_confusion_matrix)

def extract_hog_features(images):
    hog_features = []
    for baseImage in images:
        fd, hog_image = hog(
            baseImage,
            orientations=9,
            pixels_per_cell=(7, 7),
            cells_per_block=(2, 2), 
            block_norm='L2-Hys',
            visualize=True
        )
        
        hog_features.append(fd)
    
    return np.array(hog_features)

################## LBP ###################

def apply_lbp_svm(images, labels):
    lbp_features = extract_lbp_features(images, labels)
    labels = np.array(labels)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []
    confusion_matrices = []
    
    for train_index, test_index in kf.split(lbp_features):
        X_train, X_test = lbp_features[train_index], lbp_features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = svm.SVC(kernel='linear', decision_function_shape='ovo')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices.append(cm)

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_confusion_matrix = np.mean(confusion_matrices, axis=0)

    print(f'Acurácia média: {mean_accuracy:.4f}')
    print(f'Desvio padrão da acurácia: {std_accuracy:.4f}')
    print('Matriz de confusão média:')
    print(mean_confusion_matrix)

def apply_lbp_keras(images, labels):
    lbp_features = extract_lbp_features(images, labels)
    labels = np.array(labels)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []
    confusion_matrices = []

    for train_index, test_index in kf.split(lbp_features):
        X_kfold_train, X_kfold_test = lbp_features[train_index], lbp_features[test_index]
        y_kfold_train, y_kfold_test = labels[train_index], labels[test_index]

        X_train, X_val, y_train, y_val = train_test_split(X_kfold_train, y_kfold_train, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_kfold_test = scaler.transform(X_kfold_test) 

        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)
        y_kfold_test_cat = to_categorical(y_kfold_test)

        model = create_model((X_train.shape[1],), y_train.shape[1])
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0, validation_data=(X_val, y_val))

        y_pred = model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_val_classes = np.argmax(y_val, axis=1)

        acc = accuracy_score(y_val_classes, y_pred_classes)
        accuracies.append(acc)

        cm = confusion_matrix(y_val_classes, y_pred_classes)
        confusion_matrices.append(cm)

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        mean_confusion_matrix = np.mean(confusion_matrices, axis=0)

    print(f'Acurácia média: {mean_accuracy:.4f}')
    print(f'Desvio padrão da acurácia: {std_accuracy:.4f}')
    print('Matriz de confusão média:')
    print(mean_confusion_matrix)

def extract_lbp_features(images, labels, P=8, R=1):
    lbp_features = []
    for (image) in images:
        lbp = local_binary_pattern(image, P=P, R=R, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
        lbp_features.append(hist)
        
        # plt.figure(figsize=(10, 5))

        # plt.subplot(1, 2, 1)
        # plt.title('Imagem Original')
        # plt.imshow(image, cmap='gray')

        # plt.subplot(1, 2, 2)
        # plt.title('Imagem LBP')
        # plt.imshow(np.array(lbp), cmap='gray')

        # plt.show()

        # generate_image_historiogram_opencv(lbp, f"Histograma para o numero: {labels[index]}")

    return np.array(lbp_features)

####### Seguidores de fronteiras #########

def apply_boundary_following_svm(images, labels):
    boundary_following_features = extract_boundary_following_features(images)
    X_train, X_test, y_train, y_test = train_test_split(boundary_following_features, labels, test_size=0.2, random_state=42)

    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    print(f'Acurácia: {accuracy_score(y_test, y_pred)}')
    print(f'Relatório de Classificação:\n {classification_report(y_test, y_pred)}')

def extract_boundary_following_features(images):
    features = []
    for image in images:
        edges = cv2.Canny(image, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            moments = cv2.HuMoments(cv2.moments(contour)).flatten()
            feature_vector = np.hstack([perimeter, area, moments])
            features.append(feature_vector)
            break
    return np.array(features)

############# -  Utilities - ############

def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_svm_graph(features, labels):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    reduced_data = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter, label='Classes')
    plt.title('Projeção 2D das Características HOG')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.show()

def generate_image_historiogram_opencv(pixels_matrix, label):
    image = np.array(pixels_matrix, dtype=np.uint8)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    plt.figure(figsize=(10, 6))
    plt.bar(range(256), hist[:, 0], width=1) 
    plt.title(label)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    plt.show()

def get_file_data(lines):
    images = []
    labels = []

    for line in lines:
        items = line.split(' ')
        label = int(items[len(items) - 1])
        image = []
        for i in range(len(items) - 1):
            image.append(int(items[i]))

        images.append(get_file_image(image))
        labels.append(label)

    return (images, labels)

def get_file_lines():
    lines = []
    with open('base/ocr_car_numbers_rotulado.txt', 'r') as file:
        for line in file:
            lines.append(line.strip())
    return lines

def get_file_image(array):
    image = []
    for i in range(0, len(array), 35):
        image.append(array[i: i + 35])

    np_array = np.array(image, dtype=np.uint8)
    np_array = np_array * 255

    return np_array



if __name__ == "__main__":
    main()

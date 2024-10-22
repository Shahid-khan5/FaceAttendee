import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from keras_facenet import FaceNet
import matplotlib.pyplot as plt

# Initialize FaceNet
embedder = FaceNet()

def load_lfw_dataset(data_dir, n_classes=10, n_samples=20):
    X, y = [], []
    for person_name in os.listdir(data_dir)[:n_classes]:
        person_dir = os.path.join(data_dir, person_name)
        for img_name in os.listdir(person_dir)[:n_samples]:
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                X.append(img)
                y.append(person_name)
    return np.array(X), np.array(y)

def preprocess_basic(image):
    image = cv2.resize(image, (160, 160))
    return (image - 127.5) / 128.0

def preprocess_histogram_eq(image):
    image = cv2.resize(image, (160, 160))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    return (eq - 127.5) / 128.0

def preprocess_gamma(image, gamma=1.5):
    image = cv2.resize(image, (160, 160))
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return (cv2.LUT(image, table) - 127.5) / 128.0

def preprocess_clahe(image):
    image = cv2.resize(image, (160, 160))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    return (cl - 127.5) / 128.0

def preprocess_sharpen(image):
    image = cv2.resize(image, (160, 160))
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return (sharpened - 127.5) / 128.0

def extract_embeddings(images, preprocess_func):
    preprocessed = np.array([preprocess_func(img) for img in images])
    embeddings = embedder.embeddings(preprocessed)
    return embeddings

def train_classifier(X_train, y_train):
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    return clf

def evaluate_classifier(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, report

def main():
    # Load dataset
    data_dir = 'path/to/lfw/dataset'  # Replace with actual path
    X, y = load_lfw_dataset(data_dir)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessing methods
    preprocess_methods = {
        'Basic': preprocess_basic,
        'Histogram Equalization': preprocess_histogram_eq,
        'Gamma Correction': preprocess_gamma,
        'CLAHE': preprocess_clahe,
        'Sharpening': preprocess_sharpen
    }
    
    results = {}
    
    for name, preprocess_func in preprocess_methods.items():
        print(f"Processing with {name}...")
        
        # Extract embeddings
        X_train_emb = extract_embeddings(X_train, preprocess_func)
        X_test_emb = extract_embeddings(X_test, preprocess_func)
        
        # Train and evaluate
        clf = train_classifier(X_train_emb, y_train)
        accuracy, report = evaluate_classifier(clf, X_test_emb, y_test)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1-score': report['weighted avg']['f1-score']
        }
        
        print(f"{name} Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {results[name]['precision']:.4f}")
        print(f"  Recall:    {results[name]['recall']:.4f}")
        print(f"  F1-score:  {results[name]['f1-score']:.4f}")
        print()
    
    # Plot results
    metrics = ['accuracy', 'precision', 'recall', 'f1-score']
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Preprocessing Methods Comparison')
    
    for i, metric in enumerate(metrics):
        ax = axs[i // 2, i % 2]
        ax.bar(results.keys(), [results[method][metric] for method in results.keys()])
        ax.set_title(metric.capitalize())
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('preprocessing_comparison.png')
    plt.close()

if __name__ == "__main__":
    main()
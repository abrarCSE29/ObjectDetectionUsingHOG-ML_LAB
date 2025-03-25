import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
import cv2
from skimage.feature import hog
from skimage import color
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Custom transformer for RGB to Grayscale conversion
class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return np.array([color.rgb2gray(img) for img in X])

# Custom transformer for HOG features
class HOGTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        def extract_hog(img):
            features = hog(img, 
                          orientations=self.orientations,
                          pixels_per_cell=self.pixels_per_cell,
                          cells_per_block=self.cells_per_block,
                          transform_sqrt=True)
            return features
        
        return np.array([extract_hog(img) for img in X])

# Function to load images from directory
def load_images(image_dir, target_size=(64, 64)):
    images = []
    labels = []
    valid_extensions = (".jpg", ".jpeg", ".png")  # Define valid image extensions

    for class_name in os.listdir(image_dir):
        class_path = os.path.join(image_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)

                # Skip non-image files
                if not img_name.lower().endswith(valid_extensions):
                    print(f"Skipping non-image file: {img_path}")
                    continue

                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Skipping unreadable image: {img_path}")
                        continue

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, target_size)
                    images.append(img)
                    labels.append(class_name)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    return np.array(images), np.array(labels)

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Load and preprocess dataset
    # Replace with your actual image directory path
    image_dir = "dataset"
    
    try:
        X, y = load_images(image_dir)
        print(f"Loaded {len(X)} images successfully")
        
        # 2. Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Training set size: {len(X_train)}")
        print(f"Testing set size: {len(X_test)}")
        
        # 3. Create transformation pipeline
        rgb2gray = RGB2GrayTransformer()
        hog_transformer = HOGTransformer(
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(3, 3)
        )
        scaler = StandardScaler()
        
        # 4. Apply transformations
        # Convert RGB to Grayscale
        X_train_gray = rgb2gray.transform(X_train)
        X_test_gray = rgb2gray.transform(X_test)
        
        # Apply HOG transformation
        X_train_hog = hog_transformer.transform(X_train_gray)
        X_test_hog = hog_transformer.transform(X_test_gray)
        
        # Scale the features
        X_train_scaled = scaler.fit_transform(X_train_hog)
        X_test_scaled = scaler.transform(X_test_hog)
        
        # 5. Train SGD Classifier
        sgd_clf = SGDClassifier(
            random_state=42,
            max_iter=1000,
            tol=1e-3,
            learning_rate='optimal'
        )
        
        sgd_clf.fit(X_train_scaled, y_train)
        print("Model training completed")
        
        # 6. Perform testing
        y_pred = sgd_clf.predict(X_test_scaled)
        
        # 7. Generate confusion matrix and metrics
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
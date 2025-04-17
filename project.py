import cv2, numpy as np, os, matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# Configuration
DATASET_PATH = r"C:\Users\FQ4021TU\Downloads\archive (2)\UTKFace"
OUTPUT_DIR, MODELS_DIR = "output", "models"
IMG_SIZE, BATCH_SIZE, EPOCHS = (227, 227), 32, 20
os.makedirs(OUTPUT_DIR, exist_ok=True), os.makedirs(MODELS_DIR, exist_ok=True)

def load_utkface_dataset(dataset_path):
    images, ages, genders = [], [], []
    image_paths = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path) 
                if img.endswith(('.jpg', '.png'))]
    print(f"Found {len(image_paths)} image files. Loading...")
    
    for image_path in tqdm(image_paths):
        try:
            filename = os.path.basename(image_path)
            parts = filename.split('_')
            if len(parts) >= 2:
                try:
                    age, gender = int(parts[0]), int(parts[1])
                    img = cv2.imread(image_path)
                    if img is None: continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, IMG_SIZE)
                    images.append(img), ages.append(age), genders.append(gender)
                except ValueError: pass
        except Exception as e: print(f"Error processing {image_path}: {e}")
    
    return np.array(images), np.array(ages), np.array(genders)

def convert_age_to_groups(ages):
    age_groups = np.zeros_like(ages)
    for i, age in enumerate(ages):
        if age <= 2: age_groups[i] = 0
        elif age <= 6: age_groups[i] = 1
        elif age <= 12: age_groups[i] = 2
        elif age <= 20: age_groups[i] = 3
        elif age <= 32: age_groups[i] = 4
        elif age <= 43: age_groups[i] = 5
        elif age <= 53: age_groups[i] = 6
        else: age_groups[i] = 7
    return to_categorical(age_groups, num_classes=8)

def create_model(output_classes, name="model"):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(output_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

def train_models(X_train, y_age_train, y_gender_train, X_val, y_age_val, y_gender_val):
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    
    age_model, gender_model = create_model(8, "age"), create_model(2, "gender")
    
    age_history = age_model.fit(datagen.flow(X_train, y_age_train, batch_size=BATCH_SIZE),
                            validation_data=(X_val, y_age_val), epochs=EPOCHS, verbose=1)
    
    gender_history = gender_model.fit(datagen.flow(X_train, y_gender_train, batch_size=BATCH_SIZE),
                                validation_data=(X_val, y_gender_val), epochs=EPOCHS, verbose=1)
    
    age_model.save(os.path.join(MODELS_DIR, 'age_model.h5'))
    gender_model.save(os.path.join(MODELS_DIR, 'gender_model.h5'))
    
    return age_model, gender_model, age_history, gender_history

def plot_history(history, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history.history['accuracy']), ax1.plot(history.history['val_accuracy'])
    ax1.set_title(f'{model_name} Accuracy'), ax1.set_ylabel('Accuracy'), ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    ax2.plot(history.history['loss']), ax2.plot(history.history['val_loss'])
    ax2.set_title(f'{model_name} Loss'), ax2.set_ylabel('Loss'), ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{model_name.lower().replace(" ", "_")}_history.png'))
    plt.show()

def visualize_results(X_test, y_age_test, y_gender_test, age_preds, gender_preds, num_samples=5):
    age_groups = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
    gender_labels = ['Male', 'Female']
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(indices):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(X_test[idx]), plt.axis('off')
        true_age_idx, true_gender_idx = np.argmax(y_age_test[idx]), np.argmax(y_gender_test[idx])
        pred_age_idx, pred_gender_idx = np.argmax(age_preds[idx]), np.argmax(gender_preds[idx])
        age_conf, gender_conf = age_preds[idx][pred_age_idx] * 100, gender_preds[idx][pred_gender_idx] * 100
        
        title = f"True: {gender_labels[true_gender_idx]}, {age_groups[true_age_idx]}\n"
        title += f"Pred: {gender_labels[pred_gender_idx]} ({gender_conf:.1f}%), "
        title += f"{age_groups[pred_age_idx]} ({age_conf:.1f}%)"
        plt.title(title)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_results.png'))
    plt.show()
    
    # Confusion matrices 
    y_age_true, y_age_pred = np.argmax(y_age_test, axis=1), np.argmax(age_preds, axis=1)
    y_gender_true, y_gender_pred = np.argmax(y_gender_test, axis=1), np.argmax(gender_preds, axis=1)
    
    age_cm = confusion_matrix(y_age_true, y_age_pred)
    gender_cm = confusion_matrix(y_gender_true, y_gender_pred)
    
    age_cm_norm = age_cm.astype('float') / age_cm.sum(axis=1)[:, np.newaxis]
    gender_cm_norm = gender_cm.astype('float') / gender_cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrices using matplotlib 
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(age_cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Age Model Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(age_groups))
    plt.xticks(tick_marks, age_groups, rotation=45)
    plt.yticks(tick_marks, age_groups)
    plt.xlabel('Predicted Age Group'), plt.ylabel('True Age Group')
    
    # Add text annotations
    thresh = age_cm_norm.max() / 2.
    for i in range(age_cm_norm.shape[0]):
        for j in range(age_cm_norm.shape[1]):
            plt.text(j, i, f'{age_cm_norm[i, j]:.2f}',
                    horizontalalignment="center",
                    color="white" if age_cm_norm[i, j] > thresh else "black")
    
    plt.subplot(1, 2, 2)
    plt.imshow(gender_cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Gender Model Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(gender_labels))
    plt.xticks(tick_marks, gender_labels)
    plt.yticks(tick_marks, gender_labels)
    plt.xlabel('Predicted Gender'), plt.ylabel('True Gender')
    
    # Add text annotations
    thresh = gender_cm_norm.max() / 2.
    for i in range(gender_cm_norm.shape[0]):
        for j in range(gender_cm_norm.shape[1]):
            plt.text(j, i, f'{gender_cm_norm[i, j]:.2f}',
                    horizontalalignment="center",
                    color="white" if gender_cm_norm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices.png'))
    plt.show()

def detect_and_predict_realtime():
    try:
        age_model = load_model(os.path.join(MODELS_DIR, 'age_model.h5'))
        gender_model = load_model(os.path.join(MODELS_DIR, 'gender_model.h5'))
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty(): return
    except Exception as e: 
        print(f"Error loading models: {e}")
        return
    
    age_groups = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
    gender_labels = ['Male', 'Female']
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_roi = frame[y:y+h, x:x+w]
            
            try:
                face_roi_resized = cv2.resize(face_roi, IMG_SIZE)
                face_roi_rgb = cv2.cvtColor(face_roi_resized, cv2.COLOR_BGR2RGB)
                face_roi_normalized = face_roi_rgb / 255.0
                face_roi_batch = np.expand_dims(face_roi_normalized, axis=0)
                
                gender_pred = gender_model.predict(face_roi_batch, verbose=0)[0]
                age_pred = age_model.predict(face_roi_batch, verbose=0)[0]
                
                gender_idx, age_idx = np.argmax(gender_pred), np.argmax(age_pred)
                gender_conf, age_conf = gender_pred[gender_idx] * 100, age_pred[age_idx] * 100
                
                gender_text = f"{gender_labels[gender_idx]} ({gender_conf:.1f}%)"
                age_text = f"{age_groups[age_idx]} ({age_conf:.1f}%)"
                
                cv2.putText(frame, gender_text, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, age_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception as e: print(f"Error processing face: {e}")
        
        cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Gender and Age Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    try:
        print("Loading dataset...")
        X, y_age, y_gender = load_utkface_dataset(DATASET_PATH)
        
        if len(X) < 100:
            print("Warning: Very small dataset.")
            if input("Continue? (y/n): ").lower() != 'y': return
        
        print("Processing data...")
        y_age_categorical = convert_age_to_groups(y_age)
        y_gender_categorical = to_categorical(y_gender, num_classes=2)
        X = X.astype(np.float32)
        X_normalized = X / 255.0

        X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test = train_test_split(
            X_normalized, y_age_categorical, y_gender_categorical, test_size=0.2, random_state=42)
        
        X_val, X_test, y_age_val, y_age_test, y_gender_val, y_gender_test = train_test_split(
            X_test, y_age_test, y_gender_test, test_size=0.5, random_state=42)
        
        age_model_path = os.path.join(MODELS_DIR, 'age_model.h5')
        gender_model_path = os.path.join(MODELS_DIR, 'gender_model.h5')
        
        if os.path.exists(age_model_path) and os.path.exists(gender_model_path):
            if input("Use existing models? (y/n): ").lower() == 'y':
                age_model = load_model(age_model_path)
                gender_model = load_model(gender_model_path)
            else:
                age_model, gender_model, age_history, gender_history = train_models(
                    X_train, y_age_train, y_gender_train, X_val, y_age_val, y_gender_val)
                plot_history(age_history, "Age Model")
                plot_history(gender_history, "Gender Model")
        else:
            age_model, gender_model, age_history, gender_history = train_models(
                X_train, y_age_train, y_gender_train, X_val, y_age_val, y_gender_val)
            plot_history(age_history, "Age Model")
            plot_history(gender_history, "Gender Model")
        
        print("Evaluating models...")
        age_preds = age_model.predict(X_test)
        gender_preds = gender_model.predict(X_test)
        
        age_loss, age_acc = age_model.evaluate(X_test, y_age_test)
        gender_loss, gender_acc = gender_model.evaluate(X_test, y_gender_test)
        print(f"Age Model - Accuracy: {age_acc:.4f}")
        print(f"Gender Model - Accuracy: {gender_acc:.4f}")
        
        visualize_results(X_test, y_age_test, y_gender_test, age_preds, gender_preds)
        
        if input("Run real-time detection? (y/n): ").lower() == 'y':
            detect_and_predict_realtime()
            
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
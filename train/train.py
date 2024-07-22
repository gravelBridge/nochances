import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

def load_data(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            
            if "skip" in entry and entry["skip"] == True:
                continue
            
            features = []
            for category, values in entry.items():
                for key, value in values.items():
                    if isinstance(value, str) and value.isdigit():
                        value = int(value)
                    features.append(value)
            
            accept_rate = features.pop(17)  # accept_rate is at index 17
            
            data.append(features)
            labels.append(accept_rate)
    
    return np.array(data), np.array(labels)

def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(input_shape,), 
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu', 
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu', 
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu', 
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    X, y = load_data('categorization/categorized.json')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    num_classes = 4  # 0 to 3 for accept_rate categories
    model = create_model(X_train.shape[1], num_classes)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True)
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras', monitor='val_loss', save_best_only=True)
    
    history = model.fit(
        X_train_scaled, y_train, 
        epochs=300, 
        batch_size=64, 
        validation_split=0.2, 
        verbose=1, 
        callbacks=[early_stopping, model_checkpoint]
    )
    
    model = tf.keras.models.load_model('best_model.keras')
    
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    accept_rate_categories = [
        "Highly Selective (<5%)",
        "Very Selective (5-15%)",
        "Selective (15-40%)",
        "Somewhat Selective (40-60%)",
        "Minimally Selective (>60%) or Open Admission"
    ]
    
    # Group test samples by their actual accept_rate category
    samples_by_category = defaultdict(list)
    for i in range(len(y_test)):
        category = int(y_test[i])
        samples_by_category[category].append(i)
    
    # Print 10 samples from each category (or all if less than 10)
    for category in range(num_classes):
        print(f"\n--- Category: {accept_rate_categories[category]} ---")
        samples = samples_by_category[category]
        np.random.shuffle(samples)
        for i, sample_index in enumerate(samples[:10]):
            sample = X_test_scaled[sample_index].reshape(1, -1)
            prediction = model.predict(sample, verbose=0)
            predicted_class = np.argmax(prediction)
            
            print(f"\nSample {i+1}:")
            print(f"Input: {X_test[sample_index]}")
            print(f"Actual accept_rate: {accept_rate_categories[category]}")
            print(f"Predicted accept_rate: {accept_rate_categories[predicted_class]}")
            print(f"Prediction probabilities:")
            for j, prob in enumerate(prediction[0]):
                print(f"  {accept_rate_categories[j]}: {prob:.4f}")
        
        if len(samples) < 10:
            print(f"\nNote: Only {len(samples)} samples available for this category.")

if __name__ == "__main__":
    main()
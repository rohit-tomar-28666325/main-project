import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight


from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# Parameters
img_width, img_height = 224, 224
batch_size = 8
epochs = 10
num_classes = 5
validation_split = 0.2  # 20% of the data will be used for validation
test_split = 0.1
data_dir = 'C:/Users/rohit/Desktop/workspace/MS/Main Project/Project Code/main-project/datasets/dataset1'  # Add your data directory path

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=validation_split + test_split
)

# Augmentation parameters for the specific class
specific_class_augmentation = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=validation_split + test_split
)

# Generators for training and validation
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=validation_split / (validation_split + test_split)
)

validation_generator = validation_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Data preparation for testing
test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=test_split / (validation_split + test_split)
)

test_generator = test_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Function to plot histogram of classes
def plot_class_distribution(generator, title):
    classes = generator.classes
    class_counts = np.bincount(classes)
    class_labels = list(generator.class_indices.keys())
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(class_counts)), class_counts, tick_label=class_labels)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Number of samples')
    plt.show()

# Plotting class distribution for training, validation, and test datasets
plot_class_distribution(train_generator, 'Class Distribution in Training Set')
plot_class_distribution(validation_generator, 'Class Distribution in Validation Set')
plot_class_distribution(test_generator, 'Class Distribution in Test Set')

# Oversampling the specified minority class with augmentation
specific_class = '1'  # Example specific class index as a string
oversample_ratio = 3  # How many times to oversample the specific class

# Create a separate generator for the specific class with augmentation
specific_class_generator = specific_class_augmentation.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    classes=[specific_class],
    subset='training',
    shuffle=True
)

# Function to combine generators
def combined_generator(base_generator, specific_class_generator, oversample_ratio):
    while True:
        x_batch, y_batch = base_generator.next()
        for _ in range(oversample_ratio):
            x_minority, y_minority = specific_class_generator.next()
            
            # Ensure y_minority matches the shape of y_batch
            if y_minority.shape[1] == 1:
                y_minority = np.eye(num_classes)[y_minority[:, 0].astype(int)]  # One-hot encode if necessary
                
            x_batch = np.concatenate((x_batch, x_minority), axis=0)
            y_batch = np.concatenate((y_batch, y_minority), axis=0)
        
        yield x_batch, y_batch

# Combined generator for training
combined_train_generator = combined_generator(train_generator, specific_class_generator, oversample_ratio)

# Visualize the combined generator output
def visualize_combined_generator(generator, num_batches=1):
    for i, (x_batch, y_batch) in enumerate(generator):
        print(f'Batch {i}: x_batch shape = {x_batch.shape}, y_batch shape = {y_batch.shape}')
        unique, counts = np.unique(np.argmax(y_batch, axis=1), return_counts=True)
        print(f'Class distribution in batch {i}: {dict(zip(unique, counts))}')
        
        if i == num_batches - 1:  # Print only the specified number of batches for verification
            break

# Check a few batches from the combined generator
visualize_combined_generator(combined_train_generator, num_batches=3)

# Continue with model definition, compilation, and training
# Load the VGG16 model, excluding the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with combined generator
model.fit(
    combined_train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs
)
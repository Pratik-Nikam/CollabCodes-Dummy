import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # Required to run the BERT preprocessing model

# URLs for the BERT preprocessing and encoder from TF Hub
preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"

# Create KerasLayers for preprocessing and BERT encoder
bert_preprocess = hub.KerasLayer(preprocess_url, name="bert_preprocess")
bert_encoder = hub.KerasLayer(encoder_url, trainable=True, name="bert_encoder")

def build_multitask_model(num_flavors, num_dispositions):
    """
    Build a multi-task model with a shared BERT encoder and two task-specific heads.
      - num_flavors: Number of flavor classes (e.g., 60)
      - num_dispositions: Number of disposition classes (e.g., 4)
    """
    # Input: raw text string
    input_text = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
    
    # Preprocess the text (tokenization, adding [CLS] and [SEP], etc.)
    preprocessed_text = bert_preprocess(input_text)
    
    # Pass through the BERT encoder to obtain representations
    bert_outputs = bert_encoder(preprocessed_text)
    # Use the pooled output (representation of the [CLS] token)
    pooled_output = bert_outputs["pooled_output"]  # shape: (batch_size, 768)
    
    # Task 1: Flavor classification head (60+ classes)
    flavor_output = tf.keras.layers.Dense(num_flavors, activation="softmax", name="flavor")(pooled_output)
    
    # Task 2: Disposition classification head (4 classes)
    disposition_output = tf.keras.layers.Dense(num_dispositions, activation="softmax", name="disposition")(pooled_output)
    
    # Define the multi-output model
    model = tf.keras.Model(inputs=input_text, outputs=[flavor_output, disposition_output])
    return model

# Define number of classes for each task
NUM_FLAVORS = 60
NUM_DISPOSITIONS = 4

# Build the model
model = build_multitask_model(NUM_FLAVORS, NUM_DISPOSITIONS)
model.summary()

# Compile the model with appropriate losses and metrics.
# We assume the labels for each task are provided as integer indices.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
    loss={
        "flavor": tf.keras.losses.SparseCategoricalCrossentropy(),
        "disposition": tf.keras.losses.SparseCategoricalCrossentropy()
    },
    metrics={
        "flavor": tf.keras.metrics.SparseCategoricalAccuracy(name="flavor_acc"),
        "disposition": tf.keras.metrics.SparseCategoricalAccuracy(name="disposition_acc")
    }
)

# ---------------------------------------------------------
# Create a synthetic dataset for demonstration purposes.
# In practice, replace this with your actual data loading and preprocessing.

import numpy as np

# Example texts (replace these with your descriptions)
texts = [
    "The product has a unique flavor and outstanding quality.",
    "I am disappointed with the performance and customer service.",
    "This issue involves a complex flavor profile and a neutral disposition.",
    "The update delivered a vibrant flavor and a positive outlook."
]
# For demo purposes, duplicate the examples to form a dataset
texts = texts * 50  # Create 200 samples

# Synthetic labels:
# For flavors: random integers in [0, NUM_FLAVORS-1]
# For dispositions: random integers in [0, NUM_DISPOSITIONS-1]
flavor_labels = np.random.randint(0, NUM_FLAVORS, size=(len(texts),))
disposition_labels = np.random.randint(0, NUM_DISPOSITIONS, size=(len(texts),))

# Create a tf.data.Dataset from the data
batch_size = 8
dataset = tf.data.Dataset.from_tensor_slices((texts, (flavor_labels, disposition_labels)))
dataset = dataset.shuffle(buffer_size=100).batch(batch_size)

# ---------------------------------------------------------
# Train the model on the synthetic dataset

model.fit(dataset, epochs=2)

######################################################
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # Required for the BERT preprocessing model

# URLs for the BERT preprocessing and encoder from TF Hub
preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"

# Create KerasLayer instances for preprocessing and BERT encoder
bert_preprocess = hub.KerasLayer(preprocess_url, name="bert_preprocess")
bert_encoder = hub.KerasLayer(encoder_url, trainable=True, name="bert_encoder")

def build_multitask_model(num_flavors, num_dispositions, 
                          use_intermediate=True, intermediate_dim=256, dropout_rate=0.3):
    """
    Build a multi-task model using a shared BERT encoder and two task-specific heads.
    
    Parameters:
      - num_flavors: Number of flavor classes (e.g., 60)
      - num_dispositions: Number of disposition classes (e.g., 4)
      - use_intermediate: Whether to use an extra Dense layer before classification heads.
      - intermediate_dim: Dimensionality of the intermediate dense layer.
      - dropout_rate: Dropout rate for regularization.
    """
    input_text = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
    # Preprocessing: tokenizes and formats the text for BERT.
    preprocessed_text = bert_preprocess(input_text)
    # Pass through BERT encoder.
    bert_outputs = bert_encoder(preprocessed_text)
    pooled_output = bert_outputs["pooled_output"]  # shape: (batch_size, 768)

    # Optional intermediate layer with dropout.
    if use_intermediate:
        x = tf.keras.layers.Dense(intermediate_dim, activation="relu", name="intermediate_dense")(pooled_output)
        x = tf.keras.layers.Dropout(dropout_rate, name="intermediate_dropout")(x)
    else:
        x = pooled_output

    # Task 1: Flavor classification head
    flavor_output = tf.keras.layers.Dense(num_flavors, activation="softmax", name="flavor")(x)
    # Task 2: Disposition classification head
    disposition_output = tf.keras.layers.Dense(num_dispositions, activation="softmax", name="disposition")(x)
    
    model = tf.keras.Model(inputs=input_text, outputs=[flavor_output, disposition_output])
    return model

# Define the number of classes for each task
NUM_FLAVORS = 60
NUM_DISPOSITIONS = 4

# Build the model with an intermediate layer and dropout.
model = build_multitask_model(NUM_FLAVORS, NUM_DISPOSITIONS, use_intermediate=True, intermediate_dim=256, dropout_rate=0.3)
model.summary()

# ------------------------------------------------------------------
# Fine tuning strategies:
# 1. Learning Rate Scheduling: Use exponential decay.
initial_learning_rate = 3e-5
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile the model.
model.compile(
    optimizer=optimizer,
    loss={
        "flavor": tf.keras.losses.SparseCategoricalCrossentropy(),
        "disposition": tf.keras.losses.SparseCategoricalCrossentropy()
    },
    metrics={
        "flavor": tf.keras.metrics.SparseCategoricalAccuracy(name="flavor_acc"),
        "disposition": tf.keras.metrics.SparseCategoricalAccuracy(name="disposition_acc")
    }
)

# ------------------------------------------------------------------
# Create synthetic dataset for demonstration purposes.
import numpy as np

texts = [
    "The product has a unique flavor and outstanding quality.",
    "I am disappointed with the performance and customer service.",
    "This issue involves a complex flavor profile and a neutral disposition.",
    "The update delivered a vibrant flavor and a positive outlook."
]
texts = texts * 50  # 200 samples

# Generate synthetic labels.
flavor_labels = np.random.randint(0, NUM_FLAVORS, size=(len(texts),))
disposition_labels = np.random.randint(0, NUM_DISPOSITIONS, size=(len(texts),))

batch_size = 8
dataset = tf.data.Dataset.from_tensor_slices((texts, (flavor_labels, disposition_labels)))
dataset = dataset.shuffle(buffer_size=100).batch(batch_size)

# ------------------------------------------------------------------
# Callbacks for fine-tuning improvements.
callbacks = [
    # EarlyStopping: stops training when a monitored metric has stopped improving.
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
    # ReduceLROnPlateau: reduces learning rate when a metric has stopped improving.
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1)
]

# Optionally, use a validation split if you have a separate validation dataset.
# For demonstration, we will use 10% of our synthetic data as validation.
val_dataset = dataset.take(5)
train_dataset = dataset.skip(5)

# ------------------------------------------------------------------
# Strategy: Freeze BERT encoder for initial epochs, then unfreeze.
# Freeze encoder layers
for layer in model.layers:
    if layer.name.startswith("bert_encoder"):
        layer.trainable = False
print("BERT encoder frozen for initial training.")

# Train with frozen encoder for a few epochs.
initial_epochs = 2
history_frozen = model.fit(train_dataset, validation_data=val_dataset, epochs=initial_epochs, callbacks=callbacks)

# Unfreeze the encoder for further fine-tuning.
for layer in model.layers:
    if layer.name.startswith("bert_encoder"):
        layer.trainable = True
print("BERT encoder unfrozen. Continue fine-tuning.")

# Fine-tune for additional epochs.
fine_tune_epochs = 2
total_epochs = initial_epochs + fine_tune_epochs
history_finetune = model.fit(train_dataset, validation_data=val_dataset, epochs=total_epochs, initial_epoch=initial_epochs, callbacks=callbacks)

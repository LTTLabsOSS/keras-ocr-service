import tensorflow as tf

print(f"Tensorflow version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")
print(tf.config.list_physical_devices('GPU'))
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
#!/usr/bin/env python
# coding: utf-8

# In[12]:


import tensorflow as tf
from tensorflow import keras

# Working with directory library
from os import path, listdir
from os.path import isdir

# Image visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[13]:


def load_image_dataset(file_path):
    all_image_dirs = [path.join(file_path, f) for f in listdir(file_path) if not isdir(path.join(file_path, f))]
    all_image_labels = []
    for f in all_image_dirs:
        if f.split('.')[0][-3:] == 'cat':
            all_image_labels.append(0)
        else:
            all_image_labels.append(1)
    return all_image_dirs, all_image_labels


# In[14]:


all_image_dirs, all_image_labels = load_image_dataset('train')


# In[15]:


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range
    image = 2*image-1  # normalize to [-1,1] range
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label


# In[16]:


num_train_image = int(len(all_image_labels)*0.8//1)
train_image_dirs, train_label = all_image_dirs[:num_train_image], all_image_labels[:num_train_image]
test_image_dirs, test_label = all_image_dirs[num_train_image:], all_image_labels[num_train_image:]


# In[17]:


train_path_label = tf.data.Dataset.from_tensor_slices((train_image_dirs, train_label))
test_path_label = tf.data.Dataset.from_tensor_slices((test_image_dirs, test_label))


# In[18]:


train_image_label_ds = train_path_label.map(load_and_preprocess_from_path_label)
test_image_label_ds = test_path_label.map(load_and_preprocess_from_path_label).batch(1)


# In[19]:


BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_image_label_ds.shuffle(buffer_size = len(all_image_labels))
train_ds = train_ds.repeat()
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)


# In[20]:


mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable=False


# In[21]:


# Build the CNN-model

cnn_model = keras.models.Sequential([
    mobile_net,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2, activation="softmax")
])


# In[22]:


cnn_model.summary()


# In[23]:


# Compile CNN-model

cnn_model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])


# In[24]:


steps_per_epoch=tf.math.ceil(len(all_image_dirs)/BATCH_SIZE).numpy()
cnn_model.fit(train_ds, epochs=2, steps_per_epoch=steps_per_epoch, validation_data=test_image_label_ds)


# In[33]:


cnn_model.save('dog_cat.h5')


# In[ ]:





# In[35]:


from keras.preprocessing import image
import numpy as np
test_img = 'test1/2.jpg'
img = image.load_img(test_img, target_size = (192,192))
img = image.img_to_array(img, dtype=np.uint8)
img = np.array(img)/255.0
prediction = cnn_model.predict(img[np.newaxis, ...])

#print("Predicted shape",p.shape)
print("Probability:",np.max(prediction[0], axis=-1))
predicted_class = np.argmax(prediction[0], axis=-1)
print("Classified:",predicted_class,'\n')

plt.axis('off')
plt.imshow(img.squeeze())
plt.title("Loaded Image")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





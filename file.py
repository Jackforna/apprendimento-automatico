import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import cv2

ds_train, ds_info = tfds.load(
    'eurosat/rgb',
    shuffle_files=False,
    #as_supervised=True,  # Returns a tuple (img, label) instead of a dictionary {'image': img, 'label': label}
    with_info=True
)

ds_train = ds_train['train']
ds_train = ds_train.shuffle(1000, seed = 42)
train_dataset = ds_train.take(20000)
test_dataset = ds_train.skip(20000)

def generator(dataset,nolines=9):
    while True:  # Start an infinite loop
        for batch in dataset:
            images = batch["image"]
            images_np = images.numpy()

            masks = np.zeros((batch_size, 64, 64))
            for i in range(batch_size):
                for j in range(nolines):
                    start_point = (np.random.randint(0, 64 - 1), 0)
                    end_point = (np.random.randint(0, 64 - 1), 63)
                    thickness = np.random.randint(2, 3)
                    masks[i] = cv2.line(masks[i], start_point, end_point, (1), thickness)

            images_np = images_np / 255.0
            masks = np.stack(((masks),) * 3, axis=-1)

            yield (images_np * masks, images_np)

# Batch the datasets
batch_size = 8
train_dataset_batched = train_dataset.batch(batch_size)
test_dataset_batched = test_dataset.batch(batch_size)

# Create generators for the batched datasets
train_generator = generator(train_dataset_batched)
test_generator = generator(test_dataset_batched)

#Modello con autoencoder mean_mse = 0.0035 anche con aggiunta di skip connections
'''
def create_model():
    input_img = Input(shape=(64, 64, 3))  # Usa la dimensione delle immagini EuroSAT
    
    #Versione più veloce nell'esecuzione ma un po' meno precisa
    # Encoding
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoding
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    
    # Encoding
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    skip1 = x
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    skip2 = x

    # Bottleneck
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    
    # Decoding
    
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    skip2 = ZeroPadding2D(((16, 0), (16, 0)))(skip2)
    x = Add()([x, skip2])
    
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    skip1 = ZeroPadding2D(((32, 0), (32, 0)))(skip1)
    x = Add()([x, skip1])
    
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    # Compilare il modello
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    return autoencoder

autoencoder = create_model()

# Addestra il modello usando il generatore
# Definisci il numero di epoche e i passi per epoca come preferisci
autoencoder.fit(
    generator(train_dataset_batched, batch_size), 
    steps_per_epoch=20000 // batch_size,  # o len(train_dataset) // batch_size se è disponibile
    epochs=10,
    validation_data=generator(test_dataset_batched, batch_size),
    validation_steps=10000 // batch_size  # o len(test_dataset) // batch_size se è disponibile)  # Adatta questi numeri in base alle tue esigenze
)
'''

def unet_model():
    input_img = Input(shape=(64, 64, 3))
    
    # Encoding path
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    
    # Decoding path
    u5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    outputs = Conv2D(3, (1, 1), activation='sigmoid')(c7)

    model = Model(input_img, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    
    return model

# Creazione del modello U-Net
unet = unet_model()
unet.summary()

unet.fit(
    generator(train_dataset_batched, batch_size), 
    steps_per_epoch=20000 // batch_size,  # o len(train_dataset) // batch_size se è disponibile
    epochs=10,
    validation_data=generator(test_dataset_batched, batch_size),
    validation_steps=10000 // batch_size  # o len(test_dataset) // batch_size se è disponibile)  # Adatta questi numeri in base alle tue esigenze
)

mse_scores = []

for _ in range(10):  # Ripeti il processo 10 volte
    mse_values = []
    for i, (masked_imgs, original_imgs) in enumerate(test_generator):
        if i >= 1250:  # 1250 batches of size 8 give 10000 images
            break
        reconstructed_imgs = unet.predict(masked_imgs)
        mse = np.mean(np.power(original_imgs - reconstructed_imgs, 2))
        mse_values.append(mse)

    mse_scores.append(np.mean(mse_values))  # Media degli MSE per questa iterazione
# Calcolare il valore medio e la deviazione standard dell'MSE
mean_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)

print("Mean MSE:", mean_mse)
print("Standard Deviation MSE:", std_mse)

masked_imgs, original_imgs = next(test_generator)
# Prevedi (rigenera) le immagini con il modello
regenerated_imgs = unet.predict(masked_imgs)

# Numero di immagini da visualizzare
num_imgs_to_show = 3

plt.figure(figsize=(18, 6))
for i in range(num_imgs_to_show):
    # Immagine originale
    ax = plt.subplot(3, num_imgs_to_show, i + 1)
    plt.imshow(original_imgs[i])
    ax.set_title("Originale")
    plt.axis('off')

    # Immagine mascherata
    ax = plt.subplot(3, num_imgs_to_show, num_imgs_to_show + i + 1)
    plt.imshow(masked_imgs[i])
    ax.set_title("Mascherata")
    plt.axis('off')

    # Immagine rigenerata
    ax = plt.subplot(3, num_imgs_to_show, 2 * num_imgs_to_show + i + 1)
    plt.imshow(regenerated_imgs[i])
    ax.set_title("Rigenerata")
    plt.axis('off')

plt.tight_layout()
plt.show()

'''
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to
==================================================================================================
 input_1 (InputLayer)        [(None, 64, 64, 3)]          0         []

 conv2d (Conv2D)             (None, 64, 64, 64)           1792      ['input_1[0][0]']

 conv2d_1 (Conv2D)           (None, 64, 64, 64)           36928     ['conv2d[0][0]']

 max_pooling2d (MaxPooling2  (None, 32, 32, 64)           0         ['conv2d_1[0][0]']
 D)

 conv2d_2 (Conv2D)           (None, 32, 32, 128)          73856     ['max_pooling2d[0][0]']

 conv2d_3 (Conv2D)           (None, 32, 32, 128)          147584    ['conv2d_2[0][0]']

 max_pooling2d_1 (MaxPoolin  (None, 16, 16, 128)          0         ['conv2d_3[0][0]']
 g2D)

 conv2d_4 (Conv2D)           (None, 16, 16, 256)          295168    ['max_pooling2d_1[0][0]']

 conv2d_5 (Conv2D)           (None, 16, 16, 256)          590080    ['conv2d_4[0][0]']

 max_pooling2d_2 (MaxPoolin  (None, 8, 8, 256)            0         ['conv2d_5[0][0]']
 g2D)

 conv2d_6 (Conv2D)           (None, 8, 8, 512)            1180160   ['max_pooling2d_2[0][0]']

 conv2d_7 (Conv2D)           (None, 8, 8, 512)            2359808   ['conv2d_6[0][0]']

 conv2d_transpose (Conv2DTr  (None, 16, 16, 256)          524544    ['conv2d_7[0][0]']
 anspose)

 concatenate (Concatenate)   (None, 16, 16, 512)          0         ['conv2d_transpose[0][0]',
                                                                     'conv2d_5[0][0]']

 conv2d_8 (Conv2D)           (None, 16, 16, 256)          1179904   ['concatenate[0][0]']

 conv2d_9 (Conv2D)           (None, 16, 16, 256)          590080    ['conv2d_8[0][0]']

 conv2d_transpose_1 (Conv2D  (None, 32, 32, 128)          131200    ['conv2d_9[0][0]']
 Transpose)

 concatenate_1 (Concatenate  (None, 32, 32, 256)          0         ['conv2d_transpose_1[0][0]',
 )                                                                   'conv2d_3[0][0]']

 conv2d_10 (Conv2D)          (None, 32, 32, 128)          295040    ['concatenate_1[0][0]']

 conv2d_11 (Conv2D)          (None, 32, 32, 128)          147584    ['conv2d_10[0][0]']

 conv2d_transpose_2 (Conv2D  (None, 64, 64, 64)           32832     ['conv2d_11[0][0]']
 Transpose)

 concatenate_2 (Concatenate  (None, 64, 64, 128)          0         ['conv2d_transpose_2[0][0]',
 )                                                                   'conv2d_1[0][0]']

 conv2d_12 (Conv2D)          (None, 64, 64, 64)           73792     ['concatenate_2[0][0]']

 conv2d_13 (Conv2D)          (None, 64, 64, 64)           36928     ['conv2d_12[0][0]']

 conv2d_14 (Conv2D)          (None, 64, 64, 3)            195       ['conv2d_13[0][0]']

==================================================================================================
Total params: 7697475 (29.36 MB)
Trainable params: 7697475 (29.36 MB)
Non-trainable params: 0 (0.00 Byte)
__________________________________________________________________________________________________



2500/2500 [==============================] - 1355s 541ms/step - loss: 0.0047 - val_loss: 0.0034
Epoch 2/10
2500/2500 [==============================] - 1444s 578ms/step - loss: 0.0034 - val_loss: 0.0033
Epoch 3/10
2500/2500 [==============================] - 1297s 519ms/step - loss: 0.0033 - val_loss: 0.0032
Epoch 4/10
2500/2500 [==============================] - 1278s 511ms/step - loss: 0.0032 - val_loss: 0.0031
Epoch 5/10
2500/2500 [==============================] - 1329s 531ms/step - loss: 0.0031 - val_loss: 0.0030
Epoch 6/10
2500/2500 [==============================] - 1384s 554ms/step - loss: 0.0030 - val_loss: 0.0033
Epoch 7/10
2500/2500 [==============================] - 1400s 560ms/step - loss: 0.0030 - val_loss: 0.0030
Epoch 8/10
2500/2500 [==============================] - 2502s 1s/step - loss: 0.0030 - val_loss: 0.0029
Epoch 9/10
2500/2500 [==============================] - 1424s 570ms/step - loss: 0.0029 - val_loss: 0.0029
Epoch 10/10
2500/2500 [==============================] - 1355s 542ms/step - loss: 0.0029 - val_loss: 0.0028
Mean MSE: 0.0025848561667753647
Standard Deviation MSE: 1.360791075543637e-05
'''
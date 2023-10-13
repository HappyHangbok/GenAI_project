import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import ssl

_create_unverified_https_context = ssl._create_unverified_context
ssl._create_default_https_context = _create_unverified_https_context


# Ensure directories exist
if not os.path.exists('./tradition'):
    os.makedirs('./tradition')

if not os.path.exists('./gan'):
    os.makedirs('./gan')


# Define a traditional image blending function
def traditional_image_blending(img_path1, img_path2):
    # Load the images
    img1 = plt.imread(img_path1)
    img2 = plt.imread(img_path2)

    # Convert both images to RGB if one of them has an alpha channel
    if img1.shape[2] == 4:
        img1 = img1[:, :, :3]  # Remove the alpha channel
    if img2.shape[2] == 4:
        img2 = img2[:, :, :3]  # Remove the alpha channel

    # If images are not of the same size, trim them
    if img1.shape != img2.shape:
        min_height = min(img1.shape[0], img2.shape[0])
        min_width = min(img1.shape[1], img2.shape[1])

        img1 = img1[:min_height, :min_width]
        img2 = img2[:min_height, :min_width]

    # Blend the images
    blended_image = (img1 + img2) / 2.0

    return blended_image

# Load a pre-trained GAN model from TensorFlow Hub
gan_model = hub.load("https://tfhub.dev/google/progan-128/1")


def generate_and_save_images(num_images=5):
    # Traditional Image Generation
    for i in range(num_images):
        random_image1 = np.random.rand(128, 128, 3)
        random_image2 = np.random.rand(128, 128, 3)
        blended_image = traditional_image_blending('input1.png', 'input2.png')
        plt.imsave(f'./tradition/image{i}.png', blended_image)

    # GAN Image Generation
    for i in range(num_images):
        noise = tf.random.normal([1, 512])
        generated_image = gan_model.signatures['default'](noise)['default'][0]
        generated_image_np = generated_image.numpy()  # Convert tensor to numpy array
        generated_image_np = (generated_image_np + 1) / 2.0  # Scale values to range [0,1]
        plt.imsave(f'./gan/image{i}.png', generated_image_np)


generate_and_save_images()


# Compare quality: Here, I'll just use a simple measure (image variance as a proxy for detail)
def measure_image_quality(image_path):
    img = plt.imread(image_path)
    return np.var(img)


# Measure the quality
traditional_scores = [measure_image_quality(f'./tradition/image{i}.png') for i in range(5)]
gan_scores = [measure_image_quality(f'./gan/image{i}.png') for i in range(5)]

# Visualization
labels = ['Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5']
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, traditional_scores, width, label='Traditional')
rects2 = ax.bar(x + width / 2, gan_scores, width, label='GAN')

ax.set_ylabel('Image Quality (Variance)')
ax.set_title('Quality comparison between Traditional and GAN methods')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.savefig("plot.png")
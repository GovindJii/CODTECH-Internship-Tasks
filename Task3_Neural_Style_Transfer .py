import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & HYPERPARAMETERS
# -----------------------------------------------------------------------------
# Weights determine how much emphasis to put on style vs content
CONTENT_WEIGHT = 1e4
STYLE_WEIGHT = 1e-2
TOTAL_VARIATION_WEIGHT = 30  # Helps smooth out the result

# layers to use for style and content
CONTENT_LAYERS = ['block5_conv2'] 
STYLE_LAYERS = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1', 
    'block4_conv1', 
    'block5_conv1'
]

IMG_MAX_DIM = 512 # Resize images to this max dimension for speed/memory

# -----------------------------------------------------------------------------
# 2. IMAGE UTILITIES
# -----------------------------------------------------------------------------
def load_img(path_to_img):
    """Loads an image, resizes it, and adds a batch dimension."""
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = IMG_MAX_DIM / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :] # Add batch dim
    return img

def imshow(image, title=None):
    """Helper to display image."""
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)

def tensor_to_image(tensor):
    """Converts a tensor back to a valid PIL-saveable image."""
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return tensor

# -----------------------------------------------------------------------------
# 3. MODEL BUILDER (VGG19)
# -----------------------------------------------------------------------------
def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on Imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        # Preprocess input (expects [0,255] range for VGG preprocessing)
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        return {'content': content_dict, 'style': style_dict}

# -----------------------------------------------------------------------------
# 4. LOSS FUNCTIONS
# -----------------------------------------------------------------------------
def gram_matrix(input_tensor):
    """Calculates the Gram Matrix (style representation)."""
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations

def total_variation_loss(image):
    x_deltas, y_deltas = tf.image.image_gradients(image)
    return tf.reduce_mean(tf.abs(x_deltas)) + tf.reduce_mean(tf.abs(y_deltas))

def compute_loss(outputs, style_targets, content_targets, style_weight, content_weight, image):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    
    # Add Style Loss
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / len(style_outputs)

    # Add Content Loss
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / len(content_outputs)
    
    # Add Total Variation Loss (smoothness)
    tv_loss = total_variation_loss(image) * TOTAL_VARIATION_WEIGHT
    
    loss = style_loss + content_loss + tv_loss
    return loss

# -----------------------------------------------------------------------------
# 5. EXECUTION
# -----------------------------------------------------------------------------

def load_and_process_image(path, target_shape=None):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
 
    if target_shape is not None:
        img = tf.image.resize(img, target_shape)
 
    img = img[tf.newaxis, :]
    return img 

# --- DOWNLOAD SAMPLE IMAGES (If running locally, replace with your own paths) ---
# Content: A photo of a Labrador
content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
# Style: Starry Night by Van Gogh
style_path = tf.keras.utils.get_file('kandinsky5.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

print("Loading images...")
# content_image = load_img(content_path)
# style_image = load_img(style_path) 
content_image = load_and_process_image(content_path)
style_image = load_and_process_image(
    style_path,
    target_shape=content_image.shape[1:3]
)

# Initialize Model
extractor = StyleContentModel(STYLE_LAYERS, CONTENT_LAYERS)

# Calculate Targets
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

# Define Optimizer
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

# The Image to optimize (Start with Content Image)
image = tf.Variable(content_image)

# Optimization Step
@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = compute_loss(outputs, style_targets, content_targets, 
                            STYLE_WEIGHT, CONTENT_WEIGHT, image)
    
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, 0.0, 1.0))

# Training Loop
epochs = 10
steps_per_epoch = 100

print("Starting Style Transfer...")
start = time.time()

for n in range(epochs):
    for m in range(steps_per_epoch):
        train_step(image)
    print(f"Epoch {n+1}/{epochs} completed. Time: {time.time()-start:.1f}s")
    
    # Optional: Display intermediate result
    # plt.imshow(tensor_to_image(image))
    # plt.show()

end = time.time()
print(f"Total time: {end-start:.1f}s")

# -----------------------------------------------------------------------------
# 6. VISUALIZATION OF RESULTS
# -----------------------------------------------------------------------------
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 3, 2)
imshow(style_image, 'Style Image')

plt.subplot(1, 3, 3)
imshow(image, 'Styled Result')

plt.tight_layout()
plt.savefig('style_transfer_result.png')
print("Result saved as 'style_transfer_result.png'")
plt.show()
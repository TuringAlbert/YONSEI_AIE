import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

image = Image.open("/Users/juyoungkim/Development/aie/aie_opensourceSW/face_collector _aie_opensourceSW_final/pinterest_images/gentle guy_16.jpg")
image_array  = np.array(image)
print(image_array.shape)

red_channel = image_array[:,:, 0]
green_channel = image_array[:,:,1]
blue_channel = image_array[:,:,2]
print(red_channel.shape)
print(green_channel.shape)
print(blue_channel.shape)

# merged_image = np.dstack((red_channel, green_channel, blue_channel))
# merged_image = Image.fromarray(merged_image)
# merged_image.save("merged_image.jpg")
#
# gray_image = np.dot(image_array, [0.2989,0.5870, 0.1140])
# gray_image = gray_image.astype(np.uint8)
#
# flipped_image = np.fliplr(image_array)
# flipped_image = Image.fromarray(flipped_image)
# flipped_image.save("flipped_image.jpg")

blurred_image = image.filter(ImageFilter.BLUR)

sharpened_image = image.filter(ImageFilter.SHARPEN)

enhancer = ImageEnhance.Brightness(image)
brightened_image = enhancer.enhance(1.5)

enhancer = ImageEnhance.Contrast(image)
contrasted_image = enhancer.enhance(1.2)

# blurred_path = "blurred_image.jpg"
blurred_image.show()
sharpened_image.show()
brightened_image.show()
contrasted_image.show()
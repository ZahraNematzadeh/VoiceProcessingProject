import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def pad_crop_image_vit(img_array, target_shape):
    cropped_and_padded_images = []

    for img_array_single in img_array:
        height, width, channels = img_array_single.shape
        
        top_crop = 0
        top_pad = 0
        bottom_pad = 0
        
        if target_shape[0] > height:
            pad_height = target_shape[0] - height
            top_pad = pad_height // 2
            bottom_pad = pad_height - top_pad
            cropped_height = height  
        else:
            top_crop = (height - target_shape[0]) // 2
            #bottom_crop = height - target_shape[0] - top_crop
            cropped_height = target_shape[0]
            
        left_crop = 0
        left_pad = 0
        right_pad = 0
            
        if target_shape[1] > width:
            pad_width = target_shape[1] - width
            left_pad = pad_width // 2
            right_pad = pad_width - left_pad
            cropped_width = width  
        else:
            left_crop = (width - target_shape[1]) // 2
            cropped_width = target_shape[1]

        cropped_image = img_array_single[top_crop:top_crop + cropped_height,
            left_crop:left_crop + cropped_width,:,]

        padded_image = np.pad(cropped_image,
            ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),mode='mean')

        cropped_and_padded_images.append(padded_image)

    for i in range(min(3, len(cropped_and_padded_images))):
        print(cropped_and_padded_images[i].shape)
        image = cropped_and_padded_images[i]
        scalers = [MinMaxScaler() for _ in range(image.shape[2])]
        scaled_channels = [scaler.fit_transform(image[:, :, c]) for c, scaler in enumerate(scalers)]
        scaled_image = np.stack(scaled_channels, axis=2)
        print(scaled_image.shape)
        plt.imshow(scaled_image)
        plt.show()

    return np.array(cropped_and_padded_images)
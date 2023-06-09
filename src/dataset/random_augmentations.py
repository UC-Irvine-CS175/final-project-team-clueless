import augmentation
import numpy as np
import random


def perform_random_augmentations(image: np.ndarray) -> np.ndarray:

    # Setup List of random augmentations
    rotate = random.shuffle([90,180,270])[0]
    augmentations_list = ['v_flip', 'h_flip', 'rotate']
    random.shuffle(augmentations_list)
    aug_count = random.randint(1, len(augmentations_list))

    # Perform random augmentations
    for i in range(aug_count):
        if augmentations_list[i] == 'v_flip':
            v_flip = augmentation.VFlipBPS()
            image = v_flip.__call__(image)
        elif augmentations_list[i] == 'h_flip':
            h_flip = augmentation.HFlipBPS()
            image = h_flip.__call__(image)
        elif augmentations_list[i] == 'rotate':
            image = augmentation.RotateBPS(rotate)

    # Normalize the array
    normalize = augmentation.NormalizeBPS()
    image = normalize.__call__(image)

def create_augmentations_data(dir: str):
    
    return None

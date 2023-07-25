import numpy as np

def melspect_array(melspect_data):
    melspect_array = np.array([item[0] for item in melspect_data])
    melspect_array = np.stack(melspect_array, axis=0)
    melspect_array = melspect_array.reshape(
                            melspect_array.shape[0],
                            melspect_array.shape[1],
                            melspect_array.shape[2],
                            1)
    return melspect_array

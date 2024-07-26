**Optimizing Deep Learning Models for Voice Pathology Detection: A Case Study with ResNet50**

## Description
This project processes audio files and converts them into visual representations. 
You can choose to convert audio to Mel Spectrograms or Leaf images (currently leaf images does not work in this code), 
and apply different machine learning models including CNN, Transfer Learning, and Vision Transformer.

## Folder Structure

- **`docs/`**: Contains the `requirements.txt` file.
- **`models/`**: Includes model scripts:
  - `cnn.py`
  - `inceptionv3.py`
  - `resnet50.py`
  - `vit.py`
  - `xception.py`
- **`src/`**: Contains all functions used in `main.py`.
- **`helper/`**: Includes helper functions for testing.
- **`config/`**: Contains `config.py` for configuration settings.

## Configuration

You can change the default configuration in `config/config.py`. The default parameters are as follows:

- `K_fold = 10`
- `Epoch = 100`
- `Batch_size = 64`
- `num_classes = 2`
- `sample_rate = 44100`
- `max_duration = 5`
- `target_shape = (224, 224, 3)`  # Only used for Vision Transformer

## Usage

1. **Change the data directory**:
   - Update the data directory path in `config/config.py`.

2. **Run the `main.py` script**:
  Follow the prompts:
  - Enter your desired alphabet:
    **m** to convert audio files to **Mel Spectrograms**.
    **l** to convert audio files to **Leaf** images. (**Note:** Currently, this option does not work in the code.)

  - Choose the machine learning model:
    **c** to use Convolutional Neural Networks (CNN).
    **t** to use Transfer Learning:
      **r** for ResNet50
      **i** for InceptionV3
      **x** for Xception
    **v** to use Vision Transformer.
  
**View results:**
  The final performance metrics will be displayed at the end of the run.
  Plots and the best final model and weights will be saved in the **final_output_path**, which should be specified in the **config.py** file.

**Requirements**
  Python 3.8
  Required libraries (see docs/requirements.txt)

**Installation**
1. Clone the repository:
     git clone https://github.com/ZahraNematzadeh/VoiceProcessingProject.git
2. Navigate to the project directory:
     cd VoiceProcessingProject
3. Install the required dependencies:
     pip install -r docs/requirements.txt

**Contributing**
If you wish to contribute to this project, please follow the standard fork-and-pull request workflow. 
Ensure that your contributions are well-documented and include appropriate tests.

**License**
This project is licensed under the MIT License - see the LICENSE file for details.

**Contact**
Zahra Nematzadeh - zahra.nematzadeh87@gmail.com


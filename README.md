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

1. **Change the data directory**:<br>
   - Update the data directory path in `config/config.py`.

2. **Run the `main.py` script**:<br>
  Follow the prompts:<br>
  - Enter your desired alphabet:<br>
    **m** to convert audio files to **Mel Spectrograms**.<br>
    **l** to convert audio files to **Leaf** images. (**Note:** Currently, this option does not work in the code.)<br>

  - Choose the machine learning model:<br>
    **c** to use Convolutional Neural Networks (CNN).<br>
    **t** to use Transfer Learning:<br>
      **r** for ResNet50<br>
      **i** for InceptionV3<br>
      **x** for Xception<br>
    **v** to use Vision Transformer.<br>
  
**View results:** <br>
  The final performance metrics will be displayed at the end of the run.<br>
  Plots and the best final model and weights will be saved in the **final_output_path**, which should be specified in the **config.py** file.<br>

**Requirements** <br>
  Python 3.8 <br>
  Required libraries (see docs/requirements.txt) <br>

**Installation** <br>
1. Clone the repository: <br>
     git clone https://github.com/ZahraNematzadeh/VoiceProcessingProject.git <br>
2. Navigate to the project directory: <br>
     cd VoiceProcessingProject <br>
3. Install the required dependencies: <br>
     pip install -r docs/requirements.txt <br>

**Contributing** <br>
If you wish to contribute to this project, please follow the standard fork-and-pull request workflow. <br>
Ensure that your contributions are well-documented and include appropriate tests.<br>

**License** <br>
This project is licensed under the MIT License - see the LICENSE file for details. <br>

**Contact** <br>
Zahra Nematzadeh - zahra.nematzadeh87@gmail.com<br>


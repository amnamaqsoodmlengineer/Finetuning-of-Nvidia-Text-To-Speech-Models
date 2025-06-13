# Finetuning-of-Nvidia-Text-To-Speech-Models
![image](https://github.com/user-attachments/assets/9ed074bd-372a-4fe0-9103-699da5ce194e)

This repository contains the steps and scripts used to fine-tune the Nvidia TTS models tts_en_fastpitch and HiFi-GAN on a custom dataset. Below is a step-by-step guide on how the fine-tuning process was conducted.

## Steps

### 1. Transcription
The first step involved transcribing the audio files using a transcriber.

### 2. Audio Splitting
We then split the audio based on timestamps from the transcribed files using the pipeline.

### 3. Conversion to Desired JSON Format
The next step was to convert the data to the desired JSON format.

### 4. Data Preparation for FastPitch
The prepared data was then passed through the pipeline to get it ready for the fine-tuning process.

### 5. Manifest File Creation
We created a `manifest.json` file which contains the details of the dataset.

### 6. Splitting Manifest File
The `manifest.json` file was split into `dev.json` and `train.json` for development and training purposes respectively.

### 7. Test File Creation
A separate `test.json` file was also created for testing purposes.

### 8. Fine-Tuning
Finally, we fine-tuned the FastPitch and HiFi-GAN models using the `Text to Speech Finetuning using NeMo.ipynb` notebook.

## Configuration Files

### FastPitch Configuration

- **Original File**: `fastpitch_align_v1.05_From Scratch.yaml`
  - This file is for reference and is used to start fine-tuning FastPitch from scratch.

- **Resume Checkpoints File**: `fastpitch_align_v1.05_Resume_Checkpoints.yaml`
  - If FastPitch has been trained before and you wish to resume from those checkpoints, use this file. It includes the following parameters in `exp_manager`:
     - resume_if_exists: false
     - resume_ignore_no_checkpoint: false
     - resume_from_checkpoint: "results_dir/FastPitch/2024-07-10_12-20-11/checkpoints/FastPitch--val_loss=0.6365-epoch=150-last.ckpt"
### HiFiGAN Configuration

- **Original File**: `hifigan.yaml`
  - This file is for reference and is used to start training HiFiGAN from scratch.

- **Resume Checkpoints File**: `hifigan.yaml_Resume_Checkpoints.yaml`
  - If HiFiGAN has been trained before and you wish to resume from those checkpoints, use this file. It includes the following parameters in `exp_manager`:
    - resume_if_exists: false
    - resume_ignore_no_checkpoint: false
    - resume_from_checkpoint: "results_dir/HifiGan/2024-07-11_09-33-51/checkpoints/HifiGan--val_loss=0.4053-epoch=36-last.ckpt"
## Usage

1. Clone the repository.
2. Follow the steps outlined above to prepare your data.
3. Use the provided notebooks to transcribe, split, convert, prepare, and fine-tune your models.
4. Choose the appropriate configuration file based on whether you're starting from scratch or resuming from checkpoints.

## Prerequisites

- Python 3.10.12
- torch 2.0.1
- torchvision 0.15.2
- pip install cython
- pip install nemo_toolkit['all']
- Jupyter Notebook
- Required Python libraries (detailed in each notebook)

# Voice2Text
> ### Language
> - [中文](README.md)
> - [English](README_EN.md)

### Voice2Text is a voice-to-text application based on OpenAI Whisper, supporting audio, video, and real-time speech-to-text.

  ### How to Use

  #### Method 1

  1. Download the package [Voice2Text-pkg.rar](https://github.com/caiwuu/Voice2Text/releases/tag/1.0.0) and unzip it.
  2. Go to the [model repository](https://huggingface.co/Systran) to download the faster-whisper-large-v2 model and place it in the models folder.
  3. Double-click `run.bat` on Windows or `run.sh` on Linux/Mac to run.
  4. If you want to use a GPU, download and install CUDA12 yourself (the same applies to Method 2).

  #### Method 2

  1. Clone the repository.
     ```bash
     git clone https://github.com/caiwuu/Voice2Text
     cd ./Voice2Text
     ```

  2. Create a Python virtual environment.
     ```bash
     conda create -p ./env python==3.11.9
     conda activate ./env
     ```

  3. Install the dependencies.
     ```bash
     pip install -r requirements.txt
     ```

  4. Download the model into the models folder from the repository: [https://huggingface.co/Systran](https://huggingface.co/Systran). The default model is faster-whisper-large-v2; you can change the model name in the code for other models.
  
  5. Start the application.
     ```bash
     python webUI.py
     ```
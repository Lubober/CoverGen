## Getting Started

Follow these steps to set up the project environment:

1. **Create the Conda environment:**  
   Run the following command in your terminal to create an environment from the provided YAML file:

   ```bash
   conda env create -f environment.yml

2. **Activate the Conda environment:**
   ```bash
   conda activate covergen_env
2. **Install ffmpeg:**
   ```bash
   conda install -c conda-forge ffmpeg
  
## To Use The Model:
1. **Place The audio file that you want to generate an image from into the ./data/audios/ directory**
2. **Run the following command:**
   ```bash
   python main.py <filename>
Note that the file name should not contain the file extension eg. if you want to generate an image from song.mp3, the command is "python main.py song"

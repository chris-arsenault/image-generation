# Image Generation
Uses DiffUsers from HuggingFace https://huggingface.co/blog/stable_diffusion

Provides a script to run sucesssive image generation based on a text prompt set.  Has options for using each image as a base for the next, as well as generating images with diffrent styles.

## Setup
* Install miniconda
* Make sure you have CUDA installed.  For a WSL2 tutorial see: https://ubuntu.com/tutorials/enabling-gpu-acceleration-on-ubuntu-on-wsl2-with-the-nvidia-cuda-platform#1-overview
* `conda env create -f environment.yaml`
* `conda activate ldm`
* `pip install diffusers==0.2.4 transformers scipy ftfy`

## Running
* `python3 run_interleave.py`
* This will take ~5 second per 50 iterations on a RTX3090

## Options
### prompt_config
This section configures the source to read from for each image interation

| Key      | Type | Description |
| ----------- | ----------- | ----------- |
| source | Text | Name of the text file to use a source for each image set | 
| key | Text | Simple key used to name the folders and files | 
| prompt_override | Text | Used to provide a static prompt instead of reading the file.  The file is still used to control the number of images. | 
| chapter_split_regex | Regex | Regex to split the textfile into a set of chapters | 
| prompt_split_regex | Regex | Regext to split the chapters into a set of prompts | 
| max_chapters | Integer | Maximum number of chapters to run | 
| max_entries_per_chapter | Integer | Maximum number of prompts per chapter |


### generation_config
This section configures how the image is generated

| Key      | Type | Description |
| ----------- | ----------- | ----------- |
| preview | Text | Attempt to show the images as they're generated instead of just writing to file. | 
| preview_stop | Text | Command to execute to clear preview windows. | 
| preview_max_windows | Integer | Maximum number of perview image windows. | 
| save_individual | Boolean | Save each individual image, or just the final image grid. | 
| progresive | Boolean | If set to true, use each previous image as a base for the next instead of a black screen. | 
| interleave | Boolean | If set to true, cycle each last image in the prompt set.  Only works with progresive set to true. | 
| cycle_styles | Boolean | If set to true, cycle the prompts between the prompt sets.  Only works with progressive set to true. | 
| guidance_scale | Float | How much to guide towards the final prompt while genrating the image.  7.5 is decent, under 5 or over 10 generally produces poor images | 
| eta | Float | This does something.  Values 0 - 0.3 | 
| num_inference_steps | Integer | Number of steps to use, default 50.  More steps takes more time.  Under 20 steps and the image is noisy, and after 200 the image over-stylizes. |

### style_configs
This section configures multiple style to be applied to each prompt.  Each style specified creates an additional image.

| Key      | Type | Description |
| ----------- | ----------- | ----------- |
| style | Text | Styles to apply to each prompt.  Generally names of artists or art styles, but anything can be used. | 
| file_tag | Text | Short description used to create file names. | 
| weight | Float | Relative weight of the prompt in the overall prompt set.  Modifys the number of iterations run with the prompt.  0 would be no iterations, 1 would be max iterations. | 
| last_image | None | Used to store images while generation is happening. Do Not modify. | 
| images | Empty Array | Used to store images while generation is happening. Do Not modify. | 

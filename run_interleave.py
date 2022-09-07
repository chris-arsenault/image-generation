import torch
import re
import math
import os
import datetime
import functools

from torch import autocast
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DDPMPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image, ImageShow
        
prompt_config = {
    'source' : 'bible.txt',
    'key' : 'bible',
    'prompt_override' : None,
    'chapter_split_regex' : 'Gen.\d+',
    'prompt_split_regex' : '\n',
    'max_chapters' : 1,
    'max_entries_per_chapter' : 25
}

generation_config = {
    'preview' : False,
    'preview_stop' : 'pkill display',
    'preview_max_windows' : 10,
    'save_individual' : True,
    'progresive' : True,
    'interleave' : True,
    'cycle_styles' : False,
    'guidance_scale' : 7.5,
    'eta' : 0.1,
    'num_inference_steps' : 50
}

style_configs = [
    {
        'style' : 'no text hdr 4k 8k hyperreal photography nature ansel adams',
        'file_tag' : 'nature',
        'weight' : 0.8,
        'last_image' : None,
        'images' : []
    },
    {
        'style' : 'no text charcoal line art hyperfine davinci michelangelo',
        'file_tag' : 'davinci',
        'weight' : 0.9,
        'last_image' : None,
        'images' : []
    },
    {
        'style' : 'no text cyberpunk',
        'file_tag' : 'cyberpunk',
        'weight' : 0.8,
        'last_image' : None,
        'images' : []
    },
    {
        'style' : '',
        'file_tag' : 'text',
        'weight' : 0.1,
        'last_image' : None,
        'images' : []
    },
    {
        'style' : 'no text jesus comes down from the mount',
        'file_tag' : 'jesus',
        'weight' : 0.2,
        'last_image' : None,
        'images' : []
    },
    {
        'style' : 'no text god angel uderwater',
        'file_tag' : 'cyberpunk',
        'weight' : 0.2,
        'last_image' : None,
        'images' : []
    }
]

        
def create_folder():
    progresive_text = 'progresive' if generation_config['progresive'] else 'independent'
    epoch = datetime.datetime.now()
    styles = ' '.join([style['file_tag'] for style in style_configs])
    folder = f"{prompt_config['key']}/{progresive_text}/{styles}/{epoch}"
    os.system(f"mkdir -p '{folder}'")
    return folder

def chapters():
    with open(prompt_config['source']) as f:
        prompt_source = f.read()
        
    chapters = re.split(prompt_config['chapter_split_regex'], prompt_source)
    return chapters[1:1 + prompt_config['max_chapters']]
    
def prompts(chapter):
    prompts = re.split(prompt_config['prompt_split_regex'], chapter)
    return prompts[1:1 + prompt_config['max_entries_per_chapter']]


@functools.lru_cache(maxsize=None)
def create_pipelines():
    model_id = "CompVis/stable-diffusion-v1-2"
    device = "cuda"

    blank_pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
    blank_pipe = blank_pipe.to(device)

    if generation_config['progresive']:
        img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, use_auth_token=True)
        img2img_pipe = img2img_pipe.to(device)
        
    return blank_pipe, img2img_pipe
    
def create_image(prompt, style_config):
    blank_pipe, img2img_pipe = create_pipelines()
    
    p = f"{prompt} {style_config['style']}"
    
    if style_config['last_image'] is None or not generation_config['progresive']:
        last_image = blank_pipe(p, 
            guidance_scale=generation_config['guidance_scale'], 
            eta=generation_config['eta'], 
            num_inference_steps=generation_config['num_inference_steps'])
    else:
        last_image = img2img_pipe(p, 
            guidance_scale=generation_config['guidance_scale'], 
            eta=generation_config['eta'], 
            num_inference_steps=generation_config['num_inference_steps'],
            init_image=style_config['last_image'], 
            strength=style_config['weight'])
        
    return last_image['sample'][0]
    
def image_grid(imgs):
    rows = cols = math.ceil(math.sqrt(len(imgs)))
    
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

### END SETUP ###

folder = create_folder()
for chapter_index, chapter in enumerate(chapters()):
    for prompt_index, prompt in enumerate(prompts(chapter)):
        if prompt_config['prompt_override']:
            prompt = prompt_config['prompt_override']
  
        print(f"[{chapter_index}-{prompt_index}] {prompt}")
        for style_config in style_configs:
            style_config['last_image'] = create_image(prompt, style_config)
            style_config['images'].append(style_config['last_image'])
            if generation_config['save_individual']:
                style_config['last_image'].save(f"{folder}/{style_config['file_tag']} {chapter_index}-{prompt_index}.png")
            
        if generation_config['preview']:
            if prompt_index % generation_config['preview_max_windows'] == 0:
                os.system(generation_config['preview_stop'])
            ImageShow.show(style_configs[0]['last_image'])
                  
        if generation_config['interleave']:
            last_images = [s['last_image'] for s in style_configs]
            last_images = last_images[1:] + last_images[:1]
            for config_index, style_config in enumerate(style_configs):
                style_config['last_image'] = last_images[config_index]
                
        if generation_config['cycle_styles']:
            styles = [s['style'] for s in style_configs]
            styles = styles[1:] + styles[:1]
            for config_index, style_config in enumerate(style_configs):
                style_config['style'] = styles[config_index]
            
    for style_config in style_configs:
        grid = image_grid(style_config['images'])
        grid.save(f"{folder}/{style_config['file_tag']} grid-{chapter_index}.png")
    

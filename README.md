---
title: Image Morphing Transport
emoji: 🪐
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

# image_transporter

image_transporter is a python script that uses opencv to transport an image to another image, inspired by the obamify project.

## Demo

obamify demo:

![Demo Animation](cat2obama.gif)

Rem demo:

![Demo Animation](random2rem.gif)

# How to use

| Setting        | Description                                                                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| stiffness      | How much the algorithm changes the original image to make it look like the target image. Increase this if you want a more subtle transformation. |
| damping        | How much the algorithm changes the original image to make it look like the target image. Increase this if you want a more subtle transformation. |
| stretch_factor | How much the algorithm changes the original image to make it look like the target image. Increase this if you want a more subtle transformation. |
| quick_gen      | Whether to use quick generation or not. Quick generation is faster but lower quality.                                                            |
| cell_size      | Number of cells in the grid, higher cell_size = higher quality but slower                                                                        |
| source_path    | Path to the source image                                                                                                                         |
| target_path    | Path to the target image                                                                                                                         |

# Contributing

open an issue or pull request if you find any bugs or have any suggestions to improve this program

# How it works

magiccccccccccccccccccccccc

# License

sorry. instead of reducing frame, reduce the size of file, form 800 x 800 to smaller size, since i want to keep the original speed. it has to be less than 100 mb so i can upload to github
MIT License

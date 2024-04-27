[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/UwpqMYOQ)
# e4040-2023Fall-project
## TODO: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

This report provides a concise summary of the theories and experiments presented in the original paper, titled "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." and replicates the results using TensorFlow. The Swin Transformer stands out as an innovative vision transformer designed to serve as a versatile backbone in computer vision. The implemented hierarchical Transformer architecture employs shifted windows for representation computation, enhancing efficiency by confining self-attention computations to non-overlapping local windows and promoting cross-window connections. This report delves into the efficacy of Swin Transformer blocks by comparing the performance of the original model with variants that exclude Window Shift, Relative Position, and Drop Path in the context of image classification. Due to equipment constraints, Tiny Imagenet from Stanford CS231N was utilized for the experiments.

## Structure description
  - **Original Swin Transformer:** the training, validation process and results of Swin Transformer on Tiny Imagenet.
  - [[https://drive.google.com/file/d/1jWeANgyn_q4HuIGKZvG09XZYNlXtipMC/view?usp=drive_link](https://drive.google.com/file/d/1K8Nc9XVP5pOjbeJKLv3jYDhByeQyE-9G/view?usp=drive_link)]   
  - **Swin Transformer without Relative Position:** the training, validation process and results of Swin Transformer without implementing the Relative Position technique on Tiny Imagenet.
  - [https://drive.google.com/file/d/1xSb1p5-aDllHfHSxcoWgvNjOhDjw-sSL/view?usp=drive_link](https://drive.google.com/file/d/1jWeANgyn_q4HuIGKZvG09XZYNlXtipMC/view?usp=drive_link)
  - **Swin Transformer without Shift Window:** the training, validation process and results of Swin Transformer without implementing the Shift Window technique on Tiny Imagenet.
  - [https://drive.google.com/file/d/1WzyWy-_uKrXbexxyOpd13g-Fv9LS0jp6/view?usp=drive_link](https://drive.google.com/file/d/1WzyWy-_uKrXbexxyOpd13g-Fv9LS0jp6/view?usp=drive_link)
  - **Swin Transformer without Drop Path:** the training, validation process and results of Swin Transformer without implementing the Drop Path technique on Tiny Imagenet.
  - [https://drive.google.com/file/d/1K8Nc9XVP5pOjbeJKLv3jYDhByeQyE-9G/view?usp=drive_link](https://drive.google.com/file/d/1xSb1p5-aDllHfHSxcoWgvNjOhDjw-sSL/view?usp=drive_link)

# Organization of this directory
To be populated by students, as shown in previous assignments.
TODO: Create a directory/file tree
```
- e4040-2023fall-project-zkh3
  ├── Original Swin Transformer
  │   ├── utils
  │   │   ├── build_loader.py
  │   │   ├── build_model.py
  │   │   ├── config.yaml
  │   │   ├── cos_lr_warmup.py
  │   │   ├── get_config.py
  │   │   ├── logger.py
  │   │   ├── main.py
  │   │   ├── README.md
  │   │   └── swin_transformer.py
  │   ├── imagenet
  │   │   └── train
  │   └── main.ipynb
  
  ├── Swin Transformer without Drop Path
  │   ├── utils
  │   │   ├── build_loader.py
  │   │   ├── build_model.py
  │   │   ├── config.yaml
  │   │   ├── cos_lr_warmup.py
  │   │   ├── get_config.py
  │   │   ├── logger.py
  │   │   ├── main.py
  │   │   ├── README.md
  │   │   └── swin_transformer.py
  │   ├── imagenet
  │   │   └── train
  │   └── main.ipynb
  
  ├── Swin Transformer without Relative Position
  │   ├── utils
  │   │   ├── build_loader.py
  │   │   ├── build_model.py
  │   │   ├── config.yaml
  │   │   ├── cos_lr_warmup.py
  │   │   ├── get_config.py
  │   │   ├── logger.py
  │   │   ├── main.py
  │   │   ├── README.md
  │   │   └── swin_transformer.py
  │   ├── imagenet
  │   │   └── train
  │   └── main.ipynb
  
  ├── Swin Transformer without Shift Window
  │   ├── utils
  │   │   ├── build_loader.py
  │   │   ├── build_model.py
  │   │   ├── config.yaml
  │   │   ├── cos_lr_warmup.py
  │   │   ├── get_config.py
  │   │   ├── logger.py
  │   │   ├── main.py
  │   │   ├── README.md
  │   │   └── swin_transformer.py
  │   ├── imagenet
  │   │   └── train
  │   └── main.ipynb

python main.py

```

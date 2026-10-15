# Introduction to Diffusion Models

This folder is a small, notebook-based introduction to diffusion models. It uses
one-dimensional sine waves so that the noising and generation processes are easy
to visualize.

## Notebooks

Read the notebooks in this order:

1. [`diffusion_intro.ipynb`](diffusion_intro.ipynb) introduces unconditional
   diffusion, including the noise schedule, time embeddings, training, and
   sampling.
2. [`conditional_diffusion_intro.ipynb`](conditional_diffusion_intro.ipynb)
   shows how to control the generated wave with a frequency condition.
3. [`conditional_diffusion_debug.ipynb`](conditional_diffusion_debug.ipynb)
   compares a basic U-Net with an attention-based U-Net and discusses their
   limitations.

[`funs_intro.ipynb`](funs_intro.ipynb) is an optional appendix explaining the
helper functions, tensor shapes, and model components.

## Running the notebooks

From this directory, install the required packages and start Jupyter:

```bash
python -m pip install torch numpy scipy matplotlib jupyter
jupyter lab
```

The notebooks automatically use Apple Silicon, a CUDA GPU, or the CPU,
depending on what is available.

Training can take some time. Pretrained weights are included in
[`trained_parameters/`](trained_parameters/). Set `TRAIN_MODEL = False` in a
notebook to load them instead of training a new model.

## Project layout

- `funs/`: datasets, model definitions, plotting tools, and diffusion utilities
- `trained_parameters/`: pretrained model weights
- `figs/`: figures and animations used by the notebooks


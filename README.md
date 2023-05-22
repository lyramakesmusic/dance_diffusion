# dance_diffusion

A simple python wrapper for dance diffusion inference.

---

## Installation

Clone the repository and install the required libraries:

```bash
git clone https://github.com/lyramakesmusic/dancediffusion.git
cd dancediffusion
pip install ...
```

Note: Make sure you have the required trained model checkpoint file (`.ckpt`) in the project directory.

## Usage

Import `dd` and create an instance with your checkpoint path, sample size, and sample rate:

```python
from dance_diffusion import dd

ckpt_path = "path/to/your/checkpoint.ckpt"
sample_size = 131072
sample_rate = 44100

model = dd(ckpt_path, sample_size, sample_rate)
```

### Generate Audio

Generate a batch of audio clips using the model:

```python
output_folder = "outputs"
batch_size = 5
steps = 100

save_paths = model.generate_audio(output_folder, batch_size, sample_size, steps)
```

`save_paths` is a list of the file paths of generated audio clips in WAV format.


## Example

```python
from dance_diffusion import dd

ckpt_path = "path/to/your/checkpoint.ckpt"
sample_size = 131072
sample_rate = 44100
output_folder = "outputs"
batch_size = 5
steps = 100

model = DiffusionUncond(ckpt_path, sample_size, sample_rate)
save_paths = model.generate_audio(output_folder, batch_size, sample_size, steps)

print("Generated audio files:")
for path in save_paths:
    print(path)
```

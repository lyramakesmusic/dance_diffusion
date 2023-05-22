# dance_diffusion

A simple python wrapper for dance diffusion inference.

---

## Installation

Clone the repository and install the required libraries:

```bash
git clone https://github.com/lyramakesmusic/dance_diffusion.git
cd dance_diffusion
pip install -e .
pip install -r requirements.txt
```

Note: Make sure you have the required trained model checkpoint file (`.ckpt`) in the project directory.

Here is a small collection of models I've finetuned. Quality *will* vary! You'll need to do multiple generations and manually slice up the samples for good results.

| Model      | Sample Length | Sample Rate | Link |
| ----------- | ----------- | ----------- | ----------- |
| Neuro & dubstep basses | 131072 | 44100 | [Download](https://drive.google.com/file/d/1-IL74bhrZKrYbjfmEmG70dEWezFfxJgW/view?usp=sharing) (3.3GB)|
| Drum fills   | 131072 | 44100 | [Download](https://drive.google.com/file/d/1-B2S1AhyDz4eMK9n3xRnPclW4OzV7lLf/view?usp=sharing) (3.3GB)        |
| Hardstyle kicks | 65536 | 44100 | [Download](https://drive.google.com/file/d/1-S6gGj5qW1mKQP4yxNyO4XXpUzZkjYsU/view?usp=sharing) (3.3GB)  |
| Top loops | 131072 | 44100 | [Download](https://drive.google.com/file/d/1-IMK47o1HhqCSHoolsEx-cKnIijkrDDC/view?usp=sharing) (3.3GB) |
| Snares | 65536 | 44100 | Download (3.3GB) |


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
batch_size = 5
steps = 100

audio_data = model.gen(batch_size, steps)

filename = "path/to/your/output/file.wav"
audio_data.save(filename)
```

Note: The `audio_data` object is an instance of `AudioData`, which is a wrapper for a tensor that includes a `.save()` method. To extract the tensor data directly from the `AudioData` object, you can use:

```python
tensor_data = audio_data.data
```

`tensor_data` will be the PyTorch tensor containing the raw audio data.

## Full Example

```python
from dance_diffusion import dd

model = dd("path/to/your/checkpoint.ckpt", sample_size=131072, sample_rate=44100)

model.gen(batch_size=5, steps=100).save("path/to/your/output/file.wav")

print("Generated audio file saved to:")
print(filename)
```

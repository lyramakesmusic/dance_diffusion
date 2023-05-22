from setuptools import setup, find_packages

setup(
    name="dance_diffusion",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchaudio",
        "einops",
        "scipy",
        "k-diffusion",
        "v-diffusion-pytorch",
        "sample-generator",
    ],
)

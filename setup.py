from setuptools import find_packages, setup

requirements = [
    "torch",
    "numpy",
    "torchvision",
    "torchaudio",
    "transformers",
    "scikit-learn",
    "wandb",
    "datasets",
]

setup(
    name='transformer_xray',
    packages=find_packages(),
    install_requires=requirements,
)
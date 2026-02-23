from setuptools import setup, find_packages

setup(
    name='solospeech',
    version='0.1.0',
    packages=find_packages() + ["geco", "fastgeco"],
    package_dir={
        "geco": "solospeech/corrector/geco",
        "fastgeco": "solospeech/corrector/fastgeco",
    },
    install_requires=[
        'pytorch-lightning==2.4.0',
        'torch==2.4.1',
        'torchaudio==2.4.1',
        'torchvision==0.19.1',
        'wandb==0.19.1',
        'diffusers==0.30',
        'librosa==0.9.2',
        'speechbrain==1.0.2',
        'python_speech_features==0.6'
    ],
    author='Helin Wang',
    description='SoloSpeech: Enhancing Intelligibility and Quality in Target Speaker Extraction through a Cascaded Generative Pipeline',
    url='https://github.com/WangHelin1997/SoloSpeech',
    python_requires='>=3.8.19',
)

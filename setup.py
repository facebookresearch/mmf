from setuptools import setup

setup(name='pythia',
      version='0.3',
      author='Facebook AI Research',
      license='BSD',
      description="A modular research framework for multimodal vision and "
      "language research.",
      url="https://github.com/facebookresearch/pythia",
      packages=["pythia"],
      install_requires=['torch', 'torchtext', 'numpy',
                        'torchvision', 'demjson', 'tensorboardX'])

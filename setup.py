from setuptools import setup

setup(
    name='lightweight_openpose',
    version='0.0',
    install_requires=[
        'torch>=0.4.1'
        'torchvision>=0.2.1'
        'pycocotools==2.0.2'
        'opencv-python>=3.4.0.14'
        'numpy>=1.14.0'
    ],
    packages=['lightweight_openpose']
)

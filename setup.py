from setuptools import setup, find_packages

setup(name='adaptive_control_gym',
    author="Chaoyi Pan",
    author_email="pcy19@mails.tsinghua.edu.cn",
    packages=find_packages(include="adaptive_control_gym*"),
    version='0.0.0',
    install_requires=[
        'gym', 
        'pandas', 
        'seaborn', 
        'torch',
        'matplotlib', 
        'imageio',
        'wandb', 
        'control', 
        'icecream',
        'torch', 
        'tqdm', 
        'tyro', 
        'meshcat', 
        ]
)
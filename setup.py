from setuptools import setup, find_packages

setup(
    name="aether",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "tiktoken"
    ],
    entry_points={
        'console_scripts': [
            'aether-cli=aether.cli:main',
            'aether-train=aether.trainer:main',
        ],
    },
    python_requires=">=3.8",
)
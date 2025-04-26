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
            'aether-cli=cli:main',
            'aether-train=trainer:main',
        ],
    },
    python_requires=">=3.8",
)
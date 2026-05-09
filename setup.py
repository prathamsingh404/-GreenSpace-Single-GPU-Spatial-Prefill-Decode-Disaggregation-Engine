from setuptools import setup, find_packages

setup(
    name="greenspace",
    version="1.0.0",
    description="Single-GPU Spatial Prefill-Decode Disaggregation Engine via CUDA Green Contexts",
    author="Project GreenSpace Core Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "rich>=13.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)


from setuptools import setup, find_packages

setup(
    name="llamarpc",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "rich>=10.0.0",
    ],
    author="Patrick Peng",
    author_email="p.peng@protosec.ai",
    description="A Python RPC client for LLaMA",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Protosec-Research/LlamaRPC",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 
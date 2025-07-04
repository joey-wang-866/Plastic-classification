## Verified environment: 
JetPack6.2 + Orin nano

# Python Virtual Environment Setup with CUDA

This guide will walk you through setting up a Python virtual environment using `venv` on Linux, and configuring your environment to work with CUDA for GPU acceleration.

## Creating a Virtual Environment

Python's built-in `venv` module allows you to create isolated Python environments for your projects. This helps avoid dependency conflicts between different projects.

### Step 1: Create a Virtual Environment

Navigate to your project directory and create a virtual environment:

```bash
# Navigate to your project directory
cd /path/to/your/project

# Create a virtual environment named 'env'
python3 -m venv my_testing_env
```

### Step 2: Activate the Virtual Environment

Once created, you need to activate the virtual environment:

```bash
source my_testing_env/bin/activate
```

After activation, your terminal prompt should change to indicate that you're now working within the virtual environment. It typically shows the name of the environment in parentheses at the beginning of the prompt.

### Step 3: Install torch-2.5.0 torchvision-0.20.0

Follow the documentation from https://elinux.org/Jetson/L4T/TRT_Customized_Example#YoloV11_using_Pytorch, torch-2.5.0 torchvision-0.20.0 is suitable for this Jetson Orin Nano board.

```bash
wget http://jetson.webredirect.org/jp6/cu126/+f/5cf/9ed17e35cb752/torch-2.5.0-cp310-cp310-linux_aarch64.whl#sha256=5cf9ed17e35cb7523812aeda9e7d6353c437048c5a6df1dc6617650333049092

pip install torch-2.5.0-cp310-cp310-linux_aarch64.whl

wget http://jetson.webredirect.org/jp6/cu126/+f/5f9/67f920de3953f/torchvision-0.20.0-cp310-cp310-linux_aarch64.whl#sha256=5f967f920de3953f2a39d95154b1feffd5ccc06b4589e51540dc070021a9adb9

pip install torchvision-0.20.0-cp310-cp310-linux_aarch64.whl 
```

### Step 4: Install Packages

With the virtual environment activated, you can install packages using pip:

```bash
# Update pip itself if needed
pip install --upgrade pip

# To install packages with pip
pip install package

# If you want to install a specific version
pip install package=Xx.xX



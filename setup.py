from setuptools import setup, find_packages

setup(
    name="labsurv",
    version="0.0.1",
    author="Eric Monlye",
    author_email="536682885@qq.com",
    url='https://github.com/Eicmlye/labsurv',  # project main page
    description="A reinforcement learning solution to the Optimal Camera Placement (OCP) problem",
    long_description=open('README.md').read(),
    long_description_content_type="markdown",
    license="None",
    python_requires=">=3.12",
    install_requires=[
        "numpy>=1.26.4",
        "mmcv>=2.2.0",
        "gym>=0.26.2",
        "torch>=2.3.0",
        "black>=24.4.2",
        "isort>=5.13.2",
    ],
    packages=find_packages(),
    platforms="any",
)

import os
from setuptools import setup, find_packages


def _post_install():
    CACHE_DIR = os.path.expanduser("~/.cache/vision_anything")
    os.makedirs(CACHE_DIR, exist_ok=True)


setup(
    name="vision_anything",
    packages=find_packages(),
    install_requires=[],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3.9",
    description="Vision Anything: SOTA Vision Models",
    author="Jihoon Oh",
    url="https://github.com/ojh6404/vision_anything",
    author_email="ojh6404@gmail.com",
    version="0.0.1",
)

_post_install()

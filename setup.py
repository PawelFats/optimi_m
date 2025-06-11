# setup.py
from setuptools import setup, find_packages

setup(
    name="optimi_m",
    version="0.1.0",
    # говорим: искать пакеты начиная с корня проекта
    package_dir={"": "."},
    # включаем только пакеты, имя которых начинается на "src"
    packages=find_packages(where=".", include=["src", "src.*"]),
)

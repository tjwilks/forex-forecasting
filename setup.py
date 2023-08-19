from setuptools import find_packages, setup

with open('requirements.txt') as fp:
    install_requires = fp.read().splitlines()

setup(
    name='forex-forecasting',
    packages=find_packages(),
    version='0.0.1',
    description='for forecasting forex',
    author='Toby Wilkinson',
    license='',
    install_requires=install_requires
)

from setuptools import setup, find_packages

setup(
    name='wepy',
    version='0.1',
    py_modules=['wepy'],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'h5py',
        'networkx',
        'pandas'
    ],
)

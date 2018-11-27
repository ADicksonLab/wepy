from setuptools import setup, find_packages

setup(
    name='wepy',
    version='0.10.2',
    py_modules=['wepy'],
    packages=find_packages(),
    include_package_data=True,
    entry_points={'console_scripts' : ['wepy=wepy.orchestration.cli:cli']},
    install_requires=[
        'numpy',
        'h5py',
        'networkx>=2',
        'pandas',
        'dill',
        'click',
        'mdtraj',
        'scipy',
        'geomm',
        'matplotlib'
    ],
)

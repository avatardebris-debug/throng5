from setuptools import setup, find_packages

setup(
    name='throng3',
    version='0.1.0',
    description='Meta^N Recursive Self-Optimization Architecture',
    author='Edgar Lab',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
    ],
)

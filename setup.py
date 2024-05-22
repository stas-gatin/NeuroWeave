from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'A small machine learning library.'
LONG_DESCRIPTION = ('A Machine Learning library with the basic components necessary for building basic '
                    'Neural Network Models')

setup(
    name='weave',
    version=VERSION,
    author='Gabriel Niculescu, Carlos Molera, Stanislav Gatin, Patricia Pérez, Hugo Urbán',
    author_email='gabriellichu@gmail.com, info.prostaprog@gmail.com, carlosmoleracanals@gmail.com,'
                 'pperezferre@gmail.com, hugourmaz@gmail.com',
    description=DESCRIPTION,
    long_description_content_type='text/markdown',
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['h5py>=3.11.0', 'numpy>=1.26.4', 'pandas>=2.2.2', 'fastrlock>=0.8.2',
                      'python-dateutil>=2.9.0.post0', 'pytz>=2024.1', 'six>=1.16.0', 'tzdata>=2024.1'],
    extras_require={'GPU': ['cupy-cuda12.x==13.1.0']},
    python_requires='>=3.12',
    keywords=['Python', 'Machine Learning'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: GPU :: NVIDIA CUDA :: 12 :: 12.4',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Operating System :: Microsoft :: Windows',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering',
    ],
)

from setuptools import setup, find_packages

setup(
    name='iva_mt',
    version='0.2.0',
    author='cartesinus',
    author_email='msowansk@gmail.com',
    description=('A machine translation library utilizing m2m100 models, '
                 'equipped with features for generating diverse verb variants via VerbNet '
                 'and Conditional Beam Search to enrich Virtual Assistant training sets.'),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/cartesinus/iva_mt',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'transformers==4.28.1',
        'sentencepiece==0.1.99',
        'datasets',
        'peft'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',
)

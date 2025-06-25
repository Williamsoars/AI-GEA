from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='ai-gea',
    version='0.1.0',
    author='William Silva',
    author_email='Williamkauasoaresdasilva@gmail.com',
    description='AI-GEA: Artificial Intelligence for Graph Embedding Analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Williamsoars/AI-GEA',
    packages=find_packages(),
    install_requires=[
        'networkx>=2.5',
        'numpy>=1.19.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.3.0',
        'gensim>=4.0.0',
        'node2vec>=0.4.0',
        'tqdm>=4.0.0',
    ],
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='graph embedding machine-learning recommendation ai',
    project_urls={
        'Documentation': 'https://github.com/Williamsoars/AI-GEA/docs',
        'Source': 'https://github.com/Williamsoars/AI-GEA',
        'Tracker': 'https://github.com/Williamsoars/AI-GEA/issues',
    },
)

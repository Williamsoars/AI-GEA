from setuptools import setup, find_packages

setup(
    name='graph_embedding_recommender',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'networkx>=2.5',
        'numpy>=1.19.0',
        'pandas>=1.1.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.3.0',
        'gensim>=4.0.0',
        'node2vec>=0.4.0',
    ],
    python_requires='>=3.7',
    author='William Silva',
    author_email='Williamkauasoaresdasilva@gmail.com',
    description='Sistema de recomendação de embeddings para grafos com IA',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/seuusuario/seurepositorio',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],

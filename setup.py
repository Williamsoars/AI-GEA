from setuptools import setup, find_packages

setup(
    name='graph_embedding_recommender',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'networkx',
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
    ],
    author='Seu Nome',
    author_email='seu.email@exemplo.com',
    description='Sistema de recomendação de embeddings para grafos com IA',
    url='https://github.com/seuusuario/seurepositorio',
)

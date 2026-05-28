from setuptools import find_namespace_packages, setup

setup(
    name='qdiff',
    packages=find_namespace_packages(include=['qdiff*']),
    package_data={
        'qdiff': ['quarot/hadamard_utils/*.txt', 'quarot/hadamard_utils/*.pth'],
    },
    install_requires=[
        # Add any external dependencies here
    ],
)

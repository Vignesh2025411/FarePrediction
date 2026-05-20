from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'

def get_requirement(filepath:str)->List[str]:
    # this functoon return the list of requirements

    requirements=[]
    with open(filepath) as file:
        requirements=file.readlines()
        requirements=[i.replace("\n","") for i in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements






setup(
    name='FarePrediction',
    version='0.0.1',
    author='Wikki',
    author_email='vignesh893918@gmail.com',
    packages=find_packages(),
    install_requires=get_requirement('requirements.txt')
)
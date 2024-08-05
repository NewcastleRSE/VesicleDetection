from setuptools import find_packages, setup 
from typing import List

# This implimentation is based on one written by Krish Naik, and this comment is to give credit.

HYPTHON_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    Read the requirements file and return a list of requirements for installation. 
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPTHON_E_DOT in requirements:
            requirements.remove(HYPTHON_E_DOT)
    
    return requirements

setup(
    name = 'VesicleDetection', 
    version = '0.0.1', 
    author = 'Richie Dadhley', 
    author_email = 'richie.dadhley@newcastle.ac.uk', 
    packages = find_packages(), 
    install_requires = get_requirements('requirements.txt')
)
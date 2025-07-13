from setuptools import setup, find_packages

def get_requirements(file_name):
    """
    This function will return a list of requirements
    """
    with open(file_name) as file:
        requirements = file.readlines()
    
    # Remove any whitespace characters like `\n` at the end of each line
    requirements = [req.strip() for req in requirements if req.strip()]

    if '-e .' in requirements:
        requirements.remove('-e .')
    
    return requirements

setup(
    name='ml_project',
    version='0.0.1',
    author='Narendra_Reddy',
    author_email="narendrareddymolakala@gmail.com", 
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
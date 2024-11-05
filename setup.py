import os
import re
import setuptools

ROOT = os.path.dirname(__file__)
VERSION_RE = re.compile(r"__version__\s*=\s*['\"]([\w\.-]+)['\"]")

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as fh:
   requirements = fh.readlines()
   requirements = [requirement.strip().replace('\n','').replace('\r','') for requirement in requirements]
   requirements = [requirement for requirement in requirements if len(requirement) != 0 and requirement[0] != '#']

def get_version():
    init = open(os.path.join(ROOT, 'doe_xstock', '__init__.py')).read()
    return VERSION_RE.search(init).group(1)

setuptools.setup(
    name='doe_xstock',
    version=get_version(),
    author='Kingsley Nweye',
    author_email='etonwana@yahoo.com',
    description='Manage DOE\'s End Use Load Profiles for the U.S. Building Stock.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/intelligent-environments-lab/doe_xstock',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    entry_points={'console_scripts': ['doe_xstock = doe_xstock.main:main']},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7.7',
)
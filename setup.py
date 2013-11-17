from distutils.core import setup

setup(
    name='pyGP_PR',
    version='1.0.0',
    author='Marion Neumann', 'Daniel Marthaler', 'Shan Huang', 'Kristian Kersting'
    author_email='marion.neumann@uni-bonn.de.com', 'marthaler@ge.com', 'shan.huang@iais.fraunhofer.de', 'kristian.kersting@cs.tu-dortmund.de'
    packages=['bin','data','src','src.Core','src.bin','src.Tools'],
    scripts=['bin/demoRegression.py','bin/demoClassification.py','bin/demoMaunaLoa.py','bin/demoHousing.py'],
    url='https://github.com/marionmari/pyGP_PR',
    license='LICENSE.txt',
    description='Functional Gaussian Processes',
    long_description=open('README.txt').read(),
    install_requires=[
        "Python >= 2.6",
        "Numpy >= 1.7.1",
        "Scipy >= 0.12.0",
        "matplotlib >= 1.2.1",
    ],
)

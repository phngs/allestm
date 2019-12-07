from setuptools import setup, find_packages

VERSION = '1.2.0'

with open('README.md', 'r') as fh:
    long_description = fh.read()


setup(name='allestm',
      version=VERSION,
      description='Predicting various structural features of transmembrane proteins.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      python_requires=">=3.6.0",
      url='http://github.com/phngs/allestm',
      author='Dr. Peter Hönigschmid',
      author_email='hoenigschmid.peter@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
            'pandas',
            'numpy',
            'keras',
            'tensorflow',
            'scikit-learn',
            'xgboost',
            'requests'
      ],
      entry_points={
            'console_scripts': ['allestm=allestm.allestm:main']
      },
      zip_safe=False,
      include_package_data=True
)

from setuptools import setup, find_packages

setup(name='allestm',
      version='1.0',
      description='Predicting various structural features of transmembrane proteins.',
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
      zip_safe=False)

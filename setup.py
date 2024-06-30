from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'T-Sana\'s package'
LONG_DESCRIPTION = 'A package to create images, manage files and more.'

setup(
        name="t_sana_s_package", 
        version=VERSION,
        author="T-Sana",
        author_email="tsana.code@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=["numpy", "opencv-python", "sty"],
        keywords=['python', 'image'],
        classifiers= [
            "Programming Language :: Python :: 3",
            "Operating System :: Linux :: Fedora",
            "Operating System :: Microsoft :: Windows",
        ]
)
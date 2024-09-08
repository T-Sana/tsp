from setuptools import setup, find_packages

VERSION = '0.0.21' 
DESCRIPTION = 'T-Sana\'s package'
LONG_DESCRIPTION = 'A package to create images, manage files and more.'

setup(
        name="tsanap", 
        version=VERSION,
        author="T-Sana",
        author_email="tsana.code@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=["numpy", "opencv-python", "sty", "multimethod"],
        keywords=['python', 'image'],
        classifiers= [
            "Programming Language :: Python :: 3",
            "Operating System :: Linux :: Fedora",
            "Operating System :: Microsoft :: Windows",
        ]
)
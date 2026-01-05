from setuptools import setup,find_packages

def get_packages(file_path):
    with open(file_path , 'r') as file:
        packages_lst = list(file.readlines())
        packages = [i.strip() for i in packages_lst]

        if "-e ." in packages:
            packages.remove("-e .")
        return packages
        


setup(
    name = 'House_Price_App',
    version='0.0.1',
    description="Predicting house price",
    author="Kay Sap",
    author_email="tboy4198@gmail.com",
    packages=find_packages(),
    install_requires=get_packages('requirements.txt')
)
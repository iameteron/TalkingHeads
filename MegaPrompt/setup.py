from setuptools import find_packages, setup


setup(
    name="megaprompt",
    version="0.1.0",
    description="Prompt rendering toolkit for CrafText MegaPrompts",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pyyaml",
    ],
    python_requires=">=3.9",
)

from setuptools import setup, find_packages

setup(
    name="fan-predictive-maintenance",
    version="1.0.0",
    description="End-to-end Edge AI predictive maintenance system for industrial fans",
    author="Ayush Kumar Pallai",
    author_email="ayushkumarpallai020506@gmail.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "shap>=0.42.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "joblib>=1.3.0",
        "pyyaml>=6.0",
        "pytest>=7.4.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Monitoring",
    ],
)

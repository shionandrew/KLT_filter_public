from setuptools import setup, find_packages

setup(
    name="klt_filter",
    version="0.1.0",
    author="Shion Andrew",
    author_email="shionandrew@gmail.com",
    description="KL-based spatial filter for interferometric beamforming",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    url="https://github.com/shionandrew/KLT_filter_public",
    install_requires=[
        "numpy>=1.19.2",
        "scipy>=1.5.2",
        "h5py",
    ],
    extras_require={
        "chime": [
            "ch_util",
            "baseband_analysis",
            "beam_model",
            "caput",
        ],
    },
    python_requires=">=3.8",
)

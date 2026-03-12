from setuptools import setup
import os
from glob import glob

package_name = "path_smoother"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"),
            glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="your@email.com",
    description="Path smoothing and trajectory tracking for differential drive robots",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "path_smoother_node = path_smoother.path_smoother_node:main",
            "trajectory_generator_node = path_smoother.trajectory_generator_node:main",
            "trajectory_tracker_node = path_smoother.trajectory_tracker_node:main",
        ],
    },
)

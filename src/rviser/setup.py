from setuptools import setup
import os

package_name = 'rviser'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    description='rviser teleop visualizer',
    license='TODO',
    entry_points={
        'console_scripts': [
            'safe_teleop_node = rviser.safe_teleop_node:main'
        ],
    },
)
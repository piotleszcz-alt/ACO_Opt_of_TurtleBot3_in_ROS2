from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'aco_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='piotr_kaskow',
    maintainer_email='piotr_kaskow@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [ 
            'aco_node = aco_planner.aco_node:main',
            'pure_pursuit = aco_planner.pure_pursuit:main'
        ],
    },
)

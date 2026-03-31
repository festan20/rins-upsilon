from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'upsilon'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'map'), glob('map/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='stefan',
    maintainer_email='todo@todo.com',
    description='Task 1: face and ring detection with Nav2 exploration',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'face_detector = upsilon.face_detector:main',
            'ring_detector = upsilon.ring_detector:main',
            'controller = upsilon.controller:main',
            'speech = upsilon.speech:main',
        ],
    },
)

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
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'map'), glob('map/*')),
        (os.path.join('share', package_name, 'sounds'), glob('sounds/*')),
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
            'face_detector_task2 = upsilon.face_detector_task2:main',
            'ring_detector = upsilon.ring_detector:main',
            'ring_detector2 = upsilon.ring_detector2:main',
            'ring_detector_task2 = upsilon.ring_detector_task2:main',
            'cylinder_detector_task2 = upsilon.cylinder_detector_task2:main',
            'blue_line_detector = upsilon.blue_line_detector:main',
            'blue_line_follower = upsilon.blue_line_follower:main',
            'controller = upsilon.controller:main',
            'anomaly_controller = upsilon.anomaly_controller:main',
            'speech = upsilon.speech:main',
            'visualizer = upsilon.visualizer:main',
            'camera_viewer = upsilon.camera_viewer:main',
            'wasd_teleop = upsilon.wasd_teleop:main',
            'tile_detection = upsilon.tile_detection:main',
            'anomaly_detector = upsilon.anomaly_detector:main',
        ],
    },
)

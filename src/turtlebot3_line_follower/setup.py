from setuptools import find_packages, setup

package_name = 'turtlebot3_line_follower'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dfiumicelli',
    maintainer_email='dfiumicelli@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'single_line_follower = turtlebot3_line_follower.single_line_follower:main',
            'lane_keeping_open_cv = turtlebot3_line_follower.lane_keeping_open_cv:main',
        ],
    },
)

"""
Python packaging for hsa_ros2.

``data_files`` copies package.xml, launch, rviz and config into the install share tree, where ``ros2 launch``
and ``$(find-pkg-share hsa_ros2)`` look: a file not listed here is invisible after ``colcon build``.
``entry_points`` makes each ``name = module:function`` runnable as ``ros2 run hsa_ros2 <name>``.
"""

from pathlib import Path

from setuptools import find_packages, setup

package_name = 'hsa_ros2'


def files_under(subdir: str) -> list[tuple[str, list[str]]]:
    """List (install dir, [files]) for everything under ``subdir`` so it lands in the share tree."""
    return [
        (str(Path('share') / package_name / d), [str(f) for f in d.glob('*') if f.is_file()])
        for d in Path(subdir).glob('**')
    ]


setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        *files_under('launch'),
        *files_under('config'),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Amber Handal',
    maintainer_email='amberhandal2026@u.northwestern.edu',
    description='HSA wrist demo collection on the Franka FER (SpaceMouse teleop + recorder)',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'spacemouse_teleop = hsa_ros2.spacemouse_teleop:main',
            'demo_recorder = hsa_ros2.demo_recorder:main',
            'fake_camera = hsa_ros2.fake_camera:main',
            'export_dataset = hsa_ros2.export_dataset:main',
            'diag_stats = hsa_ros2.diag_stats:main',
        ],
    },
)

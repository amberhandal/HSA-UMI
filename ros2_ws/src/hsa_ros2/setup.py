"""
Python packaging for hsa_ros2.

* ``data_files`` copies package.xml, the launch files and the config files into the *install*
  tree (``install/hsa_ros2/share/hsa_ros2/...``).  That is where ``ros2 launch`` and
  ``$(find-pkg-share hsa_ros2)`` look, so a launch or yaml file you forget to list here is
  simply invisible after ``colcon build``.
* ``entry_points`` turns a Python function into an executable that ``ros2 run hsa_ros2 <name>``
  can start.  ``spacemouse_teleop = hsa_ros2.spacemouse_teleop:main`` means "the command
  spacemouse_teleop runs main() in hsa_ros2/spacemouse_teleop.py".
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
        ],
    },
)

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Dependency Parser',
    'author': 'Rohit Jain(rj288)',
    'url': 'URL to get it at.',
    'download_url': 'Where to download it.',
    'author_email': 'rj288@cornell.edu',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['parser'],
    'scripts': [],
    'name': 'dp'
}

setup(**config)

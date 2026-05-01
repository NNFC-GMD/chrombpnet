from pathlib import Path
from setuptools import setup, find_packages


def read_requirements(path):
    """Read pip-style requirements while ignoring comments and blank lines."""
    return [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


install_requires = read_requirements("requirements.txt")


config = {
    'name': 'chrombpnet',
    'author_email': 'anusri @ stanford.edu',
    'license': 'MIT',
    'license_files': ('LICENSE',),
    'include_package_data': True,
    'description': 'chrombpnet predicts chromatin accessibility from sequence',
    'download_url': 'https://github.com/kundajelab/chrombpnet',
    'version': '1.0.1',
    'packages': find_packages(),
    'python_requires': '>=3.10',
    'install_requires': install_requires,
    'zip_safe': False,
    'scripts':[
               'chrombpnet/training/models/bpnet_model.py',
               'chrombpnet/training/models/chrombpnet_with_bias_model.py'
    ],
    'entry_points': {'console_scripts': [
        'chrombpnet = chrombpnet.CHROMBPNET:main',
        'print_meme_motif_file = chrombpnet.data.__init__:print_meme_motif_file']}
}

if __name__== '__main__':
    setup(**config)

from setuptools import setup

setup(
	include_package_data=True,
	install_requires=[
		'pandas~=1.3.5',
		'numpy~=1.21.6',
		'tensorflow~=2.9.3',
		'tensorflow-probability~=0.19.0',
		'keras~=2.9.0',
		'keras-tuner~=1.2.0',
		'filelock~=3.7.1',
	],
	extra_require={
		'pc' : [ 'azure-cli', 'azureml-sdk' ]
	}
)
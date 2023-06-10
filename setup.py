from setuptools import setup

setup(
	include_package_data=True,
	install_requires=[
		'pandas>=1.3.5',
		'numpy>=1.21.6',
		'tensorflow>=2.8.0',
		'tensorflow-probability>=0.16.0',
		'keras>=2.8.0',
		'keras-tuner>=1.2.0',
	],
	extra_require={
		'pc' : [ 'azure-cli', 'azureml-sdk' ]
	}
)
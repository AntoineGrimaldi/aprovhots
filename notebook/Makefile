default: A

J=jupyter nbconvert  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=0 --allow-errors --execute
JN=$(J) --to notebook --inplace

A:
	$(JN) 2022-01-18_APROVIS_meeting.ipynb
	git commit -m'results : meeting ' 2022-01-18_APROVIS_meeting.ipynb
	git push



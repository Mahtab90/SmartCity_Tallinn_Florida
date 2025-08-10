.PHONY: etl train all

etl:
	python traffic_harmonization_pipeline.py

train:
	python modeling/train_eval_pipeline.py

all: etl train
	@echo 'Done.'

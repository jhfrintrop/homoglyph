.PHONY: all
all:


.PHONY: clean
clean:
	find . -name '*.py[cod]' -type f -delete
	find . -path '*/__pycache__/*' -delete
	find . -name __pycache__ -type d -delete
	find . -path '*/.mypy_cache/*' -delete
	find . -name .mypy_cache -type d -delete
	find . -path '*/.pytest_cache/*' -delete
	find . -name .pytest_cache -type d -delete
	find . -name .coverage -type f -delete
	rm -rf *.egg-info


.PHONY: compile
compile:
	python -m compileall homoglyph_cnn/


.PHONY: print-%
print-%:
	@echo $*=$($*)

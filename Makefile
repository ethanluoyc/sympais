.PHONY: lint
lint:
	pylint --rcfile=.pylintrc src

.PHONY: test
test:
	pytest .

.PHONY: format
format:
	yapf --style=.style.yapf -ir src examples

.PHONY: typecheck
typecheck:
	pytype src

.PHONY: isort
isort:
	isort src examples

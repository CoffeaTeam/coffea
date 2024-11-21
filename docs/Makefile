html: source/modules
	pushd .. && sed -i.bak s/docs\\\/source\\\/// README.rst && popd
	sphinx-build -M html source build
	pushd .. && mv README.rst.bak README.rst && popd

source/modules: source/reference.rst
	sphinx-autogen source/reference.rst -t source/_templates -o source/modules

clean:
	rm -rf build source/modules source/api

.PHONY: clean

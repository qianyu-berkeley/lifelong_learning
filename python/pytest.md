# Pytest

## Introduction

* Common python testing framework
  * uniitest
  * nose
  * behave
  * Robot framework
  * Pytest
* Why **pytest**?
  * powerful test serach
  * concurrency support
  * code re-use
  * great plugins

## Pytest Framework

* Use pytest.init to define how pytest find test code, functions, and classes
* Marker allow us to designate test search
  * To run mixed tests
  
    ```bash
    pytest -m "body and engine" # run test marked with both and engine
    pytest -m "body or engine" # run test marked with both or engine
    pytest -m "not entertainment" # run test marked with not marked entertainment
    ```

  * `-s` will always show print regardless if the test is passed or not
  * If we mark a class, all function inside the class will be marked
  
    ```python
    from pytest import mark

    @mark.body
    class BodyTests:

      @mark.door
      def test_body_functions(self):
        assert True
      
      def test_bumper(self):
        assert True

      def test_windshield(self):
        assert True

* Pytest `Fixture`
  * Create a `conftest.py` to create fixture and it is assessable by any file at the same level or below (no import needed)
  * fixture as a scope
    * `function`: 1 per function
    * `session`: 1 per entire test session
  
* Pytest `Reporting`

  * `pytest-html` package can general html page of test results locally

    ```bash
    pip install pytest-html
    pytest --html="result.html"
    ```

  * `pytest --junitxml="results.xml` generate xml file can be integrated with Jenkins post build action `publish a junitxml report`

    ```bash
    pytest --junitxml="BUILD_$(BUILD_NUMBER)_result.xml"
    ```

## Customize Test Runs

* [Example](https://github.com/brandonblair/elegantframeworks/tree/config_recipe)
* `parser` object to enable new input arg options
* `request` object to get input from pytest input arg options defined by parser

## Handel Skips and expected failures

* Ref: https://docs.pytest.org/en/latest/how-to/parametrize.html
* `skip` is another marker
* `xfail` is another marker

## Parameterization

Allow 1 test function to run test for each parameter in the parameters definition

* Ref: 
  * https://docs.pytest.org/en/latest/how-to/parametrize.html
  * [Example](https://github.com/brandonblair/elegantframeworks/tree/parametrize)
* Approach 1: parameterize on the fly no recommended
* Approach 2: use fixture params
* Approach 3: use data driven params such as read from a JSON file

  ```python
  # test.py
  from pytest import mark

  # approach #1
  @mark.parametrize('premier_league_club', [
      ('Arsenal'),
      ('Liverpool'),
      ('Manchester_City'),
    ]
  )
  def test_premmier_league(premier_league_club):
    print(f"{premier_league_club} is playing")

  # approach #2
  def test_browser_can_navigate_to_training_ground(browser):
    browser.get('http://techstepacademy.com/training-ground')

  # approach #3
  def test_television_turns_on_from_fixture(tv_brand_from_fixture):
    print(f"{tv_brand_from_fixture} turns on as expected")
  
  # conftest.py
  import json
  from pytest import fixture
  from selenium import webdriver
  
  data_path = 'test_data.json'

  def load_test_data(path):
      with open(path) as data_file:
          data = json.load(data_file)
          return data

  @fixture(params=[webdriver.Chrome, webdriver.Firefox, webdriver.Edge])
  def browser(request):
      driver = request.param
      drvr = driver()
      yield drvr
      drvr.quit()

  @fixture(params=load_test_data(data_path))
  def tv_brand_from_fixture(request):
      data = request.param
      return data
  
  # test_data.json
  [
      "Sony",
      "Samsung",
      "Vizio"
  ]
  ```

## Speed Test Time

* Use pytest extension `pip install pytest-xdist`
* Run pytest with `pytest -s -v -n4`
  * `-n4` means run with 4 threads
* If the tests are isolated, we can choose number of threads based on the number of test
  * `-nauto` can auto detect based on the number of processes
* If the tests are not isolated, we cannot use those capability


## Test methodology

### Unit Test

* [Example](https://github.com/BrandonBlair/elegantframeworks/tree/unittesting1)
* leverage `pip` local project installs
  * `pip install my_path_to_project` install local projects by giving path, this will install projects into pip is associated with (`site_package`)
  * `pip install -e my_path_to_project` editable install allow you to install project without copy any files. Instead, the files in the development directory are added to Python’s import path. This approach is well suited for development and is also known as a “development installation”.
    * With an editable install, you only need to perform a re-installation if you change the project metadata (eg: version, what scripts need to be generated etc). You will still need to run build commands when you need to perform a compilation for non-Python code in the project (eg: C extensions).

### How to use [Tox](https://tox.wiki/en/latest/)

* Why use tox?
  * For unit testing
  * Create seperate between actual library from the test, no need to include unnecessary python package (e.g. pytest, coverage etc)
  * Support for test multiple python environment (2.7, 3.8 etc)
* How to use tox?
  * install use pip
  * create tox.ini

    ```ini
    [tox]
    envlist = py36
    
    [testenv]
    deps = pytest  # only the libraries that is needed for test
    commands = 
      pytest
    
    [pytest]
    python_files = test_*
    python_functions = test_*
    python_classes = *Test
    testpaths = tests
    ```

### Functional/Integration test

* [Example](https://github.com/brandonblair/elegantframeworks/tree/functionaltests)
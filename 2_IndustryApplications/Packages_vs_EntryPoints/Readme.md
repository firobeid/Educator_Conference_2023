# Instructions:

I have created three separate examples that illustrate python entry points, packging and modularization in simple examples.

* Example0:
    Simple python script to run with encapsulated entry point.
* Example1:
    Another modularized method of containing an entry point in python package skeleton style
    ```python
    Example1
    |    
    │    entry_point.sh
    │    steps.md
    │
    └───entry_point
        │   day_extractor.py
        │   __init__.py
        │   __main__.py     
    ```
* Example2:
    Add a console script to your python package, create that package & add the package to your python enviroment.

    ```python
    Example2
    │   steps.md
    │
    └───root
        │   pyproject.toml
        │
        ├───dist
        │       Firo-1.0.0-py3-none-any.whl
        │       Firo-1.0.0.tar.gz
        │
        ├───entry_point
        │     day_extractor.py
        │     steps.md
        │     __init__.py
        │   
        │
        └───Firo.egg-info
                dependency_links.txt
                entry_points.txt
                PKG-INFO
                SOURCES.txt
                top_level.txt
    ```


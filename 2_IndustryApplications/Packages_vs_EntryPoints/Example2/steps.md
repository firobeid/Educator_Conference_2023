# Instructions

1. Create you main module (i.e `day_extractor.py`)
2. Creat next to your main package folder the following file `__init__.py`
    * `__init__.py`: Acts as an initialization step for marking directories on disk as python package folders(directories)
    If you have such a directory tree:
        ```python
        mydir/foo/__init__.py
        mydir/foo/module.py
        ```
    You can import the submodule as follows:
    ```python
    import foo.module
        or
    from foo import module
    ```
    ***Note***: The __init__.py file is usually empty, but can be used to export selected portions of the package under more convenient names, hold convenience functions, etc. Given the example above, the contents of the __init__ module can be accessed as: `import foo`
3. Add all the above scripts too a folder and call it whatever you want (i.e `entry_point` in my case)
4. Put the package folder(directory) in another folder and call whatever you want (i.e I called it `root`). This will encapsulate your package.
5. Create explicitly a file called `myproject.toml` next to your package `whatever you called your package` (i.e `entry_point` in my case)
6. Add the following to the snippet to add to your `myproject.toml` file:
    ```
    [project]
    name = 'Firo' --->What I called it
    requires-python = ">=3.7"
    version = "1.0.0"
    authors = [
    {name = "Firas A Obeid", email = "firas@notgonatellyou.com"},
    ]

    [project.scripts]
    day-extractor = "entry_point.day_extractor:diff_date" ----> function-name = package_folder_name.module.py_name:function_name
    ```
    This will add your console script "actual entry point to the code" to `pyproject.toml`:
    Reffer too [Toml Instructions](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) for further instructions on .toml files

7. Build your package:
    * Activate another enviroment, optional, be safer (`conda activate xxxx`) I would honestly create a new one with the python version I specified in the `pyproject.toml` file (`conda create new_env python==3.8`)
    * Run `conda activate new_env`
    * Make sure your in the `root` (project folder) and both the `myproject.toml` and packgae folder are in the same directory.
    * Run `pip install build` to install the build tool
    * Run `python -m build` to build your package
8. Install your package: 
    * Inside my `myproject.toml` I called my package`Firo` (name = 'Firo' )
    * Run `pip install dist/Firo-1.0.0-py3-none-any.whl` (`Firo` is based on your package name)
9. Now you are all set! Run `day-extractor` and the package will execute. `day-extractor` is whatever you called `function-name` under the `[project.scripts]` in the `pyproject.toml. 


# Instructions

1. Create you main module (i.e `day_extractor.py`)
2. Creat next to your main module a `__main__.py` & `__init__.py`
    * `__main__.py`: Acts as a main entry point to the local package `entry_point`
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
4. Import in the `__main__.py` your module by adding a . in the prefix (i.e `from .day_extractor import whatever_executable_function`)
5. Call in the `__main__.py` the function you imported in step 3
6. In gitbash or mac terminal, create a file called `whatever_you_want.sh` next to the `entry_point` folder and add to it the following:
    ``` python
    #!/bin/sh
    python -m entry_point
    ```
6. Simply type(run) in the terminal .`./whatever_you_want.sh` and that rund the package you created
7. If step 6 dosent work, please run this before `chmod +x whatever_you_want.sh` once and step 6 can be repeated as much as you want afterwards

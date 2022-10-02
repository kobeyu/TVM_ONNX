Steps to reproduce

1. Download TVM
    * Steps: https://tvm.apache.org/docs/install/from_source.html
    * After build TVM, we should set the environment variable in .bashrc (also in above document)
        * ```export TVM_HOME=/path/to/tvm```
        * ```export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}```  

2. Python environment
    * I use python 3.8.10 (I think other version would still work)
    * Use the requirements0923.txt to get pip package (pip install -r requirements0923.txt)

3. Directly run 
    ```
    python3 mobilenet_quent.py
    ```
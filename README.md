ADMM-Lasso
----------

Python code to solve the Lasso problem using ADMM. 

Serial and ditributed version of ADMM Lasso with Open MPI. 

Original code (written in Matlab and C++) can be found at http://www.stanford.edu/~boyd.

To run admm_lasso_MPI.py type in the terminal:

```
mpirun -np number_of_processes python lasso_admm_MPI.py -i /path/to/input_file.h5 -o /path/to/output_file.h5
```

To see a complete list of arguments type in the terminal:
```
python lasso_admm_MPI.py --help
```

The code to generate sparse matrices of normally distributed random numbers was taken from http://scipy-user.10969.n7.nabble.com/Random-sparse-matrices-td4788.html 


The ouptut of test_lasso_admm.py should look like this:

![alt tag](https://github.com/afbujan/admm_lasso/edit/master/figure1.png)

![alt tag](https://github.com/afbujan/admm_lasso/edit/master/figure2.png)

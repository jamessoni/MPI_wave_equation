## MPI Wave Equation Assignment

This assignment aims to produce a parallel solver implementation with use of MPI, for solving the wave equation.

Analysis of this solver expresses the resulting success to accrue benefits in elapsed run-time as an increase of processors occur. This is analysed and further explained in the report showing a near ideal speed-up.

The solver is ran on the DUG High Performance Clusters, yielding elapsed run-times on grid sizes of 100x100, 250x250 and 500x500 showing expected results with increasing domains. Behaviour of run-time implemented with use of cores from a single node of 2 up to 64 and two nodes amounting to 124 cores, were evaluated.

Further, the implementation makes use of block domain decomposition, with peer-to-peer message passing.

An example post-processing gif is shown below.

<a href="#"><img src="https://github.com/jamessoni/MPI_wave_equation/blob/main/animations/periodic_wave.gif" width="600"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

#### Using
For use of the program will depend on your OS and IDE/compiler of choice:
Visual Studio Community IDE users:
* Open a Visual Studio Project, add all the cloned files to the source files list
* Build project

Terminal:
* Enter directory where the executable exists
* Type:
 
        mpiexec -n 4 <project_name>.exe

(Note: make sure to delete previously created .txt files before running)

Parameters can be changed, these include:
* Boundary Conditions: periodic, dirichlet, neumann
* imax, jmax (exists in main)
* dt_out (determines frames per sec output)
* etc.

For in depth analysis, including a discussion on speed-up ratio, parallel efficiency and core number choice, refer to the report.

Hope you enjoyed the read.

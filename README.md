================================================================================
    Marion Neumann [marion dot neumann at uni-bonn dot de]
    Daniel Marthaler [marthaler at ge dot com]
    Shan Huang [shan dot huang at iais dot fraunhofer dot de]
    Kristian Kersting [kristian dot kersting at cs dot tu-dortmund dot de]

    This file is part of pyGP_FN.
    The software package is released under the BSD 2-Clause (FreeBSD) License.

    Copyright (c) by
    Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting, 30/09/2013
================================================================================

pyGP_FN is a library containing python code for Gaussian Process (GP) Regression and Classification.

pyGP_FN is a procedural implementation of Gaussian Processes.

pyGP_FN follows the structure and (a subset of) functionalities of the gpml matlab implementaion by Carl Edward Rasmussen and Hannes Nickisch (Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2013-01-21). 

This is a stable release. If you observe problems or bugs, please let us know.
NOTE: 	This implementation will be maintained as to bug fixes and corrections of problems in the execution of current functionality.  
	Future extensions will be desigend for the pyGP_OO version only which is an object-oriented implementation of GP functionality.     

Further it includes implementations of
- minimize.py implemented in python by Roland Memisevic 2008, following minimize.m (Copyright (c) Carl Edward Rasmussen (1999-2006))
- scg.py (Copyright (c) Ian T Nabney (1996-2001))
- brentmin.py (Copyright (c) Hannes Nickisch 2010-01-10)
- Mauna Loa CO2 data (Copyright (c) Pieter Tans, Aug 2012)
- FITC functionality (following matlab implementations under Copyright (c) by Ed Snelson, Carl Edward Rasmussen and Hannes Nickisch, 2011-11-02)

installing pyGP_FN
------------------
Download the archive and extract it to any local directory.
Add the local directory to your PYTHONPATH:
	export PYTHONPATH=$PYTHONPATH:/path/to/local/directory/../pyGP_FN/src/

requirements
------------------
- python 2.6 or 2.7
- scipy, numpy, and matplotlib: open-source packages for scientific computing using the Python programming language. 


acknowledgements
------------------
The following persons helped to improve this software: Roman Garnett, Maciej Kurek, Hannes Nickisch, and Zhao Xu.

**********
Use Case 1
**********

.. toctree::


Welcome to the installation guide for Use Case 1 of the Risk-based Adaptive
Monitoring Plan (RAMP) software. This software is developed as part
of the National Risk Assessment Program (NRAP), an initiative by the
Department of Energy. Its primary function is to aid in the planning and
optimization of monitoring strategies for geologic carbon storage sites,
minimizing risks and costs associated with geologic carbon storage.
By considering potential CO2 leakage scenarios, RAMP provides a comprehensive
plan by determining ideal locations and timings for seismic surveys.

1. Download a copy of RAMP using one of the following methods

    * In the command line, enter::

        git clone https://gitlab.com/NRAP/RAMP.git RAMP

    * Or login to GitLab using your credentials and go to https://gitlab.com/NRAP/RAMP,
      and click on the download button shown below.

.. image:: images/uc1/ramp_download.png
   :align: center
   :alt: RAMP GitLab repository

2. Install Python, dependencies, and your preferred IDE (Integrated Development Environment)

    * In most Linux distributions, you can install the necessary dependencies
      using the following command line instructions::

        sudo apt update
        sudo apt upgrade
        sudo apt install python3 python3-pip
        pip3 install -r /RAMP/examples/scripts/requirements.txt

    * On Mac, install python and IDE using any of the following instructions

        https://www.python.org/downloads/macos/
        https://conda.io/projects/conda/en/latest/user-guide/install/macos.html
        https://docs.spyder-ide.org/3/installation.html

      Then use the following command to install dependencies::

        conda install --file RAMP/examples/scripts/requirements.txt


    * On Windows, install python and IDE using any of the following instructions,
      although there are many different equally effective ways to install Python on Windows

        https://www.python.org/downloads/windows/
        https://conda.io/projects/conda/en/latest/user-guide/install/windows.html
        https://docs.spyder-ide.org/3/installation.html

      Then use the following command to install dependencies::

    	conda install --file RAMP/examples/scripts/requirements.txt

3. Use your EDX credentials to login at https://edx.netl.doe.gov/user/login,
   then find your EDX key by clicking on your name in the upper right, then
   copy your EDX key from the field labelled EDX-API-KEY. Note that this key
   gives essentially the same access as having your username and password,
   so take care not to email it broadly or post it anywhere publicly visible,
   especially if your EDX account has access to sensitive data.

.. image:: images/uc1/edx_page.png
   :align: center
   :alt: NETL EDX

.. image:: images/uc1/edx_key.png
   :align: center
   :alt: EDX Key

4. Navigate to the working directory in RAMP/examples/scripts

5. Edit the inputs.json file changing the EDX key to yours. Take care
   to remove the EDX key later if sharing the inputs.json file broadly.
   Also edit variables like the number of simulations to download, how long
   each stage of the multi-stage optimization should be, where to store inputs
   and outputs, etc.

6. Run the optimization using the following command::

    python3 ramp_case1_full.py inputs.json

   If you get the following error, it likely means that something is wrong with your EDX key::

       Traceback (most recent call last):
       File "C:\development\RAMP\examples\scripts\ramp_case1_full.py", line 214, in <module>
       resources = json_data['result']['resources']
       KeyError: 'result'

   If you get the following error, it likely means that the number of simulations
   you selected is incompatible with the length of the various stages you defined,
   meaning that by the time your “stage 3” of the optimization begins,
   there are no remaining leakage scenarios to attempt to detect.

.. image:: images/uc1/error2.png
   :align: center
   :alt: Error Messages #2

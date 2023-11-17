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

    * In the command line, enter ::

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

      As an alternative, users of Windows OS can follow the instructions
      provided in the file *installation_instructions_Windows.txt*
      in the folder *installation/windows*.

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

4. Navigate to the working directory *RAMP/examples/scripts*.

5. Edit the *inputs.json* or *inputs.yaml* file in order to modify
   various variables controlling the way the optimization script runs.

    a.	Change the “edx_api_key” variable to your EDX key. Take care to remove
        the EDX key later if sharing the *inputs.json* file with other users.

    b.	Edit the next four true/false variables, labelled “download_data”,
        “process_data”, “run_optimization” and “plot_results”, to specify
        whether you want all those steps run. For example, once you've downloaded the data,
        you can probably set “download_data” to false on subsequent runs.
        Once the original seismic data is processed into NRMS values you can set
        "process_data" to false. Once you've run the optimization successfully
        and are happy with the results but want to edit the plotting codes,
        you can probably set “run_optimization” to false on subsequent runs.

    c.	Leave “data_case” equal to 1, in later versions there may be other
        data cases available to work with.

    d.	Define the number or list of leakage scenarios to include
        in the optimization. Currently you can simply specify a single number X
        and the script will use all leakage scenarios 1 through X. You can also
        use a dash to specify a range, X-Y will use all leakage scenarios
        from X to Y. You can also use commas to specify an irregular list
        of leakage scenarios, eg  “1,5,9,23,24,25”

    e.	Define the directories where you'd like the various leakage scenarios,
        intermediate NRMS files and output files stored using the parameters
        “directory_seismic_data”, “directory_velocity_data”,
        “directory_nrms_data” and “directory_plots”

    f.	Define the list of potential source locations you'd like to consider
        in the optimization. You can specify an irregular list of values
        using the “sources” parameter, or you can design a set of evenly
        spaced potential sources using the “sourceNum”, “sourceMin” and
        “sourceMax” variables. If the “sources” variable is defined,
        it overrides the “sourceNum”, “sourceMin” and “sourceMax” variables.

    g.	Define the list of potential receiver locations using the same approach.

    h.	Define the seismic total duration using the “seismic_total_duration” variable.

    i.	Define the seismic sampling interval using
        the “seismic_sampling_interval” variable.

    j.	Define the NRMS threshold using the “threshold_nrms” variable.

    k.	Define how many time-steps you would like to include in each stage
        of the 3-stage optimization process using the “stage1”
        and “stage2” variables.

    l.	Define the number of monitoring plans you'd like to choose
        from using the “number_proposals” variable. While the optimization
        generates many thousands of potential monitoring plans, this limits
        the volume of information thrown at the user by narrowing it down
        to a set number of proposed monitoring plans.

6. Run the optimization using the following command, depending on which
   input file you chose to edit. Depending on your python installation,
   you may need to use "python" instead of "python3" in this command::

    python3 ramp_case1_full.py inputs.json

   or ::

    python3 ramp_case1_full.py inputs.yaml

   If you are using a visual IDE (e.g., Spyder or PyCharm) to run python,
   there may be a more complex and environment-specific way of running
   a python script with argument variables. A few examples are provided at these links:

    https://qbi-software.github.io/Python-tutorial/lessons/1-scripting.html

    https://www.jetbrains.com/help/pycharm/run-debug-configuration-python.html

   If you get the following error, it likely means that something
   is wrong with your EDX key::

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

The base output file is exported in JSON, YAML and binary formats, and
includes all monitoring plans ever constructed by the algorithm, which
can be unwieldly. Therefore a summary output file is also generated
which only includes a limited number of the best monitoring plans.
The user can define how many monitoring plans in the inputs file
(suggested 3-5 options).

The summary output file includes the list of only the seismic arrays
included in the monitoring plans, where each seismic array specifies
the number of receivers, locations of the receivers, and location
of the source. The list of monitoring plans are labelled and organized
into 3 stages, with the monitoring plans for each stage including
the arrays and deployment times, the number of leakage scenarios detected,
the list of particular scenarios detected, and the average time to detection.

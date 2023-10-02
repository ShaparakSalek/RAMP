"""
New component model class is supposed to be placed in the folder src/ramp/components.

Some IDEs allow for tabs to be automatically replaced by 4 spaces if you indicate
it in your IDE preferences. In situations other than that please don't use tabs
for indentation but rather 4 spaces.

If you do use this template, at the end of development please remove not relevant
comments and do provide your comments regarding what is happening in the code.
"""
import os
import sys
# More imports might be required if some examples of a component application
# are included with the component code.

# Since new component is based on BaseClassName we need to import it
# See examples of other components for needed typical imports
try:
    from module_name import BaseClassName
except ImportError as err:
    print('Unable to import ComponentModel class:', err)


# NewComponent is a placeholder for some meaningful name of your component
# BaseClassName is probably based on ComponentModel class of NRAP-Open-IAM
# so it has the same basic attributes and methods
# There are RAMP specific classes based on ComponentModel class: e.g.,
# DataContainer, MonitoringTechnology
class NewComponent(BaseClassName):
    """
    Here, we ask developers of the new component to provide description of
    the component, references for users to get more information about
    component and related work, as well as description of parameters, their
    default values, and description of component outputs. For examples of formatting
    this part of the documentation please see RAMP components' Python
    files.

    """
    # The bare minimum required signature of __init__ method is def __init__(self, name, parent):
    # Here, we have additional arguments arg1 and arg2
    # and dictionary of keyword arguments kwargs which, in general, is
    # a dictionary of optional arguments
    # The following method is a required method creating an instance
    # of the NewComponent class
    def __init__(self, name, parent, arg1, arg2, **kwargs):
        """
        Constructor method of NewComponent class

        :param name: name of component model
        :type name: [str]

        :param parent: the SystemModel object that the component model
            belongs to
        :type parent: SystemModel

        Here, as a good practice we recommend to provide description of all additional
        arguments of constructor method for NewComponent applicable to your model/component.
        """
        # Set up keyword arguments of the 'model' method provided by the system model
        # One need to setup model_kwargs dictionary if the model is time dependent
        # If the model depends only on time point but not on the time passed
        # between consecutive, then it's enough to define model_kwargs as
        # model_kwargs = {'time_point': 365.25}
        model_kwargs = {'time_point': 365.25, 'time_step': 365.25}

        # The super method uses the model_kwargs variable as arguments of the ComponentModel or BaseClassName
        # class’s __init__ method. It also specifies the name of the method
        # (in this case simulation_model) responsible for calculation of
        # solution.
        # If your component does not have model_kwargs the next command will look like
        # super().__init__(name, parent, model=self.simulation_model)
        super().__init__(
            name, parent, model=self.simulation_model, model_kwargs=model_kwargs)

        # Add type attribute
        self.class_type = 'NewComponent'

        # Set default parameters of the component model
        # The next 4 lines define the names of the component parameters and their
        # default values
        # This example component has 4 parameters named par1, par2, par3, par4
        # In general, The number of possible model parameters is not limited
        # by the RAMP framework. The purpose of this section of code
        # in the __init__  method is to ensure that all parameters of
        # the simulation model corresponding to the component are defined even when
        # not all parameters values are defined by users at runtime.
        # We recommend to use the same naming conventions for parameters as used
        # in other components to indicate that this is the same type of parameters.
        # If you're not sure about these conventions and/or whether the same
        # parameters were used somewhere else please ask.
        self.add_default_par('par1', value=3.0)
        self.add_default_par('par2', value=2.0)
        self.add_default_par('par3', value=-7.0)
        self.add_default_par('par4', value=0.0)

        # Define dictionary of parameters boundaries
        # The next 5 lines define the dictionary attribute pars_bounds containing
        # the upper and lower boundaries for each parameter of the simulation model.
        # This attribute is used in the method check_input_parameters of the
        # ComponentModel class. The method check_input_parameters is called
        # before the start of each simulation to check whether the provided
        # parameters satisfy the defined boundaries
        # Each entry of pars_bounds dictionary is a list of lower and upper
        # boundary, respectively, for the correspoding parameter
        self.pars_bounds = dict()
        self.pars_bounds['par1'] = [1.0, 5.5]
        self.pars_bounds['par2'] = [0.0, 10.0]
        self.pars_bounds['par3'] = [-100.0, 100.0]
        self.pars_bounds['par4'] = [-100.0, 100.0]

        # Define dictionary of temporal data limits
        # Boundaries of temporal inputs are defined by the simulation model
        # Some time-dependent simulation models might require checks of
        # temporal inputs. The check for the temporal inputs boundaries
        # should be called or implemented within the model method
        # Some components have this check implemented within a separate method
        # called check_temporal_inputs and then have this method called on
        # provided temporal inputs.
        # Similar to input parameters, the number of possible temporal inputs
        # of the simulation model is not limited by the RAMP framework.
        # For common temporal inputs we also have naming conventions: e.g.,
        # pressure, CO2saturation, which we recommend to follow for
        # simplicity of connections between components
        # temp_data_bounds is a dictionary attribute with each entry
        # being a list of 3 elements: brief description of temporal input, and
        # lower and upper boundaries
        self.temp_data_bounds = dict()
        self.temp_data_bounds['temp_input1'] = ['Temporal input 1', 1., 5.]
        self.temp_data_bounds['temp_input2'] = ['Temporal input 2', 1.5, 4.5]

        # The following lines are quite specific and assume that NewComponent
        # might need to have additional attributes.
        # Accumulators are needed if component needs track of cumulative type
        # quantities. For example, some wellbore components keep track of how much
        # CO2 and brine leaked already, or how much CO2/brine accumulated in a
        # particular aquifer layer. RAMP components might need to keep track of
        # detection times. In general, accumulators are not commonly
        # needed attribute so it is possible your component won't need them
        # either.
        # Define accumulators and their initial values
        self.add_accumulator('accumulator1', sim=0.0)
        self.add_accumulator('accumulator2', sim=1.0)

        # Define additional component model attributes
        # The next 2 lines address two additional arguments present in the signature
        # of __init__ methor: arg1 and arg2. These additional attributes (which
        # can be named according to the needs of the component) are almost always
        # needed to store some additional information about component that do not
        # fit into the description of parameters. It's hard to be specific but
        # examples of components Python files can provide ideas about what
        # these might look like
        self.additional_attr1 = arg1
        self.additional_attr2 = arg2

        # Indicate how often the component should be run
        # I suggest to not include this line in your code unless you need
        # for your component to have a different value assigned
        # Description of possible different values and their meanings is provided
        # in the developer's guide
        self.run_frequency = 2 # 2 is default

        # Define the conditional attribute if it is provided in kwargs
        # Some of the NewComponent object attributes may not be defined for
        # all instances/scenarios. The attribute conditional_attr can be assigned
        # a value, for example, only if a particular argument of the __init__
        # method was provided (in this case through kwargs argument).
        # The conditional arguments can control and define some features
        # of the simulation model defined by the developer. The names conditional_attr
        # and cond_attr are arbitrary and for illustration purposes only.
        # We recommend to name them in a way that identifies their purpose.
        if 'cond_attr' in kwargs:
            self.conditional_attr = kwargs['cond_attr']

        # This concludes the common parts of the constructor method. We
        # recommend to look at the components' Python files for more specific
        # examples of development. We also recommend to start with the components
        # that represent the same part of the system: data container, monitoring
        # technology for more applicable ideas.

        # ====================================================================
        # The model method is an instance method of the component class that
        # either calls the backend model (typical if the backend model
        # is an external code not written in Python), or is the backend itself
        # The model method must accept a dictionary of input parameter values
        # keyed by parameter names as the first argument and return a dictionary
        # of model results keyed by distinct observation names. In general,
        # arguments of the model method can be split into time-constant and
        # time-dependent arguments. All constant parameters defined by the user of the component
        # model must be passed to the model method in the dictionary p (see below).
        # The time-dependent and other types of arguments (not necessarily defined
        # by user) can be passed as keyword arguments. The signature of the model method
        # can include the default values of these arguments for situations
        # when they are not provided (see first line below; e.g., temp_input1=2.0)
        # but this is not required. In the case the default values of temporal
        # inputs are not provided, we recommend that part of the code within
        # the model method addresses this type of situation.
        # The bare minimum required signature of the model method is
        # def model(self, p):
        # Addition of all other arguments should be defined by the needs of the
        # component.

    # The following method is responsible for calculating/returning the outputs
    # associated with a given component
    # For different types of components the name of the method might be different:
    # for example, for data containers it's "export"; for monitoring technology
    # it's "process_data"; for simulation type of components it's "simulation_model"
    def model(self, p, temp_input1=2.0, temp_input2=3.0,
        time_point=365.25, time_step=365.25):
        """
        :param p: input parameters of NewComponent model
        :type p: dict

        :param temp_input1: the first of the two varying in time inputs of model
        method with default value of 2.0
        :type temp_in1: float

        :param temp_input2: the second of the two varying in time inputs of model
        method with default value of 2.0
        :type temp_in2: float

        :param time_point: time point in days at which the model output is
        to be calculated; by default, its value is 365.25 (1 year in days)
        :type time_point: float

        :param time_step: difference between the current and previous
        time points in days; by default, its value is 365.25 (1 year in days)
        :type time_point: float

        Here, as a good practice we recommend to provide description of all additional
        arguments of model method for NewComponent applicable to your model/component.

        Names temp_input1 and temp_input2 are arbitrary. Your component might have something
        like pressure, CO2 saturation, co2_rate, brine_rate. Remember about
        the naming conventions that possibly exist for your temporal inputs.
        Ask about them if you're not sure.
        """
        # Obtain the default values of the parameters from dictionary
        # of default parameters
        # If your component has parameters you'll have this line
        # What it does is it reads the default values of the component parameters
        # and in the next line updates the parameters dictionary with values
        # provided by user.
        actual_p = dict([(k,v.value) for k,v in self.default_pars.items()])
        # Update default values of parameters with the provided ones
        actual_p.update(p)

        # For the initial time point 0.0 the model should be able to return
        # the initial values of the component observations
        # This component assumes that observations are equal to zero at the
        # initial time point. If your output is not zero then it should be
        # specified here. Make sure to define values at the initial time point
        # for all your outputs
        if time_point == 0.0:
            # Define initial values of the model observations
            # The values chosen are arbitrary and the number of possible
            # observations of model is not limited in RAMP framework
            out = {}
            out['obs1'] = 0.0
            out['obs2'] = 0.0
            out['obs3'] = 0.0
            # Exit method
            return out

        # Check whether the temporal inputs satisfy the model requirements
        # and/or assumptions if needed
        # The instance method check_temporal_input can use the attribute
        # temp_data_bounds defined in the __init__ method
        # If your component has temporal inputs which require checking
        # this is the place to implement this check
        # In general, assumptions_satisfied is a flag (boolean) variable
        # You dont' have to call it this way: something simpler is fine
        # The purpose is to check whether the boundaries are met
        assumptions_satisfied = self.check_temporal_input(time, temp_input1, temp_input2)

        # Once all the checks and other required actions are performed
        # the outputs can be calculated. This is also the place to call
        # the backend model if it's developed separately.
        # The next steps depend on a particular implementation of the model method
        if assumptions_satisfied:
            # Calculate output of the component using parameters and
            # temporal keyword arguments. The signature of the backend_model
            # is not defined by the RAMP framework
            # You might need to transform parameters p or temporal inputs
            # into different units before sending them to the backend_model
            # You might need some additional info sent to the backend_model, or
            # some limited data.
            # Variable output contains output returned by backend_model
            # It might be in different units than is required by RAMP
            # so some transformation of it might be needed
            output = backend_model(p, temp_input1, temp_input2, time_point, time_step)
            # Assign values to the component accumulators
            # acc_fun1 and acc_fun2 are placeholder method names for some actions performed
            # on the variable output in order to obtain accumulators values
            # If your component need accumulators then they need to be updated
            # within the model method
            self.accumulators['accumulator1'].sim = acc_fun1(output)
            self.accumulators['accumulator2'].sim = acc_fun2(output)

            # Assign model observations
            # f1, f2, f3 are method names replacing some actions performed
            # on the variable output in order to obtain observations values
            # In most cases these actions are not needed and outputs
            # can be directly obtained from variable output if it's dictionary.
            # Again, for ideas how this part might look like see the components
            # Python files.
            out['obs1'] = f1(output)
            out['obs2'] = f2(output)
            out['obs3'] = f3(output)
            # The next type of statement is required: the model method should
            # return a dictionary out (defined) with keys corresponding to the names of
            # all possible NewComponent observations. The name out is a placeholder
            # and can be replaced with any other reasonable name.
            return out

        # Please remember that this template specifies the bare necesity
        # required from the component developer. With high probablity
        # you will have more lines and/or more complex code.
        # If you open this file in any Python Editor, don't try to run it, it won't
        # work as most of the methods used here are just name placeholders
        # and are not defined. After using this template for a
        # successful development of a new component, it would be great
        # if you could provide a feedback regarding what was missing/was useful, etc.

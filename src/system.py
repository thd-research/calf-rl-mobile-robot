from regelum.utils import rg
from regelum.system import (
    InvertedPendulum,
    ThreeWheeledRobotKinematic,
    ThreeWheeledRobotDynamic,
)
from regelum.animation import DefaultAnimation
from .animation import (
    ThreeWheeledRobotAnimationWithNewLims,
    ThreeWheeledRobotAnimationWithSpotNewLims,
    MyObjectiveAnimation
)
from regelum.callback import detach
from regelum.system import System   
from regelum.animation import (
    ObjectiveAnimation
)

# In the following two classes we want to alter their respective animation callbacks, so we:
# - detach the default animations
# - attach `DefaultAnimation` of the action and state plots
# - attach a new animation with new x- and y-limtis of [-1.3, 1.3]
#
# To learn more on customizing animations in regelum, go to https://regelum.aidynamic.io/tutorials/animations/


@ThreeWheeledRobotAnimationWithNewLims.attach
@DefaultAnimation.attach
@detach
class MyThreeWheeledRobotDynamic(ThreeWheeledRobotDynamic):
    """The parameters correspond roughly to those of Robotis TurtleBot3."""

    _parameters = {"m": 1, "I": 0.005}
    action_bounds = [[-1, 1], [-1, 1]]


@ThreeWheeledRobotAnimationWithNewLims.attach
@DefaultAnimation.attach
@detach
class MyThreeWheeledRobotKinematic(ThreeWheeledRobotKinematic):
    """The parameters correspond to those of Robotis TurtleBot3."""

    action_bounds = [[-0.22, 0.22], [-2.84, 2.84]]


@ThreeWheeledRobotAnimationWithNewLims.attach
@DefaultAnimation.attach
@detach
class MyThreeWheeledRobotKinematicCustomized(System):
    """Kinematic three-wheeled robot system implementation. """

    # These private variables are leveraged 
    # by other components within the codebase.

    # While optional, naming variables 
    # enhance usability, especially for plotting.

    _name = 'ThreeWheeledRobotKinematic'
    _system_type = 'diff_eqn'
    _dim_state = 3
    _dim_inputs = 2
    _dim_observation = 3
    _observation_naming = _state_naming = ["x_rob", "y_rob", "vartheta"]
    _inputs_naming = ["v", "omega"]
    action_bounds = [[-0.22, 0.22], [-2.84, 2.84]]

    def _compute_state_dynamics(self, time, state, inputs):
        """ Calculate the robot's state dynamics. """

        # Placeholder for the right-hand side of the differential equations
        Dstate = rg.zeros(self._dim_state, prototype=state) #

        # Element-wise calculation of the Dstate vector 
        # based on the system's differential equations
        Dstate[0] = inputs[0] * rg.cos(state[2])  # v * cos(vartheta)
        Dstate[1] = inputs[0] * rg.sin(state[2])  # v * sin(vartheta)
        Dstate[2] = inputs[1]                     # omega

        return Dstate


@ThreeWheeledRobotAnimationWithSpotNewLims.attach
@DefaultAnimation.attach
@detach
class ThreeWheeledRobotKinematicWithSpot(MyThreeWheeledRobotKinematic): ...



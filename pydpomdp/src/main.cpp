#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "madp_wrapper.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(pydpomdp, m) {
  m.doc() = "parser for .dpomdp files";
  py::class_<pydpomdp::MADPDecPOMDPDiscrete>(m, "DecPOMDP")
      .def(py::init<const std::string&>(), "Parse a given .dpomdp file",
           py::arg("dpomdp_filename"))
      .def("num_agents", &pydpomdp::MADPDecPOMDPDiscrete::num_agents,
           "Return number of agents")
      .def("num_states", &pydpomdp::MADPDecPOMDPDiscrete::num_states,
           "Return number of states")
      .def("num_joint_actions",
           &pydpomdp::MADPDecPOMDPDiscrete::num_joint_actions,
           "Return number of joint actions")
      .def("num_joint_observations",
           &pydpomdp::MADPDecPOMDPDiscrete::num_joint_observations,
           "Return number of joint observations")
      .def("num_actions", &pydpomdp::MADPDecPOMDPDiscrete::num_actions,
           "Return number of individual actions for an agent", py::arg("agent"))
      .def("num_observations",
           &pydpomdp::MADPDecPOMDPDiscrete::num_observations,
           "Return number of individual observations for an agent",
           py::arg("agent"))
      .def("agent_name", &pydpomdp::MADPDecPOMDPDiscrete::agent_name,
           "Return name of an agent", py::arg("agent"))
      .def("state_name", &pydpomdp::MADPDecPOMDPDiscrete::state_name,
           "Return name of a state", py::arg("state"))
      .def("action_name", &pydpomdp::MADPDecPOMDPDiscrete::action_name,
           "Return name of an individual action of an agent", py::arg("agent"),
           py::arg("action"))
      .def("observation_name",
           &pydpomdp::MADPDecPOMDPDiscrete::observation_name,
           "Return name of an individual observation of an agent",
           py::arg("agent"), py::arg("observation"))
     .def("joint_to_individual_action_indices", &pydpomdp::MADPDecPOMDPDiscrete::joint_to_individual_action_indices,
          "Convert joint action index to individual action indices",
          py::arg("joint_action"))
     .def("joint_to_individual_observation_indices", &pydpomdp::MADPDecPOMDPDiscrete::joint_to_individual_observation_indices,
          "Convert joint observation index to individual observation indices",
          py::arg("joint_observation"))
     .def("individual_to_joint_action_indices", &pydpomdp::MADPDecPOMDPDiscrete::individual_to_joint_action_indices,
          "Convert individual action indices to a joint action index",
          py::arg("action_indices"))
     .def("individual_to_joint_observation_indices", &pydpomdp::MADPDecPOMDPDiscrete::individual_to_joint_observation_indices,
          "Convert individual observation indices to joint observation index",
          py::arg("observation_indices"))
      .def("initial_belief_at",
           &pydpomdp::MADPDecPOMDPDiscrete::initial_belief_at,
           "Return probability of a given initial state", py::arg("state"))
      .def("reward", &pydpomdp::MADPDecPOMDPDiscrete::reward,
           "Return the reward for a given state and joint action",
           py::arg("state"), py::arg("joint_action"))
      .def("discount", &pydpomdp::MADPDecPOMDPDiscrete::discount,
           "Return the discount factor")
      .def("observation_probability",
           &pydpomdp::MADPDecPOMDPDiscrete::observation_probability,
           "Return the probability of a joint observation given a state and a "
           "joint action",
           py::arg("joint_observation"), py::arg("state"),
           py::arg("joint_action"))
      .def("transition_probability",
           &pydpomdp::MADPDecPOMDPDiscrete::transition_probability,
           "Return the probability of a new state given the current state and "
           "joint action",
           py::arg("new_state"), py::arg("state"), py::arg("joint_action"))
      .def("sample_next_state",
           &pydpomdp::MADPDecPOMDPDiscrete::sample_next_state,
           "Sample the next state given the current state and joint action and a random floating point number in the range [0,1)",
           py::arg("state"), py::arg("joint_action"), py::arg("d"))
      .def("sample_observation",
           &pydpomdp::MADPDecPOMDPDiscrete::sample_observation,
           "Sample the joint observation given the state and previous joint action and a random floating point number in the range [0,1)",
           py::arg("state"), py::arg("joint_action"), py::arg("d"))
     .def("sample_initial_state",
           &pydpomdp::MADPDecPOMDPDiscrete::sample_initial_state,
           "Sample the initial state given a random floating point number in the range [0,1)",
           py::arg("d"))
      .def("__repr__", &pydpomdp::MADPDecPOMDPDiscrete::soft_print,
           "Print human-readable description of the Dec-POMDP");

    #ifdef VERSION_INFO
        m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
    #else
        m.attr("__version__") = "dev";
    #endif
}

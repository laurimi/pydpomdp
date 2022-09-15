#ifndef MADPWRAPPER_H
#define MADPWRAPPER_H
#include <cstddef>
#include <memory>
#include <string>

namespace pydpomdp {
class MADPDecPOMDPDiscrete {
 public:
  MADPDecPOMDPDiscrete(const std::string& dpomdp_filename);
  ~MADPDecPOMDPDiscrete();
  unsigned int num_agents() const;
  unsigned int num_states() const;
  unsigned int num_joint_actions() const;
  unsigned int num_joint_observations() const;
  unsigned int num_actions(unsigned int agent) const;
  unsigned int num_observations(unsigned int agent) const;
  std::string agent_name(unsigned int agent) const;
  std::string state_name(unsigned int state_index) const;
  std::string action_name(unsigned int agent, unsigned int action_index) const;
  std::string observation_name(unsigned int agent,
                               unsigned int observation_index) const;

  std::vector<unsigned int> joint_to_individual_action_indices(unsigned int action_index) const;
  std::vector<unsigned int> joint_to_individual_observation_indices(unsigned int observation_index) const;
  unsigned int individual_to_joint_action_indices(const std::vector<unsigned int>& action_indices) const;
  unsigned int individual_to_joint_observation_indices(const std::vector<unsigned int>& observation_indices) const;

  double initial_belief_at(unsigned int state) const;
  double reward(unsigned int state, unsigned int j_act) const;
  double discount() const;
  double observation_probability(unsigned int j_obs, unsigned int state,
                                 unsigned int j_act) const;
  double transition_probability(unsigned int new_state, unsigned int state,
                                unsigned int j_act) const;
  unsigned int sample_next_state(unsigned int state, unsigned int j_act, double d) const;
  unsigned int sample_observation(unsigned int state, unsigned int j_act, double d) const;
  std::string soft_print() const;
 private:
  class MADPDecPOMDPDiscrete_impl;
  std::unique_ptr<MADPDecPOMDPDiscrete_impl> pimpl_;
};
}  // namespace pydpomdp

#endif  // MADPWRAPPER_H

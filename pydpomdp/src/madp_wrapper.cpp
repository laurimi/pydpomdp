#include "DecPOMDPDiscrete.h"
#include "ParserDPOMDPFormat_Spirit.h"
#include "madp_wrapper.h"

namespace pydpomdp {

struct MADPDecPOMDPDiscrete::MADPDecPOMDPDiscrete_impl {
  MADPDecPOMDPDiscrete_impl(const std::string& dpomdp_filename)
      : d_(std::make_unique<::DecPOMDPDiscrete>("", "", dpomdp_filename))
  {
    DPOMDPFormatParsing::ParserDPOMDPFormat_Spirit parser(d_.get());
    parser.Parse();
  }
  std::unique_ptr<::DecPOMDPDiscrete> d_;
};

MADPDecPOMDPDiscrete::MADPDecPOMDPDiscrete(const std::string& dpomdp_filename)
    : pimpl_(std::make_unique<MADPDecPOMDPDiscrete::MADPDecPOMDPDiscrete_impl>(
          dpomdp_filename)) {}

MADPDecPOMDPDiscrete::~MADPDecPOMDPDiscrete() = default;

unsigned int MADPDecPOMDPDiscrete::num_agents() const {
  return pimpl_->d_->GetNrAgents();
}
unsigned int MADPDecPOMDPDiscrete::num_states() const {
  return pimpl_->d_->GetNrStates();
}
unsigned int MADPDecPOMDPDiscrete::num_joint_actions() const {
  return pimpl_->d_->GetNrJointActions();
}
unsigned int MADPDecPOMDPDiscrete::num_joint_observations() const {
  return pimpl_->d_->GetNrJointObservations();
}
unsigned int MADPDecPOMDPDiscrete::num_actions(unsigned int agent) const {
  return pimpl_->d_->GetNrActions(agent);
}
unsigned int MADPDecPOMDPDiscrete::num_observations(unsigned int agent) const {
  return pimpl_->d_->GetNrObservations(agent);
}
std::string MADPDecPOMDPDiscrete::agent_name(unsigned int agent) const {
  return pimpl_->d_->GetAgentNameByIndex(agent);
}
std::string MADPDecPOMDPDiscrete::state_name(unsigned int state_index) const {
  return pimpl_->d_->GetState(state_index)->GetName();
}
std::string MADPDecPOMDPDiscrete::action_name(unsigned int agent,
                                              unsigned int action_index) const {
  return pimpl_->d_->GetAction(agent, action_index)->GetName();
}
std::string MADPDecPOMDPDiscrete::observation_name(
    unsigned int agent, unsigned int observation_index) const {
  return pimpl_->d_->GetObservation(agent, observation_index)->GetName();
}
std::vector<unsigned int> MADPDecPOMDPDiscrete::joint_to_individual_action_indices(unsigned int action_index) const
{
  return pimpl_->d_->JointToIndividualActionIndices(action_index);
}
std::vector<unsigned int> MADPDecPOMDPDiscrete::joint_to_individual_observation_indices(unsigned int observation_index) const
{
  return pimpl_->d_->JointToIndividualObservationIndices(observation_index);
}
unsigned int MADPDecPOMDPDiscrete::individual_to_joint_action_indices(const std::vector<unsigned int>& action_indices) const
{
  return pimpl_->d_->IndividualToJointActionIndices(action_indices);
}
unsigned int MADPDecPOMDPDiscrete::individual_to_joint_observation_indices(const std::vector<unsigned int>& observation_indices) const
{
  return pimpl_->d_->IndividualToJointObservationIndices(observation_indices);
}
double MADPDecPOMDPDiscrete::initial_belief_at(unsigned int state) const {
  return pimpl_->d_->GetInitialStateProbability(state);
}
double MADPDecPOMDPDiscrete::reward(unsigned int state,
                                    unsigned int j_act) const {
  return pimpl_->d_->GetReward(state, j_act);
}

double MADPDecPOMDPDiscrete::discount() const {
  return pimpl_->d_->GetDiscount();
}

double MADPDecPOMDPDiscrete::observation_probability(unsigned int j_obs,
                                                     unsigned int state,
                                                     unsigned int j_act) const {
  return pimpl_->d_->GetObservationModelDiscretePtr()->Get(j_act, state, j_obs);
}
double MADPDecPOMDPDiscrete::transition_probability(unsigned int new_state,
                                                    unsigned int state,
                                                    unsigned int j_act) const {
  return pimpl_->d_->GetTransitionModelDiscretePtr()->Get(state, j_act,
                                                          new_state);
}
unsigned int MADPDecPOMDPDiscrete::sample_next_state(unsigned int state, unsigned int j_act, double d) const
{
  if (d < 0.0 || d >= 1.0)
    throw std::domain_error("double input must be in the range [0,1)");

  double sum = 0.0;
  for (unsigned int i = 0; i < num_states(); ++i)
  {
    sum += transition_probability(i, state, j_act);
    if (d <= sum)
      return i;
  }
}
unsigned int MADPDecPOMDPDiscrete::sample_observation(unsigned int state, unsigned int j_act, double d) const
{
  if (d < 0.0 || d >= 1.0)
    throw std::domain_error("double input must be in the range [0,1)");

  double sum = 0.0;
  for (unsigned int i = 0; i < num_joint_observations(); ++i)
  {
    sum += observation_probability(i, state, j_act);
    if (d <= sum)
      return i;
  }
}

unsigned int MADPDecPOMDPDiscrete::sample_initial_state(double d) const
{
  if (d < 0.0 || d >= 1.0)
    throw std::domain_error("double input must be in the range [0,1)");

  double sum = 0.0;
  for (unsigned int i = 0; i < num_states(); ++i)
  {
    sum += initial_belief_at(i);
    if (d <= sum)
      return i;
  }
}

std::string MADPDecPOMDPDiscrete::soft_print() const
{
  return pimpl_->d_->SoftPrint();
}

}  // namespace pydpomdp

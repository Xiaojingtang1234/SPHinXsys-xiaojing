#include "electro_physiology.h"

namespace SPH
{
//=================================================================================================//
void ElectroPhysiologyReaction::initializeElectroPhysiologyReaction()
{
    get_production_rates_.push_back(std::bind(&ElectroPhysiologyReaction::getProductionRateIonicCurrent, this, _1));
    get_production_rates_.push_back(std::bind(&ElectroPhysiologyReaction::getProductionRateGateVariable, this, _1));
    get_production_rates_.push_back(std::bind(&ElectroPhysiologyReaction::getProductionActiveContractionStress, this, _1));

    get_loss_rates_.push_back(std::bind(&ElectroPhysiologyReaction::getLossRateIonicCurrent, this, _1));
    get_loss_rates_.push_back(std::bind(&ElectroPhysiologyReaction::getLossRateGateVariable, this, _1));
    get_loss_rates_.push_back(std::bind(&ElectroPhysiologyReaction::getLossRateActiveContractionStress, this, _1));
};
//=================================================================================================//
Real ElectroPhysiologyReaction::getProductionActiveContractionStress(LocalSpecies &species)
{
    Real voltage_dim = species[voltage_] * 100.0 - 80.0;
    Real factor = 0.1 + (1.0 - 0.1) * exp(-exp(-voltage_dim));//eq.17
    return factor * k_a_ * (voltage_dim + 80.0);//eq.16-half part
}
//=================================================================================================//
Real ElectroPhysiologyReaction::getLossRateActiveContractionStress(LocalSpecies &species)
{
    Real voltage_dim = species[voltage_] * 100.0 - 80.0;
    return 0.1 + (1.0 - 0.1) * exp(-exp(-voltage_dim));//eq.17
}
//=================================================================================================//
Real AlievPanfilowModel::getProductionRateIonicCurrent(LocalSpecies &species)
{
    Real voltage = species[voltage_];
    return -k_ * voltage * (voltage * voltage - a_ * voltage - voltage) / c_m_; //eq.42 first half part
}
//=================================================================================================//
Real AlievPanfilowModel::getLossRateIonicCurrent(LocalSpecies &species)
{
    Real gate_variable = species[gate_variable_];
    return (k_ * a_ + gate_variable) / c_m_; //eq.42 last half part
}
//=================================================================================================//
Real AlievPanfilowModel::getProductionRateGateVariable(LocalSpecies &species)
{
    Real voltage = species[voltage_];
    Real gate_variable = species[gate_variable_];
    Real temp = epsilon_ + mu_1_ * gate_variable / (mu_2_ + voltage + Eps);//eq.8 comments
    return -temp * k_ * voltage * (voltage - b_ - 1.0); //eq.8 and eq.42 dw/dt first half
}
//=================================================================================================//
Real AlievPanfilowModel::
    getLossRateGateVariable(LocalSpecies &species)
{
    Real voltage = species[voltage_];
    Real gate_variable = species[gate_variable_];
    return epsilon_ + mu_1_ * gate_variable / (mu_2_ + voltage + Eps);//eq.8 comments epsilon_ defintion
}
//=================================================================================================//


Real FitzHughNagumoModdel::getProductionRateIonicCurrent(LocalSpecies &species)
{
    Real voltage = species[voltage_];
    Real gate_variable = species[gate_variable_];
    return - 1.0 /c_m_  * (voltage * voltage * voltage  - a_ * voltage * voltage - voltage * voltage  + gate_variable) ;
}
//=================================================================================================//
Real FitzHughNagumoModdel::getLossRateIonicCurrent(LocalSpecies &species)
{
    return  a_ / c_m_; 
}
//=================================================================================================//
Real FitzHughNagumoModdel::getProductionRateGateVariable(LocalSpecies &species)
{
    Real voltage = species[voltage_];
    return  epsilon_  *  beta_  * voltage * (voltage -sigma_); 
     
}
//=================================================================================================//
Real FitzHughNagumoModdel::getLossRateGateVariable(LocalSpecies &species)
{ 
    return epsilon_ *gama_;
} 
//=================================================================================================//

} // namespace SPH

/**
 * @file 	depolarization.cpp
 * @brief 	This is the first test to validate our PED-ODE solver for solving
 * 			electrophysiology mono-domain model closed by a physiology reaction.
 * @author 	Chi Zhang and Xiangyu Hu
 */
#include "sphinxsys.h" //SPHinXsys Library.
using namespace SPH;   // Namespace cite here.
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real L = 2.5;
Real H = 2.5;
Real resolution_ref = H / 40.0;
BoundingBox system_domain_bounds(Vec2d(0.0, 0.0), Vec2d(L, H));
// observer location
StdVec<Vecd> observation_location = {Vecd(10.0, 0.5)};
//----------------------------------------------------------------------
//	Basic parameters for material properties.
//----------------------------------------------------------------------
Real diffusion_coeff = 1.0e-4;
Real bias_coeff = 0.0;
Vec2d fiber_direction(0.0, 0.0);
Real c_m = 1.0;

Real beta = 0.5;
Real gama = 1.0;
Real sigma = 0.0;
Real a = 0.1;
Real epsilon = 0.01;

Real k_a = 0.0;

  

//----------------------------------------------------------------------
//	Geometric shapes used in the system.
//----------------------------------------------------------------------
class MuscleBlock : public MultiPolygonShape
{
  public:
    explicit MuscleBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        std::vector<Vecd> shape;
        shape.push_back(Vecd(0.0, 0.0));
        shape.push_back(Vecd(0.0, H));
        shape.push_back(Vecd(L, H));
        shape.push_back(Vecd(L, 0.0));
        shape.push_back(Vecd(0.0, 0.0));
        multi_polygon_.addAPolygon(shape, ShapeBooleanOps::add);
    }
};
//----------------------------------------------------------------------
//	Application dependent initial condition.
//----------------------------------------------------------------------
class DepolarizationInitialCondition
    : public electro_physiology::ElectroPhysiologyInitialCondition
{
  protected:
    size_t voltage_;
    size_t gate_variable;

  public:
    explicit DepolarizationInitialCondition(SPHBody &sph_body)
        : electro_physiology::ElectroPhysiologyInitialCondition(sph_body)
    {
        voltage_ = particles_->diffusion_reaction_material_.AllSpeciesIndexMap()["Voltage"];
        gate_variable = particles_->diffusion_reaction_material_.AllSpeciesIndexMap()["GateVariable"];
    };

    void update(size_t index_i, Real dt)
    {
        if (0 < pos_[index_i][0] && pos_[index_i][0] < 1.25 && 0 < pos_[index_i][1] &&  pos_[index_i][1] < 1.25  )   
        {
           all_species_[voltage_][index_i] = 1.0;
        }
     

      if (0 < pos_[index_i][0]  &&  pos_[index_i][0]  < 2.5  && 1.25 < pos_[index_i][1] && pos_[index_i][1] < 2.5)   
        {
           all_species_[gate_variable][index_i] = 0.1;
        }

       


    };
};


class DepolarizationBoundaryCondition
    : public electro_physiology::ElectroPhysiologyInitialCondition
{
  protected:
    size_t voltage_;
    size_t gate_variable;

  public:
    explicit DepolarizationBoundaryCondition(SPHBody &sph_body)
        : electro_physiology::ElectroPhysiologyInitialCondition(sph_body)
    {
        voltage_ = particles_->diffusion_reaction_material_.AllSpeciesIndexMap()["Voltage"];
        gate_variable = particles_->diffusion_reaction_material_.AllSpeciesIndexMap()["GateVariable"];
    };

    void update(size_t index_i, Real dt)
    {
        if (0 < pos_[index_i][0] && pos_[index_i][0] < 2.5 && 0 < pos_[index_i][1] &&  pos_[index_i][1] < 2.0 * resolution_ref  )   
        {
           all_species_[voltage_][index_i] = 0.0;
        }
     
       
        if (0 < pos_[index_i][0] && pos_[index_i][0] < 2.5 && ( H- 2.0 * resolution_ref)  < pos_[index_i][1] &&  pos_[index_i][1] < H  )   
        {
           all_species_[voltage_][index_i] = 0.0;
        }
     
        if (0 < pos_[index_i][1] && pos_[index_i][1] < 2.5 && 0 < pos_[index_i][0] &&  pos_[index_i][0] < 2.0 * resolution_ref  )   
        {
           all_species_[voltage_][index_i] = 0.0;
        }
     
       
        if (0 < pos_[index_i][1] && pos_[index_i][1] < 2.5 && ( H- 2.0 * resolution_ref)  < pos_[index_i][0] &&  pos_[index_i][0] < H  )   
        {
           all_species_[voltage_][index_i] = 0.0;
        }
     

     
    };
};


//----------------------------------------------------------------------
//	Main program starts here.
//----------------------------------------------------------------------
int main(int ac, char *av[])
{
    //----------------------------------------------------------------------
    //	Build up the environment of a SPHSystem.
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, resolution_ref);
    sph_system.handleCommandlineOptions(ac, av);
    IOEnvironment io_environment(sph_system);
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.
    //----------------------------------------------------------------------
    SolidBody muscle_body(sph_system, makeShared<MuscleBlock>("MuscleBlock"));

 
    SharedPtr<FitzHughNagumoModdel> muscle_reaction_model_ptr = makeShared<FitzHughNagumoModdel>(k_a, beta, gama, sigma, epsilon, a, c_m);
    muscle_body.defineParticlesAndMaterial<ElectroPhysiologyParticles, MonoFieldElectroPhysiology>(
        muscle_reaction_model_ptr, TypeIdentity<DirectionalDiffusion>(), diffusion_coeff, bias_coeff, fiber_direction);
    muscle_body.generateParticles<ParticleGeneratorLattice>();

    ObserverBody voltage_observer(sph_system, "VoltageObserver");
    voltage_observer.generateParticles<ObserverParticleGenerator>(observation_location);
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //  Generally, we first define all the inner relations, then the contact relations.
    //  At last, we define the complex relaxations by combining previous defined
    //  inner and contact relations.
    //----------------------------------------------------------------------
    InnerRelation muscle_body_inner_relation(muscle_body);
    ContactRelation voltage_observer_contact_relation(voltage_observer, {&muscle_body});
    //----------------------------------------------------------------------
    //	Define the main numerical methods used in the simulation.
    //	Note that there may be data dependence on the constructors of these methods.
    //----------------------------------------------------------------------
    SimpleDynamics<DepolarizationInitialCondition> initialization(muscle_body);
    SimpleDynamics<DepolarizationBoundaryCondition> boundary(muscle_body);

    
    InteractionWithUpdate<KernelCorrectionMatrixInner> correct_configuration(muscle_body_inner_relation);
    electro_physiology::GetElectroPhysiologyTimeStepSize get_time_step_size(muscle_body);
    // Diffusion process for diffusion body.
    electro_physiology::ElectroPhysiologyDiffusionInnerRK2 diffusion_relaxation(muscle_body_inner_relation);
    // Solvers for ODE system or reactions
    electro_physiology::ElectroPhysiologyReactionRelaxationForward reaction_relaxation_forward(muscle_body);
    electro_physiology::ElectroPhysiologyReactionRelaxationBackward reaction_relaxation_backward(muscle_body);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations and observations of the simulation.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp write_states(io_environment, sph_system.real_bodies_);
    RegressionTestEnsembleAverage<ObservedQuantityRecording<Real>>
        write_recorded_voltage("Voltage", io_environment, voltage_observer_contact_relation);
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    initialization.exec();
    boundary.exec();
    correct_configuration.exec();
    //----------------------------------------------------------------------
    //	Initial states output.
    //----------------------------------------------------------------------
    write_states.writeToFile(0);
    write_recorded_voltage.writeToFile(0);
 
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    int ite = 0;
    Real T0 = 1000.0;
    Real end_time = T0;
    Real output_interval = 0.5;       /**< Time period for output */
    Real Dt = 0.01 * output_interval; /**< Time period for data observing */
    Real dt = 0.0;
    //----------------------------------------------------------------------
    //	Statistics for CPU time
    //----------------------------------------------------------------------
    TickCount t1 = TickCount::now();
    TimeInterval interval;
    //----------------------------------------------------------------------
    //	Main loop starts here.
    //----------------------------------------------------------------------
    while (GlobalStaticVariables::physical_time_ < end_time)
    {
        Real integration_time = 0.0;
        while (integration_time < output_interval)
        {
            Real relaxation_time = 0.0;
           
            while (relaxation_time < Dt)
            {

           
                if (ite % 1000 == 0)
                {
                    std::cout << "N=" << ite << " Time: "
                              << GlobalStaticVariables::physical_time_ << "	dt: "
                              << dt << "\n";
                }
                /**Strang splitting method. */
                boundary.exec();
                
                reaction_relaxation_forward.exec(0.5 * dt);
                diffusion_relaxation.exec(dt);
                reaction_relaxation_backward.exec(0.5 * dt);

                ite++;
                dt = 0.01*get_time_step_size.exec();
                relaxation_time += dt;
                integration_time += dt;
                GlobalStaticVariables::physical_time_ += dt;
            }
            write_recorded_voltage.writeToFile(ite);
        }

        TickCount t2 = TickCount::now();
        write_states.writeToFile();
        TickCount t3 = TickCount::now();
        interval += t3 - t2;
    }
    TickCount t4 = TickCount::now();

    TimeInterval tt;
    tt = t4 - t1 - interval;
    std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;
 

    return 0;
}
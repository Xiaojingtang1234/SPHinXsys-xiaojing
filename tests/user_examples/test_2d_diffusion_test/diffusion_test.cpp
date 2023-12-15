/**
 * @file 	diffusion.cpp
 * @brief 	This is the first test to validate our anisotropic diffusion solver.
 * @author 	Chi Zhang and Xiangyu Hu
 */
#include "sphinxsys.h" //SPHinXsys Library
using namespace SPH;   // Namespace cite here
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real L = 2.0;
Real H = 0.4;
Real resolution_ref = H / 20.0;
BoundingBox system_domain_bounds(Vec2d(0.0, 0.0), Vec2d(L, H));
//----------------------------------------------------------------------
//	Basic parameters for material properties.
//----------------------------------------------------------------------
Real diffusion_coeff = 0.05;
Real bias_coeff = 0.0;
Real alpha = Pi / 6.0;
Vec2d bias_direction(cos(alpha), sin(alpha));
//----------------------------------------------------------------------
//	Geometric shapes used in the case.
//----------------------------------------------------------------------
class DiffusionBlock : public MultiPolygonShape
{
  public:
    explicit DiffusionBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
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
//	Setup diffusion material properties.
//----------------------------------------------------------------------
class DiffusionMaterial : public DiffusionReaction<Solid>
{
  public:
    DiffusionMaterial() : DiffusionReaction<Solid>({"Phi"}, SharedPtr<NoReaction>())
    {
        initializeAnDiffusion<DirectionalDiffusion>("Phi", "Phi", diffusion_coeff, bias_coeff, bias_direction);
    };
};
using DiffusionParticles = DiffusionReactionParticles<SolidParticles, DiffusionMaterial>;
//----------------------------------------------------------------------
//	Application dependent initial condition.
//----------------------------------------------------------------------
class DiffusionInitialCondition
    : public DiffusionReactionInitialCondition<DiffusionParticles>
{
  protected:
    size_t phi_;

  public:
    explicit DiffusionInitialCondition(SPHBody &sph_body)
        : DiffusionReactionInitialCondition<DiffusionParticles>(sph_body)
    {
        phi_ = particles_->diffusion_reaction_material_.AllSpeciesIndexMap()["Phi"];
    };

    void update(size_t index_i, Real dt)
    {
         if (pos_[index_i][0] >= 0.3*L && pos_[index_i][0] <= 0.7 * L)
        {
            if (pos_[index_i][1] >= 0.3*H && pos_[index_i][1] <= 0.7 * H)
            {
                 all_species_[phi_][index_i] = 1.0;
      
            }
          
        } 
        //all_species_[phi_][index_i]  = 3.0 * pos_[index_i][0] * pos_[index_i][0];
       
          
    };
};



class DiffusionUpdateCondition
    : public DiffusionReactionInitialCondition<DiffusionParticles>
{
  protected:
    size_t phi_;

  public:
    explicit DiffusionUpdateCondition(SPHBody &sph_body)
        : DiffusionReactionInitialCondition<DiffusionParticles>(sph_body)
    {
        phi_ = particles_->diffusion_reaction_material_.AllSpeciesIndexMap()["Phi"];
    };

    void update(size_t index_i, Real dt)
    {
          if ((pos_[index_i][1] >= 0.8 * H) || (pos_[index_i][1] <= 0.2 * H))
        {
             all_species_[phi_][index_i]  = 1.0;
          
        }     
    
          if ((pos_[index_i][0] >= 0.9 * L) || (pos_[index_i][0] <= 0.1 * L))
        {
             all_species_[phi_][index_i]  = 1.0;
          
        }   
    };
};


 

//----------------------------------------------------------------------
//	Specify diffusion relaxation method.
//----------------------------------------------------------------------
class DiffusionBodyRelaxation
    : public DiffusionRelaxationRK2<
          DiffusionRelaxationInner<DiffusionParticles, CorrectedKernelGradientInner>>
{
  public:
    explicit DiffusionBodyRelaxation(BaseInnerRelation &inner_relation)
        : DiffusionRelaxationRK2(inner_relation){};
    virtual ~DiffusionBodyRelaxation(){};
};
//----------------------------------------------------------------------
//	An observer particle generator.
//----------------------------------------------------------------------
class TemperatureObserverParticleGenerator : public ObserverParticleGenerator
{
  public:
    explicit TemperatureObserverParticleGenerator(SPHBody &sph_body)
        : ObserverParticleGenerator(sph_body)
    {
        size_t number_of_observation_points = 11;
        Real range_of_measure = 0.9 * L;
        Real start_of_measure = 0.05 * L;

        for (size_t i = 0; i < number_of_observation_points; ++i)
        {
            Vec2d point_coordinate(range_of_measure * (Real)i / (Real)(number_of_observation_points - 1) + start_of_measure, 0.5 * H);
            positions_.push_back(point_coordinate);
        }
    }
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
    SolidBody diffusion_body(sph_system, makeShared<DiffusionBlock>("DiffusionTest"));
    diffusion_body.defineParticlesAndMaterial<DiffusionParticles, DiffusionMaterial>();
    diffusion_body.generateParticles<ParticleGeneratorLattice>();
    //----------------------------------------------------------------------
    //	Particle and body creation of fluid observers.
    //----------------------------------------------------------------------
    ObserverBody temperature_observer(sph_system, "TemperatureObserver");
    temperature_observer.generateParticles<TemperatureObserverParticleGenerator>();
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //  Generally, we first define all the inner relations, then the contact relations.
    //  At last, we define the complex relaxations by combining previous defined
    //  inner and contact relations.
    //----------------------------------------------------------------------
    InnerRelation diffusion_body_inner_relation(diffusion_body);
    ContactRelation temperature_observer_contact(temperature_observer, {&diffusion_body});
    //----------------------------------------------------------------------
    //	Define the main numerical methods used in the simulation.
    //	Note that there may be data dependence on the constructors of these methods.
    //----------------------------------------------------------------------
    DiffusionBodyRelaxation diffusion_relaxation(diffusion_body_inner_relation);
    SimpleDynamics<DiffusionInitialCondition> setup_diffusion_initial_condition(diffusion_body);
    InteractionWithUpdate<KernelCorrectionMatrixInner> correct_configuration(diffusion_body_inner_relation);
    GetDiffusionTimeStepSize<DiffusionParticles> get_time_step_size(diffusion_body);
    PeriodicConditionUsingCellLinkedList periodic_condition_y(diffusion_body, diffusion_body.getBodyShapeBounds(), yAxis);
    PeriodicConditionUsingCellLinkedList periodic_condition_x(diffusion_body, diffusion_body.getBodyShapeBounds(), xAxis);

     SimpleDynamics<DiffusionUpdateCondition> update_diffusion_condition(diffusion_body);          

    //----------------------------------------------------------------------
    //	Define the methods for I/O operations and observations of the simulation.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp write_states(io_environment, sph_system.real_bodies_);
    RegressionTestEnsembleAverage<ObservedQuantityRecording<Real>>
        write_solid_temperature("Phi", io_environment, temperature_observer_contact);

  diffusion_body.addBodyStateForRecording<Real>("PhiChangeRate");

    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
     periodic_condition_y.update_cell_linked_list_.exec();
     periodic_condition_x.update_cell_linked_list_.exec();
    sph_system.initializeSystemConfigurations();
    correct_configuration.exec();
    setup_diffusion_initial_condition.exec();
   // update_diffusion_condition.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    int ite = 0;
    Real T0 = 10.0;
    Real end_time = T0;
    Real Output_Time = 0.1 * end_time;
    Real Observe_time = 0.1 * Output_Time;
    Real dt = 0.0;
    //----------------------------------------------------------------------
    //	Statistics for CPU time
    //----------------------------------------------------------------------
    TickCount t1 = TickCount::now();
    TimeInterval interval;
    //----------------------------------------------------------------------
    //	First output before the main loop.
    //----------------------------------------------------------------------
    write_states.writeToFile();
    write_solid_temperature.writeToFile();
    //----------------------------------------------------------------------
    //	Main loop starts here.
    //----------------------------------------------------------------------
    while (GlobalStaticVariables::physical_time_ < end_time)
    {
        Real integration_time = 0.0;
        while (integration_time < Output_Time)
        {
            Real relaxation_time = 0.0;
            while (relaxation_time < Observe_time)
            {
                dt =   0.1*get_time_step_size.exec();
                diffusion_relaxation.exec(dt);
             //   update_diffusion_condition.exec();

                 if (ite < 3.0)
                {
                    write_states.writeToFile(ite);
                    write_solid_temperature.writeToFile(ite);
                }
                if (ite % 1000 == 0)
                {
                    std::cout << "N=" << ite << " Time: "
                              << GlobalStaticVariables::physical_time_ << "	dt: "
                              << dt << "\n";
                }

                ite++;
              
                relaxation_time += dt;
                integration_time += dt;
                GlobalStaticVariables::physical_time_ += dt;
            }
        write_solid_temperature.writeToFile(ite);  
      
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

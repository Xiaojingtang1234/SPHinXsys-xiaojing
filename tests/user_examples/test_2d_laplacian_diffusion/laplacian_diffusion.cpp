/**
 * @file 	Laplacian_diffusion.cpp
 * @brief 	This is a   test to validate our anisotropic laplacian algorithm.
 * @author Xiaojing Tang and Xiangyu Hu
 */
#include "sphinxsys.h" //SPHinXsys Library
using namespace SPH;   // Namespace cite here
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real L = 2.0;
Real H = 0.4;
Real resolution_ref = H / 40.0;
Real V_j = resolution_ref * resolution_ref;
BoundingBox system_domain_bounds(Vec2d(0.0, 0.0), Vec2d(L, H));
//----------------------------------------------------------------------
//	Basic parameters for material properties.
//----------------------------------------------------------------------
Real diffusion_coeff = 1.0e-4;
Real rho0 =1.0;
Real youngs_modulus = 1.0; 
Real poisson_ratio =1.0;
//----------------------------------------------------------------------
//	Geometric shapes used in the case.
//----------------------------------------------------------------------
class DiffusionBlock : public MultiPolygonShape
{
  public:
    explicit DiffusionBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        std::vector<Vec2d> shape;
        shape.push_back(Vec2d(0.0, 0.0));
        shape.push_back(Vec2d(0.0, H));
        shape.push_back(Vec2d(L, H));
        shape.push_back(Vec2d(L, 0.0));
        shape.push_back(Vec2d(0.0, 0.0));
        multi_polygon_.addAPolygon(shape, ShapeBooleanOps::add);
    }
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



using Vec2 = Eigen::Matrix<Real,2, 1>;
using Vec3 = Eigen::Matrix<Real, 3, 1>;

using Mat2 = Eigen::Matrix<Real, 2, 2>;
using Mat3 = Eigen::Matrix<Real, 3, 3>;

class LaplacianDiffusionSolid : public LinearElasticSolid 
{
  public:
    LaplacianDiffusionSolid (Real rho0, Real coeff, Real youngs_modulus, Real poisson_ratio) 
            : LinearElasticSolid(rho0, youngs_modulus, poisson_ratio), diffusion_coeff(coeff)
    {
        material_type_name_ = "LaplacianDiffusionSolid";
    }; 
    virtual ~LaplacianDiffusionSolid (){};

    Real diffusion_coeff;
    Real DiffusivityCoefficient() { return diffusion_coeff; };
};

class LaplacianDiffusionSolid;
class LaplacianDiffusionParticles : public ElasticSolidParticles
{
  public:
    LaplacianDiffusionSolid &laplacian_solid_ ;

    LaplacianDiffusionParticles(SPHBody &sph_body, LaplacianDiffusionSolid *laplacian_solid)
		: ElasticSolidParticles(sph_body, laplacian_solid), laplacian_solid_(*laplacian_solid)  { };

    virtual ~LaplacianDiffusionParticles (){};
    
     virtual void initializeOtherVariables() 
     {
        ElasticSolidParticles::initializeOtherVariables();
        registerVariable(phi_, "Phi", [&](size_t i) -> Real
                    { return Real(0.0); });
     }; 

    StdLargeVec<Real> phi_;
};


typedef DataDelegateSimple<LaplacianDiffusionParticles> LaplacianSolidDataSimple;
typedef DataDelegateInner<LaplacianDiffusionParticles> LaplacianSolidDataInner;


class NonisotropicKernelCorrectionMatrixInner : public LocalDynamics, public GeneralDataDelegateInner
{
  public:
    NonisotropicKernelCorrectionMatrixInner(BaseInnerRelation &inner_relation)
    : LocalDynamics(inner_relation.getSPHBody()),
      GeneralDataDelegateInner(inner_relation),
       B_(*particles_->registerSharedVariable<Mat2d>("KernelCorrectionMatrix")) {};
    virtual ~NonisotropicKernelCorrectionMatrixInner(){};

  protected:
    
    StdLargeVec<Mat2d> &B_;

    void interaction(size_t index_i, Real dt = 0.0)
    {
        Mat2d local_configuration = Eps * Mat2d::Identity();

        const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
        {
            Vec2d gradW_ij = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
            Vec2d r_ji = inner_neighborhood.r_ij_[n] * inner_neighborhood.e_ij_[n];
            local_configuration -= r_ji * gradW_ij.transpose();
        }
        B_[index_i] = local_configuration;
        //note that transformed B
        //B_[index_i] =  local_configuration.transpose();
    };
     
};


class LaplacianBodyRelaxation : public LocalDynamics, public LaplacianSolidDataInner
{
  public:
    LaplacianBodyRelaxation(BaseInnerRelation &inner_relation)
     : LocalDynamics(inner_relation.getSPHBody()),  LaplacianSolidDataInner(inner_relation),
      pos_(particles_->pos_), B_(particles_->B_), phi_(particles_->phi_)
     {    
        particles_->registerVariable(A1_, "FirstOrderCorrectionVectorA1", [&](size_t i) -> Vec2d { return Vec2d::Zero(); });
        particles_->registerVariable(A2_, "FirstOrderCorrectionVectorA2", [&](size_t i) -> Vec2d { return Vec2d::Zero(); });
        particles_->registerVariable(A3_, "FirstOrderCorrectionVectorA3", [&](size_t i) -> Vec2d { return Vec2d::Zero(); });

        particles_->registerVariable(SC_, "FirstOrderCorrectionMatrixSC",  [&](size_t i) -> Mat3
                    { return Mat3::Zero(); });
        particles_->registerVariable(E_, "FirstOrderCorrectionVectorE",  [&](size_t i) -> Vec2d
                    { return Vec2d::Zero(); });
        particles_->registerVariable(G_, "FirstOrderCorrectionVectorG",  [&](size_t i) -> Vec3
                    { return Vec3::Zero(); });
        particles_->registerVariable(Laplacian_, "Laplacian",  [&](size_t i) -> Vec3
                    { return Vec3::Zero(); });
     };
    virtual ~LaplacianBodyRelaxation (){};

    StdLargeVec<Vec2d> &pos_;
    StdLargeVec<Mat2d> &B_;
    StdLargeVec<Real> &phi_;

    StdLargeVec<Vec2d> A1_;
    StdLargeVec<Vec2d> A2_;
    StdLargeVec<Vec2d> A3_;

    StdLargeVec<Mat3>  SC_;
    StdLargeVec<Vec2d> E_; 
    StdLargeVec<Vec3>  G_;
    StdLargeVec<Vec3> Laplacian_; 

 protected:
 
    void initialization(size_t index_i, Real dt = 0.0)
    {
       phi_[index_i] += dt * (Laplacian_[index_i][0] + Laplacian_[index_i][1]); 
    };
 
    void interaction(size_t index_i, Real dt = 0.0)
    {
        Vec2d &pos_n_i = pos_[index_i];   
        Neighborhood &inner_neighborhood = inner_configuration_[index_i];

        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ik
        {
            size_t index_k = inner_neighborhood.j_[n];
            Vec2d gradW_ikV_k = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
            Vec2d r_ik = (pos_n_i - pos_[index_k]);

            E_[index_i] += (phi_[index_i] -phi_[index_k]) * (B_[index_i].transpose() * gradW_ikV_k);  // HOW TO DEFINE IT???
            //TO DO, HOW to express two dimensional case in three elements???????
            A1_[index_i] += r_ik[0] * r_ik[0] * (B_[index_i].transpose() * gradW_ikV_k); // HOW TO DEFINE IT???
            A2_[index_i] += r_ik[1] * r_ik[1] * (B_[index_i].transpose() * gradW_ikV_k); // HOW TO DEFINE IT???
            A3_[index_i] += r_ik[0] * r_ik[1] * (B_[index_i].transpose() * gradW_ikV_k);  // HOW TO DEFINE IT???
        }

       for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ij
        {
            size_t index_j = inner_neighborhood.j_[n];
            Vec2d r_ij = (pos_n_i - pos_[index_j]);

            Vec3 S_ = Vec3(r_ij[0] * r_ij[0], r_ij[1] * r_ij[1], r_ij[0] * r_ij[1]);
             Real FF_ = 2.0 * ( phi_[index_i]- phi_[index_j] - r_ij.dot(E_[index_i]));
            //TO DO
            Vec3 C_  = Vec3::Zero();
            C_[0] = (r_ij[0] * r_ij[0] - r_ij.dot(A1_[index_i]));  // HOW TO DEFINE IT???
            C_[1] = (r_ij[1] * r_ij[1] - r_ij.dot(A2_[index_i]));  // HOW TO DEFINE IT???
            C_[2] = (r_ij[0] * r_ij[1] - r_ij.dot(A3_[index_i])); // HOW TO DEFINE IT???

            SC_[index_i] += S_ * C_.transpose() * V_j; 
            G_[index_i] +=  S_ * FF_ * V_j;
        }

    };

       void update(size_t index_i, Real dt = 0.0)
    {
       Laplacian_[index_i] = SC_[index_i].inverse() * G_[index_i]; 
    };
 
    
};

/*
class NonisotropicFirstOrderCorrectionVectorC : public NonisotropicFirstOrderCorrectionVectorA
{
  public:
    NonisotropicFirstOrderCorrectionVectorC(BaseInnerRelation &inner_relation)
     :NonisotropicFirstOrderCorrectionVectorA(inner_relation)
     {
        particles_->registerVariable(C_, "FirstOrderCorrectionVectorC", Vec3d::Zero());
        particles_->registerVariable(SC_, "FirstOrderCorrectionMatrixSC", Mat3d::Zero());
        particles_->registerVariable(S_, "FirstOrderCorrectionVectorS", Vec3d::Zero());
     };
    virtual ~NonisotropicFirstOrderCorrectionVectorC (){};
   StdLargeVec<Vec3d>  C_;
   StdLargeVec<Mat3d>  SC_;
    StdLargeVec<Vec3d>  S_;
    
 protected:
    void interaction(size_t index_i, Real dt = 0.0)
    {
        Vecd &pos_n_i = pos_[index_i];   
        Neighborhood &inner_neighborhood = inner_configuration_[index_i];
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
        {
            size_t index_j = inner_neighborhood.j_[n];
            Vecd r_ij = (pos_n_i - pos_[index_j]);
            S_[index_i] += Vec3d(r_ij[0] * r_ij[0], r_ij[1] * r_ij[1], r_ij[0] * r_ij[1]);
  
            //TO DO
            C_[index_i][0] += r_ij[0] * r_ij[0] - r_ij.dot( A_[index_i][0]);  // HOW TO DEFINE IT???
            C_[index_i][1] += r_ij[1] * r_ij[1] - r_ij.dot( A_[index_i][1]);  // HOW TO DEFINE IT???
            C_[index_i][2] += r_ij[0] * r_ij[1] - r_ij.dot( A_[index_i][2]); // HOW TO DEFINE IT???
        }
       
       SC_[index_i] = S_[index_i] * C_[index_i].transpose(); 
    };
    
}; */

/*
class NonisotropicFirstOrderCorrectionVectorE : public LocalDynamics, public LaplacianSolidDataInner
{
  public:
    NonisotropicFirstOrderCorrectionVectorE(BaseInnerRelation &inner_relation)
     : LocalDynamics(inner_relation.getSPHBody()), LaplacianSolidDataInner(inner_relation),
     B_(particles_->B_), phi_(particles_->phi_)
     {
        particles_->registerVariable(E_, "FirstOrderCorrectionVectorE", Vecd::Zero()); 
     };
    virtual ~NonisotropicFirstOrderCorrectionVectorE (){};
   StdLargeVec<Vecd> E_;
 protected:
    StdLargeVec<Matd> &B_;
      StdLargeVec<Real> &phi_;
    void interaction(size_t index_i, Real dt = 0.0)
    {
        Neighborhood &inner_neighborhood = inner_configuration_[index_i];
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
        {
            size_t index_k = inner_neighborhood.j_[n];
            Vecd gradW_ijV_k = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
            //TO DO,  transpose or not??
            E_[index_i] += (phi_[index_i] -phi_[index_k]) * (B_[index_i] * gradW_ijV_k );  // HOW TO DEFINE IT???
             
        }
       
    };
    
};*/
/*
class NonisotropicFirstOrderCorrectionVectorG : public NonisotropicFirstOrderCorrectionVectorACE 
{
  public:
    NonisotropicFirstOrderCorrectionVectorG(BaseInnerRelation &inner_relation)
     :NonisotropicFirstOrderCorrectionVectorACE(inner_relation) 
     {
        particles_->registerVariable(G_, "FirstOrderCorrectionVectorG", Vecd::Zero());
     };
    virtual ~NonisotropicFirstOrderCorrectionVectorG (){}; 
    StdLargeVec<Vecd>  G_;
      
 protected:
    void interaction(size_t index_i, Real dt = 0.0)
    {    
        
        Vecd S_(0.0);
        Real F  =0.0;
        Vecd &pos_n_i = pos_[index_i];   
        Neighborhood &inner_neighborhood = inner_configuration_[index_i];
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
        {    
            size_t index_j = inner_neighborhood.j_[n];
            Vecd r_ij = (pos_n_i - pos_[index_j]);
            S_ += Vecd(r_ij[0] * r_ij[0], r_ij[1] * r_ij[1], r_ij[0] * r_ij[1]);
            F += 2.0 * ( phi_[index_i]- phi_[index_j] - r_ij.dot(E_[index_i]));
        }
       G_[index_i] = S_ * F ; 
    };
    
};*/

/*
class  LaplacianBodyRelaxation : public NonisotropicFirstOrderCorrectionVectorACE
{
 public:
  StdLargeVec<Vecd> Laplacian_; 
    LaplacianBodyRelaxation(BaseInnerRelation &inner_relation)
        : NonisotropicFirstOrderCorrectionVectorACE(inner_relation)
     {
          particles_->registerVariable(Laplacian_, "Laplacian", Vecd::Zero()); 
     };

    void update(size_t index_i, Real dt = 0.0)
    {
       Laplacian_[index_i]  = SC_[index_i].inverse() * G_[index_i]; 
    };
 
     virtual ~LaplacianBodyRelaxation (){}; 
   
};*/

class DiffusionInitialCondition : public LocalDynamics, public LaplacianSolidDataInner
{
  public:
    DiffusionInitialCondition(BaseInnerRelation &inner_relation)
     : LocalDynamics(inner_relation.getSPHBody()), LaplacianSolidDataInner(inner_relation),
      pos_(particles_->pos_), phi_(particles_->phi_) {};
    virtual ~DiffusionInitialCondition (){};
    
    StdLargeVec<Vec2d> &pos_;
    StdLargeVec<Real> &phi_;
 protected:
    void update(size_t index_i, Real dt = 0.0)
    {
        if (pos_[index_i][0] >= 0.45 && pos_[index_i][0] <= 0.55)
        {
            phi_[index_i] = 1.0;
        }  
  
    };
    
};

class  GetLaplacianTimeStepSize : public LocalDynamicsReduce<Real, ReduceMin>,
                             public LaplacianSolidDataSimple
{
protected:
    Real smoothing_length;
	Real dimension;

 public:
    GetLaplacianTimeStepSize(SPHBody &sph_body, Real Coefficient)
      : LocalDynamicsReduce<Real, ReduceMin>(sph_body, Real(MaxRealNumber)),
        LaplacianSolidDataSimple(sph_body) 
        {
            smoothing_length = sph_body.sph_adaptation_->ReferenceSmoothingLength();
			dimension = Real(Vec2d(0).size());
        };
   
    Real reduce(size_t index_i, Real dt)
    {
        return  0.5 * smoothing_length * smoothing_length 
				/ particles_->laplacian_solid_.DiffusivityCoefficient() / dimension;                 
    }

    virtual ~GetLaplacianTimeStepSize (){}; 
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
    SolidBody diffusion_body(sph_system, makeShared<DiffusionBlock>("DiffusionBlock"));
    diffusion_body.defineParticlesAndMaterial<LaplacianDiffusionParticles, LaplacianDiffusionSolid>(rho0, diffusion_coeff, youngs_modulus, poisson_ratio);
        
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
    
    InteractionDynamics<NonisotropicKernelCorrectionMatrixInner> correct_configuration(diffusion_body_inner_relation);

    Dynamics1Level<LaplacianBodyRelaxation> diffusion_relaxation(diffusion_body_inner_relation);
    SimpleDynamics<DiffusionInitialCondition> setup_diffusion_initial_condition(diffusion_body_inner_relation);
 
    ReduceDynamics<GetLaplacianTimeStepSize>  get_time_step_size(diffusion_body, diffusion_coeff);

   diffusion_body.addBodyStateForRecording<Real>("Phi");
   // PeriodicConditionUsingCellLinkedList periodic_condition_y(diffusion_body, diffusion_body.getBodyShapeBounds(), yAxis);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations and observations of the simulation.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp write_states(io_environment, sph_system.real_bodies_);
    RegressionTestEnsembleAverage<ObservedQuantityRecording<Real>>
        write_solid_temperature("Phi", io_environment, temperature_observer_contact);
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    //periodic_condition_y.update_cell_linked_list_.exec();
    sph_system.initializeSystemConfigurations();
    correct_configuration.exec();
    setup_diffusion_initial_condition.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    int ite = 0;
    Real T0 = 1.0;
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
                if (ite % 1 == 0)
                {
                    std::cout << "N=" << ite << " Time: "
                              << GlobalStaticVariables::physical_time_ << "	dt: "
                              << dt << "\n";
                }

                diffusion_relaxation.exec(dt);

                ite++;
                dt = get_time_step_size.exec();
                relaxation_time += dt;
                integration_time += dt;
                GlobalStaticVariables::physical_time_ += dt;
            }
        }

        TickCount t2 = TickCount::now();
        write_states.writeToFile();
        write_solid_temperature.writeToFile(ite);
        TickCount t3 = TickCount::now();
        interval += t3 - t2;
    }
    TickCount t4 = TickCount::now();

    TimeInterval tt;
    tt = t4 - t1 - interval;
    std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;

    write_solid_temperature.testResult();

    return 0;
}

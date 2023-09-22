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
Real diffusion_coeff = 1.0e-3;
Real rho0 = 1.0;
Real youngs_modulus = 1.0; 
Real poisson_ratio = 1.0;
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
        registerVariable(A1_, "FirstOrderCorrectionVectorA1", [&](size_t i) -> Vec2d { return Eps * Vec2d::Identity(); });
        registerVariable(A2_, "FirstOrderCorrectionVectorA2", [&](size_t i) -> Vec2d { return Eps * Vec2d::Identity(); });
        registerVariable(A3_, "FirstOrderCorrectionVectorA3", [&](size_t i) -> Vec2d { return Eps * Vec2d::Identity(); });

     }; 

    StdLargeVec<Real> phi_;
     StdLargeVec<Vec2d> A1_;
    StdLargeVec<Vec2d> A2_;
    StdLargeVec<Vec2d> A3_;
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

class NonisotropicKernelCorrectionMatrixInnerAC : public LocalDynamics, public LaplacianSolidDataInner
{
  public:
    NonisotropicKernelCorrectionMatrixInnerAC(BaseInnerRelation &inner_relation)
    : LocalDynamics(inner_relation.getSPHBody()),
      LaplacianSolidDataInner(inner_relation), pos_(particles_->pos_), B_(particles_->B_)
      , A1_(particles_->A1_), A2_(particles_->A2_), A3_(particles_->A3_) {};
        
    virtual ~NonisotropicKernelCorrectionMatrixInnerAC(){};

  protected:
    StdLargeVec<Vec2d> &pos_;
    StdLargeVec<Mat2d> &B_;
    StdLargeVec<Vec2d> &A1_;
    StdLargeVec<Vec2d> &A2_;
    StdLargeVec<Vec2d> &A3_;

    void interaction(size_t index_i, Real dt = 0.0)
    {   
        Vec2d &pos_n_i = pos_[index_i];   
        Neighborhood &inner_neighborhood = inner_configuration_[index_i];
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ik
        {
            size_t index_k = inner_neighborhood.j_[n];
            Vec2d gradW_ikV_k = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
            Vec2d r_ik = pos_[index_k] - pos_n_i ;

            //TO DO, HOW to express two dimensional case in three elements??????? a should be calculate only once
            A1_[index_i] += r_ik[0] * r_ik[0] * (B_[index_i].transpose() * gradW_ikV_k); // HOW TO DEFINE IT???
            A2_[index_i] += r_ik[1] * r_ik[1] * (B_[index_i].transpose() * gradW_ikV_k); // HOW TO DEFINE IT???
            A3_[index_i] += r_ik[0] * r_ik[1] * (B_[index_i].transpose() * gradW_ikV_k);  // HOW TO DEFINE IT???
        }

    };
     
};



class LaplacianBodyRelaxation : public LocalDynamics, public LaplacianSolidDataInner
{
  public:
    LaplacianBodyRelaxation(BaseInnerRelation &inner_relation)
     : LocalDynamics(inner_relation.getSPHBody()),  LaplacianSolidDataInner(inner_relation),
      pos_(particles_->pos_), B_(particles_->B_), phi_(particles_->phi_)
      , A1_(particles_->A1_), A2_(particles_->A2_), A3_(particles_->A3_) 
     {       
        particles_->registerVariable(SC_, "FirstOrderCorrectionMatrixSC",  [&](size_t i) -> Mat3
                    { return Eps * Mat3::Identity(); });
        particles_->registerVariable(E_, "FirstOrderCorrectionVectorE",  [&](size_t i) -> Vec2d
                    { return Eps * Vec2d::Identity(); });
        particles_->registerVariable(G_, "FirstOrderCorrectionVectorG",  [&](size_t i) -> Vec3
                    { return Eps * Vec3::Identity();  });
        particles_->registerVariable(Laplacian_, "Laplacian",  [&](size_t i) -> Vec3
                    { return Vec3::Zero(); });
        diffusion_coeff_=  particles_->laplacian_solid_.DiffusivityCoefficient();

        particles_->registerVariable(neigh_, "ShowingNeigh", Real(0.0));
        particles_->registerVariable(d_phi_, "d_phi_", Real(0.0));

        kk = 1.0;
             
     };
    virtual ~LaplacianBodyRelaxation (){};

    StdLargeVec<Vec2d> &pos_;
    StdLargeVec<Mat2d> &B_;
    StdLargeVec<Real> &phi_;

    StdLargeVec<Vec2d> &A1_;
    StdLargeVec<Vec2d> &A2_;
    StdLargeVec<Vec2d> &A3_;
  
    StdLargeVec<Mat3>  SC_;
    StdLargeVec<Vec2d> E_; 
    StdLargeVec<Vec3>  G_;
    StdLargeVec<Vec3> Laplacian_; 

    Real diffusion_coeff_;
    StdLargeVec<Real> d_phi_;
    StdLargeVec<Real> neigh_;   
     Real kk ;
 protected:
 
    void initialization(size_t index_i, Real dt = 0.0) {};
   
     
    void interaction(size_t index_i, Real dt = 0.0)
    {
        Vec2d E_rate = Vec2d::Zero();  
        Vec2d &pos_n_i = pos_[index_i];   
        Neighborhood &inner_neighborhood = inner_configuration_[index_i];

        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ik
        {
            size_t index_k = inner_neighborhood.j_[n];
            Vec2d gradW_ikV_k = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
            
            E_rate  += (phi_[index_k] -phi_[index_i]) * (B_[index_i].transpose() * gradW_ikV_k);  // HOW TO DEFINE IT???
           
        }
        E_[index_i] =  E_rate ;


 
        Mat3 SC_rate = Mat3::Zero();  
        Vec3 G_rate =  Vec3::Zero();

        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ij
        {
            size_t index_j = inner_neighborhood.j_[n];
            Vec2d r_ij = pos_[index_j] - pos_n_i;

            Vec3 S_ = Vec3(r_ij[0] * r_ij[0], r_ij[1] * r_ij[1], r_ij[0] * r_ij[1]);
            Real FF_ = 2.0 * ( phi_[index_j]- phi_[index_i] - r_ij.dot(E_[index_i]));
            //TO DO
            Vec3 C_ = Vec3::Zero();
            C_[0] = (r_ij[0] * r_ij[0] - r_ij.dot(A1_[index_i]));  
            C_[1] = (r_ij[1] * r_ij[1] - r_ij.dot(A2_[index_i]));   
            C_[2] = (r_ij[0] * r_ij[1] - r_ij.dot(A3_[index_i]));  
            
            SC_rate += S_ * C_.transpose() * V_j; 
            G_rate +=  S_ * FF_ * V_j;
        }
        
        SC_[index_i] = SC_rate; 
        G_[index_i] = G_rate;

       Laplacian_[index_i] = diffusion_coeff_ * SC_[index_i].inverse() * G_[index_i]; 
    };

       void update(size_t index_i, Real dt = 0.0)
    {    
        phi_[index_i] += dt * (Laplacian_[index_i][0] + Laplacian_[index_i][1]); 
    };
 
    
};
 


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
        if (pos_[index_i][0] >= 1.0 && pos_[index_i][0] <= 1.1)
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
    GetLaplacianTimeStepSize(SPHBody &sph_body)
      : LocalDynamicsReduce<Real, ReduceMin>(sph_body, Real(MaxRealNumber)),
        LaplacianSolidDataSimple(sph_body) 
        {
            smoothing_length = sph_body.sph_adaptation_->ReferenceSmoothingLength();
			dimension =2.0;
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
   InteractionDynamics< NonisotropicKernelCorrectionMatrixInnerAC> correct_second_configuration(diffusion_body_inner_relation);
    ReduceDynamics<GetLaplacianTimeStepSize>  get_time_step_size(diffusion_body);
    Dynamics1Level<LaplacianBodyRelaxation, SequencedPolicy> diffusion_relaxation(diffusion_body_inner_relation);

   SimpleDynamics<DiffusionInitialCondition> setup_diffusion_initial_condition(diffusion_body_inner_relation);

   diffusion_body.addBodyStateForRecording<Real>("Phi"); 
   diffusion_body.addBodyStateForRecording<Real>("d_phi_"); 
   diffusion_body.addBodyStateForRecording<Real>("ShowingNeigh"); 
   diffusion_body.addBodyStateForRecording<Mat2d>("KernelCorrectionMatrix"); 

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
    sph_system.initializeSystemConfigurations();
    correct_configuration.exec();
    correct_second_configuration.exec();
    setup_diffusion_initial_condition.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    int ite = 0;
    Real T0 = 1.0;
    Real end_time = T0;
    Real Output_Time = 0.01 * end_time;
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
                if (ite % 100 == 0)
                {
                    std::cout << "N=" << ite << " Time: "
                              << GlobalStaticVariables::physical_time_ << "	dt: "
                              << dt << "\n";
                }
 
                diffusion_relaxation.exec(dt);

                
                  
                ite++;
                dt = 0.1* get_time_step_size.exec();
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
 

    return 0;
}

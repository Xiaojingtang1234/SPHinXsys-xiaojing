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
 
int resolution(20);

int z_num = 20;
Real ratio_ = 4.0;
Real resolution_ref = H / z_num;
Real resolution_ref_large = ratio_ * resolution_ref;
int x_num = L / resolution_ref_large;
Vec3d scaling_vector = Vec3d(1.0, 1.0, 1.0 / ratio_); // scaling_vector for defining the anisotropic kernel
Real scaling_factor = 1.0 / ratio_;              // scaling factor to calculate the time step


Real V_j = resolution_ref_large * resolution_ref_large * resolution_ref;
BoundingBox system_domain_bounds(Vec3d(-L, -L, -H), Vec3d(2.0 * L, 2.0 * L,  2.0 * H));
Real BL = 3.0 * resolution_ref_large;
Real BH = 3.0 * resolution_ref;
//----------------------------------------------------------------------
//	Basic parameters for material properties.
//----------------------------------------------------------------------
Real diffusion_coeff = 0.05;
Real rho0 = 1.0;
Real youngs_modulus = 1.0;
Real poisson_ratio = 1.0;
//----------------------------------------------------------------------
//	Geometric shapes used in the case.
//----------------------------------------------------------------------

Vec3d halfsize_membrane(0.5 * L, 0.5 * L, 0.5 * H);
Vec3d translation_membrane(0.5 * L, 0.5 * L, 0.5 * H);


class DiffusionBlock : public ComplexShape
{
public:
	explicit DiffusionBlock(const std::string &shape_name) : ComplexShape(shape_name)
	{
		add<TriangleMeshShapeBrick>(halfsize_membrane, resolution, translation_membrane);
	
	}
};

 
AnisotropicKernel<KernelWendlandC2>
    wendland(1.15 * resolution_ref_large, scaling_vector, Vec3d(0.0, 0.0, 0.0)); // no rotation introduced

Mat3d transform_tensor = wendland.getCoordinateTransformationTensorG(scaling_vector, Vec3d(0.0, 0.0, 0.0)); // tensor
 




/** Define application dependent particle generator for thin structure. */
class AnisotropicParticleGenerator : public ParticleGenerator
{
public:
	AnisotropicParticleGenerator(SPHBody &sph_body) : ParticleGenerator(sph_body){};

	virtual void initializeGeometricVariables() override
	{
		// set particles directly
		for (int i = 0; i < x_num; i++)
		{
			for (int j = 0; j < x_num; j++)
			{
				for (int k = 0; k < z_num; k++)
				{
					Real x = (i + 0.5) * resolution_ref_large;
					Real y = (j + 0.5) * resolution_ref_large;
					Real z = (k + 0.5) * resolution_ref;
					initializePositionAndVolumetricMeasure(Vecd(x, y, z),
						(resolution_ref * resolution_ref_large * resolution_ref_large));
				
				}
			}
		}
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
            Vec3d point_coordinate(range_of_measure * (Real)i / (Real)(number_of_observation_points - 1) + start_of_measure, 0.5*L, 0.5 * H);
            positions_.push_back(point_coordinate);
        }
    }
};


class LaplacianDiffusionSolid : public LinearElasticSolid
{
  public:
    LaplacianDiffusionSolid(Real rho0, Real coeff, Real youngs_modulus, Real poisson_ratio)
        : LinearElasticSolid(rho0, youngs_modulus, poisson_ratio), diffusion_coeff(coeff)
    {
        material_type_name_ = "LaplacianDiffusionSolid";
    };
    virtual ~LaplacianDiffusionSolid(){};

    Real diffusion_coeff;
    Real DiffusivityCoefficient() { return diffusion_coeff; };
};

class LaplacianDiffusionSolid;
class LaplacianDiffusionParticles : public ElasticSolidParticles
{
  public:
    LaplacianDiffusionSolid &laplacian_solid_;

    LaplacianDiffusionParticles(SPHBody &sph_body, LaplacianDiffusionSolid *laplacian_solid)
        : ElasticSolidParticles(sph_body, laplacian_solid), laplacian_solid_(*laplacian_solid){};

    virtual ~LaplacianDiffusionParticles(){};

    virtual void initializeOtherVariables()
    {
        ElasticSolidParticles::initializeOtherVariables();
        registerVariable(phi_, "Phi", [&](size_t i) -> Real { return Real(0.0); });
        registerVariable(A1_, "FirstOrderCorrectionVectorA1", [&](size_t i) -> Vec3d { return Eps * Vec3d::Identity(); });
        registerVariable(A2_, "FirstOrderCorrectionVectorA2", [&](size_t i) -> Vec3d { return Eps * Vec3d::Identity(); });
        registerVariable(A3_, "FirstOrderCorrectionVectorA3", [&](size_t i) -> Vec3d { return Eps * Vec3d::Identity(); });
	   
        registerVariable(A4_, "FirstOrderCorrectionVectorA4", [&](size_t i) -> Vec3d { return Eps * Vec3d::Identity(); });
        registerVariable(A5_, "FirstOrderCorrectionVectorA5", [&](size_t i) -> Vec3d { return Eps * Vec3d::Identity(); });
        registerVariable(A6_, "FirstOrderCorrectionVectorA6", [&](size_t i) -> Vec3d { return Eps * Vec3d::Identity(); });
	

        addVariableToWrite<Vec3d>("FirstOrderCorrectionVectorA1");
        addVariableToWrite<Vec3d>("FirstOrderCorrectionVectorA2");
        addVariableToWrite<Vec3d>("FirstOrderCorrectionVectorA3");
    };

    StdLargeVec<Real> phi_;
    StdLargeVec<Vec3d> A1_;
    StdLargeVec<Vec3d> A2_;
    StdLargeVec<Vec3d> A3_;
    StdLargeVec<Vec3d> A4_;
    StdLargeVec<Vec3d> A5_;
    StdLargeVec<Vec3d> A6_;

};

typedef DataDelegateSimple<LaplacianDiffusionParticles> LaplacianSolidDataSimple;
typedef DataDelegateInner<LaplacianDiffusionParticles> LaplacianSolidDataInner;
typedef DataDelegateContact<LaplacianDiffusionParticles, BaseParticles, DataDelegateEmptyBase> LaplacianSolidDataContactOnly;
typedef DataDelegateComplex<LaplacianDiffusionParticles, BaseParticles> LaplacianSolidDataComplex;
typedef DataDelegateComplex<BaseParticles, BaseParticles>GeneralDataDelegateComplex;


class NonisotropicKernelCorrectionMatrixInner : public LocalDynamics, public GeneralDataDelegateInner
{
  public:
    NonisotropicKernelCorrectionMatrixInner(BaseInnerRelation &inner_relation, Real alpha = Real(0))
        : LocalDynamics(inner_relation.getSPHBody()),
          GeneralDataDelegateInner(inner_relation), 
		B_(*particles_->registerSharedVariable<Mat3d>("KernelCorrectionMatrix")) {};

    virtual ~NonisotropicKernelCorrectionMatrixInner(){};

  protected:
	  StdLargeVec<Mat3d> &B_;
 
	  void interaction(size_t index_i, Real dt = 0.0)
	  {
		  Mat3d local_configuration = Eps * Mat3d::Identity();
		  const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
		  for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
		  {
			  Vec3d gradW_ij = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
 
			  Vec3d r_ji = inner_neighborhood.r_ij_vector_[n];
			  local_configuration -= r_ji * gradW_ij.transpose();
		  }
		  B_[index_i] = local_configuration;

	  };
 

	void update(size_t index_i, Real dt)
	{
		Mat3d inverse = B_[index_i].inverse();
		B_[index_i] = inverse;	
	}
};



class NonisotropicKernelCorrectionMatrixInnerAC : public LocalDynamics, public  LaplacianSolidDataInner
{
  public:
    NonisotropicKernelCorrectionMatrixInnerAC(BaseInnerRelation &inner_relation):
                LocalDynamics(inner_relation.getSPHBody()), LaplacianSolidDataInner(inner_relation),
                 B_(particles_->B_), A1_(particles_->A1_), A2_(particles_->A2_), A3_(particles_->A3_),
                 A4_(particles_->A4_), A5_(particles_->A5_), A6_(particles_->A6_){};

    virtual ~NonisotropicKernelCorrectionMatrixInnerAC(){};

  protected:
    StdLargeVec<Mat3d> &B_;
    StdLargeVec<Vec3d> &A1_,&A2_,&A3_, &A4_,&A5_,&A6_;

    void interaction(size_t index_i, Real dt = 0.0)
    {
        Neighborhood &inner_neighborhood = inner_configuration_[index_i];
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ik
        {
            Vec3d gradW_ikV_k = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
            Vec3d r_ik = -inner_neighborhood.r_ij_vector_[n];

            A1_[index_i] += r_ik[0] * r_ik[0] * (B_[index_i].transpose() * gradW_ikV_k);
            A2_[index_i] += r_ik[1] * r_ik[1] * (B_[index_i].transpose() * gradW_ikV_k);
            A3_[index_i] += r_ik[2] * r_ik[2] * (B_[index_i].transpose() * gradW_ikV_k);

            A4_[index_i] += r_ik[0] * r_ik[1] * (B_[index_i].transpose() * gradW_ikV_k);
            A5_[index_i] += r_ik[1] * r_ik[2] * (B_[index_i].transpose() * gradW_ikV_k);
            A6_[index_i] += r_ik[2] * r_ik[0] * (B_[index_i].transpose() * gradW_ikV_k);
        }
    };
  
};

using Mat6d = Eigen::Matrix<Real, 6, 6>;
using Vec6d = Eigen::Matrix<Real, 6, 1>;
class LaplacianBodyRelaxation : public LocalDynamics, public LaplacianSolidDataInner
{
  public:
   LaplacianBodyRelaxation(BaseInnerRelation &inner_relation)
        : LocalDynamics(inner_relation.getSPHBody()), LaplacianSolidDataInner(inner_relation),
          pos_(particles_->pos_), B_(particles_->B_), phi_(particles_->phi_), A1_(particles_->A1_), A2_(particles_->A2_), A3_(particles_->A3_),
           A4_(particles_->A4_), A5_(particles_->A5_), A6_(particles_->A6_)
    {
        particles_->registerVariable(SC_, "FirstOrderCorrectionMatrixSC", [&](size_t i) -> Mat6d { return Eps * Mat6d::Identity(); });
        particles_->registerVariable(E_, "FirstOrderCorrectionVectorE", [&](size_t i) -> Vec3d { return Eps * Vec3d::Identity(); });
        particles_->registerVariable(G_, "FirstOrderCorrectionVectorG", [&](size_t i) -> Vec6d { return Eps * Vec6d::Identity(); });
        particles_->registerVariable(Laplacian_, "Laplacian", [&](size_t i) -> Vec6d { return Vec6d::Zero(); });

        particles_->registerVariable(Laplacian_x, "Laplacian_x", [&](size_t i) -> Real { return Real(0.0); });
        particles_->registerVariable(Laplacian_y, "Laplacian_y", [&](size_t i) -> Real { return Real(0.0); });
        particles_->registerVariable(Laplacian_z, "Laplacian_z", [&](size_t i) -> Real { return Real(0.0); });
		particles_->registerVariable(diffusion_dt_, "diffusion_dt", [&](size_t i) -> Real { return Real(0.0); });
		
        diffusion_coeff_ = particles_->laplacian_solid_.DiffusivityCoefficient();
    };
    virtual ~LaplacianBodyRelaxation(){};

    StdLargeVec<Vec3d> &pos_;
    StdLargeVec<Mat3d> &B_;
    StdLargeVec<Real> &phi_;

   StdLargeVec<Vec3d> &A1_,&A2_,&A3_, &A4_,&A5_,&A6_;

    StdLargeVec<Mat6d> SC_;
    StdLargeVec<Vec3d> E_;

    StdLargeVec<Vec6d> G_; 
    StdLargeVec<Vec6d> Laplacian_;

    StdLargeVec<Real> Laplacian_x, Laplacian_y, Laplacian_z, diffusion_dt_;

    Real diffusion_coeff_;

  protected:
    void initialization(size_t index_i, Real dt = 0.0){};

    void interaction(size_t index_i, Real dt = 0.0)
    {   
        Neighborhood &inner_neighborhood = inner_configuration_[index_i];
        Vec3d E_rate = Vec3d::Zero();
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ik
        {
            size_t index_k = inner_neighborhood.j_[n];
            Vec3d gradW_ikV_k = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];

            E_rate += (phi_[index_k] - phi_[index_i]) * (B_[index_i].transpose() * gradW_ikV_k); // HOW TO DEFINE IT???
        }
        E_[index_i] = E_rate;

         
        Vec6d G_rate = Vec6d::Zero();
        Mat6d SC_rate = Mat6d::Zero();
        Real H_rate = 1.0;
         for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ik
        {
            size_t index_j = inner_neighborhood.j_[n];
            Vec3d r_ij = -inner_neighborhood.r_ij_vector_[n];
            Vec3d gradW_ijV_j = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
            Vec6d S_ = Vec6d(r_ij[0] * r_ij[0], r_ij[1] * r_ij[1], r_ij[2] * r_ij[2], r_ij[0] * r_ij[1], r_ij[1] * r_ij[2], r_ij[2] * r_ij[0]);
            H_rate = r_ij.dot(B_[index_i].transpose() * gradW_ijV_j) / pow(r_ij.norm(), 4.0);
			 
            Real FF_ = 2.0 * (phi_[index_j] - phi_[index_i] - r_ij.dot(E_[index_i]));
            G_rate += S_ *H_rate * FF_;
            
             //TO DO
            Vec6d C_ = Vec6d::Zero();
            C_[0] = (r_ij[0] * r_ij[0]);
            C_[1] = (r_ij[1] * r_ij[1]);
            C_[2] = (r_ij[2] * r_ij[2]);
            C_[3] = (r_ij[0] * r_ij[1]);
            C_[4] = (r_ij[1] * r_ij[2]);
            C_[5] = (r_ij[2] * r_ij[0]);
            SC_rate += S_ *H_rate* C_.transpose();   

        }
        
        G_[index_i] = G_rate;
        SC_[index_i] = SC_rate;
        
    
    };
 
    void update(size_t index_i, Real dt = 0.0)
    {
        phi_[index_i] += dt * (Laplacian_[index_i][0] + Laplacian_[index_i][1] + Laplacian_[index_i][2]);
    };
};
 
 class DiffusionInitialCondition : public LocalDynamics, public LaplacianSolidDataInner
{
  public:
    DiffusionInitialCondition(BaseInnerRelation &inner_relation)
        : LocalDynamics(inner_relation.getSPHBody()), LaplacianSolidDataInner(inner_relation),
          pos_(particles_->pos_), phi_(particles_->phi_){};
    virtual ~DiffusionInitialCondition(){};

    StdLargeVec<Vec3d> &pos_;
    StdLargeVec<Real> &phi_;

  protected:
    void update(size_t index_i, Real dt = 0.0)
    {
        if (pos_[index_i][0] >= 0.3 * L && pos_[index_i][0] <= 0.7 * L)
        {
            if (pos_[index_i][1] >= 0.3*L && pos_[index_i][1] <= 0.7 * L)
            {
                if (pos_[index_i][2] >= 0.3*H && pos_[index_i][2] <= 0.7 * H)
                {
                    phi_[index_i] = 1.0;
                }
            }
          
        }
        //  phi_[index_i] = 3.0 * pos_[index_i][0] * pos_[index_i][0];
       
    };
};


 
class GetLaplacianTimeStepSize : public LocalDynamicsReduce<Real, ReduceMin>,
                                 public LaplacianSolidDataSimple
{
  protected:
    Real smoothing_length;

  public:
    GetLaplacianTimeStepSize(SPHBody &sph_body)
        : LocalDynamicsReduce<Real, ReduceMin>(sph_body, Real(MaxRealNumber)),
          LaplacianSolidDataSimple(sph_body)
    {
        smoothing_length = sph_body.sph_adaptation_->ReferenceSmoothingLength();
    };

    Real reduce(size_t index_i, Real dt)
    {
        return 0.5 * smoothing_length * smoothing_length * smoothing_length / particles_->laplacian_solid_.DiffusivityCoefficient() / Dimensions;
    }

    virtual ~GetLaplacianTimeStepSize(){};
};

//----------------------------------------------------------------------
//	Main program starts here.
//----------------------------------------------------------------------
int main(int ac, char *av[])
{
   
    //----------------------------------------------------------------------
    //	Build up the environment of a SPHSystem.
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, resolution_ref_large);
    sph_system.handleCommandlineOptions(ac, av);
    IOEnvironment io_environment(sph_system);
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.
    //----------------------------------------------------------------------
 
    SolidBody diffusion_body(sph_system, makeShared<DiffusionBlock>("3DDiffusionBlock"));
    diffusion_body.sph_adaptation_->resetKernel<AnisotropicKernel<KernelWendlandC2>>(scaling_vector);
    diffusion_body.defineParticlesAndMaterial<LaplacianDiffusionParticles, LaplacianDiffusionSolid>(rho0, diffusion_coeff, youngs_modulus, poisson_ratio);
    diffusion_body.generateParticles<AnisotropicParticleGenerator>();
 
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

	InteractionWithUpdate<NonisotropicKernelCorrectionMatrixInner> correct_configuration(diffusion_body_inner_relation);
	InteractionDynamics<NonisotropicKernelCorrectionMatrixInnerAC> correct_second_configuration(diffusion_body_inner_relation);
    ReduceDynamics<GetLaplacianTimeStepSize> get_time_step_size(diffusion_body);
    Dynamics1Level<LaplacianBodyRelaxation> diffusion_relaxation(diffusion_body_inner_relation);

    SimpleDynamics<DiffusionInitialCondition> setup_diffusion_initial_condition(diffusion_body_inner_relation);
    
    diffusion_body.addBodyStateForRecording<Real>("Phi");

    diffusion_body.addBodyStateForRecording<Real>("Laplacian_x");
    diffusion_body.addBodyStateForRecording<Real>("Laplacian_y");
    diffusion_body.addBodyStateForRecording<Mat3d>("KernelCorrectionMatrix");
	diffusion_body.addBodyStateForRecording<Real>("Laplacian_z");
	diffusion_body.addBodyStateForRecording<Real>("diffusion_dt");

    diffusion_body.addBodyStateForRecording<Vec3d>("FirstOrderCorrectionVectorE");
	
	PeriodicConditionUsingCellLinkedList periodic_condition_y(diffusion_body, diffusion_body.getBodyShapeBounds(), yAxis);
	PeriodicConditionUsingCellLinkedList periodic_condition_x(diffusion_body, diffusion_body.getBodyShapeBounds(), xAxis);
    PeriodicConditionUsingCellLinkedList periodic_condition_z(diffusion_body, diffusion_body.getBodyShapeBounds(), zAxis);

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
	periodic_condition_y.update_cell_linked_list_.exec();
	periodic_condition_x.update_cell_linked_list_.exec();
    periodic_condition_z.update_cell_linked_list_.exec();
    sph_system.initializeSystemConfigurations();
    correct_configuration.exec();
    correct_second_configuration.exec();
    setup_diffusion_initial_condition.exec();
     //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    int ite = 1;
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
    std::cout << transform_tensor << std::endl;
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
                dt = 0.1 *scaling_factor * get_time_step_size.exec();
                diffusion_relaxation.exec(dt);
             
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
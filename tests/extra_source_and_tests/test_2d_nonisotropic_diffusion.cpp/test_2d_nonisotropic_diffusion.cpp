/**
 * @file test_2d_nonisotropic_diffusion.cpp
 * @brief This is a test cases using anisotropic kernel for simulating solid.
 * Particle space is anisotropic in different directions of the beam.
 * @author Xiaojing Tang and Xiangyu Hu
 */
#include "sphinxsys.h"
using namespace SPH;
//------------------------------------------------------------------------------
// global parameters for the case
//------------------------------------------------------------------------------
Real L = 20.0;
Real H = 20.0;
  
int y_num = 60;
Real resolution_ref = H / y_num;
 
Real BL = 6.0 * resolution_ref;
Real BH = 6.0 * resolution_ref;
//----------------------------------------------------------------------
//	Basic parameters for material properties.
//----------------------------------------------------------------------
Real diffusion_coeff = 0.1;
Real rho0 = 1.0;
Real youngs_modulus = 1.0;
Real poisson_ratio = 1.0;
//----------------------------------------------------------------------
//	Geometric shapes used in the case.
//----------------------------------------------------------------------
 
Mat2d decomposed_transform_tensor{ 
   {0.36, 0.0},  
     {0.0, 1.0},
}; 
Mat2d inverse_decomposed_transform_tensor =  decomposed_transform_tensor.inverse();

 
std::vector<Vec2d> diffusion_shape{Vec2d(0.0, 0.0), Vec2d(0.0, H), Vec2d(L, H), Vec2d(L, 0.0), Vec2d(0.0, 0.0)};

class DiffusionBlock : public MultiPolygonShape
{
  public:
    explicit DiffusionBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(diffusion_shape, ShapeBooleanOps::add);
    }
};

 std::vector<Vec2d> boundary_shape{ Vec2d(-BL, -BH), Vec2d(-BL, H + BH), 
                                Vec2d(L + BL, H + BH), Vec2d(L + BL, -BH), Vec2d(-BL, -BH)};

class BoundaryBlock : public ComplexShape
{
  public:
    explicit BoundaryBlock(const std::string &shape_name) : ComplexShape(shape_name)
    {
        MultiPolygon outer_boundary(boundary_shape);
        add<MultiPolygonShape>(outer_boundary, "OuterBoundary");
        MultiPolygon diffusion_polygon_(diffusion_shape);
        subtract<MultiPolygonShape>(diffusion_polygon_);
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
        registerVariable(A1_, "FirstOrderCorrectionVectorA1", [&](size_t i) -> Vec2d { return Eps * Vec2d::Identity(); });
        registerVariable(A2_, "FirstOrderCorrectionVectorA2", [&](size_t i) -> Vec2d { return Eps * Vec2d::Identity(); });
        registerVariable(A3_, "FirstOrderCorrectionVectorA3", [&](size_t i) -> Vec2d { return Eps * Vec2d::Identity(); });
	
        addVariableToWrite<Vec2d>("FirstOrderCorrectionVectorA1");
        addVariableToWrite<Vec2d>("FirstOrderCorrectionVectorA2");
        addVariableToWrite<Vec2d>("FirstOrderCorrectionVectorA3");
    };

    StdLargeVec<Real> phi_;
    StdLargeVec<Vec2d> A1_;
    StdLargeVec<Vec2d> A2_;
    StdLargeVec<Vec2d> A3_;
};

typedef DataDelegateSimple<LaplacianDiffusionParticles> LaplacianSolidDataSimple;
typedef DataDelegateInner<LaplacianDiffusionParticles> LaplacianSolidDataInner; 
typedef DataDelegateContact<LaplacianDiffusionParticles, BaseParticles> LaplacianSolidDataContact;
 

 /*
class NonisotropicKernelCorrectionMatrixComplex : public LocalDynamics, public GeneralDataDelegateContact
{
  public:
  NonisotropicKernelCorrectionMatrixComplex(ComplexRelation &complex_relation, Real alpha = Real(0))
        : LocalDynamics(complex_relation.inner_relation_.getSPHBody()),
         GeneralDataDelegateContact(complex_relation.contact_relation),
      B_(*this->particles_->template registerSharedVariable<Mat2d>("KernelCorrectionMatrix"))
      {
            particles_->registerVariable(neighbour_, "neighbour", [&](size_t i) -> Real { return Real(0.0); });
      };

    virtual ~NonisotropicKernelCorrectionMatrixComplex(){};

  protected:
	  StdLargeVec<Mat2d> &B_;
      StdLargeVec<Real> neighbour_;

	  void initialization(size_t index_i, Real dt = 0.0)
	  {
		  Mat2d local_configuration = Eps * Mat2d::Identity();
		  const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
		  for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
		  {
              size_t index_j = inner_neighborhood.j_[n];
			  Vec2d gradW_ij = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
			  Vec2d r_ji = inner_neighborhood.r_ij_vector_[n];
			  local_configuration -= r_ji * gradW_ij.transpose();

                if(index_i == 1540)
                {
                   neighbour_[index_j] = 1.0;     
                }
		  }
		  B_[index_i] = local_configuration;
	  };

    void interaction(size_t index_i, Real dt = 0.0)
    {
        Mat2d local_configuration = Eps * Mat2d::Identity();
		for (size_t k = 0; k < contact_configuration_.size(); ++k)
		{
			Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
			for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
			{
                size_t index_j = contact_neighborhood.j_[n];
				Vec2d r_ji = contact_neighborhood.r_ij_vector_[n];
				Vec2d gradW_ij = contact_neighborhood.dW_ijV_j_[n] * contact_neighborhood.e_ij_[n];
				local_configuration -= r_ji * gradW_ij.transpose();

                   if(index_i == 1540)
               {
                   neighbour_[index_j] = 1.0;
                }
			}
		}
		B_[index_i] += local_configuration; 
     }; 

	void update(size_t index_i, Real dt)
	{
		Mat2d inverse = B_[index_i].inverse();
		B_[index_i] = inverse;	
     //  B_[index_i] =  Mat2d {{1.03315, 0.0}, {0.0, 1.03315},};               
	}
};


class NonisotropicKernelCorrectionMatrixComplexAC : public LocalDynamics, public LaplacianSolidDataContact
{
  public:
    NonisotropicKernelCorrectionMatrixComplexAC(ContactRelation &contact_relation): 
                LocalDynamics(contact_relation.getSPHBody()),  LaplacianSolidDataContact(contact_relation),
                 B_(particles_->B_), A1_(particles_->A1_), A2_(particles_->A2_), A3_(particles_->A3_)
                 {};
     
    virtual ~NonisotropicKernelCorrectionMatrixComplexAC(){};

  protected:
    StdLargeVec<Mat2d> &B_;
    StdLargeVec<Vec2d> &A1_,&A2_,&A3_;
   
    
    void initialization(size_t index_i, Real dt = 0.0)
    {
        Neighborhood &inner_neighborhood = inner_configuration_[index_i];
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ik
        {
            Vec2d gradW_ikV_k = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
            Vec2d r_ik = inverse_decomposed_transform_tensor * -inner_neighborhood.r_ij_vector_[n];
            
            A1_[index_i] += r_ik[0] * r_ik[0] * (B_[index_i].transpose() * gradW_ikV_k);
            A2_[index_i] += r_ik[1] * r_ik[1] * (B_[index_i].transpose() * gradW_ikV_k);
            A3_[index_i] += r_ik[0] * r_ik[1] * (B_[index_i].transpose() * gradW_ikV_k);
        }
    };

    void interaction(size_t index_i, Real dt = 0.0)
    {
         for (size_t k = 0; k < contact_configuration_.size(); ++k)
        {
            Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
            for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
            { 
                Vec2d gradW_ikV_k = contact_neighborhood.dW_ijV_j_[n] * contact_neighborhood.e_ij_[n];  
                Vec2d r_ik = inverse_decomposed_transform_tensor * -contact_neighborhood.r_ij_vector_[n];
             
                A1_[index_i] += r_ik[0] * r_ik[0] * (B_[index_i].transpose() * gradW_ikV_k);
                A2_[index_i] += r_ik[1] * r_ik[1] * (B_[index_i].transpose() * gradW_ikV_k);
                A3_[index_i] += r_ik[0] * r_ik[1] * (B_[index_i].transpose() * gradW_ikV_k);
            }
        }

    };

	void update(size_t index_i, Real dt = 0.0) {};
};

class LaplacianBodyRelaxation : public LocalDynamics, public LaplacianSolidDataContact
{
  public:
    LaplacianBodyRelaxation(ContactRelation &contact_relation): 
                LocalDynamics(contact_relation.getSPHBody()), LaplacianSolidDataContact(contact_relation),
          pos_(particles_->pos_), B_(particles_->B_), phi_(particles_->phi_), A1_(particles_->A1_), A2_(particles_->A2_), A3_(particles_->A3_)
    {
        particles_->registerVariable(SC_, "FirstOrderCorrectionMatrixSC", [&](size_t i) -> Mat3d { return Eps * Mat3d::Identity(); });
        particles_->registerVariable(E_, "FirstOrderCorrectionVectorE", [&](size_t i) -> Vec2d { return Eps * Vec2d::Identity(); });
        particles_->registerVariable(G_, "FirstOrderCorrectionVectorG", [&](size_t i) -> Vec3d { return Eps * Vec3d::Identity(); });
        particles_->registerVariable(Laplacian_, "Laplacian", [&](size_t i) -> Vec3d { return Vec3d::Zero(); });

        particles_->registerVariable(Laplacian_x, "Laplacian_x", [&](size_t i) -> Real { return Real(0.0); });
        particles_->registerVariable(Laplacian_y, "Laplacian_y", [&](size_t i) -> Real { return Real(0.0); });
        particles_->registerVariable(Laplacian_xy, "Laplacian_xy", [&](size_t i) -> Real { return Real(0.0); });
		particles_->registerVariable(diffusion_dt_, "diffusion_dt", [&](size_t i) -> Real { return Real(0.0); });
 
 
        diffusion_coeff_ = particles_->laplacian_solid_.DiffusivityCoefficient();
    };
    virtual ~LaplacianBodyRelaxation(){};

    StdLargeVec<Vec2d> &pos_;
    StdLargeVec<Mat2d> &B_;
    StdLargeVec<Real> &phi_;

    StdLargeVec<Vec2d> &A1_, &A2_, &A3_;

    StdLargeVec<Mat3d> SC_;
    StdLargeVec<Vec2d> E_;
    StdLargeVec<Vec3d> G_;
    StdLargeVec<Vec3d> Laplacian_;

    StdLargeVec<Real> Laplacian_x, Laplacian_y, Laplacian_xy, diffusion_dt_;

   
    Real diffusion_coeff_;

  protected:
    void initialization(size_t index_i, Real dt = 0.0)
    {   
        Neighborhood &inner_neighborhood = inner_configuration_[index_i];
        Vec2d E_rate = Vec2d::Zero();
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ik
        {
            size_t index_k = inner_neighborhood.j_[n];
            Vec2d gradW_ikV_k = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
          

            E_rate += (phi_[index_k] - phi_[index_i]) * (B_[index_i].transpose() * gradW_ikV_k); // HOW TO DEFINE IT???
        }
        E_[index_i] = E_rate;

         
        Vec3d G_rate = Vec3d::Zero();
        Mat3d SC_rate = Mat3d::Zero();
        Real H_rate = 1.0;
         for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ik
        {
            size_t index_j = inner_neighborhood.j_[n];
            Vec2d r_ij = inverse_decomposed_transform_tensor *  -inner_neighborhood.r_ij_vector_[n];

            Vec2d gradW_ijV_j = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
            Vec3d S_ = Vec3d(r_ij[0] * r_ij[0], r_ij[1] * r_ij[1], r_ij[0] * r_ij[1]);
            H_rate = r_ij.dot(B_[index_i].transpose() * gradW_ijV_j) / pow(r_ij.norm(), 4.0);
			 
            Real FF_ = 2.0 * (phi_[index_j] - phi_[index_i] - r_ij.dot(E_[index_i]));
            G_rate += S_ *H_rate * FF_;
            
             //TO DO
            Vec3d C_ = Vec3d::Zero();
            C_[0] = (r_ij[0] * r_ij[0]-  r_ij.dot(A1_[index_i]));
            C_[1] = (r_ij[1] * r_ij[1]-  r_ij.dot(A2_[index_i]));
            C_[2] = (r_ij[0] * r_ij[1]-  r_ij.dot(A3_[index_i]));
            SC_rate += S_ *H_rate* C_.transpose();   

        }
        
        G_[index_i] = G_rate;
        SC_[index_i] = SC_rate;
        
    
    };

    void interaction(size_t index_i, Real dt = 0.0)
    {
        Mat3d SC_rate_contact = Mat3d::Zero();
        Vec3d G_rate_contact = Vec3d::Zero();
        Real H_rate_contact = 1.0;
         for (size_t k = 0; k < contact_configuration_.size(); ++k)
        {
            Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
            for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
            {
                Vec2d r_ij = inverse_decomposed_transform_tensor * -contact_neighborhood.r_ij_vector_[n];
                Vec2d gradW_ijV_j = contact_neighborhood.dW_ijV_j_[n] * contact_neighborhood.e_ij_[n];

            
                Vec3d S_ = Vec3d(r_ij[0] * r_ij[0], r_ij[1] * r_ij[1], r_ij[0] * r_ij[1]);
                Real FF_ = 2.0 * (0.0 - r_ij.dot(E_[index_i])); ///here when it is periodic boundary condition, should notice the 0.0
                H_rate_contact = r_ij.dot(B_[index_i].transpose() * gradW_ijV_j) / pow(r_ij.norm(), 4.0);
		  
                //TO DO
                Vec3d C_ = Vec3d::Zero();
                C_[0] = (r_ij[0] * r_ij[0]-  r_ij.dot(A1_[index_i]));
                C_[1] = (r_ij[1] * r_ij[1]-  r_ij.dot(A2_[index_i]));
                C_[2] = (r_ij[0] * r_ij[1]-  r_ij.dot(A3_[index_i]));
               
                SC_rate_contact += S_ * H_rate_contact * C_.transpose();
                G_rate_contact += S_ * H_rate_contact * FF_;

            }
			SC_[index_i] += SC_rate_contact;
            G_[index_i] += G_rate_contact;
        }

        Laplacian_[index_i] = diffusion_coeff_ * SC_[index_i].inverse() * G_[index_i];

        Laplacian_x[index_i] = Laplacian_[index_i][0];
        Laplacian_y[index_i] = Laplacian_[index_i][1];
        Laplacian_xy[index_i] = Laplacian_[index_i][2];
		diffusion_dt_[index_i] = Laplacian_[index_i][0] + Laplacian_[index_i][1];
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
          pos_(particles_->pos_), phi_(particles_->phi_){};
    virtual ~DiffusionInitialCondition(){};

    StdLargeVec<Vec2d> &pos_;
    StdLargeVec<Real> &phi_;

  protected:
    void update(size_t index_i, Real dt = 0.0)
    {
         if (pos_[index_i][0] >= (0.5 * L- 0.1*L ) && pos_[index_i][0] <=  (0.5 * L + 0.1*L ))
        {
            if (pos_[index_i][1] >=  (0.5 * H - 0.1*H ) && pos_[index_i][1] <= (0.5* H + 0.1*H ))
            {
                phi_[index_i] = 1.0;
            }
        } 
       
    };
};

*/
 

class GetLaplacianTimeStepSize : public LocalDynamicsReduce<Real, ReduceMin>,
                                 public LaplacianSolidDataSimple
{
  protected:
    Real smoothing_length;

  public:
    GetLaplacianTimeStepSize(SPHBody &sph_body)
        : LocalDynamicsReduce<Real, ReduceMin>(sph_body, MaxReal),
          LaplacianSolidDataSimple(sph_body)
    {
        smoothing_length = sph_body.sph_adaptation_->ReferenceSmoothingLength();
    };

    Real reduce(size_t index_i, Real dt)
    {
        return 0.5 * smoothing_length * smoothing_length / particles_->laplacian_solid_.DiffusivityCoefficient() / Dimensions;
    }

    virtual ~GetLaplacianTimeStepSize(){};
};

//------------------------------------------------------------------------------
// the main program
//------------------------------------------------------------------------------
int main(int ac, char *av[])
{
    //----------------------------------------------------------------------
    //	Build up the environment of a SPHSystem.
    //----------------------------------------------------------------------
      
    BoundingBox system_domain_bounds(Vec2d(-L, -H), Vec2d(2.0 * L, 2.0 * H));
    SPHSystem sph_system(system_domain_bounds, resolution_ref);
    // Tag for run particle relaxation for the initial body fitted distribution.
    sph_system.setRunParticleRelaxation(true);
    // Tag for computation start with relaxed body fitted particles distribution.
    sph_system.setReloadParticles(false);
    // Handle command line arguments and override the tags for particle relaxation and reload.
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.
    //----------------------------------------------------------------------
    SolidBody diffusion_body(sph_system, makeShared<DiffusionBlock>("DiffusionBody"));
    diffusion_body.defineBodyLevelSetShape();
    diffusion_body.defineParticlesAndMaterial<LaplacianDiffusionParticles, LaplacianDiffusionSolid>
            (rho0, diffusion_coeff, youngs_modulus, poisson_ratio);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? diffusion_body.generateParticles<ParticleGeneratorReload>(diffusion_body.getName())
        : diffusion_body.generateParticles<ParticleGeneratorLattice>();
        
    SolidBody boundary_body(sph_system, makeShared<BoundaryBlock>("BoundaryBody"));
    boundary_body.defineComponentLevelSetShape("OuterBoundary");
    boundary_body.defineParticlesAndMaterial<SolidParticles, Solid>();
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? boundary_body.generateParticles<ParticleGeneratorReload>(boundary_body.getName())
        : boundary_body.generateParticles<ParticleGeneratorLattice>();




    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //	Note that the same relation should be defined only once.
    //----------------------------------------------------------------------

    InnerRelation boundary_body_inner(boundary_body);
    InnerRelation diffusion_body_inner(diffusion_body);
    ContactRelation diffusion_body_contact(diffusion_body, {&boundary_body});
    ContactRelation boundary_body_contact(boundary_body, {&diffusion_body});
    ComplexRelation diffusion_boundary_complex(boundary_body_inner, boundary_body_contact);

    //----------------------------------------------------------------------
    // Combined relations built from basic relations
    // which is only used for update configuration.
    //----------------------------------------------------------------------
  
    //----------------------------------------------------------------------
    //----------------------------------------------------------------------
    //	Run particle relaxation for body-fitted distribution if chosen.
    //----------------------------------------------------------------------
    if (sph_system.RunParticleRelaxation())
    {
        InnerRelation diffusion_body_inner(diffusion_body);
        //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        using namespace relax_dynamics;
        SimpleDynamics<RandomizeParticlePosition> random_inserted_body_particles(diffusion_body);
        SimpleDynamics<RandomizeParticlePosition> random_water_body_particles(boundary_body);
        BodyStatesRecordingToVtp write_real_body_states(sph_system.real_bodies_);
        ReloadParticleIO write_real_body_particle_reload_files(sph_system.real_bodies_);
        RelaxationStepLevelSetCorrectionInner relaxation_step_inner(diffusion_body_inner);
        RelaxationStepLevelSetCorrectionComplex relaxation_step_complex(
            ConstructorArgs(boundary_body_inner, "OuterBoundary"), boundary_body_contact);
        //----------------------------------------------------------------------
        //	Particle relaxation starts here.
        //----------------------------------------------------------------------
        random_inserted_body_particles.exec(0.25);
        random_water_body_particles.exec(0.25);
        relaxation_step_inner.SurfaceBounding().exec();
        relaxation_step_complex.SurfaceBounding().exec();
        write_real_body_states.writeToFile(0);

        int ite_p = 0;
        while (ite_p < 1000)
        {
            relaxation_step_inner.exec();
            relaxation_step_complex.exec();
            ite_p += 1;
            if (ite_p % 200 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite_p << "\n";
                write_real_body_states.writeToFile(ite_p);
            }
        }
        std::cout << "The physics relaxation process finish !" << std::endl;

        write_real_body_particle_reload_files.writeToFile(0);

        return 0;
    }
    //----------------------------------------------------------------------
    //	Define the main numerical methods used in the simulation.
    //	Note that there may be data dependence on the constructors of these methods.
    //----------------------------------------------------------------------
    
   /* Dynamics1Level<NonisotropicKernelCorrectionMatrixComplex> correct_configuration(diffusion_body_contact);
	Dynamics1Level<NonisotropicKernelCorrectionMatrixComplexAC> correct_second_configuration(diffusion_body_contact);
    ReduceDynamics<GetLaplacianTimeStepSize> get_time_step_size(diffusion_body);
    Dynamics1Level<LaplacianBodyRelaxation> diffusion_relaxation(diffusion_body_contact);
    SimpleDynamics<DiffusionInitialCondition> setup_diffusion_initial_condition(diffusion_body_inner);
  
    diffusion_body.addBodyStateForRecording<Real>("Phi");
    diffusion_body.addBodyStateForRecording<Real>("neighbour");

  
    diffusion_body.addBodyStateForRecording<Real>("Laplacian_x");
    diffusion_body.addBodyStateForRecording<Real>("Laplacian_y");
    diffusion_body.addBodyStateForRecording<Mat2d>("KernelCorrectionMatrix");
	diffusion_body.addBodyStateForRecording<Real>("Laplacian_xy");
	diffusion_body.addBodyStateForRecording<Real>("diffusion_dt");
      */
      //----------------------------------------------------------------------
    //	Define the methods for I/O operations and observations of the simulation.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp write_states(sph_system.real_bodies_);
     //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
   // correct_configuration.exec();
    //correct_second_configuration.exec();
   // setup_diffusion_initial_condition.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    int ite = 1;
    Real T0 = 1920.0;
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
               // dt =  0.1 * get_time_step_size.exec();
                dt =  10;
                //diffusion_relaxation.exec(dt);
       
             
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
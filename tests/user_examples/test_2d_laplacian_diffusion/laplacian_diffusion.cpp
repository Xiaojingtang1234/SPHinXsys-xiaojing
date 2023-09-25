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

int y_num = 20;  
Real ratio_ = 4.0; 
Real resolution_ref = H / y_num;
Real resolution_ref_large = ratio_ * resolution_ref; 
int x_num = L / resolution_ref_large;    
Vec2d scaling_vector = Vec2d(1.0, 1.0 / ratio_); // scaling_vector for defining the anisotropic kernel
Real scaling_factor = 1.0 / ratio_;              // scaling factor to calculate the time step


Real V_j = resolution_ref_large * resolution_ref;
BoundingBox system_domain_bounds(Vec2d(-L, -H), Vec2d(2.0*L, 2.0*H));
Real BL =  3.0 * resolution_ref_large;
Real BH =   3.0 * resolution_ref;
//----------------------------------------------------------------------
//	Basic parameters for material properties.
//----------------------------------------------------------------------
Real diffusion_coeff = 1.0 ;
Real rho0 = 1.0;
Real youngs_modulus = 1.0; 
Real poisson_ratio = 1.0;
//----------------------------------------------------------------------
//	Geometric shapes used in the case.
//----------------------------------------------------------------------
 std::vector<Vec2d> diffusion_shape
 {Vec2d(0.0, 0.0),Vec2d(0.0, H),Vec2d(L, H),Vec2d(L, 0.0),Vec2d(0.0, 0.0) };
class DiffusionBlock : public MultiPolygonShape
{
  public:
    explicit DiffusionBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(diffusion_shape, ShapeBooleanOps::add);
    }
};

class Boundary: public MultiPolygonShape
{
  public:
    explicit Boundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        std::vector<Vec2d> boundary;
        boundary.push_back(Vec2d(-BL, -BH));
        boundary.push_back(Vec2d(-BL, H + BH));
        boundary.push_back(Vec2d(L + BL, H + BH));
        boundary.push_back(Vec2d(L + BL, -BH));
        boundary.push_back(Vec2d(-BL, -BH));
        multi_polygon_.addAPolygon(boundary, ShapeBooleanOps::add);

 
     
        multi_polygon_.addAPolygon(diffusion_shape, ShapeBooleanOps::sub);
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



class AnisotropicParticleGenerator : public ParticleGenerator
{
  public:
 AnisotropicParticleGenerator(SPHBody &sph_body) : ParticleGenerator(sph_body){};

    virtual void initializeGeometricVariables() override
    {
        // set particles directly
        for (int i = 0; i < x_num; i++)
        {
            for (int j = 0; j < y_num; j++)
            {
                Real x =  (i + 0.5) * resolution_ref_large  ;
                Real y =  (j + 0.5) * resolution_ref ;
                initializePositionAndVolumetricMeasure(Vec2d(x, y), (resolution_ref * resolution_ref_large));
            }
        }
    }
};

   

class AnisotropicParticleGeneratorBoundary : public ParticleGenerator
{
  public:
 AnisotropicParticleGeneratorBoundary(SPHBody &sph_body) : ParticleGenerator(sph_body){};

    virtual void initializeGeometricVariables() override
    {
        // set particles directly
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < y_num + 6; j++)
            {
                Real x = -BL + (i + 0.5) * resolution_ref_large;
                Real y = -BH + (j + 0.5) * resolution_ref ;
                initializePositionAndVolumetricMeasure(Vec2d(x, y), (resolution_ref * resolution_ref_large));
            }
        }


        // set particles directly
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < y_num + 6; j++)
            {
                Real x = (x_num + i +  0.5) * resolution_ref_large;
                Real y = -BH + (j + 0.5) * resolution_ref;
                initializePositionAndVolumetricMeasure(Vec2d(x, y), (resolution_ref * resolution_ref_large));
            }
        }


        // set particles directly
        for (int i = 0; i < x_num; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Real x = (i + 0.5) * resolution_ref_large;
                Real y = -BH + (j + 0.5) * resolution_ref ;
                initializePositionAndVolumetricMeasure(Vec2d(x, y), (resolution_ref * resolution_ref_large));
            }
        }


        // set particles directly
        for (int i = 0; i < x_num; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Real x = (i + 0.5) * resolution_ref_large;
                Real y = (y_num + j + 0.5) * resolution_ref ;
                initializePositionAndVolumetricMeasure(Vec2d(x, y), (resolution_ref * resolution_ref_large));
            }
        }
    }
};

  
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
    NonisotropicKernelCorrectionMatrixInner(BaseInnerRelation &inner_relation, Real alpha = Real(0)) 
    : LocalDynamics(inner_relation.getSPHBody()),
      GeneralDataDelegateInner(inner_relation),
       alpha_(alpha), B_(*particles_->registerSharedVariable<Mat2d>("KernelCorrectionMatrix")),
       pos_(particles_->pos_)
    {     
        particles_->registerVariable(show_neighbor_, "ShowingNeighbor", [&](size_t i) -> Real
                    { return Real(0.0); });
    };
    virtual ~NonisotropicKernelCorrectionMatrixInner(){};

  protected:
     Real alpha_;
    StdLargeVec<Mat2d> &B_;
    StdLargeVec<Vec2d> &pos_;
     StdLargeVec<Real> show_neighbor_;

    void interaction(size_t index_i, Real dt = 0.0)
    {
        Mat2d local_configuration = Eps * Mat2d::Identity();
        const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
        {
            size_t index_j = inner_neighborhood.j_[n];
            Vec2d gradW_ij = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
         
            Vec2d r_ji = pos_[index_i] - pos_[index_j];
            local_configuration -= r_ji * gradW_ij.transpose();
        }
        B_[index_i] = local_configuration;
       
    };
    
    void  update(size_t index_i, Real dt)
    {
    Real det_sqr = alpha_;
    Mat2d inverse = B_[index_i].inverse();
    Real weight1_ = B_[index_i].determinant() / (B_[index_i].determinant() + det_sqr);
    Real weight2_ = det_sqr / (B_[index_i].determinant() + det_sqr);
    B_[index_i] = weight1_ * inverse + weight2_ * Mat2d::Identity();
    }
     
};


class NonisotropicKernelCorrectionMatrixComplex : public NonisotropicKernelCorrectionMatrixInner, public GeneralDataDelegateContactOnly
{
  public:
    NonisotropicKernelCorrectionMatrixComplex(ComplexRelation &complex_relation, Real alpha = Real(0))
    : NonisotropicKernelCorrectionMatrixInner(complex_relation.getInnerRelation(), alpha),
      GeneralDataDelegateContactOnly(complex_relation.getContactRelation())
      {
        for (size_t k = 0; k != contact_particles_.size(); ++k)
        {
        contact_mass_.push_back(&(contact_particles_[k]->mass_));
        contact_Vol_.push_back(&(contact_particles_[k]->Vol_));
        }
      };
     
    virtual ~NonisotropicKernelCorrectionMatrixComplex(){};

  protected: 
    StdVec<StdLargeVec<Real> *> contact_Vol_;
    StdVec<StdLargeVec<Real> *> contact_mass_;

    void interaction(size_t index_i, Real dt = 0.0)
    {
        NonisotropicKernelCorrectionMatrixInner::interaction(index_i, dt);

        Mat2d local_configuration = ZeroData<Mat2d>::value;
        for (size_t k = 0; k < contact_configuration_.size(); ++k)
        {
            Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
            for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
            {
                Vec2d gradW_ij = contact_neighborhood.dW_ijV_j_[n] * contact_neighborhood.e_ij_[n];
                Vec2d r_ji = contact_neighborhood.r_ij_[n] * contact_neighborhood.e_ij_[n];
                local_configuration -= r_ji * gradW_ij.transpose();
             }
        }

        B_[index_i] += local_configuration;
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
            Vec2d r_ik = pos_[index_k] - pos_n_i;
          
            A1_[index_i] += r_ik[0] * r_ik[0] * (B_[index_i].transpose() * gradW_ikV_k);  
            A2_[index_i] += r_ik[1] * r_ik[1] * (B_[index_i].transpose() * gradW_ikV_k);  
            A3_[index_i] += r_ik[0] * r_ik[1] * (B_[index_i].transpose() * gradW_ikV_k);  
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
        particles_->registerVariable(SC_, "FirstOrderCorrectionMatrixSC",  [&](size_t i) -> Mat3d
                    { return Eps * Mat3d::Identity(); });
        particles_->registerVariable(E_, "FirstOrderCorrectionVectorE",  [&](size_t i) -> Vec2d
                    { return Eps * Vec2d::Identity(); });
        particles_->registerVariable(G_, "FirstOrderCorrectionVectorG",  [&](size_t i) -> Vec3d
                    { return Eps * Vec3d::Identity();  });
        particles_->registerVariable(Laplacian_, "Laplacian",  [&](size_t i) -> Vec3d
                    { return Vec3d::Zero(); });

        particles_->registerVariable(Laplacian_x, "Laplacian_x", [&](size_t i) -> Real
                    { return Real(0.0); });
        particles_->registerVariable(Laplacian_y, "Laplacian_y", [&](size_t i) -> Real
                    { return Real(0.0); });
        particles_->registerVariable(Laplacian_xy, "Laplacian_xy", [&](size_t i) -> Real
                    { return Real(0.0); });

        diffusion_coeff_=  particles_->laplacian_solid_.DiffusivityCoefficient();
           
     };
    virtual ~LaplacianBodyRelaxation (){};

    StdLargeVec<Vec2d> &pos_;
    StdLargeVec<Mat2d> &B_;
    StdLargeVec<Real> &phi_;

    StdLargeVec<Vec2d> &A1_;
    StdLargeVec<Vec2d> &A2_;
    StdLargeVec<Vec2d> &A3_;
  
    StdLargeVec<Mat3d> SC_;
    StdLargeVec<Vec2d> E_; 
    StdLargeVec<Vec3d> G_;
    StdLargeVec<Vec3d> Laplacian_; 
 
    StdLargeVec<Real> Laplacian_x; 
    StdLargeVec<Real> Laplacian_y; 
    StdLargeVec<Real> Laplacian_xy; 

    Real diffusion_coeff_;
    StdLargeVec<Real> neigh_;   
  
 protected:
 
    void initialization(size_t index_i, Real dt = 0.0) {};
   
     
    void interaction(size_t index_i, Real dt = 0.0)
    {
       
        Vec2d &pos_n_i = pos_[index_i];   
        Neighborhood &inner_neighborhood = inner_configuration_[index_i];
        Vec2d E_rate = Vec2d::Zero();  
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ik
        {
            size_t index_k = inner_neighborhood.j_[n];
            Vec2d gradW_ikV_k = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
            
            E_rate += (phi_[index_k] - phi_[index_i]) * (B_[index_i].transpose() * gradW_ikV_k);  // HOW TO DEFINE IT???
           
        }
        E_[index_i] =  E_rate;


 
        Mat3d SC_rate = Mat3d::Zero();  
        Vec3d G_rate =  Vec3d::Zero();
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ij
        {
            size_t index_j = inner_neighborhood.j_[n];
            Vec2d r_ij = pos_[index_j] - pos_n_i;
            Vec2d gradW_ijV_j = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];

            Vec3d S_ = Vec3d(r_ij[0] * r_ij[0], r_ij[1] * r_ij[1], r_ij[0] * r_ij[1]);
            Real FF_ = 2.0 * ( phi_[index_j]- phi_[index_i] - r_ij.dot(E_[index_i]));

			// H_rate = (nondimensional_tensor * r_ij).dot(B_[index_i].transpose() * gradW_ijV_j) / pow((nondimensional_tensor * r_ij).norm(), 2.0);

            //TO DO
            Vec3d C_ = Vec3d::Zero();
            C_[0] = (r_ij[0] * r_ij[0] - r_ij.dot(A1_[index_i]));  
            C_[1] = (r_ij[1] * r_ij[1] - r_ij.dot(A2_[index_i]));   
            C_[2] = (r_ij[0] * r_ij[1] - r_ij.dot(A3_[index_i]));  
            
            SC_rate += S_ * C_.transpose(); 
            G_rate +=  S_ * FF_;
        }
        
        SC_[index_i] = SC_rate; 
        G_[index_i] = G_rate;

       Laplacian_[index_i] = diffusion_coeff_ * SC_[index_i].inverse() * G_[index_i]; 

       Laplacian_x[index_i]= Laplacian_[index_i][0];
       Laplacian_y[index_i]= Laplacian_[index_i][1];
       Laplacian_xy[index_i]= Laplacian_[index_i][2];
    };

       void update(size_t index_i, Real dt = 0.0)
    {    
        phi_[index_i] += dt * (Laplacian_[index_i][0] + Laplacian_[index_i][1]); 
    };
 
    
};


class LaplacianBodyRelaxationCheck : public LocalDynamics, public LaplacianSolidDataInner
{
  public:
    LaplacianBodyRelaxationCheck(BaseInnerRelation &inner_relation)
     : LocalDynamics(inner_relation.getSPHBody()),  LaplacianSolidDataInner(inner_relation),
      phi_(particles_->phi_)
      {   
        particles_->registerVariable(d_phi_, "d_phi_",  [&](size_t i) -> Real
                    { return Real(0.0); });
        diffusion_coeff_=  particles_->laplacian_solid_.DiffusivityCoefficient();
      };

       virtual ~LaplacianBodyRelaxationCheck (){};
  
protected:
    StdLargeVec<Real> &phi_;
    StdLargeVec<Real> d_phi_;
    Real diffusion_coeff_;
    
    void initialization(size_t index_i, Real dt = 0.0) {};
   
     
    void interaction(size_t index_i, Real dt = 0.0)
    {
        Real  rate_= 0.0; 
        Neighborhood &inner_neighborhood = inner_configuration_[index_i];  
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ij
        {
            size_t index_j = inner_neighborhood.j_[n]; 
            Real r_ij = inner_neighborhood.r_ij_[n];
        	Real dw_ijV_j_ = inner_neighborhood.dW_ijV_j_[n];

			//Note , here, dwij should be modified in kernel cpp
            rate_ +=  2.0 * (phi_[index_i] - phi_[index_j]) / (r_ij + TinyReal)  * dw_ijV_j_;   
                      
        }

        d_phi_[index_i] = diffusion_coeff_ * rate_; 
        
    };

    void update(size_t index_i, Real dt = 0.0)
    {    
        phi_[index_i] += dt * d_phi_[index_i] ; 
    };
 };

class GradientCheck : public LocalDynamics, public LaplacianSolidDataInner
{
  public:
    GradientCheck(BaseInnerRelation &inner_relation)
     : LocalDynamics(inner_relation.getSPHBody()),  LaplacianSolidDataInner(inner_relation),
      phi_(particles_->phi_), B_(particles_->B_)
      {   
          particles_->registerVariable(Gradient_x, "Gradient_x", [&](size_t i) -> Real
                    { return Real(0.0); });
        particles_->registerVariable(Gradient_y, "Gradient_y", [&](size_t i) -> Real
                    { return Real(0.0); });
      };

       virtual ~GradientCheck (){};
protected:
    StdLargeVec<Real> &phi_;
    StdLargeVec<Mat2d> &B_;
    StdLargeVec<Real> Gradient_x;
    StdLargeVec<Real> Gradient_y;
    
    void initialization(size_t index_i, Real dt = 0.0) {};
     
    void interaction(size_t index_i, Real dt = 0.0)
    {
        Vec2d  rate_=  Vec2d::Zero(); 
        Neighborhood &inner_neighborhood = inner_configuration_[index_i];  
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ij
        {
            size_t index_j = inner_neighborhood.j_[n]; 
            Vec2d gradW_ijV_j = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
            
            rate_ += (phi_[index_j] - phi_[index_i]) * (B_[index_i].transpose() * gradW_ijV_j);                       
        }

        Gradient_x[index_i] =  rate_[0]; 
        Gradient_y[index_i] =  rate_[1]; 
        
    };

    void update(size_t index_i, Real dt = 0.0){ };  
       
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
        phi_[index_i] = pos_[index_i][0] * pos_[index_i][0] + pos_[index_i][1] * pos_[index_i][1];        
    };
    
};

class  GetLaplacianTimeStepSize : public LocalDynamicsReduce<Real, ReduceMin>,
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
        return  0.5 * smoothing_length * smoothing_length 
				/ particles_->laplacian_solid_.DiffusivityCoefficient() / Dimensions;                 
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
    SPHSystem sph_system(system_domain_bounds, resolution_ref_large);
    sph_system.handleCommandlineOptions(ac, av);
    IOEnvironment io_environment(sph_system);
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.
    //----------------------------------------------------------------------
    SolidBody diffusion_body(sph_system, makeShared<DiffusionBlock>("DiffusionBlock"));
    diffusion_body.sph_adaptation_->resetKernel<AnisotropicKernel<KernelWendlandC2>>(scaling_vector);
    diffusion_body.defineParticlesAndMaterial<LaplacianDiffusionParticles, LaplacianDiffusionSolid>(rho0, diffusion_coeff, youngs_modulus, poisson_ratio);    
    diffusion_body.generateParticles<AnisotropicParticleGenerator>();

    SolidBody boundary_body(sph_system, makeShared<Boundary>("Boundary"));
    boundary_body.sph_adaptation_->resetKernel<AnisotropicKernel<KernelWendlandC2>>(scaling_vector);
    boundary_body.defineParticlesAndMaterial<SolidParticles, Solid>();  
    boundary_body.generateParticles<AnisotropicParticleGeneratorBoundary>();

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
    ComplexRelation diffusion_block_complex(diffusion_body, {&boundary_body});
    //----------------------------------------------------------------------
    //	Define the main numerical methods used in the simulation.
    //	Note that there may be data dependence on the constructors of these methods.
    //----------------------------------------------------------------------

   InteractionWithUpdate<NonisotropicKernelCorrectionMatrixComplex> correct_configuration(diffusion_block_complex);  
   InteractionDynamics< NonisotropicKernelCorrectionMatrixInnerAC> correct_second_configuration(diffusion_body_inner_relation);
    ReduceDynamics<GetLaplacianTimeStepSize>  get_time_step_size(diffusion_body);
    Dynamics1Level<LaplacianBodyRelaxation> diffusion_relaxation(diffusion_body_inner_relation);

    SimpleDynamics<DiffusionInitialCondition> setup_diffusion_initial_condition(diffusion_body_inner_relation);

    diffusion_body.addBodyStateForRecording<Real>("Phi"); 
 
    diffusion_body.addBodyStateForRecording<Real>("Laplacian_x");
    diffusion_body.addBodyStateForRecording<Real>("Laplacian_y"); 
    //diffusion_body.addBodyStateForRecording<Mat2d>("KernelCorrectionMatrix");
    diffusion_body.addBodyStateForRecording<Real>("Laplacian_xy");

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
                 dt = scaling_factor *  get_time_step_size.exec();

                //if (ite < 2.0)
                 //{
                    diffusion_relaxation.exec(dt);  
                    
               // }
                   if (ite % 100 == 0)
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

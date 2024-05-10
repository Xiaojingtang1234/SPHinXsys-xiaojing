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
Real L = 20.0;
Real H = 20.0;
  
int y_num = 40;
Real resolution_ref = H / y_num;
 
BoundingBox system_domain_bounds(Vec2d(-L, -H), Vec2d(2.0 * L, 2.0 * H));
Real BL = 6.0 * resolution_ref;
Real BH = 6.0 * resolution_ref;
//----------------------------------------------------------------------
//	Basic parameters for material properties.
//----------------------------------------------------------------------
Real diffusion_coeff = 1.0;
Real rho0 = 1.0;
Real youngs_modulus = 1.0;
Real poisson_ratio = 1.0;
//----------------------------------------------------------------------
//	Geometric shapes used in the case.
//----------------------------------------------------------------------
 
Mat2d transform_tensor{ 
     {0.1, 0.03},  
     {0.03, 0.03},
}; 
Mat2d inverse_decomposed_transform_tensor =  inverseCholeskyDecomposition(transform_tensor);
Mat2d  decomposed_transform_tensor =  inverse_decomposed_transform_tensor.inverse();

std::vector<Vec2d> diffusion_shape{Vec2d(0.0, 0.0), Vec2d(0.0, H), Vec2d(L, H), Vec2d(L, 0.0), Vec2d(0.0, 0.0)};
std::vector<Vec2d> sub_shape{Vec2d(0.2*L, 0.2*H), Vec2d(0.2*L, 0.4*H), Vec2d(0.4*L, 0.4*H), Vec2d(0.4*L, 0.2*H), Vec2d(0.2*L, 0.2*H)};

class DiffusionBlock : public MultiPolygonShape
{
  public:
    explicit DiffusionBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
      // Vec2d center = Vec2d(0.5 * L, 0.5 * H);
        multi_polygon_.addAPolygon(diffusion_shape, ShapeBooleanOps::add);
       // multi_polygon_.addAPolygon(sub_shape, ShapeBooleanOps::sub);
        //multi_polygon_.addACircle(center, 0.2 * L, 200 , ShapeBooleanOps::sub);
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
        registerVariable(A1_, "FirstOrderCorrectionVectorA1", [&](size_t i) -> Vec2d { return Eps * Vec2d::Identity(); });
        registerVariable(A2_, "FirstOrderCorrectionVectorA2", [&](size_t i) -> Vec2d { return Eps * Vec2d::Identity(); });
        registerVariable(A3_, "FirstOrderCorrectionVectorA3", [&](size_t i) -> Vec2d { return Eps * Vec2d::Identity(); });
        registerVariable(compensation_particle_dw_, "CompensationDw", [&](size_t i) -> Vec2d { return Eps * Vec2d::Zero(); });
        registerVariable(compensation_particle_rib_, "CompensationRib", [&](size_t i) -> Vec2d { return Eps * Vec2d::Zero(); });
	    
        
        addVariableToWrite<Vec2d>("FirstOrderCorrectionVectorA1");
        addVariableToWrite<Vec2d>("FirstOrderCorrectionVectorA2");
        addVariableToWrite<Vec2d>("FirstOrderCorrectionVectorA3");
    };

    StdLargeVec<Real> phi_;
    StdLargeVec<Vec2d> A1_;
    StdLargeVec<Vec2d> A2_;
    StdLargeVec<Vec2d> A3_;
    StdLargeVec<Vec2d> compensation_particle_dw_;
    StdLargeVec<Vec2d> compensation_particle_rib_;
};

typedef DataDelegateSimple<LaplacianDiffusionParticles> LaplacianSolidDataSimple;
typedef DataDelegateInner<LaplacianDiffusionParticles> LaplacianSolidDataInner;
typedef DataDelegateContact<LaplacianDiffusionParticles, BaseParticles, DataDelegateEmptyBase> LaplacianSolidDataContactOnly;
typedef DataDelegateComplex<LaplacianDiffusionParticles, BaseParticles> LaplacianSolidDataComplex;
typedef DataDelegateComplex<BaseParticles, BaseParticles>GeneralDataDelegateComplex;


class NonisotropicKernelCorrectionMatrix : public LocalDynamics, public LaplacianSolidDataInner
{
  public:
    NonisotropicKernelCorrectionMatrix(InnerRelation &inner_relation, Real alpha = Real(0))
        : LocalDynamics(inner_relation.getSPHBody()),
		LaplacianSolidDataInner(inner_relation), 
		B_(particles_->B_),pos_(particles_->pos_),
        compensation_particle_dw_(particles_->compensation_particle_dw_),
        compensation_particle_rib_(particles_->compensation_particle_rib_) 
        {
		   particles_->registerVariable(neighbour_, "neighbour", [&](size_t i) -> Real { return Real(0.0); });		 
		   particles_->registerVariable(distance_, "Distance", [&](size_t i) -> Real { return Real(0.0); });		 
        };

    virtual ~NonisotropicKernelCorrectionMatrix(){};

  protected:
	  StdLargeVec<Mat2d> &B_;
      StdLargeVec<Vec2d> &pos_;
      StdLargeVec<Vec2d> &compensation_particle_dw_;
      StdLargeVec<Vec2d> &compensation_particle_rib_;

	  StdLargeVec<Real> neighbour_;
	  StdLargeVec<Real> distance_;

	  void initialization(size_t index_i, Real dt = 0.0)
	  {  

          distance_[index_i] = sph_body_.body_shape_->findSignedDistance(pos_[index_i]);//this is right
          
          if (index_i ==3577)
           {
             
             std::cout<<"Distance"<< distance_[index_i]<<std::endl;
             std::cout<<"sph_body_.body_shape_->findNormalDirection(pos_[index_i]);"
                        <<sph_body_.body_shape_->findNormalDirection(pos_[index_i])<<std::endl;

           }
            if(fabs( distance_[index_i])< resolution_ref)
           {   
               //Vec2d norm_ =  sph_body_.body_shape_->findNormalDirection(pos_[index_i]); // from i to j. the previous rij is j to i
              
               const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
               Vec2d compensation = Eps * Vec2d::Identity();
                for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
                {
                    Vec2d gradW_ij = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];  
                    compensation += gradW_ij;
                }  
                 compensation_particle_dw_[index_i] = -compensation; //THIS IS RIGHT  
                Vec2d norm_ = compensation_particle_dw_[index_i] / (compensation_particle_dw_[index_i].norm() + TinyReal);
                compensation_particle_rib_[index_i] =  norm_ * distance_[index_i];    
    	  }
		 
	  };

      void interaction(size_t index_i, Real dt = 0.0)
      {
		  Mat2d local_configuration = Eps * Mat2d::Identity();
		  const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
		  for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
		  {
			  Vec2d gradW_ij = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
			  Vec2d r_ji =  inner_neighborhood.r_ij_vector_[n];
			  local_configuration -= r_ji * gradW_ij.transpose();
                
		  }
           local_configuration -= compensation_particle_rib_[index_i] 
                                           * compensation_particle_dw_[index_i].transpose(); 
		   B_[index_i] = local_configuration;
          
	  };

	void update(size_t index_i, Real dt)
	{
		Mat2d inverse = B_[index_i].inverse();
		B_[index_i] = inverse;	
                    
	} 
};



class NonisotropicKernelCorrectionMatrixAC : public LocalDynamics, public LaplacianSolidDataInner
{
  public:
    NonisotropicKernelCorrectionMatrixAC(InnerRelation &inner_relation): 
                LocalDynamics(inner_relation.getSPHBody()), LaplacianSolidDataInner(inner_relation),
                 B_(particles_->B_), A1_(particles_->A1_), A2_(particles_->A2_), A3_(particles_->A3_),
                  compensation_particle_dw_(particles_->compensation_particle_dw_),
                 compensation_particle_rib_(particles_->compensation_particle_rib_) { };
   
                 
    virtual ~NonisotropicKernelCorrectionMatrixAC(){};

  protected:
    StdLargeVec<Mat2d> &B_;
    StdLargeVec<Vec2d> &A1_,&A2_,&A3_;
    StdLargeVec<Vec2d> &compensation_particle_dw_;
    StdLargeVec<Vec2d> &compensation_particle_rib_;
    void initialization(size_t index_i, Real dt = 0.0)
    {
        Neighborhood &inner_neighborhood = inner_configuration_[index_i];
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ik
        {
            Vec2d gradW_ikV_k = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
            Vec2d r_ik = -inner_neighborhood.r_ij_vector_[n];
            
            A1_[index_i] += r_ik[0] * r_ik[0] * (B_[index_i].transpose() * gradW_ikV_k);
            A2_[index_i] += r_ik[1] * r_ik[1] * (B_[index_i].transpose() * gradW_ikV_k);
            A3_[index_i] += r_ik[0] * r_ik[1] * (B_[index_i].transpose() * gradW_ikV_k);
        }
    };

    void interaction(size_t index_i, Real dt = 0.0)
    {  
            
            A1_[index_i] +=  compensation_particle_rib_[index_i][0] * compensation_particle_rib_[index_i][0] 
                            * (B_[index_i].transpose() * compensation_particle_dw_[index_i]);
             
            A2_[index_i] +=  compensation_particle_rib_[index_i][1] * compensation_particle_rib_[index_i][1] 
                            * (B_[index_i].transpose() * compensation_particle_dw_[index_i]);
                              
            A3_[index_i] +=  compensation_particle_rib_[index_i][2] * compensation_particle_rib_[index_i][2] 
                            * (B_[index_i].transpose() * compensation_particle_dw_[index_i]);

    };

	void update(size_t index_i, Real dt = 0.0) {};
};


 
class LaplacianBodyRelaxation : public LocalDynamics, public LaplacianSolidDataInner
{
  public:
    LaplacianBodyRelaxation(InnerRelation &inner_relation): 
          LocalDynamics(inner_relation.getSPHBody()), LaplacianSolidDataInner(inner_relation),
          pos_(particles_->pos_), B_(particles_->B_), phi_(particles_->phi_), A1_(particles_->A1_), A2_(particles_->A2_), A3_(particles_->A3_)
        , compensation_particle_dw_(particles_->compensation_particle_dw_),
         compensation_particle_rib_(particles_->compensation_particle_rib_)
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
    StdLargeVec<Vec2d> &compensation_particle_dw_;
    StdLargeVec<Vec2d> &compensation_particle_rib_;

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

         //Vec2d pos_b = pos_[index_i] - compensation_particle_rib_[index_i];  //need to confirm
         //E_rate += (3.0 * pos_b[0] * pos_b[0] - phi_[index_i]) * (B_[index_i].transpose() * compensation_particle_dw_[index_i]); // HOW TO DEFINE IT???
          
        E_[index_i] = E_rate;

         
        Vec3d G_rate = Vec3d::Zero();
        Mat3d SC_rate = Mat3d::Zero();
        Real H_rate = 1.0;
         for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ik
        {
            size_t index_j = inner_neighborhood.j_[n];
            Vec2d r_ij =   -inner_neighborhood.r_ij_vector_[n];

            Vec2d gradW_ijV_j = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
            Vec3d S_ = Vec3d(r_ij[0] * r_ij[0], r_ij[1] * r_ij[1], r_ij[0] * r_ij[1]);
            H_rate = r_ij.dot(B_[index_i].transpose() * gradW_ijV_j) / pow(r_ij.norm(), 4.0);
			 
            Real FF_ = 2.0 * (phi_[index_j] - phi_[index_i] - r_ij.dot(E_[index_i]));
            G_rate += S_ *H_rate * FF_;
            
             //TO DO
            Vec3d C_ = Vec3d::Zero();
            C_[0] = (r_ij[0] * r_ij[0]- r_ij.dot(A1_[index_i]));
            C_[1] = (r_ij[1] * r_ij[1]- r_ij.dot(A2_[index_i]));
            C_[2] = (r_ij[0] * r_ij[1]- r_ij.dot(A3_[index_i]));
            SC_rate += S_ *H_rate* C_.transpose();   

        }
        
        G_[index_i] = G_rate;
        SC_[index_i] = SC_rate;
     
    };

    void interaction(size_t index_i, Real dt = 0.0)
    {
        Mat3d SC_rate_contact = Mat3d::Zero();
        Vec3d G_rate_contact = Vec3d::Zero();
        Vec2d r_ib = -compensation_particle_rib_[index_i];
        
        Vec3d S_compen =Vec3d(r_ib[0] * r_ib[0], r_ib[1] * r_ib[1], r_ib[0] * r_ib[1]);
        Real H_rate_compen = r_ib.dot(B_[index_i].transpose() * compensation_particle_dw_[index_i]) 
                              / (pow(r_ib.norm(), 4.0) + TinyReal);
        Vec3d  C_compen = Vec3d::Zero();
               C_compen[0] = (r_ib[0] * r_ib[0]- r_ib.dot(A1_[index_i]));
               C_compen[1] = (r_ib[1] * r_ib[1]- r_ib.dot(A2_[index_i]));
               C_compen[2] = (r_ib[0] * r_ib[1]- r_ib.dot(A3_[index_i]));   
        
         //Vec2d pos_b = pos_[index_i] - compensation_particle_rib_[index_i]; //need to confirm
        //Real FF_compen = 2.0 * (3.0 * pos_b[0] * pos_b[0] - phi_[index_i] - r_ib.dot(E_[index_i]));
          Real FF_compen = 2.0 * (0.0 - r_ib.dot(E_[index_i]));

        
        SC_rate_contact = S_compen * H_rate_compen * C_compen.transpose();   
        G_rate_contact = S_compen *H_rate_compen * FF_compen;

        SC_[index_i] += SC_rate_contact;
        G_[index_i] += G_rate_contact;
        
 
              
        Laplacian_[index_i] = diffusion_coeff_ * SC_[index_i].inverse() * G_[index_i];

        Laplacian_x[index_i] = Laplacian_[index_i][0];
        Laplacian_y[index_i] = Laplacian_[index_i][1];
        Laplacian_xy[index_i] = Laplacian_[index_i][2];
	 
         Mat2d Laplacian_transform = Mat2d { 
                  { Laplacian_x[index_i],  0.5 * Laplacian_xy[index_i]},  
                     { 0.5 * Laplacian_xy[index_i],  Laplacian_y[index_i]},
                 }; 
         Mat2d Laplacian_transform_noniso = decomposed_transform_tensor * Laplacian_transform * decomposed_transform_tensor.transpose() ;
         diffusion_dt_[index_i] =  Laplacian_transform_noniso(0,0) + Laplacian_transform_noniso(1,1);
 
    };

    void update(size_t index_i, Real dt = 0.0)
    {
        phi_[index_i] += dt *  diffusion_dt_[index_i];
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
       
         //phi_[index_i] =  3.0 * pos_[index_i][0] * pos_[index_i][0];
            
    };
};

class GradientCheck : public LocalDynamics, public LaplacianSolidDataInner
{
  public:
    GradientCheck(BaseInnerRelation &inner_relation)
        : LocalDynamics(inner_relation.getSPHBody()), LaplacianSolidDataInner(inner_relation),
         B_(particles_->B_), pos_(particles_->pos_), 
         compensation_particle_dw_(particles_->compensation_particle_dw_),
         compensation_particle_rib_(particles_->compensation_particle_rib_)
    {
        particles_->registerVariable(Gradient, "Gradient", [&](size_t i) -> Real { return Real(0.0); });
     };

    virtual ~GradientCheck(){};

  protected:
 
    StdLargeVec<Mat2d> &B_;
    StdLargeVec<Vec2d> &pos_;
   
    StdLargeVec<Vec2d> &compensation_particle_dw_;
    StdLargeVec<Vec2d> &compensation_particle_rib_;
    StdLargeVec<Real> Gradient;


    void initialization(size_t index_i, Real dt = 0.0){};

    void interaction(size_t index_i, Real dt = 0.0)
    {
        Real rate_ = 0.0;
        Neighborhood &inner_neighborhood = inner_configuration_[index_i];
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ij
        {
            size_t index_j = inner_neighborhood.j_[n];
            Vec2d gradW_ijV_j = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];

            rate_ += (pos_[index_j] - pos_[index_i]).dot(B_[index_i].transpose() * gradW_ijV_j);
        }

        rate_ += - compensation_particle_rib_[index_i].dot(B_[index_i].transpose() * compensation_particle_dw_[index_i]);

        Gradient[index_i] = rate_;
         
    };

    void update(size_t index_i, Real dt = 0.0){};
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
        return 0.5 * smoothing_length * smoothing_length / particles_->laplacian_solid_.DiffusivityCoefficient() / Dimensions;
    }

    virtual ~GetLaplacianTimeStepSize(){};
};



class DiffusionyRelaxationCheck : public LocalDynamics, public LaplacianSolidDataInner
{
  public:
    DiffusionyRelaxationCheck(BaseInnerRelation &inner_relation)
        : LocalDynamics(inner_relation.getSPHBody()), LaplacianSolidDataInner(inner_relation),
          phi_(particles_->phi_)
    {
        particles_->registerVariable(d_phi_, "d_phi_", [&](size_t i) -> Real { return Real(0.0); });
        diffusion_coeff_ = 1.0;
    };

    virtual ~DiffusionyRelaxationCheck(){};

  protected:
    StdLargeVec<Real> &phi_;
    StdLargeVec<Real> d_phi_;
    Real diffusion_coeff_;

    void initialization(size_t index_i, Real dt = 0.0){};

    void interaction(size_t index_i, Real dt = 0.0)
    {
        Real rate_ = 0.0;
        Neighborhood &inner_neighborhood = inner_configuration_[index_i];
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ij
        {
            size_t index_j = inner_neighborhood.j_[n];
            Real r_ij = inner_neighborhood.r_ij_[n];
            Real dw_ijV_j_ = inner_neighborhood.dW_ijV_j_[n];

            //Note , here, dwij should be modified in kernel cpp
            rate_ += 2.0 * (phi_[index_i] - phi_[index_j]) / (r_ij + TinyReal) * dw_ijV_j_;
        }

        d_phi_[index_i] = diffusion_coeff_ * rate_;
    };

    void update(size_t index_i, Real dt = 0.0)
    {
        phi_[index_i] += dt * d_phi_[index_i];
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
	 
   /** Tag for run particle relaxation for the initial body fitted distribution. */
    sph_system.setRunParticleRelaxation(false);
    /** Tag for computation start with relaxed body fitted particles distribution. */
    sph_system.setReloadParticles(true);

    sph_system.handleCommandlineOptions(ac, av);
    IOEnvironment io_environment(sph_system);

    //----------------------------------------------- -----------------------
    //	Run particle relaxation for body-fitted distribution if chosen.
    //----------------------------------------------------------------------
    if (sph_system.RunParticleRelaxation())
    { 
        SolidBody diffusion_block(sph_system, makeShared<DiffusionBlock>("DiffusionBlock"));
        diffusion_block.defineBodyLevelSetShape()->correctLevelSetSign()->writeLevelSet(io_environment);
        diffusion_block.defineParticlesAndMaterial<SolidParticles, Solid>();
        diffusion_block.generateParticles<ParticleGeneratorLattice>();
 
        InnerRelation diffusion_block_inner_relation(diffusion_block);
         //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        /** Random reset the insert body particle position. */
        SimpleDynamics<RandomizeParticlePosition> random_diffusion_body_particles(diffusion_block);
        BodyStatesRecordingToVtp write_real_body_states(io_environment, sph_system.real_bodies_);
        ReloadParticleIO write_real_body_particle_reload_files(io_environment, sph_system.real_bodies_);
        /** A  Physics relaxation step. */
        relax_dynamics::RelaxationStepInner relaxation_step_inner(diffusion_block_inner_relation);
        //----------------------------------------------------------------------
        //	Particle relaxation starts here.
        //----------------------------------------------------------------------
        random_diffusion_body_particles.exec(0.25);
        relaxation_step_inner.SurfaceBounding().exec();
        write_real_body_states.writeToFile(0);
        //----------------------------------------------------------------------
        //	Relax particles of the insert body.
        //----------------------------------------------------------------------
        int ite_p = 0;
        while (ite_p < 1000)
        {
            relaxation_step_inner.exec();
 
            ite_p += 1;
            if (ite_p % 200 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite_p << "\n";
                write_real_body_states.writeToFile(ite_p);
            }
        }
        std::cout << "The physics relaxation process finished !" << std::endl;
        /** Output results. */
        write_real_body_particle_reload_files.writeToFile(0);
        return 0;
    }
    

    //----------------------------------------------------------------------
    //	Creating body, materials and particles.
    //----------------------------------------------------------------------
    SolidBody diffusion_body(sph_system, makeShared<DiffusionBlock>("DiffusionBody"));
    diffusion_body.defineParticlesAndMaterial<LaplacianDiffusionParticles, LaplacianDiffusionSolid>(rho0, diffusion_coeff, youngs_modulus, poisson_ratio);
     (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? diffusion_body.generateParticles<ParticleGeneratorReload>(io_environment, "DiffusionBlock")
        : diffusion_body.generateParticles<ParticleGeneratorLattice>();

    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //  Generally, we first define all the inner relations, then the contact relations.
    //  At last, we define the complex relaxations by combining previous defined
    //  inner and contact relations.
    //----------------------------------------------------------------------
    InnerRelation diffusion_body_inner_relation(diffusion_body);
    //----------------------------------------------------------------------
    //	Define the main numerical methods used in the simulation.
    //	Note that there may be data dependence on the constructors of these methods.
    //----------------------------------------------------------------------
    
    

	Dynamics1Level<NonisotropicKernelCorrectionMatrix> correct_configuration(diffusion_body_inner_relation);
	Dynamics1Level<NonisotropicKernelCorrectionMatrixAC> correct_second_configuration(diffusion_body_inner_relation);
    ReduceDynamics<GetLaplacianTimeStepSize> get_time_step_size(diffusion_body);
    Dynamics1Level<LaplacianBodyRelaxation> diffusion_relaxation(diffusion_body_inner_relation);
    Dynamics1Level<GradientCheck> gradient_check(diffusion_body_inner_relation);

    SimpleDynamics<DiffusionInitialCondition> setup_diffusion_initial_condition(diffusion_body_inner_relation);
  
    diffusion_body.addBodyStateForRecording<Real>("Phi"); 
    diffusion_body.addBodyStateForRecording<Real>("diffusion_dt"); 
    diffusion_body.addBodyStateForRecording<Mat2d>("KernelCorrectionMatrix");
 

    diffusion_body.addBodyStateForRecording<Vec2d>("CompensationDw");
    diffusion_body.addBodyStateForRecording<Vec2d>("CompensationRib");
    diffusion_body.addBodyStateForRecording<Real>("Distance");
    diffusion_body.addBodyStateForRecording<Real>("Gradient");

    //----------------------------------------------------------------------
    //	Define the methods for I/O operations and observations of the simulation.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp write_states(io_environment, sph_system.real_bodies_);
    
    
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
    int ite = 1;
    Real T0 = 1920.0;
    Real end_time = T0;
    Real Output_Time = 0.03 * end_time;
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
    gradient_check.exec(0.0);
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
              
                dt = get_time_step_size.exec();
                diffusion_relaxation.exec(dt);

                if(ite < 10)
                {
                     write_states.writeToFile(ite);
                }
       
             
                if (ite % 100 == 0)
                {
                          write_states.writeToFile(ite);
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
       write_states.writeToFile(ite);
       
        TickCount t3 = TickCount::now();
        interval += t3 - t2;
    }
    TickCount t4 = TickCount::now();

    TimeInterval tt;
    tt = t4 - t1 - interval;
    std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;

    return 0;
}

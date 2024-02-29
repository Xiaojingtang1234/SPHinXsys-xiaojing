/**
 * @file 	excitation-contraction.cpp
 * @brief 	This is the case studying the electromechanics on a biventricular heart model in 3D.
 * @author 	Chi Zhang and Xiangyu Hu
 * 			Unit :
 *			time t = ms = 12.9 [-]
 * 			length l = mm
 * 			mass m = g
 *			density rho = g * (mm)^(-3)
 *			Pressure pa = g * (mm)^(-1) * (ms)^(-2)
 *			diffusion d = (mm)^(2) * (ms)^(-2)
 */
#include "sphinxsys.h" // SPHinXsys Library.
using namespace SPH;   // Namespace cite here.
/** Geometry parameter. */
/** Set the file path to the stl file. */
std::string full_path_to_stl_file = "./input/heart-new.stl";
Real length_scale = 1.0;
Real time_scale = 1.0 / 12.9;
Real stress_scale = 1.0e-6;
/** Parameters and physical properties. */
Vec3d domain_lower_bound(-65.0 * length_scale, -85.0 * length_scale, -45.0 * length_scale);
Vec3d domain_upper_bound(45.0 * length_scale, 15.0 * length_scale, 45.0 * length_scale);

Vec3d domain_lower_bound_previous(-55.0 * length_scale, -75.0 * length_scale, -35.0 * length_scale);
Vec3d domain_upper_bound_previous(35.0 * length_scale, 5.0 * length_scale, 35.0 * length_scale);


Real dp_0 = (domain_upper_bound_previous[0] - domain_lower_bound_previous[0]) / 45.0; /**< Initial particle spacing. */
/** Domain bounds of the system. */
BoundingBox system_domain_bounds(domain_lower_bound, domain_upper_bound);

/** Material properties. */
Real rho0_s = 1.06e-3;
/** Active stress factor */
Real k_a = 100 * stress_scale;
Real a0[4] = {Real(496.0) * stress_scale, Real(15196.0) * stress_scale, Real(3283.0) * stress_scale, Real(662.0) * stress_scale};
Real b0[4] = {Real(7.209), Real(20.417), Real(11.176), Real(9.466)};
/** reference stress to achieve weakly compressible condition */
Real poisson = 0.4995;
Real bulk_modulus = 2.0 * a0[0] * (1.0 + poisson) / (3.0 * (1.0 - 2.0 * poisson));
/** Electrophysiology parameters. */
std::array<std::string, 1> species_name_list{"Phi"};
Real diffusion_coeff = 0.8;
Real bias_coeff = 0.0;
/** Electrophysiology parameters. */
Real c_m = 1.0;
Real k = 8.0;
Real a = 0.01;
Real b = 0.15;
Real mu_1 = 0.2;
Real mu_2 = 0.3;
Real epsilon = 0.002;
/** Fibers and sheet. */
Vec3d fiber_direction(1.0, 0.0, 0.0);
Vec3d sheet_direction(0.0, 1.0, 0.0);

 
Mat3d decomposed_transform_tensor{ 
     {0.8944, 0.0, 0.0},  
     {0.0, 0.8944, 0.0},
     {0.0, 0.0, 0.8944},
}; 
Mat3d inverse_decomposed_transform_tensor =  decomposed_transform_tensor.inverse();


Vecd halfsize_boundary(50.0 * length_scale, 45.0 * length_scale, 40.0 * length_scale);
Vecd translation_boundary(-10.0 * length_scale, -35.0 * length_scale, 0.0 * length_scale);

class BoundaryModel : public ComplexShape
{
  public:
    explicit BoundaryModel(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<TransformShape<GeometricShapeBox>>(Transform(translation_boundary), halfsize_boundary);
        Vecd translation(-53.5 * length_scale, -70.0 * length_scale, -32.5 * length_scale);
        subtract<TriangleMeshShapeSTL>(full_path_to_stl_file, translation, length_scale);
    }
};
/**
 * Define heart geometry
 */
class Heart : public ComplexShape
{
  public:
    explicit Heart(const std::string &shape_name) : ComplexShape(shape_name)
    {
        Vecd translation(-53.5 * length_scale, -70.0 * length_scale, -32.5 * length_scale);
        add<TriangleMeshShapeSTL>(full_path_to_stl_file, translation, length_scale);
    }
};
//----------------------------------------------------------------------
//	Setup diffusion material properties.
//----------------------------------------------------------------------
class FiberDirectionDiffusion : public DiffusionReaction<LocallyOrthotropicMuscle>
{
  public:
    FiberDirectionDiffusion() : DiffusionReaction<LocallyOrthotropicMuscle>(
                                    {"Phi"}, SharedPtr<NoReaction>(),
                                    rho0_s, bulk_modulus, fiber_direction, sheet_direction, a0, b0)
    {
        initializeAnDiffusion<IsotropicDiffusion>("Phi", "Phi", diffusion_coeff);
    };
};
using FiberDirectionDiffusionParticles = DiffusionReactionParticles<ElasticSolidParticles, FiberDirectionDiffusion>;
/** Set diffusion relaxation method. */
class DiffusionRelaxation
    : public DiffusionRelaxationRK2<
          DiffusionRelaxationInner<FiberDirectionDiffusionParticles>>
{
  public:
    explicit DiffusionRelaxation(BaseInnerRelation &inner_relation)
        : DiffusionRelaxationRK2(inner_relation){};
    virtual ~DiffusionRelaxation(){};
};
/** Imposing diffusion boundary condition */
class DiffusionBCs
    : public DiffusionReactionSpeciesConstraint<BodyPartByParticle, FiberDirectionDiffusionParticles>
{
  public:
    DiffusionBCs(BodyPartByParticle &body_part, const std::string &species_name)
        : DiffusionReactionSpeciesConstraint<BodyPartByParticle, FiberDirectionDiffusionParticles>(body_part, species_name),
          pos_(particles_->pos_){};
    virtual ~DiffusionBCs(){};

    void update(size_t index_i, Real dt = 0.0)
    {
        Vecd dist_2_face = sph_body_.body_shape_->findNormalDirection(pos_[index_i]);
        Vecd face_norm = dist_2_face / (dist_2_face.norm() + 1.0e-15);

        Vecd center_norm = pos_[index_i] / (pos_[index_i].norm() + 1.0e-15);

        Real angle = face_norm.dot(center_norm);
        if (angle >= 0.0)
        {
            species_[index_i] = 1.0;
        }
        else
        {
            if (pos_[index_i][1] < -sph_body_.sph_adaptation_->ReferenceSpacing())
                species_[index_i] = 0.0;
        }
    };

  protected:
    StdLargeVec<Vecd> &pos_;
};
/** Compute Fiber and Sheet direction after diffusion */
class ComputeFiberAndSheetDirections
    : public DiffusionBasedMapping<FiberDirectionDiffusionParticles>
{
  protected:
    DiffusionReaction<LocallyOrthotropicMuscle> &diffusion_reaction_material_;
    size_t phi_;
    Real beta_epi_, beta_endo_;
    /** We define the centerline vector, which is parallel to the ventricular centerline and pointing  apex-to-base.*/
    Vecd center_line_;

  public:
    explicit ComputeFiberAndSheetDirections(SPHBody &sph_body)
        : DiffusionBasedMapping<FiberDirectionDiffusionParticles>(sph_body),
          diffusion_reaction_material_(particles_->diffusion_reaction_material_)

    {
        phi_ = diffusion_reaction_material_.AllSpeciesIndexMap()["Phi"];
        center_line_ = Vecd(0.0, 1.0, 0.0);
        beta_epi_ = -(70.0 / 180.0) * M_PI;
        beta_endo_ = (80.0 / 180.0) * M_PI;
    };
    virtual ~ComputeFiberAndSheetDirections(){};

    void update(size_t index_i, Real dt = 0.0)
    {
        /**
         * Ref: original doi.org/10.1016/j.euromechsol.2013.10.009
         * 		Present  doi.org/10.1016/j.cma.2016.05.031
         */
        /** Probe the face norm from Levelset field. */
        Vecd dist_2_face = sph_body_.body_shape_->findNormalDirection(pos_[index_i]);
        Vecd face_norm = dist_2_face / (dist_2_face.norm() + 1.0e-15);
        Vecd center_norm = pos_[index_i] / (pos_[index_i].norm() + 1.0e-15);
        if (face_norm.dot(center_norm) <= 0.0)
        {
            face_norm = -face_norm;
        }
        /** Compute the centerline's projection on the plane orthogonal to face norm. */
        Vecd circumferential_direction = getCrossProduct(center_line_, face_norm);
        Vecd cd_norm = circumferential_direction / (circumferential_direction.norm() + 1.0e-15);
        /** The rotation angle is given by beta = (beta_epi - beta_endo) phi + beta_endo */
        Real beta = (beta_epi_ - beta_endo_) * all_species_[phi_][index_i] + beta_endo_;
        /** Compute the rotation matrix through Rodrigues rotation formulation. */
        Vecd f_0 = cos(beta) * cd_norm + sin(beta) * getCrossProduct(face_norm, cd_norm) +
                   face_norm.dot(cd_norm) * (1.0 - cos(beta)) * face_norm;

        if (pos_[index_i][1] < -sph_body_.sph_adaptation_->ReferenceSpacing())
        {
            diffusion_reaction_material_.local_f0_[index_i] = f_0 / (f_0.norm() + 1.0e-15);
            diffusion_reaction_material_.local_s0_[index_i] = face_norm;
        }
        else
        {
            diffusion_reaction_material_.local_f0_[index_i] = Vecd::Zero();
            diffusion_reaction_material_.local_s0_[index_i] = Vecd::Zero();
        }
    };
};
//	define shape parameters which will be used for the constrained body part.
class MuscleBaseShapeParameters : public TriangleMeshShapeBrick::ShapeParameters
{
  public:
    MuscleBaseShapeParameters() : TriangleMeshShapeBrick::ShapeParameters()
    {
        Real l = domain_upper_bound_previous[0] - domain_lower_bound_previous[0];
        Real w = domain_upper_bound_previous[2] - domain_lower_bound_previous[2];
        halfsize_ = Vec3d(0.5 * l, 1.0 * dp_0, 0.5 * w);
        resolution_ = 20;
        translation_ = Vec3d(-10.0 * length_scale, -1.0 * dp_0, 0.0);
    }
};
//	application dependent initial condition
class ApplyStimulusCurrentSI
    : public electro_physiology::ElectroPhysiologyInitialCondition
{
  protected:
    size_t voltage_;

  public:
    explicit ApplyStimulusCurrentSI(SPHBody &sph_body)
        : electro_physiology::ElectroPhysiologyInitialCondition(sph_body)
    {
        voltage_ = particles_->diffusion_reaction_material_.AllSpeciesIndexMap()["Voltage"];
    };

    void update(size_t index_i, Real dt)
    {
        if (-30.0 * length_scale <= pos_[index_i][0] && pos_[index_i][0] <= -15.0 * length_scale)
        {
            if (-2.0 * length_scale <= pos_[index_i][1] && pos_[index_i][1] <= 0.0)
            {
                if (-3.0 * length_scale <= pos_[index_i][2] && pos_[index_i][2] <= 3.0 * length_scale)
                {
                    all_species_[voltage_][index_i] = 0.92;
                }
            }
        }
    };
};
/**
 * application dependent initial condition
 */
class ApplyStimulusCurrentSII
    : public electro_physiology::ElectroPhysiologyInitialCondition
{
  protected:
    size_t voltage_;

  public:
    explicit ApplyStimulusCurrentSII(SPHBody &sph_body)
        : electro_physiology::ElectroPhysiologyInitialCondition(sph_body)
    {
        voltage_ = particles_->diffusion_reaction_material_.AllSpeciesIndexMap()["Voltage"];
    };

    void update(size_t index_i, Real dt)
    {
        if (0.0 <= pos_[index_i][0] && pos_[index_i][0] <= 6.0 * length_scale)
        {
            if (-6.0 * length_scale <= pos_[index_i][1])
            {
                if (12.0 * length_scale <= pos_[index_i][2])
                {
                    all_species_[voltage_][index_i] = 0.95;
                }
            }
        }
    };
};
/**
 * define observer particle generator.
 */
class HeartObserverParticleGenerator : public ObserverParticleGenerator
{
  public:
    explicit HeartObserverParticleGenerator(SPHBody &sph_body) : ObserverParticleGenerator(sph_body)
    {
        /** position and volume. */
        positions_.push_back(Vecd(-45.0 * length_scale, -30.0 * length_scale, 0.0));
        positions_.push_back(Vecd(0.0, -30.0 * length_scale, 26.0 * length_scale));
        positions_.push_back(Vecd(-30.0 * length_scale, -50.0 * length_scale, 0.0));
        positions_.push_back(Vecd(0.0, -50.0 * length_scale, 20.0 * length_scale));
        positions_.push_back(Vecd(0.0, -70.0 * length_scale, 0.0));
    }
};

using Mat6d = Eigen::Matrix<Real, 6, 6>;
using Vec6d = Eigen::Matrix<Real, 6, 1>;
 
typedef DataDelegateComplex<BaseParticles, BaseParticles>GeneralDataDelegateComplex;
class NonisotropicKernelCorrectionMatrixComplex : public LocalDynamics, public GeneralDataDelegateComplex
{
  public:
    NonisotropicKernelCorrectionMatrixComplex(ComplexRelation &complex_relation, Real alpha = Real(0))
        : LocalDynamics(complex_relation.getInnerRelation().getSPHBody()),
		GeneralDataDelegateComplex(complex_relation), 
		B_(*particles_->registerSharedVariable<Mat3d>("KernelCorrectionMatrix")) {};

    virtual ~NonisotropicKernelCorrectionMatrixComplex(){};

  protected:
	  StdLargeVec<Mat3d> &B_;
 
	  void initialization(size_t index_i, Real dt = 0.0)
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

    void interaction(size_t index_i, Real dt = 0.0)
    { 
		 Mat3d local_configuration = Eps * Mat3d::Identity();
		for (size_t k = 0; k < contact_configuration_.size(); ++k)
		{
			Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
			for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
			{
				Vec3d r_ji = contact_neighborhood.r_ij_vector_[n];
				Vec3d gradW_ij = contact_neighborhood.dW_ijV_j_[n] * contact_neighborhood.e_ij_[n];
				local_configuration -= r_ji * gradW_ij.transpose();
			}
		}
		B_[index_i] += local_configuration; 
     }; 

	void update(size_t index_i, Real dt)
	{
		Mat3d inverse = B_[index_i].inverse();
		B_[index_i] = inverse;	
	}
};


class ElectroPhysiologyParticlesforLaplacian
    : public ElectroPhysiologyParticles
{
  public:
    ElectroPhysiologyParticlesforLaplacian(SPHBody &sph_body, MonoFieldElectroPhysiology *mono_field_electro_physiology)
        : ElectroPhysiologyParticles(sph_body, mono_field_electro_physiology)
        { 
          registerVariable(A1_, "FirstOrderCorrectionVectorA1", [&](size_t i) -> Vec3d { return Eps * Vec3d::Identity(); });
          registerVariable(A2_, "FirstOrderCorrectionVectorA2", [&](size_t i) -> Vec3d { return Eps * Vec3d::Identity(); });
          registerVariable(A3_, "FirstOrderCorrectionVectorA3", [&](size_t i) -> Vec3d { return Eps * Vec3d::Identity(); });
      
          registerVariable(A4_, "FirstOrderCorrectionVectorA4", [&](size_t i) -> Vec3d { return Eps * Vec3d::Identity(); });
          registerVariable(A5_, "FirstOrderCorrectionVectorA5", [&](size_t i) -> Vec3d { return Eps * Vec3d::Identity(); });
          registerVariable(A6_, "FirstOrderCorrectionVectorA6", [&](size_t i) -> Vec3d { return Eps * Vec3d::Identity(); });
    

          addVariableToWrite<Vec3d>("FirstOrderCorrectionVectorA1");
          addVariableToWrite<Vec3d>("FirstOrderCorrectionVectorA2");
          
          addVariableToWrite<Vec3d>("FirstOrderCorrectionVectorA4");
          addVariableToWrite<Vec3d>("FirstOrderCorrectionVectorA5"); 
        };

    StdLargeVec<Vec3d> A1_;
    StdLargeVec<Vec3d> A2_;
    StdLargeVec<Vec3d> A3_;
    StdLargeVec<Vec3d> A4_;
    StdLargeVec<Vec3d> A5_;
    StdLargeVec<Vec3d> A6_; 

    virtual ~ElectroPhysiologyParticlesforLaplacian(){};
    virtual ElectroPhysiologyParticlesforLaplacian *ThisObjectPtr() override { return this; };
};

/*
class NonisotropicKernelCorrectionMatrixComplexAC : public LocalDynamics, public LaplacianSolidDataComplex
{
  public:
    NonisotropicKernelCorrectionMatrixComplexAC(ComplexRelation &complex_relation): 
                LocalDynamics(complex_relation.getInnerRelation().getSPHBody()), LaplacianSolidDataComplex(complex_relation),
                 B_(particles_->B_), A1_(particles_->A1_), A2_(particles_->A2_), A3_(particles_->A3_), 
                 A4_(particles_->A4_), A5_(particles_->A5_), A6_(particles_->A6_) {};

    virtual ~NonisotropicKernelCorrectionMatrixComplexAC(){};

  protected:
    StdLargeVec<Mat3d> &B_;
    StdLargeVec<Vec3d> &A1_,&A2_,&A3_, &A4_,&A5_,&A6_;

    void initialization(size_t index_i, Real dt = 0.0)
    {
        Neighborhood &inner_neighborhood = inner_configuration_[index_i];
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ik
        {
            Vec3d gradW_ikV_k = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
            Vec3d r_ik = inverse_decomposed_transform_tensor * -inner_neighborhood.r_ij_vector_[n];
            
            A1_[index_i] += r_ik[0] * r_ik[0] * (B_[index_i].transpose() * gradW_ikV_k);
            A2_[index_i] += r_ik[1] * r_ik[1] * (B_[index_i].transpose() * gradW_ikV_k);
            A3_[index_i] += r_ik[2] * r_ik[2] * (B_[index_i].transpose() * gradW_ikV_k);

            A4_[index_i] += r_ik[0] * r_ik[1] * (B_[index_i].transpose() * gradW_ikV_k);
            A5_[index_i] += r_ik[1] * r_ik[2] * (B_[index_i].transpose() * gradW_ikV_k);
            A6_[index_i] += r_ik[2] * r_ik[0] * (B_[index_i].transpose() * gradW_ikV_k);
        }
    };

    void interaction(size_t index_i, Real dt = 0.0)
    {
         for (size_t k = 0; k < contact_configuration_.size(); ++k)
        {
            Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
            for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
            {
                Vec3d r_ik = inverse_decomposed_transform_tensor * -contact_neighborhood.r_ij_vector_[n];
                Vec3d gradW_ikV_k = contact_neighborhood.dW_ijV_j_[n] * contact_neighborhood.e_ij_[n];

                A1_[index_i] += r_ik[0] * r_ik[0] * (B_[index_i].transpose() * gradW_ikV_k);
                A2_[index_i] += r_ik[1] * r_ik[1] * (B_[index_i].transpose() * gradW_ikV_k);
                A3_[index_i] += r_ik[2] * r_ik[2] * (B_[index_i].transpose() * gradW_ikV_k);

                A4_[index_i] += r_ik[0] * r_ik[1] * (B_[index_i].transpose() * gradW_ikV_k);
                A5_[index_i] += r_ik[1] * r_ik[2] * (B_[index_i].transpose() * gradW_ikV_k);
                A6_[index_i] += r_ik[2] * r_ik[0] * (B_[index_i].transpose() * gradW_ikV_k);
            }
        }

    };

	void update(size_t index_i, Real dt = 0.0) {};
};
*/
typedef DataDelegateComplex<ElectroPhysiologyParticlesforLaplacian, BaseParticles> DiffusionReactionComplexData;
 

 class DiffusionRelaxationComplex
    : public LocalDynamics,
      public DiffusionReactionComplexData
{
  protected:
   typedef  ElectroPhysiologyParticlesforLaplacian::DiffusionReactionMaterial Material;
    Material &material_;
    StdVec<BaseDiffusion *> &all_diffusions_;
    StdVec<StdLargeVec<Real> *> &diffusion_species_; 
    StdVec<StdLargeVec<Real> *> diffusion_dt_;
   
    StdLargeVec<Vec3d> &pos_;
    StdLargeVec<Mat3d> &B_;


    //StdLargeVec<Vec3d> &A1_,&A2_,&A3_, &A4_,&A5_,&A6_;
    StdVec<StdLargeVec<Vec3d>*> E_;
    StdLargeVec<Vec3d> E_rate_;
    
    StdVec<StdLargeVec<Mat6d>> SC_;
    StdVec<StdLargeVec<Vec6d>>  G_; 
    StdLargeVec<Vec6d> Laplacian_; 
 
    StdLargeVec<Real> Laplacian_x, Laplacian_y, Laplacian_z;
    Real diffusion_coeff_;
   size_t particle_number;

  public:
   
    explicit DiffusionRelaxationComplex(ComplexRelation &complex_relation): 
      LocalDynamics(complex_relation.getSPHBody()), DiffusionReactionComplexData(complex_relation),
      material_(this->particles_->diffusion_reaction_material_),
      all_diffusions_(material_.AllDiffusions()),
      diffusion_species_(this->particles_->DiffusionSpecies()), 
      pos_(particles_->pos_), B_(particles_->B_),
      particle_number(complex_relation.getSPHBody().getBaseParticles().real_particles_bound_)
      //, A1_(particles_->A1_), A2_(particles_->A2_), A3_(particles_->A3_)
      {
       
        this->particles_->registerVariable(Laplacian_x, "Laplacian_x", [&](size_t i) -> Real { return Real(0.0); });
        this->particles_->registerVariable(Laplacian_y, "Laplacian_y", [&](size_t i) -> Real { return Real(0.0); });
        this->particles_->registerVariable(Laplacian_z, "Laplacian_z", [&](size_t i) -> Real { return Real(0.0); });
         
         std::cout<< "particle_number "<<particle_number<<std::endl;

        diffusion_dt_.resize(all_diffusions_.size());
        E_.resize(all_diffusions_.size());
        SC_.resize(all_diffusions_.size());
        G_.resize(all_diffusions_.size());


        StdVec<std::string> &all_species_names = this->particles_->AllSpeciesNames();
        IndexVector &diffusion_species_indexes = material_.DiffusionSpeciesIndexes();
        for (size_t m = 0; m != all_diffusions_.size(); ++m)
        {
            // Register specie change rate as shared variable
            std::string &diffusion_species_name = all_species_names[diffusion_species_indexes[m]];
            diffusion_dt_[m] = this->particles_->template registerSharedVariable<Real>(diffusion_species_name + "ChangeRate");
            E_[m] = this->particles_->template registerSharedVariable<Vec3d>(diffusion_species_name + "FirstOrderCorrectionVectorE");
        
            E_rate_.push_back(Vec3d::Identity()); 
      
            for (size_t i = 0; i != particle_number; ++i)
            {
                SC_[m].push_back(Mat6d::Identity()); 
                G_[m].push_back(Vec6d::Identity()); 
              
                Laplacian_.push_back(Vec6d::Identity()); 
            }

        }
      
       // note this is need to confirmed
        diffusion_coeff_ = 1.0;
    };


    virtual ~DiffusionRelaxationComplex(){}; 

    void initialization(size_t index_i, Real dt = 0.0)
    {
        for (size_t m = 0; m < this->all_diffusions_.size(); ++m)
        {
            (*this->diffusion_dt_[m])[index_i] = 0;
            E_rate_[m] = Vec3d::Zero();
            
            Neighborhood &inner_neighborhood = inner_configuration_[index_i];
            for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ik
            {
                size_t index_k = inner_neighborhood.j_[n];
                Vec3d gradW_ikV_k = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];

                E_rate_[m] += ((*this->diffusion_species_[m])[index_k] - (*this->diffusion_species_[m])[index_i]) 
                            * (B_[index_i].transpose() * gradW_ikV_k); // HOW TO DEFINE IT???
            }
           
           (*this-> E_[m])[index_i] = E_rate_[m];

            Vec6d G_rate = Vec6d::Zero();
            Mat6d SC_rate = Mat6d::Zero();
            Real H_rate = 1.0;
            for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ik
            {
                size_t index_j = inner_neighborhood.j_[n];
                Vec3d r_ij = inverse_decomposed_transform_tensor *  -inner_neighborhood.r_ij_vector_[n];

                Vec3d gradW_ijV_j = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
                Vec6d S_ = Vec6d(r_ij[0] * r_ij[0], r_ij[1] * r_ij[1], r_ij[2] * r_ij[2], r_ij[0] * r_ij[1], r_ij[1] * r_ij[2], r_ij[2] * r_ij[0]);
                H_rate = r_ij.dot(B_[index_i].transpose() * gradW_ijV_j) / pow(r_ij.norm(), 4.0);
                
                Real FF_ = 2.0 * ((*this->diffusion_species_[m])[index_j] - (*this->diffusion_species_[m])[index_i] 
                        - r_ij.dot((*this-> E_[m])[index_i]));
                G_rate += S_ *H_rate * FF_;
                
                //TO DO
                Vec6d C_ = Vec6d::Zero();
               /*C_[0] = (r_ij[0] * r_ij[0]- r_ij.dot(A1_[index_i]));
                C_[1] = (r_ij[1] * r_ij[1]- r_ij.dot(A2_[index_i]));
                C_[2] = (r_ij[2] * r_ij[2]- r_ij.dot(A3_[index_i]));
                C_[3] = (r_ij[0] * r_ij[1]- r_ij.dot(A4_[index_i]));
                C_[4] = (r_ij[1] * r_ij[2]- r_ij.dot(A5_[index_i]));
                C_[5] = (r_ij[2] * r_ij[0]- r_ij.dot(A6_[index_i]));*/ 

                C_[0] = (r_ij[0] * r_ij[0]);
                C_[1] = (r_ij[1] * r_ij[1]);
                C_[2] = (r_ij[2] * r_ij[2]);
                C_[3] = (r_ij[0] * r_ij[1]);
                C_[4] = (r_ij[1] * r_ij[2]);
                C_[5] = (r_ij[2] * r_ij[0]);
                SC_rate += S_ *H_rate* C_.transpose();   

            }

            G_[m][index_i] = G_rate;
            SC_[m][index_i] = SC_rate;
        }
    };


     void interaction(size_t index_i, Real dt = 0.0)
    {
         for (size_t m = 0; m < this->all_diffusions_.size(); ++m)
        {
          
            Mat6d SC_rate_contact = Mat6d::Zero();
            Vec6d G_rate_contact = Vec6d::Zero();
            Real H_rate_contact = 1.0;
            for (size_t k = 0; k < contact_configuration_.size(); ++k)
            {
                Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
                for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
                {
                    Vec3d r_ij = inverse_decomposed_transform_tensor * -contact_neighborhood.r_ij_vector_[n];
                    Vec3d gradW_ijV_j = contact_neighborhood.dW_ijV_j_[n] * contact_neighborhood.e_ij_[n];

                    Vec6d S_ = Vec6d(r_ij[0] * r_ij[0], r_ij[1] * r_ij[1], r_ij[2] * r_ij[2], r_ij[0] * r_ij[1], r_ij[1] * r_ij[2], r_ij[2] * r_ij[0]);
                    Real FF_ = 2.0 * (0.0 - r_ij.dot((*this-> E_[m])[index_i])); ///here when it is periodic boundary condition, should notice the 0.0
                    H_rate_contact = r_ij.dot(B_[index_i].transpose() * gradW_ijV_j) / pow(r_ij.norm(), 4.0);
            
                    //TO DO
                    Vec6d C_ = Vec6d::Zero();
                    C_[0] = (r_ij[0] * r_ij[0]);
                    C_[1] = (r_ij[1] * r_ij[1]);
                    C_[2] = (r_ij[2] * r_ij[2]);
                    C_[3] = (r_ij[0] * r_ij[1]);
                    C_[4] = (r_ij[1] * r_ij[2]);
                    C_[5] = (r_ij[2] * r_ij[0]);

                    SC_rate_contact += S_ * H_rate_contact * C_.transpose();
                    G_rate_contact += S_ * H_rate_contact * FF_;

                }
                SC_[m][index_i] += SC_rate_contact;
                G_[m][index_i] += G_rate_contact;
            }
    
            
            Laplacian_[index_i] = diffusion_coeff_ * SC_[m][index_i].inverse() * G_[m][index_i];

            Laplacian_x[index_i] = Laplacian_[index_i][0];
            Laplacian_y[index_i] = Laplacian_[index_i][1];
            Laplacian_z[index_i] = Laplacian_[index_i][2];
            (*this->diffusion_dt_[m])[index_i] = Laplacian_[index_i][0] + Laplacian_[index_i][1]+ Laplacian_[index_i][2];
        }

    };


    void update(size_t index_i, Real dt = 0.0)
    { 
        for (size_t m = 0; m < this->all_diffusions_.size(); ++m)
        {
            (*this->diffusion_species_[m])[index_i] += dt * (*this->diffusion_dt_[m])[index_i];
        }
    };

};
 

////////////////////////////////////////
/**
 * The main program.
 */
int main(int ac, char *av[])
{
    //----------------------------------------------------------------------
    //	SPHSystem section
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, dp_0);
    sph_system.setRunParticleRelaxation(false); // Tag for run particle relaxation for body-fitted distribution
    sph_system.setReloadParticles(true);       // Tag for computation with save particles distribution
#ifdef BOOST_AVAILABLE
    sph_system.handleCommandlineOptions(ac, av); // handle command line arguments
#endif
    IOEnvironment io_environment(sph_system);
    //----------------------------------------------------------------------
    //	SPH Particle relaxation section
    //----------------------------------------------------------------------
    /** check whether run particle relaxation for body fitted particle distribution. */
    if (sph_system.RunParticleRelaxation())
    {
        SolidBody herat_model(sph_system, makeShared<Heart>("HeartModel"));
        herat_model.defineBodyLevelSetShape()->correctLevelSetSign()->writeLevelSet(io_environment);
        herat_model.defineParticlesAndMaterial<FiberDirectionDiffusionParticles, FiberDirectionDiffusion>();
        herat_model.generateParticles<ParticleGeneratorLattice>();

        SolidBody boundary(sph_system, makeShared<BoundaryModel>("Boundary"));
        boundary.defineBodyLevelSetShape()->correctLevelSetSign()->writeLevelSet(io_environment);
        boundary.defineParticlesAndMaterial<SolidParticles, Solid>();
        boundary.generateParticles<ParticleGeneratorLattice>();


        /** topology */
        InnerRelation herat_model_inner(herat_model);
        InnerRelation boundary_model_inner(boundary);

        /** Random reset the relax solid particle position. */
        SimpleDynamics<RandomizeParticlePosition> random_particles(herat_model);
        SimpleDynamics<RandomizeParticlePosition> random_particles_boundary(boundary);
       /** A  Physics relaxation step. */
        relax_dynamics::RelaxationStepInner relaxation_step_inner(herat_model_inner);
        relax_dynamics::RelaxationStepInner relaxation_step_inner_boundary(boundary_model_inner);
      /** Time step for diffusion. */
        GetDiffusionTimeStepSize<FiberDirectionDiffusionParticles> get_time_step_size(herat_model);
        /** Diffusion process for diffusion body. */
        DiffusionRelaxation diffusion_relaxation(herat_model_inner);
        /** Compute the fiber and sheet after diffusion. */
        SimpleDynamics<ComputeFiberAndSheetDirections> compute_fiber_sheet(herat_model);
        /** Write the body state to Vtp file. */
        BodyStatesRecordingToVtp write_herat_model_state_to_vtp(io_environment, {herat_model});
        BodyStatesRecordingToVtp write_boundary_model_state_to_vtp(io_environment, {boundary});
      /** Write the particle reload files. */
        ReloadParticleIO write_particle_reload_files(io_environment, herat_model);
        ReloadParticleIO write_boundary_particle_reload_files(io_environment,boundary);
       //----------------------------------------------------------------------
        //	Physics relaxation starts here.
        //----------------------------------------------------------------------
        random_particles.exec(0.01);
        random_particles_boundary.exec(0.25);
        relaxation_step_inner.SurfaceBounding().exec();
        relaxation_step_inner_boundary.SurfaceBounding().exec();
        write_herat_model_state_to_vtp.writeToFile(0.0);
        write_boundary_model_state_to_vtp.writeToFile(0.0);
        //----------------------------------------------------------------------
        // From here the time stepping begins.
        //----------------------------------------------------------------------
        int ite = 0;
        int relax_step = 1000;
        int diffusion_step = 100;
        while (ite < relax_step)
        { 
            relaxation_step_inner.exec();
            relaxation_step_inner_boundary.exec();
            ite++;
            if (ite % 100 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite << "\n";
                write_herat_model_state_to_vtp.writeToFile(ite);
                write_boundary_model_state_to_vtp.writeToFile(ite);
             }
        }

        BodySurface surface_part(herat_model);
        /** constraint boundary condition for diffusion. */
        SimpleDynamics<DiffusionBCs> impose_diffusion_bc(surface_part, "Phi");
        impose_diffusion_bc.exec();

        write_herat_model_state_to_vtp.writeToFile(ite);
        write_boundary_model_state_to_vtp.writeToFile(ite);
        Real dt = get_time_step_size.exec();
        
        while (ite <= diffusion_step + relax_step)
        {
            diffusion_relaxation.exec(dt);
            impose_diffusion_bc.exec();
            if (ite % 10 == 0)
            {
                std::cout << "Diffusion steps N=" << ite - relax_step << "	dt: " << dt << "\n";
                write_herat_model_state_to_vtp.writeToFile(ite);
            }
            ite++;
        }
        compute_fiber_sheet.exec();
        ite++;
        write_herat_model_state_to_vtp.writeToFile(ite);
        compute_fiber_sheet.exec();
        write_particle_reload_files.writeToFile(0);
        write_boundary_particle_reload_files.writeToFile(0);

        return 0;
    }
    //----------------------------------------------------------------------
    //	SPH simulation section
    //----------------------------------------------------------------------
    /** create a SPH body, material and particles */
    SolidBody physiology_heart(sph_system, makeShared<Heart>("PhysiologyHeart"));
    SharedPtr<AlievPanfilowModel> muscle_reaction_model_ptr = makeShared<AlievPanfilowModel>(k_a, c_m, k, a, b, mu_1, mu_2, epsilon);
    physiology_heart.defineParticlesAndMaterial<
        ElectroPhysiologyParticlesforLaplacian, MonoFieldElectroPhysiology>(
        muscle_reaction_model_ptr, TypeIdentity<LocalDirectionalDiffusion>(), diffusion_coeff, bias_coeff, fiber_direction);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? physiology_heart.generateParticles<ParticleGeneratorReload>(io_environment, "HeartModel")
        : physiology_heart.generateParticles<ParticleGeneratorLattice>();

    SolidBody boundary_condition(sph_system, makeShared<BoundaryModel>("BoundaryCondition"));
    boundary_condition.defineParticlesAndMaterial<SolidParticles, Solid>();
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? boundary_condition.generateParticles<ParticleGeneratorReload>(io_environment, "Boundary")
        : boundary_condition.generateParticles<ParticleGeneratorLattice>();


    /** create a SPH body, material and particles */
    SolidBody mechanics_heart(sph_system, makeShared<Heart>("MechanicalHeart"));
    mechanics_heart.defineParticlesAndMaterial<
        ElasticSolidParticles, ActiveMuscle<LocallyOrthotropicMuscle>>(rho0_s, bulk_modulus, fiber_direction, sheet_direction, a0, b0);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? mechanics_heart.generateParticles<ParticleGeneratorReload>(io_environment, "HeartModel")
        : mechanics_heart.generateParticles<ParticleGeneratorLattice>();

    //----------------------------------------------------------------------
    //	SPH Observation section
    //----------------------------------------------------------------------
    ObserverBody voltage_observer(sph_system, "VoltageObserver");
    voltage_observer.generateParticles<HeartObserverParticleGenerator>();
    ObserverBody myocardium_observer(sph_system, "MyocardiumObserver");
    myocardium_observer.generateParticles<HeartObserverParticleGenerator>();
    //----------------------------------------------------------------------
    //	SPHBody relation (topology) section
    //----------------------------------------------------------------------
    InnerRelation physiology_heart_inner(physiology_heart);
    InnerRelation mechanics_body_inner(mechanics_heart);
    ContactRelation physiology_heart_contact(physiology_heart, {&mechanics_heart});
    ContactRelation mechanics_body_contact(mechanics_heart, {&physiology_heart});
    ContactRelation voltage_observer_contact(voltage_observer, {&physiology_heart});
    ContactRelation myocardium_observer_contact(myocardium_observer, {&mechanics_heart});
     
 
    ComplexRelation heart_boundary_complex(physiology_heart, {&boundary_condition});


    //----------------------------------------------------------------------
    //	SPH Method section
    //----------------------------------------------------------------------
    // Corrected configuration.
  	Dynamics1Level<NonisotropicKernelCorrectionMatrixComplex> correct_configuration_excitation(heart_boundary_complex);
    physiology_heart.addBodyStateForRecording<Mat3d>("KernelCorrectionMatrix");
    // Time step size calculation.
    electro_physiology::GetElectroPhysiologyTimeStepSize get_physiology_time_step(physiology_heart);
    // Diffusion process for diffusion body.
   
    Dynamics1Level<DiffusionRelaxationComplex>  diffusion_relaxation(heart_boundary_complex);
    // Solvers for ODE system.
    electro_physiology::ElectroPhysiologyReactionRelaxationForward reaction_relaxation_forward(physiology_heart);
    electro_physiology::ElectroPhysiologyReactionRelaxationBackward reaction_relaxation_backward(physiology_heart);
    //	Apply the Iron stimulus.
    SimpleDynamics<ApplyStimulusCurrentSI> apply_stimulus_s1(physiology_heart);
    SimpleDynamics<ApplyStimulusCurrentSII> apply_stimulus_s2(physiology_heart);
    // Active mechanics.
    InteractionWithUpdate<KernelCorrectionMatrixInner> correct_configuration_contraction(mechanics_body_inner);
    InteractionDynamics<CorrectInterpolationKernelWeights> correct_kernel_weights_for_interpolation(mechanics_body_contact);
    /** Interpolate the active contract stress from electrophysiology body. */
    InteractionDynamics<InterpolatingAQuantity<Real>>
        active_stress_interpolation(mechanics_body_contact, "ActiveContractionStress", "ActiveContractionStress");
    /** Interpolate the particle position in physiology_heart  from mechanics_heart. */
    // TODO: this is a bug, we should interpolate displacement other than position.
    InteractionDynamics<InterpolatingAQuantity<Vecd>>
        interpolation_particle_position(physiology_heart_contact, "Position", "Position");
    /** Time step size calculation. */
    ReduceDynamics<solid_dynamics::AcousticTimeStepSize> get_mechanics_time_step(mechanics_heart);
    /** active and passive stress relaxation. */
    Dynamics1Level<solid_dynamics::Integration1stHalfPK2> stress_relaxation_first_half(mechanics_body_inner);
    Dynamics1Level<solid_dynamics::Integration2ndHalf> stress_relaxation_second_half(mechanics_body_inner);
    /** Constrain region of the inserted body. */
    MuscleBaseShapeParameters muscle_base_parameters;
    BodyRegionByParticle muscle_base(mechanics_heart, makeShared<TriangleMeshShapeBrick>(muscle_base_parameters, "Holder"));
    SimpleDynamics<solid_dynamics::FixBodyPartConstraint> constraint_holder(muscle_base);
    //----------------------------------------------------------------------
    //	SPH Output section
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp write_states(io_environment, sph_system.real_bodies_);
    RegressionTestDynamicTimeWarping<ObservedQuantityRecording<Real>>
        write_voltage("Voltage", io_environment, voltage_observer_contact);
    RegressionTestDynamicTimeWarping<ObservedQuantityRecording<Vecd>>
        write_displacement("Position", io_environment, myocardium_observer_contact);
    //----------------------------------------------------------------------
    //	 Pre-simulation.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    correct_configuration_excitation.exec();
    correct_configuration_contraction.exec();
    correct_kernel_weights_for_interpolation.exec();
    /** Output initial states and observations */
    write_states.writeToFile(0);
    write_voltage.writeToFile(0);
    write_displacement.writeToFile(0);
    //----------------------------------------------------------------------
    //	 Physical parameters for main loop.
    //----------------------------------------------------------------------
    int screen_output_interval = 10;
    int ite = 0;
    int reaction_step = 2;
    Real end_time = 100;
    Real Ouput_T = end_time / 200.0;
    Real Observer_time = 0.01 * Ouput_T;
    Real dt = 0.0;   /**< Default acoustic time step sizes for physiology. */
    Real dt_s = 0.0; /**< Default acoustic time step sizes for mechanics. */
    /** Statistics for computing time. */
    TickCount t1 = TickCount::now();
    TimeInterval interval;
    std::cout << "Main Loop Starts Here : "
              << "\n";
    /** Main loop starts here. */
    while (GlobalStaticVariables::physical_time_ < end_time)
    {
        Real integration_time = 0.0;
        while (integration_time < Ouput_T)
        {
            Real relaxation_time = 0.0;
            while (relaxation_time < Observer_time)
            {
                if (ite % screen_output_interval == 0)
                {
                    std::cout << std::fixed << std::setprecision(9) << "N=" << ite << "	Time = "
                              << GlobalStaticVariables::physical_time_
                              << "	dt = " << dt
                              << "	dt_s = " << dt_s << "\n";
                }
                /** Apply stimulus excitation. */
                if (0 <= GlobalStaticVariables::physical_time_ && GlobalStaticVariables::physical_time_ <= 0.5)
                {
                    apply_stimulus_s1.exec(dt);
                }
                /** Single spiral wave. */
                // if( 60 <= GlobalStaticVariables::physical_time_
                // 	&&  GlobalStaticVariables::physical_time_ <= 65)
                // {
                // 	apply_stimulus_s2.exec(dt);
                // }
                /**Strong splitting method. */
                // forward reaction
                int ite_forward = 0;
                while (ite_forward < reaction_step)
                {
                    reaction_relaxation_forward.exec(0.5 * dt / Real(reaction_step));
                    ite_forward++;
                }
                /** 2nd Runge-Kutta scheme for diffusion. */
                diffusion_relaxation.exec(dt);

                // backward reaction
                int ite_backward = 0;
                while (ite_backward < reaction_step)
                {
                    reaction_relaxation_backward.exec(0.5 * dt / Real(reaction_step));
                    ite_backward++;
                }

                active_stress_interpolation.exec();

                Real dt_s_sum = 0.0;
                while (dt_s_sum < dt)
                {
                    dt_s = get_mechanics_time_step.exec();
                    if (dt - dt_s_sum < dt_s)
                        dt_s = dt - dt_s_sum;
                    stress_relaxation_first_half.exec(dt_s);
                    constraint_holder.exec(dt_s);
                    stress_relaxation_second_half.exec(dt_s);
                    dt_s_sum += dt_s;
                }

                ite++;
                dt = get_physiology_time_step.exec();

                relaxation_time += dt;
                integration_time += dt;
                GlobalStaticVariables::physical_time_ += dt;
            }
            write_voltage.writeToFile(ite);
            write_displacement.writeToFile(ite);
        }
        TickCount t2 = TickCount::now();
        interpolation_particle_position.exec();
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

/* -------------------------------------------------------------------------*
 *								SPHinXsys									*
 * --------------------------------------------------------------------------*
 * SPHinXsys (pronunciation: s'finksis) is an acronym from Smoothed Particle	*
 * Hydrodynamics for industrial compleX systems. It provides C++ APIs for	*
 * physical accurate simulation and aims to model coupled industrial dynamic *
 * systems including fluid, solid, multi-body dynamics and beyond with SPH	*
 * (smoothed particle hydrodynamics), a meshless computational method using	*
 * particle discretization.													*
 *																			*
 * SPHinXsys is partially funded by German Research Foundation				*
 * (Deutsche Forschungsgemeinschaft) DFG HU1527/6-1, HU1527/10-1				*
 * and HU1527/12-1.															*
 *                                                                           *
 * Portions copyright (c) 2017-2020 Technical University of Munich and		*
 * the authors' affiliations.												*
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may   *
 * not use this file except in compliance with the License. You may obtain a *
 * copy of the License at http://www.apache.org/licenses/LICENSE-2.0.        *
 *                                                                           *
 * --------------------------------------------------------------------------*/
/**
 * @file 	asph_pkj_lv_electrocontraction.h
 * @brief 	Electro-contraction of left ventricle heart model.
 * @author 	Chi ZHANG and Xiangyu HU
 * @version  0.3
 * @version 0.2.1
 * 			Chi Zhang
 * 			Unit :
 *			time t = ms = 12.9 [-]
 * 			length l = mm
 * 			mass m = g
 *			density rho = g * (mm)^(-3)
 *			Pressure pa = g * (mm)^(-1) * (ms)^(-2)
 *			diffusion d = (mm)^(2) * (ms)^(-2)
 *@version 0.3
 *			Here, the coupling with Purkinje network will be conducted.
 */
#pragma once
#include "sphinxsys.h"
using namespace SPH;
#define PI 3.1415926
/** Set the file path to the stl file. */
std::string full_path_to_lv = "./input/leftventricle.stl";
Real length_scale = 1.0;
Real time_scale = 1.0 / 12.9;
Real stress_scale = 1.0e-6;
/** Parameters and physical properties. */
Vec3d domain_lower_bound(-90.0 * length_scale, -40.0 * length_scale, -80.0 * length_scale);
Vec3d domain_upper_bound(40.0 * length_scale, 30.0 * length_scale, 50.0 * length_scale);
Real dp_0 = (domain_upper_bound[0] - domain_lower_bound[0]) / 100.0;
/** Domain bounds of the system. */
BoundingBox system_domain_bounds(domain_lower_bound, domain_upper_bound);

/** Material properties. */
Real rho0_s = 1.06e-3;
/** Active stress factor */
Real k_a = 150 * stress_scale;
Real a0[4] = {Real(496.0 * stress_scale), Real(15196.0 * stress_scale), Real(3283.0 * stress_scale), Real(662.0 * stress_scale)};
Real b0[4] = {Real(7.209), Real(20.417), Real(11.176), Real(9.466)};
/** reference stress to achieve weakly compressible condition */
Real poisson = 0.4995;
Real bulk_modulus = 2.0 * a0[0] * (1.0 + poisson) / (3.0 * (1.0 - 2.0 * poisson));
/** Electrophysiology parameters. */
Real acceleration_factor = 27.5; /** Acceleration factor for fast diffusion on purkinje network. */
Real diffusion_coeff = 0.8;
Real bias_coeff = 0.0;
/** Electrophysiology parameters. */
std::array<std::string, 1> species_name_list{"Phi"};
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
/** Purkinje Network. */
Vec3d starting_point(-21.9347 * length_scale, 4.0284 * length_scale, 0.0 * length_scale);
Vec3d second_point(-21.9347 * length_scale, 4.0284 * length_scale, -1.1089 * length_scale);


Vecd translation_boundary(-20 * length_scale, -0.0 * length_scale, -10.0 * length_scale);
Vecd halfsize_boundary(70.0 * length_scale, 40.0 * length_scale, 70.0 * length_scale);

class BoundaryModel : public ComplexShape
{
  public:
    explicit BoundaryModel(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<TransformShape<GeometricShapeBox>>(Transform(translation_boundary), halfsize_boundary, "OuterBoundary");
        Vecd translation(0.0, 0.0, 0.0);
        subtract<TriangleMeshShapeSTL>(full_path_to_lv, translation, length_scale);
    }
};

//----------------------------------------------------------------------
//	Define heart shape
//----------------------------------------------------------------------
class Heart : public ComplexShape
{
  public:
    explicit Heart(const std::string &shape_name) : ComplexShape(shape_name)
    {
        Vecd translation(0.0, 0.0, 0.0);
        add<TriangleMeshShapeSTL>(full_path_to_lv, translation, length_scale);
    }
};
//----------------------------------------------------------------------
//	Setup diffusion material properties.
//----------------------------------------------------------------------
class FiberDirectionDiffusion : public DiffusionReaction<LocallyOrthotropicMuscle>
{
  public:
    FiberDirectionDiffusion()
        : DiffusionReaction<LocallyOrthotropicMuscle>(
              {"Phi"}, SharedPtr<NoReaction>(),
              rho0_s, bulk_modulus, fiber_direction, sheet_direction, a0, b0)
    {
        initializeAnDiffusion<IsotropicDiffusion>("Phi", "Phi", diffusion_coeff);
    };
};
using FiberDirectionDiffusionParticles = DiffusionReactionParticles<ElasticSolidParticles, FiberDirectionDiffusion>;
/** Set diffusion relaxation. */
class DiffusionRelaxation
    : public DiffusionRelaxationRK2<
          DiffusionRelaxationInner<FiberDirectionDiffusionParticles>>
{
  public:
    explicit DiffusionRelaxation(InnerRelation &inner_relation)
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

        if (pos_[index_i][2] < 2.0 * sph_body_.sph_adaptation_->ReferenceSpacing())
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
        Real l = domain_upper_bound[0] - domain_lower_bound[0];
        Real w = domain_upper_bound[1] - domain_lower_bound[1];
        Real h = domain_upper_bound[2];
        halfsize_ = Vec3d(0.5 * l, 0.5 * w, 0.5 * h);
        resolution_ = 20;
        translation_ = Vec3d(-25.0 * length_scale, -5.0 * length_scale, 0.5 * h * length_scale);
    }
};
/**
 * application dependent initial condition
 */
class ApplyStimulusCurrentToMyocardium
    : public electro_physiology::ElectroPhysiologyInitialCondition
{
  protected:
    size_t voltage_;

  public:
    explicit ApplyStimulusCurrentToMyocardium(SPHBody &sph_body)
        : electro_physiology::ElectroPhysiologyInitialCondition(sph_body)
    {
        voltage_ = particles_->diffusion_reaction_material_.AllSpeciesIndexMap()["Voltage"];
    };

    void update(size_t index_i, Real dt)
    {
        if (-32.0 * length_scale <= pos_[index_i][0] && pos_[index_i][0] <= -20.0 * length_scale)
        {
            if (-5.0 * length_scale <= pos_[index_i][1] && pos_[index_i][1] <= 5.0)
            {
                if (-10.0 * length_scale <= pos_[index_i][2] && pos_[index_i][2] <= 0.0 * length_scale)
                {
                    all_species_[voltage_][index_i] = 1.0;
                }
            }
        }
    };
};
// Observer particle generator.
class HeartObserverParticleGenerator : public ObserverParticleGenerator
{
  public:
    explicit HeartObserverParticleGenerator(SPHBody &sph_body) : ObserverParticleGenerator(sph_body)
    {
        /** position and volume. */
        positions_.push_back(Vecd(-10.0, -5.0, -74));
              positions_.push_back(Vecd(-10.0, -7.35, -74));
        positions_.push_back(Vecd(-8.0, -7.35, -74));
        positions_.push_back(Vecd(-6.0, -7.35, -74));    
        positions_.push_back(Vecd(-5.0, -7.14, -74));
    }
};
/**
 * application dependent initial condition
 */
class ApplyStimulusCurrentToPKJ
    : public electro_physiology::ElectroPhysiologyInitialCondition
{
  protected:
    size_t voltage_;

  public:
    explicit ApplyStimulusCurrentToPKJ(SPHBody &sph_body)
        : electro_physiology::ElectroPhysiologyInitialCondition(sph_body)
    {
        voltage_ = particles_->diffusion_reaction_material_.AllSpeciesIndexMap()["Voltage"];
    };

    void update(size_t index_i, Real dt)
    {
        if (index_i <= 10)
        {
            all_species_[voltage_][index_i] = 1.0;
        }
    };
};

/**
 * Derived network particle generator.
 */
class NetworkGeneratorWithExtraCheck : public ParticleGeneratorNetwork
{
  protected:
    bool extraCheck(const Vecd &new_position) override
    {
        bool no_generation = false;
        if (new_position[2] > 0)
            no_generation = true;
        return no_generation;
    };

  public:
    NetworkGeneratorWithExtraCheck(SPHBody &sph_body, Vecd starting_pnt, Vecd second_pnt, int iterator, Real grad_factor)
        : ParticleGeneratorNetwork(sph_body, starting_pnt, second_pnt, iterator, grad_factor){};
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

typedef DataDelegateComplex<ElectroPhysiologyParticlesforLaplacian, BaseParticles> DiffusionReactionComplexData;
 


class NonisotropicKernelCorrectionMatrixComplexAC : public LocalDynamics, public DiffusionReactionComplexData
{
  public:
    NonisotropicKernelCorrectionMatrixComplexAC(ComplexRelation &complex_relation): 
                LocalDynamics(complex_relation.getInnerRelation().getSPHBody()),DiffusionReactionComplexData(complex_relation),
                 B_(particles_->B_), A1_(particles_->A1_), A2_(particles_->A2_), A3_(particles_->A3_), 
                 A4_(particles_->A4_), A5_(particles_->A5_), A6_(particles_->A6_)  {};
   
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
            Vec3d r_ik =  -inner_neighborhood.r_ij_vector_[n];
            
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
                Vec3d r_ik = -contact_neighborhood.r_ij_vector_[n];
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

 
class AnisotropicLocalDirectionalDiffusion : public LocalDirectionalDiffusion
{
  protected:
    StdLargeVec<Mat3d> decomposed_transform_tensor_;

  public:
    AnisotropicLocalDirectionalDiffusion(size_t diffusion_species_index, size_t gradient_species_index,
                              Real diff_cf, Real bias_diff_cf, Vecd bias_direction)
        : LocalDirectionalDiffusion(diffusion_species_index, gradient_species_index, diff_cf, bias_diff_cf, bias_direction)
    {
        material_type_name_ = "AnisotropicLocalDirectionalDiffusion";
    };
    virtual ~AnisotropicLocalDirectionalDiffusion(){};

  
   virtual void initializeLocalParameters(BaseParticles *base_particles)
   {
      LocalDirectionalDiffusion::initializeLocalParameters(base_particles);
      base_particles->registerVariable(
        decomposed_transform_tensor_, "DecomposedTransformTensor",
        [&](size_t i) -> Mat3d
        {
           Mat3d decomposed_transform_tensor =  
                    inverseCholeskyDecomposition(local_transformed_diffusivity_[i]).inverse();
          return decomposed_transform_tensor;
        });

    std::cout << "\n Anisotropic Local diffusion parameters setup finished " << std::endl;
  };

};

 class DiffusionRelaxationComplex
    : public LocalDynamics,
      public DiffusionReactionComplexData
{
  protected:
   typedef  ElectroPhysiologyParticlesforLaplacian::DiffusionReactionMaterial Material;
    Material &material_;

    StdLargeVec<Real> &voltage_;
    StdLargeVec<Real> diffusion_dt_;
   
    StdLargeVec<Vec3d> &pos_;
    StdLargeVec<Mat3d> &B_;
    StdLargeVec<Mat3d> &decomposed_transform_tensor;

    StdLargeVec<Vec3d>  E_;
    
    StdLargeVec<Mat6d> SC_;
    StdLargeVec<Vec6d> G_; 
    StdLargeVec<Vec6d> Laplacian_; 
  
   
   size_t particle_number;
   StdLargeVec<Vec3d> &A1_,&A2_,&A3_, &A4_,&A5_,&A6_;

  public: 
    explicit DiffusionRelaxationComplex(ComplexRelation &complex_relation): 
      LocalDynamics(complex_relation.getSPHBody()), DiffusionReactionComplexData(complex_relation),
      material_(this->particles_->diffusion_reaction_material_), voltage_(*particles_->registerSharedVariable<Real>("Voltage")),
      pos_(particles_->pos_), B_(particles_->B_),
        decomposed_transform_tensor(*particles_->registerSharedVariable<Mat3d>("DecomposedTransformTensor")),
        particle_number(complex_relation.getSPHBody().getBaseParticles().real_particles_bound_)
      , A1_(particles_->A1_), A2_(particles_->A2_), A3_(particles_->A3_) , A4_(particles_->A4_), A5_(particles_->A5_), A6_(particles_->A6_)
      { 
        this->particles_->registerVariable(diffusion_dt_, "VoltageChangeRate", [&](size_t i) -> Real { return Real(0.0); });
        this->particles_->registerVariable(E_, "VoltageFirstOrderCorrectionVectorE", [&](size_t i) -> Vec3d { return Vec3d::Zero(); });
         
        
        for (size_t i = 0; i != particle_number; ++i)
        {
          SC_.push_back(Mat6d::Identity()); 
          G_.push_back(Vec6d::Identity()); 
          Laplacian_.push_back(Vec6d::Identity()); 
        }
    
    };


    virtual ~DiffusionRelaxationComplex(){}; 

    void initialization(size_t index_i, Real dt = 0.0)
    {
            Vec3d E_rate = Vec3d::Zero();
            
            Neighborhood &inner_neighborhood = inner_configuration_[index_i];
            for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ik
            {
                size_t index_k = inner_neighborhood.j_[n];
                Vec3d gradW_ikV_k = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
                           
                E_rate += (voltage_[index_k] - voltage_[index_i])  * (B_[index_i].transpose() * gradW_ikV_k); // HOW TO DEFINE IT
                                           
            }
           
             E_[index_i] = E_rate;

            Vec6d G_rate = Vec6d::Zero();
       
            Mat6d SC_rate = Mat6d::Zero();
            Real H_rate = 1.0;
            for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ik
            {
                size_t index_j = inner_neighborhood.j_[n];
                Vec3d r_ij =  -inner_neighborhood.r_ij_vector_[n];

                Vec3d gradW_ijV_j = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
                Vec6d S_ = Vec6d(r_ij[0] * r_ij[0], r_ij[1] * r_ij[1], r_ij[2] * r_ij[2], r_ij[0] * r_ij[1], r_ij[1] * r_ij[2], r_ij[2] * r_ij[0]);
                H_rate = r_ij.dot(B_[index_i].transpose() * gradW_ijV_j) / pow(r_ij.norm(), 4.0);
                
                Real FF_ = 2.0 * (voltage_[index_j] - voltage_[index_i] - r_ij.dot(E_[index_i]));    
                G_rate += S_ *H_rate * FF_;
                
                //TO DO
                Vec6d C_ = Vec6d::Zero();
                C_[0] = (r_ij[0] * r_ij[0]- r_ij.dot(A1_[index_i]));
                C_[1] = (r_ij[1] * r_ij[1]- r_ij.dot(A2_[index_i]));
                C_[2] = (r_ij[2] * r_ij[2]- r_ij.dot(A3_[index_i]));
                C_[3] = (r_ij[0] * r_ij[1]- r_ij.dot(A4_[index_i]));
                C_[4] = (r_ij[1] * r_ij[2]- r_ij.dot(A5_[index_i]));
                C_[5] = (r_ij[2] * r_ij[0]- r_ij.dot(A6_[index_i])); 

                SC_rate += S_ * H_rate * C_.transpose();   

            }

            G_[index_i] = G_rate;
            SC_[index_i] = SC_rate;
        
    };


     void interaction(size_t index_i, Real dt = 0.0)
    {
          
            Mat6d SC_rate_contact = Mat6d::Zero();
            Vec6d G_rate_contact = Vec6d::Zero();
            Real H_rate_contact = 1.0;
            for (size_t k = 0; k < contact_configuration_.size(); ++k)
            {
                Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
                for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
                {
                    Vec3d r_ij = -contact_neighborhood.r_ij_vector_[n];  
                    Vec3d gradW_ijV_j = contact_neighborhood.dW_ijV_j_[n] * contact_neighborhood.e_ij_[n];

                    Vec6d S_ = Vec6d(r_ij[0] * r_ij[0], r_ij[1] * r_ij[1], r_ij[2] * r_ij[2], r_ij[0] * r_ij[1], r_ij[1] * r_ij[2], r_ij[2] * r_ij[0]);
                    Real FF_ = 2.0 * (0.0 - r_ij.dot(E_[index_i])); ///here when it is periodic boundary condition, should notice the 0.0
                    H_rate_contact = r_ij.dot(B_[index_i].transpose() * gradW_ijV_j) / pow(r_ij.norm(), 4.0);
            
                    Vec6d C_ = Vec6d::Zero();
                    C_[0] = (r_ij[0] * r_ij[0]- r_ij.dot(A1_[index_i]));
                    C_[1] = (r_ij[1] * r_ij[1]- r_ij.dot(A2_[index_i]));
                    C_[2] = (r_ij[2] * r_ij[2]- r_ij.dot(A3_[index_i]));
                    C_[3] = (r_ij[0] * r_ij[1]- r_ij.dot(A4_[index_i]));
                    C_[4] = (r_ij[1] * r_ij[2]- r_ij.dot(A5_[index_i]));
                    C_[5] = (r_ij[2] * r_ij[0]- r_ij.dot(A6_[index_i]));
 

                    SC_rate_contact += S_ * H_rate_contact * C_.transpose();
                    G_rate_contact += S_ * H_rate_contact * FF_;

                }
                SC_[index_i] += SC_rate_contact;
                G_[index_i] += G_rate_contact;
            }
    
            
         Laplacian_[index_i] =  SC_[index_i].inverse() * G_[index_i];

         Mat3d Laplacian_transform = Mat3d { 
                  { Laplacian_[index_i][0], 0.5 * Laplacian_[index_i][3],  0.5 * Laplacian_[index_i][5] },  
                     {0.5 * Laplacian_[index_i][3],   Laplacian_[index_i][1], 0.5 * Laplacian_[index_i][4]},
                      {0.5 * Laplacian_[index_i][5],  0.5 * Laplacian_[index_i][4],  Laplacian_[index_i][2]}
                 }; 
         Mat3d Laplacian_noniso = decomposed_transform_tensor[index_i]  * Laplacian_transform 
                              * decomposed_transform_tensor[index_i].transpose();
        diffusion_dt_[index_i]  =  Laplacian_noniso(0,0) + Laplacian_noniso(1,1) + Laplacian_noniso(2,2);
         
        
    };


    void update(size_t index_i, Real dt = 0.0)
    { 
      voltage_[index_i] += dt * diffusion_dt_[index_i];
    };

};

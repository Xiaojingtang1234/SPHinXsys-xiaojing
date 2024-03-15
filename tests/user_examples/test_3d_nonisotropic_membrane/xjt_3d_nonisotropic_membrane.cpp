///**
// * @file twisting_column.cpp
// * @brief This is an example of solid with classic neohookean model
// * to demonstrate the robustness of the formulation with Kirchhoff stress decomposition.
// * @author Chi Zhang  and Xiangyu Hu
// * @ref 	DOI: 10.1016/j.cma.2014.09.024
// */

#include "particle_momentum_dissipation.h"
#include "particle_momentum_dissipation.hpp"
#include "porous_media_dynamics.h"
#include "porous_media_solid.h"
#include "porous_solid_particles.h"
#include "sphinxsys.h" 

//dimensional value  (if use)
/*
 * 			Unit :
 *			time t = s 
 * 			length l = mm
 * 			mass m = g
 *			density rho = g * (mm)^(-3)
 *			Pressure pa = g * (mm)^(-1) * (s)^(-2)
 *			diffusion d = (mm)^(2) / s 
            physical_viscosity = g / mm / s
*/
using namespace SPH;
//------------------------------------------------------------------------------
// global parameters for the case
//------------------------------------------------------------------------------
Real PL = 10.0e-3 ;	/**< X-direction domain. */  // non-10 
Real PH = 0.125e-3; /**< Z-direction domain. */ // non-0.125
Real BC = 0.3 * PL;
//!!!!!!!!!!!!!!notion: system (ref_large), dt= scaling_factor * dt;
/** Domain bounds of the system. */
Vec3d domain_lower_bound(-PL, -PL, -2.0 * PH);
Vec3d domain_upper_bound(2.0 * PL, 2.0 * PL, 6.0 * PH);
BoundingBox system_domain_bounds(domain_lower_bound, domain_upper_bound);
int resolution(20);

int z_num = 6;
Real ratio_ = 1.0;
// reference particle spacing
Real resolution_ref = PH / z_num;
Real resolution_ref_large = ratio_ * resolution_ref;
Real BW = 3.0 * resolution_ref_large;

Real Volume = PL * PL * PH;  
int x_num = PL / resolution_ref_large + 7;
int y_num = x_num;

Vec3d scaling_vector = Vec3d(1.0, 1.0, 1.0 / ratio_);
Real scaling_factor = 1.0 / ratio_;

/** Define application dependent particle generator for thin structure. */
class NonisotropicParticleGenerator : public ParticleGenerator
{
public:
	NonisotropicParticleGenerator(SPHBody &sph_body) : ParticleGenerator(sph_body){};

	virtual void initializeGeometricVariables() override
	{
		// set particles directly
		for (int i = 0; i < x_num; i++)
		{
			for (int j = 0; j < y_num; j++)
			{
				for (int k = 0; k < z_num; k++)
				{
					Real x = (i + 0.5 - 3.0) * resolution_ref_large;
					Real y = (j + 0.5 - 3.0) * resolution_ref_large;
					Real z = (k + 0.5) * resolution_ref;
					initializePositionAndVolumetricMeasure(Vec3d(x, y, z),
						(resolution_ref * resolution_ref_large * resolution_ref_large));
				}
			}
		}
	}
};

Real BL = 6.0 * resolution_ref_large;
Real BH = 3.0 * resolution_ref;

class AnisotropicParticleGeneratorBoundary : public ParticleGenerator
{
  public:
    AnisotropicParticleGeneratorBoundary(SPHBody &sph_body) : ParticleGenerator(sph_body){};

    virtual void initializeGeometricVariables() override
    {
        // set particles directly
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < (y_num + 8); j++)
            {
                for (int k = 0; k < (z_num ); k++)
				{
					Real x = (i + 0.5 - 3.0 - 4.0) * resolution_ref_large;
					Real y = (j + 0.5 - 3.0 - 4.0) * resolution_ref_large;
					Real z = (k + 0.5 ) * resolution_ref;
					initializePositionAndVolumetricMeasure(Vec3d(x, y, z),
						(resolution_ref * resolution_ref_large * resolution_ref_large));
				}
				
            }
        }

 		// set particles directly
        for (int i = 0; i <  4 ; i++)
        {
            for (int j = 0; j < (y_num + 8); j++)
            {
                for (int k = 0; k < (z_num ); k++)
				{
					Real x = (x_num + i + 0.5 - 3.0 ) * resolution_ref_large;
					Real y = (j + 0.5 - 3.0  -4.0) * resolution_ref_large;
					Real z = (k + 0.5 ) * resolution_ref;
					initializePositionAndVolumetricMeasure(Vec3d(x, y, z),
						(resolution_ref * resolution_ref_large * resolution_ref_large));
				}
				
            }
        }


// set particles directly
		for (int i = 0; i < x_num; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				for (int k = 0; k < z_num; k++)
				{
					Real x = (i + 0.5 - 3.0) * resolution_ref_large;
					Real y = (j + 0.5 - 3.0 - 4.0) * resolution_ref_large;
					Real z = (k + 0.5) * resolution_ref;
					initializePositionAndVolumetricMeasure(Vec3d(x, y, z),
						(resolution_ref * resolution_ref_large * resolution_ref_large));
				}
			}
		}


// set particles directly
		for (int i = 0; i < x_num; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				for (int k = 0; k < z_num; k++)
				{
					Real x = (i + 0.5 - 3.0) * resolution_ref_large;
					Real y = (y_num + j + 0.5 - 3.0 ) * resolution_ref_large;
					Real z = (k + 0.5) * resolution_ref;
					initializePositionAndVolumetricMeasure(Vec3d(x, y, z),
						(resolution_ref * resolution_ref_large * resolution_ref_large));
				}
			}
		}



// set particles directly
		for (int i = 0; i <  (x_num + 8 ); i++)
		{
			for (int j = 0; j < (y_num + 8); j++)
			{
				for (int k = 0; k < 4; k++)
				{
					Real x = (i + 0.5 - 3.0-4.0) * resolution_ref_large;
					Real y = (j + 0.5 - 3.0-4.0) * resolution_ref_large;
					Real z = (k + 0.5 - 4.0) * resolution_ref;
					initializePositionAndVolumetricMeasure(Vec3d(x, y, z),
						(resolution_ref * resolution_ref_large * resolution_ref_large));
				}
			}
		}

		// set particles directly
		for (int i = 0; i < (x_num + 8 ); i++)
		{
			for (int j = 0; j < (y_num + 8); j++)
			{
				for (int k = 0; k < 4; k++)
				{
					Real x = (i + 0.5 - 3.0 - 4.0) * resolution_ref_large;
					Real y = (j + 0.5 - 3.0- 4.0) * resolution_ref_large;
					Real z = (z_num + k + 0.5) * resolution_ref;
					initializePositionAndVolumetricMeasure(Vec3d(x, y, z),
						(resolution_ref * resolution_ref_large * resolution_ref_large));
				}
			}
		}


         
 
    }
};

//----------------------------------------------------------------------
//	Material properties of the fluid.
//----------------------------------------------------------------------
Real rho0_s = 2.0e3;   //non-2.0e-3; 
Real poisson = 0.26316; /**< Poisson ratio. */
Real Youngs_modulus = 8.242e6; //non-8.242e6;

Real diffusivity_constant_ = 1.0e-10; //non-1.0e-4
Real fluid_initial_density_ = 1.0e3; //  non-1.0e-3;
Real water_pressure_constant_ = 3.0e6; //pa not chnaged
Real physical_viscosity = 100.0;  

Real refer_density_energy = 0.5 * water_pressure_constant_ ; 

Real End_Time = 3000.0;
Real setup_saturation_time = 450;
Real time_to_full_saturation = 0.0 * setup_saturation_time;
Real full_saturation = 0.4;


Vec3d halfsize_membrane(0.5 * PL, 0.5 * PL, 0.5 * PH);
Vec3d translation_membrane(0.5 * PL, 0.5 * PL, 0.5 * PH);

Vec3d halfsize_holder(0.5 * PL + BW, 0.5 * PL + BW, 0.5 * PH);
Vec3d translation_holder(0.5 * PL, 0.5 * PL, 0.5 * PH);

Real boundary_x = 4.0 * resolution_ref_large;
Real boundary_z = 4.0 * resolution_ref;

Vec3d halfsize_boundary(0.5 * PL + boundary_x, 0.5 * PL + boundary_x, 0.5 * PH + boundary_z );
Vec3d translation_boundary(0.5 * PL, 0.5 * PL, 0.5 * PH);


Vec3d halfsize_initial_saturation(BC / 2.0, BC / 2.0, PH / 4.0);
Vec3d translation_initial_saturation(0.5 * PL, 0.5 * PL, 0.75 * PH);

class Membrane : public ComplexShape
{
public:
	explicit Membrane(const std::string &shape_name) : ComplexShape(shape_name)
	{
		add<TriangleMeshShapeBrick>(halfsize_membrane, resolution, translation_membrane);
		add<TriangleMeshShapeBrick>(halfsize_holder, resolution, translation_holder);
	}
};

class Boundary : public ComplexShape
{
public:
	explicit Boundary(const std::string &shape_name) : ComplexShape(shape_name)
	{ 
		add<TriangleMeshShapeBrick>(halfsize_boundary, resolution, translation_boundary);
	}
};



 /**
 * @class CorrectConfiguration
 * @brief obtain the corrected initial configuration in strong form
 */

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



class SaturationInitialCondition : public multi_species_continuum::PorousMediaSaturationDynamicsInitialCondition
{
  public:
    SaturationInitialCondition(BodyPartByParticle &body_part) : multi_species_continuum::PorousMediaSaturationDynamicsInitialCondition(body_part){};
    virtual ~SaturationInitialCondition(){};

  protected:
    void update(size_t index_i, Real dt = 0.0)
    {
       // Real current_time = GlobalStaticVariables::physical_time_;
		//Real saturation = current_time < time_to_full_saturation
		//					  ? current_time * full_saturation / time_to_full_saturation
		//					  : full_saturation;
		fluid_saturation_[index_i] = full_saturation;
		fluid_mass_[index_i] = fluid_saturation_[index_i] * fluid_initial_density_ * Vol_update_[index_i];
		total_mass_[index_i] = rho_n_[index_i] * Vol_update_[index_i] + fluid_mass_[index_i]; 
 
    };
};
 
 using Mat6d = Eigen::Matrix<Real, 6, 6>;
using Vec6d = Eigen::Matrix<Real, 6, 1>;
class NonisotropicSaturationRelaxationInPorousMedia  : public LocalDynamics, public  multi_species_continuum::PorousMediaSolidDataComplex
{ 
    public:
    NonisotropicSaturationRelaxationInPorousMedia(ComplexRelation &complex_relation): 
                LocalDynamics(complex_relation.getInnerRelation().getSPHBody()),  multi_species_continuum::PorousMediaSolidDataComplex(complex_relation),  
		   pos_(particles_->pos_), B_(particles_->B_),
		    Vol_update_(particles_->Vol_update_), fluid_saturation_(particles_->fluid_saturation_),
			total_mass_(particles_->total_mass_), fluid_mass_(particles_->fluid_mass_),
			dfluid_mass_dt_(particles_->dfluid_mass_dt_), relative_fluid_flux_(particles_->relative_fluid_flux_),Vol_(particles_->Vol_),
		   particle_number(complex_relation.getInnerRelation().getSPHBody().getBaseParticles().real_particles_bound_)
	{
		   
       particles_->registerVariable(E_, "FirstOrderCorrectionVectorE", [&](size_t i) -> Vec3d { return Eps * Vec3d::Identity(); });

        std::cout<<particle_number<<std::endl;

        particles_->registerVariable(Laplacian_x, "Laplacian_x", [&](size_t i) -> Real { return Real(0.0); });
        particles_->registerVariable(Laplacian_y, "Laplacian_y", [&](size_t i) -> Real { return Real(0.0); });
        particles_->registerVariable(Laplacian_z, "Laplacian_z", [&](size_t i) -> Real { return Real(0.0); });
		 
		   for (size_t i = 0; i != particle_number; ++i)
        {
            SC_.push_back(Mat6d::Identity()); 
            G_.push_back(Vec6d::Identity()); 
            Laplacian_.push_back(Vec6d::Identity()); 
        }
 
        fluid_initial_density = particles_->porous_solid_.getFulidInitialDensity();
        diffusion_coeff_ = particles_->porous_solid_.getDiffusivityConstant();
        rho0_ = particles_->porous_solid_.ReferenceDensity();

	};

      virtual ~NonisotropicSaturationRelaxationInPorousMedia(){};

    StdLargeVec<Vec3d> &pos_;
    StdLargeVec<Mat3d> &B_;

    StdLargeVec<Real> &Vol_update_;
	StdLargeVec<Real> &fluid_saturation_;

	StdLargeVec<Real> &total_mass_;
	StdLargeVec<Real> &fluid_mass_;
	StdLargeVec<Real> &dfluid_mass_dt_;
	
	StdLargeVec<Vec3d> &relative_fluid_flux_;
	StdLargeVec<Real> &Vol_;

     
    StdLargeVec<Vec3d> E_;
    
    StdLargeVec<Mat6d> SC_;
    StdLargeVec<Vec6d> G_; 
    StdLargeVec<Vec6d> Laplacian_;
    size_t particle_number;

  
    StdLargeVec<Real> Laplacian_x, Laplacian_y, Laplacian_z;
    Real diffusion_coeff_;
     Real fluid_initial_density,rho0_;
   
   
 protected:
    void initialization(size_t index_i, Real dt = 0.0)
    {  
		 Vec3d fluid_saturation_gradient  = Vec3d::Zero();
        Neighborhood &inner_neighborhood = inner_configuration_[index_i];
        Vec3d E_rate = Vec3d::Zero();
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ik
        {
            size_t index_k = inner_neighborhood.j_[n];
            Vec3d gradW_ikV_k = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];

            E_rate += (fluid_saturation_[index_k] - fluid_saturation_[index_i]) 
						* (B_[index_i].transpose() * gradW_ikV_k); // HOW TO DEFINE IT???
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
			 
            Real FF_ =  (fluid_saturation_[index_j] - fluid_saturation_[index_i] - r_ij.dot(E_[index_i]));
            G_rate += S_ *H_rate * FF_;

            fluid_saturation_gradient -= (fluid_saturation_[index_i] - fluid_saturation_[index_j]) * gradW_ijV_j;
											
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
		relative_fluid_flux_[index_i] = -diffusion_coeff_ * fluid_initial_density
		        * fluid_saturation_[index_i] * fluid_saturation_gradient;
			
 
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
          ///here when it is isothermal boundary condition, thsi is 0.0, when boundary condition this is not 0.0 the 0.0
                Real FF_ =   ( 0.0 - r_ij.dot(E_[index_i]) ); 
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
			SC_[index_i] += SC_rate_contact;
            G_[index_i] += G_rate_contact;
        }

        Laplacian_[index_i] = diffusion_coeff_ * SC_[index_i].inverse() * G_[index_i];

        Laplacian_x[index_i] = Laplacian_[index_i][0];
        Laplacian_y[index_i] = Laplacian_[index_i][1];
        Laplacian_z[index_i] = Laplacian_[index_i][2];
		dfluid_mass_dt_[index_i] =  Vol_update_[index_i] * fluid_initial_density
								* (Laplacian_[index_i][0] + Laplacian_[index_i][1]+ Laplacian_[index_i][2]);
    };

   void update(size_t index_i, Real dt = 0.0)
    {
      fluid_mass_[index_i] += dfluid_mass_dt_[index_i] * dt;
	  fluid_mass_[index_i] =  0.995 * fluid_mass_[index_i];
		 // update total mass
	   total_mass_[index_i] = rho0_ * Vol_[index_i] + fluid_mass_[index_i];
	   //  update fluid saturation 
	   fluid_saturation_[index_i] = fluid_mass_[index_i] / fluid_initial_density / Vol_update_[index_i];
	}

}; 

class TemperatureObserverParticleGenerator : public ObserverParticleGenerator
{
  public:
    explicit TemperatureObserverParticleGenerator(SPHBody &sph_body)
        : ObserverParticleGenerator(sph_body)
    {
        size_t number_of_observation_points = 21;
        Real range_of_measure = 1.0 * PL;
        Real start_of_measure = 0.0 * PL;

        for (size_t i = 0; i < number_of_observation_points; ++i)
        {
            Vec3d point_coordinate(range_of_measure * (Real)i / (Real)(number_of_observation_points - 1) + start_of_measure, 0.5 * PL, PH / 2.0);
            positions_.push_back(point_coordinate);
        }

    }
};
 


//------------------------------------------------------------------------------
// the main program
//------------------------------------------------------------------------------

int main()
{
	//----------------------------------------------------------------------
	//	Build up the environment of a SPHSystem with global controls.
	//----------------------------------------------------------------------
	SPHSystem system(system_domain_bounds, resolution_ref_large);
	// handle command line arguments
	IOEnvironment io_environment(system);

	//----------------------------------------------------------------------
	//	Creating body, materials and particles.
	//----------------------------------------------------------------------
	// the oscillating beam
	SolidBody membrane_body(system, makeShared<Membrane>("Membrane"));

	  membrane_body.sph_adaptation_->resetKernel<AnisotropicKernel<KernelWendlandC2>>(scaling_vector);
	membrane_body.defineParticlesAndMaterial< multi_species_continuum::PorousMediaParticles,  multi_species_continuum::PorousMediaSolid>
		(rho0_s, Youngs_modulus, poisson, diffusivity_constant_,
		fluid_initial_density_, water_pressure_constant_);
	membrane_body.generateParticles<NonisotropicParticleGenerator>();


    SolidBody boundary_body(system, makeShared<Boundary>("Boundary"));
    boundary_body.sph_adaptation_->resetKernel<AnisotropicKernel<KernelWendlandC2>>(scaling_vector);
    boundary_body.defineParticlesAndMaterial<SolidParticles, Solid>();
    boundary_body.generateParticles<AnisotropicParticleGeneratorBoundary>();
 
	/** Define Observer. */
	ObserverBody my_observer(system, "MyObserver");
	my_observer.defineAdaptationRatios(1.15, 2.0);
 	my_observer.sph_adaptation_->resetKernel<AnisotropicKernel<KernelWendlandC2>>(scaling_vector);
	my_observer.generateParticles<TemperatureObserverParticleGenerator>();

	/**body relation topology */
	InnerRelation membrane_body_inner(membrane_body);
	ContactRelation my_observer_contact(my_observer, {&membrane_body});
    ComplexRelation diffusion_block_complex(membrane_body, {&boundary_body});

	/**define simple data file input and outputs functions. */
	BodyStatesRecordingToVtp write_states(io_environment, system.real_bodies_);
	BodyStatesRecordingToPlt write_statesplt(io_environment, system.real_bodies_);

	//-----------------------------------------------------------------------------
	// this section define all numerical methods will be used in this case
	//-----------------------------------------------------------------------------
	// corrected strong configuration
	Dynamics1Level<NonisotropicKernelCorrectionMatrixComplex> beam_corrected_configuration(diffusion_block_complex);

	// time step size calculation
	ReduceDynamics<solid_dynamics::AcousticTimeStepSize> computing_time_step_size(membrane_body);
	ReduceDynamics<multi_species_continuum::GetSaturationTimeStepSize> saturation_time_step_size(membrane_body);

 

	// stress relaxation for the beam
	Dynamics1Level<multi_species_continuum::PorousMediaStressRelaxationFirstHalf> stress_relaxation_first_half(membrane_body_inner);
	Dynamics1Level<multi_species_continuum::PorousMediaStressRelaxationSecondHalf> stress_relaxation_second_half(membrane_body_inner);
	// Saturation relaxation for the beam
	//Dynamics1Level<multi_species_continuum::SaturationRelaxationInPorousMedia> saturation_relaxation(membrane_body_inner);

    Dynamics1Level<NonisotropicSaturationRelaxationInPorousMedia> saturation_relaxation(diffusion_block_complex);


	SharedPtr<ComplexShape> consrtain_shape_ = makeShared<ComplexShape>("ConstrainShape");
	consrtain_shape_->add<TriangleMeshShapeBrick>(halfsize_holder, resolution, translation_holder);
	consrtain_shape_->subtract<TriangleMeshShapeBrick>(halfsize_membrane, resolution, translation_membrane);

	BodyRegionByParticle constrain_holder(membrane_body, consrtain_shape_);
	SimpleDynamics<multi_species_continuum::MomentumConstraint> clamp_constrain_beam_base(constrain_holder);
  
  
    SharedPtr<ComplexShape> saturation_shape_ = makeShared<ComplexShape>("Holder");
	saturation_shape_->add<TriangleMeshShapeBrick>(halfsize_initial_saturation, resolution, translation_initial_saturation);
	 
	BodyRegionByParticle beam_saturation_shape(membrane_body,saturation_shape_);
	SimpleDynamics<SaturationInitialCondition>
						constrain_beam_saturation(beam_saturation_shape);
	
	
	
	
	ReduceDynamics<TotalMechanicalEnergy> get_kinetic_energy_energy(membrane_body); //, refer_density_energy

	/** Damping for one ball */
	 DampingWithRandomChoice<InteractionSplit<multi_species_continuum::PorousMediaDampingPairwiseInner<Vec3d>>>
        beam_damping(0.5, membrane_body_inner, "TotalMomentum", physical_viscosity);
    //------------------------------------ ---------------------------------
	// outputs
	//-----------------------------------------------------------------------------
 
  BodyStatesRecordingToVtp write_beam_states(io_environment, system.real_bodies_);
  ObservedQuantityRecording<Vec3d>
        write_beam_tip_position("Position", io_environment, my_observer_contact);

 ObservedQuantityRecording<Real>
		write_beam_center_saturation("FluidSaturation", io_environment, my_observer_contact);

 ReducedQuantityRecording<ReduceDynamics<TotalMechanicalEnergy>>
		write_total_mechanical_energy(io_environment, membrane_body);//, refer_density_energy

	/**
	 * @brief Setup geometry and initial conditions
	 */

	int ite = 0;
	int total_ite = 0;
	int Dt_ite = 0;
	GlobalStaticVariables::physical_time_ = 0.0;

	system.initializeSystemCellLinkedLists();
	system.initializeSystemConfigurations();
	beam_corrected_configuration.exec();
	constrain_beam_saturation.exec();
	

	//----------------------------------------------------------------------
	//	Setup computing and initial conditions.
	//----------------------------------------------------------------------

	write_states.writeToFile(0);
	write_statesplt.writeToFile(0);
	write_beam_tip_position.writeToFile(0);
	write_total_mechanical_energy.writeToFile(0);
	write_beam_center_saturation.writeToFile(0);

	Real dt = 0.0; // default acoustic time step sizes
	Real D_Time = End_Time / 100.0;
	// statistics for computing time
	TickCount t1 = TickCount::now();
	TickCount::interval_t interval;

	std::string filefullpath = io_environment.output_folder_ + "/" + "3d_iterations" + ".dat";
	std::ofstream out_file(filefullpath.c_str(), std::ios::app);
	out_file << "\"run_time\""
			 << "   "
			 << "\"ite\""
			 << "   "
			 << "\"total_ite\""
			 << "   ";
	out_file << "\n";
	out_file.close();

	//-----------------------------------------------------------------------------
	// from here the time stepping begins
	//-----------------------------------------------------------------------------
	// computation loop starts
	while (GlobalStaticVariables::physical_time_ < End_Time)
	{
		Real integration_time = 0.0;
		while (integration_time < D_Time)
		{
			Real Dt = scaling_factor * scaling_factor * saturation_time_step_size.exec();
			if (GlobalStaticVariables::physical_time_ < setup_saturation_time)
			{
				constrain_beam_saturation.exec();
			}
			saturation_relaxation.exec(Dt);
           
			int stress_ite = 0;
			Real relaxation_time = 0.0;
			Real total_kinetic_energy = 1.0e8;
				Dt_ite ++;

			while (relaxation_time < Dt)
			{
				if (total_kinetic_energy > (0.2))  // not here, total mechanicla neargy need to chnage to 2000
				{
					stress_relaxation_first_half.exec(dt);
					clamp_constrain_beam_base.exec();
					beam_damping.exec(dt);
					clamp_constrain_beam_base.exec();

					stress_relaxation_second_half.exec(dt);
					// this density energy is   in total mechanical energy 
					total_kinetic_energy = get_kinetic_energy_energy.exec()  ; 

					ite++;
					stress_ite++;

					dt =  scaling_factor * SMIN(computing_time_step_size.exec(), Dt);

					if (ite % 1000 == 0)
					{
						write_total_mechanical_energy.writeToFile(ite);
						std::cout << "N=" << ite << " Time: "
								  << GlobalStaticVariables::physical_time_ << "  Dt:" << Dt << "	dt: "
								  << dt << "  Dt/ dt:" << Dt / dt << "\n";
					}
				}
                 total_ite++;
				relaxation_time += dt;
				integration_time += dt;
				GlobalStaticVariables::physical_time_ += dt;
			}
			 std::cout<< "One Diffusion finishes  "
				 << "total_kinetic_energy  " << total_kinetic_energy
				 << "     stress_ite  " << stress_ite <<  std::endl;
			write_total_mechanical_energy.writeToFile(ite);
			total_ite = total_ite + int(Dt / dt);

			std::ofstream out_file(filefullpath.c_str(), std::ios::app);
			out_file << GlobalStaticVariables::physical_time_ << "   ";
			out_file << ite << "   ";
			out_file << total_ite << "   ";
			out_file << "\n";
			out_file.close();

			write_beam_center_saturation.writeToFile(ite);
			write_beam_tip_position.writeToFile(ite);
		}
		write_states.writeToFile(ite); // should be change to nothing
		write_statesplt.writeToFile(ite);
      
        TickCount t2 = TickCount::now();
	    TickCount t3 = TickCount::now();
        interval += t3 - t2;
	}
	
	TickCount t4 = TickCount::now();
    TimeInterval tt;
	tt = t4 - t1 - interval;

	std::cout << "Total wall time for computation: " << tt.seconds() << " seconds."
			  << "  Iterations:  " << ite << std::endl;
	std::cout << "Total iterations computation:  " << GlobalStaticVariables::physical_time_ / dt
			  << "  Total iterations:  " << total_ite << std::endl;

	return 0;
}

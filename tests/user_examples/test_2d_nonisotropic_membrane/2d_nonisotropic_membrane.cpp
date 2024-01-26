/* ---------------------------------------------------------------------------*
 *            SPHinXsys: 2D menmbrane example-one body version           *
 * ----------------------------------------------------------------------------*
 * This is the one of the basic test cases, also the first case for            *
 * understanding SPH method for solid simulation.                              *
 * In this case, the constraint of the beam is implemented with                *
 * internal constrained subregion.                                             *
 * ----------------------------------------------------------------------------*/
#include "particle_momentum_dissipation.h"
#include "particle_momentum_dissipation.hpp"
#include "porous_media_dynamics.h"
#include "porous_media_solid.h"
#include "porous_solid_particles.h"
#include "sphinxsys.h"
using namespace SPH;
//------------------------------------------------------------------------------
// global parameters for the case
//------------------------------------------------------------------------------
Real PL = 5.0e-3;  // membrane length
Real PH = 0.125e-3; // membrane thickenss
Real BC = PL * 0.15;

int y_num = 4;
Real ratio_ = 1.0;
// reference particle spacing
Real resolution_ref = PH / y_num;
Real resolution_ref_large = ratio_ * resolution_ref;
Real BW = 3.0 * resolution_ref_large;

Vec2d scaling_vector = Vec2d(1.0, 1.0 / ratio_);
Real scaling_factor = 1.0 / ratio_;

int x_num = PL / resolution_ref_large + 7;

/** Domain bounds of the system. */
BoundingBox system_domain_bounds(Vec2d(-PL, -PL),
                                 Vec2d(2.0 * PL, PL));

//----------------------------------------------------------------------
//	Material properties of the fluid.
//----------------------------------------------------------------------
Real rho_0 = 2.0e3;   
Real poisson = 0.26316;
Real Youngs_modulus = 8.242e6;
Real physical_viscosity = 400.0;

Real diffusivity_constant_ = 1.0e-10;
Real fulid_initial_density_ = 1.0e3;
Real water_pressure_constant_ = 3.0e6;
Real saturation = 0.4;

Real refer_density_energy = 0.5 * water_pressure_constant_;

//----------------------------------------------------------------------
//	Geometric shapes used in the system.
//----------------------------------------------------------------------
// a membrane base shape

std::vector<Vecd> beam_base_shape{
    Vecd(-resolution_ref_large * 3.0, -PH / 2.0), Vecd(-resolution_ref_large * 3.0, PH / 2.0), Vecd(0.0, PH / 2.0),
    Vecd(0.0, -PH / 2.0), Vecd(-resolution_ref_large * 3.0, -PH / 2.0)};

// a membrane shape
std::vector<Vecd> beam_shape{Vecd(0.0, -PH / 2.0), Vecd(0.0, PH / 2.0),
                             Vecd(PL, PH / 2.0), Vecd(PL, -PH / 2.0), Vecd(0.0, -PH / 2.0)};

// a membrane end shape
std::vector<Vecd> beam_end_shape{
    Vecd(PL, -PH / 2.0), Vecd(PL, PH / 2.0),
    Vecd(PL + 4.0 * resolution_ref_large, PH / 2.0), Vecd(PL + 4.0 * resolution_ref_large, -PH / 2.0),
    Vecd(PL, -PH / 2.0)};

// a membrane saturation shape
std::vector<Vecd> beam_saturation_shape{
    Vecd(PL / 2.0 - BC, 0.0), Vecd(PL / 2.0 - BC, PH / 2.0), Vecd(PL / 2.0 + BC, PH / 2.0),
    Vecd(PL / 2.0 + BC, 0.0), Vecd(PL / 2.0 - BC, 0.0)};

// Beam observer location
StdVec<Vecd> observation_location = {Vecd(PL / 4.0, 0.0)};

//----------------------------------------------------------------------
//	Define the beam body
//----------------------------------------------------------------------
class Beam : public MultiPolygonShape
{
  public:
    explicit Beam(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(beam_base_shape, ShapeBooleanOps::add);
        multi_polygon_.addAPolygon(beam_shape, ShapeBooleanOps::add);
        multi_polygon_.addAPolygon(beam_end_shape, ShapeBooleanOps::add);
    }
};


class Boundary: public MultiPolygonShape
{
  public:
    explicit Boundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        std::vector<Vec2d> boundary;
        boundary.push_back(Vecd(-resolution_ref * 7.0, -PH / 2.0 - resolution_ref * 3.0));
        boundary.push_back(Vecd(-resolution_ref * 7.0, PH / 2.0 + resolution_ref * 3.0));
        boundary.push_back(Vec2d(PL + 8.0 * resolution_ref, PH / 2.0 + resolution_ref * 3.0));
        boundary.push_back(Vec2d(PL + 8.0 * resolution_ref, -PH / 2.0 - resolution_ref * 3.0));
        boundary.push_back(Vec2d(-resolution_ref * 7.0, -PH / 2.0 - resolution_ref * 3.0));
        multi_polygon_.addAPolygon(boundary, ShapeBooleanOps::add);

        multi_polygon_.addAPolygon(beam_base_shape, ShapeBooleanOps::sub);
        multi_polygon_.addAPolygon(beam_shape, ShapeBooleanOps::sub);
        multi_polygon_.addAPolygon(beam_end_shape, ShapeBooleanOps::sub);
    }
};

//----------------------------------------------------------------------
//	define the beam base which will be constrained.
//----------------------------------------------------------------------
MultiPolygon createBeamConstrainShape()
{
    MultiPolygon multi_polygon;
    multi_polygon.addAPolygon(beam_base_shape, ShapeBooleanOps::add);
    multi_polygon.addAPolygon(beam_shape, ShapeBooleanOps::sub);
    multi_polygon.addAPolygon(beam_end_shape, ShapeBooleanOps::add);
    return multi_polygon;
};

MultiPolygon createSaturationConstrainShape()
{
    MultiPolygon multi_polygon;
    multi_polygon.addAPolygon(beam_saturation_shape, ShapeBooleanOps::add);
    return multi_polygon;
};

//----------------------------------------------------------------------
//	application dependent initial condition
//----------------------------------------------------------------------
class SaturationInitialCondition : public multi_species_continuum::PorousMediaSaturationDynamicsInitialCondition
{
  public:
    SaturationInitialCondition(BodyPartByParticle &body_part) : multi_species_continuum::PorousMediaSaturationDynamicsInitialCondition(body_part){};
    virtual ~SaturationInitialCondition(){};

  protected:
    void update(size_t index_i, Real dt = 0.0)
    {
        fluid_saturation_[index_i] = saturation;
        fluid_mass_[index_i] = saturation * fulid_initial_density_ * Vol_update_[index_i];
        total_mass_[index_i] = rho_n_[index_i] * Vol_update_[index_i] + fluid_mass_[index_i];
    };
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
                Real x = (i + 0.5 - 3) * resolution_ref_large;
                Real y = (j + 0.5) * resolution_ref -PH / 2.0;
                initializePositionAndVolumetricMeasure(Vec2d(x, y), (resolution_ref * resolution_ref_large));
            }
        }
    }
};


class AnisotropicParticleGeneratorBoundary : public ParticleGenerator
{
  public:
    AnisotropicParticleGeneratorBoundary(SPHBody &sph_body) : ParticleGenerator(sph_body) {  };
   
      
    virtual void initializeGeometricVariables() override
    {
        // set particles directly
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < y_num + 6; j++)
            {
                Real x = (i + 0.5 - 6.0) * resolution_ref_large;
                Real y = (j + 0.5 - 3.0) * resolution_ref - PH / 2.0;
                initializePositionAndVolumetricMeasure(Vec2d(x, y), (resolution_ref * resolution_ref_large));
            }
        }

        // set particles directly
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < y_num + 6; j++)
            {
                Real x = (x_num + 0.5 - 3.0 + i) * resolution_ref_large;
                Real y = (j + 0.5 - 3.0) * resolution_ref - PH / 2.0;
                initializePositionAndVolumetricMeasure(Vec2d(x, y), (resolution_ref * resolution_ref_large));
            }
        }

        // set particles directly
        for (int i = 0; i < x_num; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Real x = (i + 0.5 - 3.0) * resolution_ref_large;
                Real y =  (j + 0.5 - 3.0) * resolution_ref - PH / 2.0;
                initializePositionAndVolumetricMeasure(Vec2d(x, y), (resolution_ref * resolution_ref_large));
            }
        }

        // set particles directly
        for (int i = 0; i < x_num; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Real x = (i + 0.5 - 3.0) * resolution_ref_large;
                Real y = (y_num + j + 0.5) * resolution_ref - PH / 2.0;
                initializePositionAndVolumetricMeasure(Vec2d(x, y), (resolution_ref * resolution_ref_large));
            }
        }
    }
};


typedef DataDelegateComplex<BaseParticles, BaseParticles>GeneralDataDelegateComplex;

class NonisotropicKernelCorrectionMatrixComplex : public LocalDynamics, public GeneralDataDelegateComplex
{
  public:
    NonisotropicKernelCorrectionMatrixComplex(ComplexRelation &complex_relation, Real alpha = Real(0))
        : LocalDynamics(complex_relation.getInnerRelation().getSPHBody()),
		GeneralDataDelegateComplex(complex_relation), 
		B_(*particles_->registerSharedVariable<Mat2d>("KernelCorrectionMatrix")) 
        {
   
       particles_->registerVariable(neigh_, "neighbour", [&](size_t i) -> Real { return Eps * Real(0.0); });
        
        };

    virtual ~NonisotropicKernelCorrectionMatrixComplex(){};
   
  protected:
    // SPHBody &contact_body_;
  //   StdLargeVec<Real> &neigh_boundary;
	 StdLargeVec<Mat2d> &B_;
     StdLargeVec<Real> neigh_;
     
   
	  void initialization(size_t index_i, Real dt = 0.0)
	  {
		  Mat2d local_configuration = Eps * Mat2d::Identity();
		  const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
		  for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
		  {   

             Real index_j = inner_neighborhood.j_[n];
             
			  Vec2d gradW_ij = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];

			  Vec2d r_ji = inner_neighborhood.r_ij_vector_[n];
			  local_configuration -= r_ji * gradW_ij.transpose();
              if (index_i == 248)
              {
                neigh_[index_j] = 1.0;
              }
              if (index_i == 64)
              {
                neigh_[index_j] = 1.0;
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
                //Real index_j = contact_neighborhood.j_[n];
              //   if (index_i == 248)
              {    
              //     neigh_boundary[index_j] = 1.0;
              }
             //if (index_i == 64)
              {
           //     neigh_boundary[index_j] = 1.0;
              }
              
				Vec2d r_ji = contact_neighborhood.r_ij_vector_[n];
				Vec2d gradW_ij = contact_neighborhood.dW_ijV_j_[n] * contact_neighborhood.e_ij_[n];
				local_configuration -= r_ji * gradW_ij.transpose();
			}
		}
		B_[index_i] += local_configuration; 
     }; 

	void update(size_t index_i, Real dt)
	{
		Mat2d inverse = B_[index_i].inverse();
		B_[index_i] = inverse;	
	}
};

class NonisotropicSaturationRelaxationInPorousMedia  : public LocalDynamics, public  multi_species_continuum::PorousMediaSolidDataComplex
{

 public:
    public:
    NonisotropicSaturationRelaxationInPorousMedia(ComplexRelation &complex_relation): 
                LocalDynamics(complex_relation.getInnerRelation().getSPHBody()),  multi_species_continuum::PorousMediaSolidDataComplex(complex_relation),  
		   pos_(particles_->pos_), B_(particles_->B_),
		    Vol_update_(particles_->Vol_update_), fluid_saturation_(particles_->fluid_saturation_),
			total_mass_(particles_->total_mass_), fluid_mass_(particles_->fluid_mass_),
			dfluid_mass_dt_(particles_->dfluid_mass_dt_), relative_fluid_flux_(particles_->relative_fluid_flux_),Vol_(particles_->Vol_),
		   particle_number(complex_relation.getInnerRelation().getSPHBody().getBaseParticles().real_particles_bound_)
	{
		   
       particles_->registerVariable(E_, "FirstOrderCorrectionVectorE", [&](size_t i) -> Vec2d { return Eps * Vec2d::Identity(); });

        std::cout<<particle_number<<std::endl;

        particles_->registerVariable(Laplacian_x, "Laplacian_x", [&](size_t i) -> Real { return Real(0.0); });
        particles_->registerVariable(Laplacian_y, "Laplacian_y", [&](size_t i) -> Real { return Real(0.0); });
        particles_->registerVariable(Laplacian_z, "Laplacian_z", [&](size_t i) -> Real { return Real(0.0); });
		 
		   for (size_t i = 0; i != particle_number; ++i)
        {
            SC_.push_back(Mat3d::Identity()); 
            G_.push_back(Vec3d::Identity()); 
            Laplacian_.push_back(Vec3d::Identity()); 
        }
 
        fluid_initial_density = particles_->porous_solid_.getFulidInitialDensity();
        diffusion_coeff_ = particles_->porous_solid_.getDiffusivityConstant();
        rho0_ = particles_->porous_solid_.ReferenceDensity();

	};

      virtual ~NonisotropicSaturationRelaxationInPorousMedia(){};

    StdLargeVec<Vec2d> &pos_;
    StdLargeVec<Mat2d> &B_;

    StdLargeVec<Real> &Vol_update_;StdLargeVec<Real> &fluid_saturation_;

	StdLargeVec<Real> &total_mass_;	StdLargeVec<Real> &fluid_mass_;
	StdLargeVec<Real> &dfluid_mass_dt_;
	
	StdLargeVec<Vec2d> &relative_fluid_flux_;
	StdLargeVec<Real> &Vol_;

     
    StdLargeVec<Vec2d> E_;
    
    StdLargeVec<Mat3d> SC_;
    StdLargeVec<Vec3d> G_; 
    StdLargeVec<Vec3d> Laplacian_;
    size_t particle_number;

  
    StdLargeVec<Real> Laplacian_x, Laplacian_y, Laplacian_z;
    Real diffusion_coeff_;
     Real fluid_initial_density,rho0_;
   
   
 protected:
    void initialization(size_t index_i, Real dt = 0.0)
    {  
		 Vec2d fluid_saturation_gradient  = Vec2d::Zero();
        Neighborhood &inner_neighborhood = inner_configuration_[index_i];
        Vec2d E_rate = Vec2d::Zero();
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ik
        {
            size_t index_k = inner_neighborhood.j_[n];
            Vec2d gradW_ikV_k = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];

            E_rate += (fluid_saturation_[index_k] - fluid_saturation_[index_i]) * (B_[index_i].transpose() * gradW_ikV_k); // HOW TO DEFINE IT???
        }
        E_[index_i] = E_rate;

         
        Vec3d G_rate = Vec3d::Zero();
        Mat3d SC_rate = Mat3d::Zero();
        Real H_rate = 1.0;
         for (size_t n = 0; n != inner_neighborhood.current_size_; ++n) // this is ik
        {
            size_t index_j = inner_neighborhood.j_[n];
            Vec2d r_ij = -inner_neighborhood.r_ij_vector_[n];
            Vec2d gradW_ijV_j = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
            Vec3d S_ = Vec3d(r_ij[0] * r_ij[0], r_ij[1] * r_ij[1], r_ij[0] * r_ij[1]);
            H_rate = r_ij.dot(B_[index_i].transpose() * gradW_ijV_j) / pow(r_ij.norm(), 4.0);
			 
            Real FF_ =  (fluid_saturation_[index_j] - fluid_saturation_[index_i] - r_ij.dot(E_[index_i]));
            G_rate += S_ *H_rate * FF_;

            fluid_saturation_gradient -= (fluid_saturation_[index_i] - fluid_saturation_[index_j]) * gradW_ijV_j;
											
             //TO DO
            Vec3d C_ = Vec3d::Zero();
            C_[0] = (r_ij[0] * r_ij[0]);
            C_[1] = (r_ij[1] * r_ij[1]);
            C_[2] = (r_ij[0] * r_ij[1]);
			 
            SC_rate += S_ *H_rate* C_.transpose();   

        }
        
        G_[index_i] = G_rate;
        SC_[index_i] = SC_rate;
		relative_fluid_flux_[index_i] = -diffusion_coeff_ * fluid_initial_density
		        * fluid_saturation_[index_i] * fluid_saturation_gradient;
			
 
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
                Vec2d r_ij = -contact_neighborhood.r_ij_vector_[n];
                Vec2d gradW_ijV_j = contact_neighborhood.dW_ijV_j_[n] * contact_neighborhood.e_ij_[n];

                Vec3d S_ =  Vec3d(r_ij[0] * r_ij[0], r_ij[1] * r_ij[1], r_ij[0] * r_ij[1]);
          ///here when it is isothermal boundary condition, thsi is 0.0, when boundary condition this is not 0.0 the 0.0
                Real FF_ =   ( 0.0 - r_ij.dot(E_[index_i]) ); 
                H_rate_contact = r_ij.dot(B_[index_i].transpose() * gradW_ijV_j) / pow(r_ij.norm(), 4.0);
		
                //TO DO
                 Vec3d C_ = Vec3d::Zero();
                 C_[0] = (r_ij[0] * r_ij[0]);
                 C_[1] = (r_ij[1] * r_ij[1]);
                 C_[2] = (r_ij[0] * r_ij[1]);
		
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



//------------------------------------------------------------------------------
// the main program
//------------------------------------------------------------------------------
int main(int ac, char *av[])
{
    //----------------------------------------------------------------------
    //	Build up the environment of a SPHSystem with global controls.
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, resolution_ref_large);
#ifdef BOOST_AVAILABLE
    // handle command line arguments
    sph_system.handleCommandlineOptions(ac, av);
#endif //----------------------------------------------------------------------
       //	Creating body, materials and particles.
       //----------------------------------------------------------------------
    SolidBody beam_body(sph_system, makeShared<Beam>("2dMembrane"));
    beam_body.sph_adaptation_->resetKernel<AnisotropicKernel<KernelWendlandC2>>(scaling_vector);
    beam_body.defineParticlesAndMaterial<multi_species_continuum::PorousMediaParticles, multi_species_continuum::PorousMediaSolid>(
        rho_0, Youngs_modulus, poisson, diffusivity_constant_, fulid_initial_density_, water_pressure_constant_);
    beam_body.generateParticles<AnisotropicParticleGenerator>();


     SolidBody boundary_body(sph_system, makeShared<Boundary>("Boundary"));
    boundary_body.sph_adaptation_->resetKernel<AnisotropicKernel<KernelWendlandC2>>(scaling_vector);
    boundary_body.defineParticlesAndMaterial<SolidParticles, Solid>();
    boundary_body.generateParticles<AnisotropicParticleGeneratorBoundary>();
   

    ObserverBody beam_observer(sph_system, "MembraneObserver");
    beam_observer.defineAdaptationRatios(1.15, 2.0);
    beam_observer.generateParticles<ObserverParticleGenerator>(observation_location);
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //  Generally, we first define all the inner relations, then the contact relations.
    //  At last, we define the complex relaxations by combining previous defined
    //  inner and contact relations.
    //----------------------------------------------------------------------
    InnerRelation beam_body_inner(beam_body);
    ContactRelation beam_observer_contact(beam_observer, {&beam_body});
    ComplexRelation beam_complex(beam_body, {&boundary_body});
   
    //-----------------------------------------------------------------------------
    // this section define all numerical methods will be used in this case
    //-----------------------------------------------------------------------------

    // corrected strong configuration
    //InteractionWithUpdate<KernelCorrectionMatrixInner> beam_corrected_configuration(beam_body_inner);
   
    Dynamics1Level<NonisotropicKernelCorrectionMatrixComplex> correct_configuration(beam_complex);
	 // time step size calculation
    ReduceDynamics<solid_dynamics::AcousticTimeStepSize> computing_time_step_size(beam_body);
    ReduceDynamics<multi_species_continuum::GetSaturationTimeStepSize> saturation_time_step_size(beam_body);

    // stress relaxation for the beam
    Dynamics1Level<multi_species_continuum::PorousMediaStressRelaxationFirstHalf> stress_relaxation_first_half(beam_body_inner);
    Dynamics1Level<multi_species_continuum::PorousMediaStressRelaxationSecondHalf> stress_relaxation_second_half(beam_body_inner);
    //Dynamics1Level<multi_species_continuum::SaturationRelaxationInPorousMedia> saturation_relaxation(beam_body_inner);
    Dynamics1Level<NonisotropicSaturationRelaxationInPorousMedia> saturation_relaxation(beam_complex);
   
    // clamping a solid body part. This is softer than a direct constraint
    BodyRegionByParticle beam_base(beam_body, makeShared<MultiPolygonShape>(createBeamConstrainShape()));
    SimpleDynamics<multi_species_continuum::MomentumConstraint> clamp_constrain_beam_base(beam_base);

    BodyRegionByParticle beam_saturation(beam_body, makeShared<MultiPolygonShape>(createSaturationConstrainShape()));
    SimpleDynamics<SaturationInitialCondition> constrain_beam_saturation(beam_saturation);
   
  // boundary_body.addBodyStateForRecording<Real>("NeighbourBoundary");
    beam_body.addBodyStateForRecording<Real>("neighbour");
    beam_body.addBodyStateForRecording<Real>("Laplacian_x");
    beam_body.addBodyStateForRecording<Real>("Laplacian_y");
    beam_body.addBodyStateForRecording<Mat2d>("KernelCorrectionMatrix");
	  
    // need to be done
    ReduceDynamics<TotalMechanicalEnergy> get_kinetic_energy(beam_body);

    /** Damping */
    DampingWithRandomChoice<InteractionSplit<multi_species_continuum::PorousMediaDampingPairwiseInner<Vec2d>>>
        beam_damping(0.5, beam_body_inner, "TotalMomentum", physical_viscosity);
    //-----------------------------------------------------------------------------
    // outputs
    //-----------------------------------------------------------------------------
    IOEnvironment io_environment(sph_system);
    BodyStatesRecordingToVtp write_beam_states(io_environment, sph_system.real_bodies_);
    // note there is a line observation

    ObservedQuantityRecording<Vecd>
        write_beam_tip_position("Position", io_environment, beam_observer_contact);

    //----------------------------------------------------------------------
    //	Setup computing and initial conditions.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
  
    correct_configuration.exec();
    constrain_beam_saturation.exec();
    int ite = 0;
    int total_ite = 0;

    GlobalStaticVariables::physical_time_ = 0.0;

    //----------------------------------------------------------------------
    //	Setup computing time-step controls.
    //----------------------------------------------------------------------

    Real End_Time = 100.0;
    Real setup_saturation_time_ = End_Time * 0.1;

    // time step size for output file
    Real D_Time = End_Time / 100.0;
    Real dt = 0.0; // default acoustic time step sizes

    // statistics for computing time
    TickCount t1 = TickCount::now();
    TimeInterval interval;
    //-----------------------------------------------------------------------------
    // from here the time stepping begins
    //-----------------------------------------------------------------------------
    write_beam_states.writeToFile(0);
    write_beam_tip_position.writeToFile(0);

    // computation loop starts
    while (GlobalStaticVariables::physical_time_ < End_Time)
    {
        Real integration_time = 0.0;
        // integrate time (loop) until the next output time
        while (integration_time < D_Time)
        {
            Real Dt = saturation_time_step_size.exec();
            if (GlobalStaticVariables::physical_time_ < setup_saturation_time_)
            {
                constrain_beam_saturation.exec();
            }
            saturation_relaxation.exec(Dt);

            int stress_ite = 0;
            Real relaxation_time = 0.0;
            Real total_kinetic_energy = 1.0e8;

            while (relaxation_time < Dt)
            {
                if (total_kinetic_energy > (5e-6 * refer_density_energy)) // this is because we change the total mehanical energy calculation
                {
                    stress_relaxation_first_half.exec(dt);
                    clamp_constrain_beam_base.exec();
                    beam_damping.exec(Dt);
                    clamp_constrain_beam_base.exec();
                    stress_relaxation_second_half.exec(dt);

                    total_kinetic_energy = get_kinetic_energy.exec();
                    ite++;
                    stress_ite++;
                    dt = SMIN(computing_time_step_size.exec(), Dt);

                    if (ite % 1000 == 0)
                    {
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

            std::cout << "One Diffusion finishes   "
                      << "total_kinetic_energy =  " << total_kinetic_energy
                      << "     stress_ite = " << stress_ite << std::endl;
        }

        TickCount t2 = TickCount::now();
        write_beam_states.writeToFile(ite);
        write_beam_tip_position.writeToFile(ite);

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

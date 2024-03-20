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


Real dp_0 = (domain_upper_bound_previous[0] - domain_lower_bound_previous[0]) / 90.0; /**< Initial particle spacing. */
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
     {0.8, 0.0, 0.0},  
     {0.0,0.8, 0.0},
     {0.0, 0.0, 0.8},
}; 
Mat3d inverse_decomposed_transform_tensor =  decomposed_transform_tensor.inverse();


Vecd halfsize_boundary(50.0 * length_scale, 45.0 * length_scale, 45.0 * length_scale);
Vecd translation_boundary(-10.0 * length_scale, -35.0 * length_scale, 0.0 * length_scale);

class BoundaryModel : public ComplexShape
{
  public:
    explicit BoundaryModel(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<TransformShape<GeometricShapeBox>>(Transform(translation_boundary), halfsize_boundary, "OuterBoundary");
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
 
 using FiberDirectionDiffusionParticles = DiffusionReactionParticles<ElasticSolidParticles, FiberDirectionDiffusion>;
/** Set diffusion relaxation method. */
using FiberDirectionDiffusionRelaxation =
    DiffusionRelaxationRK2<DiffusionRelaxation<Inner<FiberDirectionDiffusionParticles, CorrectedKernelGradientInner>>>;

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
        Vecd displ = sph_body_.initial_shape_->findNormalDirection(pos_[index_i]);
        Vecd face_norm = displ / (displ.norm() + 1.0e-15);

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
        Vecd displ = sph_body_.initial_shape_->findNormalDirection(pos_[index_i]);
        Vecd face_norm = displ / (displ.norm() + 1.0e-15);
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

/**
 * define observer particle generator.
 */ 

using Mat6d = Eigen::Matrix<Real, 6, 6>;
using Vec6d = Eigen::Matrix<Real, 6, 1>;
    
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
    sph_system.setRunParticleRelaxation(true); // Tag for run particle relaxation for body-fitted distribution
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
        SolidBody boundary(sph_system, makeShared<BoundaryModel>("Boundary"));
        boundary.defineComponentLevelSetShape("OuterBoundary");
        boundary.defineParticlesAndMaterial<SolidParticles, Solid>();
        boundary.generateParticles<ParticleGeneratorLattice>();
 
        SolidBody herat_model(sph_system, makeShared<Heart>("HeartModel"));
        herat_model.defineBodyLevelSetShape()->correctLevelSetSign()->writeLevelSet(sph_system);
        herat_model.defineParticlesAndMaterial<FiberDirectionDiffusionParticles, FiberDirectionDiffusion>();
        herat_model.generateParticles<ParticleGeneratorLattice>();
       
        /** topology */
        InnerRelation herat_model_inner(herat_model);
        InnerRelation boundary_model_inner(boundary);
        ContactRelation heart_boundary_contact(boundary, {&herat_model});
        ComplexRelation water_wall_complex(boundary_model_inner, heart_boundary_contact);
 
        /** Random reset the relax solid particle position. */
        SimpleDynamics<relax_dynamics::RandomizeParticlePosition> random_particles(herat_model);
        SimpleDynamics<relax_dynamics::RandomizeParticlePosition> random_particles_boundary(boundary);
       /** Write the body state to Vtp file. */
        BodyStatesRecordingToVtp write_herat_model_state_to_vtp({herat_model});
        BodyStatesRecordingToVtp write_boundary_model_state_to_vtp({boundary});
        ReloadParticleIO write_real_body_particle_reload_files(sph_system.real_bodies_);
      /** A  Physics relaxation step. */
        relax_dynamics::RelaxationStepLevelSetCorrectionInner relaxation_step_inner(herat_model_inner);
        relax_dynamics::RelaxationStepLevelSetCorrectionComplex relaxation_step_complex(
           ConstructorArgs(boundary_model_inner, "OuterBoundary"), heart_boundary_contact);

      /** Time step for diffusion. */
         GetDiffusionTimeStepSize<FiberDirectionDiffusionParticles> get_time_step_size(herat_model);
        /** Diffusion process for diffusion body. */
        FiberDirectionDiffusionRelaxation diffusion_relaxation(herat_model_inner);
         /** Compute the fiber and sheet after diffusion. */
        SimpleDynamics<ComputeFiberAndSheetDirections> compute_fiber_sheet(herat_model);
 
         //----------------------------------------------------------------------
        //	Physics relaxation starts here.
        //----------------------------------------------------------------------
        random_particles.exec(0.01);
        random_particles_boundary.exec(0.01);
        relaxation_step_inner.SurfaceBounding().exec();
        relaxation_step_complex.SurfaceBounding().exec();
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
            relaxation_step_complex.exec();
            ite++;
            if (ite % 100 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite << "\n";
                write_herat_model_state_to_vtp.writeToFile(ite);
                write_boundary_model_state_to_vtp.writeToFile(ite);
             }
        }
        write_herat_model_state_to_vtp.writeToFile(ite);
        write_boundary_model_state_to_vtp.writeToFile(ite);
        BodySurface surface_part(herat_model);
        /** constraint boundary condition for diffusion. */
        SimpleDynamics<DiffusionBCs> impose_diffusion_bc(surface_part, "Phi");
        impose_diffusion_bc.exec();
         
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
        write_real_body_particle_reload_files.writeToFile(0);

        return 0;
    }
  
       return 0;
}

/**
 * @file 	pkj_lv_electrocontraction.cpp
 * @brief 	This is the case studying the electromechanics on a biventricular heart model in 3D.
 * @author 	Chi Zhang and Xiangyu Hu
 * @version 0.2.1
 * 			Chi Zhang
 * 			Unit :
 *			time t = ms = 12.9 [-]
 * 			length l = mm
 * 			mass m = g
 *			density rho = g * (mm)^(-3)
 *			Pressure pa = g * (mm)^(-1) * (ms)^(-2)
 *			diffusion d = (mm)^(2) * (ms)^(-1)
 *@version 0.3
 *			Here, the coupling with Purkinje network will be conducted.
 */
/**  SPHinXsys Library. */
#include "asph_pkj_lv_electrocontraction.h"
#include "sphinxsys.h"
/** Namespace cite here. */
using namespace SPH;
/**
 * The main program.
 */
int main(int ac, char *av[])
{
    /**
     * Build up context -- a SPHSystem.
     */
    SPHSystem sph_system(system_domain_bounds, dp_0);
    /** Set the starting time. */
    GlobalStaticVariables::physical_time_ = 0.0;
    /** Tag for run particle relaxation for the initial body fitted distribution. */
    sph_system.setRunParticleRelaxation(false);
    /** Tag for reload initially relaxed particles. */
    sph_system.setReloadParticles(true);
// handle command line arguments
#ifdef BOOST_AVAILABLE
    sph_system.handleCommandlineOptions(ac, av);
#endif
    IOEnvironment io_environment(sph_system);
    //----------------------------------------------------------------------
    //	Define the same level geometric shape for all bodies
    //----------------------------------------------------------------------
    Heart triangle_mesh_heart_model("HeartModel");
    SharedPtr<LevelSetShape> level_set_heart_model =
        makeShared<LevelSetShape>(triangle_mesh_heart_model, makeShared<SPHAdaptation>(dp_0));
    level_set_heart_model->correctLevelSetSign()->writeLevelSet(io_environment);
    //----------------------------------------------------------------------
    //	SPH Particle relaxation section
    //----------------------------------------------------------------------
    /** check whether run particle relaxation for body fitted particle distribution. */
    if (sph_system.RunParticleRelaxation())
    {
        SolidBody herat_model(sph_system, level_set_heart_model);
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
        random_particles.exec(0.25);
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
    SolidBody physiology_heart(sph_system, level_set_heart_model, "PhysiologyHeart");
    SharedPtr<AlievPanfilowModel> muscle_reaction_model_ptr = makeShared<AlievPanfilowModel>(k_a, c_m, k, a, b, mu_1, mu_2, epsilon);
    physiology_heart.defineParticlesAndMaterial<
        ElectroPhysiologyParticlesforLaplacian, MonoFieldElectroPhysiology>(
        muscle_reaction_model_ptr, TypeIdentity<AnisotropicLocalDirectionalDiffusion>(), diffusion_coeff, bias_coeff, fiber_direction);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? physiology_heart.generateParticles<ParticleGeneratorReload>(io_environment, "HeartModel")
        : physiology_heart.generateParticles<ParticleGeneratorLattice>();

    /** create a SPH body, material and particles */
    SolidBody mechanics_heart(sph_system, level_set_heart_model, "MechanicalHeart");
    mechanics_heart.defineParticlesAndMaterial<
        ElasticSolidParticles, ActiveMuscle<LocallyOrthotropicMuscle>>(rho0_s, bulk_modulus, fiber_direction, sheet_direction, a0, b0);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? mechanics_heart.generateParticles<ParticleGeneratorReload>(io_environment, "HeartModel")
        : mechanics_heart.generateParticles<ParticleGeneratorLattice>();


    SolidBody boundary_condition(sph_system, makeShared<BoundaryModel>("BoundaryCondition"));
    boundary_condition.defineParticlesAndMaterial<SolidParticles, Solid>();
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? boundary_condition.generateParticles<ParticleGeneratorReload>(io_environment, "Boundary")
        : boundary_condition.generateParticles<ParticleGeneratorLattice>();

 
    //----------------------------------------------------------------------
    //	SPH Observation section
    //----------------------------------------------------------------------
    ObserverBody voltage_observer(sph_system, "VoltageObserver");
    voltage_observer.generateParticles<HeartObserverParticleGenerator>();
    ObserverBody myocardium_observer(sph_system, "MyocardiumObserver");
    myocardium_observer.generateParticles<HeartObserverParticleGenerator>();

    /** topology */
    InnerRelation physiology_heart_inner(physiology_heart);
    InnerRelation mechanics_heart_inner(mechanics_heart);
    ContactRelation physiology_heart_contact(physiology_heart, {&mechanics_heart});
    ContactRelation mechanics_heart_contact(mechanics_heart, {&physiology_heart});
    ContactRelation voltage_observer_contact(voltage_observer, {&physiology_heart});
    ContactRelation myocardium_observer_contact(myocardium_observer, {&mechanics_heart});
 
   ComplexRelation heart_boundary_complex(physiology_heart, {&boundary_condition});

     // Corrected configuration complex
 	 Dynamics1Level<NonisotropicKernelCorrectionMatrixComplex> correct_configuration_excitation(heart_boundary_complex);
    Dynamics1Level<NonisotropicKernelCorrectionMatrixComplexAC> correct_second_configuration(heart_boundary_complex);


    /** Corrected configuration. */
    //InteractionWithUpdate<KernelCorrectionMatrixInner> correct_configuration_excitation(physiology_heart_inner);
    /** Time step size calculation. */
    electro_physiology::GetElectroPhysiologyTimeStepSize get_myocardium_physiology_time_step(physiology_heart);
    /** Diffusion process for diffusion body. */ 

    Dynamics1Level<DiffusionRelaxationComplex>  myocardium_diffusion_relaxation(heart_boundary_complex); 
    //electro_physiology::ElectroPhysiologyDiffusionInnerRK2 myocardium_diffusion_relaxation(physiology_heart_inner);
    
    /** Solvers for ODE system */
    electro_physiology::ElectroPhysiologyReactionRelaxationForward myocardium_reaction_relaxation_forward(physiology_heart);
    electro_physiology::ElectroPhysiologyReactionRelaxationBackward myocardium_reaction_relaxation_backward(physiology_heart);
 
   
    /**IO for observer.*/
    BodyStatesRecordingToVtp write_states(io_environment, sph_system.real_bodies_);
    ObservedQuantityRecording<Real> write_voltage("Voltage", io_environment, voltage_observer_contact);
    ObservedQuantityRecording<Vecd> write_displacement("Position", io_environment, myocardium_observer_contact);
    /**Apply the Iron stimulus.*/
    SimpleDynamics<ApplyStimulusCurrentToMyocardium> apply_stimulus_myocardium(physiology_heart);
 
    /** Active mechanics. */
    InteractionWithUpdate<KernelCorrectionMatrixInner> correct_configuration_contraction(mechanics_heart_inner);
    /** Observer Dynamics */
    InteractionDynamics<CorrectInterpolationKernelWeights>
        correct_kernel_weights_for_interpolation(mechanics_heart_contact);
    /** Interpolate the active contract stress from electrophysiology body. */
    InteractionDynamics<InterpolatingAQuantity<Real>>
        active_stress_interpolation(mechanics_heart_contact, "ActiveContractionStress", "ActiveContractionStress");
    /** Interpolate the particle position in physiology_heart  from mechanics_heart. */
    InteractionDynamics<InterpolatingAQuantity<Vecd>>
        interpolation_particle_position(physiology_heart_contact, "Position", "Position");
    /** Time step size calculation. */
    ReduceDynamics<solid_dynamics::AcousticTimeStepSize> get_mechanics_time_step(mechanics_heart);
    /** active and passive stress relaxation. */
    Dynamics1Level<solid_dynamics::Integration1stHalfPK2> stress_relaxation_first_half(mechanics_heart_inner);
    Dynamics1Level<solid_dynamics::Integration2ndHalf> stress_relaxation_second_half(mechanics_heart_inner);
    /** Constrain region of the inserted body. */
    MuscleBaseShapeParameters muscle_base_parameters;
    BodyRegionByParticle muscle_base(mechanics_heart, makeShared<TriangleMeshShapeBrick>(muscle_base_parameters, "Holder"));
    SimpleDynamics<solid_dynamics::FixBodyPartConstraint> constraint_holder(muscle_base);
    /**
     * Pre-simulation.
     */
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    correct_configuration_excitation.exec();
     correct_second_configuration.exec(); 
    correct_configuration_contraction.exec();
    correct_kernel_weights_for_interpolation.exec();
    /**Output global basic parameters. */
    write_states.writeToFile(0);
    write_voltage.writeToFile(0);
    write_displacement.writeToFile(0);
    write_states.writeToFile(0);
    /**
     * main loop.
     */
    int screen_output_interval = 10;
    int ite = 0;
    int reaction_step = 2;
    Real end_time = 100;
    Real Ouput_T = end_time / 200.0;
    Real Observer_time = 0.01 * Ouput_T;
    Real dt_myocardium = 0.0;
    Real dt_pkj = 0.0;
    Real dt_muscle = 0.0;
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
                              << "	dt_pkj = " << dt_pkj
                              << "	dt_myocardium = " << dt_myocardium
                              << "	dt_muscle = " << dt_muscle << "\n";
                }
                /** Apply stimulus excitation. */
                  if( 0 <= GlobalStaticVariables::physical_time_
                 	&&  GlobalStaticVariables::physical_time_ <= 0.5)
                 {
                  	apply_stimulus_myocardium.exec(dt_myocardium);
                 }

            

                /**Strang splitting method. */
                int ite_forward = 0;
                while (ite_forward < reaction_step)
                {
                    myocardium_reaction_relaxation_forward.exec(0.5 * dt_myocardium / Real(reaction_step));
                    ite_forward++;
                }
                /** 2nd Runge-Kutta scheme for diffusion. */
                myocardium_diffusion_relaxation.exec(dt_myocardium);

                // backward reaction
                int ite_backward = 0;
                while (ite_backward < reaction_step)
                {
                    myocardium_reaction_relaxation_backward.exec(0.5 * dt_myocardium / Real(reaction_step));
                    ite_backward++;
                }

                active_stress_interpolation.exec();
                Real dt_muscle_sum = 0.0;
                while (dt_muscle_sum < dt_myocardium)
                {
                    dt_muscle = get_mechanics_time_step.exec();
                    if (dt_myocardium - dt_muscle_sum < dt_muscle)
                        dt_muscle = dt_myocardium - dt_muscle_sum;
                    stress_relaxation_first_half.exec(dt_muscle);
                    constraint_holder.exec(dt_muscle);
                    stress_relaxation_second_half.exec(dt_muscle);
                    dt_muscle_sum += dt_muscle;
                }

                ite++;
                dt_myocardium = get_myocardium_physiology_time_step.exec();

                relaxation_time += dt_myocardium;
                integration_time += dt_myocardium;
                GlobalStaticVariables::physical_time_ += dt_myocardium;
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
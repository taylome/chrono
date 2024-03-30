// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Radu Serban, Michael Taylor
// =============================================================================
//
// Demonstration program for M113 vehicle with continuous band tracks.
//
// =============================================================================

#include "chrono/ChConfig.h"
#include "chrono/fea/ChMeshExporter.h"
#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/terrain/RigidTerrain.h"
#include "chrono_vehicle/driver/ChPathFollowerDriver.h"
#include "chrono_vehicle/utils/ChVehiclePath.h"
#include "chrono_vehicle/tracked_vehicle/track_shoe/ChTrackShoeBand.h"
#include "chrono_vehicle/tracked_vehicle/track_assembly/ChTrackAssemblyBandANCF.h"

#include "chrono_models/vehicle/m113/M113_SimpleCVTPowertrain.h"
#include "chrono_models/vehicle/m113/M113.h"

#include "chrono/fea/ChElementShellANCF_3833_TR08.h"

#ifdef CHRONO_IRRLICHT
    #include "chrono_vehicle/tracked_vehicle/utils/ChTrackedVehicleVisualSystemIrrlicht.h"
    #define USE_IRRLICHT
#endif

#ifdef CHRONO_MUMPS
    #include "chrono_mumps/ChSolverMumps.h"
#endif

#ifdef CHRONO_PARDISO_MKL
    #include "chrono_pardisomkl/ChSolverPardisoMKL.h"
#endif

#include "chrono_thirdparty/filesystem/path.h"

using namespace chrono;
using namespace chrono::vehicle;
using namespace chrono::vehicle::m113;

using std::cout;
using std::endl;

// =============================================================================
// USER SETTINGS
// =============================================================================

// Band track type (BAND_BUSHING or BAND_ANCF)
TrackShoeType shoe_type = TrackShoeType::BAND_ANCF;

// Number of ANCF elements in one track shoe web mesh
int num_elements_length = 1;
int num_elements_width = 1;

// Enable/disable curvature constraints (ANCF_8 only)
bool constrain_curvature = true;

// Simulation step size and duration
double step_size = 2.5e-5;
//double t_end = 10.0;

// Linear solver (MUMPS, PARDISO_MKL, or SPARSE_LU)
ChSolver::Type solver_type = ChSolver::Type::SPARSE_LU;

// Verbose level
bool verbose_solver = false;
bool verbose_integrator = false;

// Output
bool output = false;
bool dbg_output = false;
bool img_output = false;
bool vtk_output = false;
bool mesh_output = false;
double img_FPS = 100;
double vtk_FPS = 100;
double mesh_FPS = 100;

// Output directories
const std::string out_dir = GetChronoOutputPath() + "M113_BAND";
const std::string img_dir = out_dir + "/IMG";
const std::string vtk_dir = out_dir + "/VTK";
const std::string mesh_dir = out_dir + "/MESH";

// =============================================================================

// Forward declarations
void AddFixedObstacles(ChSystem* system);
void WriteVehicleVTK(int frame, ChTrackedVehicle& vehicle);
void WriteMeshVTK(int frame, std::shared_ptr<fea::ChMesh> meshL, std::shared_ptr<fea::ChMesh> meshR);

// =============================================================================

int main(int argc, char* argv[]) {
    GetLog() << "Copyright (c) 2017 projectchrono.org\nChrono version: " << CHRONO_VERSION << "\n\n";

    // ANCF element type for BAND_ANCF (ANCF_4 or ANCF_8)
    ChTrackShoeBandANCF::ElementType element_type = ChTrackShoeBandANCF::ElementType::ANCF_8;

    double t_end = 12.001;

    if (argc > 1) {
        switch (argv[1][0]) {
            case int('1'):
                element_type = ChTrackShoeBandANCF::ElementType::ANCF_8;
                t_end = 1.001;
                break;
            case int('2'):
                element_type = ChTrackShoeBandANCF::ElementType::ANCF_8_CHRONO6;
                t_end = 1.001;
                break;
            case int('3'):
                element_type = ChTrackShoeBandANCF::ElementType::ANCF_8_TR05;
                t_end = 1.001;
                break;
            case int('4'):
                element_type = ChTrackShoeBandANCF::ElementType::ANCF_8_TR06;
                t_end = 1.001;
                break;
            case int('5'):
                element_type = ChTrackShoeBandANCF::ElementType::ANCF_8_TR07;
                t_end = 1.001;
                break;
            case int('6'):
                element_type = ChTrackShoeBandANCF::ElementType::ANCF_8_TR08;
                t_end = 12.001;
                break;
            case int('7'):
                element_type = ChTrackShoeBandANCF::ElementType::ANCF_8_TR09;
                t_end = 1.001;
                break;
            case int('8'):
                element_type = ChTrackShoeBandANCF::ElementType::ANCF_8_TR13;
                t_end = 1.001;
                break;
            case int('9'):
                element_type = ChTrackShoeBandANCF::ElementType::ANCF_8_M113;
                t_end = 12.001;
                break;
            default:
                std::cout << "Error: Unknown Input.\n";
                return 1;
        }
    }

    //element_type = ChTrackShoeBandANCF::ElementType::ANCF_8_TR08;
    //t_end = 12.001;
    element_type = ChTrackShoeBandANCF::ElementType::ANCF_8_TR01;
    t_end = 1.001;

    // --------------------------
    // Construct the M113 vehicle
    // --------------------------

    M113 m113;

    m113.SetTrackShoeType(shoe_type);
    m113.SetANCFTrackShoeElementType(element_type);
    m113.SetANCFTrackShoeNumElements(num_elements_length, num_elements_width);
    m113.SetANCFTrackShoeCurvatureConstraints(constrain_curvature);
    m113.SetPowertrainType(PowertrainModelType::SIMPLE_MAP);
    m113.SetDrivelineType(DrivelineTypeTV::SIMPLE);
    m113.SetBrakeType(BrakeType::SIMPLE);
    m113.SetSuspensionBushings(false);
    m113.SetGyrationMode(false);

    m113.SetContactMethod(ChContactMethod::SMC);
    m113.SetCollisionSystemType(collision::ChCollisionSystemType::BULLET);
    m113.SetChassisCollisionType(CollisionType::NONE);
    m113.SetChassisFixed(false);

    // ------------------------------------------------
    // Initialize the vehicle at the specified position
    // ------------------------------------------------

    m113.SetInitPosition(ChCoordsys<>(ChVector<>(0, 0, 0.8), QUNIT));
    m113.Initialize();

    auto& vehicle = m113.GetVehicle();
    auto sys = vehicle.GetSystem();

    int numEls = 0;
    std::shared_ptr<fea::ChMesh> meshL;
    std::shared_ptr<fea::ChMesh> meshR;
    if (shoe_type == TrackShoeType::BAND_ANCF) {
        meshL =
            std::static_pointer_cast<ChTrackAssemblyBandANCF>(vehicle.GetTrackAssembly(VehicleSide::LEFT))->GetMesh();
        meshR =
            std::static_pointer_cast<ChTrackAssemblyBandANCF>(vehicle.GetTrackAssembly(VehicleSide::RIGHT))->GetMesh();

        cout << "[FEA mesh left]  n_nodes = " << meshL->GetNnodes() << " n_elements = " << meshL->GetNelements()
             << endl;
        cout << "[FEA mesh right] n_nodes = " << meshR->GetNnodes() << " n_elements = " << meshR->GetNelements()
             << endl;
        numEls = meshL->GetNelements() + meshR->GetNelements();
    }

#ifdef USE_IRRLICHT
    // Set visualization type for vehicle components.
    //vehicle.SetChassisVisualizationType(VisualizationType::NONE);
    vehicle.SetChassisVisualizationType(VisualizationType::MESH);
    vehicle.SetSprocketVisualizationType(VisualizationType::MESH);
    vehicle.SetIdlerVisualizationType(VisualizationType::PRIMITIVES);
    vehicle.SetIdlerWheelVisualizationType(VisualizationType::MESH);
    vehicle.SetSuspensionVisualizationType(VisualizationType::PRIMITIVES);
    vehicle.SetRoadWheelVisualizationType(VisualizationType::MESH);
    vehicle.SetTrackShoeVisualizationType(VisualizationType::MESH);
#else
    // Set visualization type for vehicle components.
    vehicle.SetChassisVisualizationType(VisualizationType::NONE);
    vehicle.SetSprocketVisualizationType(VisualizationType::NONE);
    vehicle.SetIdlerVisualizationType(VisualizationType::NONE);
    vehicle.SetIdlerWheelVisualizationType(VisualizationType::NONE);
    vehicle.SetSuspensionVisualizationType(VisualizationType::NONE);
    vehicle.SetRoadWheelVisualizationType(VisualizationType::NONE);
    vehicle.SetTrackShoeVisualizationType(VisualizationType::NONE);
#endif


    // Export sprocket and shoe tread visualization meshes
    auto trimesh =
        vehicle.GetTrackAssembly(VehicleSide::LEFT)->GetSprocket()->CreateVisualizationMesh(0.15, 0.03, 0.02);
    geometry::ChTriangleMeshConnected::WriteWavefront(out_dir + "/M113_Sprocket.obj", {*trimesh});
    std::static_pointer_cast<ChTrackShoeBand>(vehicle.GetTrackShoe(LEFT, 0))->WriteTreadVisualizationMesh(out_dir);

    // Disable gravity in this simulation
    ////sys->Set_G_acc(ChVector<>(0, 0, 0));

    // --------------------------------------------------
    // Control internal collisions and contact monitoring
    // --------------------------------------------------

    // Disable contact for the FEA track meshes
    std::static_pointer_cast<ChTrackAssemblyBandANCF>(vehicle.GetTrackAssembly(LEFT))
        ->SetContactSurfaceType(ChTrackAssemblyBandANCF::ContactSurfaceType::NONE);
    std::static_pointer_cast<ChTrackAssemblyBandANCF>(vehicle.GetTrackAssembly(RIGHT))
        ->SetContactSurfaceType(ChTrackAssemblyBandANCF::ContactSurfaceType::NONE);

    // Enable contact on all tracked vehicle parts, except the left sprocket
    ////vehicle.SetCollide(TrackedCollisionFlag::ALL & (~TrackedCollisionFlag::SPROCKET_LEFT));

    // Disable contact for all tracked vehicle parts
    ////vehicle.SetCollide(TrackedCollisionFlag::NONE);

    // Disable all contacts for vehicle chassis (if chassis collision was defined)
    ////vehicle.SetChassisCollide(false);

    // Disable only contact between chassis and track shoes (if chassis collision was defined)
    ////vehicle.SetChassisVehicleCollide(false);

    // Monitor internal contacts for the chassis, left sprocket, left idler, and first shoe on the left track.
    ////vehicle.MonitorContacts(TrackedCollisionFlag::CHASSIS | TrackedCollisionFlag::SPROCKET_LEFT |
    ////                        TrackedCollisionFlag::SHOES_LEFT | TrackedCollisionFlag::IDLER_LEFT);

    // Monitor only contacts involving the chassis.
    ////vehicle.MonitorContacts(TrackedCollisionFlag::CHASSIS);

    // Render contact normals and/or contact forces.
    ////vehicle.SetRenderContactNormals(true);
    ////vehicle.SetRenderContactForces(true, 1e-4);

    // Collect contact information.
    // If enabled, number of contacts and local contact point locations are collected for all
    // monitored parts.  Data can be written to a file by invoking ChTrackedVehicle::WriteContacts().
    ////vehicle.SetContactCollection(true);

    // ------------------
    // Create the terrain
    // ------------------

    RigidTerrain terrain(sys);
    ChContactMaterialData minfo;
    minfo.mu = 0.9f;
    minfo.cr = 0.2f;
    minfo.Y = 2e7f;
    auto patch_mat = minfo.CreateMaterial(ChContactMethod::SMC);
    auto patch = terrain.AddPatch(patch_mat, CSYSNORM, 100.0, 100.0);
    patch->SetColor(ChColor(0.5f, 0.8f, 0.5f));
    patch->SetTexture(vehicle::GetDataFile("terrain/textures/tile4.jpg"), 200, 200);
    terrain.Initialize();

    // -------------------------------------------
    // Create a straight-line path follower driver
    // -------------------------------------------

    auto path = chrono::vehicle::StraightLinePath(ChVector<>(0.0, 0, 0.5), ChVector<>(100.0, 0, 0.5), 50);
    ChPathFollowerDriver driver(vehicle, path, "my_path", 5.0);
    driver.GetSteeringController().SetLookAheadDistance(5.0);
    driver.GetSteeringController().SetGains(0.5, 0, 0);
    driver.GetSpeedController().SetGains(0.6, 0.3, 0);
    driver.Initialize();

    // -------------------
    // Add fixed obstacles
    // -------------------

    AddFixedObstacles(sys);

#ifdef USE_IRRLICHT
    // ---------------------------------------
    // Create the vehicle Irrlicht application
    // ---------------------------------------

    auto vis = chrono_types::make_shared<ChTrackedVehicleVisualSystemIrrlicht>();
    vis->SetWindowTitle("M113 Band-track Vehicle Demo");
    //vis->SetChaseCamera(ChVector<>(0, 0, 0), 6.0, 0.5);
    vis->SetChaseCamera(ChVector<>(-2, 0, 0), 4, 1.0);
    ////vis->SetChaseCameraPosition(vehicle.GetPos() + ChVector<>(0, 2, 0));
    vis->SetChaseCameraMultipliers(1e-4, 10);
    vis->Initialize();
    vis->AddLightDirectional();
    vis->AddSkyBox();
    vis->AddLogo();
    vis->AttachVehicle(&vehicle);
#endif

    // -----------------
    // Initialize output
    // -----------------

    if (!filesystem::create_directory(filesystem::path(out_dir))) {
        cout << "Error creating directory " << out_dir << endl;
        return 1;
    }

    if (img_output) {
        if (!filesystem::create_directory(filesystem::path(img_dir))) {
            cout << "Error creating directory " << img_dir << endl;
            return 1;
        }
    }
    if (vtk_output) {
        if (!filesystem::create_directory(filesystem::path(vtk_dir))) {
            cout << "Error creating directory " << vtk_dir << endl;
            return 1;
        }
    }
    if (mesh_output) {
        if (!filesystem::create_directory(filesystem::path(mesh_dir))) {
            cout << "Error creating directory " << mesh_dir << endl;
            return 1;
        }
    }

    // Setup chassis position output with column headers
    utils::CSV_writer csv("\t");
    csv.stream().setf(std::ios::scientific | std::ios::showpos);
    csv.stream().precision(6);
    csv << "Time (s)"
        << "Chassis X Pos (m)"
        << "Chassis Y Pos (m)"
        << "Chassis Z Pos (m)" << endl;

    // Set up vehicle output
    ////vehicle.SetChassisOutput(true);
    ////vehicle.SetTrackAssemblyOutput(VehicleSide::LEFT, true);
    vehicle.SetOutput(ChVehicleOutput::ASCII, out_dir, "vehicle_output", 0.1);

    // Generate JSON information with available output channels
    ////vehicle.ExportComponentList(out_dir + "/component_list.json");

    // ------------------------------
    // Solver and integrator settings
    // ------------------------------

    // Linear solver
#if !defined(CHRONO_PARDISO_MKL) && !defined(CHRONO_MUMPS)
    solver_type = ChSolver::Type::SPARSE_LU;
#endif
#ifndef CHRONO_PARDISO_MKL
    if (solver_type == ChSolver::Type::PARDISO_MKL)
        solver_type = ChSolver::Type::MUMPS;
#endif
#ifndef CHRONO_MUMPS
    if (solver_type == ChSolver::Type::MUMPS)
        solver_type = ChSolver::Type::PARDISO_MKL;
#endif

    switch (solver_type) {
        case ChSolver::Type::MUMPS: {
#ifdef CHRONO_MUMPS
            auto solver = chrono_types::make_shared<ChSolverMumps>();
            solver->LockSparsityPattern(true);
            solver->EnableNullPivotDetection(true);
            solver->GetMumpsEngine().SetICNTL(14, 50);
            solver->SetVerbose(verbose_solver);
            sys->SetSolver(solver);
#endif
            break;
        }
        case ChSolver::Type::PARDISO_MKL: {
#ifdef CHRONO_PARDISO_MKL
            auto solver = chrono_types::make_shared<ChSolverPardisoMKL>();
            solver->LockSparsityPattern(true);
            solver->SetVerbose(verbose_solver);
            sys->SetSolver(solver);
#endif
            break;
        }
        default: {
            auto solver = chrono_types::make_shared<ChSolverSparseLU>();
            solver->LockSparsityPattern(true);
            solver->SetVerbose(verbose_solver);
            sys->SetSolver(solver);
            break;
        }
    }

    // Integrator
    sys->SetTimestepperType(ChTimestepper::Type::HHT);
    auto integrator = std::static_pointer_cast<ChTimestepperHHT>(sys->GetTimestepper());
    integrator->SetAlpha(-0.2);
    integrator->SetMaxiters(20);
    integrator->SetAbsTolerances(1e-4, 1e2);
    integrator->SetMode(ChTimestepperHHT::POSITION);
    integrator->SetStepControl(false);
    integrator->SetModifiedNewton(true);
    integrator->SetScaling(false);
    integrator->SetVerbose(verbose_integrator);

    // OpenMP threads
    sys->SetNumThreads(4, 4, 4);

    // ---------------
    // Simulation loop
    // ---------------

    // Inter-module communication data
    BodyStates shoe_states_left(vehicle.GetNumTrackShoes(LEFT));
    BodyStates shoe_states_right(vehicle.GetNumTrackShoes(RIGHT));
    TerrainForces shoe_forces_left(vehicle.GetNumTrackShoes(LEFT));
    TerrainForces shoe_forces_right(vehicle.GetNumTrackShoes(RIGHT));

    // Number of steps
    int sim_steps = (int)std::ceil(t_end / (step_size * 4));          // total number of simulation steps
    int img_steps = (int)std::ceil(1 / (img_FPS * (step_size * 4)));  // interval between IMG output frames
    int vtk_steps = (int)std::ceil(1 / (vtk_FPS * (step_size * 4)));  // interval between VIS postprocess output frames
    int mesh_steps = (int)std::ceil(1 / (mesh_FPS * (step_size * 4)));  // interval between mesh postprocess output frames

    // Total execution time (for integration)
    double total_timing = 0;

    // Initialize simulation frame counter
    int step_number = 0;
    int img_frame = 0;
    int vtk_frame = 0;
    int mesh_frame = 0;
    int adjusted_step_number = 0;

    //Setup variables for extra timing output
    ChMatrixNM<double, 1, 19> step_timing_stats;
    step_timing_stats.setZero();
    ChMatrixNM<double, 1, 19> step_timing_stats_total;
    step_timing_stats_total.setZero();
    
    //Get pointers to help with timing output
    auto LS = std::dynamic_pointer_cast<ChDirectSolverLS>(m113.GetSystem()->GetSolver());
    auto MeshList = m113.GetSystem()->Get_meshlist();

    //Reset Timers
    for (auto& Mesh : MeshList) {
        Mesh->ResetTimers();
        Mesh->ResetCounters();
    }
    if (LS != NULL) {  // Direct Solver
        LS->ResetTimers();
    }
    m113.GetSystem()->ResetTimers();

    int timing_steps = 40;
    int render_frame = 0;

    DriverInputs driver_inputs = driver.GetInputs();
    double step_timing = 0.0;

#ifdef USE_IRRLICHT
    vis->SetChaseCameraAngle(CH_C_DEG_TO_RAD * 60);
#endif

    while (m113.GetSystem()->GetChTime() < t_end) {
        double time = vehicle.GetChTime();
        const ChVector<>& c_pos = vehicle.GetPos();

        // File output
        if (output) {
            csv << time << c_pos.x() << c_pos.y() << c_pos.z() << endl;
        }

        // Debugging (console) output
        if (dbg_output) {
            cout << "Time: " << time << endl;
            const ChFrameMoving<>& c_ref = vehicle.GetChassisBody()->GetFrame_REF_to_abs();
            cout << "      chassis:    " << c_pos.x() << "  " << c_pos.y() << "  " << c_pos.z() << endl;
            {
                const ChVector<>& i_pos_abs = vehicle.GetTrackAssembly(LEFT)->GetIdler()->GetWheelBody()->GetPos();
                const ChVector<>& s_pos_abs = vehicle.GetTrackAssembly(LEFT)->GetSprocket()->GetGearBody()->GetPos();
                ChVector<> i_pos_rel = c_ref.TransformPointParentToLocal(i_pos_abs);
                ChVector<> s_pos_rel = c_ref.TransformPointParentToLocal(s_pos_abs);
                cout << "      L idler:    " << i_pos_rel.x() << "  " << i_pos_rel.y() << "  " << i_pos_rel.z() << endl;
                cout << "      L sprocket: " << s_pos_rel.x() << "  " << s_pos_rel.y() << "  " << s_pos_rel.z() << endl;
            }
            {
                const ChVector<>& i_pos_abs = vehicle.GetTrackAssembly(RIGHT)->GetIdler()->GetWheelBody()->GetPos();
                const ChVector<>& s_pos_abs = vehicle.GetTrackAssembly(RIGHT)->GetSprocket()->GetGearBody()->GetPos();
                ChVector<> i_pos_rel = c_ref.TransformPointParentToLocal(i_pos_abs);
                ChVector<> s_pos_rel = c_ref.TransformPointParentToLocal(s_pos_abs);
                cout << "      R idler:    " << i_pos_rel.x() << "  " << i_pos_rel.y() << "  " << i_pos_rel.z() << endl;
                cout << "      R sprocket: " << s_pos_rel.x() << "  " << s_pos_rel.y() << "  " << s_pos_rel.z() << endl;
            }
            cout << "      L suspensions (arm angles):" << endl;
            for (size_t i = 0; i < vehicle.GetTrackAssembly(LEFT)->GetNumTrackSuspensions(); i++) {
                cout << " " << vehicle.GetTrackAssembly(VehicleSide::LEFT)->GetTrackSuspension(i)->GetCarrierAngle();
            }
            cout << endl;
            cout << "      R suspensions (arm angles):" << endl;
            for (size_t i = 0; i < vehicle.GetTrackAssembly(RIGHT)->GetNumTrackSuspensions(); i++) {
                cout << " " << vehicle.GetTrackAssembly(VehicleSide::RIGHT)->GetTrackSuspension(i)->GetCarrierAngle();
            }
            cout << endl;
        }

        if (step_number % timing_steps == 0) {
            render_frame++;

            step_timing_stats(1, 17) = (step_timing_stats(1, 13) * 1e6) / (step_timing_stats(1, 15) * double(numEls));
            step_timing_stats(1, 18) = (step_timing_stats(1, 14) * 1e6) / (step_timing_stats(1, 16) * double(numEls));

            step_timing_stats_total += step_timing_stats;
            step_timing_stats_total(1, 17) = (step_timing_stats_total(1, 13) * 1e6) / (step_timing_stats_total(1, 15) * double(numEls));
            step_timing_stats_total(1, 18) = (step_timing_stats_total(1, 14) * 1e6) / (step_timing_stats_total(1, 16) * double(numEls));

            std::cout << "Render Frame: " << render_frame << " Step: " << step_number << "  Time: " << m113.GetSystem()->GetChTime() << "s,  Time for Step: " << step_timing << "s,  Avg Time for Step: " << total_timing / double(step_number) << "s,  Total Sim Time: " << total_timing << "s" << std::endl;

            const ChVector<>& c_pos = vehicle.GetPos();
            std::cout << "   Chassis Pos:   " << c_pos.x() << "  " << c_pos.y() << "  " << c_pos.z() << "  " << std::endl;
            auto chassis = vehicle.GetChassisBody();
            std::cout << "   ChassisState: " << chassis->GetPos() << "  "
                << chassis->GetRot() << "  "
                << chassis->GetPos_dt() << "  "
                << chassis->GetWvel_loc() << "  "
                << chassis->GetPos_dtdt() << "  "
                << chassis->GetWacc_loc() << endl;

            //const ChVector<>& drv_pos = vehicle.GetDriverPos();
            //std::cout << "   Driver Pos:   " << drv_pos.x() << "  " << drv_pos.y() << "  " << drv_pos.z() << "  " << std::endl;
            const ChVector<>& drv_pos2 = vehicle.GetPointLocation(ChVector<>(0.0, 0.5, 1.2));
            std::cout << "   Driver Pos:   " << drv_pos2.x() << "  " << drv_pos2.y() << "  " << drv_pos2.z() << "  " << std::endl;
            const ChVector<>& drv_vel = vehicle.GetPointVelocity(ChVector<>(0.0, 0.5, 1.2));
            std::cout << "   Driver Vel:   " << drv_vel.x() << "  " << drv_vel.y() << "  " << drv_vel.z() << "  " << std::endl;
            const ChVector<>& drv_acc_local = vehicle.GetPointAcceleration(ChVector<>(0.0, 0.5, 1.2));
            std::cout << "   Driver Accel Local:   " << drv_acc_local.x() << "  " << drv_acc_local.y() << "  " << drv_acc_local.z() << "  " << std::endl;
            const ChVector<>& drv_acc = vehicle.GetChassisBody()->PointAccelerationLocalToParent(ChVector<>(0.0, 0.5, 1.2));
            std::cout << "   Driver Accel Global:   " << drv_acc.x() << "  " << drv_acc.y() << "  " << drv_acc.z() << "  " << std::endl;

            std::cout << "   Num. contacts: " << m113.GetSystem()->GetNcontacts() << endl;

            std::cout << "   Driver Inputs: " << driver_inputs.m_throttle << "  " << driver_inputs.m_braking << "  " << driver_inputs.m_steering << endl;

            std::cout << "   Step_Total "
                << "Step_Advance "
                << "Step_Update "
                << "LS_Jacobian "
                << "LS_Setup "
                << "LS_Setup_Asm "
                << "LS_Setup_Solver "
                << "LS_Solve "
                << "LS_Solve_Asm "
                << "LS_Solve_Solver "
                << "CD_Total "
                << "CD_Broad "
                << "CD_Narrow "
                << "FEA_InternalFrc "
                << "FEA_Jacobian "
                << "FEA_InternalFrc_Calls "
                << "FEA_Jacobian_Calls "
                << "FEA_InternalFrc_AvgFunctionCall_us "
                << "FEA_Jacobian_AvgFunctionCall_us" << std::endl;

            std::cout << "   " << step_timing_stats << std::endl;
            std::cout << "   " << step_timing_stats_total << std::endl << std::endl;

            //Reset Timers
            for (auto& Mesh : MeshList) {
                Mesh->ResetTimers();
                Mesh->ResetCounters();
            }
            if (LS != NULL) {  // Direct Solver
                LS->ResetTimers();
            }
            m113.GetSystem()->ResetTimers();
            step_timing_stats.setZero();
        }

#ifdef USE_IRRLICHT
        if (!vis->Run())
            break;

        // Render scene
        vis->BeginScene();
        vis->Render();
#endif

        if (img_output && adjusted_step_number % img_steps == 0) {
#ifdef USE_IRRLICHT
            std::string filename = img_dir + "/img." + std::to_string(img_frame) + ".jpg";
            vis->WriteImageToFile(filename);
            img_frame++;
#endif
        }

        if (vtk_output && adjusted_step_number % vtk_steps == 0) {
            WriteVehicleVTK(vtk_frame, vehicle);
            if (shoe_type == TrackShoeType::BAND_ANCF)
                WriteMeshVTK(vtk_frame, meshL, meshR);
            vtk_frame++;
        }

        if (mesh_output && adjusted_step_number % mesh_steps == 0) {
            utils::CSV_writer mesh_csv("\t");
            mesh_csv.stream().setf(std::ios::scientific | std::ios::showpos);
            mesh_csv.stream().precision(6);
            mesh_csv << time << std::endl;

            int El_offset = 0;
            for (auto& Mesh : MeshList) {
                for (size_t el_idx = 0; el_idx < Mesh->GetNelements(); el_idx++){
                    auto El = std::dynamic_pointer_cast<chrono::fea::ChElementShellANCF_3833_TR08>(Mesh->GetElement(el_idx));

                    ChVector<double>pnt (0,0,0);

                    for (size_t xi_idx = 0; xi_idx < 5; xi_idx++) {
                        for (size_t eta_idx = 0; eta_idx < 5; eta_idx++) {
                            double xi = -1 + .5*xi_idx;
                            double eta = -1 + .5*eta_idx;
                            El->EvaluateSectionPoint(xi, eta, pnt);

                            mesh_csv << (El_offset+el_idx) << xi << eta << pnt << El->GetVonMissesStress(1, xi, eta, 0) << endl;
                        }
                    }
                }
                El_offset += Mesh->GetNelements();
            }

            mesh_csv.write_to_file(mesh_dir + "/mesh." + std::to_string(mesh_frame) + ".txt");
            mesh_frame++;
        }

        // Collect data from modules
        DriverInputs driver_inputs = driver.GetInputs();
        vehicle.GetTrackShoeStates(LEFT, shoe_states_left);
        vehicle.GetTrackShoeStates(RIGHT, shoe_states_right);

        // Update modules (process data from other modules)
        driver.Synchronize(time);
        terrain.Synchronize(time);
        m113.Synchronize(time, driver_inputs, shoe_forces_left, shoe_forces_right);
#ifdef USE_IRRLICHT
        vis->Synchronize("", driver_inputs);
#endif

        // Advance simulation for one timestep for all modules
        if (step_number == 120) {
            step_size = 5e-5;
            timing_steps /= 2;
            adjusted_step_number = 30;
        }
        if (step_number == 220) {
            step_size = 1e-4;
            timing_steps /= 2;
            adjusted_step_number = 80;
        }

        driver.Advance(step_size);
        terrain.Advance(step_size);
        m113.Advance(step_size);
#ifdef USE_IRRLICHT
        vis->Advance(step_size);
#endif

        // Report if the chassis experienced a collision
        if (vehicle.IsPartInContact(TrackedCollisionFlag::CHASSIS)) {
            cout << time << "  chassis contact" << endl;
        }

        // Increment frame number
        step_number++;
        adjusted_step_number++;

        //Accumulate Timers
        step_timing = sys->GetTimerStep();
        total_timing += step_timing;

        step_timing_stats(1, 0) += m113.GetSystem()->GetTimerStep();
        step_timing_stats(1, 1) += m113.GetSystem()->GetTimerAdvance();
        step_timing_stats(1, 2) += m113.GetSystem()->GetTimerUpdate();

        step_timing_stats(1, 3) += m113.GetSystem()->GetTimerJacobian();
        step_timing_stats(1, 4) += m113.GetSystem()->GetTimerLSsetup();
        step_timing_stats(1, 7) += m113.GetSystem()->GetTimerLSsolve();
        step_timing_stats(1, 10) += m113.GetSystem()->GetTimerCollision();
        step_timing_stats(1, 11) += m113.GetSystem()->GetTimerCollisionBroad();
        step_timing_stats(1, 12) += m113.GetSystem()->GetTimerCollisionNarrow();

        if (LS != NULL) {  // Direct Solver
            step_timing_stats(1, 5) += LS->GetTimeSetup_Assembly();
            step_timing_stats(1, 6) += LS->GetTimeSetup_SolverCall();
            step_timing_stats(1, 8) += LS->GetTimeSolve_Assembly();
            step_timing_stats(1, 9) += LS->GetTimeSolve_SolverCall();
        }

        // Accumulate the internal force and Jacobian timers across all the FEA mesh containers
        for (auto& Mesh : MeshList) {
            step_timing_stats(1, 13) += Mesh->GetTimeInternalForces();
            step_timing_stats(1, 14) += Mesh->GetTimeJacobianLoad();
        }
        // Accumulate the number of internal force calls (Use one mesh so that this is not accounted for extra times)
        step_timing_stats(1, 15) += MeshList.front()->GetNumCallsInternalForces();
        step_timing_stats(1, 16) += MeshList.front()->GetNumCallsJacobianLoad();

        //Reset Timers
        for (auto& Mesh : MeshList) {
            Mesh->ResetTimers();
            Mesh->ResetCounters();
        }
        if (LS != NULL) {  // Direct Solver
            LS->ResetTimers();
        }
        m113.GetSystem()->ResetTimers();

        //cout << "Step: " << step_number;
        //cout << "   Time: " << time;
        //cout << "   Number of Iterations: " << integrator->GetNumIterations();
        //cout << "   Step Time: " << step_timing;
        //cout << "   Total Time: " << total_timing;
        //cout << endl;

#ifdef USE_IRRLICHT
        vis->EndScene();
#endif
    }

    if (output) {
        csv.write_to_file(out_dir + "/chassis_position.txt");
    }

    vehicle.WriteContacts(out_dir + "/contacts.txt");

    return 0;
}

// =============================================================================

void AddFixedObstacles(ChSystem* system) {
    double radius = 2.2;
    double length = 6;

    auto obstacle = std::shared_ptr<ChBody>(system->NewBody());
    obstacle->SetPos(ChVector<>(10, 0, -1.8));
    obstacle->SetBodyFixed(true);
    obstacle->SetCollide(true);

#ifdef USE_IRRLICHT
    // Visualization
    auto shape = chrono_types::make_shared<ChCylinderShape>();
    shape->GetCylinderGeometry().p1 = ChVector<>(0, -length * 0.5, 0);
    shape->GetCylinderGeometry().p2 = ChVector<>(0, length * 0.5, 0);
    shape->GetCylinderGeometry().rad = radius;
    shape->SetTexture(vehicle::GetDataFile("terrain/textures/tile4.jpg"));
    obstacle->AddVisualShape(shape);
#endif

    // Contact
    auto obst_mat = chrono_types::make_shared<ChMaterialSurfaceSMC>();
    obst_mat->SetFriction(0.9f);
    obst_mat->SetRestitution(0.01f);
    obst_mat->SetYoungModulus(2e7f);
    obst_mat->SetPoissonRatio(0.3f);

    obstacle->GetCollisionModel()->ClearModel();
    obstacle->GetCollisionModel()->AddCylinder(obst_mat, radius, radius, length * 0.5);
    obstacle->GetCollisionModel()->BuildModel();

    system->AddBody(obstacle);
}

// =============================================================================

void WriteMeshVTK(int frame, std::shared_ptr<fea::ChMesh> meshL, std::shared_ptr<fea::ChMesh> meshR) {
    static bool generate_connectivity = true;
    if (generate_connectivity) {
        fea::ChMeshExporter::WriteMesh(meshL, vtk_dir + "/meshL_connectivity.out");
        fea::ChMeshExporter::WriteMesh(meshR, vtk_dir + "/meshR_connectivity.out");
        generate_connectivity = false;
    }
    std::string filenameL = vtk_dir + "/meshL." + std::to_string(frame) + ".vtk";
    std::string filenameR = vtk_dir + "/meshR." + std::to_string(frame) + ".vtk";
    fea::ChMeshExporter::WriteFrame(meshL, vtk_dir + "/meshL_connectivity.out", filenameL);
    fea::ChMeshExporter::WriteFrame(meshR, vtk_dir + "/meshR_connectivity.out", filenameR);
}

void WriteVehicleVTK(int frame, ChTrackedVehicle& vehicle) {
    {
        utils::CSV_writer csv(",");
        auto num_shoes_L = vehicle.GetTrackAssembly(VehicleSide::LEFT)->GetNumTrackShoes();
        auto num_shoes_R = vehicle.GetTrackAssembly(VehicleSide::RIGHT)->GetNumTrackShoes();
        for (size_t i = 0; i < num_shoes_L; i++) {
            const auto& shoe = vehicle.GetTrackAssembly(VehicleSide::LEFT)->GetTrackShoe(i)->GetShoeBody();
            csv << shoe->GetPos() << shoe->GetRot() << shoe->GetPos_dt() << shoe->GetWvel_loc() << endl;
        }
        for (size_t i = 0; i < num_shoes_R; i++) {
            const auto& shoe = vehicle.GetTrackAssembly(VehicleSide::RIGHT)->GetTrackShoe(i)->GetShoeBody();
            csv << shoe->GetPos() << shoe->GetRot() << shoe->GetPos_dt() << shoe->GetWvel_loc() << endl;
        }
        csv.write_to_file(vtk_dir + "/shoes." + std::to_string(frame) + ".vtk", "x,y,z,e0,e1,e2,e3,vx,vy,vz,ox,oy,oz");
    }

    {
        utils::CSV_writer csv(",");
        auto num_wheels_L = vehicle.GetTrackAssembly(VehicleSide::LEFT)->GetNumTrackSuspensions();
        auto num_wheels_R = vehicle.GetTrackAssembly(VehicleSide::RIGHT)->GetNumTrackSuspensions();
        for (size_t i = 0; i < num_wheels_L; i++) {
            const auto& wheel = vehicle.GetTrackAssembly(VehicleSide::LEFT)->GetTrackSuspension(i)->GetWheelBody();
            csv << wheel->GetPos() << wheel->GetRot() << wheel->GetPos_dt() << wheel->GetWvel_loc() << endl;
        }
        for (size_t i = 0; i < num_wheels_R; i++) {
            const auto& wheel = vehicle.GetTrackAssembly(VehicleSide::RIGHT)->GetTrackSuspension(i)->GetWheelBody();
            csv << wheel->GetPos() << wheel->GetRot() << wheel->GetPos_dt() << wheel->GetWvel_loc() << endl;
        }
        csv.write_to_file(vtk_dir + "/wheels." + std::to_string(frame) + ".vtk", "x,y,z,e0,e1,e2,e3,vx,vy,vz,ox,oy,oz");
    }

    {
        utils::CSV_writer csv(",");
        const auto& idlerL = vehicle.GetTrackAssembly(VehicleSide::LEFT)->GetIdler()->GetIdlerWheel()->GetBody();
        const auto& idlerR = vehicle.GetTrackAssembly(VehicleSide::RIGHT)->GetIdler()->GetIdlerWheel()->GetBody();
        csv << idlerL->GetPos() << idlerL->GetRot() << idlerL->GetPos_dt() << idlerL->GetWvel_loc() << endl;
        csv << idlerR->GetPos() << idlerR->GetRot() << idlerR->GetPos_dt() << idlerR->GetWvel_loc() << endl;
        csv.write_to_file(vtk_dir + "/idlers." + std::to_string(frame) + ".vtk", "x,y,z,e0,e1,e2,e3,vx,vy,vz,ox,oy,oz");
    }

    {
        utils::CSV_writer csv(",");
        const auto& gearL = vehicle.GetTrackAssembly(VehicleSide::LEFT)->GetSprocket()->GetGearBody();
        const auto& gearR = vehicle.GetTrackAssembly(VehicleSide::RIGHT)->GetSprocket()->GetGearBody();
        csv << gearL->GetPos() << gearL->GetRot() << gearL->GetPos_dt() << gearL->GetWvel_loc() << endl;
        csv << gearR->GetPos() << gearR->GetRot() << gearR->GetPos_dt() << gearR->GetWvel_loc() << endl;
        csv.write_to_file(vtk_dir + "/sprockets." + std::to_string(frame) + ".vtk",
                          "x,y,z,e0,e1,e2,e3,vx,vy,vz,ox,oy,oz");
    }

    {
        utils::CSV_writer csv(",");
        auto chassis = vehicle.GetChassisBody();
        csv << chassis->GetPos() << chassis->GetRot() << chassis->GetPos_dt() << chassis->GetWvel_loc() << endl;
        csv.write_to_file(vtk_dir + "/chassis." + std::to_string(frame) + ".vtk",
                          "x,y,z,e0,e1,e2,e3,vx,vy,vz,ox,oy,oz");
    }
}

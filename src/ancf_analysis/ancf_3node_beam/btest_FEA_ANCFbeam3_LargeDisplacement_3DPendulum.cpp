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
// Authors: Mike Taylor and Radu Serban
// =============================================================================
//
// Large Displacement, Large Deformation, Linear Isotropic Benchmark test for
// ANCF beam elements - Spinning 3D Pendulum with Initial Conditions and a 
// Square cross section
//
// With Modifications from:
// J. Gerstmayr and A.A. Shabana. Analysis of thin beams and cables using the 
// absolute nodal co-ordinate formulation.Nonlinear Dynamics, 45(1) : 109{130,
// 2006.
//
// =============================================================================


#include <string>

#include "chrono/ChConfig.h"

#include "chrono/physics/ChSystemSMC.h"
#include "chrono/solver/ChIterativeSolverLS.h"
#include "chrono/solver/ChDirectSolverLS.h"

//#include "chrono_postprocess/ChGnuPlot.h"
//#include "chrono_thirdparty/filesystem/path.h"

#include "chrono/fea/ChElementBeamANCF.h"
#include "chrono/fea/ChElementBeamANCF_MT01.h"
#include "chrono/fea/ChElementBeamANCF_MT02.h"
#include "chrono/fea/ChElementBeamANCF_MT03.h"
#include "chrono/fea/ChElementBeamANCF_MT04.h"
#include "chrono/fea/ChElementBeamANCF_MT05.h"
#include "chrono/fea/ChElementBeamANCF_MT06.h"
#include "chrono/fea/ChElementBeamANCF_MT07.h"
#include "chrono/fea/ChElementBeamANCF_MT08.h"
#include "chrono/fea/ChElementBeamANCF_MT09.h"
#include "chrono/fea/ChElementBeamANCF_MT10.h"
#include "chrono/fea/ChElementBeamANCF_MT11.h"
#include "chrono/fea/ChElementBeamANCF_MT12.h"
#include "chrono/fea/ChElementBeamANCF_MT13.h"
#include "chrono/fea/ChElementBeamANCF_MT14.h"
#include "chrono/fea/ChElementBeamANCF_MT15.h"
#include "chrono/fea/ChElementBeamANCF_MT16.h"
#include "chrono/fea/ChElementBeamANCF_MT17.h"
#include "chrono/fea/ChElementBeamANCF_MT18.h"
#include "chrono/fea/ChElementBeamANCF_MT19.h"
#include "chrono/fea/ChElementBeamANCF_MT20.h"
#include "chrono/fea/ChElementBeamANCF_MT21.h"
#include "chrono/fea/ChElementBeamANCF_MT22.h"
#include "chrono/fea/ChElementBeamANCF_MT23.h"
#include "chrono/fea/ChElementBeamANCF_MT24.h"
#include "chrono/fea/ChElementBeamANCF_MT25.h"
#include "chrono/fea/ChElementBeamANCF_MT26.h"
#include "chrono/fea/ChElementBeamANCF_MT27.h"
#include "chrono/fea/ChElementBeamANCF_MT28.h"
#include "chrono/fea/ChElementBeamANCF_MT29.h"
#include "chrono/fea/ChElementBeamANCF_MT30.h"
#include "chrono/fea/ChElementBeamANCF_MT31.h"
#include "chrono/fea/ChElementBeamANCF_MT32.h"
#include "chrono/fea/ChElementBeamANCF_MT33.h"
#include "chrono/fea/ChElementBeamANCF_MT34.h"
#include "chrono/fea/ChElementBeamANCF_MT35.h"
#include "chrono/fea/ChElementBeamANCF_MT36.h"
#include "chrono/fea/ChElementBeamANCF_MT37.h"
#include "chrono/fea/ChElementBeamANCF_MT38.h"
#include "chrono/fea/ChElementBeamANCF_MT60.h"
#include "chrono/fea/ChElementBeamANCF_MT61.h"
#include "chrono/fea/ChElementBeamANCF_MT62.h"

#include "chrono/fea/ChMesh.h"
#include "chrono/fea/ChVisualizationFEAmesh.h"
#include "chrono/physics/ChLoadContainer.h"
#include "chrono/fea/ChLoadsBeam.h"
#include "chrono/fea/ChLinkPointFrame.h"

#ifdef CHRONO_IRRLICHT
#include "chrono_irrlicht/ChIrrApp.h"
#endif

#ifdef CHRONO_MKL
#include "chrono_mkl/ChSolverMKL.h"
#endif

#ifdef CHRONO_MUMPS
#include "chrono_mumps/ChSolverMumps.h"
#endif

using namespace chrono;
using namespace chrono::fea;
//using namespace postprocess;

enum class SolverType { MINRES, MKL, MUMPS, SparseLU, SparseQR };

// =============================================================================

#define NUM_SKIP_STEPS 0  // number of steps for hot start
#define NUM_SIM_STEPS 100  // number of simulation steps for each benchmark
#define REPEATS 20

// =============================================================================

template <typename ElementVersion, typename MaterialVersion>
class ANCFBeamTest {
public:
    ANCFBeamTest(int num_elements, SolverType solver_type, double f);

    ~ANCFBeamTest() { delete m_system; }

    ChSystem* GetSystem() { return m_system; }
    void ExecuteStep() { m_system->DoStepDynamics(1e-3); }

    void SimulateVis();
    void NonlinearStatics() { m_system->DoStaticNonlinear(50); }
    ChVector<> GetBeamEndPointPos() { return m_nodeEndPoint->GetPos(); }

    void RunTimingTest(ChMatrixNM<double, 4, 19>& timing_stats, const std::string& test_name);

protected:
    ChSystemSMC* m_system;
    std::shared_ptr<ChNodeFEAxyzDD> m_nodeEndPoint;
    std::shared_ptr<ChNodeFEAxyzDD> m_nodeMidPoint;
    SolverType m_SolverType;
    int m_NumElements;
};

template <typename ElementVersion, typename MaterialVersion>
ANCFBeamTest<ElementVersion, MaterialVersion>::ANCFBeamTest(int num_elements, SolverType solver_type, double f) {
    m_SolverType = solver_type;
    m_NumElements = num_elements;
    m_system = new ChSystemSMC();
    m_system->Set_G_acc(ChVector<>(0, 0, -9.80665));

    // Set solver parameters
#ifndef CHRONO_MKL
    if (m_SolverType == SolverType::MKL) {
        m_SolverType = SolverType::SparseLU;
        std::cout << "WARNING! Chrono::MKL not enabled. Forcing use of Eigen SparseLU solver" << std::endl;
    }
#endif

#ifndef CHRONO_MUMPS
    if (m_SolverType == SolverType::MUMPS) {
        m_SolverType = SolverType::SparseLU;
        std::cout << "WARNING! Chrono::MUMPS not enabled. Forcing use of Eigen SparseLU solver" << std::endl;
    }
#endif

    switch (m_SolverType) {
    case SolverType::MINRES: {
        auto solver = chrono_types::make_shared<ChSolverMINRES>();
        m_system->SetSolver(solver);
        solver->SetMaxIterations(100);
        solver->SetTolerance(1e-10);
        solver->EnableDiagonalPreconditioner(true);
        solver->SetVerbose(false);

        m_system->SetSolverForceTolerance(1e-10);
        break;
    }
    case SolverType::MKL: {
#ifdef CHRONO_MKL
        auto solver = chrono_types::make_shared<ChSolverMKL>();
        solver->UseSparsityPatternLearner(false);
        solver->LockSparsityPattern(true);
        solver->SetVerbose(false);
        m_system->SetSolver(solver);
#endif
        break;
    }
    case SolverType::MUMPS: {
#ifdef CHRONO_MUMPS
        auto solver = chrono_types::make_shared<ChSolverMumps>();
        solver->UseSparsityPatternLearner(true);
        solver->LockSparsityPattern(true);
        solver->SetVerbose(false);
        m_system->SetSolver(solver);
#endif
        break;
    }
    case SolverType::SparseLU: {
        auto solver = chrono_types::make_shared<ChSolverSparseLU>();
        solver->UseSparsityPatternLearner(true);
        solver->LockSparsityPattern(true);
        solver->SetVerbose(false);
        m_system->SetSolver(solver);
        break;
    }

    case SolverType::SparseQR: {
        auto solver = chrono_types::make_shared<ChSolverSparseQR>();
        solver->UseSparsityPatternLearner(true);
        solver->LockSparsityPattern(true);
        solver->SetVerbose(false);
        m_system->SetSolver(solver);
        break;
    }
    }

    // Set up integrator
    m_system->SetTimestepperType(ChTimestepper::Type::HHT);
    auto integrator = std::static_pointer_cast<ChTimestepperHHT>(m_system->GetTimestepper());
    integrator->SetAlpha(-0.2);
    integrator->SetMaxiters(100);
    integrator->SetAbsTolerances(1e-5);
    integrator->SetMode(ChTimestepperHHT::POSITION);
    integrator->SetScaling(true);
    integrator->SetVerbose(false);
    integrator->SetModifiedNewton(true);

    // Mesh properties
    double length = 1;       // m
    double area = 10e-6 * (f*f); // m^2
    double width = std::sqrt(area);      // m
    double thickness = std::sqrt(area);  // m
    double rho = 8000 / (f*f);     // kg/m^3
    double E = 10e7 / (std::pow(f, 4)); // Pa
    double nu = 0;           // Poisson effect neglected for this model
    double k1 =
        10 * (1 + nu) / (12 + 11 * nu);  // Timoshenko shear correction coefficient for a rectangular cross-section
    double k2 = k1;                      // Timoshenko shear correction coefficient for a rectangular cross-section

    double omega_y = -4.0;  //rad/s initial angular velocity about the Y axis

    auto material = chrono_types::make_shared<MaterialVersion>(rho, E, nu, k1, k2);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    m_system->Add(mesh);

    // Setup visualization - Need to fix this section
    auto vis_surf = chrono_types::make_shared<ChVisualizationFEAmesh>(*mesh);
    vis_surf->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_SURFACE);
    vis_surf->SetWireframe(true);
    vis_surf->SetDrawInUndeformedReference(true);
    mesh->AddAsset(vis_surf);

    auto vis_node = chrono_types::make_shared<ChVisualizationFEAmesh>(*mesh);
    vis_node->SetFEMglyphType(ChVisualizationFEAmesh::E_GLYPH_NODE_DOT_POS);
    vis_node->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_NONE);
    vis_node->SetSymbolsThickness(0.01);
    mesh->AddAsset(vis_node);

    auto mvisualizebeamA = chrono_types::make_shared<ChVisualizationFEAmesh>(*(mesh.get()));
    mvisualizebeamA->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_ANCF_BEAM_AX);
    mvisualizebeamA->SetColorscaleMinMax(-0.005, 0.005);
    mvisualizebeamA->SetSmoothFaces(true);
    mvisualizebeamA->SetWireframe(false);
    mesh->AddAsset(mvisualizebeamA);

    auto mvisualizebeamC = chrono_types::make_shared<ChVisualizationFEAmesh>(*(mesh.get()));
    mvisualizebeamC->SetFEMglyphType(ChVisualizationFEAmesh::E_GLYPH_NODE_CSYS);
    mvisualizebeamC->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_NONE);
    mvisualizebeamC->SetSymbolsThickness(0.006);
    mvisualizebeamC->SetSymbolsScale(0.005);
    mvisualizebeamC->SetZbufferHide(false);
    mesh->AddAsset(mvisualizebeamC);

    // Populate the mesh container with a the nodes and elements for the meshed beam
    int num_nodes = (2 * num_elements) + 1;
    double dx = length / (num_nodes - 1);

    // Setup beam cross section gradients to initially align with the global y and z directions
    ChVector<> dir1(0, 1, 0);
    ChVector<> dir2(0, 0, 1);

    // Create a grounded body to connect the 3D pendulum to
    auto grounded = chrono_types::make_shared<ChBody>();
    grounded->SetBodyFixed(true);
    m_system->Add(grounded);

    // Create the first node and fix only its position to ground (Spherical Joint constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0, 0.0), dir1, dir2);
    mesh->AddNode(nodeA);
    auto pos_constraint = chrono_types::make_shared<ChLinkPointFrame>();
    pos_constraint->Initialize(nodeA, grounded);  // body to be connected to
    m_system->Add(pos_constraint);

    int counter_nodes = 0;
    for (int i = 1; i <= num_elements; i++) {
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(dx * (2 * i - 1), 0, 0), dir1, dir2);
        nodeC->SetPos_dt(ChVector<>(0, omega_y*(dx * (2 * i - 1)), 0));
        counter_nodes++;
        if(counter_nodes == num_elements)
            m_nodeMidPoint = nodeC;

        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(dx * (2 * i), 0, 0), dir1, dir2);
        nodeB->SetPos_dt(ChVector<>(0, omega_y*(dx * (2 * i)), 0));
        counter_nodes++;
        if (counter_nodes == num_elements)
            m_nodeMidPoint = nodeB;

        mesh->AddNode(nodeB);
        mesh->AddNode(nodeC);

        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC);
        element->SetDimensions(2 * dx, thickness, width);
        element->SetMaterial(material);
        element->SetAlphaDamp(0.01);
        element->SetGravityOn(
            true);  // Enable the efficient ANCF method for calculating the application of gravity to the element
        element->SetStrainFormulation(ElementVersion::StrainFormulation::CMPoisson);
        // element->SetStrainFormulation(ElementVersion::StrainFormulation::CMNoPoisson);
        mesh->AddElement(element);

        nodeA = nodeB;
    }

    m_nodeEndPoint = nodeA;


    mesh->SetAutomaticGravity(false);  // Turn off the default method for applying gravity to the mesh since it is less
                                       // efficient for ANCF elements

}

template <typename ElementVersion, typename MaterialVersion>
void ANCFBeamTest<ElementVersion, MaterialVersion>::SimulateVis() {
#ifdef CHRONO_IRRLICHT
    irrlicht::ChIrrApp application(m_system, L"ANCF beams", irr::core::dimension2d<irr::u32>(800, 600), false, true);
    application.AddTypicalLogo();
    application.AddTypicalSky();
    application.AddTypicalLights();
    application.AddTypicalCamera(irr::core::vector3df(-0.4f, 0.4f, 0.4f), irr::core::vector3df(0, 0, 0));

    application.AssetBindAll();
    application.AssetUpdateAll();

    while (application.GetDevice()->run()) {
        std::cout << "Time(s): " << this->m_system->GetChTime() << "  Tip Pos(m): " << this->GetBeamEndPointPos()
            << std::endl;
        application.BeginScene();
        application.DrawAll();
        irrlicht::ChIrrTools::drawSegment(application.GetVideoDriver(), ChVector<>(0), ChVector<>(1, 0, 0),
            irr::video::SColor(255, 255, 0, 0));
        irrlicht::ChIrrTools::drawSegment(application.GetVideoDriver(), ChVector<>(0), ChVector<>(0, 1, 0),
            irr::video::SColor(255, 0, 255, 0));
        irrlicht::ChIrrTools::drawSegment(application.GetVideoDriver(), ChVector<>(0), ChVector<>(0, 0, 1),
            irr::video::SColor(255, 0, 0, 255));
        ExecuteStep();
        application.EndScene();
    }
#endif
}

template <typename ElementVersion, typename MaterialVersion>
void ANCFBeamTest<ElementVersion, MaterialVersion>::RunTimingTest(ChMatrixNM<double, 4, 19>& timing_stats,
    const std::string& test_name) {
    // Timing Results entries (in seconds)
    //  - "Step_Total"
    //  - "Step_Advance"
    //  - "Step_Update"
    //  - "LS_Jacobian"
    //  - "LS_Setup"
    //  - "LS_Setup_Asm"
    //  - "LS_Setup_Solver"
    //  - "LS_Solve"
    //  - "LS_Solve_Asm"
    //  - "LS_Solve_Solver"
    //  - "CD_Total"
    //  - "CD_Broad"
    //  - "CD_Narrow"
    //  - "FEA_InternalFrc"
    //  - "FEA_Jacobian"
    //  - "FEA_InternalFrc_Calls"
    //  - "FEA_Jacobian_Calls"

    // Reset timing results since the results will be accumulated into this vector
    ChMatrixNM<double, REPEATS, 19> timing_results;
    timing_results.setZero();

    // Run the requested number of steps to warm start the system, but do not collect any timing information
    for (int i = 0; i < NUM_SKIP_STEPS; i++) {
        ExecuteStep();
    }

    ChMatrixDynamic<double> tip_displacement_z;
    tip_displacement_z.resize(NUM_SIM_STEPS*REPEATS, 2);
    //ip_displacement_z.resize(NUM_SIM_STEPS, 2);

    ChMatrixDynamic<double> midpoint_displacement;
    midpoint_displacement.resize(NUM_SIM_STEPS*REPEATS, 4);

    // Time the requested number of steps, collecting timing information (systems is not restarted between collections)
    auto LS = std::dynamic_pointer_cast<ChDirectSolverLS>(GetSystem()->GetSolver());
    auto MeshList = GetSystem()->Get_meshlist();
    for (int r = 0; r < REPEATS; r++) {
        for (int i = 0; i < NUM_SIM_STEPS; i++) {
            for (auto& Mesh : MeshList) {
                Mesh->ResetTimers();
                Mesh->ResetCounters();
            }
            if (LS != NULL) {  // Direct Solver
                LS->ResetTimers();
            }
            GetSystem()->ResetTimers();

            ExecuteStep();

            timing_results(r, 0) += GetSystem()->GetTimerStep();
            timing_results(r, 1) += GetSystem()->GetTimerAdvance();
            timing_results(r, 2) += GetSystem()->GetTimerUpdate();

            timing_results(r, 3) += GetSystem()->GetTimerJacobian();
            timing_results(r, 4) += GetSystem()->GetTimerSetup();
            timing_results(r, 7) += GetSystem()->GetTimerSolver();
            timing_results(r, 10) += GetSystem()->GetTimerCollision();
            timing_results(r, 11) += GetSystem()->GetTimerCollisionBroad();
            timing_results(r, 12) += GetSystem()->GetTimerCollisionNarrow();

            if (LS != NULL) {  // Direct Solver
                timing_results(r, 5) += LS->GetTimeSetup_Assembly();
                timing_results(r, 6) += LS->GetTimeSetup_SolverCall();
                timing_results(r, 8) += LS->GetTimeSolve_Assembly();
                timing_results(r, 9) += LS->GetTimeSolve_SolverCall();
            }

            // Accumulate the internal force and Jacobian timers across all the FEA mesh containers
            // auto MeshList = GetSystem()->Get_meshlist();
            for (auto& Mesh : MeshList) {
                timing_results(r, 13) += Mesh->GetTimeInternalForces();
                timing_results(r, 14) += Mesh->GetTimeJacobianLoad();
                timing_results(r, 15) += Mesh->GetNumCallsInternalForces();
                timing_results(r, 16) += Mesh->GetNumCallsJacobianLoad();
            }

            //if (r == (REPEATS - 1)) {
            //    tip_displacement_z(i, 0) = GetSystem()->GetChTime();
            //    tip_displacement_z(i, 1) = GetBeamEndPointPos().z();
            //}
            tip_displacement_z(i + (r * NUM_SIM_STEPS), 0) = GetSystem()->GetChTime();
            tip_displacement_z(i + (r * NUM_SIM_STEPS), 1) = GetBeamEndPointPos().z();
            midpoint_displacement(i + (r * NUM_SIM_STEPS), 0) = GetSystem()->GetChTime();
            midpoint_displacement(i + (r * NUM_SIM_STEPS), 1) = GetBeamEndPointPos().x();
            midpoint_displacement(i + (r * NUM_SIM_STEPS), 2) = GetBeamEndPointPos().y();
            midpoint_displacement(i + (r * NUM_SIM_STEPS), 3) = GetBeamEndPointPos().z();
        }
        timing_results(r, 17) = (timing_results(r, 13) * 1e6) / (timing_results(r, 15) * double(m_NumElements));
        timing_results(r, 18) = (timing_results(r, 14) * 1e6) / (timing_results(r, 16) * double(m_NumElements));
    }

    // Scale times from s to ms
    timing_results.block(0, 0, REPEATS, 15) *= 1e3;

    // Compute statistics (min, max, median, mean, std deviation)
    timing_stats.row(0) = timing_results.colwise().minCoeff();
    timing_stats.row(1) = timing_results.colwise().maxCoeff();
    timing_stats.row(2) = timing_results.colwise().mean();
    for (auto c = 0; c < timing_stats.cols(); c++) {  // compute the standard deviation column by column
        timing_stats(3, c) = std::sqrt((timing_results.col(c).array() - timing_results.col(c).mean()).square().sum() /
            (timing_results.col(c).size() - 1));
    }

    double tip_displacement_offset = (tip_displacement_z.col(1).maxCoeff() + tip_displacement_z.col(1).minCoeff()) / 2;

    std::cout << "-------------------------------------" << std::endl;
    std::cout << test_name << " - Num_Elements: " << m_NumElements << " - Linear_Solver: ";
    switch (m_SolverType) {
    case SolverType::MINRES:
        std::cout << "MINRES";
        ;
        break;
    case SolverType::MKL:
        std::cout << "MKL";
        ;
        break;
    case SolverType::MUMPS:
        std::cout << "MUMPS";
        ;
        break;
    case SolverType::SparseLU:
        std::cout << "SparseLU";
        ;
        break;
    case SolverType::SparseQR:
        std::cout << "SparseQR";
        ;
        break;
    }
#ifdef _OPENMP
    std::cout << " - Max_OMP_Threads: " << omp_get_max_threads();
#endif
    std::cout << " - Tip_Displacement_Mean = " << tip_displacement_offset << std::endl;

    std::cout << "Step_Total "
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
        << "FEA_Jacobian_Calls" 
        << "FEA_InternalFrc_AvgFunctionCall_us"
        << "FEA_Jacobian_AvgFunctionCall_us"
        << std::endl;
    for (int r = 0; r < REPEATS; r++) {
        std::cout << "Run_" << r << ":\t" << timing_results.row(r) << std::endl;
    }
    std::cout << "Min:\t" << timing_stats.row(0) << std::endl;
    std::cout << "Max:\t" << timing_stats.row(1) << std::endl;
    std::cout << "Mean:\t" << timing_stats.row(2) << std::endl;
    std::cout << "StdDev:\t" << timing_stats.row(3) << std::endl;

    //// Create (if needed) output directory
    //const std::string out_dir = GetChronoOutputPath() + "DEMO_GNUPLOT";
    //if (!filesystem::create_directory(filesystem::path(out_dir))) {
    //    std::cout << "Error creating directory " << out_dir << std::endl;
    //    return;
    //}
    //// Create the plot.
    //// NOTE. The plot shortcuts.
    //std::string filename = out_dir + "/temp_plot.gpl";
    //ChGnuPlot mplot(filename.c_str());
    //mplot.SetGrid();
    //mplot.Plot(midpoint_displacement, 0, 3, "from ChMatrix", " with lines lt 5");

    //std::string filename2 = out_dir + "/temp_plot2.gpl";
    //ChGnuPlot mplot2(filename2.c_str());
    //mplot2.SetGrid();
    //mplot2.Plot(midpoint_displacement, 1, 2, "from ChMatrix", " with lines lt 5");

    //std::string filename3 = out_dir + "/temp_plot3.gpl";
    //ChGnuPlot mplot3(filename3.c_str());
    //mplot3.SetGrid();
    //mplot3.Plot(midpoint_displacement, 1, 3, "from ChMatrix", " with lines lt 5");

}

int main(int argc, char* argv[]) {
    //ChVectorN<int, 8> num_els;
    //num_els << 8, 16, 32, 64, 128, 256, 512, 1024;

    //std::vector<SolverType> Solver = {SolverType::MINRES, SolverType::MKL, SolverType::MUMPS, SolverType::SparseLU,
    //                                  SolverType::SparseQR};
    //std::vector<SolverType> Solver = {SolverType::MKL, SolverType::MUMPS, SolverType::SparseLU, SolverType::SparseQR};

    ChVectorN<int, 1> num_els;
    num_els << 1024;
    std::vector<SolverType> Solver = { SolverType::SparseLU };

    double f = 5.0;

    ChMatrixNM<double, 4, 19> timing_stats;
    
    //ANCFBeamTest<ChElementBeamANCF_MT27, ChMaterialBeamANCF_MT27> test(8, SolverType::MUMPS, 2);
    //test.SimulateVis();

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF, ChMaterialBeamANCF> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT01, ChMaterialBeamANCF_MT01> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT01");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT02, ChMaterialBeamANCF_MT02> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT02");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT03, ChMaterialBeamANCF_MT03> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT03");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT04, ChMaterialBeamANCF_MT04> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT04");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT05, ChMaterialBeamANCF_MT05> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT05");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT06, ChMaterialBeamANCF_MT06> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT06");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT07, ChMaterialBeamANCF_MT07> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT07");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT08, ChMaterialBeamANCF_MT08> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT08");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT09, ChMaterialBeamANCF_MT09> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT09");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT10, ChMaterialBeamANCF_MT10> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT10");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT11, ChMaterialBeamANCF_MT11> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT11");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT12, ChMaterialBeamANCF_MT12> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT12");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT13, ChMaterialBeamANCF_MT13> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT13");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT14, ChMaterialBeamANCF_MT14> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT14");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT15, ChMaterialBeamANCF_MT15> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT15");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT16, ChMaterialBeamANCF_MT16> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT16");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT17, ChMaterialBeamANCF_MT17> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT17");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT18, ChMaterialBeamANCF_MT18> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT18");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT19, ChMaterialBeamANCF_MT19> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT19");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT20, ChMaterialBeamANCF_MT20> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT20");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT21, ChMaterialBeamANCF_MT21> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT21");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT22, ChMaterialBeamANCF_MT22> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT22");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT23, ChMaterialBeamANCF_MT23> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT23");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT24, ChMaterialBeamANCF_MT24> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT24");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT25, ChMaterialBeamANCF_MT25> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT25");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT26");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT27, ChMaterialBeamANCF_MT27> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT27");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT28, ChMaterialBeamANCF_MT28> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT28");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT29, ChMaterialBeamANCF_MT29> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT29");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT30, ChMaterialBeamANCF_MT30> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT30");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT31");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT32, ChMaterialBeamANCF_MT32> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT32");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT33, ChMaterialBeamANCF_MT33> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT33");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT34, ChMaterialBeamANCF_MT34> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT34");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT35, ChMaterialBeamANCF_MT35> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT35");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT36, ChMaterialBeamANCF_MT36> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT36");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT37, ChMaterialBeamANCF_MT37> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT37");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT38, ChMaterialBeamANCF_MT38> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT38");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT60, ChMaterialBeamANCF_MT60> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT60");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT61, ChMaterialBeamANCF_MT61> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT61");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT62, ChMaterialBeamANCF_MT62> test(num_els(i), ls, f);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT62");
        }
    }

    return(0);
}

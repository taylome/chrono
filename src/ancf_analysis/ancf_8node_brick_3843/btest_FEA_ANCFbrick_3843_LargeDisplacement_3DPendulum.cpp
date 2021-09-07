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
// ANCF shell elements - Simple Plate Pendulum modified for the Brick element
//
// With Modifications from:
// Aki M Mikkola and Ahmed A Shabana. A non-incremental finite element procedure
// for the analysis of large deformation of plates and shells in mechanical 
// system applications. Multibody System Dynamics, 9(3) : 283–309, 2003.
//
// =============================================================================

#include <string>

#include "chrono/ChConfig.h"

#include "chrono/parallel/ChOpenMP.h"

#include "chrono/physics/ChSystemSMC.h"
#include "chrono/solver/ChIterativeSolverLS.h"
#include "chrono/solver/ChDirectSolverLS.h"

//#include "chrono_postprocess/ChGnuPlot.h"
//#include "chrono_thirdparty/filesystem/path.h"

#include "chrono/fea/ChElementBrickANCF_3843_TR01.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR02.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR02_GQ444.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR03.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR03_GQ444.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR04.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR04_GQ444.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR05.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR05_GQ444.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR06.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR06_GQ444.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR07.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR07s.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR07s_GQ444.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR08.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR08s.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR08s_GQ444.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR08T_GQ444.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR09.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR10.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR11.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR11s.h"

#include "chrono/fea/ChMesh.h"
#include "chrono/fea/ChVisualizationFEAmesh.h"
#include "chrono/physics/ChLoadContainer.h"
#include "chrono/fea/ChLoadsBeam.h"
#include "chrono/fea/ChLinkPointFrame.h"

#ifdef CHRONO_IRRLICHT
#include "chrono_irrlicht/ChIrrApp.h"
#endif

//#undef CHRONO_PARDISO_MKL
#ifdef CHRONO_PARDISO_MKL
#include "chrono_pardisomkl/ChSolverPardisoMKL.h"
#endif

#ifdef CHRONO_MUMPS
#include "chrono_mumps/ChSolverMumps.h"
#endif

using namespace chrono;
using namespace chrono::fea;
// using namespace postprocess;

enum class SolverType { MINRES, MKL, MUMPS, SparseLU, SparseQR };

// =============================================================================

#define NUM_SKIP_STEPS 10  // number of steps for hot start
#define NUM_SIM_STEPS 100  // number of simulation steps for each benchmark
#define REPEATS 10

// =============================================================================

template <typename ElementVersion, typename MaterialVersion>
class ANCFBrickTest {
  public:
    ANCFBrickTest(int num_elements, SolverType solver_type, int NumThreads);

    ~ANCFBrickTest() { delete m_system; }

    ChSystem* GetSystem() { return m_system; }
    void ExecuteStep() { m_system->DoStepDynamics(1e-3); }

    void SimulateVis();
    void NonlinearStatics() { m_system->DoStaticNonlinear(50); }
    ChVector<> GetBeamEndPointPos() { return m_nodeEndPoint->GetPos(); }

    void RunTimingTest(ChMatrixNM<double, 4, 19>& timing_stats, const std::string& test_name);

  protected:
    ChSystemSMC* m_system;
    std::shared_ptr<ChNodeFEAxyzDDD> m_nodeEndPoint;
    SolverType m_SolverType;
    int m_NumElements;
    int m_NumThreads;
};

template <typename ElementVersion, typename MaterialVersion>
ANCFBrickTest<ElementVersion, MaterialVersion>::ANCFBrickTest(int num_elements,
                                                            SolverType solver_type,
                                                            int NumThreads) {
    m_SolverType = solver_type;
    m_NumElements = 2 * num_elements * num_elements;
    m_NumThreads = NumThreads;
    m_system = new ChSystemSMC();
    m_system->Set_G_acc(ChVector<>(0, 0, -9.80665));
    m_system->SetNumThreads(NumThreads, 1, NumThreads);

    // Set solver parameters
#ifndef CHRONO_PARDISO_MKL
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
#ifdef CHRONO_PARDISO_MKL
            auto solver = chrono_types::make_shared<ChSolverPardisoMKL>(NumThreads);
            solver->UseSparsityPatternLearner(false);
            solver->LockSparsityPattern(true);
            solver->SetVerbose(false);
            m_system->SetSolver(solver);
#endif
            break;
        }
        case SolverType::MUMPS: {
#ifdef CHRONO_MUMPS
            auto solver = chrono_types::make_shared<ChSolverMumps>(NumThreads);
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
    double length = 0.6;        // m
    double width = 0.3;         // m
    double thickness = 0.01;    // m
    double rho = 7810;          // kg/m^3
    double E = 1.0e5;            // Pa
    double nu = 0.3;            // Poisson's Ratio

    auto material = chrono_types::make_shared<MaterialVersion>(rho, E, nu);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    m_system->Add(mesh);

    // Setup visualization - Need to fix this section
    //auto vis_surf = chrono_types::make_shared<ChVisualizationFEAmesh>(*mesh);
    //vis_surf->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_SURFACE);
    //vis_surf->SetWireframe(true);
    //vis_surf->SetDrawInUndeformedReference(true);
    //mesh->AddAsset(vis_surf);

    auto vis_node = chrono_types::make_shared<ChVisualizationFEAmesh>(*mesh);
    vis_node->SetFEMglyphType(ChVisualizationFEAmesh::E_GLYPH_NODE_DOT_POS);
    vis_node->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_NONE);
    vis_node->SetSymbolsThickness(0.01);
    mesh->AddAsset(vis_node);

    // Populate the mesh container with a the nodes and elements for the meshed beam
    double dx = length / (2*num_elements);
    double dy = width / (num_elements);

    // Setup beam cross section gradients to initially align with the global y and z directions
    ChVector<> dir1(1, 0, 0);
    ChVector<> dir2(0, 1, 0);
    ChVector<> dir3(0, 0, 1);

    // Create a grounded body to connect the 3D pendulum to
    auto grounded = chrono_types::make_shared<ChBody>();
    grounded->SetBodyFixed(true);
    m_system->Add(grounded);

    // Create and add the nodes
    for (auto i = 0; i <= 2*num_elements; i++) {
        for (auto j = 0; j <= num_elements; j++) {
            auto node = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(dx * i, dy * j, 0.0), dir1, dir2, dir3);
            mesh->AddNode(node);

            // Fix only the first node's position to ground (Spherical Joint constraint)
            if ((i == 0) && (j == 0)) {
                auto pos_constraint = chrono_types::make_shared<ChLinkPointFrame>();
                pos_constraint->Initialize(node, grounded);  // body to be connected to
                m_system->Add(pos_constraint);
            }

            auto nodetop = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(dx * i, dy * j, thickness), dir1, dir2, dir3);
            mesh->AddNode(nodetop);
        }
    }

    // Create and add the elements
    for (auto i = 0; i < 2 * num_elements; i++) {
        for (auto j = 0; j < num_elements; j++) {
            int nodeA_idx = 2*j + 2 * i * (num_elements + 1);
            int nodeD_idx = 2*(j + 1) + 2 * i * (num_elements + 1);
            int nodeB_idx = 2*j + 2 * (i + 1) * (num_elements + 1);
            int nodeC_idx = 2*(j + 1) + 2 * (i + 1) * (num_elements + 1);

            int nodeE_idx = nodeA_idx + 1;
            int nodeH_idx = nodeD_idx + 1;
            int nodeF_idx = nodeB_idx + 1;
            int nodeG_idx = nodeC_idx + 1;

            // std::cout << "A:" << nodeA_idx << "  B:" << nodeB_idx << "  C:" << nodeC_idx << "  D:" << nodeD_idx << std::endl;

            auto element = chrono_types::make_shared<ElementVersion>();
            element->SetNodes(std::dynamic_pointer_cast<ChNodeFEAxyzDDD>(mesh->GetNode(nodeA_idx)),
                std::dynamic_pointer_cast<ChNodeFEAxyzDDD>(mesh->GetNode(nodeB_idx)),
                std::dynamic_pointer_cast<ChNodeFEAxyzDDD>(mesh->GetNode(nodeC_idx)),
                std::dynamic_pointer_cast<ChNodeFEAxyzDDD>(mesh->GetNode(nodeD_idx)),
                std::dynamic_pointer_cast<ChNodeFEAxyzDDD>(mesh->GetNode(nodeE_idx)),
                std::dynamic_pointer_cast<ChNodeFEAxyzDDD>(mesh->GetNode(nodeF_idx)),
                std::dynamic_pointer_cast<ChNodeFEAxyzDDD>(mesh->GetNode(nodeG_idx)),
                std::dynamic_pointer_cast<ChNodeFEAxyzDDD>(mesh->GetNode(nodeH_idx)));
            element->SetDimensions(dx, dy, thickness);
            element->SetMaterial(material);
            element->SetAlphaDamp(0.01);
            element->SetGravityOn(
                true);  // Enable the efficient ANCF method for calculating the application of gravity to the element
            mesh->AddElement(element);

            m_nodeEndPoint = std::dynamic_pointer_cast<ChNodeFEAxyzDDD>(mesh->GetNode(nodeC_idx));
        }
    }
    
    mesh->SetAutomaticGravity(false);  // Turn off the default method for applying gravity to the mesh since it is less
                                       // efficient for ANCF elements
}

template <typename ElementVersion, typename MaterialVersion>
void ANCFBrickTest<ElementVersion, MaterialVersion>::SimulateVis() {
#ifdef CHRONO_IRRLICHT
    irrlicht::ChIrrApp application(m_system, L"ANCF Bricks 3843", irr::core::dimension2d<irr::u32>(800, 600), false, true);
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
void ANCFBrickTest<ElementVersion, MaterialVersion>::RunTimingTest(ChMatrixNM<double, 4, 19>& timing_stats,
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
    tip_displacement_z.resize(NUM_SIM_STEPS * REPEATS, 2);
    // ip_displacement_z.resize(NUM_SIM_STEPS, 2);

    ChMatrixDynamic<double> midpoint_displacement;
    midpoint_displacement.resize(NUM_SIM_STEPS * REPEATS, 4);

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

            // std::cout << "Sim Time  = " << GetSystem()->GetChTime() << std::endl;

            ExecuteStep();

            timing_results(r, 0) += GetSystem()->GetTimerStep();
            timing_results(r, 1) += GetSystem()->GetTimerAdvance();
            timing_results(r, 2) += GetSystem()->GetTimerUpdate();

            timing_results(r, 3) += GetSystem()->GetTimerJacobian();
            timing_results(r, 4) += GetSystem()->GetTimerLSsetup();
            timing_results(r, 7) += GetSystem()->GetTimerLSsolve();
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

            // if (r == (REPEATS - 1)) {
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

    // std::cout << " - Num_Procs: " << ChOMP::GetNumProcs();

    std::cout << " - Requested_Threads: " << m_NumThreads;

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
              << "FEA_Jacobian_Calls "
              << "FEA_InternalFrc_AvgFunctionCall_us "
              << "FEA_Jacobian_AvgFunctionCall_us" << std::endl;
    for (int r = 0; r < REPEATS; r++) {
        std::cout << "Run_" << r << ":\t" << timing_results.row(r) << std::endl;
    }
    std::cout << "Min:\t" << timing_stats.row(0) << std::endl;
    std::cout << "Max:\t" << timing_stats.row(1) << std::endl;
    std::cout << "Mean:\t" << timing_stats.row(2) << std::endl;
    std::cout << "StdDev:\t" << timing_stats.row(3) << std::endl;

    //// Create (if needed) output directory
    // const std::string out_dir = GetChronoOutputPath() + "DEMO_GNUPLOT";
    // if (!filesystem::create_directory(filesystem::path(out_dir))) {
    //    std::cout << "Error creating directory " << out_dir << std::endl;
    //    return;
    //}
    //// Create the plot.
    //// NOTE. The plot shortcuts.
    // std::string filename = out_dir + "/temp_plot.gpl";
    // ChGnuPlot mplot(filename.c_str());
    // mplot.SetGrid();
    // mplot.Plot(midpoint_displacement, 0, 3, "from ChMatrix", " with lines lt 5");

    // std::string filename2 = out_dir + "/temp_plot2.gpl";
    // ChGnuPlot mplot2(filename2.c_str());
    // mplot2.SetGrid();
    // mplot2.Plot(midpoint_displacement, 1, 2, "from ChMatrix", " with lines lt 5");

    // std::string filename3 = out_dir + "/temp_plot3.gpl";
    // ChGnuPlot mplot3(filename3.c_str());
    // mplot3.SetGrid();
    // mplot3.Plot(midpoint_displacement, 1, 3, "from ChMatrix", " with lines lt 5");
}

int main(int argc, char* argv[]) {
    //ChVectorN<int, 8> num_els;
    //num_els << 8, 16, 32, 64, 128, 256, 512, 1024;
    //ChVectorN<int, 1> num_els;
    //num_els << 2;
    ChVectorN<int, 2> num_els;
    num_els << 2, 10; //2=>2x4=8, 10=>10x20=200 

    // std::vector<SolverType> Solver = {SolverType::MINRES, SolverType::MKL, SolverType::MUMPS, SolverType::SparseLU,
    //                                  SolverType::SparseQR};
    // std::vector<SolverType> Solver = {SolverType::MKL, SolverType::MUMPS, SolverType::SparseLU,
    // SolverType::SparseQR};
    std::vector<SolverType> Solver = { SolverType::SparseLU };
    // std::vector<SolverType> Solver = { SolverType::MKL };

    int MaxThreads = 1;
    MaxThreads = ChOMP::GetMaxThreads();
    std::cout << "GetNumProcs:\t" << ChOMP::GetNumProcs() << " Max Threads = " << MaxThreads << std::endl;

    ChMatrixNM<double, 4, 19> timing_stats;

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            int NumThreads = 1;
            // int NumThreads = MaxThreads;
            bool run = true;
            while (run) {
                if (i == 0) {
                    ANCFBrickTest<ChElementBrickANCF_3843_TR01, ChMaterialBrickANCF_3843_TR01> test(num_els(i), ls, NumThreads);
                    test.RunTimingTest(timing_stats, "ChElementBrickANCF_3843_TR01");
                }
                //if (i == 0) {
                //    ANCFBrickTest<ChElementBrickANCF_3843_TR02, ChMaterialBrickANCF_3843_TR02> test(num_els(i), ls, NumThreads);
                //    test.RunTimingTest(timing_stats, "ChElementBrickANCF_3843_TR02");
                //}
                if (i == 0) {
                    ANCFBrickTest<ChElementBrickANCF_3843_TR02_GQ444, ChMaterialBrickANCF_3843_TR02_GQ444> test(num_els(i), ls, NumThreads);
                    test.RunTimingTest(timing_stats, "ChElementBrickANCF_3843_TR02_GQ444");
                }
                //if (i == 0) {
                //    ANCFBrickTest<ChElementBrickANCF_3843_TR03, ChMaterialBrickANCF_3843_TR03> test(num_els(i), ls, NumThreads);
                //    test.RunTimingTest(timing_stats, "ChElementBrickANCF_3843_TR03");
                //}
                if (i == 0) {
                    ANCFBrickTest<ChElementBrickANCF_3843_TR03_GQ444, ChMaterialBrickANCF_3843_TR03_GQ444> test(num_els(i), ls, NumThreads);
                    test.RunTimingTest(timing_stats, "ChElementBrickANCF_3843_TR03_GQ444");
                }
                //if (i == 0) {
                //    ANCFBrickTest<ChElementBrickANCF_3843_TR04, ChMaterialBrickANCF_3843_TR04> test(num_els(i), ls, NumThreads);
                //    test.RunTimingTest(timing_stats, "ChElementBrickANCF_3843_TR04");
                //}
                if (i == 0) {
                    ANCFBrickTest<ChElementBrickANCF_3843_TR04_GQ444, ChMaterialBrickANCF_3843_TR04_GQ444> test(num_els(i), ls, NumThreads);
                    test.RunTimingTest(timing_stats, "ChElementBrickANCF_3843_TR04_GQ444");
                }
                //if (i == 0) {
                //    ANCFBrickTest<ChElementBrickANCF_3843_TR05, ChMaterialBrickANCF_3843_TR05> test(num_els(i), ls, NumThreads);
                //    test.RunTimingTest(timing_stats, "ChElementBrickANCF_3843_TR05");
                //}
                {
                    ANCFBrickTest<ChElementBrickANCF_3843_TR05_GQ444, ChMaterialBrickANCF_3843_TR05_GQ444> test(num_els(i), ls, NumThreads);
                    test.RunTimingTest(timing_stats, "ChElementBrickANCF_3843_TR05_GQ444");
                }
                //if (i == 0) {
                //    ANCFBrickTest<ChElementBrickANCF_3843_TR06, ChMaterialBrickANCF_3843_TR06> test(num_els(i), ls, NumThreads);
                //    test.RunTimingTest(timing_stats, "ChElementBrickANCF_3843_TR06");
                //}
                {
                    ANCFBrickTest<ChElementBrickANCF_3843_TR06_GQ444, ChMaterialBrickANCF_3843_TR06_GQ444> test(num_els(i), ls, NumThreads);
                    test.RunTimingTest(timing_stats, "ChElementBrickANCF_3843_TR06_GQ444");
                }
                //{
                //    ANCFBrickTest<ChElementBrickANCF_3843_TR07, ChMaterialBrickANCF_3843_TR07> test(num_els(i), ls, NumThreads);
                //    test.RunTimingTest(timing_stats, "ChElementBrickANCF_3843_TR07");
                //}
                //{
                //    ANCFBrickTest<ChElementBrickANCF_3843_TR07S, ChMaterialBrickANCF_3843_TR07S> test(num_els(i), ls, NumThreads);
                //    test.RunTimingTest(timing_stats, "ChElementBrickANCF_3843_TR07S");
                //}
                {
                    ANCFBrickTest<ChElementBrickANCF_3843_TR07S_GQ444, ChMaterialBrickANCF_3843_TR07S_GQ444> test(num_els(i), ls, NumThreads);
                    test.RunTimingTest(timing_stats, "ChElementBrickANCF_3843_TR07S_GQ444");
                }
                //{
                //    ANCFBrickTest<ChElementBrickANCF_3843_TR08, ChMaterialBrickANCF_3843_TR08> test(num_els(i), ls, NumThreads);
                //    test.RunTimingTest(timing_stats, "ChElementBrickANCF_3843_TR08");
                //}
                //{
                //    ANCFBrickTest<ChElementBrickANCF_3843_TR08S, ChMaterialBrickANCF_3843_TR08S> test(num_els(i), ls, NumThreads);
                //    test.RunTimingTest(timing_stats, "ChElementBrickANCF_3843_TR08S");
                //}
                {
                    ANCFBrickTest<ChElementBrickANCF_3843_TR08S_GQ444, ChMaterialBrickANCF_3843_TR08S_GQ444> test(num_els(i), ls, NumThreads);
                    test.RunTimingTest(timing_stats, "ChElementBrickANCF_3843_TR08S_GQ444");
                }
                {
                    ANCFBrickTest<ChElementBrickANCF_3843_TR08T_GQ444, ChMaterialBrickANCF_3843_TR08T_GQ444> test(num_els(i), ls, NumThreads);
                    test.RunTimingTest(timing_stats, "ChElementBrickANCF_3843_TR08T_GQ444");
                }
                {
                    ANCFBrickTest<ChElementBrickANCF_3843_TR09, ChMaterialBrickANCF_3843_TR09> test(num_els(i), ls, NumThreads);
                    test.RunTimingTest(timing_stats, "ChElementBrickANCF_3843_TR09");
                }
                {
                    ANCFBrickTest<ChElementBrickANCF_3843_TR10, ChMaterialBrickANCF_3843_TR10> test(num_els(i), ls, NumThreads);
                    test.RunTimingTest(timing_stats, "ChElementBrickANCF_3843_TR10");
                }
                //{
                //    ANCFBrickTest<ChElementBrickANCF_3843_TR11, ChMaterialBrickANCF_3843_TR11> test(num_els(i), ls, NumThreads);
                //    test.RunTimingTest(timing_stats, "ChElementBrickANCF_3843_TR11");
                //}
                {
                    ANCFBrickTest<ChElementBrickANCF_3843_TR11S, ChMaterialBrickANCF_3843_TR11S> test(num_els(i), ls, NumThreads);
                    test.RunTimingTest(timing_stats, "ChElementBrickANCF_3843_TR11S");
                }


                //run = false;
                if (NumThreads == MaxThreads)
                    run = false;

                if (NumThreads <= 4)
                    NumThreads *= 2;
                else  // Since computers this will be run on have a number of cores that is a multiple of 4
                    NumThreads += 4;

                if (NumThreads > MaxThreads)
                    NumThreads = MaxThreads;
            }
        }
    }

    return(0);
}
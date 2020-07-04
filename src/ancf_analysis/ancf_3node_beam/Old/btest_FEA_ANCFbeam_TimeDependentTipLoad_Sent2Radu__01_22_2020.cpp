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
// Authors: Radu Serban
// =============================================================================
//
// Benchmark test for ANCF shell or beam elements.
//
// Note that the MKL Pardiso and Mumps solvers are set to lock the sparsity
// pattern, but not to use the sparsity pattern learner.
//
// =============================================================================

#include "chrono/ChConfig.h"
#include "chrono/utils/ChBenchmark.h"

#include "chrono/physics/ChSystemSMC.h"
#include "chrono/solver/ChIterativeSolverLS.h"

#include "chrono/fea/ChElementShellANCF.h"
#include "chrono/fea/ChElementBeamANCF.h"
#include "chrono/fea/ChMesh.h"
#include "chrono/fea/ChVisualizationFEAmesh.h"

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

enum class SolverType { MINRES, MKL, MUMPS };

template <int N>
class ANCFbeam : public utils::ChBenchmarkTest {
public:
    virtual ~ANCFbeam() { delete m_system; }

    ChSystem* GetSystem() override { return m_system; }
    void ExecuteStep() override { m_system->DoStepDynamics(1e-4); }

    void SimulateVis();

protected:
    ANCFbeam(SolverType solver_type);

    ChSystemSMC* m_system;
};

template <int N>
class ANCFbeam_MINRES : public ANCFbeam<N> {
public:
    ANCFbeam_MINRES() : ANCFbeam<N>(SolverType::MINRES) {}
};

template <int N>
class ANCFbeam_MKL : public ANCFbeam<N> {
public:
    ANCFbeam_MKL() : ANCFbeam<N>(SolverType::MKL) {}
};

template <int N>
class ANCFbeam_MUMPS : public ANCFbeam<N> {
public:
    ANCFbeam_MUMPS() : ANCFbeam<N>(SolverType::MUMPS) {}
};

template <int N>
ANCFbeam<N>::ANCFbeam(SolverType solver_type) {
    m_system = new ChSystemSMC();
    m_system->Set_G_acc(ChVector<>(0, -9.8, 0));

    // Set solver parameters
#ifndef CHRONO_MKL
    if (solver_type == SolverType::MKL) {
        solver_type = SolverType::MINRES;
        std::cout << "WARNING! Chrono::MKL not enabled. Forcing use of MINRES solver" << std::endl;
    }
#endif

#ifndef CHRONO_MUMPS
    if (solver_type == SolverType::MUMPS) {
        solver_type = SolverType::MINRES;
        std::cout << "WARNING! Chrono::MUMPS not enabled. Forcing use of MINRES solver" << std::endl;
    }
#endif

    switch (solver_type) {
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
        solver->UseSparsityPatternLearner(false);
        solver->LockSparsityPattern(true);
        solver->SetVerbose(false);
        m_system->SetSolver(solver);
#endif
        break;
    }
    }

    // Set up integrator
    m_system->SetTimestepperType(ChTimestepper::Type::HHT);
    auto integrator = std::static_pointer_cast<ChTimestepperHHT>(m_system->GetTimestepper());
    integrator->SetAlpha(-0.2);
    integrator->SetMaxiters(100);
    integrator->SetAbsTolerances(1e-5);
    integrator->SetMode(ChTimestepperHHT::ACCELERATION);
    integrator->SetScaling(true);
    integrator->SetVerbose(true);

#if false  //Run the Shell Mesh Problem
    // Mesh properties
    double length = 1;
    double width = 0.1;
    double thickness = 0.01;

    double rho = 500;
    ChVector<> E(2.1e7, 2.1e7, 2.1e7);
    ChVector<> nu(0.3, 0.3, 0.3);
    ChVector<> G(8.0769231e6, 8.0769231e6, 8.0769231e6);
    auto mat = chrono_types::make_shared<ChMaterialShellANCF>(rho, E, nu, G);

    // Create mesh nodes and elements
    auto mesh = chrono_types::make_shared<ChMesh>();
    m_system->Add(mesh);

    auto vis_surf = chrono_types::make_shared<ChVisualizationFEAmesh>(*mesh);
    vis_surf->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_SURFACE);
    vis_surf->SetWireframe(true);
    vis_surf->SetDrawInUndeformedReference(true);
    mesh->AddAsset(vis_surf);

    auto vis_node = chrono_types::make_shared<ChVisualizationFEAmesh>(*mesh);
    vis_node->SetFEMglyphType(ChVisualizationFEAmesh::E_GLYPH_NODE_DOT_POS);
    vis_node->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_NONE);
    vis_node->SetSymbolsThickness(0.004);
    mesh->AddAsset(vis_node);

    int n_nodes = 2 * (1 + N);  //THIS IS WRONG & NOT USED
    double dx = length / N;
    ChVector<> dir(0, 1, 0);

    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzD>(ChVector<>(0, 0, -width / 2), dir);
    auto nodeB = chrono_types::make_shared<ChNodeFEAxyzD>(ChVector<>(0, 0, +width / 2), dir);
    nodeA->SetFixed(true);
    nodeB->SetFixed(true);
    mesh->AddNode(nodeA);
    mesh->AddNode(nodeB);

    for (int i = 1; i < N; i++) {  //CREATING N-1 SHELLS RATHER THAN N SHELLS :(
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzD>(ChVector<>(i * dx, 0, -width / 2), dir);
        auto nodeD = chrono_types::make_shared<ChNodeFEAxyzD>(ChVector<>(i * dx, 0, +width / 2), dir);
        mesh->AddNode(nodeC);
        mesh->AddNode(nodeD);

        auto element = chrono_types::make_shared<ChElementShellANCF>();
        element->SetNodes(nodeA, nodeB, nodeD, nodeC);
        element->SetDimensions(dx, width);
        element->AddLayer(thickness, 0 * CH_C_DEG_TO_RAD, mat);
        element->SetAlphaDamp(0.0);
        element->SetGravityOn(false);
        mesh->AddElement(element);

        nodeA = nodeC;
        nodeB = nodeD;
    }
#else  //Run the beam element system (Princeton Beam Experiment at 0deg beam rotation and no tip load)
    int num_elements = N;
    double beam_angle_rad = 0;
    double vert_tip_load_N = 0;


    // Mesh properties
    double length = 20 * 0.0254; //Beam dimension were originally in inches
    double width = 0.5*0.0254; //Beam dimension were originally in inches
    double thickness = 0.125*0.0254; //Beam dimension were originally in inches

    //Aluminum 7075-T651
    double rho = 2810; //kg/m^3
    double E = 71.7e9; //Pa
    double nu = 0.33;
    double G = E / (2 + (1 + nu));
    double k1 = 10 * (1 + nu) / (12 + 11 * nu);  // Timoshenko shear correction coefficient for a rectangular cross-section
    double k2 = k1;                              // Timoshenko shear correction coefficient for a rectangular cross-section

    auto material = chrono_types::make_shared<ChMaterialBeamANCF>(rho, E, nu, k1, k2);


    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    m_system->Add(mesh);

    //Setup visualization - Need to fix this section
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

    // Populate the mesh container with a the nodes and elements for the meshed beam
    int num_nodes = (2 * num_elements) + 1;
    double dx = length / (num_nodes - 1);

    //Rotate the cross section gradients to match the sign convention in the experiment
    ChVector<> dir1(0, cos(-beam_angle_rad), sin(-beam_angle_rad));
    ChVector<> dir2(0, -sin(-beam_angle_rad), cos(-beam_angle_rad));

    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0, 0.0), dir1, dir2);
    nodeA->SetFixed(true);
    mesh->AddNode(nodeA);

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(dx*(2 * i), 0, 0), dir1, dir2);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(dx*(2 * i - 1), 0, 0), dir1, dir2);
        mesh->AddNode(nodeB);
        mesh->AddNode(nodeC);

        auto element = chrono_types::make_shared<ChElementBeamANCF>();
        element->SetNodes(nodeA, nodeB, nodeC);
        element->SetDimensions(2 * dx, thickness, width);
        element->SetMaterial(material);
        element->SetAlphaDamp(0.0);
        element->SetGravityOn(true);  //Enable the efficient ANCF method for calculating the application of gravity to the element
        element->SetStrainFormulation(ChElementBeamANCF::StrainFormulation::CMPoisson);
        //element->SetStrainFormulation(ChElementBeamANCF::StrainFormulation::CMNoPoisson);
        mesh->AddElement(element);

        nodeA = nodeB;
    }

    mesh->SetAutomaticGravity(false); //Turn off the default method for applying gravity to the mesh since it is less efficient for ANCF elements

    nodeA->SetForce(ChVector<>(0, 0, -vert_tip_load_N));
#endif

}

template <int N>
void ANCFbeam<N>::SimulateVis() {
#ifdef CHRONO_IRRLICHT
    irrlicht::ChIrrApp application(m_system, L"ANCF beams", irr::core::dimension2d<irr::u32>(800, 600), false, true);
    application.AddTypicalLogo();
    application.AddTypicalSky();
    application.AddTypicalLights();
    application.AddTypicalCamera(irr::core::vector3df(-0.2f, 0.2f, 0.2f), irr::core::vector3df(0, 0, 0));

    application.AssetBindAll();
    application.AssetUpdateAll();

    while (application.GetDevice()->run()) {
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

// =============================================================================

#define NUM_SKIP_STEPS 10  // number of steps for hot start
#define NUM_SIM_STEPS 100  // number of simulation steps for each benchmark

//CH_BM_SIMULATION_LOOP(ANCFbeam08_MINRES, ANCFbeam_MINRES<8>, NUM_SKIP_STEPS, NUM_SIM_STEPS, 10);
//CH_BM_SIMULATION_LOOP(ANCFbeam16_MINRES, ANCFbeam_MINRES<16>, NUM_SKIP_STEPS, NUM_SIM_STEPS, 10);
//CH_BM_SIMULATION_LOOP(ANCFbeam32_MINRES, ANCFbeam_MINRES<32>, NUM_SKIP_STEPS, NUM_SIM_STEPS, 10);
//CH_BM_SIMULATION_LOOP(ANCFbeam64_MINRES, ANCFbeam_MINRES<64>, NUM_SKIP_STEPS, NUM_SIM_STEPS, 10);

//#ifdef CHRONO_MKL
//CH_BM_SIMULATION_LOOP(ANCFbeam08_MKL, ANCFbeam_MKL<8>, NUM_SKIP_STEPS, NUM_SIM_STEPS, 10);
//CH_BM_SIMULATION_LOOP(ANCFbeam16_MKL, ANCFbeam_MKL<16>, NUM_SKIP_STEPS, NUM_SIM_STEPS, 10);
//CH_BM_SIMULATION_LOOP(ANCFbeam32_MKL, ANCFbeam_MKL<32>, NUM_SKIP_STEPS, NUM_SIM_STEPS, 10);
//CH_BM_SIMULATION_LOOP(ANCFbeam64_MKL, ANCFbeam_MKL<64>, NUM_SKIP_STEPS, NUM_SIM_STEPS, 10);
//#endif

//#ifdef CHRONO_MUMPS
//CH_BM_SIMULATION_LOOP(ANCFbeam08_MUMPS, ANCFbeam_MUMPS<8>, NUM_SKIP_STEPS, NUM_SIM_STEPS, 10);
//CH_BM_SIMULATION_LOOP(ANCFbeam16_MUMPS, ANCFbeam_MUMPS<16>, NUM_SKIP_STEPS, NUM_SIM_STEPS, 10);
//CH_BM_SIMULATION_LOOP(ANCFbeam32_MUMPS, ANCFbeam_MUMPS<32>, NUM_SKIP_STEPS, NUM_SIM_STEPS, 10);
//CH_BM_SIMULATION_LOOP(ANCFbeam64_MUMPS, ANCFbeam_MUMPS<64>, NUM_SKIP_STEPS, NUM_SIM_STEPS, 10);
//#endif

// =============================================================================

int main(int argc, char* argv[]) {
    ::benchmark::Initialize(&argc, argv);

#ifdef CHRONO_IRRLICHT
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        ANCFbeam_MUMPS<64> test;
        test.SimulateVis();
        return 0;
    }
#endif

    ::benchmark::RunSpecifiedBenchmarks();
}

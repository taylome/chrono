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
// Small Displacement, Small Deformation, Linear Isotropic Benchmark test for
// ANCF beam elements - Square cantilevered beam with a time-dependent tip load
//
// García-Vallejo, D., Mayo, J., Escalona, J. L., & Dominguez, J. (2004).
// Efficient evaluation of the elastic forces and the Jacobian in the absolute
// nodal coordinate formulation. Nonlinear Dynamics, 35(4), 313-329.
//
// =============================================================================

//#include "mkl.h"

#include "chrono/ChConfig.h"
#include "chrono/utils/ChBenchmark.h"

#include "chrono/physics/ChSystemSMC.h"
#include "chrono/solver/ChIterativeSolverLS.h"

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
#include "chrono/fea/ChElementBeamANCF_MT30.h"
#include "chrono/fea/ChElementBeamANCF_MT31.h"
#include "chrono/fea/ChElementBeamANCF_MT32.h"

#include "chrono/fea/ChMesh.h"
#include "chrono/fea/ChVisualizationFEAmesh.h"
#include "chrono/physics/ChLoadContainer.h"
#include "chrono/fea/ChLoadsBeam.h"

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

enum class SolverType { MINRES, MKL, MUMPS, SparseLU, SparseQR };

template <int num_elements, SolverType solver_type, typename ElementVersion, typename MaterialVersion>
class ANCFBeamTest : public utils::ChBenchmarkTest {
  public:
    ANCFBeamTest();

    ~ANCFBeamTest() { delete m_system; }

    ChSystem* GetSystem() override { return m_system; }
    void ExecuteStep() override { m_system->DoStepDynamics(1e-2); }

    void SimulateVis();
    void NonlinearStatics() { m_system->DoStaticNonlinear(50); }
    ChVector<> GetBeamEndPointPos() { return m_nodeEndPoint->GetPos(); }

  protected:
    ChSystemSMC* m_system;
    std::shared_ptr<ChNodeFEAxyzDD> m_nodeEndPoint;
};

template <int num_elements, SolverType solver_type, typename ElementVersion, typename MaterialVersion>
ANCFBeamTest<num_elements, solver_type, ElementVersion, MaterialVersion>::ANCFBeamTest() {
    SolverType useSolverType = solver_type;
    m_system = new ChSystemSMC();
    m_system->Set_G_acc(ChVector<>(0, 0, -9.80665));

    // Set solver parameters
#ifndef CHRONO_MKL
    if (useSolverType == SolverType::MKL) {
        useSolverType = SolverType::MINRES;
        std::cout << "WARNING! Chrono::MKL not enabled. Forcing use of MINRES solver" << std::endl;
    }
#endif

#ifndef CHRONO_MUMPS
    if (useSolverType == SolverType::MUMPS) {
        useSolverType = SolverType::MINRES;
        std::cout << "WARNING! Chrono::MUMPS not enabled. Forcing use of MINRES solver" << std::endl;
    }
#endif

    switch (useSolverType) {
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
    // integrator->SetAbsTolerances(1e-5);
    integrator->SetAbsTolerances(1e-3);
    integrator->SetMode(ChTimestepperHHT::POSITION);
    integrator->SetScaling(true);
    integrator->SetVerbose(false);
    integrator->SetModifiedNewton(true);

    // Mesh properties
    double length = 5;       // m
    double width = 0.1;      // m
    double thickness = 0.1;  // m
    double rho = 8245.2;     // kg/m^3
    double E = 132e9;        // Pa
    double nu = 0;           // Poisson effect neglected for this model
    double G = E / (2 * (1 + nu));
    double k1 =
        10 * (1 + nu) / (12 + 11 * nu);  // Timoshenko shear correction coefficient for a rectangular cross-section
    double k2 = k1;                      // Timoshenko shear correction coefficient for a rectangular cross-section

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

    // Create the first node and fix it completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0, 0.0), dir1, dir2);
    nodeA->SetFixed(true);
    mesh->AddNode(nodeA);

    auto elementlast = chrono_types::make_shared<ElementVersion>();

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(dx * (2 * i), 0, 0), dir1, dir2);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(dx * (2 * i - 1), 0, 0), dir1, dir2);
        mesh->AddNode(nodeB);
        mesh->AddNode(nodeC);

        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC);
        element->SetDimensions(2 * dx, thickness, width);
        element->SetMaterial(material);
        element->SetAlphaDamp(0.0);
        element->SetGravityOn(
            false);  // Enable the efficient ANCF method for calculating the application of gravity to the element
        element->SetStrainFormulation(ElementVersion::StrainFormulation::CMPoisson);
        // element->SetStrainFormulation(ElementVersion::StrainFormulation::CMNoPoisson);
        mesh->AddElement(element);

        nodeA = nodeB;
        elementlast = element;
    }

    m_nodeEndPoint = nodeA;

    mesh->SetAutomaticGravity(false);  // Turn off the default method for applying gravity to the mesh since it is less
                                       // efficient for ANCF elements

    // Create a custom atomic (point) load

    class MyLoaderTimeDependentTipLoad : public ChLoaderUatomic {
      public:
        // Useful: a constructor that also sets ChLoadable
        MyLoaderTimeDependentTipLoad(std::shared_ptr<ChLoadableU> mloadable) : ChLoaderUatomic(mloadable) {}

        // Compute F=F(u), the load at U. The load is a 6-row vector, i.e.
        // a wrench: forceX, forceY, forceZ, torqueX, torqueY, torqueZ.
        virtual void ComputeF(
            const double U,              ///< normalized position along the beam axis [-1...1]
            ChVectorDynamic<>& F,        ///< Load at U
            ChVectorDynamic<>* state_x,  ///< if != 0, update state (pos. part) to this, then evaluate F
            ChVectorDynamic<>* state_w   ///< if != 0, update state (speed part) to this, then evaluate F
            ) override {
            assert(auxsystem);
            double T = auxsystem->GetChTime();
            double Fmax = -300;
            double tc = 3.5;
            double Fz = Fmax;
            if (T < tc) {
                Fz = 0.5 * Fmax * (1 - cos(CH_C_PI * T / tc));
            }

            F.setZero();
            F(2) = Fz;  // Apply the force along the global Z axis
        }

      public:
        // add auxiliary data to the class, if you need to access it during ComputeF().
        ChSystem* auxsystem;
    };

    // Create the load container and to the current system
    auto loadcontainer = chrono_types::make_shared<ChLoadContainer>();
    m_system->Add(loadcontainer);

    // Create a custom load that uses the custom loader above.
    // The ChLoad is a 'manager' for your ChLoader.
    // It is created using templates, that is instancing a ChLoad<my_loader_class>()

    std::shared_ptr<ChLoad<MyLoaderTimeDependentTipLoad> > mload(new ChLoad<MyLoaderTimeDependentTipLoad>(elementlast));
    mload->loader.auxsystem = m_system;  // initialize auxiliary data of the loader, if needed
    mload->loader.SetApplication(1.0);   // specify application point
    loadcontainer->Add(mload);           // add the load to the load container.
}

template <int num_elements, SolverType solver_type, typename ElementVersion, typename MaterialVersion>
void ANCFBeamTest<num_elements, solver_type, ElementVersion, MaterialVersion>::SimulateVis() {
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

// =============================================================================

#define NUM_SKIP_STEPS 10  // number of steps for hot start
#define NUM_SIM_STEPS 100  // number of simulation steps for each benchmark
#define REPEATS 10

//// NOTE: trick to prevent errors in expanding macros due to types that contain a comma.
// typedef ANCFBeamTest<8, SolverType::MINRES, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_008_MINRES_Org;
// typedef ANCFBeamTest<16, SolverType::MINRES, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_016_MINRES_Org;
// typedef ANCFBeamTest<32, SolverType::MINRES, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_032_MINRES_Org;
// typedef ANCFBeamTest<64, SolverType::MINRES, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_064_MINRES_Org;
// CH_BM_SIMULATION_LOOP(ANCFBeamTest_008_MINRES_Org, ANCFBeamTest_008_MINRES_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS,
// REPEATS); CH_BM_SIMULATION_LOOP(ANCFBeamTest_016_MINRES_Org, ANCFBeamTest_016_MINRES_Org, NUM_SKIP_STEPS,
// NUM_SIM_STEPS, REPEATS); CH_BM_SIMULATION_LOOP(ANCFBeamTest_032_MINRES_Org, ANCFBeamTest_032_MINRES_Org,
// NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); CH_BM_SIMULATION_LOOP(ANCFBeamTest_064_MINRES_Org,
// ANCFBeamTest_064_MINRES_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);

#ifdef CHRONO_MKL
//typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_008_MKL_Org;
//typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_016_MKL_Org;
//typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_032_MKL_Org;
//typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_064_MKL_Org;
//typedef ANCFBeamTest<128, SolverType::MKL, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_128_MKL_Org;
//typedef ANCFBeamTest<256, SolverType::MKL, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_256_MKL_Org;
//typedef ANCFBeamTest<512, SolverType::MKL, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_512_MKL_Org;
//typedef ANCFBeamTest<1024, SolverType::MKL, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_1024_MKL_Org;
//CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_Org, ANCFBeamTest_008_MKL_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_Org, ANCFBeamTest_016_MKL_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_Org, ANCFBeamTest_032_MKL_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_Org, ANCFBeamTest_064_MKL_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_128_MKL_Org, ANCFBeamTest_128_MKL_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_256_MKL_Org, ANCFBeamTest_256_MKL_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_512_MKL_Org, ANCFBeamTest_512_MKL_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_1024_MKL_Org, ANCFBeamTest_1024_MKL_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//typedef ANCFBeamTest<8, SolverType::SparseLU, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_008_SparseLU_Org;
//typedef ANCFBeamTest<16, SolverType::SparseLU, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_016_SparseLU_Org;
//typedef ANCFBeamTest<32, SolverType::SparseLU, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_032_SparseLU_Org;
//typedef ANCFBeamTest<64, SolverType::SparseLU, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_064_SparseLU_Org;
//typedef ANCFBeamTest<128, SolverType::SparseLU, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_128_SparseLU_Org;
//typedef ANCFBeamTest<256, SolverType::SparseLU, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_256_SparseLU_Org;
//typedef ANCFBeamTest<512, SolverType::SparseLU, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_512_SparseLU_Org;
//typedef ANCFBeamTest<1024, SolverType::SparseLU, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_1024_SparseLU_Org;
//CH_BM_SIMULATION_LOOP(ANCFBeam_008_SparseLU_Org, ANCFBeamTest_008_SparseLU_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_016_SparseLU_Org, ANCFBeamTest_016_SparseLU_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_032_SparseLU_Org, ANCFBeamTest_032_SparseLU_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_064_SparseLU_Org, ANCFBeamTest_064_SparseLU_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_128_SparseLU_Org, ANCFBeamTest_128_SparseLU_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_256_SparseLU_Org, ANCFBeamTest_256_SparseLU_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_512_SparseLU_Org, ANCFBeamTest_512_SparseLU_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_1024_SparseLU_Org, ANCFBeamTest_1024_SparseLU_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//typedef ANCFBeamTest<8, SolverType::SparseQR, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_008_SparseQR_Org;
//typedef ANCFBeamTest<16, SolverType::SparseQR, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_016_SparseQR_Org;
//typedef ANCFBeamTest<32, SolverType::SparseQR, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_032_SparseQR_Org;
//typedef ANCFBeamTest<64, SolverType::SparseQR, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_064_SparseQR_Org;
//typedef ANCFBeamTest<128, SolverType::SparseQR, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_128_SparseQR_Org;
//typedef ANCFBeamTest<256, SolverType::SparseQR, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_256_SparseQR_Org;
//typedef ANCFBeamTest<512, SolverType::SparseQR, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_512_SparseQR_Org;
//typedef ANCFBeamTest<1024, SolverType::SparseQR, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_1024_SparseQR_Org;
//CH_BM_SIMULATION_LOOP(ANCFBeam_008_SparseQR_Org, ANCFBeamTest_008_SparseQR_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_016_SparseQR_Org, ANCFBeamTest_016_SparseQR_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_032_SparseQR_Org, ANCFBeamTest_032_SparseQR_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_064_SparseQR_Org, ANCFBeamTest_064_SparseQR_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_128_SparseQR_Org, ANCFBeamTest_128_SparseQR_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_256_SparseQR_Org, ANCFBeamTest_256_SparseQR_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_512_SparseQR_Org, ANCFBeamTest_512_SparseQR_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_1024_SparseQR_Org, ANCFBeamTest_1024_SparseQR_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);

//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT01, ChMaterialBeamANCF_MT01> ANCFBeamTest_008_MKL_MT01;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT01, ChMaterialBeamANCF_MT01> ANCFBeamTest_016_MKL_MT01;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT01, ChMaterialBeamANCF_MT01> ANCFBeamTest_032_MKL_MT01;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT01, ChMaterialBeamANCF_MT01> ANCFBeamTest_064_MKL_MT01;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT01, ANCFBeamTest_008_MKL_MT01, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT01, ANCFBeamTest_016_MKL_MT01, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT01, ANCFBeamTest_032_MKL_MT01, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT01, ANCFBeamTest_064_MKL_MT01, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT02, ChMaterialBeamANCF_MT02> ANCFBeamTest_008_MKL_MT02;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT02, ChMaterialBeamANCF_MT02> ANCFBeamTest_016_MKL_MT02;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT02, ChMaterialBeamANCF_MT02> ANCFBeamTest_032_MKL_MT02;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT02, ChMaterialBeamANCF_MT02> ANCFBeamTest_064_MKL_MT02;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT02, ANCFBeamTest_008_MKL_MT02, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT02, ANCFBeamTest_016_MKL_MT02, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT02, ANCFBeamTest_032_MKL_MT02, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT02, ANCFBeamTest_064_MKL_MT02, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT03, ChMaterialBeamANCF_MT03> ANCFBeamTest_008_MKL_MT03;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT03, ChMaterialBeamANCF_MT03> ANCFBeamTest_016_MKL_MT03;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT03, ChMaterialBeamANCF_MT03> ANCFBeamTest_032_MKL_MT03;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT03, ChMaterialBeamANCF_MT03> ANCFBeamTest_064_MKL_MT03;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT03, ANCFBeamTest_008_MKL_MT03, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT03, ANCFBeamTest_016_MKL_MT03, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT03, ANCFBeamTest_032_MKL_MT03, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT03, ANCFBeamTest_064_MKL_MT03, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT04, ChMaterialBeamANCF_MT04> ANCFBeamTest_008_MKL_MT04;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT04, ChMaterialBeamANCF_MT04> ANCFBeamTest_016_MKL_MT04;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT04, ChMaterialBeamANCF_MT04> ANCFBeamTest_032_MKL_MT04;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT04, ChMaterialBeamANCF_MT04> ANCFBeamTest_064_MKL_MT04;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT04, ANCFBeamTest_008_MKL_MT04, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT04, ANCFBeamTest_016_MKL_MT04, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT04, ANCFBeamTest_032_MKL_MT04, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT04, ANCFBeamTest_064_MKL_MT04, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT05, ChMaterialBeamANCF_MT05> ANCFBeamTest_008_MKL_MT05;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT05, ChMaterialBeamANCF_MT05> ANCFBeamTest_016_MKL_MT05;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT05, ChMaterialBeamANCF_MT05> ANCFBeamTest_032_MKL_MT05;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT05, ChMaterialBeamANCF_MT05> ANCFBeamTest_064_MKL_MT05;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT05, ANCFBeamTest_008_MKL_MT05, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT05, ANCFBeamTest_016_MKL_MT05, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT05, ANCFBeamTest_032_MKL_MT05, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT05, ANCFBeamTest_064_MKL_MT05, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT06, ChMaterialBeamANCF_MT06> ANCFBeamTest_008_MKL_MT06;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT06, ChMaterialBeamANCF_MT06> ANCFBeamTest_016_MKL_MT06;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT06, ChMaterialBeamANCF_MT06> ANCFBeamTest_032_MKL_MT06;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT06, ChMaterialBeamANCF_MT06> ANCFBeamTest_064_MKL_MT06;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT06, ANCFBeamTest_008_MKL_MT06, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT06, ANCFBeamTest_016_MKL_MT06, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT06, ANCFBeamTest_032_MKL_MT06, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT06, ANCFBeamTest_064_MKL_MT06, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT07, ChMaterialBeamANCF_MT07> ANCFBeamTest_008_MKL_MT07;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT07, ChMaterialBeamANCF_MT07> ANCFBeamTest_016_MKL_MT07;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT07, ChMaterialBeamANCF_MT07> ANCFBeamTest_032_MKL_MT07;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT07, ChMaterialBeamANCF_MT07> ANCFBeamTest_064_MKL_MT07;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT07, ANCFBeamTest_008_MKL_MT07, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT07, ANCFBeamTest_016_MKL_MT07, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT07, ANCFBeamTest_032_MKL_MT07, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT07, ANCFBeamTest_064_MKL_MT07, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT08, ChMaterialBeamANCF_MT08> ANCFBeamTest_008_MKL_MT08;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT08, ChMaterialBeamANCF_MT08> ANCFBeamTest_016_MKL_MT08;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT08, ChMaterialBeamANCF_MT08> ANCFBeamTest_032_MKL_MT08;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT08, ChMaterialBeamANCF_MT08> ANCFBeamTest_064_MKL_MT08;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT08, ANCFBeamTest_008_MKL_MT08, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT08, ANCFBeamTest_016_MKL_MT08, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT08, ANCFBeamTest_032_MKL_MT08, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT08, ANCFBeamTest_064_MKL_MT08, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT09, ChMaterialBeamANCF_MT09> ANCFBeamTest_008_MKL_MT09;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT09, ChMaterialBeamANCF_MT09> ANCFBeamTest_016_MKL_MT09;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT09, ChMaterialBeamANCF_MT09> ANCFBeamTest_032_MKL_MT09;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT09, ChMaterialBeamANCF_MT09> ANCFBeamTest_064_MKL_MT09;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT09, ANCFBeamTest_008_MKL_MT09, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT09, ANCFBeamTest_016_MKL_MT09, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT09, ANCFBeamTest_032_MKL_MT09, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT09, ANCFBeamTest_064_MKL_MT09, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT10, ChMaterialBeamANCF_MT10> ANCFBeamTest_008_MKL_MT10;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT10, ChMaterialBeamANCF_MT10> ANCFBeamTest_016_MKL_MT10;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT10, ChMaterialBeamANCF_MT10> ANCFBeamTest_032_MKL_MT10;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT10, ChMaterialBeamANCF_MT10> ANCFBeamTest_064_MKL_MT10;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT10, ANCFBeamTest_008_MKL_MT10, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT10, ANCFBeamTest_016_MKL_MT10, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT10, ANCFBeamTest_032_MKL_MT10, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT10, ANCFBeamTest_064_MKL_MT10, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT11, ChMaterialBeamANCF_MT11> ANCFBeamTest_008_MKL_MT11;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT11, ChMaterialBeamANCF_MT11> ANCFBeamTest_016_MKL_MT11;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT11, ChMaterialBeamANCF_MT11> ANCFBeamTest_032_MKL_MT11;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT11, ChMaterialBeamANCF_MT11> ANCFBeamTest_064_MKL_MT11;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT11, ANCFBeamTest_008_MKL_MT11, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT11, ANCFBeamTest_016_MKL_MT11, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT11, ANCFBeamTest_032_MKL_MT11, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT11, ANCFBeamTest_064_MKL_MT11, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT12, ChMaterialBeamANCF_MT12> ANCFBeamTest_008_MKL_MT12;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT12, ChMaterialBeamANCF_MT12> ANCFBeamTest_016_MKL_MT12;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT12, ChMaterialBeamANCF_MT12> ANCFBeamTest_032_MKL_MT12;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT12, ChMaterialBeamANCF_MT12> ANCFBeamTest_064_MKL_MT12;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT12, ANCFBeamTest_008_MKL_MT12, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT12, ANCFBeamTest_016_MKL_MT12, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT12, ANCFBeamTest_032_MKL_MT12, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT12, ANCFBeamTest_064_MKL_MT12, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT13, ChMaterialBeamANCF_MT13> ANCFBeamTest_008_MKL_MT13;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT13, ChMaterialBeamANCF_MT13> ANCFBeamTest_016_MKL_MT13;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT13, ChMaterialBeamANCF_MT13> ANCFBeamTest_032_MKL_MT13;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT13, ChMaterialBeamANCF_MT13> ANCFBeamTest_064_MKL_MT13;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT13, ANCFBeamTest_008_MKL_MT13, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT13, ANCFBeamTest_016_MKL_MT13, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT13, ANCFBeamTest_032_MKL_MT13, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT13, ANCFBeamTest_064_MKL_MT13, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT14, ChMaterialBeamANCF_MT14> ANCFBeamTest_008_MKL_MT14;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT14, ChMaterialBeamANCF_MT14> ANCFBeamTest_016_MKL_MT14;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT14, ChMaterialBeamANCF_MT14> ANCFBeamTest_032_MKL_MT14;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT14, ChMaterialBeamANCF_MT14> ANCFBeamTest_064_MKL_MT14;
// typedef ANCFBeamTest<128, SolverType::MKL, ChElementBeamANCF_MT14, ChMaterialBeamANCF_MT14> ANCFBeamTest_128_MKL_MT14; 
// typedef ANCFBeamTest<256, SolverType::MKL, ChElementBeamANCF_MT14, ChMaterialBeamANCF_MT14> ANCFBeamTest_256_MKL_MT14; 
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT14, ANCFBeamTest_008_MKL_MT14, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT14, ANCFBeamTest_016_MKL_MT14, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT14, ANCFBeamTest_032_MKL_MT14, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT14, ANCFBeamTest_064_MKL_MT14, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_128_MKL_MT14, ANCFBeamTest_128_MKL_MT14, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_256_MKL_MT14, ANCFBeamTest_256_MKL_MT14, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);

// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT15, ChMaterialBeamANCF_MT15> ANCFBeamTest_008_MKL_MT15;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT15, ChMaterialBeamANCF_MT15> ANCFBeamTest_016_MKL_MT15;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT15, ChMaterialBeamANCF_MT15> ANCFBeamTest_032_MKL_MT15;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT15, ChMaterialBeamANCF_MT15> ANCFBeamTest_064_MKL_MT15;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT15, ANCFBeamTest_008_MKL_MT15, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT15, ANCFBeamTest_016_MKL_MT15, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT15, ANCFBeamTest_032_MKL_MT15, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT15, ANCFBeamTest_064_MKL_MT15, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT16, ChMaterialBeamANCF_MT16> ANCFBeamTest_008_MKL_MT16;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT16, ChMaterialBeamANCF_MT16> ANCFBeamTest_016_MKL_MT16;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT16, ChMaterialBeamANCF_MT16> ANCFBeamTest_032_MKL_MT16;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT16, ChMaterialBeamANCF_MT16> ANCFBeamTest_064_MKL_MT16;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT16, ANCFBeamTest_008_MKL_MT16, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT16, ANCFBeamTest_016_MKL_MT16, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT16, ANCFBeamTest_032_MKL_MT16, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT16, ANCFBeamTest_064_MKL_MT16, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT17, ChMaterialBeamANCF_MT17> ANCFBeamTest_008_MKL_MT17;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT17, ChMaterialBeamANCF_MT17> ANCFBeamTest_016_MKL_MT17;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT17, ChMaterialBeamANCF_MT17> ANCFBeamTest_032_MKL_MT17;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT17, ChMaterialBeamANCF_MT17> ANCFBeamTest_064_MKL_MT17;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT17, ANCFBeamTest_008_MKL_MT17, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT17, ANCFBeamTest_016_MKL_MT17, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT17, ANCFBeamTest_032_MKL_MT17, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT17, ANCFBeamTest_064_MKL_MT17, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT18, ChMaterialBeamANCF_MT18> ANCFBeamTest_008_MKL_MT18;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT18, ChMaterialBeamANCF_MT18> ANCFBeamTest_016_MKL_MT18;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT18, ChMaterialBeamANCF_MT18> ANCFBeamTest_032_MKL_MT18;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT18, ChMaterialBeamANCF_MT18> ANCFBeamTest_064_MKL_MT18;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT18, ANCFBeamTest_008_MKL_MT18, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT18, ANCFBeamTest_016_MKL_MT18, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT18, ANCFBeamTest_032_MKL_MT18, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT18, ANCFBeamTest_064_MKL_MT18, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT19, ChMaterialBeamANCF_MT19> ANCFBeamTest_008_MKL_MT19;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT19, ChMaterialBeamANCF_MT19> ANCFBeamTest_016_MKL_MT19;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT19, ChMaterialBeamANCF_MT19> ANCFBeamTest_032_MKL_MT19;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT19, ChMaterialBeamANCF_MT19> ANCFBeamTest_064_MKL_MT19;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT19, ANCFBeamTest_008_MKL_MT19, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT19, ANCFBeamTest_016_MKL_MT19, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT19, ANCFBeamTest_032_MKL_MT19, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT19, ANCFBeamTest_064_MKL_MT19, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT20, ChMaterialBeamANCF_MT20> ANCFBeamTest_008_MKL_MT20;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT20, ChMaterialBeamANCF_MT20> ANCFBeamTest_016_MKL_MT20;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT20, ChMaterialBeamANCF_MT20> ANCFBeamTest_032_MKL_MT20;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT20, ChMaterialBeamANCF_MT20> ANCFBeamTest_064_MKL_MT20;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT20, ANCFBeamTest_008_MKL_MT20, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT20, ANCFBeamTest_016_MKL_MT20, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT20, ANCFBeamTest_032_MKL_MT20, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT20, ANCFBeamTest_064_MKL_MT20, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT21, ChMaterialBeamANCF_MT21> ANCFBeamTest_008_MKL_MT21;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT21, ChMaterialBeamANCF_MT21> ANCFBeamTest_016_MKL_MT21;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT21, ChMaterialBeamANCF_MT21> ANCFBeamTest_032_MKL_MT21;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT21, ChMaterialBeamANCF_MT21> ANCFBeamTest_064_MKL_MT21;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT21, ANCFBeamTest_008_MKL_MT21, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT21, ANCFBeamTest_016_MKL_MT21, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT21, ANCFBeamTest_032_MKL_MT21, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT21, ANCFBeamTest_064_MKL_MT21, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT22, ChMaterialBeamANCF_MT22> ANCFBeamTest_008_MKL_MT22;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT22, ChMaterialBeamANCF_MT22> ANCFBeamTest_016_MKL_MT22;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT22, ChMaterialBeamANCF_MT22> ANCFBeamTest_032_MKL_MT22;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT22, ChMaterialBeamANCF_MT22> ANCFBeamTest_064_MKL_MT22;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT22, ANCFBeamTest_008_MKL_MT22, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT22, ANCFBeamTest_016_MKL_MT22, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT22, ANCFBeamTest_032_MKL_MT22, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT22, ANCFBeamTest_064_MKL_MT22, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT23, ChMaterialBeamANCF_MT23> ANCFBeamTest_008_MKL_MT23;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT23, ChMaterialBeamANCF_MT23> ANCFBeamTest_016_MKL_MT23;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT23, ChMaterialBeamANCF_MT23> ANCFBeamTest_032_MKL_MT23;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT23, ChMaterialBeamANCF_MT23> ANCFBeamTest_064_MKL_MT23;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT23, ANCFBeamTest_008_MKL_MT23, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT23, ANCFBeamTest_016_MKL_MT23, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT23, ANCFBeamTest_032_MKL_MT23, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT23, ANCFBeamTest_064_MKL_MT23, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT24, ChMaterialBeamANCF_MT24> ANCFBeamTest_008_MKL_MT24;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT24, ChMaterialBeamANCF_MT24> ANCFBeamTest_016_MKL_MT24;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT24, ChMaterialBeamANCF_MT24> ANCFBeamTest_032_MKL_MT24;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT24, ChMaterialBeamANCF_MT24> ANCFBeamTest_064_MKL_MT24;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT24, ANCFBeamTest_008_MKL_MT24, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT24, ANCFBeamTest_016_MKL_MT24, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT24, ANCFBeamTest_032_MKL_MT24, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT24, ANCFBeamTest_064_MKL_MT24, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT25, ChMaterialBeamANCF_MT25> ANCFBeamTest_008_MKL_MT25;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT25, ChMaterialBeamANCF_MT25> ANCFBeamTest_016_MKL_MT25;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT25, ChMaterialBeamANCF_MT25> ANCFBeamTest_032_MKL_MT25;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT25, ChMaterialBeamANCF_MT25> ANCFBeamTest_064_MKL_MT25;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT25, ANCFBeamTest_008_MKL_MT25, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT25, ANCFBeamTest_016_MKL_MT25, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT25, ANCFBeamTest_032_MKL_MT25, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT25, ANCFBeamTest_064_MKL_MT25, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_008_MKL_MT26;
//typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_016_MKL_MT26;
//typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_032_MKL_MT26;
//typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_064_MKL_MT26;
//CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT26, ANCFBeamTest_008_MKL_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT26, ANCFBeamTest_016_MKL_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT26, ANCFBeamTest_032_MKL_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT26, ANCFBeamTest_064_MKL_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
//
//typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_008_MKL_MT26;
//typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_016_MKL_MT26;
//typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_032_MKL_MT26;
//typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_064_MKL_MT26;
//typedef ANCFBeamTest<128, SolverType::MKL, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_128_MKL_MT26;
//typedef ANCFBeamTest<256, SolverType::MKL, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_256_MKL_MT26;
//typedef ANCFBeamTest<512, SolverType::MKL, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_512_MKL_MT26;
//typedef ANCFBeamTest<1024, SolverType::MKL, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_1024_MKL_MT26;
//CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT26, ANCFBeamTest_008_MKL_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT26, ANCFBeamTest_016_MKL_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT26, ANCFBeamTest_032_MKL_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT26, ANCFBeamTest_064_MKL_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_128_MKL_MT26, ANCFBeamTest_128_MKL_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_256_MKL_MT26, ANCFBeamTest_256_MKL_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_512_MKL_MT26, ANCFBeamTest_512_MKL_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_1024_MKL_MT26, ANCFBeamTest_1024_MKL_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
//typedef ANCFBeamTest<8, SolverType::SparseLU, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_008_SparseLU_MT26; 
//typedef ANCFBeamTest<16, SolverType::SparseLU, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_016_SparseLU_MT26; 
//typedef ANCFBeamTest<32, SolverType::SparseLU, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_032_SparseLU_MT26; 
//typedef ANCFBeamTest<64, SolverType::SparseLU, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_064_SparseLU_MT26; 
//typedef ANCFBeamTest<128, SolverType::SparseLU, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_128_SparseLU_MT26; 
//typedef ANCFBeamTest<256, SolverType::SparseLU, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_256_SparseLU_MT26; 
//typedef ANCFBeamTest<512, SolverType::SparseLU, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_512_SparseLU_MT26; 
//typedef ANCFBeamTest<1024, SolverType::SparseLU, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_1024_SparseLU_MT26;
//CH_BM_SIMULATION_LOOP(ANCFBeam_008_SparseLU_MT26, ANCFBeamTest_008_SparseLU_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_016_SparseLU_MT26, ANCFBeamTest_016_SparseLU_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_032_SparseLU_MT26, ANCFBeamTest_032_SparseLU_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_064_SparseLU_MT26, ANCFBeamTest_064_SparseLU_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_128_SparseLU_MT26, ANCFBeamTest_128_SparseLU_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_256_SparseLU_MT26, ANCFBeamTest_256_SparseLU_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_512_SparseLU_MT26, ANCFBeamTest_512_SparseLU_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_1024_SparseLU_MT26, ANCFBeamTest_1024_SparseLU_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
//typedef ANCFBeamTest<8, SolverType::SparseQR, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_008_SparseQR_MT26; 
//typedef ANCFBeamTest<16, SolverType::SparseQR, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_016_SparseQR_MT26; 
//typedef ANCFBeamTest<32, SolverType::SparseQR, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_032_SparseQR_MT26; 
//typedef ANCFBeamTest<64, SolverType::SparseQR, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_064_SparseQR_MT26; 
//typedef ANCFBeamTest<128, SolverType::SparseQR, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_128_SparseQR_MT26; 
//typedef ANCFBeamTest<256, SolverType::SparseQR, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_256_SparseQR_MT26; 
//typedef ANCFBeamTest<512, SolverType::SparseQR, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_512_SparseQR_MT26; 
//typedef ANCFBeamTest<1024, SolverType::SparseQR, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_1024_SparseQR_MT26;
//CH_BM_SIMULATION_LOOP(ANCFBeam_008_SparseQR_MT26, ANCFBeamTest_008_SparseQR_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_016_SparseQR_MT26, ANCFBeamTest_016_SparseQR_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_032_SparseQR_MT26, ANCFBeamTest_032_SparseQR_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_064_SparseQR_MT26, ANCFBeamTest_064_SparseQR_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_128_SparseQR_MT26, ANCFBeamTest_128_SparseQR_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_256_SparseQR_MT26, ANCFBeamTest_256_SparseQR_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_512_SparseQR_MT26, ANCFBeamTest_512_SparseQR_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_1024_SparseQR_MT26, ANCFBeamTest_1024_SparseQR_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);


//typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT27, ChMaterialBeamANCF_MT27> ANCFBeamTest_008_MKL_MT27;
//typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT27, ChMaterialBeamANCF_MT27> ANCFBeamTest_016_MKL_MT27;
//typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT27, ChMaterialBeamANCF_MT27> ANCFBeamTest_032_MKL_MT27;
//typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT27, ChMaterialBeamANCF_MT27> ANCFBeamTest_064_MKL_MT27;
//CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT27, ANCFBeamTest_008_MKL_MT27, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT27, ANCFBeamTest_016_MKL_MT27, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT27, ANCFBeamTest_032_MKL_MT27, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT27, ANCFBeamTest_064_MKL_MT27, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
//typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT28, ChMaterialBeamANCF_MT28> ANCFBeamTest_008_MKL_MT28;
//typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT28, ChMaterialBeamANCF_MT28> ANCFBeamTest_016_MKL_MT28;
//typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT28, ChMaterialBeamANCF_MT28> ANCFBeamTest_032_MKL_MT28;
//typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT28, ChMaterialBeamANCF_MT28> ANCFBeamTest_064_MKL_MT28;
//CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT28, ANCFBeamTest_008_MKL_MT28, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT28, ANCFBeamTest_016_MKL_MT28, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT28, ANCFBeamTest_032_MKL_MT28, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT28, ANCFBeamTest_064_MKL_MT28, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);

//
//
// typedef ANCFBeamTest<8, SolverType::MINRES, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26>
// ANCFBeamTest_008_MINRES_MT26; typedef ANCFBeamTest<16, SolverType::MINRES, ChElementBeamANCF_MT26,
// ChMaterialBeamANCF_MT26> ANCFBeamTest_016_MINRES_MT26; typedef ANCFBeamTest<32, SolverType::MINRES,
// ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_032_MINRES_MT26; typedef ANCFBeamTest<64,
// SolverType::MINRES, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_064_MINRES_MT26;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MINRES_MT26, ANCFBeamTest_008_MINRES_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS,
// REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MINRES_MT26, ANCFBeamTest_016_MINRES_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS,
///REPEATS); /CH_BM_SIMULATION_LOOP(ANCFBeam_032_MINRES_MT26, ANCFBeamTest_032_MINRES_MT26, NUM_SKIP_STEPS,
///NUM_SIM_STEPS, REPEATS); /CH_BM_SIMULATION_LOOP(ANCFBeam_064_MINRES_MT26, ANCFBeamTest_064_MINRES_MT26,
///NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);

//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT30, ChMaterialBeamANCF_MT30> ANCFBeamTest_008_MKL_MT30;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT30, ChMaterialBeamANCF_MT30> ANCFBeamTest_016_MKL_MT30;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT30, ChMaterialBeamANCF_MT30> ANCFBeamTest_032_MKL_MT30;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT30, ChMaterialBeamANCF_MT30> ANCFBeamTest_064_MKL_MT30;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT30, ANCFBeamTest_008_MKL_MT30, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT30, ANCFBeamTest_016_MKL_MT30, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT30, ANCFBeamTest_032_MKL_MT30, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT30, ANCFBeamTest_064_MKL_MT30, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);


//typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_008_MKL_MT31;
//typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_016_MKL_MT31;
//typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_032_MKL_MT31;
//typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_064_MKL_MT31;
//typedef ANCFBeamTest<128, SolverType::MKL, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_128_MKL_MT31;
//typedef ANCFBeamTest<256, SolverType::MKL, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_256_MKL_MT31;
//typedef ANCFBeamTest<512, SolverType::MKL, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_512_MKL_MT31;
//typedef ANCFBeamTest<1024, SolverType::MKL, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_1024_MKL_MT31; 
//CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT31, ANCFBeamTest_008_MKL_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT31, ANCFBeamTest_016_MKL_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT31, ANCFBeamTest_032_MKL_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT31, ANCFBeamTest_064_MKL_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
////CH_BM_SIMULATION_LOOP(ANCFBeam_128_MKL_MT31, ANCFBeamTest_128_MKL_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
////CH_BM_SIMULATION_LOOP(ANCFBeam_256_MKL_MT31, ANCFBeamTest_256_MKL_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
////CH_BM_SIMULATION_LOOP(ANCFBeam_512_MKL_MT31, ANCFBeamTest_512_MKL_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
////CH_BM_SIMULATION_LOOP(ANCFBeam_1024_MKL_MT31, ANCFBeamTest_1024_MKL_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
//typedef ANCFBeamTest<8, SolverType::SparseLU, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_008_SparseLU_MT31;
//typedef ANCFBeamTest<16, SolverType::SparseLU, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_016_SparseLU_MT31;
//typedef ANCFBeamTest<32, SolverType::SparseLU, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_032_SparseLU_MT31;
//typedef ANCFBeamTest<64, SolverType::SparseLU, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_064_SparseLU_MT31;
//typedef ANCFBeamTest<128, SolverType::SparseLU, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_128_SparseLU_MT31;
//typedef ANCFBeamTest<256, SolverType::SparseLU, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_256_SparseLU_MT31;
//typedef ANCFBeamTest<512, SolverType::SparseLU, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_512_SparseLU_MT31;
//typedef ANCFBeamTest<1024, SolverType::SparseLU, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_1024_SparseLU_MT31;
//CH_BM_SIMULATION_LOOP(ANCFBeam_008_SparseLU_MT31, ANCFBeamTest_008_SparseLU_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_016_SparseLU_MT31, ANCFBeamTest_016_SparseLU_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_032_SparseLU_MT31, ANCFBeamTest_032_SparseLU_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_064_SparseLU_MT31, ANCFBeamTest_064_SparseLU_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_128_SparseLU_MT31, ANCFBeamTest_128_SparseLU_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_256_SparseLU_MT31, ANCFBeamTest_256_SparseLU_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_512_SparseLU_MT31, ANCFBeamTest_512_SparseLU_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_1024_SparseLU_MT31, ANCFBeamTest_1024_SparseLU_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
//typedef ANCFBeamTest<8, SolverType::SparseQR, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_008_SparseQR_MT31;
//typedef ANCFBeamTest<16, SolverType::SparseQR, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_016_SparseQR_MT31;
//typedef ANCFBeamTest<32, SolverType::SparseQR, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_032_SparseQR_MT31;
//typedef ANCFBeamTest<64, SolverType::SparseQR, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_064_SparseQR_MT31;
//typedef ANCFBeamTest<128, SolverType::SparseQR, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_128_SparseQR_MT31;
//typedef ANCFBeamTest<256, SolverType::SparseQR, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_256_SparseQR_MT31;
//typedef ANCFBeamTest<512, SolverType::SparseQR, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_512_SparseQR_MT31;
//typedef ANCFBeamTest<1024, SolverType::SparseQR, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_1024_SparseQR_MT31;
//CH_BM_SIMULATION_LOOP(ANCFBeam_008_SparseQR_MT31, ANCFBeamTest_008_SparseQR_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_016_SparseQR_MT31, ANCFBeamTest_016_SparseQR_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_032_SparseQR_MT31, ANCFBeamTest_032_SparseQR_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_064_SparseQR_MT31, ANCFBeamTest_064_SparseQR_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_128_SparseQR_MT31, ANCFBeamTest_128_SparseQR_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_256_SparseQR_MT31, ANCFBeamTest_256_SparseQR_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_512_SparseQR_MT31, ANCFBeamTest_512_SparseQR_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_1024_SparseQR_MT31, ANCFBeamTest_1024_SparseQR_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);

//
//
//
// typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT32, ChMaterialBeamANCF_MT32> ANCFBeamTest_008_MKL_MT32;
// typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT32, ChMaterialBeamANCF_MT32> ANCFBeamTest_016_MKL_MT32;
// typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT32, ChMaterialBeamANCF_MT32> ANCFBeamTest_032_MKL_MT32;
// typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT32, ChMaterialBeamANCF_MT32> ANCFBeamTest_064_MKL_MT32;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT32, ANCFBeamTest_008_MKL_MT32, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT32, ANCFBeamTest_016_MKL_MT32, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT32, ANCFBeamTest_032_MKL_MT32, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT32, ANCFBeamTest_064_MKL_MT32, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
#endif

#ifdef CHRONO_MUMPS
 //typedef ANCFBeamTest<8, SolverType::MUMPS, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_008_MUMPS_Org;
 //typedef ANCFBeamTest<16, SolverType::MUMPS, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_016_MUMPS_Org;
 //typedef ANCFBeamTest<32, SolverType::MUMPS, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_032_MUMPS_Org;
 //typedef ANCFBeamTest<64, SolverType::MUMPS, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_064_MUMPS_Org;
 //CH_BM_SIMULATION_LOOP(ANCFBeam_008_MUMPS_Org, ANCFBeamTest_008_MUMPS_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MUMPS_Org, ANCFBeamTest_016_MUMPS_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MUMPS_Org, ANCFBeamTest_032_MUMPS_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MUMPS_Org, ANCFBeamTest_064_MUMPS_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MUMPS, ChElementBeamANCF_MT01, ChMaterialBeamANCF_MT01>
// ANCFBeamTest_008_MUMPS_MT01; typedef ANCFBeamTest<16, SolverType::MUMPS, ChElementBeamANCF_MT01,
// ChMaterialBeamANCF_MT01> ANCFBeamTest_016_MUMPS_MT01; typedef ANCFBeamTest<32, SolverType::MUMPS,
// ChElementBeamANCF_MT01, ChMaterialBeamANCF_MT01> ANCFBeamTest_032_MUMPS_MT01; typedef ANCFBeamTest<64,
// SolverType::MUMPS, ChElementBeamANCF_MT01, ChMaterialBeamANCF_MT01> ANCFBeamTest_064_MUMPS_MT01;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MUMPS_MT01, ANCFBeamTest_008_MUMPS_MT01, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MUMPS_MT01, ANCFBeamTest_016_MUMPS_MT01, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MUMPS_MT01, ANCFBeamTest_032_MUMPS_MT01, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MUMPS_MT01, ANCFBeamTest_064_MUMPS_MT01, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MUMPS, ChElementBeamANCF_MT02, ChMaterialBeamANCF_MT02>
// ANCFBeamTest_008_MUMPS_MT02; typedef ANCFBeamTest<16, SolverType::MUMPS, ChElementBeamANCF_MT02,
// ChMaterialBeamANCF_MT02> ANCFBeamTest_016_MUMPS_MT02; typedef ANCFBeamTest<32, SolverType::MUMPS,
// ChElementBeamANCF_MT02, ChMaterialBeamANCF_MT02> ANCFBeamTest_032_MUMPS_MT02; typedef ANCFBeamTest<64,
// SolverType::MUMPS, ChElementBeamANCF_MT02, ChMaterialBeamANCF_MT02> ANCFBeamTest_064_MUMPS_MT02;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MUMPS_MT02, ANCFBeamTest_008_MUMPS_MT02, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MUMPS_MT02, ANCFBeamTest_016_MUMPS_MT02, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MUMPS_MT02, ANCFBeamTest_032_MUMPS_MT02, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MUMPS_MT02, ANCFBeamTest_064_MUMPS_MT02, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MUMPS, ChElementBeamANCF_MT03, ChMaterialBeamANCF_MT03>
// ANCFBeamTest_008_MUMPS_MT03; typedef ANCFBeamTest<16, SolverType::MUMPS, ChElementBeamANCF_MT03,
// ChMaterialBeamANCF_MT03> ANCFBeamTest_016_MUMPS_MT03; typedef ANCFBeamTest<32, SolverType::MUMPS,
// ChElementBeamANCF_MT03, ChMaterialBeamANCF_MT03> ANCFBeamTest_032_MUMPS_MT03; typedef ANCFBeamTest<64,
// SolverType::MUMPS, ChElementBeamANCF_MT03, ChMaterialBeamANCF_MT03> ANCFBeamTest_064_MUMPS_MT03;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MUMPS_MT03, ANCFBeamTest_008_MUMPS_MT03, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MUMPS_MT03, ANCFBeamTest_016_MUMPS_MT03, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MUMPS_MT03, ANCFBeamTest_032_MUMPS_MT03, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MUMPS_MT03, ANCFBeamTest_064_MUMPS_MT03, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MUMPS, ChElementBeamANCF_MT04, ChMaterialBeamANCF_MT04>
// ANCFBeamTest_008_MUMPS_MT04; typedef ANCFBeamTest<16, SolverType::MUMPS, ChElementBeamANCF_MT04,
// ChMaterialBeamANCF_MT04> ANCFBeamTest_016_MUMPS_MT04; typedef ANCFBeamTest<32, SolverType::MUMPS,
// ChElementBeamANCF_MT04, ChMaterialBeamANCF_MT04> ANCFBeamTest_032_MUMPS_MT04; typedef ANCFBeamTest<64,
// SolverType::MUMPS, ChElementBeamANCF_MT04, ChMaterialBeamANCF_MT04> ANCFBeamTest_064_MUMPS_MT04;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MUMPS_MT04, ANCFBeamTest_008_MUMPS_MT04, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MUMPS_MT04, ANCFBeamTest_016_MUMPS_MT04, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MUMPS_MT04, ANCFBeamTest_032_MUMPS_MT04, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MUMPS_MT04, ANCFBeamTest_064_MUMPS_MT04, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MUMPS, ChElementBeamANCF_MT05, ChMaterialBeamANCF_MT05>
// ANCFBeamTest_008_MUMPS_MT05; typedef ANCFBeamTest<16, SolverType::MUMPS, ChElementBeamANCF_MT05,
// ChMaterialBeamANCF_MT05> ANCFBeamTest_016_MUMPS_MT05; typedef ANCFBeamTest<32, SolverType::MUMPS,
// ChElementBeamANCF_MT05, ChMaterialBeamANCF_MT05> ANCFBeamTest_032_MUMPS_MT05; typedef ANCFBeamTest<64,
// SolverType::MUMPS, ChElementBeamANCF_MT05, ChMaterialBeamANCF_MT05> ANCFBeamTest_064_MUMPS_MT05;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MUMPS_MT05, ANCFBeamTest_008_MUMPS_MT05, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MUMPS_MT05, ANCFBeamTest_016_MUMPS_MT05, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MUMPS_MT05, ANCFBeamTest_032_MUMPS_MT05, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MUMPS_MT05, ANCFBeamTest_064_MUMPS_MT05, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MUMPS, ChElementBeamANCF_MT06, ChMaterialBeamANCF_MT06>
// ANCFBeamTest_008_MUMPS_MT06; typedef ANCFBeamTest<16, SolverType::MUMPS, ChElementBeamANCF_MT06,
// ChMaterialBeamANCF_MT06> ANCFBeamTest_016_MUMPS_MT06; typedef ANCFBeamTest<32, SolverType::MUMPS,
// ChElementBeamANCF_MT06, ChMaterialBeamANCF_MT06> ANCFBeamTest_032_MUMPS_MT06; typedef ANCFBeamTest<64,
// SolverType::MUMPS, ChElementBeamANCF_MT06, ChMaterialBeamANCF_MT06> ANCFBeamTest_064_MUMPS_MT06;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MUMPS_MT06, ANCFBeamTest_008_MUMPS_MT06, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MUMPS_MT06, ANCFBeamTest_016_MUMPS_MT06, NUM_SKIP_STEPS, NUM_SIM_STEPS,
///REPEATS);
//////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MUMPS_MT06, ANCFBeamTest_032_MUMPS_MT06, NUM_SKIP_STEPS, NUM_SIM_STEPS,
///REPEATS);
//////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MUMPS_MT06, ANCFBeamTest_064_MUMPS_MT06, NUM_SKIP_STEPS, NUM_SIM_STEPS,
///REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MUMPS, ChElementBeamANCF_MT07, ChMaterialBeamANCF_MT07>
// ANCFBeamTest_008_MUMPS_MT07; typedef ANCFBeamTest<16, SolverType::MUMPS, ChElementBeamANCF_MT07,
// ChMaterialBeamANCF_MT07> ANCFBeamTest_016_MUMPS_MT07; typedef ANCFBeamTest<32, SolverType::MUMPS,
// ChElementBeamANCF_MT07, ChMaterialBeamANCF_MT07> ANCFBeamTest_032_MUMPS_MT07; typedef ANCFBeamTest<64,
// SolverType::MUMPS, ChElementBeamANCF_MT07, ChMaterialBeamANCF_MT07> ANCFBeamTest_064_MUMPS_MT07;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MUMPS_MT07, ANCFBeamTest_008_MUMPS_MT07, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MUMPS_MT07, ANCFBeamTest_016_MUMPS_MT07, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MUMPS_MT07, ANCFBeamTest_032_MUMPS_MT07, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MUMPS_MT07, ANCFBeamTest_064_MUMPS_MT07, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MUMPS, ChElementBeamANCF_MT08, ChMaterialBeamANCF_MT08>
// ANCFBeamTest_008_MUMPS_MT08; typedef ANCFBeamTest<16, SolverType::MUMPS, ChElementBeamANCF_MT08,
// ChMaterialBeamANCF_MT08> ANCFBeamTest_016_MUMPS_MT08; typedef ANCFBeamTest<32, SolverType::MUMPS,
// ChElementBeamANCF_MT08, ChMaterialBeamANCF_MT08> ANCFBeamTest_032_MUMPS_MT08; typedef ANCFBeamTest<64,
// SolverType::MUMPS, ChElementBeamANCF_MT08, ChMaterialBeamANCF_MT08> ANCFBeamTest_064_MUMPS_MT08;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MUMPS_MT08, ANCFBeamTest_008_MUMPS_MT08, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MUMPS_MT08, ANCFBeamTest_016_MUMPS_MT08, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MUMPS_MT08, ANCFBeamTest_032_MUMPS_MT08, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MUMPS_MT08, ANCFBeamTest_064_MUMPS_MT08, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MUMPS, ChElementBeamANCF_MT09, ChMaterialBeamANCF_MT09>
// ANCFBeamTest_008_MUMPS_MT09; typedef ANCFBeamTest<16, SolverType::MUMPS, ChElementBeamANCF_MT09,
// ChMaterialBeamANCF_MT09> ANCFBeamTest_016_MUMPS_MT09; typedef ANCFBeamTest<32, SolverType::MUMPS,
// ChElementBeamANCF_MT09, ChMaterialBeamANCF_MT09> ANCFBeamTest_032_MUMPS_MT09; typedef ANCFBeamTest<64,
// SolverType::MUMPS, ChElementBeamANCF_MT09, ChMaterialBeamANCF_MT09> ANCFBeamTest_064_MUMPS_MT09;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MUMPS_MT09, ANCFBeamTest_008_MUMPS_MT09, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MUMPS_MT09, ANCFBeamTest_016_MUMPS_MT09, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MUMPS_MT09, ANCFBeamTest_032_MUMPS_MT09, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MUMPS_MT09, ANCFBeamTest_064_MUMPS_MT09, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//
//
// typedef ANCFBeamTest<8, SolverType::MUMPS, ChElementBeamANCF_MT10, ChMaterialBeamANCF_MT10>
// ANCFBeamTest_008_MUMPS_MT10; typedef ANCFBeamTest<16, SolverType::MUMPS, ChElementBeamANCF_MT10,
// ChMaterialBeamANCF_MT10> ANCFBeamTest_016_MUMPS_MT10; typedef ANCFBeamTest<32, SolverType::MUMPS,
// ChElementBeamANCF_MT10, ChMaterialBeamANCF_MT10> ANCFBeamTest_032_MUMPS_MT10; typedef ANCFBeamTest<64,
// SolverType::MUMPS, ChElementBeamANCF_MT10, ChMaterialBeamANCF_MT10> ANCFBeamTest_064_MUMPS_MT10;
// CH_BM_SIMULATION_LOOP(ANCFBeam_008_MUMPS_MT10, ANCFBeamTest_008_MUMPS_MT10, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_016_MUMPS_MT10, ANCFBeamTest_016_MUMPS_MT10, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_032_MUMPS_MT10, ANCFBeamTest_032_MUMPS_MT10, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
////CH_BM_SIMULATION_LOOP(ANCFBeam_064_MUMPS_MT10, ANCFBeamTest_064_MUMPS_MT10, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
#endif








#ifdef CHRONO_MKL
typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_008_MKL_Org;
typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_016_MKL_Org;
typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_032_MKL_Org;
typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_064_MKL_Org;
typedef ANCFBeamTest<128, SolverType::MKL, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_128_MKL_Org;
typedef ANCFBeamTest<256, SolverType::MKL, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_256_MKL_Org;
typedef ANCFBeamTest<512, SolverType::MKL, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_512_MKL_Org;
typedef ANCFBeamTest<1024, SolverType::MKL, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_1024_MKL_Org;
CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_Org, ANCFBeamTest_008_MKL_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_Org, ANCFBeamTest_016_MKL_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_Org, ANCFBeamTest_032_MKL_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_Org, ANCFBeamTest_064_MKL_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_128_MKL_Org, ANCFBeamTest_128_MKL_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_256_MKL_Org, ANCFBeamTest_256_MKL_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_512_MKL_Org, ANCFBeamTest_512_MKL_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_1024_MKL_Org, ANCFBeamTest_1024_MKL_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);

typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_008_MKL_MT26;
typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_016_MKL_MT26;
typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_032_MKL_MT26;
typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_064_MKL_MT26;
typedef ANCFBeamTest<128, SolverType::MKL, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_128_MKL_MT26;
typedef ANCFBeamTest<256, SolverType::MKL, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_256_MKL_MT26;
typedef ANCFBeamTest<512, SolverType::MKL, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_512_MKL_MT26;
typedef ANCFBeamTest<1024, SolverType::MKL, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_1024_MKL_MT26;
CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT26, ANCFBeamTest_008_MKL_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT26, ANCFBeamTest_016_MKL_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT26, ANCFBeamTest_032_MKL_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT26, ANCFBeamTest_064_MKL_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_128_MKL_MT26, ANCFBeamTest_128_MKL_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_256_MKL_MT26, ANCFBeamTest_256_MKL_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_512_MKL_MT26, ANCFBeamTest_512_MKL_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_1024_MKL_MT26, ANCFBeamTest_1024_MKL_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);

typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_008_MKL_MT31;
typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_016_MKL_MT31;
typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_032_MKL_MT31;
typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_064_MKL_MT31;
typedef ANCFBeamTest<128, SolverType::MKL, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_128_MKL_MT31;
typedef ANCFBeamTest<256, SolverType::MKL, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_256_MKL_MT31;
typedef ANCFBeamTest<512, SolverType::MKL, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_512_MKL_MT31;
typedef ANCFBeamTest<1024, SolverType::MKL, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_1024_MKL_MT31; 
CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL_MT31, ANCFBeamTest_008_MKL_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL_MT31, ANCFBeamTest_016_MKL_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL_MT31, ANCFBeamTest_032_MKL_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL_MT31, ANCFBeamTest_064_MKL_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_128_MKL_MT31, ANCFBeamTest_128_MKL_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_256_MKL_MT31, ANCFBeamTest_256_MKL_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_512_MKL_MT31, ANCFBeamTest_512_MKL_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_1024_MKL_MT31, ANCFBeamTest_1024_MKL_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
#endif

#ifdef CHRONO_MUMPS
typedef ANCFBeamTest<8, SolverType::MUMPS, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_008_MUMPS_Org;
typedef ANCFBeamTest<16, SolverType::MUMPS, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_016_MUMPS_Org;
typedef ANCFBeamTest<32, SolverType::MUMPS, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_032_MUMPS_Org;
typedef ANCFBeamTest<64, SolverType::MUMPS, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_064_MUMPS_Org;
typedef ANCFBeamTest<128, SolverType::MUMPS, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_128_MUMPS_Org;
typedef ANCFBeamTest<256, SolverType::MUMPS, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_256_MUMPS_Org;
typedef ANCFBeamTest<512, SolverType::MUMPS, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_512_MUMPS_Org;
typedef ANCFBeamTest<1024, SolverType::MUMPS, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_1024_MUMPS_Org;
CH_BM_SIMULATION_LOOP(ANCFBeam_008_MUMPS_Org, ANCFBeamTest_008_MUMPS_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_016_MUMPS_Org, ANCFBeamTest_016_MUMPS_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_032_MUMPS_Org, ANCFBeamTest_032_MUMPS_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_064_MUMPS_Org, ANCFBeamTest_064_MUMPS_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_128_MUMPS_Org, ANCFBeamTest_128_MUMPS_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_256_MUMPS_Org, ANCFBeamTest_256_MUMPS_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_512_MUMPS_Org, ANCFBeamTest_512_MUMPS_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_1024_MUMPS_Org, ANCFBeamTest_1024_MUMPS_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);

typedef ANCFBeamTest<8, SolverType::MUMPS, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_008_MUMPS_MT26;
typedef ANCFBeamTest<16, SolverType::MUMPS, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_016_MUMPS_MT26;
typedef ANCFBeamTest<32, SolverType::MUMPS, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_032_MUMPS_MT26;
typedef ANCFBeamTest<64, SolverType::MUMPS, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_064_MUMPS_MT26;
typedef ANCFBeamTest<128, SolverType::MUMPS, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_128_MUMPS_MT26;
typedef ANCFBeamTest<256, SolverType::MUMPS, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_256_MUMPS_MT26;
typedef ANCFBeamTest<512, SolverType::MUMPS, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_512_MUMPS_MT26;
typedef ANCFBeamTest<1024, SolverType::MUMPS, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_1024_MUMPS_MT26;
CH_BM_SIMULATION_LOOP(ANCFBeam_008_MUMPS_MT26, ANCFBeamTest_008_MUMPS_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_016_MUMPS_MT26, ANCFBeamTest_016_MUMPS_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_032_MUMPS_MT26, ANCFBeamTest_032_MUMPS_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_064_MUMPS_MT26, ANCFBeamTest_064_MUMPS_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_128_MUMPS_MT26, ANCFBeamTest_128_MUMPS_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_256_MUMPS_MT26, ANCFBeamTest_256_MUMPS_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_512_MUMPS_MT26, ANCFBeamTest_512_MUMPS_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_1024_MUMPS_MT26, ANCFBeamTest_1024_MUMPS_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);

typedef ANCFBeamTest<8, SolverType::MUMPS, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_008_MUMPS_MT31;
typedef ANCFBeamTest<16, SolverType::MUMPS, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_016_MUMPS_MT31;
typedef ANCFBeamTest<32, SolverType::MUMPS, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_032_MUMPS_MT31;
typedef ANCFBeamTest<64, SolverType::MUMPS, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_064_MUMPS_MT31;
typedef ANCFBeamTest<128, SolverType::MUMPS, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_128_MUMPS_MT31;
typedef ANCFBeamTest<256, SolverType::MUMPS, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_256_MUMPS_MT31;
typedef ANCFBeamTest<512, SolverType::MUMPS, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_512_MUMPS_MT31;
typedef ANCFBeamTest<1024, SolverType::MUMPS, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_1024_MUMPS_MT31;
CH_BM_SIMULATION_LOOP(ANCFBeam_008_MUMPS_MT31, ANCFBeamTest_008_MUMPS_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_016_MUMPS_MT31, ANCFBeamTest_016_MUMPS_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_032_MUMPS_MT31, ANCFBeamTest_032_MUMPS_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_064_MUMPS_MT31, ANCFBeamTest_064_MUMPS_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_128_MUMPS_MT31, ANCFBeamTest_128_MUMPS_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_256_MUMPS_MT31, ANCFBeamTest_256_MUMPS_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_512_MUMPS_MT31, ANCFBeamTest_512_MUMPS_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_1024_MUMPS_MT31, ANCFBeamTest_1024_MUMPS_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
#endif


typedef ANCFBeamTest<8, SolverType::SparseLU, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_008_SparseLU_Org;
typedef ANCFBeamTest<16, SolverType::SparseLU, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_016_SparseLU_Org;
typedef ANCFBeamTest<32, SolverType::SparseLU, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_032_SparseLU_Org;
typedef ANCFBeamTest<64, SolverType::SparseLU, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_064_SparseLU_Org;
typedef ANCFBeamTest<128, SolverType::SparseLU, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_128_SparseLU_Org;
typedef ANCFBeamTest<256, SolverType::SparseLU, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_256_SparseLU_Org;
typedef ANCFBeamTest<512, SolverType::SparseLU, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_512_SparseLU_Org;
typedef ANCFBeamTest<1024, SolverType::SparseLU, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_1024_SparseLU_Org;
CH_BM_SIMULATION_LOOP(ANCFBeam_008_SparseLU_Org, ANCFBeamTest_008_SparseLU_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_016_SparseLU_Org, ANCFBeamTest_016_SparseLU_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_032_SparseLU_Org, ANCFBeamTest_032_SparseLU_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_064_SparseLU_Org, ANCFBeamTest_064_SparseLU_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_128_SparseLU_Org, ANCFBeamTest_128_SparseLU_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_256_SparseLU_Org, ANCFBeamTest_256_SparseLU_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_512_SparseLU_Org, ANCFBeamTest_512_SparseLU_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_1024_SparseLU_Org, ANCFBeamTest_1024_SparseLU_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);

typedef ANCFBeamTest<8, SolverType::SparseLU, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_008_SparseLU_MT26;
typedef ANCFBeamTest<16, SolverType::SparseLU, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_016_SparseLU_MT26;
typedef ANCFBeamTest<32, SolverType::SparseLU, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_032_SparseLU_MT26;
typedef ANCFBeamTest<64, SolverType::SparseLU, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_064_SparseLU_MT26;
typedef ANCFBeamTest<128, SolverType::SparseLU, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_128_SparseLU_MT26;
typedef ANCFBeamTest<256, SolverType::SparseLU, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_256_SparseLU_MT26;
typedef ANCFBeamTest<512, SolverType::SparseLU, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_512_SparseLU_MT26;
typedef ANCFBeamTest<1024, SolverType::SparseLU, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_1024_SparseLU_MT26;
CH_BM_SIMULATION_LOOP(ANCFBeam_008_SparseLU_MT26, ANCFBeamTest_008_SparseLU_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_016_SparseLU_MT26, ANCFBeamTest_016_SparseLU_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_032_SparseLU_MT26, ANCFBeamTest_032_SparseLU_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_064_SparseLU_MT26, ANCFBeamTest_064_SparseLU_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_128_SparseLU_MT26, ANCFBeamTest_128_SparseLU_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_256_SparseLU_MT26, ANCFBeamTest_256_SparseLU_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_512_SparseLU_MT26, ANCFBeamTest_512_SparseLU_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_1024_SparseLU_MT26, ANCFBeamTest_1024_SparseLU_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);

typedef ANCFBeamTest<8, SolverType::SparseLU, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_008_SparseLU_MT31;
typedef ANCFBeamTest<16, SolverType::SparseLU, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_016_SparseLU_MT31;
typedef ANCFBeamTest<32, SolverType::SparseLU, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_032_SparseLU_MT31;
typedef ANCFBeamTest<64, SolverType::SparseLU, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_064_SparseLU_MT31;
typedef ANCFBeamTest<128, SolverType::SparseLU, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_128_SparseLU_MT31;
typedef ANCFBeamTest<256, SolverType::SparseLU, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_256_SparseLU_MT31;
typedef ANCFBeamTest<512, SolverType::SparseLU, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_512_SparseLU_MT31;
typedef ANCFBeamTest<1024, SolverType::SparseLU, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_1024_SparseLU_MT31;
CH_BM_SIMULATION_LOOP(ANCFBeam_008_SparseLU_MT31, ANCFBeamTest_008_SparseLU_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_016_SparseLU_MT31, ANCFBeamTest_016_SparseLU_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_032_SparseLU_MT31, ANCFBeamTest_032_SparseLU_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_064_SparseLU_MT31, ANCFBeamTest_064_SparseLU_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_128_SparseLU_MT31, ANCFBeamTest_128_SparseLU_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_256_SparseLU_MT31, ANCFBeamTest_256_SparseLU_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_512_SparseLU_MT31, ANCFBeamTest_512_SparseLU_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_1024_SparseLU_MT31, ANCFBeamTest_1024_SparseLU_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);

typedef ANCFBeamTest<8, SolverType::SparseQR, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_008_SparseQR_Org;
typedef ANCFBeamTest<16, SolverType::SparseQR, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_016_SparseQR_Org;
typedef ANCFBeamTest<32, SolverType::SparseQR, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_032_SparseQR_Org;
typedef ANCFBeamTest<64, SolverType::SparseQR, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_064_SparseQR_Org;
typedef ANCFBeamTest<128, SolverType::SparseQR, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_128_SparseQR_Org;
typedef ANCFBeamTest<256, SolverType::SparseQR, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_256_SparseQR_Org;
typedef ANCFBeamTest<512, SolverType::SparseQR, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_512_SparseQR_Org;
typedef ANCFBeamTest<1024, SolverType::SparseQR, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_1024_SparseQR_Org;
CH_BM_SIMULATION_LOOP(ANCFBeam_008_SparseQR_Org, ANCFBeamTest_008_SparseQR_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_016_SparseQR_Org, ANCFBeamTest_016_SparseQR_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_032_SparseQR_Org, ANCFBeamTest_032_SparseQR_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_064_SparseQR_Org, ANCFBeamTest_064_SparseQR_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_128_SparseQR_Org, ANCFBeamTest_128_SparseQR_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_256_SparseQR_Org, ANCFBeamTest_256_SparseQR_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_512_SparseQR_Org, ANCFBeamTest_512_SparseQR_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_1024_SparseQR_Org, ANCFBeamTest_1024_SparseQR_Org, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);

typedef ANCFBeamTest<8, SolverType::SparseQR, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_008_SparseQR_MT26;
typedef ANCFBeamTest<16, SolverType::SparseQR, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_016_SparseQR_MT26;
typedef ANCFBeamTest<32, SolverType::SparseQR, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_032_SparseQR_MT26;
typedef ANCFBeamTest<64, SolverType::SparseQR, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_064_SparseQR_MT26;
typedef ANCFBeamTest<128, SolverType::SparseQR, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_128_SparseQR_MT26;
typedef ANCFBeamTest<256, SolverType::SparseQR, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_256_SparseQR_MT26;
typedef ANCFBeamTest<512, SolverType::SparseQR, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_512_SparseQR_MT26;
typedef ANCFBeamTest<1024, SolverType::SparseQR, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ANCFBeamTest_1024_SparseQR_MT26;
CH_BM_SIMULATION_LOOP(ANCFBeam_008_SparseQR_MT26, ANCFBeamTest_008_SparseQR_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_016_SparseQR_MT26, ANCFBeamTest_016_SparseQR_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_032_SparseQR_MT26, ANCFBeamTest_032_SparseQR_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_064_SparseQR_MT26, ANCFBeamTest_064_SparseQR_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_128_SparseQR_MT26, ANCFBeamTest_128_SparseQR_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_256_SparseQR_MT26, ANCFBeamTest_256_SparseQR_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_512_SparseQR_MT26, ANCFBeamTest_512_SparseQR_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_1024_SparseQR_MT26, ANCFBeamTest_1024_SparseQR_MT26, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);

typedef ANCFBeamTest<8, SolverType::SparseQR, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_008_SparseQR_MT31;
typedef ANCFBeamTest<16, SolverType::SparseQR, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_016_SparseQR_MT31;
typedef ANCFBeamTest<32, SolverType::SparseQR, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_032_SparseQR_MT31;
typedef ANCFBeamTest<64, SolverType::SparseQR, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_064_SparseQR_MT31;
typedef ANCFBeamTest<128, SolverType::SparseQR, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_128_SparseQR_MT31;
typedef ANCFBeamTest<256, SolverType::SparseQR, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_256_SparseQR_MT31;
typedef ANCFBeamTest<512, SolverType::SparseQR, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_512_SparseQR_MT31;
typedef ANCFBeamTest<1024, SolverType::SparseQR, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ANCFBeamTest_1024_SparseQR_MT31;
CH_BM_SIMULATION_LOOP(ANCFBeam_008_SparseQR_MT31, ANCFBeamTest_008_SparseQR_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_016_SparseQR_MT31, ANCFBeamTest_016_SparseQR_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_032_SparseQR_MT31, ANCFBeamTest_032_SparseQR_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_064_SparseQR_MT31, ANCFBeamTest_064_SparseQR_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFBeam_128_SparseQR_MT31, ANCFBeamTest_128_SparseQR_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_256_SparseQR_MT31, ANCFBeamTest_256_SparseQR_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_512_SparseQR_MT31, ANCFBeamTest_512_SparseQR_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS); 
//CH_BM_SIMULATION_LOOP(ANCFBeam_1024_SparseQR_MT31, ANCFBeamTest_1024_SparseQR_MT31, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);

// =============================================================================

int main(int argc, char* argv[]) {
    //printf("MKL threads = %d\n", mkl_get_max_threads());
    //printf("OpenMP threads = %d\n", omp_get_num_threads());
    //printf("Max OpenMP threads = %d\n", omp_get_max_threads());
    //printf("OpenMP Num Procs = %d\n", omp_get_num_procs());
    //printf("Setting omp_set_num_threads(4)\n");
    //omp_set_num_threads(4);
    //printf("OpenMP threads = %d\n\n", omp_get_num_threads());

    //omp_set_num_threads(8);
    //#pragma omp parallel for
    //for (int i = 0; i < omp_get_max_threads(); i++) {
    //    std::cout << omp_get_num_threads() << "\t nThreads - Thread ID: " << omp_get_thread_num() << std::endl;
    //}
    

    //omp_set_num_threads(4);

#if true
    ::benchmark::Initialize(&argc, argv);

#ifdef CHRONO_IRRLICHT
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        ANCFBeamTest<8, SolverType::MUMPS, ChElementBeamANCF, ChMaterialBeamANCF> test;
        test.SimulateVis();
        return 0;
    }
#endif

    ::benchmark::RunSpecifiedBenchmarks();
#else
    
//#define NUM_ELEMENTS 8
//
//    ANCFBeamTest<2024, SolverType::MUMPS, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> test;
//    ANCFBeamTest<NUM_ELEMENTS, SolverType::MUMPS, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> test_MT26;
//    ANCFBeamTest<NUM_ELEMENTS, SolverType::MUMPS, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> test_MT31;
//
//    for (auto i = 0; i < 1; i++) {
//        test.ExecuteStep();
//    }

    //for (auto i = 0; i < NUM_SKIP_STEPS; i++) {
    //    test_MT26.ExecuteStep();
    //    test_MT31.ExecuteStep();
    //}

    //ChTimer<> Timer_Total;
    //Timer_Total.reset();
    //Timer_Total.start();

    //for (auto i = 0; i < NUM_SIM_STEPS; i++) {
    //    test_MT26.ExecuteStep();
    //}
    //Timer_Total.stop();
    //std::cout << "MT26 - Avg Time per Step in ms:" << Timer_Total()*(100.0/ double(NUM_SIM_STEPS)) << std::endl;

    //Timer_Total.reset();
    //Timer_Total.start();

    //for (auto i = 0; i < NUM_SIM_STEPS; i++) {
    //    test_MT31.ExecuteStep();
    //}
    //Timer_Total.stop();
    //std::cout << "MT31 - Avg Time per Step in ms:" << Timer_Total()*(100.0 / double(NUM_SIM_STEPS)) << std::endl;

#endif

}

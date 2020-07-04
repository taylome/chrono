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

#include "chrono/ChConfig.h"
#include "chrono/utils/ChBenchmark.h"

#include "chrono/physics/ChSystemSMC.h"
#include "chrono/solver/ChIterativeSolverLS.h"

#include "chrono/fea/ChElementBeamANCF.h"

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

enum class SolverType {MINRES, MKL, MUMPS};

template<int num_elements, SolverType solver_type, typename ElementVersion, typename MaterialVersion>
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

template<int num_elements, SolverType solver_type, typename ElementVersion, typename MaterialVersion>
ANCFBeamTest<num_elements, solver_type, ElementVersion, MaterialVersion>::ANCFBeamTest() {

    m_system = new ChSystemSMC();
    m_system->Set_G_acc(ChVector<>(0, 0, -9.80665));

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
        solver->UseSparsityPatternLearner(true);
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
    double length = 5;      //m
    double width = 0.1;     //m
    double thickness = 0.1; //m
    double rho = 8245.2;    //kg/m^3
    double E = 132e9;       //Pa
    double nu = 0;          //Poisson effect neglected for this model
    double G = E / (2 * (1 + nu));
    double k1 = 10 * (1 + nu) / (12 + 11 * nu);  // Timoshenko shear correction coefficient for a rectangular cross-section
    double k2 = k1;                              // Timoshenko shear correction coefficient for a rectangular cross-section

    auto material = chrono_types::make_shared<MaterialVersion>(rho, E, nu, k1, k2);


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

    //Setup beam cross section gradients to initially align with the global y and z directions
    ChVector<> dir1(0, 1, 0);
    ChVector<> dir2(0, 0, 1);

    //Create the first node and fix it completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0, 0.0), dir1, dir2);
    nodeA->SetFixed(true);
    mesh->AddNode(nodeA);

    auto elementlast = chrono_types::make_shared<ElementVersion>();
    
    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(dx*(2 * i), 0, 0), dir1, dir2);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(dx*(2 * i - 1), 0, 0), dir1, dir2);
        mesh->AddNode(nodeB);
        mesh->AddNode(nodeC);

        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC);
        element->SetDimensions(2 * dx, thickness, width);
        element->SetMaterial(material);
        element->SetAlphaDamp(0.0);
        element->SetGravityOn(false);  //Enable the efficient ANCF method for calculating the application of gravity to the element
        element->SetStrainFormulation(ElementVersion::StrainFormulation::CMPoisson);
        //element->SetStrainFormulation(ElementVersion::StrainFormulation::CMNoPoisson);
        mesh->AddElement(element);

        nodeA = nodeB;
        elementlast = element;
    }

    m_nodeEndPoint = nodeA;

    mesh->SetAutomaticGravity(false); //Turn off the default method for applying gravity to the mesh since it is less efficient for ANCF elements


    // Create a custom atomic (point) load

    class MyLoaderTimeDependentTipLoad : public ChLoaderUatomic {
    public:
        // Useful: a constructor that also sets ChLoadable    
        MyLoaderTimeDependentTipLoad(std::shared_ptr<ChLoadableU> mloadable)
            : ChLoaderUatomic(mloadable) {}

        // Compute F=F(u), the load at U. The load is a 6-row vector, i.e.
        // a wrench: forceX, forceY, forceZ, torqueX, torqueY, torqueZ. 
        virtual void ComputeF(const double U,     ///< normalized position along the beam axis [-1...1]
            ChVectorDynamic<>& F,       ///< Load at U
            ChVectorDynamic<>* state_x, ///< if != 0, update state (pos. part) to this, then evaluate F
            ChVectorDynamic<>* state_w  ///< if != 0, update state (speed part) to this, then evaluate F
        ) override {
            assert(auxsystem);
            double T = auxsystem->GetChTime();
            double Fmax = -300;
            double tc = 3.5;
            double Fz = Fmax;
            if (T < tc) {
                Fz = 0.5*Fmax*(1 - cos(CH_C_PI*T / tc));
            }

            F.setZero();
            F(2) = Fz;   //Apply the force along the global Z axis
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

template<int num_elements, SolverType solver_type, typename ElementVersion, typename MaterialVersion>
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
        std::cout <<"Time(s): " << this->m_system->GetChTime() << "  Tip Pos(m): "<< this->GetBeamEndPointPos() << std::endl;
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


// NOTE: trick to prevent errors in expanding macros due to types that contain a comma.
typedef ANCFBeamTest<8, SolverType::MINRES, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_008_MINRES_Config;
typedef ANCFBeamTest<16, SolverType::MINRES, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_016_MINRES_Config;
typedef ANCFBeamTest<32, SolverType::MINRES, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_032_MINRES_Config;
typedef ANCFBeamTest<64, SolverType::MINRES, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_064_MINRES_Config;
CH_BM_SIMULATION_LOOP(ANCFBeamTest_008_MINRES, ANCFBeamTest_008_MINRES_Config, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeamTest_016_MINRES, ANCFBeamTest_016_MINRES_Config, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeamTest_032_MINRES, ANCFBeamTest_032_MINRES_Config, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeamTest_064_MINRES, ANCFBeamTest_064_MINRES_Config, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);

#ifdef CHRONO_MKL
typedef ANCFBeamTest<8, SolverType::MKL, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_008_MKL_Config;
typedef ANCFBeamTest<16, SolverType::MKL, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_016_MKL_Config;
typedef ANCFBeamTest<32, SolverType::MKL, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_032_MKL_Config;
typedef ANCFBeamTest<64, SolverType::MKL, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_064_MKL_Config;
CH_BM_SIMULATION_LOOP(ANCFBeam_008_MKL, ANCFBeamTest_008_MKL_Config, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_016_MKL, ANCFBeamTest_016_MKL_Config, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_032_MKL, ANCFBeamTest_032_MKL_Config, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_064_MKL, ANCFBeamTest_064_MKL_Config, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
#endif

#ifdef CHRONO_MUMPS
typedef ANCFBeamTest<8, SolverType::MUMPS, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_008_MUMPS_Config;
typedef ANCFBeamTest<16, SolverType::MUMPS, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_016_MUMPS_Config;
typedef ANCFBeamTest<32, SolverType::MUMPS, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_032_MUMPS_Config;
typedef ANCFBeamTest<64, SolverType::MUMPS, ChElementBeamANCF, ChMaterialBeamANCF> ANCFBeamTest_064_MUMPS_Config;
CH_BM_SIMULATION_LOOP(ANCFBeam_008_MUMPS, ANCFBeamTest_008_MUMPS_Config, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_016_MUMPS, ANCFBeamTest_016_MUMPS_Config, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_032_MUMPS, ANCFBeamTest_032_MUMPS_Config, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
CH_BM_SIMULATION_LOOP(ANCFBeam_064_MUMPS, ANCFBeamTest_064_MUMPS_Config, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
#endif

// =============================================================================

int main(int argc, char* argv[]) {
    ::benchmark::Initialize(&argc, argv);

#ifdef CHRONO_IRRLICHT
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        ANCFBeamTest<8, SolverType::MUMPS, ChElementBeamANCF, ChMaterialBeamANCF> test;
        test.SimulateVis();
        return 0;
    }
#endif

    ::benchmark::RunSpecifiedBenchmarks();
}

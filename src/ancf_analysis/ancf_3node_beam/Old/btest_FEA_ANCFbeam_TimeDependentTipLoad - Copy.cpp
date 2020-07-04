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
// Benchmark test for ANCF beam elements.
//
// Square cantilevered beam with a time-dependent tip load
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
#include "chrono/fea/ChElementBeamANCF_MT01.h"

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
enum class ElementVersion {ORIGINAL, MT01};

template <int N>
class ANCFbeam : public utils::ChBenchmarkTest {
  public:
    virtual ~ANCFbeam() { delete m_system; }

    ChSystem* GetSystem() override { return m_system; }
    void ExecuteStep() override { m_system->DoStepDynamics(1e-2); }

    void SimulateVis();
    void NonlinearStatics() { m_system->DoStaticNonlinear(50); };
    ChVector<> GetBeamEndPointPos() { return m_nodeEndPoint->GetPos(); }

  protected:
    ANCFbeam(SolverType solver_type, ElementVersion element_version);
    ChSystemSMC* m_system;
    std::shared_ptr<ChNodeFEAxyzDD> m_nodeEndPoint;
};

template <int N>
class ANCFbeam_MINRES_ORGINAL : public ANCFbeam<N> {
  public:
    ANCFbeam_MINRES_ORGINAL() : ANCFbeam<N>(SolverType::MINRES, ElementVersion::ORIGINAL) {}
};

template <int N>
class ANCFbeam_MKL_ORGINAL : public ANCFbeam<N> {
  public:
    ANCFbeam_MKL_ORGINAL() : ANCFbeam<N>(SolverType::MKL, ElementVersion::ORIGINAL) {}
};

template <int N>
class ANCFbeam_MUMPS_ORGINAL : public ANCFbeam<N> {
  public:
    ANCFbeam_MUMPS_ORGINAL() : ANCFbeam<N>(SolverType::MUMPS, ElementVersion::ORIGINAL) {}
};

template <int N>
ANCFbeam<N>::ANCFbeam(SolverType solver_type, ElementVersion element_version) {
    int num_elements = N;
    double beam_angle_rad = 0;
    double vert_tip_load_N = 0;

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
    double rho = 8245.2; //kg/m^3
    double E = 132e9; //Pa
    double nu = 0;  
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

    //Rotate the cross section gradients to match the sign convention in the experiment
    ChVector<> dir1(0, cos(-beam_angle_rad), sin(-beam_angle_rad));
    ChVector<> dir2(0, -sin(-beam_angle_rad), cos(-beam_angle_rad));

    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0, 0.0), dir1, dir2);
    nodeA->SetFixed(true);
    mesh->AddNode(nodeA);

    auto elementlast = chrono_types::make_shared<ChElementBeamANCF>();
    //switch (element_version) {
    //    case ElementVersion::ORIGINAL : auto elementlast = chrono_types::make_shared<ChElementBeamANCF>();
    //        break;
    //    case ElementVersion::MT01 : auto elementlast = chrono_types::make_shared<ChElementBeamANCF_MT01>();
    //        break;
    //}

    
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
        element->SetGravityOn(false);  //Enable the efficient ANCF method for calculating the application of gravity to the element
        element->SetStrainFormulation(ChElementBeamANCF::StrainFormulation::CMPoisson);
        //element->SetStrainFormulation(ChElementBeamANCF::StrainFormulation::CMNoPoisson);
        mesh->AddElement(element);

        nodeA = nodeB;
        elementlast = element;
    }

    m_nodeEndPoint = nodeA;

    mesh->SetAutomaticGravity(false); //Turn off the default method for applying gravity to the mesh since it is less efficient for ANCF elements

    //m_nodeEndPoint->SetForce(ChVector<>(0, 0, -vert_tip_load_N));

    // Create a custom atomic (point) load

    class MyLoaderTimeDependentTipLoad : public ChLoaderUatomic {
    public:
        // Useful: a constructor that also sets ChLoadable    
        MyLoaderTimeDependentTipLoad(std::shared_ptr<ChLoadableU> mloadable)
            : ChLoaderUatomic(mloadable) {}

        // Compute F=F(u)
        // This is the function that YOU MUST implement. It should return the 
        // load at U. For Eulero beams, loads are expected as 6-rows vectors, i.e.
        // a wrench: forceX, forceY, forceZ, torqueX, torqueY, torqueZ. 
        virtual void ComputeF(const double U,     ///< parametric coordinate in line
            ChVectorDynamic<>& F,       ///< Result F vector here, size must be = n.field coords.of loadable
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

            F.segment(0, 3) = ChVector<>(0, 0, Fz).eigen();  // load, force part
            F.segment(3, 3) = ChVector<>(0, 0, 0).eigen();   // load, torque part
        }

    public:
        // add auxiliary data to the class, if you need to access it during ComputeF().
        ChSystem* auxsystem;
    };

    // Create the load container and add it to your ChSystem 
    auto loadcontainer = chrono_types::make_shared<ChLoadContainer>();
    m_system->Add(loadcontainer);

    // Create a custom load that uses the custom loader above.
    // The ChLoad is a 'manager' for your ChLoader.
    // It is created using templates, that is instancing a ChLoad<my_loader_class>()

    std::shared_ptr<ChLoad<MyLoaderTimeDependentTipLoad> > mload(new ChLoad<MyLoaderTimeDependentTipLoad>(elementlast));
    mload->loader.auxsystem = m_system;  // initialize auxiliary data of the loader, if needed
    mload->loader.SetApplication(1.0);  // specify application point
    loadcontainer->Add(mload);          // do not forget to add the load to the load container.

}

template <int N>
void ANCFbeam<N>::SimulateVis() {
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

//// NOTE: trick to prevent erros in expanding macros due to types that contain a comma.
//typedef M113AccTest<TrackShoeType, TrackShoeType::SINGLE_PIN> sp_test_type;
//typedef M113AccTest<TrackShoeType, TrackShoeType::DOUBLE_PIN> dp_test_type;


//CH_BM_SIMULATION_LOOP(ANCFbeam08_MINRES_ORGINAL, ANCFbeam_MINRES_ORGINAL<8>, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFbeam16_MINRES_ORGINAL, ANCFbeam_MINRES_ORGINAL<16>, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFbeam32_MINRES_ORGINAL, ANCFbeam_MINRES_ORGINAL<32>, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFbeam64_MINRES_ORGINAL, ANCFbeam_MINRES_ORGINAL<64>, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);

//#ifdef CHRONO_MKL
//CH_BM_SIMULATION_LOOP(ANCFbeam08_MKL_ORGINAL, ANCFbeam_MKL_ORGINAL<8>, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFbeam16_MKL_ORGINAL, ANCFbeam_MKL_ORGINAL<16>, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFbeam32_MKL_ORGINAL, ANCFbeam_MKL_ORGINAL<32>, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFbeam64_MKL_ORGINAL, ANCFbeam_MKL_ORGINAL<64>, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//#endif

#ifdef CHRONO_MUMPS
CH_BM_SIMULATION_LOOP(ANCFbeam08_MUMPS_ORGINAL, ANCFbeam_MUMPS_ORGINAL<8>, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFbeam16_MUMPS_ORGINAL, ANCFbeam_MUMPS_ORGINAL<16>, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFbeam32_MUMPS_ORGINAL, ANCFbeam_MUMPS_ORGINAL<32>, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
//CH_BM_SIMULATION_LOOP(ANCFbeam64_MUMPS_ORGINAL, ANCFbeam_MUMPS_ORGINAL<64>, NUM_SKIP_STEPS, NUM_SIM_STEPS, REPEATS);
#endif

// =============================================================================

int main(int argc, char* argv[]) {
    ::benchmark::Initialize(&argc, argv);

#ifdef CHRONO_IRRLICHT
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        ANCFbeam_MUMPS_ORGINAL<8> test;
        test.SimulateVis();
        return 0;
    }
#endif

    ::benchmark::RunSpecifiedBenchmarks();
}

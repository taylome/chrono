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

#include <string>
//#include "unsupported/Eigen/FFT"

#include "chrono/ChConfig.h"
#include "chrono/utils/ChBenchmark.h"

#include "chrono/physics/ChSystemSMC.h"
#include "chrono/solver/ChIterativeSolverLS.h"

#include "chrono_postprocess/ChGnuPlot.h"
#include "chrono_thirdparty/filesystem/path.h"

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
using namespace postprocess;

enum class SolverType { MINRES, MKL, MUMPS, SparseLU, SparseQR };

// =============================================================================

#define NUM_SKIP_STEPS 10  // number of steps for hot start
#define NUM_SIM_STEPS 100  // number of simulation steps for each benchmark
#define REPEATS 10

// =============================================================================

template <typename ElementVersion, typename MaterialVersion>
class ANCFBeamTest {
  public:
    ANCFBeamTest(int num_elements, SolverType solver_type);

    ~ANCFBeamTest() { delete m_system; }

    ChSystem* GetSystem() { return m_system; }
    void ExecuteStep() { m_system->DoStepDynamics(1e-2); }

    void SimulateVis();
    void NonlinearStatics() { m_system->DoStaticNonlinear(50); }
    ChVector<> GetBeamEndPointPos() { return m_nodeEndPoint->GetPos(); }

    void RunTimingTest(ChMatrixNM<double, 4, 17>& timing_stats, const std::string& test_name);

  protected:
    ChSystemSMC* m_system;
    std::shared_ptr<ChNodeFEAxyzDD> m_nodeEndPoint;
    SolverType m_SolverType;
    int m_NumElements;
};

template <typename ElementVersion, typename MaterialVersion>
ANCFBeamTest<ElementVersion, MaterialVersion>::ANCFBeamTest(int num_elements, SolverType solver_type) {
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
void ANCFBeamTest<ElementVersion, MaterialVersion>::RunTimingTest(ChMatrixNM<double, 4, 17>& timing_stats,
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
    ChMatrixNM<double, REPEATS, 17> timing_results;
    timing_results.setZero();

    // Run the requested number of steps to warm start the system, but do not collect any timing information
    for (int i = 0; i < NUM_SKIP_STEPS; i++) {
        ExecuteStep();
    }

    //ChMatrixNM<double, NUM_SIM_STEPS*REPEATS, 2> tip_displacement_z;
    ChMatrixNM<double, NUM_SIM_STEPS, 2> tip_displacement_z;


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

            if (r == (REPEATS - 1)) {
                tip_displacement_z(i, 0) = GetSystem()->GetChTime();
                tip_displacement_z(i, 1) = GetBeamEndPointPos().z();
            }
            //tip_displacement_z(i + (r * NUM_SIM_STEPS), 0) = GetSystem()->GetChTime();
            //tip_displacement_z(i + (r * NUM_SIM_STEPS), 1) = GetBeamEndPointPos().z();
        }
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

    double tip_displacement_offset = (tip_displacement_z.col(1).maxCoeff() + tip_displacement_z.col(1).minCoeff())/2;

    //Eigen::VectorXcd f_values(NUM_SIM_STEPS);
    //ChVectorN<double, NUM_SIM_STEPS> freq;
    //double Fs = 1.0 / (tip_displacement_z(1, 0) - tip_displacement_z(0, 0));
    //for (auto i = 0; i < NUM_SIM_STEPS; i++) {
    //    freq(i) = Fs * i / double(NUM_SIM_STEPS);
    //}

    //Eigen::FFT<double> fft;
    //Eigen::VectorXcd f_freq(NUM_SIM_STEPS);
    //fft.fwd(f_freq, tip_displacement_z.col(1));

    //ChMatrixNM<double, NUM_SIM_STEPS, 2> fft_results;
    //fft_results.col(0) = freq;
    //for (auto i = 0; i < NUM_SIM_STEPS; i++) {
    //    fft_results(i, 0) = Fs * i / double(tip_displacement_z.rows());
    //    fft_results(i, 1) = abs(f_freq(i));
    //}

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
    std::cout << " - Max_OMP_Threads: " << omp_get_max_threads();
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
              << "FEA_Jacobian_Calls" << std::endl;
    for (int r = 0; r < REPEATS; r++) {
        std::cout << "Run_" << r << ":\t" << timing_results.row(r) << std::endl;
    }
    std::cout << "Min:\t" << timing_stats.row(0) << std::endl;
    std::cout << "Max:\t" << timing_stats.row(1) << std::endl;
    std::cout << "Mean:\t" << timing_stats.row(2) << std::endl;
    std::cout << "StdDev:\t" << timing_stats.row(3) << std::endl;

    //std::cout << fft_results << std::endl;

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
    //mplot.Plot(tip_displacement_z, 0, 1, "from ChMatrix", " with lines lt 5");


}

int main(int argc, char* argv[]) {
    ChVectorN<int, 4> num_els;
    num_els << 8, 16, 32, 64;

    //std::vector<SolverType> Solver = {SolverType::MINRES, SolverType::MKL, SolverType::MUMPS, SolverType::SparseLU,
    //                                  SolverType::SparseQR};
    std::vector<SolverType> Solver = {SolverType::MKL, SolverType::MUMPS, SolverType::SparseLU, SolverType::SparseQR};

    //ChVectorN<int, 1> num_els;
    //num_els << 8;
    //std::vector<SolverType> Solver = { SolverType::MUMPS };
    
    ChMatrixNM<double, 4, 17> timing_stats;

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF, ChMaterialBeamANCF> test(num_els(i), ls);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> test(num_els(i), ls);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT26");
        }
    }

    for (const auto& ls : Solver) {
        for (auto i = 0; i < num_els.size(); i++) {
            ANCFBeamTest<ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> test(num_els(i), ls);
            test.RunTimingTest(timing_stats, "ChElementBeamANCF_MT31");
        }
    }
}

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
// Benchmark timing tests for calculating the generalized internal force vector
// and the Jacobian of the generalized internal force vector.  No actual
// simulation is conducted since direct calls to the element's functions are made.
//
// =============================================================================

#include "chrono/ChConfig.h"

#include "chrono/physics/ChSystemSMC.h"
#include "chrono/solver/ChDirectSolverLS.h"

#include "chrono/fea/ChElementBeamANCF_3243.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR01.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR01B.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR02.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR02B.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR03.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR03B.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR04.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR04B.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR05.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR05B.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR06.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR06B.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR07.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR07B.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR08.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR08B.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR09.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR09B.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR10.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR10B.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR11.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR11B.h"

#include "chrono/fea/ChElementBeamANCF_3333.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR01.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR01B.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR02.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR02B.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR03.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR03B.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR04.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR04B.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR05.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR05B.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR06.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR06B.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR07.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR07B.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR08.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR08B.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR09.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR09B.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR10.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR10B.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR11.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR11B.h"

#include "chrono/fea/ChElementShellANCF_3443.h"
#include "chrono/fea/ChElementShellANCF_3443_TR01.h"
#include "chrono/fea/ChElementShellANCF_3443_TR01B.h"
#include "chrono/fea/ChElementShellANCF_3443_TR02.h"
#include "chrono/fea/ChElementShellANCF_3443_TR02B.h"
#include "chrono/fea/ChElementShellANCF_3443_TR03.h"
#include "chrono/fea/ChElementShellANCF_3443_TR03B.h"
#include "chrono/fea/ChElementShellANCF_3443_TR04.h"
#include "chrono/fea/ChElementShellANCF_3443_TR04B.h"
#include "chrono/fea/ChElementShellANCF_3443_TR05.h"
#include "chrono/fea/ChElementShellANCF_3443_TR05B.h"
#include "chrono/fea/ChElementShellANCF_3443_TR06.h"
#include "chrono/fea/ChElementShellANCF_3443_TR06B.h"
#include "chrono/fea/ChElementShellANCF_3443_TR07.h"
#include "chrono/fea/ChElementShellANCF_3443_TR07B.h"
#include "chrono/fea/ChElementShellANCF_3443_TR08.h"
#include "chrono/fea/ChElementShellANCF_3443_TR08B.h"
#include "chrono/fea/ChElementShellANCF_3443_TR09.h"
#include "chrono/fea/ChElementShellANCF_3443_TR09B.h"
#include "chrono/fea/ChElementShellANCF_3443_TR10.h"
#include "chrono/fea/ChElementShellANCF_3443_TR10B.h"
#include "chrono/fea/ChElementShellANCF_3443_TR11.h"
#include "chrono/fea/ChElementShellANCF_3443_TR11B.h"

#include "chrono/fea/ChElementShellANCF_3833.h"
#include "chrono/fea/ChElementShellANCF_3833_TR01.h"
#include "chrono/fea/ChElementShellANCF_3833_TR01B.h"
#include "chrono/fea/ChElementShellANCF_3833_TR02.h"
#include "chrono/fea/ChElementShellANCF_3833_TR02B.h"
#include "chrono/fea/ChElementShellANCF_3833_TR03.h"
#include "chrono/fea/ChElementShellANCF_3833_TR03B.h"
#include "chrono/fea/ChElementShellANCF_3833_TR04.h"
#include "chrono/fea/ChElementShellANCF_3833_TR04B.h"
#include "chrono/fea/ChElementShellANCF_3833_TR05.h"
#include "chrono/fea/ChElementShellANCF_3833_TR05B.h"
#include "chrono/fea/ChElementShellANCF_3833_TR06.h"
#include "chrono/fea/ChElementShellANCF_3833_TR06B.h"
#include "chrono/fea/ChElementShellANCF_3833_TR07.h"
#include "chrono/fea/ChElementShellANCF_3833_TR07B.h"
#include "chrono/fea/ChElementShellANCF_3833_TR08.h"
#include "chrono/fea/ChElementShellANCF_3833_TR08B.h"
#include "chrono/fea/ChElementShellANCF_3833_TR09.h"
#include "chrono/fea/ChElementShellANCF_3833_TR09B.h"
#include "chrono/fea/ChElementShellANCF_3833_TR10.h"
#include "chrono/fea/ChElementShellANCF_3833_TR10B.h"
#include "chrono/fea/ChElementShellANCF_3833_TR11.h"
#include "chrono/fea/ChElementShellANCF_3833_TR11B.h"

#include "chrono/fea/ChElementHexaANCF_3843.h"
#include "chrono/fea/ChElementHexaANCF_3843_TR01.h"
#include "chrono/fea/ChElementHexaANCF_3843_TR01B.h"
#include "chrono/fea/ChElementHexaANCF_3843_TR02.h"
#include "chrono/fea/ChElementHexaANCF_3843_TR02B.h"
#include "chrono/fea/ChElementHexaANCF_3843_TR03.h"
#include "chrono/fea/ChElementHexaANCF_3843_TR03B.h"
#include "chrono/fea/ChElementHexaANCF_3843_TR04.h"
#include "chrono/fea/ChElementHexaANCF_3843_TR04B.h"
#include "chrono/fea/ChElementHexaANCF_3843_TR05.h"
#include "chrono/fea/ChElementHexaANCF_3843_TR05B.h"
#include "chrono/fea/ChElementHexaANCF_3843_TR06.h"
#include "chrono/fea/ChElementHexaANCF_3843_TR06B.h"
#include "chrono/fea/ChElementHexaANCF_3843_TR07.h"
#include "chrono/fea/ChElementHexaANCF_3843_TR07B.h"
#include "chrono/fea/ChElementHexaANCF_3843_TR08.h"
#include "chrono/fea/ChElementHexaANCF_3843_TR08B.h"
#include "chrono/fea/ChElementHexaANCF_3843_TR09.h"
#include "chrono/fea/ChElementHexaANCF_3843_TR09B.h"
#include "chrono/fea/ChElementHexaANCF_3843_TR10.h"
#include "chrono/fea/ChElementHexaANCF_3843_TR10B.h"
#include "chrono/fea/ChElementHexaANCF_3843_TR11.h"
#include "chrono/fea/ChElementHexaANCF_3843_TR11B.h"

#include "chrono/fea/ChMesh.h"

using namespace chrono;
using namespace chrono::fea;

// =============================================================================

template <int num_elements, typename ElementVersion, typename MaterialVersion>
class ANCFBeam3243Test {
  public:
    ANCFBeam3243Test();

    ~ANCFBeam3243Test() { delete m_system; }

    ChSystem* GetSystem() { return m_system; }

    void PerturbNodes(const bool Use_OMP);
    double GetInternalFrc(const bool Use_OMP);
    double GetJacobian(const bool Use_OMP);

    void PrintTimingResults(const std::string& TestName, int steps);

  protected:
    ChSystemSMC* m_system;
};

template <int num_elements, typename ElementVersion, typename MaterialVersion>
ANCFBeam3243Test<num_elements, ElementVersion, MaterialVersion>::ANCFBeam3243Test() {
    m_system = new ChSystemSMC();
    m_system->Set_G_acc(ChVector<>(0, 0, -9.80665));

    // Set solver parameters
    auto solver = chrono_types::make_shared<ChSolverSparseLU>();
    solver->UseSparsityPatternLearner(true);
    solver->LockSparsityPattern(true);
    solver->SetVerbose(false);
    m_system->SetSolver(solver);

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
    double k1 =
        10 * (1 + nu) / (12 + 11 * nu);  // Timoshenko shear correction coefficient for a rectangular cross-section
    double k2 = k1;                      // Timoshenko shear correction coefficient for a rectangular cross-section

    auto material = chrono_types::make_shared<MaterialVersion>(rho, E, nu, k1, k2);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    m_system->Add(mesh);

    // Populate the mesh container with a the nodes and elements for the meshed beam
    int num_nodes = num_elements + 1;
    double dx = length / (num_nodes - 1);

    // Setup beam cross section gradients to initially align with the global y and z directions
    ChVector<> dir1(1, 0, 0);
    ChVector<> dir2(0, 1, 0);
    ChVector<> dir3(0, 0, 1);

    // Create the first node and fix it completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, 0, 0.0), dir1, dir2, dir3);
    mesh->AddNode(nodeA);

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(dx * i, 0, 0), dir1, dir2, dir3);
        mesh->AddNode(nodeB);

        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB);
        element->SetDimensions(dx, thickness, width);
        element->SetMaterial(material);
        element->SetAlphaDamp(0.01);
        mesh->AddElement(element);

        nodeA = nodeB;
    }

    m_system->Update();  // Need to call all the element SetupInital() functions
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
void ANCFBeam3243Test<num_elements, ElementVersion, MaterialVersion>::PerturbNodes(const bool Use_OMP) {
    auto MeshList = m_system->Get_meshlist();
    for (auto& Mesh : MeshList) {
        auto NodeList = Mesh->GetNodes();
        ChVector<double> Perturbation;
        if (Use_OMP) {
#pragma omp parallel for
            for (int in = 0; in < NodeList.size(); in++) {
                Perturbation.eigen().Random();
                // auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzDDD>(NodeList[in]);
                auto Node = std::static_pointer_cast<ChNodeFEAxyzDDD>(NodeList[in]);
                Node->SetPos(Node->GetPos() + 1e-6 * Perturbation);
                Node->SetD(Node->GetD() + 1e-6 * Perturbation);
                Node->SetDD(Node->GetDD() + 1e-6 * Perturbation);
                Node->SetDDD(Node->GetDDD() + 1e-6 * Perturbation);
            }
        } else {
            for (unsigned int in = 0; in < NodeList.size(); in++) {
                Perturbation.eigen().Random();
                // auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzDDD>(NodeList[in]);
                auto Node = std::static_pointer_cast<ChNodeFEAxyzDDD>(NodeList[in]);
                Node->SetPos(Node->GetPos() + 1e-6 * Perturbation);
                Node->SetD(Node->GetD() + 1e-6 * Perturbation);
                Node->SetDD(Node->GetDD() + 1e-6 * Perturbation);
                Node->SetDDD(Node->GetDDD() + 1e-6 * Perturbation);
            }
        }
    }
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
double ANCFBeam3243Test<num_elements, ElementVersion, MaterialVersion>::GetInternalFrc(const bool Use_OMP) {
    ChTimer<> timer_internal_forces;
    timer_internal_forces.reset();

    ChVectorDynamic<double> Fi(24);

    auto MeshList = m_system->Get_meshlist();
    for (auto& Mesh : MeshList) {
        auto ElementList = Mesh->GetElements();
        timer_internal_forces.start();

        if (Use_OMP) {
#pragma omp parallel for
            for (int ie = 0; ie < ElementList.size(); ie++) {
                ElementList[ie]->ComputeInternalForces(Fi);
            }
        } else {
            for (unsigned int ie = 0; ie < ElementList.size(); ie++) {
                ElementList[ie]->ComputeInternalForces(Fi);
            }
        }

        timer_internal_forces.stop();
    }
    return (timer_internal_forces());
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
double ANCFBeam3243Test<num_elements, ElementVersion, MaterialVersion>::GetJacobian(const bool Use_OMP) {
    ChTimer<> timer_KRM;
    timer_KRM.reset();

    ChMatrixNM<double, 24, 24> H;

    auto MeshList = m_system->Get_meshlist();
    for (auto& Mesh : MeshList) {
        auto ElementList = Mesh->GetElements();
        timer_KRM.start();

        if (Use_OMP) {
#pragma omp parallel for
            for (int ie = 0; ie < ElementList.size(); ie++) {
                ElementList[ie]->ComputeKRMmatricesGlobal(H, 1.0, 1.0, 1.0);
            }
        } else {
            for (unsigned int ie = 0; ie < ElementList.size(); ie++) {
                ElementList[ie]->ComputeKRMmatricesGlobal(H, 1.0, 1.0, 1.0);
            }
        }

        timer_KRM.stop();
    }
    return (timer_KRM());
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
void ANCFBeam3243Test<num_elements, ElementVersion, MaterialVersion>::PrintTimingResults(const std::string& TestName,
                                                                                         int steps) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Test Name, Num Els, Steps, Threads, ";
    std::cout << "IntFrc Max(us), IntFrc Min(us), IntFrc Mean(us), IntFrc StDev(us), ";
    std::cout << "KRM Max(us), KRM Min(us), KRM Mean(us), KRM StDev(us), " << std::endl;
    std::cout << std::flush;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Times;
    Times.resize(steps, 2);

    // Prime the test in case the internal force calcs are also needed for the Jacobian calcs
    double TimeInternalFrc = GetInternalFrc(false);
    double TimeKRM = GetJacobian(false);

    ChTimer<> Timer_Total;
    Timer_Total.reset();
    Timer_Total.start();

    // Run Single Threaded Tests
    for (auto i = 0; i < steps; i++) {
        PerturbNodes(false);
        Times(i, 0) =
            GetInternalFrc(false) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in microseconds
    }
    for (auto i = 0; i < steps; i++) {
        PerturbNodes(false);
        Times(i, 1) = GetJacobian(false) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in
                                                                            // microseconds
    }

    std::cout << TestName << ", " << num_elements << ", " << steps << ", 0, " << Times.col(0).maxCoeff() << ", "
              << Times.col(0).minCoeff() << ", " << Times.col(0).mean() << ", "
              << std::sqrt((Times.col(0).array() - Times.col(0).mean()).square().sum() / (Times.col(0).size() - 1))
              << ", " << Times.col(1).maxCoeff() << ", " << Times.col(1).minCoeff() << ", " << Times.col(1).mean()
              << ", "
              << std::sqrt((Times.col(1).array() - Times.col(1).mean()).square().sum() / (Times.col(1).size() - 1))
              << std::endl;
    std::cout << std::flush;

    // Run Multi-Threaded Tests

    int MaxThreads = 1;
    MaxThreads = ChOMP::GetNumProcs();

    int NumThreads = 1;
    bool run = true;

    int RunNum = 1;
    while (run) {
        ChOMP::SetNumThreads(NumThreads);

        for (auto i = 0; i < steps; i++) {
            PerturbNodes(true);
            Times(i, 0) =
                GetInternalFrc(true) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in microseconds
        }
        for (auto i = 0; i < steps; i++) {
            PerturbNodes(true);
            Times(i, 1) =
                GetJacobian(true) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in microseconds
        }
        std::cout << TestName << ", " << num_elements << ", " << steps << ", " << NumThreads << ", "
                  << Times.col(0).maxCoeff() << ", " << Times.col(0).minCoeff() << ", " << Times.col(0).mean() << ", "
                  << std::sqrt((Times.col(0).array() - Times.col(0).mean()).square().sum() / (Times.col(0).size() - 1))
                  << ", " << Times.col(1).maxCoeff() << ", " << Times.col(1).minCoeff() << ", " << Times.col(1).mean()
                  << ", "
                  << std::sqrt((Times.col(1).array() - Times.col(1).mean()).square().sum() / (Times.col(1).size() - 1))
                  << std::endl;
        std::cout << std::flush;

        if (NumThreads == MaxThreads)
            run = false;

        if (NumThreads <= 4)
            NumThreads *= 2;
        else  // Since computers this will be run on have a number of cores that is a multiple of 4
            NumThreads += 4;

        if (NumThreads > MaxThreads)
            NumThreads = MaxThreads;
    }

    Timer_Total.stop();
    std::cout << "Total Test Time: " << Timer_Total() << "s" << std::endl;
}

void Run_ANCFBeam_3243_Tests() {
    const int num_elements = 1024;
    int num_steps = 1000;

    // const int num_elements = 8;
    // int num_steps = 1;

    {
        ANCFBeam3243Test<num_elements, ChElementBeamANCF_3243, ChMaterialBeamANCF> Beam3243Test;
        Beam3243Test.PrintTimingResults("ChElementBeamANCF_3243_Chrono7", num_steps);
    }
    {
        ANCFBeam3243Test<num_elements, ChElementBeamANCF_3243_TR01, ChMaterialBeamANCF> Beam3243Test_TR01;
        Beam3243Test_TR01.PrintTimingResults("ChElementBeamANCF_3243_TR01", num_steps);
    }
    //{
    //    ANCFBeam3243Test<num_elements, ChElementBeamANCF_3243_TR01B, ChMaterialBeamANCF> Beam3243Test_TR01B;
    //    Beam3243Test_TR01B.PrintTimingResults("ChElementBeamANCF_3243_TR01B", num_steps);
    //}
    //{
    //    ANCFBeam3243Test<num_elements, ChElementBeamANCF_3243_TR02, ChMaterialBeamANCF> Beam3243Test_TR02;
    //    Beam3243Test_TR02.PrintTimingResults("ChElementBeamANCF_3243_TR02", num_steps);
    //}
    //{
    //    ANCFBeam3243Test<num_elements, ChElementBeamANCF_3243_TR02B, ChMaterialBeamANCF> Beam3243Test_TR02B;
    //    Beam3243Test_TR02B.PrintTimingResults("ChElementBeamANCF_3243_TR02B", num_steps);
    //}
    //{
    //    ANCFBeam3243Test<num_elements, ChElementBeamANCF_3243_TR03, ChMaterialBeamANCF> Beam3243Test_TR03;
    //    Beam3243Test_TR03.PrintTimingResults("ChElementBeamANCF_3243_TR03", num_steps);
    //}
    //{
    //    ANCFBeam3243Test<num_elements, ChElementBeamANCF_3243_TR03B, ChMaterialBeamANCF> Beam3243Test_TR03B;
    //    Beam3243Test_TR03B.PrintTimingResults("ChElementBeamANCF_3243_TR03B", num_steps);
    //}
    //{
    //    ANCFBeam3243Test<num_elements, ChElementBeamANCF_3243_TR04, ChMaterialBeamANCF> Beam3243Test_TR04;
    //    Beam3243Test_TR04.PrintTimingResults("ChElementBeamANCF_3243_TR04", num_steps);
    //}
    //{
    //    ANCFBeam3243Test<num_elements, ChElementBeamANCF_3243_TR04B, ChMaterialBeamANCF> Beam3243Test_TR04B;
    //    Beam3243Test_TR04B.PrintTimingResults("ChElementBeamANCF_3243_TR04B", num_steps);
    //}
    {
        ANCFBeam3243Test<num_elements, ChElementBeamANCF_3243_TR05, ChMaterialBeamANCF> Beam3243Test_TR05;
        Beam3243Test_TR05.PrintTimingResults("ChElementBeamANCF_3243_TR05", num_steps);
    }
    {
        ANCFBeam3243Test<num_elements, ChElementBeamANCF_3243_TR05B, ChMaterialBeamANCF> Beam3243Test_TR05B;
        Beam3243Test_TR05B.PrintTimingResults("ChElementBeamANCF_3243_TR05B", num_steps);
    }
    {
        ANCFBeam3243Test<num_elements, ChElementBeamANCF_3243_TR06, ChMaterialBeamANCF> Beam3243Test_TR06;
        Beam3243Test_TR06.PrintTimingResults("ChElementBeamANCF_3243_TR06", num_steps);
    }
    {
        ANCFBeam3243Test<num_elements, ChElementBeamANCF_3243_TR06B, ChMaterialBeamANCF> Beam3243Test_TR06B;
        Beam3243Test_TR06B.PrintTimingResults("ChElementBeamANCF_3243_TR06B", num_steps);
    }
    {
        ANCFBeam3243Test<num_elements, ChElementBeamANCF_3243_TR07, ChMaterialBeamANCF> Beam3243Test_TR07;
        Beam3243Test_TR07.PrintTimingResults("ChElementBeamANCF_3243_TR07", num_steps);
    }
    {
        ANCFBeam3243Test<num_elements, ChElementBeamANCF_3243_TR07B, ChMaterialBeamANCF> Beam3243Test_TR07B;
        Beam3243Test_TR07B.PrintTimingResults("ChElementBeamANCF_3243_TR07B", num_steps);
    }
    {
        ANCFBeam3243Test<num_elements, ChElementBeamANCF_3243_TR08, ChMaterialBeamANCF> Beam3243Test_TR08;
        Beam3243Test_TR08.PrintTimingResults("ChElementBeamANCF_3243_TR08", num_steps);
    }
    {
        ANCFBeam3243Test<num_elements, ChElementBeamANCF_3243_TR08B, ChMaterialBeamANCF> Beam3243Test_TR08B;
        Beam3243Test_TR08B.PrintTimingResults("ChElementBeamANCF_3243_TR08B", num_steps);
    }
    {
        ANCFBeam3243Test<num_elements, ChElementBeamANCF_3243_TR09, ChMaterialBeamANCF> Beam3243Test_TR09;
        Beam3243Test_TR09.PrintTimingResults("ChElementBeamANCF_3243_TR09", num_steps);
    }
    {
        ANCFBeam3243Test<num_elements, ChElementBeamANCF_3243_TR09B, ChMaterialBeamANCF> Beam3243Test_TR09B;
        Beam3243Test_TR09B.PrintTimingResults("ChElementBeamANCF_3243_TR09B", num_steps);
    }
    {
        ANCFBeam3243Test<num_elements, ChElementBeamANCF_3243_TR10, ChMaterialBeamANCF> Beam3243Test_TR10;
        Beam3243Test_TR10.PrintTimingResults("ChElementBeamANCF_3243_TR10", num_steps);
    }
    {
        ANCFBeam3243Test<num_elements, ChElementBeamANCF_3243_TR10B, ChMaterialBeamANCF> Beam3243Test_TR10B;
        Beam3243Test_TR10B.PrintTimingResults("ChElementBeamANCF_3243_TR10B", num_steps);
    }
    {
        ANCFBeam3243Test<num_elements, ChElementBeamANCF_3243_TR11, ChMaterialBeamANCF> Beam3243Test_TR11;
        Beam3243Test_TR11.PrintTimingResults("ChElementBeamANCF_3243_TR11", num_steps);
    }
    {
        ANCFBeam3243Test<num_elements, ChElementBeamANCF_3243_TR11B, ChMaterialBeamANCF> Beam3243Test_TR11B;
        Beam3243Test_TR11B.PrintTimingResults("ChElementBeamANCF_3243_TR11B", num_steps);
    }
    //{
    //    ANCFBeam3243Test<num_elements, ChElementBeamANCF_3243, ChMaterialBeamANCF> Beam3243Test;
    //    Beam3243Test.PrintTimingResults("ChElementBeamANCF_3243_Chrono7", num_steps);
    //}
}

// =============================================================================

template <int num_elements, typename ElementVersion, typename MaterialVersion>
class ANCFBeam3333Test {
  public:
    ANCFBeam3333Test();

    ~ANCFBeam3333Test() { delete m_system; }

    ChSystem* GetSystem() { return m_system; }

    void PerturbNodes(const bool Use_OMP);
    double GetInternalFrc(const bool Use_OMP);
    double GetJacobian(const bool Use_OMP);

    void PrintTimingResults(const std::string& TestName, int steps);

  protected:
    ChSystemSMC* m_system;
};

template <int num_elements, typename ElementVersion, typename MaterialVersion>
ANCFBeam3333Test<num_elements, ElementVersion, MaterialVersion>::ANCFBeam3333Test() {
    m_system = new ChSystemSMC();
    m_system->Set_G_acc(ChVector<>(0, 0, -9.80665));

    // Set solver parameters
    auto solver = chrono_types::make_shared<ChSolverSparseLU>();
    solver->UseSparsityPatternLearner(true);
    solver->LockSparsityPattern(true);
    solver->SetVerbose(false);
    m_system->SetSolver(solver);

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
    double k1 =
        10 * (1 + nu) / (12 + 11 * nu);  // Timoshenko shear correction coefficient for a rectangular cross-section
    double k2 = k1;                      // Timoshenko shear correction coefficient for a rectangular cross-section

    auto material = chrono_types::make_shared<MaterialVersion>(rho, E, nu, k1, k2);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    m_system->Add(mesh);

    // Populate the mesh container with a the nodes and elements for the meshed beam
    int num_nodes = (2 * num_elements) + 1;
    double dx = length / (num_nodes - 1);

    // Setup beam cross section gradients to initially align with the global y and z directions
    ChVector<> dir1(0, 1, 0);
    ChVector<> dir2(0, 0, 1);

    // Create the first node and fix it completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0, 0.0), dir1, dir2);
    mesh->AddNode(nodeA);

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(dx * (2 * i), 0, 0), dir1, dir2);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(dx * (2 * i - 1), 0, 0), dir1, dir2);
        mesh->AddNode(nodeB);
        mesh->AddNode(nodeC);

        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC);
        element->SetDimensions(2 * dx, thickness, width);
        element->SetMaterial(material);
        element->SetAlphaDamp(0.01);
        mesh->AddElement(element);

        nodeA = nodeB;
    }

    m_system->Update();  // Need to call all the element SetupInital() functions
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
void ANCFBeam3333Test<num_elements, ElementVersion, MaterialVersion>::PerturbNodes(const bool Use_OMP) {
    auto MeshList = m_system->Get_meshlist();
    for (auto& Mesh : MeshList) {
        auto NodeList = Mesh->GetNodes();
        ChVector<double> Perturbation;
        if (Use_OMP) {
#pragma omp parallel for
            for (int in = 0; in < NodeList.size(); in++) {
                Perturbation.eigen().Random();
                // auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzDD>(NodeList[in]);
                auto Node = std::static_pointer_cast<ChNodeFEAxyzDD>(NodeList[in]);
                Node->SetPos(Node->GetPos() + 1e-6 * Perturbation);
                Node->SetD(Node->GetD() + 1e-6 * Perturbation);
                Node->SetDD(Node->GetDD() + 1e-6 * Perturbation);
            }
        } else {
            for (unsigned int in = 0; in < NodeList.size(); in++) {
                Perturbation.eigen().Random();
                // auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzDD>(NodeList[in]);
                auto Node = std::static_pointer_cast<ChNodeFEAxyzDD>(NodeList[in]);
                Node->SetPos(Node->GetPos() + 1e-6 * Perturbation);
                Node->SetD(Node->GetD() + 1e-6 * Perturbation);
                Node->SetDD(Node->GetDD() + 1e-6 * Perturbation);
            }
        }
    }
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
double ANCFBeam3333Test<num_elements, ElementVersion, MaterialVersion>::GetInternalFrc(const bool Use_OMP) {
    ChTimer<> timer_internal_forces;
    timer_internal_forces.reset();

    ChVectorDynamic<double> Fi(27);

    auto MeshList = m_system->Get_meshlist();
    for (auto& Mesh : MeshList) {
        auto ElementList = Mesh->GetElements();
        timer_internal_forces.start();

        if (Use_OMP) {
#pragma omp parallel for
            for (int ie = 0; ie < ElementList.size(); ie++) {
                ElementList[ie]->ComputeInternalForces(Fi);
            }
        } else {
            for (unsigned int ie = 0; ie < ElementList.size(); ie++) {
                ElementList[ie]->ComputeInternalForces(Fi);
            }
        }

        timer_internal_forces.stop();
    }
    return (timer_internal_forces());
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
double ANCFBeam3333Test<num_elements, ElementVersion, MaterialVersion>::GetJacobian(const bool Use_OMP) {
    ChTimer<> timer_KRM;
    timer_KRM.reset();

    ChMatrixNM<double, 27, 27> H;

    auto MeshList = m_system->Get_meshlist();
    for (auto& Mesh : MeshList) {
        auto ElementList = Mesh->GetElements();
        timer_KRM.start();

        if (Use_OMP) {
#pragma omp parallel for
            for (int ie = 0; ie < ElementList.size(); ie++) {
                ElementList[ie]->ComputeKRMmatricesGlobal(H, 1.0, 1.0, 1.0);
            }
        } else {
            for (unsigned int ie = 0; ie < ElementList.size(); ie++) {
                ElementList[ie]->ComputeKRMmatricesGlobal(H, 1.0, 1.0, 1.0);
            }
        }

        timer_KRM.stop();
    }
    return (timer_KRM());
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
void ANCFBeam3333Test<num_elements, ElementVersion, MaterialVersion>::PrintTimingResults(const std::string& TestName,
                                                                                         int steps) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Test Name, Num Els, Steps, Threads, ";
    std::cout << "IntFrc Max(us), IntFrc Min(us), IntFrc Mean(us), IntFrc StDev(us), ";
    std::cout << "KRM Max(us), KRM Min(us), KRM Mean(us), KRM StDev(us), " << std::endl;
    std::cout << std::flush;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Times;
    Times.resize(steps, 2);

    // Prime the test in case the internal force calcs are also needed for the Jacobian calcs
    double TimeInternalFrc = GetInternalFrc(false);
    double TimeKRM = GetJacobian(false);

    ChTimer<> Timer_Total;
    Timer_Total.reset();
    Timer_Total.start();

    // Run Single Threaded Tests
    for (auto i = 0; i < steps; i++) {
        PerturbNodes(false);
        Times(i, 0) =
            GetInternalFrc(false) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in microseconds
    }
    for (auto i = 0; i < steps; i++) {
        PerturbNodes(false);
        Times(i, 1) = GetJacobian(false) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in
                                                                            // microseconds
    }

    std::cout << TestName << ", " << num_elements << ", " << steps << ", 0, " << Times.col(0).maxCoeff() << ", "
              << Times.col(0).minCoeff() << ", " << Times.col(0).mean() << ", "
              << std::sqrt((Times.col(0).array() - Times.col(0).mean()).square().sum() / (Times.col(0).size() - 1))
              << ", " << Times.col(1).maxCoeff() << ", " << Times.col(1).minCoeff() << ", " << Times.col(1).mean()
              << ", "
              << std::sqrt((Times.col(1).array() - Times.col(1).mean()).square().sum() / (Times.col(1).size() - 1))
              << std::endl;
    std::cout << std::flush;

    // Run Multi-Threaded Tests

    int MaxThreads = 1;
    MaxThreads = ChOMP::GetNumProcs();

    int NumThreads = 1;
    bool run = true;

    int RunNum = 1;
    while (run) {
        ChOMP::SetNumThreads(NumThreads);

        for (auto i = 0; i < steps; i++) {
            PerturbNodes(true);
            Times(i, 0) =
                GetInternalFrc(true) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in microseconds
        }
        for (auto i = 0; i < steps; i++) {
            PerturbNodes(true);
            Times(i, 1) =
                GetJacobian(true) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in microseconds
        }
        std::cout << TestName << ", " << num_elements << ", " << steps << ", " << NumThreads << ", "
                  << Times.col(0).maxCoeff() << ", " << Times.col(0).minCoeff() << ", " << Times.col(0).mean() << ", "
                  << std::sqrt((Times.col(0).array() - Times.col(0).mean()).square().sum() / (Times.col(0).size() - 1))
                  << ", " << Times.col(1).maxCoeff() << ", " << Times.col(1).minCoeff() << ", " << Times.col(1).mean()
                  << ", "
                  << std::sqrt((Times.col(1).array() - Times.col(1).mean()).square().sum() / (Times.col(1).size() - 1))
                  << std::endl;
        std::cout << std::flush;

        if (NumThreads == MaxThreads)
            run = false;

        if (NumThreads <= 4)
            NumThreads *= 2;
        else  // Since computers this will be run on have a number of cores that is a multiple of 4
            NumThreads += 4;

        if (NumThreads > MaxThreads)
            NumThreads = MaxThreads;
    }

    Timer_Total.stop();
    std::cout << "Total Test Time: " << Timer_Total() << "s" << std::endl;
}

void Run_ANCFBeam_3333_Tests() {
    const int num_elements = 1024;
    int num_steps = 1000;

    // const int num_elements = 8;
    // int num_steps = 1;

    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333, ChMaterialBeamANCF> Beam3333Test;
        Beam3333Test.PrintTimingResults("ChElementBeamANCF_3333_Chrono7", num_steps);
    }
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR01, ChMaterialBeamANCF> Beam3333Test_TR01;
        Beam3333Test_TR01.PrintTimingResults("ChElementBeamANCF_3333_TR01", num_steps);
    }
    //{
    //    ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR01B, ChMaterialBeamANCF> Beam3333Test_TR01B;
    //    Beam3333Test_TR01B.PrintTimingResults("ChElementBeamANCF_3333_TR01B", num_steps);
    //}
    //{
    //    ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR02, ChMaterialBeamANCF> Beam3333Test_TR02;
    //    Beam3333Test_TR02.PrintTimingResults("ChElementBeamANCF_3333_TR02", num_steps);
    //}
    //{
    //    ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR02B, ChMaterialBeamANCF> Beam3333Test_TR02B;
    //    Beam3333Test_TR02B.PrintTimingResults("ChElementBeamANCF_3333_TR02B", num_steps);
    //}
    //{
    //    ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR03, ChMaterialBeamANCF> Beam3333Test_TR03;
    //    Beam3333Test_TR03.PrintTimingResults("ChElementBeamANCF_3333_TR03", num_steps);
    //}
    //{
    //    ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR03B, ChMaterialBeamANCF> Beam3333Test_TR03B;
    //    Beam3333Test_TR03B.PrintTimingResults("ChElementBeamANCF_3333_TR03B", num_steps);
    //}
    //{
    //    ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR04, ChMaterialBeamANCF> Beam3333Test_TR04;
    //    Beam3333Test_TR04.PrintTimingResults("ChElementBeamANCF_3333_TR04", num_steps);
    //}
    //{
    //    ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR04B, ChMaterialBeamANCF> Beam3333Test_TR04B;
    //    Beam3333Test_TR04B.PrintTimingResults("ChElementBeamANCF_3333_TR04B", num_steps);
    //}
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR05, ChMaterialBeamANCF> Beam3333Test_TR05;
        Beam3333Test_TR05.PrintTimingResults("ChElementBeamANCF_3333_TR05", num_steps);
    }
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR05B, ChMaterialBeamANCF> Beam3333Test_TR05B;
        Beam3333Test_TR05B.PrintTimingResults("ChElementBeamANCF_3333_TR05B", num_steps);
    }
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR06, ChMaterialBeamANCF> Beam3333Test_TR06;
        Beam3333Test_TR06.PrintTimingResults("ChElementBeamANCF_3333_TR06", num_steps);
    }
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR06B, ChMaterialBeamANCF> Beam3333Test_TR06B;
        Beam3333Test_TR06B.PrintTimingResults("ChElementBeamANCF_3333_TR06B", num_steps);
    }
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR07, ChMaterialBeamANCF> Beam3333Test_TR07;
        Beam3333Test_TR07.PrintTimingResults("ChElementBeamANCF_3333_TR07", num_steps);
    }
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR07B, ChMaterialBeamANCF> Beam3333Test_TR07B;
        Beam3333Test_TR07B.PrintTimingResults("ChElementBeamANCF_3333_TR07B", num_steps);
    }
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR08, ChMaterialBeamANCF> Beam3333Test_TR08;
        Beam3333Test_TR08.PrintTimingResults("ChElementBeamANCF_3333_TR08", num_steps);
    }
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR08B, ChMaterialBeamANCF> Beam3333Test_TR08B;
        Beam3333Test_TR08B.PrintTimingResults("ChElementBeamANCF_3333_TR08B", num_steps);
    }
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR09, ChMaterialBeamANCF> Beam3333Test_TR09;
        Beam3333Test_TR09.PrintTimingResults("ChElementBeamANCF_3333_TR09", num_steps);
    }
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR09B, ChMaterialBeamANCF> Beam3333Test_TR09B;
        Beam3333Test_TR09B.PrintTimingResults("ChElementBeamANCF_3333_TR09B", num_steps);
    }
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR10, ChMaterialBeamANCF> Beam3333Test_TR10;
        Beam3333Test_TR10.PrintTimingResults("ChElementBeamANCF_3333_TR10", num_steps);
    }
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR10B, ChMaterialBeamANCF> Beam3333Test_TR10B;
        Beam3333Test_TR10B.PrintTimingResults("ChElementBeamANCF_3333_TR10B", num_steps);
    }
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR11, ChMaterialBeamANCF> Beam3333Test_TR11;
        Beam3333Test_TR11.PrintTimingResults("ChElementBeamANCF_3333_TR11", num_steps);
    }
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR11B, ChMaterialBeamANCF> Beam3333Test_TR11B;
        Beam3333Test_TR11B.PrintTimingResults("ChElementBeamANCF_3333_TR11B", num_steps);
    }
    //{
    //    ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333, ChMaterialBeamANCF> Beam3333Test;
    //    Beam3333Test.PrintTimingResults("ChElementBeamANCF_3333_Chrono7", num_steps);
    //}
}

// =============================================================================

template <int num_elements, typename ElementVersion, typename MaterialVersion>
class ANCFShell3443Test {
  public:
    ANCFShell3443Test(unsigned int num_layers);

    ~ANCFShell3443Test() { delete m_system; }

    ChSystem* GetSystem() { return m_system; }

    void PerturbNodes(const bool Use_OMP);
    double GetInternalFrc(const bool Use_OMP);
    double GetJacobian(const bool Use_OMP);

    void PrintTimingResults(const std::string& TestName, int steps);

  protected:
    ChSystemSMC* m_system;
    unsigned int m_num_layers;
};

template <int num_elements, typename ElementVersion, typename MaterialVersion>
ANCFShell3443Test<num_elements, ElementVersion, MaterialVersion>::ANCFShell3443Test(unsigned int num_layers) {
    m_num_layers = num_layers;
    m_system = new ChSystemSMC();
    m_system->Set_G_acc(ChVector<>(0, 0, -9.80665));

    // Set solver parameters
    auto solver = chrono_types::make_shared<ChSolverSparseLU>();
    solver->UseSparsityPatternLearner(true);
    solver->LockSparsityPattern(true);
    solver->SetVerbose(false);
    m_system->SetSolver(solver);

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
    double length = 10.0;  // m
    double width = 0.1;    // m
    double height = 0.01;  // m
    // Aluminum 7075-T651
    double rho = 2810;  // kg/m^3
    double E = 71.7e9;  // Pa
    double nu = 0.33;
    double G = E / (2 * (1 + nu));

    auto material = chrono_types::make_shared<MaterialVersion>(rho, E, nu);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    m_system->Add(mesh);

    // Populate the mesh container with a the nodes and elements for the meshed beam
    int num_nodes = 2 * num_elements + 2;
    double dx = length / (num_elements);

    // Setup beam cross section gradients to initially align with the global y and z directions
    ChVector<> dir1(1, 0, 0);
    ChVector<> dir2(0, 1, 0);
    ChVector<> dir3(0, 0, 1);

    // Create the first node and fix it completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, 0, 0.0), dir1, dir2, dir3);
    mesh->AddNode(nodeA);
    auto nodeD = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, width, 0.0), dir1, dir2, dir3);
    mesh->AddNode(nodeD);

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(dx * i, 0, 0), dir1, dir2, dir3);
        mesh->AddNode(nodeB);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(dx * i, width, 0), dir1, dir2, dir3);
        mesh->AddNode(nodeC);

        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC, nodeD);
        element->SetDimensions(dx, width);
        for (int j = 0; j < m_num_layers; j++) {
            element->AddLayer(height, 0 * CH_C_DEG_TO_RAD, material);
        }
        element->SetAlphaDamp(0.01);
        mesh->AddElement(element);

        nodeA = nodeB;
        nodeD = nodeC;
    }

    m_system->Update();  // Need to call all the element SetupInital() functions
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
void ANCFShell3443Test<num_elements, ElementVersion, MaterialVersion>::PerturbNodes(const bool Use_OMP) {
    auto MeshList = m_system->Get_meshlist();
    for (auto& Mesh : MeshList) {
        auto NodeList = Mesh->GetNodes();
        ChVector<double> Perturbation;
        if (Use_OMP) {
#pragma omp parallel for
            for (int in = 0; in < NodeList.size(); in++) {
                Perturbation.eigen().Random();
                // auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzDDD>(NodeList[in]);
                auto Node = std::static_pointer_cast<ChNodeFEAxyzDDD>(NodeList[in]);
                Node->SetPos(Node->GetPos() + 1e-6 * Perturbation);
                Node->SetD(Node->GetD() + 1e-6 * Perturbation);
                Node->SetDD(Node->GetDD() + 1e-6 * Perturbation);
                Node->SetDDD(Node->GetDDD() + 1e-6 * Perturbation);
            }
        } else {
            for (unsigned int in = 0; in < NodeList.size(); in++) {
                Perturbation.eigen().Random();
                // auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzDDD>(NodeList[in]);
                auto Node = std::static_pointer_cast<ChNodeFEAxyzDDD>(NodeList[in]);
                Node->SetPos(Node->GetPos() + 1e-6 * Perturbation);
                Node->SetD(Node->GetD() + 1e-6 * Perturbation);
                Node->SetDD(Node->GetDD() + 1e-6 * Perturbation);
                Node->SetDDD(Node->GetDDD() + 1e-6 * Perturbation);
            }
        }
    }
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
double ANCFShell3443Test<num_elements, ElementVersion, MaterialVersion>::GetInternalFrc(const bool Use_OMP) {
    ChTimer<> timer_internal_forces;
    timer_internal_forces.reset();

    ChVectorDynamic<double> Fi(48);

    auto MeshList = m_system->Get_meshlist();
    for (auto& Mesh : MeshList) {
        auto ElementList = Mesh->GetElements();
        timer_internal_forces.start();

        if (Use_OMP) {
#pragma omp parallel for
            for (int ie = 0; ie < ElementList.size(); ie++) {
                ElementList[ie]->ComputeInternalForces(Fi);
            }
        } else {
            for (unsigned int ie = 0; ie < ElementList.size(); ie++) {
                ElementList[ie]->ComputeInternalForces(Fi);
            }
        }

        timer_internal_forces.stop();
    }
    return (timer_internal_forces());
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
double ANCFShell3443Test<num_elements, ElementVersion, MaterialVersion>::GetJacobian(const bool Use_OMP) {
    ChTimer<> timer_KRM;
    timer_KRM.reset();

    ChMatrixNM<double, 48, 48> H;

    auto MeshList = m_system->Get_meshlist();
    for (auto& Mesh : MeshList) {
        auto ElementList = Mesh->GetElements();
        timer_KRM.start();

        if (Use_OMP) {
#pragma omp parallel for
            for (int ie = 0; ie < ElementList.size(); ie++) {
                ElementList[ie]->ComputeKRMmatricesGlobal(H, 1.0, 1.0, 1.0);
            }
        } else {
            for (unsigned int ie = 0; ie < ElementList.size(); ie++) {
                ElementList[ie]->ComputeKRMmatricesGlobal(H, 1.0, 1.0, 1.0);
            }
        }

        timer_KRM.stop();
    }
    return (timer_KRM());
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
void ANCFShell3443Test<num_elements, ElementVersion, MaterialVersion>::PrintTimingResults(const std::string& TestName,
                                                                                          int steps) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Test Name, Num Els, Steps, Threads, ";
    std::cout << "IntFrc Max(us), IntFrc Min(us), IntFrc Mean(us), IntFrc StDev(us), ";
    std::cout << "KRM Max(us), KRM Min(us), KRM Mean(us), KRM StDev(us), " << std::endl;
    std::cout << std::flush;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Times;
    Times.resize(steps, 2);

    // Prime the test in case the internal force calcs are also needed for the Jacobian calcs
    double TimeInternalFrc = GetInternalFrc(false);
    double TimeKRM = GetJacobian(false);

    ChTimer<> Timer_Total;
    Timer_Total.reset();
    Timer_Total.start();

    // Run Single Threaded Tests
    for (auto i = 0; i < steps; i++) {
        PerturbNodes(false);
        Times(i, 0) =
            GetInternalFrc(false) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in microseconds
    }
    for (auto i = 0; i < steps; i++) {
        PerturbNodes(false);
        Times(i, 1) = GetJacobian(false) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in
                                                                            // microseconds
    }

    std::cout << TestName << ", " << num_elements << ", " << steps << ", 0, " << Times.col(0).maxCoeff() << ", "
              << Times.col(0).minCoeff() << ", " << Times.col(0).mean() << ", "
              << std::sqrt((Times.col(0).array() - Times.col(0).mean()).square().sum() / (Times.col(0).size() - 1))
              << ", " << Times.col(1).maxCoeff() << ", " << Times.col(1).minCoeff() << ", " << Times.col(1).mean()
              << ", "
              << std::sqrt((Times.col(1).array() - Times.col(1).mean()).square().sum() / (Times.col(1).size() - 1))
              << std::endl;
    std::cout << std::flush;

    // Run Multi-Threaded Tests

    int MaxThreads = 1;
    MaxThreads = ChOMP::GetNumProcs();

    int NumThreads = 1;
    bool run = true;

    int RunNum = 1;
    while (run) {
        ChOMP::SetNumThreads(NumThreads);

        for (auto i = 0; i < steps; i++) {
            PerturbNodes(true);
            Times(i, 0) =
                GetInternalFrc(true) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in microseconds
        }
        for (auto i = 0; i < steps; i++) {
            PerturbNodes(true);
            Times(i, 1) =
                GetJacobian(true) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in microseconds
        }
        std::cout << TestName << ", " << num_elements << ", " << steps << ", " << NumThreads << ", "
                  << Times.col(0).maxCoeff() << ", " << Times.col(0).minCoeff() << ", " << Times.col(0).mean() << ", "
                  << std::sqrt((Times.col(0).array() - Times.col(0).mean()).square().sum() / (Times.col(0).size() - 1))
                  << ", " << Times.col(1).maxCoeff() << ", " << Times.col(1).minCoeff() << ", " << Times.col(1).mean()
                  << ", "
                  << std::sqrt((Times.col(1).array() - Times.col(1).mean()).square().sum() / (Times.col(1).size() - 1))
                  << std::endl;
        std::cout << std::flush;

        if (NumThreads == MaxThreads)
            run = false;

        if (NumThreads <= 4)
            NumThreads *= 2;
        else  // Since computers this will be run on have a number of cores that is a multiple of 4
            NumThreads += 4;

        if (NumThreads > MaxThreads)
            NumThreads = MaxThreads;
    }

    Timer_Total.stop();
    std::cout << "Total Test Time: " << Timer_Total() << "s" << std::endl;
}

void Run_ANCFShell_3443_Tests() {
    const int num_elements = 1024;
    int num_steps = 100;

    // const int num_elements = 8;
    // int num_steps = 1;

    {
        ANCFShell3443Test<num_elements, ChElementShellANCF_3443, ChMaterialShellANCF> Shell3443Test(1);
        Shell3443Test.PrintTimingResults("ChElementShellANCF_3443_Chrono7", num_steps);
    }
    {
        ANCFShell3443Test<num_elements, ChElementShellANCF_3443_TR01, ChMaterialShellANCF> Shell3443Test_TR01(1);
        Shell3443Test_TR01.PrintTimingResults("ChElementShellANCF_3443_TR01", num_steps);
    }
    //{
    //    ANCFShell3443Test<num_elements, ChElementShellANCF_3443_TR01B, ChMaterialShellANCF> Shell3443Test_TR01B(1);
    //    Shell3443Test_TR01B.PrintTimingResults("ChElementShellANCF_3443_TR01B", num_steps);
    //}
    //{
    //    ANCFShell3443Test<num_elements, ChElementShellANCF_3443_TR02, ChMaterialShellANCF> Shell3443Test_TR02(1);
    //    Shell3443Test_TR02.PrintTimingResults("ChElementShellANCF_3443_TR02", num_steps);
    //}
    //{
    //    ANCFShell3443Test<num_elements, ChElementShellANCF_3443_TR02B, ChMaterialShellANCF> Shell3443Test_TR02B(1);
    //    Shell3443Test_TR02B.PrintTimingResults("ChElementShellANCF_3443_TR02B", num_steps);
    //}
    //{
    //    ANCFShell3443Test<num_elements, ChElementShellANCF_3443_TR03, ChMaterialShellANCF> Shell3443Test_TR03(1);
    //    Shell3443Test_TR03.PrintTimingResults("ChElementShellANCF_3443_TR03", num_steps);
    //}
    //{
    //    ANCFShell3443Test<num_elements, ChElementShellANCF_3443_TR03B, ChMaterialShellANCF> Shell3443Test_TR03B(1);
    //    Shell3443Test_TR03B.PrintTimingResults("ChElementShellANCF_3443_TR03B", num_steps);
    //}
    //{
    //    ANCFShell3443Test<num_elements, ChElementShellANCF_3443_TR04, ChMaterialShellANCF> Shell3443Test_TR04(1);
    //    Shell3443Test_TR04.PrintTimingResults("ChElementShellANCF_3443_TR04", num_steps);
    //}
    //{
    //    ANCFShell3443Test<num_elements, ChElementShellANCF_3443_TR04B, ChMaterialShellANCF> Shell3443Test_TR04B(1);
    //    Shell3443Test_TR04B.PrintTimingResults("ChElementShellANCF_3443_TR04B", num_steps);
    //}
    {
        ANCFShell3443Test<num_elements, ChElementShellANCF_3443_TR05, ChMaterialShellANCF> Shell3443Test_TR05(1);
        Shell3443Test_TR05.PrintTimingResults("ChElementShellANCF_3443_TR05", num_steps);
    }
    {
        ANCFShell3443Test<num_elements, ChElementShellANCF_3443_TR05B, ChMaterialShellANCF> Shell3443Test_TR05B(1);
        Shell3443Test_TR05B.PrintTimingResults("ChElementShellANCF_3443_TR05B", num_steps);
    }
    {
        ANCFShell3443Test<num_elements, ChElementShellANCF_3443_TR06, ChMaterialShellANCF> Shell3443Test_TR06(1);
        Shell3443Test_TR06.PrintTimingResults("ChElementShellANCF_3443_TR06", num_steps);
    }
    {
        ANCFShell3443Test<num_elements, ChElementShellANCF_3443_TR06B, ChMaterialShellANCF> Shell3443Test_TR06B(1);
        Shell3443Test_TR06B.PrintTimingResults("ChElementShellANCF_3443_TR06B", num_steps);
    }
    {
        ANCFShell3443Test<num_elements, ChElementShellANCF_3443_TR07, ChMaterialShellANCF> Shell3443Test_TR07(1);
        Shell3443Test_TR07.PrintTimingResults("ChElementShellANCF_3443_TR07", num_steps);
    }
    {
        ANCFShell3443Test<num_elements, ChElementShellANCF_3443_TR07B, ChMaterialShellANCF> Shell3443Test_TR07B(1);
        Shell3443Test_TR07B.PrintTimingResults("ChElementShellANCF_3443_TR07B", num_steps);
    }
    {
        ANCFShell3443Test<num_elements, ChElementShellANCF_3443_TR08, ChMaterialShellANCF> Shell3443Test_TR08(1);
        Shell3443Test_TR08.PrintTimingResults("ChElementShellANCF_3443_TR08", num_steps);
    }
    {
        ANCFShell3443Test<num_elements, ChElementShellANCF_3443_TR08B, ChMaterialShellANCF> Shell3443Test_TR08B(1);
        Shell3443Test_TR08B.PrintTimingResults("ChElementShellANCF_3443_TR08B", num_steps);
    }
    {
        ANCFShell3443Test<num_elements, ChElementShellANCF_3443_TR09, ChMaterialShellANCF> Shell3443Test_TR09(1);
        Shell3443Test_TR09.PrintTimingResults("ChElementShellANCF_3443_TR09", num_steps);
    }
    {
        ANCFShell3443Test<num_elements, ChElementShellANCF_3443_TR09B, ChMaterialShellANCF> Shell3443Test_TR09B(1);
        Shell3443Test_TR09B.PrintTimingResults("ChElementShellANCF_3443_TR09B", num_steps);
    }
    {
        ANCFShell3443Test<num_elements, ChElementShellANCF_3443_TR10, ChMaterialShellANCF> Shell3443Test_TR10(1);
        Shell3443Test_TR10.PrintTimingResults("ChElementShellANCF_3443_TR10", num_steps);
    }
    {
        ANCFShell3443Test<num_elements, ChElementShellANCF_3443_TR10B, ChMaterialShellANCF> Shell3443Test_TR10B(1);
        Shell3443Test_TR10B.PrintTimingResults("ChElementShellANCF_3443_TR10B", num_steps);
    }
    {
        ANCFShell3443Test<num_elements, ChElementShellANCF_3443_TR11, ChMaterialShellANCF> Shell3443Test_TR11(1);
        Shell3443Test_TR11.PrintTimingResults("ChElementShellANCF_3443_TR11", num_steps);
    }
    {
        ANCFShell3443Test<num_elements, ChElementShellANCF_3443_TR11B, ChMaterialShellANCF> Shell3443Test_TR11B(1);
        Shell3443Test_TR11B.PrintTimingResults("ChElementShellANCF_3443_TR11B", num_steps);
    }
    //{
    //    ANCFShell3443Test<num_elements, ChElementShellANCF_3443, ChMaterialShellANCF> Shell3443Test(1);
    //    Shell3443Test.PrintTimingResults("ChElementShellANCF_3443_Chrono7", num_steps);
    //}
}

// =============================================================================

template <int num_elements, typename ElementVersion, typename MaterialVersion>
class ANCFShell3833Test {
  public:
    ANCFShell3833Test(unsigned int num_layers);

    ~ANCFShell3833Test() { delete m_system; }

    ChSystem* GetSystem() { return m_system; }

    void PerturbNodes(const bool Use_OMP);
    double GetInternalFrc(const bool Use_OMP);
    double GetJacobian(const bool Use_OMP);

    void PrintTimingResults(const std::string& TestName, int steps);

  protected:
    ChSystemSMC* m_system;
    unsigned int m_num_layers;
};

template <int num_elements, typename ElementVersion, typename MaterialVersion>
ANCFShell3833Test<num_elements, ElementVersion, MaterialVersion>::ANCFShell3833Test(unsigned int num_layers) {
    m_num_layers = num_layers;
    m_system = new ChSystemSMC();
    m_system->Set_G_acc(ChVector<>(0, 0, -9.80665));

    // Set solver parameters
    auto solver = chrono_types::make_shared<ChSolverSparseLU>();
    solver->UseSparsityPatternLearner(true);
    solver->LockSparsityPattern(true);
    solver->SetVerbose(false);
    m_system->SetSolver(solver);

    // Set up integrator
    m_system->SetTimestepperType(ChTimestepper::Type::HHT);
    auto integrator = std::static_pointer_cast<ChTimestepperHHT>(m_system->GetTimestepper());
    integrator->SetAlpha(-0.2);
    integrator->SetMaxiters(100);
    integrator->SetAbsTolerances(1e-5);
    // integrator->SetAbsTolerances(1e-3);
    integrator->SetMode(ChTimestepperHHT::POSITION);
    integrator->SetScaling(true);
    integrator->SetVerbose(false);
    integrator->SetModifiedNewton(true);

    // Mesh properties
    double length = 10.0;  // m
    double width = 0.1;    // m
    double height = 0.01;  // m
    // Aluminum 7075-T651
    double rho = 2810;  // kg/m^3
    double E = 71.7e9;  // Pa
    double nu = 0.33;
    double G = E / (2 * (1 + nu));

    auto material = chrono_types::make_shared<MaterialVersion>(rho, E, nu);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    m_system->Add(mesh);

    // Populate the mesh container with a the nodes and elements for the meshed beam
    double dx = length / (num_elements);

    // Setup shell normals to initially align with the global z direction with no curvature
    ChVector<> dir1(0, 0, 1);
    ChVector<> Curv1(0, 0, 0);

    // Create the first nodes and fix them completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0, 0.0), dir1, Curv1);
    mesh->AddNode(nodeA);
    auto nodeD = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, width, 0), dir1, Curv1);
    mesh->AddNode(nodeD);
    auto nodeH = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0.5 * width, 0), dir1, Curv1);
    mesh->AddNode(nodeH);

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i * dx, 0, 0), dir1, Curv1);
        mesh->AddNode(nodeB);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i * dx, width, 0), dir1, Curv1);
        mesh->AddNode(nodeC);
        auto nodeE = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i * dx - 0.5 * dx, 0, 0.0), dir1, Curv1);
        mesh->AddNode(nodeE);
        auto nodeF = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i * dx, 0.5 * width, 0), dir1, Curv1);
        mesh->AddNode(nodeF);
        auto nodeG = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i * dx - 0.5 * dx, width, 0), dir1, Curv1);
        mesh->AddNode(nodeG);

        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC, nodeD, nodeE, nodeF, nodeG, nodeH);
        element->SetDimensions(dx, width);
        for (int j = 0; j < m_num_layers; j++) {
            element->AddLayer(height, 0 * CH_C_DEG_TO_RAD, material);
        }
        element->SetAlphaDamp(0.01);
        mesh->AddElement(element);

        nodeA = nodeB;
        nodeD = nodeC;
        nodeH = nodeF;
    }

    m_system->Update();  // Need to call all the element SetupInital() functions
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
void ANCFShell3833Test<num_elements, ElementVersion, MaterialVersion>::PerturbNodes(const bool Use_OMP) {
    auto MeshList = m_system->Get_meshlist();
    for (auto& Mesh : MeshList) {
        auto NodeList = Mesh->GetNodes();
        ChVector<double> Perturbation;
        if (Use_OMP) {
#pragma omp parallel for
            for (int in = 0; in < NodeList.size(); in++) {
                Perturbation.eigen().Random();
                // auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzDD>(NodeList[in]);
                auto Node = std::static_pointer_cast<ChNodeFEAxyzDD>(NodeList[in]);
                Node->SetPos(Node->GetPos() + 1e-6 * Perturbation);
                Node->SetD(Node->GetD() + 1e-6 * Perturbation);
                Node->SetDD(Node->GetDD() + 1e-6 * Perturbation);
            }
        } else {
            for (unsigned int in = 0; in < NodeList.size(); in++) {
                Perturbation.eigen().Random();
                // auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzDD>(NodeList[in]);
                auto Node = std::static_pointer_cast<ChNodeFEAxyzDD>(NodeList[in]);
                Node->SetPos(Node->GetPos() + 1e-6 * Perturbation);
                Node->SetD(Node->GetD() + 1e-6 * Perturbation);
                Node->SetDD(Node->GetDD() + 1e-6 * Perturbation);
            }
        }
    }
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
double ANCFShell3833Test<num_elements, ElementVersion, MaterialVersion>::GetInternalFrc(const bool Use_OMP) {
    ChTimer<> timer_internal_forces;
    timer_internal_forces.reset();

    ChVectorDynamic<double> Fi(72);

    auto MeshList = m_system->Get_meshlist();
    for (auto& Mesh : MeshList) {
        auto ElementList = Mesh->GetElements();
        timer_internal_forces.start();

        if (Use_OMP) {
#pragma omp parallel for
            for (int ie = 0; ie < ElementList.size(); ie++) {
                ElementList[ie]->ComputeInternalForces(Fi);
            }
        } else {
            for (unsigned int ie = 0; ie < ElementList.size(); ie++) {
                ElementList[ie]->ComputeInternalForces(Fi);
            }
        }

        timer_internal_forces.stop();
    }
    return (timer_internal_forces());
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
double ANCFShell3833Test<num_elements, ElementVersion, MaterialVersion>::GetJacobian(const bool Use_OMP) {
    ChTimer<> timer_KRM;
    timer_KRM.reset();

    ChMatrixNM<double, 72, 72> H;

    auto MeshList = m_system->Get_meshlist();
    for (auto& Mesh : MeshList) {
        auto ElementList = Mesh->GetElements();
        timer_KRM.start();

        if (Use_OMP) {
#pragma omp parallel for
            for (int ie = 0; ie < ElementList.size(); ie++) {
                ElementList[ie]->ComputeKRMmatricesGlobal(H, 1.0, 1.0, 1.0);
            }
        } else {
            for (unsigned int ie = 0; ie < ElementList.size(); ie++) {
                ElementList[ie]->ComputeKRMmatricesGlobal(H, 1.0, 1.0, 1.0);
            }
        }

        timer_KRM.stop();
    }
    return (timer_KRM());
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
void ANCFShell3833Test<num_elements, ElementVersion, MaterialVersion>::PrintTimingResults(const std::string& TestName,
                                                                                          int steps) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Test Name, Num Els, Steps, Threads, ";
    std::cout << "IntFrc Max(us), IntFrc Min(us), IntFrc Mean(us), IntFrc StDev(us), ";
    std::cout << "KRM Max(us), KRM Min(us), KRM Mean(us), KRM StDev(us), " << std::endl;
    std::cout << std::flush;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Times;
    Times.resize(steps, 2);

    // Prime the test in case the internal force calcs are also needed for the Jacobian calcs
    double TimeInternalFrc = GetInternalFrc(false);
    double TimeKRM = GetJacobian(false);

    ChTimer<> Timer_Total;
    Timer_Total.reset();
    Timer_Total.start();

    // Run Single Threaded Tests
    for (auto i = 0; i < steps; i++) {
        PerturbNodes(false);
        Times(i, 0) =
            GetInternalFrc(false) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in microseconds
    }
    for (auto i = 0; i < steps; i++) {
        PerturbNodes(false);
        Times(i, 1) = GetJacobian(false) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in
                                                                            // microseconds
    }

    std::cout << TestName << ", " << num_elements << ", " << steps << ", 0, " << Times.col(0).maxCoeff() << ", "
              << Times.col(0).minCoeff() << ", " << Times.col(0).mean() << ", "
              << std::sqrt((Times.col(0).array() - Times.col(0).mean()).square().sum() / (Times.col(0).size() - 1))
              << ", " << Times.col(1).maxCoeff() << ", " << Times.col(1).minCoeff() << ", " << Times.col(1).mean()
              << ", "
              << std::sqrt((Times.col(1).array() - Times.col(1).mean()).square().sum() / (Times.col(1).size() - 1))
              << std::endl;
    std::cout << std::flush;

    // Run Multi-Threaded Tests

    int MaxThreads = 1;
    MaxThreads = ChOMP::GetNumProcs();

    int NumThreads = 1;
    bool run = true;

    int RunNum = 1;
    while (run) {
        ChOMP::SetNumThreads(NumThreads);

        for (auto i = 0; i < steps; i++) {
            PerturbNodes(true);
            Times(i, 0) =
                GetInternalFrc(true) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in microseconds
        }
        for (auto i = 0; i < steps; i++) {
            PerturbNodes(true);
            Times(i, 1) =
                GetJacobian(true) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in microseconds
        }
        std::cout << TestName << ", " << num_elements << ", " << steps << ", " << NumThreads << ", "
                  << Times.col(0).maxCoeff() << ", " << Times.col(0).minCoeff() << ", " << Times.col(0).mean() << ", "
                  << std::sqrt((Times.col(0).array() - Times.col(0).mean()).square().sum() / (Times.col(0).size() - 1))
                  << ", " << Times.col(1).maxCoeff() << ", " << Times.col(1).minCoeff() << ", " << Times.col(1).mean()
                  << ", "
                  << std::sqrt((Times.col(1).array() - Times.col(1).mean()).square().sum() / (Times.col(1).size() - 1))
                  << std::endl;
        std::cout << std::flush;

        if (NumThreads == MaxThreads)
            run = false;

        if (NumThreads <= 4)
            NumThreads *= 2;
        else  // Since computers this will be run on have a number of cores that is a multiple of 4
            NumThreads += 4;

        if (NumThreads > MaxThreads)
            NumThreads = MaxThreads;
    }

    Timer_Total.stop();
    std::cout << "Total Test Time: " << Timer_Total() << "s" << std::endl;
}

void Run_ANCFShell_3833_Tests() {
    const int num_elements = 1024;
    int num_steps = 100;

    // const int num_elements = 8;
    // int num_steps = 10;

    {
        ANCFShell3833Test<num_elements, ChElementShellANCF_3833, ChMaterialShellANCF> Shell3833Test(1);
        Shell3833Test.PrintTimingResults("ChElementShellANCF_3833_Chrono7", num_steps);
    }
    {
        ANCFShell3833Test<num_elements, ChElementShellANCF_3833_TR01, ChMaterialShellANCF> Shell3833Test_TR01(1);
        Shell3833Test_TR01.PrintTimingResults("ChElementShellANCF_3833_TR01", num_steps);
    }
    //{
    //    ANCFShell3833Test<num_elements, ChElementShellANCF_3833_TR01B, ChMaterialShellANCF> Shell3833Test_TR01B(1);
    //    Shell3833Test_TR01B.PrintTimingResults("ChElementShellANCF_3833_TR01B", num_steps);
    //}
    //{
    //    ANCFShell3833Test<num_elements, ChElementShellANCF_3833_TR02, ChMaterialShellANCF> Shell3833Test_TR02(1);
    //    Shell3833Test_TR02.PrintTimingResults("ChElementShellANCF_3833_TR02", num_steps);
    //}
    //{
    //    ANCFShell3833Test<num_elements, ChElementShellANCF_3833_TR02B, ChMaterialShellANCF> Shell3833Test_TR02B(1);
    //    Shell3833Test_TR02B.PrintTimingResults("ChElementShellANCF_3833_TR02B", num_steps);
    //}
    //{
    //    ANCFShell3833Test<num_elements, ChElementShellANCF_3833_TR03, ChMaterialShellANCF> Shell3833Test_TR03(1);
    //    Shell3833Test_TR03.PrintTimingResults("ChElementShellANCF_3833_TR03", num_steps);
    //}
    //{
    //    ANCFShell3833Test<num_elements, ChElementShellANCF_3833_TR03B, ChMaterialShellANCF> Shell3833Test_TR03B(1);
    //    Shell3833Test_TR03B.PrintTimingResults("ChElementShellANCF_3833_TR03B", num_steps);
    //}
    //{
    //    ANCFShell3833Test<num_elements, ChElementShellANCF_3833_TR04, ChMaterialShellANCF> Shell3833Test_TR04(1);
    //    Shell3833Test_TR04.PrintTimingResults("ChElementShellANCF_3833_TR04", num_steps);
    //}
    //{
    //    ANCFShell3833Test<num_elements, ChElementShellANCF_3833_TR04B, ChMaterialShellANCF> Shell3833Test_TR04B(1);
    //    Shell3833Test_TR04B.PrintTimingResults("ChElementShellANCF_3833_TR04B", num_steps);
    //}
    {
        ANCFShell3833Test<num_elements, ChElementShellANCF_3833_TR05, ChMaterialShellANCF> Shell3833Test_TR05(1);
        Shell3833Test_TR05.PrintTimingResults("ChElementShellANCF_3833_TR05", num_steps);
    }
    {
        ANCFShell3833Test<num_elements, ChElementShellANCF_3833_TR05B, ChMaterialShellANCF> Shell3833Test_TR05B(1);
        Shell3833Test_TR05B.PrintTimingResults("ChElementShellANCF_3833_TR05B", num_steps);
    }
    {
        ANCFShell3833Test<num_elements, ChElementShellANCF_3833_TR06, ChMaterialShellANCF> Shell3833Test_TR06(1);
        Shell3833Test_TR06.PrintTimingResults("ChElementShellANCF_3833_TR06", num_steps);
    }
    {
        ANCFShell3833Test<num_elements, ChElementShellANCF_3833_TR06B, ChMaterialShellANCF> Shell3833Test_TR06B(1);
        Shell3833Test_TR06B.PrintTimingResults("ChElementShellANCF_3833_TR06B", num_steps);
    }
    {
        ANCFShell3833Test<num_elements, ChElementShellANCF_3833_TR07, ChMaterialShellANCF> Shell3833Test_TR07(1);
        Shell3833Test_TR07.PrintTimingResults("ChElementShellANCF_3833_TR07", num_steps);
    }
    {
        ANCFShell3833Test<num_elements, ChElementShellANCF_3833_TR07B, ChMaterialShellANCF> Shell3833Test_TR07B(1);
        Shell3833Test_TR07B.PrintTimingResults("ChElementShellANCF_3833_TR07B", num_steps);
    }
    {
        ANCFShell3833Test<num_elements, ChElementShellANCF_3833_TR08, ChMaterialShellANCF> Shell3833Test_TR08(1);
        Shell3833Test_TR08.PrintTimingResults("ChElementShellANCF_3833_TR08", num_steps);
    }
    {
        ANCFShell3833Test<num_elements, ChElementShellANCF_3833_TR08B, ChMaterialShellANCF> Shell3833Test_TR08B(1);
        Shell3833Test_TR08B.PrintTimingResults("ChElementShellANCF_3833_TR08B", num_steps);
    }
    {
        ANCFShell3833Test<num_elements, ChElementShellANCF_3833_TR09, ChMaterialShellANCF> Shell3833Test_TR09(1);
        Shell3833Test_TR09.PrintTimingResults("ChElementShellANCF_3833_TR09", num_steps);
    }
    {
        ANCFShell3833Test<num_elements, ChElementShellANCF_3833_TR09B, ChMaterialShellANCF> Shell3833Test_TR09B(1);
        Shell3833Test_TR09B.PrintTimingResults("ChElementShellANCF_3833_TR09B", num_steps);
    }
    {
        ANCFShell3833Test<num_elements, ChElementShellANCF_3833_TR10, ChMaterialShellANCF> Shell3833Test_TR10(1);
        Shell3833Test_TR10.PrintTimingResults("ChElementShellANCF_3833_TR10", num_steps);
    }
    {
        ANCFShell3833Test<num_elements, ChElementShellANCF_3833_TR10B, ChMaterialShellANCF> Shell3833Test_TR10B(1);
        Shell3833Test_TR10B.PrintTimingResults("ChElementShellANCF_3833_TR10B", num_steps);
    }
    {
        ANCFShell3833Test<num_elements, ChElementShellANCF_3833_TR11, ChMaterialShellANCF> Shell3833Test_TR11(1);
        Shell3833Test_TR11.PrintTimingResults("ChElementShellANCF_3833_TR11", num_steps);
    }
    {
        ANCFShell3833Test<num_elements, ChElementShellANCF_3833_TR11B, ChMaterialShellANCF> Shell3833Test_TR11B(1);
        Shell3833Test_TR11B.PrintTimingResults("ChElementShellANCF_3833_TR11B", num_steps);
    }
    //{
    //    ANCFShell3833Test<num_elements, ChElementShellANCF_3833, ChMaterialShellANCF> Shell3833Test(1);
    //    Shell3833Test.PrintTimingResults("ChElementShellANCF_3833_Chrono7", num_steps);
    //}
}

// =============================================================================

template <int num_elements, typename ElementVersion, typename MaterialVersion>
class ANCFHexa3843Test {
  public:
    ANCFHexa3843Test();

    ~ANCFHexa3843Test() { delete m_system; }

    ChSystem* GetSystem() { return m_system; }

    void PerturbNodes(const bool Use_OMP);
    double GetInternalFrc(const bool Use_OMP);
    double GetJacobian(const bool Use_OMP);

    void PrintTimingResults(const std::string& TestName, int steps);

  protected:
    ChSystemSMC* m_system;
};

template <int num_elements, typename ElementVersion, typename MaterialVersion>
ANCFHexa3843Test<num_elements, ElementVersion, MaterialVersion>::ANCFHexa3843Test() {
    m_system = new ChSystemSMC();
    m_system->Set_G_acc(ChVector<>(0, 0, -9.80665));

    // Set solver parameters
    auto solver = chrono_types::make_shared<ChSolverSparseLU>();
    solver->UseSparsityPatternLearner(true);
    solver->LockSparsityPattern(true);
    solver->SetVerbose(false);
    m_system->SetSolver(solver);

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
    double length = 10.0;  // m
    double width = 0.1;    // m
    double height = 0.01;  // m
    // Aluminum 7075-T651
    double rho = 2810;  // kg/m^3
    double E = 71.7e9;  // Pa
    double nu = 0.33;
    double G = E / (2 * (1 + nu));

    auto material = chrono_types::make_shared<MaterialVersion>(rho, E, nu);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    m_system->Add(mesh);

    // Populate the mesh container with a the nodes and elements for the meshed beam
    double dx = length / (num_elements);

    // Setup Hexa normals to initially align with the global axes
    ChVector<> dir1(1, 0, 0);
    ChVector<> dir2(0, 1, 0);
    ChVector<> dir3(0, 0, 1);

    // Create the first nodes and fix them completely to ground (Cantilever constraint)
    auto nodeA =
        chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, -0.5 * width, -0.5 * width), dir1, dir2, dir3);
    mesh->AddNode(nodeA);
    auto nodeD = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, 0.5 * width, -0.5 * width), dir1, dir2, dir3);
    mesh->AddNode(nodeD);
    auto nodeE = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, -0.5 * width, 0.5 * width), dir1, dir2, dir3);
    mesh->AddNode(nodeE);
    auto nodeH = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, 0.5 * width, 0.5 * width), dir1, dir2, dir3);
    mesh->AddNode(nodeE);

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(dx * i, -0.5 * width, -0.5 * width), dir1,
                                                                dir2, dir3);
        mesh->AddNode(nodeB);
        auto nodeC =
            chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(dx * i, 0.5 * width, -0.5 * width), dir1, dir2, dir3);
        mesh->AddNode(nodeC);
        auto nodeF =
            chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(dx * i, -0.5 * width, 0.5 * width), dir1, dir2, dir3);
        mesh->AddNode(nodeF);
        auto nodeG =
            chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(dx * i, 0.5 * width, 0.5 * width), dir1, dir2, dir3);
        mesh->AddNode(nodeG);

        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC, nodeD, nodeE, nodeF, nodeG, nodeH);
        element->SetDimensions(dx, width, width);
        element->SetMaterial(material);
        element->SetAlphaDamp(0.01);
        element->SkipPrecomputation();
        mesh->AddElement(element);

        nodeA = nodeB;
        nodeD = nodeC;
        nodeE = nodeF;
        nodeH = nodeG;
    }

    m_system->Update();  // Need to call all the element SetupInital() functions
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
void ANCFHexa3843Test<num_elements, ElementVersion, MaterialVersion>::PerturbNodes(const bool Use_OMP) {
    auto MeshList = m_system->Get_meshlist();
    for (auto& Mesh : MeshList) {
        auto NodeList = Mesh->GetNodes();
        ChVector<double> Perturbation;
        if (Use_OMP) {
#pragma omp parallel for
            for (int in = 0; in < NodeList.size(); in++) {
                Perturbation.eigen().Random();
                // auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzDDD>(NodeList[in]);
                auto Node = std::static_pointer_cast<ChNodeFEAxyzDDD>(NodeList[in]);
                Node->SetPos(Node->GetPos() + 1e-6 * Perturbation);
                Node->SetD(Node->GetD() + 1e-6 * Perturbation);
                Node->SetDD(Node->GetDD() + 1e-6 * Perturbation);
                Node->SetDDD(Node->GetDDD() + 1e-6 * Perturbation);
            }
        } else {
            for (unsigned int in = 0; in < NodeList.size(); in++) {
                Perturbation.eigen().Random();
                // auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzDDD>(NodeList[in]);
                auto Node = std::static_pointer_cast<ChNodeFEAxyzDDD>(NodeList[in]);
                Node->SetPos(Node->GetPos() + 1e-6 * Perturbation);
                Node->SetD(Node->GetD() + 1e-6 * Perturbation);
                Node->SetDD(Node->GetDD() + 1e-6 * Perturbation);
                Node->SetDDD(Node->GetDDD() + 1e-6 * Perturbation);
            }
        }
    }
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
double ANCFHexa3843Test<num_elements, ElementVersion, MaterialVersion>::GetInternalFrc(const bool Use_OMP) {
    ChTimer<> timer_internal_forces;
    timer_internal_forces.reset();

    ChVectorDynamic<double> Fi(96);

    auto MeshList = m_system->Get_meshlist();
    for (auto& Mesh : MeshList) {
        auto ElementList = Mesh->GetElements();
        timer_internal_forces.start();

        if (Use_OMP) {
#pragma omp parallel for
            for (int ie = 0; ie < ElementList.size(); ie++) {
                ElementList[ie]->ComputeInternalForces(Fi);
            }
        } else {
            for (unsigned int ie = 0; ie < ElementList.size(); ie++) {
                ElementList[ie]->ComputeInternalForces(Fi);
            }
        }

        timer_internal_forces.stop();
    }
    return (timer_internal_forces());
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
double ANCFHexa3843Test<num_elements, ElementVersion, MaterialVersion>::GetJacobian(const bool Use_OMP) {
    ChTimer<> timer_KRM;
    timer_KRM.reset();

    ChMatrixNM<double, 96, 96> H;

    auto MeshList = m_system->Get_meshlist();
    for (auto& Mesh : MeshList) {
        auto ElementList = Mesh->GetElements();
        timer_KRM.start();

        if (Use_OMP) {
#pragma omp parallel for
            for (int ie = 0; ie < ElementList.size(); ie++) {
                ElementList[ie]->ComputeKRMmatricesGlobal(H, 1.0, 1.0, 1.0);
            }
        } else {
            for (unsigned int ie = 0; ie < ElementList.size(); ie++) {
                ElementList[ie]->ComputeKRMmatricesGlobal(H, 1.0, 1.0, 1.0);
            }
        }

        timer_KRM.stop();
    }
    return (timer_KRM());
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
void ANCFHexa3843Test<num_elements, ElementVersion, MaterialVersion>::PrintTimingResults(const std::string& TestName,
                                                                                         int steps) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Test Name, Num Els, Steps, Threads, ";
    std::cout << "IntFrc Max(us), IntFrc Min(us), IntFrc Mean(us), IntFrc StDev(us), ";
    std::cout << "KRM Max(us), KRM Min(us), KRM Mean(us), KRM StDev(us), " << std::endl;
    std::cout << std::flush;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Times;
    Times.resize(steps, 2);

    // Prime the test in case the internal force calcs are also needed for the Jacobian calcs
    double TimeInternalFrc = GetInternalFrc(false);
    double TimeKRM = GetJacobian(false);

    ChTimer<> Timer_Total;
    Timer_Total.reset();
    Timer_Total.start();

    // Run Single Threaded Tests
    for (auto i = 0; i < steps; i++) {
        PerturbNodes(false);
        Times(i, 0) =
            GetInternalFrc(false) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in microseconds
    }
    for (auto i = 0; i < steps; i++) {
        PerturbNodes(false);
        Times(i, 1) = GetJacobian(false) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in
                                                                            // microseconds
    }

    std::cout << TestName << ", " << num_elements << ", " << steps << ", 0, " << Times.col(0).maxCoeff() << ", "
              << Times.col(0).minCoeff() << ", " << Times.col(0).mean() << ", "
              << std::sqrt((Times.col(0).array() - Times.col(0).mean()).square().sum() / (Times.col(0).size() - 1))
              << ", " << Times.col(1).maxCoeff() << ", " << Times.col(1).minCoeff() << ", " << Times.col(1).mean()
              << ", "
              << std::sqrt((Times.col(1).array() - Times.col(1).mean()).square().sum() / (Times.col(1).size() - 1))
              << std::endl;
    std::cout << std::flush;

    // Run Multi-Threaded Tests

    int MaxThreads = 1;
    MaxThreads = ChOMP::GetNumProcs();

    int NumThreads = 1;
    bool run = true;

    int RunNum = 1;
    while (run) {
        ChOMP::SetNumThreads(NumThreads);

        for (auto i = 0; i < steps; i++) {
            PerturbNodes(true);
            Times(i, 0) =
                GetInternalFrc(true) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in microseconds
        }
        for (auto i = 0; i < steps; i++) {
            PerturbNodes(true);
            Times(i, 1) =
                GetJacobian(true) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in microseconds
        }
        std::cout << TestName << ", " << num_elements << ", " << steps << ", " << NumThreads << ", "
                  << Times.col(0).maxCoeff() << ", " << Times.col(0).minCoeff() << ", " << Times.col(0).mean() << ", "
                  << std::sqrt((Times.col(0).array() - Times.col(0).mean()).square().sum() / (Times.col(0).size() - 1))
                  << ", " << Times.col(1).maxCoeff() << ", " << Times.col(1).minCoeff() << ", " << Times.col(1).mean()
                  << ", "
                  << std::sqrt((Times.col(1).array() - Times.col(1).mean()).square().sum() / (Times.col(1).size() - 1))
                  << std::endl;
        std::cout << std::flush;

        if (NumThreads == MaxThreads)
            run = false;

        if (NumThreads <= 4)
            NumThreads *= 2;
        else  // Since computers this will be run on have a number of cores that is a multiple of 4
            NumThreads += 4;

        if (NumThreads > MaxThreads)
            NumThreads = MaxThreads;
    }

    Timer_Total.stop();
    std::cout << "Total Test Time: " << Timer_Total() << "s" << std::endl;
}

void Run_ANCFHexa_3843_Tests() {
    const int num_elements = 512;
    int num_steps = 100;

    // const int num_elements = 128;
    // int num_steps = 10;

    {
        ANCFHexa3843Test<num_elements, ChElementHexaANCF_3843, ChMaterialHexaANCF> Hexa3843Test;
        Hexa3843Test.PrintTimingResults("ChElementHexaANCF_3843_Chrono7", num_steps);
    }
    {
        ANCFHexa3843Test<num_elements, ChElementHexaANCF_3843_TR01, ChMaterialHexaANCF> Hexa3843Test_TR01;
        Hexa3843Test_TR01.PrintTimingResults("ChElementHexaANCF_3843_TR01", num_steps);
    }
    //{
    //    ANCFHexa3843Test<num_elements, ChElementHexaANCF_3843_TR01B, ChMaterialHexaANCF> Hexa3843Test_TR01B;
    //    Hexa3843Test_TR01B.PrintTimingResults("ChElementHexaANCF_3843_TR01B", num_steps);
    //}
    //{
    //    ANCFHexa3843Test<num_elements, ChElementHexaANCF_3843_TR02, ChMaterialHexaANCF> Hexa3843Test_TR02;
    //    Hexa3843Test_TR02.PrintTimingResults("ChElementHexaANCF_3843_TR02", num_steps);
    //}
    //{
    //    ANCFHexa3843Test<num_elements, ChElementHexaANCF_3843_TR02B, ChMaterialHexaANCF> Hexa3843Test_TR02B;
    //    Hexa3843Test_TR02B.PrintTimingResults("ChElementHexaANCF_3843_TR02B", num_steps);
    //}
    //{
    //    ANCFHexa3843Test<num_elements, ChElementHexaANCF_3843_TR03, ChMaterialHexaANCF> Hexa3843Test_TR03;
    //    Hexa3843Test_TR03.PrintTimingResults("ChElementHexaANCF_3843_TR03", num_steps);
    //}
    //{
    //    ANCFHexa3843Test<num_elements, ChElementHexaANCF_3843_TR03B, ChMaterialHexaANCF> Hexa3843Test_TR03B;
    //    Hexa3843Test_TR03B.PrintTimingResults("ChElementHexaANCF_3843_TR03B", num_steps);
    //}
    //{
    //    ANCFHexa3843Test<num_elements, ChElementHexaANCF_3843_TR04, ChMaterialHexaANCF> Hexa3843Test_TR04;
    //    Hexa3843Test_TR04.PrintTimingResults("ChElementHexaANCF_3843_TR04", num_steps);
    //}
    //{
    //    ANCFHexa3843Test<num_elements, ChElementHexaANCF_3843_TR04B, ChMaterialHexaANCF> Hexa3843Test_TR04B;
    //    Hexa3843Test_TR04B.PrintTimingResults("ChElementHexaANCF_3843_TR04B", num_steps);
    //}
    {
        ANCFHexa3843Test<num_elements, ChElementHexaANCF_3843_TR05, ChMaterialHexaANCF> Hexa3843Test_TR05;
        Hexa3843Test_TR05.PrintTimingResults("ChElementHexaANCF_3843_TR05", num_steps);
    }
    {
        ANCFHexa3843Test<num_elements, ChElementHexaANCF_3843_TR05B, ChMaterialHexaANCF> Hexa3843Test_TR05B;
        Hexa3843Test_TR05B.PrintTimingResults("ChElementHexaANCF_3843_TR05B", num_steps);
    }
    {
        ANCFHexa3843Test<num_elements, ChElementHexaANCF_3843_TR06, ChMaterialHexaANCF> Hexa3843Test_TR06;
        Hexa3843Test_TR06.PrintTimingResults("ChElementHexaANCF_3843_TR06", num_steps);
    }
    {
        ANCFHexa3843Test<num_elements, ChElementHexaANCF_3843_TR06B, ChMaterialHexaANCF> Hexa3843Test_TR06A;
        Hexa3843Test_TR06A.PrintTimingResults("ChElementHexaANCF_3843_TR06B", num_steps);
    }
    {
        ANCFHexa3843Test<num_elements, ChElementHexaANCF_3843_TR07, ChMaterialHexaANCF> Hexa3843Test_TR07;
        Hexa3843Test_TR07.PrintTimingResults("ChElementHexaANCF_3843_TR07", num_steps);
    }
    {
        ANCFHexa3843Test<num_elements, ChElementHexaANCF_3843_TR07B, ChMaterialHexaANCF> Hexa3843Test_TR07A;
        Hexa3843Test_TR07A.PrintTimingResults("ChElementHexaANCF_3843_TR07B", num_steps);
    }
    {
        ANCFHexa3843Test<num_elements, ChElementHexaANCF_3843_TR08, ChMaterialHexaANCF> Hexa3843Test_TR08;
        Hexa3843Test_TR08.PrintTimingResults("ChElementHexaANCF_3843_TR08", num_steps);
    }
    {
        ANCFHexa3843Test<num_elements, ChElementHexaANCF_3843_TR08B, ChMaterialHexaANCF> Hexa3843Test_TR08A;
        Hexa3843Test_TR08A.PrintTimingResults("ChElementHexaANCF_3843_TR08B", num_steps);
    }
    {
        ANCFHexa3843Test<num_elements, ChElementHexaANCF_3843_TR09, ChMaterialHexaANCF> Hexa3843Test_TR09;
        Hexa3843Test_TR09.PrintTimingResults("ChElementHexaANCF_3843_TR09", num_steps);
    }
    {
        ANCFHexa3843Test<num_elements, ChElementHexaANCF_3843_TR09B, ChMaterialHexaANCF> Hexa3843Test_TR09A;
        Hexa3843Test_TR09A.PrintTimingResults("ChElementHexaANCF_3843_TR09B", num_steps);
    }
    {
        ANCFHexa3843Test<num_elements, ChElementHexaANCF_3843_TR10, ChMaterialHexaANCF> Hexa3843Test_TR10;
        Hexa3843Test_TR10.PrintTimingResults("ChElementHexaANCF_3843_TR10", num_steps);
    }
    {
        ANCFHexa3843Test<num_elements, ChElementHexaANCF_3843_TR10B, ChMaterialHexaANCF> Hexa3843Test_TR10A;
        Hexa3843Test_TR10A.PrintTimingResults("ChElementHexaANCF_3843_TR10B", num_steps);
    }
    {
        ANCFHexa3843Test<num_elements, ChElementHexaANCF_3843_TR11, ChMaterialHexaANCF> Hexa3843Test_TR11;
        Hexa3843Test_TR11.PrintTimingResults("ChElementHexaANCF_3843_TR11", num_steps);
    }
    {
        ANCFHexa3843Test<num_elements, ChElementHexaANCF_3843_TR11B, ChMaterialHexaANCF> Hexa3843Test_TR11A;
        Hexa3843Test_TR11A.PrintTimingResults("ChElementHexaANCF_3843_TR11B", num_steps);
    }
    //{
    //    ANCFHexa3843Test<num_elements, ChElementHexaANCF_3843, ChMaterialHexaANCF> Hexa3843Test;
    //    Hexa3843Test.PrintTimingResults("ChElementHexaANCF_3843_Chrono7", num_steps);
    //}
}

// =============================================================================

int main(int argc, char* argv[]) {
    std::ios::sync_with_stdio(false);

    if (argc < 2) {
        Run_ANCFBeam_3243_Tests();
        Run_ANCFBeam_3333_Tests();
        Run_ANCFShell_3443_Tests();
        Run_ANCFShell_3833_Tests();
        Run_ANCFHexa_3843_Tests();
    } else {
        switch (argv[1][0]) {
            case int('1'):
                Run_ANCFBeam_3243_Tests();
                break;
            case int('2'):
                Run_ANCFBeam_3333_Tests();
                break;
            case int('3'):
                Run_ANCFShell_3443_Tests();
                break;
            case int('4'):
                Run_ANCFShell_3833_Tests();
                break;
            case int('5'):
                Run_ANCFHexa_3843_Tests();
                break;
            default:
                std::cout << "Error: Unknown Input.\n";
        }
    }

    return (0);
}

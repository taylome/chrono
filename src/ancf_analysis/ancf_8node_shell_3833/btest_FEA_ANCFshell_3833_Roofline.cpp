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

#include "chrono/fea/ChMesh.h"

using namespace chrono;
using namespace chrono::fea;

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
            for (unsigned int in = 0; in < NodeList.size(); in++) {
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
            for (unsigned int ie = 0; ie < ElementList.size(); ie++) {
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
            for (unsigned int ie = 0; ie < ElementList.size(); ie++) {
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
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Times;
    Times.resize(steps, 2);

    // Prime the test in case the internal force calcs are also needed for the Jacobian calcs
    // double TimeInternalFrc = GetInternalFrc(false);
    // double TimeKRM = GetJacobian(false);

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

    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Test Name, Num Els, Steps, Threads, ";
    std::cout << "IntFrc Max(us), IntFrc Min(us), IntFrc Mean(us), IntFrc StDev(us), ";
    std::cout << "KRM Max(us), KRM Min(us), KRM Mean(us), KRM StDev(us), " << std::endl;
    std::cout << TestName << ", " << num_elements << ", " << steps << ", 0, " << Times.col(0).maxCoeff() << ", "
              << Times.col(0).minCoeff() << ", " << Times.col(0).mean() << ", "
              << std::sqrt((Times.col(0).array() - Times.col(0).mean()).square().sum() / (Times.col(0).size() - 1))
              << ", " << Times.col(1).maxCoeff() << ", " << Times.col(1).minCoeff() << ", " << Times.col(1).mean()
              << ", "
              << std::sqrt((Times.col(1).array() - Times.col(1).mean()).square().sum() / (Times.col(1).size() - 1))
              << std::endl;

    ////Run Multi-Threaded Tests

    // int MaxThreads = 1;
    // MaxThreads = ChOMP::GetNumProcs();

    // int NumThreads = 1;
    // bool run = true;

    // int RunNum = 1;
    // while (run) {
    //    ChOMP::SetNumThreads(NumThreads);

    //    for (auto i = 0; i < steps; i++) {
    //        PerturbNodes(true);
    //        Times(i, 0) = GetInternalFrc(true) * (1.0e6 / double(num_elements)); //Get Time Per Function Call in
    //        microseconds
    //    }
    //    for (auto i = 0; i < steps; i++) {
    //        PerturbNodes(true);
    //        Times(i, 1) = GetJacobian(true) * (1.0e6 / double(num_elements)); //Get Time Per Function Call in
    //        microseconds
    //    }
    //    std::cout << TestName << ", " << num_elements << ", " << steps << ", " << NumThreads << ", "
    //        << Times.col(0).maxCoeff() << ", "
    //        << Times.col(0).minCoeff() << ", "
    //        << Times.col(0).mean() << ", "
    //        << std::sqrt((Times.col(0).array() - Times.col(0).mean()).square().sum() / (Times.col(0).size() - 1)) <<
    //        ", "
    //        << Times.col(1).maxCoeff() << ", "
    //        << Times.col(1).minCoeff() << ", "
    //        << Times.col(1).mean() << ", "
    //        << std::sqrt((Times.col(1).array() - Times.col(1).mean()).square().sum() / (Times.col(1).size() - 1)) <<
    //        std::endl;

    //    if (NumThreads == MaxThreads)
    //        run = false;

    //    if (NumThreads <= 4)
    //        NumThreads *= 2;
    //    else  // Since computers this will be run on have a number of cores that is a multiple of 4
    //        NumThreads += 4;

    //    if (NumThreads > MaxThreads)
    //        NumThreads = MaxThreads;
    //}

    Timer_Total.stop();
    std::cout << "Total Test Time: " << Timer_Total() << "s" << std::endl;
}

void Run_ANCFShell_3833_Tests() {
    const int num_elements = 1024;
    int num_steps = 10;

    // const int num_elements = 8;
    // int num_steps = 1;

    {
        ANCFShell3833Test<num_elements, ChElementShellANCF_3833, ChMaterialShellANCF> Shell3833Test(1);
        Shell3833Test.PrintTimingResults("ChElementShellANCF_3833_Chrono7", num_steps);
    }
    //{
    //    ANCFShell3833Test<num_elements, ChElementShellANCF_3833_TR01, ChMaterialShellANCF> Shell3833Test_TR01(1);
    //    Shell3833Test_TR01.PrintTimingResults("ChElementShellANCF_3833_TR01", num_steps);
    //}
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

int main(int argc, char* argv[]) {
    Run_ANCFShell_3833_Tests();

    return (0);
}

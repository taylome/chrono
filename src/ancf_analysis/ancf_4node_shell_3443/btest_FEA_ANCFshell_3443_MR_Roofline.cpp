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

#include "chrono/fea/ChElementShellANCF_3443_TR08.h"
#include "chrono/fea/ChElementShellANCF_3443_MR_Damp.h"
#include "chrono/fea/ChElementShellANCF_3443_MR_NoDamp.h"
#include "chrono/fea/ChElementShellANCF_3443_MR_DampNumJac.h"
#include "chrono/fea/ChElementShellANCF_3443_MR_NoDampNumJac.h"

#include "chrono/fea/ChMesh.h"

using namespace chrono;
using namespace chrono::fea;

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
    m_system->SetGravitationalAcceleration(ChVector3d(0, 0, -9.80665));

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
    integrator->SetMaxIters(100);
    // integrator->SetAbsTolerances(1e-5);
    integrator->SetAbsTolerances(1e-3);
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
    ChVector3d dir1(1, 0, 0);
    ChVector3d dir2(0, 1, 0);
    ChVector3d dir3(0, 0, 1);

    // Create the first node and fix it completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector3d(0, 0, 0.0), dir1, dir2, dir3);
    mesh->AddNode(nodeA);
    auto nodeD = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector3d(0, width, 0.0), dir1, dir2, dir3);
    mesh->AddNode(nodeD);

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector3d(dx * i, 0, 0), dir1, dir2, dir3);
        mesh->AddNode(nodeB);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector3d(dx * i, width, 0), dir1, dir2, dir3);
        mesh->AddNode(nodeC);

        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC, nodeD);
        element->SetDimensions(dx, width);
        for (int j = 0; j < m_num_layers; j++) {
            element->AddLayer(height, 0 * CH_DEG_TO_RAD, material);
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
    auto MeshList = m_system->GetMeshes();
    for (auto& Mesh : MeshList) {
        auto NodeList = Mesh->GetNodes();
        ChVector3d Perturbation;
        if (Use_OMP) {
#pragma omp parallel for
            for (int in = 0; in < NodeList.size(); in++) {
                Perturbation.eigen().Random();
                // auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzDDD>(NodeList[in]);
                auto Node = std::static_pointer_cast<ChNodeFEAxyzDDD>(NodeList[in]);
                Node->SetPos(Node->GetPos() + 1e-6 * Perturbation);
                Node->SetSlope1(Node->GetSlope1() + 1e-6 * Perturbation);
                Node->SetSlope2(Node->GetSlope2() + 1e-6 * Perturbation);
                Node->SetSlope3(Node->GetSlope3() + 1e-6 * Perturbation);
            }
        }
        else {
            for (unsigned int in = 0; in < NodeList.size(); in++) {
                Perturbation.eigen().Random();
                // auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzDDD>(NodeList[in]);
                auto Node = std::static_pointer_cast<ChNodeFEAxyzDDD>(NodeList[in]);
                Node->SetPos(Node->GetPos() + 1e-6 * Perturbation);
                Node->SetSlope1(Node->GetSlope1() + 1e-6 * Perturbation);
                Node->SetSlope2(Node->GetSlope2() + 1e-6 * Perturbation);
                Node->SetSlope3(Node->GetSlope3() + 1e-6 * Perturbation);
            }
        }
    }
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
double ANCFShell3443Test<num_elements, ElementVersion, MaterialVersion>::GetInternalFrc(const bool Use_OMP) {
    ChTimer timer_internal_forces;
    timer_internal_forces.reset();

    auto MeshList = m_system->GetMeshes();
    for (auto& Mesh : MeshList) {
        auto ElementList = Mesh->GetElements();
        timer_internal_forces.start();

        if (Use_OMP) {
#pragma omp parallel for
            for (int ie = 0; ie < ElementList.size(); ie++) {
                ChVectorDynamic<double> Fi(48);
                ElementList[ie]->ComputeInternalForces(Fi);
            }
        }
        else {
            for (unsigned int ie = 0; ie < ElementList.size(); ie++) {
                ChVectorDynamic<double> Fi(48);
                ElementList[ie]->ComputeInternalForces(Fi);
            }
        }

        timer_internal_forces.stop();
    }
    return (timer_internal_forces());
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
double ANCFShell3443Test<num_elements, ElementVersion, MaterialVersion>::GetJacobian(const bool Use_OMP) {
    ChTimer timer_KRM;
    timer_KRM.reset();

    auto MeshList = m_system->GetMeshes();
    for (auto& Mesh : MeshList) {
        auto ElementList = Mesh->GetElements();
        timer_KRM.start();

        if (Use_OMP) {
#pragma omp parallel for
            for (int ie = 0; ie < ElementList.size(); ie++) {
                ChMatrixNM<double, 48, 48> H;
                ElementList[ie]->ComputeKRMmatricesGlobal(H, 1.0, 1.0, 1.0);
            }
        }
        else {
            for (unsigned int ie = 0; ie < ElementList.size(); ie++) {
                ChMatrixNM<double, 48, 48> H;
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
    std::cout << "IntFrc Min(us), IntFrc Q1(us), IntFrc Median(us), IntFrc Q3(us), ";
    std::cout << "IntFrc Max(us), IntFrc Mean(us), IntFrc StDev(us), ";
    std::cout << "KRM Min(us), KRM Q1(us), KRM Median(us), KRM Q3(us), ";
    std::cout << "KRM Max(us), KRM Mean(us), KRM StDev(us), " << std::endl;
    std::cout << std::flush;

    Eigen::Array<double, Eigen::Dynamic, 1, Eigen::ColMajor> TimeIntFrc;
    TimeIntFrc.resize(steps, 1);
    Eigen::Array<double, Eigen::Dynamic, 1, Eigen::ColMajor> TimeJac;
    TimeJac.resize(steps, 1);

    // Prime the test in case the internal force calcs are also needed for the Jacobian calcs
    double TimeInternalFrc = GetInternalFrc(false);
    double TimeKRM = GetJacobian(false);

    ChTimer Timer_Total;
    Timer_Total.reset();
    Timer_Total.start();

    // Run Single Threaded Tests
    for (auto i = 0; i < steps; i++) {
        PerturbNodes(false);
        TimeIntFrc(i) =
            GetInternalFrc(false) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in microseconds
    }
    for (auto i = 0; i < steps; i++) {
        PerturbNodes(false);
        TimeJac(i) = GetJacobian(false) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in
                                                                            // microseconds
    }

    //std::sort(TimeIntFrc.begin(), TimeIntFrc.end());
    //std::sort(TimeJac.begin(), TimeJac.end());
    std::sort(TimeIntFrc.data(), TimeIntFrc.data() + TimeIntFrc.size());
    std::sort(TimeJac.data(), TimeJac.data() + TimeJac.size());

    //Calculate the 1st quartile, median, and 3rd quartile
    double IntFrc_Q1;
    double IntFrc_Q2;
    double IntFrc_Q3;
    double Jac_Q1;
    double Jac_Q2;
    double Jac_Q3;

    if (steps % 2 == 1) {
        IntFrc_Q2 = TimeIntFrc((steps - 1) / 2);
        Jac_Q2 = TimeJac((steps - 1) / 2);

        if (((steps - 1) / 2) % 2 == 1) {
            IntFrc_Q1 = TimeIntFrc((steps - 3) / 4);
            Jac_Q1 = TimeJac((steps - 3) / 4);
            IntFrc_Q3 = TimeIntFrc((3 * steps - 1) / 4);
            Jac_Q3 = TimeJac((3 * steps - 1) / 4);
        }
        else {
            IntFrc_Q1 = (TimeIntFrc((steps - 5) / 4) + TimeIntFrc((steps - 1) / 4)) / 2.0;
            Jac_Q1 = (TimeJac((steps - 5) / 4) + TimeJac((steps - 1) / 4)) / 2.0;
            IntFrc_Q3 = (TimeIntFrc((3 * steps - 3) / 4) + TimeIntFrc((3 * steps + 1) / 4)) / 2.0;
            Jac_Q3 = (TimeJac((3 * steps - 3) / 4) + TimeJac((3 * steps + 1) / 4)) / 2.0;
        }
    }
    else {
        IntFrc_Q2 = (TimeIntFrc(steps / 2 - 1) + TimeIntFrc((steps / 2))) / 2.0;
        Jac_Q2 = (TimeJac(steps / 2 - 1) + TimeJac((steps / 2))) / 2.0;

        if ((steps / 2) % 2 == 1) {
            IntFrc_Q1 = TimeIntFrc((steps - 2) / 4);
            Jac_Q1 = TimeJac((steps - 2) / 4);
            IntFrc_Q3 = TimeIntFrc((3 * steps - 2) / 4);
            Jac_Q3 = TimeJac((3 * steps - 2) / 4);
        }
        else {
            IntFrc_Q1 = (TimeIntFrc(steps / 4 - 1) + TimeIntFrc(steps / 4)) / 2.0;
            Jac_Q1 = (TimeJac(steps / 4 - 1) + TimeJac(steps / 4)) / 2.0;
            IntFrc_Q3 = (TimeIntFrc((3 * steps) / 4 - 1) + TimeIntFrc((3 * steps) / 4)) / 2.0;
            Jac_Q3 = (TimeJac((3 * steps) / 4 - 1) + TimeJac((3 * steps) / 4)) / 2.0;
        }
    }

    std::cout << TestName << ", " << num_elements << ", " << steps << ", 0, " << TimeIntFrc.minCoeff() << ", "
        << IntFrc_Q1 << ", " << IntFrc_Q2 << ", " << IntFrc_Q3 << ", " << TimeIntFrc.maxCoeff() << ", "
        << TimeIntFrc.mean() << ", "
        << std::sqrt((TimeIntFrc - TimeIntFrc.mean()).square().sum() / (TimeIntFrc.size() - 1)) << ", "
        << TimeJac.minCoeff() << ", " << Jac_Q1 << ", " << Jac_Q2 << ", " << Jac_Q3 << ", " << TimeJac.maxCoeff()
        << ", " << TimeJac.mean() << ", "
        << std::sqrt((TimeJac - TimeJac.mean()).square().sum() / (TimeJac.size() - 1)) << std::endl;
    std::cout << std::flush;


    //// Run Multi-Threaded Tests

    //int MaxThreads = 1;
    //MaxThreads = ChOMP::GetNumProcs();

    //int NumThreads = 1;
    //bool run = true;

    //int RunNum = 1;
    //while (run) {
    //    ChOMP::SetNumThreads(NumThreads);

    //    for (auto i = 0; i < steps; i++) {
    //        PerturbNodes(true);
    //        TimeIntFrc(i) =
    //            GetInternalFrc(true) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in microseconds
    //    }
    //    for (auto i = 0; i < steps; i++) {
    //        PerturbNodes(true);
    //        TimeJac(i) =
    //            GetJacobian(true) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in microseconds
    //    }

    //    //Sort the times
    //    //std::sort(TimeIntFrc.begin(), TimeIntFrc.end());
    //    //std::sort(TimeJac.begin(), TimeJac.end());
    //    std::sort(TimeIntFrc.data(), TimeIntFrc.data() + TimeIntFrc.size());
    //    std::sort(TimeJac.data(), TimeJac.data() + TimeJac.size());

    //    //Calculate the 1st quartile, median, and 3rd quartile
    //    if (steps % 2 == 1) {
    //        IntFrc_Q2 = TimeIntFrc((steps - 1) / 2);
    //        Jac_Q2 = TimeJac((steps - 1) / 2);

    //        if (((steps - 1) / 2) % 2 == 1) {
    //            IntFrc_Q1 = TimeIntFrc((steps - 3) / 4);
    //            Jac_Q1 = TimeJac((steps - 3) / 4);
    //            IntFrc_Q3 = TimeIntFrc((3 * steps - 1) / 4);
    //            Jac_Q3 = TimeJac((3 * steps - 1) / 4);
    //        }
    //        else {
    //            IntFrc_Q1 = (TimeIntFrc((steps - 5) / 4) + TimeIntFrc((steps - 1) / 4)) / 2.0;
    //            Jac_Q1 = (TimeJac((steps - 5) / 4) + TimeJac((steps - 1) / 4)) / 2.0;
    //            IntFrc_Q3 = (TimeIntFrc((3 * steps - 3) / 4) + TimeIntFrc((3 * steps + 1) / 4)) / 2.0;
    //            Jac_Q3 = (TimeJac((3 * steps - 3) / 4) + TimeJac((3 * steps + 1) / 4)) / 2.0;
    //        }
    //    }
    //    else {
    //        IntFrc_Q2 = (TimeIntFrc(steps / 2 - 1) + TimeIntFrc((steps / 2))) / 2.0;
    //        Jac_Q2 = (TimeJac(steps / 2 - 1) + TimeJac((steps / 2))) / 2.0;

    //        if ((steps / 2) % 2 == 1) {
    //            IntFrc_Q1 = TimeIntFrc((steps - 2) / 4);
    //            Jac_Q1 = TimeJac((steps - 2) / 4);
    //            IntFrc_Q3 = TimeIntFrc((3 * steps - 2) / 4);
    //            Jac_Q3 = TimeJac((3 * steps - 2) / 4);
    //        }
    //        else {
    //            IntFrc_Q1 = (TimeIntFrc(steps / 4 - 1) + TimeIntFrc(steps / 4)) / 2.0;
    //            Jac_Q1 = (TimeJac(steps / 4 - 1) + TimeJac(steps / 4)) / 2.0;
    //            IntFrc_Q3 = (TimeIntFrc((3 * steps) / 4 - 1) + TimeIntFrc((3 * steps) / 4)) / 2.0;
    //            Jac_Q3 = (TimeJac((3 * steps) / 4 - 1) + TimeJac((3 * steps) / 4)) / 2.0;
    //        }
    //    }

    //    std::cout << TestName << ", " << num_elements << ", " << steps << ", " << NumThreads << ", " << TimeIntFrc.minCoeff() << ", "
    //        << IntFrc_Q1 << ", " << IntFrc_Q2 << ", " << IntFrc_Q3 << ", " << TimeIntFrc.maxCoeff() << ", "
    //        << TimeIntFrc.mean() << ", "
    //        << std::sqrt((TimeIntFrc - TimeIntFrc.mean()).square().sum() / (TimeIntFrc.size() - 1)) << ", "
    //        << TimeJac.minCoeff() << ", " << Jac_Q1 << ", " << Jac_Q2 << ", " << Jac_Q3 << ", " << TimeJac.maxCoeff()
    //        << ", " << TimeJac.mean() << ", "
    //        << std::sqrt((TimeJac - TimeJac.mean()).square().sum() / (TimeJac.size() - 1)) << std::endl;
    //    std::cout << std::flush;

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

template <int num_elements, typename ElementVersion>
class ANCFShell3443Test_MR {
public:
    ANCFShell3443Test_MR(unsigned int num_layers);

    ~ANCFShell3443Test_MR() { delete m_system; }

    ChSystem* GetSystem() { return m_system; }

    void PerturbNodes(const bool Use_OMP);
    double GetInternalFrc(const bool Use_OMP);
    double GetJacobian(const bool Use_OMP);

    void PrintTimingResults(const std::string& TestName, int steps);

protected:
    ChSystemSMC* m_system;
    unsigned int m_num_layers;
};

template <int num_elements, typename ElementVersion>
ANCFShell3443Test_MR<num_elements, ElementVersion>::ANCFShell3443Test_MR(unsigned int num_layers) {
    m_num_layers = num_layers;
    m_system = new ChSystemSMC();
    m_system->SetGravitationalAcceleration(ChVector3d(0, 0, -9.80665));

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
    integrator->SetMaxIters(100);
    // integrator->SetAbsTolerances(1e-5);
    integrator->SetAbsTolerances(1e-3);
    integrator->SetVerbose(false);
    integrator->SetModifiedNewton(true);

    // Mesh properties
    double length = 10.0;  // m
    double width = 0.1;    // m
    double height = 0.01;  // m
    double rho = 7850; //kg/m^3
    double c10 = 0.8e6;  ///Pa
    double c01 = 0.2e6;    ///Pa
    double k = 1e9;    ///Pa
    double mu = 0.01 * 6 * (c10 + c01);     ///< Viscosity Coefficient
    //double mu = 0;     ///< Viscosity Coefficient

    auto material = chrono_types::make_shared<ChMaterialShellANCF_MR>(rho, c10, c01, k, mu);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    m_system->Add(mesh);

    // Populate the mesh container with a the nodes and elements for the meshed beam
    int num_nodes = 2 * num_elements + 2;
    double dx = length / (num_elements);

    // Setup beam cross section gradients to initially align with the global y and z directions
    ChVector3d dir1(1, 0, 0);
    ChVector3d dir2(0, 1, 0);
    ChVector3d dir3(0, 0, 1);

    // Create the first node and fix it completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector3d(0, 0, 0.0), dir1, dir2, dir3);
    mesh->AddNode(nodeA);
    auto nodeD = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector3d(0, width, 0.0), dir1, dir2, dir3);
    mesh->AddNode(nodeD);

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector3d(dx * i, 0, 0), dir1, dir2, dir3);
        mesh->AddNode(nodeB);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector3d(dx * i, width, 0), dir1, dir2, dir3);
        mesh->AddNode(nodeC);

        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC, nodeD);
        element->SetDimensions(dx, width);
        for (int j = 0; j < m_num_layers; j++) {
            element->AddLayer(height, material);
        }
        mesh->AddElement(element);

        nodeA = nodeB;
        nodeD = nodeC;
    }

    m_system->Update();  // Need to call all the element SetupInital() functions
}

template <int num_elements, typename ElementVersion>
void ANCFShell3443Test_MR<num_elements, ElementVersion>::PerturbNodes(const bool Use_OMP) {
    auto MeshList = m_system->GetMeshes();
    for (auto& Mesh : MeshList) {
        auto NodeList = Mesh->GetNodes();
        ChVector3d Perturbation;
        if (Use_OMP) {
#pragma omp parallel for
            for (int in = 0; in < NodeList.size(); in++) {
                Perturbation.eigen().Random();
                // auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzDDD>(NodeList[in]);
                auto Node = std::static_pointer_cast<ChNodeFEAxyzDDD>(NodeList[in]);
                Node->SetPos(Node->GetPos() + 1e-6 * Perturbation);
                Node->SetSlope1(Node->GetSlope1() + 1e-6 * Perturbation);
                Node->SetSlope2(Node->GetSlope2() + 1e-6 * Perturbation);
                Node->SetSlope3(Node->GetSlope3() + 1e-6 * Perturbation);
            }
        }
        else {
            for (unsigned int in = 0; in < NodeList.size(); in++) {
                Perturbation.eigen().Random();
                // auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzDDD>(NodeList[in]);
                auto Node = std::static_pointer_cast<ChNodeFEAxyzDDD>(NodeList[in]);
                Node->SetPos(Node->GetPos() + 1e-6 * Perturbation);
                Node->SetSlope1(Node->GetSlope1() + 1e-6 * Perturbation);
                Node->SetSlope2(Node->GetSlope2() + 1e-6 * Perturbation);
                Node->SetSlope3(Node->GetSlope3() + 1e-6 * Perturbation);
            }
        }
    }
}

template <int num_elements, typename ElementVersion>
double ANCFShell3443Test_MR<num_elements, ElementVersion>::GetInternalFrc(const bool Use_OMP) {
    ChTimer timer_internal_forces;
    timer_internal_forces.reset();

    auto MeshList = m_system->GetMeshes();
    for (auto& Mesh : MeshList) {
        auto ElementList = Mesh->GetElements();
        timer_internal_forces.start();

        if (Use_OMP) {
#pragma omp parallel for
            for (int ie = 0; ie < ElementList.size(); ie++) {
                ChVectorDynamic<double> Fi(48);
                ElementList[ie]->ComputeInternalForces(Fi);
            }
        }
        else {
            for (unsigned int ie = 0; ie < ElementList.size(); ie++) {
                ChVectorDynamic<double> Fi(48);
                ElementList[ie]->ComputeInternalForces(Fi);
            }
        }

        timer_internal_forces.stop();
    }
    return (timer_internal_forces());
}

template <int num_elements, typename ElementVersion>
double ANCFShell3443Test_MR<num_elements, ElementVersion>::GetJacobian(const bool Use_OMP) {
    ChTimer timer_KRM;
    timer_KRM.reset();

    auto MeshList = m_system->GetMeshes();
    for (auto& Mesh : MeshList) {
        auto ElementList = Mesh->GetElements();
        timer_KRM.start();

        if (Use_OMP) {
#pragma omp parallel for
            for (int ie = 0; ie < ElementList.size(); ie++) {
                ChMatrixNM<double, 48, 48> H;
                ElementList[ie]->ComputeKRMmatricesGlobal(H, 1.0, 1.0, 1.0);
            }
        }
        else {
            for (unsigned int ie = 0; ie < ElementList.size(); ie++) {
                ChMatrixNM<double, 48, 48> H;
                ElementList[ie]->ComputeKRMmatricesGlobal(H, 1.0, 1.0, 1.0);
            }
        }

        timer_KRM.stop();
    }
    return (timer_KRM());
}

template <int num_elements, typename ElementVersion>
void ANCFShell3443Test_MR<num_elements, ElementVersion>::PrintTimingResults(const std::string& TestName,
    int steps) {
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Test Name, Num Els, Steps, Threads, ";
    std::cout << "IntFrc Min(us), IntFrc Q1(us), IntFrc Median(us), IntFrc Q3(us), ";
    std::cout << "IntFrc Max(us), IntFrc Mean(us), IntFrc StDev(us), ";
    std::cout << "KRM Min(us), KRM Q1(us), KRM Median(us), KRM Q3(us), ";
    std::cout << "KRM Max(us), KRM Mean(us), KRM StDev(us), " << std::endl;
    std::cout << std::flush;

    Eigen::Array<double, Eigen::Dynamic, 1, Eigen::ColMajor> TimeIntFrc;
    TimeIntFrc.resize(steps, 1);
    Eigen::Array<double, Eigen::Dynamic, 1, Eigen::ColMajor> TimeJac;
    TimeJac.resize(steps, 1);

    // Prime the test in case the internal force calcs are also needed for the Jacobian calcs
    double TimeInternalFrc = GetInternalFrc(false);
    double TimeKRM = GetJacobian(false);

    ChTimer Timer_Total;
    Timer_Total.reset();
    Timer_Total.start();

    // Run Single Threaded Tests
    for (auto i = 0; i < steps; i++) {
        PerturbNodes(false);
        TimeIntFrc(i) =
            GetInternalFrc(false) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in microseconds
    }
    for (auto i = 0; i < steps; i++) {
        PerturbNodes(false);
        TimeJac(i) = GetJacobian(false) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in
                                                                            // microseconds
    }

    //std::sort(TimeIntFrc.begin(), TimeIntFrc.end());
    //std::sort(TimeJac.begin(), TimeJac.end());
    std::sort(TimeIntFrc.data(), TimeIntFrc.data() + TimeIntFrc.size());
    std::sort(TimeJac.data(), TimeJac.data() + TimeJac.size());

    //Calculate the 1st quartile, median, and 3rd quartile
    double IntFrc_Q1;
    double IntFrc_Q2;
    double IntFrc_Q3;
    double Jac_Q1;
    double Jac_Q2;
    double Jac_Q3;

    if (steps % 2 == 1) {
        IntFrc_Q2 = TimeIntFrc((steps - 1) / 2);
        Jac_Q2 = TimeJac((steps - 1) / 2);

        if (((steps - 1) / 2) % 2 == 1) {
            IntFrc_Q1 = TimeIntFrc((steps - 3) / 4);
            Jac_Q1 = TimeJac((steps - 3) / 4);
            IntFrc_Q3 = TimeIntFrc((3 * steps - 1) / 4);
            Jac_Q3 = TimeJac((3 * steps - 1) / 4);
        }
        else {
            IntFrc_Q1 = (TimeIntFrc((steps - 5) / 4) + TimeIntFrc((steps - 1) / 4)) / 2.0;
            Jac_Q1 = (TimeJac((steps - 5) / 4) + TimeJac((steps - 1) / 4)) / 2.0;
            IntFrc_Q3 = (TimeIntFrc((3 * steps - 3) / 4) + TimeIntFrc((3 * steps + 1) / 4)) / 2.0;
            Jac_Q3 = (TimeJac((3 * steps - 3) / 4) + TimeJac((3 * steps + 1) / 4)) / 2.0;
        }
    }
    else {
        IntFrc_Q2 = (TimeIntFrc(steps / 2 - 1) + TimeIntFrc((steps / 2))) / 2.0;
        Jac_Q2 = (TimeJac(steps / 2 - 1) + TimeJac((steps / 2))) / 2.0;

        if ((steps / 2) % 2 == 1) {
            IntFrc_Q1 = TimeIntFrc((steps - 2) / 4);
            Jac_Q1 = TimeJac((steps - 2) / 4);
            IntFrc_Q3 = TimeIntFrc((3 * steps - 2) / 4);
            Jac_Q3 = TimeJac((3 * steps - 2) / 4);
        }
        else {
            IntFrc_Q1 = (TimeIntFrc(steps / 4 - 1) + TimeIntFrc(steps / 4)) / 2.0;
            Jac_Q1 = (TimeJac(steps / 4 - 1) + TimeJac(steps / 4)) / 2.0;
            IntFrc_Q3 = (TimeIntFrc((3 * steps) / 4 - 1) + TimeIntFrc((3 * steps) / 4)) / 2.0;
            Jac_Q3 = (TimeJac((3 * steps) / 4 - 1) + TimeJac((3 * steps) / 4)) / 2.0;
        }
    }

    std::cout << TestName << ", " << num_elements << ", " << steps << ", 0, " << TimeIntFrc.minCoeff() << ", "
        << IntFrc_Q1 << ", " << IntFrc_Q2 << ", " << IntFrc_Q3 << ", " << TimeIntFrc.maxCoeff() << ", "
        << TimeIntFrc.mean() << ", "
        << std::sqrt((TimeIntFrc - TimeIntFrc.mean()).square().sum() / (TimeIntFrc.size() - 1)) << ", "
        << TimeJac.minCoeff() << ", " << Jac_Q1 << ", " << Jac_Q2 << ", " << Jac_Q3 << ", " << TimeJac.maxCoeff()
        << ", " << TimeJac.mean() << ", "
        << std::sqrt((TimeJac - TimeJac.mean()).square().sum() / (TimeJac.size() - 1)) << std::endl;
    std::cout << std::flush;


    //// Run Multi-Threaded Tests

    //int MaxThreads = 1;
    //MaxThreads = ChOMP::GetNumProcs();

    //int NumThreads = 1;
    //bool run = true;

    //int RunNum = 1;
    //while (run) {
    //    ChOMP::SetNumThreads(NumThreads);

    //    for (auto i = 0; i < steps; i++) {
    //        PerturbNodes(true);
    //        TimeIntFrc(i) =
    //            GetInternalFrc(true) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in microseconds
    //    }
    //    for (auto i = 0; i < steps; i++) {
    //        PerturbNodes(true);
    //        TimeJac(i) =
    //            GetJacobian(true) * (1.0e6 / double(num_elements));  // Get Time Per Function Call in microseconds
    //    }

    //    //Sort the times
    //    //std::sort(TimeIntFrc.begin(), TimeIntFrc.end());
    //    //std::sort(TimeJac.begin(), TimeJac.end());
    //    std::sort(TimeIntFrc.data(), TimeIntFrc.data() + TimeIntFrc.size());
    //    std::sort(TimeJac.data(), TimeJac.data() + TimeJac.size());

    //    //Calculate the 1st quartile, median, and 3rd quartile
    //    if (steps % 2 == 1) {
    //        IntFrc_Q2 = TimeIntFrc((steps - 1) / 2);
    //        Jac_Q2 = TimeJac((steps - 1) / 2);

    //        if (((steps - 1) / 2) % 2 == 1) {
    //            IntFrc_Q1 = TimeIntFrc((steps - 3) / 4);
    //            Jac_Q1 = TimeJac((steps - 3) / 4);
    //            IntFrc_Q3 = TimeIntFrc((3 * steps - 1) / 4);
    //            Jac_Q3 = TimeJac((3 * steps - 1) / 4);
    //        }
    //        else {
    //            IntFrc_Q1 = (TimeIntFrc((steps - 5) / 4) + TimeIntFrc((steps - 1) / 4)) / 2.0;
    //            Jac_Q1 = (TimeJac((steps - 5) / 4) + TimeJac((steps - 1) / 4)) / 2.0;
    //            IntFrc_Q3 = (TimeIntFrc((3 * steps - 3) / 4) + TimeIntFrc((3 * steps + 1) / 4)) / 2.0;
    //            Jac_Q3 = (TimeJac((3 * steps - 3) / 4) + TimeJac((3 * steps + 1) / 4)) / 2.0;
    //        }
    //    }
    //    else {
    //        IntFrc_Q2 = (TimeIntFrc(steps / 2 - 1) + TimeIntFrc((steps / 2))) / 2.0;
    //        Jac_Q2 = (TimeJac(steps / 2 - 1) + TimeJac((steps / 2))) / 2.0;

    //        if ((steps / 2) % 2 == 1) {
    //            IntFrc_Q1 = TimeIntFrc((steps - 2) / 4);
    //            Jac_Q1 = TimeJac((steps - 2) / 4);
    //            IntFrc_Q3 = TimeIntFrc((3 * steps - 2) / 4);
    //            Jac_Q3 = TimeJac((3 * steps - 2) / 4);
    //        }
    //        else {
    //            IntFrc_Q1 = (TimeIntFrc(steps / 4 - 1) + TimeIntFrc(steps / 4)) / 2.0;
    //            Jac_Q1 = (TimeJac(steps / 4 - 1) + TimeJac(steps / 4)) / 2.0;
    //            IntFrc_Q3 = (TimeIntFrc((3 * steps) / 4 - 1) + TimeIntFrc((3 * steps) / 4)) / 2.0;
    //            Jac_Q3 = (TimeJac((3 * steps) / 4 - 1) + TimeJac((3 * steps) / 4)) / 2.0;
    //        }
    //    }

    //    std::cout << TestName << ", " << num_elements << ", " << steps << ", " << NumThreads << ", " << TimeIntFrc.minCoeff() << ", "
    //        << IntFrc_Q1 << ", " << IntFrc_Q2 << ", " << IntFrc_Q3 << ", " << TimeIntFrc.maxCoeff() << ", "
    //        << TimeIntFrc.mean() << ", "
    //        << std::sqrt((TimeIntFrc - TimeIntFrc.mean()).square().sum() / (TimeIntFrc.size() - 1)) << ", "
    //        << TimeJac.minCoeff() << ", " << Jac_Q1 << ", " << Jac_Q2 << ", " << Jac_Q3 << ", " << TimeJac.maxCoeff()
    //        << ", " << TimeJac.mean() << ", "
    //        << std::sqrt((TimeJac - TimeJac.mean()).square().sum() / (TimeJac.size() - 1)) << std::endl;
    //    std::cout << std::flush;

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


void Run_ANCFShell_3443_Tests() {
    const int num_elements = 1024;
    int num_steps = 100;
    //const int num_elements = 1024;
    //int num_steps = 500;
    //const int num_elements = 8;
    //int num_steps = 10;

    {
        ANCFShell3443Test<num_elements, ChElementShellANCF_3443_TR08, ChMaterialShellANCF> Shell3443Test_TR08(1);
        Shell3443Test_TR08.PrintTimingResults("ChElementShellANCF_3443_TR08", num_steps);
    }

    {
        ANCFShell3443Test_MR<num_elements, ChElementShellANCF_3443_MR_Damp> Test(1);
        Test.PrintTimingResults("ChElementShellANCF_3443_MR_Damp", num_steps);
    }
    {
        ANCFShell3443Test_MR<num_elements, ChElementShellANCF_3443_MR_NoDamp> Test(1);
        Test.PrintTimingResults("ChElementShellANCF_3443_MR_NoDamp", num_steps);
    }
    {
        ANCFShell3443Test_MR<num_elements, ChElementShellANCF_3443_MR_DampNumJac> Test(1);
        Test.PrintTimingResults("ChElementShellANCF_3443_MR_DampNumJac", num_steps);
    }
    {
        ANCFShell3443Test_MR<num_elements, ChElementShellANCF_3443_MR_NoDampNumJac> Test(1);
        Test.PrintTimingResults("ChElementShellANCF_3443_MR_NoDampNumJac", num_steps);
    }

}

// =============================================================================

int main(int argc, char* argv[]) {
    
    std::ios::sync_with_stdio(false);

    Run_ANCFShell_3443_Tests();

    return (0);
}

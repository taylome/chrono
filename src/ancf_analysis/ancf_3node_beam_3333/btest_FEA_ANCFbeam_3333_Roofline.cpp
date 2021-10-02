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

#include "chrono/fea/ChElementBeamANCF.h"
#include "chrono/fea/ChElementBeamANCF_3333.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR00.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR01.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR02.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR02_GQ322.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR03.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR03_GQ322.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR04.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR04_GQ322.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR05.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR05_GQ322.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR06.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR06_GQ322.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR07.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR07s.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR07s_GQ322.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR08.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR08b.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR08s.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR08s_GQ322.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR09.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR10.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR11.h"
#include "chrono/fea/ChElementBeamANCF_3333_TR11s.h"

#include "chrono/fea/ChMesh.h"

using namespace chrono;
using namespace chrono::fea;

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
        element->SetGravityOn(
            false);  // Enable the efficient ANCF method for calculating the application of gravity to the element
        //element->SetStrainFormulation(ElementVersion::StrainFormulation::CMPoisson);
        // element->SetStrainFormulation(ElementVersion::StrainFormulation::CMNoPoisson);
        mesh->AddElement(element);

        nodeA = nodeB;
    }

    mesh->SetAutomaticGravity(false);  // Turn off the default method for applying gravity to the mesh since it is less
                                       // efficient for ANCF elements

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
                //auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzDD>(NodeList[in]);
                auto Node = std::static_pointer_cast<ChNodeFEAxyzDD>(NodeList[in]);
                Node->SetPos(Node->GetPos() + 1e-6 * Perturbation);
                Node->SetD(Node->GetD() + 1e-6 * Perturbation);
                Node->SetDD(Node->GetDD() + 1e-6 * Perturbation);
            }
        }
        else {
            for (int in = 0; in < NodeList.size(); in++) {
                Perturbation.eigen().Random();
                //auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzDD>(NodeList[in]);
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
        }
        else {
            for (int ie = 0; ie < ElementList.size(); ie++) {
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
        }
        else {
            for (int ie = 0; ie < ElementList.size(); ie++) {
                ElementList[ie]->ComputeKRMmatricesGlobal(H, 1.0, 1.0, 1.0);
            }
        }

        timer_KRM.stop();
    }
    return (timer_KRM());
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
void ANCFBeam3333Test<num_elements, ElementVersion, MaterialVersion>::PrintTimingResults(const std::string& TestName, int steps) {

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Times;
    Times.resize(steps, 2);

    //Prime the test in case the internal force calcs are also needed for the Jacobian calcs
    //double TimeInternalFrc = GetInternalFrc(false);
    //double TimeKRM = GetJacobian(false);

    ChTimer<> Timer_Total;
    Timer_Total.reset();
    Timer_Total.start();


    //Run Single Threaded Tests
    for (auto i = 0; i < steps; i++) {
        PerturbNodes(false);
        Times(i, 0) = GetInternalFrc(false) * (1.0e6 / double(num_elements)); //Get Time Per Function Call in microseconds
    }
    for (auto i = 0; i < steps; i++) {
        PerturbNodes(false);
        Times(i, 1) = GetJacobian(false) * (1.0e6 / double(num_elements)); //Get Time Per Function Call in microseconds
    }

    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Test Name, Num Els, Steps, Threads, ";
    std::cout << "IntFrc Max(us), IntFrc Min(us), IntFrc Mean(us), IntFrc StDev(us), ";
    std::cout << "KRM Max(us), KRM Min(us), KRM Mean(us), KRM StDev(us), " << std::endl;
    std::cout << TestName << ", " << num_elements << ", " << steps << ", 0, "
        << Times.col(0).maxCoeff() << ", "
        << Times.col(0).minCoeff() << ", "
        << Times.col(0).mean() << ", "
        << std::sqrt((Times.col(0).array() - Times.col(0).mean()).square().sum() / (Times.col(0).size() - 1)) << ", "
        << Times.col(1).maxCoeff() << ", "
        << Times.col(1).minCoeff() << ", "
        << Times.col(1).mean() << ", "
        << std::sqrt((Times.col(1).array() - Times.col(1).mean()).square().sum() / (Times.col(1).size() - 1)) << std::endl;



    ////Run Multi-Threaded Tests

    //int MaxThreads = 1;
    //MaxThreads = ChOMP::GetNumProcs();

    //int NumThreads = 1;
    //bool run = true;

    //int RunNum = 1;
    //while (run) {
    //    ChOMP::SetNumThreads(NumThreads);

    //    for (auto i = 0; i < steps; i++) {
    //        PerturbNodes(true);
    //        Times(i, 0) = GetInternalFrc(true) * (1.0e6 / double(num_elements)); //Get Time Per Function Call in microseconds
    //    }
    //    for (auto i = 0; i < steps; i++) {
    //        PerturbNodes(true);
    //        Times(i, 1) = GetJacobian(true) * (1.0e6 / double(num_elements)); //Get Time Per Function Call in microseconds
    //    }
    //    std::cout << TestName << ", " << num_elements << ", " << steps << ", " << NumThreads << ", "
    //        << Times.col(0).maxCoeff() << ", "
    //        << Times.col(0).minCoeff() << ", "
    //        << Times.col(0).mean() << ", "
    //        << std::sqrt((Times.col(0).array() - Times.col(0).mean()).square().sum() / (Times.col(0).size() - 1)) << ", "
    //        << Times.col(1).maxCoeff() << ", "
    //        << Times.col(1).minCoeff() << ", "
    //        << Times.col(1).mean() << ", "
    //        << std::sqrt((Times.col(1).array() - Times.col(1).mean()).square().sum() / (Times.col(1).size() - 1)) << std::endl;

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

void Run_ANCFBeam_3333_Tests() {
    const int num_elements = 1024;
    int num_steps = 1;

    //const int num_elements = 8;
    //int num_steps = 1;

    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF, ChMaterialBeamANCF> Beam3333Test_TR01;
        Beam3333Test_TR01.PrintTimingResults("ChElementBeamANCF_3333_Org", num_steps);
    }
    //{
    //    ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR00, ChMaterialBeamANCF_3333_TR00> Beam3333Test_TR00;
    //    Beam3333Test_TR00.PrintTimingResults("ChElementBeamANCF_3333_TR00", num_steps);
    //}
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR01, ChMaterialBeamANCF_3333_TR01> Beam3333Test_TR01;
        Beam3333Test_TR01.PrintTimingResults("ChElementBeamANCF_3333_TR01", num_steps);
    }
    //{
    //    ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR02, ChMaterialBeamANCF_3333_TR02> Beam3333Test_TR02;
    //    Beam3333Test_TR02.PrintTimingResults("ChElementBeamANCF_3333_TR02", num_steps);
    //}
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR02_GQ322, ChMaterialBeamANCF_3333_TR02_GQ322> Beam3333Test_TR02_GQ322;
        Beam3333Test_TR02_GQ322.PrintTimingResults("ChElementBeamANCF_3333_TR02_GQ322", num_steps);
    }
    //{
    //    ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR03, ChMaterialBeamANCF_3333_TR03> Beam3333Test_TR03;
    //    Beam3333Test_TR03.PrintTimingResults("ChElementBeamANCF_3333_TR03", num_steps);
    //}
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR03_GQ322, ChMaterialBeamANCF_3333_TR03_GQ322> Beam3333Test_TR03_GQ322;
        Beam3333Test_TR03_GQ322.PrintTimingResults("ChElementBeamANCF_3333_TR03_GQ322", num_steps);
    }
    //{
    //    ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR04, ChMaterialBeamANCF_3333_TR04> Beam3333Test_TR04;
    //    Beam3333Test_TR04.PrintTimingResults("ChElementBeamANCF_3333_TR04", num_steps);
    //}
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR04_GQ322, ChMaterialBeamANCF_3333_TR04_GQ322> Beam3333Test_TR04_GQ322;
        Beam3333Test_TR04_GQ322.PrintTimingResults("ChElementBeamANCF_3333_TR04_GQ322", num_steps);
    }
    //{
    //    ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR05, ChMaterialBeamANCF_3333_TR05> Beam3333Test_TR05;
    //    Beam3333Test_TR05.PrintTimingResults("ChElementBeamANCF_3333_TR05", num_steps);
    //}
    num_steps = 10;
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR05_GQ322, ChMaterialBeamANCF_3333_TR05_GQ322> Beam3333Test_TR05_GQ322;
        Beam3333Test_TR05_GQ322.PrintTimingResults("ChElementBeamANCF_3333_TR05_GQ322", num_steps);
    }
    //{
    //    ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR06, ChMaterialBeamANCF_3333_TR06> Beam3333Test_TR06;
    //    Beam3333Test_TR06.PrintTimingResults("ChElementBeamANCF_3333_TR06", num_steps);
    //}
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR06_GQ322, ChMaterialBeamANCF_3333_TR06_GQ322> Beam3333Test_TR06_GQ322;
        Beam3333Test_TR06_GQ322.PrintTimingResults("ChElementBeamANCF_3333_TR06_GQ322", num_steps);
    }
    //{
    //    ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR07, ChMaterialBeamANCF_3333_TR07> Beam3333Test_TR07;
    //    Beam3333Test_TR07.PrintTimingResults("ChElementBeamANCF_3333_TR07", num_steps);
    //}
    //{
    //    ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR07S, ChMaterialBeamANCF_3333_TR07S> Beam3333Test_TR07S;
    //    Beam3333Test_TR07S.PrintTimingResults("ChElementBeamANCF_3333_TR07S", num_steps);
    //}
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR07S_GQ322, ChMaterialBeamANCF_3333_TR07S_GQ322> Beam3333Test_TR07S_GQ322;
        Beam3333Test_TR07S_GQ322.PrintTimingResults("ChElementBeamANCF_3333_TR07S_GQ322", num_steps);
    }
    //{
    //    ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR08, ChMaterialBeamANCF_3333_TR08> Beam3333Test_TR08;
    //    Beam3333Test_TR08.PrintTimingResults("ChElementBeamANCF_3333_TR08", num_steps);
    //}
    //{
    //    ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR08S, ChMaterialBeamANCF_3333_TR08S> Beam3333Test_TR08S;
    //    Beam3333Test_TR08S.PrintTimingResults("ChElementBeamANCF_3333_TR08S", num_steps);
    //}
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR08S_GQ322, ChMaterialBeamANCF_3333_TR08S_GQ322> Beam3333Test_TR08S_GQ322;
        Beam3333Test_TR08S_GQ322.PrintTimingResults("ChElementBeamANCF_3333_TR08S_GQ322", num_steps);
    }
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR09, ChMaterialBeamANCF_3333_TR09> Beam3333Test_TR09;
        Beam3333Test_TR09.PrintTimingResults("ChElementBeamANCF_3333_TR09", num_steps);
    }
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR10, ChMaterialBeamANCF_3333_TR10> Beam3333Test_TR10;
        Beam3333Test_TR10.PrintTimingResults("ChElementBeamANCF_3333_TR10", num_steps);
    }
    //{
    //    ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR11, ChMaterialBeamANCF_3333_TR11> Beam3333Test_TR11;
    //    Beam3333Test_TR11.PrintTimingResults("ChElementBeamANCF_3333_TR11", num_steps);
    //}
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333_TR11S, ChMaterialBeamANCF_3333_TR11S> Beam3333Test_TR11S;
        Beam3333Test_TR11S.PrintTimingResults("ChElementBeamANCF_3333_TR11S", num_steps);
    }
    {
        ANCFBeam3333Test<num_elements, ChElementBeamANCF_3333<>, ChMaterialBeamANCF> Beam3333Test;
        Beam3333Test.PrintTimingResults("ChElementBeamANCF_3333_Final", num_steps);
    }
}

// =============================================================================

int main(int argc, char* argv[]) {

    Run_ANCFBeam_3333_Tests();

    return (0);
}

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

#include "chrono/fea/ChElementBrickANCF_3843_TR01.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR02.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR03.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR04.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR05.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR06.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR07.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR07s.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR08.h"
//#include "chrono/fea/ChElementBrickANCF_3843_TR08b.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR08s.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR09.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR10.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR11.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR11s.h"

#include "chrono/fea/ChMesh.h"

using namespace chrono;
using namespace chrono::fea;

template <int num_elements, typename ElementVersion, typename MaterialVersion>
class ANCFBrickTest {
  public:
    ANCFBrickTest();

    ~ANCFBrickTest() { delete m_system; }

    ChSystem* GetSystem() { return m_system; }

    void PerturbNodes();
    double GetInternalFrc();
    double GetJacobian();

    void PrintTimingResults(int steps);

  protected:
    ChSystemSMC* m_system;
};

template <int num_elements, typename ElementVersion, typename MaterialVersion>
ANCFBrickTest<num_elements, ElementVersion, MaterialVersion>::ANCFBrickTest() {
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
    double length = 10.0;    // m
    double width = 0.1;  // m
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

    // Setup Brick normals to initially align with the global axes
    ChVector<> dir1(1, 0, 0);
    ChVector<> dir2(0, 1, 0);
    ChVector<> dir3(0, 0, 1);

    // Create the first nodes and fix them completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, -0.5*width, -0.5*width), dir1, dir2, dir3);
    mesh->AddNode(nodeA);
    auto nodeD = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, 0.5*width, -0.5*width), dir1, dir2, dir3);
    mesh->AddNode(nodeD);
    auto nodeE = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, -0.5*width, 0.5*width), dir1, dir2, dir3);
    mesh->AddNode(nodeE);
    auto nodeH = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, 0.5*width, 0.5*width), dir1, dir2, dir3);
    mesh->AddNode(nodeE);

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(dx * i, -0.5*width, -0.5*width), dir1, dir2, dir3);
        mesh->AddNode(nodeB);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(dx * i, 0.5*width, -0.5*width), dir1, dir2, dir3);
        mesh->AddNode(nodeC);
        auto nodeF = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(dx * i, -0.5*width, 0.5*width), dir1, dir2, dir3);
        mesh->AddNode(nodeF);
        auto nodeG = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(dx * i, 0.5*width, 0.5*width), dir1, dir2, dir3);
        mesh->AddNode(nodeG);

        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC, nodeD, nodeE, nodeF, nodeG, nodeH);
        element->SetDimensions(dx, width, width);
        element->SetMaterial(material);
        element->SetAlphaDamp(0.1);
        element->SetGravityOn(
            false);  // Enable the efficient ANCF method for calculating the application of gravity to the element
        //element->SetStrainFormulation(ElementVersion::StrainFormulation::CMPoisson);
        // element->SetStrainFormulation(ElementVersion::StrainFormulation::CMNoPoisson);
        mesh->AddElement(element);

        nodeA = nodeB;
        nodeD = nodeC;
        nodeE = nodeF;
        nodeH = nodeG;
    }

    mesh->SetAutomaticGravity(false);  // Turn off the default method for applying gravity to the mesh since it is less
                                       // efficient for ANCF elements

    m_system->Update();  // Need to call all the element SetupInital() functions
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
void ANCFBrickTest<num_elements, ElementVersion, MaterialVersion>::PerturbNodes() {
    auto MeshList = m_system->Get_meshlist();
    for (auto& Mesh : MeshList) {
        auto NodeList = Mesh->GetNodes();
        ChVector<double> Perturbation;
        //#pragma omp parallel for
        for (unsigned int in = 0; in < NodeList.size(); in++) {
            Perturbation.eigen().Random();
            //auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzDDD>(NodeList[in]);
            auto Node = std::static_pointer_cast<ChNodeFEAxyzDDD>(NodeList[in]);
            Node->SetPos(Node->GetPos() + 1e-6 * Perturbation);
            Node->SetD(Node->GetD() + 1e-6 * Perturbation);
            Node->SetDD(Node->GetDD() + 1e-6 * Perturbation);
            Node->SetDDD(Node->GetDDD() + 1e-6 * Perturbation);
        }
    }
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
double ANCFBrickTest<num_elements, ElementVersion, MaterialVersion>::GetInternalFrc() {
    ChTimer<> timer_internal_forces;
    timer_internal_forces.reset();

    ChVectorDynamic<double> Fi(96);

    auto MeshList = m_system->Get_meshlist();
    for (auto& Mesh : MeshList) {
        timer_internal_forces.start();
        auto ElementList = Mesh->GetElements();

        //#pragma omp parallel for
        for (unsigned int ie = 0; ie < ElementList.size(); ie++) {
            ElementList[ie]->ComputeInternalForces(Fi);
        }
        timer_internal_forces.stop();
    }
    return (timer_internal_forces());
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
double ANCFBrickTest<num_elements, ElementVersion, MaterialVersion>::GetJacobian() {
    ChTimer<> timer_KRM;
    timer_KRM.reset();

    ChMatrixNM<double, 96, 96> H;

    auto MeshList = m_system->Get_meshlist();
    for (auto& Mesh : MeshList) {
        timer_KRM.start();
        auto ElementList = Mesh->GetElements();

        //#pragma omp parallel for
        for (unsigned int ie = 0; ie < ElementList.size(); ie++) {
            ElementList[ie]->ComputeKRMmatricesGlobal(H, 1.0, 1.0, 1.0);
        }
        timer_KRM.stop();
    }
    return (timer_KRM());
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
void ANCFBrickTest<num_elements, ElementVersion, MaterialVersion>::PrintTimingResults(int steps) {
    double TimeInternalFrc = GetInternalFrc();
    double TimeKRM = GetJacobian();

    ChTimer<> Timer_Total;
    Timer_Total.reset();
    Timer_Total.start();

    TimeInternalFrc = 0;
    TimeKRM = 0;

    for (auto i = 0; i < steps; i++) {
        PerturbNodes();
        TimeInternalFrc += GetInternalFrc();
        //TimeKRM += GetJacobian();
    }

    Timer_Total.stop();

    std::cout << TimeInternalFrc * (1.0e6 / double(steps * num_elements)) << ", ";
    std::cout << TimeKRM * (1.0e6 / double(steps * num_elements)) << ", ";
    std::cout << Timer_Total() * 1.0e3 << std::endl;
}

// =============================================================================

int main(int argc, char* argv[]) {
    double TimeInternalFrc = 0;
    double TimeKRM = 0;
    //int num_steps = 1000;
    int num_steps = 100;
#define NUM_ELEMENTS 100
//1024

    std::cout << "Element, Avg Internal Force Time per Element(micro s), Avg Jacobian Time per Element(micro s), Total "
                 "Calc Time (ms)"
              << std::endl;
    {
        ANCFBrickTest<NUM_ELEMENTS, ChElementBrickANCF_3843_TR01, ChMaterialBrickANCF_3843_TR01> BrickTest_TR01;
        std::cout << "ChElementBrickANCF_3843_TR01, ";
        BrickTest_TR01.PrintTimingResults(num_steps);
    }
    {
        ANCFBrickTest<NUM_ELEMENTS, ChElementBrickANCF_3843_TR02, ChMaterialBrickANCF_3843_TR02> BrickTest_TR02;
        std::cout << "ChElementBrickANCF_3843_TR02, ";
        BrickTest_TR02.PrintTimingResults(num_steps);
    }
    {
        ANCFBrickTest<NUM_ELEMENTS, ChElementBrickANCF_3843_TR03, ChMaterialBrickANCF_3843_TR03> BrickTest_TR03;
        std::cout << "ChElementBrickANCF_3843_TR03, ";
        BrickTest_TR03.PrintTimingResults(num_steps);
    }
    {
        ANCFBrickTest<NUM_ELEMENTS, ChElementBrickANCF_3843_TR04, ChMaterialBrickANCF_3843_TR04> BrickTest_TR04;
        std::cout << "ChElementBrickANCF_3843_TR04, ";
        BrickTest_TR04.PrintTimingResults(num_steps);
    }
    {
        ANCFBrickTest<NUM_ELEMENTS, ChElementBrickANCF_3843_TR05, ChMaterialBrickANCF_3843_TR05> BrickTest_TR05;
        std::cout << "ChElementBrickANCF_3843_TR05, ";
        BrickTest_TR05.PrintTimingResults(num_steps);
    }
    {
        ANCFBrickTest<NUM_ELEMENTS, ChElementBrickANCF_3843_TR06, ChMaterialBrickANCF_3843_TR06> BrickTest_TR06;
        std::cout << "ChElementBrickANCF_3843_TR06, ";
        BrickTest_TR06.PrintTimingResults(num_steps);
    }
    {
        ANCFBrickTest<NUM_ELEMENTS, ChElementBrickANCF_3843_TR07, ChMaterialBrickANCF_3843_TR07> BrickTest_TR07;
        std::cout << "ChElementBrickANCF_3843_TR07, ";
        BrickTest_TR07.PrintTimingResults(num_steps);
    }
    {
        ANCFBrickTest<NUM_ELEMENTS, ChElementBrickANCF_3843_TR07S, ChMaterialBrickANCF_3843_TR07S> BrickTest_TR07S;
        std::cout << "ChElementBrickANCF_3843_TR07S, ";
        BrickTest_TR07S.PrintTimingResults(num_steps);
    }
    {
        ANCFBrickTest<NUM_ELEMENTS, ChElementBrickANCF_3843_TR08, ChMaterialBrickANCF_3843_TR08> BrickTest_TR08;
        std::cout << "ChElementBrickANCF_3843_TR08, ";
        BrickTest_TR08.PrintTimingResults(num_steps);
    }
    //ANCFBrickTest<NUM_ELEMENTS, ChElementBrickANCF_3843_TR08b, ChMaterialBrickANCF_3843_TR08b> BrickTest_TR08b;
    //std::cout << "ChElementBrickANCF_3843_TR08b, ";
    //BrickTest_TR08b.PrintTimingResults(num_steps);
    {
        ANCFBrickTest<NUM_ELEMENTS, ChElementBrickANCF_3843_TR08S, ChMaterialBrickANCF_3843_TR08S> BrickTest_TR08S;
        std::cout << "ChElementBrickANCF_3843_TR08S, ";
        BrickTest_TR08S.PrintTimingResults(num_steps);
    }
    {
        ANCFBrickTest<NUM_ELEMENTS, ChElementBrickANCF_3843_TR09, ChMaterialBrickANCF_3843_TR09> BrickTest_TR09;
        std::cout << "ChElementBrickANCF_3843_TR09, ";
        BrickTest_TR09.PrintTimingResults(num_steps);
    }
    {
        ANCFBrickTest<NUM_ELEMENTS, ChElementBrickANCF_3843_TR10, ChMaterialBrickANCF_3843_TR10> BrickTest_TR10;
        std::cout << "ChElementBrickANCF_3843_TR10, ";
        BrickTest_TR10.PrintTimingResults(num_steps);
    }
    {
        ANCFBrickTest<NUM_ELEMENTS, ChElementBrickANCF_3843_TR11, ChMaterialBrickANCF_3843_TR11> BrickTest_TR11;
        std::cout << "ChElementBrickANCF_3843_TR11, ";
        BrickTest_TR11.PrintTimingResults(num_steps);
    }
    {
        ANCFBrickTest<NUM_ELEMENTS, ChElementBrickANCF_3843_TR11S, ChMaterialBrickANCF_3843_TR11S> BrickTest_TR11S;
        std::cout << "ChElementBrickANCF_3843_TR11S, ";
        BrickTest_TR11S.PrintTimingResults(num_steps);
    }

    return (0);
}

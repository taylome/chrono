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

#include "chrono/physics/ChSystemSMC.h"
#include "chrono/solver/ChDirectSolverLS.h"

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
#include "chrono/fea/ChElementBeamANCF_MT27A.h"
#include "chrono/fea/ChElementBeamANCF_MT27B.h"
#include "chrono/fea/ChElementBeamANCF_MT27C.h"
#include "chrono/fea/ChElementBeamANCF_MT28.h"
#include "chrono/fea/ChElementBeamANCF_MT29.h"
#include "chrono/fea/ChElementBeamANCF_MT30.h"
#include "chrono/fea/ChElementBeamANCF_MT31.h"
#include "chrono/fea/ChElementBeamANCF_MT32.h"
#include "chrono/fea/ChElementBeamANCF_MT33.h"
#include "chrono/fea/ChElementBeamANCF_MT34.h"
#include "chrono/fea/ChElementBeamANCF_MT35.h"
#include "chrono/fea/ChElementBeamANCF_MT36.h"
#include "chrono/fea/ChElementBeamANCF_MT37.h"
#include "chrono/fea/ChElementBeamANCF_MT38.h"
#include "chrono/fea/ChElementBeamANCF_MT39.h"
#include "chrono/fea/ChElementBeamANCF_MT40.h"
#include "chrono/fea/ChElementBeamANCF_MT41.h"
#include "chrono/fea/ChElementBeamANCF_MT60.h"
#include "chrono/fea/ChElementBeamANCF_MT61.h"
#include "chrono/fea/ChElementBeamANCF_MT62.h"
#include "chrono/fea/ChElementBeamANCF_MT63.h"
#include "chrono/fea/ChElementBeamANCF_MT64.h"

#include "chrono/fea/ChMesh.h"

using namespace chrono;
using namespace chrono::fea;

template <int num_elements, typename ElementVersion, typename MaterialVersion>
class ANCFBeamTest{
  public:
    ANCFBeamTest();

    ~ANCFBeamTest() { delete m_system; }

    ChSystem* GetSystem() { return m_system; }
	
	void PerturbNodes();
	double GetInternalFrc();
	double GetJacobian();

    void PrintTimingResults(int steps);

  protected:
    ChSystemSMC* m_system;
};

template <int num_elements, typename ElementVersion, typename MaterialVersion>
ANCFBeamTest<num_elements, ElementVersion, MaterialVersion>::ANCFBeamTest() {
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
    nodeA->SetFixed(true);
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
        element->SetStrainFormulation(ElementVersion::StrainFormulation::CMPoisson);
        // element->SetStrainFormulation(ElementVersion::StrainFormulation::CMNoPoisson);
        mesh->AddElement(element);

        nodeA = nodeB;
    }

    mesh->SetAutomaticGravity(false);  // Turn off the default method for applying gravity to the mesh since it is less
                                       // efficient for ANCF elements

	m_system->Update();  //Need to call all the element SetupInital() functions

}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
void ANCFBeamTest<num_elements, ElementVersion, MaterialVersion>::PerturbNodes() {
	auto MeshList = m_system->Get_meshlist();
	for (auto & Mesh : MeshList) {
		auto NodeList = Mesh->GetNodes();
        for (unsigned int in = 0; in < NodeList.size(); in++) {
			ChVector<double> Perturbation;
			Perturbation.eigen().Random();
            auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzDD>(NodeList[in]);
            Node->SetPos(Node->GetPos()+1e6*Perturbation);
            Node->SetD(Node->GetD() + 1e6*Perturbation);
            Node->SetDD(Node->GetDD() + 1e6*Perturbation);
		}
    }
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
double ANCFBeamTest<num_elements, ElementVersion, MaterialVersion>::GetInternalFrc() {
	ChTimer<> timer_internal_forces;
	timer_internal_forces.reset();
	
	ChVectorDynamic<double> Fi(27);
	
	auto MeshList = m_system->Get_meshlist();
	for (auto & Mesh : MeshList) {
		auto ElementList = Mesh->GetElements();
        for (unsigned int ie = 0; ie < ElementList.size(); ie++) {
            timer_internal_forces.start();
            ElementList[ie]->ComputeInternalForces(Fi);
            timer_internal_forces.stop();
        }
    }
	return(timer_internal_forces());
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
double ANCFBeamTest<num_elements, ElementVersion, MaterialVersion>::GetJacobian() {
	ChTimer<> timer_KRM;
	timer_KRM.reset();
	
	ChMatrixNM<double,27,27> H;
	
	auto MeshList = m_system->Get_meshlist();
	for (auto & Mesh : MeshList) {
        auto ElementList = Mesh->GetElements();
        for (unsigned int ie = 0; ie < ElementList.size(); ie++) {
			timer_KRM.start();
            ElementList[ie]->ComputeKRMmatricesGlobal(H, 1.0, 1.0, 1.0);
			timer_KRM.stop();
		}
    }
	return(timer_KRM());
}

template <int num_elements, typename ElementVersion, typename MaterialVersion>
void ANCFBeamTest<num_elements, ElementVersion, MaterialVersion>::PrintTimingResults(int steps) {
    
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
        TimeKRM += GetJacobian();
    }

    Timer_Total.stop();

    //std::cout << "Avg Internal Force Time per Element: " << TimeInternalFrc * (1e6 / double(steps*num_elements)) << "µs" << std::endl;
    //std::cout << "Avg Jacobian Time per Element:       " << TimeKRM * (1e6 / double(steps*num_elements)) << "µs" << std::endl;
    std::cout << TimeInternalFrc * (1.0e6 / double(steps*num_elements)) << ", ";
    std::cout << TimeKRM * (1.0e6 / double(steps*num_elements)) << ", "; 
    std::cout << Timer_Total()*1.0e3 << std::endl;
}

// =============================================================================

int main(int argc, char* argv[]) {

    double TimeInternalFrc = 0;
    double TimeKRM = 0;
    int num_steps = 10*1000;
    #define NUM_ELEMENTS 100

    std::cout << "Element, Avg Internal Force Time per Element(micro s), Avg Jacobian Time per Element(micro s), Total Calc Time (ms)" << std::endl;

	ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF, ChMaterialBeamANCF> BeamTest_Org;
    std::cout << "ChElementBeamANCF_Org, "; BeamTest_Org.PrintTimingResults(num_steps);
    
    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT01, ChMaterialBeamANCF_MT01> BeamTest_MT01;
    std::cout << "ChElementBeamANCF_MT01, "; BeamTest_MT01.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT02, ChMaterialBeamANCF_MT02> BeamTest_MT02;
    std::cout << "ChElementBeamANCF_MT02, "; BeamTest_MT02.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT03, ChMaterialBeamANCF_MT03> BeamTest_MT03;
    std::cout << "ChElementBeamANCF_MT03, "; BeamTest_MT03.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT04, ChMaterialBeamANCF_MT04> BeamTest_MT04;
    std::cout << "ChElementBeamANCF_MT04, "; BeamTest_MT04.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT05, ChMaterialBeamANCF_MT05> BeamTest_MT05;
    std::cout << "ChElementBeamANCF_MT05, "; BeamTest_MT05.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT06, ChMaterialBeamANCF_MT06> BeamTest_MT06;
    std::cout << "ChElementBeamANCF_MT06, "; BeamTest_MT06.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT07, ChMaterialBeamANCF_MT07> BeamTest_MT07;
    std::cout << "ChElementBeamANCF_MT07, "; BeamTest_MT07.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT08, ChMaterialBeamANCF_MT08> BeamTest_MT08;
    std::cout << "ChElementBeamANCF_MT08, "; BeamTest_MT08.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT09, ChMaterialBeamANCF_MT09> BeamTest_MT09;
    std::cout << "ChElementBeamANCF_MT09, "; BeamTest_MT09.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT10, ChMaterialBeamANCF_MT10> BeamTest_MT10;
    std::cout << "ChElementBeamANCF_MT10, "; BeamTest_MT10.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT11, ChMaterialBeamANCF_MT11> BeamTest_MT11;
    std::cout << "ChElementBeamANCF_MT11, "; BeamTest_MT11.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT12, ChMaterialBeamANCF_MT12> BeamTest_MT12;
    std::cout << "ChElementBeamANCF_MT12, "; BeamTest_MT12.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT13, ChMaterialBeamANCF_MT13> BeamTest_MT13;
    std::cout << "ChElementBeamANCF_MT13, "; BeamTest_MT13.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT14, ChMaterialBeamANCF_MT14> BeamTest_MT14;
    std::cout << "ChElementBeamANCF_MT14, "; BeamTest_MT14.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT15, ChMaterialBeamANCF_MT15> BeamTest_MT15;
    std::cout << "ChElementBeamANCF_MT15, "; BeamTest_MT15.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT16, ChMaterialBeamANCF_MT16> BeamTest_MT16;
    std::cout << "ChElementBeamANCF_MT16, "; BeamTest_MT16.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT17, ChMaterialBeamANCF_MT17> BeamTest_MT17;
    std::cout << "ChElementBeamANCF_MT17, "; BeamTest_MT17.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT18, ChMaterialBeamANCF_MT18> BeamTest_MT18;
    std::cout << "ChElementBeamANCF_MT18, "; BeamTest_MT18.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT19, ChMaterialBeamANCF_MT19> BeamTest_MT19;
    std::cout << "ChElementBeamANCF_MT19, "; BeamTest_MT19.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT20, ChMaterialBeamANCF_MT20> BeamTest_MT20;
    std::cout << "ChElementBeamANCF_MT20, "; BeamTest_MT20.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT21, ChMaterialBeamANCF_MT21> BeamTest_MT21;
    std::cout << "ChElementBeamANCF_MT21, "; BeamTest_MT21.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT22, ChMaterialBeamANCF_MT22> BeamTest_MT22;
    std::cout << "ChElementBeamANCF_MT22, "; BeamTest_MT22.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT23, ChMaterialBeamANCF_MT23> BeamTest_MT23;
    std::cout << "ChElementBeamANCF_MT23, "; BeamTest_MT23.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT24, ChMaterialBeamANCF_MT24> BeamTest_MT24;
    std::cout << "ChElementBeamANCF_MT24, "; BeamTest_MT24.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT25, ChMaterialBeamANCF_MT25> BeamTest_MT25;
    std::cout << "ChElementBeamANCF_MT25, "; BeamTest_MT25.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> BeamTest_MT26;
    std::cout << "ChElementBeamANCF_MT26, "; BeamTest_MT26.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT27, ChMaterialBeamANCF_MT27> BeamTest_MT27;
    std::cout << "ChElementBeamANCF_MT27, "; BeamTest_MT27.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT27A, ChMaterialBeamANCF_MT27A> BeamTest_MT27A;
    std::cout << "ChElementBeamANCF_MT27A, "; BeamTest_MT27A.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT27B, ChMaterialBeamANCF_MT27B> BeamTest_MT27B;
    std::cout << "ChElementBeamANCF_MT27B, "; BeamTest_MT27B.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT27C, ChMaterialBeamANCF_MT27C> BeamTest_MT27C;
    std::cout << "ChElementBeamANCF_MT27C, "; BeamTest_MT27C.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT28, ChMaterialBeamANCF_MT28> BeamTest_MT28;
    std::cout << "ChElementBeamANCF_MT28, "; BeamTest_MT28.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT29, ChMaterialBeamANCF_MT29> BeamTest_MT29;
    std::cout << "ChElementBeamANCF_MT29, "; BeamTest_MT29.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT30, ChMaterialBeamANCF_MT30> BeamTest_MT30;
    std::cout << "ChElementBeamANCF_MT30, "; BeamTest_MT30.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> BeamTest_MT31;
    std::cout << "ChElementBeamANCF_MT31, "; BeamTest_MT31.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT32, ChMaterialBeamANCF_MT32> BeamTest_MT32;
    std::cout << "ChElementBeamANCF_MT32, "; BeamTest_MT32.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT33, ChMaterialBeamANCF_MT33> BeamTest_MT33;
    std::cout << "ChElementBeamANCF_MT33, "; BeamTest_MT33.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT34, ChMaterialBeamANCF_MT34> BeamTest_MT34;
    std::cout << "ChElementBeamANCF_MT34, "; BeamTest_MT34.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT35, ChMaterialBeamANCF_MT35> BeamTest_MT35;
    std::cout << "ChElementBeamANCF_MT35, "; BeamTest_MT35.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT36, ChMaterialBeamANCF_MT36> BeamTest_MT36;
    std::cout << "ChElementBeamANCF_MT36, "; BeamTest_MT36.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT37, ChMaterialBeamANCF_MT37> BeamTest_MT37;
    std::cout << "ChElementBeamANCF_MT37, "; BeamTest_MT37.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT38, ChMaterialBeamANCF_MT38> BeamTest_MT38;
    std::cout << "ChElementBeamANCF_MT38, "; BeamTest_MT38.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT39, ChMaterialBeamANCF_MT39> BeamTest_MT39;
    std::cout << "ChElementBeamANCF_MT39, "; BeamTest_MT39.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT40, ChMaterialBeamANCF_MT40> BeamTest_MT40;
    std::cout << "ChElementBeamANCF_MT40, "; BeamTest_MT40.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT41, ChMaterialBeamANCF_MT41> BeamTest_MT41;
    std::cout << "ChElementBeamANCF_MT41, "; BeamTest_MT41.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT60, ChMaterialBeamANCF_MT60> BeamTest_MT60;
    std::cout << "ChElementBeamANCF_MT60, "; BeamTest_MT60.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT61, ChMaterialBeamANCF_MT61> BeamTest_MT61;
    std::cout << "ChElementBeamANCF_MT61, "; BeamTest_MT61.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT62, ChMaterialBeamANCF_MT62> BeamTest_MT62;
    std::cout << "ChElementBeamANCF_MT62, "; BeamTest_MT62.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT63, ChMaterialBeamANCF_MT63> BeamTest_MT63;
    std::cout << "ChElementBeamANCF_MT63, "; BeamTest_MT63.PrintTimingResults(num_steps);

    ANCFBeamTest<NUM_ELEMENTS, ChElementBeamANCF_MT64, ChMaterialBeamANCF_MT64> BeamTest_MT64;
    std::cout << "ChElementBeamANCF_MT64, "; BeamTest_MT64.PrintTimingResults(num_steps);

    return(0);
}

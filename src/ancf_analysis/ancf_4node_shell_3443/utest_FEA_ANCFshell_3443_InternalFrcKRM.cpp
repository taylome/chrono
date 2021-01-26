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
// Unit test for ANCF beam elements (Straight Square Beam)
//
// =============================================================================

#include "chrono/ChConfig.h"

#include "chrono/physics/ChSystemSMC.h"
#include "chrono/solver/ChDirectSolverLS.h"

#include "chrono/fea/ChElementShellANCF_3443_TR01.h"
#include "chrono/fea/ChElementShellANCF_3443_TR02.h"
#include "chrono/fea/ChElementShellANCF_3443_TR03.h"
#include "chrono/fea/ChElementShellANCF_3443_TR04.h"
#include "chrono/fea/ChElementShellANCF_3443_TR05.h"
#include "chrono/fea/ChElementShellANCF_3443_TR06.h"
#include "chrono/fea/ChElementShellANCF_3443_TR07.h"
#include "chrono/fea/ChElementShellANCF_3443_TR08.h"
#include "chrono/fea/ChElementShellANCF_3443_TR08b.h"
#include "chrono/fea/ChElementShellANCF_3443_TR09.h"
#include "chrono/fea/ChElementShellANCF_3443_TR10.h"

#include "chrono/fea/ChMesh.h"
#include "chrono/fea/ChVisualizationFEAmesh.h"
#include "chrono/physics/ChLoadContainer.h"
#include "chrono/fea/ChLoadsBeam.h"

#if defined WIN32 || defined _WIN32 || defined WIN64 || defined _WIN64
#include <windows.h>
#endif

using namespace chrono;
using namespace chrono::fea;

#define TIP_MOMENT 10.0  // Nm
#define TIP_FORCE 10.0   // N

// =============================================================================

void print_green(std::string text) {
    std::cout << "\033[1;32m" << text << "\033[0m";
}

void print_red(std::string text) {
    std::cout << "\033[1;31m" << text << "\033[0m";
}

// =============================================================================

template <typename ElementVersion, typename MaterialVersion>
class ANCFShellTest {
public:
    ANCFShellTest();

    ~ANCFShellTest() { delete m_system; }

    bool RunElementChecks(int msglvl);

    bool MassMatrixCheck(int msglvl);

    bool GeneralizedGravityForceCheck(int msglvl);

    bool GeneralizedInternalForceNoDispNoVelCheck(int msglvl);
    bool GeneralizedInternalForceSmallDispNoVelCheck(int msglvl);
    bool GeneralizedInternalForceNoDispSmallVelCheck(int msglvl);

    bool JacobianNoDispNoVelNoDampingCheck(int msglvl);
    bool JacobianSmallDispNoVelNoDampingCheck(int msglvl);

    bool JacobianNoDispNoVelWithDampingCheck(int msglvl);
    bool JacobianSmallDispNoVelWithDampingCheck(int msglvl);
    bool JacobianNoDispSmallVelWithDampingCheck(int msglvl);

    bool AxialDisplacementCheck(int msglvl);
    bool CantileverCheck(int msglvl);
    bool AxialTwistCheck(int msglvl);
    

protected:
    ChSystemSMC* m_system;
    std::shared_ptr<ElementVersion> m_element;
    std::shared_ptr<ChNodeFEAxyzDDD> m_nodeB;
};

// =============================================================================

template <typename ElementVersion, typename MaterialVersion>
ANCFShellTest<ElementVersion, MaterialVersion>::ANCFShellTest() {
    m_system = new ChSystemSMC();
    m_system->Set_G_acc(ChVector<>(0, 0, -9.80665));

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
    integrator->SetMode(ChTimestepperHHT::POSITION);
    integrator->SetScaling(true);
    integrator->SetVerbose(false);
    integrator->SetModifiedNewton(false);

    // Mesh properties (Steel)
    double length = 1.0;      //m
    double width = 1.0;     //m
    double thickness = 0.01; //m
    double rho = 7850; //kg/m^3
    double E = 210e9; //Pa
    double nu = 0.3;

    auto material = chrono_types::make_shared<MaterialVersion>(rho, E, nu);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    m_system->Add(mesh);

    // Setup beam cross section gradients to initially align with the global y and z directions
    ChVector<> dir1(1, 0, 0);   
    ChVector<> dir2(0, 1, 0);
    ChVector<> dir3(0, 0, 1);

    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, 0, 0.0), dir1, dir2, dir3);
    nodeA->SetFixed(true);
    mesh->AddNode(nodeA);
    auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(length, 0, 0), dir1, dir2, dir3);
    mesh->AddNode(nodeB);
    m_nodeB = nodeB;
    auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(length, width, 0), dir1, dir2, dir3);
    mesh->AddNode(nodeC);
    nodeC = nodeC;
    auto nodeD = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, width, 0), dir1, dir2, dir3);
    mesh->AddNode(nodeD);
    nodeD = nodeD;

    auto element = chrono_types::make_shared<ElementVersion>();
    element->SetNodes(nodeA, nodeB, nodeC, nodeD);
    element->SetDimensions(length, width, thickness);
    element->SetMaterial(material);
    element->SetAlphaDamp(0.0);
    element->SetGravityOn(false);  //Enable the efficient ANCF method for calculating the application of gravity to the element
    element->SetStrainFormulation(ElementVersion::StrainFormulation::CMPoisson);
    mesh->AddElement(element);

    m_element = element;

    mesh->SetAutomaticGravity(false); //Turn off the default method for applying gravity to the mesh since it is less efficient for ANCF elements

// =============================================================================
//  Force a call to SetupInital so that all of the pre-computation steps are called
// =============================================================================

    m_system->Update();
}

template <typename ElementVersion, typename MaterialVersion>
bool ANCFShellTest<ElementVersion, MaterialVersion>::RunElementChecks(int msglvl) {
    bool tests_passed = true;
    tests_passed = (tests_passed & MassMatrixCheck(msglvl));
    tests_passed = (tests_passed & GeneralizedGravityForceCheck(msglvl));

    tests_passed = (tests_passed & GeneralizedInternalForceNoDispNoVelCheck(msglvl));
    tests_passed = (tests_passed & GeneralizedInternalForceSmallDispNoVelCheck(msglvl));
    tests_passed = (tests_passed & GeneralizedInternalForceNoDispSmallVelCheck(msglvl));

    tests_passed = (tests_passed & JacobianNoDispNoVelNoDampingCheck(msglvl));
    tests_passed = (tests_passed & JacobianSmallDispNoVelNoDampingCheck(msglvl));
    tests_passed = (tests_passed & JacobianNoDispNoVelWithDampingCheck(msglvl));
    tests_passed = (tests_passed & JacobianSmallDispNoVelWithDampingCheck(msglvl));
    tests_passed = (tests_passed & JacobianNoDispSmallVelWithDampingCheck(msglvl));
    tests_passed = (tests_passed & JacobianNoDispSmallVelWithDampingCheck(msglvl));

    tests_passed = (tests_passed & AxialDisplacementCheck(msglvl));
    tests_passed = (tests_passed & CantileverCheck(msglvl));
    tests_passed = (tests_passed & AxialTwistCheck(msglvl));

    return(tests_passed);
}

template <typename ElementVersion, typename MaterialVersion>
bool ANCFShellTest<ElementVersion, MaterialVersion>::MassMatrixCheck(int msglvl) {
    // =============================================================================
    //  Check the Mass Matrix
    //  (Result should be nearly exact - No expected error)
    // =============================================================================
    ChMatrixNM<double, 48, 48> Expected_MassMatrix;
    Expected_MassMatrix <<
        10.7594841269841, 0, 0, 1.43605158730159, 0, 0, 1.43605158730159, 0, 0, 0, 0, 0, 3.8190873015873, 0, 0, -0.853531746031746, 0, 0, 0.619900793650794, 0, 0, 0, 0, 0, 1.22734126984127, 0, 0, -0.361349206349206, 0, 0, -0.361349206349206, 0, 0, 0, 0, 0, 3.8190873015873, 0, 0, 0.619900793650794, 0, 0, -0.853531746031746, 0, 0, 0, 0, 0,
        0, 10.7594841269841, 0, 0, 1.43605158730159, 0, 0, 1.43605158730159, 0, 0, 0, 0, 0, 3.8190873015873, 0, 0, -0.853531746031746, 0, 0, 0.619900793650794, 0, 0, 0, 0, 0, 1.22734126984127, 0, 0, -0.361349206349206, 0, 0, -0.361349206349206, 0, 0, 0, 0, 0, 3.8190873015873, 0, 0, 0.619900793650794, 0, 0, -0.853531746031746, 0, 0, 0, 0,
        0, 0, 10.7594841269841, 0, 0, 1.43605158730159, 0, 0, 1.43605158730159, 0, 0, 0, 0, 0, 3.8190873015873, 0, 0, -0.853531746031746, 0, 0, 0.619900793650794, 0, 0, 0, 0, 0, 1.22734126984127, 0, 0, -0.361349206349206, 0, 0, -0.361349206349206, 0, 0, 0, 0, 0, 3.8190873015873, 0, 0, 0.619900793650794, 0, 0, -0.853531746031746, 0, 0, 0,
        1.43605158730159, 0, 0, 0.249206349206349, 0, 0, 0.19625, 0, 0, 0, 0, 0, 0.853531746031746, 0, 0, -0.186904761904762, 0, 0, 0.130833333333333, 0, 0, 0, 0, 0, 0.361349206349206, 0, 0, -0.093452380952381, 0, 0, -0.0872222222222222, 0, 0, 0, 0, 0, 0.619900793650794, 0, 0, 0.124603174603175, 0, 0, -0.130833333333333, 0, 0, 0, 0, 0,
        0, 1.43605158730159, 0, 0, 0.249206349206349, 0, 0, 0.19625, 0, 0, 0, 0, 0, 0.853531746031746, 0, 0, -0.186904761904762, 0, 0, 0.130833333333333, 0, 0, 0, 0, 0, 0.361349206349206, 0, 0, -0.093452380952381, 0, 0, -0.0872222222222222, 0, 0, 0, 0, 0, 0.619900793650794, 0, 0, 0.124603174603175, 0, 0, -0.130833333333333, 0, 0, 0, 0,
        0, 0, 1.43605158730159, 0, 0, 0.249206349206349, 0, 0, 0.19625, 0, 0, 0, 0, 0, 0.853531746031746, 0, 0, -0.186904761904762, 0, 0, 0.130833333333333, 0, 0, 0, 0, 0, 0.361349206349206, 0, 0, -0.093452380952381, 0, 0, -0.0872222222222222, 0, 0, 0, 0, 0, 0.619900793650794, 0, 0, 0.124603174603175, 0, 0, -0.130833333333333, 0, 0, 0,
        1.43605158730159, 0, 0, 0.19625, 0, 0, 0.249206349206349, 0, 0, 0, 0, 0, 0.619900793650794, 0, 0, -0.130833333333333, 0, 0, 0.124603174603175, 0, 0, 0, 0, 0, 0.361349206349206, 0, 0, -0.0872222222222222, 0, 0, -0.093452380952381, 0, 0, 0, 0, 0, 0.853531746031746, 0, 0, 0.130833333333333, 0, 0, -0.186904761904762, 0, 0, 0, 0, 0,
        0, 1.43605158730159, 0, 0, 0.19625, 0, 0, 0.249206349206349, 0, 0, 0, 0, 0, 0.619900793650794, 0, 0, -0.130833333333333, 0, 0, 0.124603174603175, 0, 0, 0, 0, 0, 0.361349206349206, 0, 0, -0.0872222222222222, 0, 0, -0.093452380952381, 0, 0, 0, 0, 0, 0.853531746031746, 0, 0, 0.130833333333333, 0, 0, -0.186904761904762, 0, 0, 0, 0,
        0, 0, 1.43605158730159, 0, 0, 0.19625, 0, 0, 0.249206349206349, 0, 0, 0, 0, 0, 0.619900793650794, 0, 0, -0.130833333333333, 0, 0, 0.124603174603175, 0, 0, 0, 0, 0, 0.361349206349206, 0, 0, -0.0872222222222222, 0, 0, -0.093452380952381, 0, 0, 0, 0, 0, 0.853531746031746, 0, 0, 0.130833333333333, 0, 0, -0.186904761904762, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.81712962962963E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.81712962962963E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.81712962962963E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.63425925925926E-05,
        3.8190873015873, 0, 0, 0.853531746031746, 0, 0, 0.619900793650794, 0, 0, 0, 0, 0, 10.7594841269841, 0, 0, -1.43605158730159, 0, 0, 1.43605158730159, 0, 0, 0, 0, 0, 3.8190873015873, 0, 0, -0.619900793650794, 0, 0, -0.853531746031746, 0, 0, 0, 0, 0, 1.22734126984127, 0, 0, 0.361349206349206, 0, 0, -0.361349206349206, 0, 0, 0, 0, 0,
        0, 3.8190873015873, 0, 0, 0.853531746031746, 0, 0, 0.619900793650794, 0, 0, 0, 0, 0, 10.7594841269841, 0, 0, -1.43605158730159, 0, 0, 1.43605158730159, 0, 0, 0, 0, 0, 3.8190873015873, 0, 0, -0.619900793650794, 0, 0, -0.853531746031746, 0, 0, 0, 0, 0, 1.22734126984127, 0, 0, 0.361349206349206, 0, 0, -0.361349206349206, 0, 0, 0, 0,
        0, 0, 3.8190873015873, 0, 0, 0.853531746031746, 0, 0, 0.619900793650794, 0, 0, 0, 0, 0, 10.7594841269841, 0, 0, -1.43605158730159, 0, 0, 1.43605158730159, 0, 0, 0, 0, 0, 3.8190873015873, 0, 0, -0.619900793650794, 0, 0, -0.853531746031746, 0, 0, 0, 0, 0, 1.22734126984127, 0, 0, 0.361349206349206, 0, 0, -0.361349206349206, 0, 0, 0,
        -0.853531746031746, 0, 0, -0.186904761904762, 0, 0, -0.130833333333333, 0, 0, 0, 0, 0, -1.43605158730159, 0, 0, 0.249206349206349, 0, 0, -0.19625, 0, 0, 0, 0, 0, -0.619900793650794, 0, 0, 0.124603174603175, 0, 0, 0.130833333333333, 0, 0, 0, 0, 0, -0.361349206349206, 0, 0, -0.093452380952381, 0, 0, 0.0872222222222222, 0, 0, 0, 0, 0,
        0, -0.853531746031746, 0, 0, -0.186904761904762, 0, 0, -0.130833333333333, 0, 0, 0, 0, 0, -1.43605158730159, 0, 0, 0.249206349206349, 0, 0, -0.19625, 0, 0, 0, 0, 0, -0.619900793650794, 0, 0, 0.124603174603175, 0, 0, 0.130833333333333, 0, 0, 0, 0, 0, -0.361349206349206, 0, 0, -0.093452380952381, 0, 0, 0.0872222222222222, 0, 0, 0, 0,
        0, 0, -0.853531746031746, 0, 0, -0.186904761904762, 0, 0, -0.130833333333333, 0, 0, 0, 0, 0, -1.43605158730159, 0, 0, 0.249206349206349, 0, 0, -0.19625, 0, 0, 0, 0, 0, -0.619900793650794, 0, 0, 0.124603174603175, 0, 0, 0.130833333333333, 0, 0, 0, 0, 0, -0.361349206349206, 0, 0, -0.093452380952381, 0, 0, 0.0872222222222222, 0, 0, 0,
        0.619900793650794, 0, 0, 0.130833333333333, 0, 0, 0.124603174603175, 0, 0, 0, 0, 0, 1.43605158730159, 0, 0, -0.19625, 0, 0, 0.249206349206349, 0, 0, 0, 0, 0, 0.853531746031746, 0, 0, -0.130833333333333, 0, 0, -0.186904761904762, 0, 0, 0, 0, 0, 0.361349206349206, 0, 0, 0.0872222222222222, 0, 0, -0.093452380952381, 0, 0, 0, 0, 0,
        0, 0.619900793650794, 0, 0, 0.130833333333333, 0, 0, 0.124603174603175, 0, 0, 0, 0, 0, 1.43605158730159, 0, 0, -0.19625, 0, 0, 0.249206349206349, 0, 0, 0, 0, 0, 0.853531746031746, 0, 0, -0.130833333333333, 0, 0, -0.186904761904762, 0, 0, 0, 0, 0, 0.361349206349206, 0, 0, 0.0872222222222222, 0, 0, -0.093452380952381, 0, 0, 0, 0,
        0, 0, 0.619900793650794, 0, 0, 0.130833333333333, 0, 0, 0.124603174603175, 0, 0, 0, 0, 0, 1.43605158730159, 0, 0, -0.19625, 0, 0, 0.249206349206349, 0, 0, 0, 0, 0, 0.853531746031746, 0, 0, -0.130833333333333, 0, 0, -0.186904761904762, 0, 0, 0, 0, 0, 0.361349206349206, 0, 0, 0.0872222222222222, 0, 0, -0.093452380952381, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.81712962962963E-05, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.81712962962963E-05, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.81712962962963E-05,
        1.22734126984127, 0, 0, 0.361349206349206, 0, 0, 0.361349206349206, 0, 0, 0, 0, 0, 3.8190873015873, 0, 0, -0.619900793650794, 0, 0, 0.853531746031746, 0, 0, 0, 0, 0, 10.7594841269841, 0, 0, -1.43605158730159, 0, 0, -1.43605158730159, 0, 0, 0, 0, 0, 3.8190873015873, 0, 0, 0.853531746031746, 0, 0, -0.619900793650794, 0, 0, 0, 0, 0,
        0, 1.22734126984127, 0, 0, 0.361349206349206, 0, 0, 0.361349206349206, 0, 0, 0, 0, 0, 3.8190873015873, 0, 0, -0.619900793650794, 0, 0, 0.853531746031746, 0, 0, 0, 0, 0, 10.7594841269841, 0, 0, -1.43605158730159, 0, 0, -1.43605158730159, 0, 0, 0, 0, 0, 3.8190873015873, 0, 0, 0.853531746031746, 0, 0, -0.619900793650794, 0, 0, 0, 0,
        0, 0, 1.22734126984127, 0, 0, 0.361349206349206, 0, 0, 0.361349206349206, 0, 0, 0, 0, 0, 3.8190873015873, 0, 0, -0.619900793650794, 0, 0, 0.853531746031746, 0, 0, 0, 0, 0, 10.7594841269841, 0, 0, -1.43605158730159, 0, 0, -1.43605158730159, 0, 0, 0, 0, 0, 3.8190873015873, 0, 0, 0.853531746031746, 0, 0, -0.619900793650794, 0, 0, 0,
        -0.361349206349206, 0, 0, -0.093452380952381, 0, 0, -0.0872222222222222, 0, 0, 0, 0, 0, -0.619900793650794, 0, 0, 0.124603174603175, 0, 0, -0.130833333333333, 0, 0, 0, 0, 0, -1.43605158730159, 0, 0, 0.249206349206349, 0, 0, 0.19625, 0, 0, 0, 0, 0, -0.853531746031746, 0, 0, -0.186904761904762, 0, 0, 0.130833333333333, 0, 0, 0, 0, 0,
        0, -0.361349206349206, 0, 0, -0.093452380952381, 0, 0, -0.0872222222222222, 0, 0, 0, 0, 0, -0.619900793650794, 0, 0, 0.124603174603175, 0, 0, -0.130833333333333, 0, 0, 0, 0, 0, -1.43605158730159, 0, 0, 0.249206349206349, 0, 0, 0.19625, 0, 0, 0, 0, 0, -0.853531746031746, 0, 0, -0.186904761904762, 0, 0, 0.130833333333333, 0, 0, 0, 0,
        0, 0, -0.361349206349206, 0, 0, -0.093452380952381, 0, 0, -0.0872222222222222, 0, 0, 0, 0, 0, -0.619900793650794, 0, 0, 0.124603174603175, 0, 0, -0.130833333333333, 0, 0, 0, 0, 0, -1.43605158730159, 0, 0, 0.249206349206349, 0, 0, 0.19625, 0, 0, 0, 0, 0, -0.853531746031746, 0, 0, -0.186904761904762, 0, 0, 0.130833333333333, 0, 0, 0,
        -0.361349206349206, 0, 0, -0.0872222222222222, 0, 0, -0.093452380952381, 0, 0, 0, 0, 0, -0.853531746031746, 0, 0, 0.130833333333333, 0, 0, -0.186904761904762, 0, 0, 0, 0, 0, -1.43605158730159, 0, 0, 0.19625, 0, 0, 0.249206349206349, 0, 0, 0, 0, 0, -0.619900793650794, 0, 0, -0.130833333333333, 0, 0, 0.124603174603175, 0, 0, 0, 0, 0,
        0, -0.361349206349206, 0, 0, -0.0872222222222222, 0, 0, -0.093452380952381, 0, 0, 0, 0, 0, -0.853531746031746, 0, 0, 0.130833333333333, 0, 0, -0.186904761904762, 0, 0, 0, 0, 0, -1.43605158730159, 0, 0, 0.19625, 0, 0, 0.249206349206349, 0, 0, 0, 0, 0, -0.619900793650794, 0, 0, -0.130833333333333, 0, 0, 0.124603174603175, 0, 0, 0, 0,
        0, 0, -0.361349206349206, 0, 0, -0.0872222222222222, 0, 0, -0.093452380952381, 0, 0, 0, 0, 0, -0.853531746031746, 0, 0, 0.130833333333333, 0, 0, -0.186904761904762, 0, 0, 0, 0, 0, -1.43605158730159, 0, 0, 0.19625, 0, 0, 0.249206349206349, 0, 0, 0, 0, 0, -0.619900793650794, 0, 0, -0.130833333333333, 0, 0, 0.124603174603175, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1.81712962962963E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.81712962962963E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.81712962962963E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.63425925925926E-05,
        3.8190873015873, 0, 0, 0.619900793650794, 0, 0, 0.853531746031746, 0, 0, 0, 0, 0, 1.22734126984127, 0, 0, -0.361349206349206, 0, 0, 0.361349206349206, 0, 0, 0, 0, 0, 3.8190873015873, 0, 0, -0.853531746031746, 0, 0, -0.619900793650794, 0, 0, 0, 0, 0, 10.7594841269841, 0, 0, 1.43605158730159, 0, 0, -1.43605158730159, 0, 0, 0, 0, 0,
        0, 3.8190873015873, 0, 0, 0.619900793650794, 0, 0, 0.853531746031746, 0, 0, 0, 0, 0, 1.22734126984127, 0, 0, -0.361349206349206, 0, 0, 0.361349206349206, 0, 0, 0, 0, 0, 3.8190873015873, 0, 0, -0.853531746031746, 0, 0, -0.619900793650794, 0, 0, 0, 0, 0, 10.7594841269841, 0, 0, 1.43605158730159, 0, 0, -1.43605158730159, 0, 0, 0, 0,
        0, 0, 3.8190873015873, 0, 0, 0.619900793650794, 0, 0, 0.853531746031746, 0, 0, 0, 0, 0, 1.22734126984127, 0, 0, -0.361349206349206, 0, 0, 0.361349206349206, 0, 0, 0, 0, 0, 3.8190873015873, 0, 0, -0.853531746031746, 0, 0, -0.619900793650794, 0, 0, 0, 0, 0, 10.7594841269841, 0, 0, 1.43605158730159, 0, 0, -1.43605158730159, 0, 0, 0,
        0.619900793650794, 0, 0, 0.124603174603175, 0, 0, 0.130833333333333, 0, 0, 0, 0, 0, 0.361349206349206, 0, 0, -0.093452380952381, 0, 0, 0.0872222222222222, 0, 0, 0, 0, 0, 0.853531746031746, 0, 0, -0.186904761904762, 0, 0, -0.130833333333333, 0, 0, 0, 0, 0, 1.43605158730159, 0, 0, 0.249206349206349, 0, 0, -0.19625, 0, 0, 0, 0, 0,
        0, 0.619900793650794, 0, 0, 0.124603174603175, 0, 0, 0.130833333333333, 0, 0, 0, 0, 0, 0.361349206349206, 0, 0, -0.093452380952381, 0, 0, 0.0872222222222222, 0, 0, 0, 0, 0, 0.853531746031746, 0, 0, -0.186904761904762, 0, 0, -0.130833333333333, 0, 0, 0, 0, 0, 1.43605158730159, 0, 0, 0.249206349206349, 0, 0, -0.19625, 0, 0, 0, 0,
        0, 0, 0.619900793650794, 0, 0, 0.124603174603175, 0, 0, 0.130833333333333, 0, 0, 0, 0, 0, 0.361349206349206, 0, 0, -0.093452380952381, 0, 0, 0.0872222222222222, 0, 0, 0, 0, 0, 0.853531746031746, 0, 0, -0.186904761904762, 0, 0, -0.130833333333333, 0, 0, 0, 0, 0, 1.43605158730159, 0, 0, 0.249206349206349, 0, 0, -0.19625, 0, 0, 0,
        -0.853531746031746, 0, 0, -0.130833333333333, 0, 0, -0.186904761904762, 0, 0, 0, 0, 0, -0.361349206349206, 0, 0, 0.0872222222222222, 0, 0, -0.093452380952381, 0, 0, 0, 0, 0, -0.619900793650794, 0, 0, 0.130833333333333, 0, 0, 0.124603174603175, 0, 0, 0, 0, 0, -1.43605158730159, 0, 0, -0.19625, 0, 0, 0.249206349206349, 0, 0, 0, 0, 0,
        0, -0.853531746031746, 0, 0, -0.130833333333333, 0, 0, -0.186904761904762, 0, 0, 0, 0, 0, -0.361349206349206, 0, 0, 0.0872222222222222, 0, 0, -0.093452380952381, 0, 0, 0, 0, 0, -0.619900793650794, 0, 0, 0.130833333333333, 0, 0, 0.124603174603175, 0, 0, 0, 0, 0, -1.43605158730159, 0, 0, -0.19625, 0, 0, 0.249206349206349, 0, 0, 0, 0,
        0, 0, -0.853531746031746, 0, 0, -0.130833333333333, 0, 0, -0.186904761904762, 0, 0, 0, 0, 0, -0.361349206349206, 0, 0, 0.0872222222222222, 0, 0, -0.093452380952381, 0, 0, 0, 0, 0, -0.619900793650794, 0, 0, 0.130833333333333, 0, 0, 0.124603174603175, 0, 0, 0, 0, 0, -1.43605158730159, 0, 0, -0.19625, 0, 0, 0.249206349206349, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.81712962962963E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.81712962962963E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.81712962962963E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05;

    ChMatrixDynamic<double> MassMatrix;
    MassMatrix.resize(48, 48);
    m_element->ComputeKRMmatricesGlobal(MassMatrix, 0, 0, 1);
    ChMatrixNM<double, 16, 16> MassMatrix_compact;
    for (unsigned int i = 0; i < 16; i++) {
        for (unsigned int j = 0; j < 16; j++) {
            MassMatrix_compact(i, j) = MassMatrix(3 * i, 3 * j);
        }
    }

    double MaxAbsError = (MassMatrix - Expected_MassMatrix).cwiseAbs().maxCoeff();
    bool passed_test = (MaxAbsError <= 0.01);

    if (msglvl>=2) {
        std::cout << "Mass Matrix = " << std::endl;
        std::cout << MassMatrix << std::endl;

        std::cout << "Mass Matrix (Compact Form) = " << std::endl;
        std::cout << MassMatrix_compact << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Mass Matrix (Max Abs Error) = " << MaxAbsError;

        if (passed_test)
            std::cout << " - Test PASSED" << std::endl;
        else
            print_red(" - Test FAILED\n");
    }

    return (passed_test);
}

template <typename ElementVersion, typename MaterialVersion>
bool ANCFShellTest<ElementVersion, MaterialVersion>::GeneralizedGravityForceCheck(int msglvl) {
    // =============================================================================
    //  Generalized Force due to Gravity
    //  (Result should be nearly exact - No expected error)
    // =============================================================================
    ChVectorN<double, 48> Expected_InternalForceDueToGravity;
    Expected_InternalForceDueToGravity <<
        0,
        0,
        -192.45550625,
        0,
        0,
        -32.0759177083333,
        0,
        0,
        -32.0759177083333,
        0,
        0,
        0,
        0,
        0,
        -192.45550625,
        0,
        0,
        32.0759177083333,
        0,
        0,
        -32.0759177083333,
        0,
        0,
        0,
        0,
        0,
        -192.45550625,
        0,
        0,
        32.0759177083333,
        0,
        0,
        32.0759177083333,
        0,
        0,
        0,
        0,
        0,
        -192.45550625,
        0,
        0,
        -32.0759177083333,
        0,
        0,
        32.0759177083333,
        0,
        0,
        0;

    ChVectorDynamic<double> InternalForceNoDispNoVelWithGravity;
    InternalForceNoDispNoVelWithGravity.resize(48);
    m_element->SetGravityOn(true);
    m_element->ComputeInternalForces(InternalForceNoDispNoVelWithGravity);

    ChVectorDynamic<double> InternalForceNoDispNoVelNoGravity;
    InternalForceNoDispNoVelNoGravity.resize(48);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceNoDispNoVelNoGravity);

    ChVectorDynamic<double> InternalForceDueToGravity = InternalForceNoDispNoVelWithGravity - InternalForceNoDispNoVelNoGravity;

    double MaxAbsError = (InternalForceDueToGravity - Expected_InternalForceDueToGravity).cwiseAbs().maxCoeff();
    bool passed_test = (MaxAbsError <= 0.01);

    if (msglvl >= 2) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Generalized Force due to Gravity = " << std::endl;
        std::cout << InternalForceDueToGravity << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Generalized Force due to Gravity (Max Abs Error) = " << MaxAbsError;

        if (passed_test)
            std::cout << " - Test PASSED" << std::endl;
        else
            print_red(" - Test FAILED\n");
    }

    return (passed_test);
}

template <typename ElementVersion, typename MaterialVersion>
bool ANCFShellTest<ElementVersion, MaterialVersion>::GeneralizedInternalForceNoDispNoVelCheck(int msglvl) {
    // =============================================================================
    //  Check the Internal Force at Zero Displacement, Zero Velocity
    //  (should equal all 0's by definition)
    //  (Assumes that the element has not been changed from the initialized state)
    // =============================================================================

    ChVectorDynamic<double> InternalForceNoDispNoVelNoGravity;
    InternalForceNoDispNoVelNoGravity.resize(48);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceNoDispNoVelNoGravity);

    double MaxAbsError = InternalForceNoDispNoVelNoGravity.cwiseAbs().maxCoeff();
    bool passed_test = (MaxAbsError <= 0.01);

    if (msglvl >= 2) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Generalized Internal Force - No Displacement, No Velocity = " << std::endl;
        std::cout << InternalForceNoDispNoVelNoGravity << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Generalized Internal Force - No Displacement, No Velocity (Max Abs Error) = "
            << MaxAbsError;

        if (passed_test)
            std::cout << " - Test PASSED" << std::endl;
        else
            print_red(" - Test FAILED\n");
    }

    return (passed_test);
}

template <typename ElementVersion, typename MaterialVersion>
bool ANCFShellTest<ElementVersion, MaterialVersion>::GeneralizedInternalForceSmallDispNoVelCheck(int msglvl) {
    // =============================================================================
    //  Check the Internal Force at a Given Displacement 
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================

    ChVectorN<double, 48> Expected_InternalForceSmallDispNoVelNoGravity;
    Expected_InternalForceSmallDispNoVelNoGravity <<
        675.769230769231,
        110.384615384615,
        223077.600129102,
        77.5,
        27.7884615384615,
        1923.15487201901,
        92.7884615384615,
        -45.9615384615385,
        28846.2429722521,
        -141346.153846154,
        60576.9230769231,
        -309.615384615385,
        -918.076923076923,
        918.076923076923,
        -707694.147454277,
        47.2115384615385,
        -87.0192307692308,
        69230.9281733011,
        -87.0192307692308,
        47.2115384615385,
        -69230.9281733011,
        -141346.153846154,
        141346.153846154,
        -474.519230769231,
        -110.384615384615,
        -675.769230769231,
        223077.600129102,
        -45.9615384615385,
        92.7884615384615,
        -28846.2429722521,
        27.7884615384615,
        77.5,
        -1923.15487201901,
        -60576.9230769231,
        141346.153846154,
        -309.615384615385,
        352.692307692308,
        -352.692307692308,
        261538.947196073,
        65.0961538461538,
        -73.9423076923077,
        38461.6353290299,
        -73.9423076923077,
        65.0961538461538,
        -38461.6353290299,
        -60576.9230769231,
        60576.9230769231,
        -144.711538461538;

    //Setup the test conditions
    ChVector<double> OriginalPos = m_nodeB->GetPos();
    m_nodeB->SetPos(ChVector<>(m_nodeB->GetPos().x(), m_nodeB->GetPos().y(), 0.001));

    ChVectorDynamic<double> InternalForceSmallDispNoVelNoGravity;
    InternalForceSmallDispNoVelNoGravity.resize(48);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceSmallDispNoVelNoGravity);

    //Reset the element conditions back to its original values
    m_nodeB->SetPos(OriginalPos);

    double MaxAbsError = (InternalForceSmallDispNoVelNoGravity - Expected_InternalForceSmallDispNoVelNoGravity).cwiseAbs().maxCoeff();
    bool passed_test = (MaxAbsError <= 0.01*Expected_InternalForceSmallDispNoVelNoGravity.cwiseAbs().maxCoeff());

    if (msglvl >= 2) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Generalized Internal Force - Small Displacement, No Velocity = " << std::endl;
        std::cout << InternalForceSmallDispNoVelNoGravity << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Generalized Internal Force - Small Displacement, No Velocity (Max Abs Error) = "
            << MaxAbsError;

        if (passed_test)
            std::cout << " - Test PASSED" << std::endl;
        else
            print_red(" - Test FAILED\n");
    }

    return (passed_test);
}

template <typename ElementVersion, typename MaterialVersion>
bool ANCFShellTest<ElementVersion, MaterialVersion>::GeneralizedInternalForceNoDispSmallVelCheck(int msglvl) {
    // =============================================================================
    //  Check the Internal Force at a No Displacement with a Given Nodal Velocity
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================

    ChVectorN<double, 48> Expected_InternalForceNoDispSmallVelNoGravity;
    Expected_InternalForceNoDispSmallVelNoGravity <<
        0,
        0,
        2230.76923076923,
        0,
        0,
        19.2307692307692,
        0,
        0,
        288.461538461538,
        -1413.46153846154,
        605.769230769231,
        0,
        0,
        0,
        -7076.92307692308,
        0,
        0,
        692.307692307692,
        0,
        0,
        -692.307692307692,
        -1413.46153846154,
        1413.46153846154,
        0,
        0,
        0,
        2230.76923076923,
        0,
        0,
        -288.461538461538,
        0,
        0,
        -19.2307692307692,
        -605.769230769231,
        1413.46153846154,
        0,
        0,
        0,
        2615.38461538462,
        0,
        0,
        384.615384615385,
        0,
        0,
        -384.615384615385,
        -605.769230769231,
        605.769230769231,
        0;

    //Setup the test conditions
    ChVector<double> OriginalVel = m_nodeB->GetPos_dt();
    m_nodeB->SetPos_dt(ChVector<>(0.0, 0.0, 0.001));
    m_element->SetAlphaDamp(0.01);

    ChVectorDynamic<double> InternalForceNoDispSmallVelNoGravity;
    InternalForceNoDispSmallVelNoGravity.resize(48);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceNoDispSmallVelNoGravity);

    //Reset the element conditions back to its original values
    m_nodeB->SetPos_dt(OriginalVel);
    m_element->SetAlphaDamp(0.0);

    double MaxAbsError = (InternalForceNoDispSmallVelNoGravity - Expected_InternalForceNoDispSmallVelNoGravity).cwiseAbs().maxCoeff();
    bool passed_test = (MaxAbsError <= 0.01*Expected_InternalForceNoDispSmallVelNoGravity.cwiseAbs().maxCoeff());

    if (msglvl >= 2) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Generalized Internal Force - No Displacement, Small Velocity With Damping = " << std::endl;
        std::cout << InternalForceNoDispSmallVelNoGravity << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Generalized Internal Force - No Displacement, Small Velocity With Damping (Max Abs Error) = "
            << MaxAbsError;

        if (passed_test)
            std::cout << " - Test PASSED" << std::endl;
        else
            print_red(" - Test FAILED\n");
    }

    return (passed_test);

}

template <typename ElementVersion, typename MaterialVersion>
bool ANCFShellTest<ElementVersion, MaterialVersion>::JacobianNoDispNoVelNoDampingCheck(int msglvl) {
    // =============================================================================
    //  Check the Jacobian at No Displacement/Velocity - No Damping
    //  (some small error expected depending on the formulation/steps used)
    //  (The R contribution should be all zeros since Damping is not enabled)
    // =============================================================================
    ChMatrixNM<double, 48, 48> Expected_JacobianK_NoDispNoVelNoDamping;
    Expected_JacobianK_NoDispNoVelNoDamping <<
        1592307692.30769, 504807692.307692, 0, 136538461.538462, 20192307.6923077, 0, 175000000, -20192307.6923077, 0, 0, 0, -212019230.769231, -1107692307.69231, 100961538.461538, 0, 69230769.2307692, -20192307.6923077, 0, -134615384.615385, -100961538.461538, 0, 0, 0, -212019230.769231, -588461538.461538, -504807692.307692, 0, 72115384.6153846, 100961538.461538, 0, 100961538.461538, 100961538.461538, 0, 0, 0, -90865384.6153846, 103846153.846154, -100961538.461538, 0, 4807692.3076923, -100961538.461538, 0, -60576923.0769231, 20192307.6923077, 0, 0, 0, -90865384.6153846,
        504807692.307692, 1592307692.30769, 0, -20192307.6923077, 175000000, 0, 20192307.6923077, 136538461.538462, 0, 0, 0, -212019230.769231, -100961538.461538, 103846153.846154, 0, 20192307.6923077, -60576923.0769231, 0, -100961538.461538, 4807692.3076923, 0, 0, 0, -90865384.6153846, -504807692.307692, -588461538.461538, 0, 100961538.461538, 100961538.461538, 0, 100961538.461538, 72115384.6153846, 0, 0, 0, -90865384.6153846, 100961538.461538, -1107692307.69231, 0, -100961538.461538, -134615384.615385, 0, -20192307.6923077, 69230769.2307692, 0, 0, 0, -212019230.769231,
        0, 0, 707692307.692308, 0, 0, 69230769.2307692, 0, 0, 69230769.2307692, -141346153.846154, -141346153.846154, 0, 0, 0, -223076923.076923, 0, 0, 1923076.92307692, 0, 0, -28846153.8461538, -141346153.846154, -60576923.0769231, 0, 0, 0, -261538461.538462, 0, 0, 38461538.4615385, 0, 0, 38461538.4615385, -60576923.0769231, -60576923.0769231, 0, 0, 0, -223076923.076923, 0, 0, -28846153.8461538, 0, 0, 1923076.92307692, -60576923.0769231, -141346153.846154, 0,
        136538461.538462, -20192307.6923077, 0, 133333333.333333, 0, 0, 0, 14022435.8974359, 0, 0, 0, 33653846.1538462, -69230769.2307692, 20192307.6923077, 0, -37179487.1794872, -3365384.61538461, 0, 0, -14022435.8974359, 0, 0, 0, -33653846.1538462, -72115384.6153846, -100961538.461538, 0, -9935897.43589743, 16826923.0769231, 0, 0, 14022435.8974359, 0, 0, 0, -16826923.0769231, 4807692.3076923, 100961538.461538, 0, 55128205.1282051, 0, 0, 0, -14022435.8974359, 0, 0, 0, 16826923.0769231,
        20192307.6923077, 175000000, 0, 0, 62820512.8205128, 0, 14022435.8974359, 0, 0, 0, 0, -30288461.5384615, -20192307.6923077, 60576923.0769231, 0, 3365384.61538461, -29166666.6666667, 0, -14022435.8974359, 0, 0, 0, 0, -20192307.6923077, -100961538.461538, -100961538.461538, 0, 16826923.0769231, 15705128.2051282, 0, 14022435.8974359, 0, 0, 0, 0, -20192307.6923077, 100961538.461538, -134615384.615385, 0, 0, -8974358.97435897, 0, -14022435.8974359, 0, 0, 0, 0, -30288461.5384615,
        0, 0, 69230769.2307692, 0, 0, 43589743.5897436, 0, 0, 0, 22435897.4358974, -20192307.6923077, 0, 0, 0, -1923076.92307692, 0, 0, -14743589.7435897, 0, 0, 0, -22435897.4358974, -13461538.4615385, 0, 0, 0, -38461538.4615385, 0, 0, 1282051.28205128, 0, 0, 0, -11217948.7179487, -13461538.4615385, 0, 0, 0, -28846153.8461538, 0, 0, 10256410.2564103, 0, 0, 0, 11217948.7179487, -20192307.6923077, 0,
        175000000, 20192307.6923077, 0, 0, 14022435.8974359, 0, 62820512.8205128, 0, 0, 0, 0, -30288461.5384615, -134615384.615385, 100961538.461538, 0, 0, -14022435.8974359, 0, -8974358.97435897, 0, 0, 0, 0, -30288461.5384615, -100961538.461538, -100961538.461538, 0, 0, 14022435.8974359, 0, 15705128.2051282, 16826923.0769231, 0, 0, 0, -20192307.6923077, 60576923.0769231, -20192307.6923077, 0, 0, -14022435.8974359, 0, -29166666.6666667, 3365384.61538461, 0, 0, 0, -20192307.6923077,
        -20192307.6923077, 136538461.538462, 0, 14022435.8974359, 0, 0, 0, 133333333.333333, 0, 0, 0, 33653846.1538462, 100961538.461538, 4807692.3076923, 0, -14022435.8974359, 0, 0, 0, 55128205.1282051, 0, 0, 0, 16826923.0769231, -100961538.461538, -72115384.6153846, 0, 14022435.8974359, 0, 0, 16826923.0769231, -9935897.43589743, 0, 0, 0, -16826923.0769231, 20192307.6923077, -69230769.2307692, 0, -14022435.8974359, 0, 0, -3365384.61538461, -37179487.1794872, 0, 0, 0, -33653846.1538462,
        0, 0, 69230769.2307692, 0, 0, 0, 0, 0, 43589743.5897436, -20192307.6923077, 22435897.4358974, 0, 0, 0, -28846153.8461538, 0, 0, 0, 0, 0, 10256410.2564103, -20192307.6923077, 11217948.7179487, 0, 0, 0, -38461538.4615385, 0, 0, 0, 0, 0, 1282051.28205128, -13461538.4615385, -11217948.7179487, 0, 0, 0, -1923076.92307692, 0, 0, 0, 0, 0, -14743589.7435897, -13461538.4615385, -22435897.4358974, 0,
        0, 0, -141346153.846154, 0, 0, 22435897.4358974, 0, 0, -20192307.6923077, 89753685.8974359, 4206.73076923077, 0, 0, 0, 141346153.846154, 0, 0, -22435897.4358974, 0, 0, 20192307.6923077, 44865064.1025641, 841.346153846154, 0, 0, 0, 60576923.0769231, 0, 0, -11217948.7179487, 0, 0, -13461538.4615385, 22430849.3589744, -4206.73076923077, 0, 0, 0, -60576923.0769231, 0, 0, 11217948.7179487, 0, 0, 13461538.4615385, 44873477.5641026, -841.346153846154, 0,
        0, 0, -141346153.846154, 0, 0, -20192307.6923077, 0, 0, 22435897.4358974, 4206.73076923077, 89753685.8974359, 0, 0, 0, -60576923.0769231, 0, 0, 13461538.4615385, 0, 0, 11217948.7179487, -841.346153846154, 44873477.5641026, 0, 0, 0, 60576923.0769231, 0, 0, -13461538.4615385, 0, 0, -11217948.7179487, -4206.73076923077, 22430849.3589744, 0, 0, 0, 141346153.846154, 0, 0, 20192307.6923077, 0, 0, -22435897.4358974, 841.346153846154, 44865064.1025641, 0,
        -212019230.769231, -212019230.769231, 0, 33653846.1538462, -30288461.5384615, 0, -30288461.5384615, 33653846.1538462, 0, 0, 0, 314107051.282051, 212019230.769231, -90865384.6153846, 0, -33653846.1538462, 20192307.6923077, 0, 30288461.5384615, 16826923.0769231, 0, 0, 0, 157050160.25641, 90865384.6153846, 90865384.6153846, 0, -16826923.0769231, -20192307.6923077, 0, -20192307.6923077, -16826923.0769231, 0, 0, 0, 78523397.4358974, -90865384.6153846, 212019230.769231, 0, 16826923.0769231, 30288461.5384615, 0, 20192307.6923077, -33653846.1538462, 0, 0, 0, 157050160.25641,
        -1107692307.69231, -100961538.461538, 0, -69230769.2307692, -20192307.6923077, 0, -134615384.615385, 100961538.461538, 0, 0, 0, 212019230.769231, 1592307692.30769, -504807692.307692, 0, -136538461.538462, 20192307.6923077, 0, 175000000, 20192307.6923077, 0, 0, 0, 212019230.769231, 103846153.846154, 100961538.461538, 0, -4807692.3076923, -100961538.461538, 0, -60576923.0769231, -20192307.6923077, 0, 0, 0, 90865384.6153846, -588461538.461538, 504807692.307692, 0, -72115384.6153846, 100961538.461538, 0, 100961538.461538, -100961538.461538, 0, 0, 0, 90865384.6153846,
        100961538.461538, 103846153.846154, 0, 20192307.6923077, 60576923.0769231, 0, 100961538.461538, 4807692.3076923, 0, 0, 0, -90865384.6153846, -504807692.307692, 1592307692.30769, 0, -20192307.6923077, -175000000, 0, -20192307.6923077, 136538461.538462, 0, 0, 0, -212019230.769231, -100961538.461538, -1107692307.69231, 0, -100961538.461538, 134615384.615385, 0, 20192307.6923077, 69230769.2307692, 0, 0, 0, -212019230.769231, 504807692.307692, -588461538.461538, 0, 100961538.461538, -100961538.461538, 0, -100961538.461538, 72115384.6153846, 0, 0, 0, -90865384.6153846,
        0, 0, -223076923.076923, 0, 0, -1923076.92307692, 0, 0, -28846153.8461538, 141346153.846154, -60576923.0769231, 0, 0, 0, 707692307.692308, 0, 0, -69230769.2307692, 0, 0, 69230769.2307692, 141346153.846154, -141346153.846154, 0, 0, 0, -223076923.076923, 0, 0, 28846153.8461538, 0, 0, 1923076.92307692, 60576923.0769231, -141346153.846154, 0, 0, 0, -261538461.538462, 0, 0, -38461538.4615385, 0, 0, 38461538.4615385, 60576923.0769231, -60576923.0769231, 0,
        69230769.2307692, 20192307.6923077, 0, -37179487.1794872, 3365384.61538461, 0, 0, -14022435.8974359, 0, 0, 0, -33653846.1538462, -136538461.538462, -20192307.6923077, 0, 133333333.333333, 0, 0, 0, 14022435.8974359, 0, 0, 0, 33653846.1538462, -4807692.3076923, 100961538.461538, 0, 55128205.1282051, 0, 0, 0, -14022435.8974359, 0, 0, 0, 16826923.0769231, 72115384.6153846, -100961538.461538, 0, -9935897.43589743, -16826923.0769231, 0, 0, 14022435.8974359, 0, 0, 0, -16826923.0769231,
        -20192307.6923077, -60576923.0769231, 0, -3365384.61538461, -29166666.6666667, 0, -14022435.8974359, 0, 0, 0, 0, 20192307.6923077, 20192307.6923077, -175000000, 0, 0, 62820512.8205128, 0, 14022435.8974359, 0, 0, 0, 0, 30288461.5384615, 100961538.461538, 134615384.615385, 0, 0, -8974358.97435897, 0, -14022435.8974359, 0, 0, 0, 0, 30288461.5384615, -100961538.461538, 100961538.461538, 0, -16826923.0769231, 15705128.2051282, 0, 14022435.8974359, 0, 0, 0, 0, 20192307.6923077,
        0, 0, 1923076.92307692, 0, 0, -14743589.7435897, 0, 0, 0, -22435897.4358974, 13461538.4615385, 0, 0, 0, -69230769.2307692, 0, 0, 43589743.5897436, 0, 0, 0, 22435897.4358974, 20192307.6923077, 0, 0, 0, 28846153.8461538, 0, 0, 10256410.2564103, 0, 0, 0, 11217948.7179487, 20192307.6923077, 0, 0, 0, 38461538.4615385, 0, 0, 1282051.28205128, 0, 0, 0, -11217948.7179487, 13461538.4615385, 0,
        -134615384.615385, -100961538.461538, 0, 0, -14022435.8974359, 0, -8974358.97435897, 0, 0, 0, 0, 30288461.5384615, 175000000, -20192307.6923077, 0, 0, 14022435.8974359, 0, 62820512.8205128, 0, 0, 0, 0, 30288461.5384615, 60576923.0769231, 20192307.6923077, 0, 0, -14022435.8974359, 0, -29166666.6666667, -3365384.61538461, 0, 0, 0, 20192307.6923077, -100961538.461538, 100961538.461538, 0, 0, 14022435.8974359, 0, 15705128.2051282, -16826923.0769231, 0, 0, 0, 20192307.6923077,
        -100961538.461538, 4807692.3076923, 0, -14022435.8974359, 0, 0, 0, 55128205.1282051, 0, 0, 0, 16826923.0769231, 20192307.6923077, 136538461.538462, 0, 14022435.8974359, 0, 0, 0, 133333333.333333, 0, 0, 0, 33653846.1538462, -20192307.6923077, -69230769.2307692, 0, -14022435.8974359, 0, 0, 3365384.61538461, -37179487.1794872, 0, 0, 0, -33653846.1538462, 100961538.461538, -72115384.6153846, 0, 14022435.8974359, 0, 0, -16826923.0769231, -9935897.43589743, 0, 0, 0, -16826923.0769231,
        0, 0, -28846153.8461538, 0, 0, 0, 0, 0, 10256410.2564103, 20192307.6923077, 11217948.7179487, 0, 0, 0, 69230769.2307692, 0, 0, 0, 0, 0, 43589743.5897436, 20192307.6923077, 22435897.4358974, 0, 0, 0, -1923076.92307692, 0, 0, 0, 0, 0, -14743589.7435897, 13461538.4615385, -22435897.4358974, 0, 0, 0, -38461538.4615385, 0, 0, 0, 0, 0, 1282051.28205128, 13461538.4615385, -11217948.7179487, 0,
        0, 0, -141346153.846154, 0, 0, -22435897.4358974, 0, 0, -20192307.6923077, 44865064.1025641, -841.346153846154, 0, 0, 0, 141346153.846154, 0, 0, 22435897.4358974, 0, 0, 20192307.6923077, 89753685.8974359, -4206.73076923077, 0, 0, 0, 60576923.0769231, 0, 0, 11217948.7179487, 0, 0, -13461538.4615385, 44873477.5641026, 841.346153846154, 0, 0, 0, -60576923.0769231, 0, 0, -11217948.7179487, 0, 0, 13461538.4615385, 22430849.3589744, 4206.73076923077, 0,
        0, 0, -60576923.0769231, 0, 0, -13461538.4615385, 0, 0, 11217948.7179487, 841.346153846154, 44873477.5641026, 0, 0, 0, -141346153.846154, 0, 0, 20192307.6923077, 0, 0, 22435897.4358974, -4206.73076923077, 89753685.8974359, 0, 0, 0, 141346153.846154, 0, 0, -20192307.6923077, 0, 0, -22435897.4358974, -841.346153846154, 44865064.1025641, 0, 0, 0, 60576923.0769231, 0, 0, 13461538.4615385, 0, 0, -11217948.7179487, 4206.73076923077, 22430849.3589744, 0,
        -212019230.769231, -90865384.6153846, 0, -33653846.1538462, -20192307.6923077, 0, -30288461.5384615, 16826923.0769231, 0, 0, 0, 157050160.25641, 212019230.769231, -212019230.769231, 0, 33653846.1538462, 30288461.5384615, 0, 30288461.5384615, 33653846.1538462, 0, 0, 0, 314107051.282051, 90865384.6153846, 212019230.769231, 0, 16826923.0769231, -30288461.5384615, 0, -20192307.6923077, -33653846.1538462, 0, 0, 0, 157050160.25641, -90865384.6153846, 90865384.6153846, 0, -16826923.0769231, 20192307.6923077, 0, 20192307.6923077, -16826923.0769231, 0, 0, 0, 78523397.4358974,
        -588461538.461538, -504807692.307692, 0, -72115384.6153846, -100961538.461538, 0, -100961538.461538, -100961538.461538, 0, 0, 0, 90865384.6153846, 103846153.846154, -100961538.461538, 0, -4807692.3076923, 100961538.461538, 0, 60576923.0769231, -20192307.6923077, 0, 0, 0, 90865384.6153846, 1592307692.30769, 504807692.307692, 0, -136538461.538462, -20192307.6923077, 0, -175000000, 20192307.6923077, 0, 0, 0, 212019230.769231, -1107692307.69231, 100961538.461538, 0, -69230769.2307692, 20192307.6923077, 0, 134615384.615385, 100961538.461538, 0, 0, 0, 212019230.769231,
        -504807692.307692, -588461538.461538, 0, -100961538.461538, -100961538.461538, 0, -100961538.461538, -72115384.6153846, 0, 0, 0, 90865384.6153846, 100961538.461538, -1107692307.69231, 0, 100961538.461538, 134615384.615385, 0, 20192307.6923077, -69230769.2307692, 0, 0, 0, 212019230.769231, 504807692.307692, 1592307692.30769, 0, 20192307.6923077, -175000000, 0, -20192307.6923077, -136538461.538462, 0, 0, 0, 212019230.769231, -100961538.461538, 103846153.846154, 0, -20192307.6923077, 60576923.0769231, 0, 100961538.461538, -4807692.3076923, 0, 0, 0, 90865384.6153846,
        0, 0, -261538461.538462, 0, 0, -38461538.4615385, 0, 0, -38461538.4615385, 60576923.0769231, 60576923.0769231, 0, 0, 0, -223076923.076923, 0, 0, 28846153.8461538, 0, 0, -1923076.92307692, 60576923.0769231, 141346153.846154, 0, 0, 0, 707692307.692308, 0, 0, -69230769.2307692, 0, 0, -69230769.2307692, 141346153.846154, 141346153.846154, 0, 0, 0, -223076923.076923, 0, 0, -1923076.92307692, 0, 0, 28846153.8461538, 141346153.846154, 60576923.0769231, 0,
        72115384.6153846, 100961538.461538, 0, -9935897.43589743, 16826923.0769231, 0, 0, 14022435.8974359, 0, 0, 0, -16826923.0769231, -4807692.3076923, -100961538.461538, 0, 55128205.1282051, 0, 0, 0, -14022435.8974359, 0, 0, 0, 16826923.0769231, -136538461.538462, 20192307.6923077, 0, 133333333.333333, 0, 0, 0, 14022435.8974359, 0, 0, 0, 33653846.1538462, 69230769.2307692, -20192307.6923077, 0, -37179487.1794872, -3365384.61538461, 0, 0, -14022435.8974359, 0, 0, 0, -33653846.1538462,
        100961538.461538, 100961538.461538, 0, 16826923.0769231, 15705128.2051282, 0, 14022435.8974359, 0, 0, 0, 0, -20192307.6923077, -100961538.461538, 134615384.615385, 0, 0, -8974358.97435897, 0, -14022435.8974359, 0, 0, 0, 0, -30288461.5384615, -20192307.6923077, -175000000, 0, 0, 62820512.8205128, 0, 14022435.8974359, 0, 0, 0, 0, -30288461.5384615, 20192307.6923077, -60576923.0769231, 0, 3365384.61538461, -29166666.6666667, 0, -14022435.8974359, 0, 0, 0, 0, -20192307.6923077,
        0, 0, 38461538.4615385, 0, 0, 1282051.28205128, 0, 0, 0, -11217948.7179487, -13461538.4615385, 0, 0, 0, 28846153.8461538, 0, 0, 10256410.2564103, 0, 0, 0, 11217948.7179487, -20192307.6923077, 0, 0, 0, -69230769.2307692, 0, 0, 43589743.5897436, 0, 0, 0, 22435897.4358974, -20192307.6923077, 0, 0, 0, 1923076.92307692, 0, 0, -14743589.7435897, 0, 0, 0, -22435897.4358974, -13461538.4615385, 0,
        100961538.461538, 100961538.461538, 0, 0, 14022435.8974359, 0, 15705128.2051282, 16826923.0769231, 0, 0, 0, -20192307.6923077, -60576923.0769231, 20192307.6923077, 0, 0, -14022435.8974359, 0, -29166666.6666667, 3365384.61538461, 0, 0, 0, -20192307.6923077, -175000000, -20192307.6923077, 0, 0, 14022435.8974359, 0, 62820512.8205128, 0, 0, 0, 0, -30288461.5384615, 134615384.615385, -100961538.461538, 0, 0, -14022435.8974359, 0, -8974358.97435897, 0, 0, 0, 0, -30288461.5384615,
        100961538.461538, 72115384.6153846, 0, 14022435.8974359, 0, 0, 16826923.0769231, -9935897.43589743, 0, 0, 0, -16826923.0769231, -20192307.6923077, 69230769.2307692, 0, -14022435.8974359, 0, 0, -3365384.61538461, -37179487.1794872, 0, 0, 0, -33653846.1538462, 20192307.6923077, -136538461.538462, 0, 14022435.8974359, 0, 0, 0, 133333333.333333, 0, 0, 0, 33653846.1538462, -100961538.461538, -4807692.3076923, 0, -14022435.8974359, 0, 0, 0, 55128205.1282051, 0, 0, 0, 16826923.0769231,
        0, 0, 38461538.4615385, 0, 0, 0, 0, 0, 1282051.28205128, -13461538.4615385, -11217948.7179487, 0, 0, 0, 1923076.92307692, 0, 0, 0, 0, 0, -14743589.7435897, -13461538.4615385, -22435897.4358974, 0, 0, 0, -69230769.2307692, 0, 0, 0, 0, 0, 43589743.5897436, -20192307.6923077, 22435897.4358974, 0, 0, 0, 28846153.8461538, 0, 0, 0, 0, 0, 10256410.2564103, -20192307.6923077, 11217948.7179487, 0,
        0, 0, -60576923.0769231, 0, 0, -11217948.7179487, 0, 0, -13461538.4615385, 22430849.3589744, -4206.73076923077, 0, 0, 0, 60576923.0769231, 0, 0, 11217948.7179487, 0, 0, 13461538.4615385, 44873477.5641026, -841.346153846154, 0, 0, 0, 141346153.846154, 0, 0, 22435897.4358974, 0, 0, -20192307.6923077, 89753685.8974359, 4206.73076923077, 0, 0, 0, -141346153.846154, 0, 0, -22435897.4358974, 0, 0, 20192307.6923077, 44865064.1025641, 841.346153846154, 0,
        0, 0, -60576923.0769231, 0, 0, -13461538.4615385, 0, 0, -11217948.7179487, -4206.73076923077, 22430849.3589744, 0, 0, 0, -141346153.846154, 0, 0, 20192307.6923077, 0, 0, -22435897.4358974, 841.346153846154, 44865064.1025641, 0, 0, 0, 141346153.846154, 0, 0, -20192307.6923077, 0, 0, 22435897.4358974, 4206.73076923077, 89753685.8974359, 0, 0, 0, 60576923.0769231, 0, 0, 13461538.4615385, 0, 0, 11217948.7179487, -841.346153846154, 44873477.5641026, 0,
        -90865384.6153846, -90865384.6153846, 0, -16826923.0769231, -20192307.6923077, 0, -20192307.6923077, -16826923.0769231, 0, 0, 0, 78523397.4358974, 90865384.6153846, -212019230.769231, 0, 16826923.0769231, 30288461.5384615, 0, 20192307.6923077, -33653846.1538462, 0, 0, 0, 157050160.25641, 212019230.769231, 212019230.769231, 0, 33653846.1538462, -30288461.5384615, 0, -30288461.5384615, 33653846.1538462, 0, 0, 0, 314107051.282051, -212019230.769231, 90865384.6153846, 0, -33653846.1538462, 20192307.6923077, 0, 30288461.5384615, 16826923.0769231, 0, 0, 0, 157050160.25641,
        103846153.846154, 100961538.461538, 0, 4807692.3076923, 100961538.461538, 0, 60576923.0769231, 20192307.6923077, 0, 0, 0, -90865384.6153846, -588461538.461538, 504807692.307692, 0, 72115384.6153846, -100961538.461538, 0, -100961538.461538, 100961538.461538, 0, 0, 0, -90865384.6153846, -1107692307.69231, -100961538.461538, 0, 69230769.2307692, 20192307.6923077, 0, 134615384.615385, -100961538.461538, 0, 0, 0, -212019230.769231, 1592307692.30769, -504807692.307692, 0, 136538461.538462, -20192307.6923077, 0, -175000000, -20192307.6923077, 0, 0, 0, -212019230.769231,
        -100961538.461538, -1107692307.69231, 0, 100961538.461538, -134615384.615385, 0, -20192307.6923077, -69230769.2307692, 0, 0, 0, 212019230.769231, 504807692.307692, -588461538.461538, 0, -100961538.461538, 100961538.461538, 0, 100961538.461538, -72115384.6153846, 0, 0, 0, 90865384.6153846, 100961538.461538, 103846153.846154, 0, -20192307.6923077, -60576923.0769231, 0, -100961538.461538, -4807692.3076923, 0, 0, 0, 90865384.6153846, -504807692.307692, 1592307692.30769, 0, 20192307.6923077, 175000000, 0, 20192307.6923077, -136538461.538462, 0, 0, 0, 212019230.769231,
        0, 0, -223076923.076923, 0, 0, -28846153.8461538, 0, 0, -1923076.92307692, -60576923.0769231, 141346153.846154, 0, 0, 0, -261538461.538462, 0, 0, 38461538.4615385, 0, 0, -38461538.4615385, -60576923.0769231, 60576923.0769231, 0, 0, 0, -223076923.076923, 0, 0, 1923076.92307692, 0, 0, 28846153.8461538, -141346153.846154, 60576923.0769231, 0, 0, 0, 707692307.692308, 0, 0, 69230769.2307692, 0, 0, -69230769.2307692, -141346153.846154, 141346153.846154, 0,
        4807692.3076923, -100961538.461538, 0, 55128205.1282051, 0, 0, 0, -14022435.8974359, 0, 0, 0, 16826923.0769231, -72115384.6153846, 100961538.461538, 0, -9935897.43589743, -16826923.0769231, 0, 0, 14022435.8974359, 0, 0, 0, -16826923.0769231, -69230769.2307692, -20192307.6923077, 0, -37179487.1794872, 3365384.61538461, 0, 0, -14022435.8974359, 0, 0, 0, -33653846.1538462, 136538461.538462, 20192307.6923077, 0, 133333333.333333, 0, 0, 0, 14022435.8974359, 0, 0, 0, 33653846.1538462,
        -100961538.461538, -134615384.615385, 0, 0, -8974358.97435897, 0, -14022435.8974359, 0, 0, 0, 0, 30288461.5384615, 100961538.461538, -100961538.461538, 0, -16826923.0769231, 15705128.2051282, 0, 14022435.8974359, 0, 0, 0, 0, 20192307.6923077, 20192307.6923077, 60576923.0769231, 0, -3365384.61538461, -29166666.6666667, 0, -14022435.8974359, 0, 0, 0, 0, 20192307.6923077, -20192307.6923077, 175000000, 0, 0, 62820512.8205128, 0, 14022435.8974359, 0, 0, 0, 0, 30288461.5384615,
        0, 0, -28846153.8461538, 0, 0, 10256410.2564103, 0, 0, 0, 11217948.7179487, 20192307.6923077, 0, 0, 0, -38461538.4615385, 0, 0, 1282051.28205128, 0, 0, 0, -11217948.7179487, 13461538.4615385, 0, 0, 0, -1923076.92307692, 0, 0, -14743589.7435897, 0, 0, 0, -22435897.4358974, 13461538.4615385, 0, 0, 0, 69230769.2307692, 0, 0, 43589743.5897436, 0, 0, 0, 22435897.4358974, 20192307.6923077, 0,
        -60576923.0769231, -20192307.6923077, 0, 0, -14022435.8974359, 0, -29166666.6666667, -3365384.61538461, 0, 0, 0, 20192307.6923077, 100961538.461538, -100961538.461538, 0, 0, 14022435.8974359, 0, 15705128.2051282, -16826923.0769231, 0, 0, 0, 20192307.6923077, 134615384.615385, 100961538.461538, 0, 0, -14022435.8974359, 0, -8974358.97435897, 0, 0, 0, 0, 30288461.5384615, -175000000, 20192307.6923077, 0, 0, 14022435.8974359, 0, 62820512.8205128, 0, 0, 0, 0, 30288461.5384615,
        20192307.6923077, 69230769.2307692, 0, -14022435.8974359, 0, 0, 3365384.61538461, -37179487.1794872, 0, 0, 0, -33653846.1538462, -100961538.461538, 72115384.6153846, 0, 14022435.8974359, 0, 0, -16826923.0769231, -9935897.43589743, 0, 0, 0, -16826923.0769231, 100961538.461538, -4807692.3076923, 0, -14022435.8974359, 0, 0, 0, 55128205.1282051, 0, 0, 0, 16826923.0769231, -20192307.6923077, -136538461.538462, 0, 14022435.8974359, 0, 0, 0, 133333333.333333, 0, 0, 0, 33653846.1538462,
        0, 0, 1923076.92307692, 0, 0, 0, 0, 0, -14743589.7435897, 13461538.4615385, -22435897.4358974, 0, 0, 0, 38461538.4615385, 0, 0, 0, 0, 0, 1282051.28205128, 13461538.4615385, -11217948.7179487, 0, 0, 0, 28846153.8461538, 0, 0, 0, 0, 0, 10256410.2564103, 20192307.6923077, 11217948.7179487, 0, 0, 0, -69230769.2307692, 0, 0, 0, 0, 0, 43589743.5897436, 20192307.6923077, 22435897.4358974, 0,
        0, 0, -60576923.0769231, 0, 0, 11217948.7179487, 0, 0, -13461538.4615385, 44873477.5641026, 841.346153846154, 0, 0, 0, 60576923.0769231, 0, 0, -11217948.7179487, 0, 0, 13461538.4615385, 22430849.3589744, 4206.73076923077, 0, 0, 0, 141346153.846154, 0, 0, -22435897.4358974, 0, 0, -20192307.6923077, 44865064.1025641, -841.346153846154, 0, 0, 0, -141346153.846154, 0, 0, 22435897.4358974, 0, 0, 20192307.6923077, 89753685.8974359, -4206.73076923077, 0,
        0, 0, -141346153.846154, 0, 0, -20192307.6923077, 0, 0, -22435897.4358974, -841.346153846154, 44865064.1025641, 0, 0, 0, -60576923.0769231, 0, 0, 13461538.4615385, 0, 0, -11217948.7179487, 4206.73076923077, 22430849.3589744, 0, 0, 0, 60576923.0769231, 0, 0, -13461538.4615385, 0, 0, 11217948.7179487, 841.346153846154, 44873477.5641026, 0, 0, 0, 141346153.846154, 0, 0, 20192307.6923077, 0, 0, 22435897.4358974, -4206.73076923077, 89753685.8974359, 0,
        -90865384.6153846, -212019230.769231, 0, 16826923.0769231, -30288461.5384615, 0, -20192307.6923077, -33653846.1538462, 0, 0, 0, 157050160.25641, 90865384.6153846, -90865384.6153846, 0, -16826923.0769231, 20192307.6923077, 0, 20192307.6923077, -16826923.0769231, 0, 0, 0, 78523397.4358974, 212019230.769231, 90865384.6153846, 0, -33653846.1538462, -20192307.6923077, 0, -30288461.5384615, 16826923.0769231, 0, 0, 0, 157050160.25641, -212019230.769231, 212019230.769231, 0, 33653846.1538462, 30288461.5384615, 0, 30288461.5384615, 33653846.1538462, 0, 0, 0, 314107051.282051;


    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation
    ChVectorDynamic<double> InternalForceNoDispNoVelNoGravity;
    InternalForceNoDispNoVelNoGravity.resize(48);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceNoDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_NoDispNoVelNoDamping;
    JacobianK_NoDispNoVelNoDamping.resize(48, 48);
    m_element->ComputeKRMmatricesGlobal(JacobianK_NoDispNoVelNoDamping, 1.0, 0.0, 0.0);

    ChMatrixDynamic<double> JacobianR_NoDispNoVelNoDamping;
    JacobianR_NoDispNoVelNoDamping.resize(48, 48);
    m_element->ComputeKRMmatricesGlobal(JacobianR_NoDispNoVelNoDamping, 0.0, 1.0, 0.0);

    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_NoDispNoVelNoDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(48, 48);
    double MaxAbsError_JacR = JacobianR_NoDispNoVelNoDamping.cwiseAbs().maxCoeff();
    
    for (auto i = 0; i < Expected_JacobianK_NoDispNoVelNoDamping.rows(); i++) {
        for (auto j = 0; j < Expected_JacobianK_NoDispNoVelNoDamping.cols(); j++) {
            if (std::abs(Expected_JacobianK_NoDispNoVelNoDamping(i, j)) < small_terms_JacK) {
                double error = std::abs(JacobianK_NoDispNoVelNoDamping(i, j) - Expected_JacobianK_NoDispNoVelNoDamping(i, j));
                if (error > zeros_max_error_JacK)
                    zeros_max_error_JacK = error;
                percent_error_matrix_JacK(i, j) = 0.0;
            }
            else {
                double percent_error = std::abs((JacobianK_NoDispNoVelNoDamping(i, j) - Expected_JacobianK_NoDispNoVelNoDamping(i, j)) / Expected_JacobianK_NoDispNoVelNoDamping(i, j));
                if (percent_error > max_percent_error_JacK)
                    max_percent_error_JacK = percent_error;
                percent_error_matrix_JacK(i, j) = percent_error;
            }
        }
    }


    if (msglvl >= 2) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Jacobian K Term - No Displacement, No Velocity, No Damping = " << std::endl;
        std::cout << JacobianK_NoDispNoVelNoDamping << std::endl;
        std::cout << "Expected Jacobian K Term - No Displacement, No Velocity, No Damping = " << std::endl;
        std::cout << Expected_JacobianK_NoDispNoVelNoDamping << std::endl;
        std::cout << "Percent Error Jacobian K Term - No Displacement, No Velocity, No Damping = " << std::endl;
        std::cout << percent_error_matrix_JacK << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Jacobian K Term - No Displacement, No Velocity, No Damping (Max Abs Error) = "
            << (JacobianK_NoDispNoVelNoDamping - Expected_JacobianK_NoDispNoVelNoDamping).cwiseAbs().maxCoeff() << std::endl;

        std::cout << "Jacobian K Term - No Displacement, No Velocity, No Damping (Max Abs Error - Only Smaller Terms) = "
            << zeros_max_error_JacK;
        if (zeros_max_error_JacK / JacobianK_NoDispNoVelNoDamping.cwiseAbs().maxCoeff() > 0.0001) {
            print_red(" - Test FAILED\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl;
        }

        std::cout << "Jacobian K Term - No Displacement, No Velocity, No Damping (Max Abs % Error - Only Larger Terms) = "
            << max_percent_error_JacK * 100 << "%";
        if (max_percent_error_JacK > 0.01) {
            print_red(" - Test FAILED\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl;
        }
    }
    if (zeros_max_error_JacK / JacobianK_NoDispNoVelNoDamping.cwiseAbs().maxCoeff() > 0.0001) {
        passed_tests = false;
    }
    if (max_percent_error_JacK > 0.01) {
        passed_tests = false;
    }

    if (msglvl >= 2) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Jacobian R Term - No Displacement, No Velocity, No Damping = " << std::endl;
        std::cout << JacobianR_NoDispNoVelNoDamping << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Jacobian R Term - No Displacement, No Velocity, No Damping (Max Abs Error) = "
            << MaxAbsError_JacR;

        if (MaxAbsError_JacR > 0.01) {
            print_red(" - Test FAILED\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl;
        }
    }
    if (MaxAbsError_JacR > 0.01) {
        passed_tests = false;
    }

    return(passed_tests);
}

template <typename ElementVersion, typename MaterialVersion>
bool ANCFShellTest<ElementVersion, MaterialVersion>::JacobianSmallDispNoVelNoDampingCheck(int msglvl) {
    // =============================================================================
    //  Check the Jacobian at Small Displacement No Velocity - No Damping
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================
    ChMatrixNM<double, 48, 48> Expected_JacobianK_SmallDispNoVelNoDamping;
    Expected_JacobianK_SmallDispNoVelNoDamping <<
        1592308526.3354, 504807692.307692, 1230384.61538462, 136538569.423301, 20192307.6923077, 121057.692307692, 175000075.210014, -20192307.6923077, 109439.102564103, -102884.615384615, 0, -212019230.769231, -1107692984.74449, 100961538.461538, -1351538.46153846, 69230826.122019, -20192307.6923077, 141250, -134615429.029944, -100961538.461538, -106554.487179487, -102884.615384615, 0, -212019230.769231, -588461733.349919, -504807692.307692, -61923.0769230769, 72115403.3097992, 100961538.461538, 11634.6153846154, 100961568.117491, 100961538.461538, 16362.1794871794, -8653.84615384615, 0, -90865384.6153846, 103846191.75901, -100961538.461538, 183076.923076923, 4807705.00851712, -100961538.461538, 31826.9230769231, -60576936.9374215, 20192307.6923077, -39439.1025641026, -8653.84615384615, 0, -90865384.6153846,
        504807692.307692, 1592308526.3354, -61923.0769230769, -20192307.6923077, 175000107.88484, -47868.5897435897, 20192307.6923077, 136538536.748476, -79807.6923076923, 0, -102884.615384615, -212019230.769231, -100961538.461538, 103845476.793975, -220769.230769231, 20192307.6923077, -60576866.1856733, 59983.9743589744, -100961538.461538, 4807647.8931325, -103846.153846154, 0, -102884.615384615, -90865384.6153846, -504807692.307692, -588461733.349919, 99615.3846153846, 100961538.461538, 100961557.155953, 15016.0256410256, 100961538.461538, 72115414.2713376, 10576.9230769231, 0, -8653.84615384615, -90865384.6153846, 100961538.461538, -1107692269.77945, 183076.923076923, -100961538.461538, -134615371.91456, -6939.10256410257, -20192307.6923077, 69230755.3702707, -5769.23076923076, 0, -8653.84615384615, -212019230.769231,
        1230384.61538462, -61923.0769230769, 707694629.985207, 124711.538461538, -26538.4615384615, 69231059.840416, 101923.076923077, -119903.846153846, 69230989.6772458, -141346153.846154, -141346153.846154, -360096.153846154, -1230384.61538462, -61923.0769230769, -223078954.233459, 124711.538461538, 26538.4615384615, 1923266.7955442, -101923.076923077, -119903.846153846, -28846262.3782948, -141346153.846154, -60576923.0769231, -360096.153846154, -99615.3846153846, 61923.0769230769, -261538900.818989, 19134.6153846154, 25384.6153846154, 38461578.2306796, 23846.1538461538, 21442.3076923077, 38461596.1793975, -60576923.0769231, -60576923.0769231, -30288.4615384615, 99615.3846153846, 61923.0769230769, -223076774.93276, 19134.6153846154, -25384.6153846154, -28846109.8141922, -23846.1538461538, 21442.3076923077, 1923026.11955352, -60576923.0769231, -141346153.846154, -30288.4615384615,
        136538569.423301, -20192307.6923077, 124711.538461538, 133333367.595087, 0, 65657.0512820513, 5.80586080586081, 14022435.8974359, 3333.33333333333, 6891.02564102564, 0, 33653846.1538462, -69230847.1797113, 20192307.6923077, -155000, -37179495.1885188, -3365384.61538461, -9022.4358974359, -6.63919413919414, -14022435.8974359, -9503.20512820513, -14583.3333333333, 0, -33653846.1538462, -72115414.2713376, -100961538.461538, 21442.3076923077, -9935898.57313626, 16826923.0769231, -8894.23076923077, 3.90567765567765, 14022435.8974359, -3285.25641025641, 0, 0, -16826923.0769231, 4807692.02774789, 100961538.461538, 8846.15384615384, 55128210.1176174, 0, 16586.5384615385, -0.924908424908425, -14022435.8974359, -641.02564102564, 5769.23076923077, 0, 16826923.0769231,
        20192307.6923077, 175000107.88484, -26538.4615384615, 0, 62820547.082266, -11602.5641025641, 14022435.8974359, 5.80586080586081, -2291.66666666667, 0, 6891.02564102564, -30288461.5384615, -20192307.6923077, 60576845.127981, -55576.9230769231, 3365384.61538461, -29166674.6756982, 17131.4102564103, -14022435.8974359, -6.63919413919414, -15208.3333333333, 0, -14583.3333333333, -20192307.6923077, -100961538.461538, -100961568.117491, 23846.1538461538, 16826923.0769231, 15705127.0678894, 2580.12820512821, 14022435.8974359, 3.90567765567765, 1618.58974358974, 0, 0, -20192307.6923077, 100961538.461538, -134615384.895329, 58269.2307692308, 0, -8974353.98494668, 3429.48717948718, -14022435.8974359, -0.924908424908425, -5657.05128205128, 0, 5769.23076923077, -30288461.5384615,
        121057.692307692, -47868.5897435897, 69231059.840416, 65657.0512820513, -11602.5641025641, 43589839.0116027, 1746.79487179487, -4070.51282051282, 18.2211538461538, 22435897.4358974, -20192307.6923077, 24118.5897435897, -127788.461538462, -22131.4102564103, -1923310.76990317, -12387.8205128205, 11137.8205128205, -14743608.3028884, -8477.5641025641, -17467.9487179487, -12.724358974359, -22435897.4358974, -13461538.4615385, -51041.6666666667, 10576.9230769231, 16362.1794871794, -38461596.1793975, -6971.15384615385, 4246.79487179487, 1282045.22305152, -1618.58974358974, 3285.25641025641, 5, -11217948.7179487, -13461538.4615385, 0, -3846.15384615385, 53637.8205128205, -28846152.8911153, 14182.6923076923, 1987.17948717949, 10256425.7290733, 1618.58974358974, -4631.41025641026, -3.65384615384615, 11217948.7179487, -20192307.6923077, 20192.3076923077,
        175000075.210014, 20192307.6923077, 101923.076923077, 5.80586080586081, 14022435.8974359, 1746.79487179487, 62820547.058956, 0, 40416.6666666667, -13301.2820512821, 0, -30288461.5384615, -134615473.741483, 100961538.461538, -185576.923076923, 5.83791208791209, -14022435.8974359, 18573.717948718, -8974348.7605877, 0, -3397.4358974359, -14423.0769230769, 0, -30288461.5384615, -100961557.155953, -100961538.461538, 25384.6153846154, -6.41483516483517, 14022435.8974359, -5785.25641025641, 15705127.0678894, 16826923.0769231, -4246.79487179487, 0, 0, -20192307.6923077, 60576955.6874215, -20192307.6923077, 58269.2307692308, 9.80311355311355, -14022435.8974359, 11041.6666666667, -29166677.9361877, 3365384.61538461, -15464.7435897436, -1121.79487179487, 0, -20192307.6923077,
        -20192307.6923077, 136538536.748476, -119903.846153846, 14022435.8974359, 5.80586080586081, -4070.51282051282, 0, 133333367.571777, -24727.5641025641, 0, -13301.2820512821, 33653846.1538462, 100961538.461538, 4807603.18159404, 91923.0769230769, -14022435.8974359, 5.83791208791209, -8044.87179487179, 0, 55128215.3419764, -14182.6923076923, 0, -14423.0769230769, 16826923.0769231, -100961538.461538, -72115403.3097992, 19134.6153846154, 14022435.8974359, -6.41483516483517, 4118.58974358974, 16826923.0769231, -9935898.57313626, 6971.15384615385, 0, 0, -16826923.0769231, 20192307.6923077, -69230736.6202707, 8846.15384615384, -14022435.8974359, 9.80311355311355, -2099.35897435898, -3365384.61538461, -37179498.4490083, 8766.02564102564, 0, -1121.79487179487, -33653846.1538462,
        109439.102564103, -79807.6923076923, 69230989.6772458, 3333.33333333333, -2291.66666666667, 18.2211538461538, 40416.6666666667, -24727.5641025641, 43589831.3018125, -20192307.6923077, 22435897.4358974, -46554.4871794872, -167131.41025641, 75865.3846153846, -28846421.2244486, 16233.9743589744, -5785.25641025641, 19.6794871794872, -1955.1282051282, -16586.5384615385, 10256433.8059964, -20192307.6923077, 11217948.7179487, -50480.7692307692, 15016.0256410256, 11634.6153846154, -38461578.2306796, -4118.58974358974, 5785.25641025641, -15.6410256410256, -2580.12820512821, 8894.23076923077, 1282045.22305152, -13461538.4615385, -11217948.7179487, 0, 42676.2820512821, -7692.30769230769, -1922990.22211763, 8782.05128205128, -4439.10256410256, 24.9679487179487, -12804.4871794872, 12131.4102564103, -14743620.0511401, -13461538.4615385, -22435897.4358974, -3926.28205128205,
        -102884.615384615, 0, -141346153.846154, 6891.02564102564, 0, 22435897.4358974, -13301.2820512821, 0, -20192307.6923077, 89753739.3310379, 4206.73076923077, 65070.1322115384, 176923.076923077, 0, 141346153.846154, -28044.8717948718, 0, -22435897.4358974, 14423.0769230769, 0, 20192307.6923077, 44865110.2856007, 841.346153846154, 42621.3341346154, -8653.84615384615, 0, 60576923.0769231, 0, 0, -11217948.7179487, 0, 0, -13461538.4615385, 22430866.2484258, -4206.73076923077, 13461.1738782051, -65384.6153846154, 0, -60576923.0769231, -5769.23076923077, 0, 11217948.7179487, 12339.7435897436, 0, 13461538.4615385, 44873493.7503203, -841.346153846154, 20193.5136217949,
        0, -102884.615384615, -141346153.846154, 0, 6891.02564102564, -20192307.6923077, 0, -13301.2820512821, 22435897.4358974, 4206.73076923077, 89753739.3310379, -15704.7636217949, 0, 176923.076923077, -60576923.0769231, 0, -28044.8717948718, 13461538.4615385, 0, 14423.0769230769, 11217948.7179487, -841.346153846154, 44873523.7471392, -20193.8501602564, 0, -8653.84615384615, 60576923.0769231, 0, 0, -13461538.4615385, 0, 0, -11217948.7179487, -4206.73076923077, 22430866.2484258, -13460.8373397436, 0, -65384.6153846154, 141346153.846154, 0, -5769.23076923077, 20192307.6923077, 0, 12339.7435897436, -22435897.4358974, 841.346153846154, 44865080.2887818, -11217.4719551282,
        -212019230.769231, -212019230.769231, -360096.153846154, 33653846.1538462, -30288461.5384615, 24118.5897435897, -30288461.5384615, 33653846.1538462, -46554.4871794872, 65070.1322115384, -15704.7636217949, 314107175.961966, 212019230.769231, -90865384.6153846, 619230.769230769, -33653846.1538462, 20192307.6923077, -98157.0512820513, 30288461.5384615, 16826923.0769231, 50480.7692307692, 42622.1754807692, -20192.672275641, 157050268.014889, 90865384.6153846, 90865384.6153846, -30288.4615384615, -16826923.0769231, -20192307.6923077, 0, -20192307.6923077, -16826923.0769231, 0, 13460.8373397436, -13461.1738782051, 78523436.8446504, -90865384.6153846, 212019230.769231, -228846.153846154, 16826923.0769231, 30288461.5384615, -20192.3076923077, 20192307.6923077, -33653846.1538462, 43189.1025641026, 20193.0088141026, -11218.3133012821, 157050198.024649,
        -1107692984.74449, -100961538.461538, -1230384.61538462, -69230847.1797113, -20192307.6923077, -127788.461538462, -134615473.741483, 100961538.461538, -167131.41025641, 176923.076923077, 0, 212019230.769231, 1592309532.06966, -504807692.307692, 1836153.84615385, -136538620.480993, 20192307.6923077, -134519.230769231, 175000158.942532, 20192307.6923077, 152708.333333333, 271153.846153846, 0, 212019230.769231, 103845476.793975, 100961538.461538, 61923.0769230769, -4807603.18159404, -100961538.461538, 75865.3846153846, -60576845.127981, -20192307.6923077, -22131.4102564103, 176923.076923077, 0, 90865384.6153846, -588462024.11915, 504807692.307692, -667692.307692308, -72115481.4828761, 100961538.461538, -119326.923076923, 100961635.32903, -100961538.461538, 137516.025641026, 82692.3076923077, 0, 90865384.6153846,
        100961538.461538, 103845476.793975, -61923.0769230769, 20192307.6923077, 60576845.127981, -22131.4102564103, 100961538.461538, 4807603.18159404, 75865.3846153846, 0, 176923.076923077, -90865384.6153846, -504807692.307692, 1592309532.06966, -1836153.84615385, -20192307.6923077, -175000158.942532, 152708.333333333, -20192307.6923077, 136538620.480993, -134519.230769231, 0, 271153.846153846, -212019230.769231, -100961538.461538, -1107692984.74449, 1230384.61538462, -100961538.461538, 134615473.741483, -167131.41025641, 20192307.6923077, 69230847.1797113, -127788.461538462, 0, 176923.076923077, -212019230.769231, 504807692.307692, -588462024.11915, 667692.307692308, 100961538.461538, -100961635.32903, 137516.025641026, -100961538.461538, 72115481.4828761, -119326.923076923, 0, 82692.3076923077, -90865384.6153846,
        -1351538.46153846, -220769.230769231, -223078954.233459, -155000, -55576.9230769231, -1923310.76990317, -185576.923076923, 91923.0769230769, -28846421.2244486, 141346153.846154, -60576923.0769231, 619230.769230769, 1836153.84615385, -1836153.84615385, 707697826.978214, -94423.0769230769, 174038.461538462, -69231246.0583647, 174038.461538462, -94423.0769230769, 69231246.0583647, 141346153.846154, -141346153.846154, 949038.461538462, 220769.230769231, 1351538.46153846, -223078954.233459, 91923.0769230769, -185576.923076923, 28846421.2244486, -55576.9230769231, -155000, 1923310.76990317, 60576923.0769231, -141346153.846154, 619230.769230769, -705384.615384615, 705384.615384615, -261539918.511296, -130192.307692308, 147884.615384615, -38461829.0640129, 147884.615384615, -130192.307692308, 38461829.0640129, 60576923.0769231, -60576923.0769231, 289423.076923077,
        69230826.122019, 20192307.6923077, 124711.538461538, -37179495.1885188, 3365384.61538461, -12387.8205128205, 5.83791208791209, -14022435.8974359, 16233.9743589744, -28044.8717948718, 0, -33653846.1538462, -136538620.480993, -20192307.6923077, -94423.0769230769, 133333390.92842, 0, 65657.0512820513, -7.40842490842491, 14022435.8974359, -10064.1025641026, -13301.2820512821, 0, 33653846.1538462, -4807647.8931325, 100961538.461538, -119903.846153846, 55128215.3419764, 0, 16586.5384615385, -6.63919413919414, -14022435.8974359, 17467.9487179487, -14423.0769230769, 0, 16826923.0769231, 72115442.2521069, -100961538.461538, 89615.3846153846, -9935889.91929011, -16826923.0769231, 7932.69230769231, -6.82234432234432, 14022435.8974359, -13541.6666666667, -13461.5384615385, 0, -16826923.0769231,
        -20192307.6923077, -60576866.1856733, 26538.4615384615, -3365384.61538461, -29166674.6756982, 11137.8205128205, -14022435.8974359, 5.83791208791209, -5785.25641025641, 0, -28044.8717948718, 20192307.6923077, 20192307.6923077, -175000158.942532, 174038.461538462, 0, 62820570.4155994, -47628.2051282051, 14022435.8974359, -7.40842490842491, 11842.9487179487, 0, -13301.2820512821, 30288461.5384615, 100961538.461538, 134615429.029944, -101923.076923077, 0, -8974348.7605877, 1955.1282051282, -14022435.8974359, -6.63919413919414, 8477.5641025641, 0, -14423.0769230769, 30288461.5384615, -100961538.461538, 100961596.098261, -98653.8461538462, -16826923.0769231, 15705135.7217355, -17387.8205128205, 14022435.8974359, -6.82234432234432, 11041.6666666667, 0, -13461.5384615385, 20192307.6923077,
        141250, 59983.9743589744, 1923266.7955442, -9022.4358974359, 17131.4102564103, -14743608.3028884, 18573.717948718, -8044.87179487179, 19.6794871794872, -22435897.4358974, 13461538.4615385, -98157.0512820513, -134519.230769231, 152708.333333333, -69231246.0583647, 65657.0512820513, -47628.2051282051, 43589900.9346796, -11842.9487179487, 10064.1025641026, -31.2339743589744, 22435897.4358974, 20192307.6923077, -46554.4871794872, -103846.153846154, -106554.487179487, 28846262.3782948, 14182.6923076923, 3397.4358974359, 10256433.8059964, 15208.3333333333, 9503.20512820513, -12.724358974359, 11217948.7179487, 20192307.6923077, -50480.7692307692, 97115.3846153846, -106137.820512821, 38461716.8845257, 9855.76923076923, -19054.4871794872, 1282075.51151306, -15208.3333333333, 12708.3333333333, -22.948717948718, -11217948.7179487, 13461538.4615385, -47115.3846153846,
        -134615429.029944, -100961538.461538, -101923.076923077, -6.63919413919414, -14022435.8974359, -8477.5641025641, -8974348.7605877, 0, -1955.1282051282, 14423.0769230769, 0, 30288461.5384615, 175000158.942532, -20192307.6923077, 174038.461538462, -7.40842490842491, 14022435.8974359, -11842.9487179487, 62820570.4155994, 0, 47628.2051282051, 13301.2820512821, 0, 30288461.5384615, 60576866.1856733, 20192307.6923077, 26538.4615384615, 5.83791208791209, -14022435.8974359, 5785.25641025641, -29166674.6756982, -3365384.61538461, -11137.8205128205, 28044.8717948718, 0, 20192307.6923077, -100961596.098261, 100961538.461538, -98653.8461538462, -6.82234432234432, 14022435.8974359, -11041.6666666667, 15705135.7217355, -16826923.0769231, 17387.8205128205, 13461.5384615385, 0, 20192307.6923077,
        -100961538.461538, 4807647.8931325, -119903.846153846, -14022435.8974359, -6.63919413919414, -17467.9487179487, 0, 55128215.3419764, -16586.5384615385, 0, 14423.0769230769, 16826923.0769231, 20192307.6923077, 136538620.480993, -94423.0769230769, 14022435.8974359, -7.40842490842491, 10064.1025641026, 0, 133333390.92842, -65657.0512820513, 0, 13301.2820512821, 33653846.1538462, -20192307.6923077, -69230826.122019, 124711.538461538, -14022435.8974359, 5.83791208791209, -16233.9743589744, 3365384.61538461, -37179495.1885188, 12387.8205128205, 0, 28044.8717948718, -33653846.1538462, 100961538.461538, -72115442.2521069, 89615.3846153846, 14022435.8974359, -6.82234432234432, 13541.6666666667, -16826923.0769231, -9935889.91929011, -7932.69230769231, 0, 13461.5384615385, -16826923.0769231,
        -106554.487179487, -103846.153846154, -28846262.3782948, -9503.20512820513, -15208.3333333333, -12.724358974359, -3397.4358974359, -14182.6923076923, 10256433.8059964, 20192307.6923077, 11217948.7179487, 50480.7692307692, 152708.333333333, -134519.230769231, 69231246.0583647, -10064.1025641026, 11842.9487179487, -31.2339743589744, 47628.2051282051, -65657.0512820513, 43589900.9346796, 20192307.6923077, 22435897.4358974, 46554.4871794872, 59983.9743589744, 141250, -1923266.7955442, 8044.87179487179, -18573.717948718, 19.6794871794872, -17131.4102564103, 9022.4358974359, -14743608.3028884, 13461538.4615385, -22435897.4358974, 98157.0512820513, -106137.820512821, 97115.3846153846, -38461716.8845257, -12708.3333333333, 15208.3333333333, -22.948717948718, 19054.4871794872, -9855.76923076923, 1282075.51151306, 13461538.4615385, -11217948.7179487, 47115.3846153846,
        -102884.615384615, 0, -141346153.846154, -14583.3333333333, 0, -22435897.4358974, -14423.0769230769, 0, -20192307.6923077, 44865110.2856007, -841.346153846154, 42622.1754807692, 271153.846153846, 0, 141346153.846154, -13301.2820512821, 0, 22435897.4358974, 13301.2820512821, 0, 20192307.6923077, 89753780.0089786, -4206.73076923077, 65074.3389423077, -102884.615384615, 0, 60576923.0769231, 14423.0769230769, 0, 11217948.7179487, 14583.3333333333, 0, -13461538.4615385, 44873523.7471392, 841.346153846154, 20192.672275641, -65384.6153846154, 0, -60576923.0769231, -13461.5384615385, 0, -11217948.7179487, 13461.5384615385, 0, 13461538.4615385, 22430866.2467431, 4206.73076923077, 13456.9671474359,
        0, -102884.615384615, -60576923.0769231, 0, -14583.3333333333, -13461538.4615385, 0, -14423.0769230769, 11217948.7179487, 841.346153846154, 44873523.7471392, -20192.672275641, 0, 271153.846153846, -141346153.846154, 0, -13301.2820512821, 20192307.6923077, 0, 13301.2820512821, 22435897.4358974, -4206.73076923077, 89753780.0089786, -65074.3389423077, 0, -102884.615384615, 141346153.846154, 0, 14423.0769230769, -20192307.6923077, 0, 14583.3333333333, -22435897.4358974, -841.346153846154, 44865110.2856007, -42622.1754807692, 0, -65384.6153846154, 60576923.0769231, 0, -13461.5384615385, 13461538.4615385, 0, 13461.5384615385, -11217948.7179487, 4206.73076923077, 22430866.2467431, -13456.9671474359,
        -212019230.769231, -90865384.6153846, -360096.153846154, -33653846.1538462, -20192307.6923077, -51041.6666666667, -30288461.5384615, 16826923.0769231, -50480.7692307692, 42621.3341346154, -20193.8501602564, 157050268.014889, 212019230.769231, -212019230.769231, 949038.461538462, 33653846.1538462, 30288461.5384615, -46554.4871794872, 30288461.5384615, 33653846.1538462, 46554.4871794872, 65074.3389423077, -65074.3389423077, 314107270.881461, 90865384.6153846, 212019230.769231, -360096.153846154, 16826923.0769231, -30288461.5384615, 50480.7692307692, -20192307.6923077, -33653846.1538462, 51041.6666666667, 20193.8501602564, -42621.3341346154, 157050268.014889, -90865384.6153846, 90865384.6153846, -228846.153846154, -16826923.0769231, 20192307.6923077, -47115.3846153846, 20192307.6923077, -16826923.0769231, 47115.3846153846, 13456.6306089744, -13456.6306089744, 78523436.8387609,
        -588461733.349919, -504807692.307692, -99615.3846153846, -72115414.2713376, -100961538.461538, 10576.9230769231, -100961557.155953, -100961538.461538, 15016.0256410256, -8653.84615384615, 0, 90865384.6153846, 103845476.793975, -100961538.461538, 220769.230769231, -4807647.8931325, 100961538.461538, -103846.153846154, 60576866.1856733, -20192307.6923077, 59983.9743589744, -102884.615384615, 0, 90865384.6153846, 1592308526.3354, 504807692.307692, 61923.0769230769, -136538536.748476, -20192307.6923077, -79807.6923076923, -175000107.88484, 20192307.6923077, -47868.5897435897, -102884.615384615, 0, 212019230.769231, -1107692269.77945, 100961538.461538, -183076.923076923, -69230755.3702707, 20192307.6923077, -5769.23076923076, 134615371.91456, 100961538.461538, -6939.10256410257, -8653.84615384615, 0, 212019230.769231,
        -504807692.307692, -588461733.349919, 61923.0769230769, -100961538.461538, -100961568.117491, 16362.1794871794, -100961538.461538, -72115403.3097992, 11634.6153846154, 0, -8653.84615384615, 90865384.6153846, 100961538.461538, -1107692984.74449, 1351538.46153846, 100961538.461538, 134615429.029944, -106554.487179487, 20192307.6923077, -69230826.122019, 141250, 0, -102884.615384615, 212019230.769231, 504807692.307692, 1592308526.3354, -1230384.61538462, 20192307.6923077, -175000075.210014, 109439.102564103, -20192307.6923077, -136538569.423301, 121057.692307692, 0, -102884.615384615, 212019230.769231, -100961538.461538, 103846191.75901, -183076.923076923, -20192307.6923077, 60576936.9374215, -39439.1025641026, 100961538.461538, -4807705.00851712, 31826.9230769231, 0, -8653.84615384615, 90865384.6153846,
        -61923.0769230769, 99615.3846153846, -261538900.818989, 21442.3076923077, 23846.1538461538, -38461596.1793975, 25384.6153846154, 19134.6153846154, -38461578.2306796, 60576923.0769231, 60576923.0769231, -30288.4615384615, 61923.0769230769, 1230384.61538462, -223078954.233459, -119903.846153846, -101923.076923077, 28846262.3782948, 26538.4615384615, 124711.538461538, -1923266.7955442, 60576923.0769231, 141346153.846154, -360096.153846154, 61923.0769230769, -1230384.61538462, 707694629.985207, -119903.846153846, 101923.076923077, -69230989.6772458, -26538.4615384615, 124711.538461538, -69231059.840416, 141346153.846154, 141346153.846154, -360096.153846154, -61923.0769230769, -99615.3846153846, -223076774.93276, 21442.3076923077, -23846.1538461538, -1923026.11955352, -25384.6153846154, 19134.6153846154, 28846109.8141922, 141346153.846154, 60576923.0769231, -30288.4615384615,
        72115403.3097992, 100961538.461538, 19134.6153846154, -9935898.57313626, 16826923.0769231, -6971.15384615385, -6.41483516483517, 14022435.8974359, -4118.58974358974, 0, 0, -16826923.0769231, -4807603.18159404, -100961538.461538, 91923.0769230769, 55128215.3419764, 0, 14182.6923076923, 5.83791208791209, -14022435.8974359, 8044.87179487179, 14423.0769230769, 0, 16826923.0769231, -136538536.748476, 20192307.6923077, -119903.846153846, 133333367.571777, 0, 24727.5641025641, 5.80586080586081, 14022435.8974359, 4070.51282051282, 13301.2820512821, 0, 33653846.1538462, 69230736.6202707, -20192307.6923077, 8846.15384615384, -37179498.4490083, -3365384.61538461, -8766.02564102564, 9.80311355311355, -14022435.8974359, 2099.35897435898, 1121.79487179487, 0, -33653846.1538462,
        100961538.461538, 100961557.155953, 25384.6153846154, 16826923.0769231, 15705127.0678894, 4246.79487179487, 14022435.8974359, -6.41483516483517, 5785.25641025641, 0, 0, -20192307.6923077, -100961538.461538, 134615473.741483, -185576.923076923, 0, -8974348.7605877, 3397.4358974359, -14022435.8974359, 5.83791208791209, -18573.717948718, 0, 14423.0769230769, -30288461.5384615, -20192307.6923077, -175000075.210014, 101923.076923077, 0, 62820547.058956, -40416.6666666667, 14022435.8974359, 5.80586080586081, -1746.79487179487, 0, 13301.2820512821, -30288461.5384615, 20192307.6923077, -60576955.6874215, 58269.2307692308, 3365384.61538461, -29166677.9361877, 15464.7435897436, -14022435.8974359, 9.80311355311355, -11041.6666666667, 0, 1121.79487179487, -20192307.6923077,
        11634.6153846154, 15016.0256410256, 38461578.2306796, -8894.23076923077, 2580.12820512821, 1282045.22305152, -5785.25641025641, 4118.58974358974, -15.6410256410256, -11217948.7179487, -13461538.4615385, 0, 75865.3846153846, -167131.41025641, 28846421.2244486, 16586.5384615385, 1955.1282051282, 10256433.8059964, 5785.25641025641, -16233.9743589744, 19.6794871794872, 11217948.7179487, -20192307.6923077, 50480.7692307692, -79807.6923076923, 109439.102564103, -69230989.6772458, 24727.5641025641, -40416.6666666667, 43589831.3018125, 2291.66666666667, -3333.33333333333, 18.2211538461538, 22435897.4358974, -20192307.6923077, 46554.4871794872, -7692.30769230769, 42676.2820512821, 1922990.22211763, -12131.4102564103, 12804.4871794872, -14743620.0511401, 4439.10256410256, -8782.05128205128, 24.9679487179487, -22435897.4358974, -13461538.4615385, 3926.28205128205,
        100961568.117491, 100961538.461538, 23846.1538461538, 3.90567765567765, 14022435.8974359, -1618.58974358974, 15705127.0678894, 16826923.0769231, -2580.12820512821, 0, 0, -20192307.6923077, -60576845.127981, 20192307.6923077, -55576.9230769231, -6.63919413919414, -14022435.8974359, 15208.3333333333, -29166674.6756982, 3365384.61538461, -17131.4102564103, 14583.3333333333, 0, -20192307.6923077, -175000107.88484, -20192307.6923077, -26538.4615384615, 5.80586080586081, 14022435.8974359, 2291.66666666667, 62820547.082266, 0, 11602.5641025641, -6891.02564102564, 0, -30288461.5384615, 134615384.895329, -100961538.461538, 58269.2307692308, -0.924908424908425, -14022435.8974359, 5657.05128205128, -8974353.98494668, 0, -3429.48717948718, -5769.23076923077, 0, -30288461.5384615,
        100961538.461538, 72115414.2713376, 21442.3076923077, 14022435.8974359, 3.90567765567765, 3285.25641025641, 16826923.0769231, -9935898.57313626, 8894.23076923077, 0, 0, -16826923.0769231, -20192307.6923077, 69230847.1797113, -155000, -14022435.8974359, -6.63919413919414, 9503.20512820513, -3365384.61538461, -37179495.1885188, 9022.4358974359, 0, 14583.3333333333, -33653846.1538462, 20192307.6923077, -136538569.423301, 124711.538461538, 14022435.8974359, 5.80586080586081, -3333.33333333333, 0, 133333367.595087, -65657.0512820513, 0, -6891.02564102564, 33653846.1538462, -100961538.461538, -4807692.02774789, 8846.15384615384, -14022435.8974359, -0.924908424908425, 641.02564102564, 0, 55128210.1176174, -16586.5384615385, 0, -5769.23076923077, 16826923.0769231,
        16362.1794871794, 10576.9230769231, 38461596.1793975, -3285.25641025641, 1618.58974358974, 5, -4246.79487179487, 6971.15384615385, 1282045.22305152, -13461538.4615385, -11217948.7179487, 0, -22131.4102564103, -127788.461538462, 1923310.76990317, 17467.9487179487, 8477.5641025641, -12.724358974359, -11137.8205128205, 12387.8205128205, -14743608.3028884, -13461538.4615385, -22435897.4358974, 51041.6666666667, -47868.5897435897, 121057.692307692, -69231059.840416, 4070.51282051282, -1746.79487179487, 18.2211538461538, 11602.5641025641, -65657.0512820513, 43589839.0116027, -20192307.6923077, 22435897.4358974, -24118.5897435897, 53637.8205128205, -3846.15384615385, 28846152.8911153, 4631.41025641026, -1618.58974358974, -3.65384615384615, -1987.17948717949, -14182.6923076923, 10256425.7290733, -20192307.6923077, 11217948.7179487, -20192.3076923077,
        -8653.84615384615, 0, -60576923.0769231, 0, 0, -11217948.7179487, 0, 0, -13461538.4615385, 22430866.2484258, -4206.73076923077, 13460.8373397436, 176923.076923077, 0, 60576923.0769231, -14423.0769230769, 0, 11217948.7179487, 28044.8717948718, 0, 13461538.4615385, 44873523.7471392, -841.346153846154, 20193.8501602564, -102884.615384615, 0, 141346153.846154, 13301.2820512821, 0, 22435897.4358974, -6891.02564102564, 0, -20192307.6923077, 89753739.3310379, 4206.73076923077, 15704.7636217949, -65384.6153846154, 0, -141346153.846154, -12339.7435897436, 0, -22435897.4358974, 5769.23076923077, 0, 20192307.6923077, 44865080.2887818, 841.346153846154, 11217.4719551282,
        0, -8653.84615384615, -60576923.0769231, 0, 0, -13461538.4615385, 0, 0, -11217948.7179487, -4206.73076923077, 22430866.2484258, -13461.1738782051, 0, 176923.076923077, -141346153.846154, 0, -14423.0769230769, 20192307.6923077, 0, 28044.8717948718, -22435897.4358974, 841.346153846154, 44865110.2856007, -42621.3341346154, 0, -102884.615384615, 141346153.846154, 0, 13301.2820512821, -20192307.6923077, 0, -6891.02564102564, 22435897.4358974, 4206.73076923077, 89753739.3310379, -65070.1322115384, 0, -65384.6153846154, 60576923.0769231, 0, -12339.7435897436, 13461538.4615385, 0, 5769.23076923077, 11217948.7179487, -841.346153846154, 44873493.7503203, -20193.5136217949,
        -90865384.6153846, -90865384.6153846, -30288.4615384615, -16826923.0769231, -20192307.6923077, 0, -20192307.6923077, -16826923.0769231, 0, 13461.1738782051, -13460.8373397436, 78523436.8446504, 90865384.6153846, -212019230.769231, 619230.769230769, 16826923.0769231, 30288461.5384615, -50480.7692307692, 20192307.6923077, -33653846.1538462, 98157.0512820513, 20192.672275641, -42622.1754807692, 157050268.014889, 212019230.769231, 212019230.769231, -360096.153846154, 33653846.1538462, -30288461.5384615, 46554.4871794872, -30288461.5384615, 33653846.1538462, -24118.5897435897, 15704.7636217949, -65070.1322115384, 314107175.961966, -212019230.769231, 90865384.6153846, -228846.153846154, -33653846.1538462, 20192307.6923077, -43189.1025641026, 30288461.5384615, 16826923.0769231, 20192.3076923077, 11218.3133012821, -20193.0088141026, 157050198.024649,
        103846191.75901, 100961538.461538, 99615.3846153846, 4807692.02774789, 100961538.461538, -3846.15384615385, 60576955.6874215, 20192307.6923077, 42676.2820512821, -65384.6153846154, 0, -90865384.6153846, -588462024.11915, 504807692.307692, -705384.615384615, 72115442.2521069, -100961538.461538, 97115.3846153846, -100961596.098261, 100961538.461538, -106137.820512821, -65384.6153846154, 0, -90865384.6153846, -1107692269.77945, -100961538.461538, -61923.0769230769, 69230736.6202707, 20192307.6923077, -7692.30769230769, 134615384.895329, -100961538.461538, 53637.8205128205, -65384.6153846154, 0, -212019230.769231, 1592308102.13959, -504807692.307692, 667692.307692308, 136538531.84463, -20192307.6923077, 93269.2307692308, -175000070.306168, -20192307.6923077, -91137.8205128205, -65384.6153846154, 0, -212019230.769231,
        -100961538.461538, -1107692269.77945, 61923.0769230769, 100961538.461538, -134615384.895329, 53637.8205128205, -20192307.6923077, -69230736.6202707, -7692.30769230769, 0, -65384.6153846154, 212019230.769231, 504807692.307692, -588462024.11915, 705384.615384615, -100961538.461538, 100961596.098261, -106137.820512821, 100961538.461538, -72115442.2521069, 97115.3846153846, 0, -65384.6153846154, 90865384.6153846, 100961538.461538, 103846191.75901, -99615.3846153846, -20192307.6923077, -60576955.6874215, 42676.2820512821, -100961538.461538, -4807692.02774789, -3846.15384615385, 0, -65384.6153846154, 90865384.6153846, -504807692.307692, 1592308102.13959, -667692.307692308, 20192307.6923077, 175000070.306168, -91137.8205128205, 20192307.6923077, -136538531.84463, 93269.2307692308, 0, -65384.6153846154, 212019230.769231,
        183076.923076923, 183076.923076923, -223076774.93276, 8846.15384615384, 58269.2307692308, -28846152.8911153, 58269.2307692308, 8846.15384615384, -1922990.22211763, -60576923.0769231, 141346153.846154, -228846.153846154, -667692.307692308, 667692.307692308, -261539918.511296, 89615.3846153846, -98653.8461538462, 38461716.8845257, -98653.8461538462, 89615.3846153846, -38461716.8845257, -60576923.0769231, 60576923.0769231, -228846.153846154, -183076.923076923, -183076.923076923, -223076774.93276, 8846.15384615384, 58269.2307692308, 1922990.22211763, 58269.2307692308, 8846.15384615384, 28846152.8911153, -141346153.846154, 60576923.0769231, -228846.153846154, 667692.307692308, -667692.307692308, 707693468.376816, 89615.3846153846, -98653.8461538462, 69230964.9977587, -98653.8461538462, 89615.3846153846, -69230964.9977587, -141346153.846154, 141346153.846154, -228846.153846154,
        4807705.00851712, -100961538.461538, 19134.6153846154, 55128210.1176174, 0, 14182.6923076923, 9.80311355311355, -14022435.8974359, 8782.05128205128, -5769.23076923077, 0, 16826923.0769231, -72115481.4828761, 100961538.461538, -130192.307692308, -9935889.91929011, -16826923.0769231, 9855.76923076923, -6.82234432234432, 14022435.8974359, -12708.3333333333, -13461.5384615385, 0, -16826923.0769231, -69230755.3702707, -20192307.6923077, 21442.3076923077, -37179498.4490083, 3365384.61538461, -12131.4102564103, -0.924908424908425, -14022435.8974359, 4631.41025641026, -12339.7435897436, 0, -33653846.1538462, 136538531.84463, 20192307.6923077, 89615.3846153846, 133333350.007674, 0, 24727.5641025641, -12.2802197802198, 14022435.8974359, -10801.2820512821, -6891.02564102564, 0, 33653846.1538462,
        -100961538.461538, -134615371.91456, -25384.6153846154, 0, -8974353.98494668, 1987.17948717949, -14022435.8974359, 9.80311355311355, -4439.10256410256, 0, -5769.23076923077, 30288461.5384615, 100961538.461538, -100961635.32903, 147884.615384615, -16826923.0769231, 15705135.7217355, -19054.4871794872, 14022435.8974359, -6.82234432234432, 15208.3333333333, 0, -13461.5384615385, 20192307.6923077, 20192307.6923077, 60576936.9374215, -23846.1538461538, -3365384.61538461, -29166677.9361877, 12804.4871794872, -14022435.8974359, -0.924908424908425, -1618.58974358974, 0, -12339.7435897436, 20192307.6923077, -20192307.6923077, 175000070.306168, -98653.8461538462, 0, 62820529.4948534, -18814.1025641026, 14022435.8974359, -12.2802197802198, 12387.8205128205, 0, -6891.02564102564, 30288461.5384615,
        31826.9230769231, -6939.10256410257, -28846109.8141922, 16586.5384615385, 3429.48717948718, 10256425.7290733, 11041.6666666667, -2099.35897435898, 24.9679487179487, 11217948.7179487, 20192307.6923077, -20192.3076923077, -119326.923076923, 137516.025641026, -38461829.0640129, 7932.69230769231, -17387.8205128205, 1282075.51151306, -11041.6666666667, 13541.6666666667, -22.948717948718, -11217948.7179487, 13461538.4615385, -47115.3846153846, -5769.23076923076, -39439.1025641026, -1923026.11955352, -8766.02564102564, 15464.7435897436, -14743620.0511401, 5657.05128205128, 641.02564102564, -3.65384615384615, -22435897.4358974, 13461538.4615385, -43189.1025641026, 93269.2307692308, -91137.8205128205, 69230964.9977587, 24727.5641025641, -18814.1025641026, 43589789.5710433, -12387.8205128205, 10801.2820512821, -33.4775641025641, 22435897.4358974, 20192307.6923077, -24118.5897435897,
        -60576936.9374215, -20192307.6923077, -23846.1538461538, -0.924908424908425, -14022435.8974359, 1618.58974358974, -29166677.9361877, -3365384.61538461, -12804.4871794872, 12339.7435897436, 0, 20192307.6923077, 100961635.32903, -100961538.461538, 147884.615384615, -6.82234432234432, 14022435.8974359, -15208.3333333333, 15705135.7217355, -16826923.0769231, 19054.4871794872, 13461.5384615385, 0, 20192307.6923077, 134615371.91456, 100961538.461538, -25384.6153846154, 9.80311355311355, -14022435.8974359, 4439.10256410256, -8974353.98494668, 0, -1987.17948717949, 5769.23076923077, 0, 30288461.5384615, -175000070.306168, 20192307.6923077, -98653.8461538462, -12.2802197802198, 14022435.8974359, -12387.8205128205, 62820529.4948534, 0, 18814.1025641026, 6891.02564102564, 0, 30288461.5384615,
        20192307.6923077, 69230755.3702707, 21442.3076923077, -14022435.8974359, -0.924908424908425, -4631.41025641026, 3365384.61538461, -37179498.4490083, 12131.4102564103, 0, 12339.7435897436, -33653846.1538462, -100961538.461538, 72115481.4828761, -130192.307692308, 14022435.8974359, -6.82234432234432, 12708.3333333333, -16826923.0769231, -9935889.91929011, -9855.76923076923, 0, 13461.5384615385, -16826923.0769231, 100961538.461538, -4807705.00851712, 19134.6153846154, -14022435.8974359, 9.80311355311355, -8782.05128205128, 0, 55128210.1176174, -14182.6923076923, 0, 5769.23076923077, 16826923.0769231, -20192307.6923077, -136538531.84463, 89615.3846153846, 14022435.8974359, -12.2802197802198, 10801.2820512821, 0, 133333350.007674, -24727.5641025641, 0, 6891.02564102564, 33653846.1538462,
        -39439.1025641026, -5769.23076923076, 1923026.11955352, -641.02564102564, -5657.05128205128, -3.65384615384615, -15464.7435897436, 8766.02564102564, -14743620.0511401, 13461538.4615385, -22435897.4358974, 43189.1025641026, 137516.025641026, -119326.923076923, 38461829.0640129, -13541.6666666667, 11041.6666666667, -22.948717948718, 17387.8205128205, -7932.69230769231, 1282075.51151306, 13461538.4615385, -11217948.7179487, 47115.3846153846, -6939.10256410257, 31826.9230769231, 28846109.8141922, 2099.35897435898, -11041.6666666667, 24.9679487179487, -3429.48717948718, -16586.5384615385, 10256425.7290733, 20192307.6923077, 11217948.7179487, 20192.3076923077, -91137.8205128205, 93269.2307692308, -69230964.9977587, -10801.2820512821, 12387.8205128205, -33.4775641025641, 18814.1025641026, -24727.5641025641, 43589789.5710433, 20192307.6923077, 22435897.4358974, 24118.5897435897,
        -8653.84615384615, 0, -60576923.0769231, 5769.23076923077, 0, 11217948.7179487, -1121.79487179487, 0, -13461538.4615385, 44873493.7503203, 841.346153846154, 20193.0088141026, 82692.3076923077, 0, 60576923.0769231, -13461.5384615385, 0, -11217948.7179487, 13461.5384615385, 0, 13461538.4615385, 22430866.2467431, 4206.73076923077, 13456.6306089744, -8653.84615384615, 0, 141346153.846154, 1121.79487179487, 0, -22435897.4358974, -5769.23076923077, 0, -20192307.6923077, 44865080.2887818, -841.346153846154, 11218.3133012821, -65384.6153846154, 0, -141346153.846154, -6891.02564102564, 0, 22435897.4358974, 6891.02564102564, 0, 20192307.6923077, 89753698.6564626, -4206.73076923077, 15708.9703525641,
        0, -8653.84615384615, -141346153.846154, 0, 5769.23076923077, -20192307.6923077, 0, -1121.79487179487, -22435897.4358974, -841.346153846154, 44865080.2887818, -11218.3133012821, 0, 82692.3076923077, -60576923.0769231, 0, -13461.5384615385, 13461538.4615385, 0, 13461.5384615385, -11217948.7179487, 4206.73076923077, 22430866.2467431, -13456.6306089744, 0, -8653.84615384615, 60576923.0769231, 0, 1121.79487179487, -13461538.4615385, 0, -5769.23076923077, 11217948.7179487, 841.346153846154, 44873493.7503203, -20193.0088141026, 0, -65384.6153846154, 141346153.846154, 0, -6891.02564102564, 20192307.6923077, 0, 6891.02564102564, 22435897.4358974, -4206.73076923077, 89753698.6564626, -15708.9703525641,
        -90865384.6153846, -212019230.769231, -30288.4615384615, 16826923.0769231, -30288461.5384615, 20192.3076923077, -20192307.6923077, -33653846.1538462, -3926.28205128205, 20193.5136217949, -11217.4719551282, 157050198.024649, 90865384.6153846, -90865384.6153846, 289423.076923077, -16826923.0769231, 20192307.6923077, -47115.3846153846, 20192307.6923077, -16826923.0769231, 47115.3846153846, 13456.9671474359, -13456.9671474359, 78523436.8387609, 212019230.769231, 90865384.6153846, -30288.4615384615, -33653846.1538462, -20192307.6923077, 3926.28205128205, -30288461.5384615, 16826923.0769231, -20192.3076923077, 11217.4719551282, -20193.5136217949, 157050198.024649, -212019230.769231, 212019230.769231, -228846.153846154, 33653846.1538462, 30288461.5384615, -24118.5897435897, 30288461.5384615, 33653846.1538462, 24118.5897435897, 15708.9703525641, -15708.9703525641, 314107081.05425;


    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation
    
    //Setup the test conditions
    ChVector<double> OriginalPos = m_nodeB->GetPos();
    m_nodeB->SetPos(ChVector<>(m_nodeB->GetPos().x(), m_nodeB->GetPos().y(), 0.001));

    ChVectorDynamic<double> InternalForceSmallDispNoVelNoGravity;
    InternalForceSmallDispNoVelNoGravity.resize(48);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceSmallDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_SmallDispNoVelNoDamping;
    JacobianK_SmallDispNoVelNoDamping.resize(48, 48);
    m_element->ComputeKRMmatricesGlobal(JacobianK_SmallDispNoVelNoDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_SmallDispNoVelNoDamping;
    JacobianR_SmallDispNoVelNoDamping.resize(48, 48);
    m_element->ComputeKRMmatricesGlobal(JacobianR_SmallDispNoVelNoDamping, 0, 1, 0);

    //Reset the element conditions back to its original values
    m_nodeB->SetPos(OriginalPos);


    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_SmallDispNoVelNoDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(48, 48);
    double MaxAbsError_JacR = JacobianR_SmallDispNoVelNoDamping.cwiseAbs().maxCoeff();

    for (auto i = 0; i < Expected_JacobianK_SmallDispNoVelNoDamping.rows(); i++) {
        for (auto j = 0; j < Expected_JacobianK_SmallDispNoVelNoDamping.cols(); j++) {
            if (std::abs(Expected_JacobianK_SmallDispNoVelNoDamping(i, j)) < small_terms_JacK) {
                double error = std::abs(JacobianK_SmallDispNoVelNoDamping(i, j) - Expected_JacobianK_SmallDispNoVelNoDamping(i, j));
                if (error > zeros_max_error_JacK)
                    zeros_max_error_JacK = error;
                percent_error_matrix_JacK(i, j) = 0.0;
            }
            else {
                double percent_error = std::abs((JacobianK_SmallDispNoVelNoDamping(i, j) - Expected_JacobianK_SmallDispNoVelNoDamping(i, j)) / Expected_JacobianK_SmallDispNoVelNoDamping(i, j));
                if (percent_error > max_percent_error_JacK)
                    max_percent_error_JacK = percent_error;
                percent_error_matrix_JacK(i, j) = percent_error;
            }
        }
    }


    if (msglvl >= 2) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Jacobian K Term - Small Displacement, No Velocity, No Damping = " << std::endl;
        std::cout << JacobianK_SmallDispNoVelNoDamping << std::endl;
        std::cout << "Expected Jacobian K Term - Small Displacement, No Velocity, No Damping = " << std::endl;
        std::cout << Expected_JacobianK_SmallDispNoVelNoDamping << std::endl;
        std::cout << "Percent Error Jacobian K Term - Small Displacement, No Velocity, No Damping = " << std::endl;
        std::cout << percent_error_matrix_JacK << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Jacobian K Term - Small Displacement, No Velocity, No Damping (Max Abs Error) = "
            << (JacobianK_SmallDispNoVelNoDamping - Expected_JacobianK_SmallDispNoVelNoDamping).cwiseAbs().maxCoeff() << std::endl;

        std::cout << "Jacobian K Term - Small Displacement, No Velocity, No Damping (Max Abs Error - Only Smaller Terms) = "
            << zeros_max_error_JacK;
        if (zeros_max_error_JacK / JacobianK_SmallDispNoVelNoDamping.cwiseAbs().maxCoeff() > 0.0001) {
            print_red(" - Test FAILED\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl;
        }

        std::cout << "Jacobian K Term - Small Displacement, No Velocity, No Damping (Max Abs % Error - Only Larger Terms) = "
            << max_percent_error_JacK * 100 << "%";
        if (max_percent_error_JacK > 0.01) {
            print_red(" - Test FAILED\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl;
        }
    }
    if (zeros_max_error_JacK / JacobianK_SmallDispNoVelNoDamping.cwiseAbs().maxCoeff() > 0.0001) {
        passed_tests = false;
    }
    if (max_percent_error_JacK > 0.01) {
        passed_tests = false;
    }

    if (msglvl >= 2) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Jacobian R Term - Small Displacement, No Velocity, No Damping = " << std::endl;
        std::cout << JacobianR_SmallDispNoVelNoDamping << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Jacobian R Term - Small Displacement, No Velocity, No Damping (Max Abs Error) = "
            << MaxAbsError_JacR;

        if (MaxAbsError_JacR > 0.01) {
            print_red(" - Test FAILED\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl;
        }
    }
    if (MaxAbsError_JacR > 0.01) {
        passed_tests = false;
    }

    return(passed_tests);

}

template <typename ElementVersion, typename MaterialVersion>
bool ANCFShellTest<ElementVersion, MaterialVersion>::JacobianNoDispNoVelWithDampingCheck(int msglvl) {
    // =============================================================================
    //  Check the Jacobian at No Displacement/Velocity - With Damping
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================

    ChMatrixNM<double, 48, 48> Expected_JacobianK_NoDispNoVelWithDamping;
    Expected_JacobianK_NoDispNoVelWithDamping <<
        1592307692.30769, 504807692.307692, 0, 136538461.538462, 20192307.6923077, 0, 175000000, -20192307.6923077, 0, 0, 0, -212019230.769231, -1107692307.69231, 100961538.461538, 0, 69230769.2307692, -20192307.6923077, 0, -134615384.615385, -100961538.461538, 0, 0, 0, -212019230.769231, -588461538.461538, -504807692.307692, 0, 72115384.6153846, 100961538.461538, 0, 100961538.461538, 100961538.461538, 0, 0, 0, -90865384.6153846, 103846153.846154, -100961538.461538, 0, 4807692.3076923, -100961538.461538, 0, -60576923.0769231, 20192307.6923077, 0, 0, 0, -90865384.6153846,
        504807692.307692, 1592307692.30769, 0, -20192307.6923077, 175000000, 0, 20192307.6923077, 136538461.538462, 0, 0, 0, -212019230.769231, -100961538.461538, 103846153.846154, 0, 20192307.6923077, -60576923.0769231, 0, -100961538.461538, 4807692.3076923, 0, 0, 0, -90865384.6153846, -504807692.307692, -588461538.461538, 0, 100961538.461538, 100961538.461538, 0, 100961538.461538, 72115384.6153846, 0, 0, 0, -90865384.6153846, 100961538.461538, -1107692307.69231, 0, -100961538.461538, -134615384.615385, 0, -20192307.6923077, 69230769.2307692, 0, 0, 0, -212019230.769231,
        0, 0, 707692307.692308, 0, 0, 69230769.2307692, 0, 0, 69230769.2307692, -141346153.846154, -141346153.846154, 0, 0, 0, -223076923.076923, 0, 0, 1923076.92307692, 0, 0, -28846153.8461538, -141346153.846154, -60576923.0769231, 0, 0, 0, -261538461.538462, 0, 0, 38461538.4615385, 0, 0, 38461538.4615385, -60576923.0769231, -60576923.0769231, 0, 0, 0, -223076923.076923, 0, 0, -28846153.8461538, 0, 0, 1923076.92307692, -60576923.0769231, -141346153.846154, 0,
        136538461.538462, -20192307.6923077, 0, 133333333.333333, 0, 0, 0, 14022435.8974359, 0, 0, 0, 33653846.1538462, -69230769.2307692, 20192307.6923077, 0, -37179487.1794872, -3365384.61538461, 0, 0, -14022435.8974359, 0, 0, 0, -33653846.1538462, -72115384.6153846, -100961538.461538, 0, -9935897.43589743, 16826923.0769231, 0, 0, 14022435.8974359, 0, 0, 0, -16826923.0769231, 4807692.3076923, 100961538.461538, 0, 55128205.1282051, 0, 0, 0, -14022435.8974359, 0, 0, 0, 16826923.0769231,
        20192307.6923077, 175000000, 0, 0, 62820512.8205128, 0, 14022435.8974359, 0, 0, 0, 0, -30288461.5384615, -20192307.6923077, 60576923.0769231, 0, 3365384.61538461, -29166666.6666667, 0, -14022435.8974359, 0, 0, 0, 0, -20192307.6923077, -100961538.461538, -100961538.461538, 0, 16826923.0769231, 15705128.2051282, 0, 14022435.8974359, 0, 0, 0, 0, -20192307.6923077, 100961538.461538, -134615384.615385, 0, 0, -8974358.97435897, 0, -14022435.8974359, 0, 0, 0, 0, -30288461.5384615,
        0, 0, 69230769.2307692, 0, 0, 43589743.5897436, 0, 0, 0, 22435897.4358974, -20192307.6923077, 0, 0, 0, -1923076.92307692, 0, 0, -14743589.7435897, 0, 0, 0, -22435897.4358974, -13461538.4615385, 0, 0, 0, -38461538.4615385, 0, 0, 1282051.28205128, 0, 0, 0, -11217948.7179487, -13461538.4615385, 0, 0, 0, -28846153.8461538, 0, 0, 10256410.2564103, 0, 0, 0, 11217948.7179487, -20192307.6923077, 0,
        175000000, 20192307.6923077, 0, 0, 14022435.8974359, 0, 62820512.8205128, 0, 0, 0, 0, -30288461.5384615, -134615384.615385, 100961538.461538, 0, 0, -14022435.8974359, 0, -8974358.97435897, 0, 0, 0, 0, -30288461.5384615, -100961538.461538, -100961538.461538, 0, 0, 14022435.8974359, 0, 15705128.2051282, 16826923.0769231, 0, 0, 0, -20192307.6923077, 60576923.0769231, -20192307.6923077, 0, 0, -14022435.8974359, 0, -29166666.6666667, 3365384.61538461, 0, 0, 0, -20192307.6923077,
        -20192307.6923077, 136538461.538462, 0, 14022435.8974359, 0, 0, 0, 133333333.333333, 0, 0, 0, 33653846.1538462, 100961538.461538, 4807692.3076923, 0, -14022435.8974359, 0, 0, 0, 55128205.1282051, 0, 0, 0, 16826923.0769231, -100961538.461538, -72115384.6153846, 0, 14022435.8974359, 0, 0, 16826923.0769231, -9935897.43589743, 0, 0, 0, -16826923.0769231, 20192307.6923077, -69230769.2307692, 0, -14022435.8974359, 0, 0, -3365384.61538461, -37179487.1794872, 0, 0, 0, -33653846.1538462,
        0, 0, 69230769.2307692, 0, 0, 0, 0, 0, 43589743.5897436, -20192307.6923077, 22435897.4358974, 0, 0, 0, -28846153.8461538, 0, 0, 0, 0, 0, 10256410.2564103, -20192307.6923077, 11217948.7179487, 0, 0, 0, -38461538.4615385, 0, 0, 0, 0, 0, 1282051.28205128, -13461538.4615385, -11217948.7179487, 0, 0, 0, -1923076.92307692, 0, 0, 0, 0, 0, -14743589.7435897, -13461538.4615385, -22435897.4358974, 0,
        0, 0, -141346153.846154, 0, 0, 22435897.4358974, 0, 0, -20192307.6923077, 89753685.8974359, 4206.73076923077, 0, 0, 0, 141346153.846154, 0, 0, -22435897.4358974, 0, 0, 20192307.6923077, 44865064.1025641, 841.346153846154, 0, 0, 0, 60576923.0769231, 0, 0, -11217948.7179487, 0, 0, -13461538.4615385, 22430849.3589744, -4206.73076923077, 0, 0, 0, -60576923.0769231, 0, 0, 11217948.7179487, 0, 0, 13461538.4615385, 44873477.5641026, -841.346153846154, 0,
        0, 0, -141346153.846154, 0, 0, -20192307.6923077, 0, 0, 22435897.4358974, 4206.73076923077, 89753685.8974359, 0, 0, 0, -60576923.0769231, 0, 0, 13461538.4615385, 0, 0, 11217948.7179487, -841.346153846154, 44873477.5641026, 0, 0, 0, 60576923.0769231, 0, 0, -13461538.4615385, 0, 0, -11217948.7179487, -4206.73076923077, 22430849.3589744, 0, 0, 0, 141346153.846154, 0, 0, 20192307.6923077, 0, 0, -22435897.4358974, 841.346153846154, 44865064.1025641, 0,
        -212019230.769231, -212019230.769231, 0, 33653846.1538462, -30288461.5384615, 0, -30288461.5384615, 33653846.1538462, 0, 0, 0, 314107051.282051, 212019230.769231, -90865384.6153846, 0, -33653846.1538462, 20192307.6923077, 0, 30288461.5384615, 16826923.0769231, 0, 0, 0, 157050160.25641, 90865384.6153846, 90865384.6153846, 0, -16826923.0769231, -20192307.6923077, 0, -20192307.6923077, -16826923.0769231, 0, 0, 0, 78523397.4358974, -90865384.6153846, 212019230.769231, 0, 16826923.0769231, 30288461.5384615, 0, 20192307.6923077, -33653846.1538462, 0, 0, 0, 157050160.25641,
        -1107692307.69231, -100961538.461538, 0, -69230769.2307692, -20192307.6923077, 0, -134615384.615385, 100961538.461538, 0, 0, 0, 212019230.769231, 1592307692.30769, -504807692.307692, 0, -136538461.538462, 20192307.6923077, 0, 175000000, 20192307.6923077, 0, 0, 0, 212019230.769231, 103846153.846154, 100961538.461538, 0, -4807692.3076923, -100961538.461538, 0, -60576923.0769231, -20192307.6923077, 0, 0, 0, 90865384.6153846, -588461538.461538, 504807692.307692, 0, -72115384.6153846, 100961538.461538, 0, 100961538.461538, -100961538.461538, 0, 0, 0, 90865384.6153846,
        100961538.461538, 103846153.846154, 0, 20192307.6923077, 60576923.0769231, 0, 100961538.461538, 4807692.3076923, 0, 0, 0, -90865384.6153846, -504807692.307692, 1592307692.30769, 0, -20192307.6923077, -175000000, 0, -20192307.6923077, 136538461.538462, 0, 0, 0, -212019230.769231, -100961538.461538, -1107692307.69231, 0, -100961538.461538, 134615384.615385, 0, 20192307.6923077, 69230769.2307692, 0, 0, 0, -212019230.769231, 504807692.307692, -588461538.461538, 0, 100961538.461538, -100961538.461538, 0, -100961538.461538, 72115384.6153846, 0, 0, 0, -90865384.6153846,
        0, 0, -223076923.076923, 0, 0, -1923076.92307692, 0, 0, -28846153.8461538, 141346153.846154, -60576923.0769231, 0, 0, 0, 707692307.692308, 0, 0, -69230769.2307692, 0, 0, 69230769.2307692, 141346153.846154, -141346153.846154, 0, 0, 0, -223076923.076923, 0, 0, 28846153.8461538, 0, 0, 1923076.92307692, 60576923.0769231, -141346153.846154, 0, 0, 0, -261538461.538462, 0, 0, -38461538.4615385, 0, 0, 38461538.4615385, 60576923.0769231, -60576923.0769231, 0,
        69230769.2307692, 20192307.6923077, 0, -37179487.1794872, 3365384.61538461, 0, 0, -14022435.8974359, 0, 0, 0, -33653846.1538462, -136538461.538462, -20192307.6923077, 0, 133333333.333333, 0, 0, 0, 14022435.8974359, 0, 0, 0, 33653846.1538462, -4807692.3076923, 100961538.461538, 0, 55128205.1282051, 0, 0, 0, -14022435.8974359, 0, 0, 0, 16826923.0769231, 72115384.6153846, -100961538.461538, 0, -9935897.43589743, -16826923.0769231, 0, 0, 14022435.8974359, 0, 0, 0, -16826923.0769231,
        -20192307.6923077, -60576923.0769231, 0, -3365384.61538461, -29166666.6666667, 0, -14022435.8974359, 0, 0, 0, 0, 20192307.6923077, 20192307.6923077, -175000000, 0, 0, 62820512.8205128, 0, 14022435.8974359, 0, 0, 0, 0, 30288461.5384615, 100961538.461538, 134615384.615385, 0, 0, -8974358.97435897, 0, -14022435.8974359, 0, 0, 0, 0, 30288461.5384615, -100961538.461538, 100961538.461538, 0, -16826923.0769231, 15705128.2051282, 0, 14022435.8974359, 0, 0, 0, 0, 20192307.6923077,
        0, 0, 1923076.92307692, 0, 0, -14743589.7435897, 0, 0, 0, -22435897.4358974, 13461538.4615385, 0, 0, 0, -69230769.2307692, 0, 0, 43589743.5897436, 0, 0, 0, 22435897.4358974, 20192307.6923077, 0, 0, 0, 28846153.8461538, 0, 0, 10256410.2564103, 0, 0, 0, 11217948.7179487, 20192307.6923077, 0, 0, 0, 38461538.4615385, 0, 0, 1282051.28205128, 0, 0, 0, -11217948.7179487, 13461538.4615385, 0,
        -134615384.615385, -100961538.461538, 0, 0, -14022435.8974359, 0, -8974358.97435897, 0, 0, 0, 0, 30288461.5384615, 175000000, -20192307.6923077, 0, 0, 14022435.8974359, 0, 62820512.8205128, 0, 0, 0, 0, 30288461.5384615, 60576923.0769231, 20192307.6923077, 0, 0, -14022435.8974359, 0, -29166666.6666667, -3365384.61538461, 0, 0, 0, 20192307.6923077, -100961538.461538, 100961538.461538, 0, 0, 14022435.8974359, 0, 15705128.2051282, -16826923.0769231, 0, 0, 0, 20192307.6923077,
        -100961538.461538, 4807692.3076923, 0, -14022435.8974359, 0, 0, 0, 55128205.1282051, 0, 0, 0, 16826923.0769231, 20192307.6923077, 136538461.538462, 0, 14022435.8974359, 0, 0, 0, 133333333.333333, 0, 0, 0, 33653846.1538462, -20192307.6923077, -69230769.2307692, 0, -14022435.8974359, 0, 0, 3365384.61538461, -37179487.1794872, 0, 0, 0, -33653846.1538462, 100961538.461538, -72115384.6153846, 0, 14022435.8974359, 0, 0, -16826923.0769231, -9935897.43589743, 0, 0, 0, -16826923.0769231,
        0, 0, -28846153.8461538, 0, 0, 0, 0, 0, 10256410.2564103, 20192307.6923077, 11217948.7179487, 0, 0, 0, 69230769.2307692, 0, 0, 0, 0, 0, 43589743.5897436, 20192307.6923077, 22435897.4358974, 0, 0, 0, -1923076.92307692, 0, 0, 0, 0, 0, -14743589.7435897, 13461538.4615385, -22435897.4358974, 0, 0, 0, -38461538.4615385, 0, 0, 0, 0, 0, 1282051.28205128, 13461538.4615385, -11217948.7179487, 0,
        0, 0, -141346153.846154, 0, 0, -22435897.4358974, 0, 0, -20192307.6923077, 44865064.1025641, -841.346153846154, 0, 0, 0, 141346153.846154, 0, 0, 22435897.4358974, 0, 0, 20192307.6923077, 89753685.8974359, -4206.73076923077, 0, 0, 0, 60576923.0769231, 0, 0, 11217948.7179487, 0, 0, -13461538.4615385, 44873477.5641026, 841.346153846154, 0, 0, 0, -60576923.0769231, 0, 0, -11217948.7179487, 0, 0, 13461538.4615385, 22430849.3589744, 4206.73076923077, 0,
        0, 0, -60576923.0769231, 0, 0, -13461538.4615385, 0, 0, 11217948.7179487, 841.346153846154, 44873477.5641026, 0, 0, 0, -141346153.846154, 0, 0, 20192307.6923077, 0, 0, 22435897.4358974, -4206.73076923077, 89753685.8974359, 0, 0, 0, 141346153.846154, 0, 0, -20192307.6923077, 0, 0, -22435897.4358974, -841.346153846154, 44865064.1025641, 0, 0, 0, 60576923.0769231, 0, 0, 13461538.4615385, 0, 0, -11217948.7179487, 4206.73076923077, 22430849.3589744, 0,
        -212019230.769231, -90865384.6153846, 0, -33653846.1538462, -20192307.6923077, 0, -30288461.5384615, 16826923.0769231, 0, 0, 0, 157050160.25641, 212019230.769231, -212019230.769231, 0, 33653846.1538462, 30288461.5384615, 0, 30288461.5384615, 33653846.1538462, 0, 0, 0, 314107051.282051, 90865384.6153846, 212019230.769231, 0, 16826923.0769231, -30288461.5384615, 0, -20192307.6923077, -33653846.1538462, 0, 0, 0, 157050160.25641, -90865384.6153846, 90865384.6153846, 0, -16826923.0769231, 20192307.6923077, 0, 20192307.6923077, -16826923.0769231, 0, 0, 0, 78523397.4358974,
        -588461538.461538, -504807692.307692, 0, -72115384.6153846, -100961538.461538, 0, -100961538.461538, -100961538.461538, 0, 0, 0, 90865384.6153846, 103846153.846154, -100961538.461538, 0, -4807692.3076923, 100961538.461538, 0, 60576923.0769231, -20192307.6923077, 0, 0, 0, 90865384.6153846, 1592307692.30769, 504807692.307692, 0, -136538461.538462, -20192307.6923077, 0, -175000000, 20192307.6923077, 0, 0, 0, 212019230.769231, -1107692307.69231, 100961538.461538, 0, -69230769.2307692, 20192307.6923077, 0, 134615384.615385, 100961538.461538, 0, 0, 0, 212019230.769231,
        -504807692.307692, -588461538.461538, 0, -100961538.461538, -100961538.461538, 0, -100961538.461538, -72115384.6153846, 0, 0, 0, 90865384.6153846, 100961538.461538, -1107692307.69231, 0, 100961538.461538, 134615384.615385, 0, 20192307.6923077, -69230769.2307692, 0, 0, 0, 212019230.769231, 504807692.307692, 1592307692.30769, 0, 20192307.6923077, -175000000, 0, -20192307.6923077, -136538461.538462, 0, 0, 0, 212019230.769231, -100961538.461538, 103846153.846154, 0, -20192307.6923077, 60576923.0769231, 0, 100961538.461538, -4807692.3076923, 0, 0, 0, 90865384.6153846,
        0, 0, -261538461.538462, 0, 0, -38461538.4615385, 0, 0, -38461538.4615385, 60576923.0769231, 60576923.0769231, 0, 0, 0, -223076923.076923, 0, 0, 28846153.8461538, 0, 0, -1923076.92307692, 60576923.0769231, 141346153.846154, 0, 0, 0, 707692307.692308, 0, 0, -69230769.2307692, 0, 0, -69230769.2307692, 141346153.846154, 141346153.846154, 0, 0, 0, -223076923.076923, 0, 0, -1923076.92307692, 0, 0, 28846153.8461538, 141346153.846154, 60576923.0769231, 0,
        72115384.6153846, 100961538.461538, 0, -9935897.43589743, 16826923.0769231, 0, 0, 14022435.8974359, 0, 0, 0, -16826923.0769231, -4807692.3076923, -100961538.461538, 0, 55128205.1282051, 0, 0, 0, -14022435.8974359, 0, 0, 0, 16826923.0769231, -136538461.538462, 20192307.6923077, 0, 133333333.333333, 0, 0, 0, 14022435.8974359, 0, 0, 0, 33653846.1538462, 69230769.2307692, -20192307.6923077, 0, -37179487.1794872, -3365384.61538461, 0, 0, -14022435.8974359, 0, 0, 0, -33653846.1538462,
        100961538.461538, 100961538.461538, 0, 16826923.0769231, 15705128.2051282, 0, 14022435.8974359, 0, 0, 0, 0, -20192307.6923077, -100961538.461538, 134615384.615385, 0, 0, -8974358.97435897, 0, -14022435.8974359, 0, 0, 0, 0, -30288461.5384615, -20192307.6923077, -175000000, 0, 0, 62820512.8205128, 0, 14022435.8974359, 0, 0, 0, 0, -30288461.5384615, 20192307.6923077, -60576923.0769231, 0, 3365384.61538461, -29166666.6666667, 0, -14022435.8974359, 0, 0, 0, 0, -20192307.6923077,
        0, 0, 38461538.4615385, 0, 0, 1282051.28205128, 0, 0, 0, -11217948.7179487, -13461538.4615385, 0, 0, 0, 28846153.8461538, 0, 0, 10256410.2564103, 0, 0, 0, 11217948.7179487, -20192307.6923077, 0, 0, 0, -69230769.2307692, 0, 0, 43589743.5897436, 0, 0, 0, 22435897.4358974, -20192307.6923077, 0, 0, 0, 1923076.92307692, 0, 0, -14743589.7435897, 0, 0, 0, -22435897.4358974, -13461538.4615385, 0,
        100961538.461538, 100961538.461538, 0, 0, 14022435.8974359, 0, 15705128.2051282, 16826923.0769231, 0, 0, 0, -20192307.6923077, -60576923.0769231, 20192307.6923077, 0, 0, -14022435.8974359, 0, -29166666.6666667, 3365384.61538461, 0, 0, 0, -20192307.6923077, -175000000, -20192307.6923077, 0, 0, 14022435.8974359, 0, 62820512.8205128, 0, 0, 0, 0, -30288461.5384615, 134615384.615385, -100961538.461538, 0, 0, -14022435.8974359, 0, -8974358.97435897, 0, 0, 0, 0, -30288461.5384615,
        100961538.461538, 72115384.6153846, 0, 14022435.8974359, 0, 0, 16826923.0769231, -9935897.43589743, 0, 0, 0, -16826923.0769231, -20192307.6923077, 69230769.2307692, 0, -14022435.8974359, 0, 0, -3365384.61538461, -37179487.1794872, 0, 0, 0, -33653846.1538462, 20192307.6923077, -136538461.538462, 0, 14022435.8974359, 0, 0, 0, 133333333.333333, 0, 0, 0, 33653846.1538462, -100961538.461538, -4807692.3076923, 0, -14022435.8974359, 0, 0, 0, 55128205.1282051, 0, 0, 0, 16826923.0769231,
        0, 0, 38461538.4615385, 0, 0, 0, 0, 0, 1282051.28205128, -13461538.4615385, -11217948.7179487, 0, 0, 0, 1923076.92307692, 0, 0, 0, 0, 0, -14743589.7435897, -13461538.4615385, -22435897.4358974, 0, 0, 0, -69230769.2307692, 0, 0, 0, 0, 0, 43589743.5897436, -20192307.6923077, 22435897.4358974, 0, 0, 0, 28846153.8461538, 0, 0, 0, 0, 0, 10256410.2564103, -20192307.6923077, 11217948.7179487, 0,
        0, 0, -60576923.0769231, 0, 0, -11217948.7179487, 0, 0, -13461538.4615385, 22430849.3589744, -4206.73076923077, 0, 0, 0, 60576923.0769231, 0, 0, 11217948.7179487, 0, 0, 13461538.4615385, 44873477.5641026, -841.346153846154, 0, 0, 0, 141346153.846154, 0, 0, 22435897.4358974, 0, 0, -20192307.6923077, 89753685.8974359, 4206.73076923077, 0, 0, 0, -141346153.846154, 0, 0, -22435897.4358974, 0, 0, 20192307.6923077, 44865064.1025641, 841.346153846154, 0,
        0, 0, -60576923.0769231, 0, 0, -13461538.4615385, 0, 0, -11217948.7179487, -4206.73076923077, 22430849.3589744, 0, 0, 0, -141346153.846154, 0, 0, 20192307.6923077, 0, 0, -22435897.4358974, 841.346153846154, 44865064.1025641, 0, 0, 0, 141346153.846154, 0, 0, -20192307.6923077, 0, 0, 22435897.4358974, 4206.73076923077, 89753685.8974359, 0, 0, 0, 60576923.0769231, 0, 0, 13461538.4615385, 0, 0, 11217948.7179487, -841.346153846154, 44873477.5641026, 0,
        -90865384.6153846, -90865384.6153846, 0, -16826923.0769231, -20192307.6923077, 0, -20192307.6923077, -16826923.0769231, 0, 0, 0, 78523397.4358974, 90865384.6153846, -212019230.769231, 0, 16826923.0769231, 30288461.5384615, 0, 20192307.6923077, -33653846.1538462, 0, 0, 0, 157050160.25641, 212019230.769231, 212019230.769231, 0, 33653846.1538462, -30288461.5384615, 0, -30288461.5384615, 33653846.1538462, 0, 0, 0, 314107051.282051, -212019230.769231, 90865384.6153846, 0, -33653846.1538462, 20192307.6923077, 0, 30288461.5384615, 16826923.0769231, 0, 0, 0, 157050160.25641,
        103846153.846154, 100961538.461538, 0, 4807692.3076923, 100961538.461538, 0, 60576923.0769231, 20192307.6923077, 0, 0, 0, -90865384.6153846, -588461538.461538, 504807692.307692, 0, 72115384.6153846, -100961538.461538, 0, -100961538.461538, 100961538.461538, 0, 0, 0, -90865384.6153846, -1107692307.69231, -100961538.461538, 0, 69230769.2307692, 20192307.6923077, 0, 134615384.615385, -100961538.461538, 0, 0, 0, -212019230.769231, 1592307692.30769, -504807692.307692, 0, 136538461.538462, -20192307.6923077, 0, -175000000, -20192307.6923077, 0, 0, 0, -212019230.769231,
        -100961538.461538, -1107692307.69231, 0, 100961538.461538, -134615384.615385, 0, -20192307.6923077, -69230769.2307692, 0, 0, 0, 212019230.769231, 504807692.307692, -588461538.461538, 0, -100961538.461538, 100961538.461538, 0, 100961538.461538, -72115384.6153846, 0, 0, 0, 90865384.6153846, 100961538.461538, 103846153.846154, 0, -20192307.6923077, -60576923.0769231, 0, -100961538.461538, -4807692.3076923, 0, 0, 0, 90865384.6153846, -504807692.307692, 1592307692.30769, 0, 20192307.6923077, 175000000, 0, 20192307.6923077, -136538461.538462, 0, 0, 0, 212019230.769231,
        0, 0, -223076923.076923, 0, 0, -28846153.8461538, 0, 0, -1923076.92307692, -60576923.0769231, 141346153.846154, 0, 0, 0, -261538461.538462, 0, 0, 38461538.4615385, 0, 0, -38461538.4615385, -60576923.0769231, 60576923.0769231, 0, 0, 0, -223076923.076923, 0, 0, 1923076.92307692, 0, 0, 28846153.8461538, -141346153.846154, 60576923.0769231, 0, 0, 0, 707692307.692308, 0, 0, 69230769.2307692, 0, 0, -69230769.2307692, -141346153.846154, 141346153.846154, 0,
        4807692.3076923, -100961538.461538, 0, 55128205.1282051, 0, 0, 0, -14022435.8974359, 0, 0, 0, 16826923.0769231, -72115384.6153846, 100961538.461538, 0, -9935897.43589743, -16826923.0769231, 0, 0, 14022435.8974359, 0, 0, 0, -16826923.0769231, -69230769.2307692, -20192307.6923077, 0, -37179487.1794872, 3365384.61538461, 0, 0, -14022435.8974359, 0, 0, 0, -33653846.1538462, 136538461.538462, 20192307.6923077, 0, 133333333.333333, 0, 0, 0, 14022435.8974359, 0, 0, 0, 33653846.1538462,
        -100961538.461538, -134615384.615385, 0, 0, -8974358.97435897, 0, -14022435.8974359, 0, 0, 0, 0, 30288461.5384615, 100961538.461538, -100961538.461538, 0, -16826923.0769231, 15705128.2051282, 0, 14022435.8974359, 0, 0, 0, 0, 20192307.6923077, 20192307.6923077, 60576923.0769231, 0, -3365384.61538461, -29166666.6666667, 0, -14022435.8974359, 0, 0, 0, 0, 20192307.6923077, -20192307.6923077, 175000000, 0, 0, 62820512.8205128, 0, 14022435.8974359, 0, 0, 0, 0, 30288461.5384615,
        0, 0, -28846153.8461538, 0, 0, 10256410.2564103, 0, 0, 0, 11217948.7179487, 20192307.6923077, 0, 0, 0, -38461538.4615385, 0, 0, 1282051.28205128, 0, 0, 0, -11217948.7179487, 13461538.4615385, 0, 0, 0, -1923076.92307692, 0, 0, -14743589.7435897, 0, 0, 0, -22435897.4358974, 13461538.4615385, 0, 0, 0, 69230769.2307692, 0, 0, 43589743.5897436, 0, 0, 0, 22435897.4358974, 20192307.6923077, 0,
        -60576923.0769231, -20192307.6923077, 0, 0, -14022435.8974359, 0, -29166666.6666667, -3365384.61538461, 0, 0, 0, 20192307.6923077, 100961538.461538, -100961538.461538, 0, 0, 14022435.8974359, 0, 15705128.2051282, -16826923.0769231, 0, 0, 0, 20192307.6923077, 134615384.615385, 100961538.461538, 0, 0, -14022435.8974359, 0, -8974358.97435897, 0, 0, 0, 0, 30288461.5384615, -175000000, 20192307.6923077, 0, 0, 14022435.8974359, 0, 62820512.8205128, 0, 0, 0, 0, 30288461.5384615,
        20192307.6923077, 69230769.2307692, 0, -14022435.8974359, 0, 0, 3365384.61538461, -37179487.1794872, 0, 0, 0, -33653846.1538462, -100961538.461538, 72115384.6153846, 0, 14022435.8974359, 0, 0, -16826923.0769231, -9935897.43589743, 0, 0, 0, -16826923.0769231, 100961538.461538, -4807692.3076923, 0, -14022435.8974359, 0, 0, 0, 55128205.1282051, 0, 0, 0, 16826923.0769231, -20192307.6923077, -136538461.538462, 0, 14022435.8974359, 0, 0, 0, 133333333.333333, 0, 0, 0, 33653846.1538462,
        0, 0, 1923076.92307692, 0, 0, 0, 0, 0, -14743589.7435897, 13461538.4615385, -22435897.4358974, 0, 0, 0, 38461538.4615385, 0, 0, 0, 0, 0, 1282051.28205128, 13461538.4615385, -11217948.7179487, 0, 0, 0, 28846153.8461538, 0, 0, 0, 0, 0, 10256410.2564103, 20192307.6923077, 11217948.7179487, 0, 0, 0, -69230769.2307692, 0, 0, 0, 0, 0, 43589743.5897436, 20192307.6923077, 22435897.4358974, 0,
        0, 0, -60576923.0769231, 0, 0, 11217948.7179487, 0, 0, -13461538.4615385, 44873477.5641026, 841.346153846154, 0, 0, 0, 60576923.0769231, 0, 0, -11217948.7179487, 0, 0, 13461538.4615385, 22430849.3589744, 4206.73076923077, 0, 0, 0, 141346153.846154, 0, 0, -22435897.4358974, 0, 0, -20192307.6923077, 44865064.1025641, -841.346153846154, 0, 0, 0, -141346153.846154, 0, 0, 22435897.4358974, 0, 0, 20192307.6923077, 89753685.8974359, -4206.73076923077, 0,
        0, 0, -141346153.846154, 0, 0, -20192307.6923077, 0, 0, -22435897.4358974, -841.346153846154, 44865064.1025641, 0, 0, 0, -60576923.0769231, 0, 0, 13461538.4615385, 0, 0, -11217948.7179487, 4206.73076923077, 22430849.3589744, 0, 0, 0, 60576923.0769231, 0, 0, -13461538.4615385, 0, 0, 11217948.7179487, 841.346153846154, 44873477.5641026, 0, 0, 0, 141346153.846154, 0, 0, 20192307.6923077, 0, 0, 22435897.4358974, -4206.73076923077, 89753685.8974359, 0,
        -90865384.6153846, -212019230.769231, 0, 16826923.0769231, -30288461.5384615, 0, -20192307.6923077, -33653846.1538462, 0, 0, 0, 157050160.25641, 90865384.6153846, -90865384.6153846, 0, -16826923.0769231, 20192307.6923077, 0, 20192307.6923077, -16826923.0769231, 0, 0, 0, 78523397.4358974, 212019230.769231, 90865384.6153846, 0, -33653846.1538462, -20192307.6923077, 0, -30288461.5384615, 16826923.0769231, 0, 0, 0, 157050160.25641, -212019230.769231, 212019230.769231, 0, 33653846.1538462, 30288461.5384615, 0, 30288461.5384615, 33653846.1538462, 0, 0, 0, 314107051.282051;

    ChMatrixNM<double, 48, 48> Expected_JacobianR_NoDispNoVelWithDamping;
    Expected_JacobianR_NoDispNoVelWithDamping <<
        15923076.9230769, 5048076.92307692, 0, 1365384.61538462, 201923.076923077, 0, 1750000, -201923.076923077, 0, 0, 0, -2120192.30769231, -11076923.0769231, 1009615.38461538, 0, 692307.692307692, -201923.076923077, 0, -1346153.84615385, -1009615.38461538, 0, 0, 0, -2120192.30769231, -5884615.38461538, -5048076.92307692, 0, 721153.846153846, 1009615.38461538, 0, 1009615.38461538, 1009615.38461538, 0, 0, 0, -908653.846153846, 1038461.53846154, -1009615.38461538, 0, 48076.923076923, -1009615.38461538, 0, -605769.230769231, 201923.076923077, 0, 0, 0, -908653.846153846,
        5048076.92307692, 15923076.9230769, 0, -201923.076923077, 1750000, 0, 201923.076923077, 1365384.61538462, 0, 0, 0, -2120192.30769231, -1009615.38461538, 1038461.53846154, 0, 201923.076923077, -605769.230769231, 0, -1009615.38461538, 48076.923076923, 0, 0, 0, -908653.846153846, -5048076.92307692, -5884615.38461538, 0, 1009615.38461538, 1009615.38461538, 0, 1009615.38461538, 721153.846153846, 0, 0, 0, -908653.846153846, 1009615.38461538, -11076923.0769231, 0, -1009615.38461538, -1346153.84615385, 0, -201923.076923077, 692307.692307692, 0, 0, 0, -2120192.30769231,
        0, 0, 7076923.07692308, 0, 0, 692307.692307692, 0, 0, 692307.692307692, -1413461.53846154, -1413461.53846154, 0, 0, 0, -2230769.23076923, 0, 0, 19230.7692307692, 0, 0, -288461.538461538, -1413461.53846154, -605769.230769231, 0, 0, 0, -2615384.61538462, 0, 0, 384615.384615385, 0, 0, 384615.384615385, -605769.230769231, -605769.230769231, 0, 0, 0, -2230769.23076923, 0, 0, -288461.538461538, 0, 0, 19230.7692307692, -605769.230769231, -1413461.53846154, 0,
        1365384.61538462, -201923.076923077, 0, 1333333.33333333, 0, 0, 0, 140224.358974359, 0, 0, 0, 336538.461538462, -692307.692307692, 201923.076923077, 0, -371794.871794872, -33653.8461538461, 0, 0, -140224.358974359, 0, 0, 0, -336538.461538462, -721153.846153846, -1009615.38461538, 0, -99358.9743589743, 168269.230769231, 0, 0, 140224.358974359, 0, 0, 0, -168269.230769231, 48076.923076923, 1009615.38461538, 0, 551282.051282051, 0, 0, 0, -140224.358974359, 0, 0, 0, 168269.230769231,
        201923.076923077, 1750000, 0, 0, 628205.128205128, 0, 140224.358974359, 0, 0, 0, 0, -302884.615384615, -201923.076923077, 605769.230769231, 0, 33653.8461538461, -291666.666666667, 0, -140224.358974359, 0, 0, 0, 0, -201923.076923077, -1009615.38461538, -1009615.38461538, 0, 168269.230769231, 157051.282051282, 0, 140224.358974359, 0, 0, 0, 0, -201923.076923077, 1009615.38461538, -1346153.84615385, 0, 0, -89743.5897435897, 0, -140224.358974359, 0, 0, 0, 0, -302884.615384615,
        0, 0, 692307.692307692, 0, 0, 435897.435897436, 0, 0, 0, 224358.974358974, -201923.076923077, 0, 0, 0, -19230.7692307692, 0, 0, -147435.897435897, 0, 0, 0, -224358.974358974, -134615.384615385, 0, 0, 0, -384615.384615385, 0, 0, 12820.5128205128, 0, 0, 0, -112179.487179487, -134615.384615385, 0, 0, 0, -288461.538461538, 0, 0, 102564.102564103, 0, 0, 0, 112179.487179487, -201923.076923077, 0,
        1750000, 201923.076923077, 0, 0, 140224.358974359, 0, 628205.128205128, 0, 0, 0, 0, -302884.615384615, -1346153.84615385, 1009615.38461538, 0, 0, -140224.358974359, 0, -89743.5897435897, 0, 0, 0, 0, -302884.615384615, -1009615.38461538, -1009615.38461538, 0, 0, 140224.358974359, 0, 157051.282051282, 168269.230769231, 0, 0, 0, -201923.076923077, 605769.230769231, -201923.076923077, 0, 0, -140224.358974359, 0, -291666.666666667, 33653.8461538461, 0, 0, 0, -201923.076923077,
        -201923.076923077, 1365384.61538462, 0, 140224.358974359, 0, 0, 0, 1333333.33333333, 0, 0, 0, 336538.461538462, 1009615.38461538, 48076.923076923, 0, -140224.358974359, 0, 0, 0, 551282.051282051, 0, 0, 0, 168269.230769231, -1009615.38461538, -721153.846153846, 0, 140224.358974359, 0, 0, 168269.230769231, -99358.9743589743, 0, 0, 0, -168269.230769231, 201923.076923077, -692307.692307692, 0, -140224.358974359, 0, 0, -33653.8461538461, -371794.871794872, 0, 0, 0, -336538.461538462,
        0, 0, 692307.692307692, 0, 0, 0, 0, 0, 435897.435897436, -201923.076923077, 224358.974358974, 0, 0, 0, -288461.538461538, 0, 0, 0, 0, 0, 102564.102564103, -201923.076923077, 112179.487179487, 0, 0, 0, -384615.384615385, 0, 0, 0, 0, 0, 12820.5128205128, -134615.384615385, -112179.487179487, 0, 0, 0, -19230.7692307692, 0, 0, 0, 0, 0, -147435.897435897, -134615.384615385, -224358.974358974, 0,
        0, 0, -1413461.53846154, 0, 0, 224358.974358974, 0, 0, -201923.076923077, 897536.858974359, 42.0673076923077, 0, 0, 0, 1413461.53846154, 0, 0, -224358.974358974, 0, 0, 201923.076923077, 448650.641025641, 8.41346153846154, 0, 0, 0, 605769.230769231, 0, 0, -112179.487179487, 0, 0, -134615.384615385, 224308.493589744, -42.0673076923077, 0, 0, 0, -605769.230769231, 0, 0, 112179.487179487, 0, 0, 134615.384615385, 448734.775641026, -8.41346153846154, 0,
        0, 0, -1413461.53846154, 0, 0, -201923.076923077, 0, 0, 224358.974358974, 42.0673076923077, 897536.858974359, 0, 0, 0, -605769.230769231, 0, 0, 134615.384615385, 0, 0, 112179.487179487, -8.41346153846154, 448734.775641026, 0, 0, 0, 605769.230769231, 0, 0, -134615.384615385, 0, 0, -112179.487179487, -42.0673076923077, 224308.493589744, 0, 0, 0, 1413461.53846154, 0, 0, 201923.076923077, 0, 0, -224358.974358974, 8.41346153846154, 448650.641025641, 0,
        -2120192.30769231, -2120192.30769231, 0, 336538.461538462, -302884.615384615, 0, -302884.615384615, 336538.461538462, 0, 0, 0, 3141070.51282051, 2120192.30769231, -908653.846153846, 0, -336538.461538462, 201923.076923077, 0, 302884.615384615, 168269.230769231, 0, 0, 0, 1570501.6025641, 908653.846153846, 908653.846153846, 0, -168269.230769231, -201923.076923077, 0, -201923.076923077, -168269.230769231, 0, 0, 0, 785233.974358974, -908653.846153846, 2120192.30769231, 0, 168269.230769231, 302884.615384615, 0, 201923.076923077, -336538.461538462, 0, 0, 0, 1570501.6025641,
        -11076923.0769231, -1009615.38461538, 0, -692307.692307692, -201923.076923077, 0, -1346153.84615385, 1009615.38461538, 0, 0, 0, 2120192.30769231, 15923076.9230769, -5048076.92307692, 0, -1365384.61538462, 201923.076923077, 0, 1750000, 201923.076923077, 0, 0, 0, 2120192.30769231, 1038461.53846154, 1009615.38461538, 0, -48076.923076923, -1009615.38461538, 0, -605769.230769231, -201923.076923077, 0, 0, 0, 908653.846153846, -5884615.38461538, 5048076.92307692, 0, -721153.846153846, 1009615.38461538, 0, 1009615.38461538, -1009615.38461538, 0, 0, 0, 908653.846153846,
        1009615.38461538, 1038461.53846154, 0, 201923.076923077, 605769.230769231, 0, 1009615.38461538, 48076.923076923, 0, 0, 0, -908653.846153846, -5048076.92307692, 15923076.9230769, 0, -201923.076923077, -1750000, 0, -201923.076923077, 1365384.61538462, 0, 0, 0, -2120192.30769231, -1009615.38461538, -11076923.0769231, 0, -1009615.38461538, 1346153.84615385, 0, 201923.076923077, 692307.692307692, 0, 0, 0, -2120192.30769231, 5048076.92307692, -5884615.38461538, 0, 1009615.38461538, -1009615.38461538, 0, -1009615.38461538, 721153.846153846, 0, 0, 0, -908653.846153846,
        0, 0, -2230769.23076923, 0, 0, -19230.7692307692, 0, 0, -288461.538461538, 1413461.53846154, -605769.230769231, 0, 0, 0, 7076923.07692308, 0, 0, -692307.692307692, 0, 0, 692307.692307692, 1413461.53846154, -1413461.53846154, 0, 0, 0, -2230769.23076923, 0, 0, 288461.538461538, 0, 0, 19230.7692307692, 605769.230769231, -1413461.53846154, 0, 0, 0, -2615384.61538462, 0, 0, -384615.384615385, 0, 0, 384615.384615385, 605769.230769231, -605769.230769231, 0,
        692307.692307692, 201923.076923077, 0, -371794.871794872, 33653.8461538461, 0, 0, -140224.358974359, 0, 0, 0, -336538.461538462, -1365384.61538462, -201923.076923077, 0, 1333333.33333333, 0, 0, 0, 140224.358974359, 0, 0, 0, 336538.461538462, -48076.923076923, 1009615.38461538, 0, 551282.051282051, 0, 0, 0, -140224.358974359, 0, 0, 0, 168269.230769231, 721153.846153846, -1009615.38461538, 0, -99358.9743589743, -168269.230769231, 0, 0, 140224.358974359, 0, 0, 0, -168269.230769231,
        -201923.076923077, -605769.230769231, 0, -33653.8461538461, -291666.666666667, 0, -140224.358974359, 0, 0, 0, 0, 201923.076923077, 201923.076923077, -1750000, 0, 0, 628205.128205128, 0, 140224.358974359, 0, 0, 0, 0, 302884.615384615, 1009615.38461538, 1346153.84615385, 0, 0, -89743.5897435897, 0, -140224.358974359, 0, 0, 0, 0, 302884.615384615, -1009615.38461538, 1009615.38461538, 0, -168269.230769231, 157051.282051282, 0, 140224.358974359, 0, 0, 0, 0, 201923.076923077,
        0, 0, 19230.7692307692, 0, 0, -147435.897435897, 0, 0, 0, -224358.974358974, 134615.384615385, 0, 0, 0, -692307.692307692, 0, 0, 435897.435897436, 0, 0, 0, 224358.974358974, 201923.076923077, 0, 0, 0, 288461.538461538, 0, 0, 102564.102564103, 0, 0, 0, 112179.487179487, 201923.076923077, 0, 0, 0, 384615.384615385, 0, 0, 12820.5128205128, 0, 0, 0, -112179.487179487, 134615.384615385, 0,
        -1346153.84615385, -1009615.38461538, 0, 0, -140224.358974359, 0, -89743.5897435897, 0, 0, 0, 0, 302884.615384615, 1750000, -201923.076923077, 0, 0, 140224.358974359, 0, 628205.128205128, 0, 0, 0, 0, 302884.615384615, 605769.230769231, 201923.076923077, 0, 0, -140224.358974359, 0, -291666.666666667, -33653.8461538461, 0, 0, 0, 201923.076923077, -1009615.38461538, 1009615.38461538, 0, 0, 140224.358974359, 0, 157051.282051282, -168269.230769231, 0, 0, 0, 201923.076923077,
        -1009615.38461538, 48076.923076923, 0, -140224.358974359, 0, 0, 0, 551282.051282051, 0, 0, 0, 168269.230769231, 201923.076923077, 1365384.61538462, 0, 140224.358974359, 0, 0, 0, 1333333.33333333, 0, 0, 0, 336538.461538462, -201923.076923077, -692307.692307692, 0, -140224.358974359, 0, 0, 33653.8461538461, -371794.871794872, 0, 0, 0, -336538.461538462, 1009615.38461538, -721153.846153846, 0, 140224.358974359, 0, 0, -168269.230769231, -99358.9743589743, 0, 0, 0, -168269.230769231,
        0, 0, -288461.538461538, 0, 0, 0, 0, 0, 102564.102564103, 201923.076923077, 112179.487179487, 0, 0, 0, 692307.692307692, 0, 0, 0, 0, 0, 435897.435897436, 201923.076923077, 224358.974358974, 0, 0, 0, -19230.7692307692, 0, 0, 0, 0, 0, -147435.897435897, 134615.384615385, -224358.974358974, 0, 0, 0, -384615.384615385, 0, 0, 0, 0, 0, 12820.5128205128, 134615.384615385, -112179.487179487, 0,
        0, 0, -1413461.53846154, 0, 0, -224358.974358974, 0, 0, -201923.076923077, 448650.641025641, -8.41346153846154, 0, 0, 0, 1413461.53846154, 0, 0, 224358.974358974, 0, 0, 201923.076923077, 897536.858974359, -42.0673076923077, 0, 0, 0, 605769.230769231, 0, 0, 112179.487179487, 0, 0, -134615.384615385, 448734.775641026, 8.41346153846154, 0, 0, 0, -605769.230769231, 0, 0, -112179.487179487, 0, 0, 134615.384615385, 224308.493589744, 42.0673076923077, 0,
        0, 0, -605769.230769231, 0, 0, -134615.384615385, 0, 0, 112179.487179487, 8.41346153846154, 448734.775641026, 0, 0, 0, -1413461.53846154, 0, 0, 201923.076923077, 0, 0, 224358.974358974, -42.0673076923077, 897536.858974359, 0, 0, 0, 1413461.53846154, 0, 0, -201923.076923077, 0, 0, -224358.974358974, -8.41346153846154, 448650.641025641, 0, 0, 0, 605769.230769231, 0, 0, 134615.384615385, 0, 0, -112179.487179487, 42.0673076923077, 224308.493589744, 0,
        -2120192.30769231, -908653.846153846, 0, -336538.461538462, -201923.076923077, 0, -302884.615384615, 168269.230769231, 0, 0, 0, 1570501.6025641, 2120192.30769231, -2120192.30769231, 0, 336538.461538462, 302884.615384615, 0, 302884.615384615, 336538.461538462, 0, 0, 0, 3141070.51282051, 908653.846153846, 2120192.30769231, 0, 168269.230769231, -302884.615384615, 0, -201923.076923077, -336538.461538462, 0, 0, 0, 1570501.6025641, -908653.846153846, 908653.846153846, 0, -168269.230769231, 201923.076923077, 0, 201923.076923077, -168269.230769231, 0, 0, 0, 785233.974358974,
        -5884615.38461538, -5048076.92307692, 0, -721153.846153846, -1009615.38461538, 0, -1009615.38461538, -1009615.38461538, 0, 0, 0, 908653.846153846, 1038461.53846154, -1009615.38461538, 0, -48076.923076923, 1009615.38461538, 0, 605769.230769231, -201923.076923077, 0, 0, 0, 908653.846153846, 15923076.9230769, 5048076.92307692, 0, -1365384.61538462, -201923.076923077, 0, -1750000, 201923.076923077, 0, 0, 0, 2120192.30769231, -11076923.0769231, 1009615.38461538, 0, -692307.692307692, 201923.076923077, 0, 1346153.84615385, 1009615.38461538, 0, 0, 0, 2120192.30769231,
        -5048076.92307692, -5884615.38461538, 0, -1009615.38461538, -1009615.38461538, 0, -1009615.38461538, -721153.846153846, 0, 0, 0, 908653.846153846, 1009615.38461538, -11076923.0769231, 0, 1009615.38461538, 1346153.84615385, 0, 201923.076923077, -692307.692307692, 0, 0, 0, 2120192.30769231, 5048076.92307692, 15923076.9230769, 0, 201923.076923077, -1750000, 0, -201923.076923077, -1365384.61538462, 0, 0, 0, 2120192.30769231, -1009615.38461538, 1038461.53846154, 0, -201923.076923077, 605769.230769231, 0, 1009615.38461538, -48076.923076923, 0, 0, 0, 908653.846153846,
        0, 0, -2615384.61538462, 0, 0, -384615.384615385, 0, 0, -384615.384615385, 605769.230769231, 605769.230769231, 0, 0, 0, -2230769.23076923, 0, 0, 288461.538461538, 0, 0, -19230.7692307692, 605769.230769231, 1413461.53846154, 0, 0, 0, 7076923.07692308, 0, 0, -692307.692307692, 0, 0, -692307.692307692, 1413461.53846154, 1413461.53846154, 0, 0, 0, -2230769.23076923, 0, 0, -19230.7692307692, 0, 0, 288461.538461538, 1413461.53846154, 605769.230769231, 0,
        721153.846153846, 1009615.38461538, 0, -99358.9743589743, 168269.230769231, 0, 0, 140224.358974359, 0, 0, 0, -168269.230769231, -48076.923076923, -1009615.38461538, 0, 551282.051282051, 0, 0, 0, -140224.358974359, 0, 0, 0, 168269.230769231, -1365384.61538462, 201923.076923077, 0, 1333333.33333333, 0, 0, 0, 140224.358974359, 0, 0, 0, 336538.461538462, 692307.692307692, -201923.076923077, 0, -371794.871794872, -33653.8461538461, 0, 0, -140224.358974359, 0, 0, 0, -336538.461538462,
        1009615.38461538, 1009615.38461538, 0, 168269.230769231, 157051.282051282, 0, 140224.358974359, 0, 0, 0, 0, -201923.076923077, -1009615.38461538, 1346153.84615385, 0, 0, -89743.5897435897, 0, -140224.358974359, 0, 0, 0, 0, -302884.615384615, -201923.076923077, -1750000, 0, 0, 628205.128205128, 0, 140224.358974359, 0, 0, 0, 0, -302884.615384615, 201923.076923077, -605769.230769231, 0, 33653.8461538461, -291666.666666667, 0, -140224.358974359, 0, 0, 0, 0, -201923.076923077,
        0, 0, 384615.384615385, 0, 0, 12820.5128205128, 0, 0, 0, -112179.487179487, -134615.384615385, 0, 0, 0, 288461.538461538, 0, 0, 102564.102564103, 0, 0, 0, 112179.487179487, -201923.076923077, 0, 0, 0, -692307.692307692, 0, 0, 435897.435897436, 0, 0, 0, 224358.974358974, -201923.076923077, 0, 0, 0, 19230.7692307692, 0, 0, -147435.897435897, 0, 0, 0, -224358.974358974, -134615.384615385, 0,
        1009615.38461538, 1009615.38461538, 0, 0, 140224.358974359, 0, 157051.282051282, 168269.230769231, 0, 0, 0, -201923.076923077, -605769.230769231, 201923.076923077, 0, 0, -140224.358974359, 0, -291666.666666667, 33653.8461538461, 0, 0, 0, -201923.076923077, -1750000, -201923.076923077, 0, 0, 140224.358974359, 0, 628205.128205128, 0, 0, 0, 0, -302884.615384615, 1346153.84615385, -1009615.38461538, 0, 0, -140224.358974359, 0, -89743.5897435897, 0, 0, 0, 0, -302884.615384615,
        1009615.38461538, 721153.846153846, 0, 140224.358974359, 0, 0, 168269.230769231, -99358.9743589743, 0, 0, 0, -168269.230769231, -201923.076923077, 692307.692307692, 0, -140224.358974359, 0, 0, -33653.8461538461, -371794.871794872, 0, 0, 0, -336538.461538462, 201923.076923077, -1365384.61538462, 0, 140224.358974359, 0, 0, 0, 1333333.33333333, 0, 0, 0, 336538.461538462, -1009615.38461538, -48076.923076923, 0, -140224.358974359, 0, 0, 0, 551282.051282051, 0, 0, 0, 168269.230769231,
        0, 0, 384615.384615385, 0, 0, 0, 0, 0, 12820.5128205128, -134615.384615385, -112179.487179487, 0, 0, 0, 19230.7692307692, 0, 0, 0, 0, 0, -147435.897435897, -134615.384615385, -224358.974358974, 0, 0, 0, -692307.692307692, 0, 0, 0, 0, 0, 435897.435897436, -201923.076923077, 224358.974358974, 0, 0, 0, 288461.538461538, 0, 0, 0, 0, 0, 102564.102564103, -201923.076923077, 112179.487179487, 0,
        0, 0, -605769.230769231, 0, 0, -112179.487179487, 0, 0, -134615.384615385, 224308.493589744, -42.0673076923077, 0, 0, 0, 605769.230769231, 0, 0, 112179.487179487, 0, 0, 134615.384615385, 448734.775641026, -8.41346153846154, 0, 0, 0, 1413461.53846154, 0, 0, 224358.974358974, 0, 0, -201923.076923077, 897536.858974359, 42.0673076923077, 0, 0, 0, -1413461.53846154, 0, 0, -224358.974358974, 0, 0, 201923.076923077, 448650.641025641, 8.41346153846154, 0,
        0, 0, -605769.230769231, 0, 0, -134615.384615385, 0, 0, -112179.487179487, -42.0673076923077, 224308.493589744, 0, 0, 0, -1413461.53846154, 0, 0, 201923.076923077, 0, 0, -224358.974358974, 8.41346153846154, 448650.641025641, 0, 0, 0, 1413461.53846154, 0, 0, -201923.076923077, 0, 0, 224358.974358974, 42.0673076923077, 897536.858974359, 0, 0, 0, 605769.230769231, 0, 0, 134615.384615385, 0, 0, 112179.487179487, -8.41346153846154, 448734.775641026, 0,
        -908653.846153846, -908653.846153846, 0, -168269.230769231, -201923.076923077, 0, -201923.076923077, -168269.230769231, 0, 0, 0, 785233.974358974, 908653.846153846, -2120192.30769231, 0, 168269.230769231, 302884.615384615, 0, 201923.076923077, -336538.461538462, 0, 0, 0, 1570501.6025641, 2120192.30769231, 2120192.30769231, 0, 336538.461538462, -302884.615384615, 0, -302884.615384615, 336538.461538462, 0, 0, 0, 3141070.51282051, -2120192.30769231, 908653.846153846, 0, -336538.461538462, 201923.076923077, 0, 302884.615384615, 168269.230769231, 0, 0, 0, 1570501.6025641,
        1038461.53846154, 1009615.38461538, 0, 48076.923076923, 1009615.38461538, 0, 605769.230769231, 201923.076923077, 0, 0, 0, -908653.846153846, -5884615.38461538, 5048076.92307692, 0, 721153.846153846, -1009615.38461538, 0, -1009615.38461538, 1009615.38461538, 0, 0, 0, -908653.846153846, -11076923.0769231, -1009615.38461538, 0, 692307.692307692, 201923.076923077, 0, 1346153.84615385, -1009615.38461538, 0, 0, 0, -2120192.30769231, 15923076.9230769, -5048076.92307692, 0, 1365384.61538462, -201923.076923077, 0, -1750000, -201923.076923077, 0, 0, 0, -2120192.30769231,
        -1009615.38461538, -11076923.0769231, 0, 1009615.38461538, -1346153.84615385, 0, -201923.076923077, -692307.692307692, 0, 0, 0, 2120192.30769231, 5048076.92307692, -5884615.38461538, 0, -1009615.38461538, 1009615.38461538, 0, 1009615.38461538, -721153.846153846, 0, 0, 0, 908653.846153846, 1009615.38461538, 1038461.53846154, 0, -201923.076923077, -605769.230769231, 0, -1009615.38461538, -48076.923076923, 0, 0, 0, 908653.846153846, -5048076.92307692, 15923076.9230769, 0, 201923.076923077, 1750000, 0, 201923.076923077, -1365384.61538462, 0, 0, 0, 2120192.30769231,
        0, 0, -2230769.23076923, 0, 0, -288461.538461538, 0, 0, -19230.7692307692, -605769.230769231, 1413461.53846154, 0, 0, 0, -2615384.61538462, 0, 0, 384615.384615385, 0, 0, -384615.384615385, -605769.230769231, 605769.230769231, 0, 0, 0, -2230769.23076923, 0, 0, 19230.7692307692, 0, 0, 288461.538461538, -1413461.53846154, 605769.230769231, 0, 0, 0, 7076923.07692308, 0, 0, 692307.692307692, 0, 0, -692307.692307692, -1413461.53846154, 1413461.53846154, 0,
        48076.923076923, -1009615.38461538, 0, 551282.051282051, 0, 0, 0, -140224.358974359, 0, 0, 0, 168269.230769231, -721153.846153846, 1009615.38461538, 0, -99358.9743589743, -168269.230769231, 0, 0, 140224.358974359, 0, 0, 0, -168269.230769231, -692307.692307692, -201923.076923077, 0, -371794.871794872, 33653.8461538461, 0, 0, -140224.358974359, 0, 0, 0, -336538.461538462, 1365384.61538462, 201923.076923077, 0, 1333333.33333333, 0, 0, 0, 140224.358974359, 0, 0, 0, 336538.461538462,
        -1009615.38461538, -1346153.84615385, 0, 0, -89743.5897435897, 0, -140224.358974359, 0, 0, 0, 0, 302884.615384615, 1009615.38461538, -1009615.38461538, 0, -168269.230769231, 157051.282051282, 0, 140224.358974359, 0, 0, 0, 0, 201923.076923077, 201923.076923077, 605769.230769231, 0, -33653.8461538461, -291666.666666667, 0, -140224.358974359, 0, 0, 0, 0, 201923.076923077, -201923.076923077, 1750000, 0, 0, 628205.128205128, 0, 140224.358974359, 0, 0, 0, 0, 302884.615384615,
        0, 0, -288461.538461538, 0, 0, 102564.102564103, 0, 0, 0, 112179.487179487, 201923.076923077, 0, 0, 0, -384615.384615385, 0, 0, 12820.5128205128, 0, 0, 0, -112179.487179487, 134615.384615385, 0, 0, 0, -19230.7692307692, 0, 0, -147435.897435897, 0, 0, 0, -224358.974358974, 134615.384615385, 0, 0, 0, 692307.692307692, 0, 0, 435897.435897436, 0, 0, 0, 224358.974358974, 201923.076923077, 0,
        -605769.230769231, -201923.076923077, 0, 0, -140224.358974359, 0, -291666.666666667, -33653.8461538461, 0, 0, 0, 201923.076923077, 1009615.38461538, -1009615.38461538, 0, 0, 140224.358974359, 0, 157051.282051282, -168269.230769231, 0, 0, 0, 201923.076923077, 1346153.84615385, 1009615.38461538, 0, 0, -140224.358974359, 0, -89743.5897435897, 0, 0, 0, 0, 302884.615384615, -1750000, 201923.076923077, 0, 0, 140224.358974359, 0, 628205.128205128, 0, 0, 0, 0, 302884.615384615,
        201923.076923077, 692307.692307692, 0, -140224.358974359, 0, 0, 33653.8461538461, -371794.871794872, 0, 0, 0, -336538.461538462, -1009615.38461538, 721153.846153846, 0, 140224.358974359, 0, 0, -168269.230769231, -99358.9743589743, 0, 0, 0, -168269.230769231, 1009615.38461538, -48076.923076923, 0, -140224.358974359, 0, 0, 0, 551282.051282051, 0, 0, 0, 168269.230769231, -201923.076923077, -1365384.61538462, 0, 140224.358974359, 0, 0, 0, 1333333.33333333, 0, 0, 0, 336538.461538462,
        0, 0, 19230.7692307692, 0, 0, 0, 0, 0, -147435.897435897, 134615.384615385, -224358.974358974, 0, 0, 0, 384615.384615385, 0, 0, 0, 0, 0, 12820.5128205128, 134615.384615385, -112179.487179487, 0, 0, 0, 288461.538461538, 0, 0, 0, 0, 0, 102564.102564103, 201923.076923077, 112179.487179487, 0, 0, 0, -692307.692307692, 0, 0, 0, 0, 0, 435897.435897436, 201923.076923077, 224358.974358974, 0,
        0, 0, -605769.230769231, 0, 0, 112179.487179487, 0, 0, -134615.384615385, 448734.775641026, 8.41346153846154, 0, 0, 0, 605769.230769231, 0, 0, -112179.487179487, 0, 0, 134615.384615385, 224308.493589744, 42.0673076923077, 0, 0, 0, 1413461.53846154, 0, 0, -224358.974358974, 0, 0, -201923.076923077, 448650.641025641, -8.41346153846154, 0, 0, 0, -1413461.53846154, 0, 0, 224358.974358974, 0, 0, 201923.076923077, 897536.858974359, -42.0673076923077, 0,
        0, 0, -1413461.53846154, 0, 0, -201923.076923077, 0, 0, -224358.974358974, -8.41346153846154, 448650.641025641, 0, 0, 0, -605769.230769231, 0, 0, 134615.384615385, 0, 0, -112179.487179487, 42.0673076923077, 224308.493589744, 0, 0, 0, 605769.230769231, 0, 0, -134615.384615385, 0, 0, 112179.487179487, 8.41346153846154, 448734.775641026, 0, 0, 0, 1413461.53846154, 0, 0, 201923.076923077, 0, 0, 224358.974358974, -42.0673076923077, 897536.858974359, 0,
        -908653.846153846, -2120192.30769231, 0, 168269.230769231, -302884.615384615, 0, -201923.076923077, -336538.461538462, 0, 0, 0, 1570501.6025641, 908653.846153846, -908653.846153846, 0, -168269.230769231, 201923.076923077, 0, 201923.076923077, -168269.230769231, 0, 0, 0, 785233.974358974, 2120192.30769231, 908653.846153846, 0, -336538.461538462, -201923.076923077, 0, -302884.615384615, 168269.230769231, 0, 0, 0, 1570501.6025641, -2120192.30769231, 2120192.30769231, 0, 336538.461538462, 302884.615384615, 0, 302884.615384615, 336538.461538462, 0, 0, 0, 3141070.51282051;


    //Setup the test conditions
    m_element->SetAlphaDamp(0.01);

    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation
    ChVectorDynamic<double> InternalForceNoDispNoVelNoGravity;
    InternalForceNoDispNoVelNoGravity.resize(48);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceNoDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_NoDispNoVelWithDamping;
    JacobianK_NoDispNoVelWithDamping.resize(48, 48);
    m_element->ComputeKRMmatricesGlobal(JacobianK_NoDispNoVelWithDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_NoDispNoVelWithDamping;
    JacobianR_NoDispNoVelWithDamping.resize(48, 48);
    m_element->ComputeKRMmatricesGlobal(JacobianR_NoDispNoVelWithDamping, 0, 1, 0);

    //Reset the element conditions back to its original values
    m_element->SetAlphaDamp(0.0);


    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_NoDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(48, 48);
    
    double small_terms_JacR = 1e-4*Expected_JacobianR_NoDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacR = 0;
    double max_percent_error_JacR = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacR;
    percent_error_matrix_JacR.resize(48, 48);


    for (auto i = 0; i < Expected_JacobianK_NoDispNoVelWithDamping.rows(); i++) {
        for (auto j = 0; j < Expected_JacobianK_NoDispNoVelWithDamping.cols(); j++) {
            if (std::abs(Expected_JacobianK_NoDispNoVelWithDamping(i, j)) < small_terms_JacK) {
                double error = std::abs(JacobianK_NoDispNoVelWithDamping(i, j) - Expected_JacobianK_NoDispNoVelWithDamping(i, j));
                if (error > zeros_max_error_JacK)
                    zeros_max_error_JacK = error;
                percent_error_matrix_JacK(i, j) = 0.0;
            }
            else {
                double percent_error = std::abs((JacobianK_NoDispNoVelWithDamping(i, j) - Expected_JacobianK_NoDispNoVelWithDamping(i, j)) / Expected_JacobianK_NoDispNoVelWithDamping(i, j));
                if (percent_error > max_percent_error_JacK)
                    max_percent_error_JacK = percent_error;
                percent_error_matrix_JacK(i, j) = percent_error;
            }

            if (std::abs(Expected_JacobianR_NoDispNoVelWithDamping(i, j)) < small_terms_JacR) {
                double error = std::abs(JacobianR_NoDispNoVelWithDamping(i, j) - Expected_JacobianR_NoDispNoVelWithDamping(i, j));
                if (error > zeros_max_error_JacR)
                    zeros_max_error_JacR = error;
                percent_error_matrix_JacR(i, j) = 0.0;
            }
            else {
                double percent_error = std::abs((JacobianR_NoDispNoVelWithDamping(i, j) - Expected_JacobianR_NoDispNoVelWithDamping(i, j)) / Expected_JacobianR_NoDispNoVelWithDamping(i, j));
                if (percent_error > max_percent_error_JacR)
                    max_percent_error_JacR = percent_error;
                percent_error_matrix_JacR(i, j) = percent_error;
            }
        }
    }


    if (msglvl >= 2) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Jacobian K Term - No Displacement, No Velocity, With Damping = " << std::endl;
        std::cout << JacobianK_NoDispNoVelWithDamping << std::endl;
        std::cout << "Expected Jacobian K Term - No Displacement, No Velocity, With Damping = " << std::endl;
        std::cout << Expected_JacobianK_NoDispNoVelWithDamping << std::endl;
        std::cout << "Percent Error Jacobian K Term - No Displacement, No Velocity, With Damping = " << std::endl;
        std::cout << percent_error_matrix_JacK << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Jacobian K Term - No Displacement, No Velocity, With Damping (Max Abs Error) = "
            << (JacobianK_NoDispNoVelWithDamping - Expected_JacobianK_NoDispNoVelWithDamping).cwiseAbs().maxCoeff() << std::endl;

        std::cout << "Jacobian K Term - No Displacement, No Velocity, With Damping (Max Abs Error - Only Smaller Terms) = "
            << zeros_max_error_JacK;
        if (zeros_max_error_JacK / JacobianK_NoDispNoVelWithDamping.cwiseAbs().maxCoeff() > 0.0001) {
            print_red(" - Test FAILED\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl;
        }

        std::cout << "Jacobian K Term - No Displacement, No Velocity, With Damping (Max Abs % Error - Only Larger Terms) = "
            << max_percent_error_JacK * 100 << "%";
        if (max_percent_error_JacK > 0.01) {
            print_red(" - Test FAILED\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl;
        }
    }
    if (zeros_max_error_JacK / JacobianK_NoDispNoVelWithDamping.cwiseAbs().maxCoeff() > 0.0001) {
        passed_tests = false;
    }
    if (max_percent_error_JacK > 0.01) {
        passed_tests = false;
    }

    if (msglvl >= 2) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Jacobian R Term - No Displacement, No Velocity, With Damping = " << std::endl;
        std::cout << JacobianR_NoDispNoVelWithDamping << std::endl;
        std::cout << "Expected Jacobian R Term - No Displacement, No Velocity, With Damping = " << std::endl;
        std::cout << Expected_JacobianR_NoDispNoVelWithDamping << std::endl;
        std::cout << "Percent Error Jacobian R Term - No Displacement, No Velocity, With Damping = " << std::endl;
        std::cout << percent_error_matrix_JacR << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Jacobian R Term - No Displacement, No Velocity, With Damping (Max Abs Error) = "
            << (JacobianR_NoDispNoVelWithDamping - Expected_JacobianR_NoDispNoVelWithDamping).cwiseAbs().maxCoeff() << std::endl;

        std::cout << "Jacobian R Term - No Displacement, No Velocity, With Damping (Max Abs Error - Only Smaller Terms) = "
            << zeros_max_error_JacR;
        if (zeros_max_error_JacR / JacobianR_NoDispNoVelWithDamping.cwiseAbs().maxCoeff() > 0.0001) {
            print_red(" - Test FAILED\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl;
        }

        std::cout << "Jacobian R Term - No Displacement, No Velocity, With Damping (Max Abs % Error - Only Larger Terms) = "
            << max_percent_error_JacR * 100 << "%";
        if (max_percent_error_JacR > 0.01) {
            print_red(" - Test FAILED\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl;
        }
    }
    if (zeros_max_error_JacR / JacobianR_NoDispNoVelWithDamping.cwiseAbs().maxCoeff() > 0.0001) {
        passed_tests = false;
    }
    if (max_percent_error_JacR > 0.01) {
        passed_tests = false;
    }

    return(passed_tests);
}

template <typename ElementVersion, typename MaterialVersion>
bool ANCFShellTest<ElementVersion, MaterialVersion>::JacobianSmallDispNoVelWithDampingCheck(int msglvl) {
    // =============================================================================
    //  Check the Jacobian at Small Displacement No Velocity - With Damping
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================

    ChMatrixNM<double, 48, 48> Expected_JacobianK_SmallDispNoVelWithDamping;
    Expected_JacobianK_SmallDispNoVelWithDamping <<
        1592308526.3354, 504807692.307692, 1230384.61538462, 136538569.423301, 20192307.6923077, 121057.692307692, 175000075.210014, -20192307.6923077, 109439.102564103, -102884.615384615, 0, -212019230.769231, -1107692984.74449, 100961538.461538, -1351538.46153846, 69230826.122019, -20192307.6923077, 141250, -134615429.029944, -100961538.461538, -106554.487179487, -102884.615384615, 0, -212019230.769231, -588461733.349919, -504807692.307692, -61923.0769230769, 72115403.3097992, 100961538.461538, 11634.6153846154, 100961568.117491, 100961538.461538, 16362.1794871794, -8653.84615384615, 0, -90865384.6153846, 103846191.75901, -100961538.461538, 183076.923076923, 4807705.00851712, -100961538.461538, 31826.9230769231, -60576936.9374215, 20192307.6923077, -39439.1025641026, -8653.84615384615, 0, -90865384.6153846,
        504807692.307692, 1592308526.3354, -61923.0769230769, -20192307.6923077, 175000107.88484, -47868.5897435897, 20192307.6923077, 136538536.748476, -79807.6923076923, 0, -102884.615384615, -212019230.769231, -100961538.461538, 103845476.793975, -220769.230769231, 20192307.6923077, -60576866.1856733, 59983.9743589744, -100961538.461538, 4807647.8931325, -103846.153846154, 0, -102884.615384615, -90865384.6153846, -504807692.307692, -588461733.349919, 99615.3846153846, 100961538.461538, 100961557.155953, 15016.0256410256, 100961538.461538, 72115414.2713376, 10576.9230769231, 0, -8653.84615384615, -90865384.6153846, 100961538.461538, -1107692269.77945, 183076.923076923, -100961538.461538, -134615371.91456, -6939.10256410257, -20192307.6923077, 69230755.3702707, -5769.23076923076, 0, -8653.84615384615, -212019230.769231,
        1230384.61538462, -61923.0769230769, 707694629.985207, 124711.538461538, -26538.4615384615, 69231059.840416, 101923.076923077, -119903.846153846, 69230989.6772458, -141346153.846154, -141346153.846154, -360096.153846154, -1230384.61538462, -61923.0769230769, -223078954.233459, 124711.538461538, 26538.4615384615, 1923266.7955442, -101923.076923077, -119903.846153846, -28846262.3782948, -141346153.846154, -60576923.0769231, -360096.153846154, -99615.3846153846, 61923.0769230769, -261538900.818989, 19134.6153846154, 25384.6153846154, 38461578.2306796, 23846.1538461538, 21442.3076923077, 38461596.1793975, -60576923.0769231, -60576923.0769231, -30288.4615384615, 99615.3846153846, 61923.0769230769, -223076774.93276, 19134.6153846154, -25384.6153846154, -28846109.8141922, -23846.1538461538, 21442.3076923077, 1923026.11955352, -60576923.0769231, -141346153.846154, -30288.4615384615,
        136538569.423301, -20192307.6923077, 124711.538461538, 133333367.595087, 0, 65657.0512820513, 5.80586080586081, 14022435.8974359, 3333.33333333333, 6891.02564102564, 0, 33653846.1538462, -69230847.1797113, 20192307.6923077, -155000, -37179495.1885188, -3365384.61538461, -9022.4358974359, -6.63919413919414, -14022435.8974359, -9503.20512820513, -14583.3333333333, 0, -33653846.1538462, -72115414.2713376, -100961538.461538, 21442.3076923077, -9935898.57313626, 16826923.0769231, -8894.23076923077, 3.90567765567765, 14022435.8974359, -3285.25641025641, 0, 0, -16826923.0769231, 4807692.02774789, 100961538.461538, 8846.15384615384, 55128210.1176174, 0, 16586.5384615385, -0.924908424908425, -14022435.8974359, -641.02564102564, 5769.23076923077, 0, 16826923.0769231,
        20192307.6923077, 175000107.88484, -26538.4615384615, 0, 62820547.082266, -11602.5641025641, 14022435.8974359, 5.80586080586081, -2291.66666666667, 0, 6891.02564102564, -30288461.5384615, -20192307.6923077, 60576845.127981, -55576.9230769231, 3365384.61538461, -29166674.6756982, 17131.4102564103, -14022435.8974359, -6.63919413919414, -15208.3333333333, 0, -14583.3333333333, -20192307.6923077, -100961538.461538, -100961568.117491, 23846.1538461538, 16826923.0769231, 15705127.0678894, 2580.12820512821, 14022435.8974359, 3.90567765567765, 1618.58974358974, 0, 0, -20192307.6923077, 100961538.461538, -134615384.895329, 58269.2307692308, 0, -8974353.98494668, 3429.48717948718, -14022435.8974359, -0.924908424908425, -5657.05128205128, 0, 5769.23076923077, -30288461.5384615,
        121057.692307692, -47868.5897435897, 69231059.840416, 65657.0512820513, -11602.5641025641, 43589839.0116027, 1746.79487179487, -4070.51282051282, 18.2211538461538, 22435897.4358974, -20192307.6923077, 24118.5897435897, -127788.461538462, -22131.4102564103, -1923310.76990317, -12387.8205128205, 11137.8205128205, -14743608.3028884, -8477.5641025641, -17467.9487179487, -12.724358974359, -22435897.4358974, -13461538.4615385, -51041.6666666667, 10576.9230769231, 16362.1794871794, -38461596.1793975, -6971.15384615385, 4246.79487179487, 1282045.22305152, -1618.58974358974, 3285.25641025641, 5, -11217948.7179487, -13461538.4615385, 0, -3846.15384615385, 53637.8205128205, -28846152.8911153, 14182.6923076923, 1987.17948717949, 10256425.7290733, 1618.58974358974, -4631.41025641026, -3.65384615384615, 11217948.7179487, -20192307.6923077, 20192.3076923077,
        175000075.210014, 20192307.6923077, 101923.076923077, 5.80586080586081, 14022435.8974359, 1746.79487179487, 62820547.058956, 0, 40416.6666666667, -13301.2820512821, 0, -30288461.5384615, -134615473.741483, 100961538.461538, -185576.923076923, 5.83791208791209, -14022435.8974359, 18573.717948718, -8974348.7605877, 0, -3397.4358974359, -14423.0769230769, 0, -30288461.5384615, -100961557.155953, -100961538.461538, 25384.6153846154, -6.41483516483517, 14022435.8974359, -5785.25641025641, 15705127.0678894, 16826923.0769231, -4246.79487179487, 0, 0, -20192307.6923077, 60576955.6874215, -20192307.6923077, 58269.2307692308, 9.80311355311355, -14022435.8974359, 11041.6666666667, -29166677.9361877, 3365384.61538461, -15464.7435897436, -1121.79487179487, 0, -20192307.6923077,
        -20192307.6923077, 136538536.748476, -119903.846153846, 14022435.8974359, 5.80586080586081, -4070.51282051282, 0, 133333367.571777, -24727.5641025641, 0, -13301.2820512821, 33653846.1538462, 100961538.461538, 4807603.18159404, 91923.0769230769, -14022435.8974359, 5.83791208791209, -8044.87179487179, 0, 55128215.3419764, -14182.6923076923, 0, -14423.0769230769, 16826923.0769231, -100961538.461538, -72115403.3097992, 19134.6153846154, 14022435.8974359, -6.41483516483517, 4118.58974358974, 16826923.0769231, -9935898.57313626, 6971.15384615385, 0, 0, -16826923.0769231, 20192307.6923077, -69230736.6202707, 8846.15384615384, -14022435.8974359, 9.80311355311355, -2099.35897435898, -3365384.61538461, -37179498.4490083, 8766.02564102564, 0, -1121.79487179487, -33653846.1538462,
        109439.102564103, -79807.6923076923, 69230989.6772458, 3333.33333333333, -2291.66666666667, 18.2211538461538, 40416.6666666667, -24727.5641025641, 43589831.3018125, -20192307.6923077, 22435897.4358974, -46554.4871794872, -167131.41025641, 75865.3846153846, -28846421.2244486, 16233.9743589744, -5785.25641025641, 19.6794871794872, -1955.1282051282, -16586.5384615385, 10256433.8059964, -20192307.6923077, 11217948.7179487, -50480.7692307692, 15016.0256410256, 11634.6153846154, -38461578.2306796, -4118.58974358974, 5785.25641025641, -15.6410256410256, -2580.12820512821, 8894.23076923077, 1282045.22305152, -13461538.4615385, -11217948.7179487, 0, 42676.2820512821, -7692.30769230769, -1922990.22211763, 8782.05128205128, -4439.10256410256, 24.9679487179487, -12804.4871794872, 12131.4102564103, -14743620.0511401, -13461538.4615385, -22435897.4358974, -3926.28205128205,
        -102884.615384615, 0, -141346153.846154, 6891.02564102564, 0, 22435897.4358974, -13301.2820512821, 0, -20192307.6923077, 89753739.3310379, 4206.73076923077, 65070.1322115384, 176923.076923077, 0, 141346153.846154, -28044.8717948718, 0, -22435897.4358974, 14423.0769230769, 0, 20192307.6923077, 44865110.2856007, 841.346153846154, 42621.3341346154, -8653.84615384615, 0, 60576923.0769231, 0, 0, -11217948.7179487, 0, 0, -13461538.4615385, 22430866.2484258, -4206.73076923077, 13461.1738782051, -65384.6153846154, 0, -60576923.0769231, -5769.23076923077, 0, 11217948.7179487, 12339.7435897436, 0, 13461538.4615385, 44873493.7503203, -841.346153846154, 20193.5136217949,
        0, -102884.615384615, -141346153.846154, 0, 6891.02564102564, -20192307.6923077, 0, -13301.2820512821, 22435897.4358974, 4206.73076923077, 89753739.3310379, -15704.7636217949, 0, 176923.076923077, -60576923.0769231, 0, -28044.8717948718, 13461538.4615385, 0, 14423.0769230769, 11217948.7179487, -841.346153846154, 44873523.7471392, -20193.8501602564, 0, -8653.84615384615, 60576923.0769231, 0, 0, -13461538.4615385, 0, 0, -11217948.7179487, -4206.73076923077, 22430866.2484258, -13460.8373397436, 0, -65384.6153846154, 141346153.846154, 0, -5769.23076923077, 20192307.6923077, 0, 12339.7435897436, -22435897.4358974, 841.346153846154, 44865080.2887818, -11217.4719551282,
        -212019230.769231, -212019230.769231, -360096.153846154, 33653846.1538462, -30288461.5384615, 24118.5897435897, -30288461.5384615, 33653846.1538462, -46554.4871794872, 65070.1322115384, -15704.7636217949, 314107175.961966, 212019230.769231, -90865384.6153846, 619230.769230769, -33653846.1538462, 20192307.6923077, -98157.0512820513, 30288461.5384615, 16826923.0769231, 50480.7692307692, 42622.1754807692, -20192.672275641, 157050268.014889, 90865384.6153846, 90865384.6153846, -30288.4615384615, -16826923.0769231, -20192307.6923077, 0, -20192307.6923077, -16826923.0769231, 0, 13460.8373397436, -13461.1738782051, 78523436.8446504, -90865384.6153846, 212019230.769231, -228846.153846154, 16826923.0769231, 30288461.5384615, -20192.3076923077, 20192307.6923077, -33653846.1538462, 43189.1025641026, 20193.0088141026, -11218.3133012821, 157050198.024649,
        -1107692984.74449, -100961538.461538, -1230384.61538462, -69230847.1797113, -20192307.6923077, -127788.461538462, -134615473.741483, 100961538.461538, -167131.41025641, 176923.076923077, 0, 212019230.769231, 1592309532.06966, -504807692.307692, 1836153.84615385, -136538620.480993, 20192307.6923077, -134519.230769231, 175000158.942532, 20192307.6923077, 152708.333333333, 271153.846153846, 0, 212019230.769231, 103845476.793975, 100961538.461538, 61923.0769230769, -4807603.18159404, -100961538.461538, 75865.3846153846, -60576845.127981, -20192307.6923077, -22131.4102564103, 176923.076923077, 0, 90865384.6153846, -588462024.11915, 504807692.307692, -667692.307692308, -72115481.4828761, 100961538.461538, -119326.923076923, 100961635.32903, -100961538.461538, 137516.025641026, 82692.3076923077, 0, 90865384.6153846,
        100961538.461538, 103845476.793975, -61923.0769230769, 20192307.6923077, 60576845.127981, -22131.4102564103, 100961538.461538, 4807603.18159404, 75865.3846153846, 0, 176923.076923077, -90865384.6153846, -504807692.307692, 1592309532.06966, -1836153.84615385, -20192307.6923077, -175000158.942532, 152708.333333333, -20192307.6923077, 136538620.480993, -134519.230769231, 0, 271153.846153846, -212019230.769231, -100961538.461538, -1107692984.74449, 1230384.61538462, -100961538.461538, 134615473.741483, -167131.41025641, 20192307.6923077, 69230847.1797113, -127788.461538462, 0, 176923.076923077, -212019230.769231, 504807692.307692, -588462024.11915, 667692.307692308, 100961538.461538, -100961635.32903, 137516.025641026, -100961538.461538, 72115481.4828761, -119326.923076923, 0, 82692.3076923077, -90865384.6153846,
        -1351538.46153846, -220769.230769231, -223078954.233459, -155000, -55576.9230769231, -1923310.76990317, -185576.923076923, 91923.0769230769, -28846421.2244486, 141346153.846154, -60576923.0769231, 619230.769230769, 1836153.84615385, -1836153.84615385, 707697826.978214, -94423.0769230769, 174038.461538462, -69231246.0583647, 174038.461538462, -94423.0769230769, 69231246.0583647, 141346153.846154, -141346153.846154, 949038.461538462, 220769.230769231, 1351538.46153846, -223078954.233459, 91923.0769230769, -185576.923076923, 28846421.2244486, -55576.9230769231, -155000, 1923310.76990317, 60576923.0769231, -141346153.846154, 619230.769230769, -705384.615384615, 705384.615384615, -261539918.511296, -130192.307692308, 147884.615384615, -38461829.0640129, 147884.615384615, -130192.307692308, 38461829.0640129, 60576923.0769231, -60576923.0769231, 289423.076923077,
        69230826.122019, 20192307.6923077, 124711.538461538, -37179495.1885188, 3365384.61538461, -12387.8205128205, 5.83791208791209, -14022435.8974359, 16233.9743589744, -28044.8717948718, 0, -33653846.1538462, -136538620.480993, -20192307.6923077, -94423.0769230769, 133333390.92842, 0, 65657.0512820513, -7.40842490842491, 14022435.8974359, -10064.1025641026, -13301.2820512821, 0, 33653846.1538462, -4807647.8931325, 100961538.461538, -119903.846153846, 55128215.3419764, 0, 16586.5384615385, -6.63919413919414, -14022435.8974359, 17467.9487179487, -14423.0769230769, 0, 16826923.0769231, 72115442.2521069, -100961538.461538, 89615.3846153846, -9935889.91929011, -16826923.0769231, 7932.69230769231, -6.82234432234432, 14022435.8974359, -13541.6666666667, -13461.5384615385, 0, -16826923.0769231,
        -20192307.6923077, -60576866.1856733, 26538.4615384615, -3365384.61538461, -29166674.6756982, 11137.8205128205, -14022435.8974359, 5.83791208791209, -5785.25641025641, 0, -28044.8717948718, 20192307.6923077, 20192307.6923077, -175000158.942532, 174038.461538462, 0, 62820570.4155994, -47628.2051282051, 14022435.8974359, -7.40842490842491, 11842.9487179487, 0, -13301.2820512821, 30288461.5384615, 100961538.461538, 134615429.029944, -101923.076923077, 0, -8974348.7605877, 1955.1282051282, -14022435.8974359, -6.63919413919414, 8477.5641025641, 0, -14423.0769230769, 30288461.5384615, -100961538.461538, 100961596.098261, -98653.8461538462, -16826923.0769231, 15705135.7217355, -17387.8205128205, 14022435.8974359, -6.82234432234432, 11041.6666666667, 0, -13461.5384615385, 20192307.6923077,
        141250, 59983.9743589744, 1923266.7955442, -9022.4358974359, 17131.4102564103, -14743608.3028884, 18573.717948718, -8044.87179487179, 19.6794871794872, -22435897.4358974, 13461538.4615385, -98157.0512820513, -134519.230769231, 152708.333333333, -69231246.0583647, 65657.0512820513, -47628.2051282051, 43589900.9346796, -11842.9487179487, 10064.1025641026, -31.2339743589744, 22435897.4358974, 20192307.6923077, -46554.4871794872, -103846.153846154, -106554.487179487, 28846262.3782948, 14182.6923076923, 3397.4358974359, 10256433.8059964, 15208.3333333333, 9503.20512820513, -12.724358974359, 11217948.7179487, 20192307.6923077, -50480.7692307692, 97115.3846153846, -106137.820512821, 38461716.8845257, 9855.76923076923, -19054.4871794872, 1282075.51151306, -15208.3333333333, 12708.3333333333, -22.948717948718, -11217948.7179487, 13461538.4615385, -47115.3846153846,
        -134615429.029944, -100961538.461538, -101923.076923077, -6.63919413919414, -14022435.8974359, -8477.5641025641, -8974348.7605877, 0, -1955.1282051282, 14423.0769230769, 0, 30288461.5384615, 175000158.942532, -20192307.6923077, 174038.461538462, -7.40842490842491, 14022435.8974359, -11842.9487179487, 62820570.4155994, 0, 47628.2051282051, 13301.2820512821, 0, 30288461.5384615, 60576866.1856733, 20192307.6923077, 26538.4615384615, 5.83791208791209, -14022435.8974359, 5785.25641025641, -29166674.6756982, -3365384.61538461, -11137.8205128205, 28044.8717948718, 0, 20192307.6923077, -100961596.098261, 100961538.461538, -98653.8461538462, -6.82234432234432, 14022435.8974359, -11041.6666666667, 15705135.7217355, -16826923.0769231, 17387.8205128205, 13461.5384615385, 0, 20192307.6923077,
        -100961538.461538, 4807647.8931325, -119903.846153846, -14022435.8974359, -6.63919413919414, -17467.9487179487, 0, 55128215.3419764, -16586.5384615385, 0, 14423.0769230769, 16826923.0769231, 20192307.6923077, 136538620.480993, -94423.0769230769, 14022435.8974359, -7.40842490842491, 10064.1025641026, 0, 133333390.92842, -65657.0512820513, 0, 13301.2820512821, 33653846.1538462, -20192307.6923077, -69230826.122019, 124711.538461538, -14022435.8974359, 5.83791208791209, -16233.9743589744, 3365384.61538461, -37179495.1885188, 12387.8205128205, 0, 28044.8717948718, -33653846.1538462, 100961538.461538, -72115442.2521069, 89615.3846153846, 14022435.8974359, -6.82234432234432, 13541.6666666667, -16826923.0769231, -9935889.91929011, -7932.69230769231, 0, 13461.5384615385, -16826923.0769231,
        -106554.487179487, -103846.153846154, -28846262.3782948, -9503.20512820513, -15208.3333333333, -12.724358974359, -3397.4358974359, -14182.6923076923, 10256433.8059964, 20192307.6923077, 11217948.7179487, 50480.7692307692, 152708.333333333, -134519.230769231, 69231246.0583647, -10064.1025641026, 11842.9487179487, -31.2339743589744, 47628.2051282051, -65657.0512820513, 43589900.9346796, 20192307.6923077, 22435897.4358974, 46554.4871794872, 59983.9743589744, 141250, -1923266.7955442, 8044.87179487179, -18573.717948718, 19.6794871794872, -17131.4102564103, 9022.4358974359, -14743608.3028884, 13461538.4615385, -22435897.4358974, 98157.0512820513, -106137.820512821, 97115.3846153846, -38461716.8845257, -12708.3333333333, 15208.3333333333, -22.948717948718, 19054.4871794872, -9855.76923076923, 1282075.51151306, 13461538.4615385, -11217948.7179487, 47115.3846153846,
        -102884.615384615, 0, -141346153.846154, -14583.3333333333, 0, -22435897.4358974, -14423.0769230769, 0, -20192307.6923077, 44865110.2856007, -841.346153846154, 42622.1754807692, 271153.846153846, 0, 141346153.846154, -13301.2820512821, 0, 22435897.4358974, 13301.2820512821, 0, 20192307.6923077, 89753780.0089786, -4206.73076923077, 65074.3389423077, -102884.615384615, 0, 60576923.0769231, 14423.0769230769, 0, 11217948.7179487, 14583.3333333333, 0, -13461538.4615385, 44873523.7471392, 841.346153846154, 20192.672275641, -65384.6153846154, 0, -60576923.0769231, -13461.5384615385, 0, -11217948.7179487, 13461.5384615385, 0, 13461538.4615385, 22430866.2467431, 4206.73076923077, 13456.9671474359,
        0, -102884.615384615, -60576923.0769231, 0, -14583.3333333333, -13461538.4615385, 0, -14423.0769230769, 11217948.7179487, 841.346153846154, 44873523.7471392, -20192.672275641, 0, 271153.846153846, -141346153.846154, 0, -13301.2820512821, 20192307.6923077, 0, 13301.2820512821, 22435897.4358974, -4206.73076923077, 89753780.0089786, -65074.3389423077, 0, -102884.615384615, 141346153.846154, 0, 14423.0769230769, -20192307.6923077, 0, 14583.3333333333, -22435897.4358974, -841.346153846154, 44865110.2856007, -42622.1754807692, 0, -65384.6153846154, 60576923.0769231, 0, -13461.5384615385, 13461538.4615385, 0, 13461.5384615385, -11217948.7179487, 4206.73076923077, 22430866.2467431, -13456.9671474359,
        -212019230.769231, -90865384.6153846, -360096.153846154, -33653846.1538462, -20192307.6923077, -51041.6666666667, -30288461.5384615, 16826923.0769231, -50480.7692307692, 42621.3341346154, -20193.8501602564, 157050268.014889, 212019230.769231, -212019230.769231, 949038.461538462, 33653846.1538462, 30288461.5384615, -46554.4871794872, 30288461.5384615, 33653846.1538462, 46554.4871794872, 65074.3389423077, -65074.3389423077, 314107270.881461, 90865384.6153846, 212019230.769231, -360096.153846154, 16826923.0769231, -30288461.5384615, 50480.7692307692, -20192307.6923077, -33653846.1538462, 51041.6666666667, 20193.8501602564, -42621.3341346154, 157050268.014889, -90865384.6153846, 90865384.6153846, -228846.153846154, -16826923.0769231, 20192307.6923077, -47115.3846153846, 20192307.6923077, -16826923.0769231, 47115.3846153846, 13456.6306089744, -13456.6306089744, 78523436.8387609,
        -588461733.349919, -504807692.307692, -99615.3846153846, -72115414.2713376, -100961538.461538, 10576.9230769231, -100961557.155953, -100961538.461538, 15016.0256410256, -8653.84615384615, 0, 90865384.6153846, 103845476.793975, -100961538.461538, 220769.230769231, -4807647.8931325, 100961538.461538, -103846.153846154, 60576866.1856733, -20192307.6923077, 59983.9743589744, -102884.615384615, 0, 90865384.6153846, 1592308526.3354, 504807692.307692, 61923.0769230769, -136538536.748476, -20192307.6923077, -79807.6923076923, -175000107.88484, 20192307.6923077, -47868.5897435897, -102884.615384615, 0, 212019230.769231, -1107692269.77945, 100961538.461538, -183076.923076923, -69230755.3702707, 20192307.6923077, -5769.23076923076, 134615371.91456, 100961538.461538, -6939.10256410257, -8653.84615384615, 0, 212019230.769231,
        -504807692.307692, -588461733.349919, 61923.0769230769, -100961538.461538, -100961568.117491, 16362.1794871794, -100961538.461538, -72115403.3097992, 11634.6153846154, 0, -8653.84615384615, 90865384.6153846, 100961538.461538, -1107692984.74449, 1351538.46153846, 100961538.461538, 134615429.029944, -106554.487179487, 20192307.6923077, -69230826.122019, 141250, 0, -102884.615384615, 212019230.769231, 504807692.307692, 1592308526.3354, -1230384.61538462, 20192307.6923077, -175000075.210014, 109439.102564103, -20192307.6923077, -136538569.423301, 121057.692307692, 0, -102884.615384615, 212019230.769231, -100961538.461538, 103846191.75901, -183076.923076923, -20192307.6923077, 60576936.9374215, -39439.1025641026, 100961538.461538, -4807705.00851712, 31826.9230769231, 0, -8653.84615384615, 90865384.6153846,
        -61923.0769230769, 99615.3846153846, -261538900.818989, 21442.3076923077, 23846.1538461538, -38461596.1793975, 25384.6153846154, 19134.6153846154, -38461578.2306796, 60576923.0769231, 60576923.0769231, -30288.4615384615, 61923.0769230769, 1230384.61538462, -223078954.233459, -119903.846153846, -101923.076923077, 28846262.3782948, 26538.4615384615, 124711.538461538, -1923266.7955442, 60576923.0769231, 141346153.846154, -360096.153846154, 61923.0769230769, -1230384.61538462, 707694629.985207, -119903.846153846, 101923.076923077, -69230989.6772458, -26538.4615384615, 124711.538461538, -69231059.840416, 141346153.846154, 141346153.846154, -360096.153846154, -61923.0769230769, -99615.3846153846, -223076774.93276, 21442.3076923077, -23846.1538461538, -1923026.11955352, -25384.6153846154, 19134.6153846154, 28846109.8141922, 141346153.846154, 60576923.0769231, -30288.4615384615,
        72115403.3097992, 100961538.461538, 19134.6153846154, -9935898.57313626, 16826923.0769231, -6971.15384615385, -6.41483516483517, 14022435.8974359, -4118.58974358974, 0, 0, -16826923.0769231, -4807603.18159404, -100961538.461538, 91923.0769230769, 55128215.3419764, 0, 14182.6923076923, 5.83791208791209, -14022435.8974359, 8044.87179487179, 14423.0769230769, 0, 16826923.0769231, -136538536.748476, 20192307.6923077, -119903.846153846, 133333367.571777, 0, 24727.5641025641, 5.80586080586081, 14022435.8974359, 4070.51282051282, 13301.2820512821, 0, 33653846.1538462, 69230736.6202707, -20192307.6923077, 8846.15384615384, -37179498.4490083, -3365384.61538461, -8766.02564102564, 9.80311355311355, -14022435.8974359, 2099.35897435898, 1121.79487179487, 0, -33653846.1538462,
        100961538.461538, 100961557.155953, 25384.6153846154, 16826923.0769231, 15705127.0678894, 4246.79487179487, 14022435.8974359, -6.41483516483517, 5785.25641025641, 0, 0, -20192307.6923077, -100961538.461538, 134615473.741483, -185576.923076923, 0, -8974348.7605877, 3397.4358974359, -14022435.8974359, 5.83791208791209, -18573.717948718, 0, 14423.0769230769, -30288461.5384615, -20192307.6923077, -175000075.210014, 101923.076923077, 0, 62820547.058956, -40416.6666666667, 14022435.8974359, 5.80586080586081, -1746.79487179487, 0, 13301.2820512821, -30288461.5384615, 20192307.6923077, -60576955.6874215, 58269.2307692308, 3365384.61538461, -29166677.9361877, 15464.7435897436, -14022435.8974359, 9.80311355311355, -11041.6666666667, 0, 1121.79487179487, -20192307.6923077,
        11634.6153846154, 15016.0256410256, 38461578.2306796, -8894.23076923077, 2580.12820512821, 1282045.22305152, -5785.25641025641, 4118.58974358974, -15.6410256410256, -11217948.7179487, -13461538.4615385, 0, 75865.3846153846, -167131.41025641, 28846421.2244486, 16586.5384615385, 1955.1282051282, 10256433.8059964, 5785.25641025641, -16233.9743589744, 19.6794871794872, 11217948.7179487, -20192307.6923077, 50480.7692307692, -79807.6923076923, 109439.102564103, -69230989.6772458, 24727.5641025641, -40416.6666666667, 43589831.3018125, 2291.66666666667, -3333.33333333333, 18.2211538461538, 22435897.4358974, -20192307.6923077, 46554.4871794872, -7692.30769230769, 42676.2820512821, 1922990.22211763, -12131.4102564103, 12804.4871794872, -14743620.0511401, 4439.10256410256, -8782.05128205128, 24.9679487179487, -22435897.4358974, -13461538.4615385, 3926.28205128205,
        100961568.117491, 100961538.461538, 23846.1538461538, 3.90567765567765, 14022435.8974359, -1618.58974358974, 15705127.0678894, 16826923.0769231, -2580.12820512821, 0, 0, -20192307.6923077, -60576845.127981, 20192307.6923077, -55576.9230769231, -6.63919413919414, -14022435.8974359, 15208.3333333333, -29166674.6756982, 3365384.61538461, -17131.4102564103, 14583.3333333333, 0, -20192307.6923077, -175000107.88484, -20192307.6923077, -26538.4615384615, 5.80586080586081, 14022435.8974359, 2291.66666666667, 62820547.082266, 0, 11602.5641025641, -6891.02564102564, 0, -30288461.5384615, 134615384.895329, -100961538.461538, 58269.2307692308, -0.924908424908425, -14022435.8974359, 5657.05128205128, -8974353.98494668, 0, -3429.48717948718, -5769.23076923077, 0, -30288461.5384615,
        100961538.461538, 72115414.2713376, 21442.3076923077, 14022435.8974359, 3.90567765567765, 3285.25641025641, 16826923.0769231, -9935898.57313626, 8894.23076923077, 0, 0, -16826923.0769231, -20192307.6923077, 69230847.1797113, -155000, -14022435.8974359, -6.63919413919414, 9503.20512820513, -3365384.61538461, -37179495.1885188, 9022.4358974359, 0, 14583.3333333333, -33653846.1538462, 20192307.6923077, -136538569.423301, 124711.538461538, 14022435.8974359, 5.80586080586081, -3333.33333333333, 0, 133333367.595087, -65657.0512820513, 0, -6891.02564102564, 33653846.1538462, -100961538.461538, -4807692.02774789, 8846.15384615384, -14022435.8974359, -0.924908424908425, 641.02564102564, 0, 55128210.1176174, -16586.5384615385, 0, -5769.23076923077, 16826923.0769231,
        16362.1794871794, 10576.9230769231, 38461596.1793975, -3285.25641025641, 1618.58974358974, 5, -4246.79487179487, 6971.15384615385, 1282045.22305152, -13461538.4615385, -11217948.7179487, 0, -22131.4102564103, -127788.461538462, 1923310.76990317, 17467.9487179487, 8477.5641025641, -12.724358974359, -11137.8205128205, 12387.8205128205, -14743608.3028884, -13461538.4615385, -22435897.4358974, 51041.6666666667, -47868.5897435897, 121057.692307692, -69231059.840416, 4070.51282051282, -1746.79487179487, 18.2211538461538, 11602.5641025641, -65657.0512820513, 43589839.0116027, -20192307.6923077, 22435897.4358974, -24118.5897435897, 53637.8205128205, -3846.15384615385, 28846152.8911153, 4631.41025641026, -1618.58974358974, -3.65384615384615, -1987.17948717949, -14182.6923076923, 10256425.7290733, -20192307.6923077, 11217948.7179487, -20192.3076923077,
        -8653.84615384615, 0, -60576923.0769231, 0, 0, -11217948.7179487, 0, 0, -13461538.4615385, 22430866.2484258, -4206.73076923077, 13460.8373397436, 176923.076923077, 0, 60576923.0769231, -14423.0769230769, 0, 11217948.7179487, 28044.8717948718, 0, 13461538.4615385, 44873523.7471392, -841.346153846154, 20193.8501602564, -102884.615384615, 0, 141346153.846154, 13301.2820512821, 0, 22435897.4358974, -6891.02564102564, 0, -20192307.6923077, 89753739.3310379, 4206.73076923077, 15704.7636217949, -65384.6153846154, 0, -141346153.846154, -12339.7435897436, 0, -22435897.4358974, 5769.23076923077, 0, 20192307.6923077, 44865080.2887818, 841.346153846154, 11217.4719551282,
        0, -8653.84615384615, -60576923.0769231, 0, 0, -13461538.4615385, 0, 0, -11217948.7179487, -4206.73076923077, 22430866.2484258, -13461.1738782051, 0, 176923.076923077, -141346153.846154, 0, -14423.0769230769, 20192307.6923077, 0, 28044.8717948718, -22435897.4358974, 841.346153846154, 44865110.2856007, -42621.3341346154, 0, -102884.615384615, 141346153.846154, 0, 13301.2820512821, -20192307.6923077, 0, -6891.02564102564, 22435897.4358974, 4206.73076923077, 89753739.3310379, -65070.1322115384, 0, -65384.6153846154, 60576923.0769231, 0, -12339.7435897436, 13461538.4615385, 0, 5769.23076923077, 11217948.7179487, -841.346153846154, 44873493.7503203, -20193.5136217949,
        -90865384.6153846, -90865384.6153846, -30288.4615384615, -16826923.0769231, -20192307.6923077, 0, -20192307.6923077, -16826923.0769231, 0, 13461.1738782051, -13460.8373397436, 78523436.8446504, 90865384.6153846, -212019230.769231, 619230.769230769, 16826923.0769231, 30288461.5384615, -50480.7692307692, 20192307.6923077, -33653846.1538462, 98157.0512820513, 20192.672275641, -42622.1754807692, 157050268.014889, 212019230.769231, 212019230.769231, -360096.153846154, 33653846.1538462, -30288461.5384615, 46554.4871794872, -30288461.5384615, 33653846.1538462, -24118.5897435897, 15704.7636217949, -65070.1322115384, 314107175.961966, -212019230.769231, 90865384.6153846, -228846.153846154, -33653846.1538462, 20192307.6923077, -43189.1025641026, 30288461.5384615, 16826923.0769231, 20192.3076923077, 11218.3133012821, -20193.0088141026, 157050198.024649,
        103846191.75901, 100961538.461538, 99615.3846153846, 4807692.02774789, 100961538.461538, -3846.15384615385, 60576955.6874215, 20192307.6923077, 42676.2820512821, -65384.6153846154, 0, -90865384.6153846, -588462024.11915, 504807692.307692, -705384.615384615, 72115442.2521069, -100961538.461538, 97115.3846153846, -100961596.098261, 100961538.461538, -106137.820512821, -65384.6153846154, 0, -90865384.6153846, -1107692269.77945, -100961538.461538, -61923.0769230769, 69230736.6202707, 20192307.6923077, -7692.30769230769, 134615384.895329, -100961538.461538, 53637.8205128205, -65384.6153846154, 0, -212019230.769231, 1592308102.13959, -504807692.307692, 667692.307692308, 136538531.84463, -20192307.6923077, 93269.2307692308, -175000070.306168, -20192307.6923077, -91137.8205128205, -65384.6153846154, 0, -212019230.769231,
        -100961538.461538, -1107692269.77945, 61923.0769230769, 100961538.461538, -134615384.895329, 53637.8205128205, -20192307.6923077, -69230736.6202707, -7692.30769230769, 0, -65384.6153846154, 212019230.769231, 504807692.307692, -588462024.11915, 705384.615384615, -100961538.461538, 100961596.098261, -106137.820512821, 100961538.461538, -72115442.2521069, 97115.3846153846, 0, -65384.6153846154, 90865384.6153846, 100961538.461538, 103846191.75901, -99615.3846153846, -20192307.6923077, -60576955.6874215, 42676.2820512821, -100961538.461538, -4807692.02774789, -3846.15384615385, 0, -65384.6153846154, 90865384.6153846, -504807692.307692, 1592308102.13959, -667692.307692308, 20192307.6923077, 175000070.306168, -91137.8205128205, 20192307.6923077, -136538531.84463, 93269.2307692308, 0, -65384.6153846154, 212019230.769231,
        183076.923076923, 183076.923076923, -223076774.93276, 8846.15384615384, 58269.2307692308, -28846152.8911153, 58269.2307692308, 8846.15384615384, -1922990.22211763, -60576923.0769231, 141346153.846154, -228846.153846154, -667692.307692308, 667692.307692308, -261539918.511296, 89615.3846153846, -98653.8461538462, 38461716.8845257, -98653.8461538462, 89615.3846153846, -38461716.8845257, -60576923.0769231, 60576923.0769231, -228846.153846154, -183076.923076923, -183076.923076923, -223076774.93276, 8846.15384615384, 58269.2307692308, 1922990.22211763, 58269.2307692308, 8846.15384615384, 28846152.8911153, -141346153.846154, 60576923.0769231, -228846.153846154, 667692.307692308, -667692.307692308, 707693468.376816, 89615.3846153846, -98653.8461538462, 69230964.9977587, -98653.8461538462, 89615.3846153846, -69230964.9977587, -141346153.846154, 141346153.846154, -228846.153846154,
        4807705.00851712, -100961538.461538, 19134.6153846154, 55128210.1176174, 0, 14182.6923076923, 9.80311355311355, -14022435.8974359, 8782.05128205128, -5769.23076923077, 0, 16826923.0769231, -72115481.4828761, 100961538.461538, -130192.307692308, -9935889.91929011, -16826923.0769231, 9855.76923076923, -6.82234432234432, 14022435.8974359, -12708.3333333333, -13461.5384615385, 0, -16826923.0769231, -69230755.3702707, -20192307.6923077, 21442.3076923077, -37179498.4490083, 3365384.61538461, -12131.4102564103, -0.924908424908425, -14022435.8974359, 4631.41025641026, -12339.7435897436, 0, -33653846.1538462, 136538531.84463, 20192307.6923077, 89615.3846153846, 133333350.007674, 0, 24727.5641025641, -12.2802197802198, 14022435.8974359, -10801.2820512821, -6891.02564102564, 0, 33653846.1538462,
        -100961538.461538, -134615371.91456, -25384.6153846154, 0, -8974353.98494668, 1987.17948717949, -14022435.8974359, 9.80311355311355, -4439.10256410256, 0, -5769.23076923077, 30288461.5384615, 100961538.461538, -100961635.32903, 147884.615384615, -16826923.0769231, 15705135.7217355, -19054.4871794872, 14022435.8974359, -6.82234432234432, 15208.3333333333, 0, -13461.5384615385, 20192307.6923077, 20192307.6923077, 60576936.9374215, -23846.1538461538, -3365384.61538461, -29166677.9361877, 12804.4871794872, -14022435.8974359, -0.924908424908425, -1618.58974358974, 0, -12339.7435897436, 20192307.6923077, -20192307.6923077, 175000070.306168, -98653.8461538462, 0, 62820529.4948534, -18814.1025641026, 14022435.8974359, -12.2802197802198, 12387.8205128205, 0, -6891.02564102564, 30288461.5384615,
        31826.9230769231, -6939.10256410257, -28846109.8141922, 16586.5384615385, 3429.48717948718, 10256425.7290733, 11041.6666666667, -2099.35897435898, 24.9679487179487, 11217948.7179487, 20192307.6923077, -20192.3076923077, -119326.923076923, 137516.025641026, -38461829.0640129, 7932.69230769231, -17387.8205128205, 1282075.51151306, -11041.6666666667, 13541.6666666667, -22.948717948718, -11217948.7179487, 13461538.4615385, -47115.3846153846, -5769.23076923076, -39439.1025641026, -1923026.11955352, -8766.02564102564, 15464.7435897436, -14743620.0511401, 5657.05128205128, 641.02564102564, -3.65384615384615, -22435897.4358974, 13461538.4615385, -43189.1025641026, 93269.2307692308, -91137.8205128205, 69230964.9977587, 24727.5641025641, -18814.1025641026, 43589789.5710433, -12387.8205128205, 10801.2820512821, -33.4775641025641, 22435897.4358974, 20192307.6923077, -24118.5897435897,
        -60576936.9374215, -20192307.6923077, -23846.1538461538, -0.924908424908425, -14022435.8974359, 1618.58974358974, -29166677.9361877, -3365384.61538461, -12804.4871794872, 12339.7435897436, 0, 20192307.6923077, 100961635.32903, -100961538.461538, 147884.615384615, -6.82234432234432, 14022435.8974359, -15208.3333333333, 15705135.7217355, -16826923.0769231, 19054.4871794872, 13461.5384615385, 0, 20192307.6923077, 134615371.91456, 100961538.461538, -25384.6153846154, 9.80311355311355, -14022435.8974359, 4439.10256410256, -8974353.98494668, 0, -1987.17948717949, 5769.23076923077, 0, 30288461.5384615, -175000070.306168, 20192307.6923077, -98653.8461538462, -12.2802197802198, 14022435.8974359, -12387.8205128205, 62820529.4948534, 0, 18814.1025641026, 6891.02564102564, 0, 30288461.5384615,
        20192307.6923077, 69230755.3702707, 21442.3076923077, -14022435.8974359, -0.924908424908425, -4631.41025641026, 3365384.61538461, -37179498.4490083, 12131.4102564103, 0, 12339.7435897436, -33653846.1538462, -100961538.461538, 72115481.4828761, -130192.307692308, 14022435.8974359, -6.82234432234432, 12708.3333333333, -16826923.0769231, -9935889.91929011, -9855.76923076923, 0, 13461.5384615385, -16826923.0769231, 100961538.461538, -4807705.00851712, 19134.6153846154, -14022435.8974359, 9.80311355311355, -8782.05128205128, 0, 55128210.1176174, -14182.6923076923, 0, 5769.23076923077, 16826923.0769231, -20192307.6923077, -136538531.84463, 89615.3846153846, 14022435.8974359, -12.2802197802198, 10801.2820512821, 0, 133333350.007674, -24727.5641025641, 0, 6891.02564102564, 33653846.1538462,
        -39439.1025641026, -5769.23076923076, 1923026.11955352, -641.02564102564, -5657.05128205128, -3.65384615384615, -15464.7435897436, 8766.02564102564, -14743620.0511401, 13461538.4615385, -22435897.4358974, 43189.1025641026, 137516.025641026, -119326.923076923, 38461829.0640129, -13541.6666666667, 11041.6666666667, -22.948717948718, 17387.8205128205, -7932.69230769231, 1282075.51151306, 13461538.4615385, -11217948.7179487, 47115.3846153846, -6939.10256410257, 31826.9230769231, 28846109.8141922, 2099.35897435898, -11041.6666666667, 24.9679487179487, -3429.48717948718, -16586.5384615385, 10256425.7290733, 20192307.6923077, 11217948.7179487, 20192.3076923077, -91137.8205128205, 93269.2307692308, -69230964.9977587, -10801.2820512821, 12387.8205128205, -33.4775641025641, 18814.1025641026, -24727.5641025641, 43589789.5710433, 20192307.6923077, 22435897.4358974, 24118.5897435897,
        -8653.84615384615, 0, -60576923.0769231, 5769.23076923077, 0, 11217948.7179487, -1121.79487179487, 0, -13461538.4615385, 44873493.7503203, 841.346153846154, 20193.0088141026, 82692.3076923077, 0, 60576923.0769231, -13461.5384615385, 0, -11217948.7179487, 13461.5384615385, 0, 13461538.4615385, 22430866.2467431, 4206.73076923077, 13456.6306089744, -8653.84615384615, 0, 141346153.846154, 1121.79487179487, 0, -22435897.4358974, -5769.23076923077, 0, -20192307.6923077, 44865080.2887818, -841.346153846154, 11218.3133012821, -65384.6153846154, 0, -141346153.846154, -6891.02564102564, 0, 22435897.4358974, 6891.02564102564, 0, 20192307.6923077, 89753698.6564626, -4206.73076923077, 15708.9703525641,
        0, -8653.84615384615, -141346153.846154, 0, 5769.23076923077, -20192307.6923077, 0, -1121.79487179487, -22435897.4358974, -841.346153846154, 44865080.2887818, -11218.3133012821, 0, 82692.3076923077, -60576923.0769231, 0, -13461.5384615385, 13461538.4615385, 0, 13461.5384615385, -11217948.7179487, 4206.73076923077, 22430866.2467431, -13456.6306089744, 0, -8653.84615384615, 60576923.0769231, 0, 1121.79487179487, -13461538.4615385, 0, -5769.23076923077, 11217948.7179487, 841.346153846154, 44873493.7503203, -20193.0088141026, 0, -65384.6153846154, 141346153.846154, 0, -6891.02564102564, 20192307.6923077, 0, 6891.02564102564, 22435897.4358974, -4206.73076923077, 89753698.6564626, -15708.9703525641,
        -90865384.6153846, -212019230.769231, -30288.4615384615, 16826923.0769231, -30288461.5384615, 20192.3076923077, -20192307.6923077, -33653846.1538462, -3926.28205128205, 20193.5136217949, -11217.4719551282, 157050198.024649, 90865384.6153846, -90865384.6153846, 289423.076923077, -16826923.0769231, 20192307.6923077, -47115.3846153846, 20192307.6923077, -16826923.0769231, 47115.3846153846, 13456.9671474359, -13456.9671474359, 78523436.8387609, 212019230.769231, 90865384.6153846, -30288.4615384615, -33653846.1538462, -20192307.6923077, 3926.28205128205, -30288461.5384615, 16826923.0769231, -20192.3076923077, 11217.4719551282, -20193.5136217949, 157050198.024649, -212019230.769231, 212019230.769231, -228846.153846154, 33653846.1538462, 30288461.5384615, -24118.5897435897, 30288461.5384615, 33653846.1538462, 24118.5897435897, 15708.9703525641, -15708.9703525641, 314107081.05425;

    ChMatrixNM<double, 48, 48> Expected_JacobianR_SmallDispNoVelWithDamping;
    Expected_JacobianR_SmallDispNoVelWithDamping <<
        15923076.9230769, 5048076.92307692, 12303.8461538462, 1365384.61538462, 201923.076923077, 1210.57692307692, 1750000, -201923.076923077, 1094.39102564103, 0, 0, -2120192.30769231, -11076923.0769231, 1009615.38461538, -13515.3846153846, 692307.692307692, -201923.076923077, 1412.5, -1346153.84615385, -1009615.38461538, -1065.54487179487, 0, 0, -2120192.30769231, -5884615.38461538, -5048076.92307692, -619.230769230769, 721153.846153846, 1009615.38461538, 116.346153846154, 1009615.38461538, 1009615.38461538, 163.621794871795, 0, 0, -908653.846153846, 1038461.53846154, -1009615.38461538, 1830.76923076923, 48076.923076923, -1009615.38461538, 318.269230769231, -605769.230769231, 201923.076923077, -394.391025641026, 0, 0, -908653.846153846,
        5048076.92307692, 15923076.9230769, -619.230769230769, -201923.076923077, 1750000, -478.685897435897, 201923.076923077, 1365384.61538462, -798.076923076923, 0, 0, -2120192.30769231, -1009615.38461538, 1038461.53846154, -2207.69230769231, 201923.076923077, -605769.230769231, 599.839743589744, -1009615.38461538, 48076.923076923, -1038.46153846154, 0, 0, -908653.846153846, -5048076.92307692, -5884615.38461538, 996.153846153846, 1009615.38461538, 1009615.38461538, 150.160256410256, 1009615.38461538, 721153.846153846, 105.769230769231, 0, 0, -908653.846153846, 1009615.38461538, -11076923.0769231, 1830.76923076923, -1009615.38461538, -1346153.84615385, -69.3910256410257, -201923.076923077, 692307.692307692, -57.6923076923076, 0, 0, -2120192.30769231,
        12303.8461538462, -619.230769230769, 7076937.95957504, 1247.11538461538, -265.384615384615, 692309.519555765, 1019.23076923077, -1199.03846153846, 692309.144672315, -1413461.53846154, -1413461.53846154, -2572.11538461538, -12303.8461538462, -619.230769230769, -2230782.7718128, 1247.11538461538, 265.384615384615, 19232.0990429442, -1019.23076923077, -1199.03846153846, -288462.17963735, -1413461.53846154, -605769.230769231, -2572.11538461538, -996.153846153846, 619.230769230769, -2615387.05930608, 191.346153846154, 253.846153846154, 384615.59536265, 238.461538461538, 214.423076923077, 384615.665234445, -605769.230769231, -605769.230769231, -216.346153846154, 996.153846153846, 619.230769230769, -2230768.12845616, 191.346153846154, -253.846153846154, -288461.22515017, -238.461538461538, 214.423076923077, 19230.39980052, -605769.230769231, -1413461.53846154, -216.346153846154,
        1365384.61538462, -201923.076923077, 1247.11538461538, 1333333.33333333, 0, 656.570512820513, 0, 140224.358974359, 33.3333333333333, 0, 0, 336538.461538462, -692307.692307692, 201923.076923077, -1550, -371794.871794872, -33653.8461538461, -90.224358974359, 0, -140224.358974359, -95.0320512820513, 0, 0, -336538.461538462, -721153.846153846, -1009615.38461538, 214.423076923077, -99358.9743589743, 168269.230769231, -88.9423076923077, 0, 140224.358974359, -32.8525641025641, 0, 0, -168269.230769231, 48076.923076923, 1009615.38461538, 88.4615384615384, 551282.051282051, 0, 165.865384615385, 0, -140224.358974359, -6.4102564102564, 0, 0, 168269.230769231,
        201923.076923077, 1750000, -265.384615384615, 0, 628205.128205128, -116.025641025641, 140224.358974359, 0, -22.9166666666667, 0, 0, -302884.615384615, -201923.076923077, 605769.230769231, -555.769230769231, 33653.8461538461, -291666.666666667, 171.314102564103, -140224.358974359, 0, -152.083333333333, 0, 0, -201923.076923077, -1009615.38461538, -1009615.38461538, 238.461538461538, 168269.230769231, 157051.282051282, 25.8012820512821, 140224.358974359, 0, 16.1858974358974, 0, 0, -201923.076923077, 1009615.38461538, -1346153.84615385, 582.692307692308, 0, -89743.5897435897, 34.2948717948718, -140224.358974359, 0, -56.5705128205128, 0, 0, -302884.615384615,
        1210.57692307692, -478.685897435897, 692309.519555765, 656.570512820513, -116.025641025641, 435898.047498495, 17.4679487179487, -40.7051282051282, 0.12415293040293, 224358.974358974, -201923.076923077, 172.275641025641, -1277.88461538462, -221.314102564103, -19232.3282096109, -123.878205128205, 111.378205128205, -147436.002938568, -84.775641025641, -174.679487179487, -0.0608516483516484, -224358.974358974, -134615.384615385, -364.583333333333, 105.769230769231, 163.621794871795, -384615.665234445, -69.7115384615385, 42.4679487179487, 12820.4636029035, -16.1858974358974, 32.8525641025641, 0.0109432234432234, -112179.487179487, -134615.384615385, 0, -38.4615384615385, 536.378205128205, -288461.526111709, 141.826923076923, 19.8717948717949, 102564.20739661, 16.1858974358974, -46.3141025641026, -0.0272893772893773, 112179.487179487, -201923.076923077, 144.230769230769,
        1750000, 201923.076923077, 1019.23076923077, 0, 140224.358974359, 17.4679487179487, 628205.128205128, 0, 404.166666666667, 0, 0, -302884.615384615, -1346153.84615385, 1009615.38461538, -1855.76923076923, 0, -140224.358974359, 185.737179487179, -89743.5897435897, 0, -33.974358974359, 0, 0, -302884.615384615, -1009615.38461538, -1009615.38461538, 253.846153846154, 0, 140224.358974359, -57.8525641025641, 157051.282051282, 168269.230769231, -42.4679487179487, 0, 0, -201923.076923077, 605769.230769231, -201923.076923077, 582.692307692308, 0, -140224.358974359, 110.416666666667, -291666.666666667, 33653.8461538461, -154.647435897436, 0, 0, -201923.076923077,
        -201923.076923077, 1365384.61538462, -1199.03846153846, 140224.358974359, 0, -40.7051282051282, 0, 1333333.33333333, -247.275641025641, 0, 0, 336538.461538462, 1009615.38461538, 48076.923076923, 919.230769230769, -140224.358974359, 0, -80.4487179487179, 0, 551282.051282051, -141.826923076923, 0, 0, 168269.230769231, -1009615.38461538, -721153.846153846, 191.346153846154, 140224.358974359, 0, 41.1858974358974, 168269.230769231, -99358.9743589743, 69.7115384615385, 0, 0, -168269.230769231, 201923.076923077, -692307.692307692, 88.4615384615384, -140224.358974359, 0, -20.9935897435898, -33653.8461538461, -371794.871794872, 87.6602564102564, 0, 0, -336538.461538462,
        1094.39102564103, -798.076923076923, 692309.144672315, 33.3333333333333, -22.9166666666667, 0.12415293040293, 404.166666666667, -247.275641025641, 435897.970633693, -201923.076923077, 224358.974358974, -332.532051282051, -1671.3141025641, 758.653846153846, -288463.320983504, 162.339743589744, -57.8525641025641, 0.138415750915751, -19.551282051282, -165.865384615385, 102564.235922251, -201923.076923077, 112179.487179487, -360.576923076923, 150.160256410256, 116.346153846154, -384615.59536265, -41.1858974358974, 57.8525641025641, -0.0922619047619048, -25.8012820512821, 88.9423076923077, 12820.4636029035, -134615.384615385, -112179.487179487, 0, 426.762820512821, -76.9230769230769, -19230.228326161, 87.8205128205128, -44.3910256410256, 0.151648351648352, -128.044871794872, 121.314102564103, -147436.08781619, -134615.384615385, -224358.974358974, -28.0448717948718,
        0, 0, -1413461.53846154, 0, 0, 224358.974358974, 0, 0, -201923.076923077, 897536.858974359, 42.0673076923077, 650.701322115385, 0, 0, 1413461.53846154, 0, 0, -224358.974358974, 0, 0, 201923.076923077, 448650.641025641, 8.41346153846154, 426.213341346154, 0, 0, 605769.230769231, 0, 0, -112179.487179487, 0, 0, -134615.384615385, 224308.493589744, -42.0673076923077, 134.611738782051, 0, 0, -605769.230769231, 0, 0, 112179.487179487, 0, 0, 134615.384615385, 448734.775641026, -8.41346153846154, 201.935136217949,
        0, 0, -1413461.53846154, 0, 0, -201923.076923077, 0, 0, 224358.974358974, 42.0673076923077, 897536.858974359, -157.047636217949, 0, 0, -605769.230769231, 0, 0, 134615.384615385, 0, 0, 112179.487179487, -8.41346153846154, 448734.775641026, -201.938501602564, 0, 0, 605769.230769231, 0, 0, -134615.384615385, 0, 0, -112179.487179487, -42.0673076923077, 224308.493589744, -134.608373397436, 0, 0, 1413461.53846154, 0, 0, 201923.076923077, 0, 0, -224358.974358974, 8.41346153846154, 448650.641025641, -112.174719551282,
        -2120192.30769231, -2120192.30769231, -2572.11538461538, 336538.461538462, -302884.615384615, 172.275641025641, -302884.615384615, 336538.461538462, -332.532051282051, 650.701322115385, -157.047636217949, 3141071.22528364, 2120192.30769231, -908653.846153846, 4423.07692307692, -336538.461538462, 201923.076923077, -701.121794871795, 302884.615384615, 168269.230769231, 360.576923076923, 426.221754807692, -201.92672275641, 1570502.21831852, 908653.846153846, 908653.846153846, -216.346153846154, -168269.230769231, -201923.076923077, 0, -201923.076923077, -168269.230769231, 0, 134.608373397436, -134.611738782051, 785234.19955199, -908653.846153846, 2120192.30769231, -1634.61538461538, 168269.230769231, 302884.615384615, -144.230769230769, 201923.076923077, -336538.461538462, 308.49358974359, 201.930088141026, -112.183133012821, 1570501.81838431,
        -11076923.0769231, -1009615.38461538, -12303.8461538462, -692307.692307692, -201923.076923077, -1277.88461538462, -1346153.84615385, 1009615.38461538, -1671.3141025641, 0, 0, 2120192.30769231, 15923076.9230769, -5048076.92307692, 18361.5384615385, -1365384.61538462, 201923.076923077, -1345.19230769231, 1750000, 201923.076923077, 1527.08333333333, 0, 0, 2120192.30769231, 1038461.53846154, 1009615.38461538, 619.230769230769, -48076.923076923, -1009615.38461538, 758.653846153846, -605769.230769231, -201923.076923077, -221.314102564103, 0, 0, 908653.846153846, -5884615.38461538, 5048076.92307692, -6676.92307692308, -721153.846153846, 1009615.38461538, -1193.26923076923, 1009615.38461538, -1009615.38461538, 1375.16025641026, 0, 0, 908653.846153846,
        1009615.38461538, 1038461.53846154, -619.230769230769, 201923.076923077, 605769.230769231, -221.314102564103, 1009615.38461538, 48076.923076923, 758.653846153846, 0, 0, -908653.846153846, -5048076.92307692, 15923076.9230769, -18361.5384615385, -201923.076923077, -1750000, 1527.08333333333, -201923.076923077, 1365384.61538462, -1345.19230769231, 0, 0, -2120192.30769231, -1009615.38461538, -11076923.0769231, 12303.8461538462, -1009615.38461538, 1346153.84615385, -1671.3141025641, 201923.076923077, 692307.692307692, -1277.88461538462, 0, 0, -2120192.30769231, 5048076.92307692, -5884615.38461538, 6676.92307692308, 1009615.38461538, -1009615.38461538, 1375.16025641026, -1009615.38461538, 721153.846153846, -1193.26923076923, 0, 0, -908653.846153846,
        -13515.3846153846, -2207.69230769231, -2230782.7718128, -1550, -555.769230769231, -19232.3282096109, -1855.76923076923, 919.230769230769, -288463.320983504, 1413461.53846154, -605769.230769231, 4423.07692307692, 18361.5384615385, -18361.5384615385, 7076959.87216245, -944.230769230769, 1740.38461538462, -692310.871158329, 1740.38461538462, -944.230769230769, 692310.871158329, 1413461.53846154, -1413461.53846154, 6778.84615384615, 2207.69230769231, 13515.3846153846, -2230782.7718128, 919.230769230769, -1855.76923076923, 288463.320983504, -555.769230769231, -1550, 19232.3282096109, 605769.230769231, -1413461.53846154, 4423.07692307692, -7053.84615384615, 7053.84615384615, -2615394.32853685, -1301.92307692308, 1478.84615384615, -384617.321965214, 1478.84615384615, -1301.92307692308, 384617.321965214, 605769.230769231, -605769.230769231, 2067.30769230769,
        692307.692307692, 201923.076923077, 1247.11538461538, -371794.871794872, 33653.8461538461, -123.878205128205, 0, -140224.358974359, 162.339743589744, 0, 0, -336538.461538462, -1365384.61538462, -201923.076923077, -944.230769230769, 1333333.33333333, 0, 656.570512820513, 0, 140224.358974359, -100.641025641026, 0, 0, 336538.461538462, -48076.923076923, 1009615.38461538, -1199.03846153846, 551282.051282051, 0, 165.865384615385, 0, -140224.358974359, 174.679487179487, 0, 0, 168269.230769231, 721153.846153846, -1009615.38461538, 896.153846153846, -99358.9743589743, -168269.230769231, 79.3269230769231, 0, 140224.358974359, -135.416666666667, 0, 0, -168269.230769231,
        -201923.076923077, -605769.230769231, 265.384615384615, -33653.8461538461, -291666.666666667, 111.378205128205, -140224.358974359, 0, -57.8525641025641, 0, 0, 201923.076923077, 201923.076923077, -1750000, 1740.38461538462, 0, 628205.128205128, -476.282051282051, 140224.358974359, 0, 118.429487179487, 0, 0, 302884.615384615, 1009615.38461538, 1346153.84615385, -1019.23076923077, 0, -89743.5897435897, 19.551282051282, -140224.358974359, 0, 84.775641025641, 0, 0, 302884.615384615, -1009615.38461538, 1009615.38461538, -986.538461538461, -168269.230769231, 157051.282051282, -173.878205128205, 140224.358974359, 0, 110.416666666667, 0, 0, 201923.076923077,
        1412.5, 599.839743589744, 19232.0990429442, -90.224358974359, 171.314102564103, -147436.002938568, 185.737179487179, -80.4487179487179, 0.138415750915751, -224358.974358974, 134615.384615385, -701.121794871795, -1345.19230769231, 1527.08333333333, -692310.871158329, 656.570512820513, -476.282051282051, 435898.433395931, -118.429487179487, 100.641025641026, -0.238255494505494, 224358.974358974, 201923.076923077, -332.532051282051, -1038.46153846154, -1065.54487179487, 288462.17963735, 141.826923076923, 33.974358974359, 102564.235922251, 152.083333333333, 95.0320512820513, -0.0608516483516484, 112179.487179487, 201923.076923077, -360.576923076923, 971.153846153846, -1061.37820512821, 384616.592478035, 98.5576923076923, -190.544871794872, 12820.6799490574, -152.083333333333, 127.083333333333, -0.161263736263736, -112179.487179487, 134615.384615385, -336.538461538462,
        -1346153.84615385, -1009615.38461538, -1019.23076923077, 0, -140224.358974359, -84.775641025641, -89743.5897435897, 0, -19.551282051282, 0, 0, 302884.615384615, 1750000, -201923.076923077, 1740.38461538462, 0, 140224.358974359, -118.429487179487, 628205.128205128, 0, 476.282051282051, 0, 0, 302884.615384615, 605769.230769231, 201923.076923077, 265.384615384615, 0, -140224.358974359, 57.8525641025641, -291666.666666667, -33653.8461538461, -111.378205128205, 0, 0, 201923.076923077, -1009615.38461538, 1009615.38461538, -986.538461538461, 0, 140224.358974359, -110.416666666667, 157051.282051282, -168269.230769231, 173.878205128205, 0, 0, 201923.076923077,
        -1009615.38461538, 48076.923076923, -1199.03846153846, -140224.358974359, 0, -174.679487179487, 0, 551282.051282051, -165.865384615385, 0, 0, 168269.230769231, 201923.076923077, 1365384.61538462, -944.230769230769, 140224.358974359, 0, 100.641025641026, 0, 1333333.33333333, -656.570512820513, 0, 0, 336538.461538462, -201923.076923077, -692307.692307692, 1247.11538461538, -140224.358974359, 0, -162.339743589744, 33653.8461538461, -371794.871794872, 123.878205128205, 0, 0, -336538.461538462, 1009615.38461538, -721153.846153846, 896.153846153846, 140224.358974359, 0, 135.416666666667, -168269.230769231, -99358.9743589743, -79.3269230769231, 0, 0, -168269.230769231,
        -1065.54487179487, -1038.46153846154, -288462.17963735, -95.0320512820513, -152.083333333333, -0.0608516483516484, -33.974358974359, -141.826923076923, 102564.235922251, 201923.076923077, 112179.487179487, 360.576923076923, 1527.08333333333, -1345.19230769231, 692310.871158329, -100.641025641026, 118.429487179487, -0.238255494505494, 476.282051282051, -656.570512820513, 435898.433395931, 201923.076923077, 224358.974358974, 332.532051282051, 599.839743589744, 1412.5, -19232.0990429442, 80.4487179487179, -185.737179487179, 0.138415750915751, -171.314102564103, 90.224358974359, -147436.002938568, 134615.384615385, -224358.974358974, 701.121794871795, -1061.37820512821, 971.153846153846, -384616.592478035, -127.083333333333, 152.083333333333, -0.161263736263736, 190.544871794872, -98.5576923076923, 12820.6799490574, 134615.384615385, -112179.487179487, 336.538461538462,
        0, 0, -1413461.53846154, 0, 0, -224358.974358974, 0, 0, -201923.076923077, 448650.641025641, -8.41346153846154, 426.221754807692, 0, 0, 1413461.53846154, 0, 0, 224358.974358974, 0, 0, 201923.076923077, 897536.858974359, -42.0673076923077, 650.743389423077, 0, 0, 605769.230769231, 0, 0, 112179.487179487, 0, 0, -134615.384615385, 448734.775641026, 8.41346153846154, 201.92672275641, 0, 0, -605769.230769231, 0, 0, -112179.487179487, 0, 0, 134615.384615385, 224308.493589744, 42.0673076923077, 134.569671474359,
        0, 0, -605769.230769231, 0, 0, -134615.384615385, 0, 0, 112179.487179487, 8.41346153846154, 448734.775641026, -201.92672275641, 0, 0, -1413461.53846154, 0, 0, 201923.076923077, 0, 0, 224358.974358974, -42.0673076923077, 897536.858974359, -650.743389423077, 0, 0, 1413461.53846154, 0, 0, -201923.076923077, 0, 0, -224358.974358974, -8.41346153846154, 448650.641025641, -426.221754807692, 0, 0, 605769.230769231, 0, 0, 134615.384615385, 0, 0, -112179.487179487, 42.0673076923077, 224308.493589744, -134.569671474359,
        -2120192.30769231, -908653.846153846, -2572.11538461538, -336538.461538462, -201923.076923077, -364.583333333333, -302884.615384615, 168269.230769231, -360.576923076923, 426.213341346154, -201.938501602564, 1570502.21831852, 2120192.30769231, -2120192.30769231, 6778.84615384615, 336538.461538462, 302884.615384615, -332.532051282051, 302884.615384615, 336538.461538462, 332.532051282051, 650.743389423077, -650.743389423077, 3141071.76769919, 908653.846153846, 2120192.30769231, -2572.11538461538, 168269.230769231, -302884.615384615, 360.576923076923, -201923.076923077, -336538.461538462, 364.583333333333, 201.938501602564, -426.213341346154, 1570502.21831852, -908653.846153846, 908653.846153846, -1634.61538461538, -168269.230769231, 201923.076923077, -336.538461538462, 201923.076923077, -168269.230769231, 336.538461538462, 134.566306089744, -134.566306089744, 785234.199509922,
        -5884615.38461538, -5048076.92307692, -996.153846153846, -721153.846153846, -1009615.38461538, 105.769230769231, -1009615.38461538, -1009615.38461538, 150.160256410256, 0, 0, 908653.846153846, 1038461.53846154, -1009615.38461538, 2207.69230769231, -48076.923076923, 1009615.38461538, -1038.46153846154, 605769.230769231, -201923.076923077, 599.839743589744, 0, 0, 908653.846153846, 15923076.9230769, 5048076.92307692, 619.230769230769, -1365384.61538462, -201923.076923077, -798.076923076923, -1750000, 201923.076923077, -478.685897435897, 0, 0, 2120192.30769231, -11076923.0769231, 1009615.38461538, -1830.76923076923, -692307.692307692, 201923.076923077, -57.6923076923076, 1346153.84615385, 1009615.38461538, -69.3910256410257, 0, 0, 2120192.30769231,
        -5048076.92307692, -5884615.38461538, 619.230769230769, -1009615.38461538, -1009615.38461538, 163.621794871795, -1009615.38461538, -721153.846153846, 116.346153846154, 0, 0, 908653.846153846, 1009615.38461538, -11076923.0769231, 13515.3846153846, 1009615.38461538, 1346153.84615385, -1065.54487179487, 201923.076923077, -692307.692307692, 1412.5, 0, 0, 2120192.30769231, 5048076.92307692, 15923076.9230769, -12303.8461538462, 201923.076923077, -1750000, 1094.39102564103, -201923.076923077, -1365384.61538462, 1210.57692307692, 0, 0, 2120192.30769231, -1009615.38461538, 1038461.53846154, -1830.76923076923, -201923.076923077, 605769.230769231, -394.391025641026, 1009615.38461538, -48076.923076923, 318.269230769231, 0, 0, 908653.846153846,
        -619.230769230769, 996.153846153846, -2615387.05930608, 214.423076923077, 238.461538461538, -384615.665234445, 253.846153846154, 191.346153846154, -384615.59536265, 605769.230769231, 605769.230769231, -216.346153846154, 619.230769230769, 12303.8461538462, -2230782.7718128, -1199.03846153846, -1019.23076923077, 288462.17963735, 265.384615384615, 1247.11538461538, -19232.0990429442, 605769.230769231, 1413461.53846154, -2572.11538461538, 619.230769230769, -12303.8461538462, 7076937.95957504, -1199.03846153846, 1019.23076923077, -692309.144672315, -265.384615384615, 1247.11538461538, -692309.519555765, 1413461.53846154, 1413461.53846154, -2572.11538461538, -619.230769230769, -996.153846153846, -2230768.12845616, 214.423076923077, -238.461538461538, -19230.39980052, -253.846153846154, 191.346153846154, 288461.22515017, 1413461.53846154, 605769.230769231, -216.346153846154,
        721153.846153846, 1009615.38461538, 191.346153846154, -99358.9743589743, 168269.230769231, -69.7115384615385, 0, 140224.358974359, -41.1858974358974, 0, 0, -168269.230769231, -48076.923076923, -1009615.38461538, 919.230769230769, 551282.051282051, 0, 141.826923076923, 0, -140224.358974359, 80.4487179487179, 0, 0, 168269.230769231, -1365384.61538462, 201923.076923077, -1199.03846153846, 1333333.33333333, 0, 247.275641025641, 0, 140224.358974359, 40.7051282051282, 0, 0, 336538.461538462, 692307.692307692, -201923.076923077, 88.4615384615384, -371794.871794872, -33653.8461538461, -87.6602564102564, 0, -140224.358974359, 20.9935897435898, 0, 0, -336538.461538462,
        1009615.38461538, 1009615.38461538, 253.846153846154, 168269.230769231, 157051.282051282, 42.4679487179487, 140224.358974359, 0, 57.8525641025641, 0, 0, -201923.076923077, -1009615.38461538, 1346153.84615385, -1855.76923076923, 0, -89743.5897435897, 33.974358974359, -140224.358974359, 0, -185.737179487179, 0, 0, -302884.615384615, -201923.076923077, -1750000, 1019.23076923077, 0, 628205.128205128, -404.166666666667, 140224.358974359, 0, -17.4679487179487, 0, 0, -302884.615384615, 201923.076923077, -605769.230769231, 582.692307692308, 33653.8461538461, -291666.666666667, 154.647435897436, -140224.358974359, 0, -110.416666666667, 0, 0, -201923.076923077,
        116.346153846154, 150.160256410256, 384615.59536265, -88.9423076923077, 25.8012820512821, 12820.4636029035, -57.8525641025641, 41.1858974358974, -0.0922619047619048, -112179.487179487, -134615.384615385, 0, 758.653846153846, -1671.3141025641, 288463.320983504, 165.865384615385, 19.551282051282, 102564.235922251, 57.8525641025641, -162.339743589744, 0.138415750915751, 112179.487179487, -201923.076923077, 360.576923076923, -798.076923076923, 1094.39102564103, -692309.144672315, 247.275641025641, -404.166666666667, 435897.970633693, 22.9166666666667, -33.3333333333333, 0.12415293040293, 224358.974358974, -201923.076923077, 332.532051282051, -76.9230769230769, 426.762820512821, 19230.228326161, -121.314102564103, 128.044871794872, -147436.08781619, 44.3910256410256, -87.8205128205128, 0.151648351648352, -224358.974358974, -134615.384615385, 28.0448717948718,
        1009615.38461538, 1009615.38461538, 238.461538461538, 0, 140224.358974359, -16.1858974358974, 157051.282051282, 168269.230769231, -25.8012820512821, 0, 0, -201923.076923077, -605769.230769231, 201923.076923077, -555.769230769231, 0, -140224.358974359, 152.083333333333, -291666.666666667, 33653.8461538461, -171.314102564103, 0, 0, -201923.076923077, -1750000, -201923.076923077, -265.384615384615, 0, 140224.358974359, 22.9166666666667, 628205.128205128, 0, 116.025641025641, 0, 0, -302884.615384615, 1346153.84615385, -1009615.38461538, 582.692307692308, 0, -140224.358974359, 56.5705128205128, -89743.5897435897, 0, -34.2948717948718, 0, 0, -302884.615384615,
        1009615.38461538, 721153.846153846, 214.423076923077, 140224.358974359, 0, 32.8525641025641, 168269.230769231, -99358.9743589743, 88.9423076923077, 0, 0, -168269.230769231, -201923.076923077, 692307.692307692, -1550, -140224.358974359, 0, 95.0320512820513, -33653.8461538461, -371794.871794872, 90.224358974359, 0, 0, -336538.461538462, 201923.076923077, -1365384.61538462, 1247.11538461538, 140224.358974359, 0, -33.3333333333333, 0, 1333333.33333333, -656.570512820513, 0, 0, 336538.461538462, -1009615.38461538, -48076.923076923, 88.4615384615384, -140224.358974359, 0, 6.4102564102564, 0, 551282.051282051, -165.865384615385, 0, 0, 168269.230769231,
        163.621794871795, 105.769230769231, 384615.665234445, -32.8525641025641, 16.1858974358974, 0.0109432234432234, -42.4679487179487, 69.7115384615385, 12820.4636029035, -134615.384615385, -112179.487179487, 0, -221.314102564103, -1277.88461538462, 19232.3282096109, 174.679487179487, 84.775641025641, -0.0608516483516484, -111.378205128205, 123.878205128205, -147436.002938568, -134615.384615385, -224358.974358974, 364.583333333333, -478.685897435897, 1210.57692307692, -692309.519555765, 40.7051282051282, -17.4679487179487, 0.12415293040293, 116.025641025641, -656.570512820513, 435898.047498495, -201923.076923077, 224358.974358974, -172.275641025641, 536.378205128205, -38.4615384615385, 288461.526111709, 46.3141025641026, -16.1858974358974, -0.0272893772893773, -19.8717948717949, -141.826923076923, 102564.20739661, -201923.076923077, 112179.487179487, -144.230769230769,
        0, 0, -605769.230769231, 0, 0, -112179.487179487, 0, 0, -134615.384615385, 224308.493589744, -42.0673076923077, 134.608373397436, 0, 0, 605769.230769231, 0, 0, 112179.487179487, 0, 0, 134615.384615385, 448734.775641026, -8.41346153846154, 201.938501602564, 0, 0, 1413461.53846154, 0, 0, 224358.974358974, 0, 0, -201923.076923077, 897536.858974359, 42.0673076923077, 157.047636217949, 0, 0, -1413461.53846154, 0, 0, -224358.974358974, 0, 0, 201923.076923077, 448650.641025641, 8.41346153846154, 112.174719551282,
        0, 0, -605769.230769231, 0, 0, -134615.384615385, 0, 0, -112179.487179487, -42.0673076923077, 224308.493589744, -134.611738782051, 0, 0, -1413461.53846154, 0, 0, 201923.076923077, 0, 0, -224358.974358974, 8.41346153846154, 448650.641025641, -426.213341346154, 0, 0, 1413461.53846154, 0, 0, -201923.076923077, 0, 0, 224358.974358974, 42.0673076923077, 897536.858974359, -650.701322115385, 0, 0, 605769.230769231, 0, 0, 134615.384615385, 0, 0, 112179.487179487, -8.41346153846154, 448734.775641026, -201.935136217949,
        -908653.846153846, -908653.846153846, -216.346153846154, -168269.230769231, -201923.076923077, 0, -201923.076923077, -168269.230769231, 0, 134.611738782051, -134.608373397436, 785234.19955199, 908653.846153846, -2120192.30769231, 4423.07692307692, 168269.230769231, 302884.615384615, -360.576923076923, 201923.076923077, -336538.461538462, 701.121794871795, 201.92672275641, -426.221754807692, 1570502.21831852, 2120192.30769231, 2120192.30769231, -2572.11538461538, 336538.461538462, -302884.615384615, 332.532051282051, -302884.615384615, 336538.461538462, -172.275641025641, 157.047636217949, -650.701322115385, 3141071.22528364, -2120192.30769231, 908653.846153846, -1634.61538461538, -336538.461538462, 201923.076923077, -308.49358974359, 302884.615384615, 168269.230769231, 144.230769230769, 112.183133012821, -201.930088141026, 1570501.81838431,
        1038461.53846154, 1009615.38461538, 996.153846153846, 48076.923076923, 1009615.38461538, -38.4615384615385, 605769.230769231, 201923.076923077, 426.762820512821, 0, 0, -908653.846153846, -5884615.38461538, 5048076.92307692, -7053.84615384615, 721153.846153846, -1009615.38461538, 971.153846153846, -1009615.38461538, 1009615.38461538, -1061.37820512821, 0, 0, -908653.846153846, -11076923.0769231, -1009615.38461538, -619.230769230769, 692307.692307692, 201923.076923077, -76.9230769230769, 1346153.84615385, -1009615.38461538, 536.378205128205, 0, 0, -2120192.30769231, 15923076.9230769, -5048076.92307692, 6676.92307692308, 1365384.61538462, -201923.076923077, 932.692307692308, -1750000, -201923.076923077, -911.378205128205, 0, 0, -2120192.30769231,
        -1009615.38461538, -11076923.0769231, 619.230769230769, 1009615.38461538, -1346153.84615385, 536.378205128205, -201923.076923077, -692307.692307692, -76.9230769230769, 0, 0, 2120192.30769231, 5048076.92307692, -5884615.38461538, 7053.84615384615, -1009615.38461538, 1009615.38461538, -1061.37820512821, 1009615.38461538, -721153.846153846, 971.153846153846, 0, 0, 908653.846153846, 1009615.38461538, 1038461.53846154, -996.153846153846, -201923.076923077, -605769.230769231, 426.762820512821, -1009615.38461538, -48076.923076923, -38.4615384615385, 0, 0, 908653.846153846, -5048076.92307692, 15923076.9230769, -6676.92307692308, 201923.076923077, 1750000, -911.378205128205, 201923.076923077, -1365384.61538462, 932.692307692308, 0, 0, 2120192.30769231,
        1830.76923076923, 1830.76923076923, -2230768.12845616, 88.4615384615384, 582.692307692308, -288461.526111709, 582.692307692308, 88.4615384615384, -19230.228326161, -605769.230769231, 1413461.53846154, -1634.61538461538, -6676.92307692308, 6676.92307692308, -2615394.32853685, 896.153846153846, -986.538461538461, 384616.592478035, -986.538461538461, 896.153846153846, -384616.592478035, -605769.230769231, 605769.230769231, -1634.61538461538, -1830.76923076923, -1830.76923076923, -2230768.12845616, 88.4615384615384, 582.692307692308, 19230.228326161, 582.692307692308, 88.4615384615384, 288461.526111709, -1413461.53846154, 605769.230769231, -1634.61538461538, 6676.92307692308, -6676.92307692308, 7076930.58544917, 896.153846153846, -986.538461538461, 692308.946915905, -986.538461538461, 896.153846153846, -692308.946915905, -1413461.53846154, 1413461.53846154, -1634.61538461538,
        48076.923076923, -1009615.38461538, 191.346153846154, 551282.051282051, 0, 141.826923076923, 0, -140224.358974359, 87.8205128205128, 0, 0, 168269.230769231, -721153.846153846, 1009615.38461538, -1301.92307692308, -99358.9743589743, -168269.230769231, 98.5576923076923, 0, 140224.358974359, -127.083333333333, 0, 0, -168269.230769231, -692307.692307692, -201923.076923077, 214.423076923077, -371794.871794872, 33653.8461538461, -121.314102564103, 0, -140224.358974359, 46.3141025641026, 0, 0, -336538.461538462, 1365384.61538462, 201923.076923077, 896.153846153846, 1333333.33333333, 0, 247.275641025641, 0, 140224.358974359, -108.012820512821, 0, 0, 336538.461538462,
        -1009615.38461538, -1346153.84615385, -253.846153846154, 0, -89743.5897435897, 19.8717948717949, -140224.358974359, 0, -44.3910256410256, 0, 0, 302884.615384615, 1009615.38461538, -1009615.38461538, 1478.84615384615, -168269.230769231, 157051.282051282, -190.544871794872, 140224.358974359, 0, 152.083333333333, 0, 0, 201923.076923077, 201923.076923077, 605769.230769231, -238.461538461538, -33653.8461538461, -291666.666666667, 128.044871794872, -140224.358974359, 0, -16.1858974358974, 0, 0, 201923.076923077, -201923.076923077, 1750000, -986.538461538461, 0, 628205.128205128, -188.141025641026, 140224.358974359, 0, 123.878205128205, 0, 0, 302884.615384615,
        318.269230769231, -69.3910256410257, -288461.22515017, 165.865384615385, 34.2948717948718, 102564.20739661, 110.416666666667, -20.9935897435898, 0.151648351648352, 112179.487179487, 201923.076923077, -144.230769230769, -1193.26923076923, 1375.16025641026, -384617.321965214, 79.3269230769231, -173.878205128205, 12820.6799490574, -110.416666666667, 135.416666666667, -0.161263736263736, -112179.487179487, 134615.384615385, -336.538461538462, -57.6923076923076, -394.391025641026, -19230.39980052, -87.6602564102564, 154.647435897436, -147436.08781619, 56.5705128205128, 6.4102564102564, -0.0272893772893773, -224358.974358974, 134615.384615385, -308.49358974359, 932.692307692308, -911.378205128205, 692308.946915905, 247.275641025641, -188.141025641026, 435897.728967027, -123.878205128205, 108.012820512821, -0.211973443223443, 224358.974358974, 201923.076923077, -172.275641025641,
        -605769.230769231, -201923.076923077, -238.461538461538, 0, -140224.358974359, 16.1858974358974, -291666.666666667, -33653.8461538461, -128.044871794872, 0, 0, 201923.076923077, 1009615.38461538, -1009615.38461538, 1478.84615384615, 0, 140224.358974359, -152.083333333333, 157051.282051282, -168269.230769231, 190.544871794872, 0, 0, 201923.076923077, 1346153.84615385, 1009615.38461538, -253.846153846154, 0, -140224.358974359, 44.3910256410256, -89743.5897435897, 0, -19.8717948717949, 0, 0, 302884.615384615, -1750000, 201923.076923077, -986.538461538461, 0, 140224.358974359, -123.878205128205, 628205.128205128, 0, 188.141025641026, 0, 0, 302884.615384615,
        201923.076923077, 692307.692307692, 214.423076923077, -140224.358974359, 0, -46.3141025641026, 33653.8461538461, -371794.871794872, 121.314102564103, 0, 0, -336538.461538462, -1009615.38461538, 721153.846153846, -1301.92307692308, 140224.358974359, 0, 127.083333333333, -168269.230769231, -99358.9743589743, -98.5576923076923, 0, 0, -168269.230769231, 1009615.38461538, -48076.923076923, 191.346153846154, -140224.358974359, 0, -87.8205128205128, 0, 551282.051282051, -141.826923076923, 0, 0, 168269.230769231, -201923.076923077, -1365384.61538462, 896.153846153846, 140224.358974359, 0, 108.012820512821, 0, 1333333.33333333, -247.275641025641, 0, 0, 336538.461538462,
        -394.391025641026, -57.6923076923076, 19230.39980052, -6.4102564102564, -56.5705128205128, -0.0272893772893773, -154.647435897436, 87.6602564102564, -147436.08781619, 134615.384615385, -224358.974358974, 308.49358974359, 1375.16025641026, -1193.26923076923, 384617.321965214, -135.416666666667, 110.416666666667, -0.161263736263736, 173.878205128205, -79.3269230769231, 12820.6799490574, 134615.384615385, -112179.487179487, 336.538461538462, -69.3910256410257, 318.269230769231, 288461.22515017, 20.9935897435898, -110.416666666667, 0.151648351648352, -34.2948717948718, -165.865384615385, 102564.20739661, 201923.076923077, 112179.487179487, 144.230769230769, -911.378205128205, 932.692307692308, -692308.946915905, -108.012820512821, 123.878205128205, -0.211973443223443, 188.141025641026, -247.275641025641, 435897.728967027, 201923.076923077, 224358.974358974, 172.275641025641,
        0, 0, -605769.230769231, 0, 0, 112179.487179487, 0, 0, -134615.384615385, 448734.775641026, 8.41346153846154, 201.930088141026, 0, 0, 605769.230769231, 0, 0, -112179.487179487, 0, 0, 134615.384615385, 224308.493589744, 42.0673076923077, 134.566306089744, 0, 0, 1413461.53846154, 0, 0, -224358.974358974, 0, 0, -201923.076923077, 448650.641025641, -8.41346153846154, 112.183133012821, 0, 0, -1413461.53846154, 0, 0, 224358.974358974, 0, 0, 201923.076923077, 897536.858974359, -42.0673076923077, 157.089703525641,
        0, 0, -1413461.53846154, 0, 0, -201923.076923077, 0, 0, -224358.974358974, -8.41346153846154, 448650.641025641, -112.183133012821, 0, 0, -605769.230769231, 0, 0, 134615.384615385, 0, 0, -112179.487179487, 42.0673076923077, 224308.493589744, -134.566306089744, 0, 0, 605769.230769231, 0, 0, -134615.384615385, 0, 0, 112179.487179487, 8.41346153846154, 448734.775641026, -201.930088141026, 0, 0, 1413461.53846154, 0, 0, 201923.076923077, 0, 0, 224358.974358974, -42.0673076923077, 897536.858974359, -157.089703525641,
        -908653.846153846, -2120192.30769231, -216.346153846154, 168269.230769231, -302884.615384615, 144.230769230769, -201923.076923077, -336538.461538462, -28.0448717948718, 201.935136217949, -112.174719551282, 1570501.81838431, 908653.846153846, -908653.846153846, 2067.30769230769, -168269.230769231, 201923.076923077, -336.538461538462, 201923.076923077, -168269.230769231, 336.538461538462, 134.569671474359, -134.569671474359, 785234.199509922, 2120192.30769231, 908653.846153846, -216.346153846154, -336538.461538462, -201923.076923077, 28.0448717948718, -302884.615384615, 168269.230769231, -144.230769230769, 112.174719551282, -201.935136217949, 1570501.81838431, -2120192.30769231, 2120192.30769231, -1634.61538461538, 336538.461538462, 302884.615384615, -172.275641025641, 302884.615384615, 336538.461538462, 172.275641025641, 157.089703525641, -157.089703525641, 3141070.68295223;


    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation

    //Setup the test conditions
    ChVector<double> OriginalPos = m_nodeB->GetPos();
    m_nodeB->SetPos(ChVector<>(m_nodeB->GetPos().x(), m_nodeB->GetPos().y(), 0.001));
    m_element->SetAlphaDamp(0.01);

    ChVectorDynamic<double> InternalForceSmallDispNoVelNoGravity;
    InternalForceSmallDispNoVelNoGravity.resize(48);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceSmallDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_SmallDispNoVelWithDamping;
    JacobianK_SmallDispNoVelWithDamping.resize(48, 48);
    m_element->ComputeKRMmatricesGlobal(JacobianK_SmallDispNoVelWithDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_SmallDispNoVelWithDamping;
    JacobianR_SmallDispNoVelWithDamping.resize(48, 48);
    m_element->ComputeKRMmatricesGlobal(JacobianR_SmallDispNoVelWithDamping, 0, 1, 0);

    //Reset the element conditions back to its original values
    m_nodeB->SetPos(OriginalPos);
    m_element->SetAlphaDamp(0.0);


    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_SmallDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(48, 48);

    double small_terms_JacR = 1e-4*Expected_JacobianR_SmallDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacR = 0;
    double max_percent_error_JacR = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacR;
    percent_error_matrix_JacR.resize(48, 48);


    for (auto i = 0; i < Expected_JacobianK_SmallDispNoVelWithDamping.rows(); i++) {
        for (auto j = 0; j < Expected_JacobianK_SmallDispNoVelWithDamping.cols(); j++) {
            if (std::abs(Expected_JacobianK_SmallDispNoVelWithDamping(i, j)) < small_terms_JacK) {
                double error = std::abs(JacobianK_SmallDispNoVelWithDamping(i, j) - Expected_JacobianK_SmallDispNoVelWithDamping(i, j));
                if (error > zeros_max_error_JacK)
                    zeros_max_error_JacK = error;
                percent_error_matrix_JacK(i, j) = 0.0;
            }
            else {
                double percent_error = std::abs((JacobianK_SmallDispNoVelWithDamping(i, j) - Expected_JacobianK_SmallDispNoVelWithDamping(i, j)) / Expected_JacobianK_SmallDispNoVelWithDamping(i, j));
                if (percent_error > max_percent_error_JacK)
                    max_percent_error_JacK = percent_error;
                percent_error_matrix_JacK(i, j) = percent_error;
            }

            if (std::abs(Expected_JacobianR_SmallDispNoVelWithDamping(i, j)) < small_terms_JacR) {
                double error = std::abs(JacobianR_SmallDispNoVelWithDamping(i, j) - Expected_JacobianR_SmallDispNoVelWithDamping(i, j));
                if (error > zeros_max_error_JacR)
                    zeros_max_error_JacR = error;
                percent_error_matrix_JacR(i, j) = 0.0;
            }
            else {
                double percent_error = std::abs((JacobianR_SmallDispNoVelWithDamping(i, j) - Expected_JacobianR_SmallDispNoVelWithDamping(i, j)) / Expected_JacobianR_SmallDispNoVelWithDamping(i, j));
                if (percent_error > max_percent_error_JacR)
                    max_percent_error_JacR = percent_error;
                percent_error_matrix_JacR(i, j) = percent_error;
            }
        }
    }


    if (msglvl >= 2) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Jacobian K Term - Small Displacement, No Velocity, With Damping = " << std::endl;
        std::cout << JacobianK_SmallDispNoVelWithDamping << std::endl;
        std::cout << "Expected Jacobian K Term - Small Displacement, No Velocity, With Damping = " << std::endl;
        std::cout << Expected_JacobianK_SmallDispNoVelWithDamping << std::endl;
        std::cout << "Percent Error Jacobian K Term - Small Displacement, No Velocity, With Damping = " << std::endl;
        std::cout << percent_error_matrix_JacK << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Jacobian K Term - Small Displacement, No Velocity, With Damping (Max Abs Error) = "
            << (JacobianK_SmallDispNoVelWithDamping - Expected_JacobianK_SmallDispNoVelWithDamping).cwiseAbs().maxCoeff() << std::endl;

        std::cout << "Jacobian K Term - Small Displacement, No Velocity, With Damping (Max Abs Error - Only Smaller Terms) = "
            << zeros_max_error_JacK;
        if (zeros_max_error_JacK / JacobianK_SmallDispNoVelWithDamping.cwiseAbs().maxCoeff() > 0.0001) {
            print_red(" - Test FAILED\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl;
        }

        std::cout << "Jacobian K Term - Small Displacement, No Velocity, With Damping (Max Abs % Error - Only Larger Terms) = "
            << max_percent_error_JacK * 100 << "%";
        if (max_percent_error_JacK > 0.01) {
            print_red(" - Test FAILED\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl;
        }
    }
    if (zeros_max_error_JacK / JacobianK_SmallDispNoVelWithDamping.cwiseAbs().maxCoeff() > 0.0001) {
        passed_tests = false;
    }
    if (max_percent_error_JacK > 0.01) {
        passed_tests = false;
    }

    if (msglvl >= 2) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Jacobian R Term - Small Displacement, No Velocity, With Damping = " << std::endl;
        std::cout << JacobianR_SmallDispNoVelWithDamping << std::endl;
        std::cout << "Expected Jacobian R Term - Small Displacement, No Velocity, With Damping = " << std::endl;
        std::cout << Expected_JacobianR_SmallDispNoVelWithDamping << std::endl;
        std::cout << "Percent Error Jacobian R Term - Small Displacement, No Velocity, With Damping = " << std::endl;
        std::cout << percent_error_matrix_JacR << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Jacobian R Term - Small Displacement, No Velocity, With Damping (Max Abs Error) = "
            << (JacobianR_SmallDispNoVelWithDamping - Expected_JacobianR_SmallDispNoVelWithDamping).cwiseAbs().maxCoeff() << std::endl;

        std::cout << "Jacobian R Term - Small Displacement, No Velocity, With Damping (Max Abs Error - Only Smaller Terms) = "
            << zeros_max_error_JacR;
        if (zeros_max_error_JacR / JacobianR_SmallDispNoVelWithDamping.cwiseAbs().maxCoeff() > 0.0001) {
            print_red(" - Test FAILED\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl;
        }

        std::cout << "Jacobian R Term - Small Displacement, No Velocity, With Damping (Max Abs % Error - Only Larger Terms) = "
            << max_percent_error_JacR * 100 << "%";
        if (max_percent_error_JacR > 0.01) {
            print_red(" - Test FAILED\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl;
        }
    }
    if (zeros_max_error_JacR / JacobianR_SmallDispNoVelWithDamping.cwiseAbs().maxCoeff() > 0.0001) {
        passed_tests = false;
    }
    if (max_percent_error_JacR > 0.01) {
        passed_tests = false;
    }

    return(passed_tests);
}

template <typename ElementVersion, typename MaterialVersion>
bool ANCFShellTest<ElementVersion, MaterialVersion>::JacobianNoDispSmallVelWithDampingCheck(int msglvl) {
    // =============================================================================
        //  Check the Jacobian at No Displacement Small Velocity - With Damping
        //  (some small error expected depending on the formulation/steps used)
        // =============================================================================

    ChMatrixNM<double, 48, 48> Expected_JacobianK_NoDispSmallVelWithDamping;
    Expected_JacobianK_NoDispSmallVelWithDamping <<
        1592307692.30769, 504807692.307692, 12303.8461538462, 136538461.538462, 20192307.6923077, 1210.57692307692, 175000000, -20192307.6923077, 1094.39102564103, -1028.84615384615, 0, -212019230.769231, -1107692307.69231, 100961538.461538, -13515.3846153846, 69230769.2307692, -20192307.6923077, 1412.5, -134615384.615385, -100961538.461538, -1065.54487179487, -1028.84615384615, 0, -212019230.769231, -588461538.461538, -504807692.307692, -619.230769230769, 72115384.6153846, 100961538.461538, 116.346153846154, 100961538.461538, 100961538.461538, 163.621794871795, -86.5384615384616, 0, -90865384.6153846, 103846153.846154, -100961538.461538, 1830.76923076923, 4807692.3076923, -100961538.461538, 318.269230769231, -60576923.0769231, 20192307.6923077, -394.391025641026, -86.5384615384616, 0, -90865384.6153846,
        504807692.307692, 1592307692.30769, -619.230769230769, -20192307.6923077, 175000000, -478.685897435897, 20192307.6923077, 136538461.538462, -798.076923076923, 0, -1028.84615384615, -212019230.769231, -100961538.461538, 103846153.846154, -2207.69230769231, 20192307.6923077, -60576923.0769231, 599.839743589744, -100961538.461538, 4807692.3076923, -1038.46153846154, 0, -1028.84615384615, -90865384.6153846, -504807692.307692, -588461538.461538, 996.153846153846, 100961538.461538, 100961538.461538, 150.160256410256, 100961538.461538, 72115384.6153846, 105.769230769231, 0, -86.5384615384616, -90865384.6153846, 100961538.461538, -1107692307.69231, 1830.76923076923, -100961538.461538, -134615384.615385, -69.3910256410257, -20192307.6923077, 69230769.2307692, -57.6923076923076, 0, -86.5384615384616, -212019230.769231,
        0, 0, 707692307.692308, 0, 0, 69230769.2307692, 0, 0, 69230769.2307692, -141346153.846154, -141346153.846154, -2057.69230769231, 0, 0, -223076923.076923, 0, 0, 1923076.92307692, 0, 0, -28846153.8461538, -141346153.846154, -60576923.0769231, -2057.69230769231, 0, 0, -261538461.538462, 0, 0, 38461538.4615385, 0, 0, 38461538.4615385, -60576923.0769231, -60576923.0769231, -173.076923076923, 0, 0, -223076923.076923, 0, 0, -28846153.8461538, 0, 0, 1923076.92307692, -60576923.0769231, -141346153.846154, -173.076923076923,
        136538461.538462, -20192307.6923077, 1247.11538461538, 133333333.333333, 0, 656.570512820513, 0, 14022435.8974359, 33.3333333333333, 68.9102564102564, 0, 33653846.1538462, -69230769.2307692, 20192307.6923077, -1550, -37179487.1794872, -3365384.61538461, -90.224358974359, 0, -14022435.8974359, -95.0320512820513, -145.833333333333, 0, -33653846.1538462, -72115384.6153846, -100961538.461538, 214.423076923077, -9935897.43589743, 16826923.0769231, -88.9423076923077, 0, 14022435.8974359, -32.8525641025641, 0, 0, -16826923.0769231, 4807692.3076923, 100961538.461538, 88.4615384615384, 55128205.1282051, 0, 165.865384615385, 0, -14022435.8974359, -6.4102564102564, 57.6923076923077, 0, 16826923.0769231,
        20192307.6923077, 175000000, -265.384615384615, 0, 62820512.8205128, -116.025641025641, 14022435.8974359, 0, -22.9166666666667, 0, 68.9102564102564, -30288461.5384615, -20192307.6923077, 60576923.0769231, -555.769230769231, 3365384.61538461, -29166666.6666667, 171.314102564103, -14022435.8974359, 0, -152.083333333333, 0, -145.833333333333, -20192307.6923077, -100961538.461538, -100961538.461538, 238.461538461538, 16826923.0769231, 15705128.2051282, 25.8012820512821, 14022435.8974359, 0, 16.1858974358974, 0, 0, -20192307.6923077, 100961538.461538, -134615384.615385, 582.692307692308, 0, -8974358.97435897, 34.2948717948718, -14022435.8974359, 0, -56.5705128205128, 0, 57.6923076923077, -30288461.5384615,
        0, 0, 69230769.2307692, 0, 0, 43589743.5897436, 0, 0, 0, 22435897.4358974, -20192307.6923077, 137.820512820513, 0, 0, -1923076.92307692, 0, 0, -14743589.7435897, 0, 0, 0, -22435897.4358974, -13461538.4615385, -291.666666666667, 0, 0, -38461538.4615385, 0, 0, 1282051.28205128, 0, 0, 0, -11217948.7179487, -13461538.4615385, 0, 0, 0, -28846153.8461538, 0, 0, 10256410.2564103, 0, 0, 0, 11217948.7179487, -20192307.6923077, 115.384615384615,
        175000000, 20192307.6923077, 1019.23076923077, 0, 14022435.8974359, 17.4679487179487, 62820512.8205128, 0, 404.166666666667, -133.012820512821, 0, -30288461.5384615, -134615384.615385, 100961538.461538, -1855.76923076923, 0, -14022435.8974359, 185.737179487179, -8974358.97435897, 0, -33.974358974359, -144.230769230769, 0, -30288461.5384615, -100961538.461538, -100961538.461538, 253.846153846154, 0, 14022435.8974359, -57.8525641025641, 15705128.2051282, 16826923.0769231, -42.4679487179487, 0, 0, -20192307.6923077, 60576923.0769231, -20192307.6923077, 582.692307692308, 0, -14022435.8974359, 110.416666666667, -29166666.6666667, 3365384.61538461, -154.647435897436, -11.2179487179487, 0, -20192307.6923077,
        -20192307.6923077, 136538461.538462, -1199.03846153846, 14022435.8974359, 0, -40.7051282051282, 0, 133333333.333333, -247.275641025641, 0, -133.012820512821, 33653846.1538462, 100961538.461538, 4807692.3076923, 919.230769230769, -14022435.8974359, 0, -80.4487179487179, 0, 55128205.1282051, -141.826923076923, 0, -144.230769230769, 16826923.0769231, -100961538.461538, -72115384.6153846, 191.346153846154, 14022435.8974359, 0, 41.1858974358974, 16826923.0769231, -9935897.43589743, 69.7115384615385, 0, 0, -16826923.0769231, 20192307.6923077, -69230769.2307692, 88.4615384615384, -14022435.8974359, 0, -20.9935897435898, -3365384.61538461, -37179487.1794872, 87.6602564102564, 0, -11.2179487179487, -33653846.1538462,
        0, 0, 69230769.2307692, 0, 0, 0, 0, 0, 43589743.5897436, -20192307.6923077, 22435897.4358974, -266.025641025641, 0, 0, -28846153.8461538, 0, 0, 0, 0, 0, 10256410.2564103, -20192307.6923077, 11217948.7179487, -288.461538461538, 0, 0, -38461538.4615385, 0, 0, 0, 0, 0, 1282051.28205128, -13461538.4615385, -11217948.7179487, 0, 0, 0, -1923076.92307692, 0, 0, 0, 0, 0, -14743589.7435897, -13461538.4615385, -22435897.4358974, -22.4358974358974,
        -1028.84615384615, 0, -141346153.846154, 68.9102564102564, 0, 22435897.4358974, -133.012820512821, 0, -20192307.6923077, 89753685.8974359, 4206.73076923077, 650.701322115385, 1769.23076923077, 0, 141346153.846154, -280.448717948718, 0, -22435897.4358974, 144.230769230769, 0, 20192307.6923077, 44865064.1025641, 841.346153846154, 426.213341346154, -86.5384615384616, 0, 60576923.0769231, 0, 0, -11217948.7179487, 0, 0, -13461538.4615385, 22430849.3589744, -4206.73076923077, 134.611738782051, -653.846153846154, 0, -60576923.0769231, -57.6923076923077, 0, 11217948.7179487, 123.397435897436, 0, 13461538.4615385, 44873477.5641026, -841.346153846154, 201.935136217949,
        0, -1028.84615384615, -141346153.846154, 0, 68.9102564102564, -20192307.6923077, 0, -133.012820512821, 22435897.4358974, 4206.73076923077, 89753685.8974359, -157.047636217949, 0, 1769.23076923077, -60576923.0769231, 0, -280.448717948718, 13461538.4615385, 0, 144.230769230769, 11217948.7179487, -841.346153846154, 44873477.5641026, -201.938501602564, 0, -86.5384615384616, 60576923.0769231, 0, 0, -13461538.4615385, 0, 0, -11217948.7179487, -4206.73076923077, 22430849.3589744, -134.608373397436, 0, -653.846153846154, 141346153.846154, 0, -57.6923076923077, 20192307.6923077, 0, 123.397435897436, -22435897.4358974, 841.346153846154, 44865064.1025641, -112.174719551282,
        -212019230.769231, -212019230.769231, -2572.11538461538, 33653846.1538462, -30288461.5384615, 172.275641025641, -30288461.5384615, 33653846.1538462, -332.532051282051, 0, 0, 314107051.282051, 212019230.769231, -90865384.6153846, 4423.07692307692, -33653846.1538462, 20192307.6923077, -701.121794871795, 30288461.5384615, 16826923.0769231, 360.576923076923, 0, 0, 157050160.25641, 90865384.6153846, 90865384.6153846, -216.346153846154, -16826923.0769231, -20192307.6923077, 0, -20192307.6923077, -16826923.0769231, 0, 0, 0, 78523397.4358974, -90865384.6153846, 212019230.769231, -1634.61538461538, 16826923.0769231, 30288461.5384615, -144.230769230769, 20192307.6923077, -33653846.1538462, 308.49358974359, 0, 0, 157050160.25641,
        -1107692307.69231, -100961538.461538, -12303.8461538462, -69230769.2307692, -20192307.6923077, -1277.88461538462, -134615384.615385, 100961538.461538, -1671.3141025641, 1769.23076923077, 0, 212019230.769231, 1592307692.30769, -504807692.307692, 18361.5384615385, -136538461.538462, 20192307.6923077, -1345.19230769231, 175000000, 20192307.6923077, 1527.08333333333, 2711.53846153846, 0, 212019230.769231, 103846153.846154, 100961538.461538, 619.230769230769, -4807692.3076923, -100961538.461538, 758.653846153846, -60576923.0769231, -20192307.6923077, -221.314102564103, 1769.23076923077, 0, 90865384.6153846, -588461538.461538, 504807692.307692, -6676.92307692308, -72115384.6153846, 100961538.461538, -1193.26923076923, 100961538.461538, -100961538.461538, 1375.16025641026, 826.923076923077, 0, 90865384.6153846,
        100961538.461538, 103846153.846154, -619.230769230769, 20192307.6923077, 60576923.0769231, -221.314102564103, 100961538.461538, 4807692.3076923, 758.653846153846, 0, 1769.23076923077, -90865384.6153846, -504807692.307692, 1592307692.30769, -18361.5384615385, -20192307.6923077, -175000000, 1527.08333333333, -20192307.6923077, 136538461.538462, -1345.19230769231, 0, 2711.53846153846, -212019230.769231, -100961538.461538, -1107692307.69231, 12303.8461538462, -100961538.461538, 134615384.615385, -1671.3141025641, 20192307.6923077, 69230769.2307692, -1277.88461538462, 0, 1769.23076923077, -212019230.769231, 504807692.307692, -588461538.461538, 6676.92307692308, 100961538.461538, -100961538.461538, 1375.16025641026, -100961538.461538, 72115384.6153846, -1193.26923076923, 0, 826.923076923077, -90865384.6153846,
        0, 0, -223076923.076923, 0, 0, -1923076.92307692, 0, 0, -28846153.8461538, 141346153.846154, -60576923.0769231, 3538.46153846154, 0, 0, 707692307.692308, 0, 0, -69230769.2307692, 0, 0, 69230769.2307692, 141346153.846154, -141346153.846154, 5423.07692307692, 0, 0, -223076923.076923, 0, 0, 28846153.8461538, 0, 0, 1923076.92307692, 60576923.0769231, -141346153.846154, 3538.46153846154, 0, 0, -261538461.538462, 0, 0, -38461538.4615385, 0, 0, 38461538.4615385, 60576923.0769231, -60576923.0769231, 1653.84615384615,
        69230769.2307692, 20192307.6923077, 1247.11538461538, -37179487.1794872, 3365384.61538461, -123.878205128205, 0, -14022435.8974359, 162.339743589744, -280.448717948718, 0, -33653846.1538462, -136538461.538462, -20192307.6923077, -944.230769230769, 133333333.333333, 0, 656.570512820513, 0, 14022435.8974359, -100.641025641026, -133.012820512821, 0, 33653846.1538462, -4807692.3076923, 100961538.461538, -1199.03846153846, 55128205.1282051, 0, 165.865384615385, 0, -14022435.8974359, 174.679487179487, -144.230769230769, 0, 16826923.0769231, 72115384.6153846, -100961538.461538, 896.153846153846, -9935897.43589743, -16826923.0769231, 79.3269230769231, 0, 14022435.8974359, -135.416666666667, -134.615384615385, 0, -16826923.0769231,
        -20192307.6923077, -60576923.0769231, 265.384615384615, -3365384.61538461, -29166666.6666667, 111.378205128205, -14022435.8974359, 0, -57.8525641025641, 0, -280.448717948718, 20192307.6923077, 20192307.6923077, -175000000, 1740.38461538462, 0, 62820512.8205128, -476.282051282051, 14022435.8974359, 0, 118.429487179487, 0, -133.012820512821, 30288461.5384615, 100961538.461538, 134615384.615385, -1019.23076923077, 0, -8974358.97435897, 19.551282051282, -14022435.8974359, 0, 84.775641025641, 0, -144.230769230769, 30288461.5384615, -100961538.461538, 100961538.461538, -986.538461538461, -16826923.0769231, 15705128.2051282, -173.878205128205, 14022435.8974359, 0, 110.416666666667, 0, -134.615384615385, 20192307.6923077,
        0, 0, 1923076.92307692, 0, 0, -14743589.7435897, 0, 0, 0, -22435897.4358974, 13461538.4615385, -560.897435897436, 0, 0, -69230769.2307692, 0, 0, 43589743.5897436, 0, 0, 0, 22435897.4358974, 20192307.6923077, -266.025641025641, 0, 0, 28846153.8461538, 0, 0, 10256410.2564103, 0, 0, 0, 11217948.7179487, 20192307.6923077, -288.461538461538, 0, 0, 38461538.4615385, 0, 0, 1282051.28205128, 0, 0, 0, -11217948.7179487, 13461538.4615385, -269.230769230769,
        -134615384.615385, -100961538.461538, -1019.23076923077, 0, -14022435.8974359, -84.775641025641, -8974358.97435897, 0, -19.551282051282, 144.230769230769, 0, 30288461.5384615, 175000000, -20192307.6923077, 1740.38461538462, 0, 14022435.8974359, -118.429487179487, 62820512.8205128, 0, 476.282051282051, 133.012820512821, 0, 30288461.5384615, 60576923.0769231, 20192307.6923077, 265.384615384615, 0, -14022435.8974359, 57.8525641025641, -29166666.6666667, -3365384.61538461, -111.378205128205, 280.448717948718, 0, 20192307.6923077, -100961538.461538, 100961538.461538, -986.538461538461, 0, 14022435.8974359, -110.416666666667, 15705128.2051282, -16826923.0769231, 173.878205128205, 134.615384615385, 0, 20192307.6923077,
        -100961538.461538, 4807692.3076923, -1199.03846153846, -14022435.8974359, 0, -174.679487179487, 0, 55128205.1282051, -165.865384615385, 0, 144.230769230769, 16826923.0769231, 20192307.6923077, 136538461.538462, -944.230769230769, 14022435.8974359, 0, 100.641025641026, 0, 133333333.333333, -656.570512820513, 0, 133.012820512821, 33653846.1538462, -20192307.6923077, -69230769.2307692, 1247.11538461538, -14022435.8974359, 0, -162.339743589744, 3365384.61538461, -37179487.1794872, 123.878205128205, 0, 280.448717948718, -33653846.1538462, 100961538.461538, -72115384.6153846, 896.153846153846, 14022435.8974359, 0, 135.416666666667, -16826923.0769231, -9935897.43589743, -79.3269230769231, 0, 134.615384615385, -16826923.0769231,
        0, 0, -28846153.8461538, 0, 0, 0, 0, 0, 10256410.2564103, 20192307.6923077, 11217948.7179487, 288.461538461538, 0, 0, 69230769.2307692, 0, 0, 0, 0, 0, 43589743.5897436, 20192307.6923077, 22435897.4358974, 266.025641025641, 0, 0, -1923076.92307692, 0, 0, 0, 0, 0, -14743589.7435897, 13461538.4615385, -22435897.4358974, 560.897435897436, 0, 0, -38461538.4615385, 0, 0, 0, 0, 0, 1282051.28205128, 13461538.4615385, -11217948.7179487, 269.230769230769,
        -1028.84615384615, 0, -141346153.846154, -145.833333333333, 0, -22435897.4358974, -144.230769230769, 0, -20192307.6923077, 44865064.1025641, -841.346153846154, 426.221754807692, 2711.53846153846, 0, 141346153.846154, -133.012820512821, 0, 22435897.4358974, 133.012820512821, 0, 20192307.6923077, 89753685.8974359, -4206.73076923077, 650.743389423077, -1028.84615384615, 0, 60576923.0769231, 144.230769230769, 0, 11217948.7179487, 145.833333333333, 0, -13461538.4615385, 44873477.5641026, 841.346153846154, 201.92672275641, -653.846153846154, 0, -60576923.0769231, -134.615384615385, 0, -11217948.7179487, 134.615384615385, 0, 13461538.4615385, 22430849.3589744, 4206.73076923077, 134.569671474359,
        0, -1028.84615384615, -60576923.0769231, 0, -145.833333333333, -13461538.4615385, 0, -144.230769230769, 11217948.7179487, 841.346153846154, 44873477.5641026, -201.92672275641, 0, 2711.53846153846, -141346153.846154, 0, -133.012820512821, 20192307.6923077, 0, 133.012820512821, 22435897.4358974, -4206.73076923077, 89753685.8974359, -650.743389423077, 0, -1028.84615384615, 141346153.846154, 0, 144.230769230769, -20192307.6923077, 0, 145.833333333333, -22435897.4358974, -841.346153846154, 44865064.1025641, -426.221754807692, 0, -653.846153846154, 60576923.0769231, 0, -134.615384615385, 13461538.4615385, 0, 134.615384615385, -11217948.7179487, 4206.73076923077, 22430849.3589744, -134.569671474359,
        -212019230.769231, -90865384.6153846, -2572.11538461538, -33653846.1538462, -20192307.6923077, -364.583333333333, -30288461.5384615, 16826923.0769231, -360.576923076923, 0, 0, 157050160.25641, 212019230.769231, -212019230.769231, 6778.84615384615, 33653846.1538462, 30288461.5384615, -332.532051282051, 30288461.5384615, 33653846.1538462, 332.532051282051, 0, 0, 314107051.282051, 90865384.6153846, 212019230.769231, -2572.11538461538, 16826923.0769231, -30288461.5384615, 360.576923076923, -20192307.6923077, -33653846.1538462, 364.583333333333, 0, 0, 157050160.25641, -90865384.6153846, 90865384.6153846, -1634.61538461538, -16826923.0769231, 20192307.6923077, -336.538461538462, 20192307.6923077, -16826923.0769231, 336.538461538462, 0, 0, 78523397.4358974,
        -588461538.461538, -504807692.307692, -996.153846153846, -72115384.6153846, -100961538.461538, 105.769230769231, -100961538.461538, -100961538.461538, 150.160256410256, -86.5384615384616, 0, 90865384.6153846, 103846153.846154, -100961538.461538, 2207.69230769231, -4807692.3076923, 100961538.461538, -1038.46153846154, 60576923.0769231, -20192307.6923077, 599.839743589744, -1028.84615384615, 0, 90865384.6153846, 1592307692.30769, 504807692.307692, 619.230769230769, -136538461.538462, -20192307.6923077, -798.076923076923, -175000000, 20192307.6923077, -478.685897435897, -1028.84615384615, 0, 212019230.769231, -1107692307.69231, 100961538.461538, -1830.76923076923, -69230769.2307692, 20192307.6923077, -57.6923076923076, 134615384.615385, 100961538.461538, -69.3910256410257, -86.5384615384616, 0, 212019230.769231,
        -504807692.307692, -588461538.461538, 619.230769230769, -100961538.461538, -100961538.461538, 163.621794871795, -100961538.461538, -72115384.6153846, 116.346153846154, 0, -86.5384615384616, 90865384.6153846, 100961538.461538, -1107692307.69231, 13515.3846153846, 100961538.461538, 134615384.615385, -1065.54487179487, 20192307.6923077, -69230769.2307692, 1412.5, 0, -1028.84615384615, 212019230.769231, 504807692.307692, 1592307692.30769, -12303.8461538462, 20192307.6923077, -175000000, 1094.39102564103, -20192307.6923077, -136538461.538462, 1210.57692307692, 0, -1028.84615384615, 212019230.769231, -100961538.461538, 103846153.846154, -1830.76923076923, -20192307.6923077, 60576923.0769231, -394.391025641026, 100961538.461538, -4807692.3076923, 318.269230769231, 0, -86.5384615384616, 90865384.6153846,
        0, 0, -261538461.538462, 0, 0, -38461538.4615385, 0, 0, -38461538.4615385, 60576923.0769231, 60576923.0769231, -173.076923076923, 0, 0, -223076923.076923, 0, 0, 28846153.8461538, 0, 0, -1923076.92307692, 60576923.0769231, 141346153.846154, -2057.69230769231, 0, 0, 707692307.692308, 0, 0, -69230769.2307692, 0, 0, -69230769.2307692, 141346153.846154, 141346153.846154, -2057.69230769231, 0, 0, -223076923.076923, 0, 0, -1923076.92307692, 0, 0, 28846153.8461538, 141346153.846154, 60576923.0769231, -173.076923076923,
        72115384.6153846, 100961538.461538, 191.346153846154, -9935897.43589743, 16826923.0769231, -69.7115384615385, 0, 14022435.8974359, -41.1858974358974, 0, 0, -16826923.0769231, -4807692.3076923, -100961538.461538, 919.230769230769, 55128205.1282051, 0, 141.826923076923, 0, -14022435.8974359, 80.4487179487179, 144.230769230769, 0, 16826923.0769231, -136538461.538462, 20192307.6923077, -1199.03846153846, 133333333.333333, 0, 247.275641025641, 0, 14022435.8974359, 40.7051282051282, 133.012820512821, 0, 33653846.1538462, 69230769.2307692, -20192307.6923077, 88.4615384615384, -37179487.1794872, -3365384.61538461, -87.6602564102564, 0, -14022435.8974359, 20.9935897435898, 11.2179487179487, 0, -33653846.1538462,
        100961538.461538, 100961538.461538, 253.846153846154, 16826923.0769231, 15705128.2051282, 42.4679487179487, 14022435.8974359, 0, 57.8525641025641, 0, 0, -20192307.6923077, -100961538.461538, 134615384.615385, -1855.76923076923, 0, -8974358.97435897, 33.974358974359, -14022435.8974359, 0, -185.737179487179, 0, 144.230769230769, -30288461.5384615, -20192307.6923077, -175000000, 1019.23076923077, 0, 62820512.8205128, -404.166666666667, 14022435.8974359, 0, -17.4679487179487, 0, 133.012820512821, -30288461.5384615, 20192307.6923077, -60576923.0769231, 582.692307692308, 3365384.61538461, -29166666.6666667, 154.647435897436, -14022435.8974359, 0, -110.416666666667, 0, 11.2179487179487, -20192307.6923077,
        0, 0, 38461538.4615385, 0, 0, 1282051.28205128, 0, 0, 0, -11217948.7179487, -13461538.4615385, 0, 0, 0, 28846153.8461538, 0, 0, 10256410.2564103, 0, 0, 0, 11217948.7179487, -20192307.6923077, 288.461538461538, 0, 0, -69230769.2307692, 0, 0, 43589743.5897436, 0, 0, 0, 22435897.4358974, -20192307.6923077, 266.025641025641, 0, 0, 1923076.92307692, 0, 0, -14743589.7435897, 0, 0, 0, -22435897.4358974, -13461538.4615385, 22.4358974358974,
        100961538.461538, 100961538.461538, 238.461538461538, 0, 14022435.8974359, -16.1858974358974, 15705128.2051282, 16826923.0769231, -25.8012820512821, 0, 0, -20192307.6923077, -60576923.0769231, 20192307.6923077, -555.769230769231, 0, -14022435.8974359, 152.083333333333, -29166666.6666667, 3365384.61538461, -171.314102564103, 145.833333333333, 0, -20192307.6923077, -175000000, -20192307.6923077, -265.384615384615, 0, 14022435.8974359, 22.9166666666667, 62820512.8205128, 0, 116.025641025641, -68.9102564102564, 0, -30288461.5384615, 134615384.615385, -100961538.461538, 582.692307692308, 0, -14022435.8974359, 56.5705128205128, -8974358.97435897, 0, -34.2948717948718, -57.6923076923077, 0, -30288461.5384615,
        100961538.461538, 72115384.6153846, 214.423076923077, 14022435.8974359, 0, 32.8525641025641, 16826923.0769231, -9935897.43589743, 88.9423076923077, 0, 0, -16826923.0769231, -20192307.6923077, 69230769.2307692, -1550, -14022435.8974359, 0, 95.0320512820513, -3365384.61538461, -37179487.1794872, 90.224358974359, 0, 145.833333333333, -33653846.1538462, 20192307.6923077, -136538461.538462, 1247.11538461538, 14022435.8974359, 0, -33.3333333333333, 0, 133333333.333333, -656.570512820513, 0, -68.9102564102564, 33653846.1538462, -100961538.461538, -4807692.3076923, 88.4615384615384, -14022435.8974359, 0, 6.4102564102564, 0, 55128205.1282051, -165.865384615385, 0, -57.6923076923077, 16826923.0769231,
        0, 0, 38461538.4615385, 0, 0, 0, 0, 0, 1282051.28205128, -13461538.4615385, -11217948.7179487, 0, 0, 0, 1923076.92307692, 0, 0, 0, 0, 0, -14743589.7435897, -13461538.4615385, -22435897.4358974, 291.666666666667, 0, 0, -69230769.2307692, 0, 0, 0, 0, 0, 43589743.5897436, -20192307.6923077, 22435897.4358974, -137.820512820513, 0, 0, 28846153.8461538, 0, 0, 0, 0, 0, 10256410.2564103, -20192307.6923077, 11217948.7179487, -115.384615384615,
        -86.5384615384616, 0, -60576923.0769231, 0, 0, -11217948.7179487, 0, 0, -13461538.4615385, 22430849.3589744, -4206.73076923077, 134.608373397436, 1769.23076923077, 0, 60576923.0769231, -144.230769230769, 0, 11217948.7179487, 280.448717948718, 0, 13461538.4615385, 44873477.5641026, -841.346153846154, 201.938501602564, -1028.84615384615, 0, 141346153.846154, 133.012820512821, 0, 22435897.4358974, -68.9102564102564, 0, -20192307.6923077, 89753685.8974359, 4206.73076923077, 157.047636217949, -653.846153846154, 0, -141346153.846154, -123.397435897436, 0, -22435897.4358974, 57.6923076923077, 0, 20192307.6923077, 44865064.1025641, 841.346153846154, 112.174719551282,
        0, -86.5384615384616, -60576923.0769231, 0, 0, -13461538.4615385, 0, 0, -11217948.7179487, -4206.73076923077, 22430849.3589744, -134.611738782051, 0, 1769.23076923077, -141346153.846154, 0, -144.230769230769, 20192307.6923077, 0, 280.448717948718, -22435897.4358974, 841.346153846154, 44865064.1025641, -426.213341346154, 0, -1028.84615384615, 141346153.846154, 0, 133.012820512821, -20192307.6923077, 0, -68.9102564102564, 22435897.4358974, 4206.73076923077, 89753685.8974359, -650.701322115385, 0, -653.846153846154, 60576923.0769231, 0, -123.397435897436, 13461538.4615385, 0, 57.6923076923077, 11217948.7179487, -841.346153846154, 44873477.5641026, -201.935136217949,
        -90865384.6153846, -90865384.6153846, -216.346153846154, -16826923.0769231, -20192307.6923077, 0, -20192307.6923077, -16826923.0769231, 0, 0, 0, 78523397.4358974, 90865384.6153846, -212019230.769231, 4423.07692307692, 16826923.0769231, 30288461.5384615, -360.576923076923, 20192307.6923077, -33653846.1538462, 701.121794871795, 0, 0, 157050160.25641, 212019230.769231, 212019230.769231, -2572.11538461538, 33653846.1538462, -30288461.5384615, 332.532051282051, -30288461.5384615, 33653846.1538462, -172.275641025641, 0, 0, 314107051.282051, -212019230.769231, 90865384.6153846, -1634.61538461538, -33653846.1538462, 20192307.6923077, -308.49358974359, 30288461.5384615, 16826923.0769231, 144.230769230769, 0, 0, 157050160.25641,
        103846153.846154, 100961538.461538, 996.153846153846, 4807692.3076923, 100961538.461538, -38.4615384615385, 60576923.0769231, 20192307.6923077, 426.762820512821, -653.846153846154, 0, -90865384.6153846, -588461538.461538, 504807692.307692, -7053.84615384615, 72115384.6153846, -100961538.461538, 971.153846153846, -100961538.461538, 100961538.461538, -1061.37820512821, -653.846153846154, 0, -90865384.6153846, -1107692307.69231, -100961538.461538, -619.230769230769, 69230769.2307692, 20192307.6923077, -76.9230769230769, 134615384.615385, -100961538.461538, 536.378205128205, -653.846153846154, 0, -212019230.769231, 1592307692.30769, -504807692.307692, 6676.92307692308, 136538461.538462, -20192307.6923077, 932.692307692308, -175000000, -20192307.6923077, -911.378205128205, -653.846153846154, 0, -212019230.769231,
        -100961538.461538, -1107692307.69231, 619.230769230769, 100961538.461538, -134615384.615385, 536.378205128205, -20192307.6923077, -69230769.2307692, -76.9230769230769, 0, -653.846153846154, 212019230.769231, 504807692.307692, -588461538.461538, 7053.84615384615, -100961538.461538, 100961538.461538, -1061.37820512821, 100961538.461538, -72115384.6153846, 971.153846153846, 0, -653.846153846154, 90865384.6153846, 100961538.461538, 103846153.846154, -996.153846153846, -20192307.6923077, -60576923.0769231, 426.762820512821, -100961538.461538, -4807692.3076923, -38.4615384615385, 0, -653.846153846154, 90865384.6153846, -504807692.307692, 1592307692.30769, -6676.92307692308, 20192307.6923077, 175000000, -911.378205128205, 20192307.6923077, -136538461.538462, 932.692307692308, 0, -653.846153846154, 212019230.769231,
        0, 0, -223076923.076923, 0, 0, -28846153.8461538, 0, 0, -1923076.92307692, -60576923.0769231, 141346153.846154, -1307.69230769231, 0, 0, -261538461.538462, 0, 0, 38461538.4615385, 0, 0, -38461538.4615385, -60576923.0769231, 60576923.0769231, -1307.69230769231, 0, 0, -223076923.076923, 0, 0, 1923076.92307692, 0, 0, 28846153.8461538, -141346153.846154, 60576923.0769231, -1307.69230769231, 0, 0, 707692307.692308, 0, 0, 69230769.2307692, 0, 0, -69230769.2307692, -141346153.846154, 141346153.846154, -1307.69230769231,
        4807692.3076923, -100961538.461538, 191.346153846154, 55128205.1282051, 0, 141.826923076923, 0, -14022435.8974359, 87.8205128205128, -57.6923076923077, 0, 16826923.0769231, -72115384.6153846, 100961538.461538, -1301.92307692308, -9935897.43589743, -16826923.0769231, 98.5576923076923, 0, 14022435.8974359, -127.083333333333, -134.615384615385, 0, -16826923.0769231, -69230769.2307692, -20192307.6923077, 214.423076923077, -37179487.1794872, 3365384.61538461, -121.314102564103, 0, -14022435.8974359, 46.3141025641026, -123.397435897436, 0, -33653846.1538462, 136538461.538462, 20192307.6923077, 896.153846153846, 133333333.333333, 0, 247.275641025641, 0, 14022435.8974359, -108.012820512821, -68.9102564102564, 0, 33653846.1538462,
        -100961538.461538, -134615384.615385, -253.846153846154, 0, -8974358.97435897, 19.8717948717949, -14022435.8974359, 0, -44.3910256410256, 0, -57.6923076923077, 30288461.5384615, 100961538.461538, -100961538.461538, 1478.84615384615, -16826923.0769231, 15705128.2051282, -190.544871794872, 14022435.8974359, 0, 152.083333333333, 0, -134.615384615385, 20192307.6923077, 20192307.6923077, 60576923.0769231, -238.461538461538, -3365384.61538461, -29166666.6666667, 128.044871794872, -14022435.8974359, 0, -16.1858974358974, 0, -123.397435897436, 20192307.6923077, -20192307.6923077, 175000000, -986.538461538461, 0, 62820512.8205128, -188.141025641026, 14022435.8974359, 0, 123.878205128205, 0, -68.9102564102564, 30288461.5384615,
        0, 0, -28846153.8461538, 0, 0, 10256410.2564103, 0, 0, 0, 11217948.7179487, 20192307.6923077, -115.384615384615, 0, 0, -38461538.4615385, 0, 0, 1282051.28205128, 0, 0, 0, -11217948.7179487, 13461538.4615385, -269.230769230769, 0, 0, -1923076.92307692, 0, 0, -14743589.7435897, 0, 0, 0, -22435897.4358974, 13461538.4615385, -246.794871794872, 0, 0, 69230769.2307692, 0, 0, 43589743.5897436, 0, 0, 0, 22435897.4358974, 20192307.6923077, -137.820512820513,
        -60576923.0769231, -20192307.6923077, -238.461538461538, 0, -14022435.8974359, 16.1858974358974, -29166666.6666667, -3365384.61538461, -128.044871794872, 123.397435897436, 0, 20192307.6923077, 100961538.461538, -100961538.461538, 1478.84615384615, 0, 14022435.8974359, -152.083333333333, 15705128.2051282, -16826923.0769231, 190.544871794872, 134.615384615385, 0, 20192307.6923077, 134615384.615385, 100961538.461538, -253.846153846154, 0, -14022435.8974359, 44.3910256410256, -8974358.97435897, 0, -19.8717948717949, 57.6923076923077, 0, 30288461.5384615, -175000000, 20192307.6923077, -986.538461538461, 0, 14022435.8974359, -123.878205128205, 62820512.8205128, 0, 188.141025641026, 68.9102564102564, 0, 30288461.5384615,
        20192307.6923077, 69230769.2307692, 214.423076923077, -14022435.8974359, 0, -46.3141025641026, 3365384.61538461, -37179487.1794872, 121.314102564103, 0, 123.397435897436, -33653846.1538462, -100961538.461538, 72115384.6153846, -1301.92307692308, 14022435.8974359, 0, 127.083333333333, -16826923.0769231, -9935897.43589743, -98.5576923076923, 0, 134.615384615385, -16826923.0769231, 100961538.461538, -4807692.3076923, 191.346153846154, -14022435.8974359, 0, -87.8205128205128, 0, 55128205.1282051, -141.826923076923, 0, 57.6923076923077, 16826923.0769231, -20192307.6923077, -136538461.538462, 896.153846153846, 14022435.8974359, 0, 108.012820512821, 0, 133333333.333333, -247.275641025641, 0, 68.9102564102564, 33653846.1538462,
        0, 0, 1923076.92307692, 0, 0, 0, 0, 0, -14743589.7435897, 13461538.4615385, -22435897.4358974, 246.794871794872, 0, 0, 38461538.4615385, 0, 0, 0, 0, 0, 1282051.28205128, 13461538.4615385, -11217948.7179487, 269.230769230769, 0, 0, 28846153.8461538, 0, 0, 0, 0, 0, 10256410.2564103, 20192307.6923077, 11217948.7179487, 115.384615384615, 0, 0, -69230769.2307692, 0, 0, 0, 0, 0, 43589743.5897436, 20192307.6923077, 22435897.4358974, 137.820512820513,
        -86.5384615384616, 0, -60576923.0769231, 57.6923076923077, 0, 11217948.7179487, -11.2179487179487, 0, -13461538.4615385, 44873477.5641026, 841.346153846154, 201.930088141026, 826.923076923077, 0, 60576923.0769231, -134.615384615385, 0, -11217948.7179487, 134.615384615385, 0, 13461538.4615385, 22430849.3589744, 4206.73076923077, 134.566306089744, -86.5384615384616, 0, 141346153.846154, 11.2179487179487, 0, -22435897.4358974, -57.6923076923077, 0, -20192307.6923077, 44865064.1025641, -841.346153846154, 112.183133012821, -653.846153846154, 0, -141346153.846154, -68.9102564102564, 0, 22435897.4358974, 68.9102564102564, 0, 20192307.6923077, 89753685.8974359, -4206.73076923077, 157.089703525641,
        0, -86.5384615384616, -141346153.846154, 0, 57.6923076923077, -20192307.6923077, 0, -11.2179487179487, -22435897.4358974, -841.346153846154, 44865064.1025641, -112.183133012821, 0, 826.923076923077, -60576923.0769231, 0, -134.615384615385, 13461538.4615385, 0, 134.615384615385, -11217948.7179487, 4206.73076923077, 22430849.3589744, -134.566306089744, 0, -86.5384615384616, 60576923.0769231, 0, 11.2179487179487, -13461538.4615385, 0, -57.6923076923077, 11217948.7179487, 841.346153846154, 44873477.5641026, -201.930088141026, 0, -653.846153846154, 141346153.846154, 0, -68.9102564102564, 20192307.6923077, 0, 68.9102564102564, 22435897.4358974, -4206.73076923077, 89753685.8974359, -157.089703525641,
        -90865384.6153846, -212019230.769231, -216.346153846154, 16826923.0769231, -30288461.5384615, 144.230769230769, -20192307.6923077, -33653846.1538462, -28.0448717948718, 0, 0, 157050160.25641, 90865384.6153846, -90865384.6153846, 2067.30769230769, -16826923.0769231, 20192307.6923077, -336.538461538462, 20192307.6923077, -16826923.0769231, 336.538461538462, 0, 0, 78523397.4358974, 212019230.769231, 90865384.6153846, -216.346153846154, -33653846.1538462, -20192307.6923077, 28.0448717948718, -30288461.5384615, 16826923.0769231, -144.230769230769, 0, 0, 157050160.25641, -212019230.769231, 212019230.769231, -1634.61538461538, 33653846.1538462, 30288461.5384615, -172.275641025641, 30288461.5384615, 33653846.1538462, 172.275641025641, 0, 0, 314107051.282051;

    ChMatrixNM<double, 48, 48> Expected_JacobianR_NoDispSmallVelWithDamping;
    Expected_JacobianR_NoDispSmallVelWithDamping <<
        15923076.9230769, 5048076.92307692, 0, 1365384.61538462, 201923.076923077, 0, 1750000, -201923.076923077, 0, 0, 0, -2120192.30769231, -11076923.0769231, 1009615.38461538, 0, 692307.692307692, -201923.076923077, 0, -1346153.84615385, -1009615.38461538, 0, 0, 0, -2120192.30769231, -5884615.38461538, -5048076.92307692, 0, 721153.846153846, 1009615.38461538, 0, 1009615.38461538, 1009615.38461538, 0, 0, 0, -908653.846153846, 1038461.53846154, -1009615.38461538, 0, 48076.923076923, -1009615.38461538, 0, -605769.230769231, 201923.076923077, 0, 0, 0, -908653.846153846,
        5048076.92307692, 15923076.9230769, 0, -201923.076923077, 1750000, 0, 201923.076923077, 1365384.61538462, 0, 0, 0, -2120192.30769231, -1009615.38461538, 1038461.53846154, 0, 201923.076923077, -605769.230769231, 0, -1009615.38461538, 48076.923076923, 0, 0, 0, -908653.846153846, -5048076.92307692, -5884615.38461538, 0, 1009615.38461538, 1009615.38461538, 0, 1009615.38461538, 721153.846153846, 0, 0, 0, -908653.846153846, 1009615.38461538, -11076923.0769231, 0, -1009615.38461538, -1346153.84615385, 0, -201923.076923077, 692307.692307692, 0, 0, 0, -2120192.30769231,
        0, 0, 7076923.07692308, 0, 0, 692307.692307692, 0, 0, 692307.692307692, -1413461.53846154, -1413461.53846154, 0, 0, 0, -2230769.23076923, 0, 0, 19230.7692307692, 0, 0, -288461.538461538, -1413461.53846154, -605769.230769231, 0, 0, 0, -2615384.61538462, 0, 0, 384615.384615385, 0, 0, 384615.384615385, -605769.230769231, -605769.230769231, 0, 0, 0, -2230769.23076923, 0, 0, -288461.538461538, 0, 0, 19230.7692307692, -605769.230769231, -1413461.53846154, 0,
        1365384.61538462, -201923.076923077, 0, 1333333.33333333, 0, 0, 0, 140224.358974359, 0, 0, 0, 336538.461538462, -692307.692307692, 201923.076923077, 0, -371794.871794872, -33653.8461538461, 0, 0, -140224.358974359, 0, 0, 0, -336538.461538462, -721153.846153846, -1009615.38461538, 0, -99358.9743589743, 168269.230769231, 0, 0, 140224.358974359, 0, 0, 0, -168269.230769231, 48076.923076923, 1009615.38461538, 0, 551282.051282051, 0, 0, 0, -140224.358974359, 0, 0, 0, 168269.230769231,
        201923.076923077, 1750000, 0, 0, 628205.128205128, 0, 140224.358974359, 0, 0, 0, 0, -302884.615384615, -201923.076923077, 605769.230769231, 0, 33653.8461538461, -291666.666666667, 0, -140224.358974359, 0, 0, 0, 0, -201923.076923077, -1009615.38461538, -1009615.38461538, 0, 168269.230769231, 157051.282051282, 0, 140224.358974359, 0, 0, 0, 0, -201923.076923077, 1009615.38461538, -1346153.84615385, 0, 0, -89743.5897435897, 0, -140224.358974359, 0, 0, 0, 0, -302884.615384615,
        0, 0, 692307.692307692, 0, 0, 435897.435897436, 0, 0, 0, 224358.974358974, -201923.076923077, 0, 0, 0, -19230.7692307692, 0, 0, -147435.897435897, 0, 0, 0, -224358.974358974, -134615.384615385, 0, 0, 0, -384615.384615385, 0, 0, 12820.5128205128, 0, 0, 0, -112179.487179487, -134615.384615385, 0, 0, 0, -288461.538461538, 0, 0, 102564.102564103, 0, 0, 0, 112179.487179487, -201923.076923077, 0,
        1750000, 201923.076923077, 0, 0, 140224.358974359, 0, 628205.128205128, 0, 0, 0, 0, -302884.615384615, -1346153.84615385, 1009615.38461538, 0, 0, -140224.358974359, 0, -89743.5897435897, 0, 0, 0, 0, -302884.615384615, -1009615.38461538, -1009615.38461538, 0, 0, 140224.358974359, 0, 157051.282051282, 168269.230769231, 0, 0, 0, -201923.076923077, 605769.230769231, -201923.076923077, 0, 0, -140224.358974359, 0, -291666.666666667, 33653.8461538461, 0, 0, 0, -201923.076923077,
        -201923.076923077, 1365384.61538462, 0, 140224.358974359, 0, 0, 0, 1333333.33333333, 0, 0, 0, 336538.461538462, 1009615.38461538, 48076.923076923, 0, -140224.358974359, 0, 0, 0, 551282.051282051, 0, 0, 0, 168269.230769231, -1009615.38461538, -721153.846153846, 0, 140224.358974359, 0, 0, 168269.230769231, -99358.9743589743, 0, 0, 0, -168269.230769231, 201923.076923077, -692307.692307692, 0, -140224.358974359, 0, 0, -33653.8461538461, -371794.871794872, 0, 0, 0, -336538.461538462,
        0, 0, 692307.692307692, 0, 0, 0, 0, 0, 435897.435897436, -201923.076923077, 224358.974358974, 0, 0, 0, -288461.538461538, 0, 0, 0, 0, 0, 102564.102564103, -201923.076923077, 112179.487179487, 0, 0, 0, -384615.384615385, 0, 0, 0, 0, 0, 12820.5128205128, -134615.384615385, -112179.487179487, 0, 0, 0, -19230.7692307692, 0, 0, 0, 0, 0, -147435.897435897, -134615.384615385, -224358.974358974, 0,
        0, 0, -1413461.53846154, 0, 0, 224358.974358974, 0, 0, -201923.076923077, 897536.858974359, 42.0673076923077, 0, 0, 0, 1413461.53846154, 0, 0, -224358.974358974, 0, 0, 201923.076923077, 448650.641025641, 8.41346153846154, 0, 0, 0, 605769.230769231, 0, 0, -112179.487179487, 0, 0, -134615.384615385, 224308.493589744, -42.0673076923077, 0, 0, 0, -605769.230769231, 0, 0, 112179.487179487, 0, 0, 134615.384615385, 448734.775641026, -8.41346153846154, 0,
        0, 0, -1413461.53846154, 0, 0, -201923.076923077, 0, 0, 224358.974358974, 42.0673076923077, 897536.858974359, 0, 0, 0, -605769.230769231, 0, 0, 134615.384615385, 0, 0, 112179.487179487, -8.41346153846154, 448734.775641026, 0, 0, 0, 605769.230769231, 0, 0, -134615.384615385, 0, 0, -112179.487179487, -42.0673076923077, 224308.493589744, 0, 0, 0, 1413461.53846154, 0, 0, 201923.076923077, 0, 0, -224358.974358974, 8.41346153846154, 448650.641025641, 0,
        -2120192.30769231, -2120192.30769231, 0, 336538.461538462, -302884.615384615, 0, -302884.615384615, 336538.461538462, 0, 0, 0, 3141070.51282051, 2120192.30769231, -908653.846153846, 0, -336538.461538462, 201923.076923077, 0, 302884.615384615, 168269.230769231, 0, 0, 0, 1570501.6025641, 908653.846153846, 908653.846153846, 0, -168269.230769231, -201923.076923077, 0, -201923.076923077, -168269.230769231, 0, 0, 0, 785233.974358974, -908653.846153846, 2120192.30769231, 0, 168269.230769231, 302884.615384615, 0, 201923.076923077, -336538.461538462, 0, 0, 0, 1570501.6025641,
        -11076923.0769231, -1009615.38461538, 0, -692307.692307692, -201923.076923077, 0, -1346153.84615385, 1009615.38461538, 0, 0, 0, 2120192.30769231, 15923076.9230769, -5048076.92307692, 0, -1365384.61538462, 201923.076923077, 0, 1750000, 201923.076923077, 0, 0, 0, 2120192.30769231, 1038461.53846154, 1009615.38461538, 0, -48076.923076923, -1009615.38461538, 0, -605769.230769231, -201923.076923077, 0, 0, 0, 908653.846153846, -5884615.38461538, 5048076.92307692, 0, -721153.846153846, 1009615.38461538, 0, 1009615.38461538, -1009615.38461538, 0, 0, 0, 908653.846153846,
        1009615.38461538, 1038461.53846154, 0, 201923.076923077, 605769.230769231, 0, 1009615.38461538, 48076.923076923, 0, 0, 0, -908653.846153846, -5048076.92307692, 15923076.9230769, 0, -201923.076923077, -1750000, 0, -201923.076923077, 1365384.61538462, 0, 0, 0, -2120192.30769231, -1009615.38461538, -11076923.0769231, 0, -1009615.38461538, 1346153.84615385, 0, 201923.076923077, 692307.692307692, 0, 0, 0, -2120192.30769231, 5048076.92307692, -5884615.38461538, 0, 1009615.38461538, -1009615.38461538, 0, -1009615.38461538, 721153.846153846, 0, 0, 0, -908653.846153846,
        0, 0, -2230769.23076923, 0, 0, -19230.7692307692, 0, 0, -288461.538461538, 1413461.53846154, -605769.230769231, 0, 0, 0, 7076923.07692308, 0, 0, -692307.692307692, 0, 0, 692307.692307692, 1413461.53846154, -1413461.53846154, 0, 0, 0, -2230769.23076923, 0, 0, 288461.538461538, 0, 0, 19230.7692307692, 605769.230769231, -1413461.53846154, 0, 0, 0, -2615384.61538462, 0, 0, -384615.384615385, 0, 0, 384615.384615385, 605769.230769231, -605769.230769231, 0,
        692307.692307692, 201923.076923077, 0, -371794.871794872, 33653.8461538461, 0, 0, -140224.358974359, 0, 0, 0, -336538.461538462, -1365384.61538462, -201923.076923077, 0, 1333333.33333333, 0, 0, 0, 140224.358974359, 0, 0, 0, 336538.461538462, -48076.923076923, 1009615.38461538, 0, 551282.051282051, 0, 0, 0, -140224.358974359, 0, 0, 0, 168269.230769231, 721153.846153846, -1009615.38461538, 0, -99358.9743589743, -168269.230769231, 0, 0, 140224.358974359, 0, 0, 0, -168269.230769231,
        -201923.076923077, -605769.230769231, 0, -33653.8461538461, -291666.666666667, 0, -140224.358974359, 0, 0, 0, 0, 201923.076923077, 201923.076923077, -1750000, 0, 0, 628205.128205128, 0, 140224.358974359, 0, 0, 0, 0, 302884.615384615, 1009615.38461538, 1346153.84615385, 0, 0, -89743.5897435897, 0, -140224.358974359, 0, 0, 0, 0, 302884.615384615, -1009615.38461538, 1009615.38461538, 0, -168269.230769231, 157051.282051282, 0, 140224.358974359, 0, 0, 0, 0, 201923.076923077,
        0, 0, 19230.7692307692, 0, 0, -147435.897435897, 0, 0, 0, -224358.974358974, 134615.384615385, 0, 0, 0, -692307.692307692, 0, 0, 435897.435897436, 0, 0, 0, 224358.974358974, 201923.076923077, 0, 0, 0, 288461.538461538, 0, 0, 102564.102564103, 0, 0, 0, 112179.487179487, 201923.076923077, 0, 0, 0, 384615.384615385, 0, 0, 12820.5128205128, 0, 0, 0, -112179.487179487, 134615.384615385, 0,
        -1346153.84615385, -1009615.38461538, 0, 0, -140224.358974359, 0, -89743.5897435897, 0, 0, 0, 0, 302884.615384615, 1750000, -201923.076923077, 0, 0, 140224.358974359, 0, 628205.128205128, 0, 0, 0, 0, 302884.615384615, 605769.230769231, 201923.076923077, 0, 0, -140224.358974359, 0, -291666.666666667, -33653.8461538461, 0, 0, 0, 201923.076923077, -1009615.38461538, 1009615.38461538, 0, 0, 140224.358974359, 0, 157051.282051282, -168269.230769231, 0, 0, 0, 201923.076923077,
        -1009615.38461538, 48076.923076923, 0, -140224.358974359, 0, 0, 0, 551282.051282051, 0, 0, 0, 168269.230769231, 201923.076923077, 1365384.61538462, 0, 140224.358974359, 0, 0, 0, 1333333.33333333, 0, 0, 0, 336538.461538462, -201923.076923077, -692307.692307692, 0, -140224.358974359, 0, 0, 33653.8461538461, -371794.871794872, 0, 0, 0, -336538.461538462, 1009615.38461538, -721153.846153846, 0, 140224.358974359, 0, 0, -168269.230769231, -99358.9743589743, 0, 0, 0, -168269.230769231,
        0, 0, -288461.538461538, 0, 0, 0, 0, 0, 102564.102564103, 201923.076923077, 112179.487179487, 0, 0, 0, 692307.692307692, 0, 0, 0, 0, 0, 435897.435897436, 201923.076923077, 224358.974358974, 0, 0, 0, -19230.7692307692, 0, 0, 0, 0, 0, -147435.897435897, 134615.384615385, -224358.974358974, 0, 0, 0, -384615.384615385, 0, 0, 0, 0, 0, 12820.5128205128, 134615.384615385, -112179.487179487, 0,
        0, 0, -1413461.53846154, 0, 0, -224358.974358974, 0, 0, -201923.076923077, 448650.641025641, -8.41346153846154, 0, 0, 0, 1413461.53846154, 0, 0, 224358.974358974, 0, 0, 201923.076923077, 897536.858974359, -42.0673076923077, 0, 0, 0, 605769.230769231, 0, 0, 112179.487179487, 0, 0, -134615.384615385, 448734.775641026, 8.41346153846154, 0, 0, 0, -605769.230769231, 0, 0, -112179.487179487, 0, 0, 134615.384615385, 224308.493589744, 42.0673076923077, 0,
        0, 0, -605769.230769231, 0, 0, -134615.384615385, 0, 0, 112179.487179487, 8.41346153846154, 448734.775641026, 0, 0, 0, -1413461.53846154, 0, 0, 201923.076923077, 0, 0, 224358.974358974, -42.0673076923077, 897536.858974359, 0, 0, 0, 1413461.53846154, 0, 0, -201923.076923077, 0, 0, -224358.974358974, -8.41346153846154, 448650.641025641, 0, 0, 0, 605769.230769231, 0, 0, 134615.384615385, 0, 0, -112179.487179487, 42.0673076923077, 224308.493589744, 0,
        -2120192.30769231, -908653.846153846, 0, -336538.461538462, -201923.076923077, 0, -302884.615384615, 168269.230769231, 0, 0, 0, 1570501.6025641, 2120192.30769231, -2120192.30769231, 0, 336538.461538462, 302884.615384615, 0, 302884.615384615, 336538.461538462, 0, 0, 0, 3141070.51282051, 908653.846153846, 2120192.30769231, 0, 168269.230769231, -302884.615384615, 0, -201923.076923077, -336538.461538462, 0, 0, 0, 1570501.6025641, -908653.846153846, 908653.846153846, 0, -168269.230769231, 201923.076923077, 0, 201923.076923077, -168269.230769231, 0, 0, 0, 785233.974358974,
        -5884615.38461538, -5048076.92307692, 0, -721153.846153846, -1009615.38461538, 0, -1009615.38461538, -1009615.38461538, 0, 0, 0, 908653.846153846, 1038461.53846154, -1009615.38461538, 0, -48076.923076923, 1009615.38461538, 0, 605769.230769231, -201923.076923077, 0, 0, 0, 908653.846153846, 15923076.9230769, 5048076.92307692, 0, -1365384.61538462, -201923.076923077, 0, -1750000, 201923.076923077, 0, 0, 0, 2120192.30769231, -11076923.0769231, 1009615.38461538, 0, -692307.692307692, 201923.076923077, 0, 1346153.84615385, 1009615.38461538, 0, 0, 0, 2120192.30769231,
        -5048076.92307692, -5884615.38461538, 0, -1009615.38461538, -1009615.38461538, 0, -1009615.38461538, -721153.846153846, 0, 0, 0, 908653.846153846, 1009615.38461538, -11076923.0769231, 0, 1009615.38461538, 1346153.84615385, 0, 201923.076923077, -692307.692307692, 0, 0, 0, 2120192.30769231, 5048076.92307692, 15923076.9230769, 0, 201923.076923077, -1750000, 0, -201923.076923077, -1365384.61538462, 0, 0, 0, 2120192.30769231, -1009615.38461538, 1038461.53846154, 0, -201923.076923077, 605769.230769231, 0, 1009615.38461538, -48076.923076923, 0, 0, 0, 908653.846153846,
        0, 0, -2615384.61538462, 0, 0, -384615.384615385, 0, 0, -384615.384615385, 605769.230769231, 605769.230769231, 0, 0, 0, -2230769.23076923, 0, 0, 288461.538461538, 0, 0, -19230.7692307692, 605769.230769231, 1413461.53846154, 0, 0, 0, 7076923.07692308, 0, 0, -692307.692307692, 0, 0, -692307.692307692, 1413461.53846154, 1413461.53846154, 0, 0, 0, -2230769.23076923, 0, 0, -19230.7692307692, 0, 0, 288461.538461538, 1413461.53846154, 605769.230769231, 0,
        721153.846153846, 1009615.38461538, 0, -99358.9743589743, 168269.230769231, 0, 0, 140224.358974359, 0, 0, 0, -168269.230769231, -48076.923076923, -1009615.38461538, 0, 551282.051282051, 0, 0, 0, -140224.358974359, 0, 0, 0, 168269.230769231, -1365384.61538462, 201923.076923077, 0, 1333333.33333333, 0, 0, 0, 140224.358974359, 0, 0, 0, 336538.461538462, 692307.692307692, -201923.076923077, 0, -371794.871794872, -33653.8461538461, 0, 0, -140224.358974359, 0, 0, 0, -336538.461538462,
        1009615.38461538, 1009615.38461538, 0, 168269.230769231, 157051.282051282, 0, 140224.358974359, 0, 0, 0, 0, -201923.076923077, -1009615.38461538, 1346153.84615385, 0, 0, -89743.5897435897, 0, -140224.358974359, 0, 0, 0, 0, -302884.615384615, -201923.076923077, -1750000, 0, 0, 628205.128205128, 0, 140224.358974359, 0, 0, 0, 0, -302884.615384615, 201923.076923077, -605769.230769231, 0, 33653.8461538461, -291666.666666667, 0, -140224.358974359, 0, 0, 0, 0, -201923.076923077,
        0, 0, 384615.384615385, 0, 0, 12820.5128205128, 0, 0, 0, -112179.487179487, -134615.384615385, 0, 0, 0, 288461.538461538, 0, 0, 102564.102564103, 0, 0, 0, 112179.487179487, -201923.076923077, 0, 0, 0, -692307.692307692, 0, 0, 435897.435897436, 0, 0, 0, 224358.974358974, -201923.076923077, 0, 0, 0, 19230.7692307692, 0, 0, -147435.897435897, 0, 0, 0, -224358.974358974, -134615.384615385, 0,
        1009615.38461538, 1009615.38461538, 0, 0, 140224.358974359, 0, 157051.282051282, 168269.230769231, 0, 0, 0, -201923.076923077, -605769.230769231, 201923.076923077, 0, 0, -140224.358974359, 0, -291666.666666667, 33653.8461538461, 0, 0, 0, -201923.076923077, -1750000, -201923.076923077, 0, 0, 140224.358974359, 0, 628205.128205128, 0, 0, 0, 0, -302884.615384615, 1346153.84615385, -1009615.38461538, 0, 0, -140224.358974359, 0, -89743.5897435897, 0, 0, 0, 0, -302884.615384615,
        1009615.38461538, 721153.846153846, 0, 140224.358974359, 0, 0, 168269.230769231, -99358.9743589743, 0, 0, 0, -168269.230769231, -201923.076923077, 692307.692307692, 0, -140224.358974359, 0, 0, -33653.8461538461, -371794.871794872, 0, 0, 0, -336538.461538462, 201923.076923077, -1365384.61538462, 0, 140224.358974359, 0, 0, 0, 1333333.33333333, 0, 0, 0, 336538.461538462, -1009615.38461538, -48076.923076923, 0, -140224.358974359, 0, 0, 0, 551282.051282051, 0, 0, 0, 168269.230769231,
        0, 0, 384615.384615385, 0, 0, 0, 0, 0, 12820.5128205128, -134615.384615385, -112179.487179487, 0, 0, 0, 19230.7692307692, 0, 0, 0, 0, 0, -147435.897435897, -134615.384615385, -224358.974358974, 0, 0, 0, -692307.692307692, 0, 0, 0, 0, 0, 435897.435897436, -201923.076923077, 224358.974358974, 0, 0, 0, 288461.538461538, 0, 0, 0, 0, 0, 102564.102564103, -201923.076923077, 112179.487179487, 0,
        0, 0, -605769.230769231, 0, 0, -112179.487179487, 0, 0, -134615.384615385, 224308.493589744, -42.0673076923077, 0, 0, 0, 605769.230769231, 0, 0, 112179.487179487, 0, 0, 134615.384615385, 448734.775641026, -8.41346153846154, 0, 0, 0, 1413461.53846154, 0, 0, 224358.974358974, 0, 0, -201923.076923077, 897536.858974359, 42.0673076923077, 0, 0, 0, -1413461.53846154, 0, 0, -224358.974358974, 0, 0, 201923.076923077, 448650.641025641, 8.41346153846154, 0,
        0, 0, -605769.230769231, 0, 0, -134615.384615385, 0, 0, -112179.487179487, -42.0673076923077, 224308.493589744, 0, 0, 0, -1413461.53846154, 0, 0, 201923.076923077, 0, 0, -224358.974358974, 8.41346153846154, 448650.641025641, 0, 0, 0, 1413461.53846154, 0, 0, -201923.076923077, 0, 0, 224358.974358974, 42.0673076923077, 897536.858974359, 0, 0, 0, 605769.230769231, 0, 0, 134615.384615385, 0, 0, 112179.487179487, -8.41346153846154, 448734.775641026, 0,
        -908653.846153846, -908653.846153846, 0, -168269.230769231, -201923.076923077, 0, -201923.076923077, -168269.230769231, 0, 0, 0, 785233.974358974, 908653.846153846, -2120192.30769231, 0, 168269.230769231, 302884.615384615, 0, 201923.076923077, -336538.461538462, 0, 0, 0, 1570501.6025641, 2120192.30769231, 2120192.30769231, 0, 336538.461538462, -302884.615384615, 0, -302884.615384615, 336538.461538462, 0, 0, 0, 3141070.51282051, -2120192.30769231, 908653.846153846, 0, -336538.461538462, 201923.076923077, 0, 302884.615384615, 168269.230769231, 0, 0, 0, 1570501.6025641,
        1038461.53846154, 1009615.38461538, 0, 48076.923076923, 1009615.38461538, 0, 605769.230769231, 201923.076923077, 0, 0, 0, -908653.846153846, -5884615.38461538, 5048076.92307692, 0, 721153.846153846, -1009615.38461538, 0, -1009615.38461538, 1009615.38461538, 0, 0, 0, -908653.846153846, -11076923.0769231, -1009615.38461538, 0, 692307.692307692, 201923.076923077, 0, 1346153.84615385, -1009615.38461538, 0, 0, 0, -2120192.30769231, 15923076.9230769, -5048076.92307692, 0, 1365384.61538462, -201923.076923077, 0, -1750000, -201923.076923077, 0, 0, 0, -2120192.30769231,
        -1009615.38461538, -11076923.0769231, 0, 1009615.38461538, -1346153.84615385, 0, -201923.076923077, -692307.692307692, 0, 0, 0, 2120192.30769231, 5048076.92307692, -5884615.38461538, 0, -1009615.38461538, 1009615.38461538, 0, 1009615.38461538, -721153.846153846, 0, 0, 0, 908653.846153846, 1009615.38461538, 1038461.53846154, 0, -201923.076923077, -605769.230769231, 0, -1009615.38461538, -48076.923076923, 0, 0, 0, 908653.846153846, -5048076.92307692, 15923076.9230769, 0, 201923.076923077, 1750000, 0, 201923.076923077, -1365384.61538462, 0, 0, 0, 2120192.30769231,
        0, 0, -2230769.23076923, 0, 0, -288461.538461538, 0, 0, -19230.7692307692, -605769.230769231, 1413461.53846154, 0, 0, 0, -2615384.61538462, 0, 0, 384615.384615385, 0, 0, -384615.384615385, -605769.230769231, 605769.230769231, 0, 0, 0, -2230769.23076923, 0, 0, 19230.7692307692, 0, 0, 288461.538461538, -1413461.53846154, 605769.230769231, 0, 0, 0, 7076923.07692308, 0, 0, 692307.692307692, 0, 0, -692307.692307692, -1413461.53846154, 1413461.53846154, 0,
        48076.923076923, -1009615.38461538, 0, 551282.051282051, 0, 0, 0, -140224.358974359, 0, 0, 0, 168269.230769231, -721153.846153846, 1009615.38461538, 0, -99358.9743589743, -168269.230769231, 0, 0, 140224.358974359, 0, 0, 0, -168269.230769231, -692307.692307692, -201923.076923077, 0, -371794.871794872, 33653.8461538461, 0, 0, -140224.358974359, 0, 0, 0, -336538.461538462, 1365384.61538462, 201923.076923077, 0, 1333333.33333333, 0, 0, 0, 140224.358974359, 0, 0, 0, 336538.461538462,
        -1009615.38461538, -1346153.84615385, 0, 0, -89743.5897435897, 0, -140224.358974359, 0, 0, 0, 0, 302884.615384615, 1009615.38461538, -1009615.38461538, 0, -168269.230769231, 157051.282051282, 0, 140224.358974359, 0, 0, 0, 0, 201923.076923077, 201923.076923077, 605769.230769231, 0, -33653.8461538461, -291666.666666667, 0, -140224.358974359, 0, 0, 0, 0, 201923.076923077, -201923.076923077, 1750000, 0, 0, 628205.128205128, 0, 140224.358974359, 0, 0, 0, 0, 302884.615384615,
        0, 0, -288461.538461538, 0, 0, 102564.102564103, 0, 0, 0, 112179.487179487, 201923.076923077, 0, 0, 0, -384615.384615385, 0, 0, 12820.5128205128, 0, 0, 0, -112179.487179487, 134615.384615385, 0, 0, 0, -19230.7692307692, 0, 0, -147435.897435897, 0, 0, 0, -224358.974358974, 134615.384615385, 0, 0, 0, 692307.692307692, 0, 0, 435897.435897436, 0, 0, 0, 224358.974358974, 201923.076923077, 0,
        -605769.230769231, -201923.076923077, 0, 0, -140224.358974359, 0, -291666.666666667, -33653.8461538461, 0, 0, 0, 201923.076923077, 1009615.38461538, -1009615.38461538, 0, 0, 140224.358974359, 0, 157051.282051282, -168269.230769231, 0, 0, 0, 201923.076923077, 1346153.84615385, 1009615.38461538, 0, 0, -140224.358974359, 0, -89743.5897435897, 0, 0, 0, 0, 302884.615384615, -1750000, 201923.076923077, 0, 0, 140224.358974359, 0, 628205.128205128, 0, 0, 0, 0, 302884.615384615,
        201923.076923077, 692307.692307692, 0, -140224.358974359, 0, 0, 33653.8461538461, -371794.871794872, 0, 0, 0, -336538.461538462, -1009615.38461538, 721153.846153846, 0, 140224.358974359, 0, 0, -168269.230769231, -99358.9743589743, 0, 0, 0, -168269.230769231, 1009615.38461538, -48076.923076923, 0, -140224.358974359, 0, 0, 0, 551282.051282051, 0, 0, 0, 168269.230769231, -201923.076923077, -1365384.61538462, 0, 140224.358974359, 0, 0, 0, 1333333.33333333, 0, 0, 0, 336538.461538462,
        0, 0, 19230.7692307692, 0, 0, 0, 0, 0, -147435.897435897, 134615.384615385, -224358.974358974, 0, 0, 0, 384615.384615385, 0, 0, 0, 0, 0, 12820.5128205128, 134615.384615385, -112179.487179487, 0, 0, 0, 288461.538461538, 0, 0, 0, 0, 0, 102564.102564103, 201923.076923077, 112179.487179487, 0, 0, 0, -692307.692307692, 0, 0, 0, 0, 0, 435897.435897436, 201923.076923077, 224358.974358974, 0,
        0, 0, -605769.230769231, 0, 0, 112179.487179487, 0, 0, -134615.384615385, 448734.775641026, 8.41346153846154, 0, 0, 0, 605769.230769231, 0, 0, -112179.487179487, 0, 0, 134615.384615385, 224308.493589744, 42.0673076923077, 0, 0, 0, 1413461.53846154, 0, 0, -224358.974358974, 0, 0, -201923.076923077, 448650.641025641, -8.41346153846154, 0, 0, 0, -1413461.53846154, 0, 0, 224358.974358974, 0, 0, 201923.076923077, 897536.858974359, -42.0673076923077, 0,
        0, 0, -1413461.53846154, 0, 0, -201923.076923077, 0, 0, -224358.974358974, -8.41346153846154, 448650.641025641, 0, 0, 0, -605769.230769231, 0, 0, 134615.384615385, 0, 0, -112179.487179487, 42.0673076923077, 224308.493589744, 0, 0, 0, 605769.230769231, 0, 0, -134615.384615385, 0, 0, 112179.487179487, 8.41346153846154, 448734.775641026, 0, 0, 0, 1413461.53846154, 0, 0, 201923.076923077, 0, 0, 224358.974358974, -42.0673076923077, 897536.858974359, 0,
        -908653.846153846, -2120192.30769231, 0, 168269.230769231, -302884.615384615, 0, -201923.076923077, -336538.461538462, 0, 0, 0, 1570501.6025641, 908653.846153846, -908653.846153846, 0, -168269.230769231, 201923.076923077, 0, 201923.076923077, -168269.230769231, 0, 0, 0, 785233.974358974, 2120192.30769231, 908653.846153846, 0, -336538.461538462, -201923.076923077, 0, -302884.615384615, 168269.230769231, 0, 0, 0, 1570501.6025641, -2120192.30769231, 2120192.30769231, 0, 336538.461538462, 302884.615384615, 0, 302884.615384615, 336538.461538462, 0, 0, 0, 3141070.51282051;


    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation

    //Setup the test conditions
    ChVector<double> OriginalVel = m_nodeB->GetPos_dt();
    m_nodeB->SetPos_dt(ChVector<>(0.0, 0.0, 0.001));
    m_element->SetAlphaDamp(0.01);

    ChVectorDynamic<double> InternalForceNoDispSmallVelNoGravity;
    InternalForceNoDispSmallVelNoGravity.resize(48);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceNoDispSmallVelNoGravity);

    ChMatrixDynamic<double> JacobianK_NoDispSmallVelWithDamping;
    JacobianK_NoDispSmallVelWithDamping.resize(48, 48);
    m_element->ComputeKRMmatricesGlobal(JacobianK_NoDispSmallVelWithDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_NoDispSmallVelWithDamping;
    JacobianR_NoDispSmallVelWithDamping.resize(48, 48);
    m_element->ComputeKRMmatricesGlobal(JacobianR_NoDispSmallVelWithDamping, 0, 1, 0);

    //Reset the element conditions back to its original values
    m_nodeB->SetPos_dt(OriginalVel);
    m_element->SetAlphaDamp(0.0);


    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_NoDispSmallVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(48, 48);

    double small_terms_JacR = 1e-4*Expected_JacobianR_NoDispSmallVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacR = 0;
    double max_percent_error_JacR = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacR;
    percent_error_matrix_JacR.resize(48, 48);


    for (auto i = 0; i < Expected_JacobianK_NoDispSmallVelWithDamping.rows(); i++) {
        for (auto j = 0; j < Expected_JacobianK_NoDispSmallVelWithDamping.cols(); j++) {
            if (std::abs(Expected_JacobianK_NoDispSmallVelWithDamping(i, j)) < small_terms_JacK) {
                double error = std::abs(JacobianK_NoDispSmallVelWithDamping(i, j) - Expected_JacobianK_NoDispSmallVelWithDamping(i, j));
                if (error > zeros_max_error_JacK)
                    zeros_max_error_JacK = error;
                percent_error_matrix_JacK(i, j) = 0.0;
            }
            else {
                double percent_error = std::abs((JacobianK_NoDispSmallVelWithDamping(i, j) - Expected_JacobianK_NoDispSmallVelWithDamping(i, j)) / Expected_JacobianK_NoDispSmallVelWithDamping(i, j));
                if (percent_error > max_percent_error_JacK)
                    max_percent_error_JacK = percent_error;
                percent_error_matrix_JacK(i, j) = percent_error;
            }

            if (std::abs(Expected_JacobianR_NoDispSmallVelWithDamping(i, j)) < small_terms_JacR) {
                double error = std::abs(JacobianR_NoDispSmallVelWithDamping(i, j) - Expected_JacobianR_NoDispSmallVelWithDamping(i, j));
                if (error > zeros_max_error_JacR)
                    zeros_max_error_JacR = error;
                percent_error_matrix_JacR(i, j) = 0.0;
            }
            else {
                double percent_error = std::abs((JacobianR_NoDispSmallVelWithDamping(i, j) - Expected_JacobianR_NoDispSmallVelWithDamping(i, j)) / Expected_JacobianR_NoDispSmallVelWithDamping(i, j));
                if (percent_error > max_percent_error_JacR)
                    max_percent_error_JacR = percent_error;
                percent_error_matrix_JacR(i, j) = percent_error;
            }
        }
    }


    if (msglvl >= 2) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Jacobian K Term - No Displacement, Small Velocity, With Damping = " << std::endl;
        std::cout << JacobianK_NoDispSmallVelWithDamping << std::endl;
        std::cout << "Expected Jacobian K Term - No Displacement, Small Velocity, With Damping = " << std::endl;
        std::cout << Expected_JacobianK_NoDispSmallVelWithDamping << std::endl;
        std::cout << "Percent Error Jacobian K Term - No Displacement, Small Velocity, With Damping = " << std::endl;
        std::cout << percent_error_matrix_JacK << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Jacobian K Term - No Displacement, Small Velocity, With Damping (Max Abs Error) = "
            << (JacobianK_NoDispSmallVelWithDamping - Expected_JacobianK_NoDispSmallVelWithDamping).cwiseAbs().maxCoeff() << std::endl;

        std::cout << "Jacobian K Term - No Displacement, Small Velocity, With Damping (Max Abs Error - Only Smaller Terms) = "
            << zeros_max_error_JacK;
        if (zeros_max_error_JacK / JacobianK_NoDispSmallVelWithDamping.cwiseAbs().maxCoeff() > 0.0001) {
            print_red(" - Test FAILED\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl;
        }

        std::cout << "Jacobian K Term - No Displacement, Small Velocity, With Damping (Max Abs % Error - Only Larger Terms) = "
            << max_percent_error_JacK * 100 << "%";
        if (max_percent_error_JacK > 0.01) {
            print_red(" - Test FAILED\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl;
        }
    }
    if (zeros_max_error_JacK / JacobianK_NoDispSmallVelWithDamping.cwiseAbs().maxCoeff() > 0.0001) {
        passed_tests = false;
    }
    if (max_percent_error_JacK > 0.01) {
        passed_tests = false;
    }

    if (msglvl >= 2) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Jacobian R Term - No Displacement, Small Velocity, With Damping = " << std::endl;
        std::cout << JacobianR_NoDispSmallVelWithDamping << std::endl;
        std::cout << "Expected Jacobian R Term - No Displacement, Small Velocity, With Damping = " << std::endl;
        std::cout << Expected_JacobianR_NoDispSmallVelWithDamping << std::endl;
        std::cout << "Percent Error Jacobian R Term - No Displacement, Small Velocity, With Damping = " << std::endl;
        std::cout << percent_error_matrix_JacR << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Jacobian R Term - No Displacement, Small Velocity, With Damping (Max Abs Error) = "
            << (JacobianR_NoDispSmallVelWithDamping - Expected_JacobianR_NoDispSmallVelWithDamping).cwiseAbs().maxCoeff() << std::endl;

        std::cout << "Jacobian R Term - No Displacement, Small Velocity, With Damping (Max Abs Error - Only Smaller Terms) = "
            << zeros_max_error_JacR;
        if (zeros_max_error_JacR / JacobianR_NoDispSmallVelWithDamping.cwiseAbs().maxCoeff() > 0.0001) {
            print_red(" - Test FAILED\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl;
        }

        std::cout << "Jacobian R Term - No Displacement, Small Velocity, With Damping (Max Abs % Error - Only Larger Terms) = "
            << max_percent_error_JacR * 100 << "%";
        if (max_percent_error_JacR > 0.01) {
            print_red(" - Test FAILED\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl;
        }
    }
    if (zeros_max_error_JacR / JacobianR_NoDispSmallVelWithDamping.cwiseAbs().maxCoeff() > 0.0001) {
        passed_tests = false;
    }
    if (max_percent_error_JacR > 0.01) {
        passed_tests = false;
    }

    return(passed_tests);
}

template <typename ElementVersion, typename MaterialVersion>
bool ANCFShellTest<ElementVersion, MaterialVersion>::AxialDisplacementCheck(int msglvl) {
    // =============================================================================
    //  Check the Axial Displacement of a Beam compared to the analytical result
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================

    auto system = new ChSystemSMC();
    // Set gravity to 0 since this is a statics test against an analytical solution
    system->Set_G_acc(ChVector<>(0, 0, 0));

    auto solver = chrono_types::make_shared<ChSolverSparseLU>();
    solver->UseSparsityPatternLearner(true);
    solver->LockSparsityPattern(true);
    solver->SetVerbose(false);
    system->SetSolver(solver);

    // Set up integrator
    system->SetTimestepperType(ChTimestepper::Type::HHT);
    auto integrator = std::static_pointer_cast<ChTimestepperHHT>(system->GetTimestepper());
    integrator->SetAlpha(-0.2);
    integrator->SetMaxiters(100);
    integrator->SetAbsTolerances(1e-5);
    integrator->SetMode(ChTimestepperHHT::POSITION);
    integrator->SetScaling(true);
    integrator->SetVerbose(false);
    integrator->SetModifiedNewton(true);

    // Mesh properties
    int num_elements = 10;
    double length = 1.0;    // m
    double width =  0.1;  // m
    double height = 0.01;  // m
    // Aluminum 7075-T651
    double rho = 2810;  // kg/m^3
    double E = 71.7e9;  // Pa
    double nu = 0.33;
    double G = E / (2 * (1 + nu));

    auto material = chrono_types::make_shared<MaterialVersion>(rho, E, nu);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    system->Add(mesh);

    // Populate the mesh container with a the nodes and elements for the meshed beam
    int num_nodes = 2*num_elements + 2;
    double dx = length / (num_elements);

    // Setup beam cross section gradients to initially align with the global y and z directions
    ChVector<> dir1(1, 0, 0);
    ChVector<> dir2(0, 1, 0);
    ChVector<> dir3(0, 0, 1);

    // Create the first nodes and fix them completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, 0, 0.0), dir1, dir2, dir3);
    nodeA->SetFixed(true);
    mesh->AddNode(nodeA);
    auto nodeD = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, width, 0.0), dir1, dir2, dir3);
    nodeD->SetFixed(true);
    mesh->AddNode(nodeD);

    auto elementlast = chrono_types::make_shared<ElementVersion>();
    std::shared_ptr<ChNodeFEAxyzDDD> nodeEndPoint;

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(dx * i, 0, 0), dir1, dir2, dir3);
        mesh->AddNode(nodeB);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(dx * i, width, 0), dir1, dir2, dir3);
        mesh->AddNode(nodeC);

        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC, nodeD);
        element->SetDimensions(dx, width, height);
        element->SetMaterial(material);
        element->SetAlphaDamp(0.0);
        element->SetGravityOn(
            false);  // Enable the efficient ANCF method for calculating the application of gravity to the element
        element->SetStrainFormulation(ElementVersion::StrainFormulation::CMPoisson);
        // element->SetStrainFormulation(ElementVersion::StrainFormulation::CMNoPoisson);
        mesh->AddElement(element);

        nodeA = nodeB;
        nodeD = nodeC;
        elementlast = element;
    }

    nodeEndPoint = nodeA;

    mesh->SetAutomaticGravity(false);  // Turn off the default method for applying gravity to the mesh since it is less
                                       // efficient for ANCF elements

    // Create a custom atomic (point) load
    class MyLoaderTimeDependentTipLoad : public ChLoaderUVatomic {
    public:
        // Useful: a constructor that also sets ChLoadable
        MyLoaderTimeDependentTipLoad(std::shared_ptr<ChLoadableUV> mloadable) : ChLoaderUVatomic(mloadable) {}

        // Compute F=F(u), the load at U. The load is a 6-row vector, i.e.
        // a wrench: forceX, forceY, forceZ, torqueX, torqueY, torqueZ.
        virtual void ComputeF(
            const double U,              ///< normalized position along the beam axis [-1...1]
            const double V,              ///< normalized position along the beam axis [-1...1]
            ChVectorDynamic<>& F,        ///< Load at U
            ChVectorDynamic<>* state_x,  ///< if != 0, update state (pos. part) to this, then evaluate F
            ChVectorDynamic<>* state_w   ///< if != 0, update state (speed part) to this, then evaluate F
        ) override {
            assert(auxsystem);

            F.setZero();
            F(0) = TIP_FORCE;  // Apply the force along the global X axis (beam axis)
        }

    public:
        // add auxiliary data to the class, if you need to access it during ComputeF().
        ChSystem* auxsystem;
    };

    // Create the load container and to the current system
    auto loadcontainer = chrono_types::make_shared<ChLoadContainer>();
    system->Add(loadcontainer);

    // Create a custom load that uses the custom loader above.
    // The ChLoad is a 'manager' for your ChLoader.
    // It is created using templates, that is instancing a ChLoad<my_loader_class>()

    std::shared_ptr<ChLoad<MyLoaderTimeDependentTipLoad> > mload(new ChLoad<MyLoaderTimeDependentTipLoad>(elementlast));
    mload->loader.auxsystem = system;   // initialize auxiliary data of the loader, if needed
    mload->loader.SetApplication(1.0,0);  // specify application point
    loadcontainer->Add(mload);          // add the load to the load container.

    // Find the static solution for the system (final twist angle)
    system->DoStaticLinear();

    // Calculate the axial displacement of the end of the ANCF shell mesh
    ChVector<> point;
    ChQuaternion<> rot;
    elementlast->EvaluateSectionFrame(1, 0, point, rot);
    
    // For Analytical Formula, see a mechanics of materials textbook (delta = PL/AE)
    double Displacement_Theory = (TIP_FORCE * length) / (width*height*E);
    double Displacement_Model = point.x() - length;
    ChVector<> Tip_Angles = rot.Q_to_Euler123();

    double Percent_Error = (Displacement_Model - Displacement_Theory) / Displacement_Theory * 100;

    bool passed_tests = true;
    if (abs(Percent_Error) > 5.0) {
        passed_tests = false;
    }
    if ((abs(Tip_Angles.x() * CH_C_RAD_TO_DEG) > 0.001) || (abs(Tip_Angles.y() * CH_C_RAD_TO_DEG) > 0.001) || (abs(Tip_Angles.z() * CH_C_RAD_TO_DEG) > 0.001)) {
        passed_tests = false;
    }

    if (msglvl >= 2) {
        std::cout << "ANCF Tip Position: " << point << "m" << std::endl;
        std::cout << "ANCF Tip Displacement: " << Displacement_Model << "m" << std::endl;
        std::cout << "Analytical Tip Displacement: " << Displacement_Theory << "m" << std::endl;
        std::cout << "Percent Error: " << Percent_Error << "%" << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Tip Displacement Check (Percent Error less than 5%)";
        if (abs(Percent_Error) > 5.0) {
            print_red(" - Test FAILED\n\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl << std::endl;
        }
        std::cout << "Angular misalignment Checks (all angles less than 0.001 deg)";
        if ((abs(Tip_Angles.x() * CH_C_RAD_TO_DEG) > 0.001) || (abs(Tip_Angles.y() * CH_C_RAD_TO_DEG) > 0.001) || (abs(Tip_Angles.z() * CH_C_RAD_TO_DEG) > 0.001)) {
            print_red(" - Test FAILED\n\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl << std::endl;
        }
    }
    return(passed_tests);
}

template <typename ElementVersion, typename MaterialVersion>
bool ANCFShellTest<ElementVersion, MaterialVersion>::CantileverCheck(int msglvl) {
    // =============================================================================
    //  Check the Axial Displacement of a Beam compared to the analytical result
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================

    auto system = new ChSystemSMC();
    // Set gravity to 0 since this is a statics test against an analytical solution
    system->Set_G_acc(ChVector<>(0, 0, 0));

    auto solver = chrono_types::make_shared<ChSolverSparseLU>();
    solver->UseSparsityPatternLearner(true);
    solver->LockSparsityPattern(true);
    solver->SetVerbose(false);
    system->SetSolver(solver);

    // Set up integrator
    system->SetTimestepperType(ChTimestepper::Type::HHT);
    auto integrator = std::static_pointer_cast<ChTimestepperHHT>(system->GetTimestepper());
    integrator->SetAlpha(-0.2);
    integrator->SetMaxiters(100);
    integrator->SetAbsTolerances(1e-5);
    integrator->SetMode(ChTimestepperHHT::POSITION);
    integrator->SetScaling(true);
    integrator->SetVerbose(false);
    integrator->SetModifiedNewton(true);

    // Mesh properties
    int num_elements = 10;
    double length = 1.0;    // m
    double width = 0.1;  // m
    double height = 0.01;  // m
    // Aluminum 7075-T651
    double rho = 2810;  // kg/m^3
    double E = 71.7e9;  // Pa
    double nu = 0.33*0;
    double G = E / (2 * (1 + nu));

    auto material = chrono_types::make_shared<MaterialVersion>(rho, E, nu);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    system->Add(mesh);

    // Populate the mesh container with a the nodes and elements for the meshed beam
    int num_nodes = 2 * num_elements + 2;
    double dx = length / (num_elements);

    // Setup beam cross section gradients to initially align with the global y and z directions
    ChVector<> dir1(1, 0, 0);
    ChVector<> dir2(0, 1, 0);
    ChVector<> dir3(0, 0, 1);

    // Create the first nodes and fix them completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, 0, 0.0), dir1, dir2, dir3);
    nodeA->SetFixed(true);
    mesh->AddNode(nodeA);
    auto nodeD = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, width, 0.0), dir1, dir2, dir3);
    nodeD->SetFixed(true);
    mesh->AddNode(nodeD);

    auto elementlast = chrono_types::make_shared<ElementVersion>();
    std::shared_ptr<ChNodeFEAxyzDDD> nodeEndPoint;

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(dx * i, 0, 0), dir1, dir2, dir3);
        mesh->AddNode(nodeB);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(dx * i, width, 0), dir1, dir2, dir3);
        mesh->AddNode(nodeC);

        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC, nodeD);
        element->SetDimensions(dx, width, height);
        element->SetMaterial(material);
        element->SetAlphaDamp(0.0);
        element->SetGravityOn(
            false);  // Enable the efficient ANCF method for calculating the application of gravity to the element
        element->SetStrainFormulation(ElementVersion::StrainFormulation::CMPoisson);
        // element->SetStrainFormulation(ElementVersion::StrainFormulation::CMNoPoisson);
        mesh->AddElement(element);

        nodeA = nodeB;
        nodeD = nodeC;
        elementlast = element;
    }

    nodeEndPoint = nodeA;

    mesh->SetAutomaticGravity(false);  // Turn off the default method for applying gravity to the mesh since it is less
                                       // efficient for ANCF elements

    // Create a custom atomic (point) load
    class MyLoaderTimeDependentTipLoad : public ChLoaderUVatomic {
    public:
        // Useful: a constructor that also sets ChLoadable
        MyLoaderTimeDependentTipLoad(std::shared_ptr<ChLoadableUV> mloadable) : ChLoaderUVatomic(mloadable) {}

        // Compute F=F(u), the load at U. The load is a 6-row vector, i.e.
        // a wrench: forceX, forceY, forceZ, torqueX, torqueY, torqueZ.
        virtual void ComputeF(
            const double U,              ///< normalized position along the beam axis [-1...1]
            const double V,              ///< normalized position along the beam axis [-1...1]
            ChVectorDynamic<>& F,        ///< Load at U
            ChVectorDynamic<>* state_x,  ///< if != 0, update state (pos. part) to this, then evaluate F
            ChVectorDynamic<>* state_w   ///< if != 0, update state (speed part) to this, then evaluate F
        ) override {
            assert(auxsystem);

            F.setZero();
            F(2) = TIP_FORCE;  // Apply the force along the global Z axis (beam axis)
        }

    public:
        // add auxiliary data to the class, if you need to access it during ComputeF().
        ChSystem* auxsystem;
    };

    // Create the load container and to the current system
    auto loadcontainer = chrono_types::make_shared<ChLoadContainer>();
    system->Add(loadcontainer);

    // Create a custom load that uses the custom loader above.
    // The ChLoad is a 'manager' for your ChLoader.
    // It is created using templates, that is instancing a ChLoad<my_loader_class>()

    std::shared_ptr<ChLoad<MyLoaderTimeDependentTipLoad> > mload(new ChLoad<MyLoaderTimeDependentTipLoad>(elementlast));
    mload->loader.auxsystem = system;   // initialize auxiliary data of the loader, if needed
    mload->loader.SetApplication(1.0, 0.0);  // specify application point
    loadcontainer->Add(mload);          // add the load to the load container.

    // Find the nonlinear static solution for the system (final displacement)
    system->DoStaticNonlinear(50);

    // Calculate the displacement of the end of the ANCF beam mesh
    ChVector<> point;
    ChQuaternion<> rot;
    elementlast->EvaluateSectionFrame(1, 0, point, rot);

    // For Analytical Formula, see a mechanics of materials textbook (delta = PL^3/3EI)
    double I = 1.0 / 12.0 * width * std::pow(height, 3);
    double Displacement_Theory = (TIP_FORCE * std::pow(length, 3)) / (3.0*E*I);
    double Displacement_Model = point.z();
    ChVector<> Tip_Angles = rot.Q_to_Euler123();

    double Percent_Error = (Displacement_Model - Displacement_Theory) / Displacement_Theory * 100.0;

    bool passed_tests = true;
    if (abs(Percent_Error) > 5.0) {
        passed_tests = false;
    }
    if ((abs(Tip_Angles.x() * CH_C_RAD_TO_DEG) > 0.001) || (abs(Tip_Angles.z() * CH_C_RAD_TO_DEG) > 0.001)) {
        passed_tests = false;
    }

    if (msglvl >= 2) {
        std::cout << "ANCF Tip Position: " << point << "m" << std::endl;
        std::cout << "ANCF Tip Displacement: " << Displacement_Model << "m" << std::endl;
        std::cout << "Analytical Tip Displacement: " << Displacement_Theory << "m" << std::endl;
        std::cout << "Percent Error: " << Percent_Error << "%" << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Tip Displacement Check (Percent Error less than 5%)";
        if (abs(Percent_Error) > 5.0) {
            print_red(" - Test FAILED\n\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl << std::endl;
        }
        std::cout << "Angular misalignment Checks (all angles less than 0.001 deg)";
        if ((abs(Tip_Angles.x() * CH_C_RAD_TO_DEG) > 0.001) || (abs(Tip_Angles.z() * CH_C_RAD_TO_DEG) > 0.001)) {
            print_red(" - Test FAILED\n\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl << std::endl;
        }
    }
    return(passed_tests);
}

template <typename ElementVersion, typename MaterialVersion>
bool ANCFShellTest<ElementVersion, MaterialVersion>::AxialTwistCheck(int msglvl) {
    // =============================================================================
    //  Check the Axial Displacement of a Beam compared to the analytical result
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================

    auto system = new ChSystemSMC();
    // Set gravity to 0 since this is a statics test against an analytical solution
    system->Set_G_acc(ChVector<>(0, 0, 0));

    auto solver = chrono_types::make_shared<ChSolverSparseLU>();
    solver->UseSparsityPatternLearner(true);
    solver->LockSparsityPattern(true);
    solver->SetVerbose(false);
    system->SetSolver(solver);

    // Set up integrator
    system->SetTimestepperType(ChTimestepper::Type::HHT);
    auto integrator = std::static_pointer_cast<ChTimestepperHHT>(system->GetTimestepper());
    integrator->SetAlpha(-0.2);
    integrator->SetMaxiters(100);
    integrator->SetAbsTolerances(1e-5);
    integrator->SetMode(ChTimestepperHHT::POSITION);
    integrator->SetScaling(true);
    integrator->SetVerbose(false);
    integrator->SetModifiedNewton(true);

    // Mesh properties
    int num_elements = 10;
    double length = 1.0;    // m
    double width = 0.1;  // m
    double height = 0.01;  // m
    // Aluminum 7075-T651
    double rho = 2810;  // kg/m^3
    double E = 71.7e9;  // Pa
    double nu = 0.33*0;
    double G = E / (2 * (1 + nu));

    auto material = chrono_types::make_shared<MaterialVersion>(rho, E, nu);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    system->Add(mesh);

    // Populate the mesh container with a the nodes and elements for the meshed beam
    int num_nodes = 2 * num_elements + 2;
    double dx = length / (num_elements);

    // Setup beam cross section gradients to initially align with the global y and z directions
    ChVector<> dir1(1, 0, 0);
    ChVector<> dir2(0, 1, 0);
    ChVector<> dir3(0, 0, 1);

    // Create the first nodes and fix them completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, 0, 0.0), dir1, dir2, dir3);
    nodeA->SetFixed(true);
    mesh->AddNode(nodeA);
    auto nodeD = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, width, 0.0), dir1, dir2, dir3);
    nodeD->SetFixed(true);
    mesh->AddNode(nodeD);

    auto elementlast = chrono_types::make_shared<ElementVersion>();
    std::shared_ptr<ChNodeFEAxyzDDD> nodeEndPoint;

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(dx * i, 0, 0), dir1, dir2, dir3);
        mesh->AddNode(nodeB);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(dx * i, width, 0), dir1, dir2, dir3);
        mesh->AddNode(nodeC);

        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC, nodeD);
        element->SetDimensions(dx, width, height);
        element->SetMaterial(material);
        element->SetAlphaDamp(0.0);
        element->SetGravityOn(
            false);  // Enable the efficient ANCF method for calculating the application of gravity to the element
        element->SetStrainFormulation(ElementVersion::StrainFormulation::CMPoisson);
        // element->SetStrainFormulation(ElementVersion::StrainFormulation::CMNoPoisson);
        mesh->AddElement(element);

        nodeA = nodeB;
        nodeD = nodeC;
        elementlast = element;
    }

    nodeEndPoint = nodeA;

    mesh->SetAutomaticGravity(false);  // Turn off the default method for applying gravity to the mesh since it is less
                                       // efficient for ANCF elements

    // Create a custom atomic (point) load
    class MyLoaderTimeDependentTipLoad : public ChLoaderUVatomic {
    public:
        // Useful: a constructor that also sets ChLoadable
        MyLoaderTimeDependentTipLoad(std::shared_ptr<ChLoadableUV> mloadable) : ChLoaderUVatomic(mloadable) {}

        // Compute F=F(u), the load at U. The load is a 6-row vector, i.e.
        // a wrench: forceX, forceY, forceZ, torqueX, torqueY, torqueZ.
        virtual void ComputeF(
            const double U,              ///< normalized position along the beam axis [-1...1]
            const double V,              ///< normalized position along the beam axis [-1...1]
            ChVectorDynamic<>& F,        ///< Load at U
            ChVectorDynamic<>* state_x,  ///< if != 0, update state (pos. part) to this, then evaluate F
            ChVectorDynamic<>* state_w   ///< if != 0, update state (speed part) to this, then evaluate F
        ) override {
            assert(auxsystem);

            F.setZero();
            F(3) = TIP_MOMENT;  // Apply the moment about the global X axis
        }

    public:
        // add auxiliary data to the class, if you need to access it during ComputeF().
        ChSystem* auxsystem;
    };

    // Create the load container and to the current system
    auto loadcontainer = chrono_types::make_shared<ChLoadContainer>();
    system->Add(loadcontainer);

    // Create a custom load that uses the custom loader above.
    // The ChLoad is a 'manager' for your ChLoader.
    // It is created using templates, that is instancing a ChLoad<my_loader_class>()

    std::shared_ptr<ChLoad<MyLoaderTimeDependentTipLoad> > mload(new ChLoad<MyLoaderTimeDependentTipLoad>(elementlast));
    mload->loader.auxsystem = system;   // initialize auxiliary data of the loader, if needed
    mload->loader.SetApplication(1.0, 0.0);  // specify application point
    loadcontainer->Add(mload);          // add the load to the load container.

    // Find the nonlinear static solution for the system (final twist angle)
    //system->DoStaticLinear();
    system->DoStaticNonlinear(50);

    // Calculate the twist angle of the end of the ANCF beam mesh
    ChVector<> point;
    ChQuaternion<> rot;
    elementlast->EvaluateSectionFrame(1, 0, point, rot);
    ChVector<> Tip_Angles = rot.Q_to_Euler123();

    // For Analytical Formula, see: https://en.wikipedia.org/wiki/Torsion_constant
    double J = 0.312 * width*std::pow(height, 3);
    double T = TIP_MOMENT;
    double Angle_Theory = T * length / (G * J);

    double Percent_Error = (Tip_Angles.x() - Angle_Theory) / Angle_Theory * 100;

    bool passed_tests = true;
    if (abs(Percent_Error) > 20.0) {
        passed_tests = false;
    }
    if ((abs(Tip_Angles.y() * CH_C_RAD_TO_DEG) > 0.001) || (abs(Tip_Angles.z() * CH_C_RAD_TO_DEG) > 0.001)) {
        passed_tests = false;
    }

    if (msglvl >= 2) {
        std::cout << "ANCF Tip Position: " << point << "m" << std::endl;
        std::cout << "ANCF Twist Angles (Euler 123): " << Tip_Angles * CH_C_RAD_TO_DEG << "deg" << std::endl;
        std::cout << "Analytical Twist Angle: " << Angle_Theory * CH_C_RAD_TO_DEG << "deg" << std::endl;
        std::cout << "Percent Error: " << Percent_Error << "%" << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Axial Twist Angle Check (Percent Error less than 20%)";
        if (abs(Percent_Error) > 20.0) {
            print_red(" - Test FAILED\n\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl << std::endl;
        }
        std::cout << "Off axis angle Check (less than 0.001 deg)";
        if ((abs(Tip_Angles.y() * CH_C_RAD_TO_DEG) > 0.001) || (abs(Tip_Angles.z() * CH_C_RAD_TO_DEG) > 0.001)) {
            print_red(" - Test FAILED\n\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl << std::endl;
        }
    }
    return(passed_tests);
}


int main(int argc, char* argv[]) {

#if defined WIN32 || defined _WIN32 || defined WIN64 || defined _WIN64
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    GetConsoleMode(hOut, &dwMode);
    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hOut, dwMode);

    // References:
    //SetConsoleMode() and ENABLE_VIRTUAL_TERMINAL_PROCESSING?
    //https://stackoverflow.com/questions/38772468/setconsolemode-and-enable-virtual-terminal-processing

    // Windows console with ANSI colors handling
    // https://superuser.com/questions/413073/windows-console-with-ansi-colors-handling
#endif

    std::cout << "-------------------------------------" << std::endl;
    ANCFShellTest<ChElementShellANCF_3443_TR01, ChMaterialShellANCF_3443_TR01> ChElementShellANCF_3443_TR01_test;
    if (ChElementShellANCF_3443_TR01_test.RunElementChecks(0))
        print_green("ChElementShellANCF_3443_TR01 Element Checks = PASSED\n");
    else
        print_red("ChElementShellANCF_3443_TR01 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFShellTest<ChElementShellANCF_3443_TR02, ChMaterialShellANCF_3443_TR02> ChElementShellANCF_3443_TR02_test;
    if (ChElementShellANCF_3443_TR02_test.RunElementChecks(0))
        print_green("ChElementShellANCF_3443_TR02 Element Checks = PASSED\n");
    else
        print_red("ChElementShellANCF_3443_TR02 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFShellTest<ChElementShellANCF_3443_TR03, ChMaterialShellANCF_3443_TR03> ChElementShellANCF_3443_TR03_test;
    if (ChElementShellANCF_3443_TR03_test.RunElementChecks(0))
        print_green("ChElementShellANCF_3443_TR03 Element Checks = PASSED\n");
    else
        print_red("ChElementShellANCF_3443_TR03 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFShellTest<ChElementShellANCF_3443_TR04, ChMaterialShellANCF_3443_TR04> ChElementShellANCF_3443_TR04_test;
    if (ChElementShellANCF_3443_TR04_test.RunElementChecks(0))
        print_green("ChElementShellANCF_3443_TR04 Element Checks = PASSED\n");
    else
        print_red("ChElementShellANCF_3443_TR04 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFShellTest<ChElementShellANCF_3443_TR05, ChMaterialShellANCF_3443_TR05> ChElementShellANCF_3443_TR05_test;
    if (ChElementShellANCF_3443_TR05_test.RunElementChecks(0))
        print_green("ChElementShellANCF_3443_TR05 Element Checks = PASSED\n");
    else
        print_red("ChElementShellANCF_3443_TR05 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFShellTest<ChElementShellANCF_3443_TR06, ChMaterialShellANCF_3443_TR06> ChElementShellANCF_3443_TR06_test;
    if (ChElementShellANCF_3443_TR06_test.RunElementChecks(0))
        print_green("ChElementShellANCF_3443_TR06 Element Checks = PASSED\n");
    else
        print_red("ChElementShellANCF_3443_TR06 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFShellTest<ChElementShellANCF_3443_TR07, ChMaterialShellANCF_3443_TR07> ChElementShellANCF_3443_TR07_test;
    if (ChElementShellANCF_3443_TR07_test.RunElementChecks(0))
        print_green("ChElementShellANCF_3443_TR07 Element Checks = PASSED\n");
    else
        print_red("ChElementShellANCF_3443_TR07 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFShellTest<ChElementShellANCF_3443_TR08, ChMaterialShellANCF_3443_TR08> ChElementShellANCF_3443_TR08_test;
    if (ChElementShellANCF_3443_TR08_test.RunElementChecks(0))
        print_green("ChElementShellANCF_3443_TR08 Element Checks = PASSED\n");
    else
        print_red("ChElementShellANCF_3443_TR08 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFShellTest<ChElementShellANCF_3443_TR08b, ChMaterialShellANCF_3443_TR08b> ChElementShellANCF_3443_TR08b_test;
    if (ChElementShellANCF_3443_TR08b_test.RunElementChecks(0))
        print_green("ChElementShellANCF_3443_TR08b Element Checks = PASSED\n");
    else
        print_red("ChElementShellANCF_3443_TR08b Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFShellTest<ChElementShellANCF_3443_TR09, ChMaterialShellANCF_3443_TR09> ChElementShellANCF_3443_TR09_test;
    if (ChElementShellANCF_3443_TR09_test.RunElementChecks(0))
        print_green("ChElementShellANCF_3443_TR09 Element Checks = PASSED\n");
    else
        print_red("ChElementShellANCF_3443_TR09 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFShellTest<ChElementShellANCF_3443_TR10, ChMaterialShellANCF_3443_TR10> ChElementShellANCF_3443_TR10_test;
    if (ChElementShellANCF_3443_TR10_test.RunElementChecks(0))
        print_green("ChElementShellANCF_3443_TR10 Element Checks = PASSED\n");
    else
        print_red("ChElementShellANCF_3443_TR10 Element Checks = FAILED\n");

    return 0;
}

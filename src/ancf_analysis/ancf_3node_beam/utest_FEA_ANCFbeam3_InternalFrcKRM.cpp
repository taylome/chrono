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

#include "chrono/fea/ChElementBeamANCF.h"
#include "chrono/fea/ChElementBeamANCF_TR01.h"
#include "chrono/fea/ChElementBeamANCF_TR02.h"
#include "chrono/fea/ChElementBeamANCF_TR03.h"
#include "chrono/fea/ChElementBeamANCF_TR04.h"
#include "chrono/fea/ChElementBeamANCF_TR05.h"
#include "chrono/fea/ChElementBeamANCF_TR06.h"
#include "chrono/fea/ChElementBeamANCF_TR07.h"
#include "chrono/fea/ChElementBeamANCF_TR08.h"
#include "chrono/fea/ChElementBeamANCF_TR08b.h"
#include "chrono/fea/ChElementBeamANCF_TR09.h"
#include "chrono/fea/ChElementBeamANCF_TR10.h"
#include "chrono/fea/ChElementBeamANCF_TR11.h"

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
class ANCFBeamTest {
public:
    ANCFBeamTest();

    ~ANCFBeamTest() { delete m_system; }

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
    std::shared_ptr<ChNodeFEAxyzDD> m_nodeC;
};

// =============================================================================

template <typename ElementVersion, typename MaterialVersion>
ANCFBeamTest<ElementVersion, MaterialVersion>::ANCFBeamTest() {
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
    double length = 1;      //m
    double width = 0.1;     //m
    double thickness = 0.1; //m
    double rho = 7850; //kg/m^3
    double E = 210e9; //Pa
    double nu = 0.3;
    double k1 = 10 * (1 + nu) / (12 + 11 * nu);  // Timoshenko shear correction coefficient for a rectangular cross-section
    double k2 = k1;                              // Timoshenko shear correction coefficient for a rectangular cross-section

    auto material = chrono_types::make_shared<MaterialVersion>(rho, E, nu, k1, k2);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    m_system->Add(mesh);

    double beam_angle_rad = 0;
    //Rotate the cross section gradients to match the sign convention in the experiment
    ChVector<> dir1(0, cos(-beam_angle_rad), sin(-beam_angle_rad));
    ChVector<> dir2(0, -sin(-beam_angle_rad), cos(-beam_angle_rad));

    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0, 0.0), dir1, dir2);
    nodeA->SetFixed(true);
    mesh->AddNode(nodeA);
    auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(length, 0, 0), dir1, dir2);
    mesh->AddNode(nodeB);
    auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(length / 2, 0, 0), dir1, dir2);
    mesh->AddNode(nodeC);
    m_nodeC = nodeC;

    auto element = chrono_types::make_shared<ElementVersion>();
    element->SetNodes(nodeA, nodeB, nodeC);
    element->SetDimensions(length, thickness, width);
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
bool ANCFBeamTest<ElementVersion, MaterialVersion>::RunElementChecks(int msglvl) {
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

    tests_passed = (tests_passed & AxialDisplacementCheck(msglvl));
    tests_passed = (tests_passed & CantileverCheck(msglvl));
    tests_passed = (tests_passed & AxialTwistCheck(msglvl));

    return(tests_passed);
}

template <typename ElementVersion, typename MaterialVersion>
bool ANCFBeamTest<ElementVersion, MaterialVersion>::MassMatrixCheck(int msglvl) {
    // =============================================================================
    //  Check the Mass Matrix
    //  (Result should be nearly exact - No expected error)
    // =============================================================================
    ChMatrixNM<double, 27, 27> Expected_MassMatrix;
    Expected_MassMatrix <<
        10.4666666666667, 0, 0, 0, 0, 0, 0, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, 0, 0, 0, 5.23333333333333, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 10.4666666666667, 0, 0, 0, 0, 0, 0, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, 0, 0, 0, 5.23333333333333, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 10.4666666666667, 0, 0, 0, 0, 0, 0, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, 0, 0, 0, 5.23333333333333, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0.00872222222222222, 0, 0, 0, 0, 0, 0, 0, 0, -0.00218055555555556, 0, 0, 0, 0, 0, 0, 0, 0, 0.00436111111111111, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0.00872222222222222, 0, 0, 0, 0, 0, 0, 0, 0, -0.00218055555555556, 0, 0, 0, 0, 0, 0, 0, 0, 0.00436111111111111, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0.00872222222222222, 0, 0, 0, 0, 0, 0, 0, 0, -0.00218055555555556, 0, 0, 0, 0, 0, 0, 0, 0, 0.00436111111111111, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0.00872222222222222, 0, 0, 0, 0, 0, 0, 0, 0, -0.00218055555555556, 0, 0, 0, 0, 0, 0, 0, 0, 0.00436111111111111, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0.00872222222222222, 0, 0, 0, 0, 0, 0, 0, 0, -0.00218055555555556, 0, 0, 0, 0, 0, 0, 0, 0, 0.00436111111111111, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0.00872222222222222, 0, 0, 0, 0, 0, 0, 0, 0, -0.00218055555555556, 0, 0, 0, 0, 0, 0, 0, 0, 0.00436111111111111,
        -2.61666666666667, 0, 0, 0, 0, 0, 0, 0, 0, 10.4666666666667, 0, 0, 0, 0, 0, 0, 0, 0, 5.23333333333333, 0, 0, 0, 0, 0, 0, 0, 0,
        0, -2.61666666666667, 0, 0, 0, 0, 0, 0, 0, 0, 10.4666666666667, 0, 0, 0, 0, 0, 0, 0, 0, 5.23333333333333, 0, 0, 0, 0, 0, 0, 0,
        0, 0, -2.61666666666667, 0, 0, 0, 0, 0, 0, 0, 0, 10.4666666666667, 0, 0, 0, 0, 0, 0, 0, 0, 5.23333333333333, 0, 0, 0, 0, 0, 0,
        0, 0, 0, -0.00218055555555556, 0, 0, 0, 0, 0, 0, 0, 0, 0.00872222222222222, 0, 0, 0, 0, 0, 0, 0, 0, 0.00436111111111111, 0, 0, 0, 0, 0,
        0, 0, 0, 0, -0.00218055555555556, 0, 0, 0, 0, 0, 0, 0, 0, 0.00872222222222222, 0, 0, 0, 0, 0, 0, 0, 0, 0.00436111111111111, 0, 0, 0, 0,
        0, 0, 0, 0, 0, -0.00218055555555556, 0, 0, 0, 0, 0, 0, 0, 0, 0.00872222222222222, 0, 0, 0, 0, 0, 0, 0, 0, 0.00436111111111111, 0, 0, 0,
        0, 0, 0, 0, 0, 0, -0.00218055555555556, 0, 0, 0, 0, 0, 0, 0, 0, 0.00872222222222222, 0, 0, 0, 0, 0, 0, 0, 0, 0.00436111111111111, 0, 0,
        0, 0, 0, 0, 0, 0, 0, -0.00218055555555556, 0, 0, 0, 0, 0, 0, 0, 0, 0.00872222222222222, 0, 0, 0, 0, 0, 0, 0, 0, 0.00436111111111111, 0,
        0, 0, 0, 0, 0, 0, 0, 0, -0.00218055555555556, 0, 0, 0, 0, 0, 0, 0, 0, 0.00872222222222222, 0, 0, 0, 0, 0, 0, 0, 0, 0.00436111111111111,
        5.23333333333333, 0, 0, 0, 0, 0, 0, 0, 0, 5.23333333333333, 0, 0, 0, 0, 0, 0, 0, 0, 41.8666666666667, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 5.23333333333333, 0, 0, 0, 0, 0, 0, 0, 0, 5.23333333333333, 0, 0, 0, 0, 0, 0, 0, 0, 41.8666666666667, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 5.23333333333333, 0, 0, 0, 0, 0, 0, 0, 0, 5.23333333333333, 0, 0, 0, 0, 0, 0, 0, 0, 41.8666666666667, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0.00436111111111111, 0, 0, 0, 0, 0, 0, 0, 0, 0.00436111111111111, 0, 0, 0, 0, 0, 0, 0, 0, 0.0348888888888889, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0.00436111111111111, 0, 0, 0, 0, 0, 0, 0, 0, 0.00436111111111111, 0, 0, 0, 0, 0, 0, 0, 0, 0.0348888888888889, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0.00436111111111111, 0, 0, 0, 0, 0, 0, 0, 0, 0.00436111111111111, 0, 0, 0, 0, 0, 0, 0, 0, 0.0348888888888889, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0.00436111111111111, 0, 0, 0, 0, 0, 0, 0, 0, 0.00436111111111111, 0, 0, 0, 0, 0, 0, 0, 0, 0.0348888888888889, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0.00436111111111111, 0, 0, 0, 0, 0, 0, 0, 0, 0.00436111111111111, 0, 0, 0, 0, 0, 0, 0, 0, 0.0348888888888889, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0.00436111111111111, 0, 0, 0, 0, 0, 0, 0, 0, 0.00436111111111111, 0, 0, 0, 0, 0, 0, 0, 0, 0.0348888888888889;

    ChMatrixDynamic<double> MassMatrix;
    MassMatrix.resize(27, 27);
    m_element->ComputeKRMmatricesGlobal(MassMatrix, 0, 0, 1);
    ChMatrixNM<double, 9, 9> MassMatrix_compact;
    for (unsigned int i = 0; i < 9; i++) {
        for (unsigned int j = 0; j < 9; j++) {
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
bool ANCFBeamTest<ElementVersion, MaterialVersion>::GeneralizedGravityForceCheck(int msglvl) {
    // =============================================================================
    //  Generalized Force due to Gravity
    //  (Result should be nearly exact - No expected error)
    // =============================================================================
    ChVectorN<double, 27> Expected_InternalForceDueToGravity;
    Expected_InternalForceDueToGravity <<
        0,
        0,
        -128.303670833333,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -128.303670833333,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -513.214683333333,
        0,
        0,
        0,
        0,
        0,
        0;

    ChVectorDynamic<double> InternalForceNoDispNoVelWithGravity;
    InternalForceNoDispNoVelWithGravity.resize(27);
    m_element->SetGravityOn(true);
    m_element->ComputeInternalForces(InternalForceNoDispNoVelWithGravity);

    ChVectorDynamic<double> InternalForceNoDispNoVelNoGravity;
    InternalForceNoDispNoVelNoGravity.resize(27);
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
bool ANCFBeamTest<ElementVersion, MaterialVersion>::GeneralizedInternalForceNoDispNoVelCheck(int msglvl) {
    // =============================================================================
    //  Check the Internal Force at Zero Displacement, Zero Velocity
    //  (should equal all 0's by definition)
    //  (Assumes that the element has not been changed from the initialized state)
    // =============================================================================

    ChVectorDynamic<double> InternalForceNoDispNoVelNoGravity;
    InternalForceNoDispNoVelNoGravity.resize(27);
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
bool ANCFBeamTest<ElementVersion, MaterialVersion>::GeneralizedInternalForceSmallDispNoVelCheck(int msglvl) {
    // =============================================================================
    //  Check the Internal Force at a Given Displacement 
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================

    ChVectorN<double, 27> Expected_InternalForceSmallDispNoVelNoGravity;
    Expected_InternalForceSmallDispNoVelNoGravity <<
        7538.46153846154,
        0,
        1830101.54409251,
        0,
        -969.230769230769,
        0,
        -457516.339869281,
        0,
        -2067.26998491704,
        -7538.46153846154,
        0,
        1830101.54409251,
        0,
        -969.230769230769,
        0,
        457516.339869281,
        0,
        -2067.26998491704,
        0,
        0,
        -3660203.08818502,
        0,
        -1292.30769230769,
        0,
        0,
        0,
        -2756.35997988939;

    //Setup the test conditions
    ChVector<double> OriginalPos = m_nodeC->GetPos();
    m_nodeC->SetPos(ChVector<>(m_nodeC->GetPos().x(), m_nodeC->GetPos().y(), 0.001));

    ChVectorDynamic<double> InternalForceSmallDispNoVelNoGravity;
    InternalForceSmallDispNoVelNoGravity.resize(27);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceSmallDispNoVelNoGravity);

    //Reset the element conditions back to its original values
    m_nodeC->SetPos(OriginalPos);

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
bool ANCFBeamTest<ElementVersion, MaterialVersion>::GeneralizedInternalForceNoDispSmallVelCheck(int msglvl) {
    // =============================================================================
    //  Check the Internal Force at a No Displacement with a Given Nodal Velocity
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================

    ChVectorN<double, 27> Expected_InternalForceNoDispSmallVelNoGravity;
    Expected_InternalForceNoDispSmallVelNoGravity <<
        0,
        0,
        18300.6535947712,
        0,
        0,
        0,
        -4575.16339869281,
        0,
        0,
        0,
        0,
        18300.6535947712,
        0,
        0,
        0,
        4575.16339869281,
        0,
        0,
        0,
        0,
        -36601.3071895425,
        0,
        0,
        0,
        0,
        0,
        0;

    //Setup the test conditions
    ChVector<double> OriginalVel = m_nodeC->GetPos_dt();
    m_nodeC->SetPos_dt(ChVector<>(0.0, 0.0, 0.001));
    m_element->SetAlphaDamp(0.01);

    ChVectorDynamic<double> InternalForceNoDispSmallVelNoGravity;
    InternalForceNoDispSmallVelNoGravity.resize(27);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceNoDispSmallVelNoGravity);

    //Reset the element conditions back to its original values
    m_nodeC->SetPos_dt(OriginalVel);
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
bool ANCFBeamTest<ElementVersion, MaterialVersion>::JacobianNoDispNoVelNoDampingCheck(int msglvl) {
    // =============================================================================
    //  Check the Jacobian at No Displacement/Velocity - No Damping
    //  (some small error expected depending on the formulation/steps used)
    //  (The R contribution should be all zeros since Damping is not enabled)
    // =============================================================================
    ChMatrixNM<double, 27, 27> Expected_JacobianK_NoDispNoVelNoDamping;
    Expected_JacobianK_NoDispNoVelNoDamping <<
        6596153846.15385, 0, 0, 0, -605769230.769231, 0, 0, 0, -605769230.769231, 942307692.307692, 0, 0, 0, 201923076.923077, 0, 0, 0, 201923076.923077, -7538461538.46154, 0, 0, 0, -807692307.692308, 0, 0, 0, -807692307.692308,
        0, 1601307189.54248, 0, -343137254.901961, 0, 0, 0, 0, 0, 0, 228758169.934641, 0, 114379084.967320, 0, 0, 0, 0, 0, 0, -1830065359.47712, 0, -457516339.869281, 0, 0, 0, 0, 0,
        0, 0, 1601307189.54248, 0, 0, 0, -343137254.901961, 0, 0, 0, 0, 228758169.934641, 0, 0, 0, 114379084.967320, 0, 0, 0, 0, -1830065359.47712, 0, 0, 0, -457516339.869281, 0, 0,
        0, -343137254.901961, 0, 95586601.3071895, 0, 0, 0, 0, 0, 0, -114379084.967320, 0, -22292483.6601307, 0, 0, 0, 0, 0, 0, 457516339.869281, 0, 41084967.3202614, 0, 0, 0, 0, 0,
        -605769230.769231, 0, 0, 0, 378257499.581029, 0, 0, 0, 161538461.538462, -201923076.923077, 0, 0, 0, -94040137.4224904, 0, 0, 0, -40384615.3846154, 807692307.692308, 0, 0, 0, 186936483.995308, 0, 0, 0, 80769230.7692308,
        0, 0, 0, 0, 0, 109026730.350260, 0, 107692307.692308, 0, 0, 0, 0, 0, 0, -26732445.1147981, 0, -26923076.9230769, 0, 0, 0, 0, 0, 0, 52321099.3799229, 0, 53846153.8461539, 0,
        0, 0, -343137254.901961, 0, 0, 0, 95586601.3071895, 0, 0, 0, 0, -114379084.967320, 0, 0, 0, -22292483.6601307, 0, 0, 0, 0, 457516339.869281, 0, 0, 0, 41084967.3202614, 0, 0,
        0, 0, 0, 0, 0, 107692307.692308, 0, 109026730.350260, 0, 0, 0, 0, 0, 0, -26923076.9230769, 0, -26732445.1147981, 0, 0, 0, 0, 0, 0, 53846153.8461539, 0, 52321099.3799229, 0,
        -605769230.769231, 0, 0, 0, 161538461.538462, 0, 0, 0, 378257499.581029, -201923076.923077, 0, 0, 0, -40384615.3846154, 0, 0, 0, -94040137.4224904, 807692307.692308, 0, 0, 0, 80769230.7692308, 0, 0, 0, 186936483.995308,
        942307692.307692, 0, 0, 0, -201923076.923077, 0, 0, 0, -201923076.923077, 6596153846.15385, 0, 0, 0, 605769230.769231, 0, 0, 0, 605769230.769231, -7538461538.46154, 0, 0, 0, 807692307.692308, 0, 0, 0, 807692307.692308,
        0, 228758169.934641, 0, -114379084.967320, 0, 0, 0, 0, 0, 0, 1601307189.54248, 0, 343137254.901961, 0, 0, 0, 0, 0, 0, -1830065359.47712, 0, 457516339.869281, 0, 0, 0, 0, 0,
        0, 0, 228758169.934641, 0, 0, 0, -114379084.967320, 0, 0, 0, 0, 1601307189.54248, 0, 0, 0, 343137254.901961, 0, 0, 0, 0, -1830065359.47712, 0, 0, 0, 457516339.869281, 0, 0,
        0, 114379084.967320, 0, -22292483.6601307, 0, 0, 0, 0, 0, 0, 343137254.901961, 0, 95586601.3071895, 0, 0, 0, 0, 0, 0, -457516339.869281, 0, 41084967.3202614, 0, 0, 0, 0, 0,
        201923076.923077, 0, 0, 0, -94040137.4224904, 0, 0, 0, -40384615.3846154, 605769230.769231, 0, 0, 0, 378257499.581029, 0, 0, 0, 161538461.538462, -807692307.692308, 0, 0, 0, 186936483.995308, 0, 0, 0, 80769230.7692308,
        0, 0, 0, 0, 0, -26732445.1147981, 0, -26923076.9230769, 0, 0, 0, 0, 0, 0, 109026730.350260, 0, 107692307.692308, 0, 0, 0, 0, 0, 0, 52321099.3799229, 0, 53846153.8461539, 0,
        0, 0, 114379084.967320, 0, 0, 0, -22292483.6601307, 0, 0, 0, 0, 343137254.901961, 0, 0, 0, 95586601.3071895, 0, 0, 0, 0, -457516339.869281, 0, 0, 0, 41084967.3202614, 0, 0,
        0, 0, 0, 0, 0, -26923076.9230769, 0, -26732445.1147981, 0, 0, 0, 0, 0, 0, 107692307.692308, 0, 109026730.350260, 0, 0, 0, 0, 0, 0, 53846153.8461539, 0, 52321099.3799229, 0,
        201923076.923077, 0, 0, 0, -40384615.3846154, 0, 0, 0, -94040137.4224904, 605769230.769231, 0, 0, 0, 161538461.538462, 0, 0, 0, 378257499.581029, -807692307.692308, 0, 0, 0, 80769230.7692308, 0, 0, 0, 186936483.995308,
        -7538461538.46154, 0, 0, 0, 807692307.692308, 0, 0, 0, 807692307.692308, -7538461538.46154, 0, 0, 0, -807692307.692308, 0, 0, 0, -807692307.692308, 15076923076.9231, 0, 0, 0, 0, 0, 0, 0, 0,
        0, -1830065359.47712, 0, 457516339.869281, 0, 0, 0, 0, 0, 0, -1830065359.47712, 0, -457516339.869281, 0, 0, 0, 0, 0, 0, 3660130718.95425, 0, 0, 0, 0, 0, 0, 0,
        0, 0, -1830065359.47712, 0, 0, 0, 457516339.869281, 0, 0, 0, 0, -1830065359.47712, 0, 0, 0, -457516339.869281, 0, 0, 0, 0, 3660130718.95425, 0, 0, 0, 0, 0, 0,
        0, -457516339.869281, 0, 41084967.3202614, 0, 0, 0, 0, 0, 0, 457516339.869281, 0, 41084967.3202614, 0, 0, 0, 0, 0, 0, 0, 0, 375346405.228758, 0, 0, 0, 0, 0,
        -807692307.692308, 0, 0, 0, 186936483.995308, 0, 0, 0, 80769230.7692308, 807692307.692308, 0, 0, 0, 186936483.995308, 0, 0, 0, 80769230.7692308, 0, 0, 0, 0, 1510742416.62477, 0, 0, 0, 646153846.153846,
        0, 0, 0, 0, 0, 52321099.3799229, 0, 53846153.8461539, 0, 0, 0, 0, 0, 0, 52321099.3799229, 0, 53846153.8461539, 0, 0, 0, 0, 0, 0, 433819339.701693, 0, 430769230.769231, 0,
        0, 0, -457516339.869281, 0, 0, 0, 41084967.3202614, 0, 0, 0, 0, 457516339.869281, 0, 0, 0, 41084967.3202614, 0, 0, 0, 0, 0, 0, 0, 0, 375346405.228758, 0, 0,
        0, 0, 0, 0, 0, 53846153.8461539, 0, 52321099.3799229, 0, 0, 0, 0, 0, 0, 53846153.8461539, 0, 52321099.3799229, 0, 0, 0, 0, 0, 0, 430769230.769231, 0, 433819339.701693, 0,
        -807692307.692308, 0, 0, 0, 80769230.7692308, 0, 0, 0, 186936483.995308, 807692307.692308, 0, 0, 0, 80769230.7692308, 0, 0, 0, 186936483.995308, 0, 0, 0, 0, 646153846.153846, 0, 0, 0, 1510742416.62477;


    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation
    ChVectorDynamic<double> InternalForceNoDispNoVelNoGravity;
    InternalForceNoDispNoVelNoGravity.resize(27);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceNoDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_NoDispNoVelNoDamping;
    JacobianK_NoDispNoVelNoDamping.resize(27, 27);
    m_element->ComputeKRMmatricesGlobal(JacobianK_NoDispNoVelNoDamping, 1.0, 0.0, 0.0);

    ChMatrixDynamic<double> JacobianR_NoDispNoVelNoDamping;
    JacobianR_NoDispNoVelNoDamping.resize(27, 27);
    m_element->ComputeKRMmatricesGlobal(JacobianR_NoDispNoVelNoDamping, 0.0, 1.0, 0.0);

    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_NoDispNoVelNoDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(27, 27);
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
bool ANCFBeamTest<ElementVersion, MaterialVersion>::JacobianSmallDispNoVelNoDampingCheck(int msglvl) {
    // =============================================================================
    //  Check the Jacobian at Small Displacement No Velocity - No Damping
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================
    ChMatrixNM<double, 27, 27> Expected_JacobianK_SmallDispNoVelNoDamping;
    Expected_JacobianK_SmallDispNoVelNoDamping <<
        6596179476.92308, 0, 15076923.0769231, 0, -605769230.769231, 0, -1006535.94771242, 0, -605769230.769231, 942318246.153846, 0, 0, 0, 201923076.923077, 0, -91503.2679738562, 0, 201923076.923077, -7538497723.07692, 0, -15076923.0769231, 0, -807692307.692308, 0, -732026.143790850, 0, -807692307.692308,
        0, 1601332820.31171, 0, -343137254.901961, 0, -1006535.94771242, 0, -1006535.94771242, 0, 0, 228768723.780794, 0, 114379084.967320, 0, -91503.2679738562, 0, -91503.2679738562, 0, 0, -1830101544.09251, 0, -457516339.869281, 0, -732026.143790850, 0, -732026.143790850, 0,
        15076923.0769231, 0, 1601384081.85018, 0, -1776923.07692308, 0, -343137254.901961, 0, -3789994.97234791, 0, 0, 228789831.473102, 0, -161538.461538462, 0, 114379084.967320, 0, -344544.997486174, -15076923.0769231, 0, -1830173913.32328, 0, -1292307.69230769, 0, -457516339.869281, 0, -2756359.97988939,
        0, -343137254.901961, 0, 95587447.9430870, 0, 283843.137254902, 0, 0, 0, 0, -114379084.967320, 0, -22292615.5883359, 0, 0, 0, 0, 0, 0, 457516339.869281, 0, 41085221.8433384, 0, 173673.202614379, 0, 0, 0,
        -605769230.769231, 0, -1776923.07692308, 0, 378258346.216926, 0, 0, 0, 161538461.538462, -201923076.923077, 0, -161538.461538462, 0, -94040269.3506955, 0, 0, 0, -40384615.3846154, 807692307.692308, 0, 1938461.53846154, 0, 186936738.518384, 0, 0, 0, 80769230.7692308,
        0, -1006535.94771242, 0, 283843.137254902, 0, 109028549.895961, 0, 107692307.692308, 0, 0, -91503.2679738562, 0, 0, 0, -26732720.8390816, 0, -26923076.9230769, 0, 0, 1098039.21568627, 0, 173673.202614379, 0, 52321622.8284900, 0, 53846153.8461539, 0,
        -1006535.94771242, 0, -343137254.901961, 0, 0, 0, 95587447.9430870, 0, 283843.137254902, -91503.2679738562, 0, -114379084.967320, 0, 0, 0, -22292615.5883359, 0, 0, 1098039.21568627, 0, 457516339.869281, 0, 0, 0, 41085221.8433384, 0, 173673.202614379,
        0, -1006535.94771242, 0, 0, 0, 107692307.692308, 0, 109027576.986157, 0, 0, -91503.2679738562, 0, 0, 0, -26923076.9230769, 0, -26732577.0430032, 0, 0, 1098039.21568627, 0, 0, 0, 53846153.8461539, 0, 52321353.9029998, 0,
        -605769230.769231, 0, -3789994.97234791, 0, 161538461.538462, 0, 283843.137254902, 0, 378259319.126730, -201923076.923077, 0, -344544.997486174, 0, -40384615.3846154, 0, 0, 0, -94040413.1467739, 807692307.692308, 0, 4134539.96983409, 0, 80769230.7692308, 0, 173673.202614379, 0, 186937007.443875,
        942318246.153846, 0, 0, 0, -201923076.923077, 0, -91503.2679738562, 0, -201923076.923077, 6596179476.92308, 0, -15076923.0769231, 0, 605769230.769231, 0, -1006535.94771242, 0, 605769230.769231, -7538497723.07692, 0, 15076923.0769231, 0, 807692307.692308, 0, -732026.143790850, 0, 807692307.692308,
        0, 228768723.780794, 0, -114379084.967320, 0, -91503.2679738562, 0, -91503.2679738562, 0, 0, 1601332820.31171, 0, 343137254.901961, 0, -1006535.94771242, 0, -1006535.94771242, 0, 0, -1830101544.09251, 0, 457516339.869281, 0, -732026.143790850, 0, -732026.143790850, 0,
        0, 0, 228789831.473102, 0, -161538.461538462, 0, -114379084.967320, 0, -344544.997486174, -15076923.0769231, 0, 1601384081.85018, 0, -1776923.07692308, 0, 343137254.901961, 0, -3789994.97234791, 15076923.0769231, 0, -1830173913.32328, 0, -1292307.69230769, 0, 457516339.869281, 0, -2756359.97988939,
        0, 114379084.967320, 0, -22292615.5883359, 0, 0, 0, 0, 0, 0, 343137254.901961, 0, 95587447.9430870, 0, -283843.137254902, 0, 0, 0, 0, -457516339.869281, 0, 41085221.8433384, 0, -173673.202614379, 0, 0, 0,
        201923076.923077, 0, -161538.461538462, 0, -94040269.3506955, 0, 0, 0, -40384615.3846154, 605769230.769231, 0, -1776923.07692308, 0, 378258346.216926, 0, 0, 0, 161538461.538462, -807692307.692308, 0, 1938461.53846154, 0, 186936738.518384, 0, 0, 0, 80769230.7692308,
        0, -91503.2679738562, 0, 0, 0, -26732720.8390816, 0, -26923076.9230769, 0, 0, -1006535.94771242, 0, -283843.137254902, 0, 109028549.895961, 0, 107692307.692308, 0, 0, 1098039.21568627, 0, -173673.202614379, 0, 52321622.8284900, 0, 53846153.8461539, 0,
        -91503.2679738562, 0, 114379084.967320, 0, 0, 0, -22292615.5883359, 0, 0, -1006535.94771242, 0, 343137254.901961, 0, 0, 0, 95587447.9430870, 0, -283843.137254902, 1098039.21568627, 0, -457516339.869281, 0, 0, 0, 41085221.8433384, 0, -173673.202614379,
        0, -91503.2679738562, 0, 0, 0, -26923076.9230769, 0, -26732577.0430032, 0, 0, -1006535.94771242, 0, 0, 0, 107692307.692308, 0, 109027576.986157, 0, 0, 1098039.21568627, 0, 0, 0, 53846153.8461539, 0, 52321353.9029998, 0,
        201923076.923077, 0, -344544.997486174, 0, -40384615.3846154, 0, 0, 0, -94040413.1467739, 605769230.769231, 0, -3789994.97234791, 0, 161538461.538462, 0, -283843.137254902, 0, 378259319.126730, -807692307.692308, 0, 4134539.96983409, 0, 80769230.7692308, 0, -173673.202614379, 0, 186937007.443875,
        -7538497723.07692, 0, -15076923.0769231, 0, 807692307.692308, 0, 1098039.21568627, 0, 807692307.692308, -7538497723.07692, 0, 15076923.0769231, 0, -807692307.692308, 0, 1098039.21568627, 0, -807692307.692308, 15076995446.1538, 0, 0, 0, 0, 0, 1464052.28758170, 0, 0,
        0, -1830101544.09251, 0, 457516339.869281, 0, 1098039.21568627, 0, 1098039.21568627, 0, 0, -1830101544.09251, 0, -457516339.869281, 0, 1098039.21568627, 0, 1098039.21568627, 0, 0, 3660203088.18502, 0, 0, 0, 1464052.28758170, 0, 1464052.28758170, 0,
        -15076923.0769231, 0, -1830173913.32328, 0, 1938461.53846154, 0, 457516339.869281, 0, 4134539.96983409, 15076923.0769231, 0, -1830173913.32328, 0, 1938461.53846154, 0, -457516339.869281, 0, 4134539.96983409, 0, 0, 3660347826.64656, 0, 2584615.38461538, 0, 0, 0, 5512719.95977878,
        0, -457516339.869281, 0, 41085221.8433384, 0, 173673.202614379, 0, 0, 0, 0, 457516339.869281, 0, 41085221.8433384, 0, -173673.202614379, 0, 0, 0, 0, 0, 0, 375347188.490297, 0, 0, 0, 0, 0,
        -807692307.692308, 0, -1292307.69230769, 0, 186936738.518384, 0, 0, 0, 80769230.7692308, 807692307.692308, 0, -1292307.69230769, 0, 186936738.518384, 0, 0, 0, 80769230.7692308, 0, 0, 2584615.38461538, 0, 1510743199.88631, 0, 0, 0, 646153846.153846,
        0, -732026.143790850, 0, 173673.202614379, 0, 52321622.8284900, 0, 53846153.8461539, 0, 0, -732026.143790850, 0, -173673.202614379, 0, 52321622.8284900, 0, 53846153.8461539, 0, 0, 1464052.28758170, 0, 0, 0, 433821049.164538, 0, 430769230.769231, 0,
        -732026.143790850, 0, -457516339.869281, 0, 0, 0, 41085221.8433384, 0, 173673.202614379, -732026.143790850, 0, 457516339.869281, 0, 0, 0, 41085221.8433384, 0, -173673.202614379, 1464052.28758170, 0, 0, 0, 0, 0, 375347188.490297, 0, 0,
        0, -732026.143790850, 0, 0, 0, 53846153.8461539, 0, 52321353.9029998, 0, 0, -732026.143790850, 0, 0, 0, 53846153.8461539, 0, 52321353.9029998, 0, 0, 1464052.28758170, 0, 0, 0, 430769230.769231, 0, 433820122.963231, 0,
        -807692307.692308, 0, -2756359.97988939, 0, 80769230.7692308, 0, 173673.202614379, 0, 186937007.443875, 807692307.692308, 0, -2756359.97988939, 0, 80769230.7692308, 0, -173673.202614379, 0, 186937007.443875, 0, 0, 5512719.95977878, 0, 646153846.153846, 0, 0, 0, 1510744126.08762;


    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation
    
    //Setup the test conditions
    ChVector<double> OriginalPos = m_nodeC->GetPos();
    m_nodeC->SetPos(ChVector<>(m_nodeC->GetPos().x(), m_nodeC->GetPos().y(), 0.001));

    ChVectorDynamic<double> InternalForceSmallDispNoVelNoGravity;
    InternalForceSmallDispNoVelNoGravity.resize(27);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceSmallDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_SmallDispNoVelNoDamping;
    JacobianK_SmallDispNoVelNoDamping.resize(27, 27);
    m_element->ComputeKRMmatricesGlobal(JacobianK_SmallDispNoVelNoDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_SmallDispNoVelNoDamping;
    JacobianR_SmallDispNoVelNoDamping.resize(27, 27);
    m_element->ComputeKRMmatricesGlobal(JacobianR_SmallDispNoVelNoDamping, 0, 1, 0);

    //Reset the element conditions back to its original values
    m_nodeC->SetPos(OriginalPos);


    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_SmallDispNoVelNoDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(27, 27);
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
bool ANCFBeamTest<ElementVersion, MaterialVersion>::JacobianNoDispNoVelWithDampingCheck(int msglvl) {
    // =============================================================================
    //  Check the Jacobian at No Displacement/Velocity - With Damping
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================

    ChMatrixNM<double, 27, 27> Expected_JacobianK_NoDispNoVelWithDamping;
    Expected_JacobianK_NoDispNoVelWithDamping <<
        6596153846.15385, 0, 0, 0, -605769230.769231, 0, 0, 0, -605769230.769231, 942307692.307692, 0, 0, 0, 201923076.923077, 0, 0, 0, 201923076.923077, -7538461538.46154, 0, 0, 0, -807692307.692308, 0, 0, 0, -807692307.692308,
        0, 1601307189.54248, 0, -343137254.901961, 0, 0, 0, 0, 0, 0, 228758169.934641, 0, 114379084.967320, 0, 0, 0, 0, 0, 0, -1830065359.47712, 0, -457516339.869281, 0, 0, 0, 0, 0,
        0, 0, 1601307189.54248, 0, 0, 0, -343137254.901961, 0, 0, 0, 0, 228758169.934641, 0, 0, 0, 114379084.967320, 0, 0, 0, 0, -1830065359.47712, 0, 0, 0, -457516339.869281, 0, 0,
        0, -343137254.901961, 0, 95586601.3071895, 0, 0, 0, 0, 0, 0, -114379084.967320, 0, -22292483.6601307, 0, 0, 0, 0, 0, 0, 457516339.869281, 0, 41084967.3202614, 0, 0, 0, 0, 0,
        -605769230.769231, 0, 0, 0, 378257499.581029, 0, 0, 0, 161538461.538462, -201923076.923077, 0, 0, 0, -94040137.4224904, 0, 0, 0, -40384615.3846154, 807692307.692308, 0, 0, 0, 186936483.995308, 0, 0, 0, 80769230.7692308,
        0, 0, 0, 0, 0, 109026730.350260, 0, 107692307.692308, 0, 0, 0, 0, 0, 0, -26732445.1147981, 0, -26923076.9230769, 0, 0, 0, 0, 0, 0, 52321099.3799229, 0, 53846153.8461539, 0,
        0, 0, -343137254.901961, 0, 0, 0, 95586601.3071895, 0, 0, 0, 0, -114379084.967320, 0, 0, 0, -22292483.6601307, 0, 0, 0, 0, 457516339.869281, 0, 0, 0, 41084967.3202614, 0, 0,
        0, 0, 0, 0, 0, 107692307.692308, 0, 109026730.350260, 0, 0, 0, 0, 0, 0, -26923076.9230769, 0, -26732445.1147981, 0, 0, 0, 0, 0, 0, 53846153.8461539, 0, 52321099.3799229, 0,
        -605769230.769231, 0, 0, 0, 161538461.538462, 0, 0, 0, 378257499.581029, -201923076.923077, 0, 0, 0, -40384615.3846154, 0, 0, 0, -94040137.4224904, 807692307.692308, 0, 0, 0, 80769230.7692308, 0, 0, 0, 186936483.995308,
        942307692.307692, 0, 0, 0, -201923076.923077, 0, 0, 0, -201923076.923077, 6596153846.15385, 0, 0, 0, 605769230.769231, 0, 0, 0, 605769230.769231, -7538461538.46154, 0, 0, 0, 807692307.692308, 0, 0, 0, 807692307.692308,
        0, 228758169.934641, 0, -114379084.967320, 0, 0, 0, 0, 0, 0, 1601307189.54248, 0, 343137254.901961, 0, 0, 0, 0, 0, 0, -1830065359.47712, 0, 457516339.869281, 0, 0, 0, 0, 0,
        0, 0, 228758169.934641, 0, 0, 0, -114379084.967320, 0, 0, 0, 0, 1601307189.54248, 0, 0, 0, 343137254.901961, 0, 0, 0, 0, -1830065359.47712, 0, 0, 0, 457516339.869281, 0, 0,
        0, 114379084.967320, 0, -22292483.6601307, 0, 0, 0, 0, 0, 0, 343137254.901961, 0, 95586601.3071895, 0, 0, 0, 0, 0, 0, -457516339.869281, 0, 41084967.3202614, 0, 0, 0, 0, 0,
        201923076.923077, 0, 0, 0, -94040137.4224904, 0, 0, 0, -40384615.3846154, 605769230.769231, 0, 0, 0, 378257499.581029, 0, 0, 0, 161538461.538462, -807692307.692308, 0, 0, 0, 186936483.995308, 0, 0, 0, 80769230.7692308,
        0, 0, 0, 0, 0, -26732445.1147981, 0, -26923076.9230769, 0, 0, 0, 0, 0, 0, 109026730.350260, 0, 107692307.692308, 0, 0, 0, 0, 0, 0, 52321099.3799229, 0, 53846153.8461539, 0,
        0, 0, 114379084.967320, 0, 0, 0, -22292483.6601307, 0, 0, 0, 0, 343137254.901961, 0, 0, 0, 95586601.3071895, 0, 0, 0, 0, -457516339.869281, 0, 0, 0, 41084967.3202614, 0, 0,
        0, 0, 0, 0, 0, -26923076.9230769, 0, -26732445.1147981, 0, 0, 0, 0, 0, 0, 107692307.692308, 0, 109026730.350260, 0, 0, 0, 0, 0, 0, 53846153.8461539, 0, 52321099.3799229, 0,
        201923076.923077, 0, 0, 0, -40384615.3846154, 0, 0, 0, -94040137.4224904, 605769230.769231, 0, 0, 0, 161538461.538462, 0, 0, 0, 378257499.581029, -807692307.692308, 0, 0, 0, 80769230.7692308, 0, 0, 0, 186936483.995308,
        -7538461538.46154, 0, 0, 0, 807692307.692308, 0, 0, 0, 807692307.692308, -7538461538.46154, 0, 0, 0, -807692307.692308, 0, 0, 0, -807692307.692308, 15076923076.9231, 0, 0, 0, 0, 0, 0, 0, 0,
        0, -1830065359.47712, 0, 457516339.869281, 0, 0, 0, 0, 0, 0, -1830065359.47712, 0, -457516339.869281, 0, 0, 0, 0, 0, 0, 3660130718.95425, 0, 0, 0, 0, 0, 0, 0,
        0, 0, -1830065359.47712, 0, 0, 0, 457516339.869281, 0, 0, 0, 0, -1830065359.47712, 0, 0, 0, -457516339.869281, 0, 0, 0, 0, 3660130718.95425, 0, 0, 0, 0, 0, 0,
        0, -457516339.869281, 0, 41084967.3202614, 0, 0, 0, 0, 0, 0, 457516339.869281, 0, 41084967.3202614, 0, 0, 0, 0, 0, 0, 0, 0, 375346405.228758, 0, 0, 0, 0, 0,
        -807692307.692308, 0, 0, 0, 186936483.995308, 0, 0, 0, 80769230.7692308, 807692307.692308, 0, 0, 0, 186936483.995308, 0, 0, 0, 80769230.7692308, 0, 0, 0, 0, 1510742416.62477, 0, 0, 0, 646153846.153846,
        0, 0, 0, 0, 0, 52321099.3799229, 0, 53846153.8461539, 0, 0, 0, 0, 0, 0, 52321099.3799229, 0, 53846153.8461539, 0, 0, 0, 0, 0, 0, 433819339.701693, 0, 430769230.769231, 0,
        0, 0, -457516339.869281, 0, 0, 0, 41084967.3202614, 0, 0, 0, 0, 457516339.869281, 0, 0, 0, 41084967.3202614, 0, 0, 0, 0, 0, 0, 0, 0, 375346405.228758, 0, 0,
        0, 0, 0, 0, 0, 53846153.8461539, 0, 52321099.3799229, 0, 0, 0, 0, 0, 0, 53846153.8461539, 0, 52321099.3799229, 0, 0, 0, 0, 0, 0, 430769230.769231, 0, 433819339.701693, 0,
        -807692307.692308, 0, 0, 0, 80769230.7692308, 0, 0, 0, 186936483.995308, 807692307.692308, 0, 0, 0, 80769230.7692308, 0, 0, 0, 186936483.995308, 0, 0, 0, 0, 646153846.153846, 0, 0, 0, 1510742416.62477;

    ChMatrixNM<double, 27, 27> Expected_JacobianR_NoDispNoVelWithDamping;
    Expected_JacobianR_NoDispNoVelWithDamping <<
        65961538.4615385, 0, 0, 0, -6057692.30769231, 0, 0, 0, -6057692.30769231, 9423076.92307692, 0, 0, 0, 2019230.76923077, 0, 0, 0, 2019230.76923077, -75384615.3846154, 0, 0, 0, -8076923.07692308, 0, 0, 0, -8076923.07692308,
        0, 16013071.8954248, 0, -3431372.54901961, 0, 0, 0, 0, 0, 0, 2287581.69934641, 0, 1143790.84967320, 0, 0, 0, 0, 0, 0, -18300653.5947712, 0, -4575163.39869281, 0, 0, 0, 0, 0,
        0, 0, 16013071.8954248, 0, 0, 0, -3431372.54901961, 0, 0, 0, 0, 2287581.69934641, 0, 0, 0, 1143790.84967320, 0, 0, 0, 0, -18300653.5947712, 0, 0, 0, -4575163.39869281, 0, 0,
        0, -3431372.54901961, 0, 955866.013071896, 0, 0, 0, 0, 0, 0, -1143790.84967320, 0, -222924.836601307, 0, 0, 0, 0, 0, 0, 4575163.39869281, 0, 410849.673202614, 0, 0, 0, 0, 0,
        -6057692.30769231, 0, 0, 0, 3782574.99581029, 0, 0, 0, 1615384.61538462, -2019230.76923077, 0, 0, 0, -940401.374224904, 0, 0, 0, -403846.153846154, 8076923.07692308, 0, 0, 0, 1869364.83995308, 0, 0, 0, 807692.307692308,
        0, 0, 0, 0, 0, 1090267.30350260, 0, 1076923.07692308, 0, 0, 0, 0, 0, 0, -267324.451147981, 0, -269230.769230769, 0, 0, 0, 0, 0, 0, 523210.993799229, 0, 538461.538461539, 0,
        0, 0, -3431372.54901961, 0, 0, 0, 955866.013071896, 0, 0, 0, 0, -1143790.84967320, 0, 0, 0, -222924.836601307, 0, 0, 0, 0, 4575163.39869281, 0, 0, 0, 410849.673202614, 0, 0,
        0, 0, 0, 0, 0, 1076923.07692308, 0, 1090267.30350260, 0, 0, 0, 0, 0, 0, -269230.769230769, 0, -267324.451147981, 0, 0, 0, 0, 0, 0, 538461.538461539, 0, 523210.993799229, 0,
        -6057692.30769231, 0, 0, 0, 1615384.61538462, 0, 0, 0, 3782574.99581029, -2019230.76923077, 0, 0, 0, -403846.153846154, 0, 0, 0, -940401.374224904, 8076923.07692308, 0, 0, 0, 807692.307692308, 0, 0, 0, 1869364.83995308,
        9423076.92307692, 0, 0, 0, -2019230.76923077, 0, 0, 0, -2019230.76923077, 65961538.4615385, 0, 0, 0, 6057692.30769231, 0, 0, 0, 6057692.30769231, -75384615.3846154, 0, 0, 0, 8076923.07692308, 0, 0, 0, 8076923.07692308,
        0, 2287581.69934641, 0, -1143790.84967320, 0, 0, 0, 0, 0, 0, 16013071.8954248, 0, 3431372.54901961, 0, 0, 0, 0, 0, 0, -18300653.5947712, 0, 4575163.39869281, 0, 0, 0, 0, 0,
        0, 0, 2287581.69934641, 0, 0, 0, -1143790.84967320, 0, 0, 0, 0, 16013071.8954248, 0, 0, 0, 3431372.54901961, 0, 0, 0, 0, -18300653.5947712, 0, 0, 0, 4575163.39869281, 0, 0,
        0, 1143790.84967320, 0, -222924.836601307, 0, 0, 0, 0, 0, 0, 3431372.54901961, 0, 955866.013071896, 0, 0, 0, 0, 0, 0, -4575163.39869281, 0, 410849.673202614, 0, 0, 0, 0, 0,
        2019230.76923077, 0, 0, 0, -940401.374224904, 0, 0, 0, -403846.153846154, 6057692.30769231, 0, 0, 0, 3782574.99581029, 0, 0, 0, 1615384.61538462, -8076923.07692308, 0, 0, 0, 1869364.83995308, 0, 0, 0, 807692.307692308,
        0, 0, 0, 0, 0, -267324.451147981, 0, -269230.769230769, 0, 0, 0, 0, 0, 0, 1090267.30350260, 0, 1076923.07692308, 0, 0, 0, 0, 0, 0, 523210.993799229, 0, 538461.538461539, 0,
        0, 0, 1143790.84967320, 0, 0, 0, -222924.836601307, 0, 0, 0, 0, 3431372.54901961, 0, 0, 0, 955866.013071896, 0, 0, 0, 0, -4575163.39869281, 0, 0, 0, 410849.673202614, 0, 0,
        0, 0, 0, 0, 0, -269230.769230769, 0, -267324.451147981, 0, 0, 0, 0, 0, 0, 1076923.07692308, 0, 1090267.30350260, 0, 0, 0, 0, 0, 0, 538461.538461539, 0, 523210.993799229, 0,
        2019230.76923077, 0, 0, 0, -403846.153846154, 0, 0, 0, -940401.374224904, 6057692.30769231, 0, 0, 0, 1615384.61538462, 0, 0, 0, 3782574.99581029, -8076923.07692308, 0, 0, 0, 807692.307692308, 0, 0, 0, 1869364.83995308,
        -75384615.3846154, 0, 0, 0, 8076923.07692308, 0, 0, 0, 8076923.07692308, -75384615.3846154, 0, 0, 0, -8076923.07692308, 0, 0, 0, -8076923.07692308, 150769230.769231, 0, 0, 0, 0, 0, 0, 0, 0,
        0, -18300653.5947712, 0, 4575163.39869281, 0, 0, 0, 0, 0, 0, -18300653.5947712, 0, -4575163.39869281, 0, 0, 0, 0, 0, 0, 36601307.1895425, 0, 0, 0, 0, 0, 0, 0,
        0, 0, -18300653.5947712, 0, 0, 0, 4575163.39869281, 0, 0, 0, 0, -18300653.5947712, 0, 0, 0, -4575163.39869281, 0, 0, 0, 0, 36601307.1895425, 0, 0, 0, 0, 0, 0,
        0, -4575163.39869281, 0, 410849.673202614, 0, 0, 0, 0, 0, 0, 4575163.39869281, 0, 410849.673202614, 0, 0, 0, 0, 0, 0, 0, 0, 3753464.05228758, 0, 0, 0, 0, 0,
        -8076923.07692308, 0, 0, 0, 1869364.83995308, 0, 0, 0, 807692.307692308, 8076923.07692308, 0, 0, 0, 1869364.83995308, 0, 0, 0, 807692.307692308, 0, 0, 0, 0, 15107424.1662477, 0, 0, 0, 6461538.46153846,
        0, 0, 0, 0, 0, 523210.993799229, 0, 538461.538461539, 0, 0, 0, 0, 0, 0, 523210.993799229, 0, 538461.538461539, 0, 0, 0, 0, 0, 0, 4338193.39701693, 0, 4307692.30769231, 0,
        0, 0, -4575163.39869281, 0, 0, 0, 410849.673202614, 0, 0, 0, 0, 4575163.39869281, 0, 0, 0, 410849.673202614, 0, 0, 0, 0, 0, 0, 0, 0, 3753464.05228758, 0, 0,
        0, 0, 0, 0, 0, 538461.538461539, 0, 523210.993799229, 0, 0, 0, 0, 0, 0, 538461.538461539, 0, 523210.993799229, 0, 0, 0, 0, 0, 0, 4307692.30769231, 0, 4338193.39701693, 0,
        -8076923.07692308, 0, 0, 0, 807692.307692308, 0, 0, 0, 1869364.83995308, 8076923.07692308, 0, 0, 0, 807692.307692308, 0, 0, 0, 1869364.83995308, 0, 0, 0, 0, 6461538.46153846, 0, 0, 0, 15107424.1662477;


    //Setup the test conditions
    m_element->SetAlphaDamp(0.01);

    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation
    ChVectorDynamic<double> InternalForceNoDispNoVelNoGravity;
    InternalForceNoDispNoVelNoGravity.resize(27);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceNoDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_NoDispNoVelWithDamping;
    JacobianK_NoDispNoVelWithDamping.resize(27, 27);
    m_element->ComputeKRMmatricesGlobal(JacobianK_NoDispNoVelWithDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_NoDispNoVelWithDamping;
    JacobianR_NoDispNoVelWithDamping.resize(27, 27);
    m_element->ComputeKRMmatricesGlobal(JacobianR_NoDispNoVelWithDamping, 0, 1, 0);

    //Reset the element conditions back to its original values
    m_element->SetAlphaDamp(0.0);


    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_NoDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(27, 27);
    
    double small_terms_JacR = 1e-4*Expected_JacobianR_NoDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacR = 0;
    double max_percent_error_JacR = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacR;
    percent_error_matrix_JacR.resize(27, 27);


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
bool ANCFBeamTest<ElementVersion, MaterialVersion>::JacobianSmallDispNoVelWithDampingCheck(int msglvl) {
    // =============================================================================
    //  Check the Jacobian at Small Displacement No Velocity - With Damping
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================

    ChMatrixNM<double, 27, 27> Expected_JacobianK_SmallDispNoVelWithDamping;
    Expected_JacobianK_SmallDispNoVelWithDamping <<
        6596179476.92308, 0, 15076923.0769231, 0, -605769230.769231, 0, -1006535.94771242, 0, -605769230.769231, 942318246.153846, 0, 0, 0, 201923076.923077, 0, -91503.2679738562, 0, 201923076.923077, -7538497723.07692, 0, -15076923.0769231, 0, -807692307.692308, 0, -732026.143790850, 0, -807692307.692308,
        0, 1601332820.31171, 0, -343137254.901961, 0, -1006535.94771242, 0, -1006535.94771242, 0, 0, 228768723.780794, 0, 114379084.967320, 0, -91503.2679738562, 0, -91503.2679738562, 0, 0, -1830101544.09251, 0, -457516339.869281, 0, -732026.143790850, 0, -732026.143790850, 0,
        15076923.0769231, 0, 1601384081.85018, 0, -1776923.07692308, 0, -343137254.901961, 0, -3789994.97234791, 0, 0, 228789831.473102, 0, -161538.461538462, 0, 114379084.967320, 0, -344544.997486174, -15076923.0769231, 0, -1830173913.32328, 0, -1292307.69230769, 0, -457516339.869281, 0, -2756359.97988939,
        0, -343137254.901961, 0, 95587447.9430870, 0, 283843.137254902, 0, 0, 0, 0, -114379084.967320, 0, -22292615.5883359, 0, 0, 0, 0, 0, 0, 457516339.869281, 0, 41085221.8433384, 0, 173673.202614379, 0, 0, 0,
        -605769230.769231, 0, -1776923.07692308, 0, 378258346.216926, 0, 0, 0, 161538461.538462, -201923076.923077, 0, -161538.461538462, 0, -94040269.3506955, 0, 0, 0, -40384615.3846154, 807692307.692308, 0, 1938461.53846154, 0, 186936738.518384, 0, 0, 0, 80769230.7692308,
        0, -1006535.94771242, 0, 283843.137254902, 0, 109028549.895961, 0, 107692307.692308, 0, 0, -91503.2679738562, 0, 0, 0, -26732720.8390816, 0, -26923076.9230769, 0, 0, 1098039.21568627, 0, 173673.202614379, 0, 52321622.8284900, 0, 53846153.8461539, 0,
        -1006535.94771242, 0, -343137254.901961, 0, 0, 0, 95587447.9430870, 0, 283843.137254902, -91503.2679738562, 0, -114379084.967320, 0, 0, 0, -22292615.5883359, 0, 0, 1098039.21568627, 0, 457516339.869281, 0, 0, 0, 41085221.8433384, 0, 173673.202614379,
        0, -1006535.94771242, 0, 0, 0, 107692307.692308, 0, 109027576.986157, 0, 0, -91503.2679738562, 0, 0, 0, -26923076.9230769, 0, -26732577.0430032, 0, 0, 1098039.21568627, 0, 0, 0, 53846153.8461539, 0, 52321353.9029998, 0,
        -605769230.769231, 0, -3789994.97234791, 0, 161538461.538462, 0, 283843.137254902, 0, 378259319.126730, -201923076.923077, 0, -344544.997486174, 0, -40384615.3846154, 0, 0, 0, -94040413.1467739, 807692307.692308, 0, 4134539.96983409, 0, 80769230.7692308, 0, 173673.202614379, 0, 186937007.443875,
        942318246.153846, 0, 0, 0, -201923076.923077, 0, -91503.2679738562, 0, -201923076.923077, 6596179476.92308, 0, -15076923.0769231, 0, 605769230.769231, 0, -1006535.94771242, 0, 605769230.769231, -7538497723.07692, 0, 15076923.0769231, 0, 807692307.692308, 0, -732026.143790850, 0, 807692307.692308,
        0, 228768723.780794, 0, -114379084.967320, 0, -91503.2679738562, 0, -91503.2679738562, 0, 0, 1601332820.31171, 0, 343137254.901961, 0, -1006535.94771242, 0, -1006535.94771242, 0, 0, -1830101544.09251, 0, 457516339.869281, 0, -732026.143790850, 0, -732026.143790850, 0,
        0, 0, 228789831.473102, 0, -161538.461538462, 0, -114379084.967320, 0, -344544.997486174, -15076923.0769231, 0, 1601384081.85018, 0, -1776923.07692308, 0, 343137254.901961, 0, -3789994.97234791, 15076923.0769231, 0, -1830173913.32328, 0, -1292307.69230769, 0, 457516339.869281, 0, -2756359.97988939,
        0, 114379084.967320, 0, -22292615.5883359, 0, 0, 0, 0, 0, 0, 343137254.901961, 0, 95587447.9430870, 0, -283843.137254902, 0, 0, 0, 0, -457516339.869281, 0, 41085221.8433384, 0, -173673.202614379, 0, 0, 0,
        201923076.923077, 0, -161538.461538462, 0, -94040269.3506955, 0, 0, 0, -40384615.3846154, 605769230.769231, 0, -1776923.07692308, 0, 378258346.216926, 0, 0, 0, 161538461.538462, -807692307.692308, 0, 1938461.53846154, 0, 186936738.518384, 0, 0, 0, 80769230.7692308,
        0, -91503.2679738562, 0, 0, 0, -26732720.8390816, 0, -26923076.9230769, 0, 0, -1006535.94771242, 0, -283843.137254902, 0, 109028549.895961, 0, 107692307.692308, 0, 0, 1098039.21568627, 0, -173673.202614379, 0, 52321622.8284900, 0, 53846153.8461539, 0,
        -91503.2679738562, 0, 114379084.967320, 0, 0, 0, -22292615.5883359, 0, 0, -1006535.94771242, 0, 343137254.901961, 0, 0, 0, 95587447.9430870, 0, -283843.137254902, 1098039.21568627, 0, -457516339.869281, 0, 0, 0, 41085221.8433384, 0, -173673.202614379,
        0, -91503.2679738562, 0, 0, 0, -26923076.9230769, 0, -26732577.0430032, 0, 0, -1006535.94771242, 0, 0, 0, 107692307.692308, 0, 109027576.986157, 0, 0, 1098039.21568627, 0, 0, 0, 53846153.8461539, 0, 52321353.9029998, 0,
        201923076.923077, 0, -344544.997486174, 0, -40384615.3846154, 0, 0, 0, -94040413.1467739, 605769230.769231, 0, -3789994.97234791, 0, 161538461.538462, 0, -283843.137254902, 0, 378259319.126730, -807692307.692308, 0, 4134539.96983409, 0, 80769230.7692308, 0, -173673.202614379, 0, 186937007.443875,
        -7538497723.07692, 0, -15076923.0769231, 0, 807692307.692308, 0, 1098039.21568627, 0, 807692307.692308, -7538497723.07692, 0, 15076923.0769231, 0, -807692307.692308, 0, 1098039.21568627, 0, -807692307.692308, 15076995446.1538, 0, 0, 0, 0, 0, 1464052.28758170, 0, 0,
        0, -1830101544.09251, 0, 457516339.869281, 0, 1098039.21568627, 0, 1098039.21568627, 0, 0, -1830101544.09251, 0, -457516339.869281, 0, 1098039.21568627, 0, 1098039.21568627, 0, 0, 3660203088.18502, 0, 0, 0, 1464052.28758170, 0, 1464052.28758170, 0,
        -15076923.0769231, 0, -1830173913.32328, 0, 1938461.53846154, 0, 457516339.869281, 0, 4134539.96983409, 15076923.0769231, 0, -1830173913.32328, 0, 1938461.53846154, 0, -457516339.869281, 0, 4134539.96983409, 0, 0, 3660347826.64656, 0, 2584615.38461538, 0, 0, 0, 5512719.95977878,
        0, -457516339.869281, 0, 41085221.8433384, 0, 173673.202614379, 0, 0, 0, 0, 457516339.869281, 0, 41085221.8433384, 0, -173673.202614379, 0, 0, 0, 0, 0, 0, 375347188.490297, 0, 0, 0, 0, 0,
        -807692307.692308, 0, -1292307.69230769, 0, 186936738.518384, 0, 0, 0, 80769230.7692308, 807692307.692308, 0, -1292307.69230769, 0, 186936738.518384, 0, 0, 0, 80769230.7692308, 0, 0, 2584615.38461538, 0, 1510743199.88631, 0, 0, 0, 646153846.153846,
        0, -732026.143790850, 0, 173673.202614379, 0, 52321622.8284900, 0, 53846153.8461539, 0, 0, -732026.143790850, 0, -173673.202614379, 0, 52321622.8284900, 0, 53846153.8461539, 0, 0, 1464052.28758170, 0, 0, 0, 433821049.164538, 0, 430769230.769231, 0,
        -732026.143790850, 0, -457516339.869281, 0, 0, 0, 41085221.8433384, 0, 173673.202614379, -732026.143790850, 0, 457516339.869281, 0, 0, 0, 41085221.8433384, 0, -173673.202614379, 1464052.28758170, 0, 0, 0, 0, 0, 375347188.490297, 0, 0,
        0, -732026.143790850, 0, 0, 0, 53846153.8461539, 0, 52321353.9029998, 0, 0, -732026.143790850, 0, 0, 0, 53846153.8461539, 0, 52321353.9029998, 0, 0, 1464052.28758170, 0, 0, 0, 430769230.769231, 0, 433820122.963231, 0,
        -807692307.692308, 0, -2756359.97988939, 0, 80769230.7692308, 0, 173673.202614379, 0, 186937007.443875, 807692307.692308, 0, -2756359.97988939, 0, 80769230.7692308, 0, -173673.202614379, 0, 186937007.443875, 0, 0, 5512719.95977878, 0, 646153846.153846, 0, 0, 0, 1510744126.08762;

    ChMatrixNM<double, 27, 27> Expected_JacobianR_SmallDispNoVelWithDamping;
    Expected_JacobianR_SmallDispNoVelWithDamping <<
        65961538.4615385, 0, 150769.230769231, 0, -6057692.30769231, 0, 0, 0, -6057692.30769231, 9423076.92307692, 0, 0, 0, 2019230.76923077, 0, 0, 0, 2019230.76923077, -75384615.3846154, 0, -150769.230769231, 0, -8076923.07692308, 0, 0, 0, -8076923.07692308,
        0, 16013071.8954248, 0, -3431372.54901961, 0, -10065.3594771242, 0, 0, 0, 0, 2287581.69934641, 0, 1143790.84967320, 0, -915.032679738562, 0, 0, 0, 0, -18300653.5947712, 0, -4575163.39869281, 0, -7320.26143790850, 0, 0, 0,
        150769.230769231, 0, 16013584.5108095, 0, -17769.2307692308, 0, -3431372.54901961, 0, -27834.5902463550, 0, 0, 2287792.77626948, 0, -1615.38461538462, 0, 1143790.84967320, 0, -2530.41729512318, -150769.230769231, 0, -18301377.2870789, 0, -12923.0769230769, 0, -4575163.39869281, 0, -20243.3383609854,
        0, -3431372.54901961, 0, 955866.013071896, 0, 2838.43137254902, 0, 0, 0, 0, -1143790.84967320, 0, -222924.836601307, 0, 0, 0, 0, 0, 0, 4575163.39869281, 0, 410849.673202614, 0, 1736.73202614379, 0, 0, 0,
        -6057692.30769231, 0, -17769.2307692308, 0, 3782574.99581029, 0, 0, 0, 1615384.61538462, -2019230.76923077, 0, -1615.38461538462, 0, -940401.374224904, 0, 0, 0, -403846.153846154, 8076923.07692308, 0, 19384.6153846154, 0, 1869364.83995308, 0, 0, 0, 807692.307692308,
        0, -10065.3594771242, 0, 2838.43137254902, 0, 1090277.03260064, 0, 1076923.07692308, 0, 0, -915.032679738562, 0, 0, 0, -267325.889108765, 0, -269230.769230769, 0, 0, 10980.3921568627, 0, 1736.73202614379, 0, 523213.683054131, 0, 538461.538461539, 0,
        0, 0, -3431372.54901961, 0, 0, 0, 955866.013071896, 0, 2838.43137254902, 0, 0, -1143790.84967320, 0, 0, 0, -222924.836601307, 0, 0, 0, 0, 4575163.39869281, 0, 0, 0, 410849.673202614, 0, 1736.73202614379,
        0, 0, 0, 0, 0, 1076923.07692308, 0, 1090267.30350260, 0, 0, 0, 0, 0, 0, -269230.769230769, 0, -267324.451147981, 0, 0, 0, 0, 0, 0, 538461.538461539, 0, 523210.993799229, 0,
        -6057692.30769231, 0, -27834.5902463550, 0, 1615384.61538462, 0, 2838.43137254902, 0, 3782584.72490833, -2019230.76923077, 0, -2530.41729512318, 0, -403846.153846154, 0, 0, 0, -940402.812185688, 8076923.07692308, 0, 30365.0075414781, 0, 807692.307692308, 0, 1736.73202614379, 0, 1869367.52920798,
        9423076.92307692, 0, 0, 0, -2019230.76923077, 0, 0, 0, -2019230.76923077, 65961538.4615385, 0, -150769.230769231, 0, 6057692.30769231, 0, 0, 0, 6057692.30769231, -75384615.3846154, 0, 150769.230769231, 0, 8076923.07692308, 0, 0, 0, 8076923.07692308,
        0, 2287581.69934641, 0, -1143790.84967320, 0, -915.032679738562, 0, 0, 0, 0, 16013071.8954248, 0, 3431372.54901961, 0, -10065.3594771242, 0, 0, 0, 0, -18300653.5947712, 0, 4575163.39869281, 0, -7320.26143790850, 0, 0, 0,
        0, 0, 2287792.77626948, 0, -1615.38461538462, 0, -1143790.84967320, 0, -2530.41729512318, -150769.230769231, 0, 16013584.5108095, 0, -17769.2307692308, 0, 3431372.54901961, 0, -27834.5902463550, 150769.230769231, 0, -18301377.2870789, 0, -12923.0769230769, 0, 4575163.39869281, 0, -20243.3383609854,
        0, 1143790.84967320, 0, -222924.836601307, 0, 0, 0, 0, 0, 0, 3431372.54901961, 0, 955866.013071896, 0, -2838.43137254902, 0, 0, 0, 0, -4575163.39869281, 0, 410849.673202614, 0, -1736.73202614379, 0, 0, 0,
        2019230.76923077, 0, -1615.38461538462, 0, -940401.374224904, 0, 0, 0, -403846.153846154, 6057692.30769231, 0, -17769.2307692308, 0, 3782574.99581029, 0, 0, 0, 1615384.61538462, -8076923.07692308, 0, 19384.6153846154, 0, 1869364.83995308, 0, 0, 0, 807692.307692308,
        0, -915.032679738562, 0, 0, 0, -267325.889108765, 0, -269230.769230769, 0, 0, -10065.3594771242, 0, -2838.43137254902, 0, 1090277.03260064, 0, 1076923.07692308, 0, 0, 10980.3921568627, 0, -1736.73202614379, 0, 523213.683054131, 0, 538461.538461539, 0,
        0, 0, 1143790.84967320, 0, 0, 0, -222924.836601307, 0, 0, 0, 0, 3431372.54901961, 0, 0, 0, 955866.013071896, 0, -2838.43137254902, 0, 0, -4575163.39869281, 0, 0, 0, 410849.673202614, 0, -1736.73202614379,
        0, 0, 0, 0, 0, -269230.769230769, 0, -267324.451147981, 0, 0, 0, 0, 0, 0, 1076923.07692308, 0, 1090267.30350260, 0, 0, 0, 0, 0, 0, 538461.538461539, 0, 523210.993799229, 0,
        2019230.76923077, 0, -2530.41729512318, 0, -403846.153846154, 0, 0, 0, -940402.812185688, 6057692.30769231, 0, -27834.5902463550, 0, 1615384.61538462, 0, -2838.43137254902, 0, 3782584.72490833, -8076923.07692308, 0, 30365.0075414781, 0, 807692.307692308, 0, -1736.73202614379, 0, 1869367.52920798,
        -75384615.3846154, 0, -150769.230769231, 0, 8076923.07692308, 0, 0, 0, 8076923.07692308, -75384615.3846154, 0, 150769.230769231, 0, -8076923.07692308, 0, 0, 0, -8076923.07692308, 150769230.769231, 0, 0, 0, 0, 0, 0, 0, 0,
        0, -18300653.5947712, 0, 4575163.39869281, 0, 10980.3921568627, 0, 0, 0, 0, -18300653.5947712, 0, -4575163.39869281, 0, 10980.3921568627, 0, 0, 0, 0, 36601307.1895425, 0, 0, 0, 14640.5228758170, 0, 0, 0,
        -150769.230769231, 0, -18301377.2870789, 0, 19384.6153846154, 0, 4575163.39869281, 0, 30365.0075414781, 150769.230769231, 0, -18301377.2870789, 0, 19384.6153846154, 0, -4575163.39869281, 0, 30365.0075414781, 0, 0, 36602754.5741579, 0, 25846.1538461538, 0, 0, 0, 40486.6767219708,
        0, -4575163.39869281, 0, 410849.673202614, 0, 1736.73202614379, 0, 0, 0, 0, 4575163.39869281, 0, 410849.673202614, 0, -1736.73202614379, 0, 0, 0, 0, 0, 0, 3753464.05228758, 0, 0, 0, 0, 0,
        -8076923.07692308, 0, -12923.0769230769, 0, 1869364.83995308, 0, 0, 0, 807692.307692308, 8076923.07692308, 0, -12923.0769230769, 0, 1869364.83995308, 0, 0, 0, 807692.307692308, 0, 0, 25846.1538461538, 0, 15107424.1662477, 0, 0, 0, 6461538.46153846,
        0, -7320.26143790850, 0, 1736.73202614379, 0, 523213.683054131, 0, 538461.538461539, 0, 0, -7320.26143790850, 0, -1736.73202614379, 0, 523213.683054131, 0, 538461.538461539, 0, 0, 14640.5228758170, 0, 0, 0, 4338202.65903000, 0, 4307692.30769231, 0,
        0, 0, -4575163.39869281, 0, 0, 0, 410849.673202614, 0, 1736.73202614379, 0, 0, 4575163.39869281, 0, 0, 0, 410849.673202614, 0, -1736.73202614379, 0, 0, 0, 0, 0, 0, 3753464.05228758, 0, 0,
        0, 0, 0, 0, 0, 538461.538461539, 0, 523210.993799229, 0, 0, 0, 0, 0, 0, 538461.538461539, 0, 523210.993799229, 0, 0, 0, 0, 0, 0, 4307692.30769231, 0, 4338193.39701693, 0,
        -8076923.07692308, 0, -20243.3383609854, 0, 807692.307692308, 0, 1736.73202614379, 0, 1869367.52920798, 8076923.07692308, 0, -20243.3383609854, 0, 807692.307692308, 0, -1736.73202614379, 0, 1869367.52920798, 0, 0, 40486.6767219708, 0, 6461538.46153846, 0, 0, 0, 15107433.4282608;


    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation

    //Setup the test conditions
    ChVector<double> OriginalPos = m_nodeC->GetPos();
    m_nodeC->SetPos(ChVector<>(m_nodeC->GetPos().x(), m_nodeC->GetPos().y(), 0.001));
    m_element->SetAlphaDamp(0.01);

    ChVectorDynamic<double> InternalForceSmallDispNoVelNoGravity;
    InternalForceSmallDispNoVelNoGravity.resize(27);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceSmallDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_SmallDispNoVelWithDamping;
    JacobianK_SmallDispNoVelWithDamping.resize(27, 27);
    m_element->ComputeKRMmatricesGlobal(JacobianK_SmallDispNoVelWithDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_SmallDispNoVelWithDamping;
    JacobianR_SmallDispNoVelWithDamping.resize(27, 27);
    m_element->ComputeKRMmatricesGlobal(JacobianR_SmallDispNoVelWithDamping, 0, 1, 0);

    //Reset the element conditions back to its original values
    m_nodeC->SetPos(OriginalPos);
    m_element->SetAlphaDamp(0.0);


    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_SmallDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(27, 27);

    double small_terms_JacR = 1e-4*Expected_JacobianR_SmallDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacR = 0;
    double max_percent_error_JacR = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacR;
    percent_error_matrix_JacR.resize(27, 27);


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
bool ANCFBeamTest<ElementVersion, MaterialVersion>::JacobianNoDispSmallVelWithDampingCheck(int msglvl) {
    // =============================================================================
        //  Check the Jacobian at No Displacement Small Velocity - With Damping
        //  (some small error expected depending on the formulation/steps used)
        // =============================================================================

    ChMatrixNM<double, 27, 27> Expected_JacobianK_NoDispSmallVelWithDamping;
    Expected_JacobianK_NoDispSmallVelWithDamping <<
        6596153846.15385, 0, 150769.230769231, 0, -605769230.769231, 0, -10065.3594771242, 0, -605769230.769231, 942307692.307692, 0, 0, 0, 201923076.923077, 0, -915.032679738562, 0, 201923076.923077, -7538461538.46154, 0, -150769.230769231, 0, -807692307.692308, 0, -7320.26143790850, 0, -807692307.692308,
        0, 1601307189.54248, 0, -343137254.901961, 0, -10065.3594771242, 0, -10065.3594771242, 0, 0, 228758169.934641, 0, 114379084.967320, 0, -915.032679738562, 0, -915.032679738562, 0, 0, -1830065359.47712, 0, -457516339.869281, 0, -7320.26143790850, 0, -7320.26143790850, 0,
        0, 0, 1601307189.54248, 0, 0, 0, -343137254.901961, 0, -20130.7189542484, 0, 0, 228758169.934641, 0, 0, 0, 114379084.967320, 0, -1830.06535947712, 0, 0, -1830065359.47712, 0, 0, 0, -457516339.869281, 0, -14640.5228758170,
        0, -343137254.901961, 0, 95586601.3071895, 0, 2838.43137254902, 0, 0, 0, 0, -114379084.967320, 0, -22292483.6601307, 0, 0, 0, 0, 0, 0, 457516339.869281, 0, 41084967.3202614, 0, 1736.73202614379, 0, 0, 0,
        -605769230.769231, 0, -17769.2307692308, 0, 378257499.581029, 0, 0, 0, 161538461.538462, -201923076.923077, 0, -1615.38461538462, 0, -94040137.4224904, 0, 0, 0, -40384615.3846154, 807692307.692308, 0, 19384.6153846154, 0, 186936483.995308, 0, 0, 0, 80769230.7692308,
        0, 0, 0, 0, 0, 109026730.350260, 0, 107692307.692308, 0, 0, 0, 0, 0, 0, -26732445.1147981, 0, -26923076.9230769, 0, 0, 0, 0, 0, 0, 52321099.3799229, 0, 53846153.8461539, 0,
        -10065.3594771242, 0, -343137254.901961, 0, 0, 0, 95586601.3071895, 0, 2838.43137254902, -915.032679738562, 0, -114379084.967320, 0, 0, 0, -22292483.6601307, 0, 0, 10980.3921568627, 0, 457516339.869281, 0, 0, 0, 41084967.3202614, 0, 1736.73202614379,
        0, -10065.3594771242, 0, 0, 0, 107692307.692308, 0, 109026730.350260, 0, 0, -915.032679738562, 0, 0, 0, -26923076.9230769, 0, -26732445.1147981, 0, 0, 10980.3921568627, 0, 0, 0, 53846153.8461539, 0, 52321099.3799229, 0,
        -605769230.769231, 0, -27834.5902463550, 0, 161538461.538462, 0, 0, 0, 378257499.581029, -201923076.923077, 0, -2530.41729512318, 0, -40384615.3846154, 0, 0, 0, -94040137.4224904, 807692307.692308, 0, 30365.0075414781, 0, 80769230.7692308, 0, 0, 0, 186936483.995308,
        942307692.307692, 0, 0, 0, -201923076.923077, 0, -915.032679738562, 0, -201923076.923077, 6596153846.15385, 0, -150769.230769231, 0, 605769230.769231, 0, -10065.3594771242, 0, 605769230.769231, -7538461538.46154, 0, 150769.230769231, 0, 807692307.692308, 0, -7320.26143790850, 0, 807692307.692308,
        0, 228758169.934641, 0, -114379084.967320, 0, -915.032679738562, 0, -915.032679738562, 0, 0, 1601307189.54248, 0, 343137254.901961, 0, -10065.3594771242, 0, -10065.3594771242, 0, 0, -1830065359.47712, 0, 457516339.869281, 0, -7320.26143790850, 0, -7320.26143790850, 0,
        0, 0, 228758169.934641, 0, 0, 0, -114379084.967320, 0, -1830.06535947712, 0, 0, 1601307189.54248, 0, 0, 0, 343137254.901961, 0, -20130.7189542484, 0, 0, -1830065359.47712, 0, 0, 0, 457516339.869281, 0, -14640.5228758170,
        0, 114379084.967320, 0, -22292483.6601307, 0, 0, 0, 0, 0, 0, 343137254.901961, 0, 95586601.3071895, 0, -2838.43137254902, 0, 0, 0, 0, -457516339.869281, 0, 41084967.3202614, 0, -1736.73202614379, 0, 0, 0,
        201923076.923077, 0, -1615.38461538462, 0, -94040137.4224904, 0, 0, 0, -40384615.3846154, 605769230.769231, 0, -17769.2307692308, 0, 378257499.581029, 0, 0, 0, 161538461.538462, -807692307.692308, 0, 19384.6153846154, 0, 186936483.995308, 0, 0, 0, 80769230.7692308,
        0, 0, 0, 0, 0, -26732445.1147981, 0, -26923076.9230769, 0, 0, 0, 0, 0, 0, 109026730.350260, 0, 107692307.692308, 0, 0, 0, 0, 0, 0, 52321099.3799229, 0, 53846153.8461539, 0,
        -915.032679738562, 0, 114379084.967320, 0, 0, 0, -22292483.6601307, 0, 0, -10065.3594771242, 0, 343137254.901961, 0, 0, 0, 95586601.3071895, 0, -2838.43137254902, 10980.3921568627, 0, -457516339.869281, 0, 0, 0, 41084967.3202614, 0, -1736.73202614379,
        0, -915.032679738562, 0, 0, 0, -26923076.9230769, 0, -26732445.1147981, 0, 0, -10065.3594771242, 0, 0, 0, 107692307.692308, 0, 109026730.350260, 0, 0, 10980.3921568627, 0, 0, 0, 53846153.8461539, 0, 52321099.3799229, 0,
        201923076.923077, 0, -2530.41729512318, 0, -40384615.3846154, 0, 0, 0, -94040137.4224904, 605769230.769231, 0, -27834.5902463550, 0, 161538461.538462, 0, 0, 0, 378257499.581029, -807692307.692308, 0, 30365.0075414781, 0, 80769230.7692308, 0, 0, 0, 186936483.995308,
        -7538461538.46154, 0, -150769.230769231, 0, 807692307.692308, 0, 10980.3921568627, 0, 807692307.692308, -7538461538.46154, 0, 150769.230769231, 0, -807692307.692308, 0, 10980.3921568627, 0, -807692307.692308, 15076923076.9231, 0, 0, 0, 0, 0, 14640.5228758170, 0, 0,
        0, -1830065359.47712, 0, 457516339.869281, 0, 10980.3921568627, 0, 10980.3921568627, 0, 0, -1830065359.47712, 0, -457516339.869281, 0, 10980.3921568627, 0, 10980.3921568627, 0, 0, 3660130718.95425, 0, 0, 0, 14640.5228758170, 0, 14640.5228758170, 0,
        0, 0, -1830065359.47712, 0, 0, 0, 457516339.869281, 0, 21960.7843137255, 0, 0, -1830065359.47712, 0, 0, 0, -457516339.869281, 0, 21960.7843137255, 0, 0, 3660130718.95425, 0, 0, 0, 0, 0, 29281.0457516340,
        0, -457516339.869281, 0, 41084967.3202614, 0, 1736.73202614379, 0, 0, 0, 0, 457516339.869281, 0, 41084967.3202614, 0, -1736.73202614379, 0, 0, 0, 0, 0, 0, 375346405.228758, 0, 0, 0, 0, 0,
        -807692307.692308, 0, -12923.0769230769, 0, 186936483.995308, 0, 0, 0, 80769230.7692308, 807692307.692308, 0, -12923.0769230769, 0, 186936483.995308, 0, 0, 0, 80769230.7692308, 0, 0, 25846.1538461538, 0, 1510742416.62477, 0, 0, 0, 646153846.153846,
        0, 0, 0, 0, 0, 52321099.3799229, 0, 53846153.8461539, 0, 0, 0, 0, 0, 0, 52321099.3799229, 0, 53846153.8461539, 0, 0, 0, 0, 0, 0, 433819339.701693, 0, 430769230.769231, 0,
        -7320.26143790850, 0, -457516339.869281, 0, 0, 0, 41084967.3202614, 0, 1736.73202614379, -7320.26143790850, 0, 457516339.869281, 0, 0, 0, 41084967.3202614, 0, -1736.73202614379, 14640.5228758170, 0, 0, 0, 0, 0, 375346405.228758, 0, 0,
        0, -7320.26143790850, 0, 0, 0, 53846153.8461539, 0, 52321099.3799229, 0, 0, -7320.26143790850, 0, 0, 0, 53846153.8461539, 0, 52321099.3799229, 0, 0, 14640.5228758170, 0, 0, 0, 430769230.769231, 0, 433819339.701693, 0,
        -807692307.692308, 0, -20243.3383609854, 0, 80769230.7692308, 0, 0, 0, 186936483.995308, 807692307.692308, 0, -20243.3383609854, 0, 80769230.7692308, 0, 0, 0, 186936483.995308, 0, 0, 40486.6767219708, 0, 646153846.153846, 0, 0, 0, 1510742416.62477;

    ChMatrixNM<double, 27, 27> Expected_JacobianR_NoDispSmallVelWithDamping;
    Expected_JacobianR_NoDispSmallVelWithDamping <<
        65961538.4615385, 0, 0, 0, -6057692.30769231, 0, 0, 0, -6057692.30769231, 9423076.92307692, 0, 0, 0, 2019230.76923077, 0, 0, 0, 2019230.76923077, -75384615.3846154, 0, 0, 0, -8076923.07692308, 0, 0, 0, -8076923.07692308,
        0, 16013071.8954248, 0, -3431372.54901961, 0, 0, 0, 0, 0, 0, 2287581.69934641, 0, 1143790.84967320, 0, 0, 0, 0, 0, 0, -18300653.5947712, 0, -4575163.39869281, 0, 0, 0, 0, 0,
        0, 0, 16013071.8954248, 0, 0, 0, -3431372.54901961, 0, 0, 0, 0, 2287581.69934641, 0, 0, 0, 1143790.84967320, 0, 0, 0, 0, -18300653.5947712, 0, 0, 0, -4575163.39869281, 0, 0,
        0, -3431372.54901961, 0, 955866.013071896, 0, 0, 0, 0, 0, 0, -1143790.84967320, 0, -222924.836601307, 0, 0, 0, 0, 0, 0, 4575163.39869281, 0, 410849.673202614, 0, 0, 0, 0, 0,
        -6057692.30769231, 0, 0, 0, 3782574.99581029, 0, 0, 0, 1615384.61538462, -2019230.76923077, 0, 0, 0, -940401.374224904, 0, 0, 0, -403846.153846154, 8076923.07692308, 0, 0, 0, 1869364.83995308, 0, 0, 0, 807692.307692308,
        0, 0, 0, 0, 0, 1090267.30350260, 0, 1076923.07692308, 0, 0, 0, 0, 0, 0, -267324.451147981, 0, -269230.769230769, 0, 0, 0, 0, 0, 0, 523210.993799229, 0, 538461.538461539, 0,
        0, 0, -3431372.54901961, 0, 0, 0, 955866.013071896, 0, 0, 0, 0, -1143790.84967320, 0, 0, 0, -222924.836601307, 0, 0, 0, 0, 4575163.39869281, 0, 0, 0, 410849.673202614, 0, 0,
        0, 0, 0, 0, 0, 1076923.07692308, 0, 1090267.30350260, 0, 0, 0, 0, 0, 0, -269230.769230769, 0, -267324.451147981, 0, 0, 0, 0, 0, 0, 538461.538461539, 0, 523210.993799229, 0,
        -6057692.30769231, 0, 0, 0, 1615384.61538462, 0, 0, 0, 3782574.99581029, -2019230.76923077, 0, 0, 0, -403846.153846154, 0, 0, 0, -940401.374224904, 8076923.07692308, 0, 0, 0, 807692.307692308, 0, 0, 0, 1869364.83995308,
        9423076.92307692, 0, 0, 0, -2019230.76923077, 0, 0, 0, -2019230.76923077, 65961538.4615385, 0, 0, 0, 6057692.30769231, 0, 0, 0, 6057692.30769231, -75384615.3846154, 0, 0, 0, 8076923.07692308, 0, 0, 0, 8076923.07692308,
        0, 2287581.69934641, 0, -1143790.84967320, 0, 0, 0, 0, 0, 0, 16013071.8954248, 0, 3431372.54901961, 0, 0, 0, 0, 0, 0, -18300653.5947712, 0, 4575163.39869281, 0, 0, 0, 0, 0,
        0, 0, 2287581.69934641, 0, 0, 0, -1143790.84967320, 0, 0, 0, 0, 16013071.8954248, 0, 0, 0, 3431372.54901961, 0, 0, 0, 0, -18300653.5947712, 0, 0, 0, 4575163.39869281, 0, 0,
        0, 1143790.84967320, 0, -222924.836601307, 0, 0, 0, 0, 0, 0, 3431372.54901961, 0, 955866.013071896, 0, 0, 0, 0, 0, 0, -4575163.39869281, 0, 410849.673202614, 0, 0, 0, 0, 0,
        2019230.76923077, 0, 0, 0, -940401.374224904, 0, 0, 0, -403846.153846154, 6057692.30769231, 0, 0, 0, 3782574.99581029, 0, 0, 0, 1615384.61538462, -8076923.07692308, 0, 0, 0, 1869364.83995308, 0, 0, 0, 807692.307692308,
        0, 0, 0, 0, 0, -267324.451147981, 0, -269230.769230769, 0, 0, 0, 0, 0, 0, 1090267.30350260, 0, 1076923.07692308, 0, 0, 0, 0, 0, 0, 523210.993799229, 0, 538461.538461539, 0,
        0, 0, 1143790.84967320, 0, 0, 0, -222924.836601307, 0, 0, 0, 0, 3431372.54901961, 0, 0, 0, 955866.013071896, 0, 0, 0, 0, -4575163.39869281, 0, 0, 0, 410849.673202614, 0, 0,
        0, 0, 0, 0, 0, -269230.769230769, 0, -267324.451147981, 0, 0, 0, 0, 0, 0, 1076923.07692308, 0, 1090267.30350260, 0, 0, 0, 0, 0, 0, 538461.538461539, 0, 523210.993799229, 0,
        2019230.76923077, 0, 0, 0, -403846.153846154, 0, 0, 0, -940401.374224904, 6057692.30769231, 0, 0, 0, 1615384.61538462, 0, 0, 0, 3782574.99581029, -8076923.07692308, 0, 0, 0, 807692.307692308, 0, 0, 0, 1869364.83995308,
        -75384615.3846154, 0, 0, 0, 8076923.07692308, 0, 0, 0, 8076923.07692308, -75384615.3846154, 0, 0, 0, -8076923.07692308, 0, 0, 0, -8076923.07692308, 150769230.769231, 0, 0, 0, 0, 0, 0, 0, 0,
        0, -18300653.5947712, 0, 4575163.39869281, 0, 0, 0, 0, 0, 0, -18300653.5947712, 0, -4575163.39869281, 0, 0, 0, 0, 0, 0, 36601307.1895425, 0, 0, 0, 0, 0, 0, 0,
        0, 0, -18300653.5947712, 0, 0, 0, 4575163.39869281, 0, 0, 0, 0, -18300653.5947712, 0, 0, 0, -4575163.39869281, 0, 0, 0, 0, 36601307.1895425, 0, 0, 0, 0, 0, 0,
        0, -4575163.39869281, 0, 410849.673202614, 0, 0, 0, 0, 0, 0, 4575163.39869281, 0, 410849.673202614, 0, 0, 0, 0, 0, 0, 0, 0, 3753464.05228758, 0, 0, 0, 0, 0,
        -8076923.07692308, 0, 0, 0, 1869364.83995308, 0, 0, 0, 807692.307692308, 8076923.07692308, 0, 0, 0, 1869364.83995308, 0, 0, 0, 807692.307692308, 0, 0, 0, 0, 15107424.1662477, 0, 0, 0, 6461538.46153846,
        0, 0, 0, 0, 0, 523210.993799229, 0, 538461.538461539, 0, 0, 0, 0, 0, 0, 523210.993799229, 0, 538461.538461539, 0, 0, 0, 0, 0, 0, 4338193.39701693, 0, 4307692.30769231, 0,
        0, 0, -4575163.39869281, 0, 0, 0, 410849.673202614, 0, 0, 0, 0, 4575163.39869281, 0, 0, 0, 410849.673202614, 0, 0, 0, 0, 0, 0, 0, 0, 3753464.05228758, 0, 0,
        0, 0, 0, 0, 0, 538461.538461539, 0, 523210.993799229, 0, 0, 0, 0, 0, 0, 538461.538461539, 0, 523210.993799229, 0, 0, 0, 0, 0, 0, 4307692.30769231, 0, 4338193.39701693, 0,
        -8076923.07692308, 0, 0, 0, 807692.307692308, 0, 0, 0, 1869364.83995308, 8076923.07692308, 0, 0, 0, 807692.307692308, 0, 0, 0, 1869364.83995308, 0, 0, 0, 0, 6461538.46153846, 0, 0, 0, 15107424.1662477;


    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation

    //Setup the test conditions
    ChVector<double> OriginalVel = m_nodeC->GetPos_dt();
    m_nodeC->SetPos_dt(ChVector<>(0.0, 0.0, 0.001));
    m_element->SetAlphaDamp(0.01);

    ChVectorDynamic<double> InternalForceNoDispSmallVelNoGravity;
    InternalForceNoDispSmallVelNoGravity.resize(27);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceNoDispSmallVelNoGravity);

    ChMatrixDynamic<double> JacobianK_NoDispSmallVelWithDamping;
    JacobianK_NoDispSmallVelWithDamping.resize(27, 27);
    m_element->ComputeKRMmatricesGlobal(JacobianK_NoDispSmallVelWithDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_NoDispSmallVelWithDamping;
    JacobianR_NoDispSmallVelWithDamping.resize(27, 27);
    m_element->ComputeKRMmatricesGlobal(JacobianR_NoDispSmallVelWithDamping, 0, 1, 0);

    //Reset the element conditions back to its original values
    m_nodeC->SetPos_dt(OriginalVel);
    m_element->SetAlphaDamp(0.0);


    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_NoDispSmallVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(27, 27);

    double small_terms_JacR = 1e-4*Expected_JacobianR_NoDispSmallVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacR = 0;
    double max_percent_error_JacR = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacR;
    percent_error_matrix_JacR.resize(27, 27);


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
bool ANCFBeamTest<ElementVersion, MaterialVersion>::AxialDisplacementCheck(int msglvl) {
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
    int num_elements = 32;
    double length = 1;    // m
    double width = 0.01;  // m (square cross-section)
    // Aluminum 7075-T651
    double rho = 2810;  // kg/m^3
    double E = 71.7e9;  // Pa
    double nu = 0.33;
    double G = E / (2 * (1 + nu));
    double k1 =
        10 * (1 + nu) / (12 + 11 * nu);  // Timoshenko shear correction coefficient for a rectangular cross-section
    double k2 = k1;                      // Timoshenko shear correction coefficient for a rectangular cross-section

    auto material = chrono_types::make_shared<MaterialVersion>(rho, E, nu, k1, k2);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    system->Add(mesh);

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
    std::shared_ptr<ChNodeFEAxyzDD> nodeEndPoint;

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(dx * (2 * i), 0, 0), dir1, dir2);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(dx * (2 * i - 1), 0, 0), dir1, dir2);
        mesh->AddNode(nodeB);
        mesh->AddNode(nodeC);

        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC);
        element->SetDimensions(2 * dx, width, width);
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

    nodeEndPoint = nodeA;

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
    mload->loader.SetApplication(1.0);  // specify application point
    loadcontainer->Add(mload);          // add the load to the load container.

    // Find the nonlinear static solution for the system
    system->DoStaticLinear();  // Note: "system->DoStaticNonlinear(50);" does not give the correct displacement

    // Calculate the axial displacement of the end of the ANCF beam mesh
    ChVector<> point;
    ChQuaternion<> rot;
    elementlast->EvaluateSectionFrame(1.0, point, rot);

    // For Analytical Formula, see a mechanics of materials textbook (delta = PL/AE)
    double Displacement_Theory = (TIP_FORCE * length) / (width*width*E);
    double Displacement_Model = point.x() - length;
    ChVector<> Tip_Angles = rot.Q_to_Euler123();

    double Percent_Error = (Displacement_Model - Displacement_Theory) / Displacement_Theory * 100.0;

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
            print_red(" - Test FAILED\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl;
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
bool ANCFBeamTest<ElementVersion, MaterialVersion>::CantileverCheck(int msglvl) {
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
    int num_elements = 32;
    double length = 1;    // m
    double width = 0.01;  // m (square cross-section)
    // Aluminum 7075-T651
    double rho = 2810;  // kg/m^3
    double E = 71.7e9;  // Pa
    double nu = 0.33;
    double G = E / (2 * (1 + nu));
    double k1 =
        10 * (1 + nu) / (12 + 11 * nu);  // Timoshenko shear correction coefficient for a rectangular cross-section
    double k2 = k1;                      // Timoshenko shear correction coefficient for a rectangular cross-section

    auto material = chrono_types::make_shared<MaterialVersion>(rho, E, nu, k1, k2);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    system->Add(mesh);

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
    std::shared_ptr<ChNodeFEAxyzDD> nodeEndPoint;

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(dx * (2 * i), 0, 0), dir1, dir2);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(dx * (2 * i - 1), 0, 0), dir1, dir2);
        mesh->AddNode(nodeB);
        mesh->AddNode(nodeC);

        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC);
        element->SetDimensions(2 * dx, width, width);
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

    nodeEndPoint = nodeA;

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
    mload->loader.SetApplication(1.0);  // specify application point
    loadcontainer->Add(mload);          // add the load to the load container.

    // Find the nonlinear static solution for the system (final displacement)
    system->DoStaticNonlinear(50);

    // Calculate the displacement of the end of the ANCF beam mesh
    ChVector<> point;
    ChQuaternion<> rot;
    elementlast->EvaluateSectionFrame(1, point, rot);

    // For Analytical Formula, see a mechanics of materials textbook (delta = PL^3/3EI)
    double I = 1.0 / 12.0 * std::pow(width, 4);
    double Displacement_Theory = (TIP_FORCE * std::pow(length,3)) / (3.0*E*I);
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
            print_red(" - Test FAILED\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl;
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
bool ANCFBeamTest<ElementVersion, MaterialVersion>::AxialTwistCheck(int msglvl) {
    // =============================================================================
    //  Check the Axial Twist of a Beam compared to the analytical result
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
    int num_elements = 32;
    double length = 1;    // m
    double width = 0.01;  // m (square cross-section)
    // Aluminum 7075-T651
    double rho = 2810;  // kg/m^3
    double E = 71.7e9;  // Pa
    double nu = 0.33;
    double G = E / (2 * (1 + nu));
    double k1 =
        10 * (1 + nu) / (12 + 11 * nu);  // Timoshenko shear correction coefficient for a rectangular cross-section
    double k2 = k1;                      // Timoshenko shear correction coefficient for a rectangular cross-section

    auto material = chrono_types::make_shared<MaterialVersion>(rho, E, nu, k1, k2);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    system->Add(mesh);

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
    std::shared_ptr<ChNodeFEAxyzDD> nodeEndPoint;

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(dx * (2 * i), 0, 0), dir1, dir2);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(dx * (2 * i - 1), 0, 0), dir1, dir2);
        mesh->AddNode(nodeB);
        mesh->AddNode(nodeC);

        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC);
        element->SetDimensions(2 * dx, width, width);
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

    nodeEndPoint = nodeA;

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
    mload->loader.SetApplication(1.0);  // specify application point
    loadcontainer->Add(mload);          // add the load to the load container.

    // Find the static solution for the system (final twist angle)
    system->DoStaticLinear();

    // Calculate the twist angle of the end of the ANCF beam mesh
    ChVector<> point;
    ChQuaternion<> rot;
    elementlast->EvaluateSectionFrame(1, point, rot);
    ChVector<> Tip_Angles = rot.Q_to_Euler123();

    // For Analytical Formula, see: https://en.wikipedia.org/wiki/Torsion_constant
    double J = 2.25 * std::pow(0.5 * width, 4);
    double T = TIP_MOMENT;
    double Angle_Theory = T * length / (G * J);

    double Percent_Error = (Tip_Angles.x() - Angle_Theory) / Angle_Theory * 100.0;

    bool passed_tests = true;
    if (abs(Percent_Error) > 5.0) {
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
        std::cout << "Axial Twist Angle Check (Percent Error less than 5%)";
        if (abs(Percent_Error) > 5.0) {
            print_red(" - Test FAILED\n");
        }
        else {
            std::cout << " - Test PASSED" << std::endl;
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
    ANCFBeamTest<ChElementBeamANCF, ChMaterialBeamANCF> ChElementBeamANCF_test;
    if (ChElementBeamANCF_test.RunElementChecks(0))
        print_green("ChElementBeamANCF Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_TR01, ChMaterialBeamANCF_TR01> ChElementBeamANCF_TR01_test;
    if (ChElementBeamANCF_TR01_test.RunElementChecks(0))
        print_green("ChElementBeamANCF_TR01 Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_TR01 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_TR02, ChMaterialBeamANCF_TR02> ChElementBeamANCF_TR02_test;
    if (ChElementBeamANCF_TR02_test.RunElementChecks(0))
        print_green("ChElementBeamANCF_TR02 Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_TR02Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_TR03, ChMaterialBeamANCF_TR03> ChElementBeamANCF_TR03_test;
    if (ChElementBeamANCF_TR03_test.RunElementChecks(0))
        print_green("ChElementBeamANCF_TR03 Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_TR03 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_TR04, ChMaterialBeamANCF_TR04> ChElementBeamANCF_TR04_test;
    if (ChElementBeamANCF_TR04_test.RunElementChecks(0))
        print_green("ChElementBeamANCF_TR04 Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_TR04 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_TR05, ChMaterialBeamANCF_TR05> ChElementBeamANCF_TR05_test;
    if (ChElementBeamANCF_TR05_test.RunElementChecks(0))
        print_green("ChElementBeamANCF_TR05 Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_TR05 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_TR06, ChMaterialBeamANCF_TR06> ChElementBeamANCF_TR06_test;
    if (ChElementBeamANCF_TR06_test.RunElementChecks(0))
        print_green("ChElementBeamANCF_TR06 Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_TR06 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_TR07, ChMaterialBeamANCF_TR07> ChElementBeamANCF_TR07_test;
    if (ChElementBeamANCF_TR07_test.RunElementChecks(0))
        print_green("ChElementBeamANCF_TR07 Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_TR07 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_TR08, ChMaterialBeamANCF_TR08> ChElementBeamANCF_TR08_test;
    if (ChElementBeamANCF_TR08_test.RunElementChecks(0))
        print_green("ChElementBeamANCF_TR08 Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_TR08 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_TR08b, ChMaterialBeamANCF_TR08b> ChElementBeamANCF_TR08b_test;
    if (ChElementBeamANCF_TR08b_test.RunElementChecks(0))
        print_green("ChElementBeamANCF_TR08b Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_TR08b Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_TR09, ChMaterialBeamANCF_TR09> ChElementBeamANCF_TR09_test;
    if (ChElementBeamANCF_TR09_test.RunElementChecks(0))
        print_green("ChElementBeamANCF_TR09 Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_TR09 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_TR10, ChMaterialBeamANCF_TR10> ChElementBeamANCF_TR10_test;
    if (ChElementBeamANCF_TR10_test.RunElementChecks(0))
        print_green("ChElementBeamANCF_TR10 Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_TR10 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_TR11, ChMaterialBeamANCF_TR11> ChElementBeamANCF_TR11_test;
    if (ChElementBeamANCF_TR11_test.RunElementChecks(0))
        print_green("ChElementBeamANCF_TR11 Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_TR11 Element Checks = FAILED\n");

    return 0;
}

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

#include "chrono/fea/ChElementBeamANCF_3243_TR01.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR02.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR03.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR04.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR05.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR06.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR07.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR08.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR08b.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR09.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR10.h"

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
    std::shared_ptr<ChNodeFEAxyzDDD> m_nodeB;
};

// =============================================================================

template <typename ElementVersion, typename MaterialVersion>
ANCFBeamTest<ElementVersion, MaterialVersion>::ANCFBeamTest() {
    bool verbose = false;

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

    auto element = chrono_types::make_shared<ElementVersion>();
    element->SetNodes(nodeA, nodeB);
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
    ChMatrixNM<double, 24, 24> Expected_MassMatrix;
    Expected_MassMatrix <<
        29.1571428571429, 0, 0, 4.11190476190476, 0, 0, 0, 0, 0, 0, 0, 0, 10.0928571428571, 0, 0, -2.4297619047619, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 29.1571428571429, 0, 0, 4.11190476190476, 0, 0, 0, 0, 0, 0, 0, 0, 10.0928571428571, 0, 0, -2.4297619047619, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 29.1571428571429, 0, 0, 4.11190476190476, 0, 0, 0, 0, 0, 0, 0, 0, 10.0928571428571, 0, 0, -2.4297619047619, 0, 0, 0, 0, 0, 0,
        4.11190476190476, 0, 0, 0.747619047619048, 0, 0, 0, 0, 0, 0, 0, 0, 2.4297619047619, 0, 0, -0.560714285714286, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 4.11190476190476, 0, 0, 0.747619047619048, 0, 0, 0, 0, 0, 0, 0, 0, 2.4297619047619, 0, 0, -0.560714285714286, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 4.11190476190476, 0, 0, 0.747619047619048, 0, 0, 0, 0, 0, 0, 0, 0, 2.4297619047619, 0, 0, -0.560714285714286, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0.0218055555555556, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0109027777777778, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0.0218055555555556, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0109027777777778, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0.0218055555555556, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0109027777777778, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0218055555555556, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0109027777777778, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0218055555555556, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0109027777777778, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0218055555555556, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0109027777777778,
        10.0928571428571, 0, 0, 2.4297619047619, 0, 0, 0, 0, 0, 0, 0, 0, 29.1571428571429, 0, 0, -4.11190476190476, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 10.0928571428571, 0, 0, 2.4297619047619, 0, 0, 0, 0, 0, 0, 0, 0, 29.1571428571429, 0, 0, -4.11190476190476, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 10.0928571428571, 0, 0, 2.4297619047619, 0, 0, 0, 0, 0, 0, 0, 0, 29.1571428571429, 0, 0, -4.11190476190476, 0, 0, 0, 0, 0, 0,
        -2.4297619047619, 0, 0, -0.560714285714286, 0, 0, 0, 0, 0, 0, 0, 0, -4.11190476190476, 0, 0, 0.747619047619048, 0, 0, 0, 0, 0, 0, 0, 0,
        0, -2.4297619047619, 0, 0, -0.560714285714286, 0, 0, 0, 0, 0, 0, 0, 0, -4.11190476190476, 0, 0, 0.747619047619048, 0, 0, 0, 0, 0, 0, 0,
        0, 0, -2.4297619047619, 0, 0, -0.560714285714286, 0, 0, 0, 0, 0, 0, 0, 0, -4.11190476190476, 0, 0, 0.747619047619048, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0.0109027777777778, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0218055555555556, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0.0109027777777778, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0218055555555556, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0.0109027777777778, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0218055555555556, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0109027777777778, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0218055555555556, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0109027777777778, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0218055555555556, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0109027777777778, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0218055555555556;

    ChMatrixDynamic<double> MassMatrix;
    MassMatrix.resize(24, 24);
    m_element->ComputeKRMmatricesGlobal(MassMatrix, 0, 0, 1);
    ChMatrixNM<double, 8, 8> MassMatrix_compact;
    for (unsigned int i = 0; i < 8; i++) {
        for (unsigned int j = 0; j < 8; j++) {
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
    ChVectorN<double, 24> Expected_InternalForceDueToGravity;
    Expected_InternalForceDueToGravity <<
        0,
        0,
        -384.9110125,
        0,
        0,
        -64.1518354166667,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -384.9110125,
        0,
        0,
        64.1518354166667,
        0,
        0,
        0,
        0,
        0,
        0;

    ChVectorDynamic<double> InternalForceNoDispNoVelWithGravity;
    InternalForceNoDispNoVelWithGravity.resize(24);
    m_element->SetGravityOn(true);
    m_element->ComputeInternalForces(InternalForceNoDispNoVelWithGravity);

    ChVectorDynamic<double> InternalForceNoDispNoVelNoGravity;
    InternalForceNoDispNoVelNoGravity.resize(24);
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
    InternalForceNoDispNoVelNoGravity.resize(24);
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

    ChVectorN<double, 24> Expected_InternalForceSmallDispNoVelNoGravity;
    Expected_InternalForceSmallDispNoVelNoGravity <<
        2180.76923076923,
        0,
        823532.319457014,
        242.307692307692,
        0,
        68627.8144419306,
        0,
        -363.461538461538,
        0,
        -343137.254901961,
        0,
        -775.226244343891,
        -2180.76923076923,
        0,
        -823532.319457014,
        242.307692307692,
        0,
        68627.8144419306,
        0,
        -363.461538461538,
        0,
        -343137.254901961,
        0,
        -775.226244343891;

    //Setup the test conditions
    ChVector<double> OriginalPos = m_nodeB->GetPos();
    m_nodeB->SetPos(ChVector<>(m_nodeB->GetPos().x(), m_nodeB->GetPos().y(), 0.001));

    ChVectorDynamic<double> InternalForceSmallDispNoVelNoGravity;
    InternalForceSmallDispNoVelNoGravity.resize(24);
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
bool ANCFBeamTest<ElementVersion, MaterialVersion>::GeneralizedInternalForceNoDispSmallVelCheck(int msglvl) {
    // =============================================================================
    //  Check the Internal Force at a No Displacement with a Given Nodal Velocity
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================

    ChVectorN<double, 24> Expected_InternalForceNoDispSmallVelNoGravity;
    Expected_InternalForceNoDispSmallVelNoGravity <<
        0,
        0,
        8235.29411764706,
        0,
        0,
        686.274509803922,
        0,
        0,
        0,
        -3431.37254901961,
        0,
        0,
        0,
        0,
        -8235.29411764706,
        0,
        0,
        686.274509803922,
        0,
        0,
        0,
        -3431.37254901961,
        0,
        0;

    //Setup the test conditions
    ChVector<double> OriginalVel = m_nodeB->GetPos_dt();
    m_nodeB->SetPos_dt(ChVector<>(0.0, 0.0, 0.001));
    m_element->SetAlphaDamp(0.01);

    ChVectorDynamic<double> InternalForceNoDispSmallVelNoGravity;
    InternalForceNoDispSmallVelNoGravity.resize(24);
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
bool ANCFBeamTest<ElementVersion, MaterialVersion>::JacobianNoDispNoVelNoDampingCheck(int msglvl) {
    // =============================================================================
    //  Check the Jacobian at No Displacement/Velocity - No Damping
    //  (some small error expected depending on the formulation/steps used)
    //  (The R contribution should be all zeros since Damping is not enabled)
    // =============================================================================
    ChMatrixNM<double, 24, 24> Expected_JacobianK_NoDispNoVelNoDamping;
    Expected_JacobianK_NoDispNoVelNoDamping <<
        3392307692.30769, 0, 0, 282692307.692308, 0, 0, 0, -605769230.769231, 0, 0, 0, -605769230.769231, -3392307692.30769, 0, 0, 282692307.692308, 0, 0, 0, -605769230.769231, 0, 0, 0, -605769230.769231,
        0, 823529411.764706, 0, 0, 68627450.9803922, 0, -343137254.901961, 0, 0, 0, 0, 0, 0, -823529411.764706, 0, 0, 68627450.9803922, 0, -343137254.901961, 0, 0, 0, 0, 0,
        0, 0, 823529411.764706, 0, 0, 68627450.9803922, 0, 0, 0, -343137254.901961, 0, 0, 0, 0, -823529411.764706, 0, 0, 68627450.9803922, 0, 0, 0, -343137254.901961, 0, 0,
        282692307.692308, 0, 0, 376923076.923077, 0, 0, 0, 100961538.461538, 0, 0, 0, 100961538.461538, -282692307.692308, 0, 0, -94230769.2307692, 0, 0, 0, -100961538.461538, 0, 0, 0, -100961538.461538,
        0, 68627450.9803922, 0, 0, 91503267.9738562, 0, 57189542.4836601, 0, 0, 0, 0, 0, 0, -68627450.9803922, 0, 0, -22875816.9934641, 0, -57189542.4836601, 0, 0, 0, 0, 0,
        0, 0, 68627450.9803922, 0, 0, 91503267.9738562, 0, 0, 0, 57189542.4836601, 0, 0, 0, 0, -68627450.9803922, 0, 0, -22875816.9934641, 0, 0, 0, -57189542.4836601, 0, 0,
        0, -343137254.901961, 0, 0, 57189542.4836601, 0, 230508169.934641, 0, 0, 0, 0, 0, 0, 343137254.901961, 0, 0, -57189542.4836601, 0, 112629084.96732, 0, 0, 0, 0, 0,
        -605769230.769231, 0, 0, 100961538.461538, 0, 0, 0, 942879587.732529, 0, 0, 0, 403846153.846154, 605769230.769231, 0, 0, -100961538.461538, 0, 0, 0, 470581950.72901, 0, 0, 0, 201923076.923077,
        0, 0, 0, 0, 0, 0, 0, 0, 269802664.655606, 0, 269230769.230769, 0, 0, 0, 0, 0, 0, 0, 0, 0, 134043489.190548, 0, 134615384.615385, 0,
        0, 0, -343137254.901961, 0, 0, 57189542.4836601, 0, 0, 0, 230508169.934641, 0, 0, 0, 0, 343137254.901961, 0, 0, -57189542.4836601, 0, 0, 0, 112629084.96732, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 269230769.230769, 0, 269802664.655606, 0, 0, 0, 0, 0, 0, 0, 0, 0, 134615384.615385, 0, 134043489.190548, 0,
        -605769230.769231, 0, 0, 100961538.461538, 0, 0, 0, 403846153.846154, 0, 0, 0, 942879587.732529, 605769230.769231, 0, 0, -100961538.461538, 0, 0, 0, 201923076.923077, 0, 0, 0, 470581950.72901,
        -3392307692.30769, 0, 0, -282692307.692308, 0, 0, 0, 605769230.769231, 0, 0, 0, 605769230.769231, 3392307692.30769, 0, 0, -282692307.692308, 0, 0, 0, 605769230.769231, 0, 0, 0, 605769230.769231,
        0, -823529411.764706, 0, 0, -68627450.9803922, 0, 343137254.901961, 0, 0, 0, 0, 0, 0, 823529411.764706, 0, 0, -68627450.9803922, 0, 343137254.901961, 0, 0, 0, 0, 0,
        0, 0, -823529411.764706, 0, 0, -68627450.9803922, 0, 0, 0, 343137254.901961, 0, 0, 0, 0, 823529411.764706, 0, 0, -68627450.9803922, 0, 0, 0, 343137254.901961, 0, 0,
        282692307.692308, 0, 0, -94230769.2307692, 0, 0, 0, -100961538.461538, 0, 0, 0, -100961538.461538, -282692307.692308, 0, 0, 376923076.923077, 0, 0, 0, 100961538.461538, 0, 0, 0, 100961538.461538,
        0, 68627450.9803922, 0, 0, -22875816.9934641, 0, -57189542.4836601, 0, 0, 0, 0, 0, 0, -68627450.9803922, 0, 0, 91503267.9738562, 0, 57189542.4836601, 0, 0, 0, 0, 0,
        0, 0, 68627450.9803922, 0, 0, -22875816.9934641, 0, 0, 0, -57189542.4836601, 0, 0, 0, 0, -68627450.9803922, 0, 0, 91503267.9738562, 0, 0, 0, 57189542.4836601, 0, 0,
        0, -343137254.901961, 0, 0, -57189542.4836601, 0, 112629084.96732, 0, 0, 0, 0, 0, 0, 343137254.901961, 0, 0, 57189542.4836601, 0, 230508169.934641, 0, 0, 0, 0, 0,
        -605769230.769231, 0, 0, -100961538.461538, 0, 0, 0, 470581950.72901, 0, 0, 0, 201923076.923077, 605769230.769231, 0, 0, 100961538.461538, 0, 0, 0, 942879587.732529, 0, 0, 0, 403846153.846154,
        0, 0, 0, 0, 0, 0, 0, 0, 134043489.190548, 0, 134615384.615385, 0, 0, 0, 0, 0, 0, 0, 0, 0, 269802664.655606, 0, 269230769.230769, 0,
        0, 0, -343137254.901961, 0, 0, -57189542.4836601, 0, 0, 0, 112629084.96732, 0, 0, 0, 0, 343137254.901961, 0, 0, 57189542.4836601, 0, 0, 0, 230508169.934641, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 134615384.615385, 0, 134043489.190548, 0, 0, 0, 0, 0, 0, 0, 0, 0, 269230769.230769, 0, 269802664.655606, 0,
        -605769230.769231, 0, 0, -100961538.461538, 0, 0, 0, 201923076.923077, 0, 0, 0, 470581950.72901, 605769230.769231, 0, 0, 100961538.461538, 0, 0, 0, 403846153.846154, 0, 0, 0, 942879587.732529;


    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation
    ChVectorDynamic<double> InternalForceNoDispNoVelNoGravity;
    InternalForceNoDispNoVelNoGravity.resize(24);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceNoDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_NoDispNoVelNoDamping;
    JacobianK_NoDispNoVelNoDamping.resize(24, 24);
    m_element->ComputeKRMmatricesGlobal(JacobianK_NoDispNoVelNoDamping, 1.0, 0.0, 0.0);

    ChMatrixDynamic<double> JacobianR_NoDispNoVelNoDamping;
    JacobianR_NoDispNoVelNoDamping.resize(24, 24);
    m_element->ComputeKRMmatricesGlobal(JacobianR_NoDispNoVelNoDamping, 0.0, 1.0, 0.0);

    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_NoDispNoVelNoDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(24, 24);
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
    ChMatrixNM<double, 24, 24> Expected_JacobianK_SmallDispNoVelNoDamping;
    Expected_JacobianK_SmallDispNoVelNoDamping <<
        3392310600, 0, 4361538.46153846, 282692671.153846, 0, 484615.384615385, 0, -605769230.769231, 0, -411764.705882353, 0, -605769230.769231, -3392310600, 0, -4361538.46153846, 282692671.153846, 0, 484615.384615385, 0, -605769230.769231, 0, -411764.705882353, 0, -605769230.769231,
        0, 823532319.457014, 0, 0, 68627814.4419306, 0, -343137254.901961, 0, -411764.705882353, 0, -411764.705882353, 0, 0, -823532319.457014, 0, 0, 68627814.4419306, 0, -343137254.901961, 0, -411764.705882353, 0, -411764.705882353, 0,
        4361538.46153846, 0, 823538134.841629, 484615.384615385, 0, 68628541.3650076, 0, -726923.076923077, 0, -343137254.901961, 0, -1550452.48868778, -4361538.46153846, 0, -823538134.841629, 484615.384615385, 0, 68628541.3650076, 0, -726923.076923077, 0, -343137254.901961, 0, -1550452.48868778,
        282692671.153846, 0, 484615.384615385, 376923198.076923, 0, 242307.692307692, 0, 100961538.461538, 0, 0, 0, 100961538.461538, -282692671.153846, 0, -484615.384615385, -94230769.2307692, 0, -40384.6153846154, 0, -100961538.461538, 0, -68627.4509803922, 0, -100961538.461538,
        0, 68627814.4419306, 0, 0, 91503389.1277024, 0, 57189542.4836601, 0, 0, 0, 0, 0, 0, -68627814.4419306, 0, 0, -22875816.9934641, 0, -57189542.4836601, 0, -68627.4509803922, 0, -68627.4509803922, 0,
        484615.384615385, 0, 68628541.3650076, 242307.692307692, 0, 91503631.4353947, 0, 0, 0, 57189542.4836601, 0, 0, -484615.384615385, 0, -68628541.3650076, -40384.6153846154, 0, -22875816.9934641, 0, -121153.846153846, 0, -57189542.4836601, 0, -258408.74811463,
        0, -343137254.901961, 0, 0, 57189542.4836601, 0, 230508378.676948, 0, 207632.352941176, 0, 0, 0, 0, 343137254.901961, 0, 0, -57189542.4836601, 0, 112629239.686551, 0, 135504.901960784, 0, 0, 0,
        -605769230.769231, 0, -726923.076923077, 100961538.461538, 0, 0, 0, 942879796.474837, 0, 0, 0, 403846153.846154, 605769230.769231, 0, 726923.076923077, -100961538.461538, 0, -121153.846153846, 0, 470582105.44824, 0, 0, 0, 201923076.923077,
        0, -411764.705882353, 0, 0, 0, 0, 207632.352941176, 0, 269803110.792031, 0, 269230769.230769, 0, 0, 411764.705882353, 0, 0, -68627.4509803922, 0, 135504.901960784, 0, 134043818.280367, 0, 134615384.615385, 0,
        -411764.705882353, 0, -343137254.901961, 0, 0, 57189542.4836601, 0, 0, 0, 230508378.676948, 0, 207632.352941176, 411764.705882353, 0, 343137254.901961, -68627.4509803922, 0, -57189542.4836601, 0, 0, 0, 112629239.686551, 0, 135504.901960784,
        0, -411764.705882353, 0, 0, 0, 0, 0, 0, 269230769.230769, 0, 269802873.397914, 0, 0, 411764.705882353, 0, 0, -68627.4509803922, 0, 0, 0, 134615384.615385, 0, 134043643.909779, 0,
        -605769230.769231, 0, -1550452.48868778, 100961538.461538, 0, 0, 0, 403846153.846154, 0, 207632.352941176, 0, 942880033.868954, 605769230.769231, 0, 1550452.48868778, -100961538.461538, 0, -258408.74811463, 0, 201923076.923077, 0, 135504.901960784, 0, 470582279.818829,
        -3392310600, 0, -4361538.46153846, -282692671.153846, 0, -484615.384615385, 0, 605769230.769231, 0, 411764.705882353, 0, 605769230.769231, 3392310600, 0, 4361538.46153846, -282692671.153846, 0, -484615.384615385, 0, 605769230.769231, 0, 411764.705882353, 0, 605769230.769231,
        0, -823532319.457014, 0, 0, -68627814.4419306, 0, 343137254.901961, 0, 411764.705882353, 0, 411764.705882353, 0, 0, 823532319.457014, 0, 0, -68627814.4419306, 0, 343137254.901961, 0, 411764.705882353, 0, 411764.705882353, 0,
        -4361538.46153846, 0, -823538134.841629, -484615.384615385, 0, -68628541.3650076, 0, 726923.076923077, 0, 343137254.901961, 0, 1550452.48868778, 4361538.46153846, 0, 823538134.841629, -484615.384615385, 0, -68628541.3650076, 0, 726923.076923077, 0, 343137254.901961, 0, 1550452.48868778,
        282692671.153846, 0, 484615.384615385, -94230769.2307692, 0, -40384.6153846154, 0, -100961538.461538, 0, -68627.4509803922, 0, -100961538.461538, -282692671.153846, 0, -484615.384615385, 376923198.076923, 0, 242307.692307692, 0, 100961538.461538, 0, 0, 0, 100961538.461538,
        0, 68627814.4419306, 0, 0, -22875816.9934641, 0, -57189542.4836601, 0, -68627.4509803922, 0, -68627.4509803922, 0, 0, -68627814.4419306, 0, 0, 91503389.1277024, 0, 57189542.4836601, 0, 0, 0, 0, 0,
        484615.384615385, 0, 68628541.3650076, -40384.6153846154, 0, -22875816.9934641, 0, -121153.846153846, 0, -57189542.4836601, 0, -258408.74811463, -484615.384615385, 0, -68628541.3650076, 242307.692307692, 0, 91503631.4353947, 0, 0, 0, 57189542.4836601, 0, 0,
        0, -343137254.901961, 0, 0, -57189542.4836601, 0, 112629239.686551, 0, 135504.901960784, 0, 0, 0, 0, 343137254.901961, 0, 0, 57189542.4836601, 0, 230508378.676948, 0, 207632.352941176, 0, 0, 0,
        -605769230.769231, 0, -726923.076923077, -100961538.461538, 0, -121153.846153846, 0, 470582105.44824, 0, 0, 0, 201923076.923077, 605769230.769231, 0, 726923.076923077, 100961538.461538, 0, 0, 0, 942879796.474837, 0, 0, 0, 403846153.846154,
        0, -411764.705882353, 0, 0, -68627.4509803922, 0, 135504.901960784, 0, 134043818.280367, 0, 134615384.615385, 0, 0, 411764.705882353, 0, 0, 0, 0, 207632.352941176, 0, 269803110.792031, 0, 269230769.230769, 0,
        -411764.705882353, 0, -343137254.901961, -68627.4509803922, 0, -57189542.4836601, 0, 0, 0, 112629239.686551, 0, 135504.901960784, 411764.705882353, 0, 343137254.901961, 0, 0, 57189542.4836601, 0, 0, 0, 230508378.676948, 0, 207632.352941176,
        0, -411764.705882353, 0, 0, -68627.4509803922, 0, 0, 0, 134615384.615385, 0, 134043643.909779, 0, 0, 411764.705882353, 0, 0, 0, 0, 0, 0, 269230769.230769, 0, 269802873.397914, 0,
        -605769230.769231, 0, -1550452.48868778, -100961538.461538, 0, -258408.74811463, 0, 201923076.923077, 0, 135504.901960784, 0, 470582279.818829, 605769230.769231, 0, 1550452.48868778, 100961538.461538, 0, 0, 0, 403846153.846154, 0, 207632.352941176, 0, 942880033.868954;


    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation
    
    //Setup the test conditions
    ChVector<double> OriginalPos = m_nodeB->GetPos();
    m_nodeB->SetPos(ChVector<>(m_nodeB->GetPos().x(), m_nodeB->GetPos().y(), 0.001));

    ChVectorDynamic<double> InternalForceSmallDispNoVelNoGravity;
    InternalForceSmallDispNoVelNoGravity.resize(24);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceSmallDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_SmallDispNoVelNoDamping;
    JacobianK_SmallDispNoVelNoDamping.resize(24, 24);
    m_element->ComputeKRMmatricesGlobal(JacobianK_SmallDispNoVelNoDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_SmallDispNoVelNoDamping;
    JacobianR_SmallDispNoVelNoDamping.resize(24, 24);
    m_element->ComputeKRMmatricesGlobal(JacobianR_SmallDispNoVelNoDamping, 0, 1, 0);

    //Reset the element conditions back to its original values
    m_nodeB->SetPos(OriginalPos);


    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_SmallDispNoVelNoDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(24, 24);
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

    ChMatrixNM<double, 24, 24> Expected_JacobianK_NoDispNoVelWithDamping;
    Expected_JacobianK_NoDispNoVelWithDamping <<
        3392307692.30769, 0, 0, 282692307.692308, 0, 0, 0, -605769230.769231, 0, 0, 0, -605769230.769231, -3392307692.30769, 0, 0, 282692307.692308, 0, 0, 0, -605769230.769231, 0, 0, 0, -605769230.769231,
        0, 823529411.764706, 0, 0, 68627450.9803922, 0, -343137254.901961, 0, 0, 0, 0, 0, 0, -823529411.764706, 0, 0, 68627450.9803922, 0, -343137254.901961, 0, 0, 0, 0, 0,
        0, 0, 823529411.764706, 0, 0, 68627450.9803922, 0, 0, 0, -343137254.901961, 0, 0, 0, 0, -823529411.764706, 0, 0, 68627450.9803922, 0, 0, 0, -343137254.901961, 0, 0,
        282692307.692308, 0, 0, 376923076.923077, 0, 0, 0, 100961538.461538, 0, 0, 0, 100961538.461538, -282692307.692308, 0, 0, -94230769.2307692, 0, 0, 0, -100961538.461538, 0, 0, 0, -100961538.461538,
        0, 68627450.9803922, 0, 0, 91503267.9738562, 0, 57189542.4836601, 0, 0, 0, 0, 0, 0, -68627450.9803922, 0, 0, -22875816.9934641, 0, -57189542.4836601, 0, 0, 0, 0, 0,
        0, 0, 68627450.9803922, 0, 0, 91503267.9738562, 0, 0, 0, 57189542.4836601, 0, 0, 0, 0, -68627450.9803922, 0, 0, -22875816.9934641, 0, 0, 0, -57189542.4836601, 0, 0,
        0, -343137254.901961, 0, 0, 57189542.4836601, 0, 230508169.934641, 0, 0, 0, 0, 0, 0, 343137254.901961, 0, 0, -57189542.4836601, 0, 112629084.96732, 0, 0, 0, 0, 0,
        -605769230.769231, 0, 0, 100961538.461538, 0, 0, 0, 942879587.732529, 0, 0, 0, 403846153.846154, 605769230.769231, 0, 0, -100961538.461538, 0, 0, 0, 470581950.72901, 0, 0, 0, 201923076.923077,
        0, 0, 0, 0, 0, 0, 0, 0, 269802664.655606, 0, 269230769.230769, 0, 0, 0, 0, 0, 0, 0, 0, 0, 134043489.190548, 0, 134615384.615385, 0,
        0, 0, -343137254.901961, 0, 0, 57189542.4836601, 0, 0, 0, 230508169.934641, 0, 0, 0, 0, 343137254.901961, 0, 0, -57189542.4836601, 0, 0, 0, 112629084.96732, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 269230769.230769, 0, 269802664.655606, 0, 0, 0, 0, 0, 0, 0, 0, 0, 134615384.615385, 0, 134043489.190548, 0,
        -605769230.769231, 0, 0, 100961538.461538, 0, 0, 0, 403846153.846154, 0, 0, 0, 942879587.732529, 605769230.769231, 0, 0, -100961538.461538, 0, 0, 0, 201923076.923077, 0, 0, 0, 470581950.72901,
        -3392307692.30769, 0, 0, -282692307.692308, 0, 0, 0, 605769230.769231, 0, 0, 0, 605769230.769231, 3392307692.30769, 0, 0, -282692307.692308, 0, 0, 0, 605769230.769231, 0, 0, 0, 605769230.769231,
        0, -823529411.764706, 0, 0, -68627450.9803922, 0, 343137254.901961, 0, 0, 0, 0, 0, 0, 823529411.764706, 0, 0, -68627450.9803922, 0, 343137254.901961, 0, 0, 0, 0, 0,
        0, 0, -823529411.764706, 0, 0, -68627450.9803922, 0, 0, 0, 343137254.901961, 0, 0, 0, 0, 823529411.764706, 0, 0, -68627450.9803922, 0, 0, 0, 343137254.901961, 0, 0,
        282692307.692308, 0, 0, -94230769.2307692, 0, 0, 0, -100961538.461538, 0, 0, 0, -100961538.461538, -282692307.692308, 0, 0, 376923076.923077, 0, 0, 0, 100961538.461538, 0, 0, 0, 100961538.461538,
        0, 68627450.9803922, 0, 0, -22875816.9934641, 0, -57189542.4836601, 0, 0, 0, 0, 0, 0, -68627450.9803922, 0, 0, 91503267.9738562, 0, 57189542.4836601, 0, 0, 0, 0, 0,
        0, 0, 68627450.9803922, 0, 0, -22875816.9934641, 0, 0, 0, -57189542.4836601, 0, 0, 0, 0, -68627450.9803922, 0, 0, 91503267.9738562, 0, 0, 0, 57189542.4836601, 0, 0,
        0, -343137254.901961, 0, 0, -57189542.4836601, 0, 112629084.96732, 0, 0, 0, 0, 0, 0, 343137254.901961, 0, 0, 57189542.4836601, 0, 230508169.934641, 0, 0, 0, 0, 0,
        -605769230.769231, 0, 0, -100961538.461538, 0, 0, 0, 470581950.72901, 0, 0, 0, 201923076.923077, 605769230.769231, 0, 0, 100961538.461538, 0, 0, 0, 942879587.732529, 0, 0, 0, 403846153.846154,
        0, 0, 0, 0, 0, 0, 0, 0, 134043489.190548, 0, 134615384.615385, 0, 0, 0, 0, 0, 0, 0, 0, 0, 269802664.655606, 0, 269230769.230769, 0,
        0, 0, -343137254.901961, 0, 0, -57189542.4836601, 0, 0, 0, 112629084.96732, 0, 0, 0, 0, 343137254.901961, 0, 0, 57189542.4836601, 0, 0, 0, 230508169.934641, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 134615384.615385, 0, 134043489.190548, 0, 0, 0, 0, 0, 0, 0, 0, 0, 269230769.230769, 0, 269802664.655606, 0,
        -605769230.769231, 0, 0, -100961538.461538, 0, 0, 0, 201923076.923077, 0, 0, 0, 470581950.72901, 605769230.769231, 0, 0, 100961538.461538, 0, 0, 0, 403846153.846154, 0, 0, 0, 942879587.732529;

    ChMatrixNM<double, 24, 24> Expected_JacobianR_NoDispNoVelWithDamping;
    Expected_JacobianR_NoDispNoVelWithDamping <<
        33923076.9230769, 0, 0, 2826923.07692308, 0, 0, 0, -6057692.30769231, 0, 0, 0, -6057692.30769231, -33923076.9230769, 0, 0, 2826923.07692308, 0, 0, 0, -6057692.30769231, 0, 0, 0, -6057692.30769231,
        0, 8235294.11764706, 0, 0, 686274.509803922, 0, -3431372.54901961, 0, 0, 0, 0, 0, 0, -8235294.11764706, 0, 0, 686274.509803922, 0, -3431372.54901961, 0, 0, 0, 0, 0,
        0, 0, 8235294.11764706, 0, 0, 686274.509803922, 0, 0, 0, -3431372.54901961, 0, 0, 0, 0, -8235294.11764706, 0, 0, 686274.509803922, 0, 0, 0, -3431372.54901961, 0, 0,
        2826923.07692308, 0, 0, 3769230.76923077, 0, 0, 0, 1009615.38461538, 0, 0, 0, 1009615.38461538, -2826923.07692308, 0, 0, -942307.692307692, 0, 0, 0, -1009615.38461538, 0, 0, 0, -1009615.38461538,
        0, 686274.509803922, 0, 0, 915032.679738562, 0, 571895.424836601, 0, 0, 0, 0, 0, 0, -686274.509803922, 0, 0, -228758.169934641, 0, -571895.424836601, 0, 0, 0, 0, 0,
        0, 0, 686274.509803922, 0, 0, 915032.679738562, 0, 0, 0, 571895.424836601, 0, 0, 0, 0, -686274.509803922, 0, 0, -228758.169934641, 0, 0, 0, -571895.424836601, 0, 0,
        0, -3431372.54901961, 0, 0, 571895.424836601, 0, 2305081.69934641, 0, 0, 0, 0, 0, 0, 3431372.54901961, 0, 0, -571895.424836601, 0, 1126290.8496732, 0, 0, 0, 0, 0,
        -6057692.30769231, 0, 0, 1009615.38461538, 0, 0, 0, 9428795.87732529, 0, 0, 0, 4038461.53846154, 6057692.30769231, 0, 0, -1009615.38461538, 0, 0, 0, 4705819.5072901, 0, 0, 0, 2019230.76923077,
        0, 0, 0, 0, 0, 0, 0, 0, 2698026.64655606, 0, 2692307.69230769, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1340434.89190548, 0, 1346153.84615385, 0,
        0, 0, -3431372.54901961, 0, 0, 571895.424836601, 0, 0, 0, 2305081.69934641, 0, 0, 0, 0, 3431372.54901961, 0, 0, -571895.424836601, 0, 0, 0, 1126290.8496732, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 2692307.69230769, 0, 2698026.64655606, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1346153.84615385, 0, 1340434.89190548, 0,
        -6057692.30769231, 0, 0, 1009615.38461538, 0, 0, 0, 4038461.53846154, 0, 0, 0, 9428795.87732529, 6057692.30769231, 0, 0, -1009615.38461538, 0, 0, 0, 2019230.76923077, 0, 0, 0, 4705819.5072901,
        -33923076.9230769, 0, 0, -2826923.07692308, 0, 0, 0, 6057692.30769231, 0, 0, 0, 6057692.30769231, 33923076.9230769, 0, 0, -2826923.07692308, 0, 0, 0, 6057692.30769231, 0, 0, 0, 6057692.30769231,
        0, -8235294.11764706, 0, 0, -686274.509803922, 0, 3431372.54901961, 0, 0, 0, 0, 0, 0, 8235294.11764706, 0, 0, -686274.509803922, 0, 3431372.54901961, 0, 0, 0, 0, 0,
        0, 0, -8235294.11764706, 0, 0, -686274.509803922, 0, 0, 0, 3431372.54901961, 0, 0, 0, 0, 8235294.11764706, 0, 0, -686274.509803922, 0, 0, 0, 3431372.54901961, 0, 0,
        2826923.07692308, 0, 0, -942307.692307692, 0, 0, 0, -1009615.38461538, 0, 0, 0, -1009615.38461538, -2826923.07692308, 0, 0, 3769230.76923077, 0, 0, 0, 1009615.38461538, 0, 0, 0, 1009615.38461538,
        0, 686274.509803922, 0, 0, -228758.169934641, 0, -571895.424836601, 0, 0, 0, 0, 0, 0, -686274.509803922, 0, 0, 915032.679738562, 0, 571895.424836601, 0, 0, 0, 0, 0,
        0, 0, 686274.509803922, 0, 0, -228758.169934641, 0, 0, 0, -571895.424836601, 0, 0, 0, 0, -686274.509803922, 0, 0, 915032.679738562, 0, 0, 0, 571895.424836601, 0, 0,
        0, -3431372.54901961, 0, 0, -571895.424836601, 0, 1126290.8496732, 0, 0, 0, 0, 0, 0, 3431372.54901961, 0, 0, 571895.424836601, 0, 2305081.69934641, 0, 0, 0, 0, 0,
        -6057692.30769231, 0, 0, -1009615.38461538, 0, 0, 0, 4705819.5072901, 0, 0, 0, 2019230.76923077, 6057692.30769231, 0, 0, 1009615.38461538, 0, 0, 0, 9428795.87732529, 0, 0, 0, 4038461.53846154,
        0, 0, 0, 0, 0, 0, 0, 0, 1340434.89190548, 0, 1346153.84615385, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2698026.64655606, 0, 2692307.69230769, 0,
        0, 0, -3431372.54901961, 0, 0, -571895.424836601, 0, 0, 0, 1126290.8496732, 0, 0, 0, 0, 3431372.54901961, 0, 0, 571895.424836601, 0, 0, 0, 2305081.69934641, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1346153.84615385, 0, 1340434.89190548, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2692307.69230769, 0, 2698026.64655606, 0,
        -6057692.30769231, 0, 0, -1009615.38461538, 0, 0, 0, 2019230.76923077, 0, 0, 0, 4705819.5072901, 6057692.30769231, 0, 0, 1009615.38461538, 0, 0, 0, 4038461.53846154, 0, 0, 0, 9428795.87732529;


    //Setup the test conditions
    m_element->SetAlphaDamp(0.01);

    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation
    ChVectorDynamic<double> InternalForceNoDispNoVelNoGravity;
    InternalForceNoDispNoVelNoGravity.resize(24);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceNoDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_NoDispNoVelWithDamping;
    JacobianK_NoDispNoVelWithDamping.resize(24, 24);
    m_element->ComputeKRMmatricesGlobal(JacobianK_NoDispNoVelWithDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_NoDispNoVelWithDamping;
    JacobianR_NoDispNoVelWithDamping.resize(24, 24);
    m_element->ComputeKRMmatricesGlobal(JacobianR_NoDispNoVelWithDamping, 0, 1, 0);

    //Reset the element conditions back to its original values
    m_element->SetAlphaDamp(0.0);


    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_NoDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(24, 24);
    
    double small_terms_JacR = 1e-4*Expected_JacobianR_NoDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacR = 0;
    double max_percent_error_JacR = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacR;
    percent_error_matrix_JacR.resize(24, 24);


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

    ChMatrixNM<double, 24, 24> Expected_JacobianK_SmallDispNoVelWithDamping;
    Expected_JacobianK_SmallDispNoVelWithDamping <<
        3392310600, 0, 4361538.46153846, 282692671.153846, 0, 484615.384615385, 0, -605769230.769231, 0, -411764.705882353, 0, -605769230.769231, -3392310600, 0, -4361538.46153846, 282692671.153846, 0, 484615.384615385, 0, -605769230.769231, 0, -411764.705882353, 0, -605769230.769231,
        0, 823532319.457014, 0, 0, 68627814.4419306, 0, -343137254.901961, 0, -411764.705882353, 0, -411764.705882353, 0, 0, -823532319.457014, 0, 0, 68627814.4419306, 0, -343137254.901961, 0, -411764.705882353, 0, -411764.705882353, 0,
        4361538.46153846, 0, 823538134.841629, 484615.384615385, 0, 68628541.3650076, 0, -726923.076923077, 0, -343137254.901961, 0, -1550452.48868778, -4361538.46153846, 0, -823538134.841629, 484615.384615385, 0, 68628541.3650076, 0, -726923.076923077, 0, -343137254.901961, 0, -1550452.48868778,
        282692671.153846, 0, 484615.384615385, 376923198.076923, 0, 242307.692307692, 0, 100961538.461538, 0, 0, 0, 100961538.461538, -282692671.153846, 0, -484615.384615385, -94230769.2307692, 0, -40384.6153846154, 0, -100961538.461538, 0, -68627.4509803922, 0, -100961538.461538,
        0, 68627814.4419306, 0, 0, 91503389.1277024, 0, 57189542.4836601, 0, 0, 0, 0, 0, 0, -68627814.4419306, 0, 0, -22875816.9934641, 0, -57189542.4836601, 0, -68627.4509803922, 0, -68627.4509803922, 0,
        484615.384615385, 0, 68628541.3650076, 242307.692307692, 0, 91503631.4353947, 0, 0, 0, 57189542.4836601, 0, 0, -484615.384615385, 0, -68628541.3650076, -40384.6153846154, 0, -22875816.9934641, 0, -121153.846153846, 0, -57189542.4836601, 0, -258408.74811463,
        0, -343137254.901961, 0, 0, 57189542.4836601, 0, 230508378.676948, 0, 207632.352941176, 0, 0, 0, 0, 343137254.901961, 0, 0, -57189542.4836601, 0, 112629239.686551, 0, 135504.901960784, 0, 0, 0,
        -605769230.769231, 0, -726923.076923077, 100961538.461538, 0, 0, 0, 942879796.474837, 0, 0, 0, 403846153.846154, 605769230.769231, 0, 726923.076923077, -100961538.461538, 0, -121153.846153846, 0, 470582105.44824, 0, 0, 0, 201923076.923077,
        0, -411764.705882353, 0, 0, 0, 0, 207632.352941176, 0, 269803110.792031, 0, 269230769.230769, 0, 0, 411764.705882353, 0, 0, -68627.4509803922, 0, 135504.901960784, 0, 134043818.280367, 0, 134615384.615385, 0,
        -411764.705882353, 0, -343137254.901961, 0, 0, 57189542.4836601, 0, 0, 0, 230508378.676948, 0, 207632.352941176, 411764.705882353, 0, 343137254.901961, -68627.4509803922, 0, -57189542.4836601, 0, 0, 0, 112629239.686551, 0, 135504.901960784,
        0, -411764.705882353, 0, 0, 0, 0, 0, 0, 269230769.230769, 0, 269802873.397914, 0, 0, 411764.705882353, 0, 0, -68627.4509803922, 0, 0, 0, 134615384.615385, 0, 134043643.909779, 0,
        -605769230.769231, 0, -1550452.48868778, 100961538.461538, 0, 0, 0, 403846153.846154, 0, 207632.352941176, 0, 942880033.868954, 605769230.769231, 0, 1550452.48868778, -100961538.461538, 0, -258408.74811463, 0, 201923076.923077, 0, 135504.901960784, 0, 470582279.818829,
        -3392310600, 0, -4361538.46153846, -282692671.153846, 0, -484615.384615385, 0, 605769230.769231, 0, 411764.705882353, 0, 605769230.769231, 3392310600, 0, 4361538.46153846, -282692671.153846, 0, -484615.384615385, 0, 605769230.769231, 0, 411764.705882353, 0, 605769230.769231,
        0, -823532319.457014, 0, 0, -68627814.4419306, 0, 343137254.901961, 0, 411764.705882353, 0, 411764.705882353, 0, 0, 823532319.457014, 0, 0, -68627814.4419306, 0, 343137254.901961, 0, 411764.705882353, 0, 411764.705882353, 0,
        -4361538.46153846, 0, -823538134.841629, -484615.384615385, 0, -68628541.3650076, 0, 726923.076923077, 0, 343137254.901961, 0, 1550452.48868778, 4361538.46153846, 0, 823538134.841629, -484615.384615385, 0, -68628541.3650076, 0, 726923.076923077, 0, 343137254.901961, 0, 1550452.48868778,
        282692671.153846, 0, 484615.384615385, -94230769.2307692, 0, -40384.6153846154, 0, -100961538.461538, 0, -68627.4509803922, 0, -100961538.461538, -282692671.153846, 0, -484615.384615385, 376923198.076923, 0, 242307.692307692, 0, 100961538.461538, 0, 0, 0, 100961538.461538,
        0, 68627814.4419306, 0, 0, -22875816.9934641, 0, -57189542.4836601, 0, -68627.4509803922, 0, -68627.4509803922, 0, 0, -68627814.4419306, 0, 0, 91503389.1277024, 0, 57189542.4836601, 0, 0, 0, 0, 0,
        484615.384615385, 0, 68628541.3650076, -40384.6153846154, 0, -22875816.9934641, 0, -121153.846153846, 0, -57189542.4836601, 0, -258408.74811463, -484615.384615385, 0, -68628541.3650076, 242307.692307692, 0, 91503631.4353947, 0, 0, 0, 57189542.4836601, 0, 0,
        0, -343137254.901961, 0, 0, -57189542.4836601, 0, 112629239.686551, 0, 135504.901960784, 0, 0, 0, 0, 343137254.901961, 0, 0, 57189542.4836601, 0, 230508378.676948, 0, 207632.352941176, 0, 0, 0,
        -605769230.769231, 0, -726923.076923077, -100961538.461538, 0, -121153.846153846, 0, 470582105.44824, 0, 0, 0, 201923076.923077, 605769230.769231, 0, 726923.076923077, 100961538.461538, 0, 0, 0, 942879796.474837, 0, 0, 0, 403846153.846154,
        0, -411764.705882353, 0, 0, -68627.4509803922, 0, 135504.901960784, 0, 134043818.280367, 0, 134615384.615385, 0, 0, 411764.705882353, 0, 0, 0, 0, 207632.352941176, 0, 269803110.792031, 0, 269230769.230769, 0,
        -411764.705882353, 0, -343137254.901961, -68627.4509803922, 0, -57189542.4836601, 0, 0, 0, 112629239.686551, 0, 135504.901960784, 411764.705882353, 0, 343137254.901961, 0, 0, 57189542.4836601, 0, 0, 0, 230508378.676948, 0, 207632.352941176,
        0, -411764.705882353, 0, 0, -68627.4509803922, 0, 0, 0, 134615384.615385, 0, 134043643.909779, 0, 0, 411764.705882353, 0, 0, 0, 0, 0, 0, 269230769.230769, 0, 269802873.397914, 0,
        -605769230.769231, 0, -1550452.48868778, -100961538.461538, 0, -258408.74811463, 0, 201923076.923077, 0, 135504.901960784, 0, 470582279.818829, 605769230.769231, 0, 1550452.48868778, 100961538.461538, 0, 0, 0, 403846153.846154, 0, 207632.352941176, 0, 942880033.868954;

    ChMatrixNM<double, 24, 24> Expected_JacobianR_SmallDispNoVelWithDamping;
    Expected_JacobianR_SmallDispNoVelWithDamping <<
        33923076.9230769, 0, 43615.3846153846, 2826923.07692308, 0, 4846.15384615385, 0, -6057692.30769231, 0, 0, 0, -6057692.30769231, -33923076.9230769, 0, -43615.3846153846, 2826923.07692308, 0, 4846.15384615385, 0, -6057692.30769231, 0, 0, 0, -6057692.30769231,
        0, 8235294.11764706, 0, 0, 686274.509803922, 0, -3431372.54901961, 0, -4117.64705882353, 0, 0, 0, 0, -8235294.11764706, 0, 0, 686274.509803922, 0, -3431372.54901961, 0, -4117.64705882353, 0, 0, 0,
        43615.3846153846, 0, 8235352.27149321, 4846.15384615385, 0, 686281.779034691, 0, -7269.23076923077, 0, -3431372.54901961, 0, -11386.8778280543, -43615.3846153846, 0, -8235352.27149321, 4846.15384615385, 0, 686281.779034691, 0, -7269.23076923077, 0, -3431372.54901961, 0, -11386.8778280543,
        2826923.07692308, 0, 4846.15384615385, 3769230.76923077, 0, 2423.07692307692, 0, 1009615.38461538, 0, 0, 0, 1009615.38461538, -2826923.07692308, 0, -4846.15384615385, -942307.692307692, 0, -403.846153846154, 0, -1009615.38461538, 0, 0, 0, -1009615.38461538,
        0, 686274.509803922, 0, 0, 915032.679738562, 0, 571895.424836601, 0, 0, 0, 0, 0, 0, -686274.509803922, 0, 0, -228758.169934641, 0, -571895.424836601, 0, -686.274509803922, 0, 0, 0,
        4846.15384615385, 0, 686281.779034691, 2423.07692307692, 0, 915035.102815485, 0, 0, 0, 571895.424836601, 0, 0, -4846.15384615385, 0, -686281.779034691, -403.846153846154, 0, -228758.169934641, 0, -1211.53846153846, 0, -571895.424836601, 0, -1897.81297134238,
        0, -3431372.54901961, 0, 0, 571895.424836601, 0, 2305081.69934641, 0, 2076.32352941177, 0, 0, 0, 0, 3431372.54901961, 0, 0, -571895.424836601, 0, 1126290.8496732, 0, 1355.04901960784, 0, 0, 0,
        -6057692.30769231, 0, -7269.23076923077, 1009615.38461538, 0, 0, 0, 9428795.87732529, 0, 0, 0, 4038461.53846154, 6057692.30769231, 0, 7269.23076923077, -1009615.38461538, 0, -1211.53846153846, 0, 4705819.5072901, 0, 0, 0, 2019230.76923077,
        0, -4117.64705882353, 0, 0, 0, 0, 2076.32352941177, 0, 2698029.02049724, 0, 2692307.69230769, 0, 0, 4117.64705882353, 0, 0, -686.274509803922, 0, 1355.04901960784, 0, 1340436.63561136, 0, 1346153.84615385, 0,
        0, 0, -3431372.54901961, 0, 0, 571895.424836601, 0, 0, 0, 2305081.69934641, 0, 2076.32352941177, 0, 0, 3431372.54901961, 0, 0, -571895.424836601, 0, 0, 0, 1126290.8496732, 0, 1355.04901960784,
        0, 0, 0, 0, 0, 0, 0, 0, 2692307.69230769, 0, 2698026.64655606, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1346153.84615385, 0, 1340434.89190548, 0,
        -6057692.30769231, 0, -11386.8778280543, 1009615.38461538, 0, 0, 0, 4038461.53846154, 0, 2076.32352941177, 0, 9428798.25126647, 6057692.30769231, 0, 11386.8778280543, -1009615.38461538, 0, -1897.81297134238, 0, 2019230.76923077, 0, 1355.04901960784, 0, 4705821.25099598,
        -33923076.9230769, 0, -43615.3846153846, -2826923.07692308, 0, -4846.15384615385, 0, 6057692.30769231, 0, 0, 0, 6057692.30769231, 33923076.9230769, 0, 43615.3846153846, -2826923.07692308, 0, -4846.15384615385, 0, 6057692.30769231, 0, 0, 0, 6057692.30769231,
        0, -8235294.11764706, 0, 0, -686274.509803922, 0, 3431372.54901961, 0, 4117.64705882353, 0, 0, 0, 0, 8235294.11764706, 0, 0, -686274.509803922, 0, 3431372.54901961, 0, 4117.64705882353, 0, 0, 0,
        -43615.3846153846, 0, -8235352.27149321, -4846.15384615385, 0, -686281.779034691, 0, 7269.23076923077, 0, 3431372.54901961, 0, 11386.8778280543, 43615.3846153846, 0, 8235352.27149321, -4846.15384615385, 0, -686281.779034691, 0, 7269.23076923077, 0, 3431372.54901961, 0, 11386.8778280543,
        2826923.07692308, 0, 4846.15384615385, -942307.692307692, 0, -403.846153846154, 0, -1009615.38461538, 0, 0, 0, -1009615.38461538, -2826923.07692308, 0, -4846.15384615385, 3769230.76923077, 0, 2423.07692307692, 0, 1009615.38461538, 0, 0, 0, 1009615.38461538,
        0, 686274.509803922, 0, 0, -228758.169934641, 0, -571895.424836601, 0, -686.274509803922, 0, 0, 0, 0, -686274.509803922, 0, 0, 915032.679738562, 0, 571895.424836601, 0, 0, 0, 0, 0,
        4846.15384615385, 0, 686281.779034691, -403.846153846154, 0, -228758.169934641, 0, -1211.53846153846, 0, -571895.424836601, 0, -1897.81297134238, -4846.15384615385, 0, -686281.779034691, 2423.07692307692, 0, 915035.102815485, 0, 0, 0, 571895.424836601, 0, 0,
        0, -3431372.54901961, 0, 0, -571895.424836601, 0, 1126290.8496732, 0, 1355.04901960784, 0, 0, 0, 0, 3431372.54901961, 0, 0, 571895.424836601, 0, 2305081.69934641, 0, 2076.32352941177, 0, 0, 0,
        -6057692.30769231, 0, -7269.23076923077, -1009615.38461538, 0, -1211.53846153846, 0, 4705819.5072901, 0, 0, 0, 2019230.76923077, 6057692.30769231, 0, 7269.23076923077, 1009615.38461538, 0, 0, 0, 9428795.87732529, 0, 0, 0, 4038461.53846154,
        0, -4117.64705882353, 0, 0, -686.274509803922, 0, 1355.04901960784, 0, 1340436.63561136, 0, 1346153.84615385, 0, 0, 4117.64705882353, 0, 0, 0, 0, 2076.32352941177, 0, 2698029.02049724, 0, 2692307.69230769, 0,
        0, 0, -3431372.54901961, 0, 0, -571895.424836601, 0, 0, 0, 1126290.8496732, 0, 1355.04901960784, 0, 0, 3431372.54901961, 0, 0, 571895.424836601, 0, 0, 0, 2305081.69934641, 0, 2076.32352941177,
        0, 0, 0, 0, 0, 0, 0, 0, 1346153.84615385, 0, 1340434.89190548, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2692307.69230769, 0, 2698026.64655606, 0,
        -6057692.30769231, 0, -11386.8778280543, -1009615.38461538, 0, -1897.81297134238, 0, 2019230.76923077, 0, 1355.04901960784, 0, 4705821.25099598, 6057692.30769231, 0, 11386.8778280543, 1009615.38461538, 0, 0, 0, 4038461.53846154, 0, 2076.32352941177, 0, 9428798.25126647;


    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation

    //Setup the test conditions
    ChVector<double> OriginalPos = m_nodeB->GetPos();
    m_nodeB->SetPos(ChVector<>(m_nodeB->GetPos().x(), m_nodeB->GetPos().y(), 0.001));
    m_element->SetAlphaDamp(0.01);

    ChVectorDynamic<double> InternalForceSmallDispNoVelNoGravity;
    InternalForceSmallDispNoVelNoGravity.resize(24);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceSmallDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_SmallDispNoVelWithDamping;
    JacobianK_SmallDispNoVelWithDamping.resize(24, 24);
    m_element->ComputeKRMmatricesGlobal(JacobianK_SmallDispNoVelWithDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_SmallDispNoVelWithDamping;
    JacobianR_SmallDispNoVelWithDamping.resize(24, 24);
    m_element->ComputeKRMmatricesGlobal(JacobianR_SmallDispNoVelWithDamping, 0, 1, 0);

    //Reset the element conditions back to its original values
    m_nodeB->SetPos(OriginalPos);
    m_element->SetAlphaDamp(0.0);


    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_SmallDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(24, 24);

    double small_terms_JacR = 1e-4*Expected_JacobianR_SmallDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacR = 0;
    double max_percent_error_JacR = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacR;
    percent_error_matrix_JacR.resize(24, 24);


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

    ChMatrixNM<double, 24, 24> Expected_JacobianK_NoDispSmallVelWithDamping;
    Expected_JacobianK_NoDispSmallVelWithDamping <<
        3392307692.30769, 0, 43615.3846153846, 282692307.692308, 0, 4846.15384615385, 0, -605769230.769231, 0, -4117.64705882353, 0, -605769230.769231, -3392307692.30769, 0, -43615.3846153846, 282692307.692308, 0, 4846.15384615385, 0, -605769230.769231, 0, -4117.64705882353, 0, -605769230.769231,
        0, 823529411.764706, 0, 0, 68627450.9803922, 0, -343137254.901961, 0, -4117.64705882353, 0, -4117.64705882353, 0, 0, -823529411.764706, 0, 0, 68627450.9803922, 0, -343137254.901961, 0, -4117.64705882353, 0, -4117.64705882353, 0,
        0, 0, 823529411.764706, 0, 0, 68627450.9803922, 0, 0, 0, -343137254.901961, 0, -8235.29411764706, 0, 0, -823529411.764706, 0, 0, 68627450.9803922, 0, 0, 0, -343137254.901961, 0, -8235.29411764706,
        282692307.692308, 0, 4846.15384615385, 376923076.923077, 0, 2423.07692307692, 0, 100961538.461538, 0, 0, 0, 100961538.461538, -282692307.692308, 0, -4846.15384615385, -94230769.2307692, 0, -403.846153846154, 0, -100961538.461538, 0, -686.274509803922, 0, -100961538.461538,
        0, 68627450.9803922, 0, 0, 91503267.9738562, 0, 57189542.4836601, 0, 0, 0, 0, 0, 0, -68627450.9803922, 0, 0, -22875816.9934641, 0, -57189542.4836601, 0, -686.274509803922, 0, -686.274509803922, 0,
        0, 0, 68627450.9803922, 0, 0, 91503267.9738562, 0, 0, 0, 57189542.4836601, 0, 0, 0, 0, -68627450.9803922, 0, 0, -22875816.9934641, 0, 0, 0, -57189542.4836601, 0, -1372.54901960784,
        0, -343137254.901961, 0, 0, 57189542.4836601, 0, 230508169.934641, 0, 2076.32352941177, 0, 0, 0, 0, 343137254.901961, 0, 0, -57189542.4836601, 0, 112629084.96732, 0, 1355.04901960784, 0, 0, 0,
        -605769230.769231, 0, -7269.23076923077, 100961538.461538, 0, 0, 0, 942879587.732529, 0, 0, 0, 403846153.846154, 605769230.769231, 0, 7269.23076923077, -100961538.461538, 0, -1211.53846153846, 0, 470581950.72901, 0, 0, 0, 201923076.923077,
        0, 0, 0, 0, 0, 0, 0, 0, 269802664.655606, 0, 269230769.230769, 0, 0, 0, 0, 0, 0, 0, 0, 0, 134043489.190548, 0, 134615384.615385, 0,
        -4117.64705882353, 0, -343137254.901961, 0, 0, 57189542.4836601, 0, 0, 0, 230508169.934641, 0, 2076.32352941177, 4117.64705882353, 0, 343137254.901961, -686.274509803922, 0, -57189542.4836601, 0, 0, 0, 112629084.96732, 0, 1355.04901960784,
        0, -4117.64705882353, 0, 0, 0, 0, 0, 0, 269230769.230769, 0, 269802664.655606, 0, 0, 4117.64705882353, 0, 0, -686.274509803922, 0, 0, 0, 134615384.615385, 0, 134043489.190548, 0,
        -605769230.769231, 0, -11386.8778280543, 100961538.461538, 0, 0, 0, 403846153.846154, 0, 0, 0, 942879587.732529, 605769230.769231, 0, 11386.8778280543, -100961538.461538, 0, -1897.81297134238, 0, 201923076.923077, 0, 0, 0, 470581950.72901,
        -3392307692.30769, 0, -43615.3846153846, -282692307.692308, 0, -4846.15384615385, 0, 605769230.769231, 0, 4117.64705882353, 0, 605769230.769231, 3392307692.30769, 0, 43615.3846153846, -282692307.692308, 0, -4846.15384615385, 0, 605769230.769231, 0, 4117.64705882353, 0, 605769230.769231,
        0, -823529411.764706, 0, 0, -68627450.9803922, 0, 343137254.901961, 0, 4117.64705882353, 0, 4117.64705882353, 0, 0, 823529411.764706, 0, 0, -68627450.9803922, 0, 343137254.901961, 0, 4117.64705882353, 0, 4117.64705882353, 0,
        0, 0, -823529411.764706, 0, 0, -68627450.9803922, 0, 0, 0, 343137254.901961, 0, 8235.29411764706, 0, 0, 823529411.764706, 0, 0, -68627450.9803922, 0, 0, 0, 343137254.901961, 0, 8235.29411764706,
        282692307.692308, 0, 4846.15384615385, -94230769.2307692, 0, -403.846153846154, 0, -100961538.461538, 0, -686.274509803922, 0, -100961538.461538, -282692307.692308, 0, -4846.15384615385, 376923076.923077, 0, 2423.07692307692, 0, 100961538.461538, 0, 0, 0, 100961538.461538,
        0, 68627450.9803922, 0, 0, -22875816.9934641, 0, -57189542.4836601, 0, -686.274509803922, 0, -686.274509803922, 0, 0, -68627450.9803922, 0, 0, 91503267.9738562, 0, 57189542.4836601, 0, 0, 0, 0, 0,
        0, 0, 68627450.9803922, 0, 0, -22875816.9934641, 0, 0, 0, -57189542.4836601, 0, -1372.54901960784, 0, 0, -68627450.9803922, 0, 0, 91503267.9738562, 0, 0, 0, 57189542.4836601, 0, 0,
        0, -343137254.901961, 0, 0, -57189542.4836601, 0, 112629084.96732, 0, 1355.04901960784, 0, 0, 0, 0, 343137254.901961, 0, 0, 57189542.4836601, 0, 230508169.934641, 0, 2076.32352941177, 0, 0, 0,
        -605769230.769231, 0, -7269.23076923077, -100961538.461538, 0, -1211.53846153846, 0, 470581950.72901, 0, 0, 0, 201923076.923077, 605769230.769231, 0, 7269.23076923077, 100961538.461538, 0, 0, 0, 942879587.732529, 0, 0, 0, 403846153.846154,
        0, 0, 0, 0, 0, 0, 0, 0, 134043489.190548, 0, 134615384.615385, 0, 0, 0, 0, 0, 0, 0, 0, 0, 269802664.655606, 0, 269230769.230769, 0,
        -4117.64705882353, 0, -343137254.901961, -686.274509803922, 0, -57189542.4836601, 0, 0, 0, 112629084.96732, 0, 1355.04901960784, 4117.64705882353, 0, 343137254.901961, 0, 0, 57189542.4836601, 0, 0, 0, 230508169.934641, 0, 2076.32352941177,
        0, -4117.64705882353, 0, 0, -686.274509803922, 0, 0, 0, 134615384.615385, 0, 134043489.190548, 0, 0, 4117.64705882353, 0, 0, 0, 0, 0, 0, 269230769.230769, 0, 269802664.655606, 0,
        -605769230.769231, 0, -11386.8778280543, -100961538.461538, 0, -1897.81297134238, 0, 201923076.923077, 0, 0, 0, 470581950.72901, 605769230.769231, 0, 11386.8778280543, 100961538.461538, 0, 0, 0, 403846153.846154, 0, 0, 0, 942879587.732529;

    ChMatrixNM<double, 24, 24> Expected_JacobianR_NoDispSmallVelWithDamping;
    Expected_JacobianR_NoDispSmallVelWithDamping <<
        33923076.9230769, 0, 0, 2826923.07692308, 0, 0, 0, -6057692.30769231, 0, 0, 0, -6057692.30769231, -33923076.9230769, 0, 0, 2826923.07692308, 0, 0, 0, -6057692.30769231, 0, 0, 0, -6057692.30769231,
        0, 8235294.11764706, 0, 0, 686274.509803922, 0, -3431372.54901961, 0, 0, 0, 0, 0, 0, -8235294.11764706, 0, 0, 686274.509803922, 0, -3431372.54901961, 0, 0, 0, 0, 0,
        0, 0, 8235294.11764706, 0, 0, 686274.509803922, 0, 0, 0, -3431372.54901961, 0, 0, 0, 0, -8235294.11764706, 0, 0, 686274.509803922, 0, 0, 0, -3431372.54901961, 0, 0,
        2826923.07692308, 0, 0, 3769230.76923077, 0, 0, 0, 1009615.38461538, 0, 0, 0, 1009615.38461538, -2826923.07692308, 0, 0, -942307.692307692, 0, 0, 0, -1009615.38461538, 0, 0, 0, -1009615.38461538,
        0, 686274.509803922, 0, 0, 915032.679738562, 0, 571895.424836601, 0, 0, 0, 0, 0, 0, -686274.509803922, 0, 0, -228758.169934641, 0, -571895.424836601, 0, 0, 0, 0, 0,
        0, 0, 686274.509803922, 0, 0, 915032.679738562, 0, 0, 0, 571895.424836601, 0, 0, 0, 0, -686274.509803922, 0, 0, -228758.169934641, 0, 0, 0, -571895.424836601, 0, 0,
        0, -3431372.54901961, 0, 0, 571895.424836601, 0, 2305081.69934641, 0, 0, 0, 0, 0, 0, 3431372.54901961, 0, 0, -571895.424836601, 0, 1126290.8496732, 0, 0, 0, 0, 0,
        -6057692.30769231, 0, 0, 1009615.38461538, 0, 0, 0, 9428795.87732529, 0, 0, 0, 4038461.53846154, 6057692.30769231, 0, 0, -1009615.38461538, 0, 0, 0, 4705819.5072901, 0, 0, 0, 2019230.76923077,
        0, 0, 0, 0, 0, 0, 0, 0, 2698026.64655606, 0, 2692307.69230769, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1340434.89190548, 0, 1346153.84615385, 0,
        0, 0, -3431372.54901961, 0, 0, 571895.424836601, 0, 0, 0, 2305081.69934641, 0, 0, 0, 0, 3431372.54901961, 0, 0, -571895.424836601, 0, 0, 0, 1126290.8496732, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 2692307.69230769, 0, 2698026.64655606, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1346153.84615385, 0, 1340434.89190548, 0,
        -6057692.30769231, 0, 0, 1009615.38461538, 0, 0, 0, 4038461.53846154, 0, 0, 0, 9428795.87732529, 6057692.30769231, 0, 0, -1009615.38461538, 0, 0, 0, 2019230.76923077, 0, 0, 0, 4705819.5072901,
        -33923076.9230769, 0, 0, -2826923.07692308, 0, 0, 0, 6057692.30769231, 0, 0, 0, 6057692.30769231, 33923076.9230769, 0, 0, -2826923.07692308, 0, 0, 0, 6057692.30769231, 0, 0, 0, 6057692.30769231,
        0, -8235294.11764706, 0, 0, -686274.509803922, 0, 3431372.54901961, 0, 0, 0, 0, 0, 0, 8235294.11764706, 0, 0, -686274.509803922, 0, 3431372.54901961, 0, 0, 0, 0, 0,
        0, 0, -8235294.11764706, 0, 0, -686274.509803922, 0, 0, 0, 3431372.54901961, 0, 0, 0, 0, 8235294.11764706, 0, 0, -686274.509803922, 0, 0, 0, 3431372.54901961, 0, 0,
        2826923.07692308, 0, 0, -942307.692307692, 0, 0, 0, -1009615.38461538, 0, 0, 0, -1009615.38461538, -2826923.07692308, 0, 0, 3769230.76923077, 0, 0, 0, 1009615.38461538, 0, 0, 0, 1009615.38461538,
        0, 686274.509803922, 0, 0, -228758.169934641, 0, -571895.424836601, 0, 0, 0, 0, 0, 0, -686274.509803922, 0, 0, 915032.679738562, 0, 571895.424836601, 0, 0, 0, 0, 0,
        0, 0, 686274.509803922, 0, 0, -228758.169934641, 0, 0, 0, -571895.424836601, 0, 0, 0, 0, -686274.509803922, 0, 0, 915032.679738562, 0, 0, 0, 571895.424836601, 0, 0,
        0, -3431372.54901961, 0, 0, -571895.424836601, 0, 1126290.8496732, 0, 0, 0, 0, 0, 0, 3431372.54901961, 0, 0, 571895.424836601, 0, 2305081.69934641, 0, 0, 0, 0, 0,
        -6057692.30769231, 0, 0, -1009615.38461538, 0, 0, 0, 4705819.5072901, 0, 0, 0, 2019230.76923077, 6057692.30769231, 0, 0, 1009615.38461538, 0, 0, 0, 9428795.87732529, 0, 0, 0, 4038461.53846154,
        0, 0, 0, 0, 0, 0, 0, 0, 1340434.89190548, 0, 1346153.84615385, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2698026.64655606, 0, 2692307.69230769, 0,
        0, 0, -3431372.54901961, 0, 0, -571895.424836601, 0, 0, 0, 1126290.8496732, 0, 0, 0, 0, 3431372.54901961, 0, 0, 571895.424836601, 0, 0, 0, 2305081.69934641, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1346153.84615385, 0, 1340434.89190548, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2692307.69230769, 0, 2698026.64655606, 0,
        -6057692.30769231, 0, 0, -1009615.38461538, 0, 0, 0, 2019230.76923077, 0, 0, 0, 4705819.5072901, 6057692.30769231, 0, 0, 1009615.38461538, 0, 0, 0, 4038461.53846154, 0, 0, 0, 9428795.87732529;


    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation

    //Setup the test conditions
    ChVector<double> OriginalVel = m_nodeB->GetPos_dt();
    m_nodeB->SetPos_dt(ChVector<>(0.0, 0.0, 0.001));
    m_element->SetAlphaDamp(0.01);

    ChVectorDynamic<double> InternalForceNoDispSmallVelNoGravity;
    InternalForceNoDispSmallVelNoGravity.resize(24);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceNoDispSmallVelNoGravity);

    ChMatrixDynamic<double> JacobianK_NoDispSmallVelWithDamping;
    JacobianK_NoDispSmallVelWithDamping.resize(24, 24);
    m_element->ComputeKRMmatricesGlobal(JacobianK_NoDispSmallVelWithDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_NoDispSmallVelWithDamping;
    JacobianR_NoDispSmallVelWithDamping.resize(24, 24);
    m_element->ComputeKRMmatricesGlobal(JacobianR_NoDispSmallVelWithDamping, 0, 1, 0);

    //Reset the element conditions back to its original values
    m_nodeB->SetPos_dt(OriginalVel);
    m_element->SetAlphaDamp(0.0);


    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_NoDispSmallVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(24, 24);

    double small_terms_JacR = 1e-4*Expected_JacobianR_NoDispSmallVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacR = 0;
    double max_percent_error_JacR = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacR;
    percent_error_matrix_JacR.resize(24, 24);


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
    int num_nodes = num_elements + 1;
    double dx = length / (num_nodes - 1);

    // Setup beam cross section gradients to initially align with the global y and z directions
    ChVector<> dir1(1, 0, 0);
    ChVector<> dir2(0, 1, 0);
    ChVector<> dir3(0, 0, 1);

    // Create the first node and fix it completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, 0, 0.0), dir1, dir2, dir3);
    nodeA->SetFixed(true);
    mesh->AddNode(nodeA);

    auto elementlast = chrono_types::make_shared<ElementVersion>();
    std::shared_ptr<ChNodeFEAxyzDDD> nodeEndPoint;

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(dx * i, 0, 0), dir1, dir2, dir3);
        mesh->AddNode(nodeB);

        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB);
        element->SetDimensions(dx, width, width);
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

    // Find the static solution for the system (final twist angle)
    system->DoStaticLinear();

    // Calculate the axial displacement of the end of the ANCF beam mesh
    ChVector<> point;
    ChQuaternion<> rot;
    elementlast->EvaluateSectionFrame(1, point, rot);
    
    // For Analytical Formula, see a mechanics of materials textbook (delta = PL/AE)
    double Displacement_Theory = (TIP_FORCE * length) / (width*width*E);
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
    int num_nodes = num_elements + 1;
    double dx = length / (num_nodes - 1);

    // Setup beam cross section gradients to initially align with the global y and z directions
    ChVector<> dir1(1, 0, 0);
    ChVector<> dir2(0, 1, 0);
    ChVector<> dir3(0, 0, 1);

    // Create the first node and fix it completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, 0, 0.0), dir1, dir2, dir3);
    nodeA->SetFixed(true);
    mesh->AddNode(nodeA);

    auto elementlast = chrono_types::make_shared<ElementVersion>();
    std::shared_ptr<ChNodeFEAxyzDDD> nodeEndPoint;

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(dx * i, 0, 0), dir1, dir2, dir3);
        mesh->AddNode(nodeB);

        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB);
        element->SetDimensions(dx, width, width);
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
bool ANCFBeamTest<ElementVersion, MaterialVersion>::AxialTwistCheck(int msglvl) {
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
    int num_nodes = num_elements + 1;
    double dx = length / (num_nodes - 1);

    // Setup beam cross section gradients to initially align with the global y and z directions
    ChVector<> dir1(1, 0, 0);
    ChVector<> dir2(0, 1, 0);
    ChVector<> dir3(0, 0, 1);

    // Create the first node and fix it completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, 0, 0.0), dir1, dir2, dir3);
    nodeA->SetFixed(true);
    mesh->AddNode(nodeA);

    auto elementlast = chrono_types::make_shared<ElementVersion>();
    std::shared_ptr<ChNodeFEAxyzDDD> nodeEndPoint;

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(dx * i, 0, 0), dir1, dir2, dir3);
        mesh->AddNode(nodeB);

        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB);
        element->SetDimensions(dx, width, width);
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

    // Find the nonlinear static solution for the system (final twist angle)
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

    double Percent_Error = (Tip_Angles.x() - Angle_Theory) / Angle_Theory * 100;

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
    ANCFBeamTest<ChElementBeamANCF_3243_TR01, ChMaterialBeamANCF_3243_TR01> ChElementBeamANCF_3243_TR01_test;
    if (ChElementBeamANCF_3243_TR01_test.RunElementChecks(0))
        print_green("ChElementBeamANCF_3243_TR01 Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_3243_TR01 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_3243_TR02, ChMaterialBeamANCF_3243_TR02> ChElementBeamANCF_3243_TR02_test;
    if (ChElementBeamANCF_3243_TR02_test.RunElementChecks(0))
        print_green("ChElementBeamANCF_3243_TR02 Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_3243_TR02 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_3243_TR03, ChMaterialBeamANCF_3243_TR03> ChElementBeamANCF_3243_TR03_test;
    if (ChElementBeamANCF_3243_TR03_test.RunElementChecks(0))
        print_green("ChElementBeamANCF_3243_TR03 Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_3243_TR03 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_3243_TR04, ChMaterialBeamANCF_3243_TR04> ChElementBeamANCF_3243_TR04_test;
    if (ChElementBeamANCF_3243_TR04_test.RunElementChecks(0))
        print_green("ChElementBeamANCF_3243_TR04 Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_3243_TR04 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_3243_TR05, ChMaterialBeamANCF_3243_TR05> ChElementBeamANCF_3243_TR05_test;
    if (ChElementBeamANCF_3243_TR05_test.RunElementChecks(0))
        print_green("ChElementBeamANCF_3243_TR05 Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_3243_TR05 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_3243_TR06, ChMaterialBeamANCF_3243_TR06> ChElementBeamANCF_3243_TR06_test;
    if (ChElementBeamANCF_3243_TR06_test.RunElementChecks(0))
        print_green("ChElementBeamANCF_3243_TR06 Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_3243_TR06 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_3243_TR07, ChMaterialBeamANCF_3243_TR07> ChElementBeamANCF_3243_TR07_test;
    if (ChElementBeamANCF_3243_TR07_test.RunElementChecks(0))
        print_green("ChElementBeamANCF_3243_TR07 Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_3243_TR07 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_3243_TR08, ChMaterialBeamANCF_3243_TR08> ChElementBeamANCF_3243_TR08_test;
    if (ChElementBeamANCF_3243_TR08_test.RunElementChecks(0))
        print_green("ChElementBeamANCF_3243_TR08 Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_3243_TR08 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_3243_TR08b, ChMaterialBeamANCF_3243_TR08b> ChElementBeamANCF_3243_TR08b_test;
    if (ChElementBeamANCF_3243_TR08b_test.RunElementChecks(0))
        print_green("ChElementBeamANCF_3243_TR08b Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_3243_TR08b Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_3243_TR09, ChMaterialBeamANCF_3243_TR09> ChElementBeamANCF_3243_TR09_test;
    if (ChElementBeamANCF_3243_TR09_test.RunElementChecks(0))
        print_green("ChElementBeamANCF_3243_TR09 Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_3243_TR09 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_3243_TR10, ChMaterialBeamANCF_3243_TR10> ChElementBeamANCF_3243_TR10_test;
    if (ChElementBeamANCF_3243_TR10_test.RunElementChecks(0))
        print_green("ChElementBeamANCF_3243_TR10 Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_3243_TR10 Element Checks = FAILED\n");

    return 0;
}

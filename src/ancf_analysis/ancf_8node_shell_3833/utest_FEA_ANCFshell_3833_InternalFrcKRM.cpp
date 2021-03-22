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
// Unit test for ANCF 3833 shell elements
//
// =============================================================================

#include "chrono/ChConfig.h"

#include "chrono/physics/ChSystemSMC.h"
#include "chrono/solver/ChDirectSolverLS.h"

#include "chrono/fea/ChElementShellANCF_8.h"
#include "chrono/fea/ChElementShellANCF_3833_TR00.h"
#include "chrono/fea/ChElementShellANCF_3833_TR01.h"
#include "chrono/fea/ChElementShellANCF_3833_TR02.h"
#include "chrono/fea/ChElementShellANCF_3833_TR03.h"
#include "chrono/fea/ChElementShellANCF_3833_TR04.h"
#include "chrono/fea/ChElementShellANCF_3833_TR05.h"
#include "chrono/fea/ChElementShellANCF_3833_TR06.h"
#include "chrono/fea/ChElementShellANCF_3833_TR07.h"
#include "chrono/fea/ChElementShellANCF_3833_TR08.h"
#include "chrono/fea/ChElementShellANCF_3833_TR08b.h"
#include "chrono/fea/ChElementShellANCF_3833_TR09.h"
#include "chrono/fea/ChElementShellANCF_3833_TR10.h"
#include "chrono/fea/ChElementShellANCF_3833_TR11.h"

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

static const std::string ref_dir = "../data/testing/fea/";

// =============================================================================

void print_green(std::string text) {
    std::cout << "\033[1;32m" << text << "\033[0m";
}

void print_red(std::string text) {
    std::cout << "\033[1;31m" << text << "\033[0m";
}

bool load_validation_data(const std::string& filename, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& data) {
    std::ifstream file(ref_dir + filename);
    if (!file.is_open()) {
        print_red("ERROR!  Cannot open file: " + ref_dir + filename + "\n");
        return false;
    }
    for (unsigned int r = 0; r < data.rows(); r++) {
        for (unsigned int c = 0; c < data.cols(); c++) {
            file >> data(r, c);
        }
    }
    file.close();
    return true;
}

// =============================================================================

template <typename ElementVersion, typename MaterialVersion>
class ANCFOrgShellTest {
public:
    ANCFOrgShellTest();

    ~ANCFOrgShellTest() { delete m_system; }

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
    std::shared_ptr<ChNodeFEAxyzDD> m_nodeB;
};

// =============================================================================

template <typename ElementVersion, typename MaterialVersion>
ANCFOrgShellTest<ElementVersion, MaterialVersion>::ANCFOrgShellTest() {
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

    // Setup shell normals to initially align with the global z direction with no curvature
    ChVector<> dir1(0, 0, 1);
    ChVector<> Curv1(0, 0, 0);

    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0, 0.0), dir1, Curv1);
    mesh->AddNode(nodeA);
    auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(length, 0, 0), dir1, Curv1);
    mesh->AddNode(nodeB);
    m_nodeB = nodeB;
    auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(length, width, 0), dir1, Curv1);
    mesh->AddNode(nodeC);
    auto nodeD = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, width, 0), dir1, Curv1);
    mesh->AddNode(nodeD);
    auto nodeE = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0.5*length, 0, 0.0), dir1, Curv1);
    mesh->AddNode(nodeE);
    auto nodeF = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(length, 0.5*width, 0), dir1, Curv1);
    mesh->AddNode(nodeF);
    auto nodeG = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0.5*length, width, 0), dir1, Curv1);
    mesh->AddNode(nodeG);
    auto nodeH = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0.5*width, 0), dir1, Curv1);
    mesh->AddNode(nodeH);



    auto element = chrono_types::make_shared<ElementVersion>();
    element->SetNodes(nodeA, nodeB, nodeC, nodeD, nodeE, nodeF, nodeG, nodeH);
    //element->SetDimensions(length, width, thickness);
    //element->SetMaterial(material);
    element->SetDimensions(length, width);
    element->AddLayer(thickness, 0 * CH_C_DEG_TO_RAD, material);
    element->SetAlphaDamp(0.0);
    element->SetGravityOn(false);  //Enable the efficient ANCF method for calculating the application of gravity to the element
    //element->SetStrainFormulation(ElementVersion::StrainFormulation::CMPoisson);
    mesh->AddElement(element);

    m_element = element;

    mesh->SetAutomaticGravity(false); //Turn off the default method for applying gravity to the mesh since it is less efficient for ANCF elements

// =============================================================================
//  Force a call to SetupInital so that all of the pre-computation steps are called
// =============================================================================

    m_system->Update();
}

template <typename ElementVersion, typename MaterialVersion>
bool ANCFOrgShellTest<ElementVersion, MaterialVersion>::RunElementChecks(int msglvl) {
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
bool ANCFOrgShellTest<ElementVersion, MaterialVersion>::MassMatrixCheck(int msglvl) {
    // =============================================================================
    //  Check the Mass Matrix
    //  (Result should be nearly exact - No expected error)
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_MassMatrix_Compact;
    Expected_MassMatrix_Compact.resize(24, 24);
    if (!load_validation_data("UT_ANCFShell_3833_MassMatrix.txt", Expected_MassMatrix_Compact))
        return false;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_MassMatrix;
    Expected_MassMatrix.resize(3 * Expected_MassMatrix_Compact.rows(), 3 * Expected_MassMatrix_Compact.cols());
    Expected_MassMatrix.setZero();
    for (unsigned int r = 0; r < Expected_MassMatrix_Compact.rows(); r++) {
        for (unsigned int c = 0; c < Expected_MassMatrix_Compact.cols(); c++) {
            Expected_MassMatrix(3 * r, 3 * c) = Expected_MassMatrix_Compact(r, c);
            Expected_MassMatrix(3 * r + 1, 3 * c + 1) = Expected_MassMatrix_Compact(r, c);
            Expected_MassMatrix(3 * r + 2, 3 * c + 2) = Expected_MassMatrix_Compact(r, c);
        }
    }

    ChMatrixDynamic<double> MassMatrix;
    MassMatrix.resize(72, 72);
    m_element->ComputeKRMmatricesGlobal(MassMatrix, 0, 0, 1);
    ChMatrixNM<double, 24, 24> MassMatrix_compact;
    for (unsigned int i = 0; i < 24; i++) {
        for (unsigned int j = 0; j < 24; j++) {
            MassMatrix_compact(i, j) = MassMatrix(3 * i, 3 * j);
        }
    }

    double MaxAbsError = (MassMatrix - Expected_MassMatrix).cwiseAbs().maxCoeff();
    bool passed_test = (MaxAbsError <= 0.001);

    if (msglvl >= 2) {
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
bool ANCFOrgShellTest<ElementVersion, MaterialVersion>::GeneralizedGravityForceCheck(int msglvl) {
    // =============================================================================
    //  Generalized Force due to Gravity
    //  (Result should be nearly exact - No expected error)
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_InternalForceDueToGravity;
    Expected_InternalForceDueToGravity.resize(72, 1);
    if (!load_validation_data("UT_ANCFShell_3833_Grav.txt", Expected_InternalForceDueToGravity))
        return false;

    ChVectorDynamic<double> InternalForceNoDispNoVelWithGravity;
    InternalForceNoDispNoVelWithGravity.resize(72);
    m_element->SetGravityOn(true);
    m_element->ComputeInternalForces(InternalForceNoDispNoVelWithGravity);

    ChVectorDynamic<double> InternalForceNoDispNoVelNoGravity;
    InternalForceNoDispNoVelNoGravity.resize(72);
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
bool ANCFOrgShellTest<ElementVersion, MaterialVersion>::GeneralizedInternalForceNoDispNoVelCheck(int msglvl) {
    // =============================================================================
    //  Check the Internal Force at Zero Displacement, Zero Velocity
    //  (should equal all 0's by definition)
    //  (Assumes that the element has not been changed from the initialized state)
    // =============================================================================

    ChVectorDynamic<double> InternalForceNoDispNoVelNoGravity;
    InternalForceNoDispNoVelNoGravity.resize(72);
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
bool ANCFOrgShellTest<ElementVersion, MaterialVersion>::GeneralizedInternalForceSmallDispNoVelCheck(int msglvl) {
    // =============================================================================
    //  Check the Internal Force at a Given Displacement 
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_InternalForceSmallDispNoVelNoGravity;
    Expected_InternalForceSmallDispNoVelNoGravity.resize(72, 1);
    if (!load_validation_data("UT_ANCFShell_3833_IntFrcSmallDispNoVelNoGravity.txt", Expected_InternalForceSmallDispNoVelNoGravity))
        return false;

    //Setup the test conditions
    ChVector<double> OriginalPos = m_nodeB->GetPos();
    m_nodeB->SetPos(ChVector<>(m_nodeB->GetPos().x(), m_nodeB->GetPos().y(), 0.001));

    ChVectorDynamic<double> InternalForceSmallDispNoVelNoGravity;
    InternalForceSmallDispNoVelNoGravity.resize(72);
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
bool ANCFOrgShellTest<ElementVersion, MaterialVersion>::GeneralizedInternalForceNoDispSmallVelCheck(int msglvl) {
    // =============================================================================
    //  Check the Internal Force at a No Displacement with a Given Nodal Velocity
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_InternalForceNoDispSmallVelNoGravity;
    Expected_InternalForceNoDispSmallVelNoGravity.resize(72, 1);
    if (!load_validation_data("UT_ANCFShell_3833_IntFrcNoDispSmallVelNoGravity.txt", Expected_InternalForceNoDispSmallVelNoGravity))
        return false;

    //Setup the test conditions
    ChVector<double> OriginalVel = m_nodeB->GetPos_dt();
    m_nodeB->SetPos_dt(ChVector<>(0.0, 0.0, 0.001));
    m_element->SetAlphaDamp(0.01);

    ChVectorDynamic<double> InternalForceNoDispSmallVelNoGravity;
    InternalForceNoDispSmallVelNoGravity.resize(72);
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
bool ANCFOrgShellTest<ElementVersion, MaterialVersion>::JacobianNoDispNoVelNoDampingCheck(int msglvl) {
    // =============================================================================
    //  Check the Jacobian at No Displacement/Velocity - No Damping
    //  (some small error expected depending on the formulation/steps used)
    //  (The R contribution should be all zeros since Damping is not enabled)
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianK_NoDispNoVelNoDamping;
    Expected_JacobianK_NoDispNoVelNoDamping.resize(72, 72);
    if (!load_validation_data("UT_ANCFShell_3833_JacNoDispNoVelNoDamping.txt", Expected_JacobianK_NoDispNoVelNoDamping))
        return false;

    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation
    ChVectorDynamic<double> InternalForceNoDispNoVelNoGravity;
    InternalForceNoDispNoVelNoGravity.resize(72);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceNoDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_NoDispNoVelNoDamping;
    JacobianK_NoDispNoVelNoDamping.resize(72, 72);
    m_element->ComputeKRMmatricesGlobal(JacobianK_NoDispNoVelNoDamping, 1.0, 0.0, 0.0);

    ChMatrixDynamic<double> JacobianR_NoDispNoVelNoDamping;
    JacobianR_NoDispNoVelNoDamping.resize(72, 72);
    m_element->ComputeKRMmatricesGlobal(JacobianR_NoDispNoVelNoDamping, 0.0, 1.0, 0.0);

    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_NoDispNoVelNoDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(72, 72);
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
bool ANCFOrgShellTest<ElementVersion, MaterialVersion>::JacobianSmallDispNoVelNoDampingCheck(int msglvl) {
    // =============================================================================
    //  Check the Jacobian at Small Displacement No Velocity - No Damping
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianK_SmallDispNoVelNoDamping;
    Expected_JacobianK_SmallDispNoVelNoDamping.resize(72, 72);
    if (!load_validation_data("UT_ANCFShell_3833_JacSmallDispNoVelNoDamping.txt", Expected_JacobianK_SmallDispNoVelNoDamping))
        return false;

    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation

    //Setup the test conditions
    ChVector<double> OriginalPos = m_nodeB->GetPos();
    m_nodeB->SetPos(ChVector<>(m_nodeB->GetPos().x(), m_nodeB->GetPos().y(), 0.001));

    ChVectorDynamic<double> InternalForceSmallDispNoVelNoGravity;
    InternalForceSmallDispNoVelNoGravity.resize(72);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceSmallDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_SmallDispNoVelNoDamping;
    JacobianK_SmallDispNoVelNoDamping.resize(72, 72);
    m_element->ComputeKRMmatricesGlobal(JacobianK_SmallDispNoVelNoDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_SmallDispNoVelNoDamping;
    JacobianR_SmallDispNoVelNoDamping.resize(72, 72);
    m_element->ComputeKRMmatricesGlobal(JacobianR_SmallDispNoVelNoDamping, 0, 1, 0);

    //Reset the element conditions back to its original values
    m_nodeB->SetPos(OriginalPos);


    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_SmallDispNoVelNoDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(72, 72);
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
bool ANCFOrgShellTest<ElementVersion, MaterialVersion>::JacobianNoDispNoVelWithDampingCheck(int msglvl) {
    // =============================================================================
    //  Check the Jacobian at No Displacement/Velocity - With Damping
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_Jacobians;
    Expected_Jacobians.resize(2 * 72, 72);
    if (!load_validation_data("UT_ANCFShell_3833_JacNoDispNoVelWithDamping.txt", Expected_Jacobians))
        return false;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianK_NoDispNoVelWithDamping;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianR_NoDispNoVelWithDamping;
    Expected_JacobianK_NoDispNoVelWithDamping = Expected_Jacobians.block(0, 0, Expected_Jacobians.cols(), Expected_Jacobians.cols());
    Expected_JacobianR_NoDispNoVelWithDamping = Expected_Jacobians.block(Expected_Jacobians.cols(), 0, Expected_Jacobians.cols(), Expected_Jacobians.cols());

    //Setup the test conditions
    m_element->SetAlphaDamp(0.01);

    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation
    ChVectorDynamic<double> InternalForceNoDispNoVelNoGravity;
    InternalForceNoDispNoVelNoGravity.resize(72);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceNoDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_NoDispNoVelWithDamping;
    JacobianK_NoDispNoVelWithDamping.resize(72, 72);
    m_element->ComputeKRMmatricesGlobal(JacobianK_NoDispNoVelWithDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_NoDispNoVelWithDamping;
    JacobianR_NoDispNoVelWithDamping.resize(72, 72);
    m_element->ComputeKRMmatricesGlobal(JacobianR_NoDispNoVelWithDamping, 0, 1, 0);

    //Reset the element conditions back to its original values
    m_element->SetAlphaDamp(0.0);


    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_NoDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(72, 72);

    double small_terms_JacR = 1e-4*Expected_JacobianR_NoDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacR = 0;
    double max_percent_error_JacR = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacR;
    percent_error_matrix_JacR.resize(72, 72);


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
bool ANCFOrgShellTest<ElementVersion, MaterialVersion>::JacobianSmallDispNoVelWithDampingCheck(int msglvl) {
    // =============================================================================
    //  Check the Jacobian at Small Displacement No Velocity - With Damping
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_Jacobians;
    Expected_Jacobians.resize(2 * 72, 72);
    if (!load_validation_data("UT_ANCFShell_3833_JacSmallDispNoVelWithDamping.txt", Expected_Jacobians))
        return false;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianK_SmallDispNoVelWithDamping;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianR_SmallDispNoVelWithDamping;
    Expected_JacobianK_SmallDispNoVelWithDamping = Expected_Jacobians.block(0, 0, Expected_Jacobians.cols(), Expected_Jacobians.cols());
    Expected_JacobianR_SmallDispNoVelWithDamping = Expected_Jacobians.block(Expected_Jacobians.cols(), 0, Expected_Jacobians.cols(), Expected_Jacobians.cols());


    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation

    //Setup the test conditions
    ChVector<double> OriginalPos = m_nodeB->GetPos();
    m_nodeB->SetPos(ChVector<>(m_nodeB->GetPos().x(), m_nodeB->GetPos().y(), 0.001));
    m_element->SetAlphaDamp(0.01);

    ChVectorDynamic<double> InternalForceSmallDispNoVelNoGravity;
    InternalForceSmallDispNoVelNoGravity.resize(72);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceSmallDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_SmallDispNoVelWithDamping;
    JacobianK_SmallDispNoVelWithDamping.resize(72, 72);
    m_element->ComputeKRMmatricesGlobal(JacobianK_SmallDispNoVelWithDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_SmallDispNoVelWithDamping;
    JacobianR_SmallDispNoVelWithDamping.resize(72, 72);
    m_element->ComputeKRMmatricesGlobal(JacobianR_SmallDispNoVelWithDamping, 0, 1, 0);

    //Reset the element conditions back to its original values
    m_nodeB->SetPos(OriginalPos);
    m_element->SetAlphaDamp(0.0);


    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_SmallDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(72, 72);

    double small_terms_JacR = 1e-4*Expected_JacobianR_SmallDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacR = 0;
    double max_percent_error_JacR = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacR;
    percent_error_matrix_JacR.resize(72, 72);


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
bool ANCFOrgShellTest<ElementVersion, MaterialVersion>::JacobianNoDispSmallVelWithDampingCheck(int msglvl) {
    // =============================================================================
    //  Check the Jacobian at No Displacement Small Velocity - With Damping
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_Jacobians;
    Expected_Jacobians.resize(2 * 72, 72);
    if (!load_validation_data("UT_ANCFShell_3833_JacNoDispSmallVelWithDamping.txt", Expected_Jacobians))
        return false;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianK_NoDispSmallVelWithDamping;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianR_NoDispSmallVelWithDamping;
    Expected_JacobianK_NoDispSmallVelWithDamping = Expected_Jacobians.block(0, 0, Expected_Jacobians.cols(), Expected_Jacobians.cols());
    Expected_JacobianR_NoDispSmallVelWithDamping = Expected_Jacobians.block(Expected_Jacobians.cols(), 0, Expected_Jacobians.cols(), Expected_Jacobians.cols());


    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation

    //Setup the test conditions
    ChVector<double> OriginalVel = m_nodeB->GetPos_dt();
    m_nodeB->SetPos_dt(ChVector<>(0.0, 0.0, 0.001));
    m_element->SetAlphaDamp(0.01);

    ChVectorDynamic<double> InternalForceNoDispSmallVelNoGravity;
    InternalForceNoDispSmallVelNoGravity.resize(72);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceNoDispSmallVelNoGravity);

    ChMatrixDynamic<double> JacobianK_NoDispSmallVelWithDamping;
    JacobianK_NoDispSmallVelWithDamping.resize(72, 72);
    m_element->ComputeKRMmatricesGlobal(JacobianK_NoDispSmallVelWithDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_NoDispSmallVelWithDamping;
    JacobianR_NoDispSmallVelWithDamping.resize(72, 72);
    m_element->ComputeKRMmatricesGlobal(JacobianR_NoDispSmallVelWithDamping, 0, 1, 0);

    //Reset the element conditions back to its original values
    m_nodeB->SetPos_dt(OriginalVel);
    m_element->SetAlphaDamp(0.0);


    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_NoDispSmallVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(72, 72);

    double small_terms_JacR = 1e-4*Expected_JacobianR_NoDispSmallVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacR = 0;
    double max_percent_error_JacR = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacR;
    percent_error_matrix_JacR.resize(72, 72);


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
bool ANCFOrgShellTest<ElementVersion, MaterialVersion>::AxialDisplacementCheck(int msglvl) {
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
    double nu = 0.33;
    double G = E / (2 * (1 + nu));

    auto material = chrono_types::make_shared<MaterialVersion>(rho, E, nu);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    system->Add(mesh);

    // Populate the mesh container with a the nodes and elements for the meshed beam
    double dx = length / (num_elements);

    // Setup shell normals to initially align with the global z direction with no curvature
    ChVector<> dir1(0, 0, 1);
    ChVector<> Curv1(0, 0, 0);

    // Create the first nodes and fix them completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0, 0.0), dir1, Curv1);
    mesh->AddNode(nodeA);
    nodeA->SetFixed(true);
    auto nodeD = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, width, 0), dir1, Curv1);
    mesh->AddNode(nodeD);
    nodeD->SetFixed(true);
    auto nodeH = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0.5*width, 0), dir1, Curv1);
    mesh->AddNode(nodeH);
    nodeH->SetFixed(true);



    auto elementlast = chrono_types::make_shared<ElementVersion>();
    std::shared_ptr<ChNodeFEAxyzDD> nodeEndPoint;

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx, 0, 0), dir1, Curv1);
        mesh->AddNode(nodeB);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx, width, 0), dir1, Curv1);
        mesh->AddNode(nodeC);
        auto nodeE = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx - 0.5*dx, 0, 0.0), dir1, Curv1);
        mesh->AddNode(nodeE);
        auto nodeF = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx, 0.5*width, 0), dir1, Curv1);
        mesh->AddNode(nodeF);
        auto nodeG = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx - 0.5*dx, width, 0), dir1, Curv1);
        mesh->AddNode(nodeG);



        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC, nodeD, nodeE, nodeF, nodeG, nodeH);
        //element->SetDimensions(dx, width, height);
        //element->SetMaterial(material);
        element->SetDimensions(dx, width);
        element->AddLayer(height, 0 * CH_C_DEG_TO_RAD, material);
        element->SetAlphaDamp(0.0);
        element->SetGravityOn(
            false);  // Enable the efficient ANCF method for calculating the application of gravity to the element
        //element->SetStrainFormulation(ElementVersion::StrainFormulation::CMPoisson);
        // element->SetStrainFormulation(ElementVersion::StrainFormulation::CMNoPoisson);
        mesh->AddElement(element);

        nodeA = nodeB;
        nodeD = nodeC;
        nodeH = nodeF;
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
    mload->loader.SetApplication(1.0, 0);  // specify application point
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
        std::cout << "Axial Pull - Tip Displacement Check (Percent Error less than 5%) - Percent Error: " << Percent_Error << "% ";
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
bool ANCFOrgShellTest<ElementVersion, MaterialVersion>::CantileverCheck(int msglvl) {
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
    double nu = 0.33 * 0;
    double G = E / (2 * (1 + nu));

    auto material = chrono_types::make_shared<MaterialVersion>(rho, E, nu);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    system->Add(mesh);

    // Populate the mesh container with a the nodes and elements for the meshed beam
    double dx = length / (num_elements);

    // Setup shell normals to initially align with the global z direction with no curvature
    ChVector<> dir1(0, 0, 1);
    ChVector<> Curv1(0, 0, 0);

    // Create the first nodes and fix them completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0, 0.0), dir1, Curv1);
    mesh->AddNode(nodeA);
    nodeA->SetFixed(true);
    auto nodeD = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, width, 0), dir1, Curv1);
    mesh->AddNode(nodeD);
    nodeD->SetFixed(true);
    auto nodeH = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0.5*width, 0), dir1, Curv1);
    mesh->AddNode(nodeH);
    nodeH->SetFixed(true);



    auto elementlast = chrono_types::make_shared<ElementVersion>();
    std::shared_ptr<ChNodeFEAxyzDD> nodeEndPoint;

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx, 0, 0), dir1, Curv1);
        mesh->AddNode(nodeB);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx, width, 0), dir1, Curv1);
        mesh->AddNode(nodeC);
        auto nodeE = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx - 0.5*dx, 0, 0.0), dir1, Curv1);
        mesh->AddNode(nodeE);
        auto nodeF = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx, 0.5*width, 0), dir1, Curv1);
        mesh->AddNode(nodeF);
        auto nodeG = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx - 0.5*dx, width, 0), dir1, Curv1);
        mesh->AddNode(nodeG);



        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC, nodeD, nodeE, nodeF, nodeG, nodeH);
        //element->SetDimensions(dx, width, height);
        //element->SetMaterial(material);
        element->SetDimensions(dx, width);
        element->AddLayer(height, 0 * CH_C_DEG_TO_RAD, material);
        element->SetAlphaDamp(0.0);
        element->SetGravityOn(
            false);  // Enable the efficient ANCF method for calculating the application of gravity to the element
        //element->SetStrainFormulation(ElementVersion::StrainFormulation::CMPoisson);
        // element->SetStrainFormulation(ElementVersion::StrainFormulation::CMNoPoisson);
        mesh->AddElement(element);

        nodeA = nodeB;
        nodeD = nodeC;
        nodeH = nodeF;
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
        std::cout << "Cantilever Tip Displacement Check (Percent Error less than 5%) - Percent Error: " << Percent_Error << "% ";
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
bool ANCFOrgShellTest<ElementVersion, MaterialVersion>::AxialTwistCheck(int msglvl) {
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
    double nu = 0.33 * 0;
    double G = E / (2 * (1 + nu));

    auto material = chrono_types::make_shared<MaterialVersion>(rho, E, nu);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    system->Add(mesh);

    // Populate the mesh container with a the nodes and elements for the meshed beam
    double dx = length / (num_elements);

    // Setup shell normals to initially align with the global z direction with no curvature
    ChVector<> dir1(0, 0, 1);
    ChVector<> Curv1(0, 0, 0);

    // Create the first nodes and fix them completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0, 0.0), dir1, Curv1);
    mesh->AddNode(nodeA);
    nodeA->SetFixed(true);
    auto nodeD = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, width, 0), dir1, Curv1);
    mesh->AddNode(nodeD);
    nodeD->SetFixed(true);
    auto nodeH = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0.5*width, 0), dir1, Curv1);
    mesh->AddNode(nodeH);
    nodeH->SetFixed(true);



    auto elementlast = chrono_types::make_shared<ElementVersion>();
    std::shared_ptr<ChNodeFEAxyzDD> nodeEndPoint;

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx, 0, 0), dir1, Curv1);
        mesh->AddNode(nodeB);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx, width, 0), dir1, Curv1);
        mesh->AddNode(nodeC);
        auto nodeE = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx - 0.5*dx, 0, 0.0), dir1, Curv1);
        mesh->AddNode(nodeE);
        auto nodeF = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx, 0.5*width, 0), dir1, Curv1);
        mesh->AddNode(nodeF);
        auto nodeG = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx - 0.5*dx, width, 0), dir1, Curv1);
        mesh->AddNode(nodeG);



        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC, nodeD, nodeE, nodeF, nodeG, nodeH);
        //element->SetDimensions(dx, width, height);
        //element->SetMaterial(material);
        element->SetDimensions(dx, width);
        element->AddLayer(height, 0 * CH_C_DEG_TO_RAD, material);
        element->SetAlphaDamp(0.0);
        element->SetGravityOn(
            false);  // Enable the efficient ANCF method for calculating the application of gravity to the element
        //element->SetStrainFormulation(ElementVersion::StrainFormulation::CMPoisson);
        // element->SetStrainFormulation(ElementVersion::StrainFormulation::CMNoPoisson);
        mesh->AddElement(element);

        nodeA = nodeB;
        nodeD = nodeC;
        nodeH = nodeF;
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
        std::cout << "Axial Twist Angle Check (Percent Error less than 20%) - Percent Error: " << Percent_Error << "% ";
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
    std::shared_ptr<ChNodeFEAxyzDD> m_nodeB;
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

    // Setup shell normals to initially align with the global z direction with no curvature
    ChVector<> dir1(0, 0, 1);
    ChVector<> Curv1(0, 0, 0);

    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0, 0.0), dir1, Curv1);
    mesh->AddNode(nodeA);
    auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(length, 0, 0), dir1, Curv1);
    mesh->AddNode(nodeB);
    m_nodeB = nodeB;
    auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(length, width, 0), dir1, Curv1);
    mesh->AddNode(nodeC);
    auto nodeD = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, width, 0), dir1, Curv1);
    mesh->AddNode(nodeD);
    auto nodeE = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0.5*length, 0, 0.0), dir1, Curv1);
    mesh->AddNode(nodeE);
    auto nodeF = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(length, 0.5*width, 0), dir1, Curv1);
    mesh->AddNode(nodeF);
    auto nodeG = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0.5*length, width, 0), dir1, Curv1);
    mesh->AddNode(nodeG);
    auto nodeH = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0.5*width, 0), dir1, Curv1);
    mesh->AddNode(nodeH);



    auto element = chrono_types::make_shared<ElementVersion>();
    element->SetNodes(nodeA, nodeB, nodeC, nodeD, nodeE, nodeF, nodeG, nodeH);
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
    //ChMatrixNM<double, 72, 72> Expected_MassMatrix;
    //Loaded row by row to avoid a segmentation fault when compiling
    //Expected_MassMatrix.row(0) << 2.61666666666667, 0, 0, 0, 0, 0, 1.09027777777778E-05, 0, 0, 0.872222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-06, 0, 0, 1.30833333333333, 0, 0, 0, 0, 0, 5.45138888888889E-06, 0, 0, 0.872222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-06, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0;
    //Expected_MassMatrix.row(1) << 0, 2.61666666666667, 0, 0, 0, 0, 0, 1.09027777777778E-05, 0, 0, 0.872222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-06, 0, 0, 1.30833333333333, 0, 0, 0, 0, 0, 5.45138888888889E-06, 0, 0, 0.872222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-06, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0;
    //Expected_MassMatrix.row(2) << 0, 0, 2.61666666666667, 0, 0, 0, 0, 0, 1.09027777777778E-05, 0, 0, 0.872222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-06, 0, 0, 1.30833333333333, 0, 0, 0, 0, 0, 5.45138888888889E-06, 0, 0, 0.872222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-06, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05;
    //Expected_MassMatrix.row(3) << 0, 0, 0, 2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-06, 0, 0, 0, 0, 0, 0, 0, 0, 1.09027777777778E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-06, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0;
    //Expected_MassMatrix.row(4) << 0, 0, 0, 0, 2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-06, 0, 0, 0, 0, 0, 0, 0, 0, 1.09027777777778E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-06, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0;
    //Expected_MassMatrix.row(5) << 0, 0, 0, 0, 0, 2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-06, 0, 0, 0, 0, 0, 0, 0, 0, 1.09027777777778E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-06, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0;
    //Expected_MassMatrix.row(6) << 1.09027777777778E-05, 0, 0, 0, 0, 0, 8.17708333333333E-11, 0, 0, 3.63425925925926E-06, 0, 0, 0, 0, 0, 2.72569444444444E-11, 0, 0, 5.45138888888889E-06, 0, 0, 0, 0, 0, 4.08854166666667E-11, 0, 0, 3.63425925925926E-06, 0, 0, 0, 0, 0, 2.72569444444444E-11, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0;
    //Expected_MassMatrix.row(7) << 0, 1.09027777777778E-05, 0, 0, 0, 0, 0, 8.17708333333333E-11, 0, 0, 3.63425925925926E-06, 0, 0, 0, 0, 0, 2.72569444444444E-11, 0, 0, 5.45138888888889E-06, 0, 0, 0, 0, 0, 4.08854166666667E-11, 0, 0, 3.63425925925926E-06, 0, 0, 0, 0, 0, 2.72569444444444E-11, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0;
    //Expected_MassMatrix.row(8) << 0, 0, 1.09027777777778E-05, 0, 0, 0, 0, 0, 8.17708333333333E-11, 0, 0, 3.63425925925926E-06, 0, 0, 0, 0, 0, 2.72569444444444E-11, 0, 0, 5.45138888888889E-06, 0, 0, 0, 0, 0, 4.08854166666667E-11, 0, 0, 3.63425925925926E-06, 0, 0, 0, 0, 0, 2.72569444444444E-11, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11;
    //Expected_MassMatrix.row(9) << 0.872222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-06, 0, 0, 2.61666666666667, 0, 0, 0, 0, 0, 1.09027777777778E-05, 0, 0, 0.872222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-06, 0, 0, 1.30833333333333, 0, 0, 0, 0, 0, 5.45138888888889E-06, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0;
    //Expected_MassMatrix.row(10) << 0, 0.872222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-06, 0, 0, 2.61666666666667, 0, 0, 0, 0, 0, 1.09027777777778E-05, 0, 0, 0.872222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-06, 0, 0, 1.30833333333333, 0, 0, 0, 0, 0, 5.45138888888889E-06, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0;
    //Expected_MassMatrix.row(11) << 0, 0, 0.872222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-06, 0, 0, 2.61666666666667, 0, 0, 0, 0, 0, 1.09027777777778E-05, 0, 0, 0.872222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-06, 0, 0, 1.30833333333333, 0, 0, 0, 0, 0, 5.45138888888889E-06, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037;
    //Expected_MassMatrix.row(12) << 0, 0, 0, 7.26851851851852E-06, 0, 0, 0, 0, 0, 0, 0, 0, 2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-06, 0, 0, 0, 0, 0, 0, 0, 0, 1.09027777777778E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0;
    //Expected_MassMatrix.row(13) << 0, 0, 0, 0, 7.26851851851852E-06, 0, 0, 0, 0, 0, 0, 0, 0, 2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-06, 0, 0, 0, 0, 0, 0, 0, 0, 1.09027777777778E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0;
    //Expected_MassMatrix.row(14) << 0, 0, 0, 0, 0, 7.26851851851852E-06, 0, 0, 0, 0, 0, 0, 0, 0, 2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-06, 0, 0, 0, 0, 0, 0, 0, 0, 1.09027777777778E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0;
    //Expected_MassMatrix.row(15) << 3.63425925925926E-06, 0, 0, 0, 0, 0, 2.72569444444444E-11, 0, 0, 1.09027777777778E-05, 0, 0, 0, 0, 0, 8.17708333333333E-11, 0, 0, 3.63425925925926E-06, 0, 0, 0, 0, 0, 2.72569444444444E-11, 0, 0, 5.45138888888889E-06, 0, 0, 0, 0, 0, 4.08854166666667E-11, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0;
    //Expected_MassMatrix.row(16) << 0, 3.63425925925926E-06, 0, 0, 0, 0, 0, 2.72569444444444E-11, 0, 0, 1.09027777777778E-05, 0, 0, 0, 0, 0, 8.17708333333333E-11, 0, 0, 3.63425925925926E-06, 0, 0, 0, 0, 0, 2.72569444444444E-11, 0, 0, 5.45138888888889E-06, 0, 0, 0, 0, 0, 4.08854166666667E-11, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0;
    //Expected_MassMatrix.row(17) << 0, 0, 3.63425925925926E-06, 0, 0, 0, 0, 0, 2.72569444444444E-11, 0, 0, 1.09027777777778E-05, 0, 0, 0, 0, 0, 8.17708333333333E-11, 0, 0, 3.63425925925926E-06, 0, 0, 0, 0, 0, 2.72569444444444E-11, 0, 0, 5.45138888888889E-06, 0, 0, 0, 0, 0, 4.08854166666667E-11, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10;
    //Expected_MassMatrix.row(18) << 1.30833333333333, 0, 0, 0, 0, 0, 5.45138888888889E-06, 0, 0, 0.872222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-06, 0, 0, 2.61666666666667, 0, 0, 0, 0, 0, 1.09027777777778E-05, 0, 0, 0.872222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-06, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0;
    //Expected_MassMatrix.row(19) << 0, 1.30833333333333, 0, 0, 0, 0, 0, 5.45138888888889E-06, 0, 0, 0.872222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-06, 0, 0, 2.61666666666667, 0, 0, 0, 0, 0, 1.09027777777778E-05, 0, 0, 0.872222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-06, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0;
    //Expected_MassMatrix.row(20) << 0, 0, 1.30833333333333, 0, 0, 0, 0, 0, 5.45138888888889E-06, 0, 0, 0.872222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-06, 0, 0, 2.61666666666667, 0, 0, 0, 0, 0, 1.09027777777778E-05, 0, 0, 0.872222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-06, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037;
    //Expected_MassMatrix.row(21) << 0, 0, 0, 1.09027777777778E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-06, 0, 0, 0, 0, 0, 0, 0, 0, 2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-06, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0;
    //Expected_MassMatrix.row(22) << 0, 0, 0, 0, 1.09027777777778E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-06, 0, 0, 0, 0, 0, 0, 0, 0, 2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-06, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0;
    //Expected_MassMatrix.row(23) << 0, 0, 0, 0, 0, 1.09027777777778E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-06, 0, 0, 0, 0, 0, 0, 0, 0, 2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-06, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0;
    //Expected_MassMatrix.row(24) << 5.45138888888889E-06, 0, 0, 0, 0, 0, 4.08854166666667E-11, 0, 0, 3.63425925925926E-06, 0, 0, 0, 0, 0, 2.72569444444444E-11, 0, 0, 1.09027777777778E-05, 0, 0, 0, 0, 0, 8.17708333333333E-11, 0, 0, 3.63425925925926E-06, 0, 0, 0, 0, 0, 2.72569444444444E-11, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0;
    //Expected_MassMatrix.row(25) << 0, 5.45138888888889E-06, 0, 0, 0, 0, 0, 4.08854166666667E-11, 0, 0, 3.63425925925926E-06, 0, 0, 0, 0, 0, 2.72569444444444E-11, 0, 0, 1.09027777777778E-05, 0, 0, 0, 0, 0, 8.17708333333333E-11, 0, 0, 3.63425925925926E-06, 0, 0, 0, 0, 0, 2.72569444444444E-11, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0;
    //Expected_MassMatrix.row(26) << 0, 0, 5.45138888888889E-06, 0, 0, 0, 0, 0, 4.08854166666667E-11, 0, 0, 3.63425925925926E-06, 0, 0, 0, 0, 0, 2.72569444444444E-11, 0, 0, 1.09027777777778E-05, 0, 0, 0, 0, 0, 8.17708333333333E-11, 0, 0, 3.63425925925926E-06, 0, 0, 0, 0, 0, 2.72569444444444E-11, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10;
    //Expected_MassMatrix.row(27) << 0.872222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-06, 0, 0, 1.30833333333333, 0, 0, 0, 0, 0, 5.45138888888889E-06, 0, 0, 0.872222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-06, 0, 0, 2.61666666666667, 0, 0, 0, 0, 0, 1.09027777777778E-05, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0;
    //Expected_MassMatrix.row(28) << 0, 0.872222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-06, 0, 0, 1.30833333333333, 0, 0, 0, 0, 0, 5.45138888888889E-06, 0, 0, 0.872222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-06, 0, 0, 2.61666666666667, 0, 0, 0, 0, 0, 1.09027777777778E-05, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0;
    //Expected_MassMatrix.row(29) << 0, 0, 0.872222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-06, 0, 0, 1.30833333333333, 0, 0, 0, 0, 0, 5.45138888888889E-06, 0, 0, 0.872222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-06, 0, 0, 2.61666666666667, 0, 0, 0, 0, 0, 1.09027777777778E-05, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05;
    //Expected_MassMatrix.row(30) << 0, 0, 0, 7.26851851851852E-06, 0, 0, 0, 0, 0, 0, 0, 0, 1.09027777777778E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-06, 0, 0, 0, 0, 0, 0, 0, 0, 2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0;
    //Expected_MassMatrix.row(31) << 0, 0, 0, 0, 7.26851851851852E-06, 0, 0, 0, 0, 0, 0, 0, 0, 1.09027777777778E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-06, 0, 0, 0, 0, 0, 0, 0, 0, 2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0;
    //Expected_MassMatrix.row(32) << 0, 0, 0, 0, 0, 7.26851851851852E-06, 0, 0, 0, 0, 0, 0, 0, 0, 1.09027777777778E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-06, 0, 0, 0, 0, 0, 0, 0, 0, 2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0;
    //Expected_MassMatrix.row(33) << 3.63425925925926E-06, 0, 0, 0, 0, 0, 2.72569444444444E-11, 0, 0, 5.45138888888889E-06, 0, 0, 0, 0, 0, 4.08854166666667E-11, 0, 0, 3.63425925925926E-06, 0, 0, 0, 0, 0, 2.72569444444444E-11, 0, 0, 1.09027777777778E-05, 0, 0, 0, 0, 0, 8.17708333333333E-11, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0;
    //Expected_MassMatrix.row(34) << 0, 3.63425925925926E-06, 0, 0, 0, 0, 0, 2.72569444444444E-11, 0, 0, 5.45138888888889E-06, 0, 0, 0, 0, 0, 4.08854166666667E-11, 0, 0, 3.63425925925926E-06, 0, 0, 0, 0, 0, 2.72569444444444E-11, 0, 0, 1.09027777777778E-05, 0, 0, 0, 0, 0, 8.17708333333333E-11, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0;
    //Expected_MassMatrix.row(35) << 0, 0, 3.63425925925926E-06, 0, 0, 0, 0, 0, 2.72569444444444E-11, 0, 0, 5.45138888888889E-06, 0, 0, 0, 0, 0, 4.08854166666667E-11, 0, 0, 3.63425925925926E-06, 0, 0, 0, 0, 0, 2.72569444444444E-11, 0, 0, 1.09027777777778E-05, 0, 0, 0, 0, 0, 8.17708333333333E-11, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11;
    //Expected_MassMatrix.row(36) << -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, 13.9555555555556, 0, 0, 0, 0, 0, 5.81481481481482E-05, 0, 0, 8.72222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 6.97777777777778, 0, 0, 0, 0, 0, 2.90740740740741E-05, 0, 0, 8.72222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0;
    //Expected_MassMatrix.row(37) << 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, 13.9555555555556, 0, 0, 0, 0, 0, 5.81481481481482E-05, 0, 0, 8.72222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 6.97777777777778, 0, 0, 0, 0, 0, 2.90740740740741E-05, 0, 0, 8.72222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0;
    //Expected_MassMatrix.row(38) << 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, 13.9555555555556, 0, 0, 0, 0, 0, 5.81481481481482E-05, 0, 0, 8.72222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 6.97777777777778, 0, 0, 0, 0, 0, 2.90740740740741E-05, 0, 0, 8.72222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-05;
    //Expected_MassMatrix.row(39) << 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0.000116296296296296, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 5.81481481481482E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0;
    //Expected_MassMatrix.row(40) << 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0.000116296296296296, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 5.81481481481482E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0;
    //Expected_MassMatrix.row(41) << 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0.000116296296296296, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 5.81481481481482E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0;
    //Expected_MassMatrix.row(42) << -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, 5.81481481481482E-05, 0, 0, 0, 0, 0, 4.36111111111111E-10, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 2.72569444444444E-10, 0, 0, 2.90740740740741E-05, 0, 0, 0, 0, 0, 2.18055555555556E-10, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 2.72569444444444E-10, 0, 0;
    //Expected_MassMatrix.row(43) << 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, 5.81481481481482E-05, 0, 0, 0, 0, 0, 4.36111111111111E-10, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 2.72569444444444E-10, 0, 0, 2.90740740740741E-05, 0, 0, 0, 0, 0, 2.18055555555556E-10, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 2.72569444444444E-10, 0;
    //Expected_MassMatrix.row(44) << 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, 5.81481481481482E-05, 0, 0, 0, 0, 0, 4.36111111111111E-10, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 2.72569444444444E-10, 0, 0, 2.90740740740741E-05, 0, 0, 0, 0, 0, 2.18055555555556E-10, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 2.72569444444444E-10;
    //Expected_MassMatrix.row(45) << -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, 8.72222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 13.9555555555556, 0, 0, 0, 0, 0, 5.81481481481482E-05, 0, 0, 8.72222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 6.97777777777778, 0, 0, 0, 0, 0, 2.90740740740741E-05, 0, 0;
    //Expected_MassMatrix.row(46) << 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, 8.72222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 13.9555555555556, 0, 0, 0, 0, 0, 5.81481481481482E-05, 0, 0, 8.72222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 6.97777777777778, 0, 0, 0, 0, 0, 2.90740740740741E-05, 0;
    //Expected_MassMatrix.row(47) << 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, 8.72222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 13.9555555555556, 0, 0, 0, 0, 0, 5.81481481481482E-05, 0, 0, 8.72222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 6.97777777777778, 0, 0, 0, 0, 0, 2.90740740740741E-05;
    //Expected_MassMatrix.row(48) << 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0.000116296296296296, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 5.81481481481482E-05, 0, 0, 0, 0, 0;
    //Expected_MassMatrix.row(49) << 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0.000116296296296296, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 5.81481481481482E-05, 0, 0, 0, 0;
    //Expected_MassMatrix.row(50) << 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0.000116296296296296, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 5.81481481481482E-05, 0, 0, 0;
    //Expected_MassMatrix.row(51) << -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 2.72569444444444E-10, 0, 0, 5.81481481481482E-05, 0, 0, 0, 0, 0, 4.36111111111111E-10, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 2.72569444444444E-10, 0, 0, 2.90740740740741E-05, 0, 0, 0, 0, 0, 2.18055555555556E-10, 0, 0;
    //Expected_MassMatrix.row(52) << 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 2.72569444444444E-10, 0, 0, 5.81481481481482E-05, 0, 0, 0, 0, 0, 4.36111111111111E-10, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 2.72569444444444E-10, 0, 0, 2.90740740740741E-05, 0, 0, 0, 0, 0, 2.18055555555556E-10, 0;
    //Expected_MassMatrix.row(53) << 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 2.72569444444444E-10, 0, 0, 5.81481481481482E-05, 0, 0, 0, 0, 0, 4.36111111111111E-10, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 2.72569444444444E-10, 0, 0, 2.90740740740741E-05, 0, 0, 0, 0, 0, 2.18055555555556E-10;
    //Expected_MassMatrix.row(54) << -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, 6.97777777777778, 0, 0, 0, 0, 0, 2.90740740740741E-05, 0, 0, 8.72222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 13.9555555555556, 0, 0, 0, 0, 0, 5.81481481481482E-05, 0, 0, 8.72222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0;
    //Expected_MassMatrix.row(55) << 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, 6.97777777777778, 0, 0, 0, 0, 0, 2.90740740740741E-05, 0, 0, 8.72222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 13.9555555555556, 0, 0, 0, 0, 0, 5.81481481481482E-05, 0, 0, 8.72222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0;
    //Expected_MassMatrix.row(56) << 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, 6.97777777777778, 0, 0, 0, 0, 0, 2.90740740740741E-05, 0, 0, 8.72222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 13.9555555555556, 0, 0, 0, 0, 0, 5.81481481481482E-05, 0, 0, 8.72222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-05;
    //Expected_MassMatrix.row(57) << 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, 5.81481481481482E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0.000116296296296296, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0;
    //Expected_MassMatrix.row(58) << 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, 5.81481481481482E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0.000116296296296296, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0;
    //Expected_MassMatrix.row(59) << 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, 5.81481481481482E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0.000116296296296296, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0;
    //Expected_MassMatrix.row(60) << -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, 2.90740740740741E-05, 0, 0, 0, 0, 0, 2.18055555555556E-10, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 2.72569444444444E-10, 0, 0, 5.81481481481482E-05, 0, 0, 0, 0, 0, 4.36111111111111E-10, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 2.72569444444444E-10, 0, 0;
    //Expected_MassMatrix.row(61) << 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, 2.90740740740741E-05, 0, 0, 0, 0, 0, 2.18055555555556E-10, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 2.72569444444444E-10, 0, 0, 5.81481481481482E-05, 0, 0, 0, 0, 0, 4.36111111111111E-10, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 2.72569444444444E-10, 0;
    //Expected_MassMatrix.row(62) << 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, 2.90740740740741E-05, 0, 0, 0, 0, 0, 2.18055555555556E-10, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 2.72569444444444E-10, 0, 0, 5.81481481481482E-05, 0, 0, 0, 0, 0, 4.36111111111111E-10, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 2.72569444444444E-10;
    //Expected_MassMatrix.row(63) << -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, 8.72222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 6.97777777777778, 0, 0, 0, 0, 0, 2.90740740740741E-05, 0, 0, 8.72222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 13.9555555555556, 0, 0, 0, 0, 0, 5.81481481481482E-05, 0, 0;
    //Expected_MassMatrix.row(64) << 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, 8.72222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 6.97777777777778, 0, 0, 0, 0, 0, 2.90740740740741E-05, 0, 0, 8.72222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 13.9555555555556, 0, 0, 0, 0, 0, 5.81481481481482E-05, 0;
    //Expected_MassMatrix.row(65) << 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -3.48888888888889, 0, 0, 0, 0, 0, -0.000014537037037037, 0, 0, -2.61666666666667, 0, 0, 0, 0, 0, -1.09027777777778E-05, 0, 0, 8.72222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 6.97777777777778, 0, 0, 0, 0, 0, 2.90740740740741E-05, 0, 0, 8.72222222222222, 0, 0, 0, 0, 0, 3.63425925925926E-05, 0, 0, 13.9555555555556, 0, 0, 0, 0, 0, 5.81481481481482E-05;
    //Expected_MassMatrix.row(66) << 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 5.81481481481482E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0.000116296296296296, 0, 0, 0, 0, 0;
    //Expected_MassMatrix.row(67) << 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 5.81481481481482E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0.000116296296296296, 0, 0, 0, 0;
    //Expected_MassMatrix.row(68) << 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.90740740740741E-05, 0, 0, 0, 0, 0, 0, 0, 0, -2.18055555555556E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 5.81481481481482E-05, 0, 0, 0, 0, 0, 0, 0, 0, 7.26851851851852E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0.000116296296296296, 0, 0, 0;
    //Expected_MassMatrix.row(69) << -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 2.72569444444444E-10, 0, 0, 2.90740740740741E-05, 0, 0, 0, 0, 0, 2.18055555555556E-10, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 2.72569444444444E-10, 0, 0, 5.81481481481482E-05, 0, 0, 0, 0, 0, 4.36111111111111E-10, 0, 0;
    //Expected_MassMatrix.row(70) << 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 2.72569444444444E-10, 0, 0, 2.90740740740741E-05, 0, 0, 0, 0, 0, 2.18055555555556E-10, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 2.72569444444444E-10, 0, 0, 5.81481481481482E-05, 0, 0, 0, 0, 0, 4.36111111111111E-10, 0;
    //Expected_MassMatrix.row(71) << 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -0.000014537037037037, 0, 0, 0, 0, 0, -1.09027777777778E-10, 0, 0, -1.09027777777778E-05, 0, 0, 0, 0, 0, -8.17708333333333E-11, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 2.72569444444444E-10, 0, 0, 2.90740740740741E-05, 0, 0, 0, 0, 0, 2.18055555555556E-10, 0, 0, 3.63425925925926E-05, 0, 0, 0, 0, 0, 2.72569444444444E-10, 0, 0, 5.81481481481482E-05, 0, 0, 0, 0, 0, 4.36111111111111E-10;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_MassMatrix_Compact;
    Expected_MassMatrix_Compact.resize(24, 24);
    if (!load_validation_data("UT_ANCFShell_3833_MassMatrix.txt", Expected_MassMatrix_Compact))
        return false;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_MassMatrix;
    Expected_MassMatrix.resize(3 * Expected_MassMatrix_Compact.rows(), 3 * Expected_MassMatrix_Compact.cols());
    Expected_MassMatrix.setZero();
    for (unsigned int r = 0; r < Expected_MassMatrix_Compact.rows(); r++) {
        for (unsigned int c = 0; c < Expected_MassMatrix_Compact.cols(); c++) {
            Expected_MassMatrix(3 * r, 3 * c) = Expected_MassMatrix_Compact(r, c);
            Expected_MassMatrix(3 * r + 1, 3 * c + 1) = Expected_MassMatrix_Compact(r, c);
            Expected_MassMatrix(3 * r + 2, 3 * c + 2) = Expected_MassMatrix_Compact(r, c);
        }
    }

    ChMatrixDynamic<double> MassMatrix;
    MassMatrix.resize(72, 72);
    m_element->ComputeKRMmatricesGlobal(MassMatrix, 0, 0, 1);
    ChMatrixNM<double, 24, 24> MassMatrix_compact;
    for (unsigned int i = 0; i < 24; i++) {
        for (unsigned int j = 0; j < 24; j++) {
            MassMatrix_compact(i, j) = MassMatrix(3 * i, 3 * j);
        }
    }

    double MaxAbsError = (MassMatrix - Expected_MassMatrix).cwiseAbs().maxCoeff();
    bool passed_test = (MaxAbsError <= 0.001);

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
    //ChVectorN<double, 72> Expected_InternalForceDueToGravity;
    //Expected_InternalForceDueToGravity <<
    //    0,
    //    0,
    //    64.1518354166667,
    //    0,
    //    0,
    //    0,
    //    0,
    //    0,
    //    0.000267299314236111,
    //    0,
    //    0,
    //    64.1518354166667,
    //    0,
    //    0,
    //    0,
    //    0,
    //    0,
    //    0.000267299314236111,
    //    0,
    //    0,
    //    64.1518354166667,
    //    0,
    //    0,
    //    0,
    //    0,
    //    0,
    //    0.000267299314236111,
    //    0,
    //    0,
    //    64.1518354166667,
    //    0,
    //    0,
    //    0,
    //    0,
    //    0,
    //    0.000267299314236111,
    //    0,
    //    0,
    //    -256.607341666667,
    //    0,
    //    0,
    //    0,
    //    0,
    //    0,
    //    -0.00106919725694444,
    //    0,
    //    0,
    //    -256.607341666667,
    //    0,
    //    0,
    //    0,
    //    0,
    //    0,
    //    -0.00106919725694444,
    //    0,
    //    0,
    //    -256.607341666667,
    //    0,
    //    0,
    //    0,
    //    0,
    //    0,
    //    -0.00106919725694444,
    //    0,
    //    0,
    //    -256.607341666667,
    //    0,
    //    0,
    //    0,
    //    0,
    //    0,
    //    -0.00106919725694444;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_InternalForceDueToGravity;
    Expected_InternalForceDueToGravity.resize(72, 1);
    if (!load_validation_data("UT_ANCFShell_3833_Grav.txt", Expected_InternalForceDueToGravity))
        return false;

    ChVectorDynamic<double> InternalForceNoDispNoVelWithGravity;
    InternalForceNoDispNoVelWithGravity.resize(72);
    m_element->SetGravityOn(true);
    m_element->ComputeInternalForces(InternalForceNoDispNoVelWithGravity);

    ChVectorDynamic<double> InternalForceNoDispNoVelNoGravity;
    InternalForceNoDispNoVelNoGravity.resize(72);
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
    InternalForceNoDispNoVelNoGravity.resize(72);
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

    //ChVectorN<double, 72> Expected_InternalForceSmallDispNoVelNoGravity;
    //Expected_InternalForceSmallDispNoVelNoGravity <<
    //    -342.147435897436,
    //    324.198717948718,
    //    -403847.98125,
    //    35897.4358974359,
    //    -13461.5384615385,
    //    127.884615384615,
    //    -0.00142561431623932,
    //    0.00135082799145299,
    //    -1.682699921875,
    //    -2062.98076923077,
    //    2062.98076923077,
    //    -933341.907435898,
    //    -53846.1538461539,
    //    53846.1538461539,
    //    -335.416666666667,
    //    -0.00859575320512821,
    //    0.00859575320512821,
    //    -3.88892461431624,
    //    -324.198717948718,
    //    342.147435897436,
    //    -403847.98125,
    //    13461.5384615385,
    //    -35897.4358974359,
    //    127.884615384615,
    //    -0.00135082799145299,
    //    0.00142561431623932,
    //    -1.682699921875,
    //    -259.134615384615,
    //    259.134615384615,
    //    -412821.880064103,
    //    13461.5384615385,
    //    -13461.5384615385,
    //    88.6217948717949,
    //    -0.00107972756410256,
    //    0.00107972756410256,
    //    -1.72009116693376,
    //    1548.07692307692,
    //    -201.923076923077,
    //    664107.503589744,
    //    -89743.5897435898,
    //    116666.666666667,
    //    -583.333333333333,
    //    0.00645032051282051,
    //    -0.000841346153846154,
    //    2.7671145982906,
    //    201.923076923077,
    //    -1548.07692307692,
    //    664107.503589744,
    //    -116666.666666667,
    //    89743.5897435898,
    //    -583.333333333333,
    //    0.000841346153846154,
    //    -0.00645032051282051,
    //    2.7671145982906,
    //    489.102564102564,
    //    -749.358974358974,
    //    412822.371410256,
    //    0,
    //    -62820.5128205128,
    //    -237.820512820513,
    //    0.00203792735042735,
    //    -0.00312232905982906,
    //    1.7200932142094,
    //    749.358974358974,
    //    -489.102564102564,
    //    412822.371410256,
    //    62820.5128205128,
    //    0,
    //    -237.820512820513,
    //    0.00312232905982906,
    //    -0.00203792735042735,
    //    1.7200932142094;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_InternalForceSmallDispNoVelNoGravity;
    Expected_InternalForceSmallDispNoVelNoGravity.resize(72, 1);
    if (!load_validation_data("UT_ANCFShell_3833_IntFrcSmallDispNoVelNoGravity.txt", Expected_InternalForceSmallDispNoVelNoGravity))
        return false;

    //Setup the test conditions
    ChVector<double> OriginalPos = m_nodeB->GetPos();
    m_nodeB->SetPos(ChVector<>(m_nodeB->GetPos().x(), m_nodeB->GetPos().y(), 0.001));

    ChVectorDynamic<double> InternalForceSmallDispNoVelNoGravity;
    InternalForceSmallDispNoVelNoGravity.resize(72);
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

    //ChVectorN<double, 72> Expected_InternalForceNoDispSmallVelNoGravity;
    //Expected_InternalForceNoDispSmallVelNoGravity <<
    //    0,
    //    0,
    //    -4038.46153846154,
    //    358.974358974359,
    //    -134.615384615385,
    //    0,
    //    0,
    //    0,
    //    -0.0168269230769231,
    //    0,
    //    0,
    //    -9333.33333333333,
    //    -538.461538461539,
    //    538.461538461539,
    //    0,
    //    0,
    //    0,
    //    -0.0388888888888889,
    //    0,
    //    0,
    //    -4038.46153846154,
    //    134.615384615385,
    //    -358.974358974359,
    //    0,
    //    0,
    //    0,
    //    -0.0168269230769231,
    //    0,
    //    0,
    //    -4128.20512820513,
    //    134.615384615385,
    //    -134.615384615385,
    //    0,
    //    0,
    //    0,
    //    -0.0172008547008547,
    //    0,
    //    0,
    //    6641.02564102564,
    //    -897.435897435898,
    //    1166.66666666667,
    //    0,
    //    0,
    //    0,
    //    0.0276709401709402,
    //    0,
    //    0,
    //    6641.02564102564,
    //    -1166.66666666667,
    //    897.435897435898,
    //    0,
    //    0,
    //    0,
    //    0.0276709401709402,
    //    0,
    //    0,
    //    4128.20512820513,
    //    0,
    //    -628.205128205128,
    //    0,
    //    0,
    //    0,
    //    0.0172008547008547,
    //    0,
    //    0,
    //    4128.20512820513,
    //    628.205128205128,
    //    0,
    //    0,
    //    0,
    //    0,
    //    0.0172008547008547;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_InternalForceNoDispSmallVelNoGravity;
    Expected_InternalForceNoDispSmallVelNoGravity.resize(72, 1);
    if (!load_validation_data("UT_ANCFShell_3833_IntFrcNoDispSmallVelNoGravity.txt", Expected_InternalForceNoDispSmallVelNoGravity))
        return false;

    //Setup the test conditions
    ChVector<double> OriginalVel = m_nodeB->GetPos_dt();
    m_nodeB->SetPos_dt(ChVector<>(0.0, 0.0, 0.001));
    m_element->SetAlphaDamp(0.01);

    ChVectorDynamic<double> InternalForceNoDispSmallVelNoGravity;
    InternalForceNoDispSmallVelNoGravity.resize(72);
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
    //ChMatrixNM<double, 72, 72> Expected_JacobianK_NoDispNoVelNoDamping;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(0) << 2100000000, 953525641.025641, 0, 0, 0, -80769230.7692308, 8750, 3973.0235042735, 0, 1032051282.05128, -33653846.1538461, 0, 0, 0, 53846153.8461538, 4300.21367521368, -140.224358974359, 0, 928846153.846154, 392628205.128205, 0, 0, 0, 20192307.6923077, 3870.19230769231, 1635.95085470085, 0, 785256410.25641, 33653846.1538461, 0, 0, 0, 20192307.6923077, 3271.90170940171, 140.224358974359, 0, -2458974358.97436, -314102564.102564, 0, 0, 0, -134615384.615385, -10245.7264957264, -1308.76068376068, 0, -547435897.435898, -224358974.358974, 0, 0, 0, 94230769.2307692, -2280.98290598291, -934.82905982906, 0, -1310256410.25641, -224358974.358974, 0, 0, 0, 0, -5459.40170940171, -934.82905982906, 0, -529487179.48718, -583333333.333333, 0, 0, 0, -175000000, -2206.19658119658, -2430.55555555556, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(1) << 953525641.025641, 2100000000, 0, 0, 0, -80769230.7692308, 3973.0235042735, 8750, 0, 33653846.1538461, 785256410.25641, 0, 0, 0, 20192307.6923077, 140.224358974359, 3271.90170940171, 0, 392628205.128205, 928846153.846154, 0, 0, 0, 20192307.6923077, 1635.95085470085, 3870.19230769231, 0, -33653846.1538461, 1032051282.05128, 0, 0, 0, 53846153.8461538, -140.224358974359, 4300.21367521368, 0, -583333333.333333, -529487179.48718, 0, 0, 0, -175000000, -2430.55555555556, -2206.19658119658, 0, -224358974.358974, -1310256410.25641, 0, 0, 0, 0, -934.82905982906, -5459.40170940171, 0, -224358974.358974, -547435897.435898, 0, 0, 0, 94230769.2307692, -934.82905982906, -2280.98290598291, 0, -314102564.102564, -2458974358.97436, 0, 0, 0, -134615384.615385, -1308.76068376068, -10245.7264957264, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(2) << 0, 0, 933333333.333333, -53846153.8461539, -53846153.8461539, 0, 0, 0, 3888.88888888889, 0, 0, 403846153.846154, 35897435.8974359, 13461538.4615385, 0, 0, 0, 1682.69230769231, 0, 0, 412820512.820513, 13461538.4615385, 13461538.4615385, 0, 0, 0, 1720.08547008547, 0, 0, 403846153.846154, 13461538.4615385, 35897435.8974359, 0, 0, 0, 1682.69230769231, 0, 0, -664102564.102564, -89743589.7435897, -116666666.666667, 0, 0, 0, -2767.09401709402, 0, 0, -412820512.820513, 62820512.8205128, 0, 0, 0, 0, -1720.08547008547, 0, 0, -412820512.820513, 0, 62820512.8205128, 0, 0, 0, -1720.08547008547, 0, 0, -664102564.102564, -116666666.666667, -89743589.7435897, 0, 0, 0, -2767.09401709402;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(3) << 0, 0, -53846153.8461539, 26940576.9230769, 7946.04700854701, 0, 0, 0, -897.435897435898, 0, 0, -35897435.8974359, 8982959.4017094, -280.448717948718, 0, 0, 0, 299.145299145299, 0, 0, -13461538.4615385, 13469278.8461538, 3271.90170940171, 0, 0, 0, 112.179487179487, 0, 0, 13461538.4615385, 8980902.77777778, 280.448717948718, 0, 0, 0, 224.358974358974, 0, 0, 89743589.7435897, -26943568.3760684, -2617.52136752137, 0, 0, 0, -747.863247863248, 0, 0, -62820512.8205128, -35901997.8632479, -1869.65811965812, 0, 0, 0, 523.504273504273, 0, 0, 0, -35908354.7008547, -1869.65811965812, 0, 0, 0, 0, 0, 0, 62820512.8205128, -26927489.3162393, -4861.11111111111, 0, 0, 0, -1196.5811965812;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(4) << 0, 0, -53846153.8461539, 7946.04700854701, 26940576.9230769, 0, 0, 0, -897.435897435898, 0, 0, 13461538.4615385, 280.448717948718, 8980902.77777778, 0, 0, 0, 224.358974358974, 0, 0, -13461538.4615385, 3271.90170940171, 13469278.8461538, 0, 0, 0, 112.179487179487, 0, 0, -35897435.8974359, -280.448717948718, 8982959.4017094, 0, 0, 0, 299.145299145299, 0, 0, 62820512.8205128, -4861.11111111111, -26927489.3162393, 0, 0, 0, -1196.5811965812, 0, 0, 0, -1869.65811965812, -35908354.7008547, 0, 0, 0, 0, 0, 0, -62820512.8205128, -1869.65811965812, -35901997.8632479, 0, 0, 0, 523.504273504273, 0, 0, 89743589.7435897, -2617.52136752137, -26943568.3760684, 0, 0, 0, -747.863247863248;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(5) << -80769230.7692308, -80769230.7692308, 0, 0, 0, 94238547.008547, -785.25641025641, -785.25641025641, 0, -53846153.8461538, 20192307.6923077, 0, 0, 0, 31413621.7948718, 74.7863247863248, 196.314102564103, 0, -20192307.6923077, -20192307.6923077, 0, 0, 0, 47118824.7863248, 28.0448717948718, 28.0448717948718, 0, 20192307.6923077, -53846153.8461538, 0, 0, 0, 31413621.7948718, 196.314102564103, 74.7863247863248, 0, 134615384.615385, 94230769.2307692, 0, 0, 0, -94236303.4188034, -186.965811965812, -579.594017094017, 0, -94230769.2307692, 0, 0, 0, 0, -125644465.811966, 130.876068376068, 0, 0, 0, -94230769.2307692, 0, 0, 0, -125644465.811966, 0, 130.876068376068, 0, 94230769.2307692, 134615384.615385, 0, 0, 0, -94236303.4188034, -579.594017094017, -186.965811965812, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(6) << 8750, 3973.0235042735, 0, 0, 0, -785.25641025641, 224.424599358974, 0.0297976762820513, 0, 4300.21367521368, -140.224358974359, 0, 0, 0, -74.7863247863248, 74.8185763888889, -0.00105168269230769, 0, 3870.19230769231, 1635.95085470085, 0, 0, 0, -28.0448717948718, 112.208513621795, 0.0122696314102564, 0, 3271.90170940171, 140.224358974359, 0, 0, 0, 196.314102564103, 74.8108640491453, 0.00105168269230769, 0, -10245.7264957264, -1308.76068376068, 0, 0, 0, 186.965811965812, -224.435817307692, -0.00981570512820513, 0, -2280.98290598291, -934.82905982906, 0, 0, 0, -130.876068376068, -299.162406517094, -0.00701121794871795, 0, -5459.40170940171, -934.82905982906, 0, 0, 0, 0, -299.18624465812, -0.00701121794871795, 0, -2206.19658119658, -2430.55555555556, 0, 0, 0, -205.662393162393, -224.375520833333, -0.0182291666666667, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(7) << 3973.0235042735, 8750, 0, 0, 0, -785.25641025641, 0.0297976762820513, 224.424599358974, 0, 140.224358974359, 3271.90170940171, 0, 0, 0, 196.314102564103, 0.00105168269230769, 74.8108640491453, 0, 1635.95085470085, 3870.19230769231, 0, 0, 0, -28.0448717948718, 0.0122696314102564, 112.208513621795, 0, -140.224358974359, 4300.21367521368, 0, 0, 0, -74.7863247863248, -0.00105168269230769, 74.8185763888889, 0, -2430.55555555556, -2206.19658119658, 0, 0, 0, -205.662393162393, -0.0182291666666667, -224.375520833333, 0, -934.82905982906, -5459.40170940171, 0, 0, 0, 0, -0.00701121794871795, -299.18624465812, 0, -934.82905982906, -2280.98290598291, 0, 0, 0, -130.876068376068, -0.00701121794871795, -299.162406517094, 0, -1308.76068376068, -10245.7264957264, 0, 0, 0, 186.965811965812, -0.00981570512820513, -224.435817307692, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(8) << 0, 0, 3888.88888888889, -897.435897435898, -897.435897435898, 0, 0, 0, 785.285576923077, 0, 0, 1682.69230769231, -299.145299145299, 224.358974358974, 0, 0, 0, 261.764756944444, 0, 0, 1720.08547008547, -112.179487179487, -112.179487179487, 0, 0, 0, 392.641105769231, 0, 0, 1682.69230769231, 224.358974358974, -299.145299145299, 0, 0, 0, 261.764756944444, 0, 0, -2767.09401709402, 747.863247863248, 299.145299145299, 0, 0, 0, -785.277163461538, 0, 0, -1720.08547008547, -523.504273504273, 0, 0, 0, 0, -1047.02144764957, 0, 0, -1720.08547008547, 0, -523.504273504273, 0, 0, 0, -1047.02144764957, 0, 0, -2767.09401709402, 299.145299145299, 747.863247863248, 0, 0, 0, -785.277163461538;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(9) << 1032051282.05128, 33653846.1538461, 0, 0, 0, -53846153.8461538, 4300.21367521368, 140.224358974359, 0, 2100000000, -953525641.025641, 0, 0, 0, 80769230.7692308, 8750, -3973.0235042735, 0, 785256410.25641, -33653846.1538461, 0, 0, 0, -20192307.6923077, 3271.90170940171, -140.224358974359, 0, 928846153.846154, -392628205.128205, 0, 0, 0, -20192307.6923077, 3870.19230769231, -1635.95085470085, 0, -2458974358.97436, 314102564.102564, 0, 0, 0, 134615384.615385, -10245.7264957264, 1308.76068376068, 0, -529487179.48718, 583333333.333333, 0, 0, 0, 175000000, -2206.19658119658, 2430.55555555556, 0, -1310256410.25641, 224358974.358974, 0, 0, 0, 0, -5459.40170940171, 934.82905982906, 0, -547435897.435898, 224358974.358974, 0, 0, 0, -94230769.2307692, -2280.98290598291, 934.82905982906, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(10) << -33653846.1538461, 785256410.25641, 0, 0, 0, 20192307.6923077, -140.224358974359, 3271.90170940171, 0, -953525641.025641, 2100000000, 0, 0, 0, -80769230.7692308, -3973.0235042735, 8750, 0, 33653846.1538461, 1032051282.05128, 0, 0, 0, 53846153.8461538, 140.224358974359, 4300.21367521368, 0, -392628205.128205, 928846153.846154, 0, 0, 0, 20192307.6923077, -1635.95085470085, 3870.19230769231, 0, 583333333.333333, -529487179.48718, 0, 0, 0, -175000000, 2430.55555555556, -2206.19658119658, 0, 314102564.102564, -2458974358.97436, 0, 0, 0, -134615384.615385, 1308.76068376068, -10245.7264957264, 0, 224358974.358974, -547435897.435898, 0, 0, 0, 94230769.2307692, 934.82905982906, -2280.98290598291, 0, 224358974.358974, -1310256410.25641, 0, 0, 0, 0, 934.82905982906, -5459.40170940171, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(11) << 0, 0, 403846153.846154, -35897435.8974359, 13461538.4615385, 0, 0, 0, 1682.69230769231, 0, 0, 933333333.333333, 53846153.8461539, -53846153.8461539, 0, 0, 0, 3888.88888888889, 0, 0, 403846153.846154, -13461538.4615385, 35897435.8974359, 0, 0, 0, 1682.69230769231, 0, 0, 412820512.820513, -13461538.4615385, 13461538.4615385, 0, 0, 0, 1720.08547008547, 0, 0, -664102564.102564, 89743589.7435897, -116666666.666667, 0, 0, 0, -2767.09401709402, 0, 0, -664102564.102564, 116666666.666667, -89743589.7435897, 0, 0, 0, -2767.09401709402, 0, 0, -412820512.820513, 0, 62820512.8205128, 0, 0, 0, -1720.08547008547, 0, 0, -412820512.820513, -62820512.8205128, 0, 0, 0, 0, -1720.08547008547;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(12) << 0, 0, 35897435.8974359, 8982959.4017094, 280.448717948718, 0, 0, 0, -299.145299145299, 0, 0, 53846153.8461539, 26940576.9230769, -7946.04700854701, 0, 0, 0, 897.435897435898, 0, 0, -13461538.4615385, 8980902.77777778, -280.448717948718, 0, 0, 0, -224.358974358974, 0, 0, 13461538.4615385, 13469278.8461538, -3271.90170940171, 0, 0, 0, -112.179487179487, 0, 0, -89743589.7435897, -26943568.3760684, 2617.52136752137, 0, 0, 0, 747.863247863248, 0, 0, -62820512.8205128, -26927489.3162393, 4861.11111111111, 0, 0, 0, 1196.5811965812, 0, 0, 0, -35908354.7008547, 1869.65811965812, 0, 0, 0, 0, 0, 0, 62820512.8205128, -35901997.8632479, 1869.65811965812, 0, 0, 0, -523.504273504273;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(13) << 0, 0, 13461538.4615385, -280.448717948718, 8980902.77777778, 0, 0, 0, 224.358974358974, 0, 0, -53846153.8461539, -7946.04700854701, 26940576.9230769, 0, 0, 0, -897.435897435898, 0, 0, -35897435.8974359, 280.448717948718, 8982959.4017094, 0, 0, 0, 299.145299145299, 0, 0, -13461538.4615385, -3271.90170940171, 13469278.8461538, 0, 0, 0, 112.179487179487, 0, 0, 62820512.8205128, 4861.11111111111, -26927489.3162393, 0, 0, 0, -1196.5811965812, 0, 0, 89743589.7435897, 2617.52136752137, -26943568.3760684, 0, 0, 0, -747.863247863248, 0, 0, -62820512.8205128, 1869.65811965812, -35901997.8632479, 0, 0, 0, 523.504273504273, 0, 0, 0, 1869.65811965812, -35908354.7008547, 0, 0, 0, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(14) << 53846153.8461538, 20192307.6923077, 0, 0, 0, 31413621.7948718, -74.7863247863248, 196.314102564103, 0, 80769230.7692308, -80769230.7692308, 0, 0, 0, 94238547.008547, 785.25641025641, -785.25641025641, 0, -20192307.6923077, -53846153.8461538, 0, 0, 0, 31413621.7948718, -196.314102564103, 74.7863247863248, 0, 20192307.6923077, -20192307.6923077, 0, 0, 0, 47118824.7863248, -28.0448717948718, 28.0448717948718, 0, -134615384.615385, 94230769.2307692, 0, 0, 0, -94236303.4188034, 186.965811965812, -579.594017094017, 0, -94230769.2307692, 134615384.615385, 0, 0, 0, -94236303.4188034, 579.594017094017, -186.965811965812, 0, 0, -94230769.2307692, 0, 0, 0, -125644465.811966, 0, 130.876068376068, 0, 94230769.2307692, 0, 0, 0, 0, -125644465.811966, -130.876068376068, 0, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(15) << 4300.21367521368, 140.224358974359, 0, 0, 0, 74.7863247863248, 74.8185763888889, 0.00105168269230769, 0, 8750, -3973.0235042735, 0, 0, 0, 785.25641025641, 224.424599358974, -0.0297976762820513, 0, 3271.90170940171, -140.224358974359, 0, 0, 0, -196.314102564103, 74.8108640491453, -0.00105168269230769, 0, 3870.19230769231, -1635.95085470085, 0, 0, 0, 28.0448717948718, 112.208513621795, -0.0122696314102564, 0, -10245.7264957264, 1308.76068376068, 0, 0, 0, -186.965811965812, -224.435817307692, 0.00981570512820513, 0, -2206.19658119658, 2430.55555555556, 0, 0, 0, 205.662393162393, -224.375520833333, 0.0182291666666667, 0, -5459.40170940171, 934.82905982906, 0, 0, 0, 0, -299.18624465812, 0.00701121794871795, 0, -2280.98290598291, 934.82905982906, 0, 0, 0, 130.876068376068, -299.162406517094, 0.00701121794871795, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(16) << -140.224358974359, 3271.90170940171, 0, 0, 0, 196.314102564103, -0.00105168269230769, 74.8108640491453, 0, -3973.0235042735, 8750, 0, 0, 0, -785.25641025641, -0.0297976762820513, 224.424599358974, 0, 140.224358974359, 4300.21367521368, 0, 0, 0, -74.7863247863248, 0.00105168269230769, 74.8185763888889, 0, -1635.95085470085, 3870.19230769231, 0, 0, 0, -28.0448717948718, -0.0122696314102564, 112.208513621795, 0, 2430.55555555556, -2206.19658119658, 0, 0, 0, -205.662393162393, 0.0182291666666667, -224.375520833333, 0, 1308.76068376068, -10245.7264957264, 0, 0, 0, 186.965811965812, 0.00981570512820513, -224.435817307692, 0, 934.82905982906, -2280.98290598291, 0, 0, 0, -130.876068376068, 0.00701121794871795, -299.162406517094, 0, 934.82905982906, -5459.40170940171, 0, 0, 0, 0, 0.00701121794871795, -299.18624465812, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(17) << 0, 0, 1682.69230769231, 299.145299145299, 224.358974358974, 0, 0, 0, 261.764756944444, 0, 0, 3888.88888888889, 897.435897435898, -897.435897435898, 0, 0, 0, 785.285576923077, 0, 0, 1682.69230769231, -224.358974358974, -299.145299145299, 0, 0, 0, 261.764756944444, 0, 0, 1720.08547008547, 112.179487179487, -112.179487179487, 0, 0, 0, 392.641105769231, 0, 0, -2767.09401709402, -747.863247863248, 299.145299145299, 0, 0, 0, -785.277163461538, 0, 0, -2767.09401709402, -299.145299145299, 747.863247863248, 0, 0, 0, -785.277163461538, 0, 0, -1720.08547008547, 0, -523.504273504273, 0, 0, 0, -1047.02144764957, 0, 0, -1720.08547008547, 523.504273504273, 0, 0, 0, 0, -1047.02144764957;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(18) << 928846153.846154, 392628205.128205, 0, 0, 0, -20192307.6923077, 3870.19230769231, 1635.95085470085, 0, 785256410.25641, 33653846.1538461, 0, 0, 0, -20192307.6923077, 3271.90170940171, 140.224358974359, 0, 2100000000, 953525641.025641, 0, 0, 0, 80769230.7692308, 8750, 3973.0235042735, 0, 1032051282.05128, -33653846.1538461, 0, 0, 0, -53846153.8461538, 4300.21367521368, -140.224358974359, 0, -1310256410.25641, -224358974.358974, 0, 0, 0, 0, -5459.40170940171, -934.82905982906, 0, -529487179.48718, -583333333.333333, 0, 0, 0, 175000000, -2206.19658119658, -2430.55555555556, 0, -2458974358.97436, -314102564.102564, 0, 0, 0, 134615384.615385, -10245.7264957264, -1308.76068376068, 0, -547435897.435898, -224358974.358974, 0, 0, 0, -94230769.2307692, -2280.98290598291, -934.82905982906, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(19) << 392628205.128205, 928846153.846154, 0, 0, 0, -20192307.6923077, 1635.95085470085, 3870.19230769231, 0, -33653846.1538461, 1032051282.05128, 0, 0, 0, -53846153.8461538, -140.224358974359, 4300.21367521368, 0, 953525641.025641, 2100000000, 0, 0, 0, 80769230.7692308, 3973.0235042735, 8750, 0, 33653846.1538461, 785256410.25641, 0, 0, 0, -20192307.6923077, 140.224358974359, 3271.90170940171, 0, -224358974.358974, -547435897.435898, 0, 0, 0, -94230769.2307692, -934.82905982906, -2280.98290598291, 0, -314102564.102564, -2458974358.97436, 0, 0, 0, 134615384.615385, -1308.76068376068, -10245.7264957264, 0, -583333333.333333, -529487179.48718, 0, 0, 0, 175000000, -2430.55555555556, -2206.19658119658, 0, -224358974.358974, -1310256410.25641, 0, 0, 0, 0, -934.82905982906, -5459.40170940171, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(20) << 0, 0, 412820512.820513, -13461538.4615385, -13461538.4615385, 0, 0, 0, 1720.08547008547, 0, 0, 403846153.846154, -13461538.4615385, -35897435.8974359, 0, 0, 0, 1682.69230769231, 0, 0, 933333333.333333, 53846153.8461539, 53846153.8461539, 0, 0, 0, 3888.88888888889, 0, 0, 403846153.846154, -35897435.8974359, -13461538.4615385, 0, 0, 0, 1682.69230769231, 0, 0, -412820512.820513, 0, -62820512.8205128, 0, 0, 0, -1720.08547008547, 0, 0, -664102564.102564, 116666666.666667, 89743589.7435897, 0, 0, 0, -2767.09401709402, 0, 0, -664102564.102564, 89743589.7435897, 116666666.666667, 0, 0, 0, -2767.09401709402, 0, 0, -412820512.820513, -62820512.8205128, 0, 0, 0, 0, -1720.08547008547;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(21) << 0, 0, 13461538.4615385, 13469278.8461538, 3271.90170940171, 0, 0, 0, -112.179487179487, 0, 0, -13461538.4615385, 8980902.77777778, 280.448717948718, 0, 0, 0, -224.358974358974, 0, 0, 53846153.8461539, 26940576.9230769, 7946.04700854701, 0, 0, 0, 897.435897435898, 0, 0, 35897435.8974359, 8982959.4017094, -280.448717948718, 0, 0, 0, -299.145299145299, 0, 0, 0, -35908354.7008547, -1869.65811965812, 0, 0, 0, 0, 0, 0, -62820512.8205128, -26927489.3162393, -4861.11111111111, 0, 0, 0, 1196.5811965812, 0, 0, -89743589.7435897, -26943568.3760684, -2617.52136752137, 0, 0, 0, 747.863247863248, 0, 0, 62820512.8205128, -35901997.8632479, -1869.65811965812, 0, 0, 0, -523.504273504273;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(22) << 0, 0, 13461538.4615385, 3271.90170940171, 13469278.8461538, 0, 0, 0, -112.179487179487, 0, 0, 35897435.8974359, -280.448717948718, 8982959.4017094, 0, 0, 0, -299.145299145299, 0, 0, 53846153.8461539, 7946.04700854701, 26940576.9230769, 0, 0, 0, 897.435897435898, 0, 0, -13461538.4615385, 280.448717948718, 8980902.77777778, 0, 0, 0, -224.358974358974, 0, 0, 62820512.8205128, -1869.65811965812, -35901997.8632479, 0, 0, 0, -523.504273504273, 0, 0, -89743589.7435897, -2617.52136752137, -26943568.3760684, 0, 0, 0, 747.863247863248, 0, 0, -62820512.8205128, -4861.11111111111, -26927489.3162393, 0, 0, 0, 1196.5811965812, 0, 0, 0, -1869.65811965812, -35908354.7008547, 0, 0, 0, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(23) << 20192307.6923077, 20192307.6923077, 0, 0, 0, 47118824.7863248, -28.0448717948718, -28.0448717948718, 0, -20192307.6923077, 53846153.8461538, 0, 0, 0, 31413621.7948718, -196.314102564103, -74.7863247863248, 0, 80769230.7692308, 80769230.7692308, 0, 0, 0, 94238547.008547, 785.25641025641, 785.25641025641, 0, 53846153.8461538, -20192307.6923077, 0, 0, 0, 31413621.7948718, -74.7863247863248, -196.314102564103, 0, 0, 94230769.2307692, 0, 0, 0, -125644465.811966, 0, -130.876068376068, 0, -94230769.2307692, -134615384.615385, 0, 0, 0, -94236303.4188034, 579.594017094017, 186.965811965812, 0, -134615384.615385, -94230769.2307692, 0, 0, 0, -94236303.4188034, 186.965811965812, 579.594017094017, 0, 94230769.2307692, 0, 0, 0, 0, -125644465.811966, -130.876068376068, 0, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(24) << 3870.19230769231, 1635.95085470085, 0, 0, 0, 28.0448717948718, 112.208513621795, 0.0122696314102564, 0, 3271.90170940171, 140.224358974359, 0, 0, 0, -196.314102564103, 74.8108640491453, 0.00105168269230769, 0, 8750, 3973.0235042735, 0, 0, 0, 785.25641025641, 224.424599358974, 0.0297976762820513, 0, 4300.21367521368, -140.224358974359, 0, 0, 0, 74.7863247863248, 74.8185763888889, -0.00105168269230769, 0, -5459.40170940171, -934.82905982906, 0, 0, 0, 0, -299.18624465812, -0.00701121794871795, 0, -2206.19658119658, -2430.55555555556, 0, 0, 0, 205.662393162393, -224.375520833333, -0.0182291666666667, 0, -10245.7264957264, -1308.76068376068, 0, 0, 0, -186.965811965812, -224.435817307692, -0.00981570512820513, 0, -2280.98290598291, -934.82905982906, 0, 0, 0, 130.876068376068, -299.162406517094, -0.00701121794871795, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(25) << 1635.95085470085, 3870.19230769231, 0, 0, 0, 28.0448717948718, 0.0122696314102564, 112.208513621795, 0, -140.224358974359, 4300.21367521368, 0, 0, 0, 74.7863247863248, -0.00105168269230769, 74.8185763888889, 0, 3973.0235042735, 8750, 0, 0, 0, 785.25641025641, 0.0297976762820513, 224.424599358974, 0, 140.224358974359, 3271.90170940171, 0, 0, 0, -196.314102564103, 0.00105168269230769, 74.8108640491453, 0, -934.82905982906, -2280.98290598291, 0, 0, 0, 130.876068376068, -0.00701121794871795, -299.162406517094, 0, -1308.76068376068, -10245.7264957264, 0, 0, 0, -186.965811965812, -0.00981570512820513, -224.435817307692, 0, -2430.55555555556, -2206.19658119658, 0, 0, 0, 205.662393162393, -0.0182291666666667, -224.375520833333, 0, -934.82905982906, -5459.40170940171, 0, 0, 0, 0, -0.00701121794871795, -299.18624465812, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(26) << 0, 0, 1720.08547008547, 112.179487179487, 112.179487179487, 0, 0, 0, 392.641105769231, 0, 0, 1682.69230769231, -224.358974358974, 299.145299145299, 0, 0, 0, 261.764756944444, 0, 0, 3888.88888888889, 897.435897435898, 897.435897435898, 0, 0, 0, 785.285576923077, 0, 0, 1682.69230769231, 299.145299145299, -224.358974358974, 0, 0, 0, 261.764756944444, 0, 0, -1720.08547008547, 0, 523.504273504273, 0, 0, 0, -1047.02144764957, 0, 0, -2767.09401709402, -299.145299145299, -747.863247863248, 0, 0, 0, -785.277163461538, 0, 0, -2767.09401709402, -747.863247863248, -299.145299145299, 0, 0, 0, -785.277163461538, 0, 0, -1720.08547008547, 523.504273504273, 0, 0, 0, 0, -1047.02144764957;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(27) << 785256410.25641, -33653846.1538461, 0, 0, 0, 20192307.6923077, 3271.90170940171, -140.224358974359, 0, 928846153.846154, -392628205.128205, 0, 0, 0, 20192307.6923077, 3870.19230769231, -1635.95085470085, 0, 1032051282.05128, 33653846.1538461, 0, 0, 0, 53846153.8461538, 4300.21367521368, 140.224358974359, 0, 2100000000, -953525641.025641, 0, 0, 0, -80769230.7692308, 8750, -3973.0235042735, 0, -1310256410.25641, 224358974.358974, 0, 0, 0, 0, -5459.40170940171, 934.82905982906, 0, -547435897.435898, 224358974.358974, 0, 0, 0, 94230769.2307692, -2280.98290598291, 934.82905982906, 0, -2458974358.97436, 314102564.102564, 0, 0, 0, -134615384.615385, -10245.7264957264, 1308.76068376068, 0, -529487179.48718, 583333333.333333, 0, 0, 0, -175000000, -2206.19658119658, 2430.55555555556, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(28) << 33653846.1538461, 1032051282.05128, 0, 0, 0, -53846153.8461538, 140.224358974359, 4300.21367521368, 0, -392628205.128205, 928846153.846154, 0, 0, 0, -20192307.6923077, -1635.95085470085, 3870.19230769231, 0, -33653846.1538461, 785256410.25641, 0, 0, 0, -20192307.6923077, -140.224358974359, 3271.90170940171, 0, -953525641.025641, 2100000000, 0, 0, 0, 80769230.7692308, -3973.0235042735, 8750, 0, 224358974.358974, -547435897.435898, 0, 0, 0, -94230769.2307692, 934.82905982906, -2280.98290598291, 0, 224358974.358974, -1310256410.25641, 0, 0, 0, 0, 934.82905982906, -5459.40170940171, 0, 583333333.333333, -529487179.48718, 0, 0, 0, 175000000, 2430.55555555556, -2206.19658119658, 0, 314102564.102564, -2458974358.97436, 0, 0, 0, 134615384.615385, 1308.76068376068, -10245.7264957264, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(29) << 0, 0, 403846153.846154, 13461538.4615385, -35897435.8974359, 0, 0, 0, 1682.69230769231, 0, 0, 412820512.820513, 13461538.4615385, -13461538.4615385, 0, 0, 0, 1720.08547008547, 0, 0, 403846153.846154, 35897435.8974359, -13461538.4615385, 0, 0, 0, 1682.69230769231, 0, 0, 933333333.333333, -53846153.8461539, 53846153.8461539, 0, 0, 0, 3888.88888888889, 0, 0, -412820512.820513, 0, -62820512.8205128, 0, 0, 0, -1720.08547008547, 0, 0, -412820512.820513, 62820512.8205128, 0, 0, 0, 0, -1720.08547008547, 0, 0, -664102564.102564, -89743589.7435897, 116666666.666667, 0, 0, 0, -2767.09401709402, 0, 0, -664102564.102564, -116666666.666667, 89743589.7435897, 0, 0, 0, -2767.09401709402;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(30) << 0, 0, 13461538.4615385, 8980902.77777778, -280.448717948718, 0, 0, 0, 224.358974358974, 0, 0, -13461538.4615385, 13469278.8461538, -3271.90170940171, 0, 0, 0, 112.179487179487, 0, 0, -35897435.8974359, 8982959.4017094, 280.448717948718, 0, 0, 0, 299.145299145299, 0, 0, -53846153.8461539, 26940576.9230769, -7946.04700854701, 0, 0, 0, -897.435897435898, 0, 0, 0, -35908354.7008547, 1869.65811965812, 0, 0, 0, 0, 0, 0, -62820512.8205128, -35901997.8632479, 1869.65811965812, 0, 0, 0, 523.504273504273, 0, 0, 89743589.7435897, -26943568.3760684, 2617.52136752137, 0, 0, 0, -747.863247863248, 0, 0, 62820512.8205128, -26927489.3162393, 4861.11111111111, 0, 0, 0, -1196.5811965812;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(31) << 0, 0, 35897435.8974359, 280.448717948718, 8982959.4017094, 0, 0, 0, -299.145299145299, 0, 0, 13461538.4615385, -3271.90170940171, 13469278.8461538, 0, 0, 0, -112.179487179487, 0, 0, -13461538.4615385, -280.448717948718, 8980902.77777778, 0, 0, 0, -224.358974358974, 0, 0, 53846153.8461539, -7946.04700854701, 26940576.9230769, 0, 0, 0, 897.435897435898, 0, 0, 62820512.8205128, 1869.65811965812, -35901997.8632479, 0, 0, 0, -523.504273504273, 0, 0, 0, 1869.65811965812, -35908354.7008547, 0, 0, 0, 0, 0, 0, -62820512.8205128, 4861.11111111111, -26927489.3162393, 0, 0, 0, 1196.5811965812, 0, 0, -89743589.7435897, 2617.52136752137, -26943568.3760684, 0, 0, 0, 747.863247863248;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(32) << 20192307.6923077, 53846153.8461538, 0, 0, 0, 31413621.7948718, 196.314102564103, -74.7863247863248, 0, -20192307.6923077, 20192307.6923077, 0, 0, 0, 47118824.7863248, 28.0448717948718, -28.0448717948718, 0, -53846153.8461538, -20192307.6923077, 0, 0, 0, 31413621.7948718, 74.7863247863248, -196.314102564103, 0, -80769230.7692308, 80769230.7692308, 0, 0, 0, 94238547.008547, -785.25641025641, 785.25641025641, 0, 0, 94230769.2307692, 0, 0, 0, -125644465.811966, 0, -130.876068376068, 0, -94230769.2307692, 0, 0, 0, 0, -125644465.811966, 130.876068376068, 0, 0, 134615384.615385, -94230769.2307692, 0, 0, 0, -94236303.4188034, -186.965811965812, 579.594017094017, 0, 94230769.2307692, -134615384.615385, 0, 0, 0, -94236303.4188034, -579.594017094017, 186.965811965812, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(33) << 3271.90170940171, -140.224358974359, 0, 0, 0, 196.314102564103, 74.8108640491453, -0.00105168269230769, 0, 3870.19230769231, -1635.95085470085, 0, 0, 0, -28.0448717948718, 112.208513621795, -0.0122696314102564, 0, 4300.21367521368, 140.224358974359, 0, 0, 0, -74.7863247863248, 74.8185763888889, 0.00105168269230769, 0, 8750, -3973.0235042735, 0, 0, 0, -785.25641025641, 224.424599358974, -0.0297976762820513, 0, -5459.40170940171, 934.82905982906, 0, 0, 0, 0, -299.18624465812, 0.00701121794871795, 0, -2280.98290598291, 934.82905982906, 0, 0, 0, -130.876068376068, -299.162406517094, 0.00701121794871795, 0, -10245.7264957264, 1308.76068376068, 0, 0, 0, 186.965811965812, -224.435817307692, 0.00981570512820513, 0, -2206.19658119658, 2430.55555555556, 0, 0, 0, -205.662393162393, -224.375520833333, 0.0182291666666667, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(34) << 140.224358974359, 4300.21367521368, 0, 0, 0, 74.7863247863248, 0.00105168269230769, 74.8185763888889, 0, -1635.95085470085, 3870.19230769231, 0, 0, 0, 28.0448717948718, -0.0122696314102564, 112.208513621795, 0, -140.224358974359, 3271.90170940171, 0, 0, 0, -196.314102564103, -0.00105168269230769, 74.8108640491453, 0, -3973.0235042735, 8750, 0, 0, 0, 785.25641025641, -0.0297976762820513, 224.424599358974, 0, 934.82905982906, -2280.98290598291, 0, 0, 0, 130.876068376068, 0.00701121794871795, -299.162406517094, 0, 934.82905982906, -5459.40170940171, 0, 0, 0, 0, 0.00701121794871795, -299.18624465812, 0, 2430.55555555556, -2206.19658119658, 0, 0, 0, 205.662393162393, 0.0182291666666667, -224.375520833333, 0, 1308.76068376068, -10245.7264957264, 0, 0, 0, -186.965811965812, 0.00981570512820513, -224.435817307692, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(35) << 0, 0, 1682.69230769231, 224.358974358974, 299.145299145299, 0, 0, 0, 261.764756944444, 0, 0, 1720.08547008547, -112.179487179487, 112.179487179487, 0, 0, 0, 392.641105769231, 0, 0, 1682.69230769231, -299.145299145299, -224.358974358974, 0, 0, 0, 261.764756944444, 0, 0, 3888.88888888889, -897.435897435898, 897.435897435898, 0, 0, 0, 785.285576923077, 0, 0, -1720.08547008547, 0, 523.504273504273, 0, 0, 0, -1047.02144764957, 0, 0, -1720.08547008547, -523.504273504273, 0, 0, 0, 0, -1047.02144764957, 0, 0, -2767.09401709402, 747.863247863248, -299.145299145299, 0, 0, 0, -785.277163461538, 0, 0, -2767.09401709402, 299.145299145299, -747.863247863248, 0, 0, 0, -785.277163461538;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(36) << -2458974358.97436, -583333333.333333, 0, 0, 0, 134615384.615385, -10245.7264957264, -2430.55555555556, 0, -2458974358.97436, 583333333.333333, 0, 0, 0, -134615384.615385, -10245.7264957264, 2430.55555555556, 0, -1310256410.25641, -224358974.358974, 0, 0, 0, 0, -5459.40170940171, -934.82905982906, 0, -1310256410.25641, 224358974.358974, 0, 0, 0, 0, -5459.40170940171, 934.82905982906, 0, 5456410256.41026, 0, 0, 0, 0, 0, 22735.0427350427, 0, 0, 0, -897435897.435898, 0, 0, 0, -269230769.230769, 0, -3739.31623931624, 0, 2082051282.05128, 0, 0, 0, 0, 0, 8675.21367521367, 0, 0, 0, 897435897.435898, 0, 0, 0, 269230769.230769, 0, 3739.31623931624, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(37) << -314102564.102564, -529487179.48718, 0, 0, 0, 94230769.2307692, -1308.76068376068, -2206.19658119658, 0, 314102564.102564, -529487179.48718, 0, 0, 0, 94230769.2307692, 1308.76068376068, -2206.19658119658, 0, -224358974.358974, -547435897.435898, 0, 0, 0, 94230769.2307692, -934.82905982906, -2280.98290598291, 0, 224358974.358974, -547435897.435898, 0, 0, 0, 94230769.2307692, 934.82905982906, -2280.98290598291, 0, 0, 2943589743.58974, 0, 0, 0, -323076923.076923, 0, 12264.9572649573, 0, -897435897.435898, 0, 0, 0, 0, -269230769.230769, -3739.31623931624, 0, 0, 0, -789743589.74359, 0, 0, 0, -323076923.076923, 0, -3290.59829059829, 0, 897435897.435898, 0, 0, 0, 0, -269230769.230769, 3739.31623931624, 0, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(38) << 0, 0, -664102564.102564, 89743589.7435897, 62820512.8205128, 0, 0, 0, -2767.09401709402, 0, 0, -664102564.102564, -89743589.7435897, 62820512.8205128, 0, 0, 0, -2767.09401709402, 0, 0, -412820512.820513, 0, 62820512.8205128, 0, 0, 0, -1720.08547008547, 0, 0, -412820512.820513, 0, 62820512.8205128, 0, 0, 0, -1720.08547008547, 0, 0, 1866666666.66667, 0, -215384615.384615, 0, 0, 0, 7777.77777777778, 0, 0, 0, -179487179.48718, -179487179.48718, 0, 0, 0, 0, 0, 0, 287179487.179487, 0, -215384615.384615, 0, 0, 0, 1196.5811965812, 0, 0, 0, 179487179.48718, -179487179.48718, 0, 0, 0, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(39) << 0, 0, -89743589.7435897, -26943568.3760684, -4861.11111111111, 0, 0, 0, 747.863247863248, 0, 0, 89743589.7435897, -26943568.3760684, 4861.11111111111, 0, 0, 0, -747.863247863248, 0, 0, 0, -35908354.7008547, -1869.65811965812, 0, 0, 0, 0, 0, 0, 0, -35908354.7008547, 1869.65811965812, 0, 0, 0, 0, 0, 0, 0, 143635213.675214, 0, 0, 0, 0, 0, 0, 0, 179487179.48718, 89743589.7435897, -7478.63247863248, 0, 0, 0, -1495.7264957265, 0, 0, 0, 71812222.2222222, 0, 0, 0, 0, 0, 0, 0, -179487179.48718, 89743589.7435897, 7478.63247863248, 0, 0, 0, 1495.7264957265;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(40) << 0, 0, -116666666.666667, -2617.52136752137, -26927489.3162393, 0, 0, 0, 299.145299145299, 0, 0, -116666666.666667, 2617.52136752137, -26927489.3162393, 0, 0, 0, 299.145299145299, 0, 0, -62820512.8205128, -1869.65811965812, -35901997.8632479, 0, 0, 0, 523.504273504273, 0, 0, -62820512.8205128, 1869.65811965812, -35901997.8632479, 0, 0, 0, 523.504273504273, 0, 0, -215384615.384615, 0, 143614273.504274, 0, 0, 0, -3589.74358974359, 0, 0, 179487179.48718, -7478.63247863248, 89743589.7435897, 0, 0, 0, -1495.7264957265, 0, 0, 215384615.384615, 0, 71788290.5982906, 0, 0, 0, -1794.87179487179, 0, 0, 179487179.48718, 7478.63247863248, 89743589.7435897, 0, 0, 0, -1495.7264957265;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(41) << -134615384.615385, -175000000, 0, 0, 0, -94236303.4188034, 186.965811965812, -205.662393162393, 0, 134615384.615385, -175000000, 0, 0, 0, -94236303.4188034, -186.965811965812, -205.662393162393, 0, 0, -94230769.2307692, 0, 0, 0, -125644465.811966, 0, 130.876068376068, 0, 0, -94230769.2307692, 0, 0, 0, -125644465.811966, 0, 130.876068376068, 0, 0, -323076923.076923, 0, 0, 0, 502579658.119658, 0, -3141.02564102564, 0, 269230769.230769, 269230769.230769, 0, 0, 0, 314102564.102564, -373.931623931624, -373.931623931624, 0, 0, 323076923.076923, 0, 0, 0, 251284444.444444, 0, -448.717948717949, 0, -269230769.230769, 269230769.230769, 0, 0, 0, 314102564.102564, 373.931623931624, -373.931623931624, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(42) << -10245.7264957264, -2430.55555555556, 0, 0, 0, -186.965811965812, -224.435817307692, -0.0182291666666667, 0, -10245.7264957264, 2430.55555555556, 0, 0, 0, 186.965811965812, -224.435817307692, 0.0182291666666667, 0, -5459.40170940171, -934.82905982906, 0, 0, 0, 0, -299.18624465812, -0.00701121794871795, 0, -5459.40170940171, 934.82905982906, 0, 0, 0, 0, -299.18624465812, 0.00701121794871795, 0, 22735.0427350427, 0, 0, 0, 0, 0, 1196.75170940171, 0, 0, 0, -3739.31623931624, 0, 0, 0, 373.931623931624, 747.863247863248, -0.0280448717948718, 0, 8675.21367521367, 0, 0, 0, 0, 0, 598.355662393162, 0, 0, 0, 3739.31623931624, 0, 0, 0, -373.931623931624, 747.863247863248, 0.0280448717948718, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(43) << -1308.76068376068, -2206.19658119658, 0, 0, 0, -579.594017094017, -0.00981570512820513, -224.375520833333, 0, 1308.76068376068, -2206.19658119658, 0, 0, 0, -579.594017094017, 0.00981570512820513, -224.375520833333, 0, -934.82905982906, -2280.98290598291, 0, 0, 0, -130.876068376068, -0.00701121794871795, -299.162406517094, 0, 934.82905982906, -2280.98290598291, 0, 0, 0, -130.876068376068, 0.00701121794871795, -299.162406517094, 0, 0, 12264.9572649573, 0, 0, 0, -3141.02564102564, 0, 1196.67318376068, 0, -3739.31623931624, 0, 0, 0, 0, 373.931623931624, -0.0280448717948718, 747.863247863248, 0, 0, -3290.59829059829, 0, 0, 0, 448.717948717949, 0, 598.265918803419, 0, 3739.31623931624, 0, 0, 0, 0, 373.931623931624, 0.0280448717948718, 747.863247863248, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(44) << 0, 0, -2767.09401709402, -747.863247863248, -1196.5811965812, 0, 0, 0, -785.277163461538, 0, 0, -2767.09401709402, 747.863247863248, -1196.5811965812, 0, 0, 0, -785.277163461538, 0, 0, -1720.08547008547, 0, -523.504273504273, 0, 0, 0, -1047.02144764957, 0, 0, -1720.08547008547, 0, -523.504273504273, 0, 0, 0, -1047.02144764957, 0, 0, 7777.77777777778, 0, -3589.74358974359, 0, 0, 0, 4188.09252136752, 0, 0, 0, 1495.7264957265, 1495.7264957265, 0, 0, 0, 2617.52136752137, 0, 0, 1196.5811965812, 0, 1794.87179487179, 0, 0, 0, 2094.02606837607, 0, 0, 0, -1495.7264957265, 1495.7264957265, 0, 0, 0, 2617.52136752137;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(45) << -547435897.435898, -224358974.358974, 0, 0, 0, -94230769.2307692, -2280.98290598291, -934.82905982906, 0, -529487179.48718, 314102564.102564, 0, 0, 0, -94230769.2307692, -2206.19658119658, 1308.76068376068, 0, -529487179.48718, -314102564.102564, 0, 0, 0, -94230769.2307692, -2206.19658119658, -1308.76068376068, 0, -547435897.435898, 224358974.358974, 0, 0, 0, -94230769.2307692, -2280.98290598291, 934.82905982906, 0, 0, -897435897.435898, 0, 0, 0, 269230769.230769, 0, -3739.31623931624, 0, 2943589743.58974, 0, 0, 0, 0, 323076923.076923, 12264.9572649573, 0, 0, 0, 897435897.435898, 0, 0, 0, 269230769.230769, 0, 3739.31623931624, 0, -789743589.74359, 0, 0, 0, 0, 323076923.076923, -3290.59829059829, 0, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(46) << -224358974.358974, -1310256410.25641, 0, 0, 0, 0, -934.82905982906, -5459.40170940171, 0, 583333333.333333, -2458974358.97436, 0, 0, 0, 134615384.615385, 2430.55555555556, -10245.7264957264, 0, -583333333.333333, -2458974358.97436, 0, 0, 0, -134615384.615385, -2430.55555555556, -10245.7264957264, 0, 224358974.358974, -1310256410.25641, 0, 0, 0, 0, 934.82905982906, -5459.40170940171, 0, -897435897.435898, 0, 0, 0, 0, 269230769.230769, -3739.31623931624, 0, 0, 0, 5456410256.41026, 0, 0, 0, 0, 0, 22735.0427350427, 0, 897435897.435898, 0, 0, 0, 0, -269230769.230769, 3739.31623931624, 0, 0, 0, 2082051282.05128, 0, 0, 0, 0, 0, 8675.21367521367, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(47) << 0, 0, -412820512.820513, -62820512.8205128, 0, 0, 0, 0, -1720.08547008547, 0, 0, -664102564.102564, -62820512.8205128, 89743589.7435897, 0, 0, 0, -2767.09401709402, 0, 0, -664102564.102564, -62820512.8205128, -89743589.7435897, 0, 0, 0, -2767.09401709402, 0, 0, -412820512.820513, -62820512.8205128, 0, 0, 0, 0, -1720.08547008547, 0, 0, 0, 179487179.48718, 179487179.48718, 0, 0, 0, 0, 0, 0, 1866666666.66667, 215384615.384615, 0, 0, 0, 0, 7777.77777777778, 0, 0, 0, 179487179.48718, -179487179.48718, 0, 0, 0, 0, 0, 0, 287179487.179487, 215384615.384615, 0, 0, 0, 0, 1196.5811965812;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(48) << 0, 0, 62820512.8205128, -35901997.8632479, -1869.65811965812, 0, 0, 0, -523.504273504273, 0, 0, 116666666.666667, -26927489.3162393, 2617.52136752137, 0, 0, 0, -299.145299145299, 0, 0, 116666666.666667, -26927489.3162393, -2617.52136752137, 0, 0, 0, -299.145299145299, 0, 0, 62820512.8205128, -35901997.8632479, 1869.65811965812, 0, 0, 0, -523.504273504273, 0, 0, -179487179.48718, 89743589.7435897, -7478.63247863248, 0, 0, 0, 1495.7264957265, 0, 0, 215384615.384615, 143614273.504274, 0, 0, 0, 0, 3589.74358974359, 0, 0, -179487179.48718, 89743589.7435897, 7478.63247863248, 0, 0, 0, 1495.7264957265, 0, 0, -215384615.384615, 71788290.5982906, 0, 0, 0, 0, 1794.87179487179;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(49) << 0, 0, 0, -1869.65811965812, -35908354.7008547, 0, 0, 0, 0, 0, 0, -89743589.7435897, 4861.11111111111, -26943568.3760684, 0, 0, 0, 747.863247863248, 0, 0, 89743589.7435897, -4861.11111111111, -26943568.3760684, 0, 0, 0, -747.863247863248, 0, 0, 0, 1869.65811965812, -35908354.7008547, 0, 0, 0, 0, 0, 0, -179487179.48718, -7478.63247863248, 89743589.7435897, 0, 0, 0, 1495.7264957265, 0, 0, 0, 0, 143635213.675214, 0, 0, 0, 0, 0, 0, 179487179.48718, 7478.63247863248, 89743589.7435897, 0, 0, 0, -1495.7264957265, 0, 0, 0, 0, 71812222.2222222, 0, 0, 0, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(50) << 94230769.2307692, 0, 0, 0, 0, -125644465.811966, -130.876068376068, 0, 0, 175000000, -134615384.615385, 0, 0, 0, -94236303.4188034, 205.662393162393, 186.965811965812, 0, 175000000, 134615384.615385, 0, 0, 0, -94236303.4188034, 205.662393162393, -186.965811965812, 0, 94230769.2307692, 0, 0, 0, 0, -125644465.811966, -130.876068376068, 0, 0, -269230769.230769, -269230769.230769, 0, 0, 0, 314102564.102564, 373.931623931624, 373.931623931624, 0, 323076923.076923, 0, 0, 0, 0, 502579658.119658, 3141.02564102564, 0, 0, -269230769.230769, 269230769.230769, 0, 0, 0, 314102564.102564, 373.931623931624, -373.931623931624, 0, -323076923.076923, 0, 0, 0, 0, 251284444.444444, 448.717948717949, 0, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(51) << -2280.98290598291, -934.82905982906, 0, 0, 0, 130.876068376068, -299.162406517094, -0.00701121794871795, 0, -2206.19658119658, 1308.76068376068, 0, 0, 0, 579.594017094017, -224.375520833333, 0.00981570512820513, 0, -2206.19658119658, -1308.76068376068, 0, 0, 0, 579.594017094017, -224.375520833333, -0.00981570512820513, 0, -2280.98290598291, 934.82905982906, 0, 0, 0, 130.876068376068, -299.162406517094, 0.00701121794871795, 0, 0, -3739.31623931624, 0, 0, 0, -373.931623931624, 747.863247863248, -0.0280448717948718, 0, 12264.9572649573, 0, 0, 0, 0, 3141.02564102564, 1196.67318376068, 0, 0, 0, 3739.31623931624, 0, 0, 0, -373.931623931624, 747.863247863248, 0.0280448717948718, 0, -3290.59829059829, 0, 0, 0, 0, -448.717948717949, 598.265918803419, 0, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(52) << -934.82905982906, -5459.40170940171, 0, 0, 0, 0, -0.00701121794871795, -299.18624465812, 0, 2430.55555555556, -10245.7264957264, 0, 0, 0, -186.965811965812, 0.0182291666666667, -224.435817307692, 0, -2430.55555555556, -10245.7264957264, 0, 0, 0, 186.965811965812, -0.0182291666666667, -224.435817307692, 0, 934.82905982906, -5459.40170940171, 0, 0, 0, 0, 0.00701121794871795, -299.18624465812, 0, -3739.31623931624, 0, 0, 0, 0, -373.931623931624, -0.0280448717948718, 747.863247863248, 0, 0, 22735.0427350427, 0, 0, 0, 0, 0, 1196.75170940171, 0, 3739.31623931624, 0, 0, 0, 0, 373.931623931624, 0.0280448717948718, 747.863247863248, 0, 0, 8675.21367521367, 0, 0, 0, 0, 0, 598.355662393162, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(53) << 0, 0, -1720.08547008547, 523.504273504273, 0, 0, 0, 0, -1047.02144764957, 0, 0, -2767.09401709402, 1196.5811965812, -747.863247863248, 0, 0, 0, -785.277163461538, 0, 0, -2767.09401709402, 1196.5811965812, 747.863247863248, 0, 0, 0, -785.277163461538, 0, 0, -1720.08547008547, 523.504273504273, 0, 0, 0, 0, -1047.02144764957, 0, 0, 0, -1495.7264957265, -1495.7264957265, 0, 0, 0, 2617.52136752137, 0, 0, 7777.77777777778, 3589.74358974359, 0, 0, 0, 0, 4188.09252136752, 0, 0, 0, -1495.7264957265, 1495.7264957265, 0, 0, 0, 2617.52136752137, 0, 0, 1196.5811965812, -1794.87179487179, 0, 0, 0, 0, 2094.02606837607;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(54) << -1310256410.25641, -224358974.358974, 0, 0, 0, 0, -5459.40170940171, -934.82905982906, 0, -1310256410.25641, 224358974.358974, 0, 0, 0, 0, -5459.40170940171, 934.82905982906, 0, -2458974358.97436, -583333333.333333, 0, 0, 0, -134615384.615385, -10245.7264957264, -2430.55555555556, 0, -2458974358.97436, 583333333.333333, 0, 0, 0, 134615384.615385, -10245.7264957264, 2430.55555555556, 0, 2082051282.05128, 0, 0, 0, 0, 0, 8675.21367521367, 0, 0, 0, 897435897.435898, 0, 0, 0, -269230769.230769, 0, 3739.31623931624, 0, 5456410256.41026, 0, 0, 0, 0, 0, 22735.0427350427, 0, 0, 0, -897435897.435898, 0, 0, 0, 269230769.230769, 0, -3739.31623931624, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(55) << -224358974.358974, -547435897.435898, 0, 0, 0, -94230769.2307692, -934.82905982906, -2280.98290598291, 0, 224358974.358974, -547435897.435898, 0, 0, 0, -94230769.2307692, 934.82905982906, -2280.98290598291, 0, -314102564.102564, -529487179.48718, 0, 0, 0, -94230769.2307692, -1308.76068376068, -2206.19658119658, 0, 314102564.102564, -529487179.48718, 0, 0, 0, -94230769.2307692, 1308.76068376068, -2206.19658119658, 0, 0, -789743589.74359, 0, 0, 0, 323076923.076923, 0, -3290.59829059829, 0, 897435897.435898, 0, 0, 0, 0, 269230769.230769, 3739.31623931624, 0, 0, 0, 2943589743.58974, 0, 0, 0, 323076923.076923, 0, 12264.9572649573, 0, -897435897.435898, 0, 0, 0, 0, 269230769.230769, -3739.31623931624, 0, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(56) << 0, 0, -412820512.820513, 0, -62820512.8205128, 0, 0, 0, -1720.08547008547, 0, 0, -412820512.820513, 0, -62820512.8205128, 0, 0, 0, -1720.08547008547, 0, 0, -664102564.102564, -89743589.7435897, -62820512.8205128, 0, 0, 0, -2767.09401709402, 0, 0, -664102564.102564, 89743589.7435897, -62820512.8205128, 0, 0, 0, -2767.09401709402, 0, 0, 287179487.179487, 0, 215384615.384615, 0, 0, 0, 1196.5811965812, 0, 0, 0, -179487179.48718, 179487179.48718, 0, 0, 0, 0, 0, 0, 1866666666.66667, 0, 215384615.384615, 0, 0, 0, 7777.77777777778, 0, 0, 0, 179487179.48718, 179487179.48718, 0, 0, 0, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(57) << 0, 0, 0, -35908354.7008547, -1869.65811965812, 0, 0, 0, 0, 0, 0, 0, -35908354.7008547, 1869.65811965812, 0, 0, 0, 0, 0, 0, 89743589.7435897, -26943568.3760684, -4861.11111111111, 0, 0, 0, -747.863247863248, 0, 0, -89743589.7435897, -26943568.3760684, 4861.11111111111, 0, 0, 0, 747.863247863248, 0, 0, 0, 71812222.2222222, 0, 0, 0, 0, 0, 0, 0, 179487179.48718, 89743589.7435897, 7478.63247863248, 0, 0, 0, -1495.7264957265, 0, 0, 0, 143635213.675214, 0, 0, 0, 0, 0, 0, 0, -179487179.48718, 89743589.7435897, -7478.63247863248, 0, 0, 0, 1495.7264957265;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(58) << 0, 0, 62820512.8205128, -1869.65811965812, -35901997.8632479, 0, 0, 0, -523.504273504273, 0, 0, 62820512.8205128, 1869.65811965812, -35901997.8632479, 0, 0, 0, -523.504273504273, 0, 0, 116666666.666667, -2617.52136752137, -26927489.3162393, 0, 0, 0, -299.145299145299, 0, 0, 116666666.666667, 2617.52136752137, -26927489.3162393, 0, 0, 0, -299.145299145299, 0, 0, -215384615.384615, 0, 71788290.5982906, 0, 0, 0, 1794.87179487179, 0, 0, -179487179.48718, 7478.63247863248, 89743589.7435897, 0, 0, 0, 1495.7264957265, 0, 0, 215384615.384615, 0, 143614273.504274, 0, 0, 0, 3589.74358974359, 0, 0, -179487179.48718, -7478.63247863248, 89743589.7435897, 0, 0, 0, 1495.7264957265;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(59) << 0, 94230769.2307692, 0, 0, 0, -125644465.811966, 0, -130.876068376068, 0, 0, 94230769.2307692, 0, 0, 0, -125644465.811966, 0, -130.876068376068, 0, 134615384.615385, 175000000, 0, 0, 0, -94236303.4188034, -186.965811965812, 205.662393162393, 0, -134615384.615385, 175000000, 0, 0, 0, -94236303.4188034, 186.965811965812, 205.662393162393, 0, 0, -323076923.076923, 0, 0, 0, 251284444.444444, 0, 448.717948717949, 0, 269230769.230769, -269230769.230769, 0, 0, 0, 314102564.102564, -373.931623931624, 373.931623931624, 0, 0, 323076923.076923, 0, 0, 0, 502579658.119658, 0, 3141.02564102564, 0, -269230769.230769, -269230769.230769, 0, 0, 0, 314102564.102564, 373.931623931624, 373.931623931624, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(60) << -5459.40170940171, -934.82905982906, 0, 0, 0, 0, -299.18624465812, -0.00701121794871795, 0, -5459.40170940171, 934.82905982906, 0, 0, 0, 0, -299.18624465812, 0.00701121794871795, 0, -10245.7264957264, -2430.55555555556, 0, 0, 0, 186.965811965812, -224.435817307692, -0.0182291666666667, 0, -10245.7264957264, 2430.55555555556, 0, 0, 0, -186.965811965812, -224.435817307692, 0.0182291666666667, 0, 8675.21367521367, 0, 0, 0, 0, 0, 598.355662393162, 0, 0, 0, 3739.31623931624, 0, 0, 0, 373.931623931624, 747.863247863248, 0.0280448717948718, 0, 22735.0427350427, 0, 0, 0, 0, 0, 1196.75170940171, 0, 0, 0, -3739.31623931624, 0, 0, 0, -373.931623931624, 747.863247863248, -0.0280448717948718, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(61) << -934.82905982906, -2280.98290598291, 0, 0, 0, 130.876068376068, -0.00701121794871795, -299.162406517094, 0, 934.82905982906, -2280.98290598291, 0, 0, 0, 130.876068376068, 0.00701121794871795, -299.162406517094, 0, -1308.76068376068, -2206.19658119658, 0, 0, 0, 579.594017094017, -0.00981570512820513, -224.375520833333, 0, 1308.76068376068, -2206.19658119658, 0, 0, 0, 579.594017094017, 0.00981570512820513, -224.375520833333, 0, 0, -3290.59829059829, 0, 0, 0, -448.717948717949, 0, 598.265918803419, 0, 3739.31623931624, 0, 0, 0, 0, -373.931623931624, 0.0280448717948718, 747.863247863248, 0, 0, 12264.9572649573, 0, 0, 0, 3141.02564102564, 0, 1196.67318376068, 0, -3739.31623931624, 0, 0, 0, 0, -373.931623931624, -0.0280448717948718, 747.863247863248, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(62) << 0, 0, -1720.08547008547, 0, 523.504273504273, 0, 0, 0, -1047.02144764957, 0, 0, -1720.08547008547, 0, 523.504273504273, 0, 0, 0, -1047.02144764957, 0, 0, -2767.09401709402, 747.863247863248, 1196.5811965812, 0, 0, 0, -785.277163461538, 0, 0, -2767.09401709402, -747.863247863248, 1196.5811965812, 0, 0, 0, -785.277163461538, 0, 0, 1196.5811965812, 0, -1794.87179487179, 0, 0, 0, 2094.02606837607, 0, 0, 0, 1495.7264957265, -1495.7264957265, 0, 0, 0, 2617.52136752137, 0, 0, 7777.77777777778, 0, 3589.74358974359, 0, 0, 0, 4188.09252136752, 0, 0, 0, -1495.7264957265, -1495.7264957265, 0, 0, 0, 2617.52136752137;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(63) << -529487179.48718, -314102564.102564, 0, 0, 0, 94230769.2307692, -2206.19658119658, -1308.76068376068, 0, -547435897.435898, 224358974.358974, 0, 0, 0, 94230769.2307692, -2280.98290598291, 934.82905982906, 0, -547435897.435898, -224358974.358974, 0, 0, 0, 94230769.2307692, -2280.98290598291, -934.82905982906, 0, -529487179.48718, 314102564.102564, 0, 0, 0, 94230769.2307692, -2206.19658119658, 1308.76068376068, 0, 0, 897435897.435898, 0, 0, 0, -269230769.230769, 0, 3739.31623931624, 0, -789743589.74359, 0, 0, 0, 0, -323076923.076923, -3290.59829059829, 0, 0, 0, -897435897.435898, 0, 0, 0, -269230769.230769, 0, -3739.31623931624, 0, 2943589743.58974, 0, 0, 0, 0, -323076923.076923, 12264.9572649573, 0, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(64) << -583333333.333333, -2458974358.97436, 0, 0, 0, 134615384.615385, -2430.55555555556, -10245.7264957264, 0, 224358974.358974, -1310256410.25641, 0, 0, 0, 0, 934.82905982906, -5459.40170940171, 0, -224358974.358974, -1310256410.25641, 0, 0, 0, 0, -934.82905982906, -5459.40170940171, 0, 583333333.333333, -2458974358.97436, 0, 0, 0, -134615384.615385, 2430.55555555556, -10245.7264957264, 0, 897435897.435898, 0, 0, 0, 0, 269230769.230769, 3739.31623931624, 0, 0, 0, 2082051282.05128, 0, 0, 0, 0, 0, 8675.21367521367, 0, -897435897.435898, 0, 0, 0, 0, -269230769.230769, -3739.31623931624, 0, 0, 0, 5456410256.41026, 0, 0, 0, 0, 0, 22735.0427350427, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(65) << 0, 0, -664102564.102564, 62820512.8205128, 89743589.7435897, 0, 0, 0, -2767.09401709402, 0, 0, -412820512.820513, 62820512.8205128, 0, 0, 0, 0, -1720.08547008547, 0, 0, -412820512.820513, 62820512.8205128, 0, 0, 0, 0, -1720.08547008547, 0, 0, -664102564.102564, 62820512.8205128, -89743589.7435897, 0, 0, 0, -2767.09401709402, 0, 0, 0, -179487179.48718, 179487179.48718, 0, 0, 0, 0, 0, 0, 287179487.179487, -215384615.384615, 0, 0, 0, 0, 1196.5811965812, 0, 0, 0, -179487179.48718, -179487179.48718, 0, 0, 0, 0, 0, 0, 1866666666.66667, -215384615.384615, 0, 0, 0, 0, 7777.77777777778;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(66) << 0, 0, -116666666.666667, -26927489.3162393, -2617.52136752137, 0, 0, 0, 299.145299145299, 0, 0, -62820512.8205128, -35901997.8632479, 1869.65811965812, 0, 0, 0, 523.504273504273, 0, 0, -62820512.8205128, -35901997.8632479, -1869.65811965812, 0, 0, 0, 523.504273504273, 0, 0, -116666666.666667, -26927489.3162393, 2617.52136752137, 0, 0, 0, 299.145299145299, 0, 0, 179487179.48718, 89743589.7435897, 7478.63247863248, 0, 0, 0, -1495.7264957265, 0, 0, 215384615.384615, 71788290.5982906, 0, 0, 0, 0, -1794.87179487179, 0, 0, 179487179.48718, 89743589.7435897, -7478.63247863248, 0, 0, 0, -1495.7264957265, 0, 0, -215384615.384615, 143614273.504274, 0, 0, 0, 0, -3589.74358974359;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(67) << 0, 0, -89743589.7435897, -4861.11111111111, -26943568.3760684, 0, 0, 0, 747.863247863248, 0, 0, 0, 1869.65811965812, -35908354.7008547, 0, 0, 0, 0, 0, 0, 0, -1869.65811965812, -35908354.7008547, 0, 0, 0, 0, 0, 0, 89743589.7435897, 4861.11111111111, -26943568.3760684, 0, 0, 0, -747.863247863248, 0, 0, -179487179.48718, 7478.63247863248, 89743589.7435897, 0, 0, 0, 1495.7264957265, 0, 0, 0, 0, 71812222.2222222, 0, 0, 0, 0, 0, 0, 179487179.48718, -7478.63247863248, 89743589.7435897, 0, 0, 0, -1495.7264957265, 0, 0, 0, 0, 143635213.675214, 0, 0, 0, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(68) << -175000000, -134615384.615385, 0, 0, 0, -94236303.4188034, -205.662393162393, 186.965811965812, 0, -94230769.2307692, 0, 0, 0, 0, -125644465.811966, 130.876068376068, 0, 0, -94230769.2307692, 0, 0, 0, 0, -125644465.811966, 130.876068376068, 0, 0, -175000000, 134615384.615385, 0, 0, 0, -94236303.4188034, -205.662393162393, -186.965811965812, 0, 269230769.230769, -269230769.230769, 0, 0, 0, 314102564.102564, -373.931623931624, 373.931623931624, 0, 323076923.076923, 0, 0, 0, 0, 251284444.444444, -448.717948717949, 0, 0, 269230769.230769, 269230769.230769, 0, 0, 0, 314102564.102564, -373.931623931624, -373.931623931624, 0, -323076923.076923, 0, 0, 0, 0, 502579658.119658, -3141.02564102564, 0, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(69) << -2206.19658119658, -1308.76068376068, 0, 0, 0, -579.594017094017, -224.375520833333, -0.00981570512820513, 0, -2280.98290598291, 934.82905982906, 0, 0, 0, -130.876068376068, -299.162406517094, 0.00701121794871795, 0, -2280.98290598291, -934.82905982906, 0, 0, 0, -130.876068376068, -299.162406517094, -0.00701121794871795, 0, -2206.19658119658, 1308.76068376068, 0, 0, 0, -579.594017094017, -224.375520833333, 0.00981570512820513, 0, 0, 3739.31623931624, 0, 0, 0, 373.931623931624, 747.863247863248, 0.0280448717948718, 0, -3290.59829059829, 0, 0, 0, 0, 448.717948717949, 598.265918803419, 0, 0, 0, -3739.31623931624, 0, 0, 0, 373.931623931624, 747.863247863248, -0.0280448717948718, 0, 12264.9572649573, 0, 0, 0, 0, -3141.02564102564, 1196.67318376068, 0, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(70) << -2430.55555555556, -10245.7264957264, 0, 0, 0, -186.965811965812, -0.0182291666666667, -224.435817307692, 0, 934.82905982906, -5459.40170940171, 0, 0, 0, 0, 0.00701121794871795, -299.18624465812, 0, -934.82905982906, -5459.40170940171, 0, 0, 0, 0, -0.00701121794871795, -299.18624465812, 0, 2430.55555555556, -10245.7264957264, 0, 0, 0, 186.965811965812, 0.0182291666666667, -224.435817307692, 0, 3739.31623931624, 0, 0, 0, 0, -373.931623931624, 0.0280448717948718, 747.863247863248, 0, 0, 8675.21367521367, 0, 0, 0, 0, 0, 598.355662393162, 0, -3739.31623931624, 0, 0, 0, 0, 373.931623931624, -0.0280448717948718, 747.863247863248, 0, 0, 22735.0427350427, 0, 0, 0, 0, 0, 1196.75170940171, 0;
    //    Expected_JacobianK_NoDispNoVelNoDamping.row(71) << 0, 0, -2767.09401709402, -1196.5811965812, -747.863247863248, 0, 0, 0, -785.277163461538, 0, 0, -1720.08547008547, -523.504273504273, 0, 0, 0, 0, -1047.02144764957, 0, 0, -1720.08547008547, -523.504273504273, 0, 0, 0, 0, -1047.02144764957, 0, 0, -2767.09401709402, -1196.5811965812, 747.863247863248, 0, 0, 0, -785.277163461538, 0, 0, 0, 1495.7264957265, -1495.7264957265, 0, 0, 0, 2617.52136752137, 0, 0, 1196.5811965812, 1794.87179487179, 0, 0, 0, 0, 2094.02606837607, 0, 0, 0, 1495.7264957265, 1495.7264957265, 0, 0, 0, 2617.52136752137, 0, 0, 7777.77777777778, -3589.74358974359, 0, 0, 0, 0, 4188.09252136752;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianK_NoDispNoVelNoDamping;
    Expected_JacobianK_NoDispNoVelNoDamping.resize(72, 72);
    if (!load_validation_data("UT_ANCFShell_3833_JacNoDispNoVelNoDamping.txt", Expected_JacobianK_NoDispNoVelNoDamping))
        return false;

    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation
    ChVectorDynamic<double> InternalForceNoDispNoVelNoGravity;
    InternalForceNoDispNoVelNoGravity.resize(72);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceNoDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_NoDispNoVelNoDamping;
    JacobianK_NoDispNoVelNoDamping.resize(72, 72);
    m_element->ComputeKRMmatricesGlobal(JacobianK_NoDispNoVelNoDamping, 1.0, 0.0, 0.0);

    ChMatrixDynamic<double> JacobianR_NoDispNoVelNoDamping;
    JacobianR_NoDispNoVelNoDamping.resize(72, 72);
    m_element->ComputeKRMmatricesGlobal(JacobianR_NoDispNoVelNoDamping, 0.0, 1.0, 0.0);

    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_NoDispNoVelNoDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(72, 72);
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
    //ChMatrixNM<double, 72, 72> Expected_JacobianK_SmallDispNoVelNoDamping;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(0) << 2100001045.44872, 953525641.025641, -614743.58974359, 13461.5384615385, 0, -80769230.7692308, 8750.00435603633, 3973.0235042735, -2.56143162393162, 1032053109.45513, -33653846.1538461, 684294.871794872, 13461.5384615385, 0, 53846153.8461538, 4300.22128939637, -140.224358974359, 2.85122863247863, 928846689.166667, 392628205.128205, 42628.2051282051, -26923.0769230769, 0, 20192307.6923077, 3870.19453819444, 1635.95085470085, 0.177617521367521, 785256917.467949, 33653846.1538461, -49358.9743589744, -26923.0769230769, 0, 20192307.6923077, 3271.90382278312, 140.224358974359, -0.205662393162393, -2458976181.66667, -314102564.102564, -4487.17948717949, 108974.358974359, 0, -134615384.615385, -10245.7340902778, -1308.76068376068, -0.0186965811965812, -547436686.282051, -224358974.358974, 305128.205128205, 115384.615384615, 0, 94230769.2307692, -2280.98619284188, -934.82905982906, 1.27136752136752, -1310257051.02564, -224358974.358974, 4487.17948717949, 91025.641025641, 0, 0, -5459.4043792735, -934.82905982906, 0.0186965811965812, -529487842.564103, -583333333.333333, -367948.717948718, 115384.615384615, 0, -175000000, -2206.19934401709, -2430.55555555556, -1.53311965811966;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(1) << 953525641.025641, 2100001045.44872, -666346.153846154, 0, 13461.5384615385, -80769230.7692308, 3973.0235042735, 8750.00435603633, -2.77644230769231, 33653846.1538461, 785258237.660256, -648397.435897436, 0, 13461.5384615385, 20192307.6923077, 140.224358974359, 3271.9093235844, -2.70165598290598, 392628205.128205, 928846689.166667, -62820.5128205128, 0, -26923.0769230769, 20192307.6923077, 1635.95085470085, 3870.19453819444, -0.261752136752137, -33653846.1538461, 1032051789.26282, -49358.9743589744, 0, -26923.0769230769, 53846153.8461538, -140.224358974359, 4300.21578859509, -0.205662393162393, -583333333.333333, -529489002.179487, 332051.282051282, 0, 108974.358974359, -175000000, -2430.55555555556, -2206.20417574786, 1.38354700854701, -224358974.358974, -1310257199.10256, 228846.153846154, 0, 115384.615384615, 0, -934.82905982906, -5459.40499626068, 0.953525641025641, -224358974.358974, -547436538.205128, 700000, 0, 91025.641025641, 94230769.2307692, -934.82905982906, -2280.9855758547, 2.91666666666667, -314102564.102564, -2458975022.05128, 166025.641025641, 0, 115384.615384615, -134615384.615385, -1308.76068376068, -10245.729258547, 0.691773504273504;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(2) << -614743.58974359, -666346.153846154, 933336270.192308, -53846153.8461539, -53846153.8461539, 47115.3846153846, -2.56143162393162, -2.77644230769231, 3888.90112580128, 614743.58974359, -666346.153846154, 403851636.057692, 35897435.8974359, 13461538.4615385, 47115.3846153846, 2.56143162393162, -2.77644230769231, 1682.71515024038, 62820.5128205128, -42628.2051282051, 412822083.333333, 13461538.4615385, 13461538.4615385, -94230.7692307692, 0.261752136752137, -0.177617521367521, 1720.09201388889, -62820.5128205128, -42628.2051282051, 403847624.519231, 13461538.4615385, 35897435.8974359, -94230.7692307692, -0.261752136752137, -0.177617521367521, 1682.6984354968, 0, 125641.025641026, -664107800.641026, -89743589.7435897, -116666666.666667, 381410.256410256, 0, 0.523504273504274, -2767.11583600427, 430769.230769231, 296153.846153846, -412823070.512821, 62820512.8205128, 0, 403846.153846154, 1.79487179487179, 1.23397435897436, -1720.09612713675, 0, 700000, -412822352.564103, 0, 62820512.8205128, 318589.743589744, 0, 2.91666666666667, -1720.09313568376, -430769.230769231, 296153.846153846, -664104390.384615, -116666666.666667, -89743589.7435897, 403846.153846154, -1.79487179487179, 1.23397435897436, -2767.10162660256;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(3) << 13461.5384615385, 0, -53846153.8461539, 26940591.2266608, 7946.04700854701, -5517.94337606838, 0.168269230769231, 0, -897.435897435898, -73076.9230769231, 0, -35897435.8974359, 8982949.51309161, -280.448717948718, -2750.7077991453, -0.192307692307692, 0, 299.145299145299, -26923.0769230769, 0, -13461538.4615385, 13469286.1903584, 3271.90170940171, 2628.56036324786, -0.336538461538462, 0, 112.179487179487, -39102.5641025641, 0, 13461538.4615385, 8980906.4358507, 280.448717948718, 4358.56303418803, -0.387286324786325, 0, 224.358974358974, 1282.05128205128, 0, 89743589.7435897, -26943596.5322831, -2617.52136752137, -17051.3194444444, 0.913461538461539, 0, -747.863247863248, 46153.8461538462, 0, -62820512.8205128, -35902022.5172575, -1869.65811965812, -13330.7905982906, 1.15384615384615, 0, 523.504273504273, 34615.3846153846, 0, 0, -35908365.6997842, -1869.65811965812, -897.398504273504, 0.902777777777778, 0, 0, 43589.7435897436, 0, 62820512.8205128, -26927495.7320214, -4861.11111111111, -3336.39957264957, 1.14316239316239, 0, -1196.5811965812;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(4) << 0, 13461.5384615385, -53846153.8461539, 7946.04700854701, 26940591.2266608, -3467.09134615385, 0, 0.168269230769231, -897.435897435898, 0, -73076.9230769231, 13461538.4615385, 280.448717948718, 8980892.88915999, 4609.9813034188, 0, -0.192307692307692, 224.358974358974, 0, -26923.0769230769, -13461538.4615385, 3271.90170940171, 13469286.1903584, -2628.72863247863, 0, -0.336538461538462, 112.179487179487, 0, -39102.5641025641, -35897435.8974359, -280.448717948718, 8982963.05978232, -1731.18055555556, 0, -0.387286324786325, 299.145299145299, 0, 1282.05128205128, 62820512.8205128, -4861.11111111111, -26927517.4724541, 12310.4594017094, 0, 0.913461538461539, -1196.5811965812, 0, 46153.8461538462, 0, -1869.65811965812, -35908379.3548643, 9873.70192307692, 0, 1.15384615384615, 0, 0, 34615.3846153846, -62820512.8205128, -1869.65811965812, -35902008.8621774, -4609.55128205128, 0, 0.902777777777778, 523.504273504273, 0, 43589.7435897436, 89743589.7435897, -2617.52136752137, -26943574.7918504, -896.052350427351, 0, 1.14316239316239, -747.863247863248;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(5) << -80769230.7692308, -80769230.7692308, 47115.3846153846, -5517.94337606838, -3467.09134615385, 94238580.3877217, -785.25641025641, -785.25641025641, 0.588942307692308, -53846153.8461538, 20192307.6923077, -255769.230769231, -2751.28739316239, 4609.83173076923, 31413598.7315825, 74.7863247863248, 196.314102564103, -0.673076923076923, -20192307.6923077, -20192307.6923077, -94230.7692307692, 2628.72863247863, -2628.56036324786, 47118841.9254808, 28.0448717948718, 28.0448717948718, -1.17788461538462, 20192307.6923077, -53846153.8461538, -136858.974358974, 4358.45085470085, -1731.12446581197, 31413630.3327684, 196.314102564103, 74.7863247863248, -1.35550213675214, 134615384.615385, 94230769.2307692, 4487.17948717949, -17051.2820512821, 12308.7393162393, -94236369.1248344, -186.965811965812, -579.594017094017, 3.19711538461538, -94230769.2307692, 0, 161538.461538462, -13329.7435897436, 9874.26282051282, -125644523.343964, 130.876068376068, 0, 4.03846153846154, 0, -94230769.2307692, 121153.846153846, -897.435897435898, -4609.55128205128, -125644491.479006, 0, 130.876068376068, 3.15972222222222, 94230769.2307692, 134615384.615385, 152564.102564103, -3336.92307692308, -894.967948717949, -94236318.3912874, -579.594017094017, -186.965811965812, 4.00106837606838;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(6) << 8750.00435603633, 3973.0235042735, -2.56143162393162, 0.168269230769231, 0, -785.25641025641, 224.424718515576, 0.0297976762820513, -0.0459593816773504, 4300.22128939637, -140.224358974359, 2.85122863247863, -0.552884615384615, 0, -74.7863247863248, 74.818493913944, -0.00105168269230769, -0.0229487012553419, 3870.19453819444, 1635.95085470085, 0.177617521367521, -0.336538461538462, 0, -28.0448717948718, 112.208574803054, 0.0122696314102564, 0.0219030415331197, 3271.90382278312, 140.224358974359, -0.205662393162393, -0.438034188034188, 0, 196.314102564103, 74.8108945137136, 0.00105168269230769, 0.0363232438568376, -10245.7340902778, -1308.76068376068, -0.0186965811965812, 0.46474358974359, 0, 186.965811965812, -224.436051873198, -0.00981570512820513, -0.142094157318376, -2280.98619284188, -934.82905982906, 1.27136752136752, 0.865384615384615, 0, -130.876068376068, -299.162611937045, -0.00701121794871795, -0.111101575854701, -5459.4043792735, -934.82905982906, 0.0186965811965812, 0.667735042735043, 0, 0, -299.186336291392, -0.00701121794871795, -0.00747849225427351, -2206.19934401709, -2430.55555555556, -1.53311965811966, 0.844017094017094, 0, -205.662393162393, -224.375574272858, -0.0182291666666667, -0.0277892761752137;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(7) << 3973.0235042735, 8750.00435603633, -2.77644230769231, 0, 0.168269230769231, -785.25641025641, 0.0297976762820513, 224.424718515576, -0.0288669771634615, 140.224358974359, 3271.9093235844, -2.70165598290598, 0, -0.552884615384615, 196.314102564103, 0.00105168269230769, 74.8107815742004, 0.0384412760416667, 1635.95085470085, 3870.19453819444, -0.261752136752137, 0, -0.336538461538462, -28.0448717948718, 0.0122696314102564, 112.208574803054, -0.021903672542735, -140.224358974359, 4300.21578859509, -0.205662393162393, 0, -0.438034188034188, -74.7863247863248, -0.00105168269230769, 74.8186068534572, -0.0144246193910256, -2430.55555555556, -2206.20417574786, 1.38354700854701, 0, 0.46474358974359, -205.662393162393, -0.0182291666666667, -224.37575539884, 0.102574479166667, -934.82905982906, -5459.40499626068, 0.953525641025641, 0, 0.865384615384615, 0, -0.00701121794871795, -299.18645007807, 0.082272108707265, -934.82905982906, -2280.9855758547, 2.91666666666667, 0, 0.667735042735043, -130.876068376068, -0.00701121794871795, -299.162498150366, -0.0384396634615385, -1308.76068376068, -10245.729258547, 0.691773504273504, 0, 0.844017094017094, 186.965811965812, -0.00981570512820513, -224.435870747217, -0.00747344417735043;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(8) << -2.56143162393162, -2.77644230769231, 3888.90112580128, -897.435897435898, -897.435897435898, 0.588942307692308, -0.0459593816773504, -0.0288669771634615, 785.285854970694, 2.56143162393162, -2.77644230769231, 1682.71515024038, -299.145299145299, 224.358974358974, -1.93509615384615, -0.022950874732906, 0.0384407151442308, 261.764564540977, 0.261752136752137, -0.177617521367521, 1720.09201388889, -112.179487179487, -112.179487179487, -1.17788461538462, 0.021903672542735, -0.0219030415331197, 392.641248535546, -0.261752136752137, -0.177617521367521, 1682.6984354968, 224.358974358974, -299.145299145299, -1.53311965811966, 0.0363228231837607, -0.0144244090544872, 261.764828037412, 0, 0.523504273504274, -2767.11583600427, 747.863247863248, 299.145299145299, 1.62660256410256, -0.142094017094017, 0.102568028846154, -785.27771081179, 1.79487179487179, 1.23397435897436, -1720.09612713675, -523.504273504273, 0, 3.02884615384615, -0.11109764957265, 0.0822742120726496, -1047.0219269852, 0, 2.91666666666667, -1720.09313568376, 0, -523.504273504273, 2.33707264957265, -0.00747863247863248, -0.0384396634615385, -1047.02166147131, -1.79487179487179, 1.23397435897436, -2767.10162660256, 299.145299145299, 747.863247863248, 2.95405982905983, -0.0277912393162393, -0.00746937767094017, -785.277288162484;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(9) << 1032053109.45513, 33653846.1538461, 614743.58974359, -73076.9230769231, 0, -53846153.8461538, 4300.22128939637, 140.224358974359, 2.56143162393162, 2100008574.10256, -953525641.025641, 4125961.53846154, 191666.666666667, 0, 80769230.7692308, 8750.03572542735, -3973.0235042735, 17.1915064102564, 785258237.660256, -33653846.1538461, 666346.153846154, -73076.9230769231, 0, -20192307.6923077, 3271.9093235844, -140.224358974359, 2.77644230769231, 928847521.089744, -392628205.128205, 525000, -50641.0256410256, 0, -20192307.6923077, 3870.1980045406, -1635.95085470085, 2.1875, -2458979298.46154, 314102564.102564, -2598076.92307692, 333333.333333333, 0, 134615384.615385, -10245.7470769231, 1308.76068376068, -10.8253205128205, -529492118.974359, 583333333.333333, -915384.615384615, 333333.333333333, 0, 175000000, -2206.21716239316, 2430.55555555556, -3.81410256410256, -1310258268.84615, 224358974.358974, -955769.230769231, 135897.435897436, 0, 0, -5459.40945352564, 934.82905982906, -3.98237179487179, -547437756.025641, 224358974.358974, -1462820.51282051, 135897.435897436, 0, -94230769.2307692, -2280.99065010684, 934.82905982906, -6.09508547008547;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(10) << -33653846.1538461, 785258237.660256, -666346.153846154, 0, -73076.9230769231, 20192307.6923077, -140.224358974359, 3271.9093235844, -2.77644230769231, -953525641.025641, 2100008574.10256, -4125961.53846154, 0, 191666.666666667, -80769230.7692308, -3973.0235042735, 8750.03572542735, -17.1915064102564, 33653846.1538461, 1032053109.45513, -614743.58974359, 0, -73076.9230769231, 53846153.8461538, 140.224358974359, 4300.22128939637, -2.56143162393162, -392628205.128205, 928847521.089744, -525000, 0, -50641.0256410256, 20192307.6923077, -1635.95085470085, 3870.1980045406, -2.1875, 583333333.333333, -529492118.974359, 915384.615384615, 0, 333333.333333333, -175000000, 2430.55555555556, -2206.21716239316, 3.81410256410256, 314102564.102564, -2458979298.46154, 2598076.92307692, 0, 333333.333333333, -134615384.615385, 1308.76068376068, -10245.7470769231, 10.8253205128205, 224358974.358974, -547437756.025641, 1462820.51282051, 0, 135897.435897436, 94230769.2307692, 934.82905982906, -2280.99065010684, 6.09508547008547, 224358974.358974, -1310258268.84615, 955769.230769231, 0, 135897.435897436, 0, 934.82905982906, -5459.40945352564, 3.98237179487179;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(11) << 684294.871794872, -648397.435897436, 403851636.057692, -35897435.8974359, 13461538.4615385, -255769.230769231, 2.85122863247863, -2.70165598290598, 1682.71515024038, 4125961.53846154, -4125961.53846154, 933359055.641026, 53846153.8461539, -53846153.8461539, 670833.333333333, 17.1915064102564, -17.1915064102564, 3888.99606517094, 648397.435897436, -684294.871794872, 403851636.057692, -13461538.4615385, 35897435.8974359, -255769.230769231, 2.70165598290598, -2.85122863247863, 1682.71515024038, 518269.230769231, -518269.230769231, 412824614.551282, -13461538.4615385, 13461538.4615385, -177243.58974359, 2.15945512820513, -2.15945512820513, 1720.10256063034, -3096153.84615385, 403846.153846154, -664117382.564103, 89743589.7435897, -116666666.666667, 1166666.66666667, -12.900641025641, 1.68269230769231, -2767.15576068376, -403846.153846154, 3096153.84615385, -664117382.564103, 116666666.666667, -89743589.7435897, 1166666.66666667, -1.68269230769231, 12.900641025641, -2767.15576068376, -978205.128205128, 1498717.94871795, -412826088.589744, 0, 62820512.8205128, 475641.025641026, -4.0758547008547, 6.24465811965812, -1720.10870245727, -1498717.94871795, 978205.128205128, -412826088.589744, -62820512.8205128, 0, 475641.025641026, -6.24465811965812, 4.0758547008547, -1720.10870245727;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(12) << 13461.5384615385, 0, 35897435.8974359, 8982949.51309161, 280.448717948718, -2751.28739316239, -0.552884615384615, 0, -299.145299145299, 191666.666666667, 0, 53846153.8461539, 26940667.250938, -7946.04700854701, 25034.3830128205, 2.39583333333333, 0, 897.435897435898, 13461.5384615385, 0, -13461538.4615385, 8980892.88915999, -280.448717948718, -4609.83173076923, -0.552884615384615, 0, -224.358974358974, -3205.12820512821, 0, 13461538.4615385, 13469278.5049834, -3271.90170940171, -957.163461538462, -0.435363247863248, 0, -112.179487179487, -97435.8974358974, 0, -89743589.7435897, -26943528.2890256, 2617.52136752137, 15234.7596153846, 2.37179487179487, 0, 747.863247863248, -97435.8974358974, 0, -62820512.8205128, -26927449.2291966, 4861.11111111111, 11018.0128205128, 2.37179487179487, 0, 1196.5811965812, -10256.4102564103, 0, 0, -35908358.017625, 1869.65811965812, 2684.34294871795, 1.08974358974359, 0, 0, -10256.4102564103, 0, 62820512.8205128, -35902001.1800182, 1869.65811965812, 8192.93803418804, 1.08974358974359, 0, -523.504273504273;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(13) << 0, 13461.5384615385, 13461538.4615385, -280.448717948718, 8980892.88915999, 4609.83173076923, 0, -0.552884615384615, 224.358974358974, 0, 191666.666666667, -53846153.8461539, -7946.04700854701, 26940667.250938, -25034.3830128205, 0, 2.39583333333333, -897.435897435898, 0, 13461.5384615385, -35897435.8974359, 280.448717948718, 8982949.51309161, 2751.28739316239, 0, -0.552884615384615, 299.145299145299, 0, -3205.12820512821, -13461538.4615385, -3271.90170940171, 13469278.5049834, 957.163461538462, 0, -0.435363247863248, 112.179487179487, 0, -97435.8974358974, 62820512.8205128, 4861.11111111111, -26927449.2291966, -11018.0128205128, 0, 2.37179487179487, -1196.5811965812, 0, -97435.8974358974, 89743589.7435897, 2617.52136752137, -26943528.2890256, -15234.7596153846, 0, 2.37179487179487, -747.863247863248, 0, -10256.4102564103, -62820512.8205128, 1869.65811965812, -35902001.1800182, -8192.93803418804, 0, 1.08974358974359, 523.504273504273, 0, -10256.4102564103, 0, 1869.65811965812, -35908358.017625, -2684.34294871795, 0, 1.08974358974359, 0;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(14) << 53846153.8461538, 20192307.6923077, 47115.3846153846, -2750.7077991453, 4609.9813034188, 31413598.7315825, -74.7863247863248, 196.314102564103, -1.93509615384615, 80769230.7692308, -80769230.7692308, 670833.333333333, 25034.3830128205, -25034.3830128205, 94238757.8211902, 785.25641025641, -785.25641025641, 8.38541666666667, -20192307.6923077, -53846153.8461538, 47115.3846153846, -4609.9813034188, 2750.7077991453, 31413598.7315825, -196.314102564103, 74.7863247863248, -1.93509615384615, 20192307.6923077, -20192307.6923077, -11217.9487179487, -957.219551282051, 957.219551282051, 47118823.9978563, -28.0448717948718, 28.0448717948718, -1.52377136752137, -134615384.615385, 94230769.2307692, -341025.641025641, 15230.608974359, -11022.2756410256, -94236209.909812, 186.965811965812, -579.594017094017, 8.30128205128205, -94230769.2307692, 134615384.615385, -341025.641025641, 11022.2756410256, -15230.608974359, -94236209.909812, 579.594017094017, -186.965811965812, 8.30128205128205, 0, -94230769.2307692, -35897.4358974359, 2684.15598290598, -8192.63888888889, -125644473.561422, 0, 130.876068376068, 3.81410256410256, 94230769.2307692, 0, -35897.4358974359, 8192.63888888889, -2684.15598290598, -125644473.561422, -130.876068376068, 0, 3.81410256410256;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(15) << 4300.22128939637, 140.224358974359, 2.56143162393162, -0.192307692307692, 0, 74.7863247863248, 74.818493913944, 0.00105168269230769, -0.022950874732906, 8750.03572542735, -3973.0235042735, 17.1915064102564, 2.39583333333333, 0, 785.25641025641, 224.425351763667, -0.0297976762820513, 0.20846226963141, 3271.9093235844, -140.224358974359, 2.77644230769231, -0.192307692307692, 0, -196.314102564103, 74.8107815742004, -0.00105168269230769, -0.0384407151442308, 3870.1980045406, -1635.95085470085, 2.1875, -0.237713675213675, 0, 28.0448717948718, 112.208510726487, -0.0122696314102564, -0.00799641426282051, -10245.7470769231, 1308.76068376068, -10.8253205128205, 0.576923076923077, 0, -186.965811965812, -224.435483060342, 0.00981570512820513, 0.127055562232906, -2206.21716239316, 2430.55555555556, -3.81410256410256, 0.576923076923077, 0, 205.662393162393, -224.375186585983, 0.0182291666666667, 0.0918517361111111, -5459.40945352564, 934.82905982906, -3.98237179487179, 0.480769230769231, 0, 0, -299.186272226884, 0.00701121794871795, 0.0224060296474359, -2280.99065010684, 934.82905982906, -6.09508547008547, 0.480769230769231, 0, 130.876068376068, -299.162434085859, 0.00701121794871795, 0.0683303552350427;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(16) << -140.224358974359, 3271.9093235844, -2.77644230769231, 0, -0.192307692307692, 196.314102564103, -0.00105168269230769, 74.8107815742004, 0.0384407151442308, -3973.0235042735, 8750.03572542735, -17.1915064102564, 0, 2.39583333333333, -785.25641025641, -0.0297976762820513, 224.425351763667, -0.20846226963141, 140.224358974359, 4300.22128939637, -2.56143162393162, 0, -0.192307692307692, -74.7863247863248, 0.00105168269230769, 74.818493913944, 0.022950874732906, -1635.95085470085, 3870.1980045406, -2.1875, 0, -0.237713675213675, -28.0448717948718, -0.0122696314102564, 112.208510726487, 0.00799641426282051, 2430.55555555556, -2206.21716239316, 3.81410256410256, 0, 0.576923076923077, -205.662393162393, 0.0182291666666667, -224.375186585983, -0.0918517361111111, 1308.76068376068, -10245.7470769231, 10.8253205128205, 0, 0.576923076923077, 186.965811965812, 0.00981570512820513, -224.435483060342, -0.127055562232906, 934.82905982906, -2280.99065010684, 6.09508547008547, 0, 0.480769230769231, -130.876068376068, 0.00701121794871795, -299.162434085859, -0.0683303552350427, 934.82905982906, -5459.40945352564, 3.98237179487179, 0, 0.480769230769231, 0, 0.00701121794871795, -299.186272226884, -0.0224060296474359;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(17) << 2.85122863247863, -2.70165598290598, 1682.71515024038, 299.145299145299, 224.358974358974, -0.673076923076923, -0.0229487012553419, 0.0384412760416667, 261.764564540977, 17.1915064102564, -17.1915064102564, 3888.99606517094, 897.435897435898, -897.435897435898, 8.38541666666667, 0.20846226963141, -0.20846226963141, 785.287332712654, 2.70165598290598, -2.85122863247863, 1682.71515024038, -224.358974358974, -299.145299145299, -0.673076923076923, -0.0384412760416667, 0.0229487012553419, 261.764564540977, 2.15945512820513, -2.15945512820513, 1720.10256063034, 112.179487179487, -112.179487179487, -0.831997863247863, -0.00799662459935898, 0.00799662459935898, 392.641099041997, -12.900641025641, 1.68269230769231, -2767.15576068376, -747.863247863248, 299.145299145299, 2.01923076923077, 0.12703999732906, -0.0918677216880342, -785.27638365396, -1.68269230769231, 12.900641025641, -2767.15576068376, -299.145299145299, 747.863247863248, 2.01923076923077, 0.0918677216880342, -0.12703999732906, -785.27638365396, -4.0758547008547, 6.24465811965812, -1720.10870245727, 0, -523.504273504273, 1.68269230769231, 0.022405328525641, -0.0683292334401709, -1047.02151201541, -6.24465811965812, 4.0758547008547, -1720.10870245727, 523.504273504273, 0, 1.68269230769231, 0.0683292334401709, -0.022405328525641, -1047.02151201541;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(18) << 928846689.166667, 392628205.128205, 62820.5128205128, -26923.0769230769, 0, -20192307.6923077, 3870.19453819444, 1635.95085470085, 0.261752136752137, 785258237.660256, 33653846.1538461, 648397.435897436, 13461.5384615385, 0, -20192307.6923077, 3271.9093235844, 140.224358974359, 2.70165598290598, 2100001045.44872, 953525641.025641, 666346.153846154, 13461.5384615385, 0, 80769230.7692308, 8750.00435603633, 3973.0235042735, 2.77644230769231, 1032051789.26282, -33653846.1538461, 49358.9743589744, -26923.0769230769, 0, -53846153.8461538, 4300.21578859509, -140.224358974359, 0.205662393162393, -1310257199.10256, -224358974.358974, -228846.153846154, 115384.615384615, 0, 0, -5459.40499626068, -934.82905982906, -0.953525641025641, -529489002.179487, -583333333.333333, -332051.282051282, 108974.358974359, 0, 175000000, -2206.20417574786, -2430.55555555556, -1.38354700854701, -2458975022.05128, -314102564.102564, -166025.641025641, 115384.615384615, 0, 134615384.615385, -10245.729258547, -1308.76068376068, -0.691773504273504, -547436538.205128, -224358974.358974, -700000, 91025.641025641, 0, -94230769.2307692, -2280.9855758547, -934.82905982906, -2.91666666666667;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(19) << 392628205.128205, 928846689.166667, -42628.2051282051, 0, -26923.0769230769, -20192307.6923077, 1635.95085470085, 3870.19453819444, -0.177617521367521, -33653846.1538461, 1032053109.45513, -684294.871794872, 0, 13461.5384615385, -53846153.8461538, -140.224358974359, 4300.22128939637, -2.85122863247863, 953525641.025641, 2100001045.44872, 614743.58974359, 0, 13461.5384615385, 80769230.7692308, 3973.0235042735, 8750.00435603633, 2.56143162393162, 33653846.1538461, 785256917.467949, 49358.9743589744, 0, -26923.0769230769, -20192307.6923077, 140.224358974359, 3271.90382278312, 0.205662393162393, -224358974.358974, -547436686.282051, -305128.205128205, 0, 115384.615384615, -94230769.2307692, -934.82905982906, -2280.98619284188, -1.27136752136752, -314102564.102564, -2458976181.66667, 4487.17948717949, 0, 108974.358974359, 134615384.615385, -1308.76068376068, -10245.7340902778, 0.0186965811965812, -583333333.333333, -529487842.564103, 367948.717948718, 0, 115384.615384615, 175000000, -2430.55555555556, -2206.19934401709, 1.53311965811966, -224358974.358974, -1310257051.02564, -4487.17948717949, 0, 91025.641025641, 0, -934.82905982906, -5459.4043792735, -0.0186965811965812;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(20) << 42628.2051282051, -62820.5128205128, 412822083.333333, -13461538.4615385, -13461538.4615385, -94230.7692307692, 0.177617521367521, -0.261752136752137, 1720.09201388889, 666346.153846154, -614743.58974359, 403851636.057692, -13461538.4615385, -35897435.8974359, 47115.3846153846, 2.77644230769231, -2.56143162393162, 1682.71515024038, 666346.153846154, 614743.58974359, 933336270.192308, 53846153.8461539, 53846153.8461539, 47115.3846153846, 2.77644230769231, 2.56143162393162, 3888.90112580128, 42628.2051282051, 62820.5128205128, 403847624.519231, -35897435.8974359, -13461538.4615385, -94230.7692307692, 0.177617521367521, 0.261752136752137, 1682.6984354968, -296153.846153846, -430769.230769231, -412823070.512821, 0, -62820512.8205128, 403846.153846154, -1.23397435897436, -1.79487179487179, -1720.09612713675, -125641.025641026, 0, -664107800.641026, 116666666.666667, 89743589.7435897, 381410.256410256, -0.523504273504274, 0, -2767.11583600427, -296153.846153846, 430769.230769231, -664104390.384615, 89743589.7435897, 116666666.666667, 403846.153846154, -1.23397435897436, 1.79487179487179, -2767.10162660256, -700000, 0, -412822352.564103, -62820512.8205128, 0, 318589.743589744, -2.91666666666667, 0, -1720.09313568376;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(21) << -26923.0769230769, 0, 13461538.4615385, 13469286.1903584, 3271.90170940171, 2628.72863247863, -0.336538461538462, 0, -112.179487179487, -73076.9230769231, 0, -13461538.4615385, 8980892.88915999, 280.448717948718, -4609.9813034188, -0.192307692307692, 0, -224.358974358974, 13461.5384615385, 0, 53846153.8461539, 26940591.2266608, 7946.04700854701, 3467.09134615385, 0.168269230769231, 0, 897.435897435898, -39102.5641025641, 0, 35897435.8974359, 8982963.05978232, -280.448717948718, 1731.18055555556, -0.387286324786325, 0, -299.145299145299, 46153.8461538462, 0, 0, -35908379.3548643, -1869.65811965812, -9873.70192307692, 1.15384615384615, 0, 0, 1282.05128205128, 0, -62820512.8205128, -26927517.4724541, -4861.11111111111, -12310.4594017094, 0.913461538461539, 0, 1196.5811965812, 43589.7435897436, 0, -89743589.7435897, -26943574.7918504, -2617.52136752137, 896.052350427351, 1.14316239316239, 0, 747.863247863248, 34615.3846153846, 0, 62820512.8205128, -35902008.8621774, -1869.65811965812, 4609.55128205128, 0.902777777777778, 0, -523.504273504273;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(22) << 0, -26923.0769230769, 13461538.4615385, 3271.90170940171, 13469286.1903584, -2628.56036324786, 0, -0.336538461538462, -112.179487179487, 0, -73076.9230769231, 35897435.8974359, -280.448717948718, 8982949.51309161, 2750.7077991453, 0, -0.192307692307692, -299.145299145299, 0, 13461.5384615385, 53846153.8461539, 7946.04700854701, 26940591.2266608, 5517.94337606838, 0, 0.168269230769231, 897.435897435898, 0, -39102.5641025641, -13461538.4615385, 280.448717948718, 8980906.4358507, -4358.56303418803, 0, -0.387286324786325, -224.358974358974, 0, 46153.8461538462, 62820512.8205128, -1869.65811965812, -35902022.5172575, 13330.7905982906, 0, 1.15384615384615, -523.504273504273, 0, 1282.05128205128, -89743589.7435897, -2617.52136752137, -26943596.5322831, 17051.3194444444, 0, 0.913461538461539, 747.863247863248, 0, 43589.7435897436, -62820512.8205128, -4861.11111111111, -26927495.7320214, 3336.39957264957, 0, 1.14316239316239, 1196.5811965812, 0, 34615.3846153846, 0, -1869.65811965812, -35908365.6997842, 897.398504273504, 0, 0.902777777777778, 0;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(23) << 20192307.6923077, 20192307.6923077, -94230.7692307692, 2628.56036324786, -2628.72863247863, 47118841.9254808, -28.0448717948718, -28.0448717948718, -1.17788461538462, -20192307.6923077, 53846153.8461538, -255769.230769231, -4609.83173076923, 2751.28739316239, 31413598.7315825, -196.314102564103, -74.7863247863248, -0.673076923076923, 80769230.7692308, 80769230.7692308, 47115.3846153846, 3467.09134615385, 5517.94337606838, 94238580.3877217, 785.25641025641, 785.25641025641, 0.588942307692308, 53846153.8461538, -20192307.6923077, -136858.974358974, 1731.12446581197, -4358.45085470085, 31413630.3327684, -74.7863247863248, -196.314102564103, -1.35550213675214, 0, 94230769.2307692, 161538.461538462, -9874.26282051282, 13329.7435897436, -125644523.343964, 0, -130.876068376068, 4.03846153846154, -94230769.2307692, -134615384.615385, 4487.17948717949, -12308.7393162393, 17051.2820512821, -94236369.1248344, 579.594017094017, 186.965811965812, 3.19711538461538, -134615384.615385, -94230769.2307692, 152564.102564103, 894.967948717949, 3336.92307692308, -94236318.3912874, 186.965811965812, 579.594017094017, 4.00106837606838, 94230769.2307692, 0, 121153.846153846, 4609.55128205128, 897.435897435898, -125644491.479006, -130.876068376068, 0, 3.15972222222222;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(24) << 3870.19453819444, 1635.95085470085, 0.261752136752137, -0.336538461538462, 0, 28.0448717948718, 112.208574803054, 0.0122696314102564, 0.021903672542735, 3271.9093235844, 140.224358974359, 2.70165598290598, -0.552884615384615, 0, -196.314102564103, 74.8107815742004, 0.00105168269230769, -0.0384412760416667, 8750.00435603633, 3973.0235042735, 2.77644230769231, 0.168269230769231, 0, 785.25641025641, 224.424718515576, 0.0297976762820513, 0.0288669771634615, 4300.21578859509, -140.224358974359, 0.205662393162393, -0.438034188034188, 0, 74.7863247863248, 74.8186068534572, -0.00105168269230769, 0.0144246193910256, -5459.40499626068, -934.82905982906, -0.953525641025641, 0.865384615384615, 0, 0, -299.18645007807, -0.00701121794871795, -0.082272108707265, -2206.20417574786, -2430.55555555556, -1.38354700854701, 0.46474358974359, 0, 205.662393162393, -224.37575539884, -0.0182291666666667, -0.102574479166667, -10245.729258547, -1308.76068376068, -0.691773504273504, 0.844017094017094, 0, -186.965811965812, -224.435870747217, -0.00981570512820513, 0.00747344417735043, -2280.9855758547, -934.82905982906, -2.91666666666667, 0.667735042735043, 0, 130.876068376068, -299.162498150366, -0.00701121794871795, 0.0384396634615385;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(25) << 1635.95085470085, 3870.19453819444, -0.177617521367521, 0, -0.336538461538462, 28.0448717948718, 0.0122696314102564, 112.208574803054, -0.0219030415331197, -140.224358974359, 4300.22128939637, -2.85122863247863, 0, -0.552884615384615, 74.7863247863248, -0.00105168269230769, 74.818493913944, 0.0229487012553419, 3973.0235042735, 8750.00435603633, 2.56143162393162, 0, 0.168269230769231, 785.25641025641, 0.0297976762820513, 224.424718515576, 0.0459593816773504, 140.224358974359, 3271.90382278312, 0.205662393162393, 0, -0.438034188034188, -196.314102564103, 0.00105168269230769, 74.8108945137136, -0.0363232438568376, -934.82905982906, -2280.98619284188, -1.27136752136752, 0, 0.865384615384615, 130.876068376068, -0.00701121794871795, -299.162611937045, 0.111101575854701, -1308.76068376068, -10245.7340902778, 0.0186965811965812, 0, 0.46474358974359, -186.965811965812, -0.00981570512820513, -224.436051873198, 0.142094157318376, -2430.55555555556, -2206.19934401709, 1.53311965811966, 0, 0.844017094017094, 205.662393162393, -0.0182291666666667, -224.375574272858, 0.0277892761752137, -934.82905982906, -5459.4043792735, -0.0186965811965812, 0, 0.667735042735043, 0, -0.00701121794871795, -299.186336291392, 0.00747849225427351;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(26) << 0.177617521367521, -0.261752136752137, 1720.09201388889, 112.179487179487, 112.179487179487, -1.17788461538462, 0.0219030415331197, -0.021903672542735, 392.641248535546, 2.77644230769231, -2.56143162393162, 1682.71515024038, -224.358974358974, 299.145299145299, -1.93509615384615, -0.0384407151442308, 0.022950874732906, 261.764564540977, 2.77644230769231, 2.56143162393162, 3888.90112580128, 897.435897435898, 897.435897435898, 0.588942307692308, 0.0288669771634615, 0.0459593816773504, 785.285854970694, 0.177617521367521, 0.261752136752137, 1682.6984354968, 299.145299145299, -224.358974358974, -1.53311965811966, 0.0144244090544872, -0.0363228231837607, 261.764828037412, -1.23397435897436, -1.79487179487179, -1720.09612713675, 0, 523.504273504273, 3.02884615384615, -0.0822742120726496, 0.11109764957265, -1047.0219269852, -0.523504273504274, 0, -2767.11583600427, -299.145299145299, -747.863247863248, 1.62660256410256, -0.102568028846154, 0.142094017094017, -785.27771081179, -1.23397435897436, 1.79487179487179, -2767.10162660256, -747.863247863248, -299.145299145299, 2.95405982905983, 0.00746937767094017, 0.0277912393162393, -785.277288162484, -2.91666666666667, 0, -1720.09313568376, 523.504273504273, 0, 2.33707264957265, 0.0384396634615385, 0.00747863247863248, -1047.02166147131;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(27) << 785256917.467949, -33653846.1538461, -62820.5128205128, -39102.5641025641, 0, 20192307.6923077, 3271.90382278312, -140.224358974359, -0.261752136752137, 928847521.089744, -392628205.128205, 518269.230769231, -3205.12820512821, 0, 20192307.6923077, 3870.1980045406, -1635.95085470085, 2.15945512820513, 1032051789.26282, 33653846.1538461, 42628.2051282051, -39102.5641025641, 0, 53846153.8461538, 4300.21578859509, 140.224358974359, 0.177617521367521, 2100000683.71795, -953525641.025641, -525000, -3205.12820512821, 0, -80769230.7692308, 8750.00284882479, -3973.0235042735, -2.1875, -1310257259.23077, 224358974.358974, -76282.0512820513, 124358.974358974, 0, 0, -5459.40524679487, 934.82905982906, -0.31784188034188, -547436746.410256, 224358974.358974, 367948.717948718, 124358.974358974, 0, 94230769.2307692, -2280.98644337607, 934.82905982906, 1.53311965811966, -2458975042.69231, 314102564.102564, 76282.0512820513, 124358.974358974, 0, -134615384.615385, -10245.7293445513, 1308.76068376068, 0.31784188034188, -529487863.205128, 583333333.333333, -341025.641025641, 124358.974358974, 0, -175000000, -2206.19943002137, 2430.55555555556, -1.42094017094017;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(28) << 33653846.1538461, 1032051789.26282, -42628.2051282051, 0, -39102.5641025641, -53846153.8461538, 140.224358974359, 4300.21578859509, -0.177617521367521, -392628205.128205, 928847521.089744, -518269.230769231, 0, -3205.12820512821, -20192307.6923077, -1635.95085470085, 3870.1980045406, -2.15945512820513, -33653846.1538461, 785256917.467949, 62820.5128205128, 0, -39102.5641025641, -20192307.6923077, -140.224358974359, 3271.90382278312, 0.261752136752137, -953525641.025641, 2100000683.71795, 525000, 0, -3205.12820512821, 80769230.7692308, -3973.0235042735, 8750.00284882479, 2.1875, 224358974.358974, -547436746.410256, -367948.717948718, 0, 124358.974358974, -94230769.2307692, 934.82905982906, -2280.98644337607, -1.53311965811966, 224358974.358974, -1310257259.23077, 76282.0512820513, 0, 124358.974358974, 0, 934.82905982906, -5459.40524679487, 0.31784188034188, 583333333.333333, -529487863.205128, 341025.641025641, 0, 124358.974358974, 175000000, 2430.55555555556, -2206.19943002137, 1.42094017094017, 314102564.102564, -2458975042.69231, -76282.0512820513, 0, 124358.974358974, 134615384.615385, 1308.76068376068, -10245.7293445513, -0.31784188034188;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(29) << -49358.9743589744, -49358.9743589744, 403847624.519231, 13461538.4615385, -35897435.8974359, -136858.974358974, -0.205662393162393, -0.205662393162393, 1682.6984354968, 525000, -525000, 412824614.551282, 13461538.4615385, -13461538.4615385, -11217.9487179487, 2.1875, -2.1875, 1720.10256063034, 49358.9743589744, 49358.9743589744, 403847624.519231, 35897435.8974359, -13461538.4615385, -136858.974358974, 0.205662393162393, 0.205662393162393, 1682.6984354968, -525000, 525000, 933335309.48718, -53846153.8461539, 53846153.8461539, -11217.9487179487, -2.1875, 2.1875, 3888.89712286325, -161538.461538462, -457692.307692308, -412823046.282051, 0, -62820512.8205128, 435256.41025641, -0.673076923076923, -1.90705128205128, -1720.09602617521, 457692.307692308, 161538.461538462, -412823046.282051, 62820512.8205128, 0, 435256.41025641, 1.90705128205128, 0.673076923076923, -1720.09602617521, 161538.461538462, 457692.307692308, -664104540.25641, -89743589.7435897, 116666666.666667, 435256.41025641, 0.673076923076923, 1.90705128205128, -2767.10225106838, -457692.307692308, -161538.461538462, -664104540.25641, -116666666.666667, 89743589.7435897, 435256.41025641, -1.90705128205128, -0.673076923076923, -2767.10225106838;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(30) << -26923.0769230769, 0, 13461538.4615385, 8980906.4358507, -280.448717948718, 4358.45085470085, -0.438034188034188, 0, 224.358974358974, -50641.0256410256, 0, -13461538.4615385, 13469278.5049834, -3271.90170940171, -957.219551282051, -0.237713675213675, 0, 112.179487179487, -26923.0769230769, 0, -35897435.8974359, 8982963.05978232, 280.448717948718, 1731.12446581197, -0.438034188034188, 0, 299.145299145299, -3205.12820512821, 0, -53846153.8461539, 26940582.9544156, -7946.04700854701, -1927.45192307692, -0.0400641025641026, 0, -897.435897435898, 37179.4871794872, 0, 0, -35908371.6630577, 1869.65811965812, -6282.68696581197, 1.19123931623932, 0, 0, 37179.4871794872, 0, -62820512.8205128, -35902014.8254509, 1869.65811965812, -9740.5235042735, 1.19123931623932, 0, 523.504273504273, 16666.6666666667, 0, 89743589.7435897, -26943576.9074071, 2617.52136752137, -2691.67200854701, 1.10576923076923, 0, -747.863247863248, 16666.6666666667, 0, 62820512.8205128, -26927497.847578, 4861.11111111111, 2048.44017094017, 1.10576923076923, 0, -1196.5811965812;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(31) << 0, -26923.0769230769, 35897435.8974359, 280.448717948718, 8982963.05978232, -1731.12446581197, 0, -0.438034188034188, -299.145299145299, 0, -50641.0256410256, 13461538.4615385, -3271.90170940171, 13469278.5049834, 957.219551282051, 0, -0.237713675213675, -112.179487179487, 0, -26923.0769230769, -13461538.4615385, -280.448717948718, 8980906.4358507, -4358.45085470085, 0, -0.438034188034188, -224.358974358974, 0, -3205.12820512821, 53846153.8461539, -7946.04700854701, 26940582.9544156, 1927.45192307692, 0, -0.0400641025641026, 897.435897435898, 0, 37179.4871794872, 62820512.8205128, 1869.65811965812, -35902014.8254509, 9740.5235042735, 0, 1.19123931623932, -523.504273504273, 0, 37179.4871794872, 0, 1869.65811965812, -35908371.6630577, 6282.68696581197, 0, 1.19123931623932, 0, 0, 16666.6666666667, -62820512.8205128, 4861.11111111111, -26927497.847578, -2048.44017094017, 0, 1.10576923076923, 1196.5811965812, 0, 16666.6666666667, -89743589.7435897, 2617.52136752137, -26943576.9074071, 2691.67200854701, 0, 1.10576923076923, 747.863247863248;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(32) << 20192307.6923077, 53846153.8461538, -94230.7692307692, 4358.56303418803, -1731.18055555556, 31413630.3327684, 196.314102564103, -74.7863247863248, -1.53311965811966, -20192307.6923077, 20192307.6923077, -177243.58974359, -957.163461538462, 957.163461538462, 47118823.9978563, 28.0448717948718, -28.0448717948718, -0.831997863247863, -53846153.8461538, -20192307.6923077, -94230.7692307692, 1731.18055555556, -4358.56303418803, 31413630.3327684, 74.7863247863248, -196.314102564103, -1.53311965811966, -80769230.7692308, 80769230.7692308, -11217.9487179487, -1927.45192307692, 1927.45192307692, 94238561.084844, -785.25641025641, 785.25641025641, -0.140224358974359, 0, 94230769.2307692, 130128.205128205, -6283.39743589744, 9739.77564102564, -125644505.395044, 0, -130.876068376068, 4.16933760683761, -94230769.2307692, 0, 130128.205128205, -9739.77564102564, 6283.39743589744, -125644505.395044, 130.876068376068, 0, 4.16933760683761, 134615384.615385, -94230769.2307692, 58333.3333333333, -2690.96153846154, -2047.46794871795, -94236323.3284338, -186.965811965812, 579.594017094017, 3.87019230769231, 94230769.2307692, -134615384.615385, 58333.3333333333, 2047.46794871795, 2690.96153846154, -94236323.3284338, -579.594017094017, 186.965811965812, 3.87019230769231;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(33) << 3271.90382278312, -140.224358974359, -0.261752136752137, -0.387286324786325, 0, 196.314102564103, 74.8108945137136, -0.00105168269230769, 0.0363228231837607, 3870.1980045406, -1635.95085470085, 2.15945512820513, -0.435363247863248, 0, -28.0448717948718, 112.208510726487, -0.0122696314102564, -0.00799662459935898, 4300.21578859509, 140.224358974359, 0.177617521367521, -0.387286324786325, 0, -74.7863247863248, 74.8186068534572, 0.00105168269230769, 0.0144244090544872, 8750.00284882479, -3973.0235042735, -2.1875, -0.0400641025641026, 0, -785.25641025641, 224.424649594016, -0.0297976762820513, -0.016042047275641, -5459.40524679487, 934.82905982906, -0.31784188034188, 0.827991452991453, 0, 0, -299.186385977385, 0.00701121794871795, -0.0523528111645299, -2280.98644337607, 934.82905982906, 1.53311965811966, 0.827991452991453, 0, -130.876068376068, -299.16254783636, 0.00701121794871795, -0.0811850827991453, -10245.7293445513, 1308.76068376068, 0.31784188034188, 0.657051282051282, 0, 186.965811965812, -224.435888376067, 0.00981570512820513, -0.0224335136217949, -2206.19943002137, 2430.55555555556, -1.42094017094017, 0.657051282051282, 0, -205.662393162393, -224.375591901708, 0.0182291666666667, 0.017083360042735;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(34) << 140.224358974359, 4300.21578859509, -0.177617521367521, 0, -0.387286324786325, 74.7863247863248, 0.00105168269230769, 74.8186068534572, -0.0144244090544872, -1635.95085470085, 3870.1980045406, -2.15945512820513, 0, -0.435363247863248, 28.0448717948718, -0.0122696314102564, 112.208510726487, 0.00799662459935898, -140.224358974359, 3271.90382278312, 0.261752136752137, 0, -0.387286324786325, -196.314102564103, -0.00105168269230769, 74.8108945137136, -0.0363228231837607, -3973.0235042735, 8750.00284882479, 2.1875, 0, -0.0400641025641026, 785.25641025641, -0.0297976762820513, 224.424649594016, 0.016042047275641, 934.82905982906, -2280.98644337607, -1.53311965811966, 0, 0.827991452991453, 130.876068376068, 0.00701121794871795, -299.16254783636, 0.0811850827991453, 934.82905982906, -5459.40524679487, 0.31784188034188, 0, 0.827991452991453, 0, 0.00701121794871795, -299.186385977385, 0.0523528111645299, 2430.55555555556, -2206.19943002137, 1.42094017094017, 0, 0.657051282051282, 205.662393162393, 0.0182291666666667, -224.375591901708, -0.017083360042735, 1308.76068376068, -10245.7293445513, -0.31784188034188, 0, 0.657051282051282, -186.965811965812, 0.00981570512820513, -224.435888376067, 0.0224335136217949;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(35) << -0.205662393162393, -0.205662393162393, 1682.6984354968, 224.358974358974, 299.145299145299, -1.35550213675214, 0.0363232438568376, -0.0144246193910256, 261.764828037412, 2.1875, -2.1875, 1720.10256063034, -112.179487179487, 112.179487179487, -1.52377136752137, -0.00799641426282051, 0.00799641426282051, 392.641099041997, 0.205662393162393, 0.205662393162393, 1682.6984354968, -299.145299145299, -224.358974358974, -1.35550213675214, 0.0144246193910256, -0.0363232438568376, 261.764828037412, -2.1875, 2.1875, 3888.89712286325, -897.435897435898, 897.435897435898, -0.140224358974359, -0.016042047275641, 0.016042047275641, 785.285694150074, -0.673076923076923, -1.90705128205128, -1720.09602617521, 0, 523.504273504273, 2.89797008547009, -0.0523554754273504, 0.0811822783119658, -1047.02177741179, 1.90705128205128, 0.673076923076923, -1720.09602617521, -523.504273504273, 0, 2.89797008547009, -0.0811822783119658, 0.0523554754273504, -1047.02177741179, 0.673076923076923, 1.90705128205128, -2767.10225106838, 747.863247863248, -299.145299145299, 2.29967948717949, -0.0224308493589744, -0.0170797142094017, -785.277329299647, -1.90705128205128, -0.673076923076923, -2767.10225106838, 299.145299145299, -747.863247863248, 2.29967948717949, 0.0170797142094017, 0.0224308493589744, -785.277329299647;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(36) << -2458976181.66667, -583333333.333333, 0, 1282.05128205128, 0, 134615384.615385, -10245.7340902778, -2430.55555555556, 0, -2458979298.46154, 583333333.333333, -3096153.84615385, -97435.8974358974, 0, -134615384.615385, -10245.7470769231, 2430.55555555556, -12.900641025641, -1310257199.10256, -224358974.358974, -296153.846153846, 46153.8461538462, 0, 0, -5459.40499626068, -934.82905982906, -1.23397435897436, -1310257259.23077, 224358974.358974, -161538.461538462, 37179.4871794872, 0, 0, -5459.40524679487, 934.82905982906, -0.673076923076923, 5456416107.69231, 0, 1884615.38461538, -128205.128205128, 0, 0, 22735.0671153846, 0, 7.8525641025641, 689.230769230769, -897435897.435898, -215384.615384615, -215384.615384615, 0, -269230769.230769, 0.00287179487179487, -3739.31623931624, -0.897435897435898, 2082052412.82051, 0, 412820.512820513, -128205.128205128, 0, 0, 8675.21838675214, 0, 1.72008547008547, 728.717948717949, 897435897.435898, 1471794.87179487, -179487.17948718, 0, 269230769.230769, 0.00303632478632479, 3739.31623931624, 6.13247863247863;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(37) << -314102564.102564, -529489002.179487, 125641.025641026, 0, 1282.05128205128, 94230769.2307692, -1308.76068376068, -2206.20417574786, 0.523504273504274, 314102564.102564, -529492118.974359, 403846.153846154, 0, -97435.8974358974, 94230769.2307692, 1308.76068376068, -2206.21716239316, 1.68269230769231, -224358974.358974, -547436686.282051, -430769.230769231, 0, 46153.8461538462, 94230769.2307692, -934.82905982906, -2280.98619284188, -1.79487179487179, 224358974.358974, -547436746.410256, -457692.307692308, 0, 37179.4871794872, 94230769.2307692, 934.82905982906, -2280.98644337607, -1.90705128205128, 0, 2943595594.87179, -538461.538461539, 0, -128205.128205128, -323076923.076923, 0, 12264.9816452991, -2.24358974358974, -897435897.435898, 689.230769230769, 700000, 0, -215384.615384615, -269230769.230769, -3739.31623931624, 0.00287179487179487, 2.91666666666667, 0, -789742458.974359, -179487.17948718, 0, -128205.128205128, -323076923.076923, 0, -3290.59357905983, -0.747863247863248, 897435897.435898, 728.717948717949, 376923.076923077, 0, -179487.17948718, -269230769.230769, 3739.31623931624, 0.00303632478632479, 1.57051282051282;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(38) << -4487.17948717949, 332051.282051282, -664107800.641026, 89743589.7435897, 62820512.8205128, 4487.17948717949, -0.0186965811965812, 1.38354700854701, -2767.11583600427, -2598076.92307692, 915384.615384615, -664117382.564103, -89743589.7435897, 62820512.8205128, -341025.641025641, -10.8253205128205, 3.81410256410256, -2767.15576068376, -228846.153846154, -305128.205128205, -412823070.512821, 0, 62820512.8205128, 161538.461538462, -0.953525641025641, -1.27136752136752, -1720.09612713675, -76282.0512820513, -367948.717948718, -412823046.282051, 0, 62820512.8205128, 130128.205128205, -0.31784188034188, -1.53311965811966, -1720.09602617521, 1884615.38461538, -538461.538461539, 1866682777.4359, 0, -215384615.384615, -448717.948717949, 7.8525641025641, -2.24358974358974, 7777.84490598291, -700000, 215384.615384615, 3483.84615384615, -179487179.48718, -179487179.48718, -753846.153846154, -2.91666666666667, 0.897435897435898, 0.0145160256410256, 305128.205128205, -394871.794871795, 287182868.717949, 0, -215384615.384615, -448717.948717949, 1.27136752136752, -1.64529914529915, 1196.59528632479, 1417948.71794872, 143589.743589744, 2170, 179487179.48718, -179487179.48718, -628205.128205128, 5.90811965811966, 0.598290598290598, 0.00904166666666667;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(39) << 108974.358974359, 0, -89743589.7435897, -26943596.5322831, -4861.11111111111, -17051.2820512821, 0.46474358974359, 0, 747.863247863248, 333333.333333333, 0, 89743589.7435897, -26943528.2890256, 4861.11111111111, 15230.608974359, 0.576923076923077, 0, -747.863247863248, 115384.615384615, 0, 0, -35908379.3548643, -1869.65811965812, -9874.26282051282, 0.865384615384615, 0, 0, 124358.974358974, 0, 0, -35908371.6630577, 1869.65811965812, -6283.39743589744, 0.827991452991453, 0, 0, -128205.128205128, 0, 0, 143635354.236795, 0, 64631.0897435898, -1.6025641025641, 0, 0, -215384.615384615, 0, 179487179.48718, 89743666.6724103, -7478.63247863248, 39485.3846153846, -2.69230769230769, 0, -1495.7264957265, -158974.358974359, 0, 0, 71812250.1803633, 0, 7182.92735042735, -1.73076923076923, 0, 0, -179487.17948718, 0, -179487179.48718, 89743623.9804316, 7478.63247863248, -3577.47863247863, -2.24358974358974, 0, 1495.7264957265;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(40) << 0, 108974.358974359, -116666666.666667, -2617.52136752137, -26927517.4724541, 12308.7393162393, 0, 0.46474358974359, 299.145299145299, 0, 333333.333333333, -116666666.666667, 2617.52136752137, -26927449.2291966, -11022.2756410256, 0, 0.576923076923077, 299.145299145299, 0, 115384.615384615, -62820512.8205128, -1869.65811965812, -35902022.5172575, 13329.7435897436, 0, 0.865384615384615, 523.504273504273, 0, 124358.974358974, -62820512.8205128, 1869.65811965812, -35902014.8254509, 9739.77564102564, 0, 0.827991452991453, 523.504273504273, 0, -128205.128205128, -215384615.384615, 0, 143614414.065855, -82055.7692307692, 0, -1.6025641025641, -3589.74358974359, 0, -215384.615384615, 179487179.48718, -7478.63247863248, 89743666.6724103, -39481.3461538462, 0, -2.69230769230769, -1495.7264957265, 0, -158974.358974359, 215384615.384615, 0, 71788318.5564316, -5129.70085470086, 0, -1.73076923076923, -1794.87179487179, 0, -179487.17948718, 179487179.48718, 7478.63247863248, 89743623.9804316, -14355.8333333333, 0, -2.24358974358974, -1495.7264957265;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(41) << -134615384.615385, -175000000, 381410.256410256, -17051.3194444444, 12310.4594017094, -94236369.1248344, 186.965811965812, -205.662393162393, 1.62660256410256, 134615384.615385, -175000000, 1166666.66666667, 15234.7596153846, -11018.0128205128, -94236209.909812, -186.965811965812, -205.662393162393, 2.01923076923077, 0, -94230769.2307692, 403846.153846154, -9873.70192307692, 13330.7905982906, -125644523.343964, 0, 130.876068376068, 3.02884615384615, 0, -94230769.2307692, 435256.41025641, -6282.68696581197, 9740.5235042735, -125644505.395044, 0, 130.876068376068, 2.89797008547009, 0, -323076923.076923, -448717.948717949, 64631.0897435898, -82055.7692307692, 502579986.117162, 0, -3141.02564102564, -5.60897435897436, 269230769.230769, 269230769.230769, -753846.153846154, 39481.3461538462, -39485.3846153846, 314102743.618776, -373.931623931624, -373.931623931624, -9.42307692307692, 0, 323076923.076923, -556410.256410256, 7182.02991452992, -5131.49572649573, 251284509.686299, 0, -448.717948717949, -6.05769230769231, -269230769.230769, 269230769.230769, -628205.128205128, -3577.92735042735, -14357.7777777778, 314102643.992442, 373.931623931624, -373.931623931624, -7.8525641025641;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(42) << -10245.7340902778, -2430.55555555556, 0, 0.913461538461539, 0, -186.965811965812, -224.436051873198, -0.0182291666666667, -0.142094017094017, -10245.7470769231, 2430.55555555556, -12.900641025641, 2.37179487179487, 0, 186.965811965812, -224.435483060342, 0.0182291666666667, 0.12703999732906, -5459.40499626068, -934.82905982906, -1.23397435897436, 1.15384615384615, 0, 0, -299.18645007807, -0.00701121794871795, -0.0822742120726496, -5459.40524679487, 934.82905982906, -0.673076923076923, 1.19123931623932, 0, 0, -299.186385977385, 0.00701121794871795, -0.0523554754273504, 22735.0671153846, 0, 7.8525641025641, -1.6025641025641, 0, 0, 1196.75288052473, 0, 0.538520432692308, 0.00287179487179487, -3739.31623931624, -0.897435897435898, -2.69230769230769, 0, 373.931623931624, 747.863888910427, -0.0280448717948718, 0.329053098290598, 8675.21838675214, 0, 1.72008547008547, -1.85897435897436, 0, 0, 598.355895334482, 0, 0.0598419604700855, 0.00303632478632479, 3739.31623931624, 6.13247863247863, -2.24358974358974, 0, -373.931623931624, 747.863533142431, 0.0280448717948718, -0.0298685363247863;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(43) << -1308.76068376068, -2206.20417574786, 0.523504273504274, 0, 0.913461538461539, -579.594017094017, -0.00981570512820513, -224.37575539884, 0.102568028846154, 1308.76068376068, -2206.21716239316, 1.68269230769231, 0, 2.37179487179487, -579.594017094017, 0.00981570512820513, -224.375186585983, -0.0918677216880342, -934.82905982906, -2280.98619284188, -1.79487179487179, 0, 1.15384615384615, -130.876068376068, -0.00701121794871795, -299.162611937045, 0.11109764957265, 934.82905982906, -2280.98644337607, -1.90705128205128, 0, 1.19123931623932, -130.876068376068, 0.00701121794871795, -299.16254783636, 0.0811822783119658, 0, 12264.9816452991, -2.24358974358974, 0, -1.6025641025641, -3141.02564102564, 0, 1196.67435488371, -0.683777510683761, -3739.31623931624, 0.00287179487179487, 2.91666666666667, 0, -2.69230769230769, 373.931623931624, -0.0280448717948718, 747.863888910427, -0.329037954059829, 0, -3290.59357905983, -0.747863247863248, 0, -1.85897435897436, 448.717948717949, 0, 598.266151744738, -0.0427406517094017, 3739.31623931624, 0.00303632478632479, 1.57051282051282, 0, -2.24358974358974, 373.931623931624, 0.0280448717948718, 747.863533142431, -0.119646340811966;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(44) << -0.0186965811965812, 1.38354700854701, -2767.11583600427, -747.863247863248, -1196.5811965812, 3.19711538461538, -0.142094157318376, 0.102574479166667, -785.27771081179, -10.8253205128205, 3.81410256410256, -2767.15576068376, 747.863247863248, -1196.5811965812, 8.30128205128205, 0.127055562232906, -0.0918517361111111, -785.27638365396, -0.953525641025641, -1.27136752136752, -1720.09612713675, 0, -523.504273504273, 4.03846153846154, -0.082272108707265, 0.111101575854701, -1047.0219269852, -0.31784188034188, -1.53311965811966, -1720.09602617521, 0, -523.504273504273, 4.16933760683761, -0.0523528111645299, 0.0811850827991453, -1047.02177741179, 7.8525641025641, -2.24358974358974, 7777.84490598291, 0, -3589.74358974359, -5.60897435897436, 0.538520432692308, -0.683777510683761, 4188.09525406472, -2.91666666666667, 0.897435897435898, 0.0145160256410256, 1495.7264957265, 1495.7264957265, -9.42307692307692, 0.329037954059829, -0.329053098290598, 2617.52286335673, 1.27136752136752, -1.64529914529915, 1196.59528632479, 0, 1794.87179487179, -6.50641025641026, 0.0598385950854701, -0.0427473824786325, 2094.02661192904, 5.90811965811966, 0.598290598290598, 0.00904166666666667, -1495.7264957265, 1495.7264957265, -7.8525641025641, -0.029870219017094, -0.119653632478632, 2617.52203318747;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(45) << -547436686.282051, -224358974.358974, 430769.230769231, 46153.8461538462, 0, -94230769.2307692, -2280.98619284188, -934.82905982906, 1.79487179487179, -529492118.974359, 314102564.102564, -403846.153846154, -97435.8974358974, 0, -94230769.2307692, -2206.21716239316, 1308.76068376068, -1.68269230769231, -529489002.179487, -314102564.102564, -125641.025641026, 1282.05128205128, 0, -94230769.2307692, -2206.20417574786, -1308.76068376068, -0.523504273504274, -547436746.410256, 224358974.358974, 457692.307692308, 37179.4871794872, 0, -94230769.2307692, -2280.98644337607, 934.82905982906, 1.90705128205128, 689.230769230769, -897435897.435898, -700000, -215384.615384615, 0, 269230769.230769, 0.00287179487179487, -3739.31623931624, -2.91666666666667, 2943595594.87179, 0, 538461.538461539, -128205.128205128, 0, 323076923.076923, 12264.9816452991, 0, 2.24358974358974, 728.717948717949, 897435897.435898, -376923.076923077, -179487.17948718, 0, 269230769.230769, 0.00303632478632479, 3739.31623931624, -1.57051282051282, -789742458.974359, 0, 179487.17948718, -128205.128205128, 0, 323076923.076923, -3290.59357905983, 0, 0.747863247863248;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(46) << -224358974.358974, -1310257199.10256, 296153.846153846, 0, 46153.8461538462, 0, -934.82905982906, -5459.40499626068, 1.23397435897436, 583333333.333333, -2458979298.46154, 3096153.84615385, 0, -97435.8974358974, 134615384.615385, 2430.55555555556, -10245.7470769231, 12.900641025641, -583333333.333333, -2458976181.66667, 0, 0, 1282.05128205128, -134615384.615385, -2430.55555555556, -10245.7340902778, 0, 224358974.358974, -1310257259.23077, 161538.461538462, 0, 37179.4871794872, 0, 934.82905982906, -5459.40524679487, 0.673076923076923, -897435897.435898, 689.230769230769, 215384.615384615, 0, -215384.615384615, 269230769.230769, -3739.31623931624, 0.00287179487179487, 0.897435897435898, 0, 5456416107.69231, -1884615.38461538, 0, -128205.128205128, 0, 0, 22735.0671153846, -7.8525641025641, 897435897.435898, 728.717948717949, -1471794.87179487, 0, -179487.17948718, -269230769.230769, 3739.31623931624, 0.00303632478632479, -6.13247863247863, 0, 2082052412.82051, -412820.512820513, 0, -128205.128205128, 0, 0, 8675.21838675214, -1.72008547008547;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(47) << 305128.205128205, 228846.153846154, -412823070.512821, -62820512.8205128, 0, 161538.461538462, 1.27136752136752, 0.953525641025641, -1720.09612713675, -915384.615384615, 2598076.92307692, -664117382.564103, -62820512.8205128, 89743589.7435897, -341025.641025641, -3.81410256410256, 10.8253205128205, -2767.15576068376, -332051.282051282, 4487.17948717949, -664107800.641026, -62820512.8205128, -89743589.7435897, 4487.17948717949, -1.38354700854701, 0.0186965811965812, -2767.11583600427, 367948.717948718, 76282.0512820513, -412823046.282051, -62820512.8205128, 0, 130128.205128205, 1.53311965811966, 0.31784188034188, -1720.09602617521, -215384.615384615, 700000, 3483.84615384615, 179487179.48718, 179487179.48718, -753846.153846154, -0.897435897435898, 2.91666666666667, 0.0145160256410256, 538461.538461539, -1884615.38461538, 1866682777.4359, 215384615.384615, 0, -448717.948717949, 2.24358974358974, -7.8525641025641, 7777.84490598291, -143589.743589744, -1417948.71794872, 2170, 179487179.48718, -179487179.48718, -628205.128205128, -0.598290598290598, -5.90811965811966, 0.00904166666666667, 394871.794871795, -305128.205128205, 287182868.717949, 215384615.384615, 0, -448717.948717949, 1.64529914529915, -1.27136752136752, 1196.59528632479;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(48) << 115384.615384615, 0, 62820512.8205128, -35902022.5172575, -1869.65811965812, -13329.7435897436, 0.865384615384615, 0, -523.504273504273, 333333.333333333, 0, 116666666.666667, -26927449.2291966, 2617.52136752137, 11022.2756410256, 0.576923076923077, 0, -299.145299145299, 108974.358974359, 0, 116666666.666667, -26927517.4724541, -2617.52136752137, -12308.7393162393, 0.46474358974359, 0, -299.145299145299, 124358.974358974, 0, 62820512.8205128, -35902014.8254509, 1869.65811965812, -9739.77564102564, 0.827991452991453, 0, -523.504273504273, -215384.615384615, 0, -179487179.48718, 89743666.6724103, -7478.63247863248, 39481.3461538462, -2.69230769230769, 0, 1495.7264957265, -128205.128205128, 0, 215384615.384615, 143614414.065855, 0, 82055.7692307692, -1.6025641025641, 0, 3589.74358974359, -179487.17948718, 0, -179487179.48718, 89743623.9804316, 7478.63247863248, 14355.8333333333, -2.24358974358974, 0, 1495.7264957265, -158974.358974359, 0, -215384615.384615, 71788318.5564316, 0, 5129.70085470086, -1.73076923076923, 0, 1794.87179487179;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(49) << 0, 115384.615384615, 0, -1869.65811965812, -35908379.3548643, 9874.26282051282, 0, 0.865384615384615, 0, 0, 333333.333333333, -89743589.7435897, 4861.11111111111, -26943528.2890256, -15230.608974359, 0, 0.576923076923077, 747.863247863248, 0, 108974.358974359, 89743589.7435897, -4861.11111111111, -26943596.5322831, 17051.2820512821, 0, 0.46474358974359, -747.863247863248, 0, 124358.974358974, 0, 1869.65811965812, -35908371.6630577, 6283.39743589744, 0, 0.827991452991453, 0, 0, -215384.615384615, -179487179.48718, -7478.63247863248, 89743666.6724103, -39485.3846153846, 0, -2.69230769230769, 1495.7264957265, 0, -128205.128205128, 0, 0, 143635354.236795, -64631.0897435898, 0, -1.6025641025641, 0, 0, -179487.17948718, 179487179.48718, 7478.63247863248, 89743623.9804316, 3577.47863247863, 0, -2.24358974358974, -1495.7264957265, 0, -158974.358974359, 0, 0, 71812250.1803633, -7182.92735042735, 0, -1.73076923076923, 0;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(50) << 94230769.2307692, 0, 403846.153846154, -13330.7905982906, 9873.70192307692, -125644523.343964, -130.876068376068, 0, 3.02884615384615, 175000000, -134615384.615385, 1166666.66666667, 11018.0128205128, -15234.7596153846, -94236209.909812, 205.662393162393, 186.965811965812, 2.01923076923077, 175000000, 134615384.615385, 381410.256410256, -12310.4594017094, 17051.3194444444, -94236369.1248344, 205.662393162393, -186.965811965812, 1.62660256410256, 94230769.2307692, 0, 435256.41025641, -9740.5235042735, 6282.68696581197, -125644505.395044, -130.876068376068, 0, 2.89797008547009, -269230769.230769, -269230769.230769, -753846.153846154, 39485.3846153846, -39481.3461538462, 314102743.618776, 373.931623931624, 373.931623931624, -9.42307692307692, 323076923.076923, 0, -448717.948717949, 82055.7692307692, -64631.0897435898, 502579986.117162, 3141.02564102564, 0, -5.60897435897436, -269230769.230769, 269230769.230769, -628205.128205128, 14357.7777777778, 3577.92735042735, 314102643.992442, 373.931623931624, -373.931623931624, -7.8525641025641, -323076923.076923, 0, -556410.256410256, 5131.49572649573, -7182.02991452992, 251284509.686299, 448.717948717949, 0, -6.05769230769231;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(51) << -2280.98619284188, -934.82905982906, 1.79487179487179, 1.15384615384615, 0, 130.876068376068, -299.162611937045, -0.00701121794871795, -0.11109764957265, -2206.21716239316, 1308.76068376068, -1.68269230769231, 2.37179487179487, 0, 579.594017094017, -224.375186585983, 0.00981570512820513, 0.0918677216880342, -2206.20417574786, -1308.76068376068, -0.523504273504274, 0.913461538461539, 0, 579.594017094017, -224.37575539884, -0.00981570512820513, -0.102568028846154, -2280.98644337607, 934.82905982906, 1.90705128205128, 1.19123931623932, 0, 130.876068376068, -299.16254783636, 0.00701121794871795, -0.0811822783119658, 0.00287179487179487, -3739.31623931624, -2.91666666666667, -2.69230769230769, 0, -373.931623931624, 747.863888910427, -0.0280448717948718, 0.329037954059829, 12264.9816452991, 0, 2.24358974358974, -1.6025641025641, 0, 3141.02564102564, 1196.67435488371, 0, 0.683777510683761, 0.00303632478632479, 3739.31623931624, -1.57051282051282, -2.24358974358974, 0, -373.931623931624, 747.863533142431, 0.0280448717948718, 0.119646340811966, -3290.59357905983, 0, 0.747863247863248, -1.85897435897436, 0, -448.717948717949, 598.266151744738, 0, 0.0427406517094017;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(52) << -934.82905982906, -5459.40499626068, 1.23397435897436, 0, 1.15384615384615, 0, -0.00701121794871795, -299.18645007807, 0.0822742120726496, 2430.55555555556, -10245.7470769231, 12.900641025641, 0, 2.37179487179487, -186.965811965812, 0.0182291666666667, -224.435483060342, -0.12703999732906, -2430.55555555556, -10245.7340902778, 0, 0, 0.913461538461539, 186.965811965812, -0.0182291666666667, -224.436051873198, 0.142094017094017, 934.82905982906, -5459.40524679487, 0.673076923076923, 0, 1.19123931623932, 0, 0.00701121794871795, -299.186385977385, 0.0523554754273504, -3739.31623931624, 0.00287179487179487, 0.897435897435898, 0, -2.69230769230769, -373.931623931624, -0.0280448717948718, 747.863888910427, -0.329053098290598, 0, 22735.0671153846, -7.8525641025641, 0, -1.6025641025641, 0, 0, 1196.75288052473, -0.538520432692308, 3739.31623931624, 0.00303632478632479, -6.13247863247863, 0, -2.24358974358974, 373.931623931624, 0.0280448717948718, 747.863533142431, 0.0298685363247863, 0, 8675.21838675214, -1.72008547008547, 0, -1.85897435897436, 0, 0, 598.355895334482, -0.0598419604700855;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(53) << 1.27136752136752, 0.953525641025641, -1720.09612713675, 523.504273504273, 0, 4.03846153846154, -0.111101575854701, 0.082272108707265, -1047.0219269852, -3.81410256410256, 10.8253205128205, -2767.15576068376, 1196.5811965812, -747.863247863248, 8.30128205128205, 0.0918517361111111, -0.127055562232906, -785.27638365396, -1.38354700854701, 0.0186965811965812, -2767.11583600427, 1196.5811965812, 747.863247863248, 3.19711538461538, -0.102574479166667, 0.142094157318376, -785.27771081179, 1.53311965811966, 0.31784188034188, -1720.09602617521, 523.504273504273, 0, 4.16933760683761, -0.0811850827991453, 0.0523528111645299, -1047.02177741179, -0.897435897435898, 2.91666666666667, 0.0145160256410256, -1495.7264957265, -1495.7264957265, -9.42307692307692, 0.329053098290598, -0.329037954059829, 2617.52286335673, 2.24358974358974, -7.8525641025641, 7777.84490598291, 3589.74358974359, 0, -5.60897435897436, 0.683777510683761, -0.538520432692308, 4188.09525406472, -0.598290598290598, -5.90811965811966, 0.00904166666666667, -1495.7264957265, 1495.7264957265, -7.8525641025641, 0.119653632478632, 0.029870219017094, 2617.52203318747, 1.64529914529915, -1.27136752136752, 1196.59528632479, -1794.87179487179, 0, -6.50641025641026, 0.0427473824786325, -0.0598385950854701, 2094.02661192904;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(54) << -1310257051.02564, -224358974.358974, 0, 34615.3846153846, 0, 0, -5459.4043792735, -934.82905982906, 0, -1310258268.84615, 224358974.358974, -978205.128205128, -10256.4102564103, 0, 0, -5459.40945352564, 934.82905982906, -4.0758547008547, -2458975022.05128, -583333333.333333, -296153.846153846, 43589.7435897436, 0, -134615384.615385, -10245.729258547, -2430.55555555556, -1.23397435897436, -2458975042.69231, 583333333.333333, 161538.461538462, 16666.6666666667, 0, 134615384.615385, -10245.7293445513, 2430.55555555556, 0.673076923076923, 2082052412.82051, 0, 305128.205128205, -158974.358974359, 0, 0, 8675.21838675214, 0, 1.27136752136752, 728.717948717949, 897435897.435898, -143589.743589744, -179487.17948718, 0, -269230769.230769, 0.00303632478632479, 3739.31623931624, -0.598290598290598, 5456411733.33333, 0, -89743.5897435897, -15384.6153846154, 0, 0, 22735.0488888889, 0, -0.373931623931624, 509.74358974359, -897435897.435898, 1041025.64102564, -143589.743589744, 0, 269230769.230769, 0.00212393162393162, -3739.31623931624, 4.33760683760684;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(55) << -224358974.358974, -547436538.205128, 700000, 0, 34615.3846153846, -94230769.2307692, -934.82905982906, -2280.9855758547, 2.91666666666667, 224358974.358974, -547437756.025641, 1498717.94871795, 0, -10256.4102564103, -94230769.2307692, 934.82905982906, -2280.99065010684, 6.24465811965812, -314102564.102564, -529487842.564103, 430769.230769231, 0, 43589.7435897436, -94230769.2307692, -1308.76068376068, -2206.19934401709, 1.79487179487179, 314102564.102564, -529487863.205128, 457692.307692308, 0, 16666.6666666667, -94230769.2307692, 1308.76068376068, -2206.19943002137, 1.90705128205128, 0, -789742458.974359, -394871.794871795, 0, -158974.358974359, 323076923.076923, 0, -3290.59357905983, -1.64529914529915, 897435897.435898, 728.717948717949, -1417948.71794872, 0, -179487.17948718, 269230769.230769, 3739.31623931624, 0.00303632478632479, -5.90811965811966, 0, 2943591220.51282, -179487.179487179, 0, -15384.6153846154, 323076923.076923, 0, 12264.9634188034, -0.747863247863248, -897435897.435898, 509.74358974359, -1094871.79487179, 0, -143589.743589744, 269230769.230769, -3739.31623931624, 0.00212393162393162, -4.56196581196581;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(56) << 4487.17948717949, 700000, -412822352.564103, 0, -62820512.8205128, 121153.846153846, 0.0186965811965812, 2.91666666666667, -1720.09313568376, -955769.230769231, 1462820.51282051, -412826088.589744, 0, -62820512.8205128, -35897.4358974359, -3.98237179487179, 6.09508547008547, -1720.10870245727, -166025.641025641, 367948.717948718, -664104390.384615, -89743589.7435897, -62820512.8205128, 152564.102564103, -0.691773504273504, 1.53311965811966, -2767.10162660256, 76282.0512820513, 341025.641025641, -664104540.25641, 89743589.7435897, -62820512.8205128, 58333.3333333333, 0.31784188034188, 1.42094017094017, -2767.10225106838, 412820.512820513, -179487.17948718, 287182868.717949, 0, 215384615.384615, -556410.256410256, 1.72008547008547, -0.747863247863248, 1196.59528632479, -376923.076923077, -1471794.87179487, 2170, -179487179.48718, 179487179.48718, -628205.128205128, -1.57051282051282, -6.13247863247863, 0.00904166666666667, -89743.5897435897, -179487.179487179, 1866670859.48718, 0, 215384615.384615, -53846.1538461538, -0.373931623931624, -0.747863247863248, 7777.79524786325, 1094871.79487179, -1041025.64102564, 1473.58974358974, 179487179.48718, 179487179.48718, -502564.102564103, 4.56196581196581, -4.33760683760684, 0.00613995726495727;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(57) << 91025.641025641, 0, 0, -35908365.6997842, -1869.65811965812, -897.435897435898, 0.667735042735043, 0, 0, 135897.435897436, 0, 0, -35908358.017625, 1869.65811965812, 2684.15598290598, 0.480769230769231, 0, 0, 115384.615384615, 0, 89743589.7435897, -26943574.7918504, -4861.11111111111, 894.967948717949, 0.844017094017094, 0, -747.863247863248, 124358.974358974, 0, -89743589.7435897, -26943576.9074071, 4861.11111111111, -2690.96153846154, 0.657051282051282, 0, 747.863247863248, -128205.128205128, 0, 0, 71812250.1803633, 0, 7182.02991452992, -1.85897435897436, 0, 0, -179487.17948718, 0, 179487179.48718, 89743623.9804316, 7478.63247863248, 14357.7777777778, -2.24358974358974, 0, -1495.7264957265, -15384.6153846154, 0, 0, 143635260.354188, 0, -7180.23504273504, -0.192307692307692, 0, 0, -143589.743589744, 0, -179487179.48718, 89743612.0555299, -7478.63247863248, -14350.2991452991, -1.7948717948718, 0, 1495.7264957265;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(58) << 0, 91025.641025641, 62820512.8205128, -1869.65811965812, -35902008.8621774, -4609.55128205128, 0, 0.667735042735043, -523.504273504273, 0, 135897.435897436, 62820512.8205128, 1869.65811965812, -35902001.1800182, -8192.63888888889, 0, 0.480769230769231, -523.504273504273, 0, 115384.615384615, 116666666.666667, -2617.52136752137, -26927495.7320214, 3336.92307692308, 0, 0.844017094017094, -299.145299145299, 0, 124358.974358974, 116666666.666667, 2617.52136752137, -26927497.847578, -2047.46794871795, 0, 0.657051282051282, -299.145299145299, 0, -128205.128205128, -215384615.384615, 0, 71788318.5564316, -5131.49572649573, 0, -1.85897435897436, 1794.87179487179, 0, -179487.17948718, -179487179.48718, 7478.63247863248, 89743623.9804316, 3577.92735042735, 0, -2.24358974358974, 1495.7264957265, 0, -15384.6153846154, 215384615.384615, 0, 143614320.183248, 61536.9658119658, 0, -0.192307692307692, 3589.74358974359, 0, -143589.743589744, -179487179.48718, -7478.63247863248, 89743612.0555299, 14349.8504273504, 0, -1.7948717948718, 1495.7264957265;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(59) << 0, 94230769.2307692, 318589.743589744, -897.398504273504, -4609.55128205128, -125644491.479006, 0, -130.876068376068, 2.33707264957265, 0, 94230769.2307692, 475641.025641026, 2684.34294871795, -8192.93803418804, -125644473.561422, 0, -130.876068376068, 1.68269230769231, 134615384.615385, 175000000, 403846.153846154, 896.052350427351, 3336.39957264957, -94236318.3912874, -186.965811965812, 205.662393162393, 2.95405982905983, -134615384.615385, 175000000, 435256.41025641, -2691.67200854701, -2048.44017094017, -94236323.3284338, 186.965811965812, 205.662393162393, 2.29967948717949, 0, -323076923.076923, -448717.948717949, 7182.92735042735, -5129.70085470086, 251284509.686299, 0, 448.717948717949, -6.50641025641026, 269230769.230769, -269230769.230769, -628205.128205128, 14355.8333333333, 3577.47863247863, 314102643.992442, -373.931623931624, 373.931623931624, -7.8525641025641, 0, 323076923.076923, -53846.1538461538, -7180.23504273504, 61536.9658119658, 502579767.043487, 0, 3141.02564102564, -0.673076923076923, -269230769.230769, -269230769.230769, -502564.102564103, -14349.8504273504, 14350.2991452991, 314102616.166126, 373.931623931624, 373.931623931624, -6.28205128205128;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(60) << -5459.4043792735, -934.82905982906, 0, 0.902777777777778, 0, 0, -299.186336291392, -0.00701121794871795, -0.00747863247863248, -5459.40945352564, 934.82905982906, -4.0758547008547, 1.08974358974359, 0, 0, -299.186272226884, 0.00701121794871795, 0.022405328525641, -10245.729258547, -2430.55555555556, -1.23397435897436, 1.14316239316239, 0, 186.965811965812, -224.435870747217, -0.0182291666666667, 0.00746937767094017, -10245.7293445513, 2430.55555555556, 0.673076923076923, 1.10576923076923, 0, -186.965811965812, -224.435888376067, 0.0182291666666667, -0.0224308493589744, 8675.21838675214, 0, 1.27136752136752, -1.73076923076923, 0, 0, 598.355895334482, 0, 0.0598385950854701, 0.00303632478632479, 3739.31623931624, -0.598290598290598, -2.24358974358974, 0, 373.931623931624, 747.863533142431, 0.0280448717948718, 0.119653632478632, 22735.0488888889, 0, -0.373931623931624, -0.192307692307692, 0, 0, 1196.75209833675, 0, -0.0598318643162393, 0.00212393162393162, -3739.31623931624, 4.33760683760684, -1.7948717948718, 0, -373.931623931624, 747.863433776613, -0.0280448717948718, -0.119625587606838;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(61) << -934.82905982906, -2280.9855758547, 2.91666666666667, 0, 0.902777777777778, 130.876068376068, -0.00701121794871795, -299.162498150366, -0.0384396634615385, 934.82905982906, -2280.99065010684, 6.24465811965812, 0, 1.08974358974359, 130.876068376068, 0.00701121794871795, -299.162434085859, -0.0683292334401709, -1308.76068376068, -2206.19934401709, 1.79487179487179, 0, 1.14316239316239, 579.594017094017, -0.00981570512820513, -224.375574272858, 0.0277912393162393, 1308.76068376068, -2206.19943002137, 1.90705128205128, 0, 1.10576923076923, 579.594017094017, 0.00981570512820513, -224.375591901708, -0.0170797142094017, 0, -3290.59357905983, -1.64529914529915, 0, -1.73076923076923, -448.717948717949, 0, 598.266151744738, -0.0427473824786325, 3739.31623931624, 0.00303632478632479, -5.90811965811966, 0, -2.24358974358974, -373.931623931624, 0.0280448717948718, 747.863533142431, 0.029870219017094, 0, 12264.9634188034, -0.747863247863248, 0, -0.192307692307692, 3141.02564102564, 0, 1196.67357269573, 0.512814903846154, -3739.31623931624, 0.00212393162393162, -4.56196581196581, 0, -1.7948717948718, -373.931623931624, -0.0280448717948718, 747.863433776613, 0.11962390491453;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(62) << 0.0186965811965812, 2.91666666666667, -1720.09313568376, 0, 523.504273504273, 3.15972222222222, -0.00747849225427351, -0.0384396634615385, -1047.02166147131, -3.98237179487179, 6.09508547008547, -1720.10870245727, 0, 523.504273504273, 3.81410256410256, 0.0224060296474359, -0.0683303552350427, -1047.02151201541, -0.691773504273504, 1.53311965811966, -2767.10162660256, 747.863247863248, 1196.5811965812, 4.00106837606838, 0.00747344417735043, 0.0277892761752137, -785.277288162484, 0.31784188034188, 1.42094017094017, -2767.10225106838, -747.863247863248, 1196.5811965812, 3.87019230769231, -0.0224335136217949, -0.017083360042735, -785.277329299647, 1.72008547008547, -0.747863247863248, 1196.59528632479, 0, -1794.87179487179, -6.05769230769231, 0.0598419604700855, -0.0427406517094017, 2094.02661192904, -1.57051282051282, -6.13247863247863, 0.00904166666666667, 1495.7264957265, -1495.7264957265, -7.8525641025641, 0.119646340811966, 0.0298685363247863, 2617.52203318747, -0.373931623931624, -0.747863247863248, 7777.79524786325, 0, 3589.74358974359, -0.673076923076923, -0.0598318643162393, 0.512814903846154, 4188.09342890595, 4.56196581196581, -4.33760683760684, 0.00613995726495727, -1495.7264957265, -1495.7264957265, -6.28205128205128, -0.11962390491453, 0.119625587606838, 2617.5218013281;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(63) << -529487842.564103, -314102564.102564, -430769.230769231, 43589.7435897436, 0, 94230769.2307692, -2206.19934401709, -1308.76068376068, -1.79487179487179, -547437756.025641, 224358974.358974, -1498717.94871795, -10256.4102564103, 0, 94230769.2307692, -2280.99065010684, 934.82905982906, -6.24465811965812, -547436538.205128, -224358974.358974, -700000, 34615.3846153846, 0, 94230769.2307692, -2280.9855758547, -934.82905982906, -2.91666666666667, -529487863.205128, 314102564.102564, -457692.307692308, 16666.6666666667, 0, 94230769.2307692, -2206.19943002137, 1308.76068376068, -1.90705128205128, 728.717948717949, 897435897.435898, 1417948.71794872, -179487.17948718, 0, -269230769.230769, 0.00303632478632479, 3739.31623931624, 5.90811965811966, -789742458.974359, 0, 394871.794871795, -158974.358974359, 0, -323076923.076923, -3290.59357905983, 0, 1.64529914529915, 509.74358974359, -897435897.435898, 1094871.79487179, -143589.743589744, 0, -269230769.230769, 0.00212393162393162, -3739.31623931624, 4.56196581196581, 2943591220.51282, 0, 179487.179487179, -15384.6153846154, 0, -323076923.076923, 12264.9634188034, 0, 0.747863247863248;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(64) << -583333333.333333, -2458975022.05128, 296153.846153846, 0, 43589.7435897436, 134615384.615385, -2430.55555555556, -10245.729258547, 1.23397435897436, 224358974.358974, -1310258268.84615, 978205.128205128, 0, -10256.4102564103, 0, 934.82905982906, -5459.40945352564, 4.0758547008547, -224358974.358974, -1310257051.02564, 0, 0, 34615.3846153846, 0, -934.82905982906, -5459.4043792735, 0, 583333333.333333, -2458975042.69231, -161538.461538462, 0, 16666.6666666667, -134615384.615385, 2430.55555555556, -10245.7293445513, -0.673076923076923, 897435897.435898, 728.717948717949, 143589.743589744, 0, -179487.17948718, 269230769.230769, 3739.31623931624, 0.00303632478632479, 0.598290598290598, 0, 2082052412.82051, -305128.205128205, 0, -158974.358974359, 0, 0, 8675.21838675214, -1.27136752136752, -897435897.435898, 509.74358974359, -1041025.64102564, 0, -143589.743589744, -269230769.230769, -3739.31623931624, 0.00212393162393162, -4.33760683760684, 0, 5456411733.33333, 89743.5897435897, 0, -15384.6153846154, 0, 0, 22735.0488888889, 0.373931623931624;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(65) << -367948.717948718, 166025.641025641, -664104390.384615, 62820512.8205128, 89743589.7435897, 152564.102564103, -1.53311965811966, 0.691773504273504, -2767.10162660256, -1462820.51282051, 955769.230769231, -412826088.589744, 62820512.8205128, 0, -35897.4358974359, -6.09508547008547, 3.98237179487179, -1720.10870245727, -700000, -4487.17948717949, -412822352.564103, 62820512.8205128, 0, 121153.846153846, -2.91666666666667, -0.0186965811965812, -1720.09313568376, -341025.641025641, -76282.0512820513, -664104540.25641, 62820512.8205128, -89743589.7435897, 58333.3333333333, -1.42094017094017, -0.31784188034188, -2767.10225106838, 1471794.87179487, 376923.076923077, 2170, -179487179.48718, 179487179.48718, -628205.128205128, 6.13247863247863, 1.57051282051282, 0.00904166666666667, 179487.17948718, -412820.512820513, 287182868.717949, -215384615.384615, 0, -556410.256410256, 0.747863247863248, -1.72008547008547, 1196.59528632479, 1041025.64102564, -1094871.79487179, 1473.58974358974, -179487179.48718, -179487179.48718, -502564.102564103, 4.33760683760684, -4.56196581196581, 0.00613995726495727, 179487.179487179, 89743.5897435897, 1866670859.48718, -215384615.384615, 0, -53846.1538461538, 0.747863247863248, 0.373931623931624, 7777.79524786325;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(66) << 115384.615384615, 0, -116666666.666667, -26927495.7320214, -2617.52136752137, -3336.92307692308, 0.844017094017094, 0, 299.145299145299, 135897.435897436, 0, -62820512.8205128, -35902001.1800182, 1869.65811965812, 8192.63888888889, 0.480769230769231, 0, 523.504273504273, 91025.641025641, 0, -62820512.8205128, -35902008.8621774, -1869.65811965812, 4609.55128205128, 0.667735042735043, 0, 523.504273504273, 124358.974358974, 0, -116666666.666667, -26927497.847578, 2617.52136752137, 2047.46794871795, 0.657051282051282, 0, 299.145299145299, -179487.17948718, 0, 179487179.48718, 89743623.9804316, 7478.63247863248, -3577.92735042735, -2.24358974358974, 0, -1495.7264957265, -128205.128205128, 0, 215384615.384615, 71788318.5564316, 0, 5131.49572649573, -1.85897435897436, 0, -1794.87179487179, -143589.743589744, 0, 179487179.48718, 89743612.0555299, -7478.63247863248, -14349.8504273504, -1.7948717948718, 0, -1495.7264957265, -15384.6153846154, 0, -215384615.384615, 143614320.183248, 0, -61536.9658119658, -0.192307692307692, 0, -3589.74358974359;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(67) << 0, 115384.615384615, -89743589.7435897, -4861.11111111111, -26943574.7918504, -894.967948717949, 0, 0.844017094017094, 747.863247863248, 0, 135897.435897436, 0, 1869.65811965812, -35908358.017625, -2684.15598290598, 0, 0.480769230769231, 0, 0, 91025.641025641, 0, -1869.65811965812, -35908365.6997842, 897.435897435898, 0, 0.667735042735043, 0, 0, 124358.974358974, 89743589.7435897, 4861.11111111111, -26943576.9074071, 2690.96153846154, 0, 0.657051282051282, -747.863247863248, 0, -179487.17948718, -179487179.48718, 7478.63247863248, 89743623.9804316, -14357.7777777778, 0, -2.24358974358974, 1495.7264957265, 0, -128205.128205128, 0, 0, 71812250.1803633, -7182.02991452992, 0, -1.85897435897436, 0, 0, -143589.743589744, 179487179.48718, -7478.63247863248, 89743612.0555299, 14350.2991452991, 0, -1.7948717948718, -1495.7264957265, 0, -15384.6153846154, 0, 0, 143635260.354188, 7180.23504273504, 0, -0.192307692307692, 0;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(68) << -175000000, -134615384.615385, 403846.153846154, -3336.39957264957, -896.052350427351, -94236318.3912874, -205.662393162393, 186.965811965812, 2.95405982905983, -94230769.2307692, 0, 475641.025641026, 8192.93803418804, -2684.34294871795, -125644473.561422, 130.876068376068, 0, 1.68269230769231, -94230769.2307692, 0, 318589.743589744, 4609.55128205128, 897.398504273504, -125644491.479006, 130.876068376068, 0, 2.33707264957265, -175000000, 134615384.615385, 435256.41025641, 2048.44017094017, 2691.67200854701, -94236323.3284338, -205.662393162393, -186.965811965812, 2.29967948717949, 269230769.230769, -269230769.230769, -628205.128205128, -3577.47863247863, -14355.8333333333, 314102643.992442, -373.931623931624, 373.931623931624, -7.8525641025641, 323076923.076923, 0, -448717.948717949, 5129.70085470086, -7182.92735042735, 251284509.686299, -448.717948717949, 0, -6.50641025641026, 269230769.230769, 269230769.230769, -502564.102564103, -14350.2991452991, 14349.8504273504, 314102616.166126, -373.931623931624, -373.931623931624, -6.28205128205128, -323076923.076923, 0, -53846.1538461538, -61536.9658119658, 7180.23504273504, 502579767.043487, -3141.02564102564, 0, -0.673076923076923;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(69) << -2206.19934401709, -1308.76068376068, -1.79487179487179, 1.14316239316239, 0, -579.594017094017, -224.375574272858, -0.00981570512820513, -0.0277912393162393, -2280.99065010684, 934.82905982906, -6.24465811965812, 1.08974358974359, 0, -130.876068376068, -299.162434085859, 0.00701121794871795, 0.0683292334401709, -2280.9855758547, -934.82905982906, -2.91666666666667, 0.902777777777778, 0, -130.876068376068, -299.162498150366, -0.00701121794871795, 0.0384396634615385, -2206.19943002137, 1308.76068376068, -1.90705128205128, 1.10576923076923, 0, -579.594017094017, -224.375591901708, 0.00981570512820513, 0.0170797142094017, 0.00303632478632479, 3739.31623931624, 5.90811965811966, -2.24358974358974, 0, 373.931623931624, 747.863533142431, 0.0280448717948718, -0.029870219017094, -3290.59357905983, 0, 1.64529914529915, -1.73076923076923, 0, 448.717948717949, 598.266151744738, 0, 0.0427473824786325, 0.00212393162393162, -3739.31623931624, 4.56196581196581, -1.7948717948718, 0, 373.931623931624, 747.863433776613, -0.0280448717948718, -0.11962390491453, 12264.9634188034, 0, 0.747863247863248, -0.192307692307692, 0, -3141.02564102564, 1196.67357269573, 0, -0.512814903846154;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(70) << -2430.55555555556, -10245.729258547, 1.23397435897436, 0, 1.14316239316239, -186.965811965812, -0.0182291666666667, -224.435870747217, -0.00746937767094017, 934.82905982906, -5459.40945352564, 4.0758547008547, 0, 1.08974358974359, 0, 0.00701121794871795, -299.186272226884, -0.022405328525641, -934.82905982906, -5459.4043792735, 0, 0, 0.902777777777778, 0, -0.00701121794871795, -299.186336291392, 0.00747863247863248, 2430.55555555556, -10245.7293445513, -0.673076923076923, 0, 1.10576923076923, 186.965811965812, 0.0182291666666667, -224.435888376067, 0.0224308493589744, 3739.31623931624, 0.00303632478632479, 0.598290598290598, 0, -2.24358974358974, -373.931623931624, 0.0280448717948718, 747.863533142431, -0.119653632478632, 0, 8675.21838675214, -1.27136752136752, 0, -1.73076923076923, 0, 0, 598.355895334482, -0.0598385950854701, -3739.31623931624, 0.00212393162393162, -4.33760683760684, 0, -1.7948717948718, 373.931623931624, -0.0280448717948718, 747.863433776613, 0.119625587606838, 0, 22735.0488888889, 0.373931623931624, 0, -0.192307692307692, 0, 0, 1196.75209833675, 0.0598318643162393;
    //Expected_JacobianK_SmallDispNoVelNoDamping.row(71) << -1.53311965811966, 0.691773504273504, -2767.10162660256, -1196.5811965812, -747.863247863248, 4.00106837606838, -0.0277892761752137, -0.00747344417735043, -785.277288162484, -6.09508547008547, 3.98237179487179, -1720.10870245727, -523.504273504273, 0, 3.81410256410256, 0.0683303552350427, -0.0224060296474359, -1047.02151201541, -2.91666666666667, -0.0186965811965812, -1720.09313568376, -523.504273504273, 0, 3.15972222222222, 0.0384396634615385, 0.00747849225427351, -1047.02166147131, -1.42094017094017, -0.31784188034188, -2767.10225106838, -1196.5811965812, 747.863247863248, 3.87019230769231, 0.017083360042735, 0.0224335136217949, -785.277329299647, 6.13247863247863, 1.57051282051282, 0.00904166666666667, 1495.7264957265, -1495.7264957265, -7.8525641025641, -0.0298685363247863, -0.119646340811966, 2617.52203318747, 0.747863247863248, -1.72008547008547, 1196.59528632479, 1794.87179487179, 0, -6.05769230769231, 0.0427406517094017, -0.0598419604700855, 2094.02661192904, 4.33760683760684, -4.56196581196581, 0.00613995726495727, 1495.7264957265, 1495.7264957265, -6.28205128205128, -0.119625587606838, 0.11962390491453, 2617.5218013281, 0.747863247863248, 0.373931623931624, 7777.79524786325, -3589.74358974359, 0, -0.673076923076923, -0.512814903846154, 0.0598318643162393, 4188.09342890595;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianK_SmallDispNoVelNoDamping;
    Expected_JacobianK_SmallDispNoVelNoDamping.resize(72, 72);
    if (!load_validation_data("UT_ANCFShell_3833_JacSmallDispNoVelNoDamping.txt", Expected_JacobianK_SmallDispNoVelNoDamping))
        return false;

    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation
    
    //Setup the test conditions
    ChVector<double> OriginalPos = m_nodeB->GetPos();
    m_nodeB->SetPos(ChVector<>(m_nodeB->GetPos().x(), m_nodeB->GetPos().y(), 0.001));

    ChVectorDynamic<double> InternalForceSmallDispNoVelNoGravity;
    InternalForceSmallDispNoVelNoGravity.resize(72);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceSmallDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_SmallDispNoVelNoDamping;
    JacobianK_SmallDispNoVelNoDamping.resize(72, 72);
    m_element->ComputeKRMmatricesGlobal(JacobianK_SmallDispNoVelNoDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_SmallDispNoVelNoDamping;
    JacobianR_SmallDispNoVelNoDamping.resize(72, 72);
    m_element->ComputeKRMmatricesGlobal(JacobianR_SmallDispNoVelNoDamping, 0, 1, 0);

    //Reset the element conditions back to its original values
    m_nodeB->SetPos(OriginalPos);


    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_SmallDispNoVelNoDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(72, 72);
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

    //ChMatrixNM<double, 72, 72> Expected_JacobianK_NoDispNoVelWithDamping;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(0) << 2100000000, 953525641.025641, 0, 0, 0, -80769230.7692308, 8750, 3973.0235042735, 0, 1032051282.05128, -33653846.1538461, 0, 0, 0, 53846153.8461538, 4300.21367521368, -140.224358974359, 0, 928846153.846154, 392628205.128205, 0, 0, 0, 20192307.6923077, 3870.19230769231, 1635.95085470085, 0, 785256410.25641, 33653846.1538461, 0, 0, 0, 20192307.6923077, 3271.90170940171, 140.224358974359, 0, -2458974358.97436, -314102564.102564, 0, 0, 0, -134615384.615385, -10245.7264957264, -1308.76068376068, 0, -547435897.435898, -224358974.358974, 0, 0, 0, 94230769.2307692, -2280.98290598291, -934.82905982906, 0, -1310256410.25641, -224358974.358974, 0, 0, 0, 0, -5459.40170940171, -934.82905982906, 0, -529487179.48718, -583333333.333333, 0, 0, 0, -175000000, -2206.19658119658, -2430.55555555556, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(1) << 953525641.025641, 2100000000, 0, 0, 0, -80769230.7692308, 3973.0235042735, 8750, 0, 33653846.1538461, 785256410.25641, 0, 0, 0, 20192307.6923077, 140.224358974359, 3271.90170940171, 0, 392628205.128205, 928846153.846154, 0, 0, 0, 20192307.6923077, 1635.95085470085, 3870.19230769231, 0, -33653846.1538461, 1032051282.05128, 0, 0, 0, 53846153.8461538, -140.224358974359, 4300.21367521368, 0, -583333333.333333, -529487179.48718, 0, 0, 0, -175000000, -2430.55555555556, -2206.19658119658, 0, -224358974.358974, -1310256410.25641, 0, 0, 0, 0, -934.82905982906, -5459.40170940171, 0, -224358974.358974, -547435897.435898, 0, 0, 0, 94230769.2307692, -934.82905982906, -2280.98290598291, 0, -314102564.102564, -2458974358.97436, 0, 0, 0, -134615384.615385, -1308.76068376068, -10245.7264957264, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(2) << 0, 0, 933333333.333333, -53846153.8461539, -53846153.8461539, 0, 0, 0, 3888.88888888889, 0, 0, 403846153.846154, 35897435.8974359, 13461538.4615385, 0, 0, 0, 1682.69230769231, 0, 0, 412820512.820513, 13461538.4615385, 13461538.4615385, 0, 0, 0, 1720.08547008547, 0, 0, 403846153.846154, 13461538.4615385, 35897435.8974359, 0, 0, 0, 1682.69230769231, 0, 0, -664102564.102564, -89743589.7435897, -116666666.666667, 0, 0, 0, -2767.09401709402, 0, 0, -412820512.820513, 62820512.8205128, 0, 0, 0, 0, -1720.08547008547, 0, 0, -412820512.820513, 0, 62820512.8205128, 0, 0, 0, -1720.08547008547, 0, 0, -664102564.102564, -116666666.666667, -89743589.7435897, 0, 0, 0, -2767.09401709402;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(3) << 0, 0, -53846153.8461539, 26940576.9230769, 7946.04700854701, 0, 0, 0, -897.435897435898, 0, 0, -35897435.8974359, 8982959.4017094, -280.448717948718, 0, 0, 0, 299.145299145299, 0, 0, -13461538.4615385, 13469278.8461538, 3271.90170940171, 0, 0, 0, 112.179487179487, 0, 0, 13461538.4615385, 8980902.77777778, 280.448717948718, 0, 0, 0, 224.358974358974, 0, 0, 89743589.7435897, -26943568.3760684, -2617.52136752137, 0, 0, 0, -747.863247863248, 0, 0, -62820512.8205128, -35901997.8632479, -1869.65811965812, 0, 0, 0, 523.504273504273, 0, 0, 0, -35908354.7008547, -1869.65811965812, 0, 0, 0, 0, 0, 0, 62820512.8205128, -26927489.3162393, -4861.11111111111, 0, 0, 0, -1196.5811965812;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(4) << 0, 0, -53846153.8461539, 7946.04700854701, 26940576.9230769, 0, 0, 0, -897.435897435898, 0, 0, 13461538.4615385, 280.448717948718, 8980902.77777778, 0, 0, 0, 224.358974358974, 0, 0, -13461538.4615385, 3271.90170940171, 13469278.8461538, 0, 0, 0, 112.179487179487, 0, 0, -35897435.8974359, -280.448717948718, 8982959.4017094, 0, 0, 0, 299.145299145299, 0, 0, 62820512.8205128, -4861.11111111111, -26927489.3162393, 0, 0, 0, -1196.5811965812, 0, 0, 0, -1869.65811965812, -35908354.7008547, 0, 0, 0, 0, 0, 0, -62820512.8205128, -1869.65811965812, -35901997.8632479, 0, 0, 0, 523.504273504273, 0, 0, 89743589.7435897, -2617.52136752137, -26943568.3760684, 0, 0, 0, -747.863247863248;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(5) << -80769230.7692308, -80769230.7692308, 0, 0, 0, 94238547.008547, -785.25641025641, -785.25641025641, 0, -53846153.8461538, 20192307.6923077, 0, 0, 0, 31413621.7948718, 74.7863247863248, 196.314102564103, 0, -20192307.6923077, -20192307.6923077, 0, 0, 0, 47118824.7863248, 28.0448717948718, 28.0448717948718, 0, 20192307.6923077, -53846153.8461538, 0, 0, 0, 31413621.7948718, 196.314102564103, 74.7863247863248, 0, 134615384.615385, 94230769.2307692, 0, 0, 0, -94236303.4188034, -186.965811965812, -579.594017094017, 0, -94230769.2307692, 0, 0, 0, 0, -125644465.811966, 130.876068376068, 0, 0, 0, -94230769.2307692, 0, 0, 0, -125644465.811966, 0, 130.876068376068, 0, 94230769.2307692, 134615384.615385, 0, 0, 0, -94236303.4188034, -579.594017094017, -186.965811965812, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(6) << 8750, 3973.0235042735, 0, 0, 0, -785.25641025641, 224.424599358974, 0.0297976762820513, 0, 4300.21367521368, -140.224358974359, 0, 0, 0, -74.7863247863248, 74.8185763888889, -0.00105168269230769, 0, 3870.19230769231, 1635.95085470085, 0, 0, 0, -28.0448717948718, 112.208513621795, 0.0122696314102564, 0, 3271.90170940171, 140.224358974359, 0, 0, 0, 196.314102564103, 74.8108640491453, 0.00105168269230769, 0, -10245.7264957264, -1308.76068376068, 0, 0, 0, 186.965811965812, -224.435817307692, -0.00981570512820513, 0, -2280.98290598291, -934.82905982906, 0, 0, 0, -130.876068376068, -299.162406517094, -0.00701121794871795, 0, -5459.40170940171, -934.82905982906, 0, 0, 0, 0, -299.18624465812, -0.00701121794871795, 0, -2206.19658119658, -2430.55555555556, 0, 0, 0, -205.662393162393, -224.375520833333, -0.0182291666666667, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(7) << 3973.0235042735, 8750, 0, 0, 0, -785.25641025641, 0.0297976762820513, 224.424599358974, 0, 140.224358974359, 3271.90170940171, 0, 0, 0, 196.314102564103, 0.00105168269230769, 74.8108640491453, 0, 1635.95085470085, 3870.19230769231, 0, 0, 0, -28.0448717948718, 0.0122696314102564, 112.208513621795, 0, -140.224358974359, 4300.21367521368, 0, 0, 0, -74.7863247863248, -0.00105168269230769, 74.8185763888889, 0, -2430.55555555556, -2206.19658119658, 0, 0, 0, -205.662393162393, -0.0182291666666667, -224.375520833333, 0, -934.82905982906, -5459.40170940171, 0, 0, 0, 0, -0.00701121794871795, -299.18624465812, 0, -934.82905982906, -2280.98290598291, 0, 0, 0, -130.876068376068, -0.00701121794871795, -299.162406517094, 0, -1308.76068376068, -10245.7264957264, 0, 0, 0, 186.965811965812, -0.00981570512820513, -224.435817307692, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(8) << 0, 0, 3888.88888888889, -897.435897435898, -897.435897435898, 0, 0, 0, 785.285576923077, 0, 0, 1682.69230769231, -299.145299145299, 224.358974358974, 0, 0, 0, 261.764756944444, 0, 0, 1720.08547008547, -112.179487179487, -112.179487179487, 0, 0, 0, 392.641105769231, 0, 0, 1682.69230769231, 224.358974358974, -299.145299145299, 0, 0, 0, 261.764756944444, 0, 0, -2767.09401709402, 747.863247863248, 299.145299145299, 0, 0, 0, -785.277163461538, 0, 0, -1720.08547008547, -523.504273504273, 0, 0, 0, 0, -1047.02144764957, 0, 0, -1720.08547008547, 0, -523.504273504273, 0, 0, 0, -1047.02144764957, 0, 0, -2767.09401709402, 299.145299145299, 747.863247863248, 0, 0, 0, -785.277163461538;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(9) << 1032051282.05128, 33653846.1538461, 0, 0, 0, -53846153.8461538, 4300.21367521368, 140.224358974359, 0, 2100000000, -953525641.025641, 0, 0, 0, 80769230.7692308, 8750, -3973.0235042735, 0, 785256410.25641, -33653846.1538461, 0, 0, 0, -20192307.6923077, 3271.90170940171, -140.224358974359, 0, 928846153.846154, -392628205.128205, 0, 0, 0, -20192307.6923077, 3870.19230769231, -1635.95085470085, 0, -2458974358.97436, 314102564.102564, 0, 0, 0, 134615384.615385, -10245.7264957264, 1308.76068376068, 0, -529487179.48718, 583333333.333333, 0, 0, 0, 175000000, -2206.19658119658, 2430.55555555556, 0, -1310256410.25641, 224358974.358974, 0, 0, 0, 0, -5459.40170940171, 934.82905982906, 0, -547435897.435898, 224358974.358974, 0, 0, 0, -94230769.2307692, -2280.98290598291, 934.82905982906, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(10) << -33653846.1538461, 785256410.25641, 0, 0, 0, 20192307.6923077, -140.224358974359, 3271.90170940171, 0, -953525641.025641, 2100000000, 0, 0, 0, -80769230.7692308, -3973.0235042735, 8750, 0, 33653846.1538461, 1032051282.05128, 0, 0, 0, 53846153.8461538, 140.224358974359, 4300.21367521368, 0, -392628205.128205, 928846153.846154, 0, 0, 0, 20192307.6923077, -1635.95085470085, 3870.19230769231, 0, 583333333.333333, -529487179.48718, 0, 0, 0, -175000000, 2430.55555555556, -2206.19658119658, 0, 314102564.102564, -2458974358.97436, 0, 0, 0, -134615384.615385, 1308.76068376068, -10245.7264957264, 0, 224358974.358974, -547435897.435898, 0, 0, 0, 94230769.2307692, 934.82905982906, -2280.98290598291, 0, 224358974.358974, -1310256410.25641, 0, 0, 0, 0, 934.82905982906, -5459.40170940171, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(11) << 0, 0, 403846153.846154, -35897435.8974359, 13461538.4615385, 0, 0, 0, 1682.69230769231, 0, 0, 933333333.333333, 53846153.8461539, -53846153.8461539, 0, 0, 0, 3888.88888888889, 0, 0, 403846153.846154, -13461538.4615385, 35897435.8974359, 0, 0, 0, 1682.69230769231, 0, 0, 412820512.820513, -13461538.4615385, 13461538.4615385, 0, 0, 0, 1720.08547008547, 0, 0, -664102564.102564, 89743589.7435897, -116666666.666667, 0, 0, 0, -2767.09401709402, 0, 0, -664102564.102564, 116666666.666667, -89743589.7435897, 0, 0, 0, -2767.09401709402, 0, 0, -412820512.820513, 0, 62820512.8205128, 0, 0, 0, -1720.08547008547, 0, 0, -412820512.820513, -62820512.8205128, 0, 0, 0, 0, -1720.08547008547;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(12) << 0, 0, 35897435.8974359, 8982959.4017094, 280.448717948718, 0, 0, 0, -299.145299145299, 0, 0, 53846153.8461539, 26940576.9230769, -7946.04700854701, 0, 0, 0, 897.435897435898, 0, 0, -13461538.4615385, 8980902.77777778, -280.448717948718, 0, 0, 0, -224.358974358974, 0, 0, 13461538.4615385, 13469278.8461538, -3271.90170940171, 0, 0, 0, -112.179487179487, 0, 0, -89743589.7435897, -26943568.3760684, 2617.52136752137, 0, 0, 0, 747.863247863248, 0, 0, -62820512.8205128, -26927489.3162393, 4861.11111111111, 0, 0, 0, 1196.5811965812, 0, 0, 0, -35908354.7008547, 1869.65811965812, 0, 0, 0, 0, 0, 0, 62820512.8205128, -35901997.8632479, 1869.65811965812, 0, 0, 0, -523.504273504273;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(13) << 0, 0, 13461538.4615385, -280.448717948718, 8980902.77777778, 0, 0, 0, 224.358974358974, 0, 0, -53846153.8461539, -7946.04700854701, 26940576.9230769, 0, 0, 0, -897.435897435898, 0, 0, -35897435.8974359, 280.448717948718, 8982959.4017094, 0, 0, 0, 299.145299145299, 0, 0, -13461538.4615385, -3271.90170940171, 13469278.8461538, 0, 0, 0, 112.179487179487, 0, 0, 62820512.8205128, 4861.11111111111, -26927489.3162393, 0, 0, 0, -1196.5811965812, 0, 0, 89743589.7435897, 2617.52136752137, -26943568.3760684, 0, 0, 0, -747.863247863248, 0, 0, -62820512.8205128, 1869.65811965812, -35901997.8632479, 0, 0, 0, 523.504273504273, 0, 0, 0, 1869.65811965812, -35908354.7008547, 0, 0, 0, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(14) << 53846153.8461538, 20192307.6923077, 0, 0, 0, 31413621.7948718, -74.7863247863248, 196.314102564103, 0, 80769230.7692308, -80769230.7692308, 0, 0, 0, 94238547.008547, 785.25641025641, -785.25641025641, 0, -20192307.6923077, -53846153.8461538, 0, 0, 0, 31413621.7948718, -196.314102564103, 74.7863247863248, 0, 20192307.6923077, -20192307.6923077, 0, 0, 0, 47118824.7863248, -28.0448717948718, 28.0448717948718, 0, -134615384.615385, 94230769.2307692, 0, 0, 0, -94236303.4188034, 186.965811965812, -579.594017094017, 0, -94230769.2307692, 134615384.615385, 0, 0, 0, -94236303.4188034, 579.594017094017, -186.965811965812, 0, 0, -94230769.2307692, 0, 0, 0, -125644465.811966, 0, 130.876068376068, 0, 94230769.2307692, 0, 0, 0, 0, -125644465.811966, -130.876068376068, 0, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(15) << 4300.21367521368, 140.224358974359, 0, 0, 0, 74.7863247863248, 74.8185763888889, 0.00105168269230769, 0, 8750, -3973.0235042735, 0, 0, 0, 785.25641025641, 224.424599358974, -0.0297976762820513, 0, 3271.90170940171, -140.224358974359, 0, 0, 0, -196.314102564103, 74.8108640491453, -0.00105168269230769, 0, 3870.19230769231, -1635.95085470085, 0, 0, 0, 28.0448717948718, 112.208513621795, -0.0122696314102564, 0, -10245.7264957264, 1308.76068376068, 0, 0, 0, -186.965811965812, -224.435817307692, 0.00981570512820513, 0, -2206.19658119658, 2430.55555555556, 0, 0, 0, 205.662393162393, -224.375520833333, 0.0182291666666667, 0, -5459.40170940171, 934.82905982906, 0, 0, 0, 0, -299.18624465812, 0.00701121794871795, 0, -2280.98290598291, 934.82905982906, 0, 0, 0, 130.876068376068, -299.162406517094, 0.00701121794871795, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(16) << -140.224358974359, 3271.90170940171, 0, 0, 0, 196.314102564103, -0.00105168269230769, 74.8108640491453, 0, -3973.0235042735, 8750, 0, 0, 0, -785.25641025641, -0.0297976762820513, 224.424599358974, 0, 140.224358974359, 4300.21367521368, 0, 0, 0, -74.7863247863248, 0.00105168269230769, 74.8185763888889, 0, -1635.95085470085, 3870.19230769231, 0, 0, 0, -28.0448717948718, -0.0122696314102564, 112.208513621795, 0, 2430.55555555556, -2206.19658119658, 0, 0, 0, -205.662393162393, 0.0182291666666667, -224.375520833333, 0, 1308.76068376068, -10245.7264957264, 0, 0, 0, 186.965811965812, 0.00981570512820513, -224.435817307692, 0, 934.82905982906, -2280.98290598291, 0, 0, 0, -130.876068376068, 0.00701121794871795, -299.162406517094, 0, 934.82905982906, -5459.40170940171, 0, 0, 0, 0, 0.00701121794871795, -299.18624465812, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(17) << 0, 0, 1682.69230769231, 299.145299145299, 224.358974358974, 0, 0, 0, 261.764756944444, 0, 0, 3888.88888888889, 897.435897435898, -897.435897435898, 0, 0, 0, 785.285576923077, 0, 0, 1682.69230769231, -224.358974358974, -299.145299145299, 0, 0, 0, 261.764756944444, 0, 0, 1720.08547008547, 112.179487179487, -112.179487179487, 0, 0, 0, 392.641105769231, 0, 0, -2767.09401709402, -747.863247863248, 299.145299145299, 0, 0, 0, -785.277163461538, 0, 0, -2767.09401709402, -299.145299145299, 747.863247863248, 0, 0, 0, -785.277163461538, 0, 0, -1720.08547008547, 0, -523.504273504273, 0, 0, 0, -1047.02144764957, 0, 0, -1720.08547008547, 523.504273504273, 0, 0, 0, 0, -1047.02144764957;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(18) << 928846153.846154, 392628205.128205, 0, 0, 0, -20192307.6923077, 3870.19230769231, 1635.95085470085, 0, 785256410.25641, 33653846.1538461, 0, 0, 0, -20192307.6923077, 3271.90170940171, 140.224358974359, 0, 2100000000, 953525641.025641, 0, 0, 0, 80769230.7692308, 8750, 3973.0235042735, 0, 1032051282.05128, -33653846.1538461, 0, 0, 0, -53846153.8461538, 4300.21367521368, -140.224358974359, 0, -1310256410.25641, -224358974.358974, 0, 0, 0, 0, -5459.40170940171, -934.82905982906, 0, -529487179.48718, -583333333.333333, 0, 0, 0, 175000000, -2206.19658119658, -2430.55555555556, 0, -2458974358.97436, -314102564.102564, 0, 0, 0, 134615384.615385, -10245.7264957264, -1308.76068376068, 0, -547435897.435898, -224358974.358974, 0, 0, 0, -94230769.2307692, -2280.98290598291, -934.82905982906, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(19) << 392628205.128205, 928846153.846154, 0, 0, 0, -20192307.6923077, 1635.95085470085, 3870.19230769231, 0, -33653846.1538461, 1032051282.05128, 0, 0, 0, -53846153.8461538, -140.224358974359, 4300.21367521368, 0, 953525641.025641, 2100000000, 0, 0, 0, 80769230.7692308, 3973.0235042735, 8750, 0, 33653846.1538461, 785256410.25641, 0, 0, 0, -20192307.6923077, 140.224358974359, 3271.90170940171, 0, -224358974.358974, -547435897.435898, 0, 0, 0, -94230769.2307692, -934.82905982906, -2280.98290598291, 0, -314102564.102564, -2458974358.97436, 0, 0, 0, 134615384.615385, -1308.76068376068, -10245.7264957264, 0, -583333333.333333, -529487179.48718, 0, 0, 0, 175000000, -2430.55555555556, -2206.19658119658, 0, -224358974.358974, -1310256410.25641, 0, 0, 0, 0, -934.82905982906, -5459.40170940171, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(20) << 0, 0, 412820512.820513, -13461538.4615385, -13461538.4615385, 0, 0, 0, 1720.08547008547, 0, 0, 403846153.846154, -13461538.4615385, -35897435.8974359, 0, 0, 0, 1682.69230769231, 0, 0, 933333333.333333, 53846153.8461539, 53846153.8461539, 0, 0, 0, 3888.88888888889, 0, 0, 403846153.846154, -35897435.8974359, -13461538.4615385, 0, 0, 0, 1682.69230769231, 0, 0, -412820512.820513, 0, -62820512.8205128, 0, 0, 0, -1720.08547008547, 0, 0, -664102564.102564, 116666666.666667, 89743589.7435897, 0, 0, 0, -2767.09401709402, 0, 0, -664102564.102564, 89743589.7435897, 116666666.666667, 0, 0, 0, -2767.09401709402, 0, 0, -412820512.820513, -62820512.8205128, 0, 0, 0, 0, -1720.08547008547;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(21) << 0, 0, 13461538.4615385, 13469278.8461538, 3271.90170940171, 0, 0, 0, -112.179487179487, 0, 0, -13461538.4615385, 8980902.77777778, 280.448717948718, 0, 0, 0, -224.358974358974, 0, 0, 53846153.8461539, 26940576.9230769, 7946.04700854701, 0, 0, 0, 897.435897435898, 0, 0, 35897435.8974359, 8982959.4017094, -280.448717948718, 0, 0, 0, -299.145299145299, 0, 0, 0, -35908354.7008547, -1869.65811965812, 0, 0, 0, 0, 0, 0, -62820512.8205128, -26927489.3162393, -4861.11111111111, 0, 0, 0, 1196.5811965812, 0, 0, -89743589.7435897, -26943568.3760684, -2617.52136752137, 0, 0, 0, 747.863247863248, 0, 0, 62820512.8205128, -35901997.8632479, -1869.65811965812, 0, 0, 0, -523.504273504273;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(22) << 0, 0, 13461538.4615385, 3271.90170940171, 13469278.8461538, 0, 0, 0, -112.179487179487, 0, 0, 35897435.8974359, -280.448717948718, 8982959.4017094, 0, 0, 0, -299.145299145299, 0, 0, 53846153.8461539, 7946.04700854701, 26940576.9230769, 0, 0, 0, 897.435897435898, 0, 0, -13461538.4615385, 280.448717948718, 8980902.77777778, 0, 0, 0, -224.358974358974, 0, 0, 62820512.8205128, -1869.65811965812, -35901997.8632479, 0, 0, 0, -523.504273504273, 0, 0, -89743589.7435897, -2617.52136752137, -26943568.3760684, 0, 0, 0, 747.863247863248, 0, 0, -62820512.8205128, -4861.11111111111, -26927489.3162393, 0, 0, 0, 1196.5811965812, 0, 0, 0, -1869.65811965812, -35908354.7008547, 0, 0, 0, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(23) << 20192307.6923077, 20192307.6923077, 0, 0, 0, 47118824.7863248, -28.0448717948718, -28.0448717948718, 0, -20192307.6923077, 53846153.8461538, 0, 0, 0, 31413621.7948718, -196.314102564103, -74.7863247863248, 0, 80769230.7692308, 80769230.7692308, 0, 0, 0, 94238547.008547, 785.25641025641, 785.25641025641, 0, 53846153.8461538, -20192307.6923077, 0, 0, 0, 31413621.7948718, -74.7863247863248, -196.314102564103, 0, 0, 94230769.2307692, 0, 0, 0, -125644465.811966, 0, -130.876068376068, 0, -94230769.2307692, -134615384.615385, 0, 0, 0, -94236303.4188034, 579.594017094017, 186.965811965812, 0, -134615384.615385, -94230769.2307692, 0, 0, 0, -94236303.4188034, 186.965811965812, 579.594017094017, 0, 94230769.2307692, 0, 0, 0, 0, -125644465.811966, -130.876068376068, 0, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(24) << 3870.19230769231, 1635.95085470085, 0, 0, 0, 28.0448717948718, 112.208513621795, 0.0122696314102564, 0, 3271.90170940171, 140.224358974359, 0, 0, 0, -196.314102564103, 74.8108640491453, 0.00105168269230769, 0, 8750, 3973.0235042735, 0, 0, 0, 785.25641025641, 224.424599358974, 0.0297976762820513, 0, 4300.21367521368, -140.224358974359, 0, 0, 0, 74.7863247863248, 74.8185763888889, -0.00105168269230769, 0, -5459.40170940171, -934.82905982906, 0, 0, 0, 0, -299.18624465812, -0.00701121794871795, 0, -2206.19658119658, -2430.55555555556, 0, 0, 0, 205.662393162393, -224.375520833333, -0.0182291666666667, 0, -10245.7264957264, -1308.76068376068, 0, 0, 0, -186.965811965812, -224.435817307692, -0.00981570512820513, 0, -2280.98290598291, -934.82905982906, 0, 0, 0, 130.876068376068, -299.162406517094, -0.00701121794871795, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(25) << 1635.95085470085, 3870.19230769231, 0, 0, 0, 28.0448717948718, 0.0122696314102564, 112.208513621795, 0, -140.224358974359, 4300.21367521368, 0, 0, 0, 74.7863247863248, -0.00105168269230769, 74.8185763888889, 0, 3973.0235042735, 8750, 0, 0, 0, 785.25641025641, 0.0297976762820513, 224.424599358974, 0, 140.224358974359, 3271.90170940171, 0, 0, 0, -196.314102564103, 0.00105168269230769, 74.8108640491453, 0, -934.82905982906, -2280.98290598291, 0, 0, 0, 130.876068376068, -0.00701121794871795, -299.162406517094, 0, -1308.76068376068, -10245.7264957264, 0, 0, 0, -186.965811965812, -0.00981570512820513, -224.435817307692, 0, -2430.55555555556, -2206.19658119658, 0, 0, 0, 205.662393162393, -0.0182291666666667, -224.375520833333, 0, -934.82905982906, -5459.40170940171, 0, 0, 0, 0, -0.00701121794871795, -299.18624465812, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(26) << 0, 0, 1720.08547008547, 112.179487179487, 112.179487179487, 0, 0, 0, 392.641105769231, 0, 0, 1682.69230769231, -224.358974358974, 299.145299145299, 0, 0, 0, 261.764756944444, 0, 0, 3888.88888888889, 897.435897435898, 897.435897435898, 0, 0, 0, 785.285576923077, 0, 0, 1682.69230769231, 299.145299145299, -224.358974358974, 0, 0, 0, 261.764756944444, 0, 0, -1720.08547008547, 0, 523.504273504273, 0, 0, 0, -1047.02144764957, 0, 0, -2767.09401709402, -299.145299145299, -747.863247863248, 0, 0, 0, -785.277163461538, 0, 0, -2767.09401709402, -747.863247863248, -299.145299145299, 0, 0, 0, -785.277163461538, 0, 0, -1720.08547008547, 523.504273504273, 0, 0, 0, 0, -1047.02144764957;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(27) << 785256410.25641, -33653846.1538461, 0, 0, 0, 20192307.6923077, 3271.90170940171, -140.224358974359, 0, 928846153.846154, -392628205.128205, 0, 0, 0, 20192307.6923077, 3870.19230769231, -1635.95085470085, 0, 1032051282.05128, 33653846.1538461, 0, 0, 0, 53846153.8461538, 4300.21367521368, 140.224358974359, 0, 2100000000, -953525641.025641, 0, 0, 0, -80769230.7692308, 8750, -3973.0235042735, 0, -1310256410.25641, 224358974.358974, 0, 0, 0, 0, -5459.40170940171, 934.82905982906, 0, -547435897.435898, 224358974.358974, 0, 0, 0, 94230769.2307692, -2280.98290598291, 934.82905982906, 0, -2458974358.97436, 314102564.102564, 0, 0, 0, -134615384.615385, -10245.7264957264, 1308.76068376068, 0, -529487179.48718, 583333333.333333, 0, 0, 0, -175000000, -2206.19658119658, 2430.55555555556, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(28) << 33653846.1538461, 1032051282.05128, 0, 0, 0, -53846153.8461538, 140.224358974359, 4300.21367521368, 0, -392628205.128205, 928846153.846154, 0, 0, 0, -20192307.6923077, -1635.95085470085, 3870.19230769231, 0, -33653846.1538461, 785256410.25641, 0, 0, 0, -20192307.6923077, -140.224358974359, 3271.90170940171, 0, -953525641.025641, 2100000000, 0, 0, 0, 80769230.7692308, -3973.0235042735, 8750, 0, 224358974.358974, -547435897.435898, 0, 0, 0, -94230769.2307692, 934.82905982906, -2280.98290598291, 0, 224358974.358974, -1310256410.25641, 0, 0, 0, 0, 934.82905982906, -5459.40170940171, 0, 583333333.333333, -529487179.48718, 0, 0, 0, 175000000, 2430.55555555556, -2206.19658119658, 0, 314102564.102564, -2458974358.97436, 0, 0, 0, 134615384.615385, 1308.76068376068, -10245.7264957264, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(29) << 0, 0, 403846153.846154, 13461538.4615385, -35897435.8974359, 0, 0, 0, 1682.69230769231, 0, 0, 412820512.820513, 13461538.4615385, -13461538.4615385, 0, 0, 0, 1720.08547008547, 0, 0, 403846153.846154, 35897435.8974359, -13461538.4615385, 0, 0, 0, 1682.69230769231, 0, 0, 933333333.333333, -53846153.8461539, 53846153.8461539, 0, 0, 0, 3888.88888888889, 0, 0, -412820512.820513, 0, -62820512.8205128, 0, 0, 0, -1720.08547008547, 0, 0, -412820512.820513, 62820512.8205128, 0, 0, 0, 0, -1720.08547008547, 0, 0, -664102564.102564, -89743589.7435897, 116666666.666667, 0, 0, 0, -2767.09401709402, 0, 0, -664102564.102564, -116666666.666667, 89743589.7435897, 0, 0, 0, -2767.09401709402;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(30) << 0, 0, 13461538.4615385, 8980902.77777778, -280.448717948718, 0, 0, 0, 224.358974358974, 0, 0, -13461538.4615385, 13469278.8461538, -3271.90170940171, 0, 0, 0, 112.179487179487, 0, 0, -35897435.8974359, 8982959.4017094, 280.448717948718, 0, 0, 0, 299.145299145299, 0, 0, -53846153.8461539, 26940576.9230769, -7946.04700854701, 0, 0, 0, -897.435897435898, 0, 0, 0, -35908354.7008547, 1869.65811965812, 0, 0, 0, 0, 0, 0, -62820512.8205128, -35901997.8632479, 1869.65811965812, 0, 0, 0, 523.504273504273, 0, 0, 89743589.7435897, -26943568.3760684, 2617.52136752137, 0, 0, 0, -747.863247863248, 0, 0, 62820512.8205128, -26927489.3162393, 4861.11111111111, 0, 0, 0, -1196.5811965812;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(31) << 0, 0, 35897435.8974359, 280.448717948718, 8982959.4017094, 0, 0, 0, -299.145299145299, 0, 0, 13461538.4615385, -3271.90170940171, 13469278.8461538, 0, 0, 0, -112.179487179487, 0, 0, -13461538.4615385, -280.448717948718, 8980902.77777778, 0, 0, 0, -224.358974358974, 0, 0, 53846153.8461539, -7946.04700854701, 26940576.9230769, 0, 0, 0, 897.435897435898, 0, 0, 62820512.8205128, 1869.65811965812, -35901997.8632479, 0, 0, 0, -523.504273504273, 0, 0, 0, 1869.65811965812, -35908354.7008547, 0, 0, 0, 0, 0, 0, -62820512.8205128, 4861.11111111111, -26927489.3162393, 0, 0, 0, 1196.5811965812, 0, 0, -89743589.7435897, 2617.52136752137, -26943568.3760684, 0, 0, 0, 747.863247863248;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(32) << 20192307.6923077, 53846153.8461538, 0, 0, 0, 31413621.7948718, 196.314102564103, -74.7863247863248, 0, -20192307.6923077, 20192307.6923077, 0, 0, 0, 47118824.7863248, 28.0448717948718, -28.0448717948718, 0, -53846153.8461538, -20192307.6923077, 0, 0, 0, 31413621.7948718, 74.7863247863248, -196.314102564103, 0, -80769230.7692308, 80769230.7692308, 0, 0, 0, 94238547.008547, -785.25641025641, 785.25641025641, 0, 0, 94230769.2307692, 0, 0, 0, -125644465.811966, 0, -130.876068376068, 0, -94230769.2307692, 0, 0, 0, 0, -125644465.811966, 130.876068376068, 0, 0, 134615384.615385, -94230769.2307692, 0, 0, 0, -94236303.4188034, -186.965811965812, 579.594017094017, 0, 94230769.2307692, -134615384.615385, 0, 0, 0, -94236303.4188034, -579.594017094017, 186.965811965812, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(33) << 3271.90170940171, -140.224358974359, 0, 0, 0, 196.314102564103, 74.8108640491453, -0.00105168269230769, 0, 3870.19230769231, -1635.95085470085, 0, 0, 0, -28.0448717948718, 112.208513621795, -0.0122696314102564, 0, 4300.21367521368, 140.224358974359, 0, 0, 0, -74.7863247863248, 74.8185763888889, 0.00105168269230769, 0, 8750, -3973.0235042735, 0, 0, 0, -785.25641025641, 224.424599358974, -0.0297976762820513, 0, -5459.40170940171, 934.82905982906, 0, 0, 0, 0, -299.18624465812, 0.00701121794871795, 0, -2280.98290598291, 934.82905982906, 0, 0, 0, -130.876068376068, -299.162406517094, 0.00701121794871795, 0, -10245.7264957264, 1308.76068376068, 0, 0, 0, 186.965811965812, -224.435817307692, 0.00981570512820513, 0, -2206.19658119658, 2430.55555555556, 0, 0, 0, -205.662393162393, -224.375520833333, 0.0182291666666667, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(34) << 140.224358974359, 4300.21367521368, 0, 0, 0, 74.7863247863248, 0.00105168269230769, 74.8185763888889, 0, -1635.95085470085, 3870.19230769231, 0, 0, 0, 28.0448717948718, -0.0122696314102564, 112.208513621795, 0, -140.224358974359, 3271.90170940171, 0, 0, 0, -196.314102564103, -0.00105168269230769, 74.8108640491453, 0, -3973.0235042735, 8750, 0, 0, 0, 785.25641025641, -0.0297976762820513, 224.424599358974, 0, 934.82905982906, -2280.98290598291, 0, 0, 0, 130.876068376068, 0.00701121794871795, -299.162406517094, 0, 934.82905982906, -5459.40170940171, 0, 0, 0, 0, 0.00701121794871795, -299.18624465812, 0, 2430.55555555556, -2206.19658119658, 0, 0, 0, 205.662393162393, 0.0182291666666667, -224.375520833333, 0, 1308.76068376068, -10245.7264957264, 0, 0, 0, -186.965811965812, 0.00981570512820513, -224.435817307692, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(35) << 0, 0, 1682.69230769231, 224.358974358974, 299.145299145299, 0, 0, 0, 261.764756944444, 0, 0, 1720.08547008547, -112.179487179487, 112.179487179487, 0, 0, 0, 392.641105769231, 0, 0, 1682.69230769231, -299.145299145299, -224.358974358974, 0, 0, 0, 261.764756944444, 0, 0, 3888.88888888889, -897.435897435898, 897.435897435898, 0, 0, 0, 785.285576923077, 0, 0, -1720.08547008547, 0, 523.504273504273, 0, 0, 0, -1047.02144764957, 0, 0, -1720.08547008547, -523.504273504273, 0, 0, 0, 0, -1047.02144764957, 0, 0, -2767.09401709402, 747.863247863248, -299.145299145299, 0, 0, 0, -785.277163461538, 0, 0, -2767.09401709402, 299.145299145299, -747.863247863248, 0, 0, 0, -785.277163461538;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(36) << -2458974358.97436, -583333333.333333, 0, 0, 0, 134615384.615385, -10245.7264957264, -2430.55555555556, 0, -2458974358.97436, 583333333.333333, 0, 0, 0, -134615384.615385, -10245.7264957264, 2430.55555555556, 0, -1310256410.25641, -224358974.358974, 0, 0, 0, 0, -5459.40170940171, -934.82905982906, 0, -1310256410.25641, 224358974.358974, 0, 0, 0, 0, -5459.40170940171, 934.82905982906, 0, 5456410256.41026, 0, 0, 0, 0, 0, 22735.0427350427, 0, 0, 0, -897435897.435898, 0, 0, 0, -269230769.230769, 0, -3739.31623931624, 0, 2082051282.05128, 0, 0, 0, 0, 0, 8675.21367521367, 0, 0, 0, 897435897.435898, 0, 0, 0, 269230769.230769, 0, 3739.31623931624, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(37) << -314102564.102564, -529487179.48718, 0, 0, 0, 94230769.2307692, -1308.76068376068, -2206.19658119658, 0, 314102564.102564, -529487179.48718, 0, 0, 0, 94230769.2307692, 1308.76068376068, -2206.19658119658, 0, -224358974.358974, -547435897.435898, 0, 0, 0, 94230769.2307692, -934.82905982906, -2280.98290598291, 0, 224358974.358974, -547435897.435898, 0, 0, 0, 94230769.2307692, 934.82905982906, -2280.98290598291, 0, 0, 2943589743.58974, 0, 0, 0, -323076923.076923, 0, 12264.9572649573, 0, -897435897.435898, 0, 0, 0, 0, -269230769.230769, -3739.31623931624, 0, 0, 0, -789743589.74359, 0, 0, 0, -323076923.076923, 0, -3290.59829059829, 0, 897435897.435898, 0, 0, 0, 0, -269230769.230769, 3739.31623931624, 0, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(38) << 0, 0, -664102564.102564, 89743589.7435897, 62820512.8205128, 0, 0, 0, -2767.09401709402, 0, 0, -664102564.102564, -89743589.7435897, 62820512.8205128, 0, 0, 0, -2767.09401709402, 0, 0, -412820512.820513, 0, 62820512.8205128, 0, 0, 0, -1720.08547008547, 0, 0, -412820512.820513, 0, 62820512.8205128, 0, 0, 0, -1720.08547008547, 0, 0, 1866666666.66667, 0, -215384615.384615, 0, 0, 0, 7777.77777777778, 0, 0, 0, -179487179.48718, -179487179.48718, 0, 0, 0, 0, 0, 0, 287179487.179487, 0, -215384615.384615, 0, 0, 0, 1196.5811965812, 0, 0, 0, 179487179.48718, -179487179.48718, 0, 0, 0, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(39) << 0, 0, -89743589.7435897, -26943568.3760684, -4861.11111111111, 0, 0, 0, 747.863247863248, 0, 0, 89743589.7435897, -26943568.3760684, 4861.11111111111, 0, 0, 0, -747.863247863248, 0, 0, 0, -35908354.7008547, -1869.65811965812, 0, 0, 0, 0, 0, 0, 0, -35908354.7008547, 1869.65811965812, 0, 0, 0, 0, 0, 0, 0, 143635213.675214, 0, 0, 0, 0, 0, 0, 0, 179487179.48718, 89743589.7435897, -7478.63247863248, 0, 0, 0, -1495.7264957265, 0, 0, 0, 71812222.2222222, 0, 0, 0, 0, 0, 0, 0, -179487179.48718, 89743589.7435897, 7478.63247863248, 0, 0, 0, 1495.7264957265;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(40) << 0, 0, -116666666.666667, -2617.52136752137, -26927489.3162393, 0, 0, 0, 299.145299145299, 0, 0, -116666666.666667, 2617.52136752137, -26927489.3162393, 0, 0, 0, 299.145299145299, 0, 0, -62820512.8205128, -1869.65811965812, -35901997.8632479, 0, 0, 0, 523.504273504273, 0, 0, -62820512.8205128, 1869.65811965812, -35901997.8632479, 0, 0, 0, 523.504273504273, 0, 0, -215384615.384615, 0, 143614273.504274, 0, 0, 0, -3589.74358974359, 0, 0, 179487179.48718, -7478.63247863248, 89743589.7435897, 0, 0, 0, -1495.7264957265, 0, 0, 215384615.384615, 0, 71788290.5982906, 0, 0, 0, -1794.87179487179, 0, 0, 179487179.48718, 7478.63247863248, 89743589.7435897, 0, 0, 0, -1495.7264957265;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(41) << -134615384.615385, -175000000, 0, 0, 0, -94236303.4188034, 186.965811965812, -205.662393162393, 0, 134615384.615385, -175000000, 0, 0, 0, -94236303.4188034, -186.965811965812, -205.662393162393, 0, 0, -94230769.2307692, 0, 0, 0, -125644465.811966, 0, 130.876068376068, 0, 0, -94230769.2307692, 0, 0, 0, -125644465.811966, 0, 130.876068376068, 0, 0, -323076923.076923, 0, 0, 0, 502579658.119658, 0, -3141.02564102564, 0, 269230769.230769, 269230769.230769, 0, 0, 0, 314102564.102564, -373.931623931624, -373.931623931624, 0, 0, 323076923.076923, 0, 0, 0, 251284444.444444, 0, -448.717948717949, 0, -269230769.230769, 269230769.230769, 0, 0, 0, 314102564.102564, 373.931623931624, -373.931623931624, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(42) << -10245.7264957264, -2430.55555555556, 0, 0, 0, -186.965811965812, -224.435817307692, -0.0182291666666667, 0, -10245.7264957264, 2430.55555555556, 0, 0, 0, 186.965811965812, -224.435817307692, 0.0182291666666667, 0, -5459.40170940171, -934.82905982906, 0, 0, 0, 0, -299.18624465812, -0.00701121794871795, 0, -5459.40170940171, 934.82905982906, 0, 0, 0, 0, -299.18624465812, 0.00701121794871795, 0, 22735.0427350427, 0, 0, 0, 0, 0, 1196.75170940171, 0, 0, 0, -3739.31623931624, 0, 0, 0, 373.931623931624, 747.863247863248, -0.0280448717948718, 0, 8675.21367521367, 0, 0, 0, 0, 0, 598.355662393162, 0, 0, 0, 3739.31623931624, 0, 0, 0, -373.931623931624, 747.863247863248, 0.0280448717948718, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(43) << -1308.76068376068, -2206.19658119658, 0, 0, 0, -579.594017094017, -0.00981570512820513, -224.375520833333, 0, 1308.76068376068, -2206.19658119658, 0, 0, 0, -579.594017094017, 0.00981570512820513, -224.375520833333, 0, -934.82905982906, -2280.98290598291, 0, 0, 0, -130.876068376068, -0.00701121794871795, -299.162406517094, 0, 934.82905982906, -2280.98290598291, 0, 0, 0, -130.876068376068, 0.00701121794871795, -299.162406517094, 0, 0, 12264.9572649573, 0, 0, 0, -3141.02564102564, 0, 1196.67318376068, 0, -3739.31623931624, 0, 0, 0, 0, 373.931623931624, -0.0280448717948718, 747.863247863248, 0, 0, -3290.59829059829, 0, 0, 0, 448.717948717949, 0, 598.265918803419, 0, 3739.31623931624, 0, 0, 0, 0, 373.931623931624, 0.0280448717948718, 747.863247863248, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(44) << 0, 0, -2767.09401709402, -747.863247863248, -1196.5811965812, 0, 0, 0, -785.277163461538, 0, 0, -2767.09401709402, 747.863247863248, -1196.5811965812, 0, 0, 0, -785.277163461538, 0, 0, -1720.08547008547, 0, -523.504273504273, 0, 0, 0, -1047.02144764957, 0, 0, -1720.08547008547, 0, -523.504273504273, 0, 0, 0, -1047.02144764957, 0, 0, 7777.77777777778, 0, -3589.74358974359, 0, 0, 0, 4188.09252136752, 0, 0, 0, 1495.7264957265, 1495.7264957265, 0, 0, 0, 2617.52136752137, 0, 0, 1196.5811965812, 0, 1794.87179487179, 0, 0, 0, 2094.02606837607, 0, 0, 0, -1495.7264957265, 1495.7264957265, 0, 0, 0, 2617.52136752137;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(45) << -547435897.435898, -224358974.358974, 0, 0, 0, -94230769.2307692, -2280.98290598291, -934.82905982906, 0, -529487179.48718, 314102564.102564, 0, 0, 0, -94230769.2307692, -2206.19658119658, 1308.76068376068, 0, -529487179.48718, -314102564.102564, 0, 0, 0, -94230769.2307692, -2206.19658119658, -1308.76068376068, 0, -547435897.435898, 224358974.358974, 0, 0, 0, -94230769.2307692, -2280.98290598291, 934.82905982906, 0, 0, -897435897.435898, 0, 0, 0, 269230769.230769, 0, -3739.31623931624, 0, 2943589743.58974, 0, 0, 0, 0, 323076923.076923, 12264.9572649573, 0, 0, 0, 897435897.435898, 0, 0, 0, 269230769.230769, 0, 3739.31623931624, 0, -789743589.74359, 0, 0, 0, 0, 323076923.076923, -3290.59829059829, 0, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(46) << -224358974.358974, -1310256410.25641, 0, 0, 0, 0, -934.82905982906, -5459.40170940171, 0, 583333333.333333, -2458974358.97436, 0, 0, 0, 134615384.615385, 2430.55555555556, -10245.7264957264, 0, -583333333.333333, -2458974358.97436, 0, 0, 0, -134615384.615385, -2430.55555555556, -10245.7264957264, 0, 224358974.358974, -1310256410.25641, 0, 0, 0, 0, 934.82905982906, -5459.40170940171, 0, -897435897.435898, 0, 0, 0, 0, 269230769.230769, -3739.31623931624, 0, 0, 0, 5456410256.41026, 0, 0, 0, 0, 0, 22735.0427350427, 0, 897435897.435898, 0, 0, 0, 0, -269230769.230769, 3739.31623931624, 0, 0, 0, 2082051282.05128, 0, 0, 0, 0, 0, 8675.21367521367, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(47) << 0, 0, -412820512.820513, -62820512.8205128, 0, 0, 0, 0, -1720.08547008547, 0, 0, -664102564.102564, -62820512.8205128, 89743589.7435897, 0, 0, 0, -2767.09401709402, 0, 0, -664102564.102564, -62820512.8205128, -89743589.7435897, 0, 0, 0, -2767.09401709402, 0, 0, -412820512.820513, -62820512.8205128, 0, 0, 0, 0, -1720.08547008547, 0, 0, 0, 179487179.48718, 179487179.48718, 0, 0, 0, 0, 0, 0, 1866666666.66667, 215384615.384615, 0, 0, 0, 0, 7777.77777777778, 0, 0, 0, 179487179.48718, -179487179.48718, 0, 0, 0, 0, 0, 0, 287179487.179487, 215384615.384615, 0, 0, 0, 0, 1196.5811965812;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(48) << 0, 0, 62820512.8205128, -35901997.8632479, -1869.65811965812, 0, 0, 0, -523.504273504273, 0, 0, 116666666.666667, -26927489.3162393, 2617.52136752137, 0, 0, 0, -299.145299145299, 0, 0, 116666666.666667, -26927489.3162393, -2617.52136752137, 0, 0, 0, -299.145299145299, 0, 0, 62820512.8205128, -35901997.8632479, 1869.65811965812, 0, 0, 0, -523.504273504273, 0, 0, -179487179.48718, 89743589.7435897, -7478.63247863248, 0, 0, 0, 1495.7264957265, 0, 0, 215384615.384615, 143614273.504274, 0, 0, 0, 0, 3589.74358974359, 0, 0, -179487179.48718, 89743589.7435897, 7478.63247863248, 0, 0, 0, 1495.7264957265, 0, 0, -215384615.384615, 71788290.5982906, 0, 0, 0, 0, 1794.87179487179;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(49) << 0, 0, 0, -1869.65811965812, -35908354.7008547, 0, 0, 0, 0, 0, 0, -89743589.7435897, 4861.11111111111, -26943568.3760684, 0, 0, 0, 747.863247863248, 0, 0, 89743589.7435897, -4861.11111111111, -26943568.3760684, 0, 0, 0, -747.863247863248, 0, 0, 0, 1869.65811965812, -35908354.7008547, 0, 0, 0, 0, 0, 0, -179487179.48718, -7478.63247863248, 89743589.7435897, 0, 0, 0, 1495.7264957265, 0, 0, 0, 0, 143635213.675214, 0, 0, 0, 0, 0, 0, 179487179.48718, 7478.63247863248, 89743589.7435897, 0, 0, 0, -1495.7264957265, 0, 0, 0, 0, 71812222.2222222, 0, 0, 0, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(50) << 94230769.2307692, 0, 0, 0, 0, -125644465.811966, -130.876068376068, 0, 0, 175000000, -134615384.615385, 0, 0, 0, -94236303.4188034, 205.662393162393, 186.965811965812, 0, 175000000, 134615384.615385, 0, 0, 0, -94236303.4188034, 205.662393162393, -186.965811965812, 0, 94230769.2307692, 0, 0, 0, 0, -125644465.811966, -130.876068376068, 0, 0, -269230769.230769, -269230769.230769, 0, 0, 0, 314102564.102564, 373.931623931624, 373.931623931624, 0, 323076923.076923, 0, 0, 0, 0, 502579658.119658, 3141.02564102564, 0, 0, -269230769.230769, 269230769.230769, 0, 0, 0, 314102564.102564, 373.931623931624, -373.931623931624, 0, -323076923.076923, 0, 0, 0, 0, 251284444.444444, 448.717948717949, 0, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(51) << -2280.98290598291, -934.82905982906, 0, 0, 0, 130.876068376068, -299.162406517094, -0.00701121794871795, 0, -2206.19658119658, 1308.76068376068, 0, 0, 0, 579.594017094017, -224.375520833333, 0.00981570512820513, 0, -2206.19658119658, -1308.76068376068, 0, 0, 0, 579.594017094017, -224.375520833333, -0.00981570512820513, 0, -2280.98290598291, 934.82905982906, 0, 0, 0, 130.876068376068, -299.162406517094, 0.00701121794871795, 0, 0, -3739.31623931624, 0, 0, 0, -373.931623931624, 747.863247863248, -0.0280448717948718, 0, 12264.9572649573, 0, 0, 0, 0, 3141.02564102564, 1196.67318376068, 0, 0, 0, 3739.31623931624, 0, 0, 0, -373.931623931624, 747.863247863248, 0.0280448717948718, 0, -3290.59829059829, 0, 0, 0, 0, -448.717948717949, 598.265918803419, 0, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(52) << -934.82905982906, -5459.40170940171, 0, 0, 0, 0, -0.00701121794871795, -299.18624465812, 0, 2430.55555555556, -10245.7264957264, 0, 0, 0, -186.965811965812, 0.0182291666666667, -224.435817307692, 0, -2430.55555555556, -10245.7264957264, 0, 0, 0, 186.965811965812, -0.0182291666666667, -224.435817307692, 0, 934.82905982906, -5459.40170940171, 0, 0, 0, 0, 0.00701121794871795, -299.18624465812, 0, -3739.31623931624, 0, 0, 0, 0, -373.931623931624, -0.0280448717948718, 747.863247863248, 0, 0, 22735.0427350427, 0, 0, 0, 0, 0, 1196.75170940171, 0, 3739.31623931624, 0, 0, 0, 0, 373.931623931624, 0.0280448717948718, 747.863247863248, 0, 0, 8675.21367521367, 0, 0, 0, 0, 0, 598.355662393162, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(53) << 0, 0, -1720.08547008547, 523.504273504273, 0, 0, 0, 0, -1047.02144764957, 0, 0, -2767.09401709402, 1196.5811965812, -747.863247863248, 0, 0, 0, -785.277163461538, 0, 0, -2767.09401709402, 1196.5811965812, 747.863247863248, 0, 0, 0, -785.277163461538, 0, 0, -1720.08547008547, 523.504273504273, 0, 0, 0, 0, -1047.02144764957, 0, 0, 0, -1495.7264957265, -1495.7264957265, 0, 0, 0, 2617.52136752137, 0, 0, 7777.77777777778, 3589.74358974359, 0, 0, 0, 0, 4188.09252136752, 0, 0, 0, -1495.7264957265, 1495.7264957265, 0, 0, 0, 2617.52136752137, 0, 0, 1196.5811965812, -1794.87179487179, 0, 0, 0, 0, 2094.02606837607;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(54) << -1310256410.25641, -224358974.358974, 0, 0, 0, 0, -5459.40170940171, -934.82905982906, 0, -1310256410.25641, 224358974.358974, 0, 0, 0, 0, -5459.40170940171, 934.82905982906, 0, -2458974358.97436, -583333333.333333, 0, 0, 0, -134615384.615385, -10245.7264957264, -2430.55555555556, 0, -2458974358.97436, 583333333.333333, 0, 0, 0, 134615384.615385, -10245.7264957264, 2430.55555555556, 0, 2082051282.05128, 0, 0, 0, 0, 0, 8675.21367521367, 0, 0, 0, 897435897.435898, 0, 0, 0, -269230769.230769, 0, 3739.31623931624, 0, 5456410256.41026, 0, 0, 0, 0, 0, 22735.0427350427, 0, 0, 0, -897435897.435898, 0, 0, 0, 269230769.230769, 0, -3739.31623931624, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(55) << -224358974.358974, -547435897.435898, 0, 0, 0, -94230769.2307692, -934.82905982906, -2280.98290598291, 0, 224358974.358974, -547435897.435898, 0, 0, 0, -94230769.2307692, 934.82905982906, -2280.98290598291, 0, -314102564.102564, -529487179.48718, 0, 0, 0, -94230769.2307692, -1308.76068376068, -2206.19658119658, 0, 314102564.102564, -529487179.48718, 0, 0, 0, -94230769.2307692, 1308.76068376068, -2206.19658119658, 0, 0, -789743589.74359, 0, 0, 0, 323076923.076923, 0, -3290.59829059829, 0, 897435897.435898, 0, 0, 0, 0, 269230769.230769, 3739.31623931624, 0, 0, 0, 2943589743.58974, 0, 0, 0, 323076923.076923, 0, 12264.9572649573, 0, -897435897.435898, 0, 0, 0, 0, 269230769.230769, -3739.31623931624, 0, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(56) << 0, 0, -412820512.820513, 0, -62820512.8205128, 0, 0, 0, -1720.08547008547, 0, 0, -412820512.820513, 0, -62820512.8205128, 0, 0, 0, -1720.08547008547, 0, 0, -664102564.102564, -89743589.7435897, -62820512.8205128, 0, 0, 0, -2767.09401709402, 0, 0, -664102564.102564, 89743589.7435897, -62820512.8205128, 0, 0, 0, -2767.09401709402, 0, 0, 287179487.179487, 0, 215384615.384615, 0, 0, 0, 1196.5811965812, 0, 0, 0, -179487179.48718, 179487179.48718, 0, 0, 0, 0, 0, 0, 1866666666.66667, 0, 215384615.384615, 0, 0, 0, 7777.77777777778, 0, 0, 0, 179487179.48718, 179487179.48718, 0, 0, 0, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(57) << 0, 0, 0, -35908354.7008547, -1869.65811965812, 0, 0, 0, 0, 0, 0, 0, -35908354.7008547, 1869.65811965812, 0, 0, 0, 0, 0, 0, 89743589.7435897, -26943568.3760684, -4861.11111111111, 0, 0, 0, -747.863247863248, 0, 0, -89743589.7435897, -26943568.3760684, 4861.11111111111, 0, 0, 0, 747.863247863248, 0, 0, 0, 71812222.2222222, 0, 0, 0, 0, 0, 0, 0, 179487179.48718, 89743589.7435897, 7478.63247863248, 0, 0, 0, -1495.7264957265, 0, 0, 0, 143635213.675214, 0, 0, 0, 0, 0, 0, 0, -179487179.48718, 89743589.7435897, -7478.63247863248, 0, 0, 0, 1495.7264957265;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(58) << 0, 0, 62820512.8205128, -1869.65811965812, -35901997.8632479, 0, 0, 0, -523.504273504273, 0, 0, 62820512.8205128, 1869.65811965812, -35901997.8632479, 0, 0, 0, -523.504273504273, 0, 0, 116666666.666667, -2617.52136752137, -26927489.3162393, 0, 0, 0, -299.145299145299, 0, 0, 116666666.666667, 2617.52136752137, -26927489.3162393, 0, 0, 0, -299.145299145299, 0, 0, -215384615.384615, 0, 71788290.5982906, 0, 0, 0, 1794.87179487179, 0, 0, -179487179.48718, 7478.63247863248, 89743589.7435897, 0, 0, 0, 1495.7264957265, 0, 0, 215384615.384615, 0, 143614273.504274, 0, 0, 0, 3589.74358974359, 0, 0, -179487179.48718, -7478.63247863248, 89743589.7435897, 0, 0, 0, 1495.7264957265;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(59) << 0, 94230769.2307692, 0, 0, 0, -125644465.811966, 0, -130.876068376068, 0, 0, 94230769.2307692, 0, 0, 0, -125644465.811966, 0, -130.876068376068, 0, 134615384.615385, 175000000, 0, 0, 0, -94236303.4188034, -186.965811965812, 205.662393162393, 0, -134615384.615385, 175000000, 0, 0, 0, -94236303.4188034, 186.965811965812, 205.662393162393, 0, 0, -323076923.076923, 0, 0, 0, 251284444.444444, 0, 448.717948717949, 0, 269230769.230769, -269230769.230769, 0, 0, 0, 314102564.102564, -373.931623931624, 373.931623931624, 0, 0, 323076923.076923, 0, 0, 0, 502579658.119658, 0, 3141.02564102564, 0, -269230769.230769, -269230769.230769, 0, 0, 0, 314102564.102564, 373.931623931624, 373.931623931624, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(60) << -5459.40170940171, -934.82905982906, 0, 0, 0, 0, -299.18624465812, -0.00701121794871795, 0, -5459.40170940171, 934.82905982906, 0, 0, 0, 0, -299.18624465812, 0.00701121794871795, 0, -10245.7264957264, -2430.55555555556, 0, 0, 0, 186.965811965812, -224.435817307692, -0.0182291666666667, 0, -10245.7264957264, 2430.55555555556, 0, 0, 0, -186.965811965812, -224.435817307692, 0.0182291666666667, 0, 8675.21367521367, 0, 0, 0, 0, 0, 598.355662393162, 0, 0, 0, 3739.31623931624, 0, 0, 0, 373.931623931624, 747.863247863248, 0.0280448717948718, 0, 22735.0427350427, 0, 0, 0, 0, 0, 1196.75170940171, 0, 0, 0, -3739.31623931624, 0, 0, 0, -373.931623931624, 747.863247863248, -0.0280448717948718, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(61) << -934.82905982906, -2280.98290598291, 0, 0, 0, 130.876068376068, -0.00701121794871795, -299.162406517094, 0, 934.82905982906, -2280.98290598291, 0, 0, 0, 130.876068376068, 0.00701121794871795, -299.162406517094, 0, -1308.76068376068, -2206.19658119658, 0, 0, 0, 579.594017094017, -0.00981570512820513, -224.375520833333, 0, 1308.76068376068, -2206.19658119658, 0, 0, 0, 579.594017094017, 0.00981570512820513, -224.375520833333, 0, 0, -3290.59829059829, 0, 0, 0, -448.717948717949, 0, 598.265918803419, 0, 3739.31623931624, 0, 0, 0, 0, -373.931623931624, 0.0280448717948718, 747.863247863248, 0, 0, 12264.9572649573, 0, 0, 0, 3141.02564102564, 0, 1196.67318376068, 0, -3739.31623931624, 0, 0, 0, 0, -373.931623931624, -0.0280448717948718, 747.863247863248, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(62) << 0, 0, -1720.08547008547, 0, 523.504273504273, 0, 0, 0, -1047.02144764957, 0, 0, -1720.08547008547, 0, 523.504273504273, 0, 0, 0, -1047.02144764957, 0, 0, -2767.09401709402, 747.863247863248, 1196.5811965812, 0, 0, 0, -785.277163461538, 0, 0, -2767.09401709402, -747.863247863248, 1196.5811965812, 0, 0, 0, -785.277163461538, 0, 0, 1196.5811965812, 0, -1794.87179487179, 0, 0, 0, 2094.02606837607, 0, 0, 0, 1495.7264957265, -1495.7264957265, 0, 0, 0, 2617.52136752137, 0, 0, 7777.77777777778, 0, 3589.74358974359, 0, 0, 0, 4188.09252136752, 0, 0, 0, -1495.7264957265, -1495.7264957265, 0, 0, 0, 2617.52136752137;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(63) << -529487179.48718, -314102564.102564, 0, 0, 0, 94230769.2307692, -2206.19658119658, -1308.76068376068, 0, -547435897.435898, 224358974.358974, 0, 0, 0, 94230769.2307692, -2280.98290598291, 934.82905982906, 0, -547435897.435898, -224358974.358974, 0, 0, 0, 94230769.2307692, -2280.98290598291, -934.82905982906, 0, -529487179.48718, 314102564.102564, 0, 0, 0, 94230769.2307692, -2206.19658119658, 1308.76068376068, 0, 0, 897435897.435898, 0, 0, 0, -269230769.230769, 0, 3739.31623931624, 0, -789743589.74359, 0, 0, 0, 0, -323076923.076923, -3290.59829059829, 0, 0, 0, -897435897.435898, 0, 0, 0, -269230769.230769, 0, -3739.31623931624, 0, 2943589743.58974, 0, 0, 0, 0, -323076923.076923, 12264.9572649573, 0, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(64) << -583333333.333333, -2458974358.97436, 0, 0, 0, 134615384.615385, -2430.55555555556, -10245.7264957264, 0, 224358974.358974, -1310256410.25641, 0, 0, 0, 0, 934.82905982906, -5459.40170940171, 0, -224358974.358974, -1310256410.25641, 0, 0, 0, 0, -934.82905982906, -5459.40170940171, 0, 583333333.333333, -2458974358.97436, 0, 0, 0, -134615384.615385, 2430.55555555556, -10245.7264957264, 0, 897435897.435898, 0, 0, 0, 0, 269230769.230769, 3739.31623931624, 0, 0, 0, 2082051282.05128, 0, 0, 0, 0, 0, 8675.21367521367, 0, -897435897.435898, 0, 0, 0, 0, -269230769.230769, -3739.31623931624, 0, 0, 0, 5456410256.41026, 0, 0, 0, 0, 0, 22735.0427350427, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(65) << 0, 0, -664102564.102564, 62820512.8205128, 89743589.7435897, 0, 0, 0, -2767.09401709402, 0, 0, -412820512.820513, 62820512.8205128, 0, 0, 0, 0, -1720.08547008547, 0, 0, -412820512.820513, 62820512.8205128, 0, 0, 0, 0, -1720.08547008547, 0, 0, -664102564.102564, 62820512.8205128, -89743589.7435897, 0, 0, 0, -2767.09401709402, 0, 0, 0, -179487179.48718, 179487179.48718, 0, 0, 0, 0, 0, 0, 287179487.179487, -215384615.384615, 0, 0, 0, 0, 1196.5811965812, 0, 0, 0, -179487179.48718, -179487179.48718, 0, 0, 0, 0, 0, 0, 1866666666.66667, -215384615.384615, 0, 0, 0, 0, 7777.77777777778;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(66) << 0, 0, -116666666.666667, -26927489.3162393, -2617.52136752137, 0, 0, 0, 299.145299145299, 0, 0, -62820512.8205128, -35901997.8632479, 1869.65811965812, 0, 0, 0, 523.504273504273, 0, 0, -62820512.8205128, -35901997.8632479, -1869.65811965812, 0, 0, 0, 523.504273504273, 0, 0, -116666666.666667, -26927489.3162393, 2617.52136752137, 0, 0, 0, 299.145299145299, 0, 0, 179487179.48718, 89743589.7435897, 7478.63247863248, 0, 0, 0, -1495.7264957265, 0, 0, 215384615.384615, 71788290.5982906, 0, 0, 0, 0, -1794.87179487179, 0, 0, 179487179.48718, 89743589.7435897, -7478.63247863248, 0, 0, 0, -1495.7264957265, 0, 0, -215384615.384615, 143614273.504274, 0, 0, 0, 0, -3589.74358974359;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(67) << 0, 0, -89743589.7435897, -4861.11111111111, -26943568.3760684, 0, 0, 0, 747.863247863248, 0, 0, 0, 1869.65811965812, -35908354.7008547, 0, 0, 0, 0, 0, 0, 0, -1869.65811965812, -35908354.7008547, 0, 0, 0, 0, 0, 0, 89743589.7435897, 4861.11111111111, -26943568.3760684, 0, 0, 0, -747.863247863248, 0, 0, -179487179.48718, 7478.63247863248, 89743589.7435897, 0, 0, 0, 1495.7264957265, 0, 0, 0, 0, 71812222.2222222, 0, 0, 0, 0, 0, 0, 179487179.48718, -7478.63247863248, 89743589.7435897, 0, 0, 0, -1495.7264957265, 0, 0, 0, 0, 143635213.675214, 0, 0, 0, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(68) << -175000000, -134615384.615385, 0, 0, 0, -94236303.4188034, -205.662393162393, 186.965811965812, 0, -94230769.2307692, 0, 0, 0, 0, -125644465.811966, 130.876068376068, 0, 0, -94230769.2307692, 0, 0, 0, 0, -125644465.811966, 130.876068376068, 0, 0, -175000000, 134615384.615385, 0, 0, 0, -94236303.4188034, -205.662393162393, -186.965811965812, 0, 269230769.230769, -269230769.230769, 0, 0, 0, 314102564.102564, -373.931623931624, 373.931623931624, 0, 323076923.076923, 0, 0, 0, 0, 251284444.444444, -448.717948717949, 0, 0, 269230769.230769, 269230769.230769, 0, 0, 0, 314102564.102564, -373.931623931624, -373.931623931624, 0, -323076923.076923, 0, 0, 0, 0, 502579658.119658, -3141.02564102564, 0, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(69) << -2206.19658119658, -1308.76068376068, 0, 0, 0, -579.594017094017, -224.375520833333, -0.00981570512820513, 0, -2280.98290598291, 934.82905982906, 0, 0, 0, -130.876068376068, -299.162406517094, 0.00701121794871795, 0, -2280.98290598291, -934.82905982906, 0, 0, 0, -130.876068376068, -299.162406517094, -0.00701121794871795, 0, -2206.19658119658, 1308.76068376068, 0, 0, 0, -579.594017094017, -224.375520833333, 0.00981570512820513, 0, 0, 3739.31623931624, 0, 0, 0, 373.931623931624, 747.863247863248, 0.0280448717948718, 0, -3290.59829059829, 0, 0, 0, 0, 448.717948717949, 598.265918803419, 0, 0, 0, -3739.31623931624, 0, 0, 0, 373.931623931624, 747.863247863248, -0.0280448717948718, 0, 12264.9572649573, 0, 0, 0, 0, -3141.02564102564, 1196.67318376068, 0, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(70) << -2430.55555555556, -10245.7264957264, 0, 0, 0, -186.965811965812, -0.0182291666666667, -224.435817307692, 0, 934.82905982906, -5459.40170940171, 0, 0, 0, 0, 0.00701121794871795, -299.18624465812, 0, -934.82905982906, -5459.40170940171, 0, 0, 0, 0, -0.00701121794871795, -299.18624465812, 0, 2430.55555555556, -10245.7264957264, 0, 0, 0, 186.965811965812, 0.0182291666666667, -224.435817307692, 0, 3739.31623931624, 0, 0, 0, 0, -373.931623931624, 0.0280448717948718, 747.863247863248, 0, 0, 8675.21367521367, 0, 0, 0, 0, 0, 598.355662393162, 0, -3739.31623931624, 0, 0, 0, 0, 373.931623931624, -0.0280448717948718, 747.863247863248, 0, 0, 22735.0427350427, 0, 0, 0, 0, 0, 1196.75170940171, 0;
    //Expected_JacobianK_NoDispNoVelWithDamping.row(71) << 0, 0, -2767.09401709402, -1196.5811965812, -747.863247863248, 0, 0, 0, -785.277163461538, 0, 0, -1720.08547008547, -523.504273504273, 0, 0, 0, 0, -1047.02144764957, 0, 0, -1720.08547008547, -523.504273504273, 0, 0, 0, 0, -1047.02144764957, 0, 0, -2767.09401709402, -1196.5811965812, 747.863247863248, 0, 0, 0, -785.277163461538, 0, 0, 0, 1495.7264957265, -1495.7264957265, 0, 0, 0, 2617.52136752137, 0, 0, 1196.5811965812, 1794.87179487179, 0, 0, 0, 0, 2094.02606837607, 0, 0, 0, 1495.7264957265, 1495.7264957265, 0, 0, 0, 2617.52136752137, 0, 0, 7777.77777777778, -3589.74358974359, 0, 0, 0, 0, 4188.09252136752;

    //ChMatrixNM<double, 72, 72> Expected_JacobianR_NoDispNoVelWithDamping;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(0) << 21000000, 9535256.41025641, 0, 0, 0, -807692.307692308, 87.5, 39.730235042735, 0, 10320512.8205128, -336538.461538461, 0, 0, 0, 538461.538461538, 43.0021367521368, -1.40224358974359, 0, 9288461.53846154, 3926282.05128205, 0, 0, 0, 201923.076923077, 38.7019230769231, 16.3595085470085, 0, 7852564.1025641, 336538.461538461, 0, 0, 0, 201923.076923077, 32.7190170940171, 1.40224358974359, 0, -24589743.5897436, -3141025.64102564, 0, 0, 0, -1346153.84615385, -102.457264957265, -13.0876068376068, 0, -5474358.97435897, -2243589.74358974, 0, 0, 0, 942307.692307692, -22.8098290598291, -9.3482905982906, 0, -13102564.1025641, -2243589.74358974, 0, 0, 0, 0, -54.5940170940171, -9.3482905982906, 0, -5294871.7948718, -5833333.33333333, 0, 0, 0, -1750000, -22.0619658119658, -24.3055555555556, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(1) << 9535256.41025641, 21000000, 0, 0, 0, -807692.307692308, 39.730235042735, 87.5, 0, 336538.461538461, 7852564.1025641, 0, 0, 0, 201923.076923077, 1.40224358974359, 32.7190170940171, 0, 3926282.05128205, 9288461.53846154, 0, 0, 0, 201923.076923077, 16.3595085470085, 38.7019230769231, 0, -336538.461538461, 10320512.8205128, 0, 0, 0, 538461.538461538, -1.40224358974359, 43.0021367521368, 0, -5833333.33333333, -5294871.7948718, 0, 0, 0, -1750000, -24.3055555555556, -22.0619658119658, 0, -2243589.74358974, -13102564.1025641, 0, 0, 0, 0, -9.3482905982906, -54.5940170940171, 0, -2243589.74358974, -5474358.97435897, 0, 0, 0, 942307.692307692, -9.3482905982906, -22.8098290598291, 0, -3141025.64102564, -24589743.5897436, 0, 0, 0, -1346153.84615385, -13.0876068376068, -102.457264957265, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(2) << 0, 0, 9333333.33333333, -538461.538461539, -538461.538461539, 0, 0, 0, 38.8888888888889, 0, 0, 4038461.53846154, 358974.358974359, 134615.384615385, 0, 0, 0, 16.8269230769231, 0, 0, 4128205.12820513, 134615.384615385, 134615.384615385, 0, 0, 0, 17.2008547008547, 0, 0, 4038461.53846154, 134615.384615385, 358974.358974359, 0, 0, 0, 16.8269230769231, 0, 0, -6641025.64102564, -897435.897435898, -1166666.66666667, 0, 0, 0, -27.6709401709402, 0, 0, -4128205.12820513, 628205.128205128, 0, 0, 0, 0, -17.2008547008547, 0, 0, -4128205.12820513, 0, 628205.128205128, 0, 0, 0, -17.2008547008547, 0, 0, -6641025.64102564, -1166666.66666667, -897435.897435898, 0, 0, 0, -27.6709401709402;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(3) << 0, 0, -538461.538461539, 269405.769230769, 79.4604700854701, 0, 0, 0, -8.97435897435897, 0, 0, -358974.358974359, 89829.594017094, -2.80448717948718, 0, 0, 0, 2.99145299145299, 0, 0, -134615.384615385, 134692.788461538, 32.7190170940171, 0, 0, 0, 1.12179487179487, 0, 0, 134615.384615385, 89809.0277777778, 2.80448717948718, 0, 0, 0, 2.24358974358974, 0, 0, 897435.897435898, -269435.683760684, -26.1752136752137, 0, 0, 0, -7.47863247863248, 0, 0, -628205.128205128, -359019.978632479, -18.6965811965812, 0, 0, 0, 5.23504273504273, 0, 0, 0, -359083.547008547, -18.6965811965812, 0, 0, 0, 0, 0, 0, 628205.128205128, -269274.893162393, -48.6111111111111, 0, 0, 0, -11.965811965812;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(4) << 0, 0, -538461.538461539, 79.4604700854701, 269405.769230769, 0, 0, 0, -8.97435897435897, 0, 0, 134615.384615385, 2.80448717948718, 89809.0277777778, 0, 0, 0, 2.24358974358974, 0, 0, -134615.384615385, 32.7190170940171, 134692.788461538, 0, 0, 0, 1.12179487179487, 0, 0, -358974.358974359, -2.80448717948718, 89829.594017094, 0, 0, 0, 2.99145299145299, 0, 0, 628205.128205128, -48.6111111111111, -269274.893162393, 0, 0, 0, -11.965811965812, 0, 0, 0, -18.6965811965812, -359083.547008547, 0, 0, 0, 0, 0, 0, -628205.128205128, -18.6965811965812, -359019.978632479, 0, 0, 0, 5.23504273504273, 0, 0, 897435.897435898, -26.1752136752137, -269435.683760684, 0, 0, 0, -7.47863247863248;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(5) << -807692.307692308, -807692.307692308, 0, 0, 0, 942385.47008547, -7.8525641025641, -7.8525641025641, 0, -538461.538461538, 201923.076923077, 0, 0, 0, 314136.217948718, 0.747863247863248, 1.96314102564103, 0, -201923.076923077, -201923.076923077, 0, 0, 0, 471188.247863248, 0.280448717948718, 0.280448717948718, 0, 201923.076923077, -538461.538461538, 0, 0, 0, 314136.217948718, 1.96314102564103, 0.747863247863248, 0, 1346153.84615385, 942307.692307692, 0, 0, 0, -942363.034188034, -1.86965811965812, -5.79594017094017, 0, -942307.692307692, 0, 0, 0, 0, -1256444.65811966, 1.30876068376068, 0, 0, 0, -942307.692307692, 0, 0, 0, -1256444.65811966, 0, 1.30876068376068, 0, 942307.692307692, 1346153.84615385, 0, 0, 0, -942363.034188034, -5.79594017094017, -1.86965811965812, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(6) << 87.5, 39.730235042735, 0, 0, 0, -7.8525641025641, 2.24424599358974, 0.000297976762820513, 0, 43.0021367521368, -1.40224358974359, 0, 0, 0, -0.747863247863248, 0.748185763888889, -1.05168269230769E-05, 0, 38.7019230769231, 16.3595085470085, 0, 0, 0, -0.280448717948718, 1.12208513621795, 0.000122696314102564, 0, 32.7190170940171, 1.40224358974359, 0, 0, 0, 1.96314102564103, 0.748108640491453, 1.05168269230769E-05, 0, -102.457264957265, -13.0876068376068, 0, 0, 0, 1.86965811965812, -2.24435817307692, -9.81570512820513E-05, 0, -22.8098290598291, -9.3482905982906, 0, 0, 0, -1.30876068376068, -2.99162406517094, -7.01121794871795E-05, 0, -54.5940170940171, -9.3482905982906, 0, 0, 0, 0, -2.9918624465812, -7.01121794871795E-05, 0, -22.0619658119658, -24.3055555555556, 0, 0, 0, -2.05662393162393, -2.24375520833333, -0.000182291666666667, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(7) << 39.730235042735, 87.5, 0, 0, 0, -7.8525641025641, 0.000297976762820513, 2.24424599358974, 0, 1.40224358974359, 32.7190170940171, 0, 0, 0, 1.96314102564103, 1.05168269230769E-05, 0.748108640491453, 0, 16.3595085470085, 38.7019230769231, 0, 0, 0, -0.280448717948718, 0.000122696314102564, 1.12208513621795, 0, -1.40224358974359, 43.0021367521368, 0, 0, 0, -0.747863247863248, -1.05168269230769E-05, 0.748185763888889, 0, -24.3055555555556, -22.0619658119658, 0, 0, 0, -2.05662393162393, -0.000182291666666667, -2.24375520833333, 0, -9.3482905982906, -54.5940170940171, 0, 0, 0, 0, -7.01121794871795E-05, -2.9918624465812, 0, -9.3482905982906, -22.8098290598291, 0, 0, 0, -1.30876068376068, -7.01121794871795E-05, -2.99162406517094, 0, -13.0876068376068, -102.457264957265, 0, 0, 0, 1.86965811965812, -9.81570512820513E-05, -2.24435817307692, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(8) << 0, 0, 38.8888888888889, -8.97435897435897, -8.97435897435897, 0, 0, 0, 7.85285576923077, 0, 0, 16.8269230769231, -2.99145299145299, 2.24358974358974, 0, 0, 0, 2.61764756944444, 0, 0, 17.2008547008547, -1.12179487179487, -1.12179487179487, 0, 0, 0, 3.92641105769231, 0, 0, 16.8269230769231, 2.24358974358974, -2.99145299145299, 0, 0, 0, 2.61764756944444, 0, 0, -27.6709401709402, 7.47863247863248, 2.99145299145299, 0, 0, 0, -7.85277163461538, 0, 0, -17.2008547008547, -5.23504273504273, 0, 0, 0, 0, -10.4702144764957, 0, 0, -17.2008547008547, 0, -5.23504273504273, 0, 0, 0, -10.4702144764957, 0, 0, -27.6709401709402, 2.99145299145299, 7.47863247863248, 0, 0, 0, -7.85277163461538;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(9) << 10320512.8205128, 336538.461538461, 0, 0, 0, -538461.538461538, 43.0021367521368, 1.40224358974359, 0, 21000000, -9535256.41025641, 0, 0, 0, 807692.307692308, 87.5, -39.730235042735, 0, 7852564.1025641, -336538.461538461, 0, 0, 0, -201923.076923077, 32.7190170940171, -1.40224358974359, 0, 9288461.53846154, -3926282.05128205, 0, 0, 0, -201923.076923077, 38.7019230769231, -16.3595085470085, 0, -24589743.5897436, 3141025.64102564, 0, 0, 0, 1346153.84615385, -102.457264957265, 13.0876068376068, 0, -5294871.7948718, 5833333.33333333, 0, 0, 0, 1750000, -22.0619658119658, 24.3055555555556, 0, -13102564.1025641, 2243589.74358974, 0, 0, 0, 0, -54.5940170940171, 9.3482905982906, 0, -5474358.97435897, 2243589.74358974, 0, 0, 0, -942307.692307692, -22.8098290598291, 9.3482905982906, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(10) << -336538.461538461, 7852564.1025641, 0, 0, 0, 201923.076923077, -1.40224358974359, 32.7190170940171, 0, -9535256.41025641, 21000000, 0, 0, 0, -807692.307692308, -39.730235042735, 87.5, 0, 336538.461538461, 10320512.8205128, 0, 0, 0, 538461.538461538, 1.40224358974359, 43.0021367521368, 0, -3926282.05128205, 9288461.53846154, 0, 0, 0, 201923.076923077, -16.3595085470085, 38.7019230769231, 0, 5833333.33333333, -5294871.7948718, 0, 0, 0, -1750000, 24.3055555555556, -22.0619658119658, 0, 3141025.64102564, -24589743.5897436, 0, 0, 0, -1346153.84615385, 13.0876068376068, -102.457264957265, 0, 2243589.74358974, -5474358.97435897, 0, 0, 0, 942307.692307692, 9.3482905982906, -22.8098290598291, 0, 2243589.74358974, -13102564.1025641, 0, 0, 0, 0, 9.3482905982906, -54.5940170940171, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(11) << 0, 0, 4038461.53846154, -358974.358974359, 134615.384615385, 0, 0, 0, 16.8269230769231, 0, 0, 9333333.33333333, 538461.538461539, -538461.538461539, 0, 0, 0, 38.8888888888889, 0, 0, 4038461.53846154, -134615.384615385, 358974.358974359, 0, 0, 0, 16.8269230769231, 0, 0, 4128205.12820513, -134615.384615385, 134615.384615385, 0, 0, 0, 17.2008547008547, 0, 0, -6641025.64102564, 897435.897435898, -1166666.66666667, 0, 0, 0, -27.6709401709402, 0, 0, -6641025.64102564, 1166666.66666667, -897435.897435898, 0, 0, 0, -27.6709401709402, 0, 0, -4128205.12820513, 0, 628205.128205128, 0, 0, 0, -17.2008547008547, 0, 0, -4128205.12820513, -628205.128205128, 0, 0, 0, 0, -17.2008547008547;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(12) << 0, 0, 358974.358974359, 89829.594017094, 2.80448717948718, 0, 0, 0, -2.99145299145299, 0, 0, 538461.538461539, 269405.769230769, -79.4604700854701, 0, 0, 0, 8.97435897435897, 0, 0, -134615.384615385, 89809.0277777778, -2.80448717948718, 0, 0, 0, -2.24358974358974, 0, 0, 134615.384615385, 134692.788461538, -32.7190170940171, 0, 0, 0, -1.12179487179487, 0, 0, -897435.897435898, -269435.683760684, 26.1752136752137, 0, 0, 0, 7.47863247863248, 0, 0, -628205.128205128, -269274.893162393, 48.6111111111111, 0, 0, 0, 11.965811965812, 0, 0, 0, -359083.547008547, 18.6965811965812, 0, 0, 0, 0, 0, 0, 628205.128205128, -359019.978632479, 18.6965811965812, 0, 0, 0, -5.23504273504273;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(13) << 0, 0, 134615.384615385, -2.80448717948718, 89809.0277777778, 0, 0, 0, 2.24358974358974, 0, 0, -538461.538461539, -79.4604700854701, 269405.769230769, 0, 0, 0, -8.97435897435897, 0, 0, -358974.358974359, 2.80448717948718, 89829.594017094, 0, 0, 0, 2.99145299145299, 0, 0, -134615.384615385, -32.7190170940171, 134692.788461538, 0, 0, 0, 1.12179487179487, 0, 0, 628205.128205128, 48.6111111111111, -269274.893162393, 0, 0, 0, -11.965811965812, 0, 0, 897435.897435898, 26.1752136752137, -269435.683760684, 0, 0, 0, -7.47863247863248, 0, 0, -628205.128205128, 18.6965811965812, -359019.978632479, 0, 0, 0, 5.23504273504273, 0, 0, 0, 18.6965811965812, -359083.547008547, 0, 0, 0, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(14) << 538461.538461538, 201923.076923077, 0, 0, 0, 314136.217948718, -0.747863247863248, 1.96314102564103, 0, 807692.307692308, -807692.307692308, 0, 0, 0, 942385.47008547, 7.8525641025641, -7.8525641025641, 0, -201923.076923077, -538461.538461538, 0, 0, 0, 314136.217948718, -1.96314102564103, 0.747863247863248, 0, 201923.076923077, -201923.076923077, 0, 0, 0, 471188.247863248, -0.280448717948718, 0.280448717948718, 0, -1346153.84615385, 942307.692307692, 0, 0, 0, -942363.034188034, 1.86965811965812, -5.79594017094017, 0, -942307.692307692, 1346153.84615385, 0, 0, 0, -942363.034188034, 5.79594017094017, -1.86965811965812, 0, 0, -942307.692307692, 0, 0, 0, -1256444.65811966, 0, 1.30876068376068, 0, 942307.692307692, 0, 0, 0, 0, -1256444.65811966, -1.30876068376068, 0, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(15) << 43.0021367521368, 1.40224358974359, 0, 0, 0, 0.747863247863248, 0.748185763888889, 1.05168269230769E-05, 0, 87.5, -39.730235042735, 0, 0, 0, 7.8525641025641, 2.24424599358974, -0.000297976762820513, 0, 32.7190170940171, -1.40224358974359, 0, 0, 0, -1.96314102564103, 0.748108640491453, -1.05168269230769E-05, 0, 38.7019230769231, -16.3595085470085, 0, 0, 0, 0.280448717948718, 1.12208513621795, -0.000122696314102564, 0, -102.457264957265, 13.0876068376068, 0, 0, 0, -1.86965811965812, -2.24435817307692, 9.81570512820513E-05, 0, -22.0619658119658, 24.3055555555556, 0, 0, 0, 2.05662393162393, -2.24375520833333, 0.000182291666666667, 0, -54.5940170940171, 9.3482905982906, 0, 0, 0, 0, -2.9918624465812, 7.01121794871795E-05, 0, -22.8098290598291, 9.3482905982906, 0, 0, 0, 1.30876068376068, -2.99162406517094, 7.01121794871795E-05, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(16) << -1.40224358974359, 32.7190170940171, 0, 0, 0, 1.96314102564103, -1.05168269230769E-05, 0.748108640491453, 0, -39.730235042735, 87.5, 0, 0, 0, -7.8525641025641, -0.000297976762820513, 2.24424599358974, 0, 1.40224358974359, 43.0021367521368, 0, 0, 0, -0.747863247863248, 1.05168269230769E-05, 0.748185763888889, 0, -16.3595085470085, 38.7019230769231, 0, 0, 0, -0.280448717948718, -0.000122696314102564, 1.12208513621795, 0, 24.3055555555556, -22.0619658119658, 0, 0, 0, -2.05662393162393, 0.000182291666666667, -2.24375520833333, 0, 13.0876068376068, -102.457264957265, 0, 0, 0, 1.86965811965812, 9.81570512820513E-05, -2.24435817307692, 0, 9.3482905982906, -22.8098290598291, 0, 0, 0, -1.30876068376068, 7.01121794871795E-05, -2.99162406517094, 0, 9.3482905982906, -54.5940170940171, 0, 0, 0, 0, 7.01121794871795E-05, -2.9918624465812, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(17) << 0, 0, 16.8269230769231, 2.99145299145299, 2.24358974358974, 0, 0, 0, 2.61764756944444, 0, 0, 38.8888888888889, 8.97435897435897, -8.97435897435897, 0, 0, 0, 7.85285576923077, 0, 0, 16.8269230769231, -2.24358974358974, -2.99145299145299, 0, 0, 0, 2.61764756944444, 0, 0, 17.2008547008547, 1.12179487179487, -1.12179487179487, 0, 0, 0, 3.92641105769231, 0, 0, -27.6709401709402, -7.47863247863248, 2.99145299145299, 0, 0, 0, -7.85277163461538, 0, 0, -27.6709401709402, -2.99145299145299, 7.47863247863248, 0, 0, 0, -7.85277163461538, 0, 0, -17.2008547008547, 0, -5.23504273504273, 0, 0, 0, -10.4702144764957, 0, 0, -17.2008547008547, 5.23504273504273, 0, 0, 0, 0, -10.4702144764957;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(18) << 9288461.53846154, 3926282.05128205, 0, 0, 0, -201923.076923077, 38.7019230769231, 16.3595085470085, 0, 7852564.1025641, 336538.461538461, 0, 0, 0, -201923.076923077, 32.7190170940171, 1.40224358974359, 0, 21000000, 9535256.41025641, 0, 0, 0, 807692.307692308, 87.5, 39.730235042735, 0, 10320512.8205128, -336538.461538461, 0, 0, 0, -538461.538461538, 43.0021367521368, -1.40224358974359, 0, -13102564.1025641, -2243589.74358974, 0, 0, 0, 0, -54.5940170940171, -9.3482905982906, 0, -5294871.7948718, -5833333.33333333, 0, 0, 0, 1750000, -22.0619658119658, -24.3055555555556, 0, -24589743.5897436, -3141025.64102564, 0, 0, 0, 1346153.84615385, -102.457264957265, -13.0876068376068, 0, -5474358.97435897, -2243589.74358974, 0, 0, 0, -942307.692307692, -22.8098290598291, -9.3482905982906, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(19) << 3926282.05128205, 9288461.53846154, 0, 0, 0, -201923.076923077, 16.3595085470085, 38.7019230769231, 0, -336538.461538461, 10320512.8205128, 0, 0, 0, -538461.538461538, -1.40224358974359, 43.0021367521368, 0, 9535256.41025641, 21000000, 0, 0, 0, 807692.307692308, 39.730235042735, 87.5, 0, 336538.461538461, 7852564.1025641, 0, 0, 0, -201923.076923077, 1.40224358974359, 32.7190170940171, 0, -2243589.74358974, -5474358.97435897, 0, 0, 0, -942307.692307692, -9.3482905982906, -22.8098290598291, 0, -3141025.64102564, -24589743.5897436, 0, 0, 0, 1346153.84615385, -13.0876068376068, -102.457264957265, 0, -5833333.33333333, -5294871.7948718, 0, 0, 0, 1750000, -24.3055555555556, -22.0619658119658, 0, -2243589.74358974, -13102564.1025641, 0, 0, 0, 0, -9.3482905982906, -54.5940170940171, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(20) << 0, 0, 4128205.12820513, -134615.384615385, -134615.384615385, 0, 0, 0, 17.2008547008547, 0, 0, 4038461.53846154, -134615.384615385, -358974.358974359, 0, 0, 0, 16.8269230769231, 0, 0, 9333333.33333333, 538461.538461539, 538461.538461539, 0, 0, 0, 38.8888888888889, 0, 0, 4038461.53846154, -358974.358974359, -134615.384615385, 0, 0, 0, 16.8269230769231, 0, 0, -4128205.12820513, 0, -628205.128205128, 0, 0, 0, -17.2008547008547, 0, 0, -6641025.64102564, 1166666.66666667, 897435.897435898, 0, 0, 0, -27.6709401709402, 0, 0, -6641025.64102564, 897435.897435898, 1166666.66666667, 0, 0, 0, -27.6709401709402, 0, 0, -4128205.12820513, -628205.128205128, 0, 0, 0, 0, -17.2008547008547;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(21) << 0, 0, 134615.384615385, 134692.788461538, 32.7190170940171, 0, 0, 0, -1.12179487179487, 0, 0, -134615.384615385, 89809.0277777778, 2.80448717948718, 0, 0, 0, -2.24358974358974, 0, 0, 538461.538461539, 269405.769230769, 79.4604700854701, 0, 0, 0, 8.97435897435897, 0, 0, 358974.358974359, 89829.594017094, -2.80448717948718, 0, 0, 0, -2.99145299145299, 0, 0, 0, -359083.547008547, -18.6965811965812, 0, 0, 0, 0, 0, 0, -628205.128205128, -269274.893162393, -48.6111111111111, 0, 0, 0, 11.965811965812, 0, 0, -897435.897435898, -269435.683760684, -26.1752136752137, 0, 0, 0, 7.47863247863248, 0, 0, 628205.128205128, -359019.978632479, -18.6965811965812, 0, 0, 0, -5.23504273504273;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(22) << 0, 0, 134615.384615385, 32.7190170940171, 134692.788461538, 0, 0, 0, -1.12179487179487, 0, 0, 358974.358974359, -2.80448717948718, 89829.594017094, 0, 0, 0, -2.99145299145299, 0, 0, 538461.538461539, 79.4604700854701, 269405.769230769, 0, 0, 0, 8.97435897435897, 0, 0, -134615.384615385, 2.80448717948718, 89809.0277777778, 0, 0, 0, -2.24358974358974, 0, 0, 628205.128205128, -18.6965811965812, -359019.978632479, 0, 0, 0, -5.23504273504273, 0, 0, -897435.897435898, -26.1752136752137, -269435.683760684, 0, 0, 0, 7.47863247863248, 0, 0, -628205.128205128, -48.6111111111111, -269274.893162393, 0, 0, 0, 11.965811965812, 0, 0, 0, -18.6965811965812, -359083.547008547, 0, 0, 0, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(23) << 201923.076923077, 201923.076923077, 0, 0, 0, 471188.247863248, -0.280448717948718, -0.280448717948718, 0, -201923.076923077, 538461.538461538, 0, 0, 0, 314136.217948718, -1.96314102564103, -0.747863247863248, 0, 807692.307692308, 807692.307692308, 0, 0, 0, 942385.47008547, 7.8525641025641, 7.8525641025641, 0, 538461.538461538, -201923.076923077, 0, 0, 0, 314136.217948718, -0.747863247863248, -1.96314102564103, 0, 0, 942307.692307692, 0, 0, 0, -1256444.65811966, 0, -1.30876068376068, 0, -942307.692307692, -1346153.84615385, 0, 0, 0, -942363.034188034, 5.79594017094017, 1.86965811965812, 0, -1346153.84615385, -942307.692307692, 0, 0, 0, -942363.034188034, 1.86965811965812, 5.79594017094017, 0, 942307.692307692, 0, 0, 0, 0, -1256444.65811966, -1.30876068376068, 0, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(24) << 38.7019230769231, 16.3595085470085, 0, 0, 0, 0.280448717948718, 1.12208513621795, 0.000122696314102564, 0, 32.7190170940171, 1.40224358974359, 0, 0, 0, -1.96314102564103, 0.748108640491453, 1.05168269230769E-05, 0, 87.5, 39.730235042735, 0, 0, 0, 7.8525641025641, 2.24424599358974, 0.000297976762820513, 0, 43.0021367521368, -1.40224358974359, 0, 0, 0, 0.747863247863248, 0.748185763888889, -1.05168269230769E-05, 0, -54.5940170940171, -9.3482905982906, 0, 0, 0, 0, -2.9918624465812, -7.01121794871795E-05, 0, -22.0619658119658, -24.3055555555556, 0, 0, 0, 2.05662393162393, -2.24375520833333, -0.000182291666666667, 0, -102.457264957265, -13.0876068376068, 0, 0, 0, -1.86965811965812, -2.24435817307692, -9.81570512820513E-05, 0, -22.8098290598291, -9.3482905982906, 0, 0, 0, 1.30876068376068, -2.99162406517094, -7.01121794871795E-05, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(25) << 16.3595085470085, 38.7019230769231, 0, 0, 0, 0.280448717948718, 0.000122696314102564, 1.12208513621795, 0, -1.40224358974359, 43.0021367521368, 0, 0, 0, 0.747863247863248, -1.05168269230769E-05, 0.748185763888889, 0, 39.730235042735, 87.5, 0, 0, 0, 7.8525641025641, 0.000297976762820513, 2.24424599358974, 0, 1.40224358974359, 32.7190170940171, 0, 0, 0, -1.96314102564103, 1.05168269230769E-05, 0.748108640491453, 0, -9.3482905982906, -22.8098290598291, 0, 0, 0, 1.30876068376068, -7.01121794871795E-05, -2.99162406517094, 0, -13.0876068376068, -102.457264957265, 0, 0, 0, -1.86965811965812, -9.81570512820513E-05, -2.24435817307692, 0, -24.3055555555556, -22.0619658119658, 0, 0, 0, 2.05662393162393, -0.000182291666666667, -2.24375520833333, 0, -9.3482905982906, -54.5940170940171, 0, 0, 0, 0, -7.01121794871795E-05, -2.9918624465812, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(26) << 0, 0, 17.2008547008547, 1.12179487179487, 1.12179487179487, 0, 0, 0, 3.92641105769231, 0, 0, 16.8269230769231, -2.24358974358974, 2.99145299145299, 0, 0, 0, 2.61764756944444, 0, 0, 38.8888888888889, 8.97435897435897, 8.97435897435897, 0, 0, 0, 7.85285576923077, 0, 0, 16.8269230769231, 2.99145299145299, -2.24358974358974, 0, 0, 0, 2.61764756944444, 0, 0, -17.2008547008547, 0, 5.23504273504273, 0, 0, 0, -10.4702144764957, 0, 0, -27.6709401709402, -2.99145299145299, -7.47863247863248, 0, 0, 0, -7.85277163461538, 0, 0, -27.6709401709402, -7.47863247863248, -2.99145299145299, 0, 0, 0, -7.85277163461538, 0, 0, -17.2008547008547, 5.23504273504273, 0, 0, 0, 0, -10.4702144764957;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(27) << 7852564.1025641, -336538.461538461, 0, 0, 0, 201923.076923077, 32.7190170940171, -1.40224358974359, 0, 9288461.53846154, -3926282.05128205, 0, 0, 0, 201923.076923077, 38.7019230769231, -16.3595085470085, 0, 10320512.8205128, 336538.461538461, 0, 0, 0, 538461.538461538, 43.0021367521368, 1.40224358974359, 0, 21000000, -9535256.41025641, 0, 0, 0, -807692.307692308, 87.5, -39.730235042735, 0, -13102564.1025641, 2243589.74358974, 0, 0, 0, 0, -54.5940170940171, 9.3482905982906, 0, -5474358.97435897, 2243589.74358974, 0, 0, 0, 942307.692307692, -22.8098290598291, 9.3482905982906, 0, -24589743.5897436, 3141025.64102564, 0, 0, 0, -1346153.84615385, -102.457264957265, 13.0876068376068, 0, -5294871.7948718, 5833333.33333333, 0, 0, 0, -1750000, -22.0619658119658, 24.3055555555556, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(28) << 336538.461538461, 10320512.8205128, 0, 0, 0, -538461.538461538, 1.40224358974359, 43.0021367521368, 0, -3926282.05128205, 9288461.53846154, 0, 0, 0, -201923.076923077, -16.3595085470085, 38.7019230769231, 0, -336538.461538461, 7852564.1025641, 0, 0, 0, -201923.076923077, -1.40224358974359, 32.7190170940171, 0, -9535256.41025641, 21000000, 0, 0, 0, 807692.307692308, -39.730235042735, 87.5, 0, 2243589.74358974, -5474358.97435897, 0, 0, 0, -942307.692307692, 9.3482905982906, -22.8098290598291, 0, 2243589.74358974, -13102564.1025641, 0, 0, 0, 0, 9.3482905982906, -54.5940170940171, 0, 5833333.33333333, -5294871.7948718, 0, 0, 0, 1750000, 24.3055555555556, -22.0619658119658, 0, 3141025.64102564, -24589743.5897436, 0, 0, 0, 1346153.84615385, 13.0876068376068, -102.457264957265, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(29) << 0, 0, 4038461.53846154, 134615.384615385, -358974.358974359, 0, 0, 0, 16.8269230769231, 0, 0, 4128205.12820513, 134615.384615385, -134615.384615385, 0, 0, 0, 17.2008547008547, 0, 0, 4038461.53846154, 358974.358974359, -134615.384615385, 0, 0, 0, 16.8269230769231, 0, 0, 9333333.33333333, -538461.538461539, 538461.538461539, 0, 0, 0, 38.8888888888889, 0, 0, -4128205.12820513, 0, -628205.128205128, 0, 0, 0, -17.2008547008547, 0, 0, -4128205.12820513, 628205.128205128, 0, 0, 0, 0, -17.2008547008547, 0, 0, -6641025.64102564, -897435.897435898, 1166666.66666667, 0, 0, 0, -27.6709401709402, 0, 0, -6641025.64102564, -1166666.66666667, 897435.897435898, 0, 0, 0, -27.6709401709402;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(30) << 0, 0, 134615.384615385, 89809.0277777778, -2.80448717948718, 0, 0, 0, 2.24358974358974, 0, 0, -134615.384615385, 134692.788461538, -32.7190170940171, 0, 0, 0, 1.12179487179487, 0, 0, -358974.358974359, 89829.594017094, 2.80448717948718, 0, 0, 0, 2.99145299145299, 0, 0, -538461.538461539, 269405.769230769, -79.4604700854701, 0, 0, 0, -8.97435897435897, 0, 0, 0, -359083.547008547, 18.6965811965812, 0, 0, 0, 0, 0, 0, -628205.128205128, -359019.978632479, 18.6965811965812, 0, 0, 0, 5.23504273504273, 0, 0, 897435.897435898, -269435.683760684, 26.1752136752137, 0, 0, 0, -7.47863247863248, 0, 0, 628205.128205128, -269274.893162393, 48.6111111111111, 0, 0, 0, -11.965811965812;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(31) << 0, 0, 358974.358974359, 2.80448717948718, 89829.594017094, 0, 0, 0, -2.99145299145299, 0, 0, 134615.384615385, -32.7190170940171, 134692.788461538, 0, 0, 0, -1.12179487179487, 0, 0, -134615.384615385, -2.80448717948718, 89809.0277777778, 0, 0, 0, -2.24358974358974, 0, 0, 538461.538461539, -79.4604700854701, 269405.769230769, 0, 0, 0, 8.97435897435897, 0, 0, 628205.128205128, 18.6965811965812, -359019.978632479, 0, 0, 0, -5.23504273504273, 0, 0, 0, 18.6965811965812, -359083.547008547, 0, 0, 0, 0, 0, 0, -628205.128205128, 48.6111111111111, -269274.893162393, 0, 0, 0, 11.965811965812, 0, 0, -897435.897435898, 26.1752136752137, -269435.683760684, 0, 0, 0, 7.47863247863248;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(32) << 201923.076923077, 538461.538461538, 0, 0, 0, 314136.217948718, 1.96314102564103, -0.747863247863248, 0, -201923.076923077, 201923.076923077, 0, 0, 0, 471188.247863248, 0.280448717948718, -0.280448717948718, 0, -538461.538461538, -201923.076923077, 0, 0, 0, 314136.217948718, 0.747863247863248, -1.96314102564103, 0, -807692.307692308, 807692.307692308, 0, 0, 0, 942385.47008547, -7.8525641025641, 7.8525641025641, 0, 0, 942307.692307692, 0, 0, 0, -1256444.65811966, 0, -1.30876068376068, 0, -942307.692307692, 0, 0, 0, 0, -1256444.65811966, 1.30876068376068, 0, 0, 1346153.84615385, -942307.692307692, 0, 0, 0, -942363.034188034, -1.86965811965812, 5.79594017094017, 0, 942307.692307692, -1346153.84615385, 0, 0, 0, -942363.034188034, -5.79594017094017, 1.86965811965812, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(33) << 32.7190170940171, -1.40224358974359, 0, 0, 0, 1.96314102564103, 0.748108640491453, -1.05168269230769E-05, 0, 38.7019230769231, -16.3595085470085, 0, 0, 0, -0.280448717948718, 1.12208513621795, -0.000122696314102564, 0, 43.0021367521368, 1.40224358974359, 0, 0, 0, -0.747863247863248, 0.748185763888889, 1.05168269230769E-05, 0, 87.5, -39.730235042735, 0, 0, 0, -7.8525641025641, 2.24424599358974, -0.000297976762820513, 0, -54.5940170940171, 9.3482905982906, 0, 0, 0, 0, -2.9918624465812, 7.01121794871795E-05, 0, -22.8098290598291, 9.3482905982906, 0, 0, 0, -1.30876068376068, -2.99162406517094, 7.01121794871795E-05, 0, -102.457264957265, 13.0876068376068, 0, 0, 0, 1.86965811965812, -2.24435817307692, 9.81570512820513E-05, 0, -22.0619658119658, 24.3055555555556, 0, 0, 0, -2.05662393162393, -2.24375520833333, 0.000182291666666667, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(34) << 1.40224358974359, 43.0021367521368, 0, 0, 0, 0.747863247863248, 1.05168269230769E-05, 0.748185763888889, 0, -16.3595085470085, 38.7019230769231, 0, 0, 0, 0.280448717948718, -0.000122696314102564, 1.12208513621795, 0, -1.40224358974359, 32.7190170940171, 0, 0, 0, -1.96314102564103, -1.05168269230769E-05, 0.748108640491453, 0, -39.730235042735, 87.5, 0, 0, 0, 7.8525641025641, -0.000297976762820513, 2.24424599358974, 0, 9.3482905982906, -22.8098290598291, 0, 0, 0, 1.30876068376068, 7.01121794871795E-05, -2.99162406517094, 0, 9.3482905982906, -54.5940170940171, 0, 0, 0, 0, 7.01121794871795E-05, -2.9918624465812, 0, 24.3055555555556, -22.0619658119658, 0, 0, 0, 2.05662393162393, 0.000182291666666667, -2.24375520833333, 0, 13.0876068376068, -102.457264957265, 0, 0, 0, -1.86965811965812, 9.81570512820513E-05, -2.24435817307692, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(35) << 0, 0, 16.8269230769231, 2.24358974358974, 2.99145299145299, 0, 0, 0, 2.61764756944444, 0, 0, 17.2008547008547, -1.12179487179487, 1.12179487179487, 0, 0, 0, 3.92641105769231, 0, 0, 16.8269230769231, -2.99145299145299, -2.24358974358974, 0, 0, 0, 2.61764756944444, 0, 0, 38.8888888888889, -8.97435897435897, 8.97435897435897, 0, 0, 0, 7.85285576923077, 0, 0, -17.2008547008547, 0, 5.23504273504273, 0, 0, 0, -10.4702144764957, 0, 0, -17.2008547008547, -5.23504273504273, 0, 0, 0, 0, -10.4702144764957, 0, 0, -27.6709401709402, 7.47863247863248, -2.99145299145299, 0, 0, 0, -7.85277163461538, 0, 0, -27.6709401709402, 2.99145299145299, -7.47863247863248, 0, 0, 0, -7.85277163461538;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(36) << -24589743.5897436, -5833333.33333333, 0, 0, 0, 1346153.84615385, -102.457264957265, -24.3055555555556, 0, -24589743.5897436, 5833333.33333333, 0, 0, 0, -1346153.84615385, -102.457264957265, 24.3055555555556, 0, -13102564.1025641, -2243589.74358974, 0, 0, 0, 0, -54.5940170940171, -9.3482905982906, 0, -13102564.1025641, 2243589.74358974, 0, 0, 0, 0, -54.5940170940171, 9.3482905982906, 0, 54564102.5641026, 0, 0, 0, 0, 0, 227.350427350427, 0, 0, 0, -8974358.97435897, 0, 0, 0, -2692307.69230769, 0, -37.3931623931624, 0, 20820512.8205128, 0, 0, 0, 0, 0, 86.7521367521367, 0, 0, 0, 8974358.97435897, 0, 0, 0, 2692307.69230769, 0, 37.3931623931624, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(37) << -3141025.64102564, -5294871.7948718, 0, 0, 0, 942307.692307692, -13.0876068376068, -22.0619658119658, 0, 3141025.64102564, -5294871.7948718, 0, 0, 0, 942307.692307692, 13.0876068376068, -22.0619658119658, 0, -2243589.74358974, -5474358.97435897, 0, 0, 0, 942307.692307692, -9.3482905982906, -22.8098290598291, 0, 2243589.74358974, -5474358.97435897, 0, 0, 0, 942307.692307692, 9.3482905982906, -22.8098290598291, 0, 0, 29435897.4358974, 0, 0, 0, -3230769.23076923, 0, 122.649572649573, 0, -8974358.97435897, 0, 0, 0, 0, -2692307.69230769, -37.3931623931624, 0, 0, 0, -7897435.8974359, 0, 0, 0, -3230769.23076923, 0, -32.9059829059829, 0, 8974358.97435897, 0, 0, 0, 0, -2692307.69230769, 37.3931623931624, 0, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(38) << 0, 0, -6641025.64102564, 897435.897435898, 628205.128205128, 0, 0, 0, -27.6709401709402, 0, 0, -6641025.64102564, -897435.897435898, 628205.128205128, 0, 0, 0, -27.6709401709402, 0, 0, -4128205.12820513, 0, 628205.128205128, 0, 0, 0, -17.2008547008547, 0, 0, -4128205.12820513, 0, 628205.128205128, 0, 0, 0, -17.2008547008547, 0, 0, 18666666.6666667, 0, -2153846.15384615, 0, 0, 0, 77.7777777777778, 0, 0, 0, -1794871.7948718, -1794871.7948718, 0, 0, 0, 0, 0, 0, 2871794.87179487, 0, -2153846.15384615, 0, 0, 0, 11.965811965812, 0, 0, 0, 1794871.7948718, -1794871.7948718, 0, 0, 0, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(39) << 0, 0, -897435.897435898, -269435.683760684, -48.6111111111111, 0, 0, 0, 7.47863247863248, 0, 0, 897435.897435898, -269435.683760684, 48.6111111111111, 0, 0, 0, -7.47863247863248, 0, 0, 0, -359083.547008547, -18.6965811965812, 0, 0, 0, 0, 0, 0, 0, -359083.547008547, 18.6965811965812, 0, 0, 0, 0, 0, 0, 0, 1436352.13675214, 0, 0, 0, 0, 0, 0, 0, 1794871.7948718, 897435.897435898, -74.7863247863248, 0, 0, 0, -14.957264957265, 0, 0, 0, 718122.222222222, 0, 0, 0, 0, 0, 0, 0, -1794871.7948718, 897435.897435898, 74.7863247863248, 0, 0, 0, 14.957264957265;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(40) << 0, 0, -1166666.66666667, -26.1752136752137, -269274.893162393, 0, 0, 0, 2.99145299145299, 0, 0, -1166666.66666667, 26.1752136752137, -269274.893162393, 0, 0, 0, 2.99145299145299, 0, 0, -628205.128205128, -18.6965811965812, -359019.978632479, 0, 0, 0, 5.23504273504273, 0, 0, -628205.128205128, 18.6965811965812, -359019.978632479, 0, 0, 0, 5.23504273504273, 0, 0, -2153846.15384615, 0, 1436142.73504274, 0, 0, 0, -35.8974358974359, 0, 0, 1794871.7948718, -74.7863247863248, 897435.897435898, 0, 0, 0, -14.957264957265, 0, 0, 2153846.15384615, 0, 717882.905982906, 0, 0, 0, -17.9487179487179, 0, 0, 1794871.7948718, 74.7863247863248, 897435.897435898, 0, 0, 0, -14.957264957265;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(41) << -1346153.84615385, -1750000, 0, 0, 0, -942363.034188034, 1.86965811965812, -2.05662393162393, 0, 1346153.84615385, -1750000, 0, 0, 0, -942363.034188034, -1.86965811965812, -2.05662393162393, 0, 0, -942307.692307692, 0, 0, 0, -1256444.65811966, 0, 1.30876068376068, 0, 0, -942307.692307692, 0, 0, 0, -1256444.65811966, 0, 1.30876068376068, 0, 0, -3230769.23076923, 0, 0, 0, 5025796.58119658, 0, -31.4102564102564, 0, 2692307.69230769, 2692307.69230769, 0, 0, 0, 3141025.64102564, -3.73931623931624, -3.73931623931624, 0, 0, 3230769.23076923, 0, 0, 0, 2512844.44444444, 0, -4.48717948717949, 0, -2692307.69230769, 2692307.69230769, 0, 0, 0, 3141025.64102564, 3.73931623931624, -3.73931623931624, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(42) << -102.457264957265, -24.3055555555556, 0, 0, 0, -1.86965811965812, -2.24435817307692, -0.000182291666666667, 0, -102.457264957265, 24.3055555555556, 0, 0, 0, 1.86965811965812, -2.24435817307692, 0.000182291666666667, 0, -54.5940170940171, -9.3482905982906, 0, 0, 0, 0, -2.9918624465812, -7.01121794871795E-05, 0, -54.5940170940171, 9.3482905982906, 0, 0, 0, 0, -2.9918624465812, 7.01121794871795E-05, 0, 227.350427350427, 0, 0, 0, 0, 0, 11.9675170940171, 0, 0, 0, -37.3931623931624, 0, 0, 0, 3.73931623931624, 7.47863247863248, -0.000280448717948718, 0, 86.7521367521367, 0, 0, 0, 0, 0, 5.98355662393162, 0, 0, 0, 37.3931623931624, 0, 0, 0, -3.73931623931624, 7.47863247863248, 0.000280448717948718, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(43) << -13.0876068376068, -22.0619658119658, 0, 0, 0, -5.79594017094017, -9.81570512820513E-05, -2.24375520833333, 0, 13.0876068376068, -22.0619658119658, 0, 0, 0, -5.79594017094017, 9.81570512820513E-05, -2.24375520833333, 0, -9.3482905982906, -22.8098290598291, 0, 0, 0, -1.30876068376068, -7.01121794871795E-05, -2.99162406517094, 0, 9.3482905982906, -22.8098290598291, 0, 0, 0, -1.30876068376068, 7.01121794871795E-05, -2.99162406517094, 0, 0, 122.649572649573, 0, 0, 0, -31.4102564102564, 0, 11.9667318376068, 0, -37.3931623931624, 0, 0, 0, 0, 3.73931623931624, -0.000280448717948718, 7.47863247863248, 0, 0, -32.9059829059829, 0, 0, 0, 4.48717948717949, 0, 5.98265918803419, 0, 37.3931623931624, 0, 0, 0, 0, 3.73931623931624, 0.000280448717948718, 7.47863247863248, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(44) << 0, 0, -27.6709401709402, -7.47863247863248, -11.965811965812, 0, 0, 0, -7.85277163461538, 0, 0, -27.6709401709402, 7.47863247863248, -11.965811965812, 0, 0, 0, -7.85277163461538, 0, 0, -17.2008547008547, 0, -5.23504273504273, 0, 0, 0, -10.4702144764957, 0, 0, -17.2008547008547, 0, -5.23504273504273, 0, 0, 0, -10.4702144764957, 0, 0, 77.7777777777778, 0, -35.8974358974359, 0, 0, 0, 41.8809252136752, 0, 0, 0, 14.957264957265, 14.957264957265, 0, 0, 0, 26.1752136752137, 0, 0, 11.965811965812, 0, 17.9487179487179, 0, 0, 0, 20.9402606837607, 0, 0, 0, -14.957264957265, 14.957264957265, 0, 0, 0, 26.1752136752137;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(45) << -5474358.97435897, -2243589.74358974, 0, 0, 0, -942307.692307692, -22.8098290598291, -9.3482905982906, 0, -5294871.7948718, 3141025.64102564, 0, 0, 0, -942307.692307692, -22.0619658119658, 13.0876068376068, 0, -5294871.7948718, -3141025.64102564, 0, 0, 0, -942307.692307692, -22.0619658119658, -13.0876068376068, 0, -5474358.97435897, 2243589.74358974, 0, 0, 0, -942307.692307692, -22.8098290598291, 9.3482905982906, 0, 0, -8974358.97435897, 0, 0, 0, 2692307.69230769, 0, -37.3931623931624, 0, 29435897.4358974, 0, 0, 0, 0, 3230769.23076923, 122.649572649573, 0, 0, 0, 8974358.97435897, 0, 0, 0, 2692307.69230769, 0, 37.3931623931624, 0, -7897435.8974359, 0, 0, 0, 0, 3230769.23076923, -32.9059829059829, 0, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(46) << -2243589.74358974, -13102564.1025641, 0, 0, 0, 0, -9.3482905982906, -54.5940170940171, 0, 5833333.33333333, -24589743.5897436, 0, 0, 0, 1346153.84615385, 24.3055555555556, -102.457264957265, 0, -5833333.33333333, -24589743.5897436, 0, 0, 0, -1346153.84615385, -24.3055555555556, -102.457264957265, 0, 2243589.74358974, -13102564.1025641, 0, 0, 0, 0, 9.3482905982906, -54.5940170940171, 0, -8974358.97435897, 0, 0, 0, 0, 2692307.69230769, -37.3931623931624, 0, 0, 0, 54564102.5641026, 0, 0, 0, 0, 0, 227.350427350427, 0, 8974358.97435897, 0, 0, 0, 0, -2692307.69230769, 37.3931623931624, 0, 0, 0, 20820512.8205128, 0, 0, 0, 0, 0, 86.7521367521367, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(47) << 0, 0, -4128205.12820513, -628205.128205128, 0, 0, 0, 0, -17.2008547008547, 0, 0, -6641025.64102564, -628205.128205128, 897435.897435898, 0, 0, 0, -27.6709401709402, 0, 0, -6641025.64102564, -628205.128205128, -897435.897435898, 0, 0, 0, -27.6709401709402, 0, 0, -4128205.12820513, -628205.128205128, 0, 0, 0, 0, -17.2008547008547, 0, 0, 0, 1794871.7948718, 1794871.7948718, 0, 0, 0, 0, 0, 0, 18666666.6666667, 2153846.15384615, 0, 0, 0, 0, 77.7777777777778, 0, 0, 0, 1794871.7948718, -1794871.7948718, 0, 0, 0, 0, 0, 0, 2871794.87179487, 2153846.15384615, 0, 0, 0, 0, 11.965811965812;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(48) << 0, 0, 628205.128205128, -359019.978632479, -18.6965811965812, 0, 0, 0, -5.23504273504273, 0, 0, 1166666.66666667, -269274.893162393, 26.1752136752137, 0, 0, 0, -2.99145299145299, 0, 0, 1166666.66666667, -269274.893162393, -26.1752136752137, 0, 0, 0, -2.99145299145299, 0, 0, 628205.128205128, -359019.978632479, 18.6965811965812, 0, 0, 0, -5.23504273504273, 0, 0, -1794871.7948718, 897435.897435898, -74.7863247863248, 0, 0, 0, 14.957264957265, 0, 0, 2153846.15384615, 1436142.73504274, 0, 0, 0, 0, 35.8974358974359, 0, 0, -1794871.7948718, 897435.897435898, 74.7863247863248, 0, 0, 0, 14.957264957265, 0, 0, -2153846.15384615, 717882.905982906, 0, 0, 0, 0, 17.9487179487179;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(49) << 0, 0, 0, -18.6965811965812, -359083.547008547, 0, 0, 0, 0, 0, 0, -897435.897435898, 48.6111111111111, -269435.683760684, 0, 0, 0, 7.47863247863248, 0, 0, 897435.897435898, -48.6111111111111, -269435.683760684, 0, 0, 0, -7.47863247863248, 0, 0, 0, 18.6965811965812, -359083.547008547, 0, 0, 0, 0, 0, 0, -1794871.7948718, -74.7863247863248, 897435.897435898, 0, 0, 0, 14.957264957265, 0, 0, 0, 0, 1436352.13675214, 0, 0, 0, 0, 0, 0, 1794871.7948718, 74.7863247863248, 897435.897435898, 0, 0, 0, -14.957264957265, 0, 0, 0, 0, 718122.222222222, 0, 0, 0, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(50) << 942307.692307692, 0, 0, 0, 0, -1256444.65811966, -1.30876068376068, 0, 0, 1750000, -1346153.84615385, 0, 0, 0, -942363.034188034, 2.05662393162393, 1.86965811965812, 0, 1750000, 1346153.84615385, 0, 0, 0, -942363.034188034, 2.05662393162393, -1.86965811965812, 0, 942307.692307692, 0, 0, 0, 0, -1256444.65811966, -1.30876068376068, 0, 0, -2692307.69230769, -2692307.69230769, 0, 0, 0, 3141025.64102564, 3.73931623931624, 3.73931623931624, 0, 3230769.23076923, 0, 0, 0, 0, 5025796.58119658, 31.4102564102564, 0, 0, -2692307.69230769, 2692307.69230769, 0, 0, 0, 3141025.64102564, 3.73931623931624, -3.73931623931624, 0, -3230769.23076923, 0, 0, 0, 0, 2512844.44444444, 4.48717948717949, 0, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(51) << -22.8098290598291, -9.3482905982906, 0, 0, 0, 1.30876068376068, -2.99162406517094, -7.01121794871795E-05, 0, -22.0619658119658, 13.0876068376068, 0, 0, 0, 5.79594017094017, -2.24375520833333, 9.81570512820513E-05, 0, -22.0619658119658, -13.0876068376068, 0, 0, 0, 5.79594017094017, -2.24375520833333, -9.81570512820513E-05, 0, -22.8098290598291, 9.3482905982906, 0, 0, 0, 1.30876068376068, -2.99162406517094, 7.01121794871795E-05, 0, 0, -37.3931623931624, 0, 0, 0, -3.73931623931624, 7.47863247863248, -0.000280448717948718, 0, 122.649572649573, 0, 0, 0, 0, 31.4102564102564, 11.9667318376068, 0, 0, 0, 37.3931623931624, 0, 0, 0, -3.73931623931624, 7.47863247863248, 0.000280448717948718, 0, -32.9059829059829, 0, 0, 0, 0, -4.48717948717949, 5.98265918803419, 0, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(52) << -9.3482905982906, -54.5940170940171, 0, 0, 0, 0, -7.01121794871795E-05, -2.9918624465812, 0, 24.3055555555556, -102.457264957265, 0, 0, 0, -1.86965811965812, 0.000182291666666667, -2.24435817307692, 0, -24.3055555555556, -102.457264957265, 0, 0, 0, 1.86965811965812, -0.000182291666666667, -2.24435817307692, 0, 9.3482905982906, -54.5940170940171, 0, 0, 0, 0, 7.01121794871795E-05, -2.9918624465812, 0, -37.3931623931624, 0, 0, 0, 0, -3.73931623931624, -0.000280448717948718, 7.47863247863248, 0, 0, 227.350427350427, 0, 0, 0, 0, 0, 11.9675170940171, 0, 37.3931623931624, 0, 0, 0, 0, 3.73931623931624, 0.000280448717948718, 7.47863247863248, 0, 0, 86.7521367521367, 0, 0, 0, 0, 0, 5.98355662393162, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(53) << 0, 0, -17.2008547008547, 5.23504273504273, 0, 0, 0, 0, -10.4702144764957, 0, 0, -27.6709401709402, 11.965811965812, -7.47863247863248, 0, 0, 0, -7.85277163461538, 0, 0, -27.6709401709402, 11.965811965812, 7.47863247863248, 0, 0, 0, -7.85277163461538, 0, 0, -17.2008547008547, 5.23504273504273, 0, 0, 0, 0, -10.4702144764957, 0, 0, 0, -14.957264957265, -14.957264957265, 0, 0, 0, 26.1752136752137, 0, 0, 77.7777777777778, 35.8974358974359, 0, 0, 0, 0, 41.8809252136752, 0, 0, 0, -14.957264957265, 14.957264957265, 0, 0, 0, 26.1752136752137, 0, 0, 11.965811965812, -17.9487179487179, 0, 0, 0, 0, 20.9402606837607;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(54) << -13102564.1025641, -2243589.74358974, 0, 0, 0, 0, -54.5940170940171, -9.3482905982906, 0, -13102564.1025641, 2243589.74358974, 0, 0, 0, 0, -54.5940170940171, 9.3482905982906, 0, -24589743.5897436, -5833333.33333333, 0, 0, 0, -1346153.84615385, -102.457264957265, -24.3055555555556, 0, -24589743.5897436, 5833333.33333333, 0, 0, 0, 1346153.84615385, -102.457264957265, 24.3055555555556, 0, 20820512.8205128, 0, 0, 0, 0, 0, 86.7521367521367, 0, 0, 0, 8974358.97435897, 0, 0, 0, -2692307.69230769, 0, 37.3931623931624, 0, 54564102.5641026, 0, 0, 0, 0, 0, 227.350427350427, 0, 0, 0, -8974358.97435897, 0, 0, 0, 2692307.69230769, 0, -37.3931623931624, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(55) << -2243589.74358974, -5474358.97435897, 0, 0, 0, -942307.692307692, -9.3482905982906, -22.8098290598291, 0, 2243589.74358974, -5474358.97435897, 0, 0, 0, -942307.692307692, 9.3482905982906, -22.8098290598291, 0, -3141025.64102564, -5294871.7948718, 0, 0, 0, -942307.692307692, -13.0876068376068, -22.0619658119658, 0, 3141025.64102564, -5294871.7948718, 0, 0, 0, -942307.692307692, 13.0876068376068, -22.0619658119658, 0, 0, -7897435.8974359, 0, 0, 0, 3230769.23076923, 0, -32.9059829059829, 0, 8974358.97435897, 0, 0, 0, 0, 2692307.69230769, 37.3931623931624, 0, 0, 0, 29435897.4358974, 0, 0, 0, 3230769.23076923, 0, 122.649572649573, 0, -8974358.97435897, 0, 0, 0, 0, 2692307.69230769, -37.3931623931624, 0, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(56) << 0, 0, -4128205.12820513, 0, -628205.128205128, 0, 0, 0, -17.2008547008547, 0, 0, -4128205.12820513, 0, -628205.128205128, 0, 0, 0, -17.2008547008547, 0, 0, -6641025.64102564, -897435.897435898, -628205.128205128, 0, 0, 0, -27.6709401709402, 0, 0, -6641025.64102564, 897435.897435898, -628205.128205128, 0, 0, 0, -27.6709401709402, 0, 0, 2871794.87179487, 0, 2153846.15384615, 0, 0, 0, 11.965811965812, 0, 0, 0, -1794871.7948718, 1794871.7948718, 0, 0, 0, 0, 0, 0, 18666666.6666667, 0, 2153846.15384615, 0, 0, 0, 77.7777777777778, 0, 0, 0, 1794871.7948718, 1794871.7948718, 0, 0, 0, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(57) << 0, 0, 0, -359083.547008547, -18.6965811965812, 0, 0, 0, 0, 0, 0, 0, -359083.547008547, 18.6965811965812, 0, 0, 0, 0, 0, 0, 897435.897435898, -269435.683760684, -48.6111111111111, 0, 0, 0, -7.47863247863248, 0, 0, -897435.897435898, -269435.683760684, 48.6111111111111, 0, 0, 0, 7.47863247863248, 0, 0, 0, 718122.222222222, 0, 0, 0, 0, 0, 0, 0, 1794871.7948718, 897435.897435898, 74.7863247863248, 0, 0, 0, -14.957264957265, 0, 0, 0, 1436352.13675214, 0, 0, 0, 0, 0, 0, 0, -1794871.7948718, 897435.897435898, -74.7863247863248, 0, 0, 0, 14.957264957265;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(58) << 0, 0, 628205.128205128, -18.6965811965812, -359019.978632479, 0, 0, 0, -5.23504273504273, 0, 0, 628205.128205128, 18.6965811965812, -359019.978632479, 0, 0, 0, -5.23504273504273, 0, 0, 1166666.66666667, -26.1752136752137, -269274.893162393, 0, 0, 0, -2.99145299145299, 0, 0, 1166666.66666667, 26.1752136752137, -269274.893162393, 0, 0, 0, -2.99145299145299, 0, 0, -2153846.15384615, 0, 717882.905982906, 0, 0, 0, 17.9487179487179, 0, 0, -1794871.7948718, 74.7863247863248, 897435.897435898, 0, 0, 0, 14.957264957265, 0, 0, 2153846.15384615, 0, 1436142.73504274, 0, 0, 0, 35.8974358974359, 0, 0, -1794871.7948718, -74.7863247863248, 897435.897435898, 0, 0, 0, 14.957264957265;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(59) << 0, 942307.692307692, 0, 0, 0, -1256444.65811966, 0, -1.30876068376068, 0, 0, 942307.692307692, 0, 0, 0, -1256444.65811966, 0, -1.30876068376068, 0, 1346153.84615385, 1750000, 0, 0, 0, -942363.034188034, -1.86965811965812, 2.05662393162393, 0, -1346153.84615385, 1750000, 0, 0, 0, -942363.034188034, 1.86965811965812, 2.05662393162393, 0, 0, -3230769.23076923, 0, 0, 0, 2512844.44444444, 0, 4.48717948717949, 0, 2692307.69230769, -2692307.69230769, 0, 0, 0, 3141025.64102564, -3.73931623931624, 3.73931623931624, 0, 0, 3230769.23076923, 0, 0, 0, 5025796.58119658, 0, 31.4102564102564, 0, -2692307.69230769, -2692307.69230769, 0, 0, 0, 3141025.64102564, 3.73931623931624, 3.73931623931624, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(60) << -54.5940170940171, -9.3482905982906, 0, 0, 0, 0, -2.9918624465812, -7.01121794871795E-05, 0, -54.5940170940171, 9.3482905982906, 0, 0, 0, 0, -2.9918624465812, 7.01121794871795E-05, 0, -102.457264957265, -24.3055555555556, 0, 0, 0, 1.86965811965812, -2.24435817307692, -0.000182291666666667, 0, -102.457264957265, 24.3055555555556, 0, 0, 0, -1.86965811965812, -2.24435817307692, 0.000182291666666667, 0, 86.7521367521367, 0, 0, 0, 0, 0, 5.98355662393162, 0, 0, 0, 37.3931623931624, 0, 0, 0, 3.73931623931624, 7.47863247863248, 0.000280448717948718, 0, 227.350427350427, 0, 0, 0, 0, 0, 11.9675170940171, 0, 0, 0, -37.3931623931624, 0, 0, 0, -3.73931623931624, 7.47863247863248, -0.000280448717948718, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(61) << -9.3482905982906, -22.8098290598291, 0, 0, 0, 1.30876068376068, -7.01121794871795E-05, -2.99162406517094, 0, 9.3482905982906, -22.8098290598291, 0, 0, 0, 1.30876068376068, 7.01121794871795E-05, -2.99162406517094, 0, -13.0876068376068, -22.0619658119658, 0, 0, 0, 5.79594017094017, -9.81570512820513E-05, -2.24375520833333, 0, 13.0876068376068, -22.0619658119658, 0, 0, 0, 5.79594017094017, 9.81570512820513E-05, -2.24375520833333, 0, 0, -32.9059829059829, 0, 0, 0, -4.48717948717949, 0, 5.98265918803419, 0, 37.3931623931624, 0, 0, 0, 0, -3.73931623931624, 0.000280448717948718, 7.47863247863248, 0, 0, 122.649572649573, 0, 0, 0, 31.4102564102564, 0, 11.9667318376068, 0, -37.3931623931624, 0, 0, 0, 0, -3.73931623931624, -0.000280448717948718, 7.47863247863248, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(62) << 0, 0, -17.2008547008547, 0, 5.23504273504273, 0, 0, 0, -10.4702144764957, 0, 0, -17.2008547008547, 0, 5.23504273504273, 0, 0, 0, -10.4702144764957, 0, 0, -27.6709401709402, 7.47863247863248, 11.965811965812, 0, 0, 0, -7.85277163461538, 0, 0, -27.6709401709402, -7.47863247863248, 11.965811965812, 0, 0, 0, -7.85277163461538, 0, 0, 11.965811965812, 0, -17.9487179487179, 0, 0, 0, 20.9402606837607, 0, 0, 0, 14.957264957265, -14.957264957265, 0, 0, 0, 26.1752136752137, 0, 0, 77.7777777777778, 0, 35.8974358974359, 0, 0, 0, 41.8809252136752, 0, 0, 0, -14.957264957265, -14.957264957265, 0, 0, 0, 26.1752136752137;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(63) << -5294871.7948718, -3141025.64102564, 0, 0, 0, 942307.692307692, -22.0619658119658, -13.0876068376068, 0, -5474358.97435897, 2243589.74358974, 0, 0, 0, 942307.692307692, -22.8098290598291, 9.3482905982906, 0, -5474358.97435897, -2243589.74358974, 0, 0, 0, 942307.692307692, -22.8098290598291, -9.3482905982906, 0, -5294871.7948718, 3141025.64102564, 0, 0, 0, 942307.692307692, -22.0619658119658, 13.0876068376068, 0, 0, 8974358.97435897, 0, 0, 0, -2692307.69230769, 0, 37.3931623931624, 0, -7897435.8974359, 0, 0, 0, 0, -3230769.23076923, -32.9059829059829, 0, 0, 0, -8974358.97435897, 0, 0, 0, -2692307.69230769, 0, -37.3931623931624, 0, 29435897.4358974, 0, 0, 0, 0, -3230769.23076923, 122.649572649573, 0, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(64) << -5833333.33333333, -24589743.5897436, 0, 0, 0, 1346153.84615385, -24.3055555555556, -102.457264957265, 0, 2243589.74358974, -13102564.1025641, 0, 0, 0, 0, 9.3482905982906, -54.5940170940171, 0, -2243589.74358974, -13102564.1025641, 0, 0, 0, 0, -9.3482905982906, -54.5940170940171, 0, 5833333.33333333, -24589743.5897436, 0, 0, 0, -1346153.84615385, 24.3055555555556, -102.457264957265, 0, 8974358.97435897, 0, 0, 0, 0, 2692307.69230769, 37.3931623931624, 0, 0, 0, 20820512.8205128, 0, 0, 0, 0, 0, 86.7521367521367, 0, -8974358.97435897, 0, 0, 0, 0, -2692307.69230769, -37.3931623931624, 0, 0, 0, 54564102.5641026, 0, 0, 0, 0, 0, 227.350427350427, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(65) << 0, 0, -6641025.64102564, 628205.128205128, 897435.897435898, 0, 0, 0, -27.6709401709402, 0, 0, -4128205.12820513, 628205.128205128, 0, 0, 0, 0, -17.2008547008547, 0, 0, -4128205.12820513, 628205.128205128, 0, 0, 0, 0, -17.2008547008547, 0, 0, -6641025.64102564, 628205.128205128, -897435.897435898, 0, 0, 0, -27.6709401709402, 0, 0, 0, -1794871.7948718, 1794871.7948718, 0, 0, 0, 0, 0, 0, 2871794.87179487, -2153846.15384615, 0, 0, 0, 0, 11.965811965812, 0, 0, 0, -1794871.7948718, -1794871.7948718, 0, 0, 0, 0, 0, 0, 18666666.6666667, -2153846.15384615, 0, 0, 0, 0, 77.7777777777778;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(66) << 0, 0, -1166666.66666667, -269274.893162393, -26.1752136752137, 0, 0, 0, 2.99145299145299, 0, 0, -628205.128205128, -359019.978632479, 18.6965811965812, 0, 0, 0, 5.23504273504273, 0, 0, -628205.128205128, -359019.978632479, -18.6965811965812, 0, 0, 0, 5.23504273504273, 0, 0, -1166666.66666667, -269274.893162393, 26.1752136752137, 0, 0, 0, 2.99145299145299, 0, 0, 1794871.7948718, 897435.897435898, 74.7863247863248, 0, 0, 0, -14.957264957265, 0, 0, 2153846.15384615, 717882.905982906, 0, 0, 0, 0, -17.9487179487179, 0, 0, 1794871.7948718, 897435.897435898, -74.7863247863248, 0, 0, 0, -14.957264957265, 0, 0, -2153846.15384615, 1436142.73504274, 0, 0, 0, 0, -35.8974358974359;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(67) << 0, 0, -897435.897435898, -48.6111111111111, -269435.683760684, 0, 0, 0, 7.47863247863248, 0, 0, 0, 18.6965811965812, -359083.547008547, 0, 0, 0, 0, 0, 0, 0, -18.6965811965812, -359083.547008547, 0, 0, 0, 0, 0, 0, 897435.897435898, 48.6111111111111, -269435.683760684, 0, 0, 0, -7.47863247863248, 0, 0, -1794871.7948718, 74.7863247863248, 897435.897435898, 0, 0, 0, 14.957264957265, 0, 0, 0, 0, 718122.222222222, 0, 0, 0, 0, 0, 0, 1794871.7948718, -74.7863247863248, 897435.897435898, 0, 0, 0, -14.957264957265, 0, 0, 0, 0, 1436352.13675214, 0, 0, 0, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(68) << -1750000, -1346153.84615385, 0, 0, 0, -942363.034188034, -2.05662393162393, 1.86965811965812, 0, -942307.692307692, 0, 0, 0, 0, -1256444.65811966, 1.30876068376068, 0, 0, -942307.692307692, 0, 0, 0, 0, -1256444.65811966, 1.30876068376068, 0, 0, -1750000, 1346153.84615385, 0, 0, 0, -942363.034188034, -2.05662393162393, -1.86965811965812, 0, 2692307.69230769, -2692307.69230769, 0, 0, 0, 3141025.64102564, -3.73931623931624, 3.73931623931624, 0, 3230769.23076923, 0, 0, 0, 0, 2512844.44444444, -4.48717948717949, 0, 0, 2692307.69230769, 2692307.69230769, 0, 0, 0, 3141025.64102564, -3.73931623931624, -3.73931623931624, 0, -3230769.23076923, 0, 0, 0, 0, 5025796.58119658, -31.4102564102564, 0, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(69) << -22.0619658119658, -13.0876068376068, 0, 0, 0, -5.79594017094017, -2.24375520833333, -9.81570512820513E-05, 0, -22.8098290598291, 9.3482905982906, 0, 0, 0, -1.30876068376068, -2.99162406517094, 7.01121794871795E-05, 0, -22.8098290598291, -9.3482905982906, 0, 0, 0, -1.30876068376068, -2.99162406517094, -7.01121794871795E-05, 0, -22.0619658119658, 13.0876068376068, 0, 0, 0, -5.79594017094017, -2.24375520833333, 9.81570512820513E-05, 0, 0, 37.3931623931624, 0, 0, 0, 3.73931623931624, 7.47863247863248, 0.000280448717948718, 0, -32.9059829059829, 0, 0, 0, 0, 4.48717948717949, 5.98265918803419, 0, 0, 0, -37.3931623931624, 0, 0, 0, 3.73931623931624, 7.47863247863248, -0.000280448717948718, 0, 122.649572649573, 0, 0, 0, 0, -31.4102564102564, 11.9667318376068, 0, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(70) << -24.3055555555556, -102.457264957265, 0, 0, 0, -1.86965811965812, -0.000182291666666667, -2.24435817307692, 0, 9.3482905982906, -54.5940170940171, 0, 0, 0, 0, 7.01121794871795E-05, -2.9918624465812, 0, -9.3482905982906, -54.5940170940171, 0, 0, 0, 0, -7.01121794871795E-05, -2.9918624465812, 0, 24.3055555555556, -102.457264957265, 0, 0, 0, 1.86965811965812, 0.000182291666666667, -2.24435817307692, 0, 37.3931623931624, 0, 0, 0, 0, -3.73931623931624, 0.000280448717948718, 7.47863247863248, 0, 0, 86.7521367521367, 0, 0, 0, 0, 0, 5.98355662393162, 0, -37.3931623931624, 0, 0, 0, 0, 3.73931623931624, -0.000280448717948718, 7.47863247863248, 0, 0, 227.350427350427, 0, 0, 0, 0, 0, 11.9675170940171, 0;
    //Expected_JacobianR_NoDispNoVelWithDamping.row(71) << 0, 0, -27.6709401709402, -11.965811965812, -7.47863247863248, 0, 0, 0, -7.85277163461538, 0, 0, -17.2008547008547, -5.23504273504273, 0, 0, 0, 0, -10.4702144764957, 0, 0, -17.2008547008547, -5.23504273504273, 0, 0, 0, 0, -10.4702144764957, 0, 0, -27.6709401709402, -11.965811965812, 7.47863247863248, 0, 0, 0, -7.85277163461538, 0, 0, 0, 14.957264957265, -14.957264957265, 0, 0, 0, 26.1752136752137, 0, 0, 11.965811965812, 17.9487179487179, 0, 0, 0, 0, 20.9402606837607, 0, 0, 0, 14.957264957265, 14.957264957265, 0, 0, 0, 26.1752136752137, 0, 0, 77.7777777777778, -35.8974358974359, 0, 0, 0, 0, 41.8809252136752;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_Jacobians;
    Expected_Jacobians.resize(2 * 72, 72);
    if (!load_validation_data("UT_ANCFShell_3833_JacNoDispNoVelWithDamping.txt", Expected_Jacobians))
        return false;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianK_NoDispNoVelWithDamping;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianR_NoDispNoVelWithDamping;
    Expected_JacobianK_NoDispNoVelWithDamping = Expected_Jacobians.block(0, 0, Expected_Jacobians.cols(), Expected_Jacobians.cols());
    Expected_JacobianR_NoDispNoVelWithDamping = Expected_Jacobians.block(Expected_Jacobians.cols(), 0, Expected_Jacobians.cols(), Expected_Jacobians.cols());


    //Setup the test conditions
    m_element->SetAlphaDamp(0.01);

    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation
    ChVectorDynamic<double> InternalForceNoDispNoVelNoGravity;
    InternalForceNoDispNoVelNoGravity.resize(72);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceNoDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_NoDispNoVelWithDamping;
    JacobianK_NoDispNoVelWithDamping.resize(72, 72);
    m_element->ComputeKRMmatricesGlobal(JacobianK_NoDispNoVelWithDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_NoDispNoVelWithDamping;
    JacobianR_NoDispNoVelWithDamping.resize(72, 72);
    m_element->ComputeKRMmatricesGlobal(JacobianR_NoDispNoVelWithDamping, 0, 1, 0);

    //Reset the element conditions back to its original values
    m_element->SetAlphaDamp(0.0);


    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_NoDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(72, 72);
    
    double small_terms_JacR = 1e-4*Expected_JacobianR_NoDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacR = 0;
    double max_percent_error_JacR = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacR;
    percent_error_matrix_JacR.resize(72, 72);


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

    //ChMatrixNM<double, 72, 72> Expected_JacobianK_SmallDispNoVelWithDamping;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(0) << 2100001045.44872, 953525641.025641, -614743.58974359, 13461.5384615385, 0, -80769230.7692308, 8750.00435603633, 3973.0235042735, -2.56143162393162, 1032053109.45513, -33653846.1538461, 684294.871794872, 13461.5384615385, 0, 53846153.8461538, 4300.22128939637, -140.224358974359, 2.85122863247863, 928846689.166667, 392628205.128205, 42628.2051282051, -26923.0769230769, 0, 20192307.6923077, 3870.19453819444, 1635.95085470085, 0.177617521367521, 785256917.467949, 33653846.1538461, -49358.9743589744, -26923.0769230769, 0, 20192307.6923077, 3271.90382278312, 140.224358974359, -0.205662393162393, -2458976181.66667, -314102564.102564, -4487.17948717949, 108974.358974359, 0, -134615384.615385, -10245.7340902778, -1308.76068376068, -0.0186965811965812, -547436686.282051, -224358974.358974, 305128.205128205, 115384.615384615, 0, 94230769.2307692, -2280.98619284188, -934.82905982906, 1.27136752136752, -1310257051.02564, -224358974.358974, 4487.17948717949, 91025.641025641, 0, 0, -5459.4043792735, -934.82905982906, 0.0186965811965812, -529487842.564103, -583333333.333333, -367948.717948718, 115384.615384615, 0, -175000000, -2206.19934401709, -2430.55555555556, -1.53311965811966;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(1) << 953525641.025641, 2100001045.44872, -666346.153846154, 0, 13461.5384615385, -80769230.7692308, 3973.0235042735, 8750.00435603633, -2.77644230769231, 33653846.1538461, 785258237.660256, -648397.435897436, 0, 13461.5384615385, 20192307.6923077, 140.224358974359, 3271.9093235844, -2.70165598290598, 392628205.128205, 928846689.166667, -62820.5128205128, 0, -26923.0769230769, 20192307.6923077, 1635.95085470085, 3870.19453819444, -0.261752136752137, -33653846.1538461, 1032051789.26282, -49358.9743589744, 0, -26923.0769230769, 53846153.8461538, -140.224358974359, 4300.21578859509, -0.205662393162393, -583333333.333333, -529489002.179487, 332051.282051282, 0, 108974.358974359, -175000000, -2430.55555555556, -2206.20417574786, 1.38354700854701, -224358974.358974, -1310257199.10256, 228846.153846154, 0, 115384.615384615, 0, -934.82905982906, -5459.40499626068, 0.953525641025641, -224358974.358974, -547436538.205128, 700000, 0, 91025.641025641, 94230769.2307692, -934.82905982906, -2280.9855758547, 2.91666666666667, -314102564.102564, -2458975022.05128, 166025.641025641, 0, 115384.615384615, -134615384.615385, -1308.76068376068, -10245.729258547, 0.691773504273504;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(2) << -614743.58974359, -666346.153846154, 933336270.192308, -53846153.8461539, -53846153.8461539, 47115.3846153846, -2.56143162393162, -2.77644230769231, 3888.90112580128, 614743.58974359, -666346.153846154, 403851636.057692, 35897435.8974359, 13461538.4615385, 47115.3846153846, 2.56143162393162, -2.77644230769231, 1682.71515024038, 62820.5128205128, -42628.2051282051, 412822083.333333, 13461538.4615385, 13461538.4615385, -94230.7692307692, 0.261752136752137, -0.177617521367521, 1720.09201388889, -62820.5128205128, -42628.2051282051, 403847624.519231, 13461538.4615385, 35897435.8974359, -94230.7692307692, -0.261752136752137, -0.177617521367521, 1682.6984354968, 0, 125641.025641026, -664107800.641026, -89743589.7435897, -116666666.666667, 381410.256410256, 0, 0.523504273504274, -2767.11583600427, 430769.230769231, 296153.846153846, -412823070.512821, 62820512.8205128, 0, 403846.153846154, 1.79487179487179, 1.23397435897436, -1720.09612713675, 0, 700000, -412822352.564103, 0, 62820512.8205128, 318589.743589744, 0, 2.91666666666667, -1720.09313568376, -430769.230769231, 296153.846153846, -664104390.384615, -116666666.666667, -89743589.7435897, 403846.153846154, -1.79487179487179, 1.23397435897436, -2767.10162660256;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(3) << 13461.5384615385, 0, -53846153.8461539, 26940591.2266608, 7946.04700854701, -5517.94337606838, 0.168269230769231, 0, -897.435897435898, -73076.9230769231, 0, -35897435.8974359, 8982949.51309161, -280.448717948718, -2750.7077991453, -0.192307692307692, 0, 299.145299145299, -26923.0769230769, 0, -13461538.4615385, 13469286.1903584, 3271.90170940171, 2628.56036324786, -0.336538461538462, 0, 112.179487179487, -39102.5641025641, 0, 13461538.4615385, 8980906.4358507, 280.448717948718, 4358.56303418803, -0.387286324786325, 0, 224.358974358974, 1282.05128205128, 0, 89743589.7435897, -26943596.5322831, -2617.52136752137, -17051.3194444444, 0.913461538461539, 0, -747.863247863248, 46153.8461538462, 0, -62820512.8205128, -35902022.5172575, -1869.65811965812, -13330.7905982906, 1.15384615384615, 0, 523.504273504273, 34615.3846153846, 0, 0, -35908365.6997842, -1869.65811965812, -897.398504273504, 0.902777777777778, 0, 0, 43589.7435897436, 0, 62820512.8205128, -26927495.7320214, -4861.11111111111, -3336.39957264957, 1.14316239316239, 0, -1196.5811965812;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(4) << 0, 13461.5384615385, -53846153.8461539, 7946.04700854701, 26940591.2266608, -3467.09134615385, 0, 0.168269230769231, -897.435897435898, 0, -73076.9230769231, 13461538.4615385, 280.448717948718, 8980892.88915999, 4609.9813034188, 0, -0.192307692307692, 224.358974358974, 0, -26923.0769230769, -13461538.4615385, 3271.90170940171, 13469286.1903584, -2628.72863247863, 0, -0.336538461538462, 112.179487179487, 0, -39102.5641025641, -35897435.8974359, -280.448717948718, 8982963.05978232, -1731.18055555556, 0, -0.387286324786325, 299.145299145299, 0, 1282.05128205128, 62820512.8205128, -4861.11111111111, -26927517.4724541, 12310.4594017094, 0, 0.913461538461539, -1196.5811965812, 0, 46153.8461538462, 0, -1869.65811965812, -35908379.3548643, 9873.70192307692, 0, 1.15384615384615, 0, 0, 34615.3846153846, -62820512.8205128, -1869.65811965812, -35902008.8621774, -4609.55128205128, 0, 0.902777777777778, 523.504273504273, 0, 43589.7435897436, 89743589.7435897, -2617.52136752137, -26943574.7918504, -896.052350427351, 0, 1.14316239316239, -747.863247863248;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(5) << -80769230.7692308, -80769230.7692308, 47115.3846153846, -5517.94337606838, -3467.09134615385, 94238580.3877217, -785.25641025641, -785.25641025641, 0.588942307692308, -53846153.8461538, 20192307.6923077, -255769.230769231, -2751.28739316239, 4609.83173076923, 31413598.7315825, 74.7863247863248, 196.314102564103, -0.673076923076923, -20192307.6923077, -20192307.6923077, -94230.7692307692, 2628.72863247863, -2628.56036324786, 47118841.9254808, 28.0448717948718, 28.0448717948718, -1.17788461538462, 20192307.6923077, -53846153.8461538, -136858.974358974, 4358.45085470085, -1731.12446581197, 31413630.3327684, 196.314102564103, 74.7863247863248, -1.35550213675214, 134615384.615385, 94230769.2307692, 4487.17948717949, -17051.2820512821, 12308.7393162393, -94236369.1248344, -186.965811965812, -579.594017094017, 3.19711538461538, -94230769.2307692, 0, 161538.461538462, -13329.7435897436, 9874.26282051282, -125644523.343964, 130.876068376068, 0, 4.03846153846154, 0, -94230769.2307692, 121153.846153846, -897.435897435898, -4609.55128205128, -125644491.479006, 0, 130.876068376068, 3.15972222222222, 94230769.2307692, 134615384.615385, 152564.102564103, -3336.92307692308, -894.967948717949, -94236318.3912874, -579.594017094017, -186.965811965812, 4.00106837606838;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(6) << 8750.00435603633, 3973.0235042735, -2.56143162393162, 0.168269230769231, 0, -785.25641025641, 224.424718515576, 0.0297976762820513, -0.0459593816773504, 4300.22128939637, -140.224358974359, 2.85122863247863, -0.552884615384615, 0, -74.7863247863248, 74.818493913944, -0.00105168269230769, -0.0229487012553419, 3870.19453819444, 1635.95085470085, 0.177617521367521, -0.336538461538462, 0, -28.0448717948718, 112.208574803054, 0.0122696314102564, 0.0219030415331197, 3271.90382278312, 140.224358974359, -0.205662393162393, -0.438034188034188, 0, 196.314102564103, 74.8108945137136, 0.00105168269230769, 0.0363232438568376, -10245.7340902778, -1308.76068376068, -0.0186965811965812, 0.46474358974359, 0, 186.965811965812, -224.436051873198, -0.00981570512820513, -0.142094157318376, -2280.98619284188, -934.82905982906, 1.27136752136752, 0.865384615384615, 0, -130.876068376068, -299.162611937045, -0.00701121794871795, -0.111101575854701, -5459.4043792735, -934.82905982906, 0.0186965811965812, 0.667735042735043, 0, 0, -299.186336291392, -0.00701121794871795, -0.00747849225427351, -2206.19934401709, -2430.55555555556, -1.53311965811966, 0.844017094017094, 0, -205.662393162393, -224.375574272858, -0.0182291666666667, -0.0277892761752137;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(7) << 3973.0235042735, 8750.00435603633, -2.77644230769231, 0, 0.168269230769231, -785.25641025641, 0.0297976762820513, 224.424718515576, -0.0288669771634615, 140.224358974359, 3271.9093235844, -2.70165598290598, 0, -0.552884615384615, 196.314102564103, 0.00105168269230769, 74.8107815742004, 0.0384412760416667, 1635.95085470085, 3870.19453819444, -0.261752136752137, 0, -0.336538461538462, -28.0448717948718, 0.0122696314102564, 112.208574803054, -0.021903672542735, -140.224358974359, 4300.21578859509, -0.205662393162393, 0, -0.438034188034188, -74.7863247863248, -0.00105168269230769, 74.8186068534572, -0.0144246193910256, -2430.55555555556, -2206.20417574786, 1.38354700854701, 0, 0.46474358974359, -205.662393162393, -0.0182291666666667, -224.37575539884, 0.102574479166667, -934.82905982906, -5459.40499626068, 0.953525641025641, 0, 0.865384615384615, 0, -0.00701121794871795, -299.18645007807, 0.082272108707265, -934.82905982906, -2280.9855758547, 2.91666666666667, 0, 0.667735042735043, -130.876068376068, -0.00701121794871795, -299.162498150366, -0.0384396634615385, -1308.76068376068, -10245.729258547, 0.691773504273504, 0, 0.844017094017094, 186.965811965812, -0.00981570512820513, -224.435870747217, -0.00747344417735043;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(8) << -2.56143162393162, -2.77644230769231, 3888.90112580128, -897.435897435898, -897.435897435898, 0.588942307692308, -0.0459593816773504, -0.0288669771634615, 785.285854970694, 2.56143162393162, -2.77644230769231, 1682.71515024038, -299.145299145299, 224.358974358974, -1.93509615384615, -0.022950874732906, 0.0384407151442308, 261.764564540977, 0.261752136752137, -0.177617521367521, 1720.09201388889, -112.179487179487, -112.179487179487, -1.17788461538462, 0.021903672542735, -0.0219030415331197, 392.641248535546, -0.261752136752137, -0.177617521367521, 1682.6984354968, 224.358974358974, -299.145299145299, -1.53311965811966, 0.0363228231837607, -0.0144244090544872, 261.764828037412, 0, 0.523504273504274, -2767.11583600427, 747.863247863248, 299.145299145299, 1.62660256410256, -0.142094017094017, 0.102568028846154, -785.27771081179, 1.79487179487179, 1.23397435897436, -1720.09612713675, -523.504273504273, 0, 3.02884615384615, -0.11109764957265, 0.0822742120726496, -1047.0219269852, 0, 2.91666666666667, -1720.09313568376, 0, -523.504273504273, 2.33707264957265, -0.00747863247863248, -0.0384396634615385, -1047.02166147131, -1.79487179487179, 1.23397435897436, -2767.10162660256, 299.145299145299, 747.863247863248, 2.95405982905983, -0.0277912393162393, -0.00746937767094017, -785.277288162484;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(9) << 1032053109.45513, 33653846.1538461, 614743.58974359, -73076.9230769231, 0, -53846153.8461538, 4300.22128939637, 140.224358974359, 2.56143162393162, 2100008574.10256, -953525641.025641, 4125961.53846154, 191666.666666667, 0, 80769230.7692308, 8750.03572542735, -3973.0235042735, 17.1915064102564, 785258237.660256, -33653846.1538461, 666346.153846154, -73076.9230769231, 0, -20192307.6923077, 3271.9093235844, -140.224358974359, 2.77644230769231, 928847521.089744, -392628205.128205, 525000, -50641.0256410256, 0, -20192307.6923077, 3870.1980045406, -1635.95085470085, 2.1875, -2458979298.46154, 314102564.102564, -2598076.92307692, 333333.333333333, 0, 134615384.615385, -10245.7470769231, 1308.76068376068, -10.8253205128205, -529492118.974359, 583333333.333333, -915384.615384615, 333333.333333333, 0, 175000000, -2206.21716239316, 2430.55555555556, -3.81410256410256, -1310258268.84615, 224358974.358974, -955769.230769231, 135897.435897436, 0, 0, -5459.40945352564, 934.82905982906, -3.98237179487179, -547437756.025641, 224358974.358974, -1462820.51282051, 135897.435897436, 0, -94230769.2307692, -2280.99065010684, 934.82905982906, -6.09508547008547;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(10) << -33653846.1538461, 785258237.660256, -666346.153846154, 0, -73076.9230769231, 20192307.6923077, -140.224358974359, 3271.9093235844, -2.77644230769231, -953525641.025641, 2100008574.10256, -4125961.53846154, 0, 191666.666666667, -80769230.7692308, -3973.0235042735, 8750.03572542735, -17.1915064102564, 33653846.1538461, 1032053109.45513, -614743.58974359, 0, -73076.9230769231, 53846153.8461538, 140.224358974359, 4300.22128939637, -2.56143162393162, -392628205.128205, 928847521.089744, -525000, 0, -50641.0256410256, 20192307.6923077, -1635.95085470085, 3870.1980045406, -2.1875, 583333333.333333, -529492118.974359, 915384.615384615, 0, 333333.333333333, -175000000, 2430.55555555556, -2206.21716239316, 3.81410256410256, 314102564.102564, -2458979298.46154, 2598076.92307692, 0, 333333.333333333, -134615384.615385, 1308.76068376068, -10245.7470769231, 10.8253205128205, 224358974.358974, -547437756.025641, 1462820.51282051, 0, 135897.435897436, 94230769.2307692, 934.82905982906, -2280.99065010684, 6.09508547008547, 224358974.358974, -1310258268.84615, 955769.230769231, 0, 135897.435897436, 0, 934.82905982906, -5459.40945352564, 3.98237179487179;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(11) << 684294.871794872, -648397.435897436, 403851636.057692, -35897435.8974359, 13461538.4615385, -255769.230769231, 2.85122863247863, -2.70165598290598, 1682.71515024038, 4125961.53846154, -4125961.53846154, 933359055.641026, 53846153.8461539, -53846153.8461539, 670833.333333333, 17.1915064102564, -17.1915064102564, 3888.99606517094, 648397.435897436, -684294.871794872, 403851636.057692, -13461538.4615385, 35897435.8974359, -255769.230769231, 2.70165598290598, -2.85122863247863, 1682.71515024038, 518269.230769231, -518269.230769231, 412824614.551282, -13461538.4615385, 13461538.4615385, -177243.58974359, 2.15945512820513, -2.15945512820513, 1720.10256063034, -3096153.84615385, 403846.153846154, -664117382.564103, 89743589.7435897, -116666666.666667, 1166666.66666667, -12.900641025641, 1.68269230769231, -2767.15576068376, -403846.153846154, 3096153.84615385, -664117382.564103, 116666666.666667, -89743589.7435897, 1166666.66666667, -1.68269230769231, 12.900641025641, -2767.15576068376, -978205.128205128, 1498717.94871795, -412826088.589744, 0, 62820512.8205128, 475641.025641026, -4.0758547008547, 6.24465811965812, -1720.10870245727, -1498717.94871795, 978205.128205128, -412826088.589744, -62820512.8205128, 0, 475641.025641026, -6.24465811965812, 4.0758547008547, -1720.10870245727;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(12) << 13461.5384615385, 0, 35897435.8974359, 8982949.51309161, 280.448717948718, -2751.28739316239, -0.552884615384615, 0, -299.145299145299, 191666.666666667, 0, 53846153.8461539, 26940667.250938, -7946.04700854701, 25034.3830128205, 2.39583333333333, 0, 897.435897435898, 13461.5384615385, 0, -13461538.4615385, 8980892.88915999, -280.448717948718, -4609.83173076923, -0.552884615384615, 0, -224.358974358974, -3205.12820512821, 0, 13461538.4615385, 13469278.5049834, -3271.90170940171, -957.163461538462, -0.435363247863248, 0, -112.179487179487, -97435.8974358974, 0, -89743589.7435897, -26943528.2890256, 2617.52136752137, 15234.7596153846, 2.37179487179487, 0, 747.863247863248, -97435.8974358974, 0, -62820512.8205128, -26927449.2291966, 4861.11111111111, 11018.0128205128, 2.37179487179487, 0, 1196.5811965812, -10256.4102564103, 0, 0, -35908358.017625, 1869.65811965812, 2684.34294871795, 1.08974358974359, 0, 0, -10256.4102564103, 0, 62820512.8205128, -35902001.1800182, 1869.65811965812, 8192.93803418804, 1.08974358974359, 0, -523.504273504273;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(13) << 0, 13461.5384615385, 13461538.4615385, -280.448717948718, 8980892.88915999, 4609.83173076923, 0, -0.552884615384615, 224.358974358974, 0, 191666.666666667, -53846153.8461539, -7946.04700854701, 26940667.250938, -25034.3830128205, 0, 2.39583333333333, -897.435897435898, 0, 13461.5384615385, -35897435.8974359, 280.448717948718, 8982949.51309161, 2751.28739316239, 0, -0.552884615384615, 299.145299145299, 0, -3205.12820512821, -13461538.4615385, -3271.90170940171, 13469278.5049834, 957.163461538462, 0, -0.435363247863248, 112.179487179487, 0, -97435.8974358974, 62820512.8205128, 4861.11111111111, -26927449.2291966, -11018.0128205128, 0, 2.37179487179487, -1196.5811965812, 0, -97435.8974358974, 89743589.7435897, 2617.52136752137, -26943528.2890256, -15234.7596153846, 0, 2.37179487179487, -747.863247863248, 0, -10256.4102564103, -62820512.8205128, 1869.65811965812, -35902001.1800182, -8192.93803418804, 0, 1.08974358974359, 523.504273504273, 0, -10256.4102564103, 0, 1869.65811965812, -35908358.017625, -2684.34294871795, 0, 1.08974358974359, 0;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(14) << 53846153.8461538, 20192307.6923077, 47115.3846153846, -2750.7077991453, 4609.9813034188, 31413598.7315825, -74.7863247863248, 196.314102564103, -1.93509615384615, 80769230.7692308, -80769230.7692308, 670833.333333333, 25034.3830128205, -25034.3830128205, 94238757.8211902, 785.25641025641, -785.25641025641, 8.38541666666667, -20192307.6923077, -53846153.8461538, 47115.3846153846, -4609.9813034188, 2750.7077991453, 31413598.7315825, -196.314102564103, 74.7863247863248, -1.93509615384615, 20192307.6923077, -20192307.6923077, -11217.9487179487, -957.219551282051, 957.219551282051, 47118823.9978563, -28.0448717948718, 28.0448717948718, -1.52377136752137, -134615384.615385, 94230769.2307692, -341025.641025641, 15230.608974359, -11022.2756410256, -94236209.909812, 186.965811965812, -579.594017094017, 8.30128205128205, -94230769.2307692, 134615384.615385, -341025.641025641, 11022.2756410256, -15230.608974359, -94236209.909812, 579.594017094017, -186.965811965812, 8.30128205128205, 0, -94230769.2307692, -35897.4358974359, 2684.15598290598, -8192.63888888889, -125644473.561422, 0, 130.876068376068, 3.81410256410256, 94230769.2307692, 0, -35897.4358974359, 8192.63888888889, -2684.15598290598, -125644473.561422, -130.876068376068, 0, 3.81410256410256;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(15) << 4300.22128939637, 140.224358974359, 2.56143162393162, -0.192307692307692, 0, 74.7863247863248, 74.818493913944, 0.00105168269230769, -0.022950874732906, 8750.03572542735, -3973.0235042735, 17.1915064102564, 2.39583333333333, 0, 785.25641025641, 224.425351763667, -0.0297976762820513, 0.20846226963141, 3271.9093235844, -140.224358974359, 2.77644230769231, -0.192307692307692, 0, -196.314102564103, 74.8107815742004, -0.00105168269230769, -0.0384407151442308, 3870.1980045406, -1635.95085470085, 2.1875, -0.237713675213675, 0, 28.0448717948718, 112.208510726487, -0.0122696314102564, -0.00799641426282051, -10245.7470769231, 1308.76068376068, -10.8253205128205, 0.576923076923077, 0, -186.965811965812, -224.435483060342, 0.00981570512820513, 0.127055562232906, -2206.21716239316, 2430.55555555556, -3.81410256410256, 0.576923076923077, 0, 205.662393162393, -224.375186585983, 0.0182291666666667, 0.0918517361111111, -5459.40945352564, 934.82905982906, -3.98237179487179, 0.480769230769231, 0, 0, -299.186272226884, 0.00701121794871795, 0.0224060296474359, -2280.99065010684, 934.82905982906, -6.09508547008547, 0.480769230769231, 0, 130.876068376068, -299.162434085859, 0.00701121794871795, 0.0683303552350427;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(16) << -140.224358974359, 3271.9093235844, -2.77644230769231, 0, -0.192307692307692, 196.314102564103, -0.00105168269230769, 74.8107815742004, 0.0384407151442308, -3973.0235042735, 8750.03572542735, -17.1915064102564, 0, 2.39583333333333, -785.25641025641, -0.0297976762820513, 224.425351763667, -0.20846226963141, 140.224358974359, 4300.22128939637, -2.56143162393162, 0, -0.192307692307692, -74.7863247863248, 0.00105168269230769, 74.818493913944, 0.022950874732906, -1635.95085470085, 3870.1980045406, -2.1875, 0, -0.237713675213675, -28.0448717948718, -0.0122696314102564, 112.208510726487, 0.00799641426282051, 2430.55555555556, -2206.21716239316, 3.81410256410256, 0, 0.576923076923077, -205.662393162393, 0.0182291666666667, -224.375186585983, -0.0918517361111111, 1308.76068376068, -10245.7470769231, 10.8253205128205, 0, 0.576923076923077, 186.965811965812, 0.00981570512820513, -224.435483060342, -0.127055562232906, 934.82905982906, -2280.99065010684, 6.09508547008547, 0, 0.480769230769231, -130.876068376068, 0.00701121794871795, -299.162434085859, -0.0683303552350427, 934.82905982906, -5459.40945352564, 3.98237179487179, 0, 0.480769230769231, 0, 0.00701121794871795, -299.186272226884, -0.0224060296474359;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(17) << 2.85122863247863, -2.70165598290598, 1682.71515024038, 299.145299145299, 224.358974358974, -0.673076923076923, -0.0229487012553419, 0.0384412760416667, 261.764564540977, 17.1915064102564, -17.1915064102564, 3888.99606517094, 897.435897435898, -897.435897435898, 8.38541666666667, 0.20846226963141, -0.20846226963141, 785.287332712654, 2.70165598290598, -2.85122863247863, 1682.71515024038, -224.358974358974, -299.145299145299, -0.673076923076923, -0.0384412760416667, 0.0229487012553419, 261.764564540977, 2.15945512820513, -2.15945512820513, 1720.10256063034, 112.179487179487, -112.179487179487, -0.831997863247863, -0.00799662459935898, 0.00799662459935898, 392.641099041997, -12.900641025641, 1.68269230769231, -2767.15576068376, -747.863247863248, 299.145299145299, 2.01923076923077, 0.12703999732906, -0.0918677216880342, -785.27638365396, -1.68269230769231, 12.900641025641, -2767.15576068376, -299.145299145299, 747.863247863248, 2.01923076923077, 0.0918677216880342, -0.12703999732906, -785.27638365396, -4.0758547008547, 6.24465811965812, -1720.10870245727, 0, -523.504273504273, 1.68269230769231, 0.022405328525641, -0.0683292334401709, -1047.02151201541, -6.24465811965812, 4.0758547008547, -1720.10870245727, 523.504273504273, 0, 1.68269230769231, 0.0683292334401709, -0.022405328525641, -1047.02151201541;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(18) << 928846689.166667, 392628205.128205, 62820.5128205128, -26923.0769230769, 0, -20192307.6923077, 3870.19453819444, 1635.95085470085, 0.261752136752137, 785258237.660256, 33653846.1538461, 648397.435897436, 13461.5384615385, 0, -20192307.6923077, 3271.9093235844, 140.224358974359, 2.70165598290598, 2100001045.44872, 953525641.025641, 666346.153846154, 13461.5384615385, 0, 80769230.7692308, 8750.00435603633, 3973.0235042735, 2.77644230769231, 1032051789.26282, -33653846.1538461, 49358.9743589744, -26923.0769230769, 0, -53846153.8461538, 4300.21578859509, -140.224358974359, 0.205662393162393, -1310257199.10256, -224358974.358974, -228846.153846154, 115384.615384615, 0, 0, -5459.40499626068, -934.82905982906, -0.953525641025641, -529489002.179487, -583333333.333333, -332051.282051282, 108974.358974359, 0, 175000000, -2206.20417574786, -2430.55555555556, -1.38354700854701, -2458975022.05128, -314102564.102564, -166025.641025641, 115384.615384615, 0, 134615384.615385, -10245.729258547, -1308.76068376068, -0.691773504273504, -547436538.205128, -224358974.358974, -700000, 91025.641025641, 0, -94230769.2307692, -2280.9855758547, -934.82905982906, -2.91666666666667;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(19) << 392628205.128205, 928846689.166667, -42628.2051282051, 0, -26923.0769230769, -20192307.6923077, 1635.95085470085, 3870.19453819444, -0.177617521367521, -33653846.1538461, 1032053109.45513, -684294.871794872, 0, 13461.5384615385, -53846153.8461538, -140.224358974359, 4300.22128939637, -2.85122863247863, 953525641.025641, 2100001045.44872, 614743.58974359, 0, 13461.5384615385, 80769230.7692308, 3973.0235042735, 8750.00435603633, 2.56143162393162, 33653846.1538461, 785256917.467949, 49358.9743589744, 0, -26923.0769230769, -20192307.6923077, 140.224358974359, 3271.90382278312, 0.205662393162393, -224358974.358974, -547436686.282051, -305128.205128205, 0, 115384.615384615, -94230769.2307692, -934.82905982906, -2280.98619284188, -1.27136752136752, -314102564.102564, -2458976181.66667, 4487.17948717949, 0, 108974.358974359, 134615384.615385, -1308.76068376068, -10245.7340902778, 0.0186965811965812, -583333333.333333, -529487842.564103, 367948.717948718, 0, 115384.615384615, 175000000, -2430.55555555556, -2206.19934401709, 1.53311965811966, -224358974.358974, -1310257051.02564, -4487.17948717949, 0, 91025.641025641, 0, -934.82905982906, -5459.4043792735, -0.0186965811965812;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(20) << 42628.2051282051, -62820.5128205128, 412822083.333333, -13461538.4615385, -13461538.4615385, -94230.7692307692, 0.177617521367521, -0.261752136752137, 1720.09201388889, 666346.153846154, -614743.58974359, 403851636.057692, -13461538.4615385, -35897435.8974359, 47115.3846153846, 2.77644230769231, -2.56143162393162, 1682.71515024038, 666346.153846154, 614743.58974359, 933336270.192308, 53846153.8461539, 53846153.8461539, 47115.3846153846, 2.77644230769231, 2.56143162393162, 3888.90112580128, 42628.2051282051, 62820.5128205128, 403847624.519231, -35897435.8974359, -13461538.4615385, -94230.7692307692, 0.177617521367521, 0.261752136752137, 1682.6984354968, -296153.846153846, -430769.230769231, -412823070.512821, 0, -62820512.8205128, 403846.153846154, -1.23397435897436, -1.79487179487179, -1720.09612713675, -125641.025641026, 0, -664107800.641026, 116666666.666667, 89743589.7435897, 381410.256410256, -0.523504273504274, 0, -2767.11583600427, -296153.846153846, 430769.230769231, -664104390.384615, 89743589.7435897, 116666666.666667, 403846.153846154, -1.23397435897436, 1.79487179487179, -2767.10162660256, -700000, 0, -412822352.564103, -62820512.8205128, 0, 318589.743589744, -2.91666666666667, 0, -1720.09313568376;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(21) << -26923.0769230769, 0, 13461538.4615385, 13469286.1903584, 3271.90170940171, 2628.72863247863, -0.336538461538462, 0, -112.179487179487, -73076.9230769231, 0, -13461538.4615385, 8980892.88915999, 280.448717948718, -4609.9813034188, -0.192307692307692, 0, -224.358974358974, 13461.5384615385, 0, 53846153.8461539, 26940591.2266608, 7946.04700854701, 3467.09134615385, 0.168269230769231, 0, 897.435897435898, -39102.5641025641, 0, 35897435.8974359, 8982963.05978232, -280.448717948718, 1731.18055555556, -0.387286324786325, 0, -299.145299145299, 46153.8461538462, 0, 0, -35908379.3548643, -1869.65811965812, -9873.70192307692, 1.15384615384615, 0, 0, 1282.05128205128, 0, -62820512.8205128, -26927517.4724541, -4861.11111111111, -12310.4594017094, 0.913461538461539, 0, 1196.5811965812, 43589.7435897436, 0, -89743589.7435897, -26943574.7918504, -2617.52136752137, 896.052350427351, 1.14316239316239, 0, 747.863247863248, 34615.3846153846, 0, 62820512.8205128, -35902008.8621774, -1869.65811965812, 4609.55128205128, 0.902777777777778, 0, -523.504273504273;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(22) << 0, -26923.0769230769, 13461538.4615385, 3271.90170940171, 13469286.1903584, -2628.56036324786, 0, -0.336538461538462, -112.179487179487, 0, -73076.9230769231, 35897435.8974359, -280.448717948718, 8982949.51309161, 2750.7077991453, 0, -0.192307692307692, -299.145299145299, 0, 13461.5384615385, 53846153.8461539, 7946.04700854701, 26940591.2266608, 5517.94337606838, 0, 0.168269230769231, 897.435897435898, 0, -39102.5641025641, -13461538.4615385, 280.448717948718, 8980906.4358507, -4358.56303418803, 0, -0.387286324786325, -224.358974358974, 0, 46153.8461538462, 62820512.8205128, -1869.65811965812, -35902022.5172575, 13330.7905982906, 0, 1.15384615384615, -523.504273504273, 0, 1282.05128205128, -89743589.7435897, -2617.52136752137, -26943596.5322831, 17051.3194444444, 0, 0.913461538461539, 747.863247863248, 0, 43589.7435897436, -62820512.8205128, -4861.11111111111, -26927495.7320214, 3336.39957264957, 0, 1.14316239316239, 1196.5811965812, 0, 34615.3846153846, 0, -1869.65811965812, -35908365.6997842, 897.398504273504, 0, 0.902777777777778, 0;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(23) << 20192307.6923077, 20192307.6923077, -94230.7692307692, 2628.56036324786, -2628.72863247863, 47118841.9254808, -28.0448717948718, -28.0448717948718, -1.17788461538462, -20192307.6923077, 53846153.8461538, -255769.230769231, -4609.83173076923, 2751.28739316239, 31413598.7315825, -196.314102564103, -74.7863247863248, -0.673076923076923, 80769230.7692308, 80769230.7692308, 47115.3846153846, 3467.09134615385, 5517.94337606838, 94238580.3877217, 785.25641025641, 785.25641025641, 0.588942307692308, 53846153.8461538, -20192307.6923077, -136858.974358974, 1731.12446581197, -4358.45085470085, 31413630.3327684, -74.7863247863248, -196.314102564103, -1.35550213675214, 0, 94230769.2307692, 161538.461538462, -9874.26282051282, 13329.7435897436, -125644523.343964, 0, -130.876068376068, 4.03846153846154, -94230769.2307692, -134615384.615385, 4487.17948717949, -12308.7393162393, 17051.2820512821, -94236369.1248344, 579.594017094017, 186.965811965812, 3.19711538461538, -134615384.615385, -94230769.2307692, 152564.102564103, 894.967948717949, 3336.92307692308, -94236318.3912874, 186.965811965812, 579.594017094017, 4.00106837606838, 94230769.2307692, 0, 121153.846153846, 4609.55128205128, 897.435897435898, -125644491.479006, -130.876068376068, 0, 3.15972222222222;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(24) << 3870.19453819444, 1635.95085470085, 0.261752136752137, -0.336538461538462, 0, 28.0448717948718, 112.208574803054, 0.0122696314102564, 0.021903672542735, 3271.9093235844, 140.224358974359, 2.70165598290598, -0.552884615384615, 0, -196.314102564103, 74.8107815742004, 0.00105168269230769, -0.0384412760416667, 8750.00435603633, 3973.0235042735, 2.77644230769231, 0.168269230769231, 0, 785.25641025641, 224.424718515576, 0.0297976762820513, 0.0288669771634615, 4300.21578859509, -140.224358974359, 0.205662393162393, -0.438034188034188, 0, 74.7863247863248, 74.8186068534572, -0.00105168269230769, 0.0144246193910256, -5459.40499626068, -934.82905982906, -0.953525641025641, 0.865384615384615, 0, 0, -299.18645007807, -0.00701121794871795, -0.082272108707265, -2206.20417574786, -2430.55555555556, -1.38354700854701, 0.46474358974359, 0, 205.662393162393, -224.37575539884, -0.0182291666666667, -0.102574479166667, -10245.729258547, -1308.76068376068, -0.691773504273504, 0.844017094017094, 0, -186.965811965812, -224.435870747217, -0.00981570512820513, 0.00747344417735043, -2280.9855758547, -934.82905982906, -2.91666666666667, 0.667735042735043, 0, 130.876068376068, -299.162498150366, -0.00701121794871795, 0.0384396634615385;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(25) << 1635.95085470085, 3870.19453819444, -0.177617521367521, 0, -0.336538461538462, 28.0448717948718, 0.0122696314102564, 112.208574803054, -0.0219030415331197, -140.224358974359, 4300.22128939637, -2.85122863247863, 0, -0.552884615384615, 74.7863247863248, -0.00105168269230769, 74.818493913944, 0.0229487012553419, 3973.0235042735, 8750.00435603633, 2.56143162393162, 0, 0.168269230769231, 785.25641025641, 0.0297976762820513, 224.424718515576, 0.0459593816773504, 140.224358974359, 3271.90382278312, 0.205662393162393, 0, -0.438034188034188, -196.314102564103, 0.00105168269230769, 74.8108945137136, -0.0363232438568376, -934.82905982906, -2280.98619284188, -1.27136752136752, 0, 0.865384615384615, 130.876068376068, -0.00701121794871795, -299.162611937045, 0.111101575854701, -1308.76068376068, -10245.7340902778, 0.0186965811965812, 0, 0.46474358974359, -186.965811965812, -0.00981570512820513, -224.436051873198, 0.142094157318376, -2430.55555555556, -2206.19934401709, 1.53311965811966, 0, 0.844017094017094, 205.662393162393, -0.0182291666666667, -224.375574272858, 0.0277892761752137, -934.82905982906, -5459.4043792735, -0.0186965811965812, 0, 0.667735042735043, 0, -0.00701121794871795, -299.186336291392, 0.00747849225427351;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(26) << 0.177617521367521, -0.261752136752137, 1720.09201388889, 112.179487179487, 112.179487179487, -1.17788461538462, 0.0219030415331197, -0.021903672542735, 392.641248535546, 2.77644230769231, -2.56143162393162, 1682.71515024038, -224.358974358974, 299.145299145299, -1.93509615384615, -0.0384407151442308, 0.022950874732906, 261.764564540977, 2.77644230769231, 2.56143162393162, 3888.90112580128, 897.435897435898, 897.435897435898, 0.588942307692308, 0.0288669771634615, 0.0459593816773504, 785.285854970694, 0.177617521367521, 0.261752136752137, 1682.6984354968, 299.145299145299, -224.358974358974, -1.53311965811966, 0.0144244090544872, -0.0363228231837607, 261.764828037412, -1.23397435897436, -1.79487179487179, -1720.09612713675, 0, 523.504273504273, 3.02884615384615, -0.0822742120726496, 0.11109764957265, -1047.0219269852, -0.523504273504274, 0, -2767.11583600427, -299.145299145299, -747.863247863248, 1.62660256410256, -0.102568028846154, 0.142094017094017, -785.27771081179, -1.23397435897436, 1.79487179487179, -2767.10162660256, -747.863247863248, -299.145299145299, 2.95405982905983, 0.00746937767094017, 0.0277912393162393, -785.277288162484, -2.91666666666667, 0, -1720.09313568376, 523.504273504273, 0, 2.33707264957265, 0.0384396634615385, 0.00747863247863248, -1047.02166147131;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(27) << 785256917.467949, -33653846.1538461, -62820.5128205128, -39102.5641025641, 0, 20192307.6923077, 3271.90382278312, -140.224358974359, -0.261752136752137, 928847521.089744, -392628205.128205, 518269.230769231, -3205.12820512821, 0, 20192307.6923077, 3870.1980045406, -1635.95085470085, 2.15945512820513, 1032051789.26282, 33653846.1538461, 42628.2051282051, -39102.5641025641, 0, 53846153.8461538, 4300.21578859509, 140.224358974359, 0.177617521367521, 2100000683.71795, -953525641.025641, -525000, -3205.12820512821, 0, -80769230.7692308, 8750.00284882479, -3973.0235042735, -2.1875, -1310257259.23077, 224358974.358974, -76282.0512820513, 124358.974358974, 0, 0, -5459.40524679487, 934.82905982906, -0.31784188034188, -547436746.410256, 224358974.358974, 367948.717948718, 124358.974358974, 0, 94230769.2307692, -2280.98644337607, 934.82905982906, 1.53311965811966, -2458975042.69231, 314102564.102564, 76282.0512820513, 124358.974358974, 0, -134615384.615385, -10245.7293445513, 1308.76068376068, 0.31784188034188, -529487863.205128, 583333333.333333, -341025.641025641, 124358.974358974, 0, -175000000, -2206.19943002137, 2430.55555555556, -1.42094017094017;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(28) << 33653846.1538461, 1032051789.26282, -42628.2051282051, 0, -39102.5641025641, -53846153.8461538, 140.224358974359, 4300.21578859509, -0.177617521367521, -392628205.128205, 928847521.089744, -518269.230769231, 0, -3205.12820512821, -20192307.6923077, -1635.95085470085, 3870.1980045406, -2.15945512820513, -33653846.1538461, 785256917.467949, 62820.5128205128, 0, -39102.5641025641, -20192307.6923077, -140.224358974359, 3271.90382278312, 0.261752136752137, -953525641.025641, 2100000683.71795, 525000, 0, -3205.12820512821, 80769230.7692308, -3973.0235042735, 8750.00284882479, 2.1875, 224358974.358974, -547436746.410256, -367948.717948718, 0, 124358.974358974, -94230769.2307692, 934.82905982906, -2280.98644337607, -1.53311965811966, 224358974.358974, -1310257259.23077, 76282.0512820513, 0, 124358.974358974, 0, 934.82905982906, -5459.40524679487, 0.31784188034188, 583333333.333333, -529487863.205128, 341025.641025641, 0, 124358.974358974, 175000000, 2430.55555555556, -2206.19943002137, 1.42094017094017, 314102564.102564, -2458975042.69231, -76282.0512820513, 0, 124358.974358974, 134615384.615385, 1308.76068376068, -10245.7293445513, -0.31784188034188;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(29) << -49358.9743589744, -49358.9743589744, 403847624.519231, 13461538.4615385, -35897435.8974359, -136858.974358974, -0.205662393162393, -0.205662393162393, 1682.6984354968, 525000, -525000, 412824614.551282, 13461538.4615385, -13461538.4615385, -11217.9487179487, 2.1875, -2.1875, 1720.10256063034, 49358.9743589744, 49358.9743589744, 403847624.519231, 35897435.8974359, -13461538.4615385, -136858.974358974, 0.205662393162393, 0.205662393162393, 1682.6984354968, -525000, 525000, 933335309.48718, -53846153.8461539, 53846153.8461539, -11217.9487179487, -2.1875, 2.1875, 3888.89712286325, -161538.461538462, -457692.307692308, -412823046.282051, 0, -62820512.8205128, 435256.41025641, -0.673076923076923, -1.90705128205128, -1720.09602617521, 457692.307692308, 161538.461538462, -412823046.282051, 62820512.8205128, 0, 435256.41025641, 1.90705128205128, 0.673076923076923, -1720.09602617521, 161538.461538462, 457692.307692308, -664104540.25641, -89743589.7435897, 116666666.666667, 435256.41025641, 0.673076923076923, 1.90705128205128, -2767.10225106838, -457692.307692308, -161538.461538462, -664104540.25641, -116666666.666667, 89743589.7435897, 435256.41025641, -1.90705128205128, -0.673076923076923, -2767.10225106838;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(30) << -26923.0769230769, 0, 13461538.4615385, 8980906.4358507, -280.448717948718, 4358.45085470085, -0.438034188034188, 0, 224.358974358974, -50641.0256410256, 0, -13461538.4615385, 13469278.5049834, -3271.90170940171, -957.219551282051, -0.237713675213675, 0, 112.179487179487, -26923.0769230769, 0, -35897435.8974359, 8982963.05978232, 280.448717948718, 1731.12446581197, -0.438034188034188, 0, 299.145299145299, -3205.12820512821, 0, -53846153.8461539, 26940582.9544156, -7946.04700854701, -1927.45192307692, -0.0400641025641026, 0, -897.435897435898, 37179.4871794872, 0, 0, -35908371.6630577, 1869.65811965812, -6282.68696581197, 1.19123931623932, 0, 0, 37179.4871794872, 0, -62820512.8205128, -35902014.8254509, 1869.65811965812, -9740.5235042735, 1.19123931623932, 0, 523.504273504273, 16666.6666666667, 0, 89743589.7435897, -26943576.9074071, 2617.52136752137, -2691.67200854701, 1.10576923076923, 0, -747.863247863248, 16666.6666666667, 0, 62820512.8205128, -26927497.847578, 4861.11111111111, 2048.44017094017, 1.10576923076923, 0, -1196.5811965812;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(31) << 0, -26923.0769230769, 35897435.8974359, 280.448717948718, 8982963.05978232, -1731.12446581197, 0, -0.438034188034188, -299.145299145299, 0, -50641.0256410256, 13461538.4615385, -3271.90170940171, 13469278.5049834, 957.219551282051, 0, -0.237713675213675, -112.179487179487, 0, -26923.0769230769, -13461538.4615385, -280.448717948718, 8980906.4358507, -4358.45085470085, 0, -0.438034188034188, -224.358974358974, 0, -3205.12820512821, 53846153.8461539, -7946.04700854701, 26940582.9544156, 1927.45192307692, 0, -0.0400641025641026, 897.435897435898, 0, 37179.4871794872, 62820512.8205128, 1869.65811965812, -35902014.8254509, 9740.5235042735, 0, 1.19123931623932, -523.504273504273, 0, 37179.4871794872, 0, 1869.65811965812, -35908371.6630577, 6282.68696581197, 0, 1.19123931623932, 0, 0, 16666.6666666667, -62820512.8205128, 4861.11111111111, -26927497.847578, -2048.44017094017, 0, 1.10576923076923, 1196.5811965812, 0, 16666.6666666667, -89743589.7435897, 2617.52136752137, -26943576.9074071, 2691.67200854701, 0, 1.10576923076923, 747.863247863248;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(32) << 20192307.6923077, 53846153.8461538, -94230.7692307692, 4358.56303418803, -1731.18055555556, 31413630.3327684, 196.314102564103, -74.7863247863248, -1.53311965811966, -20192307.6923077, 20192307.6923077, -177243.58974359, -957.163461538462, 957.163461538462, 47118823.9978563, 28.0448717948718, -28.0448717948718, -0.831997863247863, -53846153.8461538, -20192307.6923077, -94230.7692307692, 1731.18055555556, -4358.56303418803, 31413630.3327684, 74.7863247863248, -196.314102564103, -1.53311965811966, -80769230.7692308, 80769230.7692308, -11217.9487179487, -1927.45192307692, 1927.45192307692, 94238561.084844, -785.25641025641, 785.25641025641, -0.140224358974359, 0, 94230769.2307692, 130128.205128205, -6283.39743589744, 9739.77564102564, -125644505.395044, 0, -130.876068376068, 4.16933760683761, -94230769.2307692, 0, 130128.205128205, -9739.77564102564, 6283.39743589744, -125644505.395044, 130.876068376068, 0, 4.16933760683761, 134615384.615385, -94230769.2307692, 58333.3333333333, -2690.96153846154, -2047.46794871795, -94236323.3284338, -186.965811965812, 579.594017094017, 3.87019230769231, 94230769.2307692, -134615384.615385, 58333.3333333333, 2047.46794871795, 2690.96153846154, -94236323.3284338, -579.594017094017, 186.965811965812, 3.87019230769231;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(33) << 3271.90382278312, -140.224358974359, -0.261752136752137, -0.387286324786325, 0, 196.314102564103, 74.8108945137136, -0.00105168269230769, 0.0363228231837607, 3870.1980045406, -1635.95085470085, 2.15945512820513, -0.435363247863248, 0, -28.0448717948718, 112.208510726487, -0.0122696314102564, -0.00799662459935898, 4300.21578859509, 140.224358974359, 0.177617521367521, -0.387286324786325, 0, -74.7863247863248, 74.8186068534572, 0.00105168269230769, 0.0144244090544872, 8750.00284882479, -3973.0235042735, -2.1875, -0.0400641025641026, 0, -785.25641025641, 224.424649594016, -0.0297976762820513, -0.016042047275641, -5459.40524679487, 934.82905982906, -0.31784188034188, 0.827991452991453, 0, 0, -299.186385977385, 0.00701121794871795, -0.0523528111645299, -2280.98644337607, 934.82905982906, 1.53311965811966, 0.827991452991453, 0, -130.876068376068, -299.16254783636, 0.00701121794871795, -0.0811850827991453, -10245.7293445513, 1308.76068376068, 0.31784188034188, 0.657051282051282, 0, 186.965811965812, -224.435888376067, 0.00981570512820513, -0.0224335136217949, -2206.19943002137, 2430.55555555556, -1.42094017094017, 0.657051282051282, 0, -205.662393162393, -224.375591901708, 0.0182291666666667, 0.017083360042735;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(34) << 140.224358974359, 4300.21578859509, -0.177617521367521, 0, -0.387286324786325, 74.7863247863248, 0.00105168269230769, 74.8186068534572, -0.0144244090544872, -1635.95085470085, 3870.1980045406, -2.15945512820513, 0, -0.435363247863248, 28.0448717948718, -0.0122696314102564, 112.208510726487, 0.00799662459935898, -140.224358974359, 3271.90382278312, 0.261752136752137, 0, -0.387286324786325, -196.314102564103, -0.00105168269230769, 74.8108945137136, -0.0363228231837607, -3973.0235042735, 8750.00284882479, 2.1875, 0, -0.0400641025641026, 785.25641025641, -0.0297976762820513, 224.424649594016, 0.016042047275641, 934.82905982906, -2280.98644337607, -1.53311965811966, 0, 0.827991452991453, 130.876068376068, 0.00701121794871795, -299.16254783636, 0.0811850827991453, 934.82905982906, -5459.40524679487, 0.31784188034188, 0, 0.827991452991453, 0, 0.00701121794871795, -299.186385977385, 0.0523528111645299, 2430.55555555556, -2206.19943002137, 1.42094017094017, 0, 0.657051282051282, 205.662393162393, 0.0182291666666667, -224.375591901708, -0.017083360042735, 1308.76068376068, -10245.7293445513, -0.31784188034188, 0, 0.657051282051282, -186.965811965812, 0.00981570512820513, -224.435888376067, 0.0224335136217949;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(35) << -0.205662393162393, -0.205662393162393, 1682.6984354968, 224.358974358974, 299.145299145299, -1.35550213675214, 0.0363232438568376, -0.0144246193910256, 261.764828037412, 2.1875, -2.1875, 1720.10256063034, -112.179487179487, 112.179487179487, -1.52377136752137, -0.00799641426282051, 0.00799641426282051, 392.641099041997, 0.205662393162393, 0.205662393162393, 1682.6984354968, -299.145299145299, -224.358974358974, -1.35550213675214, 0.0144246193910256, -0.0363232438568376, 261.764828037412, -2.1875, 2.1875, 3888.89712286325, -897.435897435898, 897.435897435898, -0.140224358974359, -0.016042047275641, 0.016042047275641, 785.285694150074, -0.673076923076923, -1.90705128205128, -1720.09602617521, 0, 523.504273504273, 2.89797008547009, -0.0523554754273504, 0.0811822783119658, -1047.02177741179, 1.90705128205128, 0.673076923076923, -1720.09602617521, -523.504273504273, 0, 2.89797008547009, -0.0811822783119658, 0.0523554754273504, -1047.02177741179, 0.673076923076923, 1.90705128205128, -2767.10225106838, 747.863247863248, -299.145299145299, 2.29967948717949, -0.0224308493589744, -0.0170797142094017, -785.277329299647, -1.90705128205128, -0.673076923076923, -2767.10225106838, 299.145299145299, -747.863247863248, 2.29967948717949, 0.0170797142094017, 0.0224308493589744, -785.277329299647;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(36) << -2458976181.66667, -583333333.333333, 0, 1282.05128205128, 0, 134615384.615385, -10245.7340902778, -2430.55555555556, 0, -2458979298.46154, 583333333.333333, -3096153.84615385, -97435.8974358974, 0, -134615384.615385, -10245.7470769231, 2430.55555555556, -12.900641025641, -1310257199.10256, -224358974.358974, -296153.846153846, 46153.8461538462, 0, 0, -5459.40499626068, -934.82905982906, -1.23397435897436, -1310257259.23077, 224358974.358974, -161538.461538462, 37179.4871794872, 0, 0, -5459.40524679487, 934.82905982906, -0.673076923076923, 5456416107.69231, 0, 1884615.38461538, -128205.128205128, 0, 0, 22735.0671153846, 0, 7.8525641025641, 689.230769230769, -897435897.435898, -215384.615384615, -215384.615384615, 0, -269230769.230769, 0.00287179487179487, -3739.31623931624, -0.897435897435898, 2082052412.82051, 0, 412820.512820513, -128205.128205128, 0, 0, 8675.21838675214, 0, 1.72008547008547, 728.717948717949, 897435897.435898, 1471794.87179487, -179487.17948718, 0, 269230769.230769, 0.00303632478632479, 3739.31623931624, 6.13247863247863;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(37) << -314102564.102564, -529489002.179487, 125641.025641026, 0, 1282.05128205128, 94230769.2307692, -1308.76068376068, -2206.20417574786, 0.523504273504274, 314102564.102564, -529492118.974359, 403846.153846154, 0, -97435.8974358974, 94230769.2307692, 1308.76068376068, -2206.21716239316, 1.68269230769231, -224358974.358974, -547436686.282051, -430769.230769231, 0, 46153.8461538462, 94230769.2307692, -934.82905982906, -2280.98619284188, -1.79487179487179, 224358974.358974, -547436746.410256, -457692.307692308, 0, 37179.4871794872, 94230769.2307692, 934.82905982906, -2280.98644337607, -1.90705128205128, 0, 2943595594.87179, -538461.538461539, 0, -128205.128205128, -323076923.076923, 0, 12264.9816452991, -2.24358974358974, -897435897.435898, 689.230769230769, 700000, 0, -215384.615384615, -269230769.230769, -3739.31623931624, 0.00287179487179487, 2.91666666666667, 0, -789742458.974359, -179487.17948718, 0, -128205.128205128, -323076923.076923, 0, -3290.59357905983, -0.747863247863248, 897435897.435898, 728.717948717949, 376923.076923077, 0, -179487.17948718, -269230769.230769, 3739.31623931624, 0.00303632478632479, 1.57051282051282;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(38) << -4487.17948717949, 332051.282051282, -664107800.641026, 89743589.7435897, 62820512.8205128, 4487.17948717949, -0.0186965811965812, 1.38354700854701, -2767.11583600427, -2598076.92307692, 915384.615384615, -664117382.564103, -89743589.7435897, 62820512.8205128, -341025.641025641, -10.8253205128205, 3.81410256410256, -2767.15576068376, -228846.153846154, -305128.205128205, -412823070.512821, 0, 62820512.8205128, 161538.461538462, -0.953525641025641, -1.27136752136752, -1720.09612713675, -76282.0512820513, -367948.717948718, -412823046.282051, 0, 62820512.8205128, 130128.205128205, -0.31784188034188, -1.53311965811966, -1720.09602617521, 1884615.38461538, -538461.538461539, 1866682777.4359, 0, -215384615.384615, -448717.948717949, 7.8525641025641, -2.24358974358974, 7777.84490598291, -700000, 215384.615384615, 3483.84615384615, -179487179.48718, -179487179.48718, -753846.153846154, -2.91666666666667, 0.897435897435898, 0.0145160256410256, 305128.205128205, -394871.794871795, 287182868.717949, 0, -215384615.384615, -448717.948717949, 1.27136752136752, -1.64529914529915, 1196.59528632479, 1417948.71794872, 143589.743589744, 2170, 179487179.48718, -179487179.48718, -628205.128205128, 5.90811965811966, 0.598290598290598, 0.00904166666666667;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(39) << 108974.358974359, 0, -89743589.7435897, -26943596.5322831, -4861.11111111111, -17051.2820512821, 0.46474358974359, 0, 747.863247863248, 333333.333333333, 0, 89743589.7435897, -26943528.2890256, 4861.11111111111, 15230.608974359, 0.576923076923077, 0, -747.863247863248, 115384.615384615, 0, 0, -35908379.3548643, -1869.65811965812, -9874.26282051282, 0.865384615384615, 0, 0, 124358.974358974, 0, 0, -35908371.6630577, 1869.65811965812, -6283.39743589744, 0.827991452991453, 0, 0, -128205.128205128, 0, 0, 143635354.236795, 0, 64631.0897435898, -1.6025641025641, 0, 0, -215384.615384615, 0, 179487179.48718, 89743666.6724103, -7478.63247863248, 39485.3846153846, -2.69230769230769, 0, -1495.7264957265, -158974.358974359, 0, 0, 71812250.1803633, 0, 7182.92735042735, -1.73076923076923, 0, 0, -179487.17948718, 0, -179487179.48718, 89743623.9804316, 7478.63247863248, -3577.47863247863, -2.24358974358974, 0, 1495.7264957265;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(40) << 0, 108974.358974359, -116666666.666667, -2617.52136752137, -26927517.4724541, 12308.7393162393, 0, 0.46474358974359, 299.145299145299, 0, 333333.333333333, -116666666.666667, 2617.52136752137, -26927449.2291966, -11022.2756410256, 0, 0.576923076923077, 299.145299145299, 0, 115384.615384615, -62820512.8205128, -1869.65811965812, -35902022.5172575, 13329.7435897436, 0, 0.865384615384615, 523.504273504273, 0, 124358.974358974, -62820512.8205128, 1869.65811965812, -35902014.8254509, 9739.77564102564, 0, 0.827991452991453, 523.504273504273, 0, -128205.128205128, -215384615.384615, 0, 143614414.065855, -82055.7692307692, 0, -1.6025641025641, -3589.74358974359, 0, -215384.615384615, 179487179.48718, -7478.63247863248, 89743666.6724103, -39481.3461538462, 0, -2.69230769230769, -1495.7264957265, 0, -158974.358974359, 215384615.384615, 0, 71788318.5564316, -5129.70085470086, 0, -1.73076923076923, -1794.87179487179, 0, -179487.17948718, 179487179.48718, 7478.63247863248, 89743623.9804316, -14355.8333333333, 0, -2.24358974358974, -1495.7264957265;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(41) << -134615384.615385, -175000000, 381410.256410256, -17051.3194444444, 12310.4594017094, -94236369.1248344, 186.965811965812, -205.662393162393, 1.62660256410256, 134615384.615385, -175000000, 1166666.66666667, 15234.7596153846, -11018.0128205128, -94236209.909812, -186.965811965812, -205.662393162393, 2.01923076923077, 0, -94230769.2307692, 403846.153846154, -9873.70192307692, 13330.7905982906, -125644523.343964, 0, 130.876068376068, 3.02884615384615, 0, -94230769.2307692, 435256.41025641, -6282.68696581197, 9740.5235042735, -125644505.395044, 0, 130.876068376068, 2.89797008547009, 0, -323076923.076923, -448717.948717949, 64631.0897435898, -82055.7692307692, 502579986.117162, 0, -3141.02564102564, -5.60897435897436, 269230769.230769, 269230769.230769, -753846.153846154, 39481.3461538462, -39485.3846153846, 314102743.618776, -373.931623931624, -373.931623931624, -9.42307692307692, 0, 323076923.076923, -556410.256410256, 7182.02991452992, -5131.49572649573, 251284509.686299, 0, -448.717948717949, -6.05769230769231, -269230769.230769, 269230769.230769, -628205.128205128, -3577.92735042735, -14357.7777777778, 314102643.992442, 373.931623931624, -373.931623931624, -7.8525641025641;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(42) << -10245.7340902778, -2430.55555555556, 0, 0.913461538461539, 0, -186.965811965812, -224.436051873198, -0.0182291666666667, -0.142094017094017, -10245.7470769231, 2430.55555555556, -12.900641025641, 2.37179487179487, 0, 186.965811965812, -224.435483060342, 0.0182291666666667, 0.12703999732906, -5459.40499626068, -934.82905982906, -1.23397435897436, 1.15384615384615, 0, 0, -299.18645007807, -0.00701121794871795, -0.0822742120726496, -5459.40524679487, 934.82905982906, -0.673076923076923, 1.19123931623932, 0, 0, -299.186385977385, 0.00701121794871795, -0.0523554754273504, 22735.0671153846, 0, 7.8525641025641, -1.6025641025641, 0, 0, 1196.75288052473, 0, 0.538520432692308, 0.00287179487179487, -3739.31623931624, -0.897435897435898, -2.69230769230769, 0, 373.931623931624, 747.863888910427, -0.0280448717948718, 0.329053098290598, 8675.21838675214, 0, 1.72008547008547, -1.85897435897436, 0, 0, 598.355895334482, 0, 0.0598419604700855, 0.00303632478632479, 3739.31623931624, 6.13247863247863, -2.24358974358974, 0, -373.931623931624, 747.863533142431, 0.0280448717948718, -0.0298685363247863;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(43) << -1308.76068376068, -2206.20417574786, 0.523504273504274, 0, 0.913461538461539, -579.594017094017, -0.00981570512820513, -224.37575539884, 0.102568028846154, 1308.76068376068, -2206.21716239316, 1.68269230769231, 0, 2.37179487179487, -579.594017094017, 0.00981570512820513, -224.375186585983, -0.0918677216880342, -934.82905982906, -2280.98619284188, -1.79487179487179, 0, 1.15384615384615, -130.876068376068, -0.00701121794871795, -299.162611937045, 0.11109764957265, 934.82905982906, -2280.98644337607, -1.90705128205128, 0, 1.19123931623932, -130.876068376068, 0.00701121794871795, -299.16254783636, 0.0811822783119658, 0, 12264.9816452991, -2.24358974358974, 0, -1.6025641025641, -3141.02564102564, 0, 1196.67435488371, -0.683777510683761, -3739.31623931624, 0.00287179487179487, 2.91666666666667, 0, -2.69230769230769, 373.931623931624, -0.0280448717948718, 747.863888910427, -0.329037954059829, 0, -3290.59357905983, -0.747863247863248, 0, -1.85897435897436, 448.717948717949, 0, 598.266151744738, -0.0427406517094017, 3739.31623931624, 0.00303632478632479, 1.57051282051282, 0, -2.24358974358974, 373.931623931624, 0.0280448717948718, 747.863533142431, -0.119646340811966;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(44) << -0.0186965811965812, 1.38354700854701, -2767.11583600427, -747.863247863248, -1196.5811965812, 3.19711538461538, -0.142094157318376, 0.102574479166667, -785.27771081179, -10.8253205128205, 3.81410256410256, -2767.15576068376, 747.863247863248, -1196.5811965812, 8.30128205128205, 0.127055562232906, -0.0918517361111111, -785.27638365396, -0.953525641025641, -1.27136752136752, -1720.09612713675, 0, -523.504273504273, 4.03846153846154, -0.082272108707265, 0.111101575854701, -1047.0219269852, -0.31784188034188, -1.53311965811966, -1720.09602617521, 0, -523.504273504273, 4.16933760683761, -0.0523528111645299, 0.0811850827991453, -1047.02177741179, 7.8525641025641, -2.24358974358974, 7777.84490598291, 0, -3589.74358974359, -5.60897435897436, 0.538520432692308, -0.683777510683761, 4188.09525406472, -2.91666666666667, 0.897435897435898, 0.0145160256410256, 1495.7264957265, 1495.7264957265, -9.42307692307692, 0.329037954059829, -0.329053098290598, 2617.52286335673, 1.27136752136752, -1.64529914529915, 1196.59528632479, 0, 1794.87179487179, -6.50641025641026, 0.0598385950854701, -0.0427473824786325, 2094.02661192904, 5.90811965811966, 0.598290598290598, 0.00904166666666667, -1495.7264957265, 1495.7264957265, -7.8525641025641, -0.029870219017094, -0.119653632478632, 2617.52203318747;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(45) << -547436686.282051, -224358974.358974, 430769.230769231, 46153.8461538462, 0, -94230769.2307692, -2280.98619284188, -934.82905982906, 1.79487179487179, -529492118.974359, 314102564.102564, -403846.153846154, -97435.8974358974, 0, -94230769.2307692, -2206.21716239316, 1308.76068376068, -1.68269230769231, -529489002.179487, -314102564.102564, -125641.025641026, 1282.05128205128, 0, -94230769.2307692, -2206.20417574786, -1308.76068376068, -0.523504273504274, -547436746.410256, 224358974.358974, 457692.307692308, 37179.4871794872, 0, -94230769.2307692, -2280.98644337607, 934.82905982906, 1.90705128205128, 689.230769230769, -897435897.435898, -700000, -215384.615384615, 0, 269230769.230769, 0.00287179487179487, -3739.31623931624, -2.91666666666667, 2943595594.87179, 0, 538461.538461539, -128205.128205128, 0, 323076923.076923, 12264.9816452991, 0, 2.24358974358974, 728.717948717949, 897435897.435898, -376923.076923077, -179487.17948718, 0, 269230769.230769, 0.00303632478632479, 3739.31623931624, -1.57051282051282, -789742458.974359, 0, 179487.17948718, -128205.128205128, 0, 323076923.076923, -3290.59357905983, 0, 0.747863247863248;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(46) << -224358974.358974, -1310257199.10256, 296153.846153846, 0, 46153.8461538462, 0, -934.82905982906, -5459.40499626068, 1.23397435897436, 583333333.333333, -2458979298.46154, 3096153.84615385, 0, -97435.8974358974, 134615384.615385, 2430.55555555556, -10245.7470769231, 12.900641025641, -583333333.333333, -2458976181.66667, 0, 0, 1282.05128205128, -134615384.615385, -2430.55555555556, -10245.7340902778, 0, 224358974.358974, -1310257259.23077, 161538.461538462, 0, 37179.4871794872, 0, 934.82905982906, -5459.40524679487, 0.673076923076923, -897435897.435898, 689.230769230769, 215384.615384615, 0, -215384.615384615, 269230769.230769, -3739.31623931624, 0.00287179487179487, 0.897435897435898, 0, 5456416107.69231, -1884615.38461538, 0, -128205.128205128, 0, 0, 22735.0671153846, -7.8525641025641, 897435897.435898, 728.717948717949, -1471794.87179487, 0, -179487.17948718, -269230769.230769, 3739.31623931624, 0.00303632478632479, -6.13247863247863, 0, 2082052412.82051, -412820.512820513, 0, -128205.128205128, 0, 0, 8675.21838675214, -1.72008547008547;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(47) << 305128.205128205, 228846.153846154, -412823070.512821, -62820512.8205128, 0, 161538.461538462, 1.27136752136752, 0.953525641025641, -1720.09612713675, -915384.615384615, 2598076.92307692, -664117382.564103, -62820512.8205128, 89743589.7435897, -341025.641025641, -3.81410256410256, 10.8253205128205, -2767.15576068376, -332051.282051282, 4487.17948717949, -664107800.641026, -62820512.8205128, -89743589.7435897, 4487.17948717949, -1.38354700854701, 0.0186965811965812, -2767.11583600427, 367948.717948718, 76282.0512820513, -412823046.282051, -62820512.8205128, 0, 130128.205128205, 1.53311965811966, 0.31784188034188, -1720.09602617521, -215384.615384615, 700000, 3483.84615384615, 179487179.48718, 179487179.48718, -753846.153846154, -0.897435897435898, 2.91666666666667, 0.0145160256410256, 538461.538461539, -1884615.38461538, 1866682777.4359, 215384615.384615, 0, -448717.948717949, 2.24358974358974, -7.8525641025641, 7777.84490598291, -143589.743589744, -1417948.71794872, 2170, 179487179.48718, -179487179.48718, -628205.128205128, -0.598290598290598, -5.90811965811966, 0.00904166666666667, 394871.794871795, -305128.205128205, 287182868.717949, 215384615.384615, 0, -448717.948717949, 1.64529914529915, -1.27136752136752, 1196.59528632479;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(48) << 115384.615384615, 0, 62820512.8205128, -35902022.5172575, -1869.65811965812, -13329.7435897436, 0.865384615384615, 0, -523.504273504273, 333333.333333333, 0, 116666666.666667, -26927449.2291966, 2617.52136752137, 11022.2756410256, 0.576923076923077, 0, -299.145299145299, 108974.358974359, 0, 116666666.666667, -26927517.4724541, -2617.52136752137, -12308.7393162393, 0.46474358974359, 0, -299.145299145299, 124358.974358974, 0, 62820512.8205128, -35902014.8254509, 1869.65811965812, -9739.77564102564, 0.827991452991453, 0, -523.504273504273, -215384.615384615, 0, -179487179.48718, 89743666.6724103, -7478.63247863248, 39481.3461538462, -2.69230769230769, 0, 1495.7264957265, -128205.128205128, 0, 215384615.384615, 143614414.065855, 0, 82055.7692307692, -1.6025641025641, 0, 3589.74358974359, -179487.17948718, 0, -179487179.48718, 89743623.9804316, 7478.63247863248, 14355.8333333333, -2.24358974358974, 0, 1495.7264957265, -158974.358974359, 0, -215384615.384615, 71788318.5564316, 0, 5129.70085470086, -1.73076923076923, 0, 1794.87179487179;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(49) << 0, 115384.615384615, 0, -1869.65811965812, -35908379.3548643, 9874.26282051282, 0, 0.865384615384615, 0, 0, 333333.333333333, -89743589.7435897, 4861.11111111111, -26943528.2890256, -15230.608974359, 0, 0.576923076923077, 747.863247863248, 0, 108974.358974359, 89743589.7435897, -4861.11111111111, -26943596.5322831, 17051.2820512821, 0, 0.46474358974359, -747.863247863248, 0, 124358.974358974, 0, 1869.65811965812, -35908371.6630577, 6283.39743589744, 0, 0.827991452991453, 0, 0, -215384.615384615, -179487179.48718, -7478.63247863248, 89743666.6724103, -39485.3846153846, 0, -2.69230769230769, 1495.7264957265, 0, -128205.128205128, 0, 0, 143635354.236795, -64631.0897435898, 0, -1.6025641025641, 0, 0, -179487.17948718, 179487179.48718, 7478.63247863248, 89743623.9804316, 3577.47863247863, 0, -2.24358974358974, -1495.7264957265, 0, -158974.358974359, 0, 0, 71812250.1803633, -7182.92735042735, 0, -1.73076923076923, 0;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(50) << 94230769.2307692, 0, 403846.153846154, -13330.7905982906, 9873.70192307692, -125644523.343964, -130.876068376068, 0, 3.02884615384615, 175000000, -134615384.615385, 1166666.66666667, 11018.0128205128, -15234.7596153846, -94236209.909812, 205.662393162393, 186.965811965812, 2.01923076923077, 175000000, 134615384.615385, 381410.256410256, -12310.4594017094, 17051.3194444444, -94236369.1248344, 205.662393162393, -186.965811965812, 1.62660256410256, 94230769.2307692, 0, 435256.41025641, -9740.5235042735, 6282.68696581197, -125644505.395044, -130.876068376068, 0, 2.89797008547009, -269230769.230769, -269230769.230769, -753846.153846154, 39485.3846153846, -39481.3461538462, 314102743.618776, 373.931623931624, 373.931623931624, -9.42307692307692, 323076923.076923, 0, -448717.948717949, 82055.7692307692, -64631.0897435898, 502579986.117162, 3141.02564102564, 0, -5.60897435897436, -269230769.230769, 269230769.230769, -628205.128205128, 14357.7777777778, 3577.92735042735, 314102643.992442, 373.931623931624, -373.931623931624, -7.8525641025641, -323076923.076923, 0, -556410.256410256, 5131.49572649573, -7182.02991452992, 251284509.686299, 448.717948717949, 0, -6.05769230769231;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(51) << -2280.98619284188, -934.82905982906, 1.79487179487179, 1.15384615384615, 0, 130.876068376068, -299.162611937045, -0.00701121794871795, -0.11109764957265, -2206.21716239316, 1308.76068376068, -1.68269230769231, 2.37179487179487, 0, 579.594017094017, -224.375186585983, 0.00981570512820513, 0.0918677216880342, -2206.20417574786, -1308.76068376068, -0.523504273504274, 0.913461538461539, 0, 579.594017094017, -224.37575539884, -0.00981570512820513, -0.102568028846154, -2280.98644337607, 934.82905982906, 1.90705128205128, 1.19123931623932, 0, 130.876068376068, -299.16254783636, 0.00701121794871795, -0.0811822783119658, 0.00287179487179487, -3739.31623931624, -2.91666666666667, -2.69230769230769, 0, -373.931623931624, 747.863888910427, -0.0280448717948718, 0.329037954059829, 12264.9816452991, 0, 2.24358974358974, -1.6025641025641, 0, 3141.02564102564, 1196.67435488371, 0, 0.683777510683761, 0.00303632478632479, 3739.31623931624, -1.57051282051282, -2.24358974358974, 0, -373.931623931624, 747.863533142431, 0.0280448717948718, 0.119646340811966, -3290.59357905983, 0, 0.747863247863248, -1.85897435897436, 0, -448.717948717949, 598.266151744738, 0, 0.0427406517094017;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(52) << -934.82905982906, -5459.40499626068, 1.23397435897436, 0, 1.15384615384615, 0, -0.00701121794871795, -299.18645007807, 0.0822742120726496, 2430.55555555556, -10245.7470769231, 12.900641025641, 0, 2.37179487179487, -186.965811965812, 0.0182291666666667, -224.435483060342, -0.12703999732906, -2430.55555555556, -10245.7340902778, 0, 0, 0.913461538461539, 186.965811965812, -0.0182291666666667, -224.436051873198, 0.142094017094017, 934.82905982906, -5459.40524679487, 0.673076923076923, 0, 1.19123931623932, 0, 0.00701121794871795, -299.186385977385, 0.0523554754273504, -3739.31623931624, 0.00287179487179487, 0.897435897435898, 0, -2.69230769230769, -373.931623931624, -0.0280448717948718, 747.863888910427, -0.329053098290598, 0, 22735.0671153846, -7.8525641025641, 0, -1.6025641025641, 0, 0, 1196.75288052473, -0.538520432692308, 3739.31623931624, 0.00303632478632479, -6.13247863247863, 0, -2.24358974358974, 373.931623931624, 0.0280448717948718, 747.863533142431, 0.0298685363247863, 0, 8675.21838675214, -1.72008547008547, 0, -1.85897435897436, 0, 0, 598.355895334482, -0.0598419604700855;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(53) << 1.27136752136752, 0.953525641025641, -1720.09612713675, 523.504273504273, 0, 4.03846153846154, -0.111101575854701, 0.082272108707265, -1047.0219269852, -3.81410256410256, 10.8253205128205, -2767.15576068376, 1196.5811965812, -747.863247863248, 8.30128205128205, 0.0918517361111111, -0.127055562232906, -785.27638365396, -1.38354700854701, 0.0186965811965812, -2767.11583600427, 1196.5811965812, 747.863247863248, 3.19711538461538, -0.102574479166667, 0.142094157318376, -785.27771081179, 1.53311965811966, 0.31784188034188, -1720.09602617521, 523.504273504273, 0, 4.16933760683761, -0.0811850827991453, 0.0523528111645299, -1047.02177741179, -0.897435897435898, 2.91666666666667, 0.0145160256410256, -1495.7264957265, -1495.7264957265, -9.42307692307692, 0.329053098290598, -0.329037954059829, 2617.52286335673, 2.24358974358974, -7.8525641025641, 7777.84490598291, 3589.74358974359, 0, -5.60897435897436, 0.683777510683761, -0.538520432692308, 4188.09525406472, -0.598290598290598, -5.90811965811966, 0.00904166666666667, -1495.7264957265, 1495.7264957265, -7.8525641025641, 0.119653632478632, 0.029870219017094, 2617.52203318747, 1.64529914529915, -1.27136752136752, 1196.59528632479, -1794.87179487179, 0, -6.50641025641026, 0.0427473824786325, -0.0598385950854701, 2094.02661192904;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(54) << -1310257051.02564, -224358974.358974, 0, 34615.3846153846, 0, 0, -5459.4043792735, -934.82905982906, 0, -1310258268.84615, 224358974.358974, -978205.128205128, -10256.4102564103, 0, 0, -5459.40945352564, 934.82905982906, -4.0758547008547, -2458975022.05128, -583333333.333333, -296153.846153846, 43589.7435897436, 0, -134615384.615385, -10245.729258547, -2430.55555555556, -1.23397435897436, -2458975042.69231, 583333333.333333, 161538.461538462, 16666.6666666667, 0, 134615384.615385, -10245.7293445513, 2430.55555555556, 0.673076923076923, 2082052412.82051, 0, 305128.205128205, -158974.358974359, 0, 0, 8675.21838675214, 0, 1.27136752136752, 728.717948717949, 897435897.435898, -143589.743589744, -179487.17948718, 0, -269230769.230769, 0.00303632478632479, 3739.31623931624, -0.598290598290598, 5456411733.33333, 0, -89743.5897435897, -15384.6153846154, 0, 0, 22735.0488888889, 0, -0.373931623931624, 509.74358974359, -897435897.435898, 1041025.64102564, -143589.743589744, 0, 269230769.230769, 0.00212393162393162, -3739.31623931624, 4.33760683760684;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(55) << -224358974.358974, -547436538.205128, 700000, 0, 34615.3846153846, -94230769.2307692, -934.82905982906, -2280.9855758547, 2.91666666666667, 224358974.358974, -547437756.025641, 1498717.94871795, 0, -10256.4102564103, -94230769.2307692, 934.82905982906, -2280.99065010684, 6.24465811965812, -314102564.102564, -529487842.564103, 430769.230769231, 0, 43589.7435897436, -94230769.2307692, -1308.76068376068, -2206.19934401709, 1.79487179487179, 314102564.102564, -529487863.205128, 457692.307692308, 0, 16666.6666666667, -94230769.2307692, 1308.76068376068, -2206.19943002137, 1.90705128205128, 0, -789742458.974359, -394871.794871795, 0, -158974.358974359, 323076923.076923, 0, -3290.59357905983, -1.64529914529915, 897435897.435898, 728.717948717949, -1417948.71794872, 0, -179487.17948718, 269230769.230769, 3739.31623931624, 0.00303632478632479, -5.90811965811966, 0, 2943591220.51282, -179487.179487179, 0, -15384.6153846154, 323076923.076923, 0, 12264.9634188034, -0.747863247863248, -897435897.435898, 509.74358974359, -1094871.79487179, 0, -143589.743589744, 269230769.230769, -3739.31623931624, 0.00212393162393162, -4.56196581196581;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(56) << 4487.17948717949, 700000, -412822352.564103, 0, -62820512.8205128, 121153.846153846, 0.0186965811965812, 2.91666666666667, -1720.09313568376, -955769.230769231, 1462820.51282051, -412826088.589744, 0, -62820512.8205128, -35897.4358974359, -3.98237179487179, 6.09508547008547, -1720.10870245727, -166025.641025641, 367948.717948718, -664104390.384615, -89743589.7435897, -62820512.8205128, 152564.102564103, -0.691773504273504, 1.53311965811966, -2767.10162660256, 76282.0512820513, 341025.641025641, -664104540.25641, 89743589.7435897, -62820512.8205128, 58333.3333333333, 0.31784188034188, 1.42094017094017, -2767.10225106838, 412820.512820513, -179487.17948718, 287182868.717949, 0, 215384615.384615, -556410.256410256, 1.72008547008547, -0.747863247863248, 1196.59528632479, -376923.076923077, -1471794.87179487, 2170, -179487179.48718, 179487179.48718, -628205.128205128, -1.57051282051282, -6.13247863247863, 0.00904166666666667, -89743.5897435897, -179487.179487179, 1866670859.48718, 0, 215384615.384615, -53846.1538461538, -0.373931623931624, -0.747863247863248, 7777.79524786325, 1094871.79487179, -1041025.64102564, 1473.58974358974, 179487179.48718, 179487179.48718, -502564.102564103, 4.56196581196581, -4.33760683760684, 0.00613995726495727;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(57) << 91025.641025641, 0, 0, -35908365.6997842, -1869.65811965812, -897.435897435898, 0.667735042735043, 0, 0, 135897.435897436, 0, 0, -35908358.017625, 1869.65811965812, 2684.15598290598, 0.480769230769231, 0, 0, 115384.615384615, 0, 89743589.7435897, -26943574.7918504, -4861.11111111111, 894.967948717949, 0.844017094017094, 0, -747.863247863248, 124358.974358974, 0, -89743589.7435897, -26943576.9074071, 4861.11111111111, -2690.96153846154, 0.657051282051282, 0, 747.863247863248, -128205.128205128, 0, 0, 71812250.1803633, 0, 7182.02991452992, -1.85897435897436, 0, 0, -179487.17948718, 0, 179487179.48718, 89743623.9804316, 7478.63247863248, 14357.7777777778, -2.24358974358974, 0, -1495.7264957265, -15384.6153846154, 0, 0, 143635260.354188, 0, -7180.23504273504, -0.192307692307692, 0, 0, -143589.743589744, 0, -179487179.48718, 89743612.0555299, -7478.63247863248, -14350.2991452991, -1.7948717948718, 0, 1495.7264957265;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(58) << 0, 91025.641025641, 62820512.8205128, -1869.65811965812, -35902008.8621774, -4609.55128205128, 0, 0.667735042735043, -523.504273504273, 0, 135897.435897436, 62820512.8205128, 1869.65811965812, -35902001.1800182, -8192.63888888889, 0, 0.480769230769231, -523.504273504273, 0, 115384.615384615, 116666666.666667, -2617.52136752137, -26927495.7320214, 3336.92307692308, 0, 0.844017094017094, -299.145299145299, 0, 124358.974358974, 116666666.666667, 2617.52136752137, -26927497.847578, -2047.46794871795, 0, 0.657051282051282, -299.145299145299, 0, -128205.128205128, -215384615.384615, 0, 71788318.5564316, -5131.49572649573, 0, -1.85897435897436, 1794.87179487179, 0, -179487.17948718, -179487179.48718, 7478.63247863248, 89743623.9804316, 3577.92735042735, 0, -2.24358974358974, 1495.7264957265, 0, -15384.6153846154, 215384615.384615, 0, 143614320.183248, 61536.9658119658, 0, -0.192307692307692, 3589.74358974359, 0, -143589.743589744, -179487179.48718, -7478.63247863248, 89743612.0555299, 14349.8504273504, 0, -1.7948717948718, 1495.7264957265;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(59) << 0, 94230769.2307692, 318589.743589744, -897.398504273504, -4609.55128205128, -125644491.479006, 0, -130.876068376068, 2.33707264957265, 0, 94230769.2307692, 475641.025641026, 2684.34294871795, -8192.93803418804, -125644473.561422, 0, -130.876068376068, 1.68269230769231, 134615384.615385, 175000000, 403846.153846154, 896.052350427351, 3336.39957264957, -94236318.3912874, -186.965811965812, 205.662393162393, 2.95405982905983, -134615384.615385, 175000000, 435256.41025641, -2691.67200854701, -2048.44017094017, -94236323.3284338, 186.965811965812, 205.662393162393, 2.29967948717949, 0, -323076923.076923, -448717.948717949, 7182.92735042735, -5129.70085470086, 251284509.686299, 0, 448.717948717949, -6.50641025641026, 269230769.230769, -269230769.230769, -628205.128205128, 14355.8333333333, 3577.47863247863, 314102643.992442, -373.931623931624, 373.931623931624, -7.8525641025641, 0, 323076923.076923, -53846.1538461538, -7180.23504273504, 61536.9658119658, 502579767.043487, 0, 3141.02564102564, -0.673076923076923, -269230769.230769, -269230769.230769, -502564.102564103, -14349.8504273504, 14350.2991452991, 314102616.166126, 373.931623931624, 373.931623931624, -6.28205128205128;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(60) << -5459.4043792735, -934.82905982906, 0, 0.902777777777778, 0, 0, -299.186336291392, -0.00701121794871795, -0.00747863247863248, -5459.40945352564, 934.82905982906, -4.0758547008547, 1.08974358974359, 0, 0, -299.186272226884, 0.00701121794871795, 0.022405328525641, -10245.729258547, -2430.55555555556, -1.23397435897436, 1.14316239316239, 0, 186.965811965812, -224.435870747217, -0.0182291666666667, 0.00746937767094017, -10245.7293445513, 2430.55555555556, 0.673076923076923, 1.10576923076923, 0, -186.965811965812, -224.435888376067, 0.0182291666666667, -0.0224308493589744, 8675.21838675214, 0, 1.27136752136752, -1.73076923076923, 0, 0, 598.355895334482, 0, 0.0598385950854701, 0.00303632478632479, 3739.31623931624, -0.598290598290598, -2.24358974358974, 0, 373.931623931624, 747.863533142431, 0.0280448717948718, 0.119653632478632, 22735.0488888889, 0, -0.373931623931624, -0.192307692307692, 0, 0, 1196.75209833675, 0, -0.0598318643162393, 0.00212393162393162, -3739.31623931624, 4.33760683760684, -1.7948717948718, 0, -373.931623931624, 747.863433776613, -0.0280448717948718, -0.119625587606838;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(61) << -934.82905982906, -2280.9855758547, 2.91666666666667, 0, 0.902777777777778, 130.876068376068, -0.00701121794871795, -299.162498150366, -0.0384396634615385, 934.82905982906, -2280.99065010684, 6.24465811965812, 0, 1.08974358974359, 130.876068376068, 0.00701121794871795, -299.162434085859, -0.0683292334401709, -1308.76068376068, -2206.19934401709, 1.79487179487179, 0, 1.14316239316239, 579.594017094017, -0.00981570512820513, -224.375574272858, 0.0277912393162393, 1308.76068376068, -2206.19943002137, 1.90705128205128, 0, 1.10576923076923, 579.594017094017, 0.00981570512820513, -224.375591901708, -0.0170797142094017, 0, -3290.59357905983, -1.64529914529915, 0, -1.73076923076923, -448.717948717949, 0, 598.266151744738, -0.0427473824786325, 3739.31623931624, 0.00303632478632479, -5.90811965811966, 0, -2.24358974358974, -373.931623931624, 0.0280448717948718, 747.863533142431, 0.029870219017094, 0, 12264.9634188034, -0.747863247863248, 0, -0.192307692307692, 3141.02564102564, 0, 1196.67357269573, 0.512814903846154, -3739.31623931624, 0.00212393162393162, -4.56196581196581, 0, -1.7948717948718, -373.931623931624, -0.0280448717948718, 747.863433776613, 0.11962390491453;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(62) << 0.0186965811965812, 2.91666666666667, -1720.09313568376, 0, 523.504273504273, 3.15972222222222, -0.00747849225427351, -0.0384396634615385, -1047.02166147131, -3.98237179487179, 6.09508547008547, -1720.10870245727, 0, 523.504273504273, 3.81410256410256, 0.0224060296474359, -0.0683303552350427, -1047.02151201541, -0.691773504273504, 1.53311965811966, -2767.10162660256, 747.863247863248, 1196.5811965812, 4.00106837606838, 0.00747344417735043, 0.0277892761752137, -785.277288162484, 0.31784188034188, 1.42094017094017, -2767.10225106838, -747.863247863248, 1196.5811965812, 3.87019230769231, -0.0224335136217949, -0.017083360042735, -785.277329299647, 1.72008547008547, -0.747863247863248, 1196.59528632479, 0, -1794.87179487179, -6.05769230769231, 0.0598419604700855, -0.0427406517094017, 2094.02661192904, -1.57051282051282, -6.13247863247863, 0.00904166666666667, 1495.7264957265, -1495.7264957265, -7.8525641025641, 0.119646340811966, 0.0298685363247863, 2617.52203318747, -0.373931623931624, -0.747863247863248, 7777.79524786325, 0, 3589.74358974359, -0.673076923076923, -0.0598318643162393, 0.512814903846154, 4188.09342890595, 4.56196581196581, -4.33760683760684, 0.00613995726495727, -1495.7264957265, -1495.7264957265, -6.28205128205128, -0.11962390491453, 0.119625587606838, 2617.5218013281;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(63) << -529487842.564103, -314102564.102564, -430769.230769231, 43589.7435897436, 0, 94230769.2307692, -2206.19934401709, -1308.76068376068, -1.79487179487179, -547437756.025641, 224358974.358974, -1498717.94871795, -10256.4102564103, 0, 94230769.2307692, -2280.99065010684, 934.82905982906, -6.24465811965812, -547436538.205128, -224358974.358974, -700000, 34615.3846153846, 0, 94230769.2307692, -2280.9855758547, -934.82905982906, -2.91666666666667, -529487863.205128, 314102564.102564, -457692.307692308, 16666.6666666667, 0, 94230769.2307692, -2206.19943002137, 1308.76068376068, -1.90705128205128, 728.717948717949, 897435897.435898, 1417948.71794872, -179487.17948718, 0, -269230769.230769, 0.00303632478632479, 3739.31623931624, 5.90811965811966, -789742458.974359, 0, 394871.794871795, -158974.358974359, 0, -323076923.076923, -3290.59357905983, 0, 1.64529914529915, 509.74358974359, -897435897.435898, 1094871.79487179, -143589.743589744, 0, -269230769.230769, 0.00212393162393162, -3739.31623931624, 4.56196581196581, 2943591220.51282, 0, 179487.179487179, -15384.6153846154, 0, -323076923.076923, 12264.9634188034, 0, 0.747863247863248;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(64) << -583333333.333333, -2458975022.05128, 296153.846153846, 0, 43589.7435897436, 134615384.615385, -2430.55555555556, -10245.729258547, 1.23397435897436, 224358974.358974, -1310258268.84615, 978205.128205128, 0, -10256.4102564103, 0, 934.82905982906, -5459.40945352564, 4.0758547008547, -224358974.358974, -1310257051.02564, 0, 0, 34615.3846153846, 0, -934.82905982906, -5459.4043792735, 0, 583333333.333333, -2458975042.69231, -161538.461538462, 0, 16666.6666666667, -134615384.615385, 2430.55555555556, -10245.7293445513, -0.673076923076923, 897435897.435898, 728.717948717949, 143589.743589744, 0, -179487.17948718, 269230769.230769, 3739.31623931624, 0.00303632478632479, 0.598290598290598, 0, 2082052412.82051, -305128.205128205, 0, -158974.358974359, 0, 0, 8675.21838675214, -1.27136752136752, -897435897.435898, 509.74358974359, -1041025.64102564, 0, -143589.743589744, -269230769.230769, -3739.31623931624, 0.00212393162393162, -4.33760683760684, 0, 5456411733.33333, 89743.5897435897, 0, -15384.6153846154, 0, 0, 22735.0488888889, 0.373931623931624;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(65) << -367948.717948718, 166025.641025641, -664104390.384615, 62820512.8205128, 89743589.7435897, 152564.102564103, -1.53311965811966, 0.691773504273504, -2767.10162660256, -1462820.51282051, 955769.230769231, -412826088.589744, 62820512.8205128, 0, -35897.4358974359, -6.09508547008547, 3.98237179487179, -1720.10870245727, -700000, -4487.17948717949, -412822352.564103, 62820512.8205128, 0, 121153.846153846, -2.91666666666667, -0.0186965811965812, -1720.09313568376, -341025.641025641, -76282.0512820513, -664104540.25641, 62820512.8205128, -89743589.7435897, 58333.3333333333, -1.42094017094017, -0.31784188034188, -2767.10225106838, 1471794.87179487, 376923.076923077, 2170, -179487179.48718, 179487179.48718, -628205.128205128, 6.13247863247863, 1.57051282051282, 0.00904166666666667, 179487.17948718, -412820.512820513, 287182868.717949, -215384615.384615, 0, -556410.256410256, 0.747863247863248, -1.72008547008547, 1196.59528632479, 1041025.64102564, -1094871.79487179, 1473.58974358974, -179487179.48718, -179487179.48718, -502564.102564103, 4.33760683760684, -4.56196581196581, 0.00613995726495727, 179487.179487179, 89743.5897435897, 1866670859.48718, -215384615.384615, 0, -53846.1538461538, 0.747863247863248, 0.373931623931624, 7777.79524786325;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(66) << 115384.615384615, 0, -116666666.666667, -26927495.7320214, -2617.52136752137, -3336.92307692308, 0.844017094017094, 0, 299.145299145299, 135897.435897436, 0, -62820512.8205128, -35902001.1800182, 1869.65811965812, 8192.63888888889, 0.480769230769231, 0, 523.504273504273, 91025.641025641, 0, -62820512.8205128, -35902008.8621774, -1869.65811965812, 4609.55128205128, 0.667735042735043, 0, 523.504273504273, 124358.974358974, 0, -116666666.666667, -26927497.847578, 2617.52136752137, 2047.46794871795, 0.657051282051282, 0, 299.145299145299, -179487.17948718, 0, 179487179.48718, 89743623.9804316, 7478.63247863248, -3577.92735042735, -2.24358974358974, 0, -1495.7264957265, -128205.128205128, 0, 215384615.384615, 71788318.5564316, 0, 5131.49572649573, -1.85897435897436, 0, -1794.87179487179, -143589.743589744, 0, 179487179.48718, 89743612.0555299, -7478.63247863248, -14349.8504273504, -1.7948717948718, 0, -1495.7264957265, -15384.6153846154, 0, -215384615.384615, 143614320.183248, 0, -61536.9658119658, -0.192307692307692, 0, -3589.74358974359;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(67) << 0, 115384.615384615, -89743589.7435897, -4861.11111111111, -26943574.7918504, -894.967948717949, 0, 0.844017094017094, 747.863247863248, 0, 135897.435897436, 0, 1869.65811965812, -35908358.017625, -2684.15598290598, 0, 0.480769230769231, 0, 0, 91025.641025641, 0, -1869.65811965812, -35908365.6997842, 897.435897435898, 0, 0.667735042735043, 0, 0, 124358.974358974, 89743589.7435897, 4861.11111111111, -26943576.9074071, 2690.96153846154, 0, 0.657051282051282, -747.863247863248, 0, -179487.17948718, -179487179.48718, 7478.63247863248, 89743623.9804316, -14357.7777777778, 0, -2.24358974358974, 1495.7264957265, 0, -128205.128205128, 0, 0, 71812250.1803633, -7182.02991452992, 0, -1.85897435897436, 0, 0, -143589.743589744, 179487179.48718, -7478.63247863248, 89743612.0555299, 14350.2991452991, 0, -1.7948717948718, -1495.7264957265, 0, -15384.6153846154, 0, 0, 143635260.354188, 7180.23504273504, 0, -0.192307692307692, 0;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(68) << -175000000, -134615384.615385, 403846.153846154, -3336.39957264957, -896.052350427351, -94236318.3912874, -205.662393162393, 186.965811965812, 2.95405982905983, -94230769.2307692, 0, 475641.025641026, 8192.93803418804, -2684.34294871795, -125644473.561422, 130.876068376068, 0, 1.68269230769231, -94230769.2307692, 0, 318589.743589744, 4609.55128205128, 897.398504273504, -125644491.479006, 130.876068376068, 0, 2.33707264957265, -175000000, 134615384.615385, 435256.41025641, 2048.44017094017, 2691.67200854701, -94236323.3284338, -205.662393162393, -186.965811965812, 2.29967948717949, 269230769.230769, -269230769.230769, -628205.128205128, -3577.47863247863, -14355.8333333333, 314102643.992442, -373.931623931624, 373.931623931624, -7.8525641025641, 323076923.076923, 0, -448717.948717949, 5129.70085470086, -7182.92735042735, 251284509.686299, -448.717948717949, 0, -6.50641025641026, 269230769.230769, 269230769.230769, -502564.102564103, -14350.2991452991, 14349.8504273504, 314102616.166126, -373.931623931624, -373.931623931624, -6.28205128205128, -323076923.076923, 0, -53846.1538461538, -61536.9658119658, 7180.23504273504, 502579767.043487, -3141.02564102564, 0, -0.673076923076923;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(69) << -2206.19934401709, -1308.76068376068, -1.79487179487179, 1.14316239316239, 0, -579.594017094017, -224.375574272858, -0.00981570512820513, -0.0277912393162393, -2280.99065010684, 934.82905982906, -6.24465811965812, 1.08974358974359, 0, -130.876068376068, -299.162434085859, 0.00701121794871795, 0.0683292334401709, -2280.9855758547, -934.82905982906, -2.91666666666667, 0.902777777777778, 0, -130.876068376068, -299.162498150366, -0.00701121794871795, 0.0384396634615385, -2206.19943002137, 1308.76068376068, -1.90705128205128, 1.10576923076923, 0, -579.594017094017, -224.375591901708, 0.00981570512820513, 0.0170797142094017, 0.00303632478632479, 3739.31623931624, 5.90811965811966, -2.24358974358974, 0, 373.931623931624, 747.863533142431, 0.0280448717948718, -0.029870219017094, -3290.59357905983, 0, 1.64529914529915, -1.73076923076923, 0, 448.717948717949, 598.266151744738, 0, 0.0427473824786325, 0.00212393162393162, -3739.31623931624, 4.56196581196581, -1.7948717948718, 0, 373.931623931624, 747.863433776613, -0.0280448717948718, -0.11962390491453, 12264.9634188034, 0, 0.747863247863248, -0.192307692307692, 0, -3141.02564102564, 1196.67357269573, 0, -0.512814903846154;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(70) << -2430.55555555556, -10245.729258547, 1.23397435897436, 0, 1.14316239316239, -186.965811965812, -0.0182291666666667, -224.435870747217, -0.00746937767094017, 934.82905982906, -5459.40945352564, 4.0758547008547, 0, 1.08974358974359, 0, 0.00701121794871795, -299.186272226884, -0.022405328525641, -934.82905982906, -5459.4043792735, 0, 0, 0.902777777777778, 0, -0.00701121794871795, -299.186336291392, 0.00747863247863248, 2430.55555555556, -10245.7293445513, -0.673076923076923, 0, 1.10576923076923, 186.965811965812, 0.0182291666666667, -224.435888376067, 0.0224308493589744, 3739.31623931624, 0.00303632478632479, 0.598290598290598, 0, -2.24358974358974, -373.931623931624, 0.0280448717948718, 747.863533142431, -0.119653632478632, 0, 8675.21838675214, -1.27136752136752, 0, -1.73076923076923, 0, 0, 598.355895334482, -0.0598385950854701, -3739.31623931624, 0.00212393162393162, -4.33760683760684, 0, -1.7948717948718, 373.931623931624, -0.0280448717948718, 747.863433776613, 0.119625587606838, 0, 22735.0488888889, 0.373931623931624, 0, -0.192307692307692, 0, 0, 1196.75209833675, 0.0598318643162393;
    //Expected_JacobianK_SmallDispNoVelWithDamping.row(71) << -1.53311965811966, 0.691773504273504, -2767.10162660256, -1196.5811965812, -747.863247863248, 4.00106837606838, -0.0277892761752137, -0.00747344417735043, -785.277288162484, -6.09508547008547, 3.98237179487179, -1720.10870245727, -523.504273504273, 0, 3.81410256410256, 0.0683303552350427, -0.0224060296474359, -1047.02151201541, -2.91666666666667, -0.0186965811965812, -1720.09313568376, -523.504273504273, 0, 3.15972222222222, 0.0384396634615385, 0.00747849225427351, -1047.02166147131, -1.42094017094017, -0.31784188034188, -2767.10225106838, -1196.5811965812, 747.863247863248, 3.87019230769231, 0.017083360042735, 0.0224335136217949, -785.277329299647, 6.13247863247863, 1.57051282051282, 0.00904166666666667, 1495.7264957265, -1495.7264957265, -7.8525641025641, -0.0298685363247863, -0.119646340811966, 2617.52203318747, 0.747863247863248, -1.72008547008547, 1196.59528632479, 1794.87179487179, 0, -6.05769230769231, 0.0427406517094017, -0.0598419604700855, 2094.02661192904, 4.33760683760684, -4.56196581196581, 0.00613995726495727, 1495.7264957265, 1495.7264957265, -6.28205128205128, -0.119625587606838, 0.11962390491453, 2617.5218013281, 0.747863247863248, 0.373931623931624, 7777.79524786325, -3589.74358974359, 0, -0.673076923076923, -0.512814903846154, 0.0598318643162393, 4188.09342890595;

    //ChMatrixNM<double, 72, 72> Expected_JacobianR_SmallDispNoVelWithDamping;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(0) << 21000000, 9535256.41025641, -6147.4358974359, 0, 0, -807692.307692308, 87.5, 39.730235042735, -0.0256143162393162, 10320512.8205128, -336538.461538461, 6842.94871794872, 0, 0, 538461.538461538, 43.0021367521368, -1.40224358974359, 0.0285122863247863, 9288461.53846154, 3926282.05128205, 426.282051282051, 0, 0, 201923.076923077, 38.7019230769231, 16.3595085470085, 0.00177617521367521, 7852564.1025641, 336538.461538461, -493.589743589744, 0, 0, 201923.076923077, 32.7190170940171, 1.40224358974359, -0.00205662393162393, -24589743.5897436, -3141025.64102564, -44.8717948717949, 0, 0, -1346153.84615385, -102.457264957265, -13.0876068376068, -0.000186965811965812, -5474358.97435897, -2243589.74358974, 3051.28205128205, 0, 0, 942307.692307692, -22.8098290598291, -9.3482905982906, 0.0127136752136752, -13102564.1025641, -2243589.74358974, 44.8717948717949, 0, 0, 0, -54.5940170940171, -9.3482905982906, 0.000186965811965812, -5294871.7948718, -5833333.33333333, -3679.48717948718, 0, 0, -1750000, -22.0619658119658, -24.3055555555556, -0.0153311965811966;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(1) << 9535256.41025641, 21000000, -6663.46153846154, 0, 0, -807692.307692308, 39.730235042735, 87.5, -0.0277644230769231, 336538.461538461, 7852564.1025641, -6483.97435897436, 0, 0, 201923.076923077, 1.40224358974359, 32.7190170940171, -0.0270165598290598, 3926282.05128205, 9288461.53846154, -628.205128205128, 0, 0, 201923.076923077, 16.3595085470085, 38.7019230769231, -0.00261752136752137, -336538.461538461, 10320512.8205128, -493.589743589744, 0, 0, 538461.538461538, -1.40224358974359, 43.0021367521368, -0.00205662393162393, -5833333.33333333, -5294871.7948718, 3320.51282051282, 0, 0, -1750000, -24.3055555555556, -22.0619658119658, 0.0138354700854701, -2243589.74358974, -13102564.1025641, 2288.46153846154, 0, 0, 0, -9.3482905982906, -54.5940170940171, 0.00953525641025641, -2243589.74358974, -5474358.97435897, 7000, 0, 0, 942307.692307692, -9.3482905982906, -22.8098290598291, 0.0291666666666667, -3141025.64102564, -24589743.5897436, 1660.25641025641, 0, 0, -1346153.84615385, -13.0876068376068, -102.457264957265, 0.00691773504273504;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(2) << -6147.4358974359, -6663.46153846154, 9333352.2474359, -538461.538461539, -538461.538461539, 336.538461538462, -0.0256143162393162, -0.0277644230769231, 38.8889676976496, 6147.4358974359, -6663.46153846154, 4038498.08653846, 358974.358974359, 134615.384615385, 336.538461538462, 0.0256143162393162, -0.0277644230769231, 16.8270753605769, 628.205128205128, -426.282051282051, 4128215.48012821, 134615.384615385, 134615.384615385, -673.076923076923, 0.00261752136752137, -0.00177617521367521, 17.2008978338675, -628.205128205128, -426.282051282051, 4038471.17307692, 134615.384615385, 358974.358974359, -673.076923076923, -0.00261752136752137, -0.00177617521367521, 16.8269632211538, 0, 1256.41025641026, -6641059.77948718, -897435.897435898, -1166666.66666667, 2724.35897435897, 0, 0.00523504273504274, -27.6710824145299, 4307.69230769231, 2961.53846153846, -4128222.81666667, 628205.128205128, 0, 2884.61538461538, 0.0179487179487179, 0.0123397435897436, -17.2009284027778, 0, 7000, -4128217.11794872, 0, 628205.128205128, 2275.64102564103, 0, 0.0291666666666667, -17.2009046581197, -4307.69230769231, 2961.53846153846, -6641037.27307692, -1166666.66666667, -897435.897435898, 2884.61538461538, -0.0179487179487179, 0.0123397435897436, -27.6709886378205;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(3) << 0, 0, -538461.538461539, 269405.769230769, 79.4604700854701, -55.1794337606838, 0, 0, -8.97435897435897, 0, 0, -358974.358974359, 89829.594017094, -2.80448717948718, -27.507077991453, 0, 0, 2.99145299145299, 0, 0, -134615.384615385, 134692.788461538, 32.7190170940171, 26.2856036324786, 0, 0, 1.12179487179487, 0, 0, 134615.384615385, 89809.0277777778, 2.80448717948718, 43.5856303418803, 0, 0, 2.24358974358974, 0, 0, 897435.897435898, -269435.683760684, -26.1752136752137, -170.513194444444, 0, 0, -7.47863247863248, 0, 0, -628205.128205128, -359019.978632479, -18.6965811965812, -133.307905982906, 0, 0, 5.23504273504273, 0, 0, 0, -359083.547008547, -18.6965811965812, -8.97398504273504, 0, 0, 0, 0, 0, 628205.128205128, -269274.893162393, -48.6111111111111, -33.3639957264957, 0, 0, -11.965811965812;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(4) << 0, 0, -538461.538461539, 79.4604700854701, 269405.769230769, -34.6709134615385, 0, 0, -8.97435897435897, 0, 0, 134615.384615385, 2.80448717948718, 89809.0277777778, 46.099813034188, 0, 0, 2.24358974358974, 0, 0, -134615.384615385, 32.7190170940171, 134692.788461538, -26.2872863247863, 0, 0, 1.12179487179487, 0, 0, -358974.358974359, -2.80448717948718, 89829.594017094, -17.3118055555556, 0, 0, 2.99145299145299, 0, 0, 628205.128205128, -48.6111111111111, -269274.893162393, 123.104594017094, 0, 0, -11.965811965812, 0, 0, 0, -18.6965811965812, -359083.547008547, 98.7370192307692, 0, 0, 0, 0, 0, -628205.128205128, -18.6965811965812, -359019.978632479, -46.0955128205128, 0, 0, 5.23504273504273, 0, 0, 897435.897435898, -26.1752136752137, -269435.683760684, -8.96052350427351, 0, 0, -7.47863247863248;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(5) << -807692.307692308, -807692.307692308, 336.538461538462, -55.1794337606838, -34.6709134615385, 942385.660841378, -7.8525641025641, -7.8525641025641, 0.00420673076923077, -538461.538461538, 201923.076923077, -1826.92307692308, -27.5128739316239, 46.0983173076923, 314136.086202003, 0.747863247863248, 1.96314102564103, -0.00480769230769231, -201923.076923077, -201923.076923077, -673.076923076923, 26.2872863247863, -26.2856036324786, 471188.345812762, 0.280448717948718, 0.280448717948718, -0.00841346153846154, 201923.076923077, -538461.538461538, -977.564102564103, 43.5845085470086, -17.3112446581197, 314136.266746955, 1.96314102564103, 0.747863247863248, -0.00968215811965812, 1346153.84615385, 942307.692307692, 32.0512820512821, -170.512820512821, 123.087393162393, -942363.409686197, -1.86965811965812, -5.79594017094017, 0.0228365384615385, -942307.692307692, 0, 1153.84615384615, -133.297435897436, 98.7426282051282, -1256444.98689954, 1.30876068376068, 0, 0.0288461538461538, 0, -942307.692307692, 865.384615384615, -8.97435897435897, -46.0955128205128, -1256444.80480077, 0, 1.30876068376068, 0.0225694444444444, 942307.692307692, 1346153.84615385, 1089.74358974359, -33.3692307692308, -8.94967948717949, -942363.119755053, -5.79594017094017, -1.86965811965812, 0.0285790598290598;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(6) << 87.5, 39.730235042735, -0.0256143162393162, 0, 0, -7.8525641025641, 2.24424599358974, 0.000297976762820513, -0.000459593816773504, 43.0021367521368, -1.40224358974359, 0.0285122863247863, 0, 0, -0.747863247863248, 0.748185763888889, -1.05168269230769E-05, -0.000229487012553419, 38.7019230769231, 16.3595085470085, 0.00177617521367521, 0, 0, -0.280448717948718, 1.12208513621795, 0.000122696314102564, 0.000219030415331197, 32.7190170940171, 1.40224358974359, -0.00205662393162393, 0, 0, 1.96314102564103, 0.748108640491453, 1.05168269230769E-05, 0.000363232438568376, -102.457264957265, -13.0876068376068, -0.000186965811965812, 0, 0, 1.86965811965812, -2.24435817307692, -9.81570512820513E-05, -0.00142094157318376, -22.8098290598291, -9.3482905982906, 0.0127136752136752, 0, 0, -1.30876068376068, -2.99162406517094, -7.01121794871795E-05, -0.00111101575854701, -54.5940170940171, -9.3482905982906, 0.000186965811965812, 0, 0, 0, -2.9918624465812, -7.01121794871795E-05, -0.000074784922542735, -22.0619658119658, -24.3055555555556, -0.0153311965811966, 0, 0, -2.05662393162393, -2.24375520833333, -0.000182291666666667, -0.000277892761752137;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(7) << 39.730235042735, 87.5, -0.0277644230769231, 0, 0, -7.8525641025641, 0.000297976762820513, 2.24424599358974, -0.000288669771634615, 1.40224358974359, 32.7190170940171, -0.0270165598290598, 0, 0, 1.96314102564103, 1.05168269230769E-05, 0.748108640491453, 0.000384412760416667, 16.3595085470085, 38.7019230769231, -0.00261752136752137, 0, 0, -0.280448717948718, 0.000122696314102564, 1.12208513621795, -0.00021903672542735, -1.40224358974359, 43.0021367521368, -0.00205662393162393, 0, 0, -0.747863247863248, -1.05168269230769E-05, 0.748185763888889, -0.000144246193910256, -24.3055555555556, -22.0619658119658, 0.0138354700854701, 0, 0, -2.05662393162393, -0.000182291666666667, -2.24375520833333, 0.00102574479166667, -9.3482905982906, -54.5940170940171, 0.00953525641025641, 0, 0, 0, -7.01121794871795E-05, -2.9918624465812, 0.00082272108707265, -9.3482905982906, -22.8098290598291, 0.0291666666666667, 0, 0, -1.30876068376068, -7.01121794871795E-05, -2.99162406517094, -0.000384396634615385, -13.0876068376068, -102.457264957265, 0.00691773504273504, 0, 0, 1.86965811965812, -9.81570512820513E-05, -2.24435817307692, -7.47344417735043E-05;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(8) << -0.0256143162393162, -0.0277644230769231, 38.8889676976496, -8.97435897435897, -8.97435897435897, 0.00420673076923077, -0.000459593816773504, -0.000288669771634615, 7.85285735814092, 0.0256143162393162, -0.0277644230769231, 16.8270753605769, -2.99145299145299, 2.24358974358974, -0.0138221153846154, -0.00022950874732906, 0.000384407151442308, 2.61764647015922, 0.00261752136752137, -0.00177617521367521, 17.2008978338675, -1.12179487179487, -1.12179487179487, -0.00841346153846154, 0.00021903672542735, -0.000219030415331197, 3.92641187354287, -0.00261752136752137, -0.00177617521367521, 16.8269632211538, 2.24358974358974, -2.99145299145299, -0.0109508547008547, 0.000363228231837607, -0.000144244090544872, 2.61764797572843, 0, 0.00523504273504274, -27.6710824145299, 7.47863247863248, 2.99145299145299, 0.0116185897435897, -0.00142094017094017, 0.00102568028846154, -7.85277476246284, 0.0179487179487179, 0.0123397435897436, -17.2009284027778, -5.23504273504273, 0, 0.0216346153846154, -0.0011109764957265, 0.000822742120726496, -10.4702172156525, 0, 0.0291666666666667, -17.2009046581197, 0, -5.23504273504273, 0.0166933760683761, -7.47863247863248E-05, -0.000384396634615385, -10.4702156983804, -0.0179487179487179, 0.0123397435897436, -27.6709886378205, 2.99145299145299, 7.47863247863248, 0.0211004273504274, -0.000277912393162393, -7.46937767094017E-05, -7.8527723472296;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(9) << 10320512.8205128, 336538.461538461, 6147.4358974359, 0, 0, -538461.538461538, 43.0021367521368, 1.40224358974359, 0.0256143162393162, 21000000, -9535256.41025641, 41259.6153846154, 0, 0, 807692.307692308, 87.5, -39.730235042735, 0.171915064102564, 7852564.1025641, -336538.461538461, 6663.46153846154, 0, 0, -201923.076923077, 32.7190170940171, -1.40224358974359, 0.0277644230769231, 9288461.53846154, -3926282.05128205, 5250, 0, 0, -201923.076923077, 38.7019230769231, -16.3595085470085, 0.021875, -24589743.5897436, 3141025.64102564, -25980.7692307692, 0, 0, 1346153.84615385, -102.457264957265, 13.0876068376068, -0.108253205128205, -5294871.7948718, 5833333.33333333, -9153.84615384615, 0, 0, 1750000, -22.0619658119658, 24.3055555555556, -0.0381410256410256, -13102564.1025641, 2243589.74358974, -9557.69230769231, 0, 0, 0, -54.5940170940171, 9.3482905982906, -0.0398237179487179, -5474358.97435897, 2243589.74358974, -14628.2051282051, 0, 0, -942307.692307692, -22.8098290598291, 9.3482905982906, -0.0609508547008547;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(10) << -336538.461538461, 7852564.1025641, -6663.46153846154, 0, 0, 201923.076923077, -1.40224358974359, 32.7190170940171, -0.0277644230769231, -9535256.41025641, 21000000, -41259.6153846154, 0, 0, -807692.307692308, -39.730235042735, 87.5, -0.171915064102564, 336538.461538461, 10320512.8205128, -6147.4358974359, 0, 0, 538461.538461538, 1.40224358974359, 43.0021367521368, -0.0256143162393162, -3926282.05128205, 9288461.53846154, -5250, 0, 0, 201923.076923077, -16.3595085470085, 38.7019230769231, -0.021875, 5833333.33333333, -5294871.7948718, 9153.84615384615, 0, 0, -1750000, 24.3055555555556, -22.0619658119658, 0.0381410256410256, 3141025.64102564, -24589743.5897436, 25980.7692307692, 0, 0, -1346153.84615385, 13.0876068376068, -102.457264957265, 0.108253205128205, 2243589.74358974, -5474358.97435897, 14628.2051282051, 0, 0, 942307.692307692, 9.3482905982906, -22.8098290598291, 0.0609508547008547, 2243589.74358974, -13102564.1025641, 9557.69230769231, 0, 0, 0, 9.3482905982906, -54.5940170940171, 0.0398237179487179;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(11) << 6842.94871794872, -6483.97435897436, 4038498.08653846, -358974.358974359, 134615.384615385, -1826.92307692308, 0.0285122863247863, -0.0270165598290598, 16.8270753605769, 41259.6153846154, -41259.6153846154, 9333504.81538462, 538461.538461539, -538461.538461539, 4791.66666666667, 0.171915064102564, -0.171915064102564, 38.8896033974359, 6483.97435897436, -6842.94871794872, 4038498.08653846, -134615.384615385, 358974.358974359, -1826.92307692308, 0.0270165598290598, -0.0285122863247863, 16.8270753605769, 5182.69230769231, -5182.69230769231, 4128232.47307692, -134615.384615385, 134615.384615385, -1266.02564102564, 0.0215945512820513, -0.0215945512820513, 17.2009686378205, -30961.5384615385, 4038.46153846154, -6641124.43076923, 897435.897435898, -1166666.66666667, 8333.33333333333, -0.12900641025641, 0.0168269230769231, -27.6713517948718, -4038.46153846154, 30961.5384615385, -6641124.43076923, 1166666.66666667, -897435.897435898, 8333.33333333333, -0.0168269230769231, 0.12900641025641, -27.6713517948718, -9782.05128205128, 14987.1794871794, -4128242.3, 0, 628205.128205128, 3397.4358974359, -0.040758547008547, 0.0624465811965812, -17.2010095833333, -14987.1794871794, 9782.05128205128, -4128242.3, -628205.128205128, 0, 3397.4358974359, -0.0624465811965812, 0.040758547008547, -17.2010095833333;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(12) << 0, 0, 358974.358974359, 89829.594017094, 2.80448717948718, -27.5128739316239, 0, 0, -2.99145299145299, 0, 0, 538461.538461539, 269405.769230769, -79.4604700854701, 250.343830128205, 0, 0, 8.97435897435897, 0, 0, -134615.384615385, 89809.0277777778, -2.80448717948718, -46.0983173076923, 0, 0, -2.24358974358974, 0, 0, 134615.384615385, 134692.788461538, -32.7190170940171, -9.57163461538462, 0, 0, -1.12179487179487, 0, 0, -897435.897435898, -269435.683760684, 26.1752136752137, 152.347596153846, 0, 0, 7.47863247863248, 0, 0, -628205.128205128, -269274.893162393, 48.6111111111111, 110.180128205128, 0, 0, 11.965811965812, 0, 0, 0, -359083.547008547, 18.6965811965812, 26.8434294871795, 0, 0, 0, 0, 0, 628205.128205128, -359019.978632479, 18.6965811965812, 81.9293803418804, 0, 0, -5.23504273504273;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(13) << 0, 0, 134615.384615385, -2.80448717948718, 89809.0277777778, 46.0983173076923, 0, 0, 2.24358974358974, 0, 0, -538461.538461539, -79.4604700854701, 269405.769230769, -250.343830128205, 0, 0, -8.97435897435897, 0, 0, -358974.358974359, 2.80448717948718, 89829.594017094, 27.5128739316239, 0, 0, 2.99145299145299, 0, 0, -134615.384615385, -32.7190170940171, 134692.788461538, 9.57163461538462, 0, 0, 1.12179487179487, 0, 0, 628205.128205128, 48.6111111111111, -269274.893162393, -110.180128205128, 0, 0, -11.965811965812, 0, 0, 897435.897435898, 26.1752136752137, -269435.683760684, -152.347596153846, 0, 0, -7.47863247863248, 0, 0, -628205.128205128, 18.6965811965812, -359019.978632479, -81.9293803418804, 0, 0, 5.23504273504273, 0, 0, 0, 18.6965811965812, -359083.547008547, -26.8434294871795, 0, 0, 0;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(14) << 538461.538461538, 201923.076923077, 336.538461538462, -27.507077991453, 46.099813034188, 314136.086202003, -0.747863247863248, 1.96314102564103, -0.0138221153846154, 807692.307692308, -807692.307692308, 4791.66666666667, 250.343830128205, -250.343830128205, 942386.674933291, 7.8525641025641, -7.8525641025641, 0.0598958333333333, -201923.076923077, -538461.538461538, 336.538461538462, -46.099813034188, 27.507077991453, 314136.086202003, -1.96314102564103, 0.747863247863248, -0.0138221153846154, 201923.076923077, -201923.076923077, -80.1282051282051, -9.57219551282051, 9.57219551282051, 471188.243390267, -0.280448717948718, 0.280448717948718, -0.0108840811965812, -1346153.84615385, 942307.692307692, -2435.89743589744, 152.30608974359, -110.222756410256, -942362.499968547, 1.86965811965812, -5.79594017094017, 0.0592948717948718, -942307.692307692, 1346153.84615385, -2435.89743589744, 110.222756410256, -152.30608974359, -942362.499968547, 5.79594017094017, -1.86965811965812, 0.0592948717948718, 0, -942307.692307692, -256.410256410256, 26.8415598290598, -81.9263888888889, -1256444.70244652, 0, 1.30876068376068, 0.0272435897435897, 942307.692307692, 0, -256.410256410256, 81.9263888888889, -26.8415598290598, -1256444.70244652, -1.30876068376068, 0, 0.0272435897435897;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(15) << 43.0021367521368, 1.40224358974359, 0.0256143162393162, 0, 0, 0.747863247863248, 0.748185763888889, 1.05168269230769E-05, -0.00022950874732906, 87.5, -39.730235042735, 0.171915064102564, 0, 0, 7.8525641025641, 2.24424599358974, -0.000297976762820513, 0.0020846226963141, 32.7190170940171, -1.40224358974359, 0.0277644230769231, 0, 0, -1.96314102564103, 0.748108640491453, -1.05168269230769E-05, -0.000384407151442308, 38.7019230769231, -16.3595085470085, 0.021875, 0, 0, 0.280448717948718, 1.12208513621795, -0.000122696314102564, -7.99641426282051E-05, -102.457264957265, 13.0876068376068, -0.108253205128205, 0, 0, -1.86965811965812, -2.24435817307692, 9.81570512820513E-05, 0.00127055562232906, -22.0619658119658, 24.3055555555556, -0.0381410256410256, 0, 0, 2.05662393162393, -2.24375520833333, 0.000182291666666667, 0.000918517361111111, -54.5940170940171, 9.3482905982906, -0.0398237179487179, 0, 0, 0, -2.9918624465812, 7.01121794871795E-05, 0.000224060296474359, -22.8098290598291, 9.3482905982906, -0.0609508547008547, 0, 0, 1.30876068376068, -2.99162406517094, 7.01121794871795E-05, 0.000683303552350427;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(16) << -1.40224358974359, 32.7190170940171, -0.0277644230769231, 0, 0, 1.96314102564103, -1.05168269230769E-05, 0.748108640491453, 0.000384407151442308, -39.730235042735, 87.5, -0.171915064102564, 0, 0, -7.8525641025641, -0.000297976762820513, 2.24424599358974, -0.0020846226963141, 1.40224358974359, 43.0021367521368, -0.0256143162393162, 0, 0, -0.747863247863248, 1.05168269230769E-05, 0.748185763888889, 0.00022950874732906, -16.3595085470085, 38.7019230769231, -0.021875, 0, 0, -0.280448717948718, -0.000122696314102564, 1.12208513621795, 7.99641426282051E-05, 24.3055555555556, -22.0619658119658, 0.0381410256410256, 0, 0, -2.05662393162393, 0.000182291666666667, -2.24375520833333, -0.000918517361111111, 13.0876068376068, -102.457264957265, 0.108253205128205, 0, 0, 1.86965811965812, 9.81570512820513E-05, -2.24435817307692, -0.00127055562232906, 9.3482905982906, -22.8098290598291, 0.0609508547008547, 0, 0, -1.30876068376068, 7.01121794871795E-05, -2.99162406517094, -0.000683303552350427, 9.3482905982906, -54.5940170940171, 0.0398237179487179, 0, 0, 0, 7.01121794871795E-05, -2.9918624465812, -0.000224060296474359;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(17) << 0.0285122863247863, -0.0270165598290598, 16.8270753605769, 2.99145299145299, 2.24358974358974, -0.00480769230769231, -0.000229487012553419, 0.000384412760416667, 2.61764647015922, 0.171915064102564, -0.171915064102564, 38.8896033974359, 8.97435897435897, -8.97435897435897, 0.0598958333333333, 0.0020846226963141, -0.0020846226963141, 7.85286580307961, 0.0270165598290598, -0.0285122863247863, 16.8270753605769, -2.24358974358974, -2.99145299145299, -0.00480769230769231, -0.000384412760416667, 0.000229487012553419, 2.61764647015922, 0.0215945512820513, -0.0215945512820513, 17.2009686378205, 1.12179487179487, -1.12179487179487, -0.00594284188034188, -7.99662459935897E-05, 7.99662459935897E-05, 3.92641101937305, -0.12900641025641, 0.0168269230769231, -27.6713517948718, -7.47863247863248, 2.99145299145299, 0.0144230769230769, 0.0012703999732906, -0.000918677216880342, -7.85276717901311, -0.0168269230769231, 0.12900641025641, -27.6713517948718, -2.99145299145299, 7.47863247863248, 0.0144230769230769, 0.000918677216880342, -0.0012703999732906, -7.85276717901311, -0.040758547008547, 0.0624465811965812, -17.2010095833333, 0, -5.23504273504273, 0.0120192307692308, 0.00022405328525641, -0.000683292334401709, -10.4702148444665, -0.0624465811965812, 0.040758547008547, -17.2010095833333, 5.23504273504273, 0, 0.0120192307692308, 0.000683292334401709, -0.00022405328525641, -10.4702148444665;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(18) << 9288461.53846154, 3926282.05128205, 628.205128205128, 0, 0, -201923.076923077, 38.7019230769231, 16.3595085470085, 0.00261752136752137, 7852564.1025641, 336538.461538461, 6483.97435897436, 0, 0, -201923.076923077, 32.7190170940171, 1.40224358974359, 0.0270165598290598, 21000000, 9535256.41025641, 6663.46153846154, 0, 0, 807692.307692308, 87.5, 39.730235042735, 0.0277644230769231, 10320512.8205128, -336538.461538461, 493.589743589744, 0, 0, -538461.538461538, 43.0021367521368, -1.40224358974359, 0.00205662393162393, -13102564.1025641, -2243589.74358974, -2288.46153846154, 0, 0, 0, -54.5940170940171, -9.3482905982906, -0.00953525641025641, -5294871.7948718, -5833333.33333333, -3320.51282051282, 0, 0, 1750000, -22.0619658119658, -24.3055555555556, -0.0138354700854701, -24589743.5897436, -3141025.64102564, -1660.25641025641, 0, 0, 1346153.84615385, -102.457264957265, -13.0876068376068, -0.00691773504273504, -5474358.97435897, -2243589.74358974, -7000, 0, 0, -942307.692307692, -22.8098290598291, -9.3482905982906, -0.0291666666666667;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(19) << 3926282.05128205, 9288461.53846154, -426.282051282051, 0, 0, -201923.076923077, 16.3595085470085, 38.7019230769231, -0.00177617521367521, -336538.461538461, 10320512.8205128, -6842.94871794872, 0, 0, -538461.538461538, -1.40224358974359, 43.0021367521368, -0.0285122863247863, 9535256.41025641, 21000000, 6147.4358974359, 0, 0, 807692.307692308, 39.730235042735, 87.5, 0.0256143162393162, 336538.461538461, 7852564.1025641, 493.589743589744, 0, 0, -201923.076923077, 1.40224358974359, 32.7190170940171, 0.00205662393162393, -2243589.74358974, -5474358.97435897, -3051.28205128205, 0, 0, -942307.692307692, -9.3482905982906, -22.8098290598291, -0.0127136752136752, -3141025.64102564, -24589743.5897436, 44.8717948717949, 0, 0, 1346153.84615385, -13.0876068376068, -102.457264957265, 0.000186965811965812, -5833333.33333333, -5294871.7948718, 3679.48717948718, 0, 0, 1750000, -24.3055555555556, -22.0619658119658, 0.0153311965811966, -2243589.74358974, -13102564.1025641, -44.8717948717949, 0, 0, 0, -9.3482905982906, -54.5940170940171, -0.000186965811965812;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(20) << 426.282051282051, -628.205128205128, 4128215.48012821, -134615.384615385, -134615.384615385, -673.076923076923, 0.00177617521367521, -0.00261752136752137, 17.2008978338675, 6663.46153846154, -6147.4358974359, 4038498.08653846, -134615.384615385, -358974.358974359, 336.538461538462, 0.0277644230769231, -0.0256143162393162, 16.8270753605769, 6663.46153846154, 6147.4358974359, 9333352.2474359, 538461.538461539, 538461.538461539, 336.538461538462, 0.0277644230769231, 0.0256143162393162, 38.8889676976496, 426.282051282051, 628.205128205128, 4038471.17307692, -358974.358974359, -134615.384615385, -673.076923076923, 0.00177617521367521, 0.00261752136752137, 16.8269632211538, -2961.53846153846, -4307.69230769231, -4128222.81666667, 0, -628205.128205128, 2884.61538461538, -0.0123397435897436, -0.0179487179487179, -17.2009284027778, -1256.41025641026, 0, -6641059.77948718, 1166666.66666667, 897435.897435898, 2724.35897435897, -0.00523504273504274, 0, -27.6710824145299, -2961.53846153846, 4307.69230769231, -6641037.27307692, 897435.897435898, 1166666.66666667, 2884.61538461538, -0.0123397435897436, 0.0179487179487179, -27.6709886378205, -7000, 0, -4128217.11794872, -628205.128205128, 0, 2275.64102564103, -0.0291666666666667, 0, -17.2009046581197;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(21) << 0, 0, 134615.384615385, 134692.788461538, 32.7190170940171, 26.2872863247863, 0, 0, -1.12179487179487, 0, 0, -134615.384615385, 89809.0277777778, 2.80448717948718, -46.099813034188, 0, 0, -2.24358974358974, 0, 0, 538461.538461539, 269405.769230769, 79.4604700854701, 34.6709134615385, 0, 0, 8.97435897435897, 0, 0, 358974.358974359, 89829.594017094, -2.80448717948718, 17.3118055555556, 0, 0, -2.99145299145299, 0, 0, 0, -359083.547008547, -18.6965811965812, -98.7370192307692, 0, 0, 0, 0, 0, -628205.128205128, -269274.893162393, -48.6111111111111, -123.104594017094, 0, 0, 11.965811965812, 0, 0, -897435.897435898, -269435.683760684, -26.1752136752137, 8.96052350427351, 0, 0, 7.47863247863248, 0, 0, 628205.128205128, -359019.978632479, -18.6965811965812, 46.0955128205128, 0, 0, -5.23504273504273;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(22) << 0, 0, 134615.384615385, 32.7190170940171, 134692.788461538, -26.2856036324786, 0, 0, -1.12179487179487, 0, 0, 358974.358974359, -2.80448717948718, 89829.594017094, 27.507077991453, 0, 0, -2.99145299145299, 0, 0, 538461.538461539, 79.4604700854701, 269405.769230769, 55.1794337606838, 0, 0, 8.97435897435897, 0, 0, -134615.384615385, 2.80448717948718, 89809.0277777778, -43.5856303418803, 0, 0, -2.24358974358974, 0, 0, 628205.128205128, -18.6965811965812, -359019.978632479, 133.307905982906, 0, 0, -5.23504273504273, 0, 0, -897435.897435898, -26.1752136752137, -269435.683760684, 170.513194444444, 0, 0, 7.47863247863248, 0, 0, -628205.128205128, -48.6111111111111, -269274.893162393, 33.3639957264957, 0, 0, 11.965811965812, 0, 0, 0, -18.6965811965812, -359083.547008547, 8.97398504273504, 0, 0, 0;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(23) << 201923.076923077, 201923.076923077, -673.076923076923, 26.2856036324786, -26.2872863247863, 471188.345812762, -0.280448717948718, -0.280448717948718, -0.00841346153846154, -201923.076923077, 538461.538461538, -1826.92307692308, -46.0983173076923, 27.5128739316239, 314136.086202003, -1.96314102564103, -0.747863247863248, -0.00480769230769231, 807692.307692308, 807692.307692308, 336.538461538462, 34.6709134615385, 55.1794337606838, 942385.660841378, 7.8525641025641, 7.8525641025641, 0.00420673076923077, 538461.538461538, -201923.076923077, -977.564102564103, 17.3112446581197, -43.5845085470086, 314136.266746955, -0.747863247863248, -1.96314102564103, -0.00968215811965812, 0, 942307.692307692, 1153.84615384615, -98.7426282051282, 133.297435897436, -1256444.98689954, 0, -1.30876068376068, 0.0288461538461538, -942307.692307692, -1346153.84615385, 32.0512820512821, -123.087393162393, 170.512820512821, -942363.409686197, 5.79594017094017, 1.86965811965812, 0.0228365384615385, -1346153.84615385, -942307.692307692, 1089.74358974359, 8.94967948717949, 33.3692307692308, -942363.119755053, 1.86965811965812, 5.79594017094017, 0.0285790598290598, 942307.692307692, 0, 865.384615384615, 46.0955128205128, 8.97435897435897, -1256444.80480077, -1.30876068376068, 0, 0.0225694444444444;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(24) << 38.7019230769231, 16.3595085470085, 0.00261752136752137, 0, 0, 0.280448717948718, 1.12208513621795, 0.000122696314102564, 0.00021903672542735, 32.7190170940171, 1.40224358974359, 0.0270165598290598, 0, 0, -1.96314102564103, 0.748108640491453, 1.05168269230769E-05, -0.000384412760416667, 87.5, 39.730235042735, 0.0277644230769231, 0, 0, 7.8525641025641, 2.24424599358974, 0.000297976762820513, 0.000288669771634615, 43.0021367521368, -1.40224358974359, 0.00205662393162393, 0, 0, 0.747863247863248, 0.748185763888889, -1.05168269230769E-05, 0.000144246193910256, -54.5940170940171, -9.3482905982906, -0.00953525641025641, 0, 0, 0, -2.9918624465812, -7.01121794871795E-05, -0.00082272108707265, -22.0619658119658, -24.3055555555556, -0.0138354700854701, 0, 0, 2.05662393162393, -2.24375520833333, -0.000182291666666667, -0.00102574479166667, -102.457264957265, -13.0876068376068, -0.00691773504273504, 0, 0, -1.86965811965812, -2.24435817307692, -9.81570512820513E-05, 7.47344417735043E-05, -22.8098290598291, -9.3482905982906, -0.0291666666666667, 0, 0, 1.30876068376068, -2.99162406517094, -7.01121794871795E-05, 0.000384396634615385;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(25) << 16.3595085470085, 38.7019230769231, -0.00177617521367521, 0, 0, 0.280448717948718, 0.000122696314102564, 1.12208513621795, -0.000219030415331197, -1.40224358974359, 43.0021367521368, -0.0285122863247863, 0, 0, 0.747863247863248, -1.05168269230769E-05, 0.748185763888889, 0.000229487012553419, 39.730235042735, 87.5, 0.0256143162393162, 0, 0, 7.8525641025641, 0.000297976762820513, 2.24424599358974, 0.000459593816773504, 1.40224358974359, 32.7190170940171, 0.00205662393162393, 0, 0, -1.96314102564103, 1.05168269230769E-05, 0.748108640491453, -0.000363232438568376, -9.3482905982906, -22.8098290598291, -0.0127136752136752, 0, 0, 1.30876068376068, -7.01121794871795E-05, -2.99162406517094, 0.00111101575854701, -13.0876068376068, -102.457264957265, 0.000186965811965812, 0, 0, -1.86965811965812, -9.81570512820513E-05, -2.24435817307692, 0.00142094157318376, -24.3055555555556, -22.0619658119658, 0.0153311965811966, 0, 0, 2.05662393162393, -0.000182291666666667, -2.24375520833333, 0.000277892761752137, -9.3482905982906, -54.5940170940171, -0.000186965811965812, 0, 0, 0, -7.01121794871795E-05, -2.9918624465812, 0.000074784922542735;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(26) << 0.00177617521367521, -0.00261752136752137, 17.2008978338675, 1.12179487179487, 1.12179487179487, -0.00841346153846154, 0.000219030415331197, -0.00021903672542735, 3.92641187354287, 0.0277644230769231, -0.0256143162393162, 16.8270753605769, -2.24358974358974, 2.99145299145299, -0.0138221153846154, -0.000384407151442308, 0.00022950874732906, 2.61764647015922, 0.0277644230769231, 0.0256143162393162, 38.8889676976496, 8.97435897435897, 8.97435897435897, 0.00420673076923077, 0.000288669771634615, 0.000459593816773504, 7.85285735814092, 0.00177617521367521, 0.00261752136752137, 16.8269632211538, 2.99145299145299, -2.24358974358974, -0.0109508547008547, 0.000144244090544872, -0.000363228231837607, 2.61764797572843, -0.0123397435897436, -0.0179487179487179, -17.2009284027778, 0, 5.23504273504273, 0.0216346153846154, -0.000822742120726496, 0.0011109764957265, -10.4702172156525, -0.00523504273504274, 0, -27.6710824145299, -2.99145299145299, -7.47863247863248, 0.0116185897435897, -0.00102568028846154, 0.00142094017094017, -7.85277476246284, -0.0123397435897436, 0.0179487179487179, -27.6709886378205, -7.47863247863248, -2.99145299145299, 0.0211004273504274, 7.46937767094017E-05, 0.000277912393162393, -7.8527723472296, -0.0291666666666667, 0, -17.2009046581197, 5.23504273504273, 0, 0.0166933760683761, 0.000384396634615385, 7.47863247863248E-05, -10.4702156983804;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(27) << 7852564.1025641, -336538.461538461, -628.205128205128, 0, 0, 201923.076923077, 32.7190170940171, -1.40224358974359, -0.00261752136752137, 9288461.53846154, -3926282.05128205, 5182.69230769231, 0, 0, 201923.076923077, 38.7019230769231, -16.3595085470085, 0.0215945512820513, 10320512.8205128, 336538.461538461, 426.282051282051, 0, 0, 538461.538461538, 43.0021367521368, 1.40224358974359, 0.00177617521367521, 21000000, -9535256.41025641, -5250, 0, 0, -807692.307692308, 87.5, -39.730235042735, -0.021875, -13102564.1025641, 2243589.74358974, -762.820512820513, 0, 0, 0, -54.5940170940171, 9.3482905982906, -0.0031784188034188, -5474358.97435897, 2243589.74358974, 3679.48717948718, 0, 0, 942307.692307692, -22.8098290598291, 9.3482905982906, 0.0153311965811966, -24589743.5897436, 3141025.64102564, 762.820512820513, 0, 0, -1346153.84615385, -102.457264957265, 13.0876068376068, 0.0031784188034188, -5294871.7948718, 5833333.33333333, -3410.25641025641, 0, 0, -1750000, -22.0619658119658, 24.3055555555556, -0.0142094017094017;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(28) << 336538.461538461, 10320512.8205128, -426.282051282051, 0, 0, -538461.538461538, 1.40224358974359, 43.0021367521368, -0.00177617521367521, -3926282.05128205, 9288461.53846154, -5182.69230769231, 0, 0, -201923.076923077, -16.3595085470085, 38.7019230769231, -0.0215945512820513, -336538.461538461, 7852564.1025641, 628.205128205128, 0, 0, -201923.076923077, -1.40224358974359, 32.7190170940171, 0.00261752136752137, -9535256.41025641, 21000000, 5250, 0, 0, 807692.307692308, -39.730235042735, 87.5, 0.021875, 2243589.74358974, -5474358.97435897, -3679.48717948718, 0, 0, -942307.692307692, 9.3482905982906, -22.8098290598291, -0.0153311965811966, 2243589.74358974, -13102564.1025641, 762.820512820513, 0, 0, 0, 9.3482905982906, -54.5940170940171, 0.0031784188034188, 5833333.33333333, -5294871.7948718, 3410.25641025641, 0, 0, 1750000, 24.3055555555556, -22.0619658119658, 0.0142094017094017, 3141025.64102564, -24589743.5897436, -762.820512820513, 0, 0, 1346153.84615385, 13.0876068376068, -102.457264957265, -0.0031784188034188;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(29) << -493.589743589744, -493.589743589744, 4038471.17307692, 134615.384615385, -358974.358974359, -977.564102564103, -0.00205662393162393, -0.00205662393162393, 16.8269632211538, 5250, -5250, 4128232.47307692, 134615.384615385, -134615.384615385, -80.1282051282051, 0.021875, -0.021875, 17.2009686378205, 493.589743589744, 493.589743589744, 4038471.17307692, 358974.358974359, -134615.384615385, -977.564102564103, 0.00205662393162393, 0.00205662393162393, 16.8269632211538, -5250, 5250, 9333346.25769231, -538461.538461539, 538461.538461539, -80.1282051282051, -0.021875, 0.021875, 38.8889427403846, -1615.38461538462, -4576.92307692308, -4128221.97307692, 0, -628205.128205128, 3108.97435897436, -0.00673076923076923, -0.0190705128205128, -17.2009248878205, 4576.92307692308, 1615.38461538462, -4128221.97307692, 628205.128205128, 0, 3108.97435897436, 0.0190705128205128, 0.00673076923076923, -17.2009248878205, 1615.38461538462, 4576.92307692308, -6641038.56538462, -897435.897435898, 1166666.66666667, 3108.97435897436, 0.00673076923076923, 0.0190705128205128, -27.6709940224359, -4576.92307692308, -1615.38461538462, -6641038.56538462, -1166666.66666667, 897435.897435898, 3108.97435897436, -0.0190705128205128, -0.00673076923076923, -27.6709940224359;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(30) << 0, 0, 134615.384615385, 89809.0277777778, -2.80448717948718, 43.5845085470086, 0, 0, 2.24358974358974, 0, 0, -134615.384615385, 134692.788461538, -32.7190170940171, -9.57219551282051, 0, 0, 1.12179487179487, 0, 0, -358974.358974359, 89829.594017094, 2.80448717948718, 17.3112446581197, 0, 0, 2.99145299145299, 0, 0, -538461.538461539, 269405.769230769, -79.4604700854701, -19.2745192307692, 0, 0, -8.97435897435897, 0, 0, 0, -359083.547008547, 18.6965811965812, -62.8268696581197, 0, 0, 0, 0, 0, -628205.128205128, -359019.978632479, 18.6965811965812, -97.405235042735, 0, 0, 5.23504273504273, 0, 0, 897435.897435898, -269435.683760684, 26.1752136752137, -26.9167200854701, 0, 0, -7.47863247863248, 0, 0, 628205.128205128, -269274.893162393, 48.6111111111111, 20.4844017094017, 0, 0, -11.965811965812;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(31) << 0, 0, 358974.358974359, 2.80448717948718, 89829.594017094, -17.3112446581197, 0, 0, -2.99145299145299, 0, 0, 134615.384615385, -32.7190170940171, 134692.788461538, 9.57219551282051, 0, 0, -1.12179487179487, 0, 0, -134615.384615385, -2.80448717948718, 89809.0277777778, -43.5845085470086, 0, 0, -2.24358974358974, 0, 0, 538461.538461539, -79.4604700854701, 269405.769230769, 19.2745192307692, 0, 0, 8.97435897435897, 0, 0, 628205.128205128, 18.6965811965812, -359019.978632479, 97.405235042735, 0, 0, -5.23504273504273, 0, 0, 0, 18.6965811965812, -359083.547008547, 62.8268696581197, 0, 0, 0, 0, 0, -628205.128205128, 48.6111111111111, -269274.893162393, -20.4844017094017, 0, 0, 11.965811965812, 0, 0, -897435.897435898, 26.1752136752137, -269435.683760684, 26.9167200854701, 0, 0, 7.47863247863248;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(32) << 201923.076923077, 538461.538461538, -673.076923076923, 43.5856303418803, -17.3118055555556, 314136.266746955, 1.96314102564103, -0.747863247863248, -0.0109508547008547, -201923.076923077, 201923.076923077, -1266.02564102564, -9.57163461538462, 9.57163461538462, 471188.243390267, 0.280448717948718, -0.280448717948718, -0.00594284188034188, -538461.538461538, -201923.076923077, -673.076923076923, 17.3118055555556, -43.5856303418803, 314136.266746955, 0.747863247863248, -1.96314102564103, -0.0109508547008547, -807692.307692308, 807692.307692308, -80.1282051282051, -19.2745192307692, 19.2745192307692, 942385.550535053, -7.8525641025641, 7.8525641025641, -0.00100160256410256, 0, 942307.692307692, 929.487179487179, -62.8339743589744, 97.3977564102564, -1256444.88432841, 0, -1.30876068376068, 0.0297809829059829, -942307.692307692, 0, 929.487179487179, -97.3977564102564, 62.8339743589744, -1256444.88432841, 1.30876068376068, 0, 0.0297809829059829, 1346153.84615385, -942307.692307692, 416.666666666667, -26.9096153846154, -20.4746794871795, -942363.147970951, -1.86965811965812, 5.79594017094017, 0.0276442307692308, 942307.692307692, -1346153.84615385, 416.666666666667, 20.4746794871795, 26.9096153846154, -942363.147970951, -5.79594017094017, 1.86965811965812, 0.0276442307692308;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(33) << 32.7190170940171, -1.40224358974359, -0.00261752136752137, 0, 0, 1.96314102564103, 0.748108640491453, -1.05168269230769E-05, 0.000363228231837607, 38.7019230769231, -16.3595085470085, 0.0215945512820513, 0, 0, -0.280448717948718, 1.12208513621795, -0.000122696314102564, -7.99662459935897E-05, 43.0021367521368, 1.40224358974359, 0.00177617521367521, 0, 0, -0.747863247863248, 0.748185763888889, 1.05168269230769E-05, 0.000144244090544872, 87.5, -39.730235042735, -0.021875, 0, 0, -7.8525641025641, 2.24424599358974, -0.000297976762820513, -0.00016042047275641, -54.5940170940171, 9.3482905982906, -0.0031784188034188, 0, 0, 0, -2.9918624465812, 7.01121794871795E-05, -0.000523528111645299, -22.8098290598291, 9.3482905982906, 0.0153311965811966, 0, 0, -1.30876068376068, -2.99162406517094, 7.01121794871795E-05, -0.000811850827991453, -102.457264957265, 13.0876068376068, 0.0031784188034188, 0, 0, 1.86965811965812, -2.24435817307692, 9.81570512820513E-05, -0.000224335136217949, -22.0619658119658, 24.3055555555556, -0.0142094017094017, 0, 0, -2.05662393162393, -2.24375520833333, 0.000182291666666667, 0.00017083360042735;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(34) << 1.40224358974359, 43.0021367521368, -0.00177617521367521, 0, 0, 0.747863247863248, 1.05168269230769E-05, 0.748185763888889, -0.000144244090544872, -16.3595085470085, 38.7019230769231, -0.0215945512820513, 0, 0, 0.280448717948718, -0.000122696314102564, 1.12208513621795, 7.99662459935897E-05, -1.40224358974359, 32.7190170940171, 0.00261752136752137, 0, 0, -1.96314102564103, -1.05168269230769E-05, 0.748108640491453, -0.000363228231837607, -39.730235042735, 87.5, 0.021875, 0, 0, 7.8525641025641, -0.000297976762820513, 2.24424599358974, 0.00016042047275641, 9.3482905982906, -22.8098290598291, -0.0153311965811966, 0, 0, 1.30876068376068, 7.01121794871795E-05, -2.99162406517094, 0.000811850827991453, 9.3482905982906, -54.5940170940171, 0.0031784188034188, 0, 0, 0, 7.01121794871795E-05, -2.9918624465812, 0.000523528111645299, 24.3055555555556, -22.0619658119658, 0.0142094017094017, 0, 0, 2.05662393162393, 0.000182291666666667, -2.24375520833333, -0.00017083360042735, 13.0876068376068, -102.457264957265, -0.0031784188034188, 0, 0, -1.86965811965812, 9.81570512820513E-05, -2.24435817307692, 0.000224335136217949;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(35) << -0.00205662393162393, -0.00205662393162393, 16.8269632211538, 2.24358974358974, 2.99145299145299, -0.00968215811965812, 0.000363232438568376, -0.000144246193910256, 2.61764797572843, 0.021875, -0.021875, 17.2009686378205, -1.12179487179487, 1.12179487179487, -0.0108840811965812, -7.99641426282051E-05, 7.99641426282051E-05, 3.92641101937305, 0.00205662393162393, 0.00205662393162393, 16.8269632211538, -2.99145299145299, -2.24358974358974, -0.00968215811965812, 0.000144246193910256, -0.000363232438568376, 2.61764797572843, -0.021875, 0.021875, 38.8889427403846, -8.97435897435897, 8.97435897435897, -0.00100160256410256, -0.00016042047275641, 0.00016042047275641, 7.85285643915032, -0.00673076923076923, -0.0190705128205128, -17.2009248878205, 0, 5.23504273504273, 0.0206997863247863, -0.000523554754273504, 0.000811822783119658, -10.4702163609253, 0.0190705128205128, 0.00673076923076923, -17.2009248878205, -5.23504273504273, 0, 0.0206997863247863, -0.000811822783119658, 0.000523554754273504, -10.4702163609253, 0.00673076923076923, 0.0190705128205128, -27.6709940224359, 7.47863247863248, -2.99145299145299, 0.0164262820512821, -0.000224308493589744, -0.000170797142094017, -7.85277258231272, -0.0190705128205128, -0.00673076923076923, -27.6709940224359, 2.99145299145299, -7.47863247863248, 0.0164262820512821, 0.000170797142094017, 0.000224308493589744, -7.85277258231272;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(36) << -24589743.5897436, -5833333.33333333, 0, 0, 0, 1346153.84615385, -102.457264957265, -24.3055555555556, 0, -24589743.5897436, 5833333.33333333, -30961.5384615385, 0, 0, -1346153.84615385, -102.457264957265, 24.3055555555556, -0.12900641025641, -13102564.1025641, -2243589.74358974, -2961.53846153846, 0, 0, 0, -54.5940170940171, -9.3482905982906, -0.0123397435897436, -13102564.1025641, 2243589.74358974, -1615.38461538462, 0, 0, 0, -54.5940170940171, 9.3482905982906, -0.00673076923076923, 54564102.5641026, 0, 18846.1538461538, 0, 0, 0, 227.350427350427, 0, 0.078525641025641, 0, -8974358.97435897, -2153.84615384615, 0, 0, -2692307.69230769, 0, -37.3931623931624, -0.00897435897435897, 20820512.8205128, 0, 4128.20512820513, 0, 0, 0, 86.7521367521367, 0, 0.0172008547008547, 0, 8974358.97435897, 14717.9487179487, 0, 0, 2692307.69230769, 0, 37.3931623931624, 0.0613247863247863;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(37) << -3141025.64102564, -5294871.7948718, 1256.41025641026, 0, 0, 942307.692307692, -13.0876068376068, -22.0619658119658, 0.00523504273504274, 3141025.64102564, -5294871.7948718, 4038.46153846154, 0, 0, 942307.692307692, 13.0876068376068, -22.0619658119658, 0.0168269230769231, -2243589.74358974, -5474358.97435897, -4307.69230769231, 0, 0, 942307.692307692, -9.3482905982906, -22.8098290598291, -0.0179487179487179, 2243589.74358974, -5474358.97435897, -4576.92307692308, 0, 0, 942307.692307692, 9.3482905982906, -22.8098290598291, -0.0190705128205128, 0, 29435897.4358974, -5384.61538461539, 0, 0, -3230769.23076923, 0, 122.649572649573, -0.0224358974358974, -8974358.97435897, 0, 7000, 0, 0, -2692307.69230769, -37.3931623931624, 0, 0.0291666666666667, 0, -7897435.8974359, -1794.8717948718, 0, 0, -3230769.23076923, 0, -32.9059829059829, -0.00747863247863248, 8974358.97435897, 0, 3769.23076923077, 0, 0, -2692307.69230769, 37.3931623931624, 0, 0.0157051282051282;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(38) << -44.8717948717949, 3320.51282051282, -6641059.77948718, 897435.897435898, 628205.128205128, 32.0512820512821, -0.000186965811965812, 0.0138354700854701, -27.6710824145299, -25980.7692307692, 9153.84615384615, -6641124.43076923, -897435.897435898, 628205.128205128, -2435.89743589744, -0.108253205128205, 0.0381410256410256, -27.6713517948718, -2288.46153846154, -3051.28205128205, -4128222.81666667, 0, 628205.128205128, 1153.84615384615, -0.00953525641025641, -0.0127136752136752, -17.2009284027778, -762.820512820513, -3679.48717948718, -4128221.97307692, 0, 628205.128205128, 929.487179487179, -0.0031784188034188, -0.0153311965811966, -17.2009248878205, 18846.1538461538, -5384.61538461539, 18666769.2615385, 0, -2153846.15384615, -3205.12820512821, 0.078525641025641, -0.0224358974358974, 77.7782052564103, -7000, 2153.84615384615, 27.9461538461538, -1794871.7948718, -1794871.7948718, -5384.61538461539, -0.0291666666666667, 0.00897435897435897, 0.000116442307692308, 3051.28205128205, -3948.71794871795, 2871817.37948718, 0, -2153846.15384615, -3205.12820512821, 0.0127136752136752, -0.0164529914529915, 11.9659057478632, 14179.4871794872, 1435.89743589744, 14.4128205128205, 1794871.7948718, -1794871.7948718, -4487.17948717949, 0.0590811965811966, 0.00598290598290598, 6.00534188034188E-05;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(39) << 0, 0, -897435.897435898, -269435.683760684, -48.6111111111111, -170.512820512821, 0, 0, 7.47863247863248, 0, 0, 897435.897435898, -269435.683760684, 48.6111111111111, 152.30608974359, 0, 0, -7.47863247863248, 0, 0, 0, -359083.547008547, -18.6965811965812, -98.7426282051282, 0, 0, 0, 0, 0, 0, -359083.547008547, 18.6965811965812, -62.8339743589744, 0, 0, 0, 0, 0, 0, 1436352.13675214, 0, 646.310897435898, 0, 0, 0, 0, 0, 1794871.7948718, 897435.897435898, -74.7863247863248, 394.853846153846, 0, 0, -14.957264957265, 0, 0, 0, 718122.222222222, 0, 71.8292735042735, 0, 0, 0, 0, 0, -1794871.7948718, 897435.897435898, 74.7863247863248, -35.7747863247863, 0, 0, 14.957264957265;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(40) << 0, 0, -1166666.66666667, -26.1752136752137, -269274.893162393, 123.087393162393, 0, 0, 2.99145299145299, 0, 0, -1166666.66666667, 26.1752136752137, -269274.893162393, -110.222756410256, 0, 0, 2.99145299145299, 0, 0, -628205.128205128, -18.6965811965812, -359019.978632479, 133.297435897436, 0, 0, 5.23504273504273, 0, 0, -628205.128205128, 18.6965811965812, -359019.978632479, 97.3977564102564, 0, 0, 5.23504273504273, 0, 0, -2153846.15384615, 0, 1436142.73504274, -820.557692307692, 0, 0, -35.8974358974359, 0, 0, 1794871.7948718, -74.7863247863248, 897435.897435898, -394.813461538462, 0, 0, -14.957264957265, 0, 0, 2153846.15384615, 0, 717882.905982906, -51.2970085470085, 0, 0, -17.9487179487179, 0, 0, 1794871.7948718, 74.7863247863248, 897435.897435898, -143.558333333333, 0, 0, -14.957264957265;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(41) << -1346153.84615385, -1750000, 2724.35897435897, -170.513194444444, 123.104594017094, -942363.409686197, 1.86965811965812, -2.05662393162393, 0.0116185897435897, 1346153.84615385, -1750000, 8333.33333333333, 152.347596153846, -110.180128205128, -942362.499968547, -1.86965811965812, -2.05662393162393, 0.0144230769230769, 0, -942307.692307692, 2884.61538461538, -98.7370192307692, 133.307905982906, -1256444.98689954, 0, 1.30876068376068, 0.0216346153846154, 0, -942307.692307692, 3108.97435897436, -62.8268696581197, 97.405235042735, -1256444.88432841, 0, 1.30876068376068, 0.0206997863247863, 0, -3230769.23076923, -3205.12820512821, 646.310897435898, -820.557692307692, 5025798.45555581, 0, -31.4102564102564, -0.0400641025641026, 2692307.69230769, 2692307.69230769, -5384.61538461539, 394.813461538462, -394.853846153846, 3141026.66689955, -3.73931623931624, -3.73931623931624, -0.0673076923076923, 0, 3230769.23076923, -3974.35897435897, 71.8202991452992, -51.3149572649573, 2512844.81728158, 0, -4.48717948717949, -0.0432692307692308, -2692307.69230769, 2692307.69230769, -4487.17948717949, -35.7792735042735, -143.577777777778, 3141026.097556, 3.73931623931624, -3.73931623931624, -0.0560897435897436;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(42) << -102.457264957265, -24.3055555555556, 0, 0, 0, -1.86965811965812, -2.24435817307692, -0.000182291666666667, -0.00142094017094017, -102.457264957265, 24.3055555555556, -0.12900641025641, 0, 0, 1.86965811965812, -2.24435817307692, 0.000182291666666667, 0.0012703999732906, -54.5940170940171, -9.3482905982906, -0.0123397435897436, 0, 0, 0, -2.9918624465812, -7.01121794871795E-05, -0.000822742120726496, -54.5940170940171, 9.3482905982906, -0.00673076923076923, 0, 0, 0, -2.9918624465812, 7.01121794871795E-05, -0.000523554754273504, 227.350427350427, 0, 0.078525641025641, 0, 0, 0, 11.9675170940171, 0, 0.00538520432692308, 0, -37.3931623931624, -0.00897435897435897, 0, 0, 3.73931623931624, 7.47863247863248, -0.000280448717948718, 0.00329053098290598, 86.7521367521367, 0, 0.0172008547008547, 0, 0, 0, 5.98355662393162, 0, 0.000598419604700855, 0, 37.3931623931624, 0.0613247863247863, 0, 0, -3.73931623931624, 7.47863247863248, 0.000280448717948718, -0.000298685363247863;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(43) << -13.0876068376068, -22.0619658119658, 0.00523504273504274, 0, 0, -5.79594017094017, -9.81570512820513E-05, -2.24375520833333, 0.00102568028846154, 13.0876068376068, -22.0619658119658, 0.0168269230769231, 0, 0, -5.79594017094017, 9.81570512820513E-05, -2.24375520833333, -0.000918677216880342, -9.3482905982906, -22.8098290598291, -0.0179487179487179, 0, 0, -1.30876068376068, -7.01121794871795E-05, -2.99162406517094, 0.0011109764957265, 9.3482905982906, -22.8098290598291, -0.0190705128205128, 0, 0, -1.30876068376068, 7.01121794871795E-05, -2.99162406517094, 0.000811822783119658, 0, 122.649572649573, -0.0224358974358974, 0, 0, -31.4102564102564, 0, 11.9667318376068, -0.00683777510683761, -37.3931623931624, 0, 0.0291666666666667, 0, 0, 3.73931623931624, -0.000280448717948718, 7.47863247863248, -0.00329037954059829, 0, -32.9059829059829, -0.00747863247863248, 0, 0, 4.48717948717949, 0, 5.98265918803419, -0.000427406517094017, 37.3931623931624, 0, 0.0157051282051282, 0, 0, 3.73931623931624, 0.000280448717948718, 7.47863247863248, -0.00119646340811966;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(44) << -0.000186965811965812, 0.0138354700854701, -27.6710824145299, -7.47863247863248, -11.965811965812, 0.0228365384615385, -0.00142094157318376, 0.00102574479166667, -7.85277476246284, -0.108253205128205, 0.0381410256410256, -27.6713517948718, 7.47863247863248, -11.965811965812, 0.0592948717948718, 0.00127055562232906, -0.000918517361111111, -7.85276717901311, -0.00953525641025641, -0.0127136752136752, -17.2009284027778, 0, -5.23504273504273, 0.0288461538461538, -0.00082272108707265, 0.00111101575854701, -10.4702172156525, -0.0031784188034188, -0.0153311965811966, -17.2009248878205, 0, -5.23504273504273, 0.0297809829059829, -0.000523528111645299, 0.000811850827991453, -10.4702163609253, 0.078525641025641, -0.0224358974358974, 77.7782052564103, 0, -35.8974358974359, -0.0400641025641026, 0.00538520432692308, -0.00683777510683761, 41.8809408294169, -0.0291666666666667, 0.00897435897435897, 0.000116442307692308, 14.957264957265, 14.957264957265, -0.0673076923076923, 0.00329037954059829, -0.00329053098290598, 26.1752222230955, 0.0127136752136752, -0.0164529914529915, 11.9659057478632, 0, 17.9487179487179, -0.046474358974359, 0.000598385950854701, -0.000427473824786325, 20.9402637898772, 0.0590811965811966, 0.00598290598290598, 6.00534188034188E-05, -14.957264957265, 14.957264957265, -0.0560897435897436, -0.00029870219017094, -0.00119653632478632, 26.1752174790829;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(45) << -5474358.97435897, -2243589.74358974, 4307.69230769231, 0, 0, -942307.692307692, -22.8098290598291, -9.3482905982906, 0.0179487179487179, -5294871.7948718, 3141025.64102564, -4038.46153846154, 0, 0, -942307.692307692, -22.0619658119658, 13.0876068376068, -0.0168269230769231, -5294871.7948718, -3141025.64102564, -1256.41025641026, 0, 0, -942307.692307692, -22.0619658119658, -13.0876068376068, -0.00523504273504274, -5474358.97435897, 2243589.74358974, 4576.92307692308, 0, 0, -942307.692307692, -22.8098290598291, 9.3482905982906, 0.0190705128205128, 0, -8974358.97435897, -7000, 0, 0, 2692307.69230769, 0, -37.3931623931624, -0.0291666666666667, 29435897.4358974, 0, 5384.61538461539, 0, 0, 3230769.23076923, 122.649572649573, 0, 0.0224358974358974, 0, 8974358.97435897, -3769.23076923077, 0, 0, 2692307.69230769, 0, 37.3931623931624, -0.0157051282051282, -7897435.8974359, 0, 1794.8717948718, 0, 0, 3230769.23076923, -32.9059829059829, 0, 0.00747863247863248;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(46) << -2243589.74358974, -13102564.1025641, 2961.53846153846, 0, 0, 0, -9.3482905982906, -54.5940170940171, 0.0123397435897436, 5833333.33333333, -24589743.5897436, 30961.5384615385, 0, 0, 1346153.84615385, 24.3055555555556, -102.457264957265, 0.12900641025641, -5833333.33333333, -24589743.5897436, 0, 0, 0, -1346153.84615385, -24.3055555555556, -102.457264957265, 0, 2243589.74358974, -13102564.1025641, 1615.38461538462, 0, 0, 0, 9.3482905982906, -54.5940170940171, 0.00673076923076923, -8974358.97435897, 0, 2153.84615384615, 0, 0, 2692307.69230769, -37.3931623931624, 0, 0.00897435897435897, 0, 54564102.5641026, -18846.1538461538, 0, 0, 0, 0, 227.350427350427, -0.078525641025641, 8974358.97435897, 0, -14717.9487179487, 0, 0, -2692307.69230769, 37.3931623931624, 0, -0.0613247863247863, 0, 20820512.8205128, -4128.20512820513, 0, 0, 0, 0, 86.7521367521367, -0.0172008547008547;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(47) << 3051.28205128205, 2288.46153846154, -4128222.81666667, -628205.128205128, 0, 1153.84615384615, 0.0127136752136752, 0.00953525641025641, -17.2009284027778, -9153.84615384615, 25980.7692307692, -6641124.43076923, -628205.128205128, 897435.897435898, -2435.89743589744, -0.0381410256410256, 0.108253205128205, -27.6713517948718, -3320.51282051282, 44.8717948717949, -6641059.77948718, -628205.128205128, -897435.897435898, 32.0512820512821, -0.0138354700854701, 0.000186965811965812, -27.6710824145299, 3679.48717948718, 762.820512820513, -4128221.97307692, -628205.128205128, 0, 929.487179487179, 0.0153311965811966, 0.0031784188034188, -17.2009248878205, -2153.84615384615, 7000, 27.9461538461538, 1794871.7948718, 1794871.7948718, -5384.61538461539, -0.00897435897435897, 0.0291666666666667, 0.000116442307692308, 5384.61538461539, -18846.1538461538, 18666769.2615385, 2153846.15384615, 0, -3205.12820512821, 0.0224358974358974, -0.078525641025641, 77.7782052564103, -1435.89743589744, -14179.4871794872, 14.4128205128205, 1794871.7948718, -1794871.7948718, -4487.17948717949, -0.00598290598290598, -0.0590811965811966, 6.00534188034188E-05, 3948.71794871795, -3051.28205128205, 2871817.37948718, 2153846.15384615, 0, -3205.12820512821, 0.0164529914529915, -0.0127136752136752, 11.9659057478632;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(48) << 0, 0, 628205.128205128, -359019.978632479, -18.6965811965812, -133.297435897436, 0, 0, -5.23504273504273, 0, 0, 1166666.66666667, -269274.893162393, 26.1752136752137, 110.222756410256, 0, 0, -2.99145299145299, 0, 0, 1166666.66666667, -269274.893162393, -26.1752136752137, -123.087393162393, 0, 0, -2.99145299145299, 0, 0, 628205.128205128, -359019.978632479, 18.6965811965812, -97.3977564102564, 0, 0, -5.23504273504273, 0, 0, -1794871.7948718, 897435.897435898, -74.7863247863248, 394.813461538462, 0, 0, 14.957264957265, 0, 0, 2153846.15384615, 1436142.73504274, 0, 820.557692307692, 0, 0, 35.8974358974359, 0, 0, -1794871.7948718, 897435.897435898, 74.7863247863248, 143.558333333333, 0, 0, 14.957264957265, 0, 0, -2153846.15384615, 717882.905982906, 0, 51.2970085470085, 0, 0, 17.9487179487179;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(49) << 0, 0, 0, -18.6965811965812, -359083.547008547, 98.7426282051282, 0, 0, 0, 0, 0, -897435.897435898, 48.6111111111111, -269435.683760684, -152.30608974359, 0, 0, 7.47863247863248, 0, 0, 897435.897435898, -48.6111111111111, -269435.683760684, 170.512820512821, 0, 0, -7.47863247863248, 0, 0, 0, 18.6965811965812, -359083.547008547, 62.8339743589744, 0, 0, 0, 0, 0, -1794871.7948718, -74.7863247863248, 897435.897435898, -394.853846153846, 0, 0, 14.957264957265, 0, 0, 0, 0, 1436352.13675214, -646.310897435898, 0, 0, 0, 0, 0, 1794871.7948718, 74.7863247863248, 897435.897435898, 35.7747863247863, 0, 0, -14.957264957265, 0, 0, 0, 0, 718122.222222222, -71.8292735042735, 0, 0, 0;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(50) << 942307.692307692, 0, 2884.61538461538, -133.307905982906, 98.7370192307692, -1256444.98689954, -1.30876068376068, 0, 0.0216346153846154, 1750000, -1346153.84615385, 8333.33333333333, 110.180128205128, -152.347596153846, -942362.499968547, 2.05662393162393, 1.86965811965812, 0.0144230769230769, 1750000, 1346153.84615385, 2724.35897435897, -123.104594017094, 170.513194444444, -942363.409686197, 2.05662393162393, -1.86965811965812, 0.0116185897435897, 942307.692307692, 0, 3108.97435897436, -97.405235042735, 62.8268696581197, -1256444.88432841, -1.30876068376068, 0, 0.0206997863247863, -2692307.69230769, -2692307.69230769, -5384.61538461539, 394.853846153846, -394.813461538462, 3141026.66689955, 3.73931623931624, 3.73931623931624, -0.0673076923076923, 3230769.23076923, 0, -3205.12820512821, 820.557692307692, -646.310897435898, 5025798.45555581, 31.4102564102564, 0, -0.0400641025641026, -2692307.69230769, 2692307.69230769, -4487.17948717949, 143.577777777778, 35.7792735042735, 3141026.097556, 3.73931623931624, -3.73931623931624, -0.0560897435897436, -3230769.23076923, 0, -3974.35897435897, 51.3149572649573, -71.8202991452992, 2512844.81728158, 4.48717948717949, 0, -0.0432692307692308;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(51) << -22.8098290598291, -9.3482905982906, 0.0179487179487179, 0, 0, 1.30876068376068, -2.99162406517094, -7.01121794871795E-05, -0.0011109764957265, -22.0619658119658, 13.0876068376068, -0.0168269230769231, 0, 0, 5.79594017094017, -2.24375520833333, 9.81570512820513E-05, 0.000918677216880342, -22.0619658119658, -13.0876068376068, -0.00523504273504274, 0, 0, 5.79594017094017, -2.24375520833333, -9.81570512820513E-05, -0.00102568028846154, -22.8098290598291, 9.3482905982906, 0.0190705128205128, 0, 0, 1.30876068376068, -2.99162406517094, 7.01121794871795E-05, -0.000811822783119658, 0, -37.3931623931624, -0.0291666666666667, 0, 0, -3.73931623931624, 7.47863247863248, -0.000280448717948718, 0.00329037954059829, 122.649572649573, 0, 0.0224358974358974, 0, 0, 31.4102564102564, 11.9667318376068, 0, 0.00683777510683761, 0, 37.3931623931624, -0.0157051282051282, 0, 0, -3.73931623931624, 7.47863247863248, 0.000280448717948718, 0.00119646340811966, -32.9059829059829, 0, 0.00747863247863248, 0, 0, -4.48717948717949, 5.98265918803419, 0, 0.000427406517094017;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(52) << -9.3482905982906, -54.5940170940171, 0.0123397435897436, 0, 0, 0, -7.01121794871795E-05, -2.9918624465812, 0.000822742120726496, 24.3055555555556, -102.457264957265, 0.12900641025641, 0, 0, -1.86965811965812, 0.000182291666666667, -2.24435817307692, -0.0012703999732906, -24.3055555555556, -102.457264957265, 0, 0, 0, 1.86965811965812, -0.000182291666666667, -2.24435817307692, 0.00142094017094017, 9.3482905982906, -54.5940170940171, 0.00673076923076923, 0, 0, 0, 7.01121794871795E-05, -2.9918624465812, 0.000523554754273504, -37.3931623931624, 0, 0.00897435897435897, 0, 0, -3.73931623931624, -0.000280448717948718, 7.47863247863248, -0.00329053098290598, 0, 227.350427350427, -0.078525641025641, 0, 0, 0, 0, 11.9675170940171, -0.00538520432692308, 37.3931623931624, 0, -0.0613247863247863, 0, 0, 3.73931623931624, 0.000280448717948718, 7.47863247863248, 0.000298685363247863, 0, 86.7521367521367, -0.0172008547008547, 0, 0, 0, 0, 5.98355662393162, -0.000598419604700855;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(53) << 0.0127136752136752, 0.00953525641025641, -17.2009284027778, 5.23504273504273, 0, 0.0288461538461538, -0.00111101575854701, 0.00082272108707265, -10.4702172156525, -0.0381410256410256, 0.108253205128205, -27.6713517948718, 11.965811965812, -7.47863247863248, 0.0592948717948718, 0.000918517361111111, -0.00127055562232906, -7.85276717901311, -0.0138354700854701, 0.000186965811965812, -27.6710824145299, 11.965811965812, 7.47863247863248, 0.0228365384615385, -0.00102574479166667, 0.00142094157318376, -7.85277476246284, 0.0153311965811966, 0.0031784188034188, -17.2009248878205, 5.23504273504273, 0, 0.0297809829059829, -0.000811850827991453, 0.000523528111645299, -10.4702163609253, -0.00897435897435897, 0.0291666666666667, 0.000116442307692308, -14.957264957265, -14.957264957265, -0.0673076923076923, 0.00329053098290598, -0.00329037954059829, 26.1752222230955, 0.0224358974358974, -0.078525641025641, 77.7782052564103, 35.8974358974359, 0, -0.0400641025641026, 0.00683777510683761, -0.00538520432692308, 41.8809408294169, -0.00598290598290598, -0.0590811965811966, 6.00534188034188E-05, -14.957264957265, 14.957264957265, -0.0560897435897436, 0.00119653632478632, 0.00029870219017094, 26.1752174790829, 0.0164529914529915, -0.0127136752136752, 11.9659057478632, -17.9487179487179, 0, -0.046474358974359, 0.000427473824786325, -0.000598385950854701, 20.9402637898772;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(54) << -13102564.1025641, -2243589.74358974, 0, 0, 0, 0, -54.5940170940171, -9.3482905982906, 0, -13102564.1025641, 2243589.74358974, -9782.05128205128, 0, 0, 0, -54.5940170940171, 9.3482905982906, -0.040758547008547, -24589743.5897436, -5833333.33333333, -2961.53846153846, 0, 0, -1346153.84615385, -102.457264957265, -24.3055555555556, -0.0123397435897436, -24589743.5897436, 5833333.33333333, 1615.38461538462, 0, 0, 1346153.84615385, -102.457264957265, 24.3055555555556, 0.00673076923076923, 20820512.8205128, 0, 3051.28205128205, 0, 0, 0, 86.7521367521367, 0, 0.0127136752136752, 0, 8974358.97435897, -1435.89743589744, 0, 0, -2692307.69230769, 0, 37.3931623931624, -0.00598290598290598, 54564102.5641026, 0, -897.435897435897, 0, 0, 0, 227.350427350427, 0, -0.00373931623931624, 0, -8974358.97435897, 10410.2564102564, 0, 0, 2692307.69230769, 0, -37.3931623931624, 0.0433760683760684;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(55) << -2243589.74358974, -5474358.97435897, 7000, 0, 0, -942307.692307692, -9.3482905982906, -22.8098290598291, 0.0291666666666667, 2243589.74358974, -5474358.97435897, 14987.1794871794, 0, 0, -942307.692307692, 9.3482905982906, -22.8098290598291, 0.0624465811965812, -3141025.64102564, -5294871.7948718, 4307.69230769231, 0, 0, -942307.692307692, -13.0876068376068, -22.0619658119658, 0.0179487179487179, 3141025.64102564, -5294871.7948718, 4576.92307692308, 0, 0, -942307.692307692, 13.0876068376068, -22.0619658119658, 0.0190705128205128, 0, -7897435.8974359, -3948.71794871795, 0, 0, 3230769.23076923, 0, -32.9059829059829, -0.0164529914529915, 8974358.97435897, 0, -14179.4871794872, 0, 0, 2692307.69230769, 37.3931623931624, 0, -0.0590811965811966, 0, 29435897.4358974, -1794.87179487179, 0, 0, 3230769.23076923, 0, 122.649572649573, -0.00747863247863248, -8974358.97435897, 0, -10948.7179487179, 0, 0, 2692307.69230769, -37.3931623931624, 0, -0.0456196581196581;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(56) << 44.8717948717949, 7000, -4128217.11794872, 0, -628205.128205128, 865.384615384615, 0.000186965811965812, 0.0291666666666667, -17.2009046581197, -9557.69230769231, 14628.2051282051, -4128242.3, 0, -628205.128205128, -256.410256410256, -0.0398237179487179, 0.0609508547008547, -17.2010095833333, -1660.25641025641, 3679.48717948718, -6641037.27307692, -897435.897435898, -628205.128205128, 1089.74358974359, -0.00691773504273504, 0.0153311965811966, -27.6709886378205, 762.820512820513, 3410.25641025641, -6641038.56538462, 897435.897435898, -628205.128205128, 416.666666666667, 0.0031784188034188, 0.0142094017094017, -27.6709940224359, 4128.20512820513, -1794.8717948718, 2871817.37948718, 0, 2153846.15384615, -3974.35897435897, 0.0172008547008547, -0.00747863247863248, 11.9659057478632, -3769.23076923077, -14717.9487179487, 14.4128205128205, -1794871.7948718, 1794871.7948718, -4487.17948717949, -0.0157051282051282, -0.0613247863247863, 6.00534188034188E-05, -897.435897435897, -1794.87179487179, 18666693.825641, 0, 2153846.15384615, -384.615384615385, -0.00373931623931624, -0.00747863247863248, 77.777890940171, 10948.7179487179, -10410.2564102564, 9.63846153846154, 1794871.7948718, 1794871.7948718, -3589.74358974359, 0.0456196581196581, -0.0433760683760684, 4.01602564102564E-05;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(57) << 0, 0, 0, -359083.547008547, -18.6965811965812, -8.97435897435897, 0, 0, 0, 0, 0, 0, -359083.547008547, 18.6965811965812, 26.8415598290598, 0, 0, 0, 0, 0, 897435.897435898, -269435.683760684, -48.6111111111111, 8.94967948717949, 0, 0, -7.47863247863248, 0, 0, -897435.897435898, -269435.683760684, 48.6111111111111, -26.9096153846154, 0, 0, 7.47863247863248, 0, 0, 0, 718122.222222222, 0, 71.8202991452992, 0, 0, 0, 0, 0, 1794871.7948718, 897435.897435898, 74.7863247863248, 143.577777777778, 0, 0, -14.957264957265, 0, 0, 0, 1436352.13675214, 0, -71.8023504273504, 0, 0, 0, 0, 0, -1794871.7948718, 897435.897435898, -74.7863247863248, -143.502991452991, 0, 0, 14.957264957265;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(58) << 0, 0, 628205.128205128, -18.6965811965812, -359019.978632479, -46.0955128205128, 0, 0, -5.23504273504273, 0, 0, 628205.128205128, 18.6965811965812, -359019.978632479, -81.9263888888889, 0, 0, -5.23504273504273, 0, 0, 1166666.66666667, -26.1752136752137, -269274.893162393, 33.3692307692308, 0, 0, -2.99145299145299, 0, 0, 1166666.66666667, 26.1752136752137, -269274.893162393, -20.4746794871795, 0, 0, -2.99145299145299, 0, 0, -2153846.15384615, 0, 717882.905982906, -51.3149572649573, 0, 0, 17.9487179487179, 0, 0, -1794871.7948718, 74.7863247863248, 897435.897435898, 35.7792735042735, 0, 0, 14.957264957265, 0, 0, 2153846.15384615, 0, 1436142.73504274, 615.369658119658, 0, 0, 35.8974358974359, 0, 0, -1794871.7948718, -74.7863247863248, 897435.897435898, 143.498504273504, 0, 0, 14.957264957265;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(59) << 0, 942307.692307692, 2275.64102564103, -8.97398504273504, -46.0955128205128, -1256444.80480077, 0, -1.30876068376068, 0.0166933760683761, 0, 942307.692307692, 3397.4358974359, 26.8434294871795, -81.9293803418804, -1256444.70244652, 0, -1.30876068376068, 0.0120192307692308, 1346153.84615385, 1750000, 2884.61538461538, 8.96052350427351, 33.3639957264957, -942363.119755053, -1.86965811965812, 2.05662393162393, 0.0211004273504274, -1346153.84615385, 1750000, 3108.97435897436, -26.9167200854701, -20.4844017094017, -942363.147970951, 1.86965811965812, 2.05662393162393, 0.0164262820512821, 0, -3230769.23076923, -3205.12820512821, 71.8292735042735, -51.2970085470085, 2512844.81728158, 0, 4.48717948717949, -0.046474358974359, 2692307.69230769, -2692307.69230769, -4487.17948717949, 143.558333333333, 35.7747863247863, 3141026.097556, -3.73931623931624, 3.73931623931624, -0.0560897435897436, 0, 3230769.23076923, -384.615384615385, -71.8023504273504, 615.369658119658, 5025797.20364513, 0, 31.4102564102564, -0.00480769230769231, -2692307.69230769, -2692307.69230769, -3589.74358974359, -143.498504273504, 143.502991452991, 3141025.93854186, 3.73931623931624, 3.73931623931624, -0.0448717948717949;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(60) << -54.5940170940171, -9.3482905982906, 0, 0, 0, 0, -2.9918624465812, -7.01121794871795E-05, -7.47863247863248E-05, -54.5940170940171, 9.3482905982906, -0.040758547008547, 0, 0, 0, -2.9918624465812, 7.01121794871795E-05, 0.00022405328525641, -102.457264957265, -24.3055555555556, -0.0123397435897436, 0, 0, 1.86965811965812, -2.24435817307692, -0.000182291666666667, 7.46937767094017E-05, -102.457264957265, 24.3055555555556, 0.00673076923076923, 0, 0, -1.86965811965812, -2.24435817307692, 0.000182291666666667, -0.000224308493589744, 86.7521367521367, 0, 0.0127136752136752, 0, 0, 0, 5.98355662393162, 0, 0.000598385950854701, 0, 37.3931623931624, -0.00598290598290598, 0, 0, 3.73931623931624, 7.47863247863248, 0.000280448717948718, 0.00119653632478632, 227.350427350427, 0, -0.00373931623931624, 0, 0, 0, 11.9675170940171, 0, -0.000598318643162393, 0, -37.3931623931624, 0.0433760683760684, 0, 0, -3.73931623931624, 7.47863247863248, -0.000280448717948718, -0.00119625587606838;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(61) << -9.3482905982906, -22.8098290598291, 0.0291666666666667, 0, 0, 1.30876068376068, -7.01121794871795E-05, -2.99162406517094, -0.000384396634615385, 9.3482905982906, -22.8098290598291, 0.0624465811965812, 0, 0, 1.30876068376068, 7.01121794871795E-05, -2.99162406517094, -0.000683292334401709, -13.0876068376068, -22.0619658119658, 0.0179487179487179, 0, 0, 5.79594017094017, -9.81570512820513E-05, -2.24375520833333, 0.000277912393162393, 13.0876068376068, -22.0619658119658, 0.0190705128205128, 0, 0, 5.79594017094017, 9.81570512820513E-05, -2.24375520833333, -0.000170797142094017, 0, -32.9059829059829, -0.0164529914529915, 0, 0, -4.48717948717949, 0, 5.98265918803419, -0.000427473824786325, 37.3931623931624, 0, -0.0590811965811966, 0, 0, -3.73931623931624, 0.000280448717948718, 7.47863247863248, 0.00029870219017094, 0, 122.649572649573, -0.00747863247863248, 0, 0, 31.4102564102564, 0, 11.9667318376068, 0.00512814903846154, -37.3931623931624, 0, -0.0456196581196581, 0, 0, -3.73931623931624, -0.000280448717948718, 7.47863247863248, 0.0011962390491453;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(62) << 0.000186965811965812, 0.0291666666666667, -17.2009046581197, 0, 5.23504273504273, 0.0225694444444444, -0.000074784922542735, -0.000384396634615385, -10.4702156983804, -0.0398237179487179, 0.0609508547008547, -17.2010095833333, 0, 5.23504273504273, 0.0272435897435897, 0.000224060296474359, -0.000683303552350427, -10.4702148444665, -0.00691773504273504, 0.0153311965811966, -27.6709886378205, 7.47863247863248, 11.965811965812, 0.0285790598290598, 7.47344417735043E-05, 0.000277892761752137, -7.8527723472296, 0.0031784188034188, 0.0142094017094017, -27.6709940224359, -7.47863247863248, 11.965811965812, 0.0276442307692308, -0.000224335136217949, -0.00017083360042735, -7.85277258231272, 0.0172008547008547, -0.00747863247863248, 11.9659057478632, 0, -17.9487179487179, -0.0432692307692308, 0.000598419604700855, -0.000427406517094017, 20.9402637898772, -0.0157051282051282, -0.0613247863247863, 6.00534188034188E-05, 14.957264957265, -14.957264957265, -0.0560897435897436, 0.00119646340811966, 0.000298685363247863, 26.1752174790829, -0.00373931623931624, -0.00747863247863248, 77.777890940171, 0, 35.8974358974359, -0.00480769230769231, -0.000598318643162393, 0.00512814903846154, 41.8809303997091, 0.0456196581196581, -0.0433760683760684, 4.01602564102564E-05, -14.957264957265, -14.957264957265, -0.0448717948717949, -0.0011962390491453, 0.00119625587606838, 26.1752161541474;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(63) << -5294871.7948718, -3141025.64102564, -4307.69230769231, 0, 0, 942307.692307692, -22.0619658119658, -13.0876068376068, -0.0179487179487179, -5474358.97435897, 2243589.74358974, -14987.1794871794, 0, 0, 942307.692307692, -22.8098290598291, 9.3482905982906, -0.0624465811965812, -5474358.97435897, -2243589.74358974, -7000, 0, 0, 942307.692307692, -22.8098290598291, -9.3482905982906, -0.0291666666666667, -5294871.7948718, 3141025.64102564, -4576.92307692308, 0, 0, 942307.692307692, -22.0619658119658, 13.0876068376068, -0.0190705128205128, 0, 8974358.97435897, 14179.4871794872, 0, 0, -2692307.69230769, 0, 37.3931623931624, 0.0590811965811966, -7897435.8974359, 0, 3948.71794871795, 0, 0, -3230769.23076923, -32.9059829059829, 0, 0.0164529914529915, 0, -8974358.97435897, 10948.7179487179, 0, 0, -2692307.69230769, 0, -37.3931623931624, 0.0456196581196581, 29435897.4358974, 0, 1794.87179487179, 0, 0, -3230769.23076923, 122.649572649573, 0, 0.00747863247863248;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(64) << -5833333.33333333, -24589743.5897436, 2961.53846153846, 0, 0, 1346153.84615385, -24.3055555555556, -102.457264957265, 0.0123397435897436, 2243589.74358974, -13102564.1025641, 9782.05128205128, 0, 0, 0, 9.3482905982906, -54.5940170940171, 0.040758547008547, -2243589.74358974, -13102564.1025641, 0, 0, 0, 0, -9.3482905982906, -54.5940170940171, 0, 5833333.33333333, -24589743.5897436, -1615.38461538462, 0, 0, -1346153.84615385, 24.3055555555556, -102.457264957265, -0.00673076923076923, 8974358.97435897, 0, 1435.89743589744, 0, 0, 2692307.69230769, 37.3931623931624, 0, 0.00598290598290598, 0, 20820512.8205128, -3051.28205128205, 0, 0, 0, 0, 86.7521367521367, -0.0127136752136752, -8974358.97435897, 0, -10410.2564102564, 0, 0, -2692307.69230769, -37.3931623931624, 0, -0.0433760683760684, 0, 54564102.5641026, 897.435897435897, 0, 0, 0, 0, 227.350427350427, 0.00373931623931624;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(65) << -3679.48717948718, 1660.25641025641, -6641037.27307692, 628205.128205128, 897435.897435898, 1089.74358974359, -0.0153311965811966, 0.00691773504273504, -27.6709886378205, -14628.2051282051, 9557.69230769231, -4128242.3, 628205.128205128, 0, -256.410256410256, -0.0609508547008547, 0.0398237179487179, -17.2010095833333, -7000, -44.8717948717949, -4128217.11794872, 628205.128205128, 0, 865.384615384615, -0.0291666666666667, -0.000186965811965812, -17.2009046581197, -3410.25641025641, -762.820512820513, -6641038.56538462, 628205.128205128, -897435.897435898, 416.666666666667, -0.0142094017094017, -0.0031784188034188, -27.6709940224359, 14717.9487179487, 3769.23076923077, 14.4128205128205, -1794871.7948718, 1794871.7948718, -4487.17948717949, 0.0613247863247863, 0.0157051282051282, 6.00534188034188E-05, 1794.8717948718, -4128.20512820513, 2871817.37948718, -2153846.15384615, 0, -3974.35897435897, 0.00747863247863248, -0.0172008547008547, 11.9659057478632, 10410.2564102564, -10948.7179487179, 9.63846153846154, -1794871.7948718, -1794871.7948718, -3589.74358974359, 0.0433760683760684, -0.0456196581196581, 4.01602564102564E-05, 1794.87179487179, 897.435897435897, 18666693.825641, -2153846.15384615, 0, -384.615384615385, 0.00747863247863248, 0.00373931623931624, 77.777890940171;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(66) << 0, 0, -1166666.66666667, -269274.893162393, -26.1752136752137, -33.3692307692308, 0, 0, 2.99145299145299, 0, 0, -628205.128205128, -359019.978632479, 18.6965811965812, 81.9263888888889, 0, 0, 5.23504273504273, 0, 0, -628205.128205128, -359019.978632479, -18.6965811965812, 46.0955128205128, 0, 0, 5.23504273504273, 0, 0, -1166666.66666667, -269274.893162393, 26.1752136752137, 20.4746794871795, 0, 0, 2.99145299145299, 0, 0, 1794871.7948718, 897435.897435898, 74.7863247863248, -35.7792735042735, 0, 0, -14.957264957265, 0, 0, 2153846.15384615, 717882.905982906, 0, 51.3149572649573, 0, 0, -17.9487179487179, 0, 0, 1794871.7948718, 897435.897435898, -74.7863247863248, -143.498504273504, 0, 0, -14.957264957265, 0, 0, -2153846.15384615, 1436142.73504274, 0, -615.369658119658, 0, 0, -35.8974358974359;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(67) << 0, 0, -897435.897435898, -48.6111111111111, -269435.683760684, -8.94967948717949, 0, 0, 7.47863247863248, 0, 0, 0, 18.6965811965812, -359083.547008547, -26.8415598290598, 0, 0, 0, 0, 0, 0, -18.6965811965812, -359083.547008547, 8.97435897435897, 0, 0, 0, 0, 0, 897435.897435898, 48.6111111111111, -269435.683760684, 26.9096153846154, 0, 0, -7.47863247863248, 0, 0, -1794871.7948718, 74.7863247863248, 897435.897435898, -143.577777777778, 0, 0, 14.957264957265, 0, 0, 0, 0, 718122.222222222, -71.8202991452992, 0, 0, 0, 0, 0, 1794871.7948718, -74.7863247863248, 897435.897435898, 143.502991452991, 0, 0, -14.957264957265, 0, 0, 0, 0, 1436352.13675214, 71.8023504273504, 0, 0, 0;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(68) << -1750000, -1346153.84615385, 2884.61538461538, -33.3639957264957, -8.96052350427351, -942363.119755053, -2.05662393162393, 1.86965811965812, 0.0211004273504274, -942307.692307692, 0, 3397.4358974359, 81.9293803418804, -26.8434294871795, -1256444.70244652, 1.30876068376068, 0, 0.0120192307692308, -942307.692307692, 0, 2275.64102564103, 46.0955128205128, 8.97398504273504, -1256444.80480077, 1.30876068376068, 0, 0.0166933760683761, -1750000, 1346153.84615385, 3108.97435897436, 20.4844017094017, 26.9167200854701, -942363.147970951, -2.05662393162393, -1.86965811965812, 0.0164262820512821, 2692307.69230769, -2692307.69230769, -4487.17948717949, -35.7747863247863, -143.558333333333, 3141026.097556, -3.73931623931624, 3.73931623931624, -0.0560897435897436, 3230769.23076923, 0, -3205.12820512821, 51.2970085470085, -71.8292735042735, 2512844.81728158, -4.48717948717949, 0, -0.046474358974359, 2692307.69230769, 2692307.69230769, -3589.74358974359, -143.502991452991, 143.498504273504, 3141025.93854186, -3.73931623931624, -3.73931623931624, -0.0448717948717949, -3230769.23076923, 0, -384.615384615385, -615.369658119658, 71.8023504273504, 5025797.20364513, -31.4102564102564, 0, -0.00480769230769231;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(69) << -22.0619658119658, -13.0876068376068, -0.0179487179487179, 0, 0, -5.79594017094017, -2.24375520833333, -9.81570512820513E-05, -0.000277912393162393, -22.8098290598291, 9.3482905982906, -0.0624465811965812, 0, 0, -1.30876068376068, -2.99162406517094, 7.01121794871795E-05, 0.000683292334401709, -22.8098290598291, -9.3482905982906, -0.0291666666666667, 0, 0, -1.30876068376068, -2.99162406517094, -7.01121794871795E-05, 0.000384396634615385, -22.0619658119658, 13.0876068376068, -0.0190705128205128, 0, 0, -5.79594017094017, -2.24375520833333, 9.81570512820513E-05, 0.000170797142094017, 0, 37.3931623931624, 0.0590811965811966, 0, 0, 3.73931623931624, 7.47863247863248, 0.000280448717948718, -0.00029870219017094, -32.9059829059829, 0, 0.0164529914529915, 0, 0, 4.48717948717949, 5.98265918803419, 0, 0.000427473824786325, 0, -37.3931623931624, 0.0456196581196581, 0, 0, 3.73931623931624, 7.47863247863248, -0.000280448717948718, -0.0011962390491453, 122.649572649573, 0, 0.00747863247863248, 0, 0, -31.4102564102564, 11.9667318376068, 0, -0.00512814903846154;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(70) << -24.3055555555556, -102.457264957265, 0.0123397435897436, 0, 0, -1.86965811965812, -0.000182291666666667, -2.24435817307692, -7.46937767094017E-05, 9.3482905982906, -54.5940170940171, 0.040758547008547, 0, 0, 0, 7.01121794871795E-05, -2.9918624465812, -0.00022405328525641, -9.3482905982906, -54.5940170940171, 0, 0, 0, 0, -7.01121794871795E-05, -2.9918624465812, 7.47863247863248E-05, 24.3055555555556, -102.457264957265, -0.00673076923076923, 0, 0, 1.86965811965812, 0.000182291666666667, -2.24435817307692, 0.000224308493589744, 37.3931623931624, 0, 0.00598290598290598, 0, 0, -3.73931623931624, 0.000280448717948718, 7.47863247863248, -0.00119653632478632, 0, 86.7521367521367, -0.0127136752136752, 0, 0, 0, 0, 5.98355662393162, -0.000598385950854701, -37.3931623931624, 0, -0.0433760683760684, 0, 0, 3.73931623931624, -0.000280448717948718, 7.47863247863248, 0.00119625587606838, 0, 227.350427350427, 0.00373931623931624, 0, 0, 0, 0, 11.9675170940171, 0.000598318643162393;
    //Expected_JacobianR_SmallDispNoVelWithDamping.row(71) << -0.0153311965811966, 0.00691773504273504, -27.6709886378205, -11.965811965812, -7.47863247863248, 0.0285790598290598, -0.000277892761752137, -7.47344417735043E-05, -7.8527723472296, -0.0609508547008547, 0.0398237179487179, -17.2010095833333, -5.23504273504273, 0, 0.0272435897435897, 0.000683303552350427, -0.000224060296474359, -10.4702148444665, -0.0291666666666667, -0.000186965811965812, -17.2009046581197, -5.23504273504273, 0, 0.0225694444444444, 0.000384396634615385, 0.000074784922542735, -10.4702156983804, -0.0142094017094017, -0.0031784188034188, -27.6709940224359, -11.965811965812, 7.47863247863248, 0.0276442307692308, 0.00017083360042735, 0.000224335136217949, -7.85277258231272, 0.0613247863247863, 0.0157051282051282, 6.00534188034188E-05, 14.957264957265, -14.957264957265, -0.0560897435897436, -0.000298685363247863, -0.00119646340811966, 26.1752174790829, 0.00747863247863248, -0.0172008547008547, 11.9659057478632, 17.9487179487179, 0, -0.0432692307692308, 0.000427406517094017, -0.000598419604700855, 20.9402637898772, 0.0433760683760684, -0.0456196581196581, 4.01602564102564E-05, 14.957264957265, 14.957264957265, -0.0448717948717949, -0.00119625587606838, 0.0011962390491453, 26.1752161541474, 0.00747863247863248, 0.00373931623931624, 77.777890940171, -35.8974358974359, 0, -0.00480769230769231, -0.00512814903846154, 0.000598318643162393, 41.8809303997091;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_Jacobians;
    Expected_Jacobians.resize(2 * 72, 72);
    if (!load_validation_data("UT_ANCFShell_3833_JacSmallDispNoVelWithDamping.txt", Expected_Jacobians))
        return false;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianK_SmallDispNoVelWithDamping;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianR_SmallDispNoVelWithDamping;
    Expected_JacobianK_SmallDispNoVelWithDamping = Expected_Jacobians.block(0, 0, Expected_Jacobians.cols(), Expected_Jacobians.cols());
    Expected_JacobianR_SmallDispNoVelWithDamping = Expected_Jacobians.block(Expected_Jacobians.cols(), 0, Expected_Jacobians.cols(), Expected_Jacobians.cols());


    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation

    //Setup the test conditions
    ChVector<double> OriginalPos = m_nodeB->GetPos();
    m_nodeB->SetPos(ChVector<>(m_nodeB->GetPos().x(), m_nodeB->GetPos().y(), 0.001));
    m_element->SetAlphaDamp(0.01);

    ChVectorDynamic<double> InternalForceSmallDispNoVelNoGravity;
    InternalForceSmallDispNoVelNoGravity.resize(72);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceSmallDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_SmallDispNoVelWithDamping;
    JacobianK_SmallDispNoVelWithDamping.resize(72, 72);
    m_element->ComputeKRMmatricesGlobal(JacobianK_SmallDispNoVelWithDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_SmallDispNoVelWithDamping;
    JacobianR_SmallDispNoVelWithDamping.resize(72, 72);
    m_element->ComputeKRMmatricesGlobal(JacobianR_SmallDispNoVelWithDamping, 0, 1, 0);

    //Reset the element conditions back to its original values
    m_nodeB->SetPos(OriginalPos);
    m_element->SetAlphaDamp(0.0);


    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_SmallDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(72, 72);

    double small_terms_JacR = 1e-4*Expected_JacobianR_SmallDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacR = 0;
    double max_percent_error_JacR = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacR;
    percent_error_matrix_JacR.resize(72, 72);


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

    //ChMatrixNM<double, 72, 72> Expected_JacobianK_NoDispSmallVelWithDamping;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(0) << 2100000000, 953525641.025641, -6147.4358974359, 134.615384615385, 0, -80769230.7692308, 8750, 3973.0235042735, -0.0256143162393162, 1032051282.05128, -33653846.1538461, 6842.94871794872, 134.615384615385, 0, 53846153.8461538, 4300.21367521368, -140.224358974359, 0.0285122863247863, 928846153.846154, 392628205.128205, 426.282051282051, -269.230769230769, 0, 20192307.6923077, 3870.19230769231, 1635.95085470085, 0.00177617521367521, 785256410.25641, 33653846.1538461, -493.589743589744, -269.230769230769, 0, 20192307.6923077, 3271.90170940171, 140.224358974359, -0.00205662393162393, -2458974358.97436, -314102564.102564, -44.8717948717949, 1089.74358974359, 0, -134615384.615385, -10245.7264957264, -1308.76068376068, -0.000186965811965812, -547435897.435898, -224358974.358974, 3051.28205128205, 1153.84615384615, 0, 94230769.2307692, -2280.98290598291, -934.82905982906, 0.0127136752136752, -1310256410.25641, -224358974.358974, 44.8717948717949, 910.25641025641, 0, 0, -5459.40170940171, -934.82905982906, 0.000186965811965812, -529487179.48718, -583333333.333333, -3679.48717948718, 1153.84615384615, 0, -175000000, -2206.19658119658, -2430.55555555556, -0.0153311965811966;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(1) << 953525641.025641, 2100000000, -6663.46153846154, 0, 134.615384615385, -80769230.7692308, 3973.0235042735, 8750, -0.0277644230769231, 33653846.1538461, 785256410.25641, -6483.97435897436, 0, 134.615384615385, 20192307.6923077, 140.224358974359, 3271.90170940171, -0.0270165598290598, 392628205.128205, 928846153.846154, -628.205128205128, 0, -269.230769230769, 20192307.6923077, 1635.95085470085, 3870.19230769231, -0.00261752136752137, -33653846.1538461, 1032051282.05128, -493.589743589744, 0, -269.230769230769, 53846153.8461538, -140.224358974359, 4300.21367521368, -0.00205662393162393, -583333333.333333, -529487179.48718, 3320.51282051282, 0, 1089.74358974359, -175000000, -2430.55555555556, -2206.19658119658, 0.0138354700854701, -224358974.358974, -1310256410.25641, 2288.46153846154, 0, 1153.84615384615, 0, -934.82905982906, -5459.40170940171, 0.00953525641025641, -224358974.358974, -547435897.435898, 7000, 0, 910.25641025641, 94230769.2307692, -934.82905982906, -2280.98290598291, 0.0291666666666667, -314102564.102564, -2458974358.97436, 1660.25641025641, 0, 1153.84615384615, -134615384.615385, -1308.76068376068, -10245.7264957264, 0.00691773504273504;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(2) << 0, 0, 933333333.333333, -53846153.8461539, -53846153.8461539, 269.230769230769, 0, 0, 3888.88888888889, 0, 0, 403846153.846154, 35897435.8974359, 13461538.4615385, 269.230769230769, 0, 0, 1682.69230769231, 0, 0, 412820512.820513, 13461538.4615385, 13461538.4615385, -538.461538461539, 0, 0, 1720.08547008547, 0, 0, 403846153.846154, 13461538.4615385, 35897435.8974359, -538.461538461539, 0, 0, 1682.69230769231, 0, 0, -664102564.102564, -89743589.7435897, -116666666.666667, 2179.48717948718, 0, 0, -2767.09401709402, 0, 0, -412820512.820513, 62820512.8205128, 0, 2307.69230769231, 0, 0, -1720.08547008547, 0, 0, -412820512.820513, 0, 62820512.8205128, 1820.51282051282, 0, 0, -1720.08547008547, 0, 0, -664102564.102564, -116666666.666667, -89743589.7435897, 2307.69230769231, 0, 0, -2767.09401709402;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(3) << 134.615384615385, 0, -53846153.8461539, 26940576.9230769, 7946.04700854701, -55.1794337606838, 0.00168269230769231, 0, -897.435897435898, -730.769230769231, 0, -35897435.8974359, 8982959.4017094, -280.448717948718, -27.507077991453, -0.00192307692307692, 0, 299.145299145299, -269.230769230769, 0, -13461538.4615385, 13469278.8461538, 3271.90170940171, 26.2856036324786, -0.00336538461538462, 0, 112.179487179487, -391.025641025641, 0, 13461538.4615385, 8980902.77777778, 280.448717948718, 43.5856303418803, -0.00387286324786325, 0, 224.358974358974, 12.8205128205128, 0, 89743589.7435897, -26943568.3760684, -2617.52136752137, -170.513194444444, 0.00913461538461539, 0, -747.863247863248, 461.538461538462, 0, -62820512.8205128, -35901997.8632479, -1869.65811965812, -133.307905982906, 0.0115384615384615, 0, 523.504273504273, 346.153846153846, 0, 0, -35908354.7008547, -1869.65811965812, -8.97398504273504, 0.00902777777777778, 0, 0, 435.897435897436, 0, 62820512.8205128, -26927489.3162393, -4861.11111111111, -33.3639957264957, 0.0114316239316239, 0, -1196.5811965812;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(4) << 0, 134.615384615385, -53846153.8461539, 7946.04700854701, 26940576.9230769, -34.6709134615385, 0, 0.00168269230769231, -897.435897435898, 0, -730.769230769231, 13461538.4615385, 280.448717948718, 8980902.77777778, 46.099813034188, 0, -0.00192307692307692, 224.358974358974, 0, -269.230769230769, -13461538.4615385, 3271.90170940171, 13469278.8461538, -26.2872863247863, 0, -0.00336538461538462, 112.179487179487, 0, -391.025641025641, -35897435.8974359, -280.448717948718, 8982959.4017094, -17.3118055555556, 0, -0.00387286324786325, 299.145299145299, 0, 12.8205128205128, 62820512.8205128, -4861.11111111111, -26927489.3162393, 123.104594017094, 0, 0.00913461538461539, -1196.5811965812, 0, 461.538461538462, 0, -1869.65811965812, -35908354.7008547, 98.7370192307692, 0, 0.0115384615384615, 0, 0, 346.153846153846, -62820512.8205128, -1869.65811965812, -35901997.8632479, -46.0955128205128, 0, 0.00902777777777778, 523.504273504273, 0, 435.897435897436, 89743589.7435897, -2617.52136752137, -26943568.3760684, -8.96052350427351, 0, 0.0114316239316239, -747.863247863248;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(5) << -80769230.7692308, -80769230.7692308, 336.538461538462, 0, 0, 94238547.008547, -785.25641025641, -785.25641025641, 0.00364583333333333, -53846153.8461538, 20192307.6923077, -1826.92307692308, 0, 0, 31413621.7948718, 74.7863247863248, 196.314102564103, -0.00536858974358974, -20192307.6923077, -20192307.6923077, -673.076923076923, 0, 0, 47118824.7863248, 28.0448717948718, 28.0448717948718, -0.00729166666666667, 20192307.6923077, -53846153.8461538, -977.564102564103, 0, 0, 31413621.7948718, 196.314102564103, 74.7863247863248, -0.00856036324786325, 134615384.615385, 94230769.2307692, 32.0512820512821, 0, 0, -94236303.4188034, -186.965811965812, -579.594017094017, 0.0182959401709402, -94230769.2307692, 0, 1153.84615384615, 0, 0, -125644465.811966, 130.876068376068, 0, 0.0240384615384615, 0, -94230769.2307692, 865.384615384615, 0, 0, -125644465.811966, 0, 130.876068376068, 0.0187767094017094, 94230769.2307692, 134615384.615385, 1089.74358974359, 0, 0, -94236303.4188034, -579.594017094017, -186.965811965812, 0.0237713675213675;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(6) << 8750, 3973.0235042735, -0.0256143162393162, 0.00168269230769231, 0, -785.25641025641, 224.424599358974, 0.0297976762820513, -0.000459593816773504, 4300.21367521368, -140.224358974359, 0.0285122863247863, -0.00552884615384615, 0, -74.7863247863248, 74.8185763888889, -0.00105168269230769, -0.000229487012553419, 3870.19230769231, 1635.95085470085, 0.00177617521367521, -0.00336538461538462, 0, -28.0448717948718, 112.208513621795, 0.0122696314102564, 0.000219030415331197, 3271.90170940171, 140.224358974359, -0.00205662393162393, -0.00438034188034188, 0, 196.314102564103, 74.8108640491453, 0.00105168269230769, 0.000363232438568376, -10245.7264957264, -1308.76068376068, -0.000186965811965812, 0.0046474358974359, 0, 186.965811965812, -224.435817307692, -0.00981570512820513, -0.00142094157318376, -2280.98290598291, -934.82905982906, 0.0127136752136752, 0.00865384615384615, 0, -130.876068376068, -299.162406517094, -0.00701121794871795, -0.00111101575854701, -5459.40170940171, -934.82905982906, 0.000186965811965812, 0.00667735042735043, 0, 0, -299.18624465812, -0.00701121794871795, -0.000074784922542735, -2206.19658119658, -2430.55555555556, -0.0153311965811966, 0.00844017094017094, 0, -205.662393162393, -224.375520833333, -0.0182291666666667, -0.000277892761752137;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(7) << 3973.0235042735, 8750, -0.0277644230769231, 0, 0.00168269230769231, -785.25641025641, 0.0297976762820513, 224.424599358974, -0.000288669771634615, 140.224358974359, 3271.90170940171, -0.0270165598290598, 0, -0.00552884615384615, 196.314102564103, 0.00105168269230769, 74.8108640491453, 0.000384412760416667, 1635.95085470085, 3870.19230769231, -0.00261752136752137, 0, -0.00336538461538462, -28.0448717948718, 0.0122696314102564, 112.208513621795, -0.00021903672542735, -140.224358974359, 4300.21367521368, -0.00205662393162393, 0, -0.00438034188034188, -74.7863247863248, -0.00105168269230769, 74.8185763888889, -0.000144246193910256, -2430.55555555556, -2206.19658119658, 0.0138354700854701, 0, 0.0046474358974359, -205.662393162393, -0.0182291666666667, -224.375520833333, 0.00102574479166667, -934.82905982906, -5459.40170940171, 0.00953525641025641, 0, 0.00865384615384615, 0, -0.00701121794871795, -299.18624465812, 0.00082272108707265, -934.82905982906, -2280.98290598291, 0.0291666666666667, 0, 0.00667735042735043, -130.876068376068, -0.00701121794871795, -299.162406517094, -0.000384396634615385, -1308.76068376068, -10245.7264957264, 0.00691773504273504, 0, 0.00844017094017094, 186.965811965812, -0.00981570512820513, -224.435817307692, -7.47344417735043E-05;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(8) << 0, 0, 3888.88888888889, -897.435897435898, -897.435897435898, 0.00392628205128205, 0, 0, 785.285576923077, 0, 0, 1682.69230769231, -299.145299145299, 224.358974358974, -0.0141025641025641, 0, 0, 261.764756944444, 0, 0, 1720.08547008547, -112.179487179487, -112.179487179487, -0.0078525641025641, 0, 0, 392.641105769231, 0, 0, 1682.69230769231, 224.358974358974, -299.145299145299, -0.0103899572649573, 0, 0, 261.764756944444, 0, 0, -2767.09401709402, 747.863247863248, 299.145299145299, 0.0093482905982906, 0, 0, -785.277163461538, 0, 0, -1720.08547008547, -523.504273504273, 0, 0.0192307692307692, 0, 0, -1047.02144764957, 0, 0, -1720.08547008547, 0, -523.504273504273, 0.0147970085470085, 0, 0, -1047.02144764957, 0, 0, -2767.09401709402, 299.145299145299, 747.863247863248, 0.0186965811965812, 0, 0, -785.277163461538;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(9) << 1032051282.05128, 33653846.1538461, 6147.4358974359, -730.769230769231, 0, -53846153.8461538, 4300.21367521368, 140.224358974359, 0.0256143162393162, 2100000000, -953525641.025641, 41259.6153846154, 1916.66666666667, 0, 80769230.7692308, 8750, -3973.0235042735, 0.171915064102564, 785256410.25641, -33653846.1538461, 6663.46153846154, -730.769230769231, 0, -20192307.6923077, 3271.90170940171, -140.224358974359, 0.0277644230769231, 928846153.846154, -392628205.128205, 5250, -506.410256410256, 0, -20192307.6923077, 3870.19230769231, -1635.95085470085, 0.021875, -2458974358.97436, 314102564.102564, -25980.7692307692, 3333.33333333333, 0, 134615384.615385, -10245.7264957264, 1308.76068376068, -0.108253205128205, -529487179.48718, 583333333.333333, -9153.84615384615, 3333.33333333333, 0, 175000000, -2206.19658119658, 2430.55555555556, -0.0381410256410256, -1310256410.25641, 224358974.358974, -9557.69230769231, 1358.97435897436, 0, 0, -5459.40170940171, 934.82905982906, -0.0398237179487179, -547435897.435898, 224358974.358974, -14628.2051282051, 1358.97435897436, 0, -94230769.2307692, -2280.98290598291, 934.82905982906, -0.0609508547008547;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(10) << -33653846.1538461, 785256410.25641, -6663.46153846154, 0, -730.769230769231, 20192307.6923077, -140.224358974359, 3271.90170940171, -0.0277644230769231, -953525641.025641, 2100000000, -41259.6153846154, 0, 1916.66666666667, -80769230.7692308, -3973.0235042735, 8750, -0.171915064102564, 33653846.1538461, 1032051282.05128, -6147.4358974359, 0, -730.769230769231, 53846153.8461538, 140.224358974359, 4300.21367521368, -0.0256143162393162, -392628205.128205, 928846153.846154, -5250, 0, -506.410256410256, 20192307.6923077, -1635.95085470085, 3870.19230769231, -0.021875, 583333333.333333, -529487179.48718, 9153.84615384615, 0, 3333.33333333333, -175000000, 2430.55555555556, -2206.19658119658, 0.0381410256410256, 314102564.102564, -2458974358.97436, 25980.7692307692, 0, 3333.33333333333, -134615384.615385, 1308.76068376068, -10245.7264957264, 0.108253205128205, 224358974.358974, -547435897.435898, 14628.2051282051, 0, 1358.97435897436, 94230769.2307692, 934.82905982906, -2280.98290598291, 0.0609508547008547, 224358974.358974, -1310256410.25641, 9557.69230769231, 0, 1358.97435897436, 0, 934.82905982906, -5459.40170940171, 0.0398237179487179;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(11) << 0, 0, 403846153.846154, -35897435.8974359, 13461538.4615385, -1461.53846153846, 0, 0, 1682.69230769231, 0, 0, 933333333.333333, 53846153.8461539, -53846153.8461539, 3833.33333333333, 0, 0, 3888.88888888889, 0, 0, 403846153.846154, -13461538.4615385, 35897435.8974359, -1461.53846153846, 0, 0, 1682.69230769231, 0, 0, 412820512.820513, -13461538.4615385, 13461538.4615385, -1012.82051282051, 0, 0, 1720.08547008547, 0, 0, -664102564.102564, 89743589.7435897, -116666666.666667, 6666.66666666667, 0, 0, -2767.09401709402, 0, 0, -664102564.102564, 116666666.666667, -89743589.7435897, 6666.66666666667, 0, 0, -2767.09401709402, 0, 0, -412820512.820513, 0, 62820512.8205128, 2717.94871794872, 0, 0, -1720.08547008547, 0, 0, -412820512.820513, -62820512.8205128, 0, 2717.94871794872, 0, 0, -1720.08547008547;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(12) << 134.615384615385, 0, 35897435.8974359, 8982959.4017094, 280.448717948718, -27.5128739316239, -0.00552884615384615, 0, -299.145299145299, 1916.66666666667, 0, 53846153.8461539, 26940576.9230769, -7946.04700854701, 250.343830128205, 0.0239583333333333, 0, 897.435897435898, 134.615384615385, 0, -13461538.4615385, 8980902.77777778, -280.448717948718, -46.0983173076923, -0.00552884615384615, 0, -224.358974358974, -32.0512820512821, 0, 13461538.4615385, 13469278.8461538, -3271.90170940171, -9.57163461538462, -0.00435363247863248, 0, -112.179487179487, -974.358974358974, 0, -89743589.7435897, -26943568.3760684, 2617.52136752137, 152.347596153846, 0.0237179487179487, 0, 747.863247863248, -974.358974358974, 0, -62820512.8205128, -26927489.3162393, 4861.11111111111, 110.180128205128, 0.0237179487179487, 0, 1196.5811965812, -102.564102564103, 0, 0, -35908354.7008547, 1869.65811965812, 26.8434294871795, 0.0108974358974359, 0, 0, -102.564102564103, 0, 62820512.8205128, -35901997.8632479, 1869.65811965812, 81.9293803418804, 0.0108974358974359, 0, -523.504273504273;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(13) << 0, 134.615384615385, 13461538.4615385, -280.448717948718, 8980902.77777778, 46.0983173076923, 0, -0.00552884615384615, 224.358974358974, 0, 1916.66666666667, -53846153.8461539, -7946.04700854701, 26940576.9230769, -250.343830128205, 0, 0.0239583333333333, -897.435897435898, 0, 134.615384615385, -35897435.8974359, 280.448717948718, 8982959.4017094, 27.5128739316239, 0, -0.00552884615384615, 299.145299145299, 0, -32.0512820512821, -13461538.4615385, -3271.90170940171, 13469278.8461538, 9.57163461538462, 0, -0.00435363247863248, 112.179487179487, 0, -974.358974358974, 62820512.8205128, 4861.11111111111, -26927489.3162393, -110.180128205128, 0, 0.0237179487179487, -1196.5811965812, 0, -974.358974358974, 89743589.7435897, 2617.52136752137, -26943568.3760684, -152.347596153846, 0, 0.0237179487179487, -747.863247863248, 0, -102.564102564103, -62820512.8205128, 1869.65811965812, -35901997.8632479, -81.9293803418804, 0, 0.0108974358974359, 523.504273504273, 0, -102.564102564103, 0, 1869.65811965812, -35908354.7008547, -26.8434294871795, 0, 0.0108974358974359, 0;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(14) << 53846153.8461538, 20192307.6923077, 336.538461538462, 0, 0, 31413621.7948718, -74.7863247863248, 196.314102564103, -0.0107772435897436, 80769230.7692308, -80769230.7692308, 4791.66666666667, 0, 0, 94238547.008547, 785.25641025641, -785.25641025641, 0.0519097222222222, -20192307.6923077, -53846153.8461538, 336.538461538462, 0, 0, 31413621.7948718, -196.314102564103, 74.7863247863248, -0.0107772435897436, 20192307.6923077, -20192307.6923077, -80.1282051282051, 0, 0, 47118824.7863248, -28.0448717948718, 28.0448717948718, -0.00877403846153846, -134615384.615385, 94230769.2307692, -2435.89743589744, 0, 0, -94236303.4188034, 186.965811965812, -579.594017094017, 0.0454059829059829, -94230769.2307692, 134615384.615385, -2435.89743589744, 0, 0, -94236303.4188034, 579.594017094017, -186.965811965812, 0.0454059829059829, 0, -94230769.2307692, -256.410256410256, 0, 0, -125644465.811966, 0, 130.876068376068, 0.0215811965811966, 94230769.2307692, 0, -256.410256410256, 0, 0, -125644465.811966, -130.876068376068, 0, 0.0215811965811966;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(15) << 4300.21367521368, 140.224358974359, 0.0256143162393162, -0.00192307692307692, 0, 74.7863247863248, 74.8185763888889, 0.00105168269230769, -0.00022950874732906, 8750, -3973.0235042735, 0.171915064102564, 0.0239583333333333, 0, 785.25641025641, 224.424599358974, -0.0297976762820513, 0.0020846226963141, 3271.90170940171, -140.224358974359, 0.0277644230769231, -0.00192307692307692, 0, -196.314102564103, 74.8108640491453, -0.00105168269230769, -0.000384407151442308, 3870.19230769231, -1635.95085470085, 0.021875, -0.00237713675213675, 0, 28.0448717948718, 112.208513621795, -0.0122696314102564, -7.99641426282051E-05, -10245.7264957264, 1308.76068376068, -0.108253205128205, 0.00576923076923077, 0, -186.965811965812, -224.435817307692, 0.00981570512820513, 0.00127055562232906, -2206.19658119658, 2430.55555555556, -0.0381410256410256, 0.00576923076923077, 0, 205.662393162393, -224.375520833333, 0.0182291666666667, 0.000918517361111111, -5459.40170940171, 934.82905982906, -0.0398237179487179, 0.00480769230769231, 0, 0, -299.18624465812, 0.00701121794871795, 0.000224060296474359, -2280.98290598291, 934.82905982906, -0.0609508547008547, 0.00480769230769231, 0, 130.876068376068, -299.162406517094, 0.00701121794871795, 0.000683303552350427;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(16) << -140.224358974359, 3271.90170940171, -0.0277644230769231, 0, -0.00192307692307692, 196.314102564103, -0.00105168269230769, 74.8108640491453, 0.000384407151442308, -3973.0235042735, 8750, -0.171915064102564, 0, 0.0239583333333333, -785.25641025641, -0.0297976762820513, 224.424599358974, -0.0020846226963141, 140.224358974359, 4300.21367521368, -0.0256143162393162, 0, -0.00192307692307692, -74.7863247863248, 0.00105168269230769, 74.8185763888889, 0.00022950874732906, -1635.95085470085, 3870.19230769231, -0.021875, 0, -0.00237713675213675, -28.0448717948718, -0.0122696314102564, 112.208513621795, 7.99641426282051E-05, 2430.55555555556, -2206.19658119658, 0.0381410256410256, 0, 0.00576923076923077, -205.662393162393, 0.0182291666666667, -224.375520833333, -0.000918517361111111, 1308.76068376068, -10245.7264957264, 0.108253205128205, 0, 0.00576923076923077, 186.965811965812, 0.00981570512820513, -224.435817307692, -0.00127055562232906, 934.82905982906, -2280.98290598291, 0.0609508547008547, 0, 0.00480769230769231, -130.876068376068, 0.00701121794871795, -299.162406517094, -0.000683303552350427, 934.82905982906, -5459.40170940171, 0.0398237179487179, 0, 0.00480769230769231, 0, 0.00701121794871795, -299.18624465812, -0.000224060296474359;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(17) << 0, 0, 1682.69230769231, 299.145299145299, 224.358974358974, -0.00328525641025641, 0, 0, 261.764756944444, 0, 0, 3888.88888888889, 897.435897435898, -897.435897435898, 0.0559027777777778, 0, 0, 785.285576923077, 0, 0, 1682.69230769231, -224.358974358974, -299.145299145299, -0.00328525641025641, 0, 0, 261.764756944444, 0, 0, 1720.08547008547, 112.179487179487, -112.179487179487, -0.00488782051282051, 0, 0, 392.641105769231, 0, 0, -2767.09401709402, -747.863247863248, 299.145299145299, 0.00747863247863248, 0, 0, -785.277163461538, 0, 0, -2767.09401709402, -299.145299145299, 747.863247863248, 0.00747863247863248, 0, 0, -785.277163461538, 0, 0, -1720.08547008547, 0, -523.504273504273, 0.00918803418803419, 0, 0, -1047.02144764957, 0, 0, -1720.08547008547, 523.504273504273, 0, 0.00918803418803419, 0, 0, -1047.02144764957;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(18) << 928846153.846154, 392628205.128205, 628.205128205128, -269.230769230769, 0, -20192307.6923077, 3870.19230769231, 1635.95085470085, 0.00261752136752137, 785256410.25641, 33653846.1538461, 6483.97435897436, 134.615384615385, 0, -20192307.6923077, 3271.90170940171, 140.224358974359, 0.0270165598290598, 2100000000, 953525641.025641, 6663.46153846154, 134.615384615385, 0, 80769230.7692308, 8750, 3973.0235042735, 0.0277644230769231, 1032051282.05128, -33653846.1538461, 493.589743589744, -269.230769230769, 0, -53846153.8461538, 4300.21367521368, -140.224358974359, 0.00205662393162393, -1310256410.25641, -224358974.358974, -2288.46153846154, 1153.84615384615, 0, 0, -5459.40170940171, -934.82905982906, -0.00953525641025641, -529487179.48718, -583333333.333333, -3320.51282051282, 1089.74358974359, 0, 175000000, -2206.19658119658, -2430.55555555556, -0.0138354700854701, -2458974358.97436, -314102564.102564, -1660.25641025641, 1153.84615384615, 0, 134615384.615385, -10245.7264957264, -1308.76068376068, -0.00691773504273504, -547435897.435898, -224358974.358974, -7000, 910.25641025641, 0, -94230769.2307692, -2280.98290598291, -934.82905982906, -0.0291666666666667;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(19) << 392628205.128205, 928846153.846154, -426.282051282051, 0, -269.230769230769, -20192307.6923077, 1635.95085470085, 3870.19230769231, -0.00177617521367521, -33653846.1538461, 1032051282.05128, -6842.94871794872, 0, 134.615384615385, -53846153.8461538, -140.224358974359, 4300.21367521368, -0.0285122863247863, 953525641.025641, 2100000000, 6147.4358974359, 0, 134.615384615385, 80769230.7692308, 3973.0235042735, 8750, 0.0256143162393162, 33653846.1538461, 785256410.25641, 493.589743589744, 0, -269.230769230769, -20192307.6923077, 140.224358974359, 3271.90170940171, 0.00205662393162393, -224358974.358974, -547435897.435898, -3051.28205128205, 0, 1153.84615384615, -94230769.2307692, -934.82905982906, -2280.98290598291, -0.0127136752136752, -314102564.102564, -2458974358.97436, 44.8717948717949, 0, 1089.74358974359, 134615384.615385, -1308.76068376068, -10245.7264957264, 0.000186965811965812, -583333333.333333, -529487179.48718, 3679.48717948718, 0, 1153.84615384615, 175000000, -2430.55555555556, -2206.19658119658, 0.0153311965811966, -224358974.358974, -1310256410.25641, -44.8717948717949, 0, 910.25641025641, 0, -934.82905982906, -5459.40170940171, -0.000186965811965812;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(20) << 0, 0, 412820512.820513, -13461538.4615385, -13461538.4615385, -538.461538461539, 0, 0, 1720.08547008547, 0, 0, 403846153.846154, -13461538.4615385, -35897435.8974359, 269.230769230769, 0, 0, 1682.69230769231, 0, 0, 933333333.333333, 53846153.8461539, 53846153.8461539, 269.230769230769, 0, 0, 3888.88888888889, 0, 0, 403846153.846154, -35897435.8974359, -13461538.4615385, -538.461538461539, 0, 0, 1682.69230769231, 0, 0, -412820512.820513, 0, -62820512.8205128, 2307.69230769231, 0, 0, -1720.08547008547, 0, 0, -664102564.102564, 116666666.666667, 89743589.7435897, 2179.48717948718, 0, 0, -2767.09401709402, 0, 0, -664102564.102564, 89743589.7435897, 116666666.666667, 2307.69230769231, 0, 0, -2767.09401709402, 0, 0, -412820512.820513, -62820512.8205128, 0, 1820.51282051282, 0, 0, -1720.08547008547;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(21) << -269.230769230769, 0, 13461538.4615385, 13469278.8461538, 3271.90170940171, 26.2872863247863, -0.00336538461538462, 0, -112.179487179487, -730.769230769231, 0, -13461538.4615385, 8980902.77777778, 280.448717948718, -46.099813034188, -0.00192307692307692, 0, -224.358974358974, 134.615384615385, 0, 53846153.8461539, 26940576.9230769, 7946.04700854701, 34.6709134615385, 0.00168269230769231, 0, 897.435897435898, -391.025641025641, 0, 35897435.8974359, 8982959.4017094, -280.448717948718, 17.3118055555556, -0.00387286324786325, 0, -299.145299145299, 461.538461538462, 0, 0, -35908354.7008547, -1869.65811965812, -98.7370192307692, 0.0115384615384615, 0, 0, 12.8205128205128, 0, -62820512.8205128, -26927489.3162393, -4861.11111111111, -123.104594017094, 0.00913461538461539, 0, 1196.5811965812, 435.897435897436, 0, -89743589.7435897, -26943568.3760684, -2617.52136752137, 8.96052350427351, 0.0114316239316239, 0, 747.863247863248, 346.153846153846, 0, 62820512.8205128, -35901997.8632479, -1869.65811965812, 46.0955128205128, 0.00902777777777778, 0, -523.504273504273;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(22) << 0, -269.230769230769, 13461538.4615385, 3271.90170940171, 13469278.8461538, -26.2856036324786, 0, -0.00336538461538462, -112.179487179487, 0, -730.769230769231, 35897435.8974359, -280.448717948718, 8982959.4017094, 27.507077991453, 0, -0.00192307692307692, -299.145299145299, 0, 134.615384615385, 53846153.8461539, 7946.04700854701, 26940576.9230769, 55.1794337606838, 0, 0.00168269230769231, 897.435897435898, 0, -391.025641025641, -13461538.4615385, 280.448717948718, 8980902.77777778, -43.5856303418803, 0, -0.00387286324786325, -224.358974358974, 0, 461.538461538462, 62820512.8205128, -1869.65811965812, -35901997.8632479, 133.307905982906, 0, 0.0115384615384615, -523.504273504273, 0, 12.8205128205128, -89743589.7435897, -2617.52136752137, -26943568.3760684, 170.513194444444, 0, 0.00913461538461539, 747.863247863248, 0, 435.897435897436, -62820512.8205128, -4861.11111111111, -26927489.3162393, 33.3639957264957, 0, 0.0114316239316239, 1196.5811965812, 0, 346.153846153846, 0, -1869.65811965812, -35908354.7008547, 8.97398504273504, 0, 0.00902777777777778, 0;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(23) << 20192307.6923077, 20192307.6923077, -673.076923076923, 0, 0, 47118824.7863248, -28.0448717948718, -28.0448717948718, -0.00729166666666667, -20192307.6923077, 53846153.8461538, -1826.92307692308, 0, 0, 31413621.7948718, -196.314102564103, -74.7863247863248, -0.00536858974358974, 80769230.7692308, 80769230.7692308, 336.538461538462, 0, 0, 94238547.008547, 785.25641025641, 785.25641025641, 0.00364583333333333, 53846153.8461538, -20192307.6923077, -977.564102564103, 0, 0, 31413621.7948718, -74.7863247863248, -196.314102564103, -0.00856036324786325, 0, 94230769.2307692, 1153.84615384615, 0, 0, -125644465.811966, 0, -130.876068376068, 0.0240384615384615, -94230769.2307692, -134615384.615385, 32.0512820512821, 0, 0, -94236303.4188034, 579.594017094017, 186.965811965812, 0.0182959401709402, -134615384.615385, -94230769.2307692, 1089.74358974359, 0, 0, -94236303.4188034, 186.965811965812, 579.594017094017, 0.0237713675213675, 94230769.2307692, 0, 865.384615384615, 0, 0, -125644465.811966, -130.876068376068, 0, 0.0187767094017094;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(24) << 3870.19230769231, 1635.95085470085, 0.00261752136752137, -0.00336538461538462, 0, 28.0448717948718, 112.208513621795, 0.0122696314102564, 0.00021903672542735, 3271.90170940171, 140.224358974359, 0.0270165598290598, -0.00552884615384615, 0, -196.314102564103, 74.8108640491453, 0.00105168269230769, -0.000384412760416667, 8750, 3973.0235042735, 0.0277644230769231, 0.00168269230769231, 0, 785.25641025641, 224.424599358974, 0.0297976762820513, 0.000288669771634615, 4300.21367521368, -140.224358974359, 0.00205662393162393, -0.00438034188034188, 0, 74.7863247863248, 74.8185763888889, -0.00105168269230769, 0.000144246193910256, -5459.40170940171, -934.82905982906, -0.00953525641025641, 0.00865384615384615, 0, 0, -299.18624465812, -0.00701121794871795, -0.00082272108707265, -2206.19658119658, -2430.55555555556, -0.0138354700854701, 0.0046474358974359, 0, 205.662393162393, -224.375520833333, -0.0182291666666667, -0.00102574479166667, -10245.7264957264, -1308.76068376068, -0.00691773504273504, 0.00844017094017094, 0, -186.965811965812, -224.435817307692, -0.00981570512820513, 7.47344417735043E-05, -2280.98290598291, -934.82905982906, -0.0291666666666667, 0.00667735042735043, 0, 130.876068376068, -299.162406517094, -0.00701121794871795, 0.000384396634615385;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(25) << 1635.95085470085, 3870.19230769231, -0.00177617521367521, 0, -0.00336538461538462, 28.0448717948718, 0.0122696314102564, 112.208513621795, -0.000219030415331197, -140.224358974359, 4300.21367521368, -0.0285122863247863, 0, -0.00552884615384615, 74.7863247863248, -0.00105168269230769, 74.8185763888889, 0.000229487012553419, 3973.0235042735, 8750, 0.0256143162393162, 0, 0.00168269230769231, 785.25641025641, 0.0297976762820513, 224.424599358974, 0.000459593816773504, 140.224358974359, 3271.90170940171, 0.00205662393162393, 0, -0.00438034188034188, -196.314102564103, 0.00105168269230769, 74.8108640491453, -0.000363232438568376, -934.82905982906, -2280.98290598291, -0.0127136752136752, 0, 0.00865384615384615, 130.876068376068, -0.00701121794871795, -299.162406517094, 0.00111101575854701, -1308.76068376068, -10245.7264957264, 0.000186965811965812, 0, 0.0046474358974359, -186.965811965812, -0.00981570512820513, -224.435817307692, 0.00142094157318376, -2430.55555555556, -2206.19658119658, 0.0153311965811966, 0, 0.00844017094017094, 205.662393162393, -0.0182291666666667, -224.375520833333, 0.000277892761752137, -934.82905982906, -5459.40170940171, -0.000186965811965812, 0, 0.00667735042735043, 0, -0.00701121794871795, -299.18624465812, 0.000074784922542735;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(26) << 0, 0, 1720.08547008547, 112.179487179487, 112.179487179487, -0.0078525641025641, 0, 0, 392.641105769231, 0, 0, 1682.69230769231, -224.358974358974, 299.145299145299, -0.0141025641025641, 0, 0, 261.764756944444, 0, 0, 3888.88888888889, 897.435897435898, 897.435897435898, 0.00392628205128205, 0, 0, 785.285576923077, 0, 0, 1682.69230769231, 299.145299145299, -224.358974358974, -0.0103899572649573, 0, 0, 261.764756944444, 0, 0, -1720.08547008547, 0, 523.504273504273, 0.0192307692307692, 0, 0, -1047.02144764957, 0, 0, -2767.09401709402, -299.145299145299, -747.863247863248, 0.0093482905982906, 0, 0, -785.277163461538, 0, 0, -2767.09401709402, -747.863247863248, -299.145299145299, 0.0186965811965812, 0, 0, -785.277163461538, 0, 0, -1720.08547008547, 523.504273504273, 0, 0.0147970085470085, 0, 0, -1047.02144764957;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(27) << 785256410.25641, -33653846.1538461, -628.205128205128, -391.025641025641, 0, 20192307.6923077, 3271.90170940171, -140.224358974359, -0.00261752136752137, 928846153.846154, -392628205.128205, 5182.69230769231, -32.0512820512821, 0, 20192307.6923077, 3870.19230769231, -1635.95085470085, 0.0215945512820513, 1032051282.05128, 33653846.1538461, 426.282051282051, -391.025641025641, 0, 53846153.8461538, 4300.21367521368, 140.224358974359, 0.00177617521367521, 2100000000, -953525641.025641, -5250, -32.0512820512821, 0, -80769230.7692308, 8750, -3973.0235042735, -0.021875, -1310256410.25641, 224358974.358974, -762.820512820513, 1243.58974358974, 0, 0, -5459.40170940171, 934.82905982906, -0.0031784188034188, -547435897.435898, 224358974.358974, 3679.48717948718, 1243.58974358974, 0, 94230769.2307692, -2280.98290598291, 934.82905982906, 0.0153311965811966, -2458974358.97436, 314102564.102564, 762.820512820513, 1243.58974358974, 0, -134615384.615385, -10245.7264957264, 1308.76068376068, 0.0031784188034188, -529487179.48718, 583333333.333333, -3410.25641025641, 1243.58974358974, 0, -175000000, -2206.19658119658, 2430.55555555556, -0.0142094017094017;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(28) << 33653846.1538461, 1032051282.05128, -426.282051282051, 0, -391.025641025641, -53846153.8461538, 140.224358974359, 4300.21367521368, -0.00177617521367521, -392628205.128205, 928846153.846154, -5182.69230769231, 0, -32.0512820512821, -20192307.6923077, -1635.95085470085, 3870.19230769231, -0.0215945512820513, -33653846.1538461, 785256410.25641, 628.205128205128, 0, -391.025641025641, -20192307.6923077, -140.224358974359, 3271.90170940171, 0.00261752136752137, -953525641.025641, 2100000000, 5250, 0, -32.0512820512821, 80769230.7692308, -3973.0235042735, 8750, 0.021875, 224358974.358974, -547435897.435898, -3679.48717948718, 0, 1243.58974358974, -94230769.2307692, 934.82905982906, -2280.98290598291, -0.0153311965811966, 224358974.358974, -1310256410.25641, 762.820512820513, 0, 1243.58974358974, 0, 934.82905982906, -5459.40170940171, 0.0031784188034188, 583333333.333333, -529487179.48718, 3410.25641025641, 0, 1243.58974358974, 175000000, 2430.55555555556, -2206.19658119658, 0.0142094017094017, 314102564.102564, -2458974358.97436, -762.820512820513, 0, 1243.58974358974, 134615384.615385, 1308.76068376068, -10245.7264957264, -0.0031784188034188;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(29) << 0, 0, 403846153.846154, 13461538.4615385, -35897435.8974359, -782.051282051282, 0, 0, 1682.69230769231, 0, 0, 412820512.820513, 13461538.4615385, -13461538.4615385, -64.1025641025641, 0, 0, 1720.08547008547, 0, 0, 403846153.846154, 35897435.8974359, -13461538.4615385, -782.051282051282, 0, 0, 1682.69230769231, 0, 0, 933333333.333333, -53846153.8461539, 53846153.8461539, -64.1025641025641, 0, 0, 3888.88888888889, 0, 0, -412820512.820513, 0, -62820512.8205128, 2487.17948717949, 0, 0, -1720.08547008547, 0, 0, -412820512.820513, 62820512.8205128, 0, 2487.17948717949, 0, 0, -1720.08547008547, 0, 0, -664102564.102564, -89743589.7435897, 116666666.666667, 2487.17948717949, 0, 0, -2767.09401709402, 0, 0, -664102564.102564, -116666666.666667, 89743589.7435897, 2487.17948717949, 0, 0, -2767.09401709402;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(30) << -269.230769230769, 0, 13461538.4615385, 8980902.77777778, -280.448717948718, 43.5845085470086, -0.00438034188034188, 0, 224.358974358974, -506.410256410256, 0, -13461538.4615385, 13469278.8461538, -3271.90170940171, -9.57219551282051, -0.00237713675213675, 0, 112.179487179487, -269.230769230769, 0, -35897435.8974359, 8982959.4017094, 280.448717948718, 17.3112446581197, -0.00438034188034188, 0, 299.145299145299, -32.0512820512821, 0, -53846153.8461539, 26940576.9230769, -7946.04700854701, -19.2745192307692, -0.000400641025641026, 0, -897.435897435898, 371.794871794872, 0, 0, -35908354.7008547, 1869.65811965812, -62.8268696581197, 0.0119123931623932, 0, 0, 371.794871794872, 0, -62820512.8205128, -35901997.8632479, 1869.65811965812, -97.405235042735, 0.0119123931623932, 0, 523.504273504273, 166.666666666667, 0, 89743589.7435897, -26943568.3760684, 2617.52136752137, -26.9167200854701, 0.0110576923076923, 0, -747.863247863248, 166.666666666667, 0, 62820512.8205128, -26927489.3162393, 4861.11111111111, 20.4844017094017, 0.0110576923076923, 0, -1196.5811965812;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(31) << 0, -269.230769230769, 35897435.8974359, 280.448717948718, 8982959.4017094, -17.3112446581197, 0, -0.00438034188034188, -299.145299145299, 0, -506.410256410256, 13461538.4615385, -3271.90170940171, 13469278.8461538, 9.57219551282051, 0, -0.00237713675213675, -112.179487179487, 0, -269.230769230769, -13461538.4615385, -280.448717948718, 8980902.77777778, -43.5845085470086, 0, -0.00438034188034188, -224.358974358974, 0, -32.0512820512821, 53846153.8461539, -7946.04700854701, 26940576.9230769, 19.2745192307692, 0, -0.000400641025641026, 897.435897435898, 0, 371.794871794872, 62820512.8205128, 1869.65811965812, -35901997.8632479, 97.405235042735, 0, 0.0119123931623932, -523.504273504273, 0, 371.794871794872, 0, 1869.65811965812, -35908354.7008547, 62.8268696581197, 0, 0.0119123931623932, 0, 0, 166.666666666667, -62820512.8205128, 4861.11111111111, -26927489.3162393, -20.4844017094017, 0, 0.0110576923076923, 1196.5811965812, 0, 166.666666666667, -89743589.7435897, 2617.52136752137, -26943568.3760684, 26.9167200854701, 0, 0.0110576923076923, 747.863247863248;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(32) << 20192307.6923077, 53846153.8461538, -673.076923076923, 0, 0, 31413621.7948718, 196.314102564103, -74.7863247863248, -0.0093215811965812, -20192307.6923077, 20192307.6923077, -1266.02564102564, 0, 0, 47118824.7863248, 28.0448717948718, -28.0448717948718, -0.00580929487179487, -53846153.8461538, -20192307.6923077, -673.076923076923, 0, 0, 31413621.7948718, 74.7863247863248, -196.314102564103, -0.0093215811965812, -80769230.7692308, 80769230.7692308, -80.1282051282051, 0, 0, 94238547.008547, -785.25641025641, 785.25641025641, -0.000868055555555556, 0, 94230769.2307692, 929.487179487179, 0, 0, -125644465.811966, 0, -130.876068376068, 0.024599358974359, -94230769.2307692, 0, 929.487179487179, 0, 0, -125644465.811966, 130.876068376068, 0, 0.024599358974359, 134615384.615385, -94230769.2307692, 416.666666666667, 0, 0, -94236303.4188034, -186.965811965812, 579.594017094017, 0.0224626068376068, 94230769.2307692, -134615384.615385, 416.666666666667, 0, 0, -94236303.4188034, -579.594017094017, 186.965811965812, 0.0224626068376068;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(33) << 3271.90170940171, -140.224358974359, -0.00261752136752137, -0.00387286324786325, 0, 196.314102564103, 74.8108640491453, -0.00105168269230769, 0.000363228231837607, 3870.19230769231, -1635.95085470085, 0.0215945512820513, -0.00435363247863248, 0, -28.0448717948718, 112.208513621795, -0.0122696314102564, -7.99662459935897E-05, 4300.21367521368, 140.224358974359, 0.00177617521367521, -0.00387286324786325, 0, -74.7863247863248, 74.8185763888889, 0.00105168269230769, 0.000144244090544872, 8750, -3973.0235042735, -0.021875, -0.000400641025641026, 0, -785.25641025641, 224.424599358974, -0.0297976762820513, -0.00016042047275641, -5459.40170940171, 934.82905982906, -0.0031784188034188, 0.00827991452991453, 0, 0, -299.18624465812, 0.00701121794871795, -0.000523528111645299, -2280.98290598291, 934.82905982906, 0.0153311965811966, 0.00827991452991453, 0, -130.876068376068, -299.162406517094, 0.00701121794871795, -0.000811850827991453, -10245.7264957264, 1308.76068376068, 0.0031784188034188, 0.00657051282051282, 0, 186.965811965812, -224.435817307692, 0.00981570512820513, -0.000224335136217949, -2206.19658119658, 2430.55555555556, -0.0142094017094017, 0.00657051282051282, 0, -205.662393162393, -224.375520833333, 0.0182291666666667, 0.00017083360042735;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(34) << 140.224358974359, 4300.21367521368, -0.00177617521367521, 0, -0.00387286324786325, 74.7863247863248, 0.00105168269230769, 74.8185763888889, -0.000144244090544872, -1635.95085470085, 3870.19230769231, -0.0215945512820513, 0, -0.00435363247863248, 28.0448717948718, -0.0122696314102564, 112.208513621795, 7.99662459935897E-05, -140.224358974359, 3271.90170940171, 0.00261752136752137, 0, -0.00387286324786325, -196.314102564103, -0.00105168269230769, 74.8108640491453, -0.000363228231837607, -3973.0235042735, 8750, 0.021875, 0, -0.000400641025641026, 785.25641025641, -0.0297976762820513, 224.424599358974, 0.00016042047275641, 934.82905982906, -2280.98290598291, -0.0153311965811966, 0, 0.00827991452991453, 130.876068376068, 0.00701121794871795, -299.162406517094, 0.000811850827991453, 934.82905982906, -5459.40170940171, 0.0031784188034188, 0, 0.00827991452991453, 0, 0.00701121794871795, -299.18624465812, 0.000523528111645299, 2430.55555555556, -2206.19658119658, 0.0142094017094017, 0, 0.00657051282051282, 205.662393162393, 0.0182291666666667, -224.375520833333, -0.00017083360042735, 1308.76068376068, -10245.7264957264, -0.0031784188034188, 0, 0.00657051282051282, -186.965811965812, 0.00981570512820513, -224.435817307692, 0.000224335136217949;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(35) << 0, 0, 1682.69230769231, 224.358974358974, 299.145299145299, -0.00886752136752137, 0, 0, 261.764756944444, 0, 0, 1720.08547008547, -112.179487179487, 112.179487179487, -0.0108173076923077, 0, 0, 392.641105769231, 0, 0, 1682.69230769231, -299.145299145299, -224.358974358974, -0.00886752136752137, 0, 0, 261.764756944444, 0, 0, 3888.88888888889, -897.435897435898, 897.435897435898, -0.00093482905982906, 0, 0, 785.285576923077, 0, 0, -1720.08547008547, 0, 523.504273504273, 0.0181089743589744, 0, 0, -1047.02144764957, 0, 0, -1720.08547008547, -523.504273504273, 0, 0.0181089743589744, 0, 0, -1047.02144764957, 0, 0, -2767.09401709402, 747.863247863248, -299.145299145299, 0.0138354700854701, 0, 0, -785.277163461538, 0, 0, -2767.09401709402, 299.145299145299, -747.863247863248, 0.0138354700854701, 0, 0, -785.277163461538;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(36) << -2458974358.97436, -583333333.333333, 0, 12.8205128205128, 0, 134615384.615385, -10245.7264957264, -2430.55555555556, 0, -2458974358.97436, 583333333.333333, -30961.5384615385, -974.358974358974, 0, -134615384.615385, -10245.7264957264, 2430.55555555556, -0.12900641025641, -1310256410.25641, -224358974.358974, -2961.53846153846, 461.538461538462, 0, 0, -5459.40170940171, -934.82905982906, -0.0123397435897436, -1310256410.25641, 224358974.358974, -1615.38461538462, 371.794871794872, 0, 0, -5459.40170940171, 934.82905982906, -0.00673076923076923, 5456410256.41026, 0, 18846.1538461538, -1282.05128205128, 0, 0, 22735.0427350427, 0, 0.078525641025641, 0, -897435897.435898, -2153.84615384615, -2153.84615384615, 0, -269230769.230769, 0, -3739.31623931624, -0.00897435897435897, 2082051282.05128, 0, 4128.20512820513, -1282.05128205128, 0, 0, 8675.21367521367, 0, 0.0172008547008547, 0, 897435897.435898, 14717.9487179487, -1794.8717948718, 0, 269230769.230769, 0, 3739.31623931624, 0.0613247863247863;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(37) << -314102564.102564, -529487179.48718, 1256.41025641026, 0, 12.8205128205128, 94230769.2307692, -1308.76068376068, -2206.19658119658, 0.00523504273504274, 314102564.102564, -529487179.48718, 4038.46153846154, 0, -974.358974358974, 94230769.2307692, 1308.76068376068, -2206.19658119658, 0.0168269230769231, -224358974.358974, -547435897.435898, -4307.69230769231, 0, 461.538461538462, 94230769.2307692, -934.82905982906, -2280.98290598291, -0.0179487179487179, 224358974.358974, -547435897.435898, -4576.92307692308, 0, 371.794871794872, 94230769.2307692, 934.82905982906, -2280.98290598291, -0.0190705128205128, 0, 2943589743.58974, -5384.61538461539, 0, -1282.05128205128, -323076923.076923, 0, 12264.9572649573, -0.0224358974358974, -897435897.435898, 0, 7000, 0, -2153.84615384615, -269230769.230769, -3739.31623931624, 0, 0.0291666666666667, 0, -789743589.74359, -1794.8717948718, 0, -1282.05128205128, -323076923.076923, 0, -3290.59829059829, -0.00747863247863248, 897435897.435898, 0, 3769.23076923077, 0, -1794.8717948718, -269230769.230769, 3739.31623931624, 0, 0.0157051282051282;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(38) << 0, 0, -664102564.102564, 89743589.7435897, 62820512.8205128, 25.6410256410256, 0, 0, -2767.09401709402, 0, 0, -664102564.102564, -89743589.7435897, 62820512.8205128, -1948.71794871795, 0, 0, -2767.09401709402, 0, 0, -412820512.820513, 0, 62820512.8205128, 923.076923076923, 0, 0, -1720.08547008547, 0, 0, -412820512.820513, 0, 62820512.8205128, 743.589743589744, 0, 0, -1720.08547008547, 0, 0, 1866666666.66667, 0, -215384615.384615, -2564.10256410256, 0, 0, 7777.77777777778, 0, 0, 0, -179487179.48718, -179487179.48718, -4307.69230769231, 0, 0, 0, 0, 0, 287179487.179487, 0, -215384615.384615, -2564.10256410256, 0, 0, 1196.5811965812, 0, 0, 0, 179487179.48718, -179487179.48718, -3589.74358974359, 0, 0, 0;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(39) << 1089.74358974359, 0, -89743589.7435897, -26943568.3760684, -4861.11111111111, -170.512820512821, 0.0046474358974359, 0, 747.863247863248, 3333.33333333333, 0, 89743589.7435897, -26943568.3760684, 4861.11111111111, 152.30608974359, 0.00576923076923077, 0, -747.863247863248, 1153.84615384615, 0, 0, -35908354.7008547, -1869.65811965812, -98.7426282051282, 0.00865384615384615, 0, 0, 1243.58974358974, 0, 0, -35908354.7008547, 1869.65811965812, -62.8339743589744, 0.00827991452991453, 0, 0, -1282.05128205128, 0, 0, 143635213.675214, 0, 646.310897435898, -0.016025641025641, 0, 0, -2153.84615384615, 0, 179487179.48718, 89743589.7435897, -7478.63247863248, 394.853846153846, -0.0269230769230769, 0, -1495.7264957265, -1589.74358974359, 0, 0, 71812222.2222222, 0, 71.8292735042735, -0.0173076923076923, 0, 0, -1794.8717948718, 0, -179487179.48718, 89743589.7435897, 7478.63247863248, -35.7747863247863, -0.0224358974358974, 0, 1495.7264957265;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(40) << 0, 1089.74358974359, -116666666.666667, -2617.52136752137, -26927489.3162393, 123.087393162393, 0, 0.0046474358974359, 299.145299145299, 0, 3333.33333333333, -116666666.666667, 2617.52136752137, -26927489.3162393, -110.222756410256, 0, 0.00576923076923077, 299.145299145299, 0, 1153.84615384615, -62820512.8205128, -1869.65811965812, -35901997.8632479, 133.297435897436, 0, 0.00865384615384615, 523.504273504273, 0, 1243.58974358974, -62820512.8205128, 1869.65811965812, -35901997.8632479, 97.3977564102564, 0, 0.00827991452991453, 523.504273504273, 0, -1282.05128205128, -215384615.384615, 0, 143614273.504274, -820.557692307692, 0, -0.016025641025641, -3589.74358974359, 0, -2153.84615384615, 179487179.48718, -7478.63247863248, 89743589.7435897, -394.813461538462, 0, -0.0269230769230769, -1495.7264957265, 0, -1589.74358974359, 215384615.384615, 0, 71788290.5982906, -51.2970085470085, 0, -0.0173076923076923, -1794.87179487179, 0, -1794.8717948718, 179487179.48718, 7478.63247863248, 89743589.7435897, -143.558333333333, 0, -0.0224358974358974, -1495.7264957265;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(41) << -134615384.615385, -175000000, 2724.35897435897, 0, 0, -94236303.4188034, 186.965811965812, -205.662393162393, 0.0115651709401709, 134615384.615385, -175000000, 8333.33333333333, 0, 0, -94236303.4188034, -186.965811965812, -205.662393162393, 0.018482905982906, 0, -94230769.2307692, 2884.61538461538, 0, 0, -125644465.811966, 0, 130.876068376068, 0.0197115384615385, 0, -94230769.2307692, 3108.97435897436, 0, 0, -125644465.811966, 0, 130.876068376068, 0.019150641025641, 0, -323076923.076923, -3205.12820512821, 0, 0, 502579658.119658, 0, -3141.02564102564, -0.0347222222222222, 269230769.230769, 269230769.230769, -5384.61538461539, 0, 0, 314102564.102564, -373.931623931624, -373.931623931624, -0.0583333333333333, 0, 323076923.076923, -3974.35897435897, 0, 0, 251284444.444444, 0, -448.717948717949, -0.0379273504273504, -269230769.230769, 269230769.230769, -4487.17948717949, 0, 0, 314102564.102564, 373.931623931624, -373.931623931624, -0.0486111111111111;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(42) << -10245.7264957264, -2430.55555555556, 0, 0.00913461538461539, 0, -186.965811965812, -224.435817307692, -0.0182291666666667, -0.00142094017094017, -10245.7264957264, 2430.55555555556, -0.12900641025641, 0.0237179487179487, 0, 186.965811965812, -224.435817307692, 0.0182291666666667, 0.0012703999732906, -5459.40170940171, -934.82905982906, -0.0123397435897436, 0.0115384615384615, 0, 0, -299.18624465812, -0.00701121794871795, -0.000822742120726496, -5459.40170940171, 934.82905982906, -0.00673076923076923, 0.0119123931623932, 0, 0, -299.18624465812, 0.00701121794871795, -0.000523554754273504, 22735.0427350427, 0, 0.078525641025641, -0.016025641025641, 0, 0, 1196.75170940171, 0, 0.00538520432692308, 0, -3739.31623931624, -0.00897435897435897, -0.0269230769230769, 0, 373.931623931624, 747.863247863248, -0.0280448717948718, 0.00329053098290598, 8675.21367521367, 0, 0.0172008547008547, -0.0185897435897436, 0, 0, 598.355662393162, 0, 0.000598419604700855, 0, 3739.31623931624, 0.0613247863247863, -0.0224358974358974, 0, -373.931623931624, 747.863247863248, 0.0280448717948718, -0.000298685363247863;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(43) << -1308.76068376068, -2206.19658119658, 0.00523504273504274, 0, 0.00913461538461539, -579.594017094017, -0.00981570512820513, -224.375520833333, 0.00102568028846154, 1308.76068376068, -2206.19658119658, 0.0168269230769231, 0, 0.0237179487179487, -579.594017094017, 0.00981570512820513, -224.375520833333, -0.000918677216880342, -934.82905982906, -2280.98290598291, -0.0179487179487179, 0, 0.0115384615384615, -130.876068376068, -0.00701121794871795, -299.162406517094, 0.0011109764957265, 934.82905982906, -2280.98290598291, -0.0190705128205128, 0, 0.0119123931623932, -130.876068376068, 0.00701121794871795, -299.162406517094, 0.000811822783119658, 0, 12264.9572649573, -0.0224358974358974, 0, -0.016025641025641, -3141.02564102564, 0, 1196.67318376068, -0.00683777510683761, -3739.31623931624, 0, 0.0291666666666667, 0, -0.0269230769230769, 373.931623931624, -0.0280448717948718, 747.863247863248, -0.00329037954059829, 0, -3290.59829059829, -0.00747863247863248, 0, -0.0185897435897436, 448.717948717949, 0, 598.265918803419, -0.000427406517094017, 3739.31623931624, 0, 0.0157051282051282, 0, -0.0224358974358974, 373.931623931624, 0.0280448717948718, 747.863247863248, -0.00119646340811966;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(44) << 0, 0, -2767.09401709402, -747.863247863248, -1196.5811965812, 0.0228098290598291, 0, 0, -785.277163461538, 0, 0, -2767.09401709402, 747.863247863248, -1196.5811965812, 0.0613247863247863, 0, 0, -785.277163461538, 0, 0, -1720.08547008547, 0, -523.504273504273, 0.0278846153846154, 0, 0, -1047.02144764957, 0, 0, -1720.08547008547, 0, -523.504273504273, 0.0290064102564103, 0, 0, -1047.02144764957, 0, 0, 7777.77777777778, 0, -3589.74358974359, -0.0373931623931624, 0, 0, 4188.09252136752, 0, 0, 0, 1495.7264957265, 1495.7264957265, -0.0628205128205128, 0, 0, 2617.52136752137, 0, 0, 1196.5811965812, 0, 1794.87179487179, -0.0438034188034188, 0, 0, 2094.02606837607, 0, 0, 0, -1495.7264957265, 1495.7264957265, -0.0523504273504274, 0, 0, 2617.52136752137;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(45) << -547435897.435898, -224358974.358974, 4307.69230769231, 461.538461538462, 0, -94230769.2307692, -2280.98290598291, -934.82905982906, 0.0179487179487179, -529487179.48718, 314102564.102564, -4038.46153846154, -974.358974358974, 0, -94230769.2307692, -2206.19658119658, 1308.76068376068, -0.0168269230769231, -529487179.48718, -314102564.102564, -1256.41025641026, 12.8205128205128, 0, -94230769.2307692, -2206.19658119658, -1308.76068376068, -0.00523504273504274, -547435897.435898, 224358974.358974, 4576.92307692308, 371.794871794872, 0, -94230769.2307692, -2280.98290598291, 934.82905982906, 0.0190705128205128, 0, -897435897.435898, -7000, -2153.84615384615, 0, 269230769.230769, 0, -3739.31623931624, -0.0291666666666667, 2943589743.58974, 0, 5384.61538461539, -1282.05128205128, 0, 323076923.076923, 12264.9572649573, 0, 0.0224358974358974, 0, 897435897.435898, -3769.23076923077, -1794.8717948718, 0, 269230769.230769, 0, 3739.31623931624, -0.0157051282051282, -789743589.74359, 0, 1794.8717948718, -1282.05128205128, 0, 323076923.076923, -3290.59829059829, 0, 0.00747863247863248;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(46) << -224358974.358974, -1310256410.25641, 2961.53846153846, 0, 461.538461538462, 0, -934.82905982906, -5459.40170940171, 0.0123397435897436, 583333333.333333, -2458974358.97436, 30961.5384615385, 0, -974.358974358974, 134615384.615385, 2430.55555555556, -10245.7264957264, 0.12900641025641, -583333333.333333, -2458974358.97436, 0, 0, 12.8205128205128, -134615384.615385, -2430.55555555556, -10245.7264957264, 0, 224358974.358974, -1310256410.25641, 1615.38461538462, 0, 371.794871794872, 0, 934.82905982906, -5459.40170940171, 0.00673076923076923, -897435897.435898, 0, 2153.84615384615, 0, -2153.84615384615, 269230769.230769, -3739.31623931624, 0, 0.00897435897435897, 0, 5456410256.41026, -18846.1538461538, 0, -1282.05128205128, 0, 0, 22735.0427350427, -0.078525641025641, 897435897.435898, 0, -14717.9487179487, 0, -1794.8717948718, -269230769.230769, 3739.31623931624, 0, -0.0613247863247863, 0, 2082051282.05128, -4128.20512820513, 0, -1282.05128205128, 0, 0, 8675.21367521367, -0.0172008547008547;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(47) << 0, 0, -412820512.820513, -62820512.8205128, 0, 923.076923076923, 0, 0, -1720.08547008547, 0, 0, -664102564.102564, -62820512.8205128, 89743589.7435897, -1948.71794871795, 0, 0, -2767.09401709402, 0, 0, -664102564.102564, -62820512.8205128, -89743589.7435897, 25.6410256410256, 0, 0, -2767.09401709402, 0, 0, -412820512.820513, -62820512.8205128, 0, 743.589743589744, 0, 0, -1720.08547008547, 0, 0, 0, 179487179.48718, 179487179.48718, -4307.69230769231, 0, 0, 0, 0, 0, 1866666666.66667, 215384615.384615, 0, -2564.10256410256, 0, 0, 7777.77777777778, 0, 0, 0, 179487179.48718, -179487179.48718, -3589.74358974359, 0, 0, 0, 0, 0, 287179487.179487, 215384615.384615, 0, -2564.10256410256, 0, 0, 1196.5811965812;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(48) << 1153.84615384615, 0, 62820512.8205128, -35901997.8632479, -1869.65811965812, -133.297435897436, 0.00865384615384615, 0, -523.504273504273, 3333.33333333333, 0, 116666666.666667, -26927489.3162393, 2617.52136752137, 110.222756410256, 0.00576923076923077, 0, -299.145299145299, 1089.74358974359, 0, 116666666.666667, -26927489.3162393, -2617.52136752137, -123.087393162393, 0.0046474358974359, 0, -299.145299145299, 1243.58974358974, 0, 62820512.8205128, -35901997.8632479, 1869.65811965812, -97.3977564102564, 0.00827991452991453, 0, -523.504273504273, -2153.84615384615, 0, -179487179.48718, 89743589.7435897, -7478.63247863248, 394.813461538462, -0.0269230769230769, 0, 1495.7264957265, -1282.05128205128, 0, 215384615.384615, 143614273.504274, 0, 820.557692307692, -0.016025641025641, 0, 3589.74358974359, -1794.8717948718, 0, -179487179.48718, 89743589.7435897, 7478.63247863248, 143.558333333333, -0.0224358974358974, 0, 1495.7264957265, -1589.74358974359, 0, -215384615.384615, 71788290.5982906, 0, 51.2970085470085, -0.0173076923076923, 0, 1794.87179487179;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(49) << 0, 1153.84615384615, 0, -1869.65811965812, -35908354.7008547, 98.7426282051282, 0, 0.00865384615384615, 0, 0, 3333.33333333333, -89743589.7435897, 4861.11111111111, -26943568.3760684, -152.30608974359, 0, 0.00576923076923077, 747.863247863248, 0, 1089.74358974359, 89743589.7435897, -4861.11111111111, -26943568.3760684, 170.512820512821, 0, 0.0046474358974359, -747.863247863248, 0, 1243.58974358974, 0, 1869.65811965812, -35908354.7008547, 62.8339743589744, 0, 0.00827991452991453, 0, 0, -2153.84615384615, -179487179.48718, -7478.63247863248, 89743589.7435897, -394.853846153846, 0, -0.0269230769230769, 1495.7264957265, 0, -1282.05128205128, 0, 0, 143635213.675214, -646.310897435898, 0, -0.016025641025641, 0, 0, -1794.8717948718, 179487179.48718, 7478.63247863248, 89743589.7435897, 35.7747863247863, 0, -0.0224358974358974, -1495.7264957265, 0, -1589.74358974359, 0, 0, 71812222.2222222, -71.8292735042735, 0, -0.0173076923076923, 0;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(50) << 94230769.2307692, 0, 2884.61538461538, 0, 0, -125644465.811966, -130.876068376068, 0, 0.0197115384615385, 175000000, -134615384.615385, 8333.33333333333, 0, 0, -94236303.4188034, 205.662393162393, 186.965811965812, 0.018482905982906, 175000000, 134615384.615385, 2724.35897435897, 0, 0, -94236303.4188034, 205.662393162393, -186.965811965812, 0.0115651709401709, 94230769.2307692, 0, 3108.97435897436, 0, 0, -125644465.811966, -130.876068376068, 0, 0.019150641025641, -269230769.230769, -269230769.230769, -5384.61538461539, 0, 0, 314102564.102564, 373.931623931624, 373.931623931624, -0.0583333333333333, 323076923.076923, 0, -3205.12820512821, 0, 0, 502579658.119658, 3141.02564102564, 0, -0.0347222222222222, -269230769.230769, 269230769.230769, -4487.17948717949, 0, 0, 314102564.102564, 373.931623931624, -373.931623931624, -0.0486111111111111, -323076923.076923, 0, -3974.35897435897, 0, 0, 251284444.444444, 448.717948717949, 0, -0.0379273504273504;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(51) << -2280.98290598291, -934.82905982906, 0.0179487179487179, 0.0115384615384615, 0, 130.876068376068, -299.162406517094, -0.00701121794871795, -0.0011109764957265, -2206.19658119658, 1308.76068376068, -0.0168269230769231, 0.0237179487179487, 0, 579.594017094017, -224.375520833333, 0.00981570512820513, 0.000918677216880342, -2206.19658119658, -1308.76068376068, -0.00523504273504274, 0.00913461538461539, 0, 579.594017094017, -224.375520833333, -0.00981570512820513, -0.00102568028846154, -2280.98290598291, 934.82905982906, 0.0190705128205128, 0.0119123931623932, 0, 130.876068376068, -299.162406517094, 0.00701121794871795, -0.000811822783119658, 0, -3739.31623931624, -0.0291666666666667, -0.0269230769230769, 0, -373.931623931624, 747.863247863248, -0.0280448717948718, 0.00329037954059829, 12264.9572649573, 0, 0.0224358974358974, -0.016025641025641, 0, 3141.02564102564, 1196.67318376068, 0, 0.00683777510683761, 0, 3739.31623931624, -0.0157051282051282, -0.0224358974358974, 0, -373.931623931624, 747.863247863248, 0.0280448717948718, 0.00119646340811966, -3290.59829059829, 0, 0.00747863247863248, -0.0185897435897436, 0, -448.717948717949, 598.265918803419, 0, 0.000427406517094017;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(52) << -934.82905982906, -5459.40170940171, 0.0123397435897436, 0, 0.0115384615384615, 0, -0.00701121794871795, -299.18624465812, 0.000822742120726496, 2430.55555555556, -10245.7264957264, 0.12900641025641, 0, 0.0237179487179487, -186.965811965812, 0.0182291666666667, -224.435817307692, -0.0012703999732906, -2430.55555555556, -10245.7264957264, 0, 0, 0.00913461538461539, 186.965811965812, -0.0182291666666667, -224.435817307692, 0.00142094017094017, 934.82905982906, -5459.40170940171, 0.00673076923076923, 0, 0.0119123931623932, 0, 0.00701121794871795, -299.18624465812, 0.000523554754273504, -3739.31623931624, 0, 0.00897435897435897, 0, -0.0269230769230769, -373.931623931624, -0.0280448717948718, 747.863247863248, -0.00329053098290598, 0, 22735.0427350427, -0.078525641025641, 0, -0.016025641025641, 0, 0, 1196.75170940171, -0.00538520432692308, 3739.31623931624, 0, -0.0613247863247863, 0, -0.0224358974358974, 373.931623931624, 0.0280448717948718, 747.863247863248, 0.000298685363247863, 0, 8675.21367521367, -0.0172008547008547, 0, -0.0185897435897436, 0, 0, 598.355662393162, -0.000598419604700855;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(53) << 0, 0, -1720.08547008547, 523.504273504273, 0, 0.0278846153846154, 0, 0, -1047.02144764957, 0, 0, -2767.09401709402, 1196.5811965812, -747.863247863248, 0.0613247863247863, 0, 0, -785.277163461538, 0, 0, -2767.09401709402, 1196.5811965812, 747.863247863248, 0.0228098290598291, 0, 0, -785.277163461538, 0, 0, -1720.08547008547, 523.504273504273, 0, 0.0290064102564103, 0, 0, -1047.02144764957, 0, 0, 0, -1495.7264957265, -1495.7264957265, -0.0628205128205128, 0, 0, 2617.52136752137, 0, 0, 7777.77777777778, 3589.74358974359, 0, -0.0373931623931624, 0, 0, 4188.09252136752, 0, 0, 0, -1495.7264957265, 1495.7264957265, -0.0523504273504274, 0, 0, 2617.52136752137, 0, 0, 1196.5811965812, -1794.87179487179, 0, -0.0438034188034188, 0, 0, 2094.02606837607;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(54) << -1310256410.25641, -224358974.358974, 0, 346.153846153846, 0, 0, -5459.40170940171, -934.82905982906, 0, -1310256410.25641, 224358974.358974, -9782.05128205128, -102.564102564103, 0, 0, -5459.40170940171, 934.82905982906, -0.040758547008547, -2458974358.97436, -583333333.333333, -2961.53846153846, 435.897435897436, 0, -134615384.615385, -10245.7264957264, -2430.55555555556, -0.0123397435897436, -2458974358.97436, 583333333.333333, 1615.38461538462, 166.666666666667, 0, 134615384.615385, -10245.7264957264, 2430.55555555556, 0.00673076923076923, 2082051282.05128, 0, 3051.28205128205, -1589.74358974359, 0, 0, 8675.21367521367, 0, 0.0127136752136752, 0, 897435897.435898, -1435.89743589744, -1794.8717948718, 0, -269230769.230769, 0, 3739.31623931624, -0.00598290598290598, 5456410256.41026, 0, -897.435897435897, -153.846153846154, 0, 0, 22735.0427350427, 0, -0.00373931623931624, 0, -897435897.435898, 10410.2564102564, -1435.89743589744, 0, 269230769.230769, 0, -3739.31623931624, 0.0433760683760684;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(55) << -224358974.358974, -547435897.435898, 7000, 0, 346.153846153846, -94230769.2307692, -934.82905982906, -2280.98290598291, 0.0291666666666667, 224358974.358974, -547435897.435898, 14987.1794871794, 0, -102.564102564103, -94230769.2307692, 934.82905982906, -2280.98290598291, 0.0624465811965812, -314102564.102564, -529487179.48718, 4307.69230769231, 0, 435.897435897436, -94230769.2307692, -1308.76068376068, -2206.19658119658, 0.0179487179487179, 314102564.102564, -529487179.48718, 4576.92307692308, 0, 166.666666666667, -94230769.2307692, 1308.76068376068, -2206.19658119658, 0.0190705128205128, 0, -789743589.74359, -3948.71794871795, 0, -1589.74358974359, 323076923.076923, 0, -3290.59829059829, -0.0164529914529915, 897435897.435898, 0, -14179.4871794872, 0, -1794.8717948718, 269230769.230769, 3739.31623931624, 0, -0.0590811965811966, 0, 2943589743.58974, -1794.87179487179, 0, -153.846153846154, 323076923.076923, 0, 12264.9572649573, -0.00747863247863248, -897435897.435898, 0, -10948.7179487179, 0, -1435.89743589744, 269230769.230769, -3739.31623931624, 0, -0.0456196581196581;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(56) << 0, 0, -412820512.820513, 0, -62820512.8205128, 692.307692307692, 0, 0, -1720.08547008547, 0, 0, -412820512.820513, 0, -62820512.8205128, -205.128205128205, 0, 0, -1720.08547008547, 0, 0, -664102564.102564, -89743589.7435897, -62820512.8205128, 871.794871794872, 0, 0, -2767.09401709402, 0, 0, -664102564.102564, 89743589.7435897, -62820512.8205128, 333.333333333333, 0, 0, -2767.09401709402, 0, 0, 287179487.179487, 0, 215384615.384615, -3179.48717948718, 0, 0, 1196.5811965812, 0, 0, 0, -179487179.48718, 179487179.48718, -3589.74358974359, 0, 0, 0, 0, 0, 1866666666.66667, 0, 215384615.384615, -307.692307692308, 0, 0, 7777.77777777778, 0, 0, 0, 179487179.48718, 179487179.48718, -2871.79487179487, 0, 0, 0;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(57) << 910.25641025641, 0, 0, -35908354.7008547, -1869.65811965812, -8.97435897435897, 0.00667735042735043, 0, 0, 1358.97435897436, 0, 0, -35908354.7008547, 1869.65811965812, 26.8415598290598, 0.00480769230769231, 0, 0, 1153.84615384615, 0, 89743589.7435897, -26943568.3760684, -4861.11111111111, 8.94967948717949, 0.00844017094017094, 0, -747.863247863248, 1243.58974358974, 0, -89743589.7435897, -26943568.3760684, 4861.11111111111, -26.9096153846154, 0.00657051282051282, 0, 747.863247863248, -1282.05128205128, 0, 0, 71812222.2222222, 0, 71.8202991452992, -0.0185897435897436, 0, 0, -1794.8717948718, 0, 179487179.48718, 89743589.7435897, 7478.63247863248, 143.577777777778, -0.0224358974358974, 0, -1495.7264957265, -153.846153846154, 0, 0, 143635213.675214, 0, -71.8023504273504, -0.00192307692307692, 0, 0, -1435.89743589744, 0, -179487179.48718, 89743589.7435897, -7478.63247863248, -143.502991452991, -0.0179487179487179, 0, 1495.7264957265;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(58) << 0, 910.25641025641, 62820512.8205128, -1869.65811965812, -35901997.8632479, -46.0955128205128, 0, 0.00667735042735043, -523.504273504273, 0, 1358.97435897436, 62820512.8205128, 1869.65811965812, -35901997.8632479, -81.9263888888889, 0, 0.00480769230769231, -523.504273504273, 0, 1153.84615384615, 116666666.666667, -2617.52136752137, -26927489.3162393, 33.3692307692308, 0, 0.00844017094017094, -299.145299145299, 0, 1243.58974358974, 116666666.666667, 2617.52136752137, -26927489.3162393, -20.4746794871795, 0, 0.00657051282051282, -299.145299145299, 0, -1282.05128205128, -215384615.384615, 0, 71788290.5982906, -51.3149572649573, 0, -0.0185897435897436, 1794.87179487179, 0, -1794.8717948718, -179487179.48718, 7478.63247863248, 89743589.7435897, 35.7792735042735, 0, -0.0224358974358974, 1495.7264957265, 0, -153.846153846154, 215384615.384615, 0, 143614273.504274, 615.369658119658, 0, -0.00192307692307692, 3589.74358974359, 0, -1435.89743589744, -179487179.48718, -7478.63247863248, 89743589.7435897, 143.498504273504, 0, -0.0179487179487179, 1495.7264957265;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(59) << 0, 94230769.2307692, 2275.64102564103, 0, 0, -125644465.811966, 0, -130.876068376068, 0.0152510683760684, 0, 94230769.2307692, 3397.4358974359, 0, 0, -125644465.811966, 0, -130.876068376068, 0.0124465811965812, 134615384.615385, 175000000, 2884.61538461538, 0, 0, -94236303.4188034, -186.965811965812, 205.662393162393, 0.019284188034188, -134615384.615385, 175000000, 3108.97435897436, 0, 0, -94236303.4188034, 186.965811965812, 205.662393162393, 0.0157318376068376, 0, -323076923.076923, -3205.12820512821, 0, 0, 251284444.444444, 0, 448.717948717949, -0.0398504273504274, 269230769.230769, -269230769.230769, -4487.17948717949, 0, 0, 314102564.102564, -373.931623931624, 373.931623931624, -0.0486111111111111, 0, 323076923.076923, -384.615384615385, 0, 0, 502579658.119658, 0, 3141.02564102564, -0.00416666666666667, -269230769.230769, -269230769.230769, -3589.74358974359, 0, 0, 314102564.102564, 373.931623931624, 373.931623931624, -0.0388888888888889;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(60) << -5459.40170940171, -934.82905982906, 0, 0.00902777777777778, 0, 0, -299.18624465812, -0.00701121794871795, -7.47863247863248E-05, -5459.40170940171, 934.82905982906, -0.040758547008547, 0.0108974358974359, 0, 0, -299.18624465812, 0.00701121794871795, 0.00022405328525641, -10245.7264957264, -2430.55555555556, -0.0123397435897436, 0.0114316239316239, 0, 186.965811965812, -224.435817307692, -0.0182291666666667, 7.46937767094017E-05, -10245.7264957264, 2430.55555555556, 0.00673076923076923, 0.0110576923076923, 0, -186.965811965812, -224.435817307692, 0.0182291666666667, -0.000224308493589744, 8675.21367521367, 0, 0.0127136752136752, -0.0173076923076923, 0, 0, 598.355662393162, 0, 0.000598385950854701, 0, 3739.31623931624, -0.00598290598290598, -0.0224358974358974, 0, 373.931623931624, 747.863247863248, 0.0280448717948718, 0.00119653632478632, 22735.0427350427, 0, -0.00373931623931624, -0.00192307692307692, 0, 0, 1196.75170940171, 0, -0.000598318643162393, 0, -3739.31623931624, 0.0433760683760684, -0.0179487179487179, 0, -373.931623931624, 747.863247863248, -0.0280448717948718, -0.00119625587606838;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(61) << -934.82905982906, -2280.98290598291, 0.0291666666666667, 0, 0.00902777777777778, 130.876068376068, -0.00701121794871795, -299.162406517094, -0.000384396634615385, 934.82905982906, -2280.98290598291, 0.0624465811965812, 0, 0.0108974358974359, 130.876068376068, 0.00701121794871795, -299.162406517094, -0.000683292334401709, -1308.76068376068, -2206.19658119658, 0.0179487179487179, 0, 0.0114316239316239, 579.594017094017, -0.00981570512820513, -224.375520833333, 0.000277912393162393, 1308.76068376068, -2206.19658119658, 0.0190705128205128, 0, 0.0110576923076923, 579.594017094017, 0.00981570512820513, -224.375520833333, -0.000170797142094017, 0, -3290.59829059829, -0.0164529914529915, 0, -0.0173076923076923, -448.717948717949, 0, 598.265918803419, -0.000427473824786325, 3739.31623931624, 0, -0.0590811965811966, 0, -0.0224358974358974, -373.931623931624, 0.0280448717948718, 747.863247863248, 0.00029870219017094, 0, 12264.9572649573, -0.00747863247863248, 0, -0.00192307692307692, 3141.02564102564, 0, 1196.67318376068, 0.00512814903846154, -3739.31623931624, 0, -0.0456196581196581, 0, -0.0179487179487179, -373.931623931624, -0.0280448717948718, 747.863247863248, 0.0011962390491453;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(62) << 0, 0, -1720.08547008547, 0, 523.504273504273, 0.0218482905982906, 0, 0, -1047.02144764957, 0, 0, -1720.08547008547, 0, 523.504273504273, 0.027457264957265, 0, 0, -1047.02144764957, 0, 0, -2767.09401709402, 747.863247863248, 1196.5811965812, 0.0276709401709402, 0, 0, -785.277163461538, 0, 0, -2767.09401709402, -747.863247863248, 1196.5811965812, 0.0272970085470085, 0, 0, -785.277163461538, 0, 0, 1196.5811965812, 0, -1794.87179487179, -0.039957264957265, 0, 0, 2094.02606837607, 0, 0, 0, 1495.7264957265, -1495.7264957265, -0.0523504273504274, 0, 0, 2617.52136752137, 0, 0, 7777.77777777778, 0, 3589.74358974359, -0.00448717948717949, 0, 0, 4188.09252136752, 0, 0, 0, -1495.7264957265, -1495.7264957265, -0.0418803418803419, 0, 0, 2617.52136752137;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(63) << -529487179.48718, -314102564.102564, -4307.69230769231, 435.897435897436, 0, 94230769.2307692, -2206.19658119658, -1308.76068376068, -0.0179487179487179, -547435897.435898, 224358974.358974, -14987.1794871794, -102.564102564103, 0, 94230769.2307692, -2280.98290598291, 934.82905982906, -0.0624465811965812, -547435897.435898, -224358974.358974, -7000, 346.153846153846, 0, 94230769.2307692, -2280.98290598291, -934.82905982906, -0.0291666666666667, -529487179.48718, 314102564.102564, -4576.92307692308, 166.666666666667, 0, 94230769.2307692, -2206.19658119658, 1308.76068376068, -0.0190705128205128, 0, 897435897.435898, 14179.4871794872, -1794.8717948718, 0, -269230769.230769, 0, 3739.31623931624, 0.0590811965811966, -789743589.74359, 0, 3948.71794871795, -1589.74358974359, 0, -323076923.076923, -3290.59829059829, 0, 0.0164529914529915, 0, -897435897.435898, 10948.7179487179, -1435.89743589744, 0, -269230769.230769, 0, -3739.31623931624, 0.0456196581196581, 2943589743.58974, 0, 1794.87179487179, -153.846153846154, 0, -323076923.076923, 12264.9572649573, 0, 0.00747863247863248;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(64) << -583333333.333333, -2458974358.97436, 2961.53846153846, 0, 435.897435897436, 134615384.615385, -2430.55555555556, -10245.7264957264, 0.0123397435897436, 224358974.358974, -1310256410.25641, 9782.05128205128, 0, -102.564102564103, 0, 934.82905982906, -5459.40170940171, 0.040758547008547, -224358974.358974, -1310256410.25641, 0, 0, 346.153846153846, 0, -934.82905982906, -5459.40170940171, 0, 583333333.333333, -2458974358.97436, -1615.38461538462, 0, 166.666666666667, -134615384.615385, 2430.55555555556, -10245.7264957264, -0.00673076923076923, 897435897.435898, 0, 1435.89743589744, 0, -1794.8717948718, 269230769.230769, 3739.31623931624, 0, 0.00598290598290598, 0, 2082051282.05128, -3051.28205128205, 0, -1589.74358974359, 0, 0, 8675.21367521367, -0.0127136752136752, -897435897.435898, 0, -10410.2564102564, 0, -1435.89743589744, -269230769.230769, -3739.31623931624, 0, -0.0433760683760684, 0, 5456410256.41026, 897.435897435897, 0, -153.846153846154, 0, 0, 22735.0427350427, 0.00373931623931624;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(65) << 0, 0, -664102564.102564, 62820512.8205128, 89743589.7435897, 871.794871794872, 0, 0, -2767.09401709402, 0, 0, -412820512.820513, 62820512.8205128, 0, -205.128205128205, 0, 0, -1720.08547008547, 0, 0, -412820512.820513, 62820512.8205128, 0, 692.307692307692, 0, 0, -1720.08547008547, 0, 0, -664102564.102564, 62820512.8205128, -89743589.7435897, 333.333333333333, 0, 0, -2767.09401709402, 0, 0, 0, -179487179.48718, 179487179.48718, -3589.74358974359, 0, 0, 0, 0, 0, 287179487.179487, -215384615.384615, 0, -3179.48717948718, 0, 0, 1196.5811965812, 0, 0, 0, -179487179.48718, -179487179.48718, -2871.79487179487, 0, 0, 0, 0, 0, 1866666666.66667, -215384615.384615, 0, -307.692307692308, 0, 0, 7777.77777777778;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(66) << 1153.84615384615, 0, -116666666.666667, -26927489.3162393, -2617.52136752137, -33.3692307692308, 0.00844017094017094, 0, 299.145299145299, 1358.97435897436, 0, -62820512.8205128, -35901997.8632479, 1869.65811965812, 81.9263888888889, 0.00480769230769231, 0, 523.504273504273, 910.25641025641, 0, -62820512.8205128, -35901997.8632479, -1869.65811965812, 46.0955128205128, 0.00667735042735043, 0, 523.504273504273, 1243.58974358974, 0, -116666666.666667, -26927489.3162393, 2617.52136752137, 20.4746794871795, 0.00657051282051282, 0, 299.145299145299, -1794.8717948718, 0, 179487179.48718, 89743589.7435897, 7478.63247863248, -35.7792735042735, -0.0224358974358974, 0, -1495.7264957265, -1282.05128205128, 0, 215384615.384615, 71788290.5982906, 0, 51.3149572649573, -0.0185897435897436, 0, -1794.87179487179, -1435.89743589744, 0, 179487179.48718, 89743589.7435897, -7478.63247863248, -143.498504273504, -0.0179487179487179, 0, -1495.7264957265, -153.846153846154, 0, -215384615.384615, 143614273.504274, 0, -615.369658119658, -0.00192307692307692, 0, -3589.74358974359;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(67) << 0, 1153.84615384615, -89743589.7435897, -4861.11111111111, -26943568.3760684, -8.94967948717949, 0, 0.00844017094017094, 747.863247863248, 0, 1358.97435897436, 0, 1869.65811965812, -35908354.7008547, -26.8415598290598, 0, 0.00480769230769231, 0, 0, 910.25641025641, 0, -1869.65811965812, -35908354.7008547, 8.97435897435897, 0, 0.00667735042735043, 0, 0, 1243.58974358974, 89743589.7435897, 4861.11111111111, -26943568.3760684, 26.9096153846154, 0, 0.00657051282051282, -747.863247863248, 0, -1794.8717948718, -179487179.48718, 7478.63247863248, 89743589.7435897, -143.577777777778, 0, -0.0224358974358974, 1495.7264957265, 0, -1282.05128205128, 0, 0, 71812222.2222222, -71.8202991452992, 0, -0.0185897435897436, 0, 0, -1435.89743589744, 179487179.48718, -7478.63247863248, 89743589.7435897, 143.502991452991, 0, -0.0179487179487179, -1495.7264957265, 0, -153.846153846154, 0, 0, 143635213.675214, 71.8023504273504, 0, -0.00192307692307692, 0;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(68) << -175000000, -134615384.615385, 2884.61538461538, 0, 0, -94236303.4188034, -205.662393162393, 186.965811965812, 0.019284188034188, -94230769.2307692, 0, 3397.4358974359, 0, 0, -125644465.811966, 130.876068376068, 0, 0.0124465811965812, -94230769.2307692, 0, 2275.64102564103, 0, 0, -125644465.811966, 130.876068376068, 0, 0.0152510683760684, -175000000, 134615384.615385, 3108.97435897436, 0, 0, -94236303.4188034, -205.662393162393, -186.965811965812, 0.0157318376068376, 269230769.230769, -269230769.230769, -4487.17948717949, 0, 0, 314102564.102564, -373.931623931624, 373.931623931624, -0.0486111111111111, 323076923.076923, 0, -3205.12820512821, 0, 0, 251284444.444444, -448.717948717949, 0, -0.0398504273504274, 269230769.230769, 269230769.230769, -3589.74358974359, 0, 0, 314102564.102564, -373.931623931624, -373.931623931624, -0.0388888888888889, -323076923.076923, 0, -384.615384615385, 0, 0, 502579658.119658, -3141.02564102564, 0, -0.00416666666666667;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(69) << -2206.19658119658, -1308.76068376068, -0.0179487179487179, 0.0114316239316239, 0, -579.594017094017, -224.375520833333, -0.00981570512820513, -0.000277912393162393, -2280.98290598291, 934.82905982906, -0.0624465811965812, 0.0108974358974359, 0, -130.876068376068, -299.162406517094, 0.00701121794871795, 0.000683292334401709, -2280.98290598291, -934.82905982906, -0.0291666666666667, 0.00902777777777778, 0, -130.876068376068, -299.162406517094, -0.00701121794871795, 0.000384396634615385, -2206.19658119658, 1308.76068376068, -0.0190705128205128, 0.0110576923076923, 0, -579.594017094017, -224.375520833333, 0.00981570512820513, 0.000170797142094017, 0, 3739.31623931624, 0.0590811965811966, -0.0224358974358974, 0, 373.931623931624, 747.863247863248, 0.0280448717948718, -0.00029870219017094, -3290.59829059829, 0, 0.0164529914529915, -0.0173076923076923, 0, 448.717948717949, 598.265918803419, 0, 0.000427473824786325, 0, -3739.31623931624, 0.0456196581196581, -0.0179487179487179, 0, 373.931623931624, 747.863247863248, -0.0280448717948718, -0.0011962390491453, 12264.9572649573, 0, 0.00747863247863248, -0.00192307692307692, 0, -3141.02564102564, 1196.67318376068, 0, -0.00512814903846154;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(70) << -2430.55555555556, -10245.7264957264, 0.0123397435897436, 0, 0.0114316239316239, -186.965811965812, -0.0182291666666667, -224.435817307692, -7.46937767094017E-05, 934.82905982906, -5459.40170940171, 0.040758547008547, 0, 0.0108974358974359, 0, 0.00701121794871795, -299.18624465812, -0.00022405328525641, -934.82905982906, -5459.40170940171, 0, 0, 0.00902777777777778, 0, -0.00701121794871795, -299.18624465812, 7.47863247863248E-05, 2430.55555555556, -10245.7264957264, -0.00673076923076923, 0, 0.0110576923076923, 186.965811965812, 0.0182291666666667, -224.435817307692, 0.000224308493589744, 3739.31623931624, 0, 0.00598290598290598, 0, -0.0224358974358974, -373.931623931624, 0.0280448717948718, 747.863247863248, -0.00119653632478632, 0, 8675.21367521367, -0.0127136752136752, 0, -0.0173076923076923, 0, 0, 598.355662393162, -0.000598385950854701, -3739.31623931624, 0, -0.0433760683760684, 0, -0.0179487179487179, 373.931623931624, -0.0280448717948718, 747.863247863248, 0.00119625587606838, 0, 22735.0427350427, 0.00373931623931624, 0, -0.00192307692307692, 0, 0, 1196.75170940171, 0.000598318643162393;
    //Expected_JacobianK_NoDispSmallVelWithDamping.row(71) << 0, 0, -2767.09401709402, -1196.5811965812, -747.863247863248, 0.0276709401709402, 0, 0, -785.277163461538, 0, 0, -1720.08547008547, -523.504273504273, 0, 0.027457264957265, 0, 0, -1047.02144764957, 0, 0, -1720.08547008547, -523.504273504273, 0, 0.0218482905982906, 0, 0, -1047.02144764957, 0, 0, -2767.09401709402, -1196.5811965812, 747.863247863248, 0.0272970085470085, 0, 0, -785.277163461538, 0, 0, 0, 1495.7264957265, -1495.7264957265, -0.0523504273504274, 0, 0, 2617.52136752137, 0, 0, 1196.5811965812, 1794.87179487179, 0, -0.039957264957265, 0, 0, 2094.02606837607, 0, 0, 0, 1495.7264957265, 1495.7264957265, -0.0418803418803419, 0, 0, 2617.52136752137, 0, 0, 7777.77777777778, -3589.74358974359, 0, -0.00448717948717949, 0, 0, 4188.09252136752;

    //ChMatrixNM<double, 72, 72> Expected_JacobianR_NoDispSmallVelWithDamping;
    //
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(0) << 21000000, 9535256.41025641, 0, 0, 0, -807692.307692308, 87.5, 39.730235042735, 0, 10320512.8205128, -336538.461538461, 0, 0, 0, 538461.538461538, 43.0021367521368, -1.40224358974359, 0, 9288461.53846154, 3926282.05128205, 0, 0, 0, 201923.076923077, 38.7019230769231, 16.3595085470085, 0, 7852564.1025641, 336538.461538461, 0, 0, 0, 201923.076923077, 32.7190170940171, 1.40224358974359, 0, -24589743.5897436, -3141025.64102564, 0, 0, 0, -1346153.84615385, -102.457264957265, -13.0876068376068, 0, -5474358.97435897, -2243589.74358974, 0, 0, 0, 942307.692307692, -22.8098290598291, -9.3482905982906, 0, -13102564.1025641, -2243589.74358974, 0, 0, 0, 0, -54.5940170940171, -9.3482905982906, 0, -5294871.7948718, -5833333.33333333, 0, 0, 0, -1750000, -22.0619658119658, -24.3055555555556, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(1) << 9535256.41025641, 21000000, 0, 0, 0, -807692.307692308, 39.730235042735, 87.5, 0, 336538.461538461, 7852564.1025641, 0, 0, 0, 201923.076923077, 1.40224358974359, 32.7190170940171, 0, 3926282.05128205, 9288461.53846154, 0, 0, 0, 201923.076923077, 16.3595085470085, 38.7019230769231, 0, -336538.461538461, 10320512.8205128, 0, 0, 0, 538461.538461538, -1.40224358974359, 43.0021367521368, 0, -5833333.33333333, -5294871.7948718, 0, 0, 0, -1750000, -24.3055555555556, -22.0619658119658, 0, -2243589.74358974, -13102564.1025641, 0, 0, 0, 0, -9.3482905982906, -54.5940170940171, 0, -2243589.74358974, -5474358.97435897, 0, 0, 0, 942307.692307692, -9.3482905982906, -22.8098290598291, 0, -3141025.64102564, -24589743.5897436, 0, 0, 0, -1346153.84615385, -13.0876068376068, -102.457264957265, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(2) << 0, 0, 9333333.33333333, -538461.538461539, -538461.538461539, 0, 0, 0, 38.8888888888889, 0, 0, 4038461.53846154, 358974.358974359, 134615.384615385, 0, 0, 0, 16.8269230769231, 0, 0, 4128205.12820513, 134615.384615385, 134615.384615385, 0, 0, 0, 17.2008547008547, 0, 0, 4038461.53846154, 134615.384615385, 358974.358974359, 0, 0, 0, 16.8269230769231, 0, 0, -6641025.64102564, -897435.897435898, -1166666.66666667, 0, 0, 0, -27.6709401709402, 0, 0, -4128205.12820513, 628205.128205128, 0, 0, 0, 0, -17.2008547008547, 0, 0, -4128205.12820513, 0, 628205.128205128, 0, 0, 0, -17.2008547008547, 0, 0, -6641025.64102564, -1166666.66666667, -897435.897435898, 0, 0, 0, -27.6709401709402;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(3) << 0, 0, -538461.538461539, 269405.769230769, 79.4604700854701, 0, 0, 0, -8.97435897435897, 0, 0, -358974.358974359, 89829.594017094, -2.80448717948718, 0, 0, 0, 2.99145299145299, 0, 0, -134615.384615385, 134692.788461538, 32.7190170940171, 0, 0, 0, 1.12179487179487, 0, 0, 134615.384615385, 89809.0277777778, 2.80448717948718, 0, 0, 0, 2.24358974358974, 0, 0, 897435.897435898, -269435.683760684, -26.1752136752137, 0, 0, 0, -7.47863247863248, 0, 0, -628205.128205128, -359019.978632479, -18.6965811965812, 0, 0, 0, 5.23504273504273, 0, 0, 0, -359083.547008547, -18.6965811965812, 0, 0, 0, 0, 0, 0, 628205.128205128, -269274.893162393, -48.6111111111111, 0, 0, 0, -11.965811965812;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(4) << 0, 0, -538461.538461539, 79.4604700854701, 269405.769230769, 0, 0, 0, -8.97435897435897, 0, 0, 134615.384615385, 2.80448717948718, 89809.0277777778, 0, 0, 0, 2.24358974358974, 0, 0, -134615.384615385, 32.7190170940171, 134692.788461538, 0, 0, 0, 1.12179487179487, 0, 0, -358974.358974359, -2.80448717948718, 89829.594017094, 0, 0, 0, 2.99145299145299, 0, 0, 628205.128205128, -48.6111111111111, -269274.893162393, 0, 0, 0, -11.965811965812, 0, 0, 0, -18.6965811965812, -359083.547008547, 0, 0, 0, 0, 0, 0, -628205.128205128, -18.6965811965812, -359019.978632479, 0, 0, 0, 5.23504273504273, 0, 0, 897435.897435898, -26.1752136752137, -269435.683760684, 0, 0, 0, -7.47863247863248;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(5) << -807692.307692308, -807692.307692308, 0, 0, 0, 942385.47008547, -7.8525641025641, -7.8525641025641, 0, -538461.538461538, 201923.076923077, 0, 0, 0, 314136.217948718, 0.747863247863248, 1.96314102564103, 0, -201923.076923077, -201923.076923077, 0, 0, 0, 471188.247863248, 0.280448717948718, 0.280448717948718, 0, 201923.076923077, -538461.538461538, 0, 0, 0, 314136.217948718, 1.96314102564103, 0.747863247863248, 0, 1346153.84615385, 942307.692307692, 0, 0, 0, -942363.034188034, -1.86965811965812, -5.79594017094017, 0, -942307.692307692, 0, 0, 0, 0, -1256444.65811966, 1.30876068376068, 0, 0, 0, -942307.692307692, 0, 0, 0, -1256444.65811966, 0, 1.30876068376068, 0, 942307.692307692, 1346153.84615385, 0, 0, 0, -942363.034188034, -5.79594017094017, -1.86965811965812, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(6) << 87.5, 39.730235042735, 0, 0, 0, -7.8525641025641, 2.24424599358974, 0.000297976762820513, 0, 43.0021367521368, -1.40224358974359, 0, 0, 0, -0.747863247863248, 0.748185763888889, -1.05168269230769E-05, 0, 38.7019230769231, 16.3595085470085, 0, 0, 0, -0.280448717948718, 1.12208513621795, 0.000122696314102564, 0, 32.7190170940171, 1.40224358974359, 0, 0, 0, 1.96314102564103, 0.748108640491453, 1.05168269230769E-05, 0, -102.457264957265, -13.0876068376068, 0, 0, 0, 1.86965811965812, -2.24435817307692, -9.81570512820513E-05, 0, -22.8098290598291, -9.3482905982906, 0, 0, 0, -1.30876068376068, -2.99162406517094, -7.01121794871795E-05, 0, -54.5940170940171, -9.3482905982906, 0, 0, 0, 0, -2.9918624465812, -7.01121794871795E-05, 0, -22.0619658119658, -24.3055555555556, 0, 0, 0, -2.05662393162393, -2.24375520833333, -0.000182291666666667, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(7) << 39.730235042735, 87.5, 0, 0, 0, -7.8525641025641, 0.000297976762820513, 2.24424599358974, 0, 1.40224358974359, 32.7190170940171, 0, 0, 0, 1.96314102564103, 1.05168269230769E-05, 0.748108640491453, 0, 16.3595085470085, 38.7019230769231, 0, 0, 0, -0.280448717948718, 0.000122696314102564, 1.12208513621795, 0, -1.40224358974359, 43.0021367521368, 0, 0, 0, -0.747863247863248, -1.05168269230769E-05, 0.748185763888889, 0, -24.3055555555556, -22.0619658119658, 0, 0, 0, -2.05662393162393, -0.000182291666666667, -2.24375520833333, 0, -9.3482905982906, -54.5940170940171, 0, 0, 0, 0, -7.01121794871795E-05, -2.9918624465812, 0, -9.3482905982906, -22.8098290598291, 0, 0, 0, -1.30876068376068, -7.01121794871795E-05, -2.99162406517094, 0, -13.0876068376068, -102.457264957265, 0, 0, 0, 1.86965811965812, -9.81570512820513E-05, -2.24435817307692, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(8) << 0, 0, 38.8888888888889, -8.97435897435897, -8.97435897435897, 0, 0, 0, 7.85285576923077, 0, 0, 16.8269230769231, -2.99145299145299, 2.24358974358974, 0, 0, 0, 2.61764756944444, 0, 0, 17.2008547008547, -1.12179487179487, -1.12179487179487, 0, 0, 0, 3.92641105769231, 0, 0, 16.8269230769231, 2.24358974358974, -2.99145299145299, 0, 0, 0, 2.61764756944444, 0, 0, -27.6709401709402, 7.47863247863248, 2.99145299145299, 0, 0, 0, -7.85277163461538, 0, 0, -17.2008547008547, -5.23504273504273, 0, 0, 0, 0, -10.4702144764957, 0, 0, -17.2008547008547, 0, -5.23504273504273, 0, 0, 0, -10.4702144764957, 0, 0, -27.6709401709402, 2.99145299145299, 7.47863247863248, 0, 0, 0, -7.85277163461538;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(9) << 10320512.8205128, 336538.461538461, 0, 0, 0, -538461.538461538, 43.0021367521368, 1.40224358974359, 0, 21000000, -9535256.41025641, 0, 0, 0, 807692.307692308, 87.5, -39.730235042735, 0, 7852564.1025641, -336538.461538461, 0, 0, 0, -201923.076923077, 32.7190170940171, -1.40224358974359, 0, 9288461.53846154, -3926282.05128205, 0, 0, 0, -201923.076923077, 38.7019230769231, -16.3595085470085, 0, -24589743.5897436, 3141025.64102564, 0, 0, 0, 1346153.84615385, -102.457264957265, 13.0876068376068, 0, -5294871.7948718, 5833333.33333333, 0, 0, 0, 1750000, -22.0619658119658, 24.3055555555556, 0, -13102564.1025641, 2243589.74358974, 0, 0, 0, 0, -54.5940170940171, 9.3482905982906, 0, -5474358.97435897, 2243589.74358974, 0, 0, 0, -942307.692307692, -22.8098290598291, 9.3482905982906, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(10) << -336538.461538461, 7852564.1025641, 0, 0, 0, 201923.076923077, -1.40224358974359, 32.7190170940171, 0, -9535256.41025641, 21000000, 0, 0, 0, -807692.307692308, -39.730235042735, 87.5, 0, 336538.461538461, 10320512.8205128, 0, 0, 0, 538461.538461538, 1.40224358974359, 43.0021367521368, 0, -3926282.05128205, 9288461.53846154, 0, 0, 0, 201923.076923077, -16.3595085470085, 38.7019230769231, 0, 5833333.33333333, -5294871.7948718, 0, 0, 0, -1750000, 24.3055555555556, -22.0619658119658, 0, 3141025.64102564, -24589743.5897436, 0, 0, 0, -1346153.84615385, 13.0876068376068, -102.457264957265, 0, 2243589.74358974, -5474358.97435897, 0, 0, 0, 942307.692307692, 9.3482905982906, -22.8098290598291, 0, 2243589.74358974, -13102564.1025641, 0, 0, 0, 0, 9.3482905982906, -54.5940170940171, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(11) << 0, 0, 4038461.53846154, -358974.358974359, 134615.384615385, 0, 0, 0, 16.8269230769231, 0, 0, 9333333.33333333, 538461.538461539, -538461.538461539, 0, 0, 0, 38.8888888888889, 0, 0, 4038461.53846154, -134615.384615385, 358974.358974359, 0, 0, 0, 16.8269230769231, 0, 0, 4128205.12820513, -134615.384615385, 134615.384615385, 0, 0, 0, 17.2008547008547, 0, 0, -6641025.64102564, 897435.897435898, -1166666.66666667, 0, 0, 0, -27.6709401709402, 0, 0, -6641025.64102564, 1166666.66666667, -897435.897435898, 0, 0, 0, -27.6709401709402, 0, 0, -4128205.12820513, 0, 628205.128205128, 0, 0, 0, -17.2008547008547, 0, 0, -4128205.12820513, -628205.128205128, 0, 0, 0, 0, -17.2008547008547;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(12) << 0, 0, 358974.358974359, 89829.594017094, 2.80448717948718, 0, 0, 0, -2.99145299145299, 0, 0, 538461.538461539, 269405.769230769, -79.4604700854701, 0, 0, 0, 8.97435897435897, 0, 0, -134615.384615385, 89809.0277777778, -2.80448717948718, 0, 0, 0, -2.24358974358974, 0, 0, 134615.384615385, 134692.788461538, -32.7190170940171, 0, 0, 0, -1.12179487179487, 0, 0, -897435.897435898, -269435.683760684, 26.1752136752137, 0, 0, 0, 7.47863247863248, 0, 0, -628205.128205128, -269274.893162393, 48.6111111111111, 0, 0, 0, 11.965811965812, 0, 0, 0, -359083.547008547, 18.6965811965812, 0, 0, 0, 0, 0, 0, 628205.128205128, -359019.978632479, 18.6965811965812, 0, 0, 0, -5.23504273504273;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(13) << 0, 0, 134615.384615385, -2.80448717948718, 89809.0277777778, 0, 0, 0, 2.24358974358974, 0, 0, -538461.538461539, -79.4604700854701, 269405.769230769, 0, 0, 0, -8.97435897435897, 0, 0, -358974.358974359, 2.80448717948718, 89829.594017094, 0, 0, 0, 2.99145299145299, 0, 0, -134615.384615385, -32.7190170940171, 134692.788461538, 0, 0, 0, 1.12179487179487, 0, 0, 628205.128205128, 48.6111111111111, -269274.893162393, 0, 0, 0, -11.965811965812, 0, 0, 897435.897435898, 26.1752136752137, -269435.683760684, 0, 0, 0, -7.47863247863248, 0, 0, -628205.128205128, 18.6965811965812, -359019.978632479, 0, 0, 0, 5.23504273504273, 0, 0, 0, 18.6965811965812, -359083.547008547, 0, 0, 0, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(14) << 538461.538461538, 201923.076923077, 0, 0, 0, 314136.217948718, -0.747863247863248, 1.96314102564103, 0, 807692.307692308, -807692.307692308, 0, 0, 0, 942385.47008547, 7.8525641025641, -7.8525641025641, 0, -201923.076923077, -538461.538461538, 0, 0, 0, 314136.217948718, -1.96314102564103, 0.747863247863248, 0, 201923.076923077, -201923.076923077, 0, 0, 0, 471188.247863248, -0.280448717948718, 0.280448717948718, 0, -1346153.84615385, 942307.692307692, 0, 0, 0, -942363.034188034, 1.86965811965812, -5.79594017094017, 0, -942307.692307692, 1346153.84615385, 0, 0, 0, -942363.034188034, 5.79594017094017, -1.86965811965812, 0, 0, -942307.692307692, 0, 0, 0, -1256444.65811966, 0, 1.30876068376068, 0, 942307.692307692, 0, 0, 0, 0, -1256444.65811966, -1.30876068376068, 0, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(15) << 43.0021367521368, 1.40224358974359, 0, 0, 0, 0.747863247863248, 0.748185763888889, 1.05168269230769E-05, 0, 87.5, -39.730235042735, 0, 0, 0, 7.8525641025641, 2.24424599358974, -0.000297976762820513, 0, 32.7190170940171, -1.40224358974359, 0, 0, 0, -1.96314102564103, 0.748108640491453, -1.05168269230769E-05, 0, 38.7019230769231, -16.3595085470085, 0, 0, 0, 0.280448717948718, 1.12208513621795, -0.000122696314102564, 0, -102.457264957265, 13.0876068376068, 0, 0, 0, -1.86965811965812, -2.24435817307692, 9.81570512820513E-05, 0, -22.0619658119658, 24.3055555555556, 0, 0, 0, 2.05662393162393, -2.24375520833333, 0.000182291666666667, 0, -54.5940170940171, 9.3482905982906, 0, 0, 0, 0, -2.9918624465812, 7.01121794871795E-05, 0, -22.8098290598291, 9.3482905982906, 0, 0, 0, 1.30876068376068, -2.99162406517094, 7.01121794871795E-05, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(16) << -1.40224358974359, 32.7190170940171, 0, 0, 0, 1.96314102564103, -1.05168269230769E-05, 0.748108640491453, 0, -39.730235042735, 87.5, 0, 0, 0, -7.8525641025641, -0.000297976762820513, 2.24424599358974, 0, 1.40224358974359, 43.0021367521368, 0, 0, 0, -0.747863247863248, 1.05168269230769E-05, 0.748185763888889, 0, -16.3595085470085, 38.7019230769231, 0, 0, 0, -0.280448717948718, -0.000122696314102564, 1.12208513621795, 0, 24.3055555555556, -22.0619658119658, 0, 0, 0, -2.05662393162393, 0.000182291666666667, -2.24375520833333, 0, 13.0876068376068, -102.457264957265, 0, 0, 0, 1.86965811965812, 9.81570512820513E-05, -2.24435817307692, 0, 9.3482905982906, -22.8098290598291, 0, 0, 0, -1.30876068376068, 7.01121794871795E-05, -2.99162406517094, 0, 9.3482905982906, -54.5940170940171, 0, 0, 0, 0, 7.01121794871795E-05, -2.9918624465812, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(17) << 0, 0, 16.8269230769231, 2.99145299145299, 2.24358974358974, 0, 0, 0, 2.61764756944444, 0, 0, 38.8888888888889, 8.97435897435897, -8.97435897435897, 0, 0, 0, 7.85285576923077, 0, 0, 16.8269230769231, -2.24358974358974, -2.99145299145299, 0, 0, 0, 2.61764756944444, 0, 0, 17.2008547008547, 1.12179487179487, -1.12179487179487, 0, 0, 0, 3.92641105769231, 0, 0, -27.6709401709402, -7.47863247863248, 2.99145299145299, 0, 0, 0, -7.85277163461538, 0, 0, -27.6709401709402, -2.99145299145299, 7.47863247863248, 0, 0, 0, -7.85277163461538, 0, 0, -17.2008547008547, 0, -5.23504273504273, 0, 0, 0, -10.4702144764957, 0, 0, -17.2008547008547, 5.23504273504273, 0, 0, 0, 0, -10.4702144764957;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(18) << 9288461.53846154, 3926282.05128205, 0, 0, 0, -201923.076923077, 38.7019230769231, 16.3595085470085, 0, 7852564.1025641, 336538.461538461, 0, 0, 0, -201923.076923077, 32.7190170940171, 1.40224358974359, 0, 21000000, 9535256.41025641, 0, 0, 0, 807692.307692308, 87.5, 39.730235042735, 0, 10320512.8205128, -336538.461538461, 0, 0, 0, -538461.538461538, 43.0021367521368, -1.40224358974359, 0, -13102564.1025641, -2243589.74358974, 0, 0, 0, 0, -54.5940170940171, -9.3482905982906, 0, -5294871.7948718, -5833333.33333333, 0, 0, 0, 1750000, -22.0619658119658, -24.3055555555556, 0, -24589743.5897436, -3141025.64102564, 0, 0, 0, 1346153.84615385, -102.457264957265, -13.0876068376068, 0, -5474358.97435897, -2243589.74358974, 0, 0, 0, -942307.692307692, -22.8098290598291, -9.3482905982906, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(19) << 3926282.05128205, 9288461.53846154, 0, 0, 0, -201923.076923077, 16.3595085470085, 38.7019230769231, 0, -336538.461538461, 10320512.8205128, 0, 0, 0, -538461.538461538, -1.40224358974359, 43.0021367521368, 0, 9535256.41025641, 21000000, 0, 0, 0, 807692.307692308, 39.730235042735, 87.5, 0, 336538.461538461, 7852564.1025641, 0, 0, 0, -201923.076923077, 1.40224358974359, 32.7190170940171, 0, -2243589.74358974, -5474358.97435897, 0, 0, 0, -942307.692307692, -9.3482905982906, -22.8098290598291, 0, -3141025.64102564, -24589743.5897436, 0, 0, 0, 1346153.84615385, -13.0876068376068, -102.457264957265, 0, -5833333.33333333, -5294871.7948718, 0, 0, 0, 1750000, -24.3055555555556, -22.0619658119658, 0, -2243589.74358974, -13102564.1025641, 0, 0, 0, 0, -9.3482905982906, -54.5940170940171, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(20) << 0, 0, 4128205.12820513, -134615.384615385, -134615.384615385, 0, 0, 0, 17.2008547008547, 0, 0, 4038461.53846154, -134615.384615385, -358974.358974359, 0, 0, 0, 16.8269230769231, 0, 0, 9333333.33333333, 538461.538461539, 538461.538461539, 0, 0, 0, 38.8888888888889, 0, 0, 4038461.53846154, -358974.358974359, -134615.384615385, 0, 0, 0, 16.8269230769231, 0, 0, -4128205.12820513, 0, -628205.128205128, 0, 0, 0, -17.2008547008547, 0, 0, -6641025.64102564, 1166666.66666667, 897435.897435898, 0, 0, 0, -27.6709401709402, 0, 0, -6641025.64102564, 897435.897435898, 1166666.66666667, 0, 0, 0, -27.6709401709402, 0, 0, -4128205.12820513, -628205.128205128, 0, 0, 0, 0, -17.2008547008547;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(21) << 0, 0, 134615.384615385, 134692.788461538, 32.7190170940171, 0, 0, 0, -1.12179487179487, 0, 0, -134615.384615385, 89809.0277777778, 2.80448717948718, 0, 0, 0, -2.24358974358974, 0, 0, 538461.538461539, 269405.769230769, 79.4604700854701, 0, 0, 0, 8.97435897435897, 0, 0, 358974.358974359, 89829.594017094, -2.80448717948718, 0, 0, 0, -2.99145299145299, 0, 0, 0, -359083.547008547, -18.6965811965812, 0, 0, 0, 0, 0, 0, -628205.128205128, -269274.893162393, -48.6111111111111, 0, 0, 0, 11.965811965812, 0, 0, -897435.897435898, -269435.683760684, -26.1752136752137, 0, 0, 0, 7.47863247863248, 0, 0, 628205.128205128, -359019.978632479, -18.6965811965812, 0, 0, 0, -5.23504273504273;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(22) << 0, 0, 134615.384615385, 32.7190170940171, 134692.788461538, 0, 0, 0, -1.12179487179487, 0, 0, 358974.358974359, -2.80448717948718, 89829.594017094, 0, 0, 0, -2.99145299145299, 0, 0, 538461.538461539, 79.4604700854701, 269405.769230769, 0, 0, 0, 8.97435897435897, 0, 0, -134615.384615385, 2.80448717948718, 89809.0277777778, 0, 0, 0, -2.24358974358974, 0, 0, 628205.128205128, -18.6965811965812, -359019.978632479, 0, 0, 0, -5.23504273504273, 0, 0, -897435.897435898, -26.1752136752137, -269435.683760684, 0, 0, 0, 7.47863247863248, 0, 0, -628205.128205128, -48.6111111111111, -269274.893162393, 0, 0, 0, 11.965811965812, 0, 0, 0, -18.6965811965812, -359083.547008547, 0, 0, 0, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(23) << 201923.076923077, 201923.076923077, 0, 0, 0, 471188.247863248, -0.280448717948718, -0.280448717948718, 0, -201923.076923077, 538461.538461538, 0, 0, 0, 314136.217948718, -1.96314102564103, -0.747863247863248, 0, 807692.307692308, 807692.307692308, 0, 0, 0, 942385.47008547, 7.8525641025641, 7.8525641025641, 0, 538461.538461538, -201923.076923077, 0, 0, 0, 314136.217948718, -0.747863247863248, -1.96314102564103, 0, 0, 942307.692307692, 0, 0, 0, -1256444.65811966, 0, -1.30876068376068, 0, -942307.692307692, -1346153.84615385, 0, 0, 0, -942363.034188034, 5.79594017094017, 1.86965811965812, 0, -1346153.84615385, -942307.692307692, 0, 0, 0, -942363.034188034, 1.86965811965812, 5.79594017094017, 0, 942307.692307692, 0, 0, 0, 0, -1256444.65811966, -1.30876068376068, 0, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(24) << 38.7019230769231, 16.3595085470085, 0, 0, 0, 0.280448717948718, 1.12208513621795, 0.000122696314102564, 0, 32.7190170940171, 1.40224358974359, 0, 0, 0, -1.96314102564103, 0.748108640491453, 1.05168269230769E-05, 0, 87.5, 39.730235042735, 0, 0, 0, 7.8525641025641, 2.24424599358974, 0.000297976762820513, 0, 43.0021367521368, -1.40224358974359, 0, 0, 0, 0.747863247863248, 0.748185763888889, -1.05168269230769E-05, 0, -54.5940170940171, -9.3482905982906, 0, 0, 0, 0, -2.9918624465812, -7.01121794871795E-05, 0, -22.0619658119658, -24.3055555555556, 0, 0, 0, 2.05662393162393, -2.24375520833333, -0.000182291666666667, 0, -102.457264957265, -13.0876068376068, 0, 0, 0, -1.86965811965812, -2.24435817307692, -9.81570512820513E-05, 0, -22.8098290598291, -9.3482905982906, 0, 0, 0, 1.30876068376068, -2.99162406517094, -7.01121794871795E-05, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(25) << 16.3595085470085, 38.7019230769231, 0, 0, 0, 0.280448717948718, 0.000122696314102564, 1.12208513621795, 0, -1.40224358974359, 43.0021367521368, 0, 0, 0, 0.747863247863248, -1.05168269230769E-05, 0.748185763888889, 0, 39.730235042735, 87.5, 0, 0, 0, 7.8525641025641, 0.000297976762820513, 2.24424599358974, 0, 1.40224358974359, 32.7190170940171, 0, 0, 0, -1.96314102564103, 1.05168269230769E-05, 0.748108640491453, 0, -9.3482905982906, -22.8098290598291, 0, 0, 0, 1.30876068376068, -7.01121794871795E-05, -2.99162406517094, 0, -13.0876068376068, -102.457264957265, 0, 0, 0, -1.86965811965812, -9.81570512820513E-05, -2.24435817307692, 0, -24.3055555555556, -22.0619658119658, 0, 0, 0, 2.05662393162393, -0.000182291666666667, -2.24375520833333, 0, -9.3482905982906, -54.5940170940171, 0, 0, 0, 0, -7.01121794871795E-05, -2.9918624465812, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(26) << 0, 0, 17.2008547008547, 1.12179487179487, 1.12179487179487, 0, 0, 0, 3.92641105769231, 0, 0, 16.8269230769231, -2.24358974358974, 2.99145299145299, 0, 0, 0, 2.61764756944444, 0, 0, 38.8888888888889, 8.97435897435897, 8.97435897435897, 0, 0, 0, 7.85285576923077, 0, 0, 16.8269230769231, 2.99145299145299, -2.24358974358974, 0, 0, 0, 2.61764756944444, 0, 0, -17.2008547008547, 0, 5.23504273504273, 0, 0, 0, -10.4702144764957, 0, 0, -27.6709401709402, -2.99145299145299, -7.47863247863248, 0, 0, 0, -7.85277163461538, 0, 0, -27.6709401709402, -7.47863247863248, -2.99145299145299, 0, 0, 0, -7.85277163461538, 0, 0, -17.2008547008547, 5.23504273504273, 0, 0, 0, 0, -10.4702144764957;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(27) << 7852564.1025641, -336538.461538461, 0, 0, 0, 201923.076923077, 32.7190170940171, -1.40224358974359, 0, 9288461.53846154, -3926282.05128205, 0, 0, 0, 201923.076923077, 38.7019230769231, -16.3595085470085, 0, 10320512.8205128, 336538.461538461, 0, 0, 0, 538461.538461538, 43.0021367521368, 1.40224358974359, 0, 21000000, -9535256.41025641, 0, 0, 0, -807692.307692308, 87.5, -39.730235042735, 0, -13102564.1025641, 2243589.74358974, 0, 0, 0, 0, -54.5940170940171, 9.3482905982906, 0, -5474358.97435897, 2243589.74358974, 0, 0, 0, 942307.692307692, -22.8098290598291, 9.3482905982906, 0, -24589743.5897436, 3141025.64102564, 0, 0, 0, -1346153.84615385, -102.457264957265, 13.0876068376068, 0, -5294871.7948718, 5833333.33333333, 0, 0, 0, -1750000, -22.0619658119658, 24.3055555555556, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(28) << 336538.461538461, 10320512.8205128, 0, 0, 0, -538461.538461538, 1.40224358974359, 43.0021367521368, 0, -3926282.05128205, 9288461.53846154, 0, 0, 0, -201923.076923077, -16.3595085470085, 38.7019230769231, 0, -336538.461538461, 7852564.1025641, 0, 0, 0, -201923.076923077, -1.40224358974359, 32.7190170940171, 0, -9535256.41025641, 21000000, 0, 0, 0, 807692.307692308, -39.730235042735, 87.5, 0, 2243589.74358974, -5474358.97435897, 0, 0, 0, -942307.692307692, 9.3482905982906, -22.8098290598291, 0, 2243589.74358974, -13102564.1025641, 0, 0, 0, 0, 9.3482905982906, -54.5940170940171, 0, 5833333.33333333, -5294871.7948718, 0, 0, 0, 1750000, 24.3055555555556, -22.0619658119658, 0, 3141025.64102564, -24589743.5897436, 0, 0, 0, 1346153.84615385, 13.0876068376068, -102.457264957265, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(29) << 0, 0, 4038461.53846154, 134615.384615385, -358974.358974359, 0, 0, 0, 16.8269230769231, 0, 0, 4128205.12820513, 134615.384615385, -134615.384615385, 0, 0, 0, 17.2008547008547, 0, 0, 4038461.53846154, 358974.358974359, -134615.384615385, 0, 0, 0, 16.8269230769231, 0, 0, 9333333.33333333, -538461.538461539, 538461.538461539, 0, 0, 0, 38.8888888888889, 0, 0, -4128205.12820513, 0, -628205.128205128, 0, 0, 0, -17.2008547008547, 0, 0, -4128205.12820513, 628205.128205128, 0, 0, 0, 0, -17.2008547008547, 0, 0, -6641025.64102564, -897435.897435898, 1166666.66666667, 0, 0, 0, -27.6709401709402, 0, 0, -6641025.64102564, -1166666.66666667, 897435.897435898, 0, 0, 0, -27.6709401709402;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(30) << 0, 0, 134615.384615385, 89809.0277777778, -2.80448717948718, 0, 0, 0, 2.24358974358974, 0, 0, -134615.384615385, 134692.788461538, -32.7190170940171, 0, 0, 0, 1.12179487179487, 0, 0, -358974.358974359, 89829.594017094, 2.80448717948718, 0, 0, 0, 2.99145299145299, 0, 0, -538461.538461539, 269405.769230769, -79.4604700854701, 0, 0, 0, -8.97435897435897, 0, 0, 0, -359083.547008547, 18.6965811965812, 0, 0, 0, 0, 0, 0, -628205.128205128, -359019.978632479, 18.6965811965812, 0, 0, 0, 5.23504273504273, 0, 0, 897435.897435898, -269435.683760684, 26.1752136752137, 0, 0, 0, -7.47863247863248, 0, 0, 628205.128205128, -269274.893162393, 48.6111111111111, 0, 0, 0, -11.965811965812;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(31) << 0, 0, 358974.358974359, 2.80448717948718, 89829.594017094, 0, 0, 0, -2.99145299145299, 0, 0, 134615.384615385, -32.7190170940171, 134692.788461538, 0, 0, 0, -1.12179487179487, 0, 0, -134615.384615385, -2.80448717948718, 89809.0277777778, 0, 0, 0, -2.24358974358974, 0, 0, 538461.538461539, -79.4604700854701, 269405.769230769, 0, 0, 0, 8.97435897435897, 0, 0, 628205.128205128, 18.6965811965812, -359019.978632479, 0, 0, 0, -5.23504273504273, 0, 0, 0, 18.6965811965812, -359083.547008547, 0, 0, 0, 0, 0, 0, -628205.128205128, 48.6111111111111, -269274.893162393, 0, 0, 0, 11.965811965812, 0, 0, -897435.897435898, 26.1752136752137, -269435.683760684, 0, 0, 0, 7.47863247863248;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(32) << 201923.076923077, 538461.538461538, 0, 0, 0, 314136.217948718, 1.96314102564103, -0.747863247863248, 0, -201923.076923077, 201923.076923077, 0, 0, 0, 471188.247863248, 0.280448717948718, -0.280448717948718, 0, -538461.538461538, -201923.076923077, 0, 0, 0, 314136.217948718, 0.747863247863248, -1.96314102564103, 0, -807692.307692308, 807692.307692308, 0, 0, 0, 942385.47008547, -7.8525641025641, 7.8525641025641, 0, 0, 942307.692307692, 0, 0, 0, -1256444.65811966, 0, -1.30876068376068, 0, -942307.692307692, 0, 0, 0, 0, -1256444.65811966, 1.30876068376068, 0, 0, 1346153.84615385, -942307.692307692, 0, 0, 0, -942363.034188034, -1.86965811965812, 5.79594017094017, 0, 942307.692307692, -1346153.84615385, 0, 0, 0, -942363.034188034, -5.79594017094017, 1.86965811965812, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(33) << 32.7190170940171, -1.40224358974359, 0, 0, 0, 1.96314102564103, 0.748108640491453, -1.05168269230769E-05, 0, 38.7019230769231, -16.3595085470085, 0, 0, 0, -0.280448717948718, 1.12208513621795, -0.000122696314102564, 0, 43.0021367521368, 1.40224358974359, 0, 0, 0, -0.747863247863248, 0.748185763888889, 1.05168269230769E-05, 0, 87.5, -39.730235042735, 0, 0, 0, -7.8525641025641, 2.24424599358974, -0.000297976762820513, 0, -54.5940170940171, 9.3482905982906, 0, 0, 0, 0, -2.9918624465812, 7.01121794871795E-05, 0, -22.8098290598291, 9.3482905982906, 0, 0, 0, -1.30876068376068, -2.99162406517094, 7.01121794871795E-05, 0, -102.457264957265, 13.0876068376068, 0, 0, 0, 1.86965811965812, -2.24435817307692, 9.81570512820513E-05, 0, -22.0619658119658, 24.3055555555556, 0, 0, 0, -2.05662393162393, -2.24375520833333, 0.000182291666666667, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(34) << 1.40224358974359, 43.0021367521368, 0, 0, 0, 0.747863247863248, 1.05168269230769E-05, 0.748185763888889, 0, -16.3595085470085, 38.7019230769231, 0, 0, 0, 0.280448717948718, -0.000122696314102564, 1.12208513621795, 0, -1.40224358974359, 32.7190170940171, 0, 0, 0, -1.96314102564103, -1.05168269230769E-05, 0.748108640491453, 0, -39.730235042735, 87.5, 0, 0, 0, 7.8525641025641, -0.000297976762820513, 2.24424599358974, 0, 9.3482905982906, -22.8098290598291, 0, 0, 0, 1.30876068376068, 7.01121794871795E-05, -2.99162406517094, 0, 9.3482905982906, -54.5940170940171, 0, 0, 0, 0, 7.01121794871795E-05, -2.9918624465812, 0, 24.3055555555556, -22.0619658119658, 0, 0, 0, 2.05662393162393, 0.000182291666666667, -2.24375520833333, 0, 13.0876068376068, -102.457264957265, 0, 0, 0, -1.86965811965812, 9.81570512820513E-05, -2.24435817307692, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(35) << 0, 0, 16.8269230769231, 2.24358974358974, 2.99145299145299, 0, 0, 0, 2.61764756944444, 0, 0, 17.2008547008547, -1.12179487179487, 1.12179487179487, 0, 0, 0, 3.92641105769231, 0, 0, 16.8269230769231, -2.99145299145299, -2.24358974358974, 0, 0, 0, 2.61764756944444, 0, 0, 38.8888888888889, -8.97435897435897, 8.97435897435897, 0, 0, 0, 7.85285576923077, 0, 0, -17.2008547008547, 0, 5.23504273504273, 0, 0, 0, -10.4702144764957, 0, 0, -17.2008547008547, -5.23504273504273, 0, 0, 0, 0, -10.4702144764957, 0, 0, -27.6709401709402, 7.47863247863248, -2.99145299145299, 0, 0, 0, -7.85277163461538, 0, 0, -27.6709401709402, 2.99145299145299, -7.47863247863248, 0, 0, 0, -7.85277163461538;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(36) << -24589743.5897436, -5833333.33333333, 0, 0, 0, 1346153.84615385, -102.457264957265, -24.3055555555556, 0, -24589743.5897436, 5833333.33333333, 0, 0, 0, -1346153.84615385, -102.457264957265, 24.3055555555556, 0, -13102564.1025641, -2243589.74358974, 0, 0, 0, 0, -54.5940170940171, -9.3482905982906, 0, -13102564.1025641, 2243589.74358974, 0, 0, 0, 0, -54.5940170940171, 9.3482905982906, 0, 54564102.5641026, 0, 0, 0, 0, 0, 227.350427350427, 0, 0, 0, -8974358.97435897, 0, 0, 0, -2692307.69230769, 0, -37.3931623931624, 0, 20820512.8205128, 0, 0, 0, 0, 0, 86.7521367521367, 0, 0, 0, 8974358.97435897, 0, 0, 0, 2692307.69230769, 0, 37.3931623931624, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(37) << -3141025.64102564, -5294871.7948718, 0, 0, 0, 942307.692307692, -13.0876068376068, -22.0619658119658, 0, 3141025.64102564, -5294871.7948718, 0, 0, 0, 942307.692307692, 13.0876068376068, -22.0619658119658, 0, -2243589.74358974, -5474358.97435897, 0, 0, 0, 942307.692307692, -9.3482905982906, -22.8098290598291, 0, 2243589.74358974, -5474358.97435897, 0, 0, 0, 942307.692307692, 9.3482905982906, -22.8098290598291, 0, 0, 29435897.4358974, 0, 0, 0, -3230769.23076923, 0, 122.649572649573, 0, -8974358.97435897, 0, 0, 0, 0, -2692307.69230769, -37.3931623931624, 0, 0, 0, -7897435.8974359, 0, 0, 0, -3230769.23076923, 0, -32.9059829059829, 0, 8974358.97435897, 0, 0, 0, 0, -2692307.69230769, 37.3931623931624, 0, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(38) << 0, 0, -6641025.64102564, 897435.897435898, 628205.128205128, 0, 0, 0, -27.6709401709402, 0, 0, -6641025.64102564, -897435.897435898, 628205.128205128, 0, 0, 0, -27.6709401709402, 0, 0, -4128205.12820513, 0, 628205.128205128, 0, 0, 0, -17.2008547008547, 0, 0, -4128205.12820513, 0, 628205.128205128, 0, 0, 0, -17.2008547008547, 0, 0, 18666666.6666667, 0, -2153846.15384615, 0, 0, 0, 77.7777777777778, 0, 0, 0, -1794871.7948718, -1794871.7948718, 0, 0, 0, 0, 0, 0, 2871794.87179487, 0, -2153846.15384615, 0, 0, 0, 11.965811965812, 0, 0, 0, 1794871.7948718, -1794871.7948718, 0, 0, 0, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(39) << 0, 0, -897435.897435898, -269435.683760684, -48.6111111111111, 0, 0, 0, 7.47863247863248, 0, 0, 897435.897435898, -269435.683760684, 48.6111111111111, 0, 0, 0, -7.47863247863248, 0, 0, 0, -359083.547008547, -18.6965811965812, 0, 0, 0, 0, 0, 0, 0, -359083.547008547, 18.6965811965812, 0, 0, 0, 0, 0, 0, 0, 1436352.13675214, 0, 0, 0, 0, 0, 0, 0, 1794871.7948718, 897435.897435898, -74.7863247863248, 0, 0, 0, -14.957264957265, 0, 0, 0, 718122.222222222, 0, 0, 0, 0, 0, 0, 0, -1794871.7948718, 897435.897435898, 74.7863247863248, 0, 0, 0, 14.957264957265;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(40) << 0, 0, -1166666.66666667, -26.1752136752137, -269274.893162393, 0, 0, 0, 2.99145299145299, 0, 0, -1166666.66666667, 26.1752136752137, -269274.893162393, 0, 0, 0, 2.99145299145299, 0, 0, -628205.128205128, -18.6965811965812, -359019.978632479, 0, 0, 0, 5.23504273504273, 0, 0, -628205.128205128, 18.6965811965812, -359019.978632479, 0, 0, 0, 5.23504273504273, 0, 0, -2153846.15384615, 0, 1436142.73504274, 0, 0, 0, -35.8974358974359, 0, 0, 1794871.7948718, -74.7863247863248, 897435.897435898, 0, 0, 0, -14.957264957265, 0, 0, 2153846.15384615, 0, 717882.905982906, 0, 0, 0, -17.9487179487179, 0, 0, 1794871.7948718, 74.7863247863248, 897435.897435898, 0, 0, 0, -14.957264957265;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(41) << -1346153.84615385, -1750000, 0, 0, 0, -942363.034188034, 1.86965811965812, -2.05662393162393, 0, 1346153.84615385, -1750000, 0, 0, 0, -942363.034188034, -1.86965811965812, -2.05662393162393, 0, 0, -942307.692307692, 0, 0, 0, -1256444.65811966, 0, 1.30876068376068, 0, 0, -942307.692307692, 0, 0, 0, -1256444.65811966, 0, 1.30876068376068, 0, 0, -3230769.23076923, 0, 0, 0, 5025796.58119658, 0, -31.4102564102564, 0, 2692307.69230769, 2692307.69230769, 0, 0, 0, 3141025.64102564, -3.73931623931624, -3.73931623931624, 0, 0, 3230769.23076923, 0, 0, 0, 2512844.44444444, 0, -4.48717948717949, 0, -2692307.69230769, 2692307.69230769, 0, 0, 0, 3141025.64102564, 3.73931623931624, -3.73931623931624, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(42) << -102.457264957265, -24.3055555555556, 0, 0, 0, -1.86965811965812, -2.24435817307692, -0.000182291666666667, 0, -102.457264957265, 24.3055555555556, 0, 0, 0, 1.86965811965812, -2.24435817307692, 0.000182291666666667, 0, -54.5940170940171, -9.3482905982906, 0, 0, 0, 0, -2.9918624465812, -7.01121794871795E-05, 0, -54.5940170940171, 9.3482905982906, 0, 0, 0, 0, -2.9918624465812, 7.01121794871795E-05, 0, 227.350427350427, 0, 0, 0, 0, 0, 11.9675170940171, 0, 0, 0, -37.3931623931624, 0, 0, 0, 3.73931623931624, 7.47863247863248, -0.000280448717948718, 0, 86.7521367521367, 0, 0, 0, 0, 0, 5.98355662393162, 0, 0, 0, 37.3931623931624, 0, 0, 0, -3.73931623931624, 7.47863247863248, 0.000280448717948718, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(43) << -13.0876068376068, -22.0619658119658, 0, 0, 0, -5.79594017094017, -9.81570512820513E-05, -2.24375520833333, 0, 13.0876068376068, -22.0619658119658, 0, 0, 0, -5.79594017094017, 9.81570512820513E-05, -2.24375520833333, 0, -9.3482905982906, -22.8098290598291, 0, 0, 0, -1.30876068376068, -7.01121794871795E-05, -2.99162406517094, 0, 9.3482905982906, -22.8098290598291, 0, 0, 0, -1.30876068376068, 7.01121794871795E-05, -2.99162406517094, 0, 0, 122.649572649573, 0, 0, 0, -31.4102564102564, 0, 11.9667318376068, 0, -37.3931623931624, 0, 0, 0, 0, 3.73931623931624, -0.000280448717948718, 7.47863247863248, 0, 0, -32.9059829059829, 0, 0, 0, 4.48717948717949, 0, 5.98265918803419, 0, 37.3931623931624, 0, 0, 0, 0, 3.73931623931624, 0.000280448717948718, 7.47863247863248, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(44) << 0, 0, -27.6709401709402, -7.47863247863248, -11.965811965812, 0, 0, 0, -7.85277163461538, 0, 0, -27.6709401709402, 7.47863247863248, -11.965811965812, 0, 0, 0, -7.85277163461538, 0, 0, -17.2008547008547, 0, -5.23504273504273, 0, 0, 0, -10.4702144764957, 0, 0, -17.2008547008547, 0, -5.23504273504273, 0, 0, 0, -10.4702144764957, 0, 0, 77.7777777777778, 0, -35.8974358974359, 0, 0, 0, 41.8809252136752, 0, 0, 0, 14.957264957265, 14.957264957265, 0, 0, 0, 26.1752136752137, 0, 0, 11.965811965812, 0, 17.9487179487179, 0, 0, 0, 20.9402606837607, 0, 0, 0, -14.957264957265, 14.957264957265, 0, 0, 0, 26.1752136752137;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(45) << -5474358.97435897, -2243589.74358974, 0, 0, 0, -942307.692307692, -22.8098290598291, -9.3482905982906, 0, -5294871.7948718, 3141025.64102564, 0, 0, 0, -942307.692307692, -22.0619658119658, 13.0876068376068, 0, -5294871.7948718, -3141025.64102564, 0, 0, 0, -942307.692307692, -22.0619658119658, -13.0876068376068, 0, -5474358.97435897, 2243589.74358974, 0, 0, 0, -942307.692307692, -22.8098290598291, 9.3482905982906, 0, 0, -8974358.97435897, 0, 0, 0, 2692307.69230769, 0, -37.3931623931624, 0, 29435897.4358974, 0, 0, 0, 0, 3230769.23076923, 122.649572649573, 0, 0, 0, 8974358.97435897, 0, 0, 0, 2692307.69230769, 0, 37.3931623931624, 0, -7897435.8974359, 0, 0, 0, 0, 3230769.23076923, -32.9059829059829, 0, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(46) << -2243589.74358974, -13102564.1025641, 0, 0, 0, 0, -9.3482905982906, -54.5940170940171, 0, 5833333.33333333, -24589743.5897436, 0, 0, 0, 1346153.84615385, 24.3055555555556, -102.457264957265, 0, -5833333.33333333, -24589743.5897436, 0, 0, 0, -1346153.84615385, -24.3055555555556, -102.457264957265, 0, 2243589.74358974, -13102564.1025641, 0, 0, 0, 0, 9.3482905982906, -54.5940170940171, 0, -8974358.97435897, 0, 0, 0, 0, 2692307.69230769, -37.3931623931624, 0, 0, 0, 54564102.5641026, 0, 0, 0, 0, 0, 227.350427350427, 0, 8974358.97435897, 0, 0, 0, 0, -2692307.69230769, 37.3931623931624, 0, 0, 0, 20820512.8205128, 0, 0, 0, 0, 0, 86.7521367521367, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(47) << 0, 0, -4128205.12820513, -628205.128205128, 0, 0, 0, 0, -17.2008547008547, 0, 0, -6641025.64102564, -628205.128205128, 897435.897435898, 0, 0, 0, -27.6709401709402, 0, 0, -6641025.64102564, -628205.128205128, -897435.897435898, 0, 0, 0, -27.6709401709402, 0, 0, -4128205.12820513, -628205.128205128, 0, 0, 0, 0, -17.2008547008547, 0, 0, 0, 1794871.7948718, 1794871.7948718, 0, 0, 0, 0, 0, 0, 18666666.6666667, 2153846.15384615, 0, 0, 0, 0, 77.7777777777778, 0, 0, 0, 1794871.7948718, -1794871.7948718, 0, 0, 0, 0, 0, 0, 2871794.87179487, 2153846.15384615, 0, 0, 0, 0, 11.965811965812;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(48) << 0, 0, 628205.128205128, -359019.978632479, -18.6965811965812, 0, 0, 0, -5.23504273504273, 0, 0, 1166666.66666667, -269274.893162393, 26.1752136752137, 0, 0, 0, -2.99145299145299, 0, 0, 1166666.66666667, -269274.893162393, -26.1752136752137, 0, 0, 0, -2.99145299145299, 0, 0, 628205.128205128, -359019.978632479, 18.6965811965812, 0, 0, 0, -5.23504273504273, 0, 0, -1794871.7948718, 897435.897435898, -74.7863247863248, 0, 0, 0, 14.957264957265, 0, 0, 2153846.15384615, 1436142.73504274, 0, 0, 0, 0, 35.8974358974359, 0, 0, -1794871.7948718, 897435.897435898, 74.7863247863248, 0, 0, 0, 14.957264957265, 0, 0, -2153846.15384615, 717882.905982906, 0, 0, 0, 0, 17.9487179487179;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(49) << 0, 0, 0, -18.6965811965812, -359083.547008547, 0, 0, 0, 0, 0, 0, -897435.897435898, 48.6111111111111, -269435.683760684, 0, 0, 0, 7.47863247863248, 0, 0, 897435.897435898, -48.6111111111111, -269435.683760684, 0, 0, 0, -7.47863247863248, 0, 0, 0, 18.6965811965812, -359083.547008547, 0, 0, 0, 0, 0, 0, -1794871.7948718, -74.7863247863248, 897435.897435898, 0, 0, 0, 14.957264957265, 0, 0, 0, 0, 1436352.13675214, 0, 0, 0, 0, 0, 0, 1794871.7948718, 74.7863247863248, 897435.897435898, 0, 0, 0, -14.957264957265, 0, 0, 0, 0, 718122.222222222, 0, 0, 0, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(50) << 942307.692307692, 0, 0, 0, 0, -1256444.65811966, -1.30876068376068, 0, 0, 1750000, -1346153.84615385, 0, 0, 0, -942363.034188034, 2.05662393162393, 1.86965811965812, 0, 1750000, 1346153.84615385, 0, 0, 0, -942363.034188034, 2.05662393162393, -1.86965811965812, 0, 942307.692307692, 0, 0, 0, 0, -1256444.65811966, -1.30876068376068, 0, 0, -2692307.69230769, -2692307.69230769, 0, 0, 0, 3141025.64102564, 3.73931623931624, 3.73931623931624, 0, 3230769.23076923, 0, 0, 0, 0, 5025796.58119658, 31.4102564102564, 0, 0, -2692307.69230769, 2692307.69230769, 0, 0, 0, 3141025.64102564, 3.73931623931624, -3.73931623931624, 0, -3230769.23076923, 0, 0, 0, 0, 2512844.44444444, 4.48717948717949, 0, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(51) << -22.8098290598291, -9.3482905982906, 0, 0, 0, 1.30876068376068, -2.99162406517094, -7.01121794871795E-05, 0, -22.0619658119658, 13.0876068376068, 0, 0, 0, 5.79594017094017, -2.24375520833333, 9.81570512820513E-05, 0, -22.0619658119658, -13.0876068376068, 0, 0, 0, 5.79594017094017, -2.24375520833333, -9.81570512820513E-05, 0, -22.8098290598291, 9.3482905982906, 0, 0, 0, 1.30876068376068, -2.99162406517094, 7.01121794871795E-05, 0, 0, -37.3931623931624, 0, 0, 0, -3.73931623931624, 7.47863247863248, -0.000280448717948718, 0, 122.649572649573, 0, 0, 0, 0, 31.4102564102564, 11.9667318376068, 0, 0, 0, 37.3931623931624, 0, 0, 0, -3.73931623931624, 7.47863247863248, 0.000280448717948718, 0, -32.9059829059829, 0, 0, 0, 0, -4.48717948717949, 5.98265918803419, 0, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(52) << -9.3482905982906, -54.5940170940171, 0, 0, 0, 0, -7.01121794871795E-05, -2.9918624465812, 0, 24.3055555555556, -102.457264957265, 0, 0, 0, -1.86965811965812, 0.000182291666666667, -2.24435817307692, 0, -24.3055555555556, -102.457264957265, 0, 0, 0, 1.86965811965812, -0.000182291666666667, -2.24435817307692, 0, 9.3482905982906, -54.5940170940171, 0, 0, 0, 0, 7.01121794871795E-05, -2.9918624465812, 0, -37.3931623931624, 0, 0, 0, 0, -3.73931623931624, -0.000280448717948718, 7.47863247863248, 0, 0, 227.350427350427, 0, 0, 0, 0, 0, 11.9675170940171, 0, 37.3931623931624, 0, 0, 0, 0, 3.73931623931624, 0.000280448717948718, 7.47863247863248, 0, 0, 86.7521367521367, 0, 0, 0, 0, 0, 5.98355662393162, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(53) << 0, 0, -17.2008547008547, 5.23504273504273, 0, 0, 0, 0, -10.4702144764957, 0, 0, -27.6709401709402, 11.965811965812, -7.47863247863248, 0, 0, 0, -7.85277163461538, 0, 0, -27.6709401709402, 11.965811965812, 7.47863247863248, 0, 0, 0, -7.85277163461538, 0, 0, -17.2008547008547, 5.23504273504273, 0, 0, 0, 0, -10.4702144764957, 0, 0, 0, -14.957264957265, -14.957264957265, 0, 0, 0, 26.1752136752137, 0, 0, 77.7777777777778, 35.8974358974359, 0, 0, 0, 0, 41.8809252136752, 0, 0, 0, -14.957264957265, 14.957264957265, 0, 0, 0, 26.1752136752137, 0, 0, 11.965811965812, -17.9487179487179, 0, 0, 0, 0, 20.9402606837607;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(54) << -13102564.1025641, -2243589.74358974, 0, 0, 0, 0, -54.5940170940171, -9.3482905982906, 0, -13102564.1025641, 2243589.74358974, 0, 0, 0, 0, -54.5940170940171, 9.3482905982906, 0, -24589743.5897436, -5833333.33333333, 0, 0, 0, -1346153.84615385, -102.457264957265, -24.3055555555556, 0, -24589743.5897436, 5833333.33333333, 0, 0, 0, 1346153.84615385, -102.457264957265, 24.3055555555556, 0, 20820512.8205128, 0, 0, 0, 0, 0, 86.7521367521367, 0, 0, 0, 8974358.97435897, 0, 0, 0, -2692307.69230769, 0, 37.3931623931624, 0, 54564102.5641026, 0, 0, 0, 0, 0, 227.350427350427, 0, 0, 0, -8974358.97435897, 0, 0, 0, 2692307.69230769, 0, -37.3931623931624, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(55) << -2243589.74358974, -5474358.97435897, 0, 0, 0, -942307.692307692, -9.3482905982906, -22.8098290598291, 0, 2243589.74358974, -5474358.97435897, 0, 0, 0, -942307.692307692, 9.3482905982906, -22.8098290598291, 0, -3141025.64102564, -5294871.7948718, 0, 0, 0, -942307.692307692, -13.0876068376068, -22.0619658119658, 0, 3141025.64102564, -5294871.7948718, 0, 0, 0, -942307.692307692, 13.0876068376068, -22.0619658119658, 0, 0, -7897435.8974359, 0, 0, 0, 3230769.23076923, 0, -32.9059829059829, 0, 8974358.97435897, 0, 0, 0, 0, 2692307.69230769, 37.3931623931624, 0, 0, 0, 29435897.4358974, 0, 0, 0, 3230769.23076923, 0, 122.649572649573, 0, -8974358.97435897, 0, 0, 0, 0, 2692307.69230769, -37.3931623931624, 0, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(56) << 0, 0, -4128205.12820513, 0, -628205.128205128, 0, 0, 0, -17.2008547008547, 0, 0, -4128205.12820513, 0, -628205.128205128, 0, 0, 0, -17.2008547008547, 0, 0, -6641025.64102564, -897435.897435898, -628205.128205128, 0, 0, 0, -27.6709401709402, 0, 0, -6641025.64102564, 897435.897435898, -628205.128205128, 0, 0, 0, -27.6709401709402, 0, 0, 2871794.87179487, 0, 2153846.15384615, 0, 0, 0, 11.965811965812, 0, 0, 0, -1794871.7948718, 1794871.7948718, 0, 0, 0, 0, 0, 0, 18666666.6666667, 0, 2153846.15384615, 0, 0, 0, 77.7777777777778, 0, 0, 0, 1794871.7948718, 1794871.7948718, 0, 0, 0, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(57) << 0, 0, 0, -359083.547008547, -18.6965811965812, 0, 0, 0, 0, 0, 0, 0, -359083.547008547, 18.6965811965812, 0, 0, 0, 0, 0, 0, 897435.897435898, -269435.683760684, -48.6111111111111, 0, 0, 0, -7.47863247863248, 0, 0, -897435.897435898, -269435.683760684, 48.6111111111111, 0, 0, 0, 7.47863247863248, 0, 0, 0, 718122.222222222, 0, 0, 0, 0, 0, 0, 0, 1794871.7948718, 897435.897435898, 74.7863247863248, 0, 0, 0, -14.957264957265, 0, 0, 0, 1436352.13675214, 0, 0, 0, 0, 0, 0, 0, -1794871.7948718, 897435.897435898, -74.7863247863248, 0, 0, 0, 14.957264957265;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(58) << 0, 0, 628205.128205128, -18.6965811965812, -359019.978632479, 0, 0, 0, -5.23504273504273, 0, 0, 628205.128205128, 18.6965811965812, -359019.978632479, 0, 0, 0, -5.23504273504273, 0, 0, 1166666.66666667, -26.1752136752137, -269274.893162393, 0, 0, 0, -2.99145299145299, 0, 0, 1166666.66666667, 26.1752136752137, -269274.893162393, 0, 0, 0, -2.99145299145299, 0, 0, -2153846.15384615, 0, 717882.905982906, 0, 0, 0, 17.9487179487179, 0, 0, -1794871.7948718, 74.7863247863248, 897435.897435898, 0, 0, 0, 14.957264957265, 0, 0, 2153846.15384615, 0, 1436142.73504274, 0, 0, 0, 35.8974358974359, 0, 0, -1794871.7948718, -74.7863247863248, 897435.897435898, 0, 0, 0, 14.957264957265;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(59) << 0, 942307.692307692, 0, 0, 0, -1256444.65811966, 0, -1.30876068376068, 0, 0, 942307.692307692, 0, 0, 0, -1256444.65811966, 0, -1.30876068376068, 0, 1346153.84615385, 1750000, 0, 0, 0, -942363.034188034, -1.86965811965812, 2.05662393162393, 0, -1346153.84615385, 1750000, 0, 0, 0, -942363.034188034, 1.86965811965812, 2.05662393162393, 0, 0, -3230769.23076923, 0, 0, 0, 2512844.44444444, 0, 4.48717948717949, 0, 2692307.69230769, -2692307.69230769, 0, 0, 0, 3141025.64102564, -3.73931623931624, 3.73931623931624, 0, 0, 3230769.23076923, 0, 0, 0, 5025796.58119658, 0, 31.4102564102564, 0, -2692307.69230769, -2692307.69230769, 0, 0, 0, 3141025.64102564, 3.73931623931624, 3.73931623931624, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(60) << -54.5940170940171, -9.3482905982906, 0, 0, 0, 0, -2.9918624465812, -7.01121794871795E-05, 0, -54.5940170940171, 9.3482905982906, 0, 0, 0, 0, -2.9918624465812, 7.01121794871795E-05, 0, -102.457264957265, -24.3055555555556, 0, 0, 0, 1.86965811965812, -2.24435817307692, -0.000182291666666667, 0, -102.457264957265, 24.3055555555556, 0, 0, 0, -1.86965811965812, -2.24435817307692, 0.000182291666666667, 0, 86.7521367521367, 0, 0, 0, 0, 0, 5.98355662393162, 0, 0, 0, 37.3931623931624, 0, 0, 0, 3.73931623931624, 7.47863247863248, 0.000280448717948718, 0, 227.350427350427, 0, 0, 0, 0, 0, 11.9675170940171, 0, 0, 0, -37.3931623931624, 0, 0, 0, -3.73931623931624, 7.47863247863248, -0.000280448717948718, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(61) << -9.3482905982906, -22.8098290598291, 0, 0, 0, 1.30876068376068, -7.01121794871795E-05, -2.99162406517094, 0, 9.3482905982906, -22.8098290598291, 0, 0, 0, 1.30876068376068, 7.01121794871795E-05, -2.99162406517094, 0, -13.0876068376068, -22.0619658119658, 0, 0, 0, 5.79594017094017, -9.81570512820513E-05, -2.24375520833333, 0, 13.0876068376068, -22.0619658119658, 0, 0, 0, 5.79594017094017, 9.81570512820513E-05, -2.24375520833333, 0, 0, -32.9059829059829, 0, 0, 0, -4.48717948717949, 0, 5.98265918803419, 0, 37.3931623931624, 0, 0, 0, 0, -3.73931623931624, 0.000280448717948718, 7.47863247863248, 0, 0, 122.649572649573, 0, 0, 0, 31.4102564102564, 0, 11.9667318376068, 0, -37.3931623931624, 0, 0, 0, 0, -3.73931623931624, -0.000280448717948718, 7.47863247863248, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(62) << 0, 0, -17.2008547008547, 0, 5.23504273504273, 0, 0, 0, -10.4702144764957, 0, 0, -17.2008547008547, 0, 5.23504273504273, 0, 0, 0, -10.4702144764957, 0, 0, -27.6709401709402, 7.47863247863248, 11.965811965812, 0, 0, 0, -7.85277163461538, 0, 0, -27.6709401709402, -7.47863247863248, 11.965811965812, 0, 0, 0, -7.85277163461538, 0, 0, 11.965811965812, 0, -17.9487179487179, 0, 0, 0, 20.9402606837607, 0, 0, 0, 14.957264957265, -14.957264957265, 0, 0, 0, 26.1752136752137, 0, 0, 77.7777777777778, 0, 35.8974358974359, 0, 0, 0, 41.8809252136752, 0, 0, 0, -14.957264957265, -14.957264957265, 0, 0, 0, 26.1752136752137;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(63) << -5294871.7948718, -3141025.64102564, 0, 0, 0, 942307.692307692, -22.0619658119658, -13.0876068376068, 0, -5474358.97435897, 2243589.74358974, 0, 0, 0, 942307.692307692, -22.8098290598291, 9.3482905982906, 0, -5474358.97435897, -2243589.74358974, 0, 0, 0, 942307.692307692, -22.8098290598291, -9.3482905982906, 0, -5294871.7948718, 3141025.64102564, 0, 0, 0, 942307.692307692, -22.0619658119658, 13.0876068376068, 0, 0, 8974358.97435897, 0, 0, 0, -2692307.69230769, 0, 37.3931623931624, 0, -7897435.8974359, 0, 0, 0, 0, -3230769.23076923, -32.9059829059829, 0, 0, 0, -8974358.97435897, 0, 0, 0, -2692307.69230769, 0, -37.3931623931624, 0, 29435897.4358974, 0, 0, 0, 0, -3230769.23076923, 122.649572649573, 0, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(64) << -5833333.33333333, -24589743.5897436, 0, 0, 0, 1346153.84615385, -24.3055555555556, -102.457264957265, 0, 2243589.74358974, -13102564.1025641, 0, 0, 0, 0, 9.3482905982906, -54.5940170940171, 0, -2243589.74358974, -13102564.1025641, 0, 0, 0, 0, -9.3482905982906, -54.5940170940171, 0, 5833333.33333333, -24589743.5897436, 0, 0, 0, -1346153.84615385, 24.3055555555556, -102.457264957265, 0, 8974358.97435897, 0, 0, 0, 0, 2692307.69230769, 37.3931623931624, 0, 0, 0, 20820512.8205128, 0, 0, 0, 0, 0, 86.7521367521367, 0, -8974358.97435897, 0, 0, 0, 0, -2692307.69230769, -37.3931623931624, 0, 0, 0, 54564102.5641026, 0, 0, 0, 0, 0, 227.350427350427, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(65) << 0, 0, -6641025.64102564, 628205.128205128, 897435.897435898, 0, 0, 0, -27.6709401709402, 0, 0, -4128205.12820513, 628205.128205128, 0, 0, 0, 0, -17.2008547008547, 0, 0, -4128205.12820513, 628205.128205128, 0, 0, 0, 0, -17.2008547008547, 0, 0, -6641025.64102564, 628205.128205128, -897435.897435898, 0, 0, 0, -27.6709401709402, 0, 0, 0, -1794871.7948718, 1794871.7948718, 0, 0, 0, 0, 0, 0, 2871794.87179487, -2153846.15384615, 0, 0, 0, 0, 11.965811965812, 0, 0, 0, -1794871.7948718, -1794871.7948718, 0, 0, 0, 0, 0, 0, 18666666.6666667, -2153846.15384615, 0, 0, 0, 0, 77.7777777777778;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(66) << 0, 0, -1166666.66666667, -269274.893162393, -26.1752136752137, 0, 0, 0, 2.99145299145299, 0, 0, -628205.128205128, -359019.978632479, 18.6965811965812, 0, 0, 0, 5.23504273504273, 0, 0, -628205.128205128, -359019.978632479, -18.6965811965812, 0, 0, 0, 5.23504273504273, 0, 0, -1166666.66666667, -269274.893162393, 26.1752136752137, 0, 0, 0, 2.99145299145299, 0, 0, 1794871.7948718, 897435.897435898, 74.7863247863248, 0, 0, 0, -14.957264957265, 0, 0, 2153846.15384615, 717882.905982906, 0, 0, 0, 0, -17.9487179487179, 0, 0, 1794871.7948718, 897435.897435898, -74.7863247863248, 0, 0, 0, -14.957264957265, 0, 0, -2153846.15384615, 1436142.73504274, 0, 0, 0, 0, -35.8974358974359;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(67) << 0, 0, -897435.897435898, -48.6111111111111, -269435.683760684, 0, 0, 0, 7.47863247863248, 0, 0, 0, 18.6965811965812, -359083.547008547, 0, 0, 0, 0, 0, 0, 0, -18.6965811965812, -359083.547008547, 0, 0, 0, 0, 0, 0, 897435.897435898, 48.6111111111111, -269435.683760684, 0, 0, 0, -7.47863247863248, 0, 0, -1794871.7948718, 74.7863247863248, 897435.897435898, 0, 0, 0, 14.957264957265, 0, 0, 0, 0, 718122.222222222, 0, 0, 0, 0, 0, 0, 1794871.7948718, -74.7863247863248, 897435.897435898, 0, 0, 0, -14.957264957265, 0, 0, 0, 0, 1436352.13675214, 0, 0, 0, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(68) << -1750000, -1346153.84615385, 0, 0, 0, -942363.034188034, -2.05662393162393, 1.86965811965812, 0, -942307.692307692, 0, 0, 0, 0, -1256444.65811966, 1.30876068376068, 0, 0, -942307.692307692, 0, 0, 0, 0, -1256444.65811966, 1.30876068376068, 0, 0, -1750000, 1346153.84615385, 0, 0, 0, -942363.034188034, -2.05662393162393, -1.86965811965812, 0, 2692307.69230769, -2692307.69230769, 0, 0, 0, 3141025.64102564, -3.73931623931624, 3.73931623931624, 0, 3230769.23076923, 0, 0, 0, 0, 2512844.44444444, -4.48717948717949, 0, 0, 2692307.69230769, 2692307.69230769, 0, 0, 0, 3141025.64102564, -3.73931623931624, -3.73931623931624, 0, -3230769.23076923, 0, 0, 0, 0, 5025796.58119658, -31.4102564102564, 0, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(69) << -22.0619658119658, -13.0876068376068, 0, 0, 0, -5.79594017094017, -2.24375520833333, -9.81570512820513E-05, 0, -22.8098290598291, 9.3482905982906, 0, 0, 0, -1.30876068376068, -2.99162406517094, 7.01121794871795E-05, 0, -22.8098290598291, -9.3482905982906, 0, 0, 0, -1.30876068376068, -2.99162406517094, -7.01121794871795E-05, 0, -22.0619658119658, 13.0876068376068, 0, 0, 0, -5.79594017094017, -2.24375520833333, 9.81570512820513E-05, 0, 0, 37.3931623931624, 0, 0, 0, 3.73931623931624, 7.47863247863248, 0.000280448717948718, 0, -32.9059829059829, 0, 0, 0, 0, 4.48717948717949, 5.98265918803419, 0, 0, 0, -37.3931623931624, 0, 0, 0, 3.73931623931624, 7.47863247863248, -0.000280448717948718, 0, 122.649572649573, 0, 0, 0, 0, -31.4102564102564, 11.9667318376068, 0, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(70) << -24.3055555555556, -102.457264957265, 0, 0, 0, -1.86965811965812, -0.000182291666666667, -2.24435817307692, 0, 9.3482905982906, -54.5940170940171, 0, 0, 0, 0, 7.01121794871795E-05, -2.9918624465812, 0, -9.3482905982906, -54.5940170940171, 0, 0, 0, 0, -7.01121794871795E-05, -2.9918624465812, 0, 24.3055555555556, -102.457264957265, 0, 0, 0, 1.86965811965812, 0.000182291666666667, -2.24435817307692, 0, 37.3931623931624, 0, 0, 0, 0, -3.73931623931624, 0.000280448717948718, 7.47863247863248, 0, 0, 86.7521367521367, 0, 0, 0, 0, 0, 5.98355662393162, 0, -37.3931623931624, 0, 0, 0, 0, 3.73931623931624, -0.000280448717948718, 7.47863247863248, 0, 0, 227.350427350427, 0, 0, 0, 0, 0, 11.9675170940171, 0;
    //Expected_JacobianR_NoDispSmallVelWithDamping.row(71) << 0, 0, -27.6709401709402, -11.965811965812, -7.47863247863248, 0, 0, 0, -7.85277163461538, 0, 0, -17.2008547008547, -5.23504273504273, 0, 0, 0, 0, -10.4702144764957, 0, 0, -17.2008547008547, -5.23504273504273, 0, 0, 0, 0, -10.4702144764957, 0, 0, -27.6709401709402, -11.965811965812, 7.47863247863248, 0, 0, 0, -7.85277163461538, 0, 0, 0, 14.957264957265, -14.957264957265, 0, 0, 0, 26.1752136752137, 0, 0, 11.965811965812, 17.9487179487179, 0, 0, 0, 0, 20.9402606837607, 0, 0, 0, 14.957264957265, 14.957264957265, 0, 0, 0, 26.1752136752137, 0, 0, 77.7777777777778, -35.8974358974359, 0, 0, 0, 0, 41.8809252136752;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_Jacobians;
    Expected_Jacobians.resize(2 * 72, 72);
    if (!load_validation_data("UT_ANCFShell_3833_JacNoDispSmallVelWithDamping.txt", Expected_Jacobians))
        return false;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianK_NoDispSmallVelWithDamping;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianR_NoDispSmallVelWithDamping;
    Expected_JacobianK_NoDispSmallVelWithDamping = Expected_Jacobians.block(0, 0, Expected_Jacobians.cols(), Expected_Jacobians.cols());
    Expected_JacobianR_NoDispSmallVelWithDamping = Expected_Jacobians.block(Expected_Jacobians.cols(), 0, Expected_Jacobians.cols(), Expected_Jacobians.cols());


    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation

    //Setup the test conditions
    ChVector<double> OriginalVel = m_nodeB->GetPos_dt();
    m_nodeB->SetPos_dt(ChVector<>(0.0, 0.0, 0.001));
    m_element->SetAlphaDamp(0.01);

    ChVectorDynamic<double> InternalForceNoDispSmallVelNoGravity;
    InternalForceNoDispSmallVelNoGravity.resize(72);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceNoDispSmallVelNoGravity);

    ChMatrixDynamic<double> JacobianK_NoDispSmallVelWithDamping;
    JacobianK_NoDispSmallVelWithDamping.resize(72, 72);
    m_element->ComputeKRMmatricesGlobal(JacobianK_NoDispSmallVelWithDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_NoDispSmallVelWithDamping;
    JacobianR_NoDispSmallVelWithDamping.resize(72, 72);
    m_element->ComputeKRMmatricesGlobal(JacobianR_NoDispSmallVelWithDamping, 0, 1, 0);

    //Reset the element conditions back to its original values
    m_nodeB->SetPos_dt(OriginalVel);
    m_element->SetAlphaDamp(0.0);


    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_NoDispSmallVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(72, 72);

    double small_terms_JacR = 1e-4*Expected_JacobianR_NoDispSmallVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacR = 0;
    double max_percent_error_JacR = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacR;
    percent_error_matrix_JacR.resize(72, 72);


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
    double dx = length / (num_elements);

    // Setup shell normals to initially align with the global z direction with no curvature
    ChVector<> dir1(0, 0, 1);
    ChVector<> Curv1(0, 0, 0);

    // Create the first nodes and fix them completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0, 0.0), dir1, Curv1);
    mesh->AddNode(nodeA);
    nodeA->SetFixed(true);
    auto nodeD = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, width, 0), dir1, Curv1);
    mesh->AddNode(nodeD);
    nodeD->SetFixed(true);
    auto nodeH = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0.5*width, 0), dir1, Curv1);
    mesh->AddNode(nodeH);
    nodeH->SetFixed(true);



    auto elementlast = chrono_types::make_shared<ElementVersion>();
    std::shared_ptr<ChNodeFEAxyzDD> nodeEndPoint;

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx, 0, 0), dir1, Curv1);
        mesh->AddNode(nodeB);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx, width, 0), dir1, Curv1);
        mesh->AddNode(nodeC);
        auto nodeE = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx - 0.5*dx, 0, 0.0), dir1, Curv1);
        mesh->AddNode(nodeE);
        auto nodeF = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx, 0.5*width, 0), dir1, Curv1);
        mesh->AddNode(nodeF);
        auto nodeG = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx - 0.5*dx, width, 0), dir1, Curv1);
        mesh->AddNode(nodeG);



        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC, nodeD, nodeE, nodeF, nodeG, nodeH);
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
        nodeH = nodeF;
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
        std::cout << "Axial Pull - Tip Displacement Check (Percent Error less than 5%) - Percent Error: " << Percent_Error << "% ";
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
    double dx = length / (num_elements);

    // Setup shell normals to initially align with the global z direction with no curvature
    ChVector<> dir1(0, 0, 1);
    ChVector<> Curv1(0, 0, 0);

    // Create the first nodes and fix them completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0, 0.0), dir1, Curv1);
    mesh->AddNode(nodeA);
    nodeA->SetFixed(true);
    auto nodeD = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, width, 0), dir1, Curv1);
    mesh->AddNode(nodeD);
    nodeD->SetFixed(true);
    auto nodeH = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0.5*width, 0), dir1, Curv1);
    mesh->AddNode(nodeH);
    nodeH->SetFixed(true);



    auto elementlast = chrono_types::make_shared<ElementVersion>();
    std::shared_ptr<ChNodeFEAxyzDD> nodeEndPoint;

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx, 0, 0), dir1, Curv1);
        mesh->AddNode(nodeB);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx, width, 0), dir1, Curv1);
        mesh->AddNode(nodeC);
        auto nodeE = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx - 0.5*dx, 0, 0.0), dir1, Curv1);
        mesh->AddNode(nodeE);
        auto nodeF = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx, 0.5*width, 0), dir1, Curv1);
        mesh->AddNode(nodeF);
        auto nodeG = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx - 0.5*dx, width, 0), dir1, Curv1);
        mesh->AddNode(nodeG);



        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC, nodeD, nodeE, nodeF, nodeG, nodeH);
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
        nodeH = nodeF;
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
        std::cout << "Cantilever Tip Displacement Check (Percent Error less than 5%) - Percent Error: " << Percent_Error << "% ";
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
    double dx = length / (num_elements);

    // Setup shell normals to initially align with the global z direction with no curvature
    ChVector<> dir1(0, 0, 1);
    ChVector<> Curv1(0, 0, 0);

    // Create the first nodes and fix them completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0, 0.0), dir1, Curv1);
    mesh->AddNode(nodeA);
    nodeA->SetFixed(true);
    auto nodeD = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, width, 0), dir1, Curv1);
    mesh->AddNode(nodeD);
    nodeD->SetFixed(true);
    auto nodeH = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0.5*width, 0), dir1, Curv1);
    mesh->AddNode(nodeH);
    nodeH->SetFixed(true);



    auto elementlast = chrono_types::make_shared<ElementVersion>();
    std::shared_ptr<ChNodeFEAxyzDD> nodeEndPoint;

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx, 0, 0), dir1, Curv1);
        mesh->AddNode(nodeB);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx, width, 0), dir1, Curv1);
        mesh->AddNode(nodeC);
        auto nodeE = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx - 0.5*dx, 0, 0.0), dir1, Curv1);
        mesh->AddNode(nodeE);
        auto nodeF = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx, 0.5*width, 0), dir1, Curv1);
        mesh->AddNode(nodeF);
        auto nodeG = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(i*dx - 0.5*dx, width, 0), dir1, Curv1);
        mesh->AddNode(nodeG);



        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC, nodeD, nodeE, nodeF, nodeG, nodeH);
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
        nodeH = nodeF;
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
        std::cout << "Axial Twist Angle Check (Percent Error less than 20%) - Percent Error: " << Percent_Error << "% ";
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
    ANCFOrgShellTest<ChElementShellANCF_8, ChMaterialShellANCF> ChElementShellANCF_3833_Org_test;
    if (ChElementShellANCF_3833_Org_test.RunElementChecks(1))
        print_green("ChElementShellANCF_8 Element Checks = PASSED\n");
    else
        print_red("ChElementShellANCF_8 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFOrgShellTest<ChElementShellANCF_3833_TR00, ChMaterialShellANCF> ChElementShellANCF_3833_TR00_test;
    if (ChElementShellANCF_3833_TR00_test.RunElementChecks(1))
        print_green("ChElementShellANCF_3833_TR00 Element Checks = PASSED\n");
    else
        print_red("ChElementShellANCF_3833_TR00 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFShellTest<ChElementShellANCF_3833_TR01, ChMaterialShellANCF_3833_TR01> ChElementShellANCF_3833_TR01_test;
    if (ChElementShellANCF_3833_TR01_test.RunElementChecks(0))
        print_green("ChElementShellANCF_3833_TR01 Element Checks = PASSED\n");
    else
        print_red("ChElementShellANCF_3833_TR01 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFShellTest<ChElementShellANCF_3833_TR02, ChMaterialShellANCF_3833_TR02> ChElementShellANCF_3833_TR02_test;
    if (ChElementShellANCF_3833_TR02_test.RunElementChecks(0))
        print_green("ChElementShellANCF_3833_TR02 Element Checks = PASSED\n");
    else
        print_red("ChElementShellANCF_3833_TR02 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFShellTest<ChElementShellANCF_3833_TR03, ChMaterialShellANCF_3833_TR03> ChElementShellANCF_3833_TR03_test;
    if (ChElementShellANCF_3833_TR03_test.RunElementChecks(0))
        print_green("ChElementShellANCF_3833_TR03 Element Checks = PASSED\n");
    else
        print_red("ChElementShellANCF_3833_TR03 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFShellTest<ChElementShellANCF_3833_TR04, ChMaterialShellANCF_3833_TR04> ChElementShellANCF_3833_TR04_test;
    if (ChElementShellANCF_3833_TR04_test.RunElementChecks(0))
        print_green("ChElementShellANCF_3833_TR04 Element Checks = PASSED\n");
    else
        print_red("ChElementShellANCF_3833_TR04 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFShellTest<ChElementShellANCF_3833_TR05, ChMaterialShellANCF_3833_TR05> ChElementShellANCF_3833_TR05_test;
    if (ChElementShellANCF_3833_TR05_test.RunElementChecks(0))
        print_green("ChElementShellANCF_3833_TR05 Element Checks = PASSED\n");
    else
        print_red("ChElementShellANCF_3833_TR05 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFShellTest<ChElementShellANCF_3833_TR06, ChMaterialShellANCF_3833_TR06> ChElementShellANCF_3833_TR06_test;
    if (ChElementShellANCF_3833_TR06_test.RunElementChecks(0))
        print_green("ChElementShellANCF_3833_TR06 Element Checks = PASSED\n");
    else
        print_red("ChElementShellANCF_3833_TR06 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFShellTest<ChElementShellANCF_3833_TR07, ChMaterialShellANCF_3833_TR07> ChElementShellANCF_3833_TR07_test;
    if (ChElementShellANCF_3833_TR07_test.RunElementChecks(0))
        print_green("ChElementShellANCF_3833_TR07 Element Checks = PASSED\n");
    else
        print_red("ChElementShellANCF_3833_TR07 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFShellTest<ChElementShellANCF_3833_TR08, ChMaterialShellANCF_3833_TR08> ChElementShellANCF_3833_TR08_test;
    if (ChElementShellANCF_3833_TR08_test.RunElementChecks(0))
        print_green("ChElementShellANCF_3833_TR08 Element Checks = PASSED\n");
    else
        print_red("ChElementShellANCF_3833_TR08 Element Checks = FAILED\n");

    //std::cout << "-------------------------------------" << std::endl;
    //ANCFShellTest<ChElementShellANCF_3833_TR08b, ChMaterialShellANCF_3833_TR08b> ChElementShellANCF_3833_TR08b_test;
    //if (ChElementShellANCF_3833_TR08b_test.RunElementChecks(0))
    //    print_green("ChElementShellANCF_3833_TR08b Element Checks = PASSED\n");
    //else
    //    print_red("ChElementShellANCF_3833_TR08b Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFShellTest<ChElementShellANCF_3833_TR09, ChMaterialShellANCF_3833_TR09> ChElementShellANCF_3833_TR09_test;
    if (ChElementShellANCF_3833_TR09_test.RunElementChecks(0))
        print_green("ChElementShellANCF_3833_TR09 Element Checks = PASSED\n");
    else
        print_red("ChElementShellANCF_3833_TR09 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFShellTest<ChElementShellANCF_3833_TR10, ChMaterialShellANCF_3833_TR10> ChElementShellANCF_3833_TR10_test;
    if (ChElementShellANCF_3833_TR10_test.RunElementChecks(0))
        print_green("ChElementShellANCF_3833_TR10 Element Checks = PASSED\n");
    else
        print_red("ChElementShellANCF_3833_TR10 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFShellTest<ChElementShellANCF_3833_TR11, ChMaterialShellANCF_3833_TR11> ChElementShellANCF_3833_TR11_test;
    if (ChElementShellANCF_3833_TR11_test.RunElementChecks(0))
        print_green("ChElementShellANCF_3833_TR11 Element Checks = PASSED\n");
    else
        print_red("ChElementShellANCF_3833_TR11 Element Checks = FAILED\n");

    return 0;
}

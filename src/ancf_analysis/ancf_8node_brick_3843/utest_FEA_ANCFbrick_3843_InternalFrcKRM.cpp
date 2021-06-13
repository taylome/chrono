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
// Unit test for ANCF brick elements
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
#include "chrono/fea/ChElementBrickANCF_3843_TR06_GQ444.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR07.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR07s.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR07s_GQ444.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR08.h"
//#include "chrono/fea/ChElementBrickANCF_3843_TR08b.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR08s.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR08s_GQ444.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR08T_GQ444.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR09.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR10.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR11.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR11s.h"

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
#define TIP_FORCE 100.0   // N

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
class ANCFBrickTest {
public:
    ANCFBrickTest();

    ~ANCFBrickTest() { delete m_system; }

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
ANCFBrickTest<ElementVersion, MaterialVersion>::ANCFBrickTest() {
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
    double height = 1.0; //m
    double rho = 7850; //kg/m^3
    double E = 210e9; //Pa
    double nu = 0.3;

    auto material = chrono_types::make_shared<MaterialVersion>(rho, E, nu);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    m_system->Add(mesh);

    // Setup Brick gradients to initially align with the global x, y, and z directions
    ChVector<> dir1(1, 0, 0);
    ChVector<> dir2(0, 1, 0);
    ChVector<> dir3(0, 0, 1);

    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, 0, 0.0), dir1, dir2, dir3);
    mesh->AddNode(nodeA);
    auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(length, 0, 0), dir1, dir2, dir3);
    mesh->AddNode(nodeB);
    m_nodeB = nodeB;
    auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(length, width, 0), dir1, dir2, dir3);
    mesh->AddNode(nodeC);
    auto nodeD = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, width, 0), dir1, dir2, dir3);
    mesh->AddNode(nodeD);
    auto nodeE = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, 0, height), dir1, dir2, dir3);
    mesh->AddNode(nodeE);
    auto nodeF = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(length,0, height), dir1, dir2, dir3);
    mesh->AddNode(nodeF);
    auto nodeG = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(length, width, height), dir1, dir2, dir3);
    mesh->AddNode(nodeG);
    auto nodeH = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0,width, height), dir1, dir2, dir3);
    mesh->AddNode(nodeH);



    auto element = chrono_types::make_shared<ElementVersion>();
    element->SetNodes(nodeA, nodeB, nodeC, nodeD, nodeE, nodeF, nodeG, nodeH);
    element->SetDimensions(length, width, height);
    element->SetMaterial(material);
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
bool ANCFBrickTest<ElementVersion, MaterialVersion>::RunElementChecks(int msglvl) {
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

    msglvl = 2;
    //tests_passed = (tests_passed & AxialDisplacementCheck(msglvl));
    //tests_passed = (tests_passed & CantileverCheck(msglvl));
    //tests_passed = (tests_passed & AxialTwistCheck(msglvl));

    return(tests_passed);
}

template <typename ElementVersion, typename MaterialVersion>
bool ANCFBrickTest<ElementVersion, MaterialVersion>::MassMatrixCheck(int msglvl) {
    // =============================================================================
    //  Check the Mass Matrix
    //  (Result should be nearly exact - No expected error)
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_MassMatrix_Compact;
    Expected_MassMatrix_Compact.resize(32, 32);
    if (!load_validation_data("UT_ANCFBrick_3843_MassMatrix.txt", Expected_MassMatrix_Compact))
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
    MassMatrix.resize(96, 96);
    m_element->ComputeKRMmatricesGlobal(MassMatrix, 0, 0, 1);
    ChMatrixNM<double, 32, 32> MassMatrix_compact;
    for (unsigned int i = 0; i < 32; i++) {
        for (unsigned int j = 0; j < 32; j++) {
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
bool ANCFBrickTest<ElementVersion, MaterialVersion>::GeneralizedGravityForceCheck(int msglvl) {
    // =============================================================================
    //  Generalized Force due to Gravity
    //  (Result should be nearly exact - No expected error)
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_InternalForceDueToGravity;
    Expected_InternalForceDueToGravity.resize(96, 1);
    if (!load_validation_data("UT_ANCFBrick_3843_Grav.txt", Expected_InternalForceDueToGravity))
        return false;

    ChVectorDynamic<double> InternalForceNoDispNoVelWithGravity;
    InternalForceNoDispNoVelWithGravity.resize(96);
    m_element->SetGravityOn(true);
    m_element->ComputeInternalForces(InternalForceNoDispNoVelWithGravity);

    ChVectorDynamic<double> InternalForceNoDispNoVelNoGravity;
    InternalForceNoDispNoVelNoGravity.resize(96);
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
bool ANCFBrickTest<ElementVersion, MaterialVersion>::GeneralizedInternalForceNoDispNoVelCheck(int msglvl) {
    // =============================================================================
    //  Check the Internal Force at Zero Displacement, Zero Velocity
    //  (should equal all 0's by definition)
    //  (Assumes that the element has not been changed from the initialized state)
    // =============================================================================

    ChVectorDynamic<double> InternalForceNoDispNoVelNoGravity;
    InternalForceNoDispNoVelNoGravity.resize(96);
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
bool ANCFBrickTest<ElementVersion, MaterialVersion>::GeneralizedInternalForceSmallDispNoVelCheck(int msglvl) {
    // =============================================================================
    //  Check the Internal Force at a Given Displacement 
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_InternalForceSmallDispNoVelNoGravity;
    Expected_InternalForceSmallDispNoVelNoGravity.resize(96, 1);
    if (!load_validation_data("UT_ANCFBrick_3843_IntFrcSmallDispNoVelNoGravity.txt", Expected_InternalForceSmallDispNoVelNoGravity))
        return false;

    //Setup the test conditions
    ChVector<double> OriginalPos = m_nodeB->GetPos();
    m_nodeB->SetPos(ChVector<>(m_nodeB->GetPos().x(), m_nodeB->GetPos().y(), 0.001));

    ChVectorDynamic<double> InternalForceSmallDispNoVelNoGravity;
    InternalForceSmallDispNoVelNoGravity.resize(96);
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
bool ANCFBrickTest<ElementVersion, MaterialVersion>::GeneralizedInternalForceNoDispSmallVelCheck(int msglvl) {
    // =============================================================================
    //  Check the Internal Force at a No Displacement with a Given Nodal Velocity
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_InternalForceNoDispSmallVelNoGravity;
    Expected_InternalForceNoDispSmallVelNoGravity.resize(96, 1);
    if (!load_validation_data("UT_ANCFBrick_3843_IntFrcNoDispSmallVelNoGravity.txt", Expected_InternalForceNoDispSmallVelNoGravity))
        return false;

    //Setup the test conditions
    ChVector<double> OriginalVel = m_nodeB->GetPos_dt();
    m_nodeB->SetPos_dt(ChVector<>(0.0, 0.0, 0.001));
    m_element->SetAlphaDamp(0.01);

    ChVectorDynamic<double> InternalForceNoDispSmallVelNoGravity;
    InternalForceNoDispSmallVelNoGravity.resize(96);
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
bool ANCFBrickTest<ElementVersion, MaterialVersion>::JacobianNoDispNoVelNoDampingCheck(int msglvl) {
    // =============================================================================
    //  Check the Jacobian at No Displacement/Velocity - No Damping
    //  (some small error expected depending on the formulation/steps used)
    //  (The R contribution should be all zeros since Damping is not enabled)
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianK_NoDispNoVelNoDamping;
    Expected_JacobianK_NoDispNoVelNoDamping.resize(96, 96);
    if (!load_validation_data("UT_ANCFBrick_3843_JacNoDispNoVelNoDamping.txt", Expected_JacobianK_NoDispNoVelNoDamping))
        return false;

    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation
    ChVectorDynamic<double> InternalForceNoDispNoVelNoGravity;
    InternalForceNoDispNoVelNoGravity.resize(96);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceNoDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_NoDispNoVelNoDamping;
    JacobianK_NoDispNoVelNoDamping.resize(96, 96);
    m_element->ComputeKRMmatricesGlobal(JacobianK_NoDispNoVelNoDamping, 1.0, 0.0, 0.0);

    ChMatrixDynamic<double> JacobianR_NoDispNoVelNoDamping;
    JacobianR_NoDispNoVelNoDamping.resize(96, 96);
    m_element->ComputeKRMmatricesGlobal(JacobianR_NoDispNoVelNoDamping, 0.0, 1.0, 0.0);

    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_NoDispNoVelNoDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(96, 96);
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
bool ANCFBrickTest<ElementVersion, MaterialVersion>::JacobianSmallDispNoVelNoDampingCheck(int msglvl) {
    // =============================================================================
    //  Check the Jacobian at Small Displacement No Velocity - No Damping
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianK_SmallDispNoVelNoDamping;
    Expected_JacobianK_SmallDispNoVelNoDamping.resize(96, 96);
    if (!load_validation_data("UT_ANCFBrick_3843_JacSmallDispNoVelNoDamping.txt", Expected_JacobianK_SmallDispNoVelNoDamping))
        return false;

    //Ensure that the internal force is recalculated in case the results are expected
    //by the Jacobian Calculation
    
    //Setup the test conditions
    ChVector<double> OriginalPos = m_nodeB->GetPos();
    m_nodeB->SetPos(ChVector<>(m_nodeB->GetPos().x(), m_nodeB->GetPos().y(), 0.001));

    ChVectorDynamic<double> InternalForceSmallDispNoVelNoGravity;
    InternalForceSmallDispNoVelNoGravity.resize(96);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceSmallDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_SmallDispNoVelNoDamping;
    JacobianK_SmallDispNoVelNoDamping.resize(96, 96);
    m_element->ComputeKRMmatricesGlobal(JacobianK_SmallDispNoVelNoDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_SmallDispNoVelNoDamping;
    JacobianR_SmallDispNoVelNoDamping.resize(96, 96);
    m_element->ComputeKRMmatricesGlobal(JacobianR_SmallDispNoVelNoDamping, 0, 1, 0);

    //Reset the element conditions back to its original values
    m_nodeB->SetPos(OriginalPos);


    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_SmallDispNoVelNoDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(96, 96);
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
bool ANCFBrickTest<ElementVersion, MaterialVersion>::JacobianNoDispNoVelWithDampingCheck(int msglvl) {
    // =============================================================================
    //  Check the Jacobian at No Displacement/Velocity - With Damping
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_Jacobians;
    Expected_Jacobians.resize(2 * 96, 96);
    if (!load_validation_data("UT_ANCFBrick_3843_JacNoDispNoVelWithDamping.txt", Expected_Jacobians))
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
    InternalForceNoDispNoVelNoGravity.resize(96);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceNoDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_NoDispNoVelWithDamping;
    JacobianK_NoDispNoVelWithDamping.resize(96, 96);
    m_element->ComputeKRMmatricesGlobal(JacobianK_NoDispNoVelWithDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_NoDispNoVelWithDamping;
    JacobianR_NoDispNoVelWithDamping.resize(96, 96);
    m_element->ComputeKRMmatricesGlobal(JacobianR_NoDispNoVelWithDamping, 0, 1, 0);

    //Reset the element conditions back to its original values
    m_element->SetAlphaDamp(0.0);


    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_NoDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(96, 96);
    
    double small_terms_JacR = 1e-4*Expected_JacobianR_NoDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacR = 0;
    double max_percent_error_JacR = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacR;
    percent_error_matrix_JacR.resize(96, 96);


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
bool ANCFBrickTest<ElementVersion, MaterialVersion>::JacobianSmallDispNoVelWithDampingCheck(int msglvl) {
    // =============================================================================
    //  Check the Jacobian at Small Displacement No Velocity - With Damping
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_Jacobians;
    Expected_Jacobians.resize(2 * 96, 96);
    if (!load_validation_data("UT_ANCFBrick_3843_JacSmallDispNoVelWithDamping.txt", Expected_Jacobians))
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
    InternalForceSmallDispNoVelNoGravity.resize(96);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceSmallDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_SmallDispNoVelWithDamping;
    JacobianK_SmallDispNoVelWithDamping.resize(96, 96);
    m_element->ComputeKRMmatricesGlobal(JacobianK_SmallDispNoVelWithDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_SmallDispNoVelWithDamping;
    JacobianR_SmallDispNoVelWithDamping.resize(96, 96);
    m_element->ComputeKRMmatricesGlobal(JacobianR_SmallDispNoVelWithDamping, 0, 1, 0);

    //Reset the element conditions back to its original values
    m_nodeB->SetPos(OriginalPos);
    m_element->SetAlphaDamp(0.0);


    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_SmallDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(96, 96);

    double small_terms_JacR = 1e-4*Expected_JacobianR_SmallDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacR = 0;
    double max_percent_error_JacR = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacR;
    percent_error_matrix_JacR.resize(96, 96);


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
bool ANCFBrickTest<ElementVersion, MaterialVersion>::JacobianNoDispSmallVelWithDampingCheck(int msglvl) {
    // =============================================================================
    //  Check the Jacobian at No Displacement Small Velocity - With Damping
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_Jacobians;
    Expected_Jacobians.resize(2 * 96, 96);
    if (!load_validation_data("UT_ANCFBrick_3843_JacNoDispSmallVelWithDamping.txt", Expected_Jacobians))
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
    InternalForceNoDispSmallVelNoGravity.resize(96);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceNoDispSmallVelNoGravity);

    ChMatrixDynamic<double> JacobianK_NoDispSmallVelWithDamping;
    JacobianK_NoDispSmallVelWithDamping.resize(96, 96);
    m_element->ComputeKRMmatricesGlobal(JacobianK_NoDispSmallVelWithDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_NoDispSmallVelWithDamping;
    JacobianR_NoDispSmallVelWithDamping.resize(96, 96);
    m_element->ComputeKRMmatricesGlobal(JacobianR_NoDispSmallVelWithDamping, 0, 1, 0);

    //Reset the element conditions back to its original values
    m_nodeB->SetPos_dt(OriginalVel);
    m_element->SetAlphaDamp(0.0);


    bool passed_tests = true;

    double small_terms_JacK = 1e-4*Expected_JacobianK_NoDispSmallVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(96, 96);

    double small_terms_JacR = 1e-4*Expected_JacobianR_NoDispSmallVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacR = 0;
    double max_percent_error_JacR = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacR;
    percent_error_matrix_JacR.resize(96, 96);


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
bool ANCFBrickTest<ElementVersion, MaterialVersion>::AxialDisplacementCheck(int msglvl) {
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
    int num_elements = 1;
    double length = 1;    // m
    double width = 0.01;  // m (square cross-section)
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

    // Setup Brick normals to initially align with the global axes
    ChVector<> dir1(1, 0, 0);
    ChVector<> dir2(0, 1, 0);
    ChVector<> dir3(0, 0, 1);

    // Create the first nodes and fix them completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, -0.5*width, -0.5*width), dir1, dir2, dir3);
    mesh->AddNode(nodeA);
    nodeA->SetFixed(true);
    auto nodeD = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, 0.5*width, -0.5*width), dir1, dir2, dir3);
    mesh->AddNode(nodeD);
    nodeD->SetFixed(true);
    auto nodeE = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, -0.5*width, 0.5*width), dir1, dir2, dir3);
    mesh->AddNode(nodeE);
    nodeE->SetFixed(true);
    auto nodeH = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, 0.5*width, 0.5*width), dir1, dir2, dir3);
    mesh->AddNode(nodeE);
    nodeE->SetFixed(true);


    auto elementlast = chrono_types::make_shared<ElementVersion>();
    std::shared_ptr<ChNodeFEAxyzDD> nodeEndPoint;

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
        elementlast = element;
    }

    nodeEndPoint = nodeA;

    mesh->SetAutomaticGravity(false);  // Turn off the default method for applying gravity to the mesh since it is less
                                       // efficient for ANCF elements

    // Create a custom atomic (point) load
    class MyLoaderTimeDependentTipLoad : public ChLoaderUVWatomic {
    public:
        // Useful: a constructor that also sets ChLoadable
        MyLoaderTimeDependentTipLoad(std::shared_ptr<ChLoadableUVW> mloadable) : ChLoaderUVWatomic(mloadable, 0, 0, 0) {}

        // Compute F=F(u), the load at U. The load is a 6-row vector, i.e.
        // a wrench: forceX, forceY, forceZ, torqueX, torqueY, torqueZ.
        virtual void ComputeF(
            const double U,              ///< normalized position along the beam axis [-1...1]
            const double V,              ///< normalized position along the beam axis [-1...1]
            const double W,              ///< normalized position along the beam axis [-1...1]
            ChVectorDynamic<>& F,        ///< Load at U
            ChVectorDynamic<>* state_x,  ///< if != 0, update state (pos. part) to this, then evaluate F
            ChVectorDynamic<>* state_w   ///< if != 0, update state (speed part) to this, then evaluate F
        ) override {
            assert(auxsystem);

            F.setZero();
            F(0) = TIP_FORCE/4.0;  // Apply the force along the global X axis (beam axis)
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
    mload->loader.SetApplication(1.0,0.0,0.0);  // specify application point
    loadcontainer->Add(mload);          // add the load to the load container.

    system->Update();
    ChVectorDynamic<double> Fi(96);
    elementlast->ComputeInternalForces(Fi);
    std::cout << "Pre Call: " << Fi << std::endl;
    // Find the nonlinear static solution for the system (final displacement)
    //system->DoStaticLinear();
    //system->DoStaticNonlinear(50);
    for(auto step = 0; step<100; step++)
        system->DoStepDynamics(1e-3);

    //elementlast->ComputeInternalForces(Fi);
    std::cout << "Post Call: " << Fi << std::endl;
    ChVectorDynamic<double> Qi(96);
    double detJ;
    ChVectorDynamic<>* state_x;
    ChVectorDynamic<>* state_w;
    ChVectorDynamic<double> F(6);
    F.setZero();
    F(0) = TIP_FORCE;
    elementlast->ComputeNF(1, 0, 0, Qi, detJ, F, state_x, state_w);
    std::cout << "Fi-Qi: " << Fi-Qi << std::endl;


    // Calculate the axial displacement of the end of the ANCF Brick mesh
    ChVector<> point;
    ChQuaternion<> rot;
    elementlast->EvaluateSectionFrame(1, 0, 0, point, rot);
    
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
bool ANCFBrickTest<ElementVersion, MaterialVersion>::CantileverCheck(int msglvl) {
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
    int num_elements = 1;
    double length = 1;    // m
    double width = 0.01;  // m (square cross-section)
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

    // Setup Brick normals to initially align with the global axes
    ChVector<> dir1(1, 0, 0);
    ChVector<> dir2(0, 1, 0);
    ChVector<> dir3(0, 0, 1);

    // Create the first nodes and fix them completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, -0.5*width, -0.5*width), dir1, dir2, dir3);
    mesh->AddNode(nodeA);
    nodeA->SetFixed(true);
    auto nodeD = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, 0.5*width, -0.5*width), dir1, dir2, dir3);
    mesh->AddNode(nodeD);
    nodeD->SetFixed(true);
    auto nodeE = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, -0.5*width, 0.5*width), dir1, dir2, dir3);
    mesh->AddNode(nodeE);
    nodeE->SetFixed(true);
    auto nodeH = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, 0.5*width, 0.5*width), dir1, dir2, dir3);
    mesh->AddNode(nodeE);
    nodeE->SetFixed(true);


    auto elementlast = chrono_types::make_shared<ElementVersion>();
    std::shared_ptr<ChNodeFEAxyzDD> nodeEndPoint;

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
        element->SetAlphaDamp(0.0);
        element->SetGravityOn(
            false);  // Enable the efficient ANCF method for calculating the application of gravity to the element
        //element->SetStrainFormulation(ElementVersion::StrainFormulation::CMPoisson);
        // element->SetStrainFormulation(ElementVersion::StrainFormulation::CMNoPoisson);
        mesh->AddElement(element);

        nodeA = nodeB;
        nodeD = nodeC;
        nodeE = nodeF;
        nodeH = nodeG;
        elementlast = element;
    }

    nodeEndPoint = nodeA;

    mesh->SetAutomaticGravity(false);  // Turn off the default method for applying gravity to the mesh since it is less
                                       // efficient for ANCF elements

    // Create a custom atomic (point) load
    class MyLoaderTimeDependentTipLoad : public ChLoaderUVWatomic {
    public:
        // Useful: a constructor that also sets ChLoadable
        MyLoaderTimeDependentTipLoad(std::shared_ptr<ChLoadableUVW> mloadable) : ChLoaderUVWatomic(mloadable, 0, 0, 0) {}

        // Compute F=F(u), the load at U. The load is a 6-row vector, i.e.
        // a wrench: forceX, forceY, forceZ, torqueX, torqueY, torqueZ.
        virtual void ComputeF(
            const double U,              ///< normalized position along the beam axis [-1...1]
            const double V,              ///< normalized position along the beam axis [-1...1]
            const double W,              ///< normalized position along the beam axis [-1...1]
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
    mload->loader.SetApplication(1.0, 0.0, 0.0);  // specify application point
    loadcontainer->Add(mload);          // add the load to the load container.


    ChVectorDynamic<double> Fi(96);
    elementlast->ComputeInternalForces(Fi);
    std::cout << "Pre Call: " << Fi << std::endl;
    // Find the nonlinear static solution for the system (final displacement)
    system->DoStaticLinear();
    //system->DoStaticNonlinear(50);
    std::cout << "Post Call: "<<Fi << std::endl;

    // Calculate the displacement of the end of the ANCF beam mesh
    ChVector<> point;
    ChQuaternion<> rot;
    elementlast->EvaluateSectionFrame(1, 0, 0, point, rot);

    // For Analytical Formula, see a mechanics of materials textbook (delta = PL^3/3EI)
    double I = 1.0 / 12.0 * width * std::pow(width, 3);
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
bool ANCFBrickTest<ElementVersion, MaterialVersion>::AxialTwistCheck(int msglvl) {
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
    int num_elements = 100;
    double length = 1;    // m
    double width = 0.01;  // m (square cross-section)
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

    // Setup Brick normals to initially align with the global axes
    ChVector<> dir1(1, 0, 0);
    ChVector<> dir2(0, 1, 0);
    ChVector<> dir3(0, 0, 1);

    // Create the first nodes and fix them completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, -0.5*width, -0.5*width), dir1, dir2, dir3);
    mesh->AddNode(nodeA);
    nodeA->SetFixed(true);
    auto nodeD = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, 0.5*width, -0.5*width), dir1, dir2, dir3);
    mesh->AddNode(nodeD);
    nodeD->SetFixed(true);
    auto nodeE = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, -0.5*width, 0.5*width), dir1, dir2, dir3);
    mesh->AddNode(nodeE);
    nodeE->SetFixed(true);
    auto nodeH = chrono_types::make_shared<ChNodeFEAxyzDDD>(ChVector<>(0, 0.5*width, 0.5*width), dir1, dir2, dir3);
    mesh->AddNode(nodeE);
    nodeE->SetFixed(true);


    auto elementlast = chrono_types::make_shared<ElementVersion>();
    std::shared_ptr<ChNodeFEAxyzDD> nodeEndPoint;

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
        element->SetAlphaDamp(0.0);
        element->SetGravityOn(
            false);  // Enable the efficient ANCF method for calculating the application of gravity to the element
        //element->SetStrainFormulation(ElementVersion::StrainFormulation::CMPoisson);
        // element->SetStrainFormulation(ElementVersion::StrainFormulation::CMNoPoisson);
        mesh->AddElement(element);

        nodeA = nodeB;
        nodeD = nodeC;
        nodeE = nodeF;
        nodeH = nodeG;
        elementlast = element;
    }

    nodeEndPoint = nodeA;

    mesh->SetAutomaticGravity(false);  // Turn off the default method for applying gravity to the mesh since it is less
                                       // efficient for ANCF elements

    // Create a custom atomic (point) load
    class MyLoaderTimeDependentTipLoad : public ChLoaderUVWatomic {
    public:
        // Useful: a constructor that also sets ChLoadable
        MyLoaderTimeDependentTipLoad(std::shared_ptr<ChLoadableUVW> mloadable) : ChLoaderUVWatomic(mloadable, 0, 0, 0) {}

        // Compute F=F(u), the load at U. The load is a 6-row vector, i.e.
        // a wrench: forceX, forceY, forceZ, torqueX, torqueY, torqueZ.
        virtual void ComputeF(
            const double U,              ///< normalized position along the beam axis [-1...1]
            const double V,              ///< normalized position along the beam axis [-1...1]
            const double W,              ///< normalized position along the beam axis [-1...1]
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
    mload->loader.SetApplication(1.0, 0.0, 0.0);  // specify application point
    loadcontainer->Add(mload);          // add the load to the load container.

    // Find the nonlinear static solution for the system (final twist angle)
    system->DoStaticLinear();
    //system->DoStaticNonlinear(50);

    // Calculate the twist angle of the end of the ANCF beam mesh
    ChVector<> point;
    ChQuaternion<> rot;
    elementlast->EvaluateSectionFrame(1, 0, 0, point, rot);
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
        std::cout << "Axial Twist Angle Check (Percent Error less than 20%) - Percent Error: " << Percent_Error << "% ";
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
    ANCFBrickTest<ChElementBrickANCF_3843_TR01, ChMaterialBrickANCF_3843_TR01> ChElementBrickANCF_3843_TR01_test;
    if (ChElementBrickANCF_3843_TR01_test.RunElementChecks(0))
        print_green("ChElementBrickANCF_3843_TR01 Element Checks = PASSED\n");
    else
        print_red("ChElementBrickANCF_3843_TR01 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBrickTest<ChElementBrickANCF_3843_TR02, ChMaterialBrickANCF_3843_TR02> ChElementBrickANCF_3843_TR02_test;
    if (ChElementBrickANCF_3843_TR02_test.RunElementChecks(0))
        print_green("ChElementBrickANCF_3843_TR02 Element Checks = PASSED\n");
    else
        print_red("ChElementBrickANCF_3843_TR02 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBrickTest<ChElementBrickANCF_3843_TR03, ChMaterialBrickANCF_3843_TR03> ChElementBrickANCF_3843_TR03_test;
    if (ChElementBrickANCF_3843_TR03_test.RunElementChecks(0))
        print_green("ChElementBrickANCF_3843_TR03 Element Checks = PASSED\n");
    else
        print_red("ChElementBrickANCF_3843_TR03 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBrickTest<ChElementBrickANCF_3843_TR04, ChMaterialBrickANCF_3843_TR04> ChElementBrickANCF_3843_TR04_test;
    if (ChElementBrickANCF_3843_TR04_test.RunElementChecks(0))
        print_green("ChElementBrickANCF_3843_TR04 Element Checks = PASSED\n");
    else
        print_red("ChElementBrickANCF_3843_TR04 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBrickTest<ChElementBrickANCF_3843_TR05, ChMaterialBrickANCF_3843_TR05> ChElementBrickANCF_3843_TR05_test;
    if (ChElementBrickANCF_3843_TR05_test.RunElementChecks(0))
        print_green("ChElementBrickANCF_3843_TR05 Element Checks = PASSED\n");
    else
        print_red("ChElementBrickANCF_3843_TR05 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBrickTest<ChElementBrickANCF_3843_TR06, ChMaterialBrickANCF_3843_TR06> ChElementBrickANCF_3843_TR06_test;
    if (ChElementBrickANCF_3843_TR06_test.RunElementChecks(0))
        print_green("ChElementBrickANCF_3843_TR06 Element Checks = PASSED\n");
    else
        print_red("ChElementBrickANCF_3843_TR06 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBrickTest<ChElementBrickANCF_3843_TR06_GQ444, ChMaterialBrickANCF_3843_TR06_GQ444> ChElementBrickANCF_3843_TR06_GQ444_test;
    if (ChElementBrickANCF_3843_TR06_GQ444_test.RunElementChecks(0))
        print_green("ChElementBrickANCF_3843_TR06_GQ444 Element Checks = PASSED\n");
    else
        print_red("ChElementBrickANCF_3843_TR06_GQ444 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBrickTest<ChElementBrickANCF_3843_TR07, ChMaterialBrickANCF_3843_TR07> ChElementBrickANCF_3843_TR07_test;
    if (ChElementBrickANCF_3843_TR07_test.RunElementChecks(0))
        print_green("ChElementBrickANCF_3843_TR07 Element Checks = PASSED\n");
    else
        print_red("ChElementBrickANCF_3843_TR07 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBrickTest<ChElementBrickANCF_3843_TR07S, ChMaterialBrickANCF_3843_TR07S> ChElementBrickANCF_3843_TR07S_test;
    if (ChElementBrickANCF_3843_TR07S_test.RunElementChecks(0))
        print_green("ChElementBrickANCF_3843_TR07S Element Checks = PASSED\n");
    else
        print_red("ChElementBrickANCF_3843_TR07S Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBrickTest<ChElementBrickANCF_3843_TR07S_GQ444, ChMaterialBrickANCF_3843_TR07S_GQ444> ChElementBrickANCF_3843_TR07S_GQ444_test;
    if (ChElementBrickANCF_3843_TR07S_GQ444_test.RunElementChecks(0))
        print_green("ChElementBrickANCF_3843_TR07S_GQ444 Element Checks = PASSED\n");
    else
        print_red("ChElementBrickANCF_3843_TR07S_GQ444 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBrickTest<ChElementBrickANCF_3843_TR08, ChMaterialBrickANCF_3843_TR08> ChElementBrickANCF_3843_TR08_test;
    if (ChElementBrickANCF_3843_TR08_test.RunElementChecks(0))
        print_green("ChElementBrickANCF_3843_TR08 Element Checks = PASSED\n");
    else
        print_red("ChElementBrickANCF_3843_TR08 Element Checks = FAILED\n");

    //std::cout << "-------------------------------------" << std::endl;
    //ANCFBrickTest<ChElementBrickANCF_3843_TR08b, ChMaterialBrickANCF_3843_TR08b> ChElementBrickANCF_3843_TR08b_test;
    //if (ChElementBrickANCF_3843_TR08b_test.RunElementChecks(0))
    //    print_green("ChElementBrickANCF_3843_TR08b Element Checks = PASSED\n");
    //else
    //    print_red("ChElementBrickANCF_3843_TR08b Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBrickTest<ChElementBrickANCF_3843_TR08S, ChMaterialBrickANCF_3843_TR08S> ChElementBrickANCF_3843_TR08S_test;
    if (ChElementBrickANCF_3843_TR08S_test.RunElementChecks(0))
        print_green("ChElementBrickANCF_3843_TR08S Element Checks = PASSED\n");
    else
        print_red("ChElementBrickANCF_3843_TR08S Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBrickTest<ChElementBrickANCF_3843_TR08S_GQ444, ChMaterialBrickANCF_3843_TR08S_GQ444> ChElementBrickANCF_3843_TR08S_GQ444_test;
    if (ChElementBrickANCF_3843_TR08S_GQ444_test.RunElementChecks(0))
        print_green("ChElementBrickANCF_3843_TR08S_GQ444 Element Checks = PASSED\n");
    else
        print_red("ChElementBrickANCF_3843_TR08S_GQ444 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBrickTest<ChElementBrickANCF_3843_TR08T_GQ444, ChMaterialBrickANCF_3843_TR08T_GQ444> ChElementBrickANCF_3843_TR08T_GQ444_test;
    if (ChElementBrickANCF_3843_TR08T_GQ444_test.RunElementChecks(0))
        print_green("ChElementBrickANCF_3843_TR08T_GQ444 Element Checks = PASSED\n");
    else
        print_red("ChElementBrickANCF_3843_TR08T_GQ444 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBrickTest<ChElementBrickANCF_3843_TR09, ChMaterialBrickANCF_3843_TR09> ChElementBrickANCF_3843_TR09_test;
    if (ChElementBrickANCF_3843_TR09_test.RunElementChecks(0))
        print_green("ChElementBrickANCF_3843_TR09 Element Checks = PASSED\n");
    else
        print_red("ChElementBrickANCF_3843_TR09 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBrickTest<ChElementBrickANCF_3843_TR10, ChMaterialBrickANCF_3843_TR10> ChElementBrickANCF_3843_TR10_test;
    if (ChElementBrickANCF_3843_TR10_test.RunElementChecks(0))
        print_green("ChElementBrickANCF_3843_TR10 Element Checks = PASSED\n");
    else
        print_red("ChElementBrickANCF_3843_TR10 Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBrickTest<ChElementBrickANCF_3843_TR11, ChMaterialBrickANCF_3843_TR11> ChElementBrickANCF_3843_TR11_test;
    if (ChElementBrickANCF_3843_TR11_test.RunElementChecks(0))
        print_green("ChElementBrickANCF_3843_TR11 Element Checks = PASSED\n");
    else
        print_red("ChElementBrickANCF_3843_TR11 Element Checks = FAILED\n"); 

    std::cout << "-------------------------------------" << std::endl;
    ANCFBrickTest<ChElementBrickANCF_3843_TR11S, ChMaterialBrickANCF_3843_TR11S> ChElementBrickANCF_3843_TR11S_test;
    if (ChElementBrickANCF_3843_TR11S_test.RunElementChecks(0))
        print_green("ChElementBrickANCF_3843_TR11S Element Checks = PASSED\n");
    else
        print_red("ChElementBrickANCF_3843_TR11S Element Checks = FAILED\n"); 

    return 0;
}
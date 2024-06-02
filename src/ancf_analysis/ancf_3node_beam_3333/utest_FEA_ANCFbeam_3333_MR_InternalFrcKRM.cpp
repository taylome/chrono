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

#include "chrono/fea/ChElementBeamANCF_3333_MR_Damp.h"
#include "chrono/fea/ChElementBeamANCF_3333_MR_NoDamp.h"
#include "chrono/fea/ChElementBeamANCF_3333_MR_DampNumJac.h"
#include "chrono/fea/ChElementBeamANCF_3333_MR_NoDampNumJac.h"

#include "chrono/fea/ChMesh.h"

#include "chrono/physics/ChLoadContainer.h"
#include "chrono/fea/ChLoadsBeam.h"

#if defined WIN32 || defined _WIN32 || defined WIN64 || defined _WIN64
    #include <windows.h>
#endif

using namespace chrono;
using namespace chrono::fea;

#define TIP_MOMENT 1.0  // Nm
#define TIP_FORCE 1.0   // N
#define Jac_Error 0.33   // Percent error as decimal

// =============================================================================

static const std::string ref_dir = "../data/testing/fea/";

// =============================================================================

void print_green(std::string text) {
    std::cout << "\033[1;32m" << text << "\033[0m";
}

void print_red(std::string text) {
    std::cout << "\033[1;31m" << text << "\033[0m";
}

bool load_validation_data(const std::string& filename,
                          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& data) {
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

template <typename ElementVersion>
class ANCFBeamTest {
  public:
    static const int NSF = 9;  ///< number of shape functions

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
    bool CantileverTipLoadCheck(int msglvl);
    bool CantileverGravityCheck(int msglvl);
    bool AxialTwistCheck(int msglvl);
    bool NonlinearAxialDisplacementCheck(int msglvl);

  protected:
    ChSystemSMC* m_system;
    std::shared_ptr<ElementVersion> m_element;
    std::shared_ptr<ChNodeFEAxyzDD> m_nodeC;
    std::shared_ptr<ChMaterialBeamANCF_MR> m_material;
};

// =============================================================================

template <typename ElementVersion>
ANCFBeamTest<ElementVersion>::ANCFBeamTest() {
    m_system = new ChSystemSMC();
    m_system->SetGravitationalAcceleration(ChVector3d(0, 0, -9.80665));

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
    integrator->SetAbsTolerances(1e-5);
    integrator->SetVerbose(false);
    integrator->SetModifiedNewton(false);

    // Mesh properties
    double length = 1.0;      //m
    double width = 0.1;     //m
    double thickness = 0.1; //m
    double rho = 7850; //kg/m^3
    double c10 = 0.8e6;  ///Pa
    double c01 = 0.2e6;    ///Pa
    double k = 1e9;    ///Pa
    double mu = 0.01 * 6 * (c10 + c01);     ///< Viscosity Coefficient

    m_material = chrono_types::make_shared<ChMaterialBeamANCF_MR>(rho, c10, c01, k, mu);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    m_system->Add(mesh);

    // Rotate the cross section gradients to match the sign convention in the experiment
    ChVector3d dir1(0, 1, 0);
    ChVector3d dir2(0, 0, 1);

    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector3d(0, 0, 0.0), dir1, dir2);
    nodeA->SetFixed(true);
    mesh->AddNode(nodeA);
    auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector3d(length, 0, 0), dir1, dir2);
    mesh->AddNode(nodeB);
    auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector3d(length / 2, 0, 0), dir1, dir2);
    mesh->AddNode(nodeC);
    m_nodeC = nodeC;

    auto element = chrono_types::make_shared<ElementVersion>();
    element->SetNodes(nodeA, nodeB, nodeC);
    element->SetDimensions(length, thickness, width);
    element->SetMaterial(m_material);
    mesh->AddElement(element);

    m_element = element;

    // =============================================================================
    //  Force a call to SetupInital so that all of the pre-computation steps are called
    // =============================================================================

    m_system->Update();
}

template <typename ElementVersion>
bool ANCFBeamTest<ElementVersion>::RunElementChecks(int msglvl) {
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

    tests_passed = (tests_passed & AxialDisplacementCheck(msglvl));
    tests_passed = (tests_passed & CantileverTipLoadCheck(msglvl));
    tests_passed = (tests_passed & CantileverGravityCheck(msglvl));
    tests_passed = (tests_passed & AxialTwistCheck(msglvl));
    tests_passed = (tests_passed & NonlinearAxialDisplacementCheck(msglvl));

    return (tests_passed);
}

template <typename ElementVersion>
bool ANCFBeamTest<ElementVersion>::MassMatrixCheck(int msglvl) {
    // =============================================================================
    //  Check the Mass Matrix
    //  (Result should be nearly exact - No expected error)
    // =============================================================================
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_MassMatrix_Compact;
    Expected_MassMatrix_Compact.resize(NSF, NSF);
    if (!load_validation_data("UT_ANCFBeam_3333_MassMatrix.txt", Expected_MassMatrix_Compact))
        return false;

    // Due to the known sparsity and repetitive pattern of the mass matrix, only 1 entries is saved per 3x3 block.  For
    // the unit expand out the reference solution to its full sparse and repetitive size for checking against the return
    // solution from the element.
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

    // Obtain the mass matrix from the Jacobian calculation by setting the scaling factor on the mass matrix to 1 and
    // the scaling on the partial derivatives with respect to the nodal coordinates and time derivative of the nodal
    // coordinates to zeros.
    ChMatrixDynamic<double> MassMatrix;
    MassMatrix.resize(3 * NSF, 3 * NSF);
    m_element->ComputeKRMmatricesGlobal(MassMatrix, 0, 0, 1);

    double MaxAbsError = (MassMatrix - Expected_MassMatrix).cwiseAbs().maxCoeff();
    // "Exact" numeric integration is used in the element formulation so no significant error is expected in this
    // solution
    bool passed_test = (MaxAbsError <= 0.001);

    if (msglvl >= 2) {
        std::cout << "Mass Matrix = " << std::endl;
        std::cout << MassMatrix << std::endl;

        // Reduce the mass matrix to the same compact form that is stored in the reference solution file to make
        // debugging easier.
        ChMatrixNM<double, NSF, NSF> MassMatrix_compact;
        for (unsigned int i = 0; i < NSF; i++) {
            for (unsigned int j = 0; j < NSF; j++) {
                MassMatrix_compact(i, j) = MassMatrix(3 * i, 3 * j);
            }
        }

        std::cout << "Mass Matrix (Compact Form) = " << std::endl;
        std::cout << MassMatrix_compact << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Mass Matrix (Max Abs Error) = " << MaxAbsError;

        if (passed_test)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");
    }

    return (passed_test);
}

template <typename ElementVersion>
bool ANCFBeamTest<ElementVersion>::GeneralizedGravityForceCheck(int msglvl) {
    // =============================================================================
    //  Generalized Force due to Gravity
    //  (Result should be nearly exact - No expected error)
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_InternalForceDueToGravity;
    Expected_InternalForceDueToGravity.resize(3 * NSF, 1);
    if (!load_validation_data("UT_ANCFBeam_3333_Grav.txt", Expected_InternalForceDueToGravity))
        return false;

    // For efficiency, a fast ANCF specific generalized force due to gravity is defined. This is included in the
    // ComputeGravityForces method for the element
    ChVectorDynamic<double> GeneralizedForceDueToGravity;
    GeneralizedForceDueToGravity.resize(3 * NSF);
    m_element->ComputeGravityForces(GeneralizedForceDueToGravity, m_system->GetGravitationalAcceleration());

    // "Exact" numeric integration is used in the element formulation so no significant error is expected in this
    // solution
    double MaxAbsError = (GeneralizedForceDueToGravity - Expected_InternalForceDueToGravity).cwiseAbs().maxCoeff();
    bool passed_test = (MaxAbsError <= 0.001);

    if (msglvl >= 2) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Generalized Force due to Gravity = " << std::endl;
        std::cout << GeneralizedForceDueToGravity << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Generalized Force due to Gravity (Max Abs Error) = " << MaxAbsError;

        if (passed_test)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");
    }

    return (passed_test);
}

template <typename ElementVersion>
bool ANCFBeamTest<ElementVersion>::GeneralizedInternalForceNoDispNoVelCheck(int msglvl) {
    // =============================================================================
    //  Check the Internal Force at Zero Displacement, Zero Velocity
    //  (should equal all 0's by definition)
    //  (Assumes that the element has not been changed from the initialized state)
    // =============================================================================

    ChVectorDynamic<double> InternalForceNoDispNoVel;
    InternalForceNoDispNoVel.resize(3 * NSF);
    m_element->ComputeInternalForces(InternalForceNoDispNoVel);

    double MaxAbsError = InternalForceNoDispNoVel.cwiseAbs().maxCoeff();
    bool passed_test = (MaxAbsError <= 0.001);

    if (msglvl >= 2) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Generalized Internal Force - No Displacement, No Velocity = " << std::endl;
        std::cout << InternalForceNoDispNoVel << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Generalized Internal Force - No Displacement, No Velocity (Max Abs Error) = " << MaxAbsError;

        if (passed_test)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");
    }

    return (passed_test);
}

template <typename ElementVersion>
bool ANCFBeamTest<ElementVersion>::GeneralizedInternalForceSmallDispNoVelCheck(int msglvl) {
    // =============================================================================
    //  Check the Internal Force at a given displacement of one node
    //  (some small error is expected depending on the formulation/steps used)
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_InternalForceSmallDispNoVel;
    Expected_InternalForceSmallDispNoVel.resize(3 * NSF, 1);
    if (!load_validation_data("UT_ANCFBeam_3333_MR_IntFrcSmallDispNoVel.txt", Expected_InternalForceSmallDispNoVel))
        return false;

    // Setup the test conditions
    ChVector3d OriginalPos = m_nodeC->GetPos();
    m_nodeC->SetPos(ChVector3d(m_nodeC->GetPos().x(), m_nodeC->GetPos().y(), 0.001));

    ChVectorDynamic<double> InternalForceSmallDispNoVel;
    InternalForceSmallDispNoVel.resize(3 * NSF);
    m_element->ComputeInternalForces(InternalForceSmallDispNoVel);

    // Reset the element conditions back to its original values
    m_nodeC->SetPos(OriginalPos);

    double MaxAbsError = (InternalForceSmallDispNoVel - Expected_InternalForceSmallDispNoVel).cwiseAbs().maxCoeff();
    bool passed_test = (MaxAbsError <= 0.01 * Expected_InternalForceSmallDispNoVel.cwiseAbs().maxCoeff());

    if (msglvl >= 2) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Generalized Internal Force - Small Displacement, No Velocity = " << std::endl;
        std::cout << InternalForceSmallDispNoVel << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Generalized Internal Force - Small Displacement, No Velocity (Max Abs Error) = " << MaxAbsError
                  << " (" << MaxAbsError / Expected_InternalForceSmallDispNoVel.cwiseAbs().maxCoeff() * 100 << "%)";

        if (passed_test)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");
    }

    return (passed_test);
}

template <typename ElementVersion>
bool ANCFBeamTest<ElementVersion>::GeneralizedInternalForceNoDispSmallVelCheck(int msglvl) {
    // =============================================================================
    //  Check the Internal Force at with no nodal displacements, but a given nodal velocity
    //  (some small error is expected depending on the formulation/steps used)
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_InternalForceNoDispSmallVel;
    Expected_InternalForceNoDispSmallVel.resize(3 * NSF, 1);
    if (!load_validation_data("UT_ANCFBeam_3333_MR_IntFrcNoDispSmallVel.txt", Expected_InternalForceNoDispSmallVel))
        return false;

    // Setup the test conditions
    ChVector3d OriginalVel = m_nodeC->GetPosDt();
    m_nodeC->SetPosDt(ChVector3d(0.0, 0.0, 0.001));

    ChVectorDynamic<double> InternalForceNoDispSmallVel;
    InternalForceNoDispSmallVel.resize(3 * NSF);
    m_element->ComputeInternalForces(InternalForceNoDispSmallVel);

    // Reset the element conditions back to its original values
    m_nodeC->SetPosDt(OriginalVel);

    double MaxAbsError = (InternalForceNoDispSmallVel - Expected_InternalForceNoDispSmallVel).cwiseAbs().maxCoeff();
    bool passed_test = (MaxAbsError <= 0.01 * Expected_InternalForceNoDispSmallVel.cwiseAbs().maxCoeff());

    if (msglvl >= 2) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Generalized Internal Force - No Displacement, Small Velocity With Damping = " << std::endl;
        std::cout << InternalForceNoDispSmallVel << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Generalized Internal Force - Small Displacement, No Velocity (Max Abs Error) = " << MaxAbsError
                  << " (" << MaxAbsError / Expected_InternalForceNoDispSmallVel.cwiseAbs().maxCoeff() * 100 << "%)";

        if (passed_test)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");
    }

    return (passed_test);
}

template <typename ElementVersion>
bool ANCFBeamTest<ElementVersion>::JacobianNoDispNoVelNoDampingCheck(int msglvl) {
    // =============================================================================
    //  Check the Jacobian at No Displacement/Velocity - No Damping
    //  (some small error is expected depending on the formulation/steps used)
    //  (The R contribution should be all zeros since damping is not enabled)
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianK_NoDispNoVelNoDamping;
    Expected_JacobianK_NoDispNoVelNoDamping.resize(3 * NSF, 3 * NSF);
    if (!load_validation_data("UT_ANCFBeam_3333_MR_JacNoDispNoVelNoDamping.txt", Expected_JacobianK_NoDispNoVelNoDamping))
        return false;

    //Set the material damping to 0
    double old_mu = m_material->Get_mu();
    m_material->Set_mu(0);

    // Ensure that the internal force is recalculated in case the results are expected
    // by the Jacobian Calculation
    ChVectorDynamic<double> InternalForceNoDispNoVel;
    InternalForceNoDispNoVel.resize(3 * NSF);
    m_element->ComputeInternalForces(InternalForceNoDispNoVel);

    ChMatrixDynamic<double> JacobianK_NoDispNoVelNoDamping;
    JacobianK_NoDispNoVelNoDamping.resize(3 * NSF, 3 * NSF);
    m_element->ComputeKRMmatricesGlobal(JacobianK_NoDispNoVelNoDamping, 1.0, 0.0, 0.0);

    ChMatrixDynamic<double> JacobianR_NoDispNoVelNoDamping;
    JacobianR_NoDispNoVelNoDamping.resize(3 * NSF, 3 * NSF);
    m_element->ComputeKRMmatricesGlobal(JacobianR_NoDispNoVelNoDamping, 0.0, 1.0, 0.0);

    //Reset the material damping
    m_material->Set_mu(old_mu);

    double small_terms_JacK = 1e-4 * Expected_JacobianK_NoDispNoVelNoDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(3 * NSF, 3 * NSF);
    double MaxAbsError_JacR = JacobianR_NoDispNoVelNoDamping.cwiseAbs().maxCoeff();

    for (auto i = 0; i < Expected_JacobianK_NoDispNoVelNoDamping.rows(); i++) {
        for (auto j = 0; j < Expected_JacobianK_NoDispNoVelNoDamping.cols(); j++) {
            if (std::abs(Expected_JacobianK_NoDispNoVelNoDamping(i, j)) < small_terms_JacK) {
                double error =
                    std::abs(JacobianK_NoDispNoVelNoDamping(i, j) - Expected_JacobianK_NoDispNoVelNoDamping(i, j));
                if (error > zeros_max_error_JacK)
                    zeros_max_error_JacK = error;
                // zero the percent error since small difference in small terms can lead to large percentage error that
                // do not really matter and can make it difficult to spot important large percent differences on the
                // large terms
                percent_error_matrix_JacK(i, j) = 0.0;
            } else {
                double percent_error =
                    std::abs((JacobianK_NoDispNoVelNoDamping(i, j) - Expected_JacobianK_NoDispNoVelNoDamping(i, j)) /
                             Expected_JacobianK_NoDispNoVelNoDamping(i, j));
                if (percent_error > max_percent_error_JacK)
                    max_percent_error_JacK = percent_error;
                percent_error_matrix_JacK(i, j) = percent_error;
            }
        }
    }

    // Run the Jacobian Checks
    bool passed_JacobianK_SmallTems =
        zeros_max_error_JacK / JacobianK_NoDispNoVelNoDamping.cwiseAbs().maxCoeff() < 1e-4;
    bool passed_JacobianK_LargeTems = max_percent_error_JacK < Jac_Error;
    bool passed_JacobianR = MaxAbsError_JacR / JacobianK_NoDispNoVelNoDamping.cwiseAbs().maxCoeff() < 1e-6;
    bool passed_tests = passed_JacobianK_LargeTems && passed_JacobianK_SmallTems && passed_JacobianR;

    // Print the results for the K terms (partial derivatives with respect to the nodal coordinates)
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
                  << (JacobianK_NoDispNoVelNoDamping - Expected_JacobianK_NoDispNoVelNoDamping).cwiseAbs().maxCoeff()
                  << std::endl;

        std::cout
            << "Jacobian K Term - No Displacement, No Velocity, No Damping (Max Abs Error - Only Smaller Terms) = "
            << zeros_max_error_JacK << " ( "
            << (zeros_max_error_JacK / Expected_JacobianK_NoDispNoVelNoDamping.cwiseAbs().maxCoeff() * 100)
            << "% of Max Term)";
        if (passed_JacobianK_SmallTems)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");

        std::cout
            << "Jacobian K Term - No Displacement, No Velocity, No Damping (Max Abs % Error - Only Larger Terms) = "
            << max_percent_error_JacK * 100 << "%";
        if (passed_JacobianK_LargeTems)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");
    }

    // Print the results for the R term (partial derivatives with respect to the time derivative of the nodal
    // coordinates)
    if (msglvl >= 2) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Jacobian R Term - No Displacement, No Velocity, No Damping = " << std::endl;
        std::cout << JacobianR_NoDispNoVelNoDamping << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Jacobian R Term - No Displacement, No Velocity, No Damping (Max Abs Error) = "
                  << MaxAbsError_JacR;

        if (passed_JacobianR)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");
    }

    return (passed_tests);
}

template <typename ElementVersion>
bool ANCFBeamTest<ElementVersion>::JacobianSmallDispNoVelNoDampingCheck(int msglvl) {
    // =============================================================================
    //  Check the Jacobian at Small Displacement No Velocity - No Damping
    //  (some small error is expected depending on the formulation/steps used)
    //  (The R contribution should be all zeros since damping is not enabled)
    // ==========================================================================================================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianK_SmallDispNoVelNoDamping;
    Expected_JacobianK_SmallDispNoVelNoDamping.resize(3 * NSF, 3 * NSF);
    if (!load_validation_data("UT_ANCFBeam_3333_MR_JacSmallDispNoVelNoDamping.txt",
                              Expected_JacobianK_SmallDispNoVelNoDamping))
        return false;

    //Set the material damping to 0
    double old_mu = m_material->Get_mu();
    m_material->Set_mu(0);

    // Setup the test conditions
    ChVector3d OriginalPos = m_nodeC->GetPos();
    m_nodeC->SetPos(ChVector3d(m_nodeC->GetPos().x(), m_nodeC->GetPos().y(), 0.001));

    // Ensure that the internal force is recalculated in case the results are expected
    // by the Jacobian Calculation
    ChVectorDynamic<double> InternalForceSmallDispNoVel;
    InternalForceSmallDispNoVel.resize(3 * NSF);
    m_element->ComputeInternalForces(InternalForceSmallDispNoVel);

    ChMatrixDynamic<double> JacobianK_SmallDispNoVelNoDamping;
    JacobianK_SmallDispNoVelNoDamping.resize(3 * NSF, 3 * NSF);
    m_element->ComputeKRMmatricesGlobal(JacobianK_SmallDispNoVelNoDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_SmallDispNoVelNoDamping;
    JacobianR_SmallDispNoVelNoDamping.resize(3 * NSF, 3 * NSF);
    m_element->ComputeKRMmatricesGlobal(JacobianR_SmallDispNoVelNoDamping, 0, 1, 0);

    // Reset the element conditions back to its original values
    m_nodeC->SetPos(OriginalPos);

    //Reset the material damping
    m_material->Set_mu(old_mu);

    double small_terms_JacK = 1e-4 * Expected_JacobianK_SmallDispNoVelNoDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(3 * NSF, 3 * NSF);
    double MaxAbsError_JacR = JacobianR_SmallDispNoVelNoDamping.cwiseAbs().maxCoeff();

    for (auto i = 0; i < Expected_JacobianK_SmallDispNoVelNoDamping.rows(); i++) {
        for (auto j = 0; j < Expected_JacobianK_SmallDispNoVelNoDamping.cols(); j++) {
            if (std::abs(Expected_JacobianK_SmallDispNoVelNoDamping(i, j)) < small_terms_JacK) {
                double error = std::abs(JacobianK_SmallDispNoVelNoDamping(i, j) -
                                        Expected_JacobianK_SmallDispNoVelNoDamping(i, j));
                if (error > zeros_max_error_JacK)
                    zeros_max_error_JacK = error;
                // zero the percent error since small difference in small terms can lead to large percentage error that
                // do not really matter and can make it difficult to spot important large percent differences on the
                // large terms
                percent_error_matrix_JacK(i, j) = 0.0;
            } else {
                double percent_error = std::abs(
                    (JacobianK_SmallDispNoVelNoDamping(i, j) - Expected_JacobianK_SmallDispNoVelNoDamping(i, j)) /
                    Expected_JacobianK_SmallDispNoVelNoDamping(i, j));
                if (percent_error > max_percent_error_JacK)
                    max_percent_error_JacK = percent_error;
                percent_error_matrix_JacK(i, j) = percent_error;
            }
        }
    }

    // Run the Jacobian Checks
    bool passed_JacobianK_SmallTems =
        zeros_max_error_JacK / JacobianK_SmallDispNoVelNoDamping.cwiseAbs().maxCoeff() < 1e-4;
    bool passed_JacobianK_LargeTems = max_percent_error_JacK < Jac_Error;
    bool passed_JacobianR = MaxAbsError_JacR / JacobianK_SmallDispNoVelNoDamping.cwiseAbs().maxCoeff() < 1e-6;
    bool passed_tests = passed_JacobianK_LargeTems && passed_JacobianK_SmallTems && passed_JacobianR;

    // Print the results for the K terms (partial derivatives with respect to the nodal coordinates)
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
        std::cout
            << "Jacobian K Term - Small Displacement, No Velocity, No Damping (Max Abs Error) = "
            << (JacobianK_SmallDispNoVelNoDamping - Expected_JacobianK_SmallDispNoVelNoDamping).cwiseAbs().maxCoeff()
            << std::endl;

        std::cout
            << "Jacobian K Term - Small Displacement, No Velocity, No Damping (Max Abs Error - Only Smaller Terms) = "
            << zeros_max_error_JacK << " ( "
            << (zeros_max_error_JacK / Expected_JacobianK_SmallDispNoVelNoDamping.cwiseAbs().maxCoeff() * 100)
            << "% of Max Term)";
        if (passed_JacobianK_SmallTems)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");

        std::cout
            << "Jacobian K Term - Small Displacement, No Velocity, No Damping (Max Abs % Error - Only Larger Terms) = "
            << max_percent_error_JacK * 100 << "%";
        if (passed_JacobianK_LargeTems)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");
    }

    // Print the results for the R term (partial derivatives with respect to the time derivative of the nodal
    // coordinates)
    if (msglvl >= 2) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Jacobian R Term - Small Displacement, No Velocity, No Damping = " << std::endl;
        std::cout << JacobianR_SmallDispNoVelNoDamping << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Jacobian R Term - Small Displacement, No Velocity, No Damping (Max Abs Error) = "
                  << MaxAbsError_JacR;

        if (passed_JacobianR)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");
    }

    return (passed_tests);
}

template <typename ElementVersion>
bool ANCFBeamTest<ElementVersion>::JacobianNoDispNoVelWithDampingCheck(int msglvl) {
    // =============================================================================
    //  Check the Jacobian at No Displacement/Velocity - With Damping
    //  (some small error is expected depending on the formulation/steps used)
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_Jacobians;
    Expected_Jacobians.resize(2 * 3 * NSF, 3 * NSF);
    if (!load_validation_data("UT_ANCFBeam_3333_MR_JacNoDispNoVelWithDamping.txt", Expected_Jacobians))
        return false;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianK_NoDispNoVelWithDamping;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianR_NoDispNoVelWithDamping;
    Expected_JacobianK_NoDispNoVelWithDamping =
        Expected_Jacobians.block(0, 0, Expected_Jacobians.cols(), Expected_Jacobians.cols());
    Expected_JacobianR_NoDispNoVelWithDamping =
        Expected_Jacobians.block(Expected_Jacobians.cols(), 0, Expected_Jacobians.cols(), Expected_Jacobians.cols());

    // Ensure that the internal force is recalculated in case the results are expected
    // by the Jacobian Calculation
    ChVectorDynamic<double> InternalForceNoDispNoVel;
    InternalForceNoDispNoVel.resize(3 * NSF);
    m_element->ComputeInternalForces(InternalForceNoDispNoVel);

    ChMatrixDynamic<double> JacobianK_NoDispNoVelWithDamping;
    JacobianK_NoDispNoVelWithDamping.resize(3 * NSF, 3 * NSF);
    m_element->ComputeKRMmatricesGlobal(JacobianK_NoDispNoVelWithDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_NoDispNoVelWithDamping;
    JacobianR_NoDispNoVelWithDamping.resize(3 * NSF, 3 * NSF);
    m_element->ComputeKRMmatricesGlobal(JacobianR_NoDispNoVelWithDamping, 0, 1, 0);

    double small_terms_JacK = 1e-4 * Expected_JacobianK_NoDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(3 * NSF, 3 * NSF);

    double small_terms_JacR = 1e-4 * Expected_JacobianR_NoDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacR = 0;
    double max_percent_error_JacR = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacR;
    percent_error_matrix_JacR.resize(3 * NSF, 3 * NSF);

    for (auto i = 0; i < Expected_JacobianK_NoDispNoVelWithDamping.rows(); i++) {
        for (auto j = 0; j < Expected_JacobianK_NoDispNoVelWithDamping.cols(); j++) {
            if (std::abs(Expected_JacobianK_NoDispNoVelWithDamping(i, j)) < small_terms_JacK) {
                double error =
                    std::abs(JacobianK_NoDispNoVelWithDamping(i, j) - Expected_JacobianK_NoDispNoVelWithDamping(i, j));
                if (error > zeros_max_error_JacK)
                    zeros_max_error_JacK = error;
                // zero the percent error since small difference in small terms can lead to large percentage error that
                // do not really matter and can make it difficult to spot important large percent differences on the
                // large terms
                percent_error_matrix_JacK(i, j) = 0.0;
            } else {
                double percent_error = std::abs(
                    (JacobianK_NoDispNoVelWithDamping(i, j) - Expected_JacobianK_NoDispNoVelWithDamping(i, j)) /
                    Expected_JacobianK_NoDispNoVelWithDamping(i, j));
                if (percent_error > max_percent_error_JacK)
                    max_percent_error_JacK = percent_error;
                percent_error_matrix_JacK(i, j) = percent_error;
            }

            if (std::abs(Expected_JacobianR_NoDispNoVelWithDamping(i, j)) < small_terms_JacR) {
                double error =
                    std::abs(JacobianR_NoDispNoVelWithDamping(i, j) - Expected_JacobianR_NoDispNoVelWithDamping(i, j));
                if (error > zeros_max_error_JacR)
                    zeros_max_error_JacR = error;
                // zero the percent error since small difference in small terms can lead to large percentage error that
                // do not really matter and can make it difficult to spot important large percent differences on the
                // large terms
                percent_error_matrix_JacR(i, j) = 0.0;
            } else {
                double percent_error = std::abs(
                    (JacobianR_NoDispNoVelWithDamping(i, j) - Expected_JacobianR_NoDispNoVelWithDamping(i, j)) /
                    Expected_JacobianR_NoDispNoVelWithDamping(i, j));
                if (percent_error > max_percent_error_JacR)
                    max_percent_error_JacR = percent_error;
                percent_error_matrix_JacR(i, j) = percent_error;
            }
        }
    }

    // Run the Jacobian Checks
    bool passed_JacobianK_SmallTems =
        zeros_max_error_JacK / JacobianK_NoDispNoVelWithDamping.cwiseAbs().maxCoeff() < 1e-4;
    bool passed_JacobianK_LargeTems = max_percent_error_JacK < Jac_Error;
    bool passed_JacobianR_SmallTems =
        zeros_max_error_JacR / JacobianR_NoDispNoVelWithDamping.cwiseAbs().maxCoeff() < 1e-4;
    bool passed_JacobianR_LargeTems = max_percent_error_JacR < Jac_Error;
    bool passed_tests = passed_JacobianK_LargeTems && passed_JacobianK_SmallTems && passed_JacobianR_SmallTems &&
                        passed_JacobianR_LargeTems;

    // Print the results for the K terms (partial derivatives with respect to the nodal coordinates)
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
        std::cout
            << "Jacobian K Term - No Displacement, No Velocity, With Damping (Max Abs Error) = "
            << (JacobianK_NoDispNoVelWithDamping - Expected_JacobianK_NoDispNoVelWithDamping).cwiseAbs().maxCoeff()
            << std::endl;

        std::cout
            << "Jacobian K Term - No Displacement, No Velocity, With Damping (Max Abs Error - Only Smaller Terms) = "
            << zeros_max_error_JacK << " ( "
            << (zeros_max_error_JacK / Expected_JacobianK_NoDispNoVelWithDamping.cwiseAbs().maxCoeff() * 100)
            << "% of Max Term)";
        if (passed_JacobianK_SmallTems)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");

        std::cout
            << "Jacobian K Term - No Displacement, No Velocity, With Damping (Max Abs % Error - Only Larger Terms) = "
            << max_percent_error_JacK * 100 << "%";
        if (passed_JacobianK_LargeTems)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");
    }

    // Print the results for the R term (partial derivatives with respect to the time derivative of the nodal
    // coordinates)
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
        std::cout
            << "Jacobian R Term - No Displacement, No Velocity, With Damping (Max Abs Error) = "
            << (JacobianR_NoDispNoVelWithDamping - Expected_JacobianR_NoDispNoVelWithDamping).cwiseAbs().maxCoeff()
            << std::endl;

        std::cout
            << "Jacobian R Term - No Displacement, No Velocity, With Damping (Max Abs Error - Only Smaller Terms) = "
            << zeros_max_error_JacR << " ( "
            << (zeros_max_error_JacK / Expected_JacobianR_NoDispNoVelWithDamping.cwiseAbs().maxCoeff() * 100)
            << "% of Max Term)";
        if (passed_JacobianR_SmallTems)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");

        std::cout
            << "Jacobian R Term - No Displacement, No Velocity, With Damping (Max Abs % Error - Only Larger Terms) = "
            << max_percent_error_JacR * 100 << "%";
        if (passed_JacobianR_LargeTems)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");
    }

    return (passed_tests);
}

template <typename ElementVersion>
bool ANCFBeamTest<ElementVersion>::JacobianSmallDispNoVelWithDampingCheck(int msglvl) {
    // =============================================================================
    //  Check the Jacobian at Small Displacement No Velocity - With Damping
    //  (some small error is expected depending on the formulation/steps used)
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_Jacobians;
    Expected_Jacobians.resize(2 * 3 * NSF, 3 * NSF);
    if (!load_validation_data("UT_ANCFBeam_3333_MR_JacSmallDispNoVelWithDamping.txt", Expected_Jacobians))
        return false;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianK_SmallDispNoVelWithDamping;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianR_SmallDispNoVelWithDamping;
    Expected_JacobianK_SmallDispNoVelWithDamping =
        Expected_Jacobians.block(0, 0, Expected_Jacobians.cols(), Expected_Jacobians.cols());
    Expected_JacobianR_SmallDispNoVelWithDamping =
        Expected_Jacobians.block(Expected_Jacobians.cols(), 0, Expected_Jacobians.cols(), Expected_Jacobians.cols());

    // Setup the test conditions
    ChVector3d OriginalPos = m_nodeC->GetPos();
    m_nodeC->SetPos(ChVector3d(m_nodeC->GetPos().x(), m_nodeC->GetPos().y(), 0.001));

    // Ensure that the internal force is recalculated in case the results are expected
    // by the Jacobian Calculation
    ChVectorDynamic<double> InternalForceSmallDispNoVel;
    InternalForceSmallDispNoVel.resize(3 * NSF);
    m_element->ComputeInternalForces(InternalForceSmallDispNoVel);

    ChMatrixDynamic<double> JacobianK_SmallDispNoVelWithDamping;
    JacobianK_SmallDispNoVelWithDamping.resize(3 * NSF, 3 * NSF);
    m_element->ComputeKRMmatricesGlobal(JacobianK_SmallDispNoVelWithDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_SmallDispNoVelWithDamping;
    JacobianR_SmallDispNoVelWithDamping.resize(3 * NSF, 3 * NSF);
    m_element->ComputeKRMmatricesGlobal(JacobianR_SmallDispNoVelWithDamping, 0, 1, 0);

    // Reset the element conditions back to its original values
    m_nodeC->SetPos(OriginalPos);

    double small_terms_JacK = 1e-4 * Expected_JacobianK_SmallDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(3 * NSF, 3 * NSF);

    double small_terms_JacR = 1e-4 * Expected_JacobianR_SmallDispNoVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacR = 0;
    double max_percent_error_JacR = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacR;
    percent_error_matrix_JacR.resize(3 * NSF, 3 * NSF);

    for (auto i = 0; i < Expected_JacobianK_SmallDispNoVelWithDamping.rows(); i++) {
        for (auto j = 0; j < Expected_JacobianK_SmallDispNoVelWithDamping.cols(); j++) {
            if (std::abs(Expected_JacobianK_SmallDispNoVelWithDamping(i, j)) < small_terms_JacK) {
                double error = std::abs(JacobianK_SmallDispNoVelWithDamping(i, j) -
                                        Expected_JacobianK_SmallDispNoVelWithDamping(i, j));
                if (error > zeros_max_error_JacK)
                    zeros_max_error_JacK = error;
                // zero the percent error since small difference in small terms can lead to large percentage error that
                // do not really matter and can make it difficult to spot important large percent differences on the
                // large terms
                percent_error_matrix_JacK(i, j) = 0.0;
            } else {
                double percent_error = std::abs(
                    (JacobianK_SmallDispNoVelWithDamping(i, j) - Expected_JacobianK_SmallDispNoVelWithDamping(i, j)) /
                    Expected_JacobianK_SmallDispNoVelWithDamping(i, j));
                if (percent_error > max_percent_error_JacK)
                    max_percent_error_JacK = percent_error;
                percent_error_matrix_JacK(i, j) = percent_error;
            }

            if (std::abs(Expected_JacobianR_SmallDispNoVelWithDamping(i, j)) < small_terms_JacR) {
                double error = std::abs(JacobianR_SmallDispNoVelWithDamping(i, j) -
                                        Expected_JacobianR_SmallDispNoVelWithDamping(i, j));
                if (error > zeros_max_error_JacR)
                    zeros_max_error_JacR = error;
                // zero the percent error since small difference in small terms can lead to large percentage error that
                // do not really matter and can make it difficult to spot important large percent differences on the
                // large terms
                percent_error_matrix_JacR(i, j) = 0.0;
            } else {
                double percent_error = std::abs(
                    (JacobianR_SmallDispNoVelWithDamping(i, j) - Expected_JacobianR_SmallDispNoVelWithDamping(i, j)) /
                    Expected_JacobianR_SmallDispNoVelWithDamping(i, j));
                if (percent_error > max_percent_error_JacR)
                    max_percent_error_JacR = percent_error;
                percent_error_matrix_JacR(i, j) = percent_error;
            }
        }
    }

    // Run the Jacobian Checks
    bool passed_JacobianK_SmallTems =
        zeros_max_error_JacK / JacobianK_SmallDispNoVelWithDamping.cwiseAbs().maxCoeff() < 1e-4;
    bool passed_JacobianK_LargeTems = max_percent_error_JacK < Jac_Error;
    bool passed_JacobianR_SmallTems =
        zeros_max_error_JacR / JacobianR_SmallDispNoVelWithDamping.cwiseAbs().maxCoeff() < 1e-4;
    bool passed_JacobianR_LargeTems = max_percent_error_JacR < Jac_Error;
    bool passed_tests = passed_JacobianK_LargeTems && passed_JacobianK_SmallTems && passed_JacobianR_SmallTems &&
                        passed_JacobianR_LargeTems;

    // Print the results for the K terms (partial derivatives with respect to the nodal coordinates)
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
                  << (JacobianK_SmallDispNoVelWithDamping - Expected_JacobianK_SmallDispNoVelWithDamping)
                         .cwiseAbs()
                         .maxCoeff()
                  << std::endl;

        std::cout
            << "Jacobian K Term - Small Displacement, No Velocity, With Damping (Max Abs Error - Only Smaller Terms) = "
            << zeros_max_error_JacK << " ( "
            << (zeros_max_error_JacK / Expected_JacobianK_SmallDispNoVelWithDamping.cwiseAbs().maxCoeff() * 100)
            << "% of Max Term)";
        if (passed_JacobianK_SmallTems)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");

        std::cout << "Jacobian K Term - Small Displacement, No Velocity, With Damping (Max Abs % Error - Only Larger "
                     "Terms) = "
                  << max_percent_error_JacK * 100 << "%";
        if (passed_JacobianK_LargeTems)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");
    }

    // Print the results for the R term (partial derivatives with respect to the time derivative of the nodal
    // coordinates)
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
                  << (JacobianR_SmallDispNoVelWithDamping - Expected_JacobianR_SmallDispNoVelWithDamping)
                         .cwiseAbs()
                         .maxCoeff()
                  << std::endl;

        std::cout
            << "Jacobian R Term - Small Displacement, No Velocity, With Damping (Max Abs Error - Only Smaller Terms) = "
            << zeros_max_error_JacR << " ( "
            << (zeros_max_error_JacK / Expected_JacobianR_SmallDispNoVelWithDamping.cwiseAbs().maxCoeff() * 100)
            << "% of Max Term)";
        if (passed_JacobianR_SmallTems)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");

        std::cout << "Jacobian R Term - Small Displacement, No Velocity, With Damping (Max Abs % Error - Only Larger "
                     "Terms) = "
                  << max_percent_error_JacR * 100 << "%";
        if (passed_JacobianR_LargeTems)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");
    }

    return (passed_tests);
}

template <typename ElementVersion>
bool ANCFBeamTest<ElementVersion>::JacobianNoDispSmallVelWithDampingCheck(int msglvl) {
    // =============================================================================
    //  Check the Jacobian at No Displacement Small Velocity - With Damping
    //  (some small error is expected depending on the formulation/steps used)
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_Jacobians;
    Expected_Jacobians.resize(2 * 3 * NSF, 3 * NSF);
    if (!load_validation_data("UT_ANCFBeam_3333_MR_JacNoDispSmallVelWithDamping.txt", Expected_Jacobians))
        return false;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianK_NoDispSmallVelWithDamping;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Expected_JacobianR_NoDispSmallVelWithDamping;
    Expected_JacobianK_NoDispSmallVelWithDamping =
        Expected_Jacobians.block(0, 0, Expected_Jacobians.cols(), Expected_Jacobians.cols());
    Expected_JacobianR_NoDispSmallVelWithDamping =
        Expected_Jacobians.block(Expected_Jacobians.cols(), 0, Expected_Jacobians.cols(), Expected_Jacobians.cols());

    // Setup the test conditions
    ChVector3d OriginalVel = m_nodeC->GetPosDt();
    m_nodeC->SetPosDt(ChVector3d(0.0, 0.0, 0.001));

    // Ensure that the internal force is recalculated in case the results are expected
    // by the Jacobian Calculation
    ChVectorDynamic<double> InternalForceNoDispSmallVel;
    InternalForceNoDispSmallVel.resize(3 * NSF);
    m_element->ComputeInternalForces(InternalForceNoDispSmallVel);

    ChMatrixDynamic<double> JacobianK_NoDispSmallVelWithDamping;
    JacobianK_NoDispSmallVelWithDamping.resize(3 * NSF, 3 * NSF);
    m_element->ComputeKRMmatricesGlobal(JacobianK_NoDispSmallVelWithDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_NoDispSmallVelWithDamping;
    JacobianR_NoDispSmallVelWithDamping.resize(3 * NSF, 3 * NSF);
    m_element->ComputeKRMmatricesGlobal(JacobianR_NoDispSmallVelWithDamping, 0, 1, 0);

    // Reset the element conditions back to its original values
    m_nodeC->SetPosDt(OriginalVel);

    double small_terms_JacK = 1e-4 * Expected_JacobianK_NoDispSmallVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacK = 0;
    double max_percent_error_JacK = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacK;
    percent_error_matrix_JacK.resize(3 * NSF, 3 * NSF);

    double small_terms_JacR = 1e-4 * Expected_JacobianR_NoDispSmallVelWithDamping.cwiseAbs().maxCoeff();
    double zeros_max_error_JacR = 0;
    double max_percent_error_JacR = 0;
    ChMatrixDynamic<double> percent_error_matrix_JacR;
    percent_error_matrix_JacR.resize(3 * NSF, 3 * NSF);

    for (auto i = 0; i < Expected_JacobianK_NoDispSmallVelWithDamping.rows(); i++) {
        for (auto j = 0; j < Expected_JacobianK_NoDispSmallVelWithDamping.cols(); j++) {
            if (std::abs(Expected_JacobianK_NoDispSmallVelWithDamping(i, j)) < small_terms_JacK) {
                double error = std::abs(JacobianK_NoDispSmallVelWithDamping(i, j) -
                                        Expected_JacobianK_NoDispSmallVelWithDamping(i, j));
                if (error > zeros_max_error_JacK)
                    zeros_max_error_JacK = error;
                // zero the percent error since small difference in small terms can lead to large percentage error that
                // do not really matter and can make it difficult to spot important large percent differences on the
                // large terms
                percent_error_matrix_JacK(i, j) = 0.0;
            } else {
                double percent_error = std::abs(
                    (JacobianK_NoDispSmallVelWithDamping(i, j) - Expected_JacobianK_NoDispSmallVelWithDamping(i, j)) /
                    Expected_JacobianK_NoDispSmallVelWithDamping(i, j));
                if (percent_error > max_percent_error_JacK)
                    max_percent_error_JacK = percent_error;
                percent_error_matrix_JacK(i, j) = percent_error;
            }

            if (std::abs(Expected_JacobianR_NoDispSmallVelWithDamping(i, j)) < small_terms_JacR) {
                double error = std::abs(JacobianR_NoDispSmallVelWithDamping(i, j) -
                                        Expected_JacobianR_NoDispSmallVelWithDamping(i, j));
                if (error > zeros_max_error_JacR)
                    zeros_max_error_JacR = error;
                // zero the percent error since small difference in small terms can lead to large percentage error that
                // do not really matter and can make it difficult to spot important large percent differences on the
                // large terms
                percent_error_matrix_JacR(i, j) = 0.0;
            } else {
                double percent_error = std::abs(
                    (JacobianR_NoDispSmallVelWithDamping(i, j) - Expected_JacobianR_NoDispSmallVelWithDamping(i, j)) /
                    Expected_JacobianR_NoDispSmallVelWithDamping(i, j));
                if (percent_error > max_percent_error_JacR)
                    max_percent_error_JacR = percent_error;
                percent_error_matrix_JacR(i, j) = percent_error;
            }
        }
    }

    // Run the Jacobian Checks
    bool passed_JacobianK_SmallTems =
        zeros_max_error_JacK / JacobianK_NoDispSmallVelWithDamping.cwiseAbs().maxCoeff() < 1e-4;
    bool passed_JacobianK_LargeTems = max_percent_error_JacK < Jac_Error;
    bool passed_JacobianR_SmallTems =
        zeros_max_error_JacR / JacobianR_NoDispSmallVelWithDamping.cwiseAbs().maxCoeff() < 1e-4;
    bool passed_JacobianR_LargeTems = max_percent_error_JacR < Jac_Error;
    bool passed_tests = passed_JacobianK_LargeTems && passed_JacobianK_SmallTems && passed_JacobianR_SmallTems &&
                        passed_JacobianR_LargeTems;

    // Print the results for the K terms (partial derivatives with respect to the nodal coordinates)
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
                  << (JacobianK_NoDispSmallVelWithDamping - Expected_JacobianK_NoDispSmallVelWithDamping)
                         .cwiseAbs()
                         .maxCoeff()
                  << std::endl;

        std::cout
            << "Jacobian K Term - No Displacement, Small Velocity, With Damping (Max Abs Error - Only Smaller Terms) = "
            << zeros_max_error_JacK << " ( "
            << (zeros_max_error_JacK / Expected_JacobianK_NoDispSmallVelWithDamping.cwiseAbs().maxCoeff() * 100)
            << "% of Max Term)";
        if (passed_JacobianK_SmallTems)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");

        std::cout << "Jacobian K Term - No Displacement, Small Velocity, With Damping (Max Abs % Error - Only Larger "
                     "Terms) = "
                  << max_percent_error_JacK * 100 << "%";
        if (passed_JacobianK_LargeTems)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");
    }

    // Print the results for the R term (partial derivatives with respect to the time derivative of the nodal
    // coordinates)
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
                  << (JacobianR_NoDispSmallVelWithDamping - Expected_JacobianR_NoDispSmallVelWithDamping)
                         .cwiseAbs()
                         .maxCoeff()
                  << std::endl;

        std::cout
            << "Jacobian R Term - No Displacement, Small Velocity, With Damping (Max Abs Error - Only Smaller Terms) = "
            << zeros_max_error_JacR << " ( "
            << (zeros_max_error_JacK / Expected_JacobianR_NoDispSmallVelWithDamping.cwiseAbs().maxCoeff() * 100)
            << "% of Max Term)";
        if (passed_JacobianR_SmallTems)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");

        std::cout << "Jacobian R Term - No Displacement, Small Velocity, With Damping (Max Abs % Error - Only Larger "
                     "Terms) = "
                  << max_percent_error_JacR * 100 << "%";
        if (passed_JacobianR_LargeTems)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");
    }

    return (passed_tests);
}

template <typename ElementVersion>
bool ANCFBeamTest<ElementVersion>::AxialDisplacementCheck(int msglvl) {
    // =============================================================================
    //  Check the axial displacement of a beam compared to the analytical result
    //  (some small error is expected based on assumptions and boundary conditions)
    // =============================================================================

    auto system = new ChSystemSMC();
    // Set gravity to 0 since this is a statics test against an analytical solution
    system->SetGravitationalAcceleration(ChVector3d(0, 0, 0));

    auto solver = chrono_types::make_shared<ChSolverSparseLU>();
    solver->UseSparsityPatternLearner(true);
    solver->LockSparsityPattern(true);
    solver->SetVerbose(false);
    system->SetSolver(solver);

    // Set up integrator
    system->SetTimestepperType(ChTimestepper::Type::HHT);
    auto integrator = std::static_pointer_cast<ChTimestepperHHT>(system->GetTimestepper());
    integrator->SetAlpha(-0.2);
    integrator->SetMaxIters(100);
    integrator->SetAbsTolerances(1e-5);
    integrator->SetVerbose(false);
    integrator->SetModifiedNewton(true);

    // Mesh properties
    int num_elements = 32;
    double length = 1;      // m
    double width = 0.01;    // m (square cross-section)
    double height = width;  // m (square cross-section)
    double rho = 7850; //kg/m^3
    double c10 = 0.8e6;  ///Pa
    double c01 = 0.2e6;    ///Pa
    double k = 1e9;    ///Pa
    double mu = 0.01 * 6 * (c10 + c01);     ///< Viscosity Coefficient

    double G = 2 * (c10 + c01); // Pa
    double E = 9 * k*G / (3 * k + G);  // Pa
    double nu = (3 * k - 2 * G) / (2 * (3 * k + G));

    auto material = chrono_types::make_shared<ChMaterialBeamANCF_MR>(rho, c10, c01, k, mu);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    system->Add(mesh);

    // Populate the mesh container with a the nodes and elements for the meshed beam
    int num_nodes = (2 * num_elements) + 1;
    double dx = length / (num_nodes - 1);

    // Setup beam cross section gradients to initially align with the global y and z directions
    ChVector3d dir1(0, 1, 0);
    ChVector3d dir2(0, 0, 1);

    // Create the first node and fix it completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector3d(0, 0, 0.0), dir1, dir2);
    nodeA->SetFixed(true);
    mesh->AddNode(nodeA);

    auto elementlast = chrono_types::make_shared<ElementVersion>();
    std::shared_ptr<ChNodeFEAxyzDD> nodeEndPoint;

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector3d(dx * (2 * i), 0, 0), dir1, dir2);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector3d(dx * (2 * i - 1), 0, 0), dir1, dir2);
        mesh->AddNode(nodeB);
        mesh->AddNode(nodeC);

        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC);
        element->SetDimensions(2 * dx, width, width);
        element->SetMaterial(material);
        mesh->AddElement(element);

        nodeA = nodeB;
        elementlast = element;
    }

    nodeEndPoint = nodeA;

    mesh->SetAutomaticGravity(
        false);  // Turn off the gravity at the mesh level since it is not applied in this test.  This step is not
                 // required since the acceleration due to gravity was already set to all zeros.

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

    auto loader = chrono_types::make_shared<MyLoaderTimeDependentTipLoad>(elementlast);
    loader->auxsystem = system;   // initialize auxiliary data of the loader, if needed
    loader->SetApplication(1.0);  // specify application point
    auto load = chrono_types::make_shared<ChLoad>(loader);
    loadcontainer->Add(load);  // add the load to the load container.

    // Find the nonlinear static solution for the system
    system->DoStaticNonlinear(50);

    // Calculate the axial displacement of the end of the ANCF beam mesh
    ChVector3d point;
    ChQuaternion<> rot;
    elementlast->EvaluateSectionFrame(1.0, point, rot);

    // For Analytical Formula, see a mechanics of materials textbook (delta = (P*L)/(A*E))
    double Displacement_Theory = (TIP_FORCE * length) / (width * height * E);
    double Displacement_Model = point.x() - length;
    ChVector3d Tip_Angles = rot.GetCardanAnglesXYZ();

    double Percent_Error = (Displacement_Model - Displacement_Theory) / Displacement_Theory * 100;

    bool passed_displacement = abs(Percent_Error) < 6.0;
    bool passed_angles = (abs(Tip_Angles.x() * CH_RAD_TO_DEG) < 0.001) &&
                         (abs(Tip_Angles.y() * CH_RAD_TO_DEG) < 0.001) &&
                         (abs(Tip_Angles.z() * CH_RAD_TO_DEG) < 0.001);
    bool passed_tests = passed_displacement && passed_angles;

    if (msglvl >= 2) {
        std::cout << "Axial Pull Test - ANCF Tip Position: " << point << "m" << std::endl;
        std::cout << "Axial Pull Test - ANCF Tip Displacement: " << Displacement_Model << "m" << std::endl;
        std::cout << "Axial Pull Test - Analytical Tip Displacement: " << Displacement_Theory << "m" << std::endl;
        std::cout << "Axial Pull Test - ANCF Tip Angles: (" << Tip_Angles.x() * CH_RAD_TO_DEG << ", "
                  << Tip_Angles.y() * CH_RAD_TO_DEG << ", " << Tip_Angles.z() * CH_RAD_TO_DEG << ")deg"
                  << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Axial Pull Test - Tip Displacement Check (Percent Error less than 6%) = " << Percent_Error << "%";
        if (passed_displacement)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");

        std::cout << "Axial Pull Test - Angular misalignment Checks (all angles less than 0.001 deg)";
        if (passed_angles)
            print_green(" - Test PASSED\n\n");
        else
            print_red(" - Test FAILED\n\n");
    }

    return (passed_tests);
}

template <typename ElementVersion>
bool ANCFBeamTest<ElementVersion>::CantileverTipLoadCheck(int msglvl) {
    // =============================================================================
    //  Check the vertical tip displacement of a cantilever beam with a tip load compared to the analytical result
    //  (some small error is expected based on assumptions and boundary conditions)
    // =============================================================================

    auto system = new ChSystemSMC();
    // Set gravity to 0 since this is a statics test against an analytical solution
    system->SetGravitationalAcceleration(ChVector3d(0, 0, 0));

    auto solver = chrono_types::make_shared<ChSolverSparseLU>();
    solver->UseSparsityPatternLearner(true);
    solver->LockSparsityPattern(true);
    solver->SetVerbose(false);
    system->SetSolver(solver);

    // Set up integrator
    system->SetTimestepperType(ChTimestepper::Type::HHT);
    auto integrator = std::static_pointer_cast<ChTimestepperHHT>(system->GetTimestepper());
    integrator->SetAlpha(-0.2);
    integrator->SetMaxIters(100);
    integrator->SetAbsTolerances(1e-5);
    integrator->SetVerbose(false);
    integrator->SetModifiedNewton(true);

    // Mesh properties
    int num_elements = 32;
    double length = 1;      // m
    double width = 0.1;    // m (square cross-section)
    double height = width;  // m (square cross-section)
    double rho = 7850; //kg/m^3
    double c10 = 0.8e6;  ///Pa
    double c01 = 0.2e6;    ///Pa
    double k = 1e9;    ///Pa
    double mu = 0.01 * 6 * (c10 + c01);     ///< Viscosity Coefficient

    double G = 2 * (c10 + c01); // Pa
    double E = 9 * k*G / (3 * k + G);  // Pa
    double nu = (3 * k - 2 * G) / (2 * (3 * k + G));

    auto material = chrono_types::make_shared<ChMaterialBeamANCF_MR>(rho, c10, c01, k, mu);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    system->Add(mesh);

    // Populate the mesh container with a the nodes and elements for the meshed beam
    int num_nodes = (2 * num_elements) + 1;
    double dx = length / (num_nodes - 1);

    // Setup beam cross section gradients to initially align with the global y and z directions
    ChVector3d dir1(0, 1, 0);
    ChVector3d dir2(0, 0, 1);

    // Create the first node and fix it completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector3d(0, 0, 0.0), dir1, dir2);
    nodeA->SetFixed(true);
    mesh->AddNode(nodeA);

    auto elementlast = chrono_types::make_shared<ElementVersion>();
    std::shared_ptr<ChNodeFEAxyzDD> nodeEndPoint;

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector3d(dx * (2 * i), 0, 0), dir1, dir2);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector3d(dx * (2 * i - 1), 0, 0), dir1, dir2);
        mesh->AddNode(nodeB);
        mesh->AddNode(nodeC);

        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC);
        element->SetDimensions(2 * dx, width, width);
        element->SetMaterial(material);
        mesh->AddElement(element);

        nodeA = nodeB;
        elementlast = element;
    }

    nodeEndPoint = nodeA;

    mesh->SetAutomaticGravity(
        false);  // Turn off the gravity at the mesh level since it is not applied in this test.  This step is not
                 // required since the acceleration due to gravity was already set to all zeros.

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

    auto loader = chrono_types::make_shared<MyLoaderTimeDependentTipLoad>(elementlast);
    loader->auxsystem = system;   // initialize auxiliary data of the loader, if needed
    loader->SetApplication(1.0);  // specify application point
    auto load = chrono_types::make_shared<ChLoad>(loader);
    loadcontainer->Add(load);  // add the load to the load container.

    // Find the static solution for the system (final displacement)
    system->DoStaticLinear();
    system->DoStaticLinear();
    system->DoStaticLinear();

    // Calculate the displacement of the end of the ANCF beam mesh
    ChVector3d point;
    ChQuaternion<> rot;
    elementlast->EvaluateSectionFrame(1, point, rot);

    // For Analytical Formula, see a mechanics of materials textbook (delta = (P*L^3)/(3*E*I))
    double I = 1.0 / 12.0 * width * std::pow(height, 3);
    double Displacement_Theory = (TIP_FORCE * std::pow(length, 3)) / (3.0 * E * I);
    double Displacement_Model = point.z();
    ChVector3d Tip_Angles = rot.GetCardanAnglesXYZ();

    double Percent_Error = (Displacement_Model - Displacement_Theory) / Displacement_Theory * 100.0;

    bool passed_displacement = abs(Percent_Error) < 2.0;
    // check the off-axis angles which should be zeros
    bool passed_angles =
        (abs(Tip_Angles.x() * CH_RAD_TO_DEG) < 0.001) && (abs(Tip_Angles.z() * CH_RAD_TO_DEG) < 0.001);
    bool passed_tests = passed_displacement && passed_angles;

    if (msglvl >= 2) {
        std::cout << "Cantilever Beam (Tip Load) - ANCF Tip Position: " << point << "m" << std::endl;
        std::cout << "Cantilever Beam (Tip Load) - ANCF Tip Displacement: " << Displacement_Model << "m" << std::endl;
        std::cout << "Cantilever Beam (Tip Load) - Analytical Tip Displacement: " << Displacement_Theory << "m"
                  << std::endl;
        std::cout << "Cantilever Beam (Tip Load) - ANCF Tip Angles: (" << Tip_Angles.x() * CH_RAD_TO_DEG << ", "
                  << Tip_Angles.y() * CH_RAD_TO_DEG << ", " << Tip_Angles.z() * CH_RAD_TO_DEG << ")deg"
                  << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Cantilever Beam (Tip Load) - Tip Displacement Check (Percent Error less than 2%) = "
                  << Percent_Error << "%";
        if (passed_displacement)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");

        std::cout
            << "Cantilever Beam (Tip Load) - Off-axis Angular misalignment Checks (all angles less than 0.001 deg)";
        if (passed_angles)
            print_green(" - Test PASSED\n\n");
        else
            print_red(" - Test FAILED\n\n");
    }

    return (passed_tests);
}

template <typename ElementVersion>
bool ANCFBeamTest<ElementVersion>::CantileverGravityCheck(int msglvl) {
    // =============================================================================
    //  Check the vertical tip displacement of a cantilever beam with a uniform gravity load compared to the analytical
    //  result (some small error is expected based on assumptions and boundary conditions)
    // =============================================================================

    auto system = new ChSystemSMC();
    double g = -9.80665;
    system->SetGravitationalAcceleration(ChVector3d(0, 0, g));

    auto solver = chrono_types::make_shared<ChSolverSparseLU>();
    solver->UseSparsityPatternLearner(true);
    solver->LockSparsityPattern(true);
    solver->SetVerbose(false);
    system->SetSolver(solver);

    // Set up integrator
    system->SetTimestepperType(ChTimestepper::Type::HHT);
    auto integrator = std::static_pointer_cast<ChTimestepperHHT>(system->GetTimestepper());
    integrator->SetAlpha(-0.2);
    integrator->SetMaxIters(100);
    integrator->SetAbsTolerances(1e-5);
    integrator->SetVerbose(false);
    integrator->SetModifiedNewton(true);

    // Mesh properties
    int num_elements = 32;
    double length = 1;      // m
    double width = 0.1;    // m (square cross-section)
    double height = width;  // m (square cross-section)
    double rho = 7850*0.001; //kg/m^3
    double c10 = 0.8e6;  ///Pa
    double c01 = 0.2e6;    ///Pa
    double k = 1e9;    ///Pa
    double mu = 0.01 * 6 * (c10 + c01);     ///< Viscosity Coefficient

    double G = 2 * (c10 + c01); // Pa
    double E = 9 * k*G / (3 * k + G);  // Pa
    double nu = (3 * k - 2 * G) / (2 * (3 * k + G));

    auto material = chrono_types::make_shared<ChMaterialBeamANCF_MR>(rho, c10, c01, k, mu);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    system->Add(mesh);

    // Populate the mesh container with a the nodes and elements for the meshed beam
    int num_nodes = (2 * num_elements) + 1;
    double dx = length / (num_nodes - 1);

    // Setup beam cross section gradients to initially align with the global y and z directions
    ChVector3d dir1(0, 1, 0);
    ChVector3d dir2(0, 0, 1);

    // Create the first node and fix it completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector3d(0, 0, 0.0), dir1, dir2);
    nodeA->SetFixed(true);
    mesh->AddNode(nodeA);

    auto elementlast = chrono_types::make_shared<ElementVersion>();
    std::shared_ptr<ChNodeFEAxyzDD> nodeEndPoint;

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector3d(dx * (2 * i), 0, 0), dir1, dir2);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector3d(dx * (2 * i - 1), 0, 0), dir1, dir2);
        mesh->AddNode(nodeB);
        mesh->AddNode(nodeC);

        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC);
        element->SetDimensions(2 * dx, width, width);
        element->SetMaterial(material);
        mesh->AddElement(element);

        nodeA = nodeB;
        elementlast = element;
    }

    nodeEndPoint = nodeA;

    // Find the static solution for the system (final displacement)
    system->DoStaticLinear();
    system->DoStaticLinear();
    system->DoStaticLinear();
    system->DoStaticLinear();

    // Calculate the displacement of the end of the ANCF beam mesh
    ChVector3d point;
    ChQuaternion<> rot;
    elementlast->EvaluateSectionFrame(1, point, rot);

    // For Analytical Formula, see a mechanics of materials textbook (delta = (q*L^4)/(8*E*I))
    double I = 1.0 / 12.0 * width * std::pow(height, 3);
    double Displacement_Theory = (rho * width * height * g * std::pow(length, 4)) / (8.0 * E * I);
    double Displacement_Model = point.z();
    ChVector3d Tip_Angles = rot.GetCardanAnglesXYZ();

    double Percent_Error = (Displacement_Model - Displacement_Theory) / Displacement_Theory * 100.0;

    bool passed_displacement = abs(Percent_Error) < 2.0;
    // check the off-axis angles which should be zeros
    bool passed_angles =
        (abs(Tip_Angles.x() * CH_RAD_TO_DEG) < 0.001) && (abs(Tip_Angles.z() * CH_RAD_TO_DEG) < 0.001);
    bool passed_tests = passed_displacement && passed_angles;

    if (msglvl >= 2) {
        std::cout << "Cantilever Beam (Gravity Load) - ANCF Tip Position: " << point << "m" << std::endl;
        std::cout << "Cantilever Beam (Gravity Load) - ANCF Tip Displacement: " << Displacement_Model << "m"
                  << std::endl;
        std::cout << "Cantilever Beam (Gravity Load) - Analytical Tip Displacement: " << Displacement_Theory << "m"
                  << std::endl;
        std::cout << "Cantilever Beam (Gravity Load) - ANCF Tip Angles: (" << Tip_Angles.x() * CH_RAD_TO_DEG << ", "
                  << Tip_Angles.y() * CH_RAD_TO_DEG << ", " << Tip_Angles.z() * CH_RAD_TO_DEG << ")deg"
                  << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Cantilever Beam (Gravity Load) - Tip Displacement Check (Percent Error less than 2%) = "
                  << Percent_Error << "%";
        if (passed_displacement)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");

        std::cout
            << "Cantilever Beam (Gravity Load) - Off-axis Angular misalignment Checks (all angles less than 0.001 deg)";
        if (passed_angles)
            print_green(" - Test PASSED\n\n");
        else
            print_red(" - Test FAILED\n\n");
    }

    return (passed_tests);
}

template <typename ElementVersion>
bool ANCFBeamTest<ElementVersion>::AxialTwistCheck(int msglvl) {
    // =============================================================================
    //  Check the axial twist angle of a beam compared to the analytical result with an applied torque about the beam
    //  axis (some small error is expected based on assumptions and boundary conditions)
    // =============================================================================

    auto system = new ChSystemSMC();
    // Set gravity to 0 since this is a statics test against an analytical solution
    system->SetGravitationalAcceleration(ChVector3d(0, 0, 0));

    auto solver = chrono_types::make_shared<ChSolverSparseLU>();
    solver->UseSparsityPatternLearner(true);
    solver->LockSparsityPattern(true);
    solver->SetVerbose(false);
    system->SetSolver(solver);

    // Set up integrator
    system->SetTimestepperType(ChTimestepper::Type::HHT);
    auto integrator = std::static_pointer_cast<ChTimestepperHHT>(system->GetTimestepper());
    integrator->SetAlpha(-0.2);
    integrator->SetMaxIters(100);
    integrator->SetAbsTolerances(1e-5);
    integrator->SetVerbose(false);
    integrator->SetModifiedNewton(true);

    // Mesh properties
    int num_elements = 32;
    double length = 1;      // m
    double width = 0.1;    // m (square cross-section)
    double height = width;  // m (square cross-section)
    double rho = 7850; //kg/m^3
    double c10 = 0.8e6;  ///Pa
    double c01 = 0.2e6;    ///Pa
    double k = 1e9;    ///Pa
    double mu = 0.01 * 6 * (c10 + c01);     ///< Viscosity Coefficient

    double G = 2 * (c10 + c01); // Pa
    double E = 9 * k*G / (3 * k + G);  // Pa
    double nu = (3 * k - 2 * G) / (2 * (3 * k + G));

    auto material = chrono_types::make_shared<ChMaterialBeamANCF_MR>(rho, c10, c01, k, mu);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    system->Add(mesh);

    // Populate the mesh container with a the nodes and elements for the meshed beam
    int num_nodes = (2 * num_elements) + 1;
    double dx = length / (num_nodes - 1);

    // Setup beam cross section gradients to initially align with the global y and z directions
    ChVector3d dir1(0, 1, 0);
    ChVector3d dir2(0, 0, 1);

    // Create the first node and fix it completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector3d(0, 0, 0.0), dir1, dir2);
    nodeA->SetFixed(true);
    mesh->AddNode(nodeA);

    auto elementlast = chrono_types::make_shared<ElementVersion>();
    std::shared_ptr<ChNodeFEAxyzDD> nodeEndPoint;

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector3d(dx * (2 * i), 0, 0), dir1, dir2);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector3d(dx * (2 * i - 1), 0, 0), dir1, dir2);
        mesh->AddNode(nodeB);
        mesh->AddNode(nodeC);

        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC);
        element->SetDimensions(2 * dx, width, width);
        element->SetMaterial(material);
        mesh->AddElement(element);

        nodeA = nodeB;
        elementlast = element;
    }

    nodeEndPoint = nodeA;

    mesh->SetAutomaticGravity(
        false);  // Turn off the gravity at the mesh level since it is not applied in this test.  This step is not
                 // required since the acceleration due to gravity was already set to all zeros.

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

    auto loader = chrono_types::make_shared<MyLoaderTimeDependentTipLoad>(elementlast);
    loader->auxsystem = system;   // initialize auxiliary data of the loader, if needed
    loader->SetApplication(1.0);  // specify application point
    auto load = chrono_types::make_shared<ChLoad>(loader);
    loadcontainer->Add(load);  // add the load to the load container.

    // Find the static solution for the system (final twist angle)
    system->DoStaticLinear();
    system->DoStaticLinear();

    // Calculate the twist angle of the end of the ANCF beam mesh
    ChVector3d point;
    ChQuaternion<> rot;
    elementlast->EvaluateSectionFrame(1, point, rot);
    ChVector3d Tip_Angles = rot.GetCardanAnglesXYZ();

    // For Analytical Formula, see: https://en.wikipedia.org/wiki/Torsion_constant
    double J = 2.25 * std::pow(0.5 * width, 4);
    double Angle_Theory = TIP_MOMENT * length / (G * J);

    double Percent_Error = (Tip_Angles.x() - Angle_Theory) / Angle_Theory * 100;

    bool passed_twist = abs(Percent_Error) < 20.0;
    // check the off-axis angles which should be zeros
    bool passed_angles =
        (abs(Tip_Angles.y() * CH_RAD_TO_DEG) < 0.001) && (abs(Tip_Angles.z() * CH_RAD_TO_DEG) < 0.001);
    bool passed_tests = passed_twist && passed_angles;

    if (msglvl >= 2) {
        std::cout << "Axial Twist - ANCF Tip Position: " << point << "m" << std::endl;
        std::cout << "Axial Twist - ANCF Twist Angles (Euler 123): " << Tip_Angles * CH_RAD_TO_DEG << "deg"
                  << std::endl;
        std::cout << "Axial Twist - Analytical Twist Angle: " << Angle_Theory * CH_RAD_TO_DEG << "deg" << std::endl;
        std::cout << "Axial Twist - ANCF Tip Angles: (" << Tip_Angles.x() * CH_RAD_TO_DEG << ", "
                  << Tip_Angles.y() * CH_RAD_TO_DEG << ", " << Tip_Angles.z() * CH_RAD_TO_DEG << ")deg"
                  << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Axial Twist - Twist Angle Check (Percent Error less than 20%) = " << Percent_Error << "%";
        if (passed_twist)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");

        std::cout << "Axial Twist - Off-axis Angular misalignment Checks (all angles less than 0.001 deg)";
        if (passed_angles)
            print_green(" - Test PASSED\n\n");
        else
            print_red(" - Test FAILED\n\n");
    }

    return (passed_tests);
}

template <typename ElementVersion>
bool ANCFBeamTest<ElementVersion>::NonlinearAxialDisplacementCheck(int msglvl) {
    // =============================================================================
    //  Check the axial displacement of a beam compared to the analytical result
    //  (some small error is expected based on assumptions and boundary conditions)
    // =============================================================================

    auto system = new ChSystemSMC();
    // Set gravity to 0 since this is a statics test against an analytical solution
    system->SetGravitationalAcceleration(ChVector3d(0, 0, 0));

    auto solver = chrono_types::make_shared<ChSolverSparseLU>();
    solver->UseSparsityPatternLearner(true);
    solver->LockSparsityPattern(true);
    solver->SetVerbose(false);
    system->SetSolver(solver);

    // Set up integrator
    system->SetTimestepperType(ChTimestepper::Type::HHT);
    auto integrator = std::static_pointer_cast<ChTimestepperHHT>(system->GetTimestepper());
    integrator->SetAlpha(-0.2);
    integrator->SetMaxIters(100);
    integrator->SetAbsTolerances(1e-5);
    integrator->SetVerbose(false);
    integrator->SetModifiedNewton(true);

    // Mesh properties
    int num_elements = 10;
    double length = 1.0;      //m
    double width = 0.1;     //m (square cross-section)
    double rho = 7850; //kg/m^3
    double c10 = 0.334e6;  //Pa
    double c01 = -337;    //Pa
    double k = 0.1e9;    //Pa
    double mu = 0 * 0.01 * 6 * (c10 + c01);     ///< Viscosity Coefficient


    double G = 2 * (c10 + c01); // Pa
    double E = 9 * k*G / (3 * k + G);  // Pa
    double nu = (3 * k - 2 * G) / (2 * (3 * k + G));

    auto material = chrono_types::make_shared<ChMaterialBeamANCF_MR>(rho, c10, c01, k, mu);

    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    system->Add(mesh);

    // Populate the mesh container with a the nodes and elements for the meshed beam
    int num_nodes = (2 * num_elements) + 1;
    double dx = length / (num_nodes - 1);

    // Setup beam cross section gradients to initially align with the global y and z directions
    ChVector3d dir1(0, 1, 0);
    ChVector3d dir2(0, 0, 1);

    // Create the first node and fix it completely to ground (Cantilever constraint)
    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector3d(0, 0, 0.0), dir1, dir2);
    nodeA->SetFixed(true);
    mesh->AddNode(nodeA);

    auto elementlast = chrono_types::make_shared<ElementVersion>();
    std::shared_ptr<ChNodeFEAxyzDD> nodeEndPoint;

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector3d(dx * (2 * i), 0, 0), dir1, dir2);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector3d(dx * (2 * i - 1), 0, 0), dir1, dir2);
        mesh->AddNode(nodeB);
        mesh->AddNode(nodeC);

        auto element = chrono_types::make_shared<ElementVersion>();
        element->SetNodes(nodeA, nodeB, nodeC);
        element->SetDimensions(2 * dx, width, width);
        element->SetMaterial(material);
        mesh->AddElement(element);

        nodeA = nodeB;
        elementlast = element;
    }

    nodeEndPoint = nodeA;

    mesh->SetAutomaticGravity(
        false);  // Turn off the gravity at the mesh level since it is not applied in this test.  This step is not
                 // required since the acceleration due to gravity was already set to all zeros.

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
            F(0) = m_Load;  // Apply the force along the global X axis (beam axis)
        }

        void SetLoad(double Load) { m_Load = Load; }
    public:
        // add auxiliary data to the class, if you need to access it during ComputeF().
        ChSystem* auxsystem;
    private:
        double m_Load;
    };

    // Create the load container and to the current system
    auto loadcontainer = chrono_types::make_shared<ChLoadContainer>();
    system->Add(loadcontainer);

    // Create a custom load that uses the custom loader above.
    // The ChLoad is a 'manager' for your ChLoader.
    // It is created using templates, that is instancing a ChLoad<my_loader_class>()

    auto loader = chrono_types::make_shared<MyLoaderTimeDependentTipLoad>(elementlast);
    loader->auxsystem = system;   // initialize auxiliary data of the loader, if needed
    loader->SetApplication(1.0);  // specify application point
    auto load = chrono_types::make_shared<ChLoad>(loader);
    loadcontainer->Add(load);  // add the load to the load container.

    ChVector3d point;
    ChQuaternion<> rot;

    // Find the nonlinear static solution for the system (final displacement)
    double test_load = 500;
    loader->SetLoad(test_load);
    system->DoStaticLinear();
    system->DoStaticLinear();
    //system->DoStaticNonlinear(50);
    elementlast->EvaluateSectionFrame(1, point, rot);
    std::cout << "Load: " << test_load << "N  - Displacement: " << point.x() - length << "m" << std::endl;

    // Find the nonlinear static solution for the system (final displacement)
    test_load = 2000;
    loader->SetLoad(test_load);
    system->DoStaticLinear();
    system->DoStaticLinear();
    //system->DoStaticNonlinear(50);
    elementlast->EvaluateSectionFrame(1, point, rot);
    std::cout << "Load: " << test_load << "N  - Displacement: " << point.x() - length << "m" << std::endl;

    // Find the nonlinear static solution for the system (final displacement)
    test_load = 5000;
    loader->SetLoad(test_load);
    system->DoStaticLinear();
    system->DoStaticLinear();
    //system->DoStaticNonlinear(50);
    elementlast->EvaluateSectionFrame(1, point, rot);
    std::cout << "Load: " << test_load << "N  - Displacement: " << point.x() - length << "m" << std::endl;

    // Find the nonlinear static solution for the system (final displacement)
    test_load = 6500;
    loader->SetLoad(test_load);
    system->DoStaticLinear();
    system->DoStaticLinear();
    //system->DoStaticNonlinear(50);
    elementlast->EvaluateSectionFrame(1, point, rot);
    std::cout << "Load: " << test_load << "N  - Displacement: " << point.x() - length << "m" << std::endl;

    // Find the nonlinear static solution for the system (final displacement)
    test_load = 8000;
    loader->SetLoad(test_load);
    system->DoStaticLinear();
    system->DoStaticLinear();
    //system->DoStaticNonlinear(50);
    elementlast->EvaluateSectionFrame(1, point, rot);
    std::cout << "Load: " << test_load << "N  - Displacement: " << point.x() - length << "m" << std::endl;

    // Find the nonlinear static solution for the system (final displacement)
    test_load = 10000;
    loader->SetLoad(test_load);
    system->DoStaticLinear();
    system->DoStaticLinear();
    //system->DoStaticNonlinear(50);
    elementlast->EvaluateSectionFrame(1, point, rot);
    std::cout << "Load: " << test_load << "N  - Displacement: " << point.x() - length << "m" << std::endl;

    // Find the static solution for the system (final axial displacement)
    system->DoStaticLinear();

    // Calculate the axial displacement of the end of the ANCF beam mesh
    elementlast->EvaluateSectionFrame(1, point, rot);

    //Obrezkov Matikainen Harish - A finite element for soft tissue deformation based on the absolute nodal coordinate formulation
    double Displacement_Theory = 0.80483;
    double Displacement_Model = point.x() - length;
    ChVector3d Tip_Angles = rot.GetCardanAnglesXYZ();

    double Percent_Error = (Displacement_Model - Displacement_Theory) / Displacement_Theory * 100;

    bool passed_displacement = abs(Percent_Error) < 5.0;
    bool passed_angles = (abs(Tip_Angles.x() * CH_RAD_TO_DEG) < 0.001) &&
        (abs(Tip_Angles.y() * CH_RAD_TO_DEG) < 0.001) &&
        (abs(Tip_Angles.z() * CH_RAD_TO_DEG) < 0.001);
    bool passed_tests = passed_displacement && passed_angles;

    if (msglvl >= 2) {
        std::cout << "Axial Pull Test - ANCF Tip Position: " << point << "m" << std::endl;
        std::cout << "Axial Pull Test - ANCF Tip Displacement: " << Displacement_Model << "m" << std::endl;
        std::cout << "Axial Pull Test - Analytical Tip Displacement: " << Displacement_Theory << "m" << std::endl;
        std::cout << "Axial Pull Test - ANCF Tip Angles: (" << Tip_Angles.x() * CH_RAD_TO_DEG << ", "
            << Tip_Angles.y() * CH_RAD_TO_DEG << ", " << Tip_Angles.z() * CH_RAD_TO_DEG << ")deg"
            << std::endl;
    }
    if (msglvl >= 1) {
        std::cout << "Axial Pull Test - Tip Displacement Check (Percent Error less than 5%) = " << Percent_Error << "%";
        if (passed_displacement)
            print_green(" - Test PASSED\n");
        else
            print_red(" - Test FAILED\n");

        std::cout << "Axial Pull Test - Angular misalignment Checks (all angles less than 0.001 deg)";
        if (passed_angles)
            print_green(" - Test PASSED\n\n");
        else
            print_red(" - Test FAILED\n\n");
    }

    return (passed_tests);
}

int main(int argc, char* argv[]) {
#if defined WIN32 || defined _WIN32 || defined WIN64 || defined _WIN64
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    GetConsoleMode(hOut, &dwMode);
    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hOut, dwMode);

    // References:
    // SetConsoleMode() and ENABLE_VIRTUAL_TERMINAL_PROCESSING?
    // https://stackoverflow.com/questions/38772468/setconsolemode-and-enable-virtual-terminal-processing

    // Windows console with ANSI colors handling
    // https://superuser.com/questions/413073/windows-console-with-ansi-colors-handling
#endif

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_3333_MR_Damp> ChElementBeamANCF_3333_MR_Damp_test;
    if (ChElementBeamANCF_3333_MR_Damp_test.RunElementChecks(1))
        print_green("ChElementBeamANCF_3333_MR_Damp Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_3333_MR_Damp Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_3333_MR_NoDamp> ChElementBeamANCF_3333_MR_NoDamp_test;
    if (ChElementBeamANCF_3333_MR_NoDamp_test.RunElementChecks(1))
        print_green("ChElementBeamANCF_3333_MR_NoDamp Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_3333_MR_NoDamp Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_3333_MR_DampNumJac> ChElementBeamANCF_3333_MR_DampNumJac_test;
    if (ChElementBeamANCF_3333_MR_DampNumJac_test.RunElementChecks(1))
        print_green("ChElementBeamANCF_3333_MR_DampNumJac Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_3333_MR_DampNumJac Element Checks = FAILED\n");

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_3333_MR_NoDampNumJac> ChElementBeamANCF_3333_MR_NoDampNumJac_test;
    if (ChElementBeamANCF_3333_MR_NoDampNumJac_test.RunElementChecks(1))
        print_green("ChElementBeamANCF_3333_MR_NoDampNumJac Element Checks = PASSED\n");
    else
        print_red("ChElementBeamANCF_3333_MR_NoDampNumJac Element Checks = FAILED\n");

    return 0;
}

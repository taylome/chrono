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
#include "chrono/utils/ChBenchmark.h"

#include "chrono/physics/ChSystemSMC.h"
#include "chrono/solver/ChIterativeSolverLS.h"

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
#include "chrono/fea/ChElementBeamANCF_MT28.h"
#include "chrono/fea/ChElementBeamANCF_MT30.h"
#include "chrono/fea/ChElementBeamANCF_MT31.h"
#include "chrono/fea/ChElementBeamANCF_MT32.h"

#include "chrono/fea/ChMesh.h"
#include "chrono/fea/ChVisualizationFEAmesh.h"
#include "chrono/physics/ChLoadContainer.h"
#include "chrono/fea/ChLoadsBeam.h"


using namespace chrono;
using namespace chrono::fea;


// =============================================================================

template <typename ElementVersion, typename MaterialVersion>
class ANCFBeamTest {
public:
    ANCFBeamTest();

    ~ANCFBeamTest() { delete m_system; }

    bool RunElementChecks(bool verbose);

protected:
    ChSystemSMC* m_system;
    std::shared_ptr<ElementVersion> m_element;
    std::shared_ptr<ChNodeFEAxyzDD> m_nodeC;
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
    double G = E / (2 + (1 + nu));
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
bool ANCFBeamTest<ElementVersion, MaterialVersion>::RunElementChecks(bool verbose) {
    // =============================================================================
    //  Check the Mass Matrix and Generalized Force due to Gravity
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

    if (verbose) {
        std::cout << "Mass Matrix = " << std::endl;
        std::cout << MassMatrix << std::endl;

        std::cout << "Mass Matrix (Compact Form) = " << std::endl;
        std::cout << MassMatrix_compact << std::endl;
    }
    std::cout << "Mass Matrix (Max Abs Error) = "
        << (MassMatrix - Expected_MassMatrix).cwiseAbs().maxCoeff() << std::endl;


    // =============================================================================
    //  Check the Mass Matrix and Generalized Force due to Gravity
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

    if (verbose) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Generalized Force due to Gravity = " << std::endl;
        std::cout << InternalForceDueToGravity << std::endl;
    }
    std::cout << "Generalized Force due to Gravity (Max Abs Error) = "
        << (InternalForceDueToGravity - Expected_InternalForceDueToGravity).cwiseAbs().maxCoeff() << std::endl;


    // =============================================================================
    //  Check the Internal Force at Zero Displacement 
    //  (should equal all 0's by definition)
    // =============================================================================

    if (verbose) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Generalized Internal Force - No Displacement, No Velocity = " << std::endl;
        std::cout << InternalForceNoDispNoVelNoGravity << std::endl;
    }
    std::cout << "Generalized Internal Force - No Displacement, No Velocity (Max Abs Error) = "
        << InternalForceNoDispNoVelNoGravity.cwiseAbs().maxCoeff() << std::endl;


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


    m_nodeC->SetPos(ChVector<>(m_nodeC->GetPos().x(), m_nodeC->GetPos().y(), 0.001));

    ChVectorDynamic<double> InternalForceSmallDispNoVelNoGravity;
    InternalForceSmallDispNoVelNoGravity.resize(27);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceSmallDispNoVelNoGravity);

    m_nodeC->SetPos(ChVector<>(m_nodeC->GetPos().x(), m_nodeC->GetPos().y(), 0.0));
    
    if (verbose) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Generalized Internal Force - Small Displacement, No Velocity = " << std::endl;
        std::cout << InternalForceSmallDispNoVelNoGravity << std::endl;
    }
    std::cout << "Generalized Internal Force - Small Displacement, No Velocity (Max Abs Error) = "
        << (InternalForceSmallDispNoVelNoGravity - Expected_InternalForceSmallDispNoVelNoGravity).cwiseAbs().maxCoeff() << std::endl;


    // =============================================================================
    //  Check the Internal Force at a No Displacement with a Given Nodal Velocity
    //  (some small error expected depending on the formulation/steps used)
    // =============================================================================

    ChVectorN<double, 27> Expected_InternalForceSmallDispSmallVelNoGravity;
    Expected_InternalForceSmallDispSmallVelNoGravity <<
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

    m_nodeC->SetPos_dt(ChVector<>(0.0, 0.0, 0.001));
    m_element->SetAlphaDamp(0.01);

    ChVectorDynamic<double> InternalForceSmallDispSmallVelNoGravity;
    InternalForceSmallDispSmallVelNoGravity.resize(27);
    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceSmallDispSmallVelNoGravity);

    m_nodeC->SetPos_dt(ChVector<>(0.0, 0.0, 0.0));
    m_element->SetAlphaDamp(0.0);

    if (verbose) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Generalized Internal Force - No Displacement, Small Velocity = " << std::endl;
        std::cout << InternalForceSmallDispSmallVelNoGravity << std::endl;
    }
    std::cout << "Generalized Internal Force - No Displacement, Small Velocity (Max Abs Error) = "
        << (InternalForceSmallDispSmallVelNoGravity - Expected_InternalForceSmallDispSmallVelNoGravity).cwiseAbs().maxCoeff() << std::endl;
    

    // =============================================================================
    //  Check the Jacobian at No Displacement/Velocity - No Damping
    //  (some small error expected depending on the formulation/steps used)
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
    m_element->ComputeInternalForces(InternalForceNoDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_NoDispNoVelNoDamping;
    JacobianK_NoDispNoVelNoDamping.resize(27, 27);
    m_element->ComputeKRMmatricesGlobal(JacobianK_NoDispNoVelNoDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_NoDispNoVelNoDamping;
    JacobianR_NoDispNoVelNoDamping.resize(27, 27);
    m_element->ComputeKRMmatricesGlobal(JacobianR_NoDispNoVelNoDamping, 0, 1, 0);

    double zeros_max_error = 0;
    double max_percent_error = 0;
    ChMatrixDynamic<double> percent_error_matrix;
    percent_error_matrix.resize(27, 27);
    for (auto i = 0; i < Expected_JacobianK_NoDispNoVelNoDamping.rows(); i++) {
        for (auto j = 0; j < Expected_JacobianK_NoDispNoVelNoDamping.cols(); j++) {
            if (std::abs(Expected_JacobianK_NoDispNoVelNoDamping(i, j)) < 100) {
                double error = std::abs(JacobianK_NoDispNoVelNoDamping(i, j) - Expected_JacobianK_NoDispNoVelNoDamping(i, j));
                if (error > zeros_max_error)
                    zeros_max_error = error;
                percent_error_matrix(i, j) = 0.0;
            }
            else {
                double percent_error = std::abs((JacobianK_NoDispNoVelNoDamping(i, j) - Expected_JacobianK_NoDispNoVelNoDamping(i, j)) / Expected_JacobianK_NoDispNoVelNoDamping(i, j));
                if (percent_error > max_percent_error)
                    max_percent_error = percent_error;
                percent_error_matrix(i, j) = percent_error;
            }
        }
    }

    verbose = true;
    if (verbose) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Jacobian K Term - No Displacement, No Velocity, No Damping = " << std::endl;
        std::cout << JacobianK_NoDispNoVelNoDamping << std::endl;
        std::cout << "Expected Jacobian K Term - No Displacement, No Velocity, No Damping = " << std::endl;
        std::cout << Expected_JacobianK_NoDispNoVelNoDamping << std::endl;
        std::cout << "Percent Error Jacobian K Term - No Displacement, No Velocity, No Damping = " << std::endl;
        std::cout << percent_error_matrix << std::endl;
    }
    std::cout << "Jacobian K Term - No Displacement, No Velocity, No Damping (Max Abs Error) = "
        << (JacobianK_NoDispNoVelNoDamping - Expected_JacobianK_NoDispNoVelNoDamping).cwiseAbs().maxCoeff() << std::endl;
    std::cout << "Jacobian K Term - No Displacement, No Velocity, No Damping (Max Abs Error - Only Terms < 100) = "
        << zeros_max_error << std::endl;
    std::cout << "Jacobian K Term - No Displacement, No Velocity, No Damping (Max Abs % Error - Only Terms >= 100) = "
        << max_percent_error*100 << "%" << std::endl;

    if (verbose) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Jacobian R Term - No Displacement, No Velocity, No Damping = " << std::endl;
        std::cout << JacobianR_NoDispNoVelNoDamping << std::endl;
    }
    std::cout << "Jacobian R Term - No Displacement, No Velocity, No Damping (Max Abs Error) = "
        << JacobianR_NoDispNoVelNoDamping.cwiseAbs().maxCoeff() << std::endl;
    

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


    m_element->SetAlphaDamp(0.01);

    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceSmallDispSmallVelNoGravity);

    ChMatrixDynamic<double> JacobianK_NoDispNoVelWithDamping;
    JacobianK_NoDispNoVelWithDamping.resize(27, 27);
    m_element->ComputeKRMmatricesGlobal(JacobianK_NoDispNoVelWithDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_NoDispNoVelWithDamping;
    JacobianR_NoDispNoVelWithDamping.resize(27, 27);
    m_element->ComputeKRMmatricesGlobal(JacobianR_NoDispNoVelWithDamping, 0, 1, 0);

    m_element->SetAlphaDamp(0.0);

    if (verbose) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Jacobian K Term - No Displacement, No Velocity, With Damping = " << std::endl;
        std::cout << JacobianK_NoDispNoVelWithDamping << std::endl;
        std::cout << "Expected Jacobian K Term - No Displacement, No Velocity, With Damping = " << std::endl;
        std::cout << Expected_JacobianK_NoDispNoVelWithDamping << std::endl;
    }
    std::cout << "Jacobian K Term - No Displacement, No Velocity, With Damping (Max Abs Error) = "
        << (JacobianK_NoDispNoVelWithDamping - Expected_JacobianK_NoDispNoVelWithDamping).cwiseAbs().maxCoeff() << std::endl;
    if (verbose) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Jacobian R Term - No Displacement, No Velocity, With Damping = " << std::endl;
        std::cout << JacobianR_NoDispNoVelWithDamping << std::endl;
        std::cout << "Expected Jacobian R Term - No Displacement, No Velocity, With Damping = " << std::endl;
        std::cout << Expected_JacobianR_NoDispNoVelWithDamping << std::endl;
    }
    std::cout << "Jacobian R Term - No Displacement, No Velocity, With Damping (Max Abs Error) = "
        << (JacobianR_NoDispNoVelWithDamping - Expected_JacobianR_NoDispNoVelWithDamping).cwiseAbs().maxCoeff() << std::endl;

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
    m_nodeC->SetPos(ChVector<>(m_nodeC->GetPos().x(), m_nodeC->GetPos().y(), 0.001));

    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceSmallDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_SmallDispNoVelNoDamping;
    JacobianK_SmallDispNoVelNoDamping.resize(27, 27);
    m_element->ComputeKRMmatricesGlobal(JacobianK_SmallDispNoVelNoDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_SmallDispNoVelNoDamping;
    JacobianR_SmallDispNoVelNoDamping.resize(27, 27);
    m_element->ComputeKRMmatricesGlobal(JacobianR_SmallDispNoVelNoDamping, 0, 1, 0);

    m_nodeC->SetPos(ChVector<>(m_nodeC->GetPos().x(), m_nodeC->GetPos().y(), 0.0));


    for (auto i = 0; i < Expected_JacobianK_SmallDispNoVelNoDamping.rows(); i++) {
        for (auto j = 0; j < Expected_JacobianK_SmallDispNoVelNoDamping.cols(); j++) {
            if (std::abs(Expected_JacobianK_SmallDispNoVelNoDamping(i, j)) < 100) {
                double error = std::abs(JacobianK_SmallDispNoVelNoDamping(i, j) - Expected_JacobianK_SmallDispNoVelNoDamping(i, j));
                if (error > zeros_max_error)
                    zeros_max_error = error;
                percent_error_matrix(i, j) = 0.0;
            }
            else {
                double percent_error = std::abs((JacobianK_SmallDispNoVelNoDamping(i, j) - Expected_JacobianK_SmallDispNoVelNoDamping(i, j)) / Expected_JacobianK_SmallDispNoVelNoDamping(i, j));
                if (percent_error > max_percent_error)
                    max_percent_error = percent_error;
                percent_error_matrix(i, j) = percent_error;
            }
        }
    }

    verbose = true;
    if (verbose) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Jacobian K Term - Small Displacement, No Velocity, No Damping = " << std::endl;
        std::cout << JacobianK_SmallDispNoVelNoDamping << std::endl;
        std::cout << "Expected Jacobian K Term - Small Displacement, No Velocity, No Damping = " << std::endl;
        std::cout << Expected_JacobianK_SmallDispNoVelNoDamping << std::endl;
        std::cout << "Percent Error Jacobian K Term - Small Displacement, No Velocity, No Damping = " << std::endl;
        std::cout << percent_error_matrix << std::endl;
    }
    std::cout << "Jacobian K Term - Small Displacement, No Velocity, No Damping (Max Abs Error) = "
        << (JacobianK_SmallDispNoVelNoDamping - Expected_JacobianK_SmallDispNoVelNoDamping).cwiseAbs().maxCoeff() << std::endl;
    std::cout << "Jacobian K Term - Small Displacement, No Velocity, No Damping (Max Abs Error - Only Terms < 100) = "
        << zeros_max_error << std::endl;
    std::cout << "Jacobian K Term - Small Displacement, No Velocity, No Damping (Max Abs % Error - Only Terms >= 100) = "
        << max_percent_error * 100 << "%" << std::endl;

    if (verbose) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Jacobian R Term - Small Displacement, No Velocity, No Damping = " << std::endl;
        std::cout << JacobianR_SmallDispNoVelNoDamping << std::endl;
    }
    std::cout << "Jacobian R Term - Small Displacement, No Velocity, No Damping (Max Abs Error) = "
        << JacobianR_SmallDispNoVelNoDamping.cwiseAbs().maxCoeff() << std::endl;


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


    m_nodeC->SetPos(ChVector<>(m_nodeC->GetPos().x(), m_nodeC->GetPos().y(), 0.001));
    m_element->SetAlphaDamp(0.01);

    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceSmallDispNoVelNoGravity);

    ChMatrixDynamic<double> JacobianK_SmallDispNoVelWithDamping;
    JacobianK_SmallDispNoVelWithDamping.resize(27, 27);
    m_element->ComputeKRMmatricesGlobal(JacobianK_SmallDispNoVelWithDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_SmallDispNoVelWithDamping;
    JacobianR_SmallDispNoVelWithDamping.resize(27, 27);
    m_element->ComputeKRMmatricesGlobal(JacobianR_SmallDispNoVelWithDamping, 0, 1, 0);

    m_element->SetAlphaDamp(0.0);
    m_nodeC->SetPos(ChVector<>(m_nodeC->GetPos().x(), m_nodeC->GetPos().y(), 0.0));


    if (verbose) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Jacobian K Term - Small Displacement, No Velocity, With Damping = " << std::endl;
        std::cout << JacobianK_SmallDispNoVelWithDamping << std::endl;
        std::cout << "Expected Jacobian K Term - Small Displacement, No Velocity, With Damping = " << std::endl;
        std::cout << Expected_JacobianK_SmallDispNoVelWithDamping << std::endl;
    }
    std::cout << "Jacobian K Term - Small Displacement, No Velocity, With Damping (Max Abs Error) = "
        << (JacobianK_SmallDispNoVelWithDamping - Expected_JacobianK_SmallDispNoVelWithDamping).cwiseAbs().maxCoeff() << std::endl;
    if (verbose) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Jacobian R Term - Small Displacement, No Velocity, No Damping = " << std::endl;
        std::cout << JacobianR_SmallDispNoVelWithDamping << std::endl;
        std::cout << "Expected Jacobian R Term - Small Displacement, No Velocity, With Damping = " << std::endl;
        std::cout << Expected_JacobianR_SmallDispNoVelWithDamping << std::endl;
    }
    std::cout << "Jacobian R Term - Small Displacement, No Velocity, With Damping (Max Abs Error) = "
        << (JacobianR_SmallDispNoVelWithDamping - Expected_JacobianR_SmallDispNoVelWithDamping).cwiseAbs().maxCoeff() << std::endl;

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


    m_nodeC->SetPos_dt(ChVector<>(0.0, 0.0, 0.001));
    m_element->SetAlphaDamp(0.01);

    m_element->SetGravityOn(false);
    m_element->ComputeInternalForces(InternalForceSmallDispSmallVelNoGravity);

    ChMatrixDynamic<double> JacobianK_NoDispSmallVelWithDamping;
    JacobianK_NoDispSmallVelWithDamping.resize(27, 27);
    m_element->ComputeKRMmatricesGlobal(JacobianK_NoDispSmallVelWithDamping, 1, 0, 0);

    ChMatrixDynamic<double> JacobianR_NoDispSmallVelWithDamping;
    JacobianR_NoDispSmallVelWithDamping.resize(27, 27);
    m_element->ComputeKRMmatricesGlobal(JacobianR_NoDispSmallVelWithDamping, 0, 1, 0);

    m_element->SetAlphaDamp(0.0);
    m_nodeC->SetPos_dt(ChVector<>(0.0, 0.0, 0.0));

    if (verbose) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Jacobian K Term - No Displacement, Small Velocity, With Damping = " << std::endl;
        std::cout << JacobianK_NoDispSmallVelWithDamping << std::endl;
        std::cout << "Expected Jacobian K Term - No Displacement, Small Velocity, With Damping = " << std::endl;
        std::cout << Expected_JacobianK_NoDispSmallVelWithDamping << std::endl;
    }
    std::cout << "Jacobian K Term - No Displacement, Small Velocity, With Damping (Max Abs Error) = "
        << (JacobianK_NoDispSmallVelWithDamping - Expected_JacobianK_NoDispSmallVelWithDamping).cwiseAbs().maxCoeff() << std::endl;
    if (verbose) {
        std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
        std::cout << "Jacobian R Term - No Displacement, Small Velocity, With Damping = " << std::endl;
        std::cout << JacobianR_NoDispSmallVelWithDamping << std::endl;
        std::cout << "Expected Jacobian R Term - No Displacement, Small Velocity, With Damping = " << std::endl;
        std::cout << Expected_JacobianR_NoDispSmallVelWithDamping << std::endl;
    }
    std::cout << "Jacobian R Term - No Displacement, Small Velocity, With Damping (Max Abs Error) = "
        << (JacobianR_NoDispSmallVelWithDamping - Expected_JacobianR_NoDispSmallVelWithDamping).cwiseAbs().maxCoeff() << std::endl;


    return 1;
}

int main(int argc, char* argv[]) {

    //std::cout << "-------------------------------------" << std::endl;
    //ANCFBeamTest<ChElementBeamANCF, ChMaterialBeamANCF> ChElementBeamANCF_test;
    //if (ChElementBeamANCF_test.RunElementChecks(true))
    //    std::cout << "ChElementBeamANCF Element Checks = PASSED" << std::endl;
    //else
    //    std::cout << "ChElementBeamANCF Element Checks = FAILED" << std::endl;

    std::cout << "-------------------------------------" << std::endl;
    ANCFBeamTest<ChElementBeamANCF_MT01, ChMaterialBeamANCF_MT01> ChElementBeamANCF_MT01_test;
    if (ChElementBeamANCF_MT01_test.RunElementChecks(false))
        std::cout << "ChElementBeamANCF_MT01 Element Checks = PASSED" << std::endl;
    else
        std::cout << "ChElementBeamANCF_MT01 Element Checks = FAILED" << std::endl;

    //std::cout << "-------------------------------------" << std::endl;
    //ANCFBeamTest<ChElementBeamANCF_MT07, ChMaterialBeamANCF_MT07> ChElementBeamANCF_MT07_test;
    //if (ChElementBeamANCF_MT07_test.RunElementChecks(false))
    //    std::cout << "ChElementBeamANCF_MT07 Element Checks = PASSED" << std::endl;
    //else
    //    std::cout << "ChElementBeamANCF_MT07 Element Checks = FAILED" << std::endl;

    //std::cout << "-------------------------------------" << std::endl;
    //ANCFBeamTest<ChElementBeamANCF_MT26, ChMaterialBeamANCF_MT26> ChElementBeamANCF_MT26_test;
    //if (ChElementBeamANCF_MT26_test.RunElementChecks(true))
    //    std::cout << "ChElementBeamANCF_MT26 Element Checks = PASSED" << std::endl;
    //else
    //    std::cout << "ChElementBeamANCF_MT26 Element Checks = FAILED" << std::endl;

    //std::cout << "-------------------------------------" << std::endl;
    //ANCFBeamTest<ChElementBeamANCF_MT30, ChMaterialBeamANCF_MT30> ChElementBeamANCF_MT30_test;
    //if (ChElementBeamANCF_MT30_test.RunElementChecks(false))
    //    std::cout << "ChElementBeamANCF_MT30 Element Checks = PASSED" << std::endl;
    //else
    //    std::cout << "ChElementBeamANCF_MT30 Element Checks = FAILED" << std::endl;

    //std::cout << "-------------------------------------" << std::endl;
    //ANCFBeamTest<ChElementBeamANCF_MT31, ChMaterialBeamANCF_MT31> ChElementBeamANCF_MT31_test;
    //if (ChElementBeamANCF_MT31_test.RunElementChecks(true))
    //    std::cout << "ChElementBeamANCF_MT31 Element Checks = PASSED" << std::endl;
    //else
    //    std::cout << "ChElementBeamANCF_MT31 Element Checks = FAILED" << std::endl;

    return 0;
}
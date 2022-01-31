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
// Authors: Michael Taylor, Antonio Recuero, Radu Serban
// =============================================================================
// Fully Parameterized ANCF brick element with 8 nodes (96DOF). A Description of this element can be found in:
// Olshevskiy, A., Dmitrochenko, O., & Kim, C. W. (2014). Three-dimensional solid brick element using slopes in the
// absolute nodal coordinate formulation. Journal of Computational and Nonlinear Dynamics, 9(2).
// =============================================================================
// Two term Mooney-Rivlin Hyperelastic Material Law with penalty term for incompressibility with the option for a single coefficient nonlinear KV Damping
// =============================================================================

#include "chrono/fea/ChElementHexaANCF_3843_MR_V5.h"
#include "chrono/physics/ChSystem.h"

namespace chrono {
namespace fea {

// ------------------------------------------------------------------------------
// Constructor
// ------------------------------------------------------------------------------

ChElementHexaANCF_3843_MR_V5::ChElementHexaANCF_3843_MR_V5()
    : m_lenX(0), m_lenY(0), m_lenZ(0) {
    m_nodes.resize(8);
}

// ------------------------------------------------------------------------------
// Set element nodes
// ------------------------------------------------------------------------------

void ChElementHexaANCF_3843_MR_V5::SetNodes(std::shared_ptr<ChNodeFEAxyzDDD> nodeA,
                                      std::shared_ptr<ChNodeFEAxyzDDD> nodeB,
                                      std::shared_ptr<ChNodeFEAxyzDDD> nodeC,
                                      std::shared_ptr<ChNodeFEAxyzDDD> nodeD,
                                      std::shared_ptr<ChNodeFEAxyzDDD> nodeE,
                                      std::shared_ptr<ChNodeFEAxyzDDD> nodeF,
                                      std::shared_ptr<ChNodeFEAxyzDDD> nodeG,
                                      std::shared_ptr<ChNodeFEAxyzDDD> nodeH) {
    assert(nodeA);
    assert(nodeB);
    assert(nodeC);
    assert(nodeD);
    assert(nodeE);
    assert(nodeF);
    assert(nodeG);
    assert(nodeH);

    m_nodes[0] = nodeA;
    m_nodes[1] = nodeB;
    m_nodes[2] = nodeC;
    m_nodes[3] = nodeD;
    m_nodes[4] = nodeE;
    m_nodes[5] = nodeF;
    m_nodes[6] = nodeG;
    m_nodes[7] = nodeH;

    std::vector<ChVariables*> mvars;
    mvars.push_back(&m_nodes[0]->Variables());
    mvars.push_back(&m_nodes[0]->Variables_D());
    mvars.push_back(&m_nodes[0]->Variables_DD());
    mvars.push_back(&m_nodes[0]->Variables_DDD());
    mvars.push_back(&m_nodes[1]->Variables());
    mvars.push_back(&m_nodes[1]->Variables_D());
    mvars.push_back(&m_nodes[1]->Variables_DD());
    mvars.push_back(&m_nodes[1]->Variables_DDD());
    mvars.push_back(&m_nodes[2]->Variables());
    mvars.push_back(&m_nodes[2]->Variables_D());
    mvars.push_back(&m_nodes[2]->Variables_DD());
    mvars.push_back(&m_nodes[2]->Variables_DDD());
    mvars.push_back(&m_nodes[3]->Variables());
    mvars.push_back(&m_nodes[3]->Variables_D());
    mvars.push_back(&m_nodes[3]->Variables_DD());
    mvars.push_back(&m_nodes[3]->Variables_DDD());
    mvars.push_back(&m_nodes[4]->Variables());
    mvars.push_back(&m_nodes[4]->Variables_D());
    mvars.push_back(&m_nodes[4]->Variables_DD());
    mvars.push_back(&m_nodes[4]->Variables_DDD());
    mvars.push_back(&m_nodes[5]->Variables());
    mvars.push_back(&m_nodes[5]->Variables_D());
    mvars.push_back(&m_nodes[5]->Variables_DD());
    mvars.push_back(&m_nodes[5]->Variables_DDD());
    mvars.push_back(&m_nodes[6]->Variables());
    mvars.push_back(&m_nodes[6]->Variables_D());
    mvars.push_back(&m_nodes[6]->Variables_DD());
    mvars.push_back(&m_nodes[6]->Variables_DDD());
    mvars.push_back(&m_nodes[7]->Variables());
    mvars.push_back(&m_nodes[7]->Variables_D());
    mvars.push_back(&m_nodes[7]->Variables_DD());
    mvars.push_back(&m_nodes[7]->Variables_DDD());

    Kmatr.SetVariables(mvars);

    // Initial positions and slopes of the element nodes
    // These values define the reference configuration of the element
    CalcCoordMatrix(m_ebar0);

    // Check to see if SetupInitial has already been called.  If so, the precomputed matrices will need to be re-generated.  If not, this will be handled once
    // SetupInitial is called.
    if (m_SD.size() > 0) {
        PrecomputeInternalForceMatricesWeights();
    }
}

// -----------------------------------------------------------------------------
// Element Settings
// -----------------------------------------------------------------------------

// Specify the element dimensions.

void ChElementHexaANCF_3843_MR_V5::SetDimensions(double lenX, double lenY, double lenZ) {
    m_lenX = lenX;
    m_lenY = lenY;
    m_lenZ = lenZ;

    // Check to see if SetupInitial has already been called.  If so, the precomputed matrices will need to be re-generated.  If not, this will be handled once
    // SetupInitial is called.
    if (m_SD.size() > 0) {
        PrecomputeInternalForceMatricesWeights();
    }
}

// Specify the element material.

void ChElementHexaANCF_3843_MR_V5::SetMaterial(std::shared_ptr<ChMaterialHexaANCF_MR> brick_mat) {
    m_material = brick_mat;
}

// -----------------------------------------------------------------------------
// Evaluate Strains and Stresses
// -----------------------------------------------------------------------------
// These functions are designed for single function calls.  If these values are needed at the same points in the element
// through out the simulation, then the adjusted normalized shape function derivative matrix (Sxi_D) for each query
// point should be cached and saved to increase the execution speed
// -----------------------------------------------------------------------------

// Get the Green-Lagrange strain tensor at the normalized element coordinates (xi, eta, zeta) [-1...1]

ChMatrix33<> ChElementHexaANCF_3843_MR_V5::GetGreenLagrangeStrain(const double xi, const double eta, const double zeta) {
    MatrixNx3c Sxi_D;  // Matrix of normalized shape function derivatives
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    ChMatrix33<double> J_0xi;  // Element Jacobian between the reference configuration and normalized configuration
    J_0xi.noalias() = m_ebar0 * Sxi_D;

    Sxi_D = Sxi_D * J_0xi.inverse();  // Adjust the shape function derivative matrix to account for the potentially
                                      // distorted reference configuration

    Matrix3xN e_bar;  // Element coordinates in matrix form
    CalcCoordMatrix(e_bar);

    // Calculate the Deformation Gradient at the current point
    ChMatrixNMc<double, 3, 3> F = e_bar * Sxi_D;

    ChMatrix33<> I3x3;
    I3x3.setIdentity();
    return 0.5 * (F.transpose() * F - I3x3);
}

// Get the 2nd Piola-Kirchoff stress tensor at the normalized element coordinates (xi, eta, zeta) [-1...1] at the
// current state of the element.

ChMatrix33<> ChElementHexaANCF_3843_MR_V5::GetPK2Stress(const double xi, const double eta, const double zeta) {
    MatrixNx3c Sxi_D;  // Matrix of normalized shape function derivatives
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    ChMatrix33<double> J_0xi;  // Element Jacobian between the reference configuration and normalized configuration
    J_0xi.noalias() = m_ebar0 * Sxi_D;

    Sxi_D = Sxi_D * J_0xi.inverse();  // Adjust the shape function derivative matrix to account for the potentially
                                      // distorted reference configuration

    Matrix3xN ebar;  // Element coordinates in matrix form
    CalcCoordMatrix(ebar);

    // Calculate the Deformation Gradient at the current point
    ChMatrixNMc<double, 3, 3> F = ebar * Sxi_D;

    // Calculate the contribution to the 2nd Piola-Kirchoff stress tensor from the Mooney-Rivlin material law
    // Formula from: Cheng, J., & Zhang, L. T. (2018). A general approach to derive stress and elasticity tensors for hyperelastic isotropic and anisotropic biomaterials. International journal of computational methods, 15(04), 1850028.
    // https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6377211/
    double J = F.determinant();
    ChMatrix33<> C = F.transpose()*F;
    ChMatrix33<> Csquare = C*C;
    ChMatrix33<> Cbar = std::pow(J, -2.0 / 3.0)*C;
    ChMatrix33<> I3x3;
    I3x3.setIdentity();
    double J_m23 = std::pow(J, -2.0 / 3.0);
    double I1 = C.trace();
    double I1bar = std::pow(J, -2.0 / 3.0)*I1;
    double I2 = 0.5*(I1*I1 - Csquare.trace());
    double I2bar = std::pow(J, -4.0 / 3.0)*I2;
    double mu1 = 2 * GetMaterial()->Get_c10();
    double mu2 = 2 * GetMaterial()->Get_c01();

    ChMatrix33<> SPK2 =
        (std::pow(J, 1.0 / 3.0) * (J - 1) * GetMaterial()->Get_k() - J_m23 / 3.0 * (mu1 * I1bar + 2 * mu2 * I2bar)) *
            Cbar.inverse() +
        J_m23 * (mu1 + mu2 * I1bar) * I3x3 - J_m23 * mu2 * Cbar;

    
    // Calculate the contribution to the 2nd Piola-Kirchoff stress tensor from the single coefficient nonlinear Kelvin-Voigt Viscoelastic material law

    Matrix3xN edot_bar;  // Element coordinates in matrix form
    CalcCoordDerivMatrix(edot_bar);

    // Calculate the time derivative of the Deformation Gradient at the current point
    ChMatrixNMc<double, 3, 3> Fdot = edot_bar * Sxi_D;

    // Calculate the time derivative of the Greens-Lagrange strain tensor at the current point
    ChMatrix33<> Edot = 0.5 * (F.transpose()*Fdot + Fdot.transpose()*F);
    ChMatrix33<> CInv = C.inverse();
    SPK2.noalias() += J * GetMaterial()->Get_mu()*CInv*Edot*CInv;

    return SPK2;
}

// Get the von Mises stress value at the normalized element coordinates (xi, eta, zeta) [-1...1] at the current
// state of the element.

double ChElementHexaANCF_3843_MR_V5::GetVonMissesStress(const double xi, const double eta, const double zeta) {
    MatrixNx3c Sxi_D;  // Matrix of normalized shape function derivatives
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    ChMatrix33<double> J_0xi;  // Element Jacobian between the reference configuration and normalized configuration
    J_0xi.noalias() = m_ebar0 * Sxi_D;

    Sxi_D = Sxi_D * J_0xi.inverse();  // Adjust the shape function derivative matrix to account for the potentially
                                      // distorted reference configuration

    Matrix3xN ebar;  // Element coordinates in matrix form
    CalcCoordMatrix(ebar);

    // Calculate the Deformation Gradient at the current point
    ChMatrixNMc<double, 3, 3> F = ebar * Sxi_D;

    // Calculate the contribution to the 2nd Piola-Kirchoff stress tensor from the Mooney-Rivlin material law
    // Formula from: Cheng, J., & Zhang, L. T. (2018). A general approach to derive stress and elasticity tensors for hyperelastic isotropic and anisotropic biomaterials. International journal of computational methods, 15(04), 1850028.
    // https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6377211/
    double J = F.determinant();
    ChMatrix33<> C = F.transpose()*F;
    ChMatrix33<> Csquare = C * C;
    ChMatrix33<> Cbar = std::pow(J, -2.0 / 3.0)*C;
    ChMatrix33<> I3x3;
    I3x3.setIdentity();
    double J_m23 = std::pow(J, -2.0 / 3.0);
    double I1 = C.trace();
    double I1bar = std::pow(J, -2.0 / 3.0)*I1;
    double I2 = 0.5*(I1*I1 - Csquare.trace());
    double I2bar = std::pow(J, -4.0 / 3.0)*I2;
    double mu1 = 2 * GetMaterial()->Get_c10();
    double mu2 = 2 * GetMaterial()->Get_c01();

    ChMatrix33<> SPK2 =
        (std::pow(J, 1.0 / 3.0) * (J - 1) * GetMaterial()->Get_k() - J_m23 / 3.0 * (mu1 * I1bar + 2 * mu2 * I2bar)) *
        Cbar.inverse() +
        J_m23 * (mu1 + mu2 * I1bar) * I3x3 - J_m23 * mu2 * Cbar;


    // Calculate the contribution to the 2nd Piola-Kirchoff stress tensor from the single coefficient nonlinear Kelvin-Voigt Viscoelastic material law

    Matrix3xN edot_bar;  // Element coordinates in matrix form
    CalcCoordDerivMatrix(edot_bar);

    // Calculate the time derivative of the Deformation Gradient at the current point
    ChMatrixNMc<double, 3, 3> Fdot = edot_bar * Sxi_D;

    // Calculate the time derivative of the Greens-Lagrange strain tensor at the current point
    ChMatrix33<> Edot = 0.5 * (F.transpose()*Fdot + Fdot.transpose()*F);
    ChMatrix33<> CInv = C.inverse();
    SPK2.noalias() += J * GetMaterial()->Get_mu()*CInv*Edot*CInv;


    // Convert from 2ndPK Stress to Cauchy Stress
    ChMatrix33<double> S = (F * SPK2 * F.transpose()) / J;
    double SVonMises =
        sqrt(0.5 * ((S(0, 0) - S(1, 1)) * (S(0, 0) - S(1, 1)) + (S(1, 1) - S(2, 2)) * (S(1, 1) - S(2, 2)) +
                    (S(2, 2) - S(0, 0)) * (S(2, 2) - S(0, 0))) +
             3 * (S(1, 2) * S(1, 2) + S(2, 0) * S(2, 0) + S(0, 1) * S(0, 1)));

    return (SVonMises);
}

// -----------------------------------------------------------------------------
// Interface to ChElementBase base class
// -----------------------------------------------------------------------------

// Initial element setup.

void ChElementHexaANCF_3843_MR_V5::SetupInitial(ChSystem* system) {
    // Store the initial nodal coordinates. These values define the reference configuration of the element.
    CalcCoordMatrix(m_ebar0);

    // Compute and store the constant mass matrix and the matrix used to multiply the acceleration due to gravity to get
    // the generalized gravitational force vector for the element
    ComputeMassMatrixAndGravityForce();

    // Compute any required matrices and vectors for the generalized internal force and Jacobian calculations
    PrecomputeInternalForceMatricesWeights();
}

// Fill the D vector with the current field values at the element nodes.

void ChElementHexaANCF_3843_MR_V5::GetStateBlock(ChVectorDynamic<>& mD) {
    mD.segment(0, 3) = m_nodes[0]->GetPos().eigen();
    mD.segment(3, 3) = m_nodes[0]->GetD().eigen();
    mD.segment(6, 3) = m_nodes[0]->GetDD().eigen();
    mD.segment(9, 3) = m_nodes[0]->GetDDD().eigen();

    mD.segment(12, 3) = m_nodes[1]->GetPos().eigen();
    mD.segment(15, 3) = m_nodes[1]->GetD().eigen();
    mD.segment(18, 3) = m_nodes[1]->GetDD().eigen();
    mD.segment(21, 3) = m_nodes[1]->GetDDD().eigen();

    mD.segment(24, 3) = m_nodes[2]->GetPos().eigen();
    mD.segment(27, 3) = m_nodes[2]->GetD().eigen();
    mD.segment(30, 3) = m_nodes[2]->GetDD().eigen();
    mD.segment(33, 3) = m_nodes[2]->GetDDD().eigen();

    mD.segment(36, 3) = m_nodes[3]->GetPos().eigen();
    mD.segment(39, 3) = m_nodes[3]->GetD().eigen();
    mD.segment(42, 3) = m_nodes[3]->GetDD().eigen();
    mD.segment(45, 3) = m_nodes[3]->GetDDD().eigen();

    mD.segment(48, 3) = m_nodes[4]->GetPos().eigen();
    mD.segment(51, 3) = m_nodes[4]->GetD().eigen();
    mD.segment(54, 3) = m_nodes[4]->GetDD().eigen();
    mD.segment(57, 3) = m_nodes[4]->GetDDD().eigen();

    mD.segment(60, 3) = m_nodes[5]->GetPos().eigen();
    mD.segment(63, 3) = m_nodes[5]->GetD().eigen();
    mD.segment(66, 3) = m_nodes[5]->GetDD().eigen();
    mD.segment(69, 3) = m_nodes[5]->GetDDD().eigen();

    mD.segment(72, 3) = m_nodes[6]->GetPos().eigen();
    mD.segment(75, 3) = m_nodes[6]->GetD().eigen();
    mD.segment(78, 3) = m_nodes[6]->GetDD().eigen();
    mD.segment(81, 3) = m_nodes[6]->GetDDD().eigen();

    mD.segment(84, 3) = m_nodes[7]->GetPos().eigen();
    mD.segment(87, 3) = m_nodes[7]->GetD().eigen();
    mD.segment(90, 3) = m_nodes[7]->GetDD().eigen();
    mD.segment(93, 3) = m_nodes[7]->GetDDD().eigen();
}

// State update.

void ChElementHexaANCF_3843_MR_V5::Update() {
    ChElementGeneric::Update();
}

// Return the mass matrix in full sparse form.

void ChElementHexaANCF_3843_MR_V5::ComputeMmatrixGlobal(ChMatrixRef M) {
    M.setZero();

    // Mass Matrix is Stored in Compact Upper Triangular Form
    // Expand it out into its Full Sparse Symmetric Form
    unsigned int idx = 0;
    for (unsigned int i = 0; i < NSF; i++) {
        for (unsigned int j = i; j < NSF; j++) {
            M(3 * i, 3 * j) = m_MassMatrix(idx);
            M(3 * i + 1, 3 * j + 1) = m_MassMatrix(idx);
            M(3 * i + 2, 3 * j + 2) = m_MassMatrix(idx);
            if (i != j) {
                M(3 * j, 3 * i) = m_MassMatrix(idx);
                M(3 * j + 1, 3 * i + 1) = m_MassMatrix(idx);
                M(3 * j + 2, 3 * i + 2) = m_MassMatrix(idx);
            }
            idx++;
        }
    }
}

// This class computes and adds corresponding masses to ElementGeneric member m_TotalMass

void ChElementHexaANCF_3843_MR_V5::ComputeNodalMass() {
    m_nodes[0]->m_TotalMass += m_MassMatrix(0) + m_MassMatrix(4) + m_MassMatrix(8) + m_MassMatrix(12) +
                               m_MassMatrix(16) + m_MassMatrix(20) + m_MassMatrix(24) + m_MassMatrix(28);
    m_nodes[1]->m_TotalMass += m_MassMatrix(4) + m_MassMatrix(122) + m_MassMatrix(126) + m_MassMatrix(130) +
                               m_MassMatrix(134) + m_MassMatrix(138) + m_MassMatrix(142) + m_MassMatrix(146);
    m_nodes[2]->m_TotalMass += m_MassMatrix(8) + m_MassMatrix(126) + m_MassMatrix(228) + m_MassMatrix(232) +
                               m_MassMatrix(236) + m_MassMatrix(240) + m_MassMatrix(244) + m_MassMatrix(248);
    m_nodes[3]->m_TotalMass += m_MassMatrix(12) + m_MassMatrix(130) + m_MassMatrix(232) + m_MassMatrix(318) +
                               m_MassMatrix(322) + m_MassMatrix(326) + m_MassMatrix(330) + m_MassMatrix(334);
    m_nodes[4]->m_TotalMass += m_MassMatrix(16) + m_MassMatrix(134) + m_MassMatrix(236) + m_MassMatrix(322) +
                               m_MassMatrix(392) + m_MassMatrix(396) + m_MassMatrix(400) + m_MassMatrix(404);
    m_nodes[5]->m_TotalMass += m_MassMatrix(20) + m_MassMatrix(138) + m_MassMatrix(240) + m_MassMatrix(326) +
                               m_MassMatrix(396) + m_MassMatrix(450) + m_MassMatrix(454) + m_MassMatrix(458);
    m_nodes[6]->m_TotalMass += m_MassMatrix(24) + m_MassMatrix(142) + m_MassMatrix(244) + m_MassMatrix(330) +
                               m_MassMatrix(400) + m_MassMatrix(454) + m_MassMatrix(492) + m_MassMatrix(496);
    m_nodes[7]->m_TotalMass += m_MassMatrix(28) + m_MassMatrix(146) + m_MassMatrix(248) + m_MassMatrix(334) +
                               m_MassMatrix(404) + m_MassMatrix(458) + m_MassMatrix(496) + m_MassMatrix(518);
}

// Compute the generalized internal force vector for the current nodal coordinates and set the value in the Fi vector.

void ChElementHexaANCF_3843_MR_V5::ComputeInternalForces(ChVectorDynamic<>& Fi) {
    
    if (GetMaterial()->Get_mu() != 0) {
        ComputeInternalForceDamping(Fi);
    }
    else {
        ComputeInternalForceNoDamping(Fi);
    }
}

// Calculate the global matrix H as a linear combination of K, R, and M:
//   H = Mfactor * [M] + Kfactor * [K] + Rfactor * [R]

void ChElementHexaANCF_3843_MR_V5::ComputeKRMmatricesGlobal(ChMatrixRef H, double Kfactor, double Rfactor, double Mfactor) {

    if (GetMaterial()->Get_mu() != 0) {
        ComputeInternalJacobianDamping(H, Kfactor, Rfactor, Mfactor);
    }
    else {
        ComputeInternalJacobianNoDamping(H, Kfactor, Mfactor);
    }
}

// Compute the generalized force vector due to gravity using the efficient ANCF specific method
void ChElementHexaANCF_3843_MR_V5::ComputeGravityForces(ChVectorDynamic<>& Fg, const ChVector<>& G_acc) {
    assert(Fg.size() == 3 * NSF);

    // Calculate and add the generalized force due to gravity to the generalized internal force vector for the element.
    // The generalized force due to gravity could be computed once prior to the start of the simulation if gravity was
    // assumed constant throughout the entire simulation.  However, this implementation assumes that the acceleration
    // due to gravity, while a constant for the entire system, can change from step to step which could be useful for
    // gravity loaded units tests as an example.  The generalized force due to gravity is calculated in compact matrix
    // form and is pre-mapped to the desired vector format
    Eigen::Map<MatrixNx3> GravForceCompact(Fg.data(), NSF, 3);
    GravForceCompact = m_GravForceScale * G_acc.eigen().transpose();
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

void ChElementHexaANCF_3843_MR_V5::EvaluateElementFrame(const double xi,
                                                  const double eta,
                                                  const double zeta,
                                                  ChVector<>& point,
                                                  ChQuaternion<>& rot) {
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, eta, zeta);
    VectorN Sxi_xi_compact;
    Calc_Sxi_xi_compact(Sxi_xi_compact, xi, eta, zeta);
    VectorN Sxi_eta_compact;
    Calc_Sxi_eta_compact(Sxi_eta_compact, xi, eta, zeta);

    Matrix3xN e_bar;
    CalcCoordMatrix(e_bar);

    // r = S*e written in compact form
    point = e_bar * Sxi_compact;

    // Since ANCF does not use rotations, calculate an approximate
    // rotation based off the position vector gradients
    ChVector<double> Xdir = e_bar * Sxi_xi_compact * 2 / m_lenX;
    ChVector<double> Ydir = e_bar * Sxi_eta_compact * 2 / m_lenY;

    // Since the position vector gradients are not in general orthogonal,
    // set the Dx direction tangent to the brick xi axis and
    // compute the Dy and Dz directions by using a
    // Gram-Schmidt orthonormalization, guided by the brick eta axis
    ChMatrix33<> msect;
    msect.Set_A_Xdir(Xdir, Ydir);

    rot = msect.Get_A_quaternion();
}

void ChElementHexaANCF_3843_MR_V5::EvaluateElementPoint(const double xi,
                                                  const double eta,
                                                  const double zeta,
                                                  ChVector<>& point) {
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, eta, zeta);

    Matrix3xN e_bar;
    CalcCoordMatrix(e_bar);

    // r = S*e written in compact form
    point = e_bar * Sxi_compact;
}

void ChElementHexaANCF_3843_MR_V5::EvaluateElementVel(double xi, double eta, const double zeta, ChVector<>& Result) {
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, eta, zeta);

    Matrix3xN e_bardot;
    CalcCoordDerivMatrix(e_bardot);

    // rdot = S*edot written in compact form
    Result = e_bardot * Sxi_compact;
}

// -----------------------------------------------------------------------------
// Functions for ChLoadable interface
// -----------------------------------------------------------------------------

// Gets all the DOFs packed in a single vector (position part).

void ChElementHexaANCF_3843_MR_V5::LoadableGetStateBlock_x(int block_offset, ChState& mD) {
    mD.segment(block_offset + 0, 3) = m_nodes[0]->GetPos().eigen();
    mD.segment(block_offset + 3, 3) = m_nodes[0]->GetD().eigen();
    mD.segment(block_offset + 6, 3) = m_nodes[0]->GetDD().eigen();
    mD.segment(block_offset + 9, 3) = m_nodes[0]->GetDDD().eigen();

    mD.segment(block_offset + 12, 3) = m_nodes[1]->GetPos().eigen();
    mD.segment(block_offset + 15, 3) = m_nodes[1]->GetD().eigen();
    mD.segment(block_offset + 18, 3) = m_nodes[1]->GetDD().eigen();
    mD.segment(block_offset + 21, 3) = m_nodes[1]->GetDDD().eigen();

    mD.segment(block_offset + 24, 3) = m_nodes[2]->GetPos().eigen();
    mD.segment(block_offset + 27, 3) = m_nodes[2]->GetD().eigen();
    mD.segment(block_offset + 30, 3) = m_nodes[2]->GetDD().eigen();
    mD.segment(block_offset + 33, 3) = m_nodes[2]->GetDDD().eigen();

    mD.segment(block_offset + 36, 3) = m_nodes[3]->GetPos().eigen();
    mD.segment(block_offset + 39, 3) = m_nodes[3]->GetD().eigen();
    mD.segment(block_offset + 42, 3) = m_nodes[3]->GetDD().eigen();
    mD.segment(block_offset + 45, 3) = m_nodes[3]->GetDDD().eigen();

    mD.segment(block_offset + 48, 3) = m_nodes[4]->GetPos().eigen();
    mD.segment(block_offset + 51, 3) = m_nodes[4]->GetD().eigen();
    mD.segment(block_offset + 54, 3) = m_nodes[4]->GetDD().eigen();
    mD.segment(block_offset + 57, 3) = m_nodes[4]->GetDDD().eigen();

    mD.segment(block_offset + 60, 3) = m_nodes[5]->GetPos().eigen();
    mD.segment(block_offset + 63, 3) = m_nodes[5]->GetD().eigen();
    mD.segment(block_offset + 66, 3) = m_nodes[5]->GetDD().eigen();
    mD.segment(block_offset + 69, 3) = m_nodes[5]->GetDDD().eigen();

    mD.segment(block_offset + 72, 3) = m_nodes[6]->GetPos().eigen();
    mD.segment(block_offset + 75, 3) = m_nodes[6]->GetD().eigen();
    mD.segment(block_offset + 78, 3) = m_nodes[6]->GetDD().eigen();
    mD.segment(block_offset + 81, 3) = m_nodes[6]->GetDDD().eigen();

    mD.segment(block_offset + 84, 3) = m_nodes[7]->GetPos().eigen();
    mD.segment(block_offset + 87, 3) = m_nodes[7]->GetD().eigen();
    mD.segment(block_offset + 90, 3) = m_nodes[7]->GetDD().eigen();
    mD.segment(block_offset + 93, 3) = m_nodes[7]->GetDDD().eigen();
}

// Gets all the DOFs packed in a single vector (velocity part).

void ChElementHexaANCF_3843_MR_V5::LoadableGetStateBlock_w(int block_offset, ChStateDelta& mD) {
    mD.segment(block_offset + 0, 3) = m_nodes[0]->GetPos_dt().eigen();
    mD.segment(block_offset + 3, 3) = m_nodes[0]->GetD_dt().eigen();
    mD.segment(block_offset + 6, 3) = m_nodes[0]->GetDD_dt().eigen();
    mD.segment(block_offset + 9, 3) = m_nodes[0]->GetDDD_dt().eigen();

    mD.segment(block_offset + 12, 3) = m_nodes[1]->GetPos_dt().eigen();
    mD.segment(block_offset + 15, 3) = m_nodes[1]->GetD_dt().eigen();
    mD.segment(block_offset + 18, 3) = m_nodes[1]->GetDD_dt().eigen();
    mD.segment(block_offset + 21, 3) = m_nodes[1]->GetDDD_dt().eigen();

    mD.segment(block_offset + 24, 3) = m_nodes[2]->GetPos_dt().eigen();
    mD.segment(block_offset + 27, 3) = m_nodes[2]->GetD_dt().eigen();
    mD.segment(block_offset + 30, 3) = m_nodes[2]->GetDD_dt().eigen();
    mD.segment(block_offset + 33, 3) = m_nodes[2]->GetDDD_dt().eigen();

    mD.segment(block_offset + 36, 3) = m_nodes[3]->GetPos_dt().eigen();
    mD.segment(block_offset + 39, 3) = m_nodes[3]->GetD_dt().eigen();
    mD.segment(block_offset + 42, 3) = m_nodes[3]->GetDD_dt().eigen();
    mD.segment(block_offset + 45, 3) = m_nodes[3]->GetDDD_dt().eigen();

    mD.segment(block_offset + 48, 3) = m_nodes[4]->GetPos_dt().eigen();
    mD.segment(block_offset + 51, 3) = m_nodes[4]->GetD_dt().eigen();
    mD.segment(block_offset + 54, 3) = m_nodes[4]->GetDD_dt().eigen();
    mD.segment(block_offset + 57, 3) = m_nodes[4]->GetDDD_dt().eigen();

    mD.segment(block_offset + 60, 3) = m_nodes[5]->GetPos_dt().eigen();
    mD.segment(block_offset + 63, 3) = m_nodes[5]->GetD_dt().eigen();
    mD.segment(block_offset + 66, 3) = m_nodes[5]->GetDD_dt().eigen();
    mD.segment(block_offset + 69, 3) = m_nodes[5]->GetDDD_dt().eigen();

    mD.segment(block_offset + 72, 3) = m_nodes[6]->GetPos_dt().eigen();
    mD.segment(block_offset + 75, 3) = m_nodes[6]->GetD_dt().eigen();
    mD.segment(block_offset + 78, 3) = m_nodes[6]->GetDD_dt().eigen();
    mD.segment(block_offset + 81, 3) = m_nodes[6]->GetDDD_dt().eigen();

    mD.segment(block_offset + 84, 3) = m_nodes[7]->GetPos_dt().eigen();
    mD.segment(block_offset + 87, 3) = m_nodes[7]->GetD_dt().eigen();
    mD.segment(block_offset + 90, 3) = m_nodes[7]->GetDD_dt().eigen();
    mD.segment(block_offset + 93, 3) = m_nodes[7]->GetDDD_dt().eigen();
}

/// Increment all DOFs using a delta.

void ChElementHexaANCF_3843_MR_V5::LoadableStateIncrement(const unsigned int off_x,
                                                    ChState& x_new,
                                                    const ChState& x,
                                                    const unsigned int off_v,
                                                    const ChStateDelta& Dv) {
    m_nodes[0]->NodeIntStateIncrement(off_x, x_new, x, off_v, Dv);
    m_nodes[1]->NodeIntStateIncrement(off_x + 12, x_new, x, off_v + 12, Dv);
    m_nodes[2]->NodeIntStateIncrement(off_x + 24, x_new, x, off_v + 24, Dv);
    m_nodes[3]->NodeIntStateIncrement(off_x + 36, x_new, x, off_v + 36, Dv);
    m_nodes[4]->NodeIntStateIncrement(off_x + 48, x_new, x, off_v + 48, Dv);
    m_nodes[5]->NodeIntStateIncrement(off_x + 60, x_new, x, off_v + 60, Dv);
    m_nodes[6]->NodeIntStateIncrement(off_x + 72, x_new, x, off_v + 72, Dv);
    m_nodes[7]->NodeIntStateIncrement(off_x + 84, x_new, x, off_v + 84, Dv);
}

// Get the pointers to the contained ChVariables, appending to the mvars vector.

void ChElementHexaANCF_3843_MR_V5::LoadableGetVariables(std::vector<ChVariables*>& mvars) {
    for (int i = 0; i < m_nodes.size(); ++i) {
        mvars.push_back(&m_nodes[i]->Variables());
        mvars.push_back(&m_nodes[i]->Variables_D());
        mvars.push_back(&m_nodes[i]->Variables_DD());
        mvars.push_back(&m_nodes[i]->Variables_DDD());
    }
}

// Evaluate N'*F, which is the projection of the applied point force and moment at the coordinates (xi,eta,zeta)
// This calculation takes a slightly different form for ANCF elements
// For this ANCF element, only the first 6 entries in F are used in the calculation.  The first three entries is
// the applied force in global coordinates and the second 3 entries is the applied moment in global space.

void ChElementHexaANCF_3843_MR_V5::ComputeNF(
    const double xi,             // parametric coordinate in volume
    const double eta,            // parametric coordinate in volume
    const double zeta,           // parametric coordinate in volume
    ChVectorDynamic<>& Qi,       // Return result of N'*F  here, maybe with offset block_offset
    double& detJ,                // Return det[J] here
    const ChVectorDynamic<>& F,  // Input F vector, size is = n.field coords.
    ChVectorDynamic<>* state_x,  // if != 0, update state (pos. part) to this, then evaluate Q
    ChVectorDynamic<>* state_w   // if != 0, update state (speed part) to this, then evaluate Q
) {
    // Compute the generalized force vector for the applied force component using the compact form of the shape
    // functions.  This requires a reshaping of the calculated matrix to get it into the correct vector order (just a
    // reinterpretation of the data since the matrix is in row-major format)
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, eta, zeta);
    MatrixNx3 QiCompact;

    QiCompact = Sxi_compact * F.segment(0, 3).transpose();

    Eigen::Map<Vector3N> QiReshaped(QiCompact.data(), QiCompact.size());
    Qi = QiReshaped;

    // Compute the generalized force vector for the applied moment component
    // See: Antonio M Recuero, Javier F Aceituno, Jose L Escalona, and Ahmed A Shabana.
    // A nonlinear approach for modeling rail flexibility using the absolute nodal coordinate
    // formulation. Nonlinear Dynamics, 83(1-2):463-481, 2016.

    Matrix3xN e_bar;
    CalcCoordMatrix(e_bar);

    MatrixNx3c Sxi_D;
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    ChMatrix33<double> J_Cxi;
    ChMatrix33<double> J_Cxi_Inv;

    J_Cxi.noalias() = e_bar * Sxi_D;
    J_Cxi_Inv = J_Cxi.inverse();

    // Compute the unique pieces that make up the moment projection matrix "G"
    VectorN G_A = Sxi_D.col(0).transpose() * J_Cxi_Inv(0, 0) + Sxi_D.col(1).transpose() * J_Cxi_Inv(1, 0) +
                  Sxi_D.col(2).transpose() * J_Cxi_Inv(2, 0);
    VectorN G_B = Sxi_D.col(0).transpose() * J_Cxi_Inv(0, 1) + Sxi_D.col(1).transpose() * J_Cxi_Inv(1, 1) +
                  Sxi_D.col(2).transpose() * J_Cxi_Inv(2, 1);
    VectorN G_C = Sxi_D.col(0).transpose() * J_Cxi_Inv(0, 2) + Sxi_D.col(1).transpose() * J_Cxi_Inv(1, 2) +
                  Sxi_D.col(2).transpose() * J_Cxi_Inv(2, 2);

    ChVectorN<double, 3> M_scaled = 0.5 * F.segment(3, 3);

    // Compute G'M without actually forming the complete matrix "G" (since it has a sparsity pattern to it)
    for (unsigned int i = 0; i < NSF; i++) {
        Qi(3 * i) += M_scaled(1) * G_C(i) - M_scaled(2) * G_B(i);
        Qi((3 * i) + 1) += M_scaled(2) * G_A(i) - M_scaled(0) * G_C(i);
        Qi((3 * i) + 2) += M_scaled(0) * G_B(i) - M_scaled(1) * G_A(i);
    }

    // Compute the element Jacobian between the current configuration and the normalized configuration
    // This is different than the element Jacobian between the reference configuration and the normalized
    //  configuration used in the internal force calculations.  For this calculation, this is the ratio between the
    //  actual differential volume and the normalized differential volume.  The determinate of the element Jacobian is
    //  used to calculate this volume ratio for potential use in Gauss-Quadrature or similar numeric integration.
    detJ = J_Cxi.determinant();
}

// Return the element density (needed for ChLoaderVolumeGravity).

double ChElementHexaANCF_3843_MR_V5::GetDensity() {
    return GetMaterial()->Get_rho();
}

// -----------------------------------------------------------------------------
// Mass Matrix & Generalized Force Due to Gravity Calculation
// -----------------------------------------------------------------------------

void ChElementHexaANCF_3843_MR_V5::ComputeMassMatrixAndGravityForce() {
    // For this element, the mass matrix integrand is of order 7 in xi, 7 in eta, and 7 in zeta.
    // 4 GQ Points are needed in the xi, eta, and zeta directions for exact integration of the element's mass matrix,
    // even if the reference configuration is not straight. Since the major pieces of the generalized force due to
    // gravity can also be used to calculate the mass matrix, these calculations are performed at the same time.  Only
    // the matrix that scales the acceleration due to gravity is calculated at this time so that any changes to the
    // acceleration due to gravity in the system are correctly accounted for in the generalized internal force
    // calculation.

    ChQuadratureTables* GQTable = GetStaticGQTables();
    unsigned int GQ_idx_xi_eta_zeta = 3;  // 4 Point Gauss-Quadrature;

    // Mass Matrix in its compact matrix form.  Since the mass matrix is symmetric, just the upper diagonal entries will
    // be stored.
    ChMatrixNM<double, NSF, NSF> MassMatrixCompactSquare;

    // Set these to zeros since they will be incremented as the vector/matrix is calculated
    MassMatrixCompactSquare.setZero();
    m_GravForceScale.setZero();

    double rho = GetMaterial()->Get_rho();  // Density of the material for the element

    // Sum the contribution to the mass matrix and generalized force due to gravity at the current point
    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi_eta_zeta].size(); it_xi++) {
        for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[GQ_idx_xi_eta_zeta].size(); it_eta++) {
            for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[GQ_idx_xi_eta_zeta].size(); it_zeta++) {
                double GQ_weight = GQTable->Weight[GQ_idx_xi_eta_zeta][it_xi] *
                                   GQTable->Weight[GQ_idx_xi_eta_zeta][it_eta] *
                                   GQTable->Weight[GQ_idx_xi_eta_zeta][it_zeta];
                double xi = GQTable->Lroots[GQ_idx_xi_eta_zeta][it_xi];
                double eta = GQTable->Lroots[GQ_idx_xi_eta_zeta][it_eta];
                double zeta = GQTable->Lroots[GQ_idx_xi_eta_zeta][it_zeta];
                double det_J_0xi = Calc_det_J_0xi(xi, eta, zeta);  // determinant of the element Jacobian (volume ratio)

                VectorN Sxi_compact;  // Vector of the Unique Normalized Shape Functions
                Calc_Sxi_compact(Sxi_compact, xi, eta, zeta);

                m_GravForceScale += (GQ_weight * rho * det_J_0xi) * Sxi_compact;
                MassMatrixCompactSquare += (GQ_weight * rho * det_J_0xi) * Sxi_compact * Sxi_compact.transpose();
            }
        }
    }

    // Store just the unique entries in the Mass Matrix in Compact Upper Triangular Form
    // since the full Mass Matrix is both sparse and symmetric
    unsigned int idx = 0;
    for (unsigned int i = 0; i < NSF; i++) {
        for (unsigned int j = i; j < NSF; j++) {
            m_MassMatrix(idx) = MassMatrixCompactSquare(i, j);
            idx++;
        }
    }
}

// Precalculate constant matrices for the internal force calculations when using the "Continuous Integration" style
// method

void ChElementHexaANCF_3843_MR_V5::PrecomputeInternalForceMatricesWeights() {
    ChQuadratureTables* GQTable = GetStaticGQTables();
    unsigned int GQ_idx_xi_eta_zeta = NP - 1;  // Gauss-Quadrature table index for xi, eta, and zeta

    m_SD.resize(NSF, 3 * NIP);
    m_kGQ.resize(NIP, 1);

    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi_eta_zeta].size(); it_xi++) {
        for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[GQ_idx_xi_eta_zeta].size(); it_eta++) {
            for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[GQ_idx_xi_eta_zeta].size(); it_zeta++) {
                double GQ_weight = GQTable->Weight[GQ_idx_xi_eta_zeta][it_xi] *
                                   GQTable->Weight[GQ_idx_xi_eta_zeta][it_eta] *
                                   GQTable->Weight[GQ_idx_xi_eta_zeta][it_zeta];
                double xi = GQTable->Lroots[GQ_idx_xi_eta_zeta][it_xi];
                double eta = GQTable->Lroots[GQ_idx_xi_eta_zeta][it_eta];
                double zeta = GQTable->Lroots[GQ_idx_xi_eta_zeta][it_zeta];
                auto index =
                    it_zeta + it_eta * GQTable->Lroots[GQ_idx_xi_eta_zeta].size() +
                    it_xi * GQTable->Lroots[GQ_idx_xi_eta_zeta].size() * GQTable->Lroots[GQ_idx_xi_eta_zeta].size();
                ChMatrix33<double>
                    J_0xi;         // Element Jacobian between the reference configuration and normalized configuration
                MatrixNx3c Sxi_D;  // Matrix of normalized shape function derivatives

                Calc_Sxi_D(Sxi_D, xi, eta, zeta);
                J_0xi.noalias() = m_ebar0 * Sxi_D;

                // Adjust the shape function derivative matrix to account for a potentially deformed reference state
                // SD_precompute_D = Sxi_D * J_0xi.inverse();
                m_SD.block(0, 3 * index, NSF, 3) = Sxi_D * J_0xi.inverse();
                m_kGQ(index) = -J_0xi.determinant() * GQ_weight;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Elastic force calculation
// -----------------------------------------------------------------------------

void ChElementHexaANCF_3843_MR_V5::ComputeInternalForceDamping(ChVectorDynamic<>& Fi) {
    assert(Fi.size() == 3 * NSF);

    // Element coordinates in matrix form
    Matrix3xN ebar;
    CalcCoordMatrix(ebar);

    // Element coordinate time derivatives in matrix form
    Matrix3xN ebardot;
    CalcCoordDerivMatrix(ebardot);

    //MatrixNx3 QiCompact;
    //QiCompact.setZero();
    //Fi.setZero();
    //Eigen::Map<MatrixNx3> FiCompact(Fi.data(), NSF, 3);


    double c10 = GetMaterial()->Get_c10();
    double c01 = GetMaterial()->Get_c01();
    double k = GetMaterial()->Get_k();
    double mu = GetMaterial()->Get_mu();

    ChMatrixNM<double, 3 * NIP, 3> CombinedF;

    //Calculate the contribution from each Gauss-quadrature point, point by point
    for (unsigned int GQpnt = 0; GQpnt < NIP; GQpnt++) {
        MatrixNx3c Sbar_xi_D = m_SD.block<NSF, 3>(0, 3 * GQpnt);

        // Calculate the Deformation Gradient at the current point
        ChMatrixNMc<double, 3, 3> F = ebar * Sbar_xi_D;

        //Calculate the Right Cauchy-Green deformation tensor (symmetric):
        ChMatrix33<> C = F.transpose()*F;

        //Calculate the 1st invariant of the Right Cauchy-Green deformation tensor: trace(C)
        double I1 = C(0, 0) + C(1, 1) + C(2, 2);

        //Calculate the 2nd invariant of the Right Cauchy-Green deformation tensor: 1/2*(trace(C)^2-trace(C^2))
        double I2 = 0.5*(I1*I1 - C(0, 0)*C(0, 0) - C(1, 1)*C(1, 1) - C(2, 2)*C(2, 2)) - C(0, 1)*C(0, 1) - C(0, 2)*C(0, 2) - C(1, 2)*C(1, 2);

        //Calculate the determinate of F (commonly denoted as J)
        double detF = F(0, 0)*F(1, 1)*F(2, 2) + F(0, 1)*F(1, 2)*F(2, 0) + F(0, 2)*F(1, 0)*F(2, 1)
            - F(0, 0)*F(1, 2)*F(2, 1) - F(0, 1)*F(1, 0)*F(2, 2) - F(0, 2)*F(1, 1)*F(2, 0);

        double detF_m2_3 = std::pow(detF, -2.0 / 3.0);       // detF^(-2/3)

        double scale1 = m_kGQ(GQpnt) * 2.0 * c10 * detF_m2_3;
        double scale2 = m_kGQ(GQpnt) * 2.0 * c01 * detF_m2_3 * detF_m2_3;
        double scale3 = scale1 + scale2 * I1;
        double scale4 = m_kGQ(GQpnt) * k * (detF - 1.0) - (scale1*I1 + 2.0 * scale2*I2) / (3.0 * detF);

        //--------------------------------------------------------------------------------
        // 2 term Mooney-Rivlin Material Combined with the Nonlinear Viscoelastic Contribution
        //--------------------------------------------------------------------------------

        // Calculate the time derivative of the Deformation Gradient at the current point
        ChMatrixNMc<double, 3, 3> Fdot = ebardot * Sbar_xi_D;

        // Calculate the time derivative of the Greens-Lagrange strain tensor at the current point
        ChMatrix33<> Edot = 0.5 * (F.transpose()*Fdot + Fdot.transpose()*F);

        double kGQmu_over_detF3 = m_kGQ(GQpnt) * mu / (detF*detF*detF);

        double CInv00 = C(1, 1) * C(2, 2) - C(1, 2) * C(1, 2);
        double CInv11 = C(0, 0) * C(2, 2) - C(0, 2) * C(0, 2);
        double CInv22 = C(0, 0) * C(1, 1) - C(0, 1) * C(0, 1);
        double CInv01 = C(0, 2) * C(1, 2) - C(0, 1) * C(2, 2);
        double CInv02 = C(0, 1) * C(1, 2) - C(0, 2) * C(1, 1);
        double CInv12 = C(0, 2) * C(0, 1) - C(0, 0) * C(1, 2);

        ChMatrixNMc<double, 3, 3> SPK2_NLKV;
        SPK2_NLKV(0, 0) = kGQmu_over_detF3 * (Edot(0, 0)*CInv00*CInv00 + 2 * Edot(0, 1)*CInv00*CInv01 + 2 * Edot(0, 2)*CInv00*CInv02 + Edot(1, 1) * CInv01*CInv01 + 2 * Edot(1, 2)*CInv01*CInv02 + Edot(2, 2) * CInv02*CInv02);
        SPK2_NLKV(1, 0) = kGQmu_over_detF3 * (Edot(0, 0)*CInv00*CInv01 + Edot(0, 1) * (CInv01*CInv01 + CInv00 * CInv11) + Edot(0, 2) * (CInv00*CInv12 + CInv01 * CInv02) + Edot(1, 1) * CInv01*CInv11 + Edot(1, 2) * (CInv01*CInv12 + CInv11 * CInv02) + Edot(2, 2) * CInv02*CInv12);
        SPK2_NLKV(2, 0) = kGQmu_over_detF3 * (Edot(0, 0)*CInv00*CInv02 + Edot(0, 1) * (CInv00*CInv12 + CInv01 * CInv02) + Edot(0, 2) * (CInv02*CInv02 + CInv00 * CInv22) + Edot(1, 1) * CInv01*CInv12 + Edot(1, 2) * (CInv01*CInv22 + CInv02 * CInv12) + Edot(2, 2) * CInv02*CInv22);

        SPK2_NLKV(0, 1) = SPK2_NLKV(1, 0);
        SPK2_NLKV(1, 1) = kGQmu_over_detF3 * (Edot(0, 0)*CInv01*CInv01 + 2 * Edot(0, 1)*CInv01*CInv11 + 2 * Edot(0, 2)*CInv01*CInv12 + Edot(1, 1) * CInv11*CInv11 + 2 * Edot(1, 2)*CInv11*CInv12 + Edot(2, 2) * CInv12*CInv12);
        SPK2_NLKV(2, 1) = kGQmu_over_detF3 * (Edot(0, 0)*CInv01*CInv02 + Edot(0, 1) * (CInv01*CInv12 + CInv11 * CInv02) + Edot(0, 2) * (CInv01*CInv22 + CInv02 * CInv12) + Edot(1, 1) * CInv11*CInv12 + Edot(1, 2) * (CInv12*CInv12 + CInv11 * CInv22) + Edot(2, 2) * CInv12*CInv22);

        SPK2_NLKV(0, 2) = SPK2_NLKV(2, 0);
        SPK2_NLKV(1, 2) = SPK2_NLKV(2, 1);
        SPK2_NLKV(2, 2) = kGQmu_over_detF3 * (Edot(0, 0)*CInv02*CInv02 + 2 * Edot(0, 1)*CInv02*CInv12 + 2 * Edot(0, 2)*CInv02*CInv22 + Edot(1, 1) * CInv12*CInv12 + 2 * Edot(1, 2)*CInv12*CInv22 + Edot(2, 2) * CInv22*CInv22);


        CombinedF.block<3,3>(3 * GQpnt,0).noalias() = (SPK2_NLKV - scale2 * C) * F.transpose() + scale3 * F.transpose();
        CombinedF(3 * GQpnt+0, 0) += scale4 * (F(1, 1)*F(2, 2) - F(1, 2)*F(2, 1));
        CombinedF(3 * GQpnt+0, 1) += scale4 * (F(0, 2)*F(2, 1) - F(0, 1)*F(2, 2));
        CombinedF(3 * GQpnt+0, 2) += scale4 * (F(0, 1)*F(1, 2) - F(0, 2)*F(1, 1));

        CombinedF(3 * GQpnt+1, 0) += scale4 * (F(1, 2)*F(2, 0) - F(1, 0)*F(2, 2));
        CombinedF(3 * GQpnt+1, 1) += scale4 * (F(0, 0)*F(2, 2) - F(0, 2)*F(2, 0));
        CombinedF(3 * GQpnt+1, 2) += scale4 * (F(0, 2)*F(1, 0) - F(0, 0)*F(1, 2));

        CombinedF(3 * GQpnt+2, 0) += scale4 * (F(1, 0)*F(2, 1) - F(1, 1)*F(2, 0));
        CombinedF(3 * GQpnt+2, 1) += scale4 * (F(0, 1)*F(2, 0) - F(0, 0)*F(2, 1));
        CombinedF(3 * GQpnt+2, 2) += scale4 * (F(0, 0)*F(1, 1) - F(0, 1)*F(1, 0));
    }

    Eigen::Map<MatrixNx3> FiCompact(Fi.data(), NSF, 3);
    FiCompact.noalias() = m_SD * CombinedF;
}

void ChElementHexaANCF_3843_MR_V5::ComputeInternalForceNoDamping(ChVectorDynamic<>& Fi) {
    assert(Fi.size() == 3 * NSF);

    // Element coordinates in matrix form
    Matrix3xN ebar;
    CalcCoordMatrix(ebar);

    // Element coordinate time derivatives in matrix form
    Matrix3xN ebardot;
    CalcCoordDerivMatrix(ebardot);

    double c10 = GetMaterial()->Get_c10();
    double c01 = GetMaterial()->Get_c01();
    double k = GetMaterial()->Get_k();

    ChMatrixNM<double, 3 * NIP, 3> CombinedF;

    //Calculate the contribution from each Gauss-quadrature point, point by point
    for (unsigned int GQpnt = 0; GQpnt < NIP; GQpnt++) {
        MatrixNx3c Sbar_xi_D = m_SD.block<NSF, 3>(0, 3 * GQpnt);

        // Calculate the Deformation Gradient at the current point
        ChMatrixNMc<double, 3, 3> F = ebar * Sbar_xi_D;

        //Calculate the Right Cauchy-Green deformation tensor (symmetric):
        //ChMatrix33<> C = F.transpose()*F;
        ChMatrixNMc<double, 3, 3> C = F.transpose()*F;

        //Calculate the 1st invariant of the Right Cauchy-Green deformation tensor: trace(C)
        double I1 = C(0, 0) + C(1, 1) + C(2, 2);

        //Calculate the 2nd invariant of the Right Cauchy-Green deformation tensor: 1/2*(trace(C)^2-trace(C^2))
        double I2 = 0.5*(I1*I1 - C(0, 0)*C(0, 0) - C(1, 1)*C(1, 1) - C(2, 2)*C(2, 2)) - C(0, 1)*C(0, 1) - C(0, 2)*C(0, 2) - C(1, 2)*C(1, 2);

        //Calculate the determinate of F (commonly denoted as J)
        double detF = F(0, 0)*F(1, 1)*F(2, 2) + F(0, 1)*F(1, 2)*F(2, 0) + F(0, 2)*F(1, 0)*F(2, 1)
            - F(0, 0)*F(1, 2)*F(2, 1) - F(0, 1)*F(1, 0)*F(2, 2) - F(0, 2)*F(1, 1)*F(2, 0);

        double detF_m2_3 = std::pow(detF, -2.0 / 3.0);       // detF^(-2/3)

        double scale1 = m_kGQ(GQpnt) * 2.0 * c10 * detF_m2_3;
        double scale2 = m_kGQ(GQpnt) * 2.0 * c01 * detF_m2_3 * detF_m2_3;
        double scale3 = scale1 + scale2 * I1;
        double scale4 = m_kGQ(GQpnt) * k * (detF - 1.0) - (scale1*I1 + 2.0 * scale2*I2) / (3.0 * detF);
        
        CombinedF.block<3, 3>(3 * GQpnt, 0).noalias() = scale3 * F.transpose() - scale2 * C *F.transpose();
        CombinedF(3 * GQpnt + 0, 0) += scale4 * (F(1, 1)*F(2, 2) - F(1, 2)*F(2, 1));
        CombinedF(3 * GQpnt + 0, 1) += scale4 * (F(0, 2)*F(2, 1) - F(0, 1)*F(2, 2));
        CombinedF(3 * GQpnt + 0, 2) += scale4 * (F(0, 1)*F(1, 2) - F(0, 2)*F(1, 1));

        CombinedF(3 * GQpnt + 1, 0) += scale4 * (F(1, 2)*F(2, 0) - F(1, 0)*F(2, 2));
        CombinedF(3 * GQpnt + 1, 1) += scale4 * (F(0, 0)*F(2, 2) - F(0, 2)*F(2, 0));
        CombinedF(3 * GQpnt + 1, 2) += scale4 * (F(0, 2)*F(1, 0) - F(0, 0)*F(1, 2));

        CombinedF(3 * GQpnt + 2, 0) += scale4 * (F(1, 0)*F(2, 1) - F(1, 1)*F(2, 0));
        CombinedF(3 * GQpnt + 2, 1) += scale4 * (F(0, 1)*F(2, 0) - F(0, 0)*F(2, 1));
        CombinedF(3 * GQpnt + 2, 2) += scale4 * (F(0, 0)*F(1, 1) - F(0, 1)*F(1, 0));

        //Check to see if this below is any faster...was used to debug the vectorized code version
        //CombinedF(3 * GQpnt + 0, 0) = scale4 * (F(1, 1)*F(2, 2) - F(1, 2)*F(2, 1)) + scale3 * F(0, 0) - scale2 * (C(0, 0)*F(0, 0) + C(0, 1)*F(0, 1) + C(0, 2)*F(0, 2));
        //CombinedF(3 * GQpnt + 0, 1) = scale4 * (F(0, 2)*F(2, 1) - F(0, 1)*F(2, 2)) + scale3 * F(1, 0) - scale2 * (C(0, 0)*F(1, 0) + C(0, 1)*F(1, 1) + C(0, 2)*F(1, 2));
        //CombinedF(3 * GQpnt + 0, 2) = scale4 * (F(0, 1)*F(1, 2) - F(0, 2)*F(1, 1)) + scale3 * F(2, 0) - scale2 * (C(0, 0)*F(2, 0) + C(0, 1)*F(2, 1) + C(0, 2)*F(2, 2));

        //CombinedF(3 * GQpnt + 1, 0) = scale4 * (F(1, 2)*F(2, 0) - F(1, 0)*F(2, 2)) + scale3 * F(0, 1) - scale2 * (C(1, 0)*F(0, 0) + C(1, 1)*F(0, 1) + C(1, 2)*F(0, 2));
        //CombinedF(3 * GQpnt + 1, 1) = scale4 * (F(0, 0)*F(2, 2) - F(0, 2)*F(2, 0)) + scale3 * F(1, 1) - scale2 * (C(1, 0)*F(1, 0) + C(1, 1)*F(1, 1) + C(1, 2)*F(1, 2));
        //CombinedF(3 * GQpnt + 1, 2) = scale4 * (F(0, 2)*F(1, 0) - F(0, 0)*F(1, 2)) + scale3 * F(2, 1) - scale2 * (C(1, 0)*F(2, 0) + C(1, 1)*F(2, 1) + C(1, 2)*F(2, 2));

        //CombinedF(3 * GQpnt + 2, 0) = scale4 * (F(1, 0)*F(2, 1) - F(1, 1)*F(2, 0)) + scale3 * F(0, 2) - scale2 * (C(2, 0)*F(0, 0) + C(2, 1)*F(0, 1) + C(2, 2)*F(0, 2));
        //CombinedF(3 * GQpnt + 2, 1) = scale4 * (F(0, 1)*F(2, 0) - F(0, 0)*F(2, 1)) + scale3 * F(1, 2) - scale2 * (C(2, 0)*F(1, 0) + C(2, 1)*F(1, 1) + C(2, 2)*F(1, 2));
        //CombinedF(3 * GQpnt + 2, 2) = scale4 * (F(0, 0)*F(1, 1) - F(0, 1)*F(1, 0)) + scale3 * F(2, 2) - scale2 * (C(2, 0)*F(2, 0) + C(2, 1)*F(2, 1) + C(2, 2)*F(2, 2));
    }

    Eigen::Map<MatrixNx3> FiCompact(Fi.data(), NSF, 3);
    FiCompact.noalias() = m_SD * CombinedF;
}


// -----------------------------------------------------------------------------
// Jacobians of internal forces
// -----------------------------------------------------------------------------

void ChElementHexaANCF_3843_MR_V5::ComputeInternalJacobianDamping(ChMatrixRef H, double Kfactor, double Rfactor, double Mfactor) {
    assert((H.rows() == 3 * NSF) && (H.cols() == 3 * NSF));

    // The integrated quantity represents the 96x96 Jacobian
//      Kfactor * [K] + Rfactor * [R]
// Note that the matrices with current nodal coordinates and velocities are
// already available in m_d and m_d_dt (as set in ComputeInternalForces).
// Similarly, the ANS strain and strain derivatives are already available in
// m_strainANS and m_strainANS_D (as calculated in ComputeInternalForces).

    // Element coordinates in matrix form
    Matrix3xN ebar;
    CalcCoordMatrix(ebar);

    // Element coordinate time derivatives in matrix form
    Matrix3xN ebardot;
    CalcCoordDerivMatrix(ebardot);

    H.setZero();

    double c10 = GetMaterial()->Get_c10();
    double c01 = GetMaterial()->Get_c01();
    double k = GetMaterial()->Get_k();
    double mu = GetMaterial()->Get_mu();

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> JacCompact;
    JacCompact.resize(NSF, NSF);
    JacCompact.setZero();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> FA;
    FA.resize(NSF, NSF);
    FA.setZero();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> FB;
    FB.resize(NSF, NSF);
    FB.setZero();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> FC;
    FC.resize(NSF, NSF);
    FC.setZero();

    //Calculate the contribution from each Gauss-quadrature point, point by point
    for (unsigned int GQpnt = 0; GQpnt < NIP; GQpnt++) {
        MatrixNx3c Sbar_xi_D = m_SD.block<NSF, 3>(0, 3 * GQpnt);

        // Calculate the Deformation Gradient at the current point
        ChMatrixNMc<double, 3, 3> F = ebar * Sbar_xi_D;

        //Calculate the Right Cauchy-Green deformation tensor (symmetric):
        ChMatrix33<> C = F.transpose()*F;

        //Calculate the 1st invariant of the Right Cauchy-Green deformation tensor: trace(C)
        double I1 = C(0, 0) + C(1, 1) + C(2, 2);

        //Calculate the 2nd invariant of the Right Cauchy-Green deformation tensor: 1/2*(trace(C)^2-trace(C^2))
        double I2 = 0.5*(I1*I1 - C(0, 0)*C(0, 0) - C(1, 1)*C(1, 1) - C(2, 2)*C(2, 2)) - C(0, 1)*C(0, 1) - C(0, 2)*C(0, 2) - C(1, 2)*C(1, 2);

        //Calculate the determinate of F (commonly denoted as J)
        double detF = F(0, 0)*F(1, 1)*F(2, 2) + F(0, 1)*F(1, 2)*F(2, 0) + F(0, 2)*F(1, 0)*F(2, 1)
            - F(0, 0)*F(1, 2)*F(2, 1) - F(0, 1)*F(1, 0)*F(2, 2) - F(0, 2)*F(1, 1)*F(2, 0);

        double detF_m2_3 = std::pow(detF, -2.0 / 3.0);       // detF^(-2/3)

        double scale1 = -Kfactor * m_kGQ(GQpnt) * 2.0 * c10 * detF_m2_3;
        double scale2 = -Kfactor * m_kGQ(GQpnt) * 2.0 * c01 * detF_m2_3 * detF_m2_3;
        double scale3 = scale1 + scale2 * I1;
        double scale4 = -Kfactor * m_kGQ(GQpnt) * k * (detF - 1.0) - (scale1*I1 + 2.0 * scale2*I2) / (3.0 * detF);
        double scale5 = 0.5*((5 * I1*scale1 + 14 * I2*scale2) / (9 * detF*detF) - Kfactor * m_kGQ(GQpnt) * k);
        double scale6 = -2 * (scale1 + 2 * I1*scale2) / (3 * detF);
        double scale7 = 4 * scale2 / (3 * detF);

        ChMatrixNM<double, 1, 3> detF_S0;
        detF_S0 << F(1, 1)*F(2, 2) - F(1, 2)*F(2, 1), F(0, 2)*F(2, 1) - F(0, 1)*F(2, 2), F(0, 1)*F(1, 2) - F(0, 2)*F(1, 1);
        ChMatrixNM<double, 1, 3> detF_S1;
        detF_S1 << F(1, 2)*F(2, 0) - F(1, 0)*F(2, 2), F(0, 0)*F(2, 2) - F(0, 2)*F(2, 0), F(0, 2)*F(1, 0) - F(0, 0)*F(1, 2);
        ChMatrixNM<double, 1, 3> detF_S2;
        detF_S2 << F(1, 0)*F(2, 1) - F(1, 1)*F(2, 0), F(0, 1)*F(2, 0) - F(0, 0)*F(2, 1), F(0, 0)*F(1, 1) - F(0, 1)*F(1, 0);

        ChMatrixNMc<double, 3 * NSF, 9> Left;
        //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Left;
        //Left.resize(3 * NSF, 9);
        Eigen::Map<MatrixNx3> Left_HalfdC00_de_T_Compact(Left.col(0).data(), NSF, 3);
        Eigen::Map<MatrixNx3> Left_HalfdC11_de_T_Compact(Left.col(1).data(), NSF, 3);
        Eigen::Map<MatrixNx3> Left_HalfdC22_de_T_Compact(Left.col(2).data(), NSF, 3);
        Eigen::Map<MatrixNx3> Left_dC12_de_T_Compact(Left.col(3).data(), NSF, 3);
        Eigen::Map<MatrixNx3> Left_dC02_de_T_Compact(Left.col(4).data(), NSF, 3);
        Eigen::Map<MatrixNx3> Left_dC01_de_T_Compact(Left.col(5).data(), NSF, 3);
        Eigen::Map<MatrixNx3> Left_dI1_de_T_Compact(Left.col(6).data(), NSF, 3);
        Eigen::Map<MatrixNx3> Left_Combined_Compact(Left.col(7).data(), NSF, 3);
        Eigen::Map<MatrixNx3> Left_ddetF_de_T_Compact(Left.col(8).data(), NSF, 3);

        Left_HalfdC00_de_T_Compact = Sbar_xi_D.col(0)*(F.col(0).transpose());
        Left_HalfdC11_de_T_Compact = Sbar_xi_D.col(1)*(F.col(1).transpose());
        Left_HalfdC22_de_T_Compact = Sbar_xi_D.col(2)*(F.col(2).transpose());
        Left_dC12_de_T_Compact = Sbar_xi_D.col(1)*(F.col(2).transpose()) + Sbar_xi_D.col(2)*(F.col(1).transpose());
        Left_dC02_de_T_Compact = Sbar_xi_D.col(0)*(F.col(2).transpose()) + Sbar_xi_D.col(2)*(F.col(0).transpose());
        Left_dC01_de_T_Compact = Sbar_xi_D.col(0)*(F.col(1).transpose()) + Sbar_xi_D.col(1)*(F.col(0).transpose());

        Left_dI1_de_T_Compact = Sbar_xi_D.col(0)*(2 * F.col(0).transpose()) +
            Sbar_xi_D.col(1)*(2 * F.col(1).transpose()) +
            Sbar_xi_D.col(2)*(2 * F.col(2).transpose());

        Left_Combined_Compact = Sbar_xi_D.col(0)*(scale6*F.col(0).transpose() + scale7 * C.row(0)*F.transpose() + scale5 * detF_S0) +
            Sbar_xi_D.col(1)*(scale6*F.col(1).transpose() + scale7 * C.row(1)*F.transpose() + scale5 * detF_S1) +
            Sbar_xi_D.col(2)*(scale6*F.col(2).transpose() + scale7 * C.row(2)*F.transpose() + scale5 * detF_S2);

        Left_ddetF_de_T_Compact = Sbar_xi_D.col(0)*detF_S0 + Sbar_xi_D.col(1)*detF_S1 + Sbar_xi_D.col(2)*detF_S2;

        ChMatrixNMc<double, 3 * NSF, 9> Right;
        //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Right;
        //Right.resize(3 * NSF, 9);
        Right.col(0) = (-2.0*scale2) * Left.col(0);
        Right.col(1) = (-2.0*scale2) * Left.col(1);
        Right.col(2) = (-2.0*scale2) * Left.col(2);
        Right.col(3) = -scale2 * Left.col(3);
        Right.col(4) = -scale2 * Left.col(4);
        Right.col(5) = -scale2 * Left.col(5);
        Right.col(6) = 0.5*scale2*Left.col(6);
        Right.col(7) = Left.col(8);
        Right.col(8) = Left.col(7);

        //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> LeftRight;
        //LeftRight = Left * Right.transpose();

        H.noalias() += Left * Right.transpose();

        JacCompact.noalias() += Sbar_xi_D.col(0)*(scale3*Sbar_xi_D.col(0).transpose() - scale2 * C.row(0)*Sbar_xi_D.transpose())
            + Sbar_xi_D.col(1)*(scale3*Sbar_xi_D.col(1).transpose() - scale2 * C.row(1)*Sbar_xi_D.transpose())
            + Sbar_xi_D.col(2)*(scale3*Sbar_xi_D.col(2).transpose() - scale2 * C.row(2)*Sbar_xi_D.transpose());

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> FA_Mult;
        FA_Mult.resize(NSF, 3);
        FA_Mult.col(0) = (scale4 * F(0, 2))*Sbar_xi_D.col(1) - (scale4 * F(0, 1))*Sbar_xi_D.col(2);
        FA_Mult.col(1) = (scale4 * F(0, 0))*Sbar_xi_D.col(2) - (scale4 * F(0, 2))*Sbar_xi_D.col(0);
        FA_Mult.col(2) = (scale4 * F(0, 1))*Sbar_xi_D.col(0) - (scale4 * F(0, 0))*Sbar_xi_D.col(1);
        FA.noalias() += Sbar_xi_D * FA_Mult.transpose();

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> FB_Mult;
        FB_Mult.resize(NSF, 3);
        FB_Mult.col(0) = (scale4 * F(1, 2))*Sbar_xi_D.col(1) - (scale4 * F(1, 1))*Sbar_xi_D.col(2);
        FB_Mult.col(1) = (scale4 * F(1, 0))*Sbar_xi_D.col(2) - (scale4 * F(1, 2))*Sbar_xi_D.col(0);
        FB_Mult.col(2) = (scale4 * F(1, 1))*Sbar_xi_D.col(0) - (scale4 * F(1, 0))*Sbar_xi_D.col(1);
        FB.noalias() += Sbar_xi_D * FB_Mult.transpose();

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> FC_Mult;
        FC_Mult.resize(NSF, 3);
        FC_Mult.col(0) = (scale4 * F(2, 2))*Sbar_xi_D.col(1) - (scale4 * F(2, 1))*Sbar_xi_D.col(2);
        FC_Mult.col(1) = (scale4 * F(2, 0))*Sbar_xi_D.col(2) - (scale4 * F(2, 2))*Sbar_xi_D.col(0);
        FC_Mult.col(2) = (scale4 * F(2, 1))*Sbar_xi_D.col(0) - (scale4 * F(2, 0))*Sbar_xi_D.col(1);
        FC.noalias() += Sbar_xi_D * FC_Mult.transpose();

        
        //--------------------------------------------------------------------------------
        // 2 term Mooney-Rivlin Material Combined with the Nonlinear Viscoelastic Contribution
        //--------------------------------------------------------------------------------

        // Calculate the time derivative of the Deformation Gradient at the current point
        ChMatrixNMc<double, 3, 3> Fdot = ebardot * Sbar_xi_D;

        // Calculate the time derivative of the Greens-Lagrange strain tensor at the current point
        ChMatrix33<> Edot = 0.5 * (F.transpose()*Fdot + Fdot.transpose()*F);

        double kGQmu_over_detF3 = m_kGQ(GQpnt) * mu / (detF*detF*detF);

        double CInv00 = C(1, 1) * C(2, 2) - C(1, 2) * C(1, 2);
        double CInv11 = C(0, 0) * C(2, 2) - C(0, 2) * C(0, 2);
        double CInv22 = C(0, 0) * C(1, 1) - C(0, 1) * C(0, 1);
        double CInv01 = C(0, 2) * C(1, 2) - C(0, 1) * C(2, 2);
        double CInv02 = C(0, 1) * C(1, 2) - C(0, 2) * C(1, 1);
        double CInv12 = C(0, 2) * C(0, 1) - C(0, 0) * C(1, 2);

        double SPK2_NLKV_0 = kGQmu_over_detF3 * (Edot(0, 0)*CInv00*CInv00 + 2 * Edot(0, 1)*CInv00*CInv01 + 2 * Edot(0, 2)*CInv00*CInv02 + Edot(1, 1) * CInv01*CInv01 + 2 * Edot(1, 2)*CInv01*CInv02 + Edot(2, 2) * CInv02*CInv02);
        double SPK2_NLKV_1 = kGQmu_over_detF3 * (Edot(0, 0)*CInv01*CInv01 + 2 * Edot(0, 1)*CInv01*CInv11 + 2 * Edot(0, 2)*CInv01*CInv12 + Edot(1, 1) * CInv11*CInv11 + 2 * Edot(1, 2)*CInv11*CInv12 + Edot(2, 2) * CInv12*CInv12);
        double SPK2_NLKV_2 = kGQmu_over_detF3 * (Edot(0, 0)*CInv02*CInv02 + 2 * Edot(0, 1)*CInv02*CInv12 + 2 * Edot(0, 2)*CInv02*CInv22 + Edot(1, 1) * CInv12*CInv12 + 2 * Edot(1, 2)*CInv12*CInv22 + Edot(2, 2) * CInv22*CInv22);
        double SPK2_NLKV_3 = kGQmu_over_detF3 * (Edot(0, 0)*CInv01*CInv02 + Edot(0, 1) * (CInv01*CInv12 + CInv11 * CInv02) + Edot(0, 2) * (CInv01*CInv22 + CInv02 * CInv12) + Edot(1, 1) * CInv11*CInv12 + Edot(1, 2) * (CInv12*CInv12 + CInv11 * CInv22) + Edot(2, 2) * CInv12*CInv22);
        double SPK2_NLKV_4 = kGQmu_over_detF3 * (Edot(0, 0)*CInv00*CInv02 + Edot(0, 1) * (CInv00*CInv12 + CInv01 * CInv02) + Edot(0, 2) * (CInv02*CInv02 + CInv00 * CInv22) + Edot(1, 1) * CInv01*CInv12 + Edot(1, 2) * (CInv01*CInv22 + CInv02 * CInv12) + Edot(2, 2) * CInv02*CInv22);
        double SPK2_NLKV_5 = kGQmu_over_detF3 * (Edot(0, 0)*CInv00*CInv01 + Edot(0, 1) * (CInv01*CInv01 + CInv00 * CInv11) + Edot(0, 2) * (CInv00*CInv12 + CInv01 * CInv02) + Edot(1, 1) * CInv01*CInv11 + Edot(1, 2) * (CInv01*CInv12 + CInv11 * CInv02) + Edot(2, 2) * CInv02*CInv12);

        JacCompact.noalias() += Sbar_xi_D.col(0)*((-Kfactor * SPK2_NLKV_0)*Sbar_xi_D.col(0).transpose() + (-Kfactor * SPK2_NLKV_5)*Sbar_xi_D.col(1).transpose() + (-Kfactor * SPK2_NLKV_4)*Sbar_xi_D.col(2).transpose())
            + Sbar_xi_D.col(1)*((-Kfactor * SPK2_NLKV_5)*Sbar_xi_D.col(0).transpose() + (-Kfactor * SPK2_NLKV_1)*Sbar_xi_D.col(1).transpose() + (-Kfactor * SPK2_NLKV_3)*Sbar_xi_D.col(2).transpose())
            + Sbar_xi_D.col(2)*((-Kfactor * SPK2_NLKV_4)*Sbar_xi_D.col(0).transpose() + (-Kfactor * SPK2_NLKV_3)*Sbar_xi_D.col(1).transpose() + (-Kfactor * SPK2_NLKV_2)*Sbar_xi_D.col(2).transpose());

        double SPK2_Scale = 3 * Kfactor / detF;

        double EdotC1_CInvC1 = Edot(0, 0) * CInv00 + Edot(0, 1) * CInv01 + Edot(0, 2) * CInv02;
        double EdotC2_CInvC1 = Edot(0, 1) * CInv00 + Edot(1, 1) * CInv01 + Edot(1, 2) * CInv02;
        double EdotC3_CInvC1 = Edot(0, 2) * CInv00 + Edot(1, 2) * CInv01 + Edot(2, 2) * CInv02;

        double EdotC1_CInvC2 = Edot(0, 0) * CInv01 + Edot(0, 1) * CInv11 + Edot(0, 2) * CInv12;
        double EdotC2_CInvC2 = Edot(0, 1) * CInv01 + Edot(1, 1) * CInv11 + Edot(1, 2) * CInv12;
        double EdotC3_CInvC2 = Edot(0, 2) * CInv01 + Edot(1, 2) * CInv11 + Edot(2, 2) * CInv12;

        double EdotC1_CInvC3 = Edot(0, 0) * CInv02 + Edot(0, 1) * CInv12 + Edot(0, 2) * CInv22;
        double EdotC2_CInvC3 = Edot(0, 1) * CInv02 + Edot(1, 1) * CInv12 + Edot(1, 2) * CInv22;
        double EdotC3_CInvC3 = Edot(0, 2) * CInv02 + Edot(1, 2) * CInv12 + Edot(2, 2) * CInv22;

        double KK = -Kfactor * kGQmu_over_detF3;
        double KR = -Rfactor * kGQmu_over_detF3;

        double ScaleNLKV1_11cF = KK * (0) + KR * (CInv00*CInv00);
        double ScaleNLKV1_22cF = KK * (4 * C(2, 2)*EdotC1_CInvC1 - 4 * C(0, 2)*EdotC3_CInvC1) + KR * (CInv01*CInv01);
        double ScaleNLKV1_33cF = KK * (4 * C(1, 1)*EdotC1_CInvC1 - 4 * C(0, 1)*EdotC2_CInvC1) + KR * (CInv02*CInv02);
        double ScaleNLKV1_12cF = KK * (2 * C(1, 2)*EdotC3_CInvC1 - 2 * C(2, 2)*EdotC2_CInvC1) + KR * (CInv00*CInv01);
        double ScaleNLKV1_13cF = KK * (2 * C(1, 2)*EdotC2_CInvC1 - 2 * C(1, 1)*EdotC3_CInvC1) + KR * (CInv00*CInv02);
        double ScaleNLKV1_23cF = KK * (2 * C(0, 1)*EdotC3_CInvC1 + 2 * C(0, 2)*EdotC2_CInvC1 - 4 * C(1, 2)*EdotC1_CInvC1) + KR * (CInv01*CInv02);

        double ScaleNLKV2_11cF = KK * (4 * C(2, 2)*EdotC2_CInvC2 - 4 * C(1, 2)*EdotC3_CInvC2) + KR * (CInv01*CInv01);
        double ScaleNLKV2_22cF = KK * (0) + KR * (CInv11*CInv11);
        double ScaleNLKV2_33cF = KK * (4 * C(0, 0)*EdotC2_CInvC2 - 4 * C(0, 1)*EdotC1_CInvC2) + KR * (CInv12*CInv12);
        double ScaleNLKV2_12cF = KK * (2 * C(0, 2)*EdotC3_CInvC2 - 2 * C(2, 2)*EdotC1_CInvC2) + KR * (CInv01*CInv11);
        double ScaleNLKV2_13cF = KK * (2 * C(0, 1)*EdotC3_CInvC2 + 2 * C(1, 2)*EdotC1_CInvC2 - 4 * C(0, 2)*EdotC2_CInvC2) + KR * (CInv01*CInv12);
        double ScaleNLKV2_23cF = KK * (2 * C(0, 2)*EdotC1_CInvC2 - 2 * C(0, 0)*EdotC3_CInvC2) + KR * (CInv11*CInv12);

        double ScaleNLKV3_11cF = KK * (4 * C(1, 1)*EdotC3_CInvC3 - 4 * C(1, 2)*EdotC2_CInvC3) + KR * (CInv02*CInv02);
        double ScaleNLKV3_22cF = KK * (4 * C(0, 0)*EdotC3_CInvC3 - 4 * C(0, 2)*EdotC1_CInvC3) + KR * (CInv12*CInv12);
        double ScaleNLKV3_33cF = KK * (0) + KR * (CInv22*CInv22);
        double ScaleNLKV3_12cF = KK * (2 * C(0, 2)*EdotC2_CInvC3 + 2 * C(1, 2)*EdotC1_CInvC3 - 4 * C(0, 1)*EdotC3_CInvC3) + KR * (CInv02*CInv12);
        double ScaleNLKV3_13cF = KK * (2 * C(0, 1)*EdotC2_CInvC3 - 2 * C(1, 1)*EdotC1_CInvC3) + KR * (CInv02*CInv22);
        double ScaleNLKV3_23cF = KK * (2 * C(0, 1)*EdotC1_CInvC3 - 2 * C(0, 0)*EdotC2_CInvC3) + KR * (CInv12*CInv22);

        double ScaleNLKV4_11cF = KK * (2 * C(2, 2)*EdotC2_CInvC3 + 2 * C(1, 1)*EdotC3_CInvC2 - 2 * C(1, 2)*(EdotC2_CInvC2 + EdotC3_CInvC3)) + KR * (CInv01*CInv02);
        double ScaleNLKV4_22cF = KK * (2 * C(0, 0)*EdotC3_CInvC2 - 2 * C(0, 2)*EdotC1_CInvC2) + KR * (CInv11*CInv12);
        double ScaleNLKV4_33cF = KK * (2 * C(0, 0)*EdotC2_CInvC3 - 2 * C(0, 1)*EdotC1_CInvC3) + KR * (CInv12*CInv22);
        double ScaleNLKV4_12cF = KK * (C(0, 2)*(EdotC2_CInvC2 + EdotC3_CInvC3) + C(1, 2) * EdotC1_CInvC2 - 2 * C(0, 1)*EdotC3_CInvC2 - C(2, 2) * EdotC1_CInvC3) + KR * 1 / 2 * (CInv01*CInv12 + CInv11 * CInv02);
        double ScaleNLKV4_13cF = KK * (C(0, 1)*(EdotC2_CInvC2 + EdotC3_CInvC3) + C(1, 2) * EdotC1_CInvC3 - 2 * C(0, 2)*EdotC2_CInvC3 - C(1, 1) * EdotC1_CInvC2) + KR * 1 / 2 * (CInv01*CInv22 + CInv02 * CInv12);
        double ScaleNLKV4_23cF = KK * (C(0, 1)*EdotC1_CInvC2 + C(0, 2) * EdotC1_CInvC3 - C(0, 0) * (EdotC2_CInvC2 + EdotC3_CInvC3)) + KR * 1 / 2 * (CInv12*CInv12 + CInv11 * CInv22);

        double ScaleNLKV5_11cF = KK * (2 * C(1, 1)*EdotC3_CInvC1 - 2 * C(1, 2)*EdotC2_CInvC1) + KR * (CInv00*CInv02);
        double ScaleNLKV5_22cF = KK * (2 * C(0, 0)*EdotC3_CInvC1 + 2 * C(2, 2)*EdotC1_CInvC3 - 2 * C(0, 2)*(EdotC1_CInvC1 + EdotC3_CInvC3)) + KR * (CInv01*CInv12);
        double ScaleNLKV5_33cF = KK * (2 * C(1, 1)*EdotC1_CInvC3 - 2 * C(0, 1)*EdotC2_CInvC3) + KR * (CInv02*CInv22);
        double ScaleNLKV5_12cF = KK * (C(0, 2)*EdotC2_CInvC1 + C(1, 2) * (EdotC1_CInvC1 + EdotC3_CInvC3) - 2 * C(0, 1)*EdotC3_CInvC1 - C(2, 2) * EdotC2_CInvC3) + KR * 1 / 2 * (CInv00*CInv12 + CInv01 * CInv02);
        double ScaleNLKV5_13cF = KK * (C(0, 1)*EdotC2_CInvC1 + C(1, 2) * EdotC2_CInvC3 - C(1, 1) * (EdotC1_CInvC1 + EdotC3_CInvC3)) + KR * 1 / 2 * (CInv00*CInv22 + CInv02 * CInv02);
        double ScaleNLKV5_23cF = KK * (C(0, 1)*(EdotC1_CInvC1 + EdotC3_CInvC3) + C(0, 2) * EdotC2_CInvC3 - 2 * C(1, 2)*EdotC1_CInvC3 - C(0, 0) * EdotC2_CInvC1) + KR * 1 / 2 * (CInv01*CInv22 + CInv02 * CInv12);

        double ScaleNLKV6_11cF = KK * (2 * C(2, 2)*EdotC2_CInvC1 - 2 * C(1, 2)*EdotC3_CInvC1) + KR * (CInv00*CInv01);
        double ScaleNLKV6_22cF = KK * (2 * C(2, 2)*EdotC1_CInvC2 - 2 * C(0, 2)*EdotC3_CInvC2) + KR * (CInv01*CInv11);
        double ScaleNLKV6_33cF = KK * (2 * C(0, 0)*EdotC2_CInvC1 + 2 * C(1, 1)*EdotC1_CInvC2 - 2 * C(0, 1)*(EdotC1_CInvC1 + EdotC2_CInvC2)) + KR * (CInv02*CInv12);
        double ScaleNLKV6_12cF = KK * (C(0, 2)*EdotC3_CInvC1 + C(1, 2) * EdotC3_CInvC2 - C(2, 2) * (EdotC1_CInvC1 + EdotC2_CInvC2)) + KR * 1 / 2 * (CInv00*CInv11 + CInv01 * CInv01);
        double ScaleNLKV6_13cF = KK * (C(0, 1)*EdotC3_CInvC1 + C(1, 2) * (EdotC1_CInvC1 + EdotC2_CInvC2) - 2 * C(0, 2)*EdotC2_CInvC1 - C(1, 1) * EdotC3_CInvC2) + KR * 1 / 2 * (CInv00*CInv12 + CInv01 * CInv02);
        double ScaleNLKV6_23cF = KK * (C(0, 1)*EdotC3_CInvC2 + C(0, 2) * (EdotC1_CInvC1 + EdotC2_CInvC2) - 2 * C(1, 2)*EdotC1_CInvC2 - C(0, 0) * EdotC3_CInvC1) + KR * 1 / 2 * (CInv01*CInv12 + CInv11 * CInv02);

        double ScaleNLKV1_11cFdot = KK * (CInv00*CInv00);
        double ScaleNLKV1_22cFdot = KK * (CInv01*CInv01);
        double ScaleNLKV1_33cFdot = KK * (CInv02*CInv02);
        double ScaleNLKV1_12cFdot = KK * (CInv00*CInv01);
        double ScaleNLKV1_13cFdot = KK * (CInv00*CInv02);
        double ScaleNLKV1_23cFdot = KK * (CInv01*CInv02);

        double ScaleNLKV2_11cFdot = KK * (CInv01*CInv01);
        double ScaleNLKV2_22cFdot = KK * (CInv11*CInv11);
        double ScaleNLKV2_33cFdot = KK * (CInv12*CInv12);
        double ScaleNLKV2_12cFdot = KK * (CInv01*CInv11);
        double ScaleNLKV2_13cFdot = KK * (CInv01*CInv12);
        double ScaleNLKV2_23cFdot = KK * (CInv11*CInv12);

        double ScaleNLKV3_11cFdot = KK * (CInv02*CInv02);
        double ScaleNLKV3_22cFdot = KK * (CInv12*CInv12);
        double ScaleNLKV3_33cFdot = KK * (CInv22*CInv22);
        double ScaleNLKV3_12cFdot = KK * (CInv02*CInv12);
        double ScaleNLKV3_13cFdot = KK * (CInv02*CInv22);
        double ScaleNLKV3_23cFdot = KK * (CInv12*CInv22);

        double ScaleNLKV4_11cFdot = KK * (CInv01*CInv02);
        double ScaleNLKV4_22cFdot = KK * (CInv11*CInv12);
        double ScaleNLKV4_33cFdot = KK * (CInv12*CInv22);
        double ScaleNLKV4_12cFdot = KK * 0.5 * (CInv01*CInv12 + CInv11 * CInv02);
        double ScaleNLKV4_13cFdot = KK * 0.5 * (CInv01*CInv22 + CInv02 * CInv12);
        double ScaleNLKV4_23cFdot = KK * 0.5 * (CInv12*CInv12 + CInv11 * CInv22);

        double ScaleNLKV5_11cFdot = KK * (CInv00*CInv02);
        double ScaleNLKV5_22cFdot = KK * (CInv01*CInv12);
        double ScaleNLKV5_33cFdot = KK * (CInv02*CInv22);
        double ScaleNLKV5_12cFdot = KK * 0.5 * (CInv00*CInv12 + CInv01 * CInv02);
        double ScaleNLKV5_13cFdot = KK * 0.5 * (CInv00*CInv22 + CInv02 * CInv02);
        double ScaleNLKV5_23cFdot = KK * 0.5 * (CInv01*CInv22 + CInv02 * CInv12);

        double ScaleNLKV6_11cFdot = KK * (CInv00*CInv01);
        double ScaleNLKV6_22cFdot = KK * (CInv01*CInv11);
        double ScaleNLKV6_33cFdot = KK * (CInv02*CInv12);
        double ScaleNLKV6_12cFdot = KK * 0.5 * (CInv00*CInv11 + CInv01 * CInv01);
        double ScaleNLKV6_13cFdot = KK * 0.5 * (CInv00*CInv12 + CInv01 * CInv02);
        double ScaleNLKV6_23cFdot = KK * 0.5 * (CInv01*CInv12 + CInv11 * CInv02);

        //ChMatrixNMc<double, 3 * NSF, 6> Partial_SPK2_NLKV_de_combined;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Partial_SPK2_NLKV_de_combined;
        Partial_SPK2_NLKV_de_combined.resize(3 * NSF, 6);

        Eigen::Map<MatrixNx3> Partial_SPK2_NLKV_0_de_combined(Partial_SPK2_NLKV_de_combined.col(0).data(), NSF, 3);
        Partial_SPK2_NLKV_0_de_combined =
            Sbar_xi_D.col(0)*(SPK2_Scale*SPK2_NLKV_0*detF_S0 + ScaleNLKV1_11cFdot * Fdot.col(0).transpose() + ScaleNLKV1_12cFdot * Fdot.col(1).transpose() + ScaleNLKV1_13cFdot * Fdot.col(2).transpose() + ScaleNLKV1_11cF * F.col(0).transpose() + ScaleNLKV1_12cF * F.col(1).transpose() + ScaleNLKV1_13cF * F.col(2).transpose()) +
            Sbar_xi_D.col(1)*(SPK2_Scale*SPK2_NLKV_0*detF_S1 + ScaleNLKV1_12cFdot * Fdot.col(0).transpose() + ScaleNLKV1_22cFdot * Fdot.col(1).transpose() + ScaleNLKV1_23cFdot * Fdot.col(2).transpose() + ScaleNLKV1_12cF * F.col(0).transpose() + ScaleNLKV1_22cF * F.col(1).transpose() + ScaleNLKV1_23cF * F.col(2).transpose()) +
            Sbar_xi_D.col(2)*(SPK2_Scale*SPK2_NLKV_0*detF_S2 + ScaleNLKV1_13cFdot * Fdot.col(0).transpose() + ScaleNLKV1_23cFdot * Fdot.col(1).transpose() + ScaleNLKV1_33cFdot * Fdot.col(2).transpose() + ScaleNLKV1_13cF * F.col(0).transpose() + ScaleNLKV1_23cF * F.col(1).transpose() + ScaleNLKV1_33cF * F.col(2).transpose());

        Eigen::Map<MatrixNx3> Partial_SPK2_NLKV_1_de_combined(Partial_SPK2_NLKV_de_combined.col(1).data(), NSF, 3);
        Partial_SPK2_NLKV_1_de_combined =
            Sbar_xi_D.col(0)*(SPK2_Scale*SPK2_NLKV_1*detF_S0 + ScaleNLKV2_11cFdot * Fdot.col(0).transpose() + ScaleNLKV2_12cFdot * Fdot.col(1).transpose() + ScaleNLKV2_13cFdot * Fdot.col(2).transpose() + ScaleNLKV2_11cF * F.col(0).transpose() + ScaleNLKV2_12cF * F.col(1).transpose() + ScaleNLKV2_13cF * F.col(2).transpose()) +
            Sbar_xi_D.col(1)*(SPK2_Scale*SPK2_NLKV_1*detF_S1 + ScaleNLKV2_12cFdot * Fdot.col(0).transpose() + ScaleNLKV2_22cFdot * Fdot.col(1).transpose() + ScaleNLKV2_23cFdot * Fdot.col(2).transpose() + ScaleNLKV2_12cF * F.col(0).transpose() + ScaleNLKV2_22cF * F.col(1).transpose() + ScaleNLKV2_23cF * F.col(2).transpose()) +
            Sbar_xi_D.col(2)*(SPK2_Scale*SPK2_NLKV_1*detF_S2 + ScaleNLKV2_13cFdot * Fdot.col(0).transpose() + ScaleNLKV2_23cFdot * Fdot.col(1).transpose() + ScaleNLKV2_33cFdot * Fdot.col(2).transpose() + ScaleNLKV2_13cF * F.col(0).transpose() + ScaleNLKV2_23cF * F.col(1).transpose() + ScaleNLKV2_33cF * F.col(2).transpose());

        Eigen::Map<MatrixNx3> Partial_SPK2_NLKV_2_de_combined(Partial_SPK2_NLKV_de_combined.col(2).data(), NSF, 3);
        Partial_SPK2_NLKV_2_de_combined =
            Sbar_xi_D.col(0)*(SPK2_Scale*SPK2_NLKV_2*detF_S0 + ScaleNLKV3_11cFdot * Fdot.col(0).transpose() + ScaleNLKV3_12cFdot * Fdot.col(1).transpose() + ScaleNLKV3_13cFdot * Fdot.col(2).transpose() + ScaleNLKV3_11cF * F.col(0).transpose() + ScaleNLKV3_12cF * F.col(1).transpose() + ScaleNLKV3_13cF * F.col(2).transpose()) +
            Sbar_xi_D.col(1)*(SPK2_Scale*SPK2_NLKV_2*detF_S1 + ScaleNLKV3_12cFdot * Fdot.col(0).transpose() + ScaleNLKV3_22cFdot * Fdot.col(1).transpose() + ScaleNLKV3_23cFdot * Fdot.col(2).transpose() + ScaleNLKV3_12cF * F.col(0).transpose() + ScaleNLKV3_22cF * F.col(1).transpose() + ScaleNLKV3_23cF * F.col(2).transpose()) +
            Sbar_xi_D.col(2)*(SPK2_Scale*SPK2_NLKV_2*detF_S2 + ScaleNLKV3_13cFdot * Fdot.col(0).transpose() + ScaleNLKV3_23cFdot * Fdot.col(1).transpose() + ScaleNLKV3_33cFdot * Fdot.col(2).transpose() + ScaleNLKV3_13cF * F.col(0).transpose() + ScaleNLKV3_23cF * F.col(1).transpose() + ScaleNLKV3_33cF * F.col(2).transpose());

        Eigen::Map<MatrixNx3> Partial_SPK2_NLKV_3_de_combined(Partial_SPK2_NLKV_de_combined.col(3).data(), NSF, 3);
        Partial_SPK2_NLKV_3_de_combined =
            Sbar_xi_D.col(0)*(SPK2_Scale*SPK2_NLKV_3*detF_S0 + ScaleNLKV4_11cFdot * Fdot.col(0).transpose() + ScaleNLKV4_12cFdot * Fdot.col(1).transpose() + ScaleNLKV4_13cFdot * Fdot.col(2).transpose() + ScaleNLKV4_11cF * F.col(0).transpose() + ScaleNLKV4_12cF * F.col(1).transpose() + ScaleNLKV4_13cF * F.col(2).transpose()) +
            Sbar_xi_D.col(1)*(SPK2_Scale*SPK2_NLKV_3*detF_S1 + ScaleNLKV4_12cFdot * Fdot.col(0).transpose() + ScaleNLKV4_22cFdot * Fdot.col(1).transpose() + ScaleNLKV4_23cFdot * Fdot.col(2).transpose() + ScaleNLKV4_12cF * F.col(0).transpose() + ScaleNLKV4_22cF * F.col(1).transpose() + ScaleNLKV4_23cF * F.col(2).transpose()) +
            Sbar_xi_D.col(2)*(SPK2_Scale*SPK2_NLKV_3*detF_S2 + ScaleNLKV4_13cFdot * Fdot.col(0).transpose() + ScaleNLKV4_23cFdot * Fdot.col(1).transpose() + ScaleNLKV4_33cFdot * Fdot.col(2).transpose() + ScaleNLKV4_13cF * F.col(0).transpose() + ScaleNLKV4_23cF * F.col(1).transpose() + ScaleNLKV4_33cF * F.col(2).transpose());

        Eigen::Map<MatrixNx3> Partial_SPK2_NLKV_4_de_combined(Partial_SPK2_NLKV_de_combined.col(4).data(), NSF, 3);
        Partial_SPK2_NLKV_4_de_combined =
            Sbar_xi_D.col(0)*(SPK2_Scale*SPK2_NLKV_4*detF_S0 + ScaleNLKV5_11cFdot * Fdot.col(0).transpose() + ScaleNLKV5_12cFdot * Fdot.col(1).transpose() + ScaleNLKV5_13cFdot * Fdot.col(2).transpose() + ScaleNLKV5_11cF * F.col(0).transpose() + ScaleNLKV5_12cF * F.col(1).transpose() + ScaleNLKV5_13cF * F.col(2).transpose()) +
            Sbar_xi_D.col(1)*(SPK2_Scale*SPK2_NLKV_4*detF_S1 + ScaleNLKV5_12cFdot * Fdot.col(0).transpose() + ScaleNLKV5_22cFdot * Fdot.col(1).transpose() + ScaleNLKV5_23cFdot * Fdot.col(2).transpose() + ScaleNLKV5_12cF * F.col(0).transpose() + ScaleNLKV5_22cF * F.col(1).transpose() + ScaleNLKV5_23cF * F.col(2).transpose()) +
            Sbar_xi_D.col(2)*(SPK2_Scale*SPK2_NLKV_4*detF_S2 + ScaleNLKV5_13cFdot * Fdot.col(0).transpose() + ScaleNLKV5_23cFdot * Fdot.col(1).transpose() + ScaleNLKV5_33cFdot * Fdot.col(2).transpose() + ScaleNLKV5_13cF * F.col(0).transpose() + ScaleNLKV5_23cF * F.col(1).transpose() + ScaleNLKV5_33cF * F.col(2).transpose());

        Eigen::Map<MatrixNx3> Partial_SPK2_NLKV_5_de_combined(Partial_SPK2_NLKV_de_combined.col(5).data(), NSF, 3);
        Partial_SPK2_NLKV_5_de_combined =
            Sbar_xi_D.col(0)*(SPK2_Scale*SPK2_NLKV_5*detF_S0 + ScaleNLKV6_11cFdot * Fdot.col(0).transpose() + ScaleNLKV6_12cFdot * Fdot.col(1).transpose() + ScaleNLKV6_13cFdot * Fdot.col(2).transpose() + ScaleNLKV6_11cF * F.col(0).transpose() + ScaleNLKV6_12cF * F.col(1).transpose() + ScaleNLKV6_13cF * F.col(2).transpose()) +
            Sbar_xi_D.col(1)*(SPK2_Scale*SPK2_NLKV_5*detF_S1 + ScaleNLKV6_12cFdot * Fdot.col(0).transpose() + ScaleNLKV6_22cFdot * Fdot.col(1).transpose() + ScaleNLKV6_23cFdot * Fdot.col(2).transpose() + ScaleNLKV6_12cF * F.col(0).transpose() + ScaleNLKV6_22cF * F.col(1).transpose() + ScaleNLKV6_23cF * F.col(2).transpose()) +
            Sbar_xi_D.col(2)*(SPK2_Scale*SPK2_NLKV_5*detF_S2 + ScaleNLKV6_13cFdot * Fdot.col(0).transpose() + ScaleNLKV6_23cFdot * Fdot.col(1).transpose() + ScaleNLKV6_33cFdot * Fdot.col(2).transpose() + ScaleNLKV6_13cF * F.col(0).transpose() + ScaleNLKV6_23cF * F.col(1).transpose() + ScaleNLKV6_33cF * F.col(2).transpose());

        H.noalias() += Left.block<3 * NSF, 6>(0, 0) * Partial_SPK2_NLKV_de_combined.transpose();
    }

    for (unsigned int i = 0; i < NSF; i++) {
        for (unsigned int j = 0; j < NSF; j++) {
            int idx = (j >= i) ? ((NSF*(NSF - 1) - (NSF - i)*(NSF - i - 1)) / 2 + j) : ((NSF*(NSF - 1) - (NSF - j)*(NSF - j - 1)) / 2 + i);
            H(3 * i, 3 * j) += JacCompact(i, j) + Mfactor * m_MassMatrix(idx);
            H(3 * i, 3 * j + 1) += FC(i, j);
            H(3 * i, 3 * j + 2) -= FB(i, j);

            H(3 * i + 1, 3 * j) -= FC(i, j);
            H(3 * i + 1, 3 * j + 1) += JacCompact(i, j) + Mfactor * m_MassMatrix(idx);
            H(3 * i + 1, 3 * j + 2) += FA(i, j);

            H(3 * i + 2, 3 * j) += FB(i, j);
            H(3 * i + 2, 3 * j + 1) -= FA(i, j);
            H(3 * i + 2, 3 * j + 2) += JacCompact(i, j) + Mfactor * m_MassMatrix(idx);
        }
    }

}

void ChElementHexaANCF_3843_MR_V5::ComputeInternalJacobianNoDamping(ChMatrixRef H, double Kfactor, double Mfactor) {
    assert((H.rows() == 3 * NSF) && (H.cols() == 3 * NSF));

    // The integrated quantity represents the 96x96 Jacobian
//      Kfactor * [K] + Rfactor * [R]
// Note that the matrices with current nodal coordinates and velocities are
// already available in m_d and m_d_dt (as set in ComputeInternalForces).
// Similarly, the ANS strain and strain derivatives are already available in
// m_strainANS and m_strainANS_D (as calculated in ComputeInternalForces).

    // Element coordinates in matrix form
    Matrix3xN ebar;
    CalcCoordMatrix(ebar);

    // Element coordinate time derivatives in matrix form
    Matrix3xN ebardot;
    CalcCoordDerivMatrix(ebardot);

    H.setZero();

    double c10 = GetMaterial()->Get_c10();
    double c01 = GetMaterial()->Get_c01();
    double k = GetMaterial()->Get_k();

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> JacCompact;
    JacCompact.resize(NSF, NSF);
    JacCompact.setZero();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> FA;
    FA.resize(NSF, NSF);
    FA.setZero();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> FB;
    FB.resize(NSF, NSF);
    FB.setZero();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> FC;
    FC.resize(NSF, NSF);
    FC.setZero();

    //Calculate the contribution from each Gauss-quadrature point, point by point
    for (unsigned int GQpnt = 0; GQpnt < NIP; GQpnt++) {
        MatrixNx3c Sbar_xi_D = m_SD.block<NSF, 3>(0, 3 * GQpnt);

        // Calculate the Deformation Gradient at the current point
        ChMatrixNMc<double, 3, 3> F = ebar * Sbar_xi_D;

        //Calculate the Right Cauchy-Green deformation tensor (symmetric):
        ChMatrix33<> C = F.transpose()*F;

        //Calculate the 1st invariant of the Right Cauchy-Green deformation tensor: trace(C)
        double I1 = C(0, 0) + C(1, 1) + C(2, 2);

        //Calculate the 2nd invariant of the Right Cauchy-Green deformation tensor: 1/2*(trace(C)^2-trace(C^2))
        double I2 = 0.5*(I1*I1 - C(0, 0)*C(0, 0) - C(1, 1)*C(1, 1) - C(2, 2)*C(2, 2)) - C(0, 1)*C(0, 1) - C(0, 2)*C(0, 2) - C(1, 2)*C(1, 2);

        //Calculate the determinate of F (commonly denoted as J)
        double detF = F(0, 0)*F(1, 1)*F(2, 2) + F(0, 1)*F(1, 2)*F(2, 0) + F(0, 2)*F(1, 0)*F(2, 1)
            - F(0, 0)*F(1, 2)*F(2, 1) - F(0, 1)*F(1, 0)*F(2, 2) - F(0, 2)*F(1, 1)*F(2, 0);

        double detF_m2_3 = std::pow(detF, -2.0 / 3.0);       // detF^(-2/3)

        double scale1 = -Kfactor * m_kGQ(GQpnt) * 2.0 * c10 * detF_m2_3;
        double scale2 = -Kfactor * m_kGQ(GQpnt) * 2.0 * c01 * detF_m2_3 * detF_m2_3;
        double scale3 = scale1 + scale2 * I1;
        double scale4 = -Kfactor * m_kGQ(GQpnt) * k * (detF - 1.0) - (scale1*I1 + 2.0 * scale2*I2) / (3.0 * detF);
        double scale5 = 0.5*((5 * I1*scale1 + 14 * I2*scale2) / (9 * detF*detF) - Kfactor * m_kGQ(GQpnt) * k);
        double scale6 = -2 * (scale1 + 2 * I1*scale2) / (3 * detF);
        double scale7 = 4 * scale2 / (3 * detF);

        ChMatrixNM<double, 1, 3> detF_S0;
        detF_S0 << F(1, 1)*F(2, 2) - F(1, 2)*F(2, 1), F(0, 2)*F(2, 1) - F(0, 1)*F(2, 2), F(0, 1)*F(1, 2) - F(0, 2)*F(1, 1);
        ChMatrixNM<double, 1, 3> detF_S1;
        detF_S1 << F(1, 2)*F(2, 0) - F(1, 0)*F(2, 2), F(0, 0)*F(2, 2) - F(0, 2)*F(2, 0), F(0, 2)*F(1, 0) - F(0, 0)*F(1, 2);
        ChMatrixNM<double, 1, 3> detF_S2;
        detF_S2 << F(1, 0)*F(2, 1) - F(1, 1)*F(2, 0), F(0, 1)*F(2, 0) - F(0, 0)*F(2, 1), F(0, 0)*F(1, 1) - F(0, 1)*F(1, 0);

        ChMatrixNMc<double, 3 * NSF, 9> Left;
        //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Left;
        //Left.resize(3 * NSF, 9);
        Eigen::Map<MatrixNx3> Left_HalfdC00_de_T_Compact(Left.col(0).data(), NSF, 3);
        Eigen::Map<MatrixNx3> Left_HalfdC11_de_T_Compact(Left.col(1).data(), NSF, 3);
        Eigen::Map<MatrixNx3> Left_HalfdC22_de_T_Compact(Left.col(2).data(), NSF, 3);
        Eigen::Map<MatrixNx3> Left_dC12_de_T_Compact(Left.col(3).data(), NSF, 3);
        Eigen::Map<MatrixNx3> Left_dC02_de_T_Compact(Left.col(4).data(), NSF, 3);
        Eigen::Map<MatrixNx3> Left_dC01_de_T_Compact(Left.col(5).data(), NSF, 3);
        Eigen::Map<MatrixNx3> Left_dI1_de_T_Compact(Left.col(6).data(), NSF, 3);
        Eigen::Map<MatrixNx3> Left_Combined_Compact(Left.col(7).data(), NSF, 3);
        Eigen::Map<MatrixNx3> Left_ddetF_de_T_Compact(Left.col(8).data(), NSF, 3);

        Left_HalfdC00_de_T_Compact = Sbar_xi_D.col(0)*(F.col(0).transpose());
        Left_HalfdC11_de_T_Compact = Sbar_xi_D.col(1)*(F.col(1).transpose());
        Left_HalfdC22_de_T_Compact = Sbar_xi_D.col(2)*(F.col(2).transpose());
        Left_dC12_de_T_Compact = Sbar_xi_D.col(1)*(F.col(2).transpose()) + Sbar_xi_D.col(2)*(F.col(1).transpose());
        Left_dC02_de_T_Compact = Sbar_xi_D.col(0)*(F.col(2).transpose()) + Sbar_xi_D.col(2)*(F.col(0).transpose());
        Left_dC01_de_T_Compact = Sbar_xi_D.col(0)*(F.col(1).transpose()) + Sbar_xi_D.col(1)*(F.col(0).transpose());

        Left_dI1_de_T_Compact = Sbar_xi_D.col(0)*(2 * F.col(0).transpose()) +
            Sbar_xi_D.col(1)*(2 * F.col(1).transpose()) +
            Sbar_xi_D.col(2)*(2 * F.col(2).transpose());

        Left_Combined_Compact = Sbar_xi_D.col(0)*(scale6*F.col(0).transpose() + scale7 * C.row(0)*F.transpose() + scale5 * detF_S0) +
            Sbar_xi_D.col(1)*(scale6*F.col(1).transpose() + scale7 * C.row(1)*F.transpose() + scale5 * detF_S1) +
            Sbar_xi_D.col(2)*(scale6*F.col(2).transpose() + scale7 * C.row(2)*F.transpose() + scale5 * detF_S2);

        Left_ddetF_de_T_Compact = Sbar_xi_D.col(0)*detF_S0 + Sbar_xi_D.col(1)*detF_S1 + Sbar_xi_D.col(2)*detF_S2;

        ChMatrixNMc<double, 3 * NSF, 9> Right;
        //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Right;
        //Right.resize(3 * NSF, 9);
        Right.col(0) = (-2.0*scale2) * Left.col(0);
        Right.col(1) = (-2.0*scale2) * Left.col(1);
        Right.col(2) = (-2.0*scale2) * Left.col(2);
        Right.col(3) = -scale2 * Left.col(3);
        Right.col(4) = -scale2 * Left.col(4);
        Right.col(5) = -scale2 * Left.col(5);
        Right.col(6) = 0.5*scale2*Left.col(6);
        Right.col(7) = Left.col(8);
        Right.col(8) = Left.col(7);

        //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> LeftRight;
        //LeftRight = Left * Right.transpose();

        H.noalias() += Left * Right.transpose();

        JacCompact.noalias() += Sbar_xi_D.col(0)*(scale3*Sbar_xi_D.col(0).transpose() - scale2 * C.row(0)*Sbar_xi_D.transpose())
            + Sbar_xi_D.col(1)*(scale3*Sbar_xi_D.col(1).transpose() - scale2 * C.row(1)*Sbar_xi_D.transpose())
            + Sbar_xi_D.col(2)*(scale3*Sbar_xi_D.col(2).transpose() - scale2 * C.row(2)*Sbar_xi_D.transpose());

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> FA_Mult;
        FA_Mult.resize(NSF, 3);
        FA_Mult.col(0) = (scale4 * F(0, 2))*Sbar_xi_D.col(1) - (scale4 * F(0, 1))*Sbar_xi_D.col(2);
        FA_Mult.col(1) = (scale4 * F(0, 0))*Sbar_xi_D.col(2) - (scale4 * F(0, 2))*Sbar_xi_D.col(0);
        FA_Mult.col(2) = (scale4 * F(0, 1))*Sbar_xi_D.col(0) - (scale4 * F(0, 0))*Sbar_xi_D.col(1);
        FA.noalias() += Sbar_xi_D * FA_Mult.transpose();

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> FB_Mult;
        FB_Mult.resize(NSF, 3);
        FB_Mult.col(0) = (scale4 * F(1, 2))*Sbar_xi_D.col(1) - (scale4 * F(1, 1))*Sbar_xi_D.col(2);
        FB_Mult.col(1) = (scale4 * F(1, 0))*Sbar_xi_D.col(2) - (scale4 * F(1, 2))*Sbar_xi_D.col(0);
        FB_Mult.col(2) = (scale4 * F(1, 1))*Sbar_xi_D.col(0) - (scale4 * F(1, 0))*Sbar_xi_D.col(1);
        FB.noalias() += Sbar_xi_D * FB_Mult.transpose();

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> FC_Mult;
        FC_Mult.resize(NSF, 3);
        FC_Mult.col(0) = (scale4 * F(2, 2))*Sbar_xi_D.col(1) - (scale4 * F(2, 1))*Sbar_xi_D.col(2);
        FC_Mult.col(1) = (scale4 * F(2, 0))*Sbar_xi_D.col(2) - (scale4 * F(2, 2))*Sbar_xi_D.col(0);
        FC_Mult.col(2) = (scale4 * F(2, 1))*Sbar_xi_D.col(0) - (scale4 * F(2, 0))*Sbar_xi_D.col(1);
        FC.noalias() += Sbar_xi_D * FC_Mult.transpose();
    }

    for (unsigned int i = 0; i < NSF; i++) {
        for (unsigned int j = 0; j < NSF; j++) {
            int idx = (j >= i) ? ((NSF*(NSF - 1) - (NSF - i)*(NSF - i - 1)) / 2 + j) : ((NSF*(NSF - 1) - (NSF - j)*(NSF - j - 1)) / 2 + i);
            H(3 * i, 3 * j) += JacCompact(i, j) + Mfactor * m_MassMatrix(idx);
            H(3 * i, 3 * j + 1) += FC(i, j);
            H(3 * i, 3 * j + 2) -= FB(i, j);

            H(3 * i + 1, 3 * j) -= FC(i, j);
            H(3 * i + 1, 3 * j + 1) += JacCompact(i, j) + Mfactor * m_MassMatrix(idx);
            H(3 * i + 1, 3 * j + 2) += FA(i, j);

            H(3 * i + 2, 3 * j) += FB(i, j);
            H(3 * i + 2, 3 * j + 1) -= FA(i, j);
            H(3 * i + 2, 3 * j + 2) += JacCompact(i, j) + Mfactor * m_MassMatrix(idx);
        }
    }
}


// -----------------------------------------------------------------------------
// Shape functions
// -----------------------------------------------------------------------------

// Nx1 Vector Form of the Normalized Shape Functions
// [s1; s2; s3; ...]

void ChElementHexaANCF_3843_MR_V5::Calc_Sxi_compact(VectorN& Sxi_compact, double xi, double eta, double zeta) {
    Sxi_compact(0) =
        0.0625 * (zeta - 1) * (xi - 1) * (eta - 1) * (eta * eta + eta + xi * xi + xi + zeta * zeta + zeta - 2);
    Sxi_compact(1) = 0.03125 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (zeta - 1) * (eta - 1);
    Sxi_compact(2) = 0.03125 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (zeta - 1) * (xi - 1);
    Sxi_compact(3) = 0.03125 * m_lenZ * (zeta + 1) * (zeta - 1) * (zeta - 1) * (xi - 1) * (eta - 1);
    Sxi_compact(4) =
        -0.0625 * (zeta - 1) * (xi + 1) * (eta - 1) * (eta * eta + eta + xi * xi - xi + zeta * zeta + zeta - 2);
    Sxi_compact(5) = 0.03125 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (zeta - 1) * (eta - 1);
    Sxi_compact(6) = -0.03125 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (zeta - 1) * (xi + 1);
    Sxi_compact(7) = -0.03125 * m_lenZ * (zeta + 1) * (zeta - 1) * (zeta - 1) * (xi + 1) * (eta - 1);
    Sxi_compact(8) =
        0.0625 * (zeta - 1) * (xi + 1) * (eta + 1) * (eta * eta - eta + xi * xi - xi + zeta * zeta + zeta - 2);
    Sxi_compact(9) = -0.03125 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (zeta - 1) * (eta + 1);
    Sxi_compact(10) = -0.03125 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (zeta - 1) * (xi + 1);
    Sxi_compact(11) = 0.03125 * m_lenZ * (zeta + 1) * (zeta - 1) * (zeta - 1) * (xi + 1) * (eta + 1);
    Sxi_compact(12) =
        -0.0625 * (zeta - 1) * (xi - 1) * (eta + 1) * (eta * eta - eta + xi * xi + xi + zeta * zeta + zeta - 2);
    Sxi_compact(13) = -0.03125 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (zeta - 1) * (eta + 1);
    Sxi_compact(14) = 0.03125 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (zeta - 1) * (xi - 1);
    Sxi_compact(15) = -0.03125 * m_lenZ * (zeta + 1) * (zeta - 1) * (zeta - 1) * (xi - 1) * (eta + 1);
    Sxi_compact(16) =
        -0.0625 * (zeta + 1) * (xi - 1) * (eta - 1) * (eta * eta + eta + xi * xi + xi + zeta * zeta - zeta - 2);
    Sxi_compact(17) = -0.03125 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (zeta + 1) * (eta - 1);
    Sxi_compact(18) = -0.03125 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (zeta + 1) * (xi - 1);
    Sxi_compact(19) = 0.03125 * m_lenZ * (zeta - 1) * (zeta + 1) * (zeta + 1) * (xi - 1) * (eta - 1);
    Sxi_compact(20) =
        0.0625 * (zeta + 1) * (xi + 1) * (eta - 1) * (eta * eta + eta + xi * xi - xi + zeta * zeta - zeta - 2);
    Sxi_compact(21) = -0.03125 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (zeta + 1) * (eta - 1);
    Sxi_compact(22) = 0.03125 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (zeta + 1) * (xi + 1);
    Sxi_compact(23) = -0.03125 * m_lenZ * (zeta - 1) * (zeta + 1) * (zeta + 1) * (xi + 1) * (eta - 1);
    Sxi_compact(24) =
        -0.0625 * (zeta + 1) * (xi + 1) * (eta + 1) * (eta * eta - eta + xi * xi - xi + zeta * zeta - zeta - 2);
    Sxi_compact(25) = 0.03125 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (zeta + 1) * (eta + 1);
    Sxi_compact(26) = 0.03125 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (zeta + 1) * (xi + 1);
    Sxi_compact(27) = 0.03125 * m_lenZ * (zeta - 1) * (zeta + 1) * (zeta + 1) * (xi + 1) * (eta + 1);
    Sxi_compact(28) =
        0.0625 * (zeta + 1) * (xi - 1) * (eta + 1) * (eta * eta - eta + xi * xi + xi + zeta * zeta - zeta - 2);
    Sxi_compact(29) = 0.03125 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (zeta + 1) * (eta + 1);
    Sxi_compact(30) = -0.03125 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (zeta + 1) * (xi - 1);
    Sxi_compact(31) = -0.03125 * m_lenZ * (zeta - 1) * (zeta + 1) * (zeta + 1) * (xi - 1) * (eta + 1);
}

// Nx1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to xi
// [s1; s2; s3; ...]

void ChElementHexaANCF_3843_MR_V5::Calc_Sxi_xi_compact(VectorN& Sxi_xi_compact, double xi, double eta, double zeta) {
    Sxi_xi_compact(0) = 0.0625 * (zeta - 1) * (eta - 1) * (eta * eta + eta + 3 * xi * xi + zeta * zeta + zeta - 3);
    Sxi_xi_compact(1) = 0.03125 * m_lenX * (3 * xi + 1) * (xi - 1) * (zeta - 1) * (eta - 1);
    Sxi_xi_compact(2) = 0.03125 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (zeta - 1);
    Sxi_xi_compact(3) = 0.03125 * m_lenZ * (zeta + 1) * (zeta - 1) * (zeta - 1) * (eta - 1);
    Sxi_xi_compact(4) = -0.0625 * (zeta - 1) * (eta - 1) * (eta * eta + eta + 3 * xi * xi + zeta * zeta + zeta - 3);
    Sxi_xi_compact(5) = 0.03125 * m_lenX * (xi + 1) * (3 * xi - 1) * (zeta - 1) * (eta - 1);
    Sxi_xi_compact(6) = -0.03125 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (zeta - 1);
    Sxi_xi_compact(7) = -0.03125 * m_lenZ * (zeta + 1) * (zeta - 1) * (zeta - 1) * (eta - 1);
    Sxi_xi_compact(8) = 0.0625 * (zeta - 1) * (eta + 1) * (eta * eta - eta + 3 * xi * xi + zeta * zeta + zeta - 3);
    Sxi_xi_compact(9) = -0.03125 * m_lenX * (xi + 1) * (3 * xi - 1) * (zeta - 1) * (eta + 1);
    Sxi_xi_compact(10) = -0.03125 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (zeta - 1);
    Sxi_xi_compact(11) = 0.03125 * m_lenZ * (zeta + 1) * (zeta - 1) * (zeta - 1) * (eta + 1);
    Sxi_xi_compact(12) = -0.0625 * (zeta - 1) * (eta + 1) * (eta * eta - eta + 3 * xi * xi + zeta * zeta + zeta - 3);
    Sxi_xi_compact(13) = -0.03125 * m_lenX * (3 * xi + 1) * (xi - 1) * (zeta - 1) * (eta + 1);
    Sxi_xi_compact(14) = 0.03125 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (zeta - 1);
    Sxi_xi_compact(15) = -0.03125 * m_lenZ * (zeta + 1) * (zeta - 1) * (zeta - 1) * (eta + 1);
    Sxi_xi_compact(16) = -0.0625 * (zeta + 1) * (eta - 1) * (eta * eta + eta + 3 * xi * xi + zeta * zeta - zeta - 3);
    Sxi_xi_compact(17) = -0.03125 * m_lenX * (3 * xi + 1) * (xi - 1) * (zeta + 1) * (eta - 1);
    Sxi_xi_compact(18) = -0.03125 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (zeta + 1);
    Sxi_xi_compact(19) = 0.03125 * m_lenZ * (zeta - 1) * (zeta + 1) * (zeta + 1) * (eta - 1);
    Sxi_xi_compact(20) = 0.0625 * (zeta + 1) * (eta - 1) * (eta * eta + eta + 3 * xi * xi + zeta * zeta - zeta - 3);
    Sxi_xi_compact(21) = -0.03125 * m_lenX * (xi + 1) * (3 * xi - 1) * (zeta + 1) * (eta - 1);
    Sxi_xi_compact(22) = 0.03125 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (zeta + 1);
    Sxi_xi_compact(23) = -0.03125 * m_lenZ * (zeta - 1) * (zeta + 1) * (zeta + 1) * (eta - 1);
    Sxi_xi_compact(24) = -0.0625 * (zeta + 1) * (eta + 1) * (eta * eta - eta + 3 * xi * xi + zeta * zeta - zeta - 3);
    Sxi_xi_compact(25) = 0.03125 * m_lenX * (xi + 1) * (3 * xi - 1) * (zeta + 1) * (eta + 1);
    Sxi_xi_compact(26) = 0.03125 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (zeta + 1);
    Sxi_xi_compact(27) = 0.03125 * m_lenZ * (zeta - 1) * (zeta + 1) * (zeta + 1) * (eta + 1);
    Sxi_xi_compact(28) = 0.0625 * (zeta + 1) * (eta + 1) * (eta * eta - eta + 3 * xi * xi + zeta * zeta - zeta - 3);
    Sxi_xi_compact(29) = 0.03125 * m_lenX * (3 * xi + 1) * (xi - 1) * (zeta + 1) * (eta + 1);
    Sxi_xi_compact(30) = -0.03125 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (zeta + 1);
    Sxi_xi_compact(31) = -0.03125 * m_lenZ * (zeta - 1) * (zeta + 1) * (zeta + 1) * (eta + 1);
}

// Nx1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to eta
// [s1; s2; s3; ...]

void ChElementHexaANCF_3843_MR_V5::Calc_Sxi_eta_compact(VectorN& Sxi_eta_compact, double xi, double eta, double zeta) {
    Sxi_eta_compact(0) = 0.0625 * (zeta - 1) * (xi - 1) * (3 * eta * eta + xi * xi + xi + zeta * zeta + zeta - 3);
    Sxi_eta_compact(1) = 0.03125 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (zeta - 1);
    Sxi_eta_compact(2) = 0.03125 * m_lenY * (3 * eta + 1) * (eta - 1) * (zeta - 1) * (xi - 1);
    Sxi_eta_compact(3) = 0.03125 * m_lenZ * (zeta + 1) * (zeta - 1) * (zeta - 1) * (xi - 1);
    Sxi_eta_compact(4) = -0.0625 * (zeta - 1) * (xi + 1) * (3 * eta * eta + xi * xi - xi + zeta * zeta + zeta - 3);
    Sxi_eta_compact(5) = 0.03125 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (zeta - 1);
    Sxi_eta_compact(6) = -0.03125 * m_lenY * (3 * eta + 1) * (eta - 1) * (zeta - 1) * (xi + 1);
    Sxi_eta_compact(7) = -0.03125 * m_lenZ * (zeta + 1) * (zeta - 1) * (zeta - 1) * (xi + 1);
    Sxi_eta_compact(8) = 0.0625 * (zeta - 1) * (xi + 1) * (3 * eta * eta + xi * xi - xi + zeta * zeta + zeta - 3);
    Sxi_eta_compact(9) = -0.03125 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (zeta - 1);
    Sxi_eta_compact(10) = -0.03125 * m_lenY * (eta + 1) * (3 * eta - 1) * (zeta - 1) * (xi + 1);
    Sxi_eta_compact(11) = 0.03125 * m_lenZ * (zeta + 1) * (zeta - 1) * (zeta - 1) * (xi + 1);
    Sxi_eta_compact(12) = -0.0625 * (zeta - 1) * (xi - 1) * (3 * eta * eta + xi * xi + xi + zeta * zeta + zeta - 3);
    Sxi_eta_compact(13) = -0.03125 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (zeta - 1);
    Sxi_eta_compact(14) = 0.03125 * m_lenY * (eta + 1) * (3 * eta - 1) * (zeta - 1) * (xi - 1);
    Sxi_eta_compact(15) = -0.03125 * m_lenZ * (zeta + 1) * (zeta - 1) * (zeta - 1) * (xi - 1);
    Sxi_eta_compact(16) = -0.0625 * (zeta + 1) * (xi - 1) * (3 * eta * eta + xi * xi + xi + zeta * zeta - zeta - 3);
    Sxi_eta_compact(17) = -0.03125 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (zeta + 1);
    Sxi_eta_compact(18) = -0.03125 * m_lenY * (3 * eta + 1) * (eta - 1) * (zeta + 1) * (xi - 1);
    Sxi_eta_compact(19) = 0.03125 * m_lenZ * (zeta - 1) * (zeta + 1) * (zeta + 1) * (xi - 1);
    Sxi_eta_compact(20) = 0.0625 * (zeta + 1) * (xi + 1) * (3 * eta * eta + xi * xi - xi + zeta * zeta - zeta - 3);
    Sxi_eta_compact(21) = -0.03125 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (zeta + 1);
    Sxi_eta_compact(22) = 0.03125 * m_lenY * (3 * eta + 1) * (eta - 1) * (zeta + 1) * (xi + 1);
    Sxi_eta_compact(23) = -0.03125 * m_lenZ * (zeta - 1) * (zeta + 1) * (zeta + 1) * (xi + 1);
    Sxi_eta_compact(24) = -0.0625 * (zeta + 1) * (xi + 1) * (3 * eta * eta + xi * xi - xi + zeta * zeta - zeta - 3);
    Sxi_eta_compact(25) = 0.03125 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (zeta + 1);
    Sxi_eta_compact(26) = 0.03125 * m_lenY * (eta + 1) * (3 * eta - 1) * (zeta + 1) * (xi + 1);
    Sxi_eta_compact(27) = 0.03125 * m_lenZ * (zeta - 1) * (zeta + 1) * (zeta + 1) * (xi + 1);
    Sxi_eta_compact(28) = 0.0625 * (zeta + 1) * (xi - 1) * (3 * eta * eta + xi * xi + xi + zeta * zeta - zeta - 3);
    Sxi_eta_compact(29) = 0.03125 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (zeta + 1);
    Sxi_eta_compact(30) = -0.03125 * m_lenY * (eta + 1) * (3 * eta - 1) * (zeta + 1) * (xi - 1);
    Sxi_eta_compact(31) = -0.03125 * m_lenZ * (zeta - 1) * (zeta + 1) * (zeta + 1) * (xi - 1);
}

// Nx1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to zeta
// [s1; s2; s3; ...]

void ChElementHexaANCF_3843_MR_V5::Calc_Sxi_zeta_compact(VectorN& Sxi_zeta_compact, double xi, double eta, double zeta) {
    Sxi_zeta_compact(0) = 0.0625 * (xi - 1) * (eta - 1) * (eta * eta + eta + xi * xi + xi + 3 * zeta * zeta - 3);
    Sxi_zeta_compact(1) = 0.03125 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (eta - 1);
    Sxi_zeta_compact(2) = 0.03125 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (xi - 1);
    Sxi_zeta_compact(3) = 0.03125 * m_lenZ * (3 * zeta + 1) * (zeta - 1) * (xi - 1) * (eta - 1);
    Sxi_zeta_compact(4) = -0.0625 * (xi + 1) * (eta - 1) * (eta * eta + eta + xi * xi - xi + 3 * zeta * zeta - 3);
    Sxi_zeta_compact(5) = 0.03125 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (eta - 1);
    Sxi_zeta_compact(6) = -0.03125 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (xi + 1);
    Sxi_zeta_compact(7) = -0.03125 * m_lenZ * (3 * zeta + 1) * (zeta - 1) * (xi + 1) * (eta - 1);
    Sxi_zeta_compact(8) = 0.0625 * (xi + 1) * (eta + 1) * (eta * eta - eta + xi * xi - xi + 3 * zeta * zeta - 3);
    Sxi_zeta_compact(9) = -0.03125 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (eta + 1);
    Sxi_zeta_compact(10) = -0.03125 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (xi + 1);
    Sxi_zeta_compact(11) = 0.03125 * m_lenZ * (3 * zeta + 1) * (zeta - 1) * (xi + 1) * (eta + 1);
    Sxi_zeta_compact(12) = -0.0625 * (xi - 1) * (eta + 1) * (eta * eta - eta + xi * xi + xi + 3 * zeta * zeta - 3);
    Sxi_zeta_compact(13) = -0.03125 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (eta + 1);
    Sxi_zeta_compact(14) = 0.03125 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (xi - 1);
    Sxi_zeta_compact(15) = -0.03125 * m_lenZ * (3 * zeta + 1) * (zeta - 1) * (xi - 1) * (eta + 1);
    Sxi_zeta_compact(16) = -0.0625 * (xi - 1) * (eta - 1) * (eta * eta + eta + xi * xi + xi + 3 * zeta * zeta - 3);
    Sxi_zeta_compact(17) = -0.03125 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (eta - 1);
    Sxi_zeta_compact(18) = -0.03125 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (xi - 1);
    Sxi_zeta_compact(19) = 0.03125 * m_lenZ * (zeta + 1) * (3 * zeta - 1) * (xi - 1) * (eta - 1);
    Sxi_zeta_compact(20) = 0.0625 * (xi + 1) * (eta - 1) * (eta * eta + eta + xi * xi - xi + 3 * zeta * zeta - 3);
    Sxi_zeta_compact(21) = -0.03125 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (eta - 1);
    Sxi_zeta_compact(22) = 0.03125 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (xi + 1);
    Sxi_zeta_compact(23) = -0.03125 * m_lenZ * (zeta + 1) * (3 * zeta - 1) * (xi + 1) * (eta - 1);
    Sxi_zeta_compact(24) = -0.0625 * (xi + 1) * (eta + 1) * (eta * eta - eta + xi * xi - xi + 3 * zeta * zeta - 3);
    Sxi_zeta_compact(25) = 0.03125 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (eta + 1);
    Sxi_zeta_compact(26) = 0.03125 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (xi + 1);
    Sxi_zeta_compact(27) = 0.03125 * m_lenZ * (zeta + 1) * (3 * zeta - 1) * (xi + 1) * (eta + 1);
    Sxi_zeta_compact(28) = 0.0625 * (xi - 1) * (eta + 1) * (eta * eta - eta + xi * xi + xi + 3 * zeta * zeta - 3);
    Sxi_zeta_compact(29) = 0.03125 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (eta + 1);
    Sxi_zeta_compact(30) = -0.03125 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (xi - 1);
    Sxi_zeta_compact(31) = -0.03125 * m_lenZ * (zeta + 1) * (3 * zeta - 1) * (xi - 1) * (eta + 1);
}

// Nx3 compact form of the partial derivatives of Normalized Shape Functions with respect to xi, eta, and zeta by
// columns

void ChElementHexaANCF_3843_MR_V5::Calc_Sxi_D(MatrixNx3c& Sxi_D, double xi, double eta, double zeta) {
    VectorN Sxi_D_col;
    Calc_Sxi_xi_compact(Sxi_D_col, xi, eta, zeta);
    Sxi_D.col(0) = Sxi_D_col;

    Calc_Sxi_eta_compact(Sxi_D_col, xi, eta, zeta);
    Sxi_D.col(1) = Sxi_D_col;

    Calc_Sxi_zeta_compact(Sxi_D_col, xi, eta, zeta);
    Sxi_D.col(2) = Sxi_D_col;
}

// -----------------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------------

void ChElementHexaANCF_3843_MR_V5::CalcCoordVector(Vector3N& e) {
    e.segment(0, 3) = m_nodes[0]->GetPos().eigen();
    e.segment(3, 3) = m_nodes[0]->GetD().eigen();
    e.segment(6, 3) = m_nodes[0]->GetDD().eigen();
    e.segment(9, 3) = m_nodes[0]->GetDDD().eigen();

    e.segment(12, 3) = m_nodes[1]->GetPos().eigen();
    e.segment(15, 3) = m_nodes[1]->GetD().eigen();
    e.segment(18, 3) = m_nodes[1]->GetDD().eigen();
    e.segment(21, 3) = m_nodes[1]->GetDDD().eigen();

    e.segment(24, 3) = m_nodes[2]->GetPos().eigen();
    e.segment(27, 3) = m_nodes[2]->GetD().eigen();
    e.segment(30, 3) = m_nodes[2]->GetDD().eigen();
    e.segment(33, 3) = m_nodes[2]->GetDDD().eigen();

    e.segment(36, 3) = m_nodes[3]->GetPos().eigen();
    e.segment(39, 3) = m_nodes[3]->GetD().eigen();
    e.segment(42, 3) = m_nodes[3]->GetDD().eigen();
    e.segment(45, 3) = m_nodes[3]->GetDDD().eigen();

    e.segment(48, 3) = m_nodes[4]->GetPos().eigen();
    e.segment(51, 3) = m_nodes[4]->GetD().eigen();
    e.segment(54, 3) = m_nodes[4]->GetDD().eigen();
    e.segment(57, 3) = m_nodes[4]->GetDDD().eigen();

    e.segment(60, 3) = m_nodes[5]->GetPos().eigen();
    e.segment(63, 3) = m_nodes[5]->GetD().eigen();
    e.segment(66, 3) = m_nodes[5]->GetDD().eigen();
    e.segment(69, 3) = m_nodes[5]->GetDDD().eigen();

    e.segment(72, 3) = m_nodes[6]->GetPos().eigen();
    e.segment(75, 3) = m_nodes[6]->GetD().eigen();
    e.segment(78, 3) = m_nodes[6]->GetDD().eigen();
    e.segment(81, 3) = m_nodes[6]->GetDDD().eigen();

    e.segment(84, 3) = m_nodes[7]->GetPos().eigen();
    e.segment(87, 3) = m_nodes[7]->GetD().eigen();
    e.segment(90, 3) = m_nodes[7]->GetDD().eigen();
    e.segment(93, 3) = m_nodes[7]->GetDDD().eigen();
}

void ChElementHexaANCF_3843_MR_V5::CalcCoordMatrix(Matrix3xN& ebar) {
    ebar.col(0) = m_nodes[0]->GetPos().eigen();
    ebar.col(1) = m_nodes[0]->GetD().eigen();
    ebar.col(2) = m_nodes[0]->GetDD().eigen();
    ebar.col(3) = m_nodes[0]->GetDDD().eigen();

    ebar.col(4) = m_nodes[1]->GetPos().eigen();
    ebar.col(5) = m_nodes[1]->GetD().eigen();
    ebar.col(6) = m_nodes[1]->GetDD().eigen();
    ebar.col(7) = m_nodes[1]->GetDDD().eigen();

    ebar.col(8) = m_nodes[2]->GetPos().eigen();
    ebar.col(9) = m_nodes[2]->GetD().eigen();
    ebar.col(10) = m_nodes[2]->GetDD().eigen();
    ebar.col(11) = m_nodes[2]->GetDDD().eigen();

    ebar.col(12) = m_nodes[3]->GetPos().eigen();
    ebar.col(13) = m_nodes[3]->GetD().eigen();
    ebar.col(14) = m_nodes[3]->GetDD().eigen();
    ebar.col(15) = m_nodes[3]->GetDDD().eigen();

    ebar.col(16) = m_nodes[4]->GetPos().eigen();
    ebar.col(17) = m_nodes[4]->GetD().eigen();
    ebar.col(18) = m_nodes[4]->GetDD().eigen();
    ebar.col(19) = m_nodes[4]->GetDDD().eigen();

    ebar.col(20) = m_nodes[5]->GetPos().eigen();
    ebar.col(21) = m_nodes[5]->GetD().eigen();
    ebar.col(22) = m_nodes[5]->GetDD().eigen();
    ebar.col(23) = m_nodes[5]->GetDDD().eigen();

    ebar.col(24) = m_nodes[6]->GetPos().eigen();
    ebar.col(25) = m_nodes[6]->GetD().eigen();
    ebar.col(26) = m_nodes[6]->GetDD().eigen();
    ebar.col(27) = m_nodes[6]->GetDDD().eigen();

    ebar.col(28) = m_nodes[7]->GetPos().eigen();
    ebar.col(29) = m_nodes[7]->GetD().eigen();
    ebar.col(30) = m_nodes[7]->GetDD().eigen();
    ebar.col(31) = m_nodes[7]->GetDDD().eigen();
}

void ChElementHexaANCF_3843_MR_V5::CalcCoordDerivVector(Vector3N& edot) {
    edot.segment(0, 3) = m_nodes[0]->GetPos_dt().eigen();
    edot.segment(3, 3) = m_nodes[0]->GetD_dt().eigen();
    edot.segment(6, 3) = m_nodes[0]->GetDD_dt().eigen();
    edot.segment(9, 3) = m_nodes[0]->GetDDD_dt().eigen();

    edot.segment(12, 3) = m_nodes[1]->GetPos_dt().eigen();
    edot.segment(15, 3) = m_nodes[1]->GetD_dt().eigen();
    edot.segment(18, 3) = m_nodes[1]->GetDD_dt().eigen();
    edot.segment(21, 3) = m_nodes[1]->GetDDD_dt().eigen();

    edot.segment(24, 3) = m_nodes[2]->GetPos_dt().eigen();
    edot.segment(27, 3) = m_nodes[2]->GetD_dt().eigen();
    edot.segment(30, 3) = m_nodes[2]->GetDD_dt().eigen();
    edot.segment(33, 3) = m_nodes[2]->GetDDD_dt().eigen();

    edot.segment(36, 3) = m_nodes[3]->GetPos_dt().eigen();
    edot.segment(39, 3) = m_nodes[3]->GetD_dt().eigen();
    edot.segment(42, 3) = m_nodes[3]->GetDD_dt().eigen();
    edot.segment(45, 3) = m_nodes[3]->GetDDD_dt().eigen();

    edot.segment(48, 3) = m_nodes[4]->GetPos_dt().eigen();
    edot.segment(51, 3) = m_nodes[4]->GetD_dt().eigen();
    edot.segment(54, 3) = m_nodes[4]->GetDD_dt().eigen();
    edot.segment(57, 3) = m_nodes[4]->GetDDD_dt().eigen();

    edot.segment(60, 3) = m_nodes[5]->GetPos_dt().eigen();
    edot.segment(63, 3) = m_nodes[5]->GetD_dt().eigen();
    edot.segment(66, 3) = m_nodes[5]->GetDD_dt().eigen();
    edot.segment(69, 3) = m_nodes[5]->GetDDD_dt().eigen();

    edot.segment(72, 3) = m_nodes[6]->GetPos_dt().eigen();
    edot.segment(75, 3) = m_nodes[6]->GetD_dt().eigen();
    edot.segment(78, 3) = m_nodes[6]->GetDD_dt().eigen();
    edot.segment(81, 3) = m_nodes[6]->GetDDD_dt().eigen();

    edot.segment(84, 3) = m_nodes[7]->GetPos_dt().eigen();
    edot.segment(87, 3) = m_nodes[7]->GetD_dt().eigen();
    edot.segment(90, 3) = m_nodes[7]->GetDD_dt().eigen();
    edot.segment(93, 3) = m_nodes[7]->GetDDD_dt().eigen();
}

void ChElementHexaANCF_3843_MR_V5::CalcCoordDerivMatrix(Matrix3xN& ebardot) {
    ebardot.col(0) = m_nodes[0]->GetPos_dt().eigen();
    ebardot.col(1) = m_nodes[0]->GetD_dt().eigen();
    ebardot.col(2) = m_nodes[0]->GetDD_dt().eigen();
    ebardot.col(3) = m_nodes[0]->GetDDD_dt().eigen();

    ebardot.col(4) = m_nodes[1]->GetPos_dt().eigen();
    ebardot.col(5) = m_nodes[1]->GetD_dt().eigen();
    ebardot.col(6) = m_nodes[1]->GetDD_dt().eigen();
    ebardot.col(7) = m_nodes[1]->GetDDD_dt().eigen();

    ebardot.col(8) = m_nodes[2]->GetPos_dt().eigen();
    ebardot.col(9) = m_nodes[2]->GetD_dt().eigen();
    ebardot.col(10) = m_nodes[2]->GetDD_dt().eigen();
    ebardot.col(11) = m_nodes[2]->GetDDD_dt().eigen();

    ebardot.col(12) = m_nodes[3]->GetPos_dt().eigen();
    ebardot.col(13) = m_nodes[3]->GetD_dt().eigen();
    ebardot.col(14) = m_nodes[3]->GetDD_dt().eigen();
    ebardot.col(15) = m_nodes[3]->GetDDD_dt().eigen();

    ebardot.col(16) = m_nodes[4]->GetPos_dt().eigen();
    ebardot.col(17) = m_nodes[4]->GetD_dt().eigen();
    ebardot.col(18) = m_nodes[4]->GetDD_dt().eigen();
    ebardot.col(19) = m_nodes[4]->GetDDD_dt().eigen();

    ebardot.col(20) = m_nodes[5]->GetPos_dt().eigen();
    ebardot.col(21) = m_nodes[5]->GetD_dt().eigen();
    ebardot.col(22) = m_nodes[5]->GetDD_dt().eigen();
    ebardot.col(23) = m_nodes[5]->GetDDD_dt().eigen();

    ebardot.col(24) = m_nodes[6]->GetPos_dt().eigen();
    ebardot.col(25) = m_nodes[6]->GetD_dt().eigen();
    ebardot.col(26) = m_nodes[6]->GetDD_dt().eigen();
    ebardot.col(27) = m_nodes[6]->GetDDD_dt().eigen();

    ebardot.col(28) = m_nodes[7]->GetPos_dt().eigen();
    ebardot.col(29) = m_nodes[7]->GetD_dt().eigen();
    ebardot.col(30) = m_nodes[7]->GetDD_dt().eigen();
    ebardot.col(31) = m_nodes[7]->GetDDD_dt().eigen();
}

void ChElementHexaANCF_3843_MR_V5::CalcCombinedCoordMatrix(MatrixNx6& ebar_ebardot) {
    ebar_ebardot.template block<1, 3>(0, 0) = m_nodes[0]->GetPos().eigen();
    ebar_ebardot.template block<1, 3>(0, 3) = m_nodes[0]->GetPos_dt().eigen();
    ebar_ebardot.template block<1, 3>(1, 0) = m_nodes[0]->GetD().eigen();
    ebar_ebardot.template block<1, 3>(1, 3) = m_nodes[0]->GetD_dt().eigen();
    ebar_ebardot.template block<1, 3>(2, 0) = m_nodes[0]->GetDD().eigen();
    ebar_ebardot.template block<1, 3>(2, 3) = m_nodes[0]->GetDD_dt().eigen();
    ebar_ebardot.template block<1, 3>(3, 0) = m_nodes[0]->GetDDD().eigen();
    ebar_ebardot.template block<1, 3>(3, 3) = m_nodes[0]->GetDDD_dt().eigen();

    ebar_ebardot.template block<1, 3>(4, 0) = m_nodes[1]->GetPos().eigen();
    ebar_ebardot.template block<1, 3>(4, 3) = m_nodes[1]->GetPos_dt().eigen();
    ebar_ebardot.template block<1, 3>(5, 0) = m_nodes[1]->GetD().eigen();
    ebar_ebardot.template block<1, 3>(5, 3) = m_nodes[1]->GetD_dt().eigen();
    ebar_ebardot.template block<1, 3>(6, 0) = m_nodes[1]->GetDD().eigen();
    ebar_ebardot.template block<1, 3>(6, 3) = m_nodes[1]->GetDD_dt().eigen();
    ebar_ebardot.template block<1, 3>(7, 0) = m_nodes[1]->GetDDD().eigen();
    ebar_ebardot.template block<1, 3>(7, 3) = m_nodes[1]->GetDDD_dt().eigen();

    ebar_ebardot.template block<1, 3>(8, 0) = m_nodes[2]->GetPos().eigen();
    ebar_ebardot.template block<1, 3>(8, 3) = m_nodes[2]->GetPos_dt().eigen();
    ebar_ebardot.template block<1, 3>(9, 0) = m_nodes[2]->GetD().eigen();
    ebar_ebardot.template block<1, 3>(9, 3) = m_nodes[2]->GetD_dt().eigen();
    ebar_ebardot.template block<1, 3>(10, 0) = m_nodes[2]->GetDD().eigen();
    ebar_ebardot.template block<1, 3>(10, 3) = m_nodes[2]->GetDD_dt().eigen();
    ebar_ebardot.template block<1, 3>(11, 0) = m_nodes[2]->GetDDD().eigen();
    ebar_ebardot.template block<1, 3>(11, 3) = m_nodes[2]->GetDDD_dt().eigen();

    ebar_ebardot.template block<1, 3>(12, 0) = m_nodes[3]->GetPos().eigen();
    ebar_ebardot.template block<1, 3>(12, 3) = m_nodes[3]->GetPos_dt().eigen();
    ebar_ebardot.template block<1, 3>(13, 0) = m_nodes[3]->GetD().eigen();
    ebar_ebardot.template block<1, 3>(13, 3) = m_nodes[3]->GetD_dt().eigen();
    ebar_ebardot.template block<1, 3>(14, 0) = m_nodes[3]->GetDD().eigen();
    ebar_ebardot.template block<1, 3>(14, 3) = m_nodes[3]->GetDD_dt().eigen();
    ebar_ebardot.template block<1, 3>(15, 0) = m_nodes[3]->GetDDD().eigen();
    ebar_ebardot.template block<1, 3>(15, 3) = m_nodes[3]->GetDDD_dt().eigen();

    ebar_ebardot.template block<1, 3>(16, 0) = m_nodes[4]->GetPos().eigen();
    ebar_ebardot.template block<1, 3>(16, 3) = m_nodes[4]->GetPos_dt().eigen();
    ebar_ebardot.template block<1, 3>(17, 0) = m_nodes[4]->GetD().eigen();
    ebar_ebardot.template block<1, 3>(17, 3) = m_nodes[4]->GetD_dt().eigen();
    ebar_ebardot.template block<1, 3>(18, 0) = m_nodes[4]->GetDD().eigen();
    ebar_ebardot.template block<1, 3>(18, 3) = m_nodes[4]->GetDD_dt().eigen();
    ebar_ebardot.template block<1, 3>(19, 0) = m_nodes[4]->GetDDD().eigen();
    ebar_ebardot.template block<1, 3>(19, 3) = m_nodes[4]->GetDDD_dt().eigen();

    ebar_ebardot.template block<1, 3>(20, 0) = m_nodes[5]->GetPos().eigen();
    ebar_ebardot.template block<1, 3>(20, 3) = m_nodes[5]->GetPos_dt().eigen();
    ebar_ebardot.template block<1, 3>(21, 0) = m_nodes[5]->GetD().eigen();
    ebar_ebardot.template block<1, 3>(21, 3) = m_nodes[5]->GetD_dt().eigen();
    ebar_ebardot.template block<1, 3>(22, 0) = m_nodes[5]->GetDD().eigen();
    ebar_ebardot.template block<1, 3>(22, 3) = m_nodes[5]->GetDD_dt().eigen();
    ebar_ebardot.template block<1, 3>(23, 0) = m_nodes[5]->GetDDD().eigen();
    ebar_ebardot.template block<1, 3>(23, 3) = m_nodes[5]->GetDDD_dt().eigen();

    ebar_ebardot.template block<1, 3>(24, 0) = m_nodes[6]->GetPos().eigen();
    ebar_ebardot.template block<1, 3>(24, 3) = m_nodes[6]->GetPos_dt().eigen();
    ebar_ebardot.template block<1, 3>(25, 0) = m_nodes[6]->GetD().eigen();
    ebar_ebardot.template block<1, 3>(25, 3) = m_nodes[6]->GetD_dt().eigen();
    ebar_ebardot.template block<1, 3>(26, 0) = m_nodes[6]->GetDD().eigen();
    ebar_ebardot.template block<1, 3>(26, 3) = m_nodes[6]->GetDD_dt().eigen();
    ebar_ebardot.template block<1, 3>(27, 0) = m_nodes[6]->GetDDD().eigen();
    ebar_ebardot.template block<1, 3>(27, 3) = m_nodes[6]->GetDDD_dt().eigen();

    ebar_ebardot.template block<1, 3>(28, 0) = m_nodes[7]->GetPos().eigen();
    ebar_ebardot.template block<1, 3>(28, 3) = m_nodes[7]->GetPos_dt().eigen();
    ebar_ebardot.template block<1, 3>(29, 0) = m_nodes[7]->GetD().eigen();
    ebar_ebardot.template block<1, 3>(29, 3) = m_nodes[7]->GetD_dt().eigen();
    ebar_ebardot.template block<1, 3>(30, 0) = m_nodes[7]->GetDD().eigen();
    ebar_ebardot.template block<1, 3>(30, 3) = m_nodes[7]->GetDD_dt().eigen();
    ebar_ebardot.template block<1, 3>(31, 0) = m_nodes[7]->GetDDD().eigen();
    ebar_ebardot.template block<1, 3>(31, 3) = m_nodes[7]->GetDDD_dt().eigen();
}

// Calculate the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element

void ChElementHexaANCF_3843_MR_V5::Calc_J_0xi(ChMatrix33<double>& J_0xi, double xi, double eta, double zeta) {
    MatrixNx3c Sxi_D;
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    J_0xi = m_ebar0 * Sxi_D;
}

// Calculate the determinant of the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element

double ChElementHexaANCF_3843_MR_V5::Calc_det_J_0xi(double xi, double eta, double zeta) {
    ChMatrix33<double> J_0xi;
    Calc_J_0xi(J_0xi, xi, eta, zeta);

    return (J_0xi.determinant());
}

////////////////////////////////////////////////////////////////

//#ifndef CH_QUADRATURE_STATIC_TABLES
#define CH_QUADRATURE_STATIC_TABLES 10
ChQuadratureTables static_tables_3843_MR_V5(1, CH_QUADRATURE_STATIC_TABLES);
//#endif // !CH_QUADRATURE_STATIC_TABLES

ChQuadratureTables* ChElementHexaANCF_3843_MR_V5::GetStaticGQTables() {
    return &static_tables_3843_MR_V5;
}

}  // namespace fea
}  // namespace chrono

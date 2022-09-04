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
//
// =============================================================================
// Two term Mooney-Rivlin Hyperelastic Material Law with penalty term for incompressibility with the option for a single
// coefficient nonlinear KV Damping
// =============================================================================
// A description of the material law can be found in: Orzechowski, G., & Fraczek, J. (2015). Nearly incompressible
// nonlinear material models in the large deformation analysis of beams using ANCF. Nonlinear Dynamics, 82(1), 451-464.
//
// A description of the damping law can be found in: Kubler, L., Eberhard, P., & Geisler, J. (2003). Flexible multibody
// systems with large deformations and nonlinear structural damping using absolute nodal coordinates. Nonlinear
// Dynamics, 34(1), 31-52.
// =============================================================================

#include "chrono/fea/ChElementHexaANCF_3843_MR_Damp.h"
#include "chrono/physics/ChSystem.h"

namespace chrono {
namespace fea {

// ------------------------------------------------------------------------------
// Constructor
// ------------------------------------------------------------------------------

ChElementHexaANCF_3843_MR_Damp::ChElementHexaANCF_3843_MR_Damp()
    : m_lenX(0), m_lenY(0), m_lenZ(0) {
    m_nodes.resize(8);
}

// ------------------------------------------------------------------------------
// Set element nodes
// ------------------------------------------------------------------------------

void ChElementHexaANCF_3843_MR_Damp::SetNodes(std::shared_ptr<ChNodeFEAxyzDDD> nodeA,
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
}

// -----------------------------------------------------------------------------
// Element Settings
// -----------------------------------------------------------------------------

// Specify the element dimensions.

void ChElementHexaANCF_3843_MR_Damp::SetDimensions(double lenX, double lenY, double lenZ) {
    m_lenX = lenX;
    m_lenY = lenY;
    m_lenZ = lenZ;
}

// Specify the element material.

void ChElementHexaANCF_3843_MR_Damp::SetMaterial(std::shared_ptr<ChMaterialHexaANCF_MR> brick_mat) {
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

ChMatrix33<> ChElementHexaANCF_3843_MR_Damp::GetGreenLagrangeStrain(const double xi, const double eta, const double zeta) {
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

ChMatrix33<> ChElementHexaANCF_3843_MR_Damp::GetPK2Stress(const double xi, const double eta, const double zeta) {
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

    return SPK2;
}

// Get the von Mises stress value at the normalized element coordinates (xi, eta, zeta) [-1...1] at the current
// state of the element.

double ChElementHexaANCF_3843_MR_Damp::GetVonMissesStress(const double xi, const double eta, const double zeta) {
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

void ChElementHexaANCF_3843_MR_Damp::SetupInitial(ChSystem* system) {
    // Store the initial nodal coordinates. These values define the reference configuration of the element.
    CalcCoordMatrix(m_ebar0);

    // Compute and store the constant mass matrix and the matrix used to multiply the acceleration due to gravity to get
    // the generalized gravitational force vector for the element
    ComputeMassMatrixAndGravityForce();

    // Compute Pre-computed matrices and vectors for the generalized internal force calcualtions
    PrecomputeInternalForceMatricesWeights();
}

// Fill the D vector with the current field values at the element nodes.

void ChElementHexaANCF_3843_MR_Damp::GetStateBlock(ChVectorDynamic<>& mD) {
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

void ChElementHexaANCF_3843_MR_Damp::Update() {
    ChElementGeneric::Update();
}

// Return the mass matrix in full sparse form.

void ChElementHexaANCF_3843_MR_Damp::ComputeMmatrixGlobal(ChMatrixRef M) {
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

void ChElementHexaANCF_3843_MR_Damp::ComputeNodalMass() {
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

void ChElementHexaANCF_3843_MR_Damp::ComputeInternalForces(ChVectorDynamic<>& Fi) {

    //if (GetMaterial()->Get_mu() != 0) {
        ComputeInternalForceDamping(Fi);
    //}
    //else {
    //    ComputeInternalForceNoDamping(Fi);
    //}
}

// Calculate the global matrix H as a linear combination of K, R, and M:
//   H = Mfactor * [M] + Kfactor * [K] + Rfactor * [R]

void ChElementHexaANCF_3843_MR_Damp::ComputeKRMmatricesGlobal(ChMatrixRef H, double Kfactor, double Rfactor, double Mfactor) {

    //if (GetMaterial()->Get_mu() != 0) {
        ComputeInternalJacobianDamping(H, Kfactor, Rfactor, Mfactor);
    //}
    //else {
    //    ComputeInternalJacobianNoDamping(H, Kfactor, Mfactor);
    //}
}

// Compute the generalized force vector due to gravity using the efficient ANCF specific method
void ChElementHexaANCF_3843_MR_Damp::ComputeGravityForces(ChVectorDynamic<>& Fg, const ChVector<>& G_acc) {
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

void ChElementHexaANCF_3843_MR_Damp::EvaluateElementFrame(const double xi,
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

void ChElementHexaANCF_3843_MR_Damp::EvaluateElementPoint(const double xi,
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

void ChElementHexaANCF_3843_MR_Damp::EvaluateElementVel(double xi, double eta, const double zeta, ChVector<>& Result) {
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

void ChElementHexaANCF_3843_MR_Damp::LoadableGetStateBlock_x(int block_offset, ChState& mD) {
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

void ChElementHexaANCF_3843_MR_Damp::LoadableGetStateBlock_w(int block_offset, ChStateDelta& mD) {
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

void ChElementHexaANCF_3843_MR_Damp::LoadableStateIncrement(const unsigned int off_x,
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

void ChElementHexaANCF_3843_MR_Damp::LoadableGetVariables(std::vector<ChVariables*>& mvars) {
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

void ChElementHexaANCF_3843_MR_Damp::ComputeNF(
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

double ChElementHexaANCF_3843_MR_Damp::GetDensity() {
    return GetMaterial()->Get_rho();
}

// -----------------------------------------------------------------------------
// Mass Matrix & Generalized Force Due to Gravity Calculation
// -----------------------------------------------------------------------------

void ChElementHexaANCF_3843_MR_Damp::ComputeMassMatrixAndGravityForce() {
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
    MatrixNxN MassMatrixCompactSquare;

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

void ChElementHexaANCF_3843_MR_Damp::PrecomputeInternalForceMatricesWeights() {
    ChQuadratureTables* GQTable = GetStaticGQTables();

    m_SD.resize(NSF, 3 * NIP);
    m_kGQ.resize(1, NIP);

    // Precalculate the matrices of normalized shape function derivatives corrected for a potentially non-straight
    // reference configuration & GQ Weights times the determinant of the element Jacobian for later use in the
    // generalized internal force and Jacobian calculations.
    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[NP - 1].size(); it_xi++) {
        for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[NP - 1].size(); it_eta++) {
            for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[NP - 1].size(); it_zeta++) {
                double GQ_weight = GQTable->Weight[NP - 1][it_xi] * GQTable->Weight[NP - 1][it_eta] *
                                    GQTable->Weight[NP - 1][it_zeta];
                double xi = GQTable->Lroots[NP - 1][it_xi];
                double eta = GQTable->Lroots[NP - 1][it_eta];
                double zeta = GQTable->Lroots[NP - 1][it_zeta];
                auto index = it_zeta + it_eta * GQTable->Lroots[NP - 1].size() +
                                it_xi * GQTable->Lroots[NP - 1].size() * GQTable->Lroots[NP - 1].size();
                ChMatrix33<double> J_0xi;  // Element Jacobian between the reference and normalized configurations
                MatrixNx3c Sxi_D;          // Matrix of normalized shape function derivatives

                Calc_Sxi_D(Sxi_D, xi, eta, zeta);
                J_0xi.noalias() = m_ebar0 * Sxi_D;

                // Adjust the shape function derivative matrix to account for a potentially deformed reference state
                m_kGQ(index) = -J_0xi.determinant() * GQ_weight;
                ChMatrixNM<double, NSF, 3> SD = Sxi_D * J_0xi.inverse();

                // Group all of the columns together in blocks with the shape function derivatives for the section
                // that does not include the Poisson effect at the beginning of m_SD
                m_SD.col(index) = SD.col(0);
                m_SD.col(index + NIP) = SD.col(1);
                m_SD.col(index + 2 * NIP) = SD.col(2);
            }
        }
    }
}


// -----------------------------------------------------------------------------
// Elastic force calculation
// -----------------------------------------------------------------------------

void ChElementHexaANCF_3843_MR_Damp::ComputeInternalForceDamping(ChVectorDynamic<>& Fi) {
    assert(Fi.size() == 3 * NSF);

    // Retrieve the nodal coordinates and nodal coordinate time derivatives
    Matrix6xN ebar_ebardot;
    CalcCombinedCoordMatrix(ebar_ebardot);

    // =============================================================================
    // Calculate the deformation gradient and time derivative of the deformation gradient for all Gauss quadrature
    // points in a single matrix multiplication.  Note that since the shape function derivative matrix is ordered by
    // columns, the resulting deformation gradient will be ordered by block matrix (column vectors) components
    //      [F11     F12     F13    ]
    //      [F21     F22     F23    ]
    // FC = [F31     F32     F33    ]
    //      [Fdot11  Fdot12  Fdot13 ]
    //      [Fdot21  Fdot22  Fdot23 ]
    //      [Fdot31  Fdot32  Fdot33 ]
    // =============================================================================

    ChMatrixNM<double, 6, 3 * NIP> FC = ebar_ebardot * m_SD;  // ChMatrixNM<double, 6, 3 * NIP>

    Eigen::Map<ArrayNIP> F11(FC.block<1, NIP>(0, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F12(FC.block<1, NIP>(0, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F13(FC.block<1, NIP>(0, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> F21(FC.block<1, NIP>(1, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F22(FC.block<1, NIP>(1, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F23(FC.block<1, NIP>(1, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> F31(FC.block<1, NIP>(2, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F32(FC.block<1, NIP>(2, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F33(FC.block<1, NIP>(2, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> Fdot11(FC.block<1, NIP>(3, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> Fdot12(FC.block<1, NIP>(3, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> Fdot13(FC.block<1, NIP>(3, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> Fdot21(FC.block<1, NIP>(4, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> Fdot22(FC.block<1, NIP>(4, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> Fdot23(FC.block<1, NIP>(4, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> Fdot31(FC.block<1, NIP>(5, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> Fdot32(FC.block<1, NIP>(5, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> Fdot33(FC.block<1, NIP>(5, 2 * NIP).data(), 1, NIP);


    //Calculate the Right Cauchy-Green deformation tensor (symmetric):
    ArrayNIP C11 = F11 * F11 + F21 * F21 + F31 * F31;
    ArrayNIP C22 = F12 * F12 + F22 * F22 + F32 * F32;
    ArrayNIP C33 = F13 * F13 + F23 * F23 + F33 * F33;
    ArrayNIP C12 = F11 * F12 + F21 * F22 + F31 * F32;
    ArrayNIP C13 = F11 * F13 + F21 * F23 + F31 * F33;
    ArrayNIP C23 = F12 * F13 + F22 * F23 + F32 * F33;

    //Calculate the 1st invariant of the Right Cauchy-Green deformation tensor: trace(C)
    ArrayNIP I1 = C11 + C22 + C33;

    //Calculate the 2nd invariant of the Right Cauchy-Green deformation tensor: 1/2*(trace(C)^2-trace(C^2))
    ArrayNIP I2 = 0.5*(I1*I1 - C11 * C11 - C22 * C22 - C33 * C33) - C12 * C12 - C13 * C13 - C23 * C23;

    //Calculate the determinate of F (commonly denoted as J)
    ArrayNIP detF = F11 * F22*F33 + F12 * F23*F31 + F13 * F21*F32 - F11 * F23*F32 - F12 * F21*F33 - F13 * F22*F31;

    ArrayNIP detF_m2_3 = detF.pow(-2.0 / 3.0);

    double c10 = GetMaterial()->Get_c10();
    double c01 = GetMaterial()->Get_c01();
    double k = GetMaterial()->Get_k();
    double mu = GetMaterial()->Get_mu();

    ArrayNIP kGQmu_over_detF3 = mu * m_kGQ / (detF*detF*detF);

    // Calculate the time derivative of the Greens-Lagrange strain tensor at the current point (symmetric):
    ArrayNIP Edot11 = F11 * Fdot11 + F21 * Fdot21 + F31 * Fdot31;
    ArrayNIP Edot22 = F12 * Fdot12 + F22 * Fdot22 + F32 * Fdot32;
    ArrayNIP Edot33 = F13 * Fdot13 + F23 * Fdot23 + F33 * Fdot33;
    ArrayNIP Edot12 = 0.5*(F11 * Fdot12 + F12 * Fdot11 + F21 * Fdot22 + F22 * Fdot21 + F31 * Fdot32 + F32 * Fdot31);
    ArrayNIP Edot13 = 0.5*(F11 * Fdot13 + F13 * Fdot11 + F21 * Fdot23 + F23 * Fdot21 + F31 * Fdot33 + F33 * Fdot31);
    ArrayNIP Edot23 = 0.5*(F12 * Fdot13 + F13 * Fdot12 + F22 * Fdot23 + F23 * Fdot22 + F32 * Fdot33 + F33 * Fdot32);

    //Inverse of C times detF^2 (since all the detF terms will be handled together)
    ArrayNIP CInv11 = C22 * C33 - C23 * C23;
    ArrayNIP CInv22 = C11 * C33 - C13 * C13;
    ArrayNIP CInv33 = C11 * C22 - C12 * C12;
    ArrayNIP CInv12 = C13 * C23 - C12 * C33;
    ArrayNIP CInv13 = C12 * C23 - C13 * C22;
    ArrayNIP CInv23 = C13 * C12 - C11 * C23;

    //Calculate the Stress from the viscosity law
    ArrayNIP SPK2_NLKV_1 = kGQmu_over_detF3 * (Edot11*CInv11*CInv11 + 2.0 * Edot12*CInv11*CInv12 + 2.0 * Edot13*CInv11*CInv13 + Edot22 * CInv12*CInv12 + 2.0 * Edot23*CInv12*CInv13 + Edot33 * CInv13*CInv13);
    ArrayNIP SPK2_NLKV_2 = kGQmu_over_detF3 * (Edot11*CInv12*CInv12 + 2.0 * Edot12*CInv12*CInv22 + 2.0 * Edot13*CInv12*CInv23 + Edot22 * CInv22*CInv22 + 2.0 * Edot23*CInv22*CInv23 + Edot33 * CInv23*CInv23);
    ArrayNIP SPK2_NLKV_3 = kGQmu_over_detF3 * (Edot11*CInv13*CInv13 + 2.0 * Edot12*CInv13*CInv23 + 2.0 * Edot13*CInv13*CInv33 + Edot22 * CInv23*CInv23 + 2.0 * Edot23*CInv23*CInv33 + Edot33 * CInv33*CInv33);
    ArrayNIP SPK2_NLKV_4 = kGQmu_over_detF3 * (Edot11*CInv12*CInv13 + Edot12 * (CInv12*CInv23 + CInv22 * CInv13) + Edot13 * (CInv12*CInv33 + CInv13 * CInv23) + Edot22 * CInv22*CInv23 + Edot23 * (CInv23*CInv23 + CInv22 * CInv33) + Edot33 * CInv23*CInv33);
    ArrayNIP SPK2_NLKV_5 = kGQmu_over_detF3 * (Edot11*CInv11*CInv13 + Edot12 * (CInv11*CInv23 + CInv12 * CInv13) + Edot13 * (CInv13*CInv13 + CInv11 * CInv33) + Edot22 * CInv12*CInv23 + Edot23 * (CInv12*CInv33 + CInv13 * CInv23) + Edot33 * CInv13*CInv33);
    ArrayNIP SPK2_NLKV_6 = kGQmu_over_detF3 * (Edot11*CInv11*CInv12 + Edot12 * (CInv12*CInv12 + CInv11 * CInv22) + Edot13 * (CInv11*CInv23 + CInv12 * CInv13) + Edot22 * CInv12*CInv22 + Edot23 * (CInv12*CInv23 + CInv22 * CInv13) + Edot33 * CInv13*CInv23);

    ArrayNIP s0 = 2.0 * m_kGQ * detF_m2_3;
    ArrayNIP s2 = -c01 * s0 * detF_m2_3;
    s0 *= c10;
    ArrayNIP s1 = s0 - s2 * I1;
    ArrayNIP s3 = k * (detF - 1.0)*m_kGQ - (I1*s0 - 2 * I2 * s2) / (3.0 * detF);

    ChMatrixNMc<double, 3 * NIP, 3> P_Block;
    P_Block.block<NIP, 1>(0, 0) = s3 * (F22 * F33 - F23 * F32) + (s1 + s2 * C11) * F11 + s2 * (C12 * F12 + C13 * F13) + F11 * SPK2_NLKV_1 + F12 * SPK2_NLKV_6 + F13 * SPK2_NLKV_5;
    P_Block.block<NIP, 1>(NIP, 0) = s3 * (F23 * F31 - F21 * F33) + (s1 + s2 * C22) * F12 + s2 * (C12 * F11 + C23 * F13) + F11 * SPK2_NLKV_6 + F12 * SPK2_NLKV_2 + F13 * SPK2_NLKV_4;
    P_Block.block<NIP, 1>(2 * NIP, 0) = s3 * (F21 * F32 - F22 * F31) + (s1 + s2 * C33) * F13 + s2 * (C13 * F11 + C23 * F12) + F11 * SPK2_NLKV_5 + F12 * SPK2_NLKV_4 + F13 * SPK2_NLKV_3;
    P_Block.block<NIP, 1>(0, 1) = s3 * (F13 * F32 - F12 * F33) + (s1 + s2 * C11) * F21 + s2 * (C12 * F22 + C13 * F23) + F21 * SPK2_NLKV_1 + F22 * SPK2_NLKV_6 + F23 * SPK2_NLKV_5;
    P_Block.block<NIP, 1>(NIP, 1) = s3 * (F11 * F33 - F13 * F31) + (s1 + s2 * C22) * F22 + s2 * (C12 * F21 + C23 * F23) + F21 * SPK2_NLKV_6 + F22 * SPK2_NLKV_2 + F23 * SPK2_NLKV_4;
    P_Block.block<NIP, 1>(2 * NIP, 1) = s3 * (F12 * F31 - F11 * F32) + (s1 + s2 * C33) * F23 + s2 * (C13 * F21 + C23 * F22) + F21 * SPK2_NLKV_5 + F22 * SPK2_NLKV_4 + F23 * SPK2_NLKV_3;
    P_Block.block<NIP, 1>(0, 2) = s3 * (F12 * F23 - F13 * F22) + (s1 + s2 * C11) * F31 + s2 * (C12 * F32 + C13 * F33) + F31 * SPK2_NLKV_1 + F32 * SPK2_NLKV_6 + F33 * SPK2_NLKV_5;
    P_Block.block<NIP, 1>(NIP, 2) = s3 * (F13 * F21 - F11 * F23) + (s1 + s2 * C22) * F32 + s2 * (C12 * F31 + C23 * F33) + F31 * SPK2_NLKV_6 + F32 * SPK2_NLKV_2 + F33 * SPK2_NLKV_4;
    P_Block.block<NIP, 1>(2 * NIP, 2) = s3 * (F11 * F22 - F12 * F21) + (s1 + s2 * C33) * F33 + s2 * (C13 * F31 + C23 * F32) + F31 * SPK2_NLKV_5 + F32 * SPK2_NLKV_4 + F33 * SPK2_NLKV_3;

    Eigen::Map<MatrixNx3> FiCompact(Fi.data(), NSF, 3);
    FiCompact.noalias() = m_SD * P_Block;
}

void ChElementHexaANCF_3843_MR_Damp::ComputeInternalForceNoDamping(ChVectorDynamic<>& Fi) {
    assert(Fi.size() == 3 * NSF);

    // Element coordinates in matrix form
    Matrix3xN ebar;
    CalcCoordMatrix(ebar);

    // =============================================================================
    // Calculate the deformation gradient for all Gauss quadrature points in a single matrix multiplication.  Note
    // that since the shape function derivative matrix is ordered by columns, the resulting deformation gradient
    // will be ordered by block matrix (row vectors) components
    //      [F11     F12     F13    ]
    // FC = [F21     F22     F23    ]
    //      [F31     F32     F33    ]
    // =============================================================================

    ChMatrixNM<double, 3, 3 * NIP> FC = ebar * m_SD;

    Eigen::Map<ArrayNIP> F11(FC.block<1, NIP>(0, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F12(FC.block<1, NIP>(0, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F13(FC.block<1, NIP>(0, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> F21(FC.block<1, NIP>(1, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F22(FC.block<1, NIP>(1, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F23(FC.block<1, NIP>(1, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> F31(FC.block<1, NIP>(2, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F32(FC.block<1, NIP>(2, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F33(FC.block<1, NIP>(2, 2 * NIP).data(), 1, NIP);

    //Calculate the Right Cauchy-Green deformation tensor (symmetric):
    ArrayNIP C11 = F11 * F11 + F21 * F21 + F31 * F31;
    ArrayNIP C22 = F12 * F12 + F22 * F22 + F32 * F32;
    ArrayNIP C33 = F13 * F13 + F23 * F23 + F33 * F33;
    ArrayNIP C12 = F11 * F12 + F21 * F22 + F31 * F32;
    ArrayNIP C13 = F11 * F13 + F21 * F23 + F31 * F33;
    ArrayNIP C23 = F12 * F13 + F22 * F23 + F32 * F33;

    //Calculate the 1st invariant of the Right Cauchy-Green deformation tensor: trace(C)
    ArrayNIP I1 = C11 + C22 + C33;

    //Calculate the 2nd invariant of the Right Cauchy-Green deformation tensor: 1/2*(trace(C)^2-trace(C^2))
    ArrayNIP I2 = 0.5*(I1*I1 - C11*C11 - C22*C22 - C33*C33) - C12*C12 - C13*C13 - C23*C23;

    //Calculate the determinate of F (commonly denoted as J)
    ArrayNIP detF = F11*F22*F33 + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 - F13*F22*F31;

    ArrayNIP detF_m2_3 = detF.pow(-2.0 / 3.0);

    double c10 = GetMaterial()->Get_c10();
    double c01 = GetMaterial()->Get_c01();
    double k = GetMaterial()->Get_k();

    ArrayNIP s0 = 2.0 * m_kGQ * detF_m2_3;
    ArrayNIP s2 = -c01 * s0 * detF_m2_3;
    s0 *= c10;
    ArrayNIP s1 = s0 - s2 * I1;
    ArrayNIP s3 = k * (detF - 1.0)*m_kGQ - (I1*s0 - 2 * I2 * s2) / (3.0 * detF);

    ChMatrixNMc<double, 3 * NIP, 3> P_Block;
    P_Block.block<NIP, 1>(0, 0) = s3 * (F22 * F33 - F23 * F32) + (s1 + s2 * C11) * F11 + s2 * (C12 * F12 + C13 * F13);
    P_Block.block<NIP, 1>(NIP, 0) = s3 * (F23 * F31 - F21 * F33) + (s1 + s2 * C22) * F12 + s2 * (C12 * F11 + C23 * F13);
    P_Block.block<NIP, 1>(2 * NIP, 0) = s3 * (F21 * F32 - F22 * F31) + (s1 + s2 * C33) * F13 + s2 * (C13 * F11 + C23 * F12);
    P_Block.block<NIP, 1>(0, 1) = s3 * (F13 * F32 - F12 * F33) + (s1 + s2 * C11) * F21 + s2 * (C12 * F22 + C13 * F23);
    P_Block.block<NIP, 1>(NIP, 1) = s3 * (F11 * F33 - F13 * F31) + (s1 + s2 * C22) * F22 + s2 * (C12 * F21 + C23 * F23);
    P_Block.block<NIP, 1>(2 * NIP, 1) = s3 * (F12 * F31 - F11 * F32) + (s1 + s2 * C33) * F23 + s2 * (C13 * F21 + C23 * F22);
    P_Block.block<NIP, 1>(0, 2) = s3 * (F12 * F23 - F13 * F22) + (s1 + s2 * C11) * F31 + s2 * (C12 * F32 + C13 * F33);
    P_Block.block<NIP, 1>(NIP, 2) = s3 * (F13 * F21 - F11 * F23) + (s1 + s2 * C22) * F32 + s2 * (C12 * F31 + C23 * F33);
    P_Block.block<NIP, 1>(2 * NIP, 2) = s3 * (F11 * F22 - F12 * F21) + (s1 + s2 * C33) * F33 + s2 * (C13 * F31 + C23 * F32);

    Eigen::Map<MatrixNx3> FiCompact(Fi.data(), NSF, 3);
    FiCompact.noalias() = m_SD * P_Block;
}


// -----------------------------------------------------------------------------
// Jacobians of internal forces
// -----------------------------------------------------------------------------

void ChElementHexaANCF_3843_MR_Damp::ComputeInternalJacobianDamping(ChMatrixRef H, double Kfactor, double Rfactor, double Mfactor) {
    assert((H.rows() == 3 * NSF) && (H.cols() == 3 * NSF));

    // Retrieve the nodal coordinates and nodal coordinate time derivatives
    Matrix6xN ebar_ebardot;
    CalcCombinedCoordMatrix(ebar_ebardot);

    // =============================================================================
    // Calculate the deformation gradient and time derivative of the deformation gradient for all Gauss quadrature
    // points in a single matrix multiplication.  Note that since the shape function derivative matrix is ordered by
    // columns, the resulting deformation gradient will be ordered by block matrix (column vectors) components
    //      [F11     F12     F13    ]
    //      [F21     F22     F23    ]
    // FC = [F31     F32     F33    ]
    //      [Fdot11  Fdot12  Fdot13 ]
    //      [Fdot21  Fdot22  Fdot23 ]
    //      [Fdot31  Fdot32  Fdot33 ]
    // =============================================================================

    ChMatrixNM<double, 6, 3 * NIP> FC = ebar_ebardot * m_SD;  // ChMatrixNM<double, 6, 3 * NIP>

    Eigen::Map<ArrayNIP> F11(FC.block<1, NIP>(0, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F12(FC.block<1, NIP>(0, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F13(FC.block<1, NIP>(0, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> F21(FC.block<1, NIP>(1, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F22(FC.block<1, NIP>(1, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F23(FC.block<1, NIP>(1, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> F31(FC.block<1, NIP>(2, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F32(FC.block<1, NIP>(2, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F33(FC.block<1, NIP>(2, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> Fdot11(FC.block<1, NIP>(3, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> Fdot12(FC.block<1, NIP>(3, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> Fdot13(FC.block<1, NIP>(3, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> Fdot21(FC.block<1, NIP>(4, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> Fdot22(FC.block<1, NIP>(4, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> Fdot23(FC.block<1, NIP>(4, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> Fdot31(FC.block<1, NIP>(5, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> Fdot32(FC.block<1, NIP>(5, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> Fdot33(FC.block<1, NIP>(5, 2 * NIP).data(), 1, NIP);


    //Calculate the Right Cauchy-Green deformation tensor (symmetric):
    ArrayNIP C11 = F11 * F11 + F21 * F21 + F31 * F31;
    ArrayNIP C22 = F12 * F12 + F22 * F22 + F32 * F32;
    ArrayNIP C33 = F13 * F13 + F23 * F23 + F33 * F33;
    ArrayNIP C12 = F11 * F12 + F21 * F22 + F31 * F32;
    ArrayNIP C13 = F11 * F13 + F21 * F23 + F31 * F33;
    ArrayNIP C23 = F12 * F13 + F22 * F23 + F32 * F33;

    //Calculate the 1st invariant of the Right Cauchy-Green deformation tensor: trace(C)
    ArrayNIP I1 = C11 + C22 + C33;

    //Calculate the 2nd invariant of the Right Cauchy-Green deformation tensor: 1/2*(trace(C)^2-trace(C^2))
    ArrayNIP I2 = 0.5*(I1*I1 - C11 * C11 - C22 * C22 - C33 * C33) - C12 * C12 - C13 * C13 - C23 * C23;

    //Calculate the determinate of F (commonly denoted as J)
    ArrayNIP detF = F11 * F22*F33 + F12 * F23*F31 + F13 * F21*F32 - F11 * F23*F32 - F12 * F21*F33 - F13 * F22*F31;

    ArrayNIP detF_m2_3 = detF.pow(-2.0 / 3.0);

    double c10 = GetMaterial()->Get_c10();
    double c01 = GetMaterial()->Get_c01();
    double k = GetMaterial()->Get_k();
    double mu = GetMaterial()->Get_mu();

    ArrayNIP kGQmu_over_detF3 = mu * m_kGQ / (detF*detF*detF);
    ArrayNIP KK = -Kfactor * kGQmu_over_detF3;
    ArrayNIP KR = -Rfactor * kGQmu_over_detF3;
    ArrayNIP SPK2_Scale = -3.0 / detF;

    // Calculate the time derivative of the Greens-Lagrange strain tensor at the current point (symmetric):
    ArrayNIP Edot11 = F11 * Fdot11 + F21 * Fdot21 + F31 * Fdot31;
    ArrayNIP Edot22 = F12 * Fdot12 + F22 * Fdot22 + F32 * Fdot32;
    ArrayNIP Edot33 = F13 * Fdot13 + F23 * Fdot23 + F33 * Fdot33;
    ArrayNIP Edot12 = 0.5*(F11 * Fdot12 + F12 * Fdot11 + F21 * Fdot22 + F22 * Fdot21 + F31 * Fdot32 + F32 * Fdot31);
    ArrayNIP Edot13 = 0.5*(F11 * Fdot13 + F13 * Fdot11 + F21 * Fdot23 + F23 * Fdot21 + F31 * Fdot33 + F33 * Fdot31);
    ArrayNIP Edot23 = 0.5*(F12 * Fdot13 + F13 * Fdot12 + F22 * Fdot23 + F23 * Fdot22 + F32 * Fdot33 + F33 * Fdot32);

    //Inverse of C times detF^2 (since all the detF terms will be handled together)
    ArrayNIP CInv11 = C22 * C33 - C23 * C23;
    ArrayNIP CInv22 = C11 * C33 - C13 * C13;
    ArrayNIP CInv33 = C11 * C22 - C12 * C12;
    ArrayNIP CInv12 = C13 * C23 - C12 * C33;
    ArrayNIP CInv13 = C12 * C23 - C13 * C22;
    ArrayNIP CInv23 = C13 * C12 - C11 * C23;

    //Calculate the Stress from the viscosity law (scaled by -Kfactor * m_kGQ)
    ArrayNIP SPK2_NLKV_1 = KK * (Edot11*CInv11*CInv11 + 2.0 * Edot12*CInv11*CInv12 + 2.0 * Edot13*CInv11*CInv13 + Edot22 * CInv12*CInv12 + 2.0 * Edot23*CInv12*CInv13 + Edot33 * CInv13*CInv13);
    ArrayNIP SPK2_NLKV_2 = KK * (Edot11*CInv12*CInv12 + 2.0 * Edot12*CInv12*CInv22 + 2.0 * Edot13*CInv12*CInv23 + Edot22 * CInv22*CInv22 + 2.0 * Edot23*CInv22*CInv23 + Edot33 * CInv23*CInv23);
    ArrayNIP SPK2_NLKV_3 = KK * (Edot11*CInv13*CInv13 + 2.0 * Edot12*CInv13*CInv23 + 2.0 * Edot13*CInv13*CInv33 + Edot22 * CInv23*CInv23 + 2.0 * Edot23*CInv23*CInv33 + Edot33 * CInv33*CInv33);
    ArrayNIP SPK2_NLKV_4 = KK * (Edot11*CInv12*CInv13 + Edot12 * (CInv12*CInv23 + CInv22 * CInv13) + Edot13 * (CInv12*CInv33 + CInv13 * CInv23) + Edot22 * CInv22*CInv23 + Edot23 * (CInv23*CInv23 + CInv22 * CInv33) + Edot33 * CInv23*CInv33);
    ArrayNIP SPK2_NLKV_5 = KK * (Edot11*CInv11*CInv13 + Edot12 * (CInv11*CInv23 + CInv12 * CInv13) + Edot13 * (CInv13*CInv13 + CInv11 * CInv33) + Edot22 * CInv12*CInv23 + Edot23 * (CInv12*CInv33 + CInv13 * CInv23) + Edot33 * CInv13*CInv33);
    ArrayNIP SPK2_NLKV_6 = KK * (Edot11*CInv11*CInv12 + Edot12 * (CInv12*CInv12 + CInv11 * CInv22) + Edot13 * (CInv11*CInv23 + CInv12 * CInv13) + Edot22 * CInv12*CInv22 + Edot23 * (CInv12*CInv23 + CInv22 * CInv13) + Edot33 * CInv13*CInv23);

    ArrayNIP EdotC1_CInvC1 = Edot11 * CInv11 + Edot12 * CInv12 + Edot13 * CInv13;
    ArrayNIP EdotC2_CInvC1 = Edot12 * CInv11 + Edot22 * CInv12 + Edot23 * CInv13;
    ArrayNIP EdotC3_CInvC1 = Edot13 * CInv11 + Edot23 * CInv12 + Edot33 * CInv13;
    ArrayNIP EdotC1_CInvC2 = Edot11 * CInv12 + Edot12 * CInv22 + Edot13 * CInv23;
    ArrayNIP EdotC2_CInvC2 = Edot12 * CInv12 + Edot22 * CInv22 + Edot23 * CInv23;
    ArrayNIP EdotC3_CInvC2 = Edot13 * CInv12 + Edot23 * CInv22 + Edot33 * CInv23;
    ArrayNIP EdotC1_CInvC3 = Edot11 * CInv13 + Edot12 * CInv23 + Edot13 * CInv33;
    ArrayNIP EdotC2_CInvC3 = Edot12 * CInv13 + Edot22 * CInv23 + Edot23 * CInv33;
    ArrayNIP EdotC3_CInvC3 = Edot13 * CInv13 + Edot23 * CInv23 + Edot33 * CInv33;


    ArrayNIP BA11 = F22 * F33 - F23 * F32;
    ArrayNIP BA21 = F23 * F31 - F21 * F33;
    ArrayNIP BA31 = F21 * F32 - F22 * F31;
    ArrayNIP BA12 = F13 * F32 - F12 * F33;
    ArrayNIP BA22 = F11 * F33 - F13 * F31;
    ArrayNIP BA32 = F12 * F31 - F11 * F32;
    ArrayNIP BA13 = F12 * F23 - F13 * F22;
    ArrayNIP BA23 = F13 * F21 - F11 * F23;
    ArrayNIP BA33 = F11 * F22 - F12 * F21;

    ArrayNIP s1 = (-Kfactor * -4.0 / 3.0 * c10) * detF_m2_3 / detF * m_kGQ;
    ArrayNIP s2 = (-Kfactor * 4.0 * c01) * detF_m2_3*detF_m2_3 * m_kGQ;
    ArrayNIP s3 = (2.0 / 3.0) * s2 / detF;

    ChMatrixNM<double, 3 * NSF, NIP> Left;
    ChMatrixNM<double, 3 * NSF, NIP> Right;
    Matrix3Nx3N Jac;

    //Calculate the Contribution from the dC11 / de terms
    {
        ArrayNIP ScaleNLKV_11cFdot = CInv11 * CInv11;
        ArrayNIP ScaleNLKV_22cFdot = CInv12 * CInv12;
        ArrayNIP ScaleNLKV_33cFdot = CInv13 * CInv13;
        ArrayNIP ScaleNLKV_12cFdot = CInv11 * CInv12;
        ArrayNIP ScaleNLKV_13cFdot = CInv11 * CInv13;
        ArrayNIP ScaleNLKV_23cFdot = CInv12 * CInv13;
        ArrayNIP ScaleNLKV_11cF = KR * ScaleNLKV_11cFdot;
        ArrayNIP ScaleNLKV_22cF = 4.0 * KK * (C33 * EdotC1_CInvC1 - C13 * EdotC3_CInvC1) + KR * ScaleNLKV_22cFdot;
        ArrayNIP ScaleNLKV_33cF = 4.0 * KK * (C22 * EdotC1_CInvC1 - C12 * EdotC2_CInvC1) + KR * ScaleNLKV_33cFdot;
        ArrayNIP ScaleNLKV_12cF = 2.0 * KK * (C23 * EdotC3_CInvC1 - C33 * EdotC2_CInvC1) + KR * ScaleNLKV_12cFdot;
        ArrayNIP ScaleNLKV_13cF = 2.0 * KK * (C23 * EdotC2_CInvC1 - C22 * EdotC3_CInvC1) + KR * ScaleNLKV_13cFdot;
        ArrayNIP ScaleNLKV_23cF = 2.0 * KK * (C12 * EdotC3_CInvC1 + C13 * EdotC2_CInvC1 - 2.0 * C23 * EdotC1_CInvC1) + KR * ScaleNLKV_23cFdot;
        ScaleNLKV_11cFdot *= KK;
        ScaleNLKV_22cFdot *= KK;
        ScaleNLKV_33cFdot *= KK;
        ScaleNLKV_12cFdot *= KK;
        ScaleNLKV_13cFdot *= KK;
        ScaleNLKV_23cFdot *= KK;

        ArrayNIP Partial_SPK2_NLKV_1_de_combined11 = SPK2_Scale * SPK2_NLKV_1 * BA22 + ScaleNLKV_11cFdot * Fdot11 + ScaleNLKV_12cFdot * Fdot12 + ScaleNLKV_13cFdot * Fdot13 + ScaleNLKV_11cF * F11 + ScaleNLKV_12cF * F12 + ScaleNLKV_13cF * F13;
        ArrayNIP Partial_SPK2_NLKV_1_de_combined12 = SPK2_Scale * SPK2_NLKV_1 * BA23 + ScaleNLKV_11cFdot * Fdot21 + ScaleNLKV_12cFdot * Fdot22 + ScaleNLKV_13cFdot * Fdot23 + ScaleNLKV_11cF * F21 + ScaleNLKV_12cF * F22 + ScaleNLKV_13cF * F23;
        ArrayNIP Partial_SPK2_NLKV_1_de_combined13 = SPK2_Scale * SPK2_NLKV_1 * BA13 + ScaleNLKV_11cFdot * Fdot31 + ScaleNLKV_12cFdot * Fdot32 + ScaleNLKV_13cFdot * Fdot33 + ScaleNLKV_11cF * F31 + ScaleNLKV_12cF * F32 + ScaleNLKV_13cF * F33;
        ArrayNIP Partial_SPK2_NLKV_1_de_combined21 = SPK2_Scale * SPK2_NLKV_1 * BA32 + ScaleNLKV_12cFdot * Fdot11 + ScaleNLKV_22cFdot * Fdot12 + ScaleNLKV_23cFdot * Fdot13 + ScaleNLKV_12cF * F11 + ScaleNLKV_22cF * F12 + ScaleNLKV_23cF * F13;
        ArrayNIP Partial_SPK2_NLKV_1_de_combined22 = SPK2_Scale * SPK2_NLKV_1 * BA33 + ScaleNLKV_12cFdot * Fdot21 + ScaleNLKV_22cFdot * Fdot22 + ScaleNLKV_23cFdot * Fdot23 + ScaleNLKV_12cF * F21 + ScaleNLKV_22cF * F22 + ScaleNLKV_23cF * F23;
        ArrayNIP Partial_SPK2_NLKV_1_de_combined23 = SPK2_Scale * SPK2_NLKV_1 * BA23 + ScaleNLKV_12cFdot * Fdot31 + ScaleNLKV_22cFdot * Fdot32 + ScaleNLKV_23cFdot * Fdot33 + ScaleNLKV_12cF * F31 + ScaleNLKV_22cF * F32 + ScaleNLKV_23cF * F33;
        ArrayNIP Partial_SPK2_NLKV_1_de_combined31 = SPK2_Scale * SPK2_NLKV_1 * BA31 + ScaleNLKV_13cFdot * Fdot11 + ScaleNLKV_23cFdot * Fdot12 + ScaleNLKV_33cFdot * Fdot13 + ScaleNLKV_13cF * F11 + ScaleNLKV_23cF * F12 + ScaleNLKV_33cF * F13;
        ArrayNIP Partial_SPK2_NLKV_1_de_combined32 = SPK2_Scale * SPK2_NLKV_1 * BA32 + ScaleNLKV_13cFdot * Fdot21 + ScaleNLKV_23cFdot * Fdot22 + ScaleNLKV_33cFdot * Fdot23 + ScaleNLKV_13cF * F21 + ScaleNLKV_23cF * F22 + ScaleNLKV_33cF * F23;
        ArrayNIP Partial_SPK2_NLKV_1_de_combined33 = SPK2_Scale * SPK2_NLKV_1 * BA33 + ScaleNLKV_13cFdot * Fdot31 + ScaleNLKV_23cFdot * Fdot32 + ScaleNLKV_33cFdot * Fdot33 + ScaleNLKV_13cF * F31 + ScaleNLKV_23cF * F32 + ScaleNLKV_33cF * F33;


        ArrayNIP s4 = s1 + s3 * (C11 - I1);

        ArrayNIP S1A = s4 * BA11 + Partial_SPK2_NLKV_1_de_combined11;
        ArrayNIP S2A = s4 * BA21 + s2 * F12 + Partial_SPK2_NLKV_1_de_combined21;
        ArrayNIP S3A = s4 * BA31 + s2 * F13 + Partial_SPK2_NLKV_1_de_combined31;

        ArrayNIP S1B = s4 * BA12 + Partial_SPK2_NLKV_1_de_combined12;
        ArrayNIP S2B = s4 * BA22 + s2 * F22 + Partial_SPK2_NLKV_1_de_combined22;
        ArrayNIP S3B = s4 * BA32 + s2 * F23 + Partial_SPK2_NLKV_1_de_combined32;

        ArrayNIP S1C = s4 * BA13 + Partial_SPK2_NLKV_1_de_combined13;
        ArrayNIP S2C = s4 * BA23 + s2 * F32 + Partial_SPK2_NLKV_1_de_combined23;
        ArrayNIP S3C = s4 * BA33 + s2 * F33 + Partial_SPK2_NLKV_1_de_combined33;

        for (auto i = 0; i < NSF; i++) {
            Left.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * F11;
            Left.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * F21;
            Left.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * F31;

            Right.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1A +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2A +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3A;
            Right.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1B +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2B +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3B;
            Right.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1C +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2C +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3C;
        }
        Jac.noalias() = Left * Right.transpose();
    }
    //Calculate the Contribution from the dC22 / de terms
    {
        ArrayNIP ScaleNLKV_11cFdot = CInv12 * CInv12;
        ArrayNIP ScaleNLKV_22cFdot = CInv22 * CInv22;
        ArrayNIP ScaleNLKV_33cFdot = CInv23 * CInv23;
        ArrayNIP ScaleNLKV_12cFdot = CInv12 * CInv22;
        ArrayNIP ScaleNLKV_13cFdot = CInv12 * CInv23;
        ArrayNIP ScaleNLKV_23cFdot = CInv22 * CInv23;
        ArrayNIP ScaleNLKV_11cF = 4.0 * KK * (C33 * EdotC2_CInvC2 - C23 * EdotC3_CInvC2) + KR * ScaleNLKV_11cFdot;
        ArrayNIP ScaleNLKV_22cF = KR * ScaleNLKV_22cFdot;
        ArrayNIP ScaleNLKV_33cF = 4.0 * KK * (C11 * EdotC2_CInvC2 - C12 * EdotC1_CInvC2) + KR * ScaleNLKV_33cFdot;
        ArrayNIP ScaleNLKV_12cF = 2.0 * KK * (C13 * EdotC3_CInvC2 - C33 * EdotC1_CInvC2) + KR * ScaleNLKV_12cFdot;
        ArrayNIP ScaleNLKV_13cF = 2.0 * KK * (C12 * EdotC3_CInvC2 + C23 * EdotC1_CInvC2 - 2.0 * C13 * EdotC2_CInvC2) + KR * ScaleNLKV_13cFdot;
        ArrayNIP ScaleNLKV_23cF = 2.0 * KK * (C13 * EdotC1_CInvC2 - C11 * EdotC3_CInvC2) + KR * ScaleNLKV_23cFdot;
        ScaleNLKV_11cFdot *= KK;
        ScaleNLKV_22cFdot *= KK;
        ScaleNLKV_33cFdot *= KK;
        ScaleNLKV_12cFdot *= KK;
        ScaleNLKV_13cFdot *= KK;
        ScaleNLKV_23cFdot *= KK;

        ArrayNIP Partial_SPK2_NLKV_2_de_combined11 = SPK2_Scale * SPK2_NLKV_2 * BA22 + ScaleNLKV_11cFdot * Fdot11 + ScaleNLKV_12cFdot * Fdot12 + ScaleNLKV_13cFdot * Fdot13 + ScaleNLKV_11cF * F11 + ScaleNLKV_12cF * F12 + ScaleNLKV_13cF * F13;
        ArrayNIP Partial_SPK2_NLKV_2_de_combined12 = SPK2_Scale * SPK2_NLKV_2 * BA23 + ScaleNLKV_11cFdot * Fdot21 + ScaleNLKV_12cFdot * Fdot22 + ScaleNLKV_13cFdot * Fdot23 + ScaleNLKV_11cF * F21 + ScaleNLKV_12cF * F22 + ScaleNLKV_13cF * F23;
        ArrayNIP Partial_SPK2_NLKV_2_de_combined13 = SPK2_Scale * SPK2_NLKV_2 * BA13 + ScaleNLKV_11cFdot * Fdot31 + ScaleNLKV_12cFdot * Fdot32 + ScaleNLKV_13cFdot * Fdot33 + ScaleNLKV_11cF * F31 + ScaleNLKV_12cF * F32 + ScaleNLKV_13cF * F33;
        ArrayNIP Partial_SPK2_NLKV_2_de_combined21 = SPK2_Scale * SPK2_NLKV_2 * BA32 + ScaleNLKV_12cFdot * Fdot11 + ScaleNLKV_22cFdot * Fdot12 + ScaleNLKV_23cFdot * Fdot13 + ScaleNLKV_12cF * F11 + ScaleNLKV_22cF * F12 + ScaleNLKV_23cF * F13;
        ArrayNIP Partial_SPK2_NLKV_2_de_combined22 = SPK2_Scale * SPK2_NLKV_2 * BA33 + ScaleNLKV_12cFdot * Fdot21 + ScaleNLKV_22cFdot * Fdot22 + ScaleNLKV_23cFdot * Fdot23 + ScaleNLKV_12cF * F21 + ScaleNLKV_22cF * F22 + ScaleNLKV_23cF * F23;
        ArrayNIP Partial_SPK2_NLKV_2_de_combined23 = SPK2_Scale * SPK2_NLKV_2 * BA23 + ScaleNLKV_12cFdot * Fdot31 + ScaleNLKV_22cFdot * Fdot32 + ScaleNLKV_23cFdot * Fdot33 + ScaleNLKV_12cF * F31 + ScaleNLKV_22cF * F32 + ScaleNLKV_23cF * F33;
        ArrayNIP Partial_SPK2_NLKV_2_de_combined31 = SPK2_Scale * SPK2_NLKV_2 * BA31 + ScaleNLKV_13cFdot * Fdot11 + ScaleNLKV_23cFdot * Fdot12 + ScaleNLKV_33cFdot * Fdot13 + ScaleNLKV_13cF * F11 + ScaleNLKV_23cF * F12 + ScaleNLKV_33cF * F13;
        ArrayNIP Partial_SPK2_NLKV_2_de_combined32 = SPK2_Scale * SPK2_NLKV_2 * BA32 + ScaleNLKV_13cFdot * Fdot21 + ScaleNLKV_23cFdot * Fdot22 + ScaleNLKV_33cFdot * Fdot23 + ScaleNLKV_13cF * F21 + ScaleNLKV_23cF * F22 + ScaleNLKV_33cF * F23;
        ArrayNIP Partial_SPK2_NLKV_2_de_combined33 = SPK2_Scale * SPK2_NLKV_2 * BA33 + ScaleNLKV_13cFdot * Fdot31 + ScaleNLKV_23cFdot * Fdot32 + ScaleNLKV_33cFdot * Fdot33 + ScaleNLKV_13cF * F31 + ScaleNLKV_23cF * F32 + ScaleNLKV_33cF * F33;

        
        ArrayNIP s4 = s1 + s3 * (C22 - I1);

        ArrayNIP S1A = s4 * BA11 + s2 * F11 + Partial_SPK2_NLKV_2_de_combined11;
        ArrayNIP S2A = s4 * BA21 + Partial_SPK2_NLKV_2_de_combined21;
        ArrayNIP S3A = s4 * BA31 + s2 * F13 + Partial_SPK2_NLKV_2_de_combined31;

        ArrayNIP S1B = s4 * BA12 + s2 * F21 + Partial_SPK2_NLKV_2_de_combined12;
        ArrayNIP S2B = s4 * BA22 + Partial_SPK2_NLKV_2_de_combined22;
        ArrayNIP S3B = s4 * BA32 + s2 * F23 + Partial_SPK2_NLKV_2_de_combined32;

        ArrayNIP S1C = s4 * BA13 + s2 * F31 + Partial_SPK2_NLKV_2_de_combined13;
        ArrayNIP S2C = s4 * BA23 + Partial_SPK2_NLKV_2_de_combined23;
        ArrayNIP S3C = s4 * BA33 + s2 * F33 + Partial_SPK2_NLKV_2_de_combined33;

        for (auto i = 0; i < NSF; i++) {
            Left.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 1 * NIP).array() * F12;
            Left.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 1 * NIP).array() * F22;
            Left.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 1 * NIP).array() * F32;
            Right.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1A +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2A +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3A;
            Right.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1B +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2B +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3B;
            Right.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1C +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2C +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3C;
        }
        Jac.noalias() += Left * Right.transpose();
    }
    //Calculate the Contribution from the dC33 / de terms
    {
        ArrayNIP ScaleNLKV_11cFdot = CInv13 * CInv13;
        ArrayNIP ScaleNLKV_22cFdot = CInv23 * CInv23;
        ArrayNIP ScaleNLKV_33cFdot = CInv33 * CInv33;
        ArrayNIP ScaleNLKV_12cFdot = CInv13 * CInv23;
        ArrayNIP ScaleNLKV_13cFdot = CInv13 * CInv33;
        ArrayNIP ScaleNLKV_23cFdot = CInv23 * CInv33;
        ArrayNIP ScaleNLKV_11cF = 4.0 * KK * (C22 * EdotC3_CInvC3 - C23 * EdotC2_CInvC3) + KR * ScaleNLKV_11cFdot;
        ArrayNIP ScaleNLKV_22cF = 4.0 * KK * (C11 * EdotC3_CInvC3 - C13 * EdotC1_CInvC3) + KR * ScaleNLKV_22cFdot;
        ArrayNIP ScaleNLKV_33cF = KR * ScaleNLKV_33cFdot;
        ArrayNIP ScaleNLKV_12cF = 2.0 * KK * (C13 * EdotC2_CInvC3 + C23 * EdotC1_CInvC3 - 2.0 * C12 * EdotC3_CInvC3) + KR * ScaleNLKV_12cFdot;
        ArrayNIP ScaleNLKV_13cF = 2.0 * KK * (C12 * EdotC2_CInvC3 - C22 * EdotC1_CInvC3) + KR * ScaleNLKV_13cFdot;
        ArrayNIP ScaleNLKV_23cF = 2.0 * KK * (C12 * EdotC1_CInvC3 - C11 * EdotC2_CInvC3) + KR * ScaleNLKV_23cFdot;
        ScaleNLKV_11cFdot *= KK;
        ScaleNLKV_22cFdot *= KK;
        ScaleNLKV_33cFdot *= KK;
        ScaleNLKV_12cFdot *= KK;
        ScaleNLKV_13cFdot *= KK;
        ScaleNLKV_23cFdot *= KK;

        ArrayNIP Partial_SPK2_NLKV_3_de_combined11 = SPK2_Scale * SPK2_NLKV_3 * BA22 + ScaleNLKV_11cFdot * Fdot11 + ScaleNLKV_12cFdot * Fdot12 + ScaleNLKV_13cFdot * Fdot13 + ScaleNLKV_11cF * F11 + ScaleNLKV_12cF * F12 + ScaleNLKV_13cF * F13;
        ArrayNIP Partial_SPK2_NLKV_3_de_combined12 = SPK2_Scale * SPK2_NLKV_3 * BA23 + ScaleNLKV_11cFdot * Fdot21 + ScaleNLKV_12cFdot * Fdot22 + ScaleNLKV_13cFdot * Fdot23 + ScaleNLKV_11cF * F21 + ScaleNLKV_12cF * F22 + ScaleNLKV_13cF * F23;
        ArrayNIP Partial_SPK2_NLKV_3_de_combined13 = SPK2_Scale * SPK2_NLKV_3 * BA13 + ScaleNLKV_11cFdot * Fdot31 + ScaleNLKV_12cFdot * Fdot32 + ScaleNLKV_13cFdot * Fdot33 + ScaleNLKV_11cF * F31 + ScaleNLKV_12cF * F32 + ScaleNLKV_13cF * F33;
        ArrayNIP Partial_SPK2_NLKV_3_de_combined21 = SPK2_Scale * SPK2_NLKV_3 * BA32 + ScaleNLKV_12cFdot * Fdot11 + ScaleNLKV_22cFdot * Fdot12 + ScaleNLKV_23cFdot * Fdot13 + ScaleNLKV_12cF * F11 + ScaleNLKV_22cF * F12 + ScaleNLKV_23cF * F13;
        ArrayNIP Partial_SPK2_NLKV_3_de_combined22 = SPK2_Scale * SPK2_NLKV_3 * BA33 + ScaleNLKV_12cFdot * Fdot21 + ScaleNLKV_22cFdot * Fdot22 + ScaleNLKV_23cFdot * Fdot23 + ScaleNLKV_12cF * F21 + ScaleNLKV_22cF * F22 + ScaleNLKV_23cF * F23;
        ArrayNIP Partial_SPK2_NLKV_3_de_combined23 = SPK2_Scale * SPK2_NLKV_3 * BA23 + ScaleNLKV_12cFdot * Fdot31 + ScaleNLKV_22cFdot * Fdot32 + ScaleNLKV_23cFdot * Fdot33 + ScaleNLKV_12cF * F31 + ScaleNLKV_22cF * F32 + ScaleNLKV_23cF * F33;
        ArrayNIP Partial_SPK2_NLKV_3_de_combined31 = SPK2_Scale * SPK2_NLKV_3 * BA31 + ScaleNLKV_13cFdot * Fdot11 + ScaleNLKV_23cFdot * Fdot12 + ScaleNLKV_33cFdot * Fdot13 + ScaleNLKV_13cF * F11 + ScaleNLKV_23cF * F12 + ScaleNLKV_33cF * F13;
        ArrayNIP Partial_SPK2_NLKV_3_de_combined32 = SPK2_Scale * SPK2_NLKV_3 * BA32 + ScaleNLKV_13cFdot * Fdot21 + ScaleNLKV_23cFdot * Fdot22 + ScaleNLKV_33cFdot * Fdot23 + ScaleNLKV_13cF * F21 + ScaleNLKV_23cF * F22 + ScaleNLKV_33cF * F23;
        ArrayNIP Partial_SPK2_NLKV_3_de_combined33 = SPK2_Scale * SPK2_NLKV_3 * BA33 + ScaleNLKV_13cFdot * Fdot31 + ScaleNLKV_23cFdot * Fdot32 + ScaleNLKV_33cFdot * Fdot33 + ScaleNLKV_13cF * F31 + ScaleNLKV_23cF * F32 + ScaleNLKV_33cF * F33;


        ArrayNIP s4 = s1 + s3 * (C33 - I1);

        ArrayNIP S1A = s4 * BA11 + s2 * F11 + Partial_SPK2_NLKV_3_de_combined11;
        ArrayNIP S2A = s4 * BA21 + s2 * F12 + Partial_SPK2_NLKV_3_de_combined21;
        ArrayNIP S3A = s4 * BA31 + Partial_SPK2_NLKV_3_de_combined31;

        ArrayNIP S1B = s4 * BA12 + s2 * F21 + Partial_SPK2_NLKV_3_de_combined12;
        ArrayNIP S2B = s4 * BA22 + s2 * F22 + Partial_SPK2_NLKV_3_de_combined22;
        ArrayNIP S3B = s4 * BA32 + Partial_SPK2_NLKV_3_de_combined32;

        ArrayNIP S1C = s4 * BA13 + s2 * F31 + Partial_SPK2_NLKV_3_de_combined13;
        ArrayNIP S2C = s4 * BA23 + s2 * F32 + Partial_SPK2_NLKV_3_de_combined23;
        ArrayNIP S3C = s4 * BA33 + Partial_SPK2_NLKV_3_de_combined33;

        for (auto i = 0; i < NSF; i++) {
            Left.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 2 * NIP).array() * F13;
            Left.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 2 * NIP).array() * F23;
            Left.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 2 * NIP).array() * F33;
            Right.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1A +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2A +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3A;
            Right.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1B +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2B +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3B;
            Right.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1C +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2C +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3C;
        }
        Jac.noalias() += Left * Right.transpose();
    }

    ArrayNIP s5 = -0.5 * s2;
    // Calculate the Contribution from the dC23/de terms
    {
        ArrayNIP ScaleNLKV_11cFdot = CInv12 * CInv13;
        ArrayNIP ScaleNLKV_22cFdot = CInv22 * CInv23;
        ArrayNIP ScaleNLKV_33cFdot = CInv23 * CInv33;
        ArrayNIP ScaleNLKV_12cFdot = 0.5 * (CInv12 * CInv23 + CInv22 * CInv13);
        ArrayNIP ScaleNLKV_13cFdot = 0.5 * (CInv12 * CInv33 + CInv13 * CInv23);
        ArrayNIP ScaleNLKV_23cFdot = 0.5 * (CInv23 * CInv23 + CInv22 * CInv33);
        ArrayNIP ScaleNLKV_11cF = 2.0 * KK * (C33 * EdotC2_CInvC3 + C22 * EdotC3_CInvC2 - C23 * (EdotC2_CInvC2 + EdotC3_CInvC3)) + KR * ScaleNLKV_11cFdot;
        ArrayNIP ScaleNLKV_22cF = 2.0 * KK * (C11 * EdotC3_CInvC2 - C13 * EdotC1_CInvC2) + KR * ScaleNLKV_22cFdot;
        ArrayNIP ScaleNLKV_33cF = 2.0 * KK * (C11 * EdotC2_CInvC3 - C12 * EdotC1_CInvC3) + KR * ScaleNLKV_33cFdot;
        ArrayNIP ScaleNLKV_12cF = KK * (C13 * (EdotC2_CInvC2 + EdotC3_CInvC3) + C23 * EdotC1_CInvC2 - 2.0 * C12 * EdotC3_CInvC2 - C33 * EdotC1_CInvC3) + KR * ScaleNLKV_12cFdot;
        ArrayNIP ScaleNLKV_13cF = KK * (C12 * (EdotC2_CInvC2 + EdotC3_CInvC3) + C23 * EdotC1_CInvC3 - 2.0 * C13 * EdotC2_CInvC3 - C22 * EdotC1_CInvC2) + KR * ScaleNLKV_13cFdot;
        ArrayNIP ScaleNLKV_23cF = KK * (C12 * EdotC1_CInvC2 + C13 * EdotC1_CInvC3 - C11 * (EdotC2_CInvC2 + EdotC3_CInvC3)) + KR * ScaleNLKV_23cFdot;
        ScaleNLKV_11cFdot *= KK;
        ScaleNLKV_22cFdot *= KK;
        ScaleNLKV_33cFdot *= KK;
        ScaleNLKV_12cFdot *= KK;
        ScaleNLKV_13cFdot *= KK;
        ScaleNLKV_23cFdot *= KK;

        ArrayNIP Partial_SPK2_NLKV_4_de_combined11 = SPK2_Scale * SPK2_NLKV_4 * BA22 + ScaleNLKV_11cFdot * Fdot11 + ScaleNLKV_12cFdot * Fdot12 + ScaleNLKV_13cFdot * Fdot13 + ScaleNLKV_11cF * F11 + ScaleNLKV_12cF * F12 + ScaleNLKV_13cF * F13;
        ArrayNIP Partial_SPK2_NLKV_4_de_combined12 = SPK2_Scale * SPK2_NLKV_4 * BA23 + ScaleNLKV_11cFdot * Fdot21 + ScaleNLKV_12cFdot * Fdot22 + ScaleNLKV_13cFdot * Fdot23 + ScaleNLKV_11cF * F21 + ScaleNLKV_12cF * F22 + ScaleNLKV_13cF * F23;
        ArrayNIP Partial_SPK2_NLKV_4_de_combined13 = SPK2_Scale * SPK2_NLKV_4 * BA13 + ScaleNLKV_11cFdot * Fdot31 + ScaleNLKV_12cFdot * Fdot32 + ScaleNLKV_13cFdot * Fdot33 + ScaleNLKV_11cF * F31 + ScaleNLKV_12cF * F32 + ScaleNLKV_13cF * F33;
        ArrayNIP Partial_SPK2_NLKV_4_de_combined21 = SPK2_Scale * SPK2_NLKV_4 * BA32 + ScaleNLKV_12cFdot * Fdot11 + ScaleNLKV_22cFdot * Fdot12 + ScaleNLKV_23cFdot * Fdot13 + ScaleNLKV_12cF * F11 + ScaleNLKV_22cF * F12 + ScaleNLKV_23cF * F13;
        ArrayNIP Partial_SPK2_NLKV_4_de_combined22 = SPK2_Scale * SPK2_NLKV_4 * BA33 + ScaleNLKV_12cFdot * Fdot21 + ScaleNLKV_22cFdot * Fdot22 + ScaleNLKV_23cFdot * Fdot23 + ScaleNLKV_12cF * F21 + ScaleNLKV_22cF * F22 + ScaleNLKV_23cF * F23;
        ArrayNIP Partial_SPK2_NLKV_4_de_combined23 = SPK2_Scale * SPK2_NLKV_4 * BA23 + ScaleNLKV_12cFdot * Fdot31 + ScaleNLKV_22cFdot * Fdot32 + ScaleNLKV_23cFdot * Fdot33 + ScaleNLKV_12cF * F31 + ScaleNLKV_22cF * F32 + ScaleNLKV_23cF * F33;
        ArrayNIP Partial_SPK2_NLKV_4_de_combined31 = SPK2_Scale * SPK2_NLKV_4 * BA31 + ScaleNLKV_13cFdot * Fdot11 + ScaleNLKV_23cFdot * Fdot12 + ScaleNLKV_33cFdot * Fdot13 + ScaleNLKV_13cF * F11 + ScaleNLKV_23cF * F12 + ScaleNLKV_33cF * F13;
        ArrayNIP Partial_SPK2_NLKV_4_de_combined32 = SPK2_Scale * SPK2_NLKV_4 * BA32 + ScaleNLKV_13cFdot * Fdot21 + ScaleNLKV_23cFdot * Fdot22 + ScaleNLKV_33cFdot * Fdot23 + ScaleNLKV_13cF * F21 + ScaleNLKV_23cF * F22 + ScaleNLKV_33cF * F23;
        ArrayNIP Partial_SPK2_NLKV_4_de_combined33 = SPK2_Scale * SPK2_NLKV_4 * BA33 + ScaleNLKV_13cFdot * Fdot31 + ScaleNLKV_23cFdot * Fdot32 + ScaleNLKV_33cFdot * Fdot33 + ScaleNLKV_13cF * F31 + ScaleNLKV_23cF * F32 + ScaleNLKV_33cF * F33;


        ArrayNIP s4 = s3 * C23;

        ArrayNIP S1A = s4 * BA11 + Partial_SPK2_NLKV_4_de_combined11;
        ArrayNIP S2A = s4 * BA21 + s5 * F13 + Partial_SPK2_NLKV_4_de_combined21;
        ArrayNIP S3A = s4 * BA31 + s5 * F12 + Partial_SPK2_NLKV_4_de_combined31;

        ArrayNIP S1B = s4 * BA12 + Partial_SPK2_NLKV_4_de_combined12;
        ArrayNIP S2B = s4 * BA22 + s5 * F23 + Partial_SPK2_NLKV_4_de_combined22;
        ArrayNIP S3B = s4 * BA32 + s5 * F22 + Partial_SPK2_NLKV_4_de_combined32;

        ArrayNIP S1C = s4 * BA13 + Partial_SPK2_NLKV_4_de_combined13;
        ArrayNIP S2C = s4 * BA23 + s5 * F33 + Partial_SPK2_NLKV_4_de_combined23;
        ArrayNIP S3C = s4 * BA33 + s5 * F32 + Partial_SPK2_NLKV_4_de_combined33;

        for (auto i = 0; i < NSF; i++) {
            Left.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 1 * NIP).array() * F13 + m_SD.block<1, NIP>(i, 2 * NIP).array() * F12;
            Left.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 1 * NIP).array() * F23 + m_SD.block<1, NIP>(i, 2 * NIP).array() * F22;
            Left.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 1 * NIP).array() * F33 + m_SD.block<1, NIP>(i, 2 * NIP).array() * F32;
            Right.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1A +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2A +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3A;
            Right.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1B +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2B +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3B;
            Right.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1C +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2C +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3C;
        }
        Jac.noalias() += Left * Right.transpose();
    }
    // Calculate the Contribution from the dC13/de terms
    {
        ArrayNIP ScaleNLKV_11cFdot = CInv11 * CInv13;
        ArrayNIP ScaleNLKV_22cFdot = CInv12 * CInv23;
        ArrayNIP ScaleNLKV_33cFdot = CInv13 * CInv33;
        ArrayNIP ScaleNLKV_12cFdot = 0.5 * (CInv11 * CInv23 + CInv12 * CInv13);
        ArrayNIP ScaleNLKV_13cFdot = 0.5 * (CInv11 * CInv33 + CInv13 * CInv13);
        ArrayNIP ScaleNLKV_23cFdot = 0.5 * (CInv12 * CInv33 + CInv13 * CInv23);
        ArrayNIP ScaleNLKV_11cF = 2.0 * KK * (C22 * EdotC3_CInvC1 - C23 * EdotC2_CInvC1) + KR * ScaleNLKV_11cFdot;
        ArrayNIP ScaleNLKV_22cF = 2.0 * KK * (C11 * EdotC3_CInvC1 + C33 * EdotC1_CInvC3 - C13 * (EdotC1_CInvC1 + EdotC3_CInvC3)) + KR * ScaleNLKV_22cFdot;
        ArrayNIP ScaleNLKV_33cF = 2.0 * KK * (C22 * EdotC1_CInvC3 - C12 * EdotC2_CInvC3) + KR * ScaleNLKV_33cFdot;
        ArrayNIP ScaleNLKV_12cF = KK * (C13 * EdotC2_CInvC1 + C23 * (EdotC1_CInvC1 + EdotC3_CInvC3) - 2.0 * C12 * EdotC3_CInvC1 - C33 * EdotC2_CInvC3) + KR * ScaleNLKV_12cFdot;
        ArrayNIP ScaleNLKV_13cF = KK * (C12 * EdotC2_CInvC1 + C23 * EdotC2_CInvC3 - C22 * (EdotC1_CInvC1 + EdotC3_CInvC3)) + KR * ScaleNLKV_13cFdot;
        ArrayNIP ScaleNLKV_23cF = KK * (C12 * (EdotC1_CInvC1 + EdotC3_CInvC3) + C13 * EdotC2_CInvC3 - 2.0 * C23 * EdotC1_CInvC3 - C11 * EdotC2_CInvC1) + KR * ScaleNLKV_23cFdot;
        ScaleNLKV_11cFdot *= KK;
        ScaleNLKV_22cFdot *= KK;
        ScaleNLKV_33cFdot *= KK;
        ScaleNLKV_12cFdot *= KK;
        ScaleNLKV_13cFdot *= KK;
        ScaleNLKV_23cFdot *= KK;

        ArrayNIP Partial_SPK2_NLKV_5_de_combined11 = SPK2_Scale * SPK2_NLKV_5 * BA22 + ScaleNLKV_11cFdot * Fdot11 + ScaleNLKV_12cFdot * Fdot12 + ScaleNLKV_13cFdot * Fdot13 + ScaleNLKV_11cF * F11 + ScaleNLKV_12cF * F12 + ScaleNLKV_13cF * F13;
        ArrayNIP Partial_SPK2_NLKV_5_de_combined12 = SPK2_Scale * SPK2_NLKV_5 * BA23 + ScaleNLKV_11cFdot * Fdot21 + ScaleNLKV_12cFdot * Fdot22 + ScaleNLKV_13cFdot * Fdot23 + ScaleNLKV_11cF * F21 + ScaleNLKV_12cF * F22 + ScaleNLKV_13cF * F23;
        ArrayNIP Partial_SPK2_NLKV_5_de_combined13 = SPK2_Scale * SPK2_NLKV_5 * BA13 + ScaleNLKV_11cFdot * Fdot31 + ScaleNLKV_12cFdot * Fdot32 + ScaleNLKV_13cFdot * Fdot33 + ScaleNLKV_11cF * F31 + ScaleNLKV_12cF * F32 + ScaleNLKV_13cF * F33;
        ArrayNIP Partial_SPK2_NLKV_5_de_combined21 = SPK2_Scale * SPK2_NLKV_5 * BA32 + ScaleNLKV_12cFdot * Fdot11 + ScaleNLKV_22cFdot * Fdot12 + ScaleNLKV_23cFdot * Fdot13 + ScaleNLKV_12cF * F11 + ScaleNLKV_22cF * F12 + ScaleNLKV_23cF * F13;
        ArrayNIP Partial_SPK2_NLKV_5_de_combined22 = SPK2_Scale * SPK2_NLKV_5 * BA33 + ScaleNLKV_12cFdot * Fdot21 + ScaleNLKV_22cFdot * Fdot22 + ScaleNLKV_23cFdot * Fdot23 + ScaleNLKV_12cF * F21 + ScaleNLKV_22cF * F22 + ScaleNLKV_23cF * F23;
        ArrayNIP Partial_SPK2_NLKV_5_de_combined23 = SPK2_Scale * SPK2_NLKV_5 * BA23 + ScaleNLKV_12cFdot * Fdot31 + ScaleNLKV_22cFdot * Fdot32 + ScaleNLKV_23cFdot * Fdot33 + ScaleNLKV_12cF * F31 + ScaleNLKV_22cF * F32 + ScaleNLKV_23cF * F33;
        ArrayNIP Partial_SPK2_NLKV_5_de_combined31 = SPK2_Scale * SPK2_NLKV_5 * BA31 + ScaleNLKV_13cFdot * Fdot11 + ScaleNLKV_23cFdot * Fdot12 + ScaleNLKV_33cFdot * Fdot13 + ScaleNLKV_13cF * F11 + ScaleNLKV_23cF * F12 + ScaleNLKV_33cF * F13;
        ArrayNIP Partial_SPK2_NLKV_5_de_combined32 = SPK2_Scale * SPK2_NLKV_5 * BA32 + ScaleNLKV_13cFdot * Fdot21 + ScaleNLKV_23cFdot * Fdot22 + ScaleNLKV_33cFdot * Fdot23 + ScaleNLKV_13cF * F21 + ScaleNLKV_23cF * F22 + ScaleNLKV_33cF * F23;
        ArrayNIP Partial_SPK2_NLKV_5_de_combined33 = SPK2_Scale * SPK2_NLKV_5 * BA33 + ScaleNLKV_13cFdot * Fdot31 + ScaleNLKV_23cFdot * Fdot32 + ScaleNLKV_33cFdot * Fdot33 + ScaleNLKV_13cF * F31 + ScaleNLKV_23cF * F32 + ScaleNLKV_33cF * F33;


        ArrayNIP s4 = s3 * C13;

        ArrayNIP S1A = s4 * BA11 + s5 * F13 + Partial_SPK2_NLKV_5_de_combined11;
        ArrayNIP S2A = s4 * BA21 + Partial_SPK2_NLKV_5_de_combined21;
        ArrayNIP S3A = s4 * BA31 + s5 * F11 + Partial_SPK2_NLKV_5_de_combined31;

        ArrayNIP S1B = s4 * BA12 + s5 * F23 + Partial_SPK2_NLKV_5_de_combined12;
        ArrayNIP S2B = s4 * BA22 + Partial_SPK2_NLKV_5_de_combined22;
        ArrayNIP S3B = s4 * BA32 + s5 * F21 + Partial_SPK2_NLKV_5_de_combined32;

        ArrayNIP S1C = s4 * BA13 + s5 * F33 + Partial_SPK2_NLKV_5_de_combined13;
        ArrayNIP S2C = s4 * BA23 + Partial_SPK2_NLKV_5_de_combined23;
        ArrayNIP S3C = s4 * BA33 + s5 * F31 + Partial_SPK2_NLKV_5_de_combined33;

        for (auto i = 0; i < NSF; i++) {
            Left.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * F13 + m_SD.block<1, NIP>(i, 2 * NIP).array() * F11;
            Left.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * F23 + m_SD.block<1, NIP>(i, 2 * NIP).array() * F21;
            Left.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * F33 + m_SD.block<1, NIP>(i, 2 * NIP).array() * F31;
            Right.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1A +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2A +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3A;
            Right.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1B +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2B +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3B;
            Right.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1C +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2C +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3C;
        }
        Jac.noalias() += Left * Right.transpose();
    }
    // Calculate the Contribution from the dC12/de terms
    {
        ArrayNIP ScaleNLKV_11cFdot = CInv11 * CInv12;
        ArrayNIP ScaleNLKV_22cFdot = CInv12 * CInv22;
        ArrayNIP ScaleNLKV_33cFdot = CInv13 * CInv23;
        ArrayNIP ScaleNLKV_12cFdot = 0.5 * (CInv11 * CInv22 + CInv12 * CInv12);
        ArrayNIP ScaleNLKV_13cFdot = 0.5 * (CInv11 * CInv23 + CInv12 * CInv13);
        ArrayNIP ScaleNLKV_23cFdot = 0.5 * (CInv12 * CInv23 + CInv22 * CInv13);
        ArrayNIP ScaleNLKV_11cF = 2.0 * KK * (C33 * EdotC2_CInvC1 - C23 * EdotC3_CInvC1) + KR * ScaleNLKV_11cFdot;
        ArrayNIP ScaleNLKV_22cF = 2.0 * KK * (C33 * EdotC1_CInvC2 - C13 * EdotC3_CInvC2) + KR * ScaleNLKV_22cFdot;
        ArrayNIP ScaleNLKV_33cF = 2.0 * KK * (C11 * EdotC2_CInvC1 + C22 * EdotC1_CInvC2 - C12 * (EdotC1_CInvC1 + EdotC2_CInvC2)) + KR * ScaleNLKV_33cFdot;
        ArrayNIP ScaleNLKV_12cF = KK * (C13 * EdotC3_CInvC1 + C23 * EdotC3_CInvC2 - C33 * (EdotC1_CInvC1 + EdotC2_CInvC2)) + KR * ScaleNLKV_12cFdot;
        ArrayNIP ScaleNLKV_13cF = KK * (C12 * EdotC3_CInvC1 + C23 * (EdotC1_CInvC1 + EdotC2_CInvC2) - 2.0 * C13 * EdotC2_CInvC1 - C22 * EdotC3_CInvC2) + KR * ScaleNLKV_13cFdot;
        ArrayNIP ScaleNLKV_23cF = KK * (C12 * EdotC3_CInvC2 + C13 * (EdotC1_CInvC1 + EdotC2_CInvC2) - 2.0 * C23 * EdotC1_CInvC2 - C11 * EdotC3_CInvC1) + KR * ScaleNLKV_23cFdot;
        ScaleNLKV_11cFdot *= KK;
        ScaleNLKV_22cFdot *= KK;
        ScaleNLKV_33cFdot *= KK;
        ScaleNLKV_12cFdot *= KK;
        ScaleNLKV_13cFdot *= KK;
        ScaleNLKV_23cFdot *= KK;

        ArrayNIP Partial_SPK2_NLKV_6_de_combined11 = SPK2_Scale * SPK2_NLKV_6 * BA22 + ScaleNLKV_11cFdot * Fdot11 + ScaleNLKV_12cFdot * Fdot12 + ScaleNLKV_13cFdot * Fdot13 + ScaleNLKV_11cF * F11 + ScaleNLKV_12cF * F12 + ScaleNLKV_13cF * F13;
        ArrayNIP Partial_SPK2_NLKV_6_de_combined12 = SPK2_Scale * SPK2_NLKV_6 * BA23 + ScaleNLKV_11cFdot * Fdot21 + ScaleNLKV_12cFdot * Fdot22 + ScaleNLKV_13cFdot * Fdot23 + ScaleNLKV_11cF * F21 + ScaleNLKV_12cF * F22 + ScaleNLKV_13cF * F23;
        ArrayNIP Partial_SPK2_NLKV_6_de_combined13 = SPK2_Scale * SPK2_NLKV_6 * BA13 + ScaleNLKV_11cFdot * Fdot31 + ScaleNLKV_12cFdot * Fdot32 + ScaleNLKV_13cFdot * Fdot33 + ScaleNLKV_11cF * F31 + ScaleNLKV_12cF * F32 + ScaleNLKV_13cF * F33;
        ArrayNIP Partial_SPK2_NLKV_6_de_combined21 = SPK2_Scale * SPK2_NLKV_6 * BA32 + ScaleNLKV_12cFdot * Fdot11 + ScaleNLKV_22cFdot * Fdot12 + ScaleNLKV_23cFdot * Fdot13 + ScaleNLKV_12cF * F11 + ScaleNLKV_22cF * F12 + ScaleNLKV_23cF * F13;
        ArrayNIP Partial_SPK2_NLKV_6_de_combined22 = SPK2_Scale * SPK2_NLKV_6 * BA33 + ScaleNLKV_12cFdot * Fdot21 + ScaleNLKV_22cFdot * Fdot22 + ScaleNLKV_23cFdot * Fdot23 + ScaleNLKV_12cF * F21 + ScaleNLKV_22cF * F22 + ScaleNLKV_23cF * F23;
        ArrayNIP Partial_SPK2_NLKV_6_de_combined23 = SPK2_Scale * SPK2_NLKV_6 * BA23 + ScaleNLKV_12cFdot * Fdot31 + ScaleNLKV_22cFdot * Fdot32 + ScaleNLKV_23cFdot * Fdot33 + ScaleNLKV_12cF * F31 + ScaleNLKV_22cF * F32 + ScaleNLKV_23cF * F33;
        ArrayNIP Partial_SPK2_NLKV_6_de_combined31 = SPK2_Scale * SPK2_NLKV_6 * BA31 + ScaleNLKV_13cFdot * Fdot11 + ScaleNLKV_23cFdot * Fdot12 + ScaleNLKV_33cFdot * Fdot13 + ScaleNLKV_13cF * F11 + ScaleNLKV_23cF * F12 + ScaleNLKV_33cF * F13;
        ArrayNIP Partial_SPK2_NLKV_6_de_combined32 = SPK2_Scale * SPK2_NLKV_6 * BA32 + ScaleNLKV_13cFdot * Fdot21 + ScaleNLKV_23cFdot * Fdot22 + ScaleNLKV_33cFdot * Fdot23 + ScaleNLKV_13cF * F21 + ScaleNLKV_23cF * F22 + ScaleNLKV_33cF * F23;
        ArrayNIP Partial_SPK2_NLKV_6_de_combined33 = SPK2_Scale * SPK2_NLKV_6 * BA33 + ScaleNLKV_13cFdot * Fdot31 + ScaleNLKV_23cFdot * Fdot32 + ScaleNLKV_33cFdot * Fdot33 + ScaleNLKV_13cF * F31 + ScaleNLKV_23cF * F32 + ScaleNLKV_33cF * F33;


        ArrayNIP s4 = s3 * C12;

        ArrayNIP S1A = s4 * BA11 + s5 * F12 + Partial_SPK2_NLKV_6_de_combined11;
        ArrayNIP S2A = s4 * BA21 + s5 * F11 + Partial_SPK2_NLKV_6_de_combined21;
        ArrayNIP S3A = s4 * BA31 + Partial_SPK2_NLKV_6_de_combined31;

        ArrayNIP S1B = s4 * BA12 + s5 * F22 + Partial_SPK2_NLKV_6_de_combined12;
        ArrayNIP S2B = s4 * BA22 + s5 * F21 + Partial_SPK2_NLKV_6_de_combined22;
        ArrayNIP S3B = s4 * BA32 + Partial_SPK2_NLKV_6_de_combined32;

        ArrayNIP S1C = s4 * BA13 + s5 * F32 + Partial_SPK2_NLKV_6_de_combined13;
        ArrayNIP S2C = s4 * BA23 + s5 * F31 + Partial_SPK2_NLKV_6_de_combined23;
        ArrayNIP S3C = s4 * BA33 + Partial_SPK2_NLKV_6_de_combined33;

        for (auto i = 0; i < NSF; i++) {
            Left.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * F12 + m_SD.block<1, NIP>(i, 1 * NIP).array() * F11;
            Left.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * F22 + m_SD.block<1, NIP>(i, 1 * NIP).array() * F21;
            Left.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * F32 + m_SD.block<1, NIP>(i, 1 * NIP).array() * F31;
            Right.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1A +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2A +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3A;
            Right.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1B +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2B +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3B;
            Right.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1C +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2C +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3C;
        }
        Jac.noalias() += Left * Right.transpose();
    }


    // Calculate the Contribution from the ddetF/de terms
    {
        ArrayNIP s4 = -Kfactor * k * m_kGQ + (-5.0 / 6.0*s1*I1 + 7.0 / 6.0*s3*I2) / detF;
        ArrayNIP s6 = s1 - s3 * I1;

        ArrayNIP S1A = s4 * BA11 + (s6 + s3 * C11)*F11 + s3 * (C12*F12 + C12 * F13);
        ArrayNIP S2A = s4 * BA21 + (s6 + s3 * C22)*F12 + s3 * (C12*F11 + C23 * F13);
        ArrayNIP S3A = s4 * BA31 + (s6 + s3 * C33)*F13 + s3 * (C13*F11 + C23 * F12);

        ArrayNIP S1B = s4 * BA12 + (s6 + s3 * C11)*F21 + s3 * (C12*F22 + C13 * F23);
        ArrayNIP S2B = s4 * BA22 + (s6 + s3 * C22)*F22 + s3 * (C12*F21 + C23 * F23);
        ArrayNIP S3B = s4 * BA32 + (s6 + s3 * C33)*F23 + s3 * (C13*F21 + C23 * F22);

        ArrayNIP S1C = s4 * BA13 + (s6 + s3 * C11)*F31 + s3 * (C12*F32 + C13 * F33);
        ArrayNIP S2C = s4 * BA23 + (s6 + s3 * C22)*F32 + s3 * (C12*F31 + C23 * F33);
        ArrayNIP S3C = s4 * BA33 + (s6 + s3 * C33)*F33 + s3 * (C13*F31 + C23 * F32);

        for (auto i = 0; i < NSF; i++) {
            Left.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * BA11 +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * BA21 +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * BA31;
            Left.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * BA12 +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * BA22 +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * BA32;
            Left.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * BA13 +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * BA23 +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * BA33;

            Right.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1A +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2A +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3A;
            Right.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1B +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2B +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3B;
            Right.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1C +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2C +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3C;
        }
        Jac.noalias() += Left * Right.transpose();
    }

    //Calculate the contribution from the Mass Matrix and Expand component
    ArrayNIP s7 = (-Kfactor * 2.0 * c10)*m_kGQ*detF_m2_3 - s5 * I1;
    ArrayNIP s8 = -Kfactor * m_kGQ * (k * (detF - 1.0) - 2.0 / 3.0 * c10 * I1*detF_m2_3 / detF - 4.0 / 3.0 * c01 * I2*detF_m2_3*detF_m2_3 / detF);

    ArrayNIP sC11 = s7 + s5 * C11 + SPK2_NLKV_1;
    ArrayNIP sC22 = s7 + s5 * C22 + SPK2_NLKV_2;
    ArrayNIP sC33 = s7 + s5 * C33 + SPK2_NLKV_3;
    ArrayNIP sC12 = s5 * C12 + SPK2_NLKV_6;
    ArrayNIP sC13 = s5 * C13 + SPK2_NLKV_5;
    ArrayNIP sC23 = s5 * C23 + SPK2_NLKV_4;

    ArrayNIP sF11 = s8 * F11;
    ArrayNIP sF12 = s8 * F12;
    ArrayNIP sF13 = s8 * F13;
    ArrayNIP sF21 = s8 * F21;
    ArrayNIP sF22 = s8 * F22;
    ArrayNIP sF23 = s8 * F23;
    ArrayNIP sF31 = s8 * F31;
    ArrayNIP sF32 = s8 * F32;
    ArrayNIP sF33 = s8 * F33;

    unsigned int idx = 0;
    for (unsigned int i = 0; i < (NSF - 1); i++) {
        ChVectorN<double, 3 * NIP> scaled_SD_row_i;
        scaled_SD_row_i.segment(0 * NIP, NIP).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * sC11 +
            m_SD.block<1, NIP>(i, 1 * NIP).array() * sC12 +
            m_SD.block<1, NIP>(i, 2 * NIP).array() * sC13;
        scaled_SD_row_i.segment(1 * NIP, NIP).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * sC12 +
            m_SD.block<1, NIP>(i, 1 * NIP).array() * sC22 +
            m_SD.block<1, NIP>(i, 2 * NIP).array() * sC23;
        scaled_SD_row_i.segment(2 * NIP, NIP).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * sC13 +
            m_SD.block<1, NIP>(i, 1 * NIP).array() * sC23 +
            m_SD.block<1, NIP>(i, 2 * NIP).array() * sC33;

        ChVectorN<double, 3 * NIP> scaled_B2_row_i;
        scaled_B2_row_i.segment(0 * NIP, NIP).array() = m_SD.block<1, NIP>(i, 1 * NIP).array()*sF33 - m_SD.block<1, NIP>(i, 2 * NIP).array()*sF32;
        scaled_B2_row_i.segment(1 * NIP, NIP).array() = m_SD.block<1, NIP>(i, 2 * NIP).array()*sF31 - m_SD.block<1, NIP>(i, 0 * NIP).array()*sF33;
        scaled_B2_row_i.segment(2 * NIP, NIP).array() = m_SD.block<1, NIP>(i, 0 * NIP).array()*sF32 - m_SD.block<1, NIP>(i, 1 * NIP).array()*sF31;

        ChVectorN<double, 3 * NIP> scaled_B3_row_i;
        scaled_B3_row_i.segment(0 * NIP, NIP).array() = m_SD.block<1, NIP>(i, 2 * NIP).array()*sF22 - m_SD.block<1, NIP>(i, 1 * NIP).array()*sF23;
        scaled_B3_row_i.segment(1 * NIP, NIP).array() = m_SD.block<1, NIP>(i, 0 * NIP).array()*sF23 - m_SD.block<1, NIP>(i, 2 * NIP).array()*sF21;
        scaled_B3_row_i.segment(2 * NIP, NIP).array() = m_SD.block<1, NIP>(i, 1 * NIP).array()*sF21 - m_SD.block<1, NIP>(i, 0 * NIP).array()*sF22;

        ChVectorN<double, 3 * NIP> scaled_B4_row_i;
        scaled_B4_row_i.segment(0 * NIP, NIP).array() = m_SD.block<1, NIP>(i, 1 * NIP).array()*sF13 - m_SD.block<1, NIP>(i, 2 * NIP).array()*sF12;
        scaled_B4_row_i.segment(1 * NIP, NIP).array() = m_SD.block<1, NIP>(i, 2 * NIP).array()*sF11 - m_SD.block<1, NIP>(i, 0 * NIP).array()*sF13;
        scaled_B4_row_i.segment(2 * NIP, NIP).array() = m_SD.block<1, NIP>(i, 0 * NIP).array()*sF12 - m_SD.block<1, NIP>(i, 1 * NIP).array()*sF11;

        double d_diag = Mfactor * m_MassMatrix(idx) + (scaled_SD_row_i.dot(m_SD.row(i)));

        Jac(3 * i, 3 * i) += d_diag;
        Jac(3 * i + 1, 3 * i + 1) += d_diag;
        Jac(3 * i + 2, 3 * i + 2) += d_diag;
        idx++;

        for (unsigned int j = (i + 1); j < NSF; j++) {
            double d = Mfactor * m_MassMatrix(idx) + (scaled_SD_row_i.dot(m_SD.row(j)));
            double B2 = scaled_B2_row_i.dot(m_SD.row(j));
            double B3 = scaled_B3_row_i.dot(m_SD.row(j));
            double B4 = scaled_B4_row_i.dot(m_SD.row(j));

            Jac(3 * i + 0, 3 * j + 0) += d;
            Jac(3 * i + 0, 3 * j + 1) -= B2;
            Jac(3 * i + 0, 3 * j + 2) -= B3;

            Jac(3 * i + 1, 3 * j + 0) += B2;
            Jac(3 * i + 1, 3 * j + 1) += d;
            Jac(3 * i + 1, 3 * j + 2) -= B4;

            Jac(3 * i + 2, 3 * j + 0) += B3;
            Jac(3 * i + 2, 3 * j + 1) += B4;
            Jac(3 * i + 2, 3 * j + 2) += d;

            Jac(3 * j + 0, 3 * i + 0) += d;
            Jac(3 * j + 0, 3 * i + 1) += B2;
            Jac(3 * j + 0, 3 * i + 2) += B3;

            Jac(3 * j + 1, 3 * i + 0) -= B2;
            Jac(3 * j + 1, 3 * i + 1) += d;
            Jac(3 * j + 1, 3 * i + 2) += B4;

            Jac(3 * j + 2, 3 * i + 0) -= B3;
            Jac(3 * j + 2, 3 * i + 1) -= B4;
            Jac(3 * j + 2, 3 * i + 2) += d;

            idx++;
        }
    }

    //No need to calculate B2, B3, or B4 for the last point since they equal 0
    ChVectorN<double, 3 * NIP> scaled_SD_row_i;
    scaled_SD_row_i.segment(0 * NIP, NIP).array() = m_SD.block<1, NIP>(NSF - 1, 0 * NIP).array() * sC11 +
        m_SD.block<1, NIP>(NSF - 1, 1 * NIP).array() * sC12 +
        m_SD.block<1, NIP>(NSF - 1, 2 * NIP).array() * sC13;
    scaled_SD_row_i.segment(1 * NIP, NIP).array() = m_SD.block<1, NIP>(NSF - 1, 0 * NIP).array() * sC12 +
        m_SD.block<1, NIP>(NSF - 1, 1 * NIP).array() * sC22 +
        m_SD.block<1, NIP>(NSF - 1, 2 * NIP).array() * sC23;
    scaled_SD_row_i.segment(2 * NIP, NIP).array() = m_SD.block<1, NIP>(NSF - 1, 0 * NIP).array() * sC13 +
        m_SD.block<1, NIP>(NSF - 1, 1 * NIP).array() * sC23 +
        m_SD.block<1, NIP>(NSF - 1, 2 * NIP).array() * sC33;

    double d_diag = Mfactor * m_MassMatrix(idx) + (scaled_SD_row_i.dot(m_SD.row(NSF - 1)));

    Jac(3 * (NSF - 1) + 0, 3 * (NSF - 1) + 0) += d_diag;
    Jac(3 * (NSF - 1) + 1, 3 * (NSF - 1) + 1) += d_diag;
    Jac(3 * (NSF - 1) + 2, 3 * (NSF - 1) + 2) += d_diag;

    H.noalias() = Jac;
}

void ChElementHexaANCF_3843_MR_Damp::ComputeInternalJacobianNoDamping(ChMatrixRef H, double Kfactor, double Mfactor) {
    assert((H.rows() == 3 * NSF) && (H.cols() == 3 * NSF));

    // Element coordinates in matrix form
    Matrix3xN ebar;
    CalcCoordMatrix(ebar);

    // =============================================================================
    // Calculate the deformation gradient for all Gauss quadrature points in a single matrix multiplication.  Note
    // that since the shape function derivative matrix is ordered by columns, the resulting deformation gradient
    // will be ordered by block matrix (row vectors) components
    //      [F11     F12     F13    ]
    // FC = [F21     F22     F23    ]
    //      [F31     F32     F33    ]
    // =============================================================================

    ChMatrixNM<double, 3, 3 * NIP> FC = ebar * m_SD;

    Eigen::Map<ArrayNIP> F11(FC.block<1, NIP>(0, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F12(FC.block<1, NIP>(0, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F13(FC.block<1, NIP>(0, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> F21(FC.block<1, NIP>(1, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F22(FC.block<1, NIP>(1, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F23(FC.block<1, NIP>(1, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> F31(FC.block<1, NIP>(2, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F32(FC.block<1, NIP>(2, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F33(FC.block<1, NIP>(2, 2 * NIP).data(), 1, NIP);

    //Calculate the Right Cauchy-Green deformation tensor (symmetric):
    ArrayNIP C11 = F11 * F11 + F21 * F21 + F31 * F31;
    ArrayNIP C22 = F12 * F12 + F22 * F22 + F32 * F32;
    ArrayNIP C33 = F13 * F13 + F23 * F23 + F33 * F33;
    ArrayNIP C12 = F11 * F12 + F21 * F22 + F31 * F32;
    ArrayNIP C13 = F11 * F13 + F21 * F23 + F31 * F33;
    ArrayNIP C23 = F12 * F13 + F22 * F23 + F32 * F33;

    //Calculate the 1st invariant of the Right Cauchy-Green deformation tensor: trace(C)
    ArrayNIP I1 = C11 + C22 + C33;

    //Calculate the 2nd invariant of the Right Cauchy-Green deformation tensor: 1/2*(trace(C)^2-trace(C^2))
    ArrayNIP I2 = 0.5*(I1*I1 - C11 * C11 - C22 * C22 - C33 * C33) - C12 * C12 - C13 * C13 - C23 * C23;

    //Calculate the determinate of F (commonly denoted as J)
    ArrayNIP detF = F11 * F22*F33 + F12 * F23*F31 + F13 * F21*F32 - F11 * F23*F32 - F12 * F21*F33 - F13 * F22*F31;

    ArrayNIP detF_m2_3 = detF.pow(-2.0 / 3.0);

    double c10 = GetMaterial()->Get_c10();
    double c01 = GetMaterial()->Get_c01();
    double k = GetMaterial()->Get_k();

    ArrayNIP BA11 = F22*F33 - F23*F32;
    ArrayNIP BA21 = F23*F31 - F21*F33;
    ArrayNIP BA31 = F21*F32 - F22*F31;
    ArrayNIP BA12 = F13*F32 - F12*F33;
    ArrayNIP BA22 = F11*F33 - F13*F31;
    ArrayNIP BA32 = F12*F31 - F11*F32;
    ArrayNIP BA13 = F12*F23 - F13*F22;
    ArrayNIP BA23 = F13*F21 - F11*F23;
    ArrayNIP BA33 = F11*F22 - F12*F21;

    ArrayNIP s1 = (-Kfactor * -4.0 / 3.0 * c10) * detF_m2_3 / detF* m_kGQ;
    ArrayNIP s2 = (-Kfactor * 4.0 * c01) * detF_m2_3*detF_m2_3 * m_kGQ;
    ArrayNIP s3 = (2.0 / 3.0) * s2 / detF;

    ChMatrixNM<double, 3 * NSF, NIP> Left;
    ChMatrixNM<double, 3 * NSF, NIP> Right;
    Matrix3Nx3N Jac;

    //Calculate the Contribution from the dC11 / de terms
    {
        ArrayNIP s4 = s1 + s3 * (C11 - I1);

        ArrayNIP S1A = s4*BA11;
        ArrayNIP S2A = s4*BA21 + s2*F12;
        ArrayNIP S3A = s4*BA31 + s2*F13;

        ArrayNIP S1B = s4*BA12;
        ArrayNIP S2B = s4*BA22 + s2*F22;
        ArrayNIP S3B = s4*BA32 + s2*F23;

        ArrayNIP S1C = s4*BA13;
        ArrayNIP S2C = s4*BA23 + s2*F32;
        ArrayNIP S3C = s4*BA33 + s2*F33;

        for (auto i = 0; i < NSF; i++) {
            Left.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * F11;
            Left.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * F21;
            Left.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * F31;

            Right.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1A +
                                                        m_SD.block<1, NIP>(i, 1 * NIP).array() * S2A +
                                                        m_SD.block<1, NIP>(i, 2 * NIP).array() * S3A;
            Right.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1B +
                                                        m_SD.block<1, NIP>(i, 1 * NIP).array() * S2B +
                                                        m_SD.block<1, NIP>(i, 2 * NIP).array() * S3B;
            Right.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1C +
                                                        m_SD.block<1, NIP>(i, 1 * NIP).array() * S2C +
                                                        m_SD.block<1, NIP>(i, 2 * NIP).array() * S3C;
        }
        Jac.noalias() = Left * Right.transpose();
    }
    //Calculate the Contribution from the dC22 / de terms
    {
        ArrayNIP s4 = s1 + s3 * (C22 - I1);

        ArrayNIP S1A = s4*BA11 + s2*F11;
        ArrayNIP S2A = s4*BA21;
        ArrayNIP S3A = s4*BA31 + s2*F13;

        ArrayNIP S1B = s4*BA12 + s2*F21;
        ArrayNIP S2B = s4*BA22;
        ArrayNIP S3B = s4*BA32 + s2*F23;

        ArrayNIP S1C = s4*BA13 + s2*F31;
        ArrayNIP S2C = s4*BA23;
        ArrayNIP S3C = s4*BA33 + s2*F33;

        for (auto i = 0; i < NSF; i++) {
            Left.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 1 * NIP).array() * F12;
            Left.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 1 * NIP).array() * F22;
            Left.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 1 * NIP).array() * F32;
            Right.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1A +
                                                        m_SD.block<1, NIP>(i, 1 * NIP).array() * S2A +
                                                        m_SD.block<1, NIP>(i, 2 * NIP).array() * S3A;
            Right.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1B +
                                                        m_SD.block<1, NIP>(i, 1 * NIP).array() * S2B +
                                                        m_SD.block<1, NIP>(i, 2 * NIP).array() * S3B;
            Right.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1C +
                                                        m_SD.block<1, NIP>(i, 1 * NIP).array() * S2C +
                                                        m_SD.block<1, NIP>(i, 2 * NIP).array() * S3C;
        }
        Jac.noalias() += Left * Right.transpose();
    }
    //Calculate the Contribution from the dC33 / de terms
    {
        ArrayNIP s4 = s1 + s3 * (C33 - I1);

        ArrayNIP S1A = s4*BA11 + s2*F11;
        ArrayNIP S2A = s4*BA21 + s2*F12;
        ArrayNIP S3A = s4*BA31;

        ArrayNIP S1B = s4*BA12 + s2*F21;
        ArrayNIP S2B = s4*BA22 + s2*F22;
        ArrayNIP S3B = s4*BA32;

        ArrayNIP S1C = s4*BA13 + s2*F31;
        ArrayNIP S2C = s4*BA23 + s2*F32;
        ArrayNIP S3C = s4*BA33;

        for (auto i = 0; i < NSF; i++) {
            Left.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 2 * NIP).array() * F13;
            Left.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 2 * NIP).array() * F23;
            Left.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 2 * NIP).array() * F33;
            Right.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1A +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2A +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3A;
            Right.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1B +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2B +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3B;
            Right.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1C +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2C +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3C;
        }
        Jac.noalias() += Left * Right.transpose();
    }

    ArrayNIP s5 = -0.5 * s2;
    // Calculate the Contribution from the dC23/de terms
    {
        ArrayNIP s4 = s3 * C23;

        ArrayNIP S1A = s4 * BA11;
        ArrayNIP S2A = s4 * BA21 + s5 * F13;
        ArrayNIP S3A = s4 * BA31 + s5 * F12;

        ArrayNIP S1B = s4 * BA12;
        ArrayNIP S2B = s4 * BA22 + s5 * F23;
        ArrayNIP S3B = s4 * BA32 + s5 * F22;

        ArrayNIP S1C = s4 * BA13;
        ArrayNIP S2C = s4 * BA23 + s5 * F33;
        ArrayNIP S3C = s4 * BA33 + s5 * F32;

        for (auto i = 0; i < NSF; i++) {
            Left.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 1 * NIP).array() * F13 + m_SD.block<1, NIP>(i, 2 * NIP).array() * F12;
            Left.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 1 * NIP).array() * F23 + m_SD.block<1, NIP>(i, 2 * NIP).array() * F22;
            Left.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 1 * NIP).array() * F33 + m_SD.block<1, NIP>(i, 2 * NIP).array() * F32;
            Right.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1A +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2A +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3A;
            Right.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1B +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2B +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3B;
            Right.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1C +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2C +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3C;
        }
        Jac.noalias() += Left * Right.transpose();
    }
    // Calculate the Contribution from the dC13/de terms
    {
        ArrayNIP s4 = s3 * C13;

        ArrayNIP S1A = s4*BA11 + s5*F13;
        ArrayNIP S2A = s4*BA21;
        ArrayNIP S3A = s4*BA31 + s5*F11;

        ArrayNIP S1B = s4*BA12 + s5*F23;
        ArrayNIP S2B = s4*BA22;
        ArrayNIP S3B = s4*BA32 + s5*F21;

        ArrayNIP S1C = s4*BA13 + s5*F33;
        ArrayNIP S2C = s4*BA23;
        ArrayNIP S3C = s4*BA33 + s5*F31;

        for (auto i = 0; i < NSF; i++) {
            Left.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * F13 + m_SD.block<1, NIP>(i, 2 * NIP).array() * F11;
            Left.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * F23 + m_SD.block<1, NIP>(i, 2 * NIP).array() * F21;
            Left.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * F33 + m_SD.block<1, NIP>(i, 2 * NIP).array() * F31;
            Right.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1A +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2A +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3A;
            Right.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1B +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2B +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3B;
            Right.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1C +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2C +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3C;
        }
        Jac.noalias() += Left * Right.transpose();
    }
    // Calculate the Contribution from the dC12/de terms
    {
        ArrayNIP s4 = s3 * C12;

        ArrayNIP S1A = s4 * BA11 + s5 * F12;
        ArrayNIP S2A = s4 * BA21 + s5 * F11;
        ArrayNIP S3A = s4 * BA31;

        ArrayNIP S1B = s4 * BA12 + s5 * F22;
        ArrayNIP S2B = s4 * BA22 + s5 * F21;
        ArrayNIP S3B = s4 * BA32;

        ArrayNIP S1C = s4 * BA13 + s5 * F32;
        ArrayNIP S2C = s4 * BA23 + s5 * F31;
        ArrayNIP S3C = s4 * BA33;

        for (auto i = 0; i < NSF; i++) {
            Left.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * F12 + m_SD.block<1, NIP>(i, 1 * NIP).array() * F11;
            Left.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * F22 + m_SD.block<1, NIP>(i, 1 * NIP).array() * F21;
            Left.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * F32 + m_SD.block<1, NIP>(i, 1 * NIP).array() * F31;
            Right.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1A +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2A +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3A;
            Right.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1B +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2B +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3B;
            Right.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1C +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2C +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3C;
        }
        Jac.noalias() += Left * Right.transpose();
    }
    // Calculate the Contribution from the ddetF/de terms
    {
        ArrayNIP s4 = -Kfactor * k * m_kGQ + (-5.0 / 6.0*s1*I1 + 7.0 / 6.0*s3*I2) / detF;
        ArrayNIP s6 = s1 - s3*I1;

        ArrayNIP S1A = s4*BA11 + (s6 + s3*C11)*F11 + s3 * (C12*F12 + C12*F13);
        ArrayNIP S2A = s4*BA21 + (s6 + s3*C22)*F12 + s3 * (C12*F11 + C23*F13);
        ArrayNIP S3A = s4*BA31 + (s6 + s3*C33)*F13 + s3 * (C13*F11 + C23*F12);

        ArrayNIP S1B = s4*BA12 + (s6 + s3*C11)*F21 + s3 * (C12*F22 + C13*F23);
        ArrayNIP S2B = s4*BA22 + (s6 + s3*C22)*F22 + s3 * (C12*F21 + C23*F23);
        ArrayNIP S3B = s4*BA32 + (s6 + s3*C33)*F23 + s3 * (C13*F21 + C23*F22);

        ArrayNIP S1C = s4*BA13 + (s6 + s3*C11)*F31 + s3 * (C12*F32 + C13*F33);
        ArrayNIP S2C = s4*BA23 + (s6 + s3*C22)*F32 + s3 * (C12*F31 + C23*F33);
        ArrayNIP S3C = s4*BA33 + (s6 + s3*C33)*F33 + s3 * (C13*F31 + C23*F32);

        for (auto i = 0; i < NSF; i++) {
            Left.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * BA11 +
                                                       m_SD.block<1, NIP>(i, 1 * NIP).array() * BA21 +
                                                       m_SD.block<1, NIP>(i, 2 * NIP).array() * BA31;
            Left.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * BA12 +
                                                       m_SD.block<1, NIP>(i, 1 * NIP).array() * BA22 +
                                                       m_SD.block<1, NIP>(i, 2 * NIP).array() * BA32;
            Left.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * BA13 +
                                                       m_SD.block<1, NIP>(i, 1 * NIP).array() * BA23 +
                                                       m_SD.block<1, NIP>(i, 2 * NIP).array() * BA33;

            Right.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1A +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2A +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3A;
            Right.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1B +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2B +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3B;
            Right.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1C +
                m_SD.block<1, NIP>(i, 1 * NIP).array() * S2C +
                m_SD.block<1, NIP>(i, 2 * NIP).array() * S3C;
        }
        Jac.noalias() += Left * Right.transpose();
    }

    //Calculate the contribution from the Mass Matrix and Expand component
    ArrayNIP s7 = (-Kfactor * 2.0 * c10)*m_kGQ*detF_m2_3 - s5*I1;
    ArrayNIP s8 = -Kfactor * m_kGQ * (k * (detF - 1.0) - 2.0 / 3.0 * c10 * I1*detF_m2_3 / detF - 4.0 / 3.0 * c01 * I2*detF_m2_3*detF_m2_3 / detF);

    ArrayNIP sC11 = s7 + s5 * C11;
    ArrayNIP sC22 = s7 + s5 * C22;
    ArrayNIP sC33 = s7 + s5 * C33;
    ArrayNIP sC12 = s5*C12;
    ArrayNIP sC13 = s5*C13;
    ArrayNIP sC23 = s5*C23;

    ArrayNIP sF11 = s8*F11;
    ArrayNIP sF12 = s8*F12;
    ArrayNIP sF13 = s8*F13;
    ArrayNIP sF21 = s8*F21;
    ArrayNIP sF22 = s8*F22;
    ArrayNIP sF23 = s8*F23;
    ArrayNIP sF31 = s8*F31;
    ArrayNIP sF32 = s8*F32;
    ArrayNIP sF33 = s8*F33;

    unsigned int idx = 0;
    for (unsigned int i = 0; i < (NSF-1); i++) {
        ChVectorN<double, 3 * NIP> scaled_SD_row_i;
        scaled_SD_row_i.segment(0 * NIP, NIP).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * sC11 +
                                                        m_SD.block<1, NIP>(i, 1 * NIP).array() * sC12 +
                                                        m_SD.block<1, NIP>(i, 2 * NIP).array() * sC13;
        scaled_SD_row_i.segment(1 * NIP, NIP).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * sC12 +
                                                        m_SD.block<1, NIP>(i, 1 * NIP).array() * sC22 +
                                                        m_SD.block<1, NIP>(i, 2 * NIP).array() * sC23;
        scaled_SD_row_i.segment(2 * NIP, NIP).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * sC13 +
                                                        m_SD.block<1, NIP>(i, 1 * NIP).array() * sC23 +
                                                        m_SD.block<1, NIP>(i, 2 * NIP).array() * sC33;

        ChVectorN<double, 3 * NIP> scaled_B2_row_i;
        scaled_B2_row_i.segment(0 * NIP, NIP).array() = m_SD.block<1, NIP>(i, 1 * NIP).array()*sF33 - m_SD.block<1, NIP>(i, 2 * NIP).array()*sF32;
        scaled_B2_row_i.segment(1 * NIP, NIP).array() = m_SD.block<1, NIP>(i, 2 * NIP).array()*sF31 - m_SD.block<1, NIP>(i, 0 * NIP).array()*sF33;
        scaled_B2_row_i.segment(2 * NIP, NIP).array() = m_SD.block<1, NIP>(i, 0 * NIP).array()*sF32 - m_SD.block<1, NIP>(i, 1 * NIP).array()*sF31;

        ChVectorN<double, 3 * NIP> scaled_B3_row_i;
        scaled_B3_row_i.segment(0 * NIP, NIP).array() = m_SD.block<1, NIP>(i, 2 * NIP).array()*sF22 - m_SD.block<1, NIP>(i, 1 * NIP).array()*sF23;
        scaled_B3_row_i.segment(1 * NIP, NIP).array() = m_SD.block<1, NIP>(i, 0 * NIP).array()*sF23 - m_SD.block<1, NIP>(i, 2 * NIP).array()*sF21;
        scaled_B3_row_i.segment(2 * NIP, NIP).array() = m_SD.block<1, NIP>(i, 1 * NIP).array()*sF21 - m_SD.block<1, NIP>(i, 0 * NIP).array()*sF22;

        ChVectorN<double, 3 * NIP> scaled_B4_row_i;
        scaled_B4_row_i.segment(0 * NIP, NIP).array() = m_SD.block<1, NIP>(i, 1 * NIP).array()*sF13 - m_SD.block<1, NIP>(i, 2 * NIP).array()*sF12;
        scaled_B4_row_i.segment(1 * NIP, NIP).array() = m_SD.block<1, NIP>(i, 2 * NIP).array()*sF11 - m_SD.block<1, NIP>(i, 0 * NIP).array()*sF13;
        scaled_B4_row_i.segment(2 * NIP, NIP).array() = m_SD.block<1, NIP>(i, 0 * NIP).array()*sF12 - m_SD.block<1, NIP>(i, 1 * NIP).array()*sF11;

        double d_diag = Mfactor * m_MassMatrix(idx) + (scaled_SD_row_i.dot(m_SD.row(i)));

        Jac(3 * i, 3 * i) += d_diag;
        Jac(3 * i + 1, 3 * i + 1) += d_diag;
        Jac(3 * i + 2, 3 * i + 2) += d_diag;
        idx++;

        for (unsigned int j = (i+1); j < NSF; j++) {
            double d = Mfactor * m_MassMatrix(idx) + (scaled_SD_row_i.dot(m_SD.row(j)));
            double B2 = scaled_B2_row_i.dot(m_SD.row(j));
            double B3 = scaled_B3_row_i.dot(m_SD.row(j));
            double B4 = scaled_B4_row_i.dot(m_SD.row(j));

            Jac(3 * i + 0, 3 * j + 0) += d;
            Jac(3 * i + 0, 3 * j + 1) -= B2;
            Jac(3 * i + 0, 3 * j + 2) -= B3;

            Jac(3 * i + 1, 3 * j + 0) += B2;
            Jac(3 * i + 1, 3 * j + 1) += d;
            Jac(3 * i + 1, 3 * j + 2) -= B4;

            Jac(3 * i + 2, 3 * j + 0) += B3;
            Jac(3 * i + 2, 3 * j + 1) += B4;
            Jac(3 * i + 2, 3 * j + 2) += d;

            Jac(3 * j + 0, 3 * i + 0) += d;
            Jac(3 * j + 0, 3 * i + 1) += B2;
            Jac(3 * j + 0, 3 * i + 2) += B3;

            Jac(3 * j + 1, 3 * i + 0) -= B2;
            Jac(3 * j + 1, 3 * i + 1) += d;
            Jac(3 * j + 1, 3 * i + 2) += B4;

            Jac(3 * j + 2, 3 * i + 0) -= B3;
            Jac(3 * j + 2, 3 * i + 1) -= B4;
            Jac(3 * j + 2, 3 * i + 2) += d;

            idx++;
        }
    }

    //No need to calculate B2, B3, or B4 for the last point since they equal 0
    ChVectorN<double, 3 * NIP> scaled_SD_row_i;
    scaled_SD_row_i.segment(0 * NIP, NIP).array() = m_SD.block<1, NIP>(NSF-1, 0 * NIP).array() * sC11 +
        m_SD.block<1, NIP>(NSF-1, 1 * NIP).array() * sC12 +
        m_SD.block<1, NIP>(NSF-1, 2 * NIP).array() * sC13;
    scaled_SD_row_i.segment(1 * NIP, NIP).array() = m_SD.block<1, NIP>(NSF - 1, 0 * NIP).array() * sC12 +
        m_SD.block<1, NIP>(NSF-1, 1 * NIP).array() * sC22 +
        m_SD.block<1, NIP>(NSF-1, 2 * NIP).array() * sC23;
    scaled_SD_row_i.segment(2 * NIP, NIP).array() = m_SD.block<1, NIP>(NSF - 1, 0 * NIP).array() * sC13 +
        m_SD.block<1, NIP>(NSF-1, 1 * NIP).array() * sC23 +
        m_SD.block<1, NIP>(NSF-1, 2 * NIP).array() * sC33;

    double d_diag = Mfactor * m_MassMatrix(idx) + (scaled_SD_row_i.dot(m_SD.row(NSF - 1)));

    Jac(3 * (NSF - 1) + 0, 3 * (NSF - 1) + 0) += d_diag;
    Jac(3 * (NSF - 1) + 1, 3 * (NSF - 1) + 1) += d_diag;
    Jac(3 * (NSF - 1) + 2, 3 * (NSF - 1) + 2) += d_diag;

    H.noalias() = Jac;
}

// -----------------------------------------------------------------------------
// Shape functions
// -----------------------------------------------------------------------------

// Nx1 Vector Form of the Normalized Shape Functions
// [s1; s2; s3; ...]

void ChElementHexaANCF_3843_MR_Damp::Calc_Sxi_compact(VectorN& Sxi_compact, double xi, double eta, double zeta) {
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

void ChElementHexaANCF_3843_MR_Damp::Calc_Sxi_xi_compact(VectorN& Sxi_xi_compact, double xi, double eta, double zeta) {
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

void ChElementHexaANCF_3843_MR_Damp::Calc_Sxi_eta_compact(VectorN& Sxi_eta_compact, double xi, double eta, double zeta) {
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

void ChElementHexaANCF_3843_MR_Damp::Calc_Sxi_zeta_compact(VectorN& Sxi_zeta_compact, double xi, double eta, double zeta) {
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

void ChElementHexaANCF_3843_MR_Damp::Calc_Sxi_D(MatrixNx3c& Sxi_D, double xi, double eta, double zeta) {
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

void ChElementHexaANCF_3843_MR_Damp::CalcCoordVector(Vector3N& e) {
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

void ChElementHexaANCF_3843_MR_Damp::CalcCoordMatrix(Matrix3xN& ebar) {
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

void ChElementHexaANCF_3843_MR_Damp::CalcCoordDerivVector(Vector3N& edot) {
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

void ChElementHexaANCF_3843_MR_Damp::CalcCoordDerivMatrix(Matrix3xN& ebardot) {
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

void ChElementHexaANCF_3843_MR_Damp::CalcCombinedCoordMatrix(Matrix6xN& ebar_ebardot) {
    ebar_ebardot.block<3, 1>(0, 0) = m_nodes[0]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 0) = m_nodes[0]->GetPos_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 1) = m_nodes[0]->GetD().eigen();
    ebar_ebardot.block<3, 1>(3, 1) = m_nodes[0]->GetD_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 2) = m_nodes[0]->GetDD().eigen();
    ebar_ebardot.block<3, 1>(3, 2) = m_nodes[0]->GetDD_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 3) = m_nodes[0]->GetDDD().eigen();
    ebar_ebardot.block<3, 1>(3, 3) = m_nodes[0]->GetDDD_dt().eigen();

    ebar_ebardot.block<3, 1>(0, 4) = m_nodes[1]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 4) = m_nodes[1]->GetPos_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 5) = m_nodes[1]->GetD().eigen();
    ebar_ebardot.block<3, 1>(3, 5) = m_nodes[1]->GetD_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 6) = m_nodes[1]->GetDD().eigen();
    ebar_ebardot.block<3, 1>(3, 6) = m_nodes[1]->GetDD_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 7) = m_nodes[1]->GetDDD().eigen();
    ebar_ebardot.block<3, 1>(3, 7) = m_nodes[1]->GetDDD_dt().eigen();

    ebar_ebardot.block<3, 1>(0, 8) = m_nodes[2]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 8) = m_nodes[2]->GetPos_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 9) = m_nodes[2]->GetD().eigen();
    ebar_ebardot.block<3, 1>(3, 9) = m_nodes[2]->GetD_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 10) = m_nodes[2]->GetDD().eigen();
    ebar_ebardot.block<3, 1>(3, 10) = m_nodes[2]->GetDD_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 11) = m_nodes[2]->GetDDD().eigen();
    ebar_ebardot.block<3, 1>(3, 11) = m_nodes[2]->GetDDD_dt().eigen();

    ebar_ebardot.block<3, 1>(0, 12) = m_nodes[3]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 12) = m_nodes[3]->GetPos_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 13) = m_nodes[3]->GetD().eigen();
    ebar_ebardot.block<3, 1>(3, 13) = m_nodes[3]->GetD_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 14) = m_nodes[3]->GetDD().eigen();
    ebar_ebardot.block<3, 1>(3, 14) = m_nodes[3]->GetDD_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 15) = m_nodes[3]->GetDDD().eigen();
    ebar_ebardot.block<3, 1>(3, 15) = m_nodes[3]->GetDDD_dt().eigen();

    ebar_ebardot.block<3, 1>(0, 16) = m_nodes[4]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 16) = m_nodes[4]->GetPos_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 17) = m_nodes[4]->GetD().eigen();
    ebar_ebardot.block<3, 1>(3, 17) = m_nodes[4]->GetD_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 18) = m_nodes[4]->GetDD().eigen();
    ebar_ebardot.block<3, 1>(3, 18) = m_nodes[4]->GetDD_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 19) = m_nodes[4]->GetDDD().eigen();
    ebar_ebardot.block<3, 1>(3, 19) = m_nodes[4]->GetDDD_dt().eigen();

    ebar_ebardot.block<3, 1>(0, 20) = m_nodes[5]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 20) = m_nodes[5]->GetPos_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 21) = m_nodes[5]->GetD().eigen();
    ebar_ebardot.block<3, 1>(3, 21) = m_nodes[5]->GetD_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 22) = m_nodes[5]->GetDD().eigen();
    ebar_ebardot.block<3, 1>(3, 22) = m_nodes[5]->GetDD_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 23) = m_nodes[5]->GetDDD().eigen();
    ebar_ebardot.block<3, 1>(3, 23) = m_nodes[5]->GetDDD_dt().eigen();

    ebar_ebardot.block<3, 1>(0, 24) = m_nodes[6]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 24) = m_nodes[6]->GetPos_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 25) = m_nodes[6]->GetD().eigen();
    ebar_ebardot.block<3, 1>(3, 25) = m_nodes[6]->GetD_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 26) = m_nodes[6]->GetDD().eigen();
    ebar_ebardot.block<3, 1>(3, 26) = m_nodes[6]->GetDD_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 27) = m_nodes[6]->GetDDD().eigen();
    ebar_ebardot.block<3, 1>(3, 27) = m_nodes[6]->GetDDD_dt().eigen();

    ebar_ebardot.block<3, 1>(0, 28) = m_nodes[7]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 28) = m_nodes[7]->GetPos_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 29) = m_nodes[7]->GetD().eigen();
    ebar_ebardot.block<3, 1>(3, 29) = m_nodes[7]->GetD_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 30) = m_nodes[7]->GetDD().eigen();
    ebar_ebardot.block<3, 1>(3, 30) = m_nodes[7]->GetDD_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 31) = m_nodes[7]->GetDDD().eigen();
    ebar_ebardot.block<3, 1>(3, 31) = m_nodes[7]->GetDDD_dt().eigen();
}

// Calculate the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element

void ChElementHexaANCF_3843_MR_Damp::Calc_J_0xi(ChMatrix33<double>& J_0xi, double xi, double eta, double zeta) {
    MatrixNx3c Sxi_D;
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    J_0xi = m_ebar0 * Sxi_D;
}

// Calculate the determinant of the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element

double ChElementHexaANCF_3843_MR_Damp::Calc_det_J_0xi(double xi, double eta, double zeta) {
    ChMatrix33<double> J_0xi;
    Calc_J_0xi(J_0xi, xi, eta, zeta);

    return (J_0xi.determinant());
}

////////////////////////////////////////////////////////////////

//#ifndef CH_QUADRATURE_STATIC_TABLES
#define CH_QUADRATURE_STATIC_TABLES 10
ChQuadratureTables static_tables_3843_MR_Damp(1, CH_QUADRATURE_STATIC_TABLES);
//#endif // !CH_QUADRATURE_STATIC_TABLES

ChQuadratureTables* ChElementHexaANCF_3843_MR_Damp::GetStaticGQTables() {
    return &static_tables_3843_MR_Damp;
}

}  // namespace fea
}  // namespace chrono

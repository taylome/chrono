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
// Fully Parameterized ANCF shell element with 4 nodes (48DOF). A Description of this element can be found in: Aki M
// Mikkola and Ahmed A Shabana. A non-incremental finite element procedure for the analysis of large deformation of
// plates and shells in mechanical system applications. Multibody System Dynamics, 9(3) : 283�309, 2003.
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

#include "chrono/fea/ChElementShellANCF_3443_MR_DampNumJac.h"
#include "chrono/physics/ChSystem.h"

namespace chrono {
namespace fea {

// ------------------------------------------------------------------------------
// Constructor
// ------------------------------------------------------------------------------

ChElementShellANCF_3443_MR_DampNumJac::ChElementShellANCF_3443_MR_DampNumJac()
    : m_numLayers(0), m_lenX(0), m_lenY(0), m_thicknessZ(0), m_midsurfoffset(0) {
    m_nodes.resize(4);
}

// ------------------------------------------------------------------------------
// Set element nodes
// ------------------------------------------------------------------------------

void ChElementShellANCF_3443_MR_DampNumJac::SetNodes(std::shared_ptr<ChNodeFEAxyzDDD> nodeA,
                                            std::shared_ptr<ChNodeFEAxyzDDD> nodeB,
                                            std::shared_ptr<ChNodeFEAxyzDDD> nodeC,
                                            std::shared_ptr<ChNodeFEAxyzDDD> nodeD) {
    assert(nodeA);
    assert(nodeB);
    assert(nodeC);
    assert(nodeD);

    m_nodes[0] = nodeA;
    m_nodes[1] = nodeB;
    m_nodes[2] = nodeC;
    m_nodes[3] = nodeD;

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

    Kmatr.SetVariables(mvars);

    // Initial positions and slopes of the element nodes
    // These values define the reference configuration of the element
    CalcCoordMatrix(m_ebar0);
}

// -----------------------------------------------------------------------------
// Add a layer.
// -----------------------------------------------------------------------------

void ChElementShellANCF_3443_MR_DampNumJac::AddLayer(double thickness,
                                            std::shared_ptr<ChMaterialShellANCF_MR> material) {
    m_layers.push_back(Layer(thickness, material));
    m_layer_zoffsets.push_back(m_thicknessZ);
    m_numLayers += 1;
    m_thicknessZ += thickness;
}

// -----------------------------------------------------------------------------
// Element Settings
// -----------------------------------------------------------------------------

// Specify the element dimensions.

void ChElementShellANCF_3443_MR_DampNumJac::SetDimensions(double lenX, double lenY) {
    m_lenX = lenX;
    m_lenY = lenY;
}

// Offset the midsurface of the composite shell element.  A positive value shifts the element's midsurface upward
// along the elements zeta direction.  The offset should be provided in model units.

void ChElementShellANCF_3443_MR_DampNumJac::SetMidsurfaceOffset(const double offset) {
    m_midsurfoffset = offset;
}

// -----------------------------------------------------------------------------
// Evaluate Strains and Stresses
// -----------------------------------------------------------------------------
// These functions are designed for single function calls.  If these values are needed at the same points in the element
// through out the simulation, then the adjusted normalized shape function derivative matrix (Sxi_D) for each query
// point should be cached and saved to increase the execution speed
// -----------------------------------------------------------------------------

// Get the Green-Lagrange strain tensor at the normalized element coordinates (xi, eta, zeta) [-1...1]

ChMatrix33<> ChElementShellANCF_3443_MR_DampNumJac::GetGreenLagrangeStrain(const double xi,
                                                                  const double eta,
                                                                  const double zeta) {
    MatrixNx3c Sxi_D;  // Matrix of normalized shape function derivatives
    Calc_Sxi_D(Sxi_D, xi, eta, zeta, m_thicknessZ, m_midsurfoffset);

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

// Get the 2nd Piola-Kirchoff stress tensor at the normalized **layer** coordinates (xi, eta, layer_zeta) at the current
// state of the element for the specified layer number (0 indexed) since the stress can be discontinuous at the layer
// boundary.   "layer_zeta" spans -1 to 1 from the bottom surface to the top surface

ChMatrix33<> ChElementShellANCF_3443_MR_DampNumJac::GetPK2Stress(const double layer,
                                                        const double xi,
                                                        const double eta,
                                                        const double layer_zeta) {
    MatrixNx3c Sxi_D;  // Matrix of normalized shape function derivatives
    double layer_midsurface_offset =
        -m_thicknessZ / 2 + m_layer_zoffsets[layer] + m_layers[layer].Get_thickness() / 2 + m_midsurfoffset;
    Calc_Sxi_D(Sxi_D, xi, eta, layer_zeta, m_layers[layer].Get_thickness(), layer_midsurface_offset);

    ChMatrix33<double> J_0xi;  // Element Jacobian between the reference configuration and normalized configuration
    J_0xi.noalias() = m_ebar0 * Sxi_D;

    Sxi_D = Sxi_D * J_0xi.inverse();  // Adjust the shape function derivative matrix to account for the potentially
                                      // distorted reference configuration

    Matrix3xN e_bar;  // Element coordinates in matrix form
    CalcCoordMatrix(e_bar);

    // Calculate the Deformation Gradient at the current point
    ChMatrixNMc<double, 3, 3> F = e_bar * Sxi_D;

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
    double mu1 = 2 * m_layers[layer].GetMaterial()->Get_c10();
    double mu2 = 2 * m_layers[layer].GetMaterial()->Get_c01();

    ChMatrix33<> SPK2 =
        (std::pow(J, 1.0 / 3.0) * (J - 1) * m_layers[layer].GetMaterial()->Get_k() - J_m23 / 3.0 * (mu1 * I1bar + 2 * mu2 * I2bar)) *
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
    SPK2.noalias() += J * m_layers[layer].GetMaterial()->Get_mu()*CInv*Edot*CInv;

    return SPK2;
}

// Get the von Mises stress value at the normalized **layer** coordinates (xi, eta, layer_zeta) at the current state
// of the element for the specified layer number (0 indexed) since the stress can be discontinuous at the layer
// boundary.  "layer_zeta" spans -1 to 1 from the bottom surface to the top surface

double ChElementShellANCF_3443_MR_DampNumJac::GetVonMissesStress(const double layer,
                                                        const double xi,
                                                        const double eta,
                                                        const double layer_zeta) {
    MatrixNx3c Sxi_D;  // Matrix of normalized shape function derivatives
    double layer_midsurface_offset =
        -m_thicknessZ / 2 + m_layer_zoffsets[layer] + m_layers[layer].Get_thickness() / 2 + m_midsurfoffset;
    Calc_Sxi_D(Sxi_D, xi, eta, layer_zeta, m_layers[layer].Get_thickness(), layer_midsurface_offset);

    ChMatrix33<double> J_0xi;  // Element Jacobian between the reference configuration and normalized configuration
    J_0xi.noalias() = m_ebar0 * Sxi_D;

    Sxi_D = Sxi_D * J_0xi.inverse();  // Adjust the shape function derivative matrix to account for the potentially
                                      // distorted reference configuration

    Matrix3xN e_bar;  // Element coordinates in matrix form
    CalcCoordMatrix(e_bar);

    // Calculate the Deformation Gradient at the current point
    ChMatrixNMc<double, 3, 3> F = e_bar * Sxi_D;

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
    double mu1 = 2 * m_layers[layer].GetMaterial()->Get_c10();
    double mu2 = 2 * m_layers[layer].GetMaterial()->Get_c01();

    ChMatrix33<> SPK2 =
        (std::pow(J, 1.0 / 3.0) * (J - 1) * m_layers[layer].GetMaterial()->Get_k() - J_m23 / 3.0 * (mu1 * I1bar + 2 * mu2 * I2bar)) *
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
    SPK2.noalias() += J * m_layers[layer].GetMaterial()->Get_mu()*CInv*Edot*CInv;


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

void ChElementShellANCF_3443_MR_DampNumJac::SetupInitial(ChSystem* system) {
    // Store the initial nodal coordinates. These values define the reference configuration of the element.
    CalcCoordMatrix(m_ebar0);

    // Compute and store the constant mass matrix and the matrix used to multiply the acceleration due to gravity to get
    // the generalized gravitational force vector for the element
    ComputeMassMatrixAndGravityForce();

    // Compute Pre-computed matrices and vectors for the generalized internal force calcualtions
    PrecomputeInternalForceMatricesWeights();
}

// Fill the D vector with the current field values at the element nodes.

void ChElementShellANCF_3443_MR_DampNumJac::GetStateBlock(ChVectorDynamic<>& mD) {
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
}

// State update.

void ChElementShellANCF_3443_MR_DampNumJac::Update() {
    ChElementGeneric::Update();
}

// Return the mass matrix in full sparse form.

void ChElementShellANCF_3443_MR_DampNumJac::ComputeMmatrixGlobal(ChMatrixRef M) {
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

void ChElementShellANCF_3443_MR_DampNumJac::ComputeNodalMass() {
    m_nodes[0]->m_TotalMass += m_MassMatrix(0) + m_MassMatrix(4) + m_MassMatrix(8) + m_MassMatrix(12);
    m_nodes[1]->m_TotalMass += m_MassMatrix(4) + m_MassMatrix(58) + m_MassMatrix(62) + m_MassMatrix(66);
    m_nodes[2]->m_TotalMass += m_MassMatrix(8) + m_MassMatrix(62) + m_MassMatrix(100) + m_MassMatrix(104);
    m_nodes[3]->m_TotalMass += m_MassMatrix(12) + m_MassMatrix(66) + m_MassMatrix(104) + m_MassMatrix(126);
}

// Compute the generalized internal force vector for the current nodal coordinates and set the value in the Fi vector.

void ChElementShellANCF_3443_MR_DampNumJac::ComputeInternalForces(ChVectorDynamic<>& Fi) {
    //if (GetMaterial()->Get_mu() != 0) {
        // Retrieve the nodal coordinates and nodal coordinate time derivatives
    Matrix6xN ebar_ebardot;
    CalcCombinedCoordMatrix(ebar_ebardot);
    ComputeInternalForceDamping(Fi, ebar_ebardot);
    //}
    //else {
    //    // Element coordinates in matrix form
    //    Matrix3xN ebar;
    //    CalcCoordMatrix(ebar);
    //    ComputeInternalForceNoDamping(Fi, ebar);
    //}
}

// Calculate the global matrix H as a linear combination of K, R, and M:
//   H = Mfactor * [M] + Kfactor * [K] + Rfactor * [R]

void ChElementShellANCF_3443_MR_DampNumJac::ComputeKRMmatricesGlobal(ChMatrixRef H,
                                                            double Kfactor,
                                                            double Rfactor,
                                                            double Mfactor) {

    //if (GetMaterial()->Get_mu() != 0) {
    ComputeInternalJacobianDamping(H, Kfactor, Rfactor, Mfactor);
    //}
    //else {
    //    ComputeInternalJacobianNoDamping(H, Kfactor, Mfactor);
    //}
}

// Compute the generalized force vector due to gravity using the efficient ANCF specific method
void ChElementShellANCF_3443_MR_DampNumJac::ComputeGravityForces(ChVectorDynamic<>& Fg, const ChVector<>& G_acc) {
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
// Interface to ChElementShell base class
// -----------------------------------------------------------------------------

void ChElementShellANCF_3443_MR_DampNumJac::EvaluateSectionFrame(const double xi,
                                                        const double eta,
                                                        ChVector<>& point,
                                                        ChQuaternion<>& rot) {
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, eta, 0, m_thicknessZ, m_midsurfoffset);
    VectorN Sxi_xi_compact;
    Calc_Sxi_xi_compact(Sxi_xi_compact, xi, eta, 0, m_thicknessZ, m_midsurfoffset);
    VectorN Sxi_eta_compact;
    Calc_Sxi_eta_compact(Sxi_eta_compact, xi, eta, 0, m_thicknessZ, m_midsurfoffset);

    Matrix3xN e_bar;
    CalcCoordMatrix(e_bar);

    // r = S*e written in compact form
    point = e_bar * Sxi_compact;

    // Since ANCF does not use rotations, calculate an approximate
    // rotation based off the position vector gradients
    ChVector<double> MidsurfaceX = e_bar * Sxi_xi_compact * 2 / m_lenX;
    ChVector<double> MidsurfaceY = e_bar * Sxi_eta_compact * 2 / m_lenY;

    // Since the position vector gradients are not in general orthogonal,
    // set the Dx direction tangent to the shell xi axis and
    // compute the Dy and Dz directions by using a
    // Gram-Schmidt orthonormalization, guided by the shell eta axis
    ChMatrix33<> msect;
    msect.Set_A_Xdir(MidsurfaceX, MidsurfaceY);

    rot = msect.Get_A_quaternion();
}

void ChElementShellANCF_3443_MR_DampNumJac::EvaluateSectionPoint(const double xi, const double eta, ChVector<>& point) {
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, eta, 0, m_thicknessZ, m_midsurfoffset);

    Matrix3xN e_bar;
    CalcCoordMatrix(e_bar);

    // r = S*e written in compact form
    point = e_bar * Sxi_compact;
}

void ChElementShellANCF_3443_MR_DampNumJac::EvaluateSectionVelNorm(const double xi, const double eta, ChVector<>& Result) {
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, eta, 0, m_thicknessZ, m_midsurfoffset);

    Matrix3xN e_bardot;
    CalcCoordDerivMatrix(e_bardot);

    // rdot = S*edot written in compact form
    Result = e_bardot * Sxi_compact;
}

// -----------------------------------------------------------------------------
// Functions for ChLoadable interface
// -----------------------------------------------------------------------------

// Gets all the DOFs packed in a single vector (position part).

void ChElementShellANCF_3443_MR_DampNumJac::LoadableGetStateBlock_x(int block_offset, ChState& mD) {
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
}

// Gets all the DOFs packed in a single vector (velocity part).

void ChElementShellANCF_3443_MR_DampNumJac::LoadableGetStateBlock_w(int block_offset, ChStateDelta& mD) {
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
}

/// Increment all DOFs using a delta.

void ChElementShellANCF_3443_MR_DampNumJac::LoadableStateIncrement(const unsigned int off_x,
                                                          ChState& x_new,
                                                          const ChState& x,
                                                          const unsigned int off_v,
                                                          const ChStateDelta& Dv) {
    m_nodes[0]->NodeIntStateIncrement(off_x, x_new, x, off_v, Dv);
    m_nodes[1]->NodeIntStateIncrement(off_x + 12, x_new, x, off_v + 12, Dv);
    m_nodes[2]->NodeIntStateIncrement(off_x + 24, x_new, x, off_v + 24, Dv);
    m_nodes[3]->NodeIntStateIncrement(off_x + 36, x_new, x, off_v + 36, Dv);
}

// Get the pointers to the contained ChVariables, appending to the mvars vector.

void ChElementShellANCF_3443_MR_DampNumJac::LoadableGetVariables(std::vector<ChVariables*>& mvars) {
    for (int i = 0; i < m_nodes.size(); ++i) {
        mvars.push_back(&m_nodes[i]->Variables());
        mvars.push_back(&m_nodes[i]->Variables_D());
        mvars.push_back(&m_nodes[i]->Variables_DD());
        mvars.push_back(&m_nodes[i]->Variables_DDD());
    }
}

// Evaluate N'*F, which is the projection of the applied point force and moment at the midsurface coordinates (xi,eta,0)
// This calculation takes a slightly different form for ANCF elements
// For this ANCF element, only the first 6 entries in F are used in the calculation.  The first three entries is
// the applied force in global coordinates and the second 3 entries is the applied moment in global space.

void ChElementShellANCF_3443_MR_DampNumJac::ComputeNF(
    const double xi,             // parametric coordinate in surface
    const double eta,            // parametric coordinate in surface
    ChVectorDynamic<>& Qi,       // Return result of Q = N'*F  here
    double& detJ,                // Return det[J] here
    const ChVectorDynamic<>& F,  // Input F vector, size is =n. field coords.
    ChVectorDynamic<>* state_x,  // if != 0, update state (pos. part) to this, then evaluate Q
    ChVectorDynamic<>* state_w   // if != 0, update state (speed part) to this, then evaluate Q
) {
    // Compute the generalized force vector for the applied force component using the compact form of the shape
    // functions.  This requires a reshaping of the calculated matrix to get it into the correct vector order (just a
    // reinterpretation of the data since the matrix is in row-major format)
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, eta, 0, m_thicknessZ, m_midsurfoffset);
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
    Calc_Sxi_D(Sxi_D, xi, eta, 0, m_thicknessZ, m_midsurfoffset);

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
    //  actual differential area and the normalized differential area.  The cross product is
    //  used to calculate this area ratio for potential use in Gauss-Quadrature or similar numeric integration.
    detJ = J_Cxi.col(0).cross(J_Cxi.col(1)).norm();
}

// Evaluate N'*F, which is the projection of the applied point force and moment at the coordinates (xi,eta,zeta)
// This calculation takes a slightly different form for ANCF elements
// For this ANCF element, only the first 6 entries in F are used in the calculation.  The first three entries is
// the applied force in global coordinates and the second 3 entries is the applied moment in global space.

void ChElementShellANCF_3443_MR_DampNumJac::ComputeNF(
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
    Calc_Sxi_compact(Sxi_compact, xi, eta, zeta, m_thicknessZ, m_midsurfoffset);
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
    Calc_Sxi_D(Sxi_D, xi, eta, zeta, m_thicknessZ, m_midsurfoffset);

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

// Calculate the average element density (needed for ChLoaderVolumeGravity).

double ChElementShellANCF_3443_MR_DampNumJac::GetDensity() {
    double tot_density = 0;
    for (int kl = 0; kl < m_numLayers; kl++) {
        double rho = m_layers[kl].GetMaterial()->Get_rho();
        double layerthick = m_layers[kl].Get_thickness();
        tot_density += rho * layerthick;
    }
    return tot_density / m_thicknessZ;
}

// Calculate normal to the midsurface at coordinates (xi, eta).

ChVector<> ChElementShellANCF_3443_MR_DampNumJac::ComputeNormal(const double xi, const double eta) {
    VectorN Sxi_zeta_compact;
    Calc_Sxi_zeta_compact(Sxi_zeta_compact, xi, eta, 0, m_thicknessZ, m_midsurfoffset);

    Matrix3xN e_bar;
    CalcCoordMatrix(e_bar);

    // Calculate the position vector gradient with respect to zeta at the current point (whose length may not equal 1)
    ChVector<> r_zeta = e_bar * Sxi_zeta_compact;

    return r_zeta.GetNormalized();
}

// -----------------------------------------------------------------------------
// Mass Matrix & Generalized Force Due to Gravity Calculation
// -----------------------------------------------------------------------------

void ChElementShellANCF_3443_MR_DampNumJac::ComputeMassMatrixAndGravityForce() {
    // For this element, the mass matrix integrand is of order 12 in xi, 12 in eta, and 4 in zeta.
    // 7 GQ Points are needed in the xi & eta directions and 3 GQ Points are needed in the zeta direction for
    // exact integration of the element's mass matrix, even if the reference configuration is not straight. Since the
    // major pieces of the generalized force due to gravity can also be used to calculate the mass matrix, these
    // calculations are performed at the same time.  Only the matrix that scales the acceleration due to gravity is
    // calculated at this time so that any changes to the acceleration due to gravity in the system are correctly
    // accounted for in the generalized internal force calculation.

    ChQuadratureTables* GQTable = GetStaticGQTables();
    unsigned int GQ_idx_xi_eta = 6;  // 7 Point Gauss-Quadrature;
    unsigned int GQ_idx_zeta = 2;    // 3 Point Gauss-Quadrature;

    // Mass Matrix in its compact matrix form.  Since the mass matrix is symmetric, just the upper diagonal entries will
    // be stored.
    MatrixNxN MassMatrixCompactSquare;

    // Set these to zeros since they will be incremented as the vector/matrix is calculated
    MassMatrixCompactSquare.setZero();
    m_GravForceScale.setZero();

    for (size_t kl = 0; kl < m_numLayers; kl++) {
        double rho = m_layers[kl].GetMaterial()->Get_rho();  // Density of the material for the current layer
        double thickness = m_layers[kl].Get_thickness();
        double zoffset = m_layer_zoffsets[kl];
        double layer_midsurface_offset =
            -m_thicknessZ / 2 + m_layer_zoffsets[kl] + m_layers[kl].Get_thickness() / 2 + m_midsurfoffset;

        // Sum the contribution to the mass matrix and generalized force due to gravity at the current point
        for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi_eta].size(); it_xi++) {
            for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[GQ_idx_xi_eta].size(); it_eta++) {
                for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[GQ_idx_zeta].size(); it_zeta++) {
                    double GQ_weight = GQTable->Weight[GQ_idx_xi_eta][it_xi] * GQTable->Weight[GQ_idx_xi_eta][it_eta] *
                                       GQTable->Weight[GQ_idx_zeta][it_zeta];
                    double xi = GQTable->Lroots[GQ_idx_xi_eta][it_xi];
                    double eta = GQTable->Lroots[GQ_idx_xi_eta][it_eta];
                    double zeta = GQTable->Lroots[GQ_idx_zeta][it_zeta];
                    double det_J_0xi = Calc_det_J_0xi(xi, eta, zeta, thickness,
                                                      zoffset);  // determinant of the element Jacobian (volume ratio)

                    VectorN Sxi_compact;  // Vector of the Unique Normalized Shape Functions
                    Calc_Sxi_compact(Sxi_compact, xi, eta, zeta, thickness, layer_midsurface_offset);

                    m_GravForceScale += (GQ_weight * rho * det_J_0xi) * Sxi_compact;
                    MassMatrixCompactSquare += (GQ_weight * rho * det_J_0xi) * Sxi_compact * Sxi_compact.transpose();
                }
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

// Precalculate constant matrices and scalars for the internal force calculations
void ChElementShellANCF_3443_MR_DampNumJac::PrecomputeInternalForceMatricesWeights() {
    ChQuadratureTables* GQTable = GetStaticGQTables();
    unsigned int GQ_idx_xi_eta = NP - 1;  // Gauss-Quadrature table index for xi and eta
    unsigned int GQ_idx_zeta = NT - 1;    // Gauss-Quadrature table index for zeta

    m_SD.resize(m_numLayers * NSF, 3 * NIP);
    m_kGQ.resize(1, m_numLayers * NIP);

    for (size_t kl = 0; kl < m_numLayers; kl++) {
        double thickness = m_layers[kl].Get_thickness();
        double layer_midsurface_offset = -m_thicknessZ / 2 + m_layer_zoffsets[kl] + thickness / 2 + m_midsurfoffset;

        for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi_eta].size(); it_xi++) {
            for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[GQ_idx_xi_eta].size(); it_eta++) {
                for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[GQ_idx_zeta].size(); it_zeta++) {
                    double GQ_weight = GQTable->Weight[GQ_idx_xi_eta][it_xi] * GQTable->Weight[GQ_idx_xi_eta][it_eta] *
                                       GQTable->Weight[GQ_idx_zeta][it_zeta];
                    double xi = GQTable->Lroots[GQ_idx_xi_eta][it_xi];
                    double eta = GQTable->Lroots[GQ_idx_xi_eta][it_eta];
                    double zeta = GQTable->Lroots[GQ_idx_zeta][it_zeta];
                    auto index = it_zeta + it_eta * GQTable->Lroots[GQ_idx_zeta].size() +
                                 it_xi * GQTable->Lroots[GQ_idx_zeta].size() * GQTable->Lroots[GQ_idx_xi_eta].size();
                    ChMatrix33<double>
                        J_0xi;  // Element Jacobian between the reference configuration and normalized configuration
                    MatrixNx3c Sxi_D;  // Matrix of normalized shape function derivatives

                    Calc_Sxi_D(Sxi_D, xi, eta, zeta, thickness, layer_midsurface_offset);
                    J_0xi.noalias() = m_ebar0 * Sxi_D;

                    // Adjust the shape function derivative matrix to account for a potentially deformed reference state
                    ChMatrixNM<double, NSF, 3> SD_precompute_D = Sxi_D * J_0xi.inverse();
                    m_kGQ(index + NIP * kl) = -J_0xi.determinant() * GQ_weight;

                    // Group all of the columns together in blocks, and then layer by layer across the entire element
                    m_SD.block<NSF, 1>(kl * NSF, index + 0 * NIP) = SD_precompute_D.col(0);
                    m_SD.block<NSF, 1>(kl * NSF, index + 1 * NIP) = SD_precompute_D.col(1);
                    m_SD.block<NSF, 1>(kl * NSF, index + 2 * NIP) = SD_precompute_D.col(2);
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Elastic force calculation
// -----------------------------------------------------------------------------

void ChElementShellANCF_3443_MR_DampNumJac::ComputeInternalForceDamping(ChVectorDynamic<>& Fi, Matrix6xN& ebar_ebardot) {
    assert(Fi.size() == 3 * NSF);


    // Zero out the QiCompact since the contribution from each GQ point will be added to it
    MatrixNx3 QiCompact;
    QiCompact.setZero();

    for (size_t kl = 0; kl < m_numLayers; kl++) {
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

    ChMatrixNM<double, 6, 3 * NIP> FC = ebar_ebardot * m_SD.block<NSF, 3 * NIP>(kl * NSF, 0);

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

    double c10 = m_layers[kl].GetMaterial()->Get_c10();
    double c01 = m_layers[kl].GetMaterial()->Get_c01();
    double k = m_layers[kl].GetMaterial()->Get_k();
    double mu = m_layers[kl].GetMaterial()->Get_mu();

    ArrayNIP kGQmu_over_detF3 = mu * m_kGQ.block<1, NIP>(0, kl * NIP) / (detF*detF*detF);

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

    ArrayNIP s0 = 2.0 * m_kGQ.block<1, NIP>(0, kl * NIP) * detF_m2_3;
    ArrayNIP s2 = -c01 * s0 * detF_m2_3;
    s0 *= c10;
    ArrayNIP s1 = s0 - s2 * I1;
    ArrayNIP s3 = k * (detF - 1.0)*m_kGQ.block<1, NIP>(0, kl * NIP) - (I1*s0 - 2 * I2 * s2) / (3.0 * detF);

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

    // =============================================================================
    // Multiply the scaled first Piola-Kirchoff stresses by the shape function derivative matrix to get the
    // generalized force vector in matrix form (in the correct order if its calculated in row-major memory layout)
    // =============================================================================

    QiCompact.noalias() += m_SD.block<NSF, 3 * NIP>(kl * NSF, 0) * P_Block;
    }

    Eigen::Map<ChVectorN<double, 3 * NSF>> QiReshaped(QiCompact.data(), QiCompact.size());
    Fi.noalias() = QiReshaped;
}

void ChElementShellANCF_3443_MR_DampNumJac::ComputeInternalForceNoDamping(ChVectorDynamic<>& Fi, Matrix3xN& ebar) {
    assert(Fi.size() == 3 * NSF);

    // Zero out the QiCompact since the contribution from each GQ point will be added to it
    MatrixNx3 QiCompact;
    QiCompact.setZero();

    for (size_t kl = 0; kl < m_numLayers; kl++) {
    // =============================================================================
    // Calculate the deformation gradient for all Gauss quadrature points in a single matrix multiplication.  Note
    // that since the shape function derivative matrix is ordered by columns, the resulting deformation gradient
    // will be ordered by block matrix (row vectors) components
    //      [F11     F12     F13    ]
    // FC = [F21     F22     F23    ]
    //      [F31     F32     F33    ]
    // =============================================================================

    ChMatrixNM<double, 3, 3 * NIP> FC = ebar * m_SD.block<NSF, 3 * NIP>(kl * NSF, 0);

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

    double c10 = m_layers[kl].GetMaterial()->Get_c10();
    double c01 = m_layers[kl].GetMaterial()->Get_c01();
    double k = m_layers[kl].GetMaterial()->Get_k();

    ArrayNIP s0 = 2.0 * m_kGQ.block<1, NIP>(0, kl * NIP) * detF_m2_3;
    ArrayNIP s2 = -c01 * s0 * detF_m2_3;
    s0 *= c10;
    ArrayNIP s1 = s0 - s2 * I1;
    ArrayNIP s3 = k * (detF - 1.0)*m_kGQ.block<1, NIP>(0, kl * NIP) - (I1*s0 - 2 * I2 * s2) / (3.0 * detF);

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

    // =============================================================================
    // Multiply the scaled first Piola-Kirchoff stresses by the shape function derivative matrix to get the
    // generalized force vector in matrix form (in the correct order if its calculated in row-major memory layout)
    // =============================================================================

    QiCompact.noalias() += m_SD.block<NSF, 3 * NIP>(kl * NSF, 0) * P_Block;
    }

    Eigen::Map<ChVectorN<double, 3 * NSF>> QiReshaped(QiCompact.data(), QiCompact.size());
    Fi.noalias() = QiReshaped;
}


// -----------------------------------------------------------------------------
// Jacobians of internal forces
// -----------------------------------------------------------------------------

void ChElementShellANCF_3443_MR_DampNumJac::ComputeInternalJacobianDamping(ChMatrixRef H, double Kfactor, double Rfactor, double Mfactor) {
    assert((H.rows() == 3 * NSF) && (H.cols() == 3 * NSF));

    ChVectorDynamic<double> FiOrignal(3 * NSF);
    ChVectorDynamic<double> FiDelta(3 * NSF);

    Matrix6xN ebar_ebardot;
    CalcCombinedCoordMatrix(ebar_ebardot);

    double delta = 1e-6;

    Matrix3Nx3N Jac;

    // Compute the Jacobian via numerical differentiation of the generalized internal force vector
    // Since the generalized force vector due to gravity is a constant, it doesn't affect this
    // Jacobian calculation
    ComputeInternalForceDamping(FiOrignal, ebar_ebardot);
    for (unsigned int i = 0; i < (3 * NSF); i++) {
        ebar_ebardot(i % 3, i / 3) += delta;
        ComputeInternalForceDamping(FiDelta, ebar_ebardot);
        Jac.col(i).noalias() = -Kfactor / delta * (FiDelta - FiOrignal);
        ebar_ebardot(i % 3, i / 3) -= delta;

        ebar_ebardot(i % 3 + 3, i / 3) += delta;
        ComputeInternalForceDamping(FiDelta, ebar_ebardot);
        Jac.col(i).noalias() += -Rfactor / delta * (FiDelta - FiOrignal);
        ebar_ebardot(i % 3 + 3, i / 3) -= delta;
    }

    // Add in the contribution from the Mass Matrix which is stored in compact upper triangular form
    unsigned int idx = 0;
    for (unsigned int j = 0; j < NSF; j++) {
        for (unsigned int i = j; i < NSF; i++) {
            Jac(3 * i, 3 * j) += Mfactor * m_MassMatrix(idx);
            Jac(3 * i + 1, 3 * j + 1) += Mfactor * m_MassMatrix(idx);
            Jac(3 * i + 2, 3 * j + 2) += Mfactor * m_MassMatrix(idx);
            if (i != j) {
                Jac(3 * j, 3 * i) += Mfactor * m_MassMatrix(idx);
                Jac(3 * j + 1, 3 * i + 1) += Mfactor * m_MassMatrix(idx);
                Jac(3 * j + 2, 3 * i + 2) += Mfactor * m_MassMatrix(idx);
            }
            idx++;
        }
    }

    H.noalias() = Jac;
}

void ChElementShellANCF_3443_MR_DampNumJac::ComputeInternalJacobianNoDamping(ChMatrixRef H, double Kfactor, double Mfactor) {
    assert((H.rows() == 3 * NSF) && (H.cols() == 3 * NSF));

    ChVectorDynamic<double> FiOrignal(3 * NSF);
    ChVectorDynamic<double> FiDelta(3 * NSF);

    Matrix3xN ebar;
    CalcCoordMatrix(ebar);

    double delta = 1e-6;

    Matrix3Nx3N Jac;

    // Compute the Jacobian via numerical differentiation of the generalized internal force vector
    // Since the generalized force vector due to gravity is a constant, it doesn't affect this
    // Jacobian calculation
    ComputeInternalForceNoDamping(FiOrignal, ebar);
    for (unsigned int i = 0; i < (3 * NSF); i++) {
        ebar(i % 3, i / 3) += delta;
        ComputeInternalForceNoDamping(FiDelta, ebar);
        Jac.col(i).noalias() = -Kfactor / delta * (FiDelta - FiOrignal);
        ebar(i % 3, i / 3) -= delta;
    }

    // Add in the contribution from the Mass Matrix which is stored in compact upper triangular form
    unsigned int idx = 0;
    for (unsigned int j = 0; j < NSF; j++) {
        for (unsigned int i = j; i < NSF; i++) {
            Jac(3 * i, 3 * j) += Mfactor * m_MassMatrix(idx);
            Jac(3 * i + 1, 3 * j + 1) += Mfactor * m_MassMatrix(idx);
            Jac(3 * i + 2, 3 * j + 2) += Mfactor * m_MassMatrix(idx);
            if (i != j) {
                Jac(3 * j, 3 * i) += Mfactor * m_MassMatrix(idx);
                Jac(3 * j + 1, 3 * i + 1) += Mfactor * m_MassMatrix(idx);
                Jac(3 * j + 2, 3 * i + 2) += Mfactor * m_MassMatrix(idx);
            }
            idx++;
        }
    }

    H.noalias() = Jac;
}

// -----------------------------------------------------------------------------
// Shape functions
// -----------------------------------------------------------------------------

// Nx1 Vector Form of the Normalized Shape Functions
// [s1; s2; s3; ...]

void ChElementShellANCF_3443_MR_DampNumJac::Calc_Sxi_compact(VectorN& Sxi_compact,
                                                    double xi,
                                                    double eta,
                                                    double zeta,
                                                    double thickness,
                                                    double zoffset) {
    Sxi_compact(0) = -0.125 * (xi - 1) * (eta - 1) * (eta * eta + eta + xi * xi + xi - 2);
    Sxi_compact(1) = -0.0625 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (eta - 1);
    Sxi_compact(2) = -0.0625 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (xi - 1);
    Sxi_compact(3) = -0.125 * (xi - 1) * (eta - 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);

    Sxi_compact(4) = 0.125 * (xi + 1) * (eta - 1) * (eta * eta + eta + xi * xi - xi - 2);
    Sxi_compact(5) = -0.0625 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (eta - 1);
    Sxi_compact(6) = 0.0625 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (xi + 1);
    Sxi_compact(7) = 0.125 * (xi + 1) * (eta - 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);

    Sxi_compact(8) = -0.125 * (xi + 1) * (eta + 1) * (eta * eta - eta + xi * xi - xi - 2);
    Sxi_compact(9) = 0.0625 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (eta + 1);
    Sxi_compact(10) = 0.0625 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (xi + 1);
    Sxi_compact(11) = -0.125 * (xi + 1) * (eta + 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);

    Sxi_compact(12) = 0.125 * (xi - 1) * (eta + 1) * (eta * eta - eta + xi * xi + xi - 2);
    Sxi_compact(13) = 0.0625 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (eta + 1);
    Sxi_compact(14) = -0.0625 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (xi - 1);
    Sxi_compact(15) = 0.125 * (xi - 1) * (eta + 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
}

// Nx1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to xi
// [s1; s2; s3; ...]

void ChElementShellANCF_3443_MR_DampNumJac::Calc_Sxi_xi_compact(VectorN& Sxi_xi_compact,
                                                       double xi,
                                                       double eta,
                                                       double zeta,
                                                       double thickness,
                                                       double zoffset) {
    Sxi_xi_compact(0) = -0.125 * (eta - 1) * (eta * eta + eta + 3 * xi * xi - 3);
    Sxi_xi_compact(1) = -0.0625 * m_lenX * (3 * xi + 1) * (xi - 1) * (eta - 1);
    Sxi_xi_compact(2) = -0.0625 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1);
    Sxi_xi_compact(3) = -0.125 * (eta - 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);

    Sxi_xi_compact(4) = 0.125 * (eta - 1) * (eta * eta + eta + 3 * xi * xi - 3);
    Sxi_xi_compact(5) = -0.0625 * m_lenX * (xi + 1) * (3 * xi - 1) * (eta - 1);
    Sxi_xi_compact(6) = 0.0625 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1);
    Sxi_xi_compact(7) = 0.125 * (eta - 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);

    Sxi_xi_compact(8) = -0.125 * (eta + 1) * (eta * eta - eta + 3 * xi * xi - 3);
    Sxi_xi_compact(9) = 0.0625 * m_lenX * (xi + 1) * (3 * xi - 1) * (eta + 1);
    Sxi_xi_compact(10) = 0.0625 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1);
    Sxi_xi_compact(11) = -0.125 * (eta + 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);

    Sxi_xi_compact(12) = 0.125 * (eta + 1) * (eta * eta - eta + 3 * xi * xi - 3);
    Sxi_xi_compact(13) = 0.0625 * m_lenX * (3 * xi + 1) * (xi - 1) * (eta + 1);
    Sxi_xi_compact(14) = -0.0625 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1);
    Sxi_xi_compact(15) = 0.125 * (eta + 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
}

// Nx1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to eta
// [s1; s2; s3; ...]

void ChElementShellANCF_3443_MR_DampNumJac::Calc_Sxi_eta_compact(VectorN& Sxi_eta_compact,
                                                        double xi,
                                                        double eta,
                                                        double zeta,
                                                        double thickness,
                                                        double zoffset) {
    Sxi_eta_compact(0) = -0.125 * (xi - 1) * (3 * eta * eta + xi * xi + xi - 3);
    Sxi_eta_compact(1) = -0.0625 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1);
    Sxi_eta_compact(2) = -0.0625 * m_lenY * (3 * eta + 1) * (eta - 1) * (xi - 1);
    Sxi_eta_compact(3) = -0.125 * (xi - 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);

    Sxi_eta_compact(4) = 0.125 * (xi + 1) * (3 * eta * eta + xi * xi - xi - 3);
    Sxi_eta_compact(5) = -0.0625 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1);
    Sxi_eta_compact(6) = 0.0625 * m_lenY * (3 * eta + 1) * (eta - 1) * (xi + 1);
    Sxi_eta_compact(7) = 0.125 * (xi + 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);

    Sxi_eta_compact(8) = -0.125 * (xi + 1) * (3 * eta * eta + xi * xi - xi - 3);
    Sxi_eta_compact(9) = 0.0625 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1);
    Sxi_eta_compact(10) = 0.0625 * m_lenY * (eta + 1) * (3 * eta - 1) * (xi + 1);
    Sxi_eta_compact(11) = -0.125 * (xi + 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);

    Sxi_eta_compact(12) = 0.125 * (xi - 1) * (3 * eta * eta + xi * xi + xi - 3);
    Sxi_eta_compact(13) = 0.0625 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1);
    Sxi_eta_compact(14) = -0.0625 * m_lenY * (eta + 1) * (3 * eta - 1) * (xi - 1);
    Sxi_eta_compact(15) = 0.125 * (xi - 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
}

// Nx1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to zeta
// [s1; s2; s3; ...]

void ChElementShellANCF_3443_MR_DampNumJac::Calc_Sxi_zeta_compact(VectorN& Sxi_zeta_compact,
                                                         double xi,
                                                         double eta,
                                                         double zeta,
                                                         double thickness,
                                                         double zoffset) {
    Sxi_zeta_compact(0) = 0.0;
    Sxi_zeta_compact(1) = 0.0;
    Sxi_zeta_compact(2) = 0.0;
    Sxi_zeta_compact(3) = 0.125 * thickness * (xi - 1) * (eta - 1);

    Sxi_zeta_compact(4) = 0.0;
    Sxi_zeta_compact(5) = 0.0;
    Sxi_zeta_compact(6) = 0.0;
    Sxi_zeta_compact(7) = -0.125 * thickness * (xi + 1) * (eta - 1);

    Sxi_zeta_compact(8) = 0.0;
    Sxi_zeta_compact(9) = 0.0;
    Sxi_zeta_compact(10) = 0.0;
    Sxi_zeta_compact(11) = 0.125 * thickness * (xi + 1) * (eta + 1);

    Sxi_zeta_compact(12) = 0.0;
    Sxi_zeta_compact(13) = 0.0;
    Sxi_zeta_compact(14) = 0.0;
    Sxi_zeta_compact(15) = -0.125 * thickness * (xi - 1) * (eta + 1);
}

// Nx3 compact form of the partial derivatives of Normalized Shape Functions with respect to xi, eta, and zeta by
// columns

void ChElementShellANCF_3443_MR_DampNumJac::Calc_Sxi_D(MatrixNx3c& Sxi_D,
                                              double xi,
                                              double eta,
                                              double zeta,
                                              double thickness,
                                              double zoffset) {
    VectorN Sxi_D_col;
    Calc_Sxi_xi_compact(Sxi_D_col, xi, eta, zeta, thickness, zoffset);
    Sxi_D.col(0) = Sxi_D_col;

    Calc_Sxi_eta_compact(Sxi_D_col, xi, eta, zeta, thickness, zoffset);
    Sxi_D.col(1) = Sxi_D_col;

    Calc_Sxi_zeta_compact(Sxi_D_col, xi, eta, zeta, thickness, zoffset);
    Sxi_D.col(2) = Sxi_D_col;
}

// -----------------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------------

void ChElementShellANCF_3443_MR_DampNumJac::CalcCoordVector(Vector3N& e) {
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
}

void ChElementShellANCF_3443_MR_DampNumJac::CalcCoordMatrix(Matrix3xN& ebar) {
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
}

void ChElementShellANCF_3443_MR_DampNumJac::CalcCoordDerivVector(Vector3N& edot) {
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
}

void ChElementShellANCF_3443_MR_DampNumJac::CalcCoordDerivMatrix(Matrix3xN& ebardot) {
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
}

void ChElementShellANCF_3443_MR_DampNumJac::CalcCombinedCoordMatrix(Matrix6xN& ebar_ebardot) {
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
}

// Calculate the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element

void ChElementShellANCF_3443_MR_DampNumJac::Calc_J_0xi(ChMatrix33<double>& J_0xi,
                                              double xi,
                                              double eta,
                                              double zeta,
                                              double thickness,
                                              double zoffset) {
    MatrixNx3c Sxi_D;
    Calc_Sxi_D(Sxi_D, xi, eta, zeta, thickness, zoffset);

    J_0xi = m_ebar0 * Sxi_D;
}

// Calculate the determinant of the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element

double ChElementShellANCF_3443_MR_DampNumJac::Calc_det_J_0xi(double xi,
                                                    double eta,
                                                    double zeta,
                                                    double thickness,
                                                    double zoffset) {
    ChMatrix33<double> J_0xi;
    Calc_J_0xi(J_0xi, xi, eta, zeta, thickness, zoffset);

    return (J_0xi.determinant());
}

void ChElementShellANCF_3443_MR_DampNumJac::RotateReorderStiffnessMatrix(ChMatrixNM<double, 6, 6>& D, double theta) {
    // Reorder the stiffness matrix from the order assumed in ChMaterialShellANCF.h
    //  E = [E11,E22,2*E12,E33,2*E13,2*E23]
    // to the order assumed in this element formulation
    //  E = [E11,E22,E33,2*E23,2*E13,2*E12]
    // Note that the 6x6 stiffness matrix is symmetric

    ChMatrixNM<double, 6, 6> D_Reordered;
    D_Reordered << D(0, 0), D(0, 1), D(0, 3), D(0, 5), D(0, 4), D(0, 2), D(1, 0), D(1, 1), D(1, 3), D(1, 5), D(1, 4),
        D(1, 2), D(3, 0), D(3, 1), D(3, 3), D(3, 5), D(3, 4), D(3, 2), D(5, 0), D(5, 1), D(5, 3), D(5, 5), D(5, 4),
        D(5, 2), D(4, 0), D(4, 1), D(4, 3), D(4, 5), D(4, 4), D(4, 2), D(2, 0), D(2, 1), D(2, 3), D(2, 5), D(2, 4),
        D(2, 2);

    // Stiffness Tensor Rotation Matrix From:
    // http://solidmechanics.org/text/Chapter3_2/Chapter3_2.htm

    ChMatrixNM<double, 6, 6> K;
    K << std::cos(theta) * std::cos(theta), std::sin(theta) * std::sin(theta), 0, 0, 0,
        2 * std::cos(theta) * std::sin(theta), std::sin(theta) * std::sin(theta), std::cos(theta) * std::cos(theta), 0,
        0, 0, -2 * std::cos(theta) * std::sin(theta), 0, 0, 1, 0, 0, 0, 0, 0, 0, std::cos(theta), std::sin(theta), 0, 0,
        0, 0, -std::sin(theta), std::cos(theta), 0, -std::cos(theta) * std::sin(theta),
        std::cos(theta) * std::sin(theta), 0, 0, 0,
        std::cos(theta) * std::cos(theta) - std::sin(theta) * std::sin(theta);

    D = K * D_Reordered * K.transpose();
}

////////////////////////////////////////////////////////////////

//#ifndef CH_QUADRATURE_STATIC_TABLES
#define CH_QUADRATURE_STATIC_TABLES 10
ChQuadratureTables static_tables_3443_MR_DampNumJac(1, CH_QUADRATURE_STATIC_TABLES);
//#endif // !CH_QUADRATURE_STATIC_TABLES

ChQuadratureTables* ChElementShellANCF_3443_MR_DampNumJac::GetStaticGQTables() {
    return &static_tables_3443_MR_DampNumJac;
}

////////////////////////////////////////////////////////////////

// ============================================================================
// Implementation of ChElementShellANCF_3443_MR_DampNumJac::Layer methods
// ============================================================================

// Private constructor (a layer can be created only by adding it to an element)

ChElementShellANCF_3443_MR_DampNumJac::Layer::Layer(double thickness,
                                           std::shared_ptr<ChMaterialShellANCF_MR> material)
    : m_thickness(thickness), m_material(material) {}

}  // namespace fea
}  // namespace chrono

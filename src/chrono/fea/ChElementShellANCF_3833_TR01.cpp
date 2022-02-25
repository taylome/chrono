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
// Higher order ANCF shell element with 8 nodes. Description of this element (3833_TR01) and its internal forces may be
// found in: Henrik Ebel, Marko K Matikainen, Vesa-Ville Hurskainen, and Aki Mikkola. Analysis of high-order
// quadrilateral plate elements based on the absolute nodal coordinate formulation for three - dimensional
// elasticity.Advances in Mechanical Engineering, 9(6) : 1687814017705069, 2017.
// =============================================================================
//
// =============================================================================
// TR01 = Direct Textbook Translation
// =============================================================================
// Mass Matrix = Compact Upper Triangular
// Full Number of GQ Points
// Nodal Coordinates in Vector Form (Lots of multiplications by zeros)
// Textbook Mathematics
// No Precomputing Quantities
// Numeric Jacobian
// =============================================================================

#include "chrono/fea/ChElementShellANCF_3833_TR01.h"
#include "chrono/physics/ChSystem.h"

namespace chrono {
namespace fea {

// ------------------------------------------------------------------------------
// Constructor
// ------------------------------------------------------------------------------
ChElementShellANCF_3833_TR01::ChElementShellANCF_3833_TR01()
    : m_numLayers(0), m_lenX(0), m_lenY(0), m_thicknessZ(0), m_midsurfoffset(0), m_Alpha(0), m_damping_enabled(false) {
    m_nodes.resize(8);
}

// ------------------------------------------------------------------------------
// Set element nodes
// ------------------------------------------------------------------------------

void ChElementShellANCF_3833_TR01::SetNodes(std::shared_ptr<ChNodeFEAxyzDD> nodeA,
                                            std::shared_ptr<ChNodeFEAxyzDD> nodeB,
                                            std::shared_ptr<ChNodeFEAxyzDD> nodeC,
                                            std::shared_ptr<ChNodeFEAxyzDD> nodeD,
                                            std::shared_ptr<ChNodeFEAxyzDD> nodeE,
                                            std::shared_ptr<ChNodeFEAxyzDD> nodeF,
                                            std::shared_ptr<ChNodeFEAxyzDD> nodeG,
                                            std::shared_ptr<ChNodeFEAxyzDD> nodeH) {
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
    mvars.push_back(&m_nodes[1]->Variables());
    mvars.push_back(&m_nodes[1]->Variables_D());
    mvars.push_back(&m_nodes[1]->Variables_DD());
    mvars.push_back(&m_nodes[2]->Variables());
    mvars.push_back(&m_nodes[2]->Variables_D());
    mvars.push_back(&m_nodes[2]->Variables_DD());
    mvars.push_back(&m_nodes[3]->Variables());
    mvars.push_back(&m_nodes[3]->Variables_D());
    mvars.push_back(&m_nodes[3]->Variables_DD());
    mvars.push_back(&m_nodes[4]->Variables());
    mvars.push_back(&m_nodes[4]->Variables_D());
    mvars.push_back(&m_nodes[4]->Variables_DD());
    mvars.push_back(&m_nodes[5]->Variables());
    mvars.push_back(&m_nodes[5]->Variables_D());
    mvars.push_back(&m_nodes[5]->Variables_DD());
    mvars.push_back(&m_nodes[6]->Variables());
    mvars.push_back(&m_nodes[6]->Variables_D());
    mvars.push_back(&m_nodes[6]->Variables_DD());
    mvars.push_back(&m_nodes[7]->Variables());
    mvars.push_back(&m_nodes[7]->Variables_D());
    mvars.push_back(&m_nodes[7]->Variables_DD());

    Kmatr.SetVariables(mvars);

    // Initial positions and slopes of the element nodes
    // These values define the reference configuration of the element
    CalcCoordMatrix(m_ebar0);
    CalcCoordVector(m_e0);
}

// -----------------------------------------------------------------------------
// Add a layer.
// -----------------------------------------------------------------------------

void ChElementShellANCF_3833_TR01::AddLayer(double thickness,
                                            double theta,
                                            std::shared_ptr<ChMaterialShellANCF> material) {
    m_layers.push_back(Layer(thickness, theta, material));
    m_layer_zoffsets.push_back(m_thicknessZ);
    m_numLayers += 1;
    m_thicknessZ += thickness;
}

// -----------------------------------------------------------------------------
// Element Settings
// -----------------------------------------------------------------------------

// Specify the element dimensions.

void ChElementShellANCF_3833_TR01::SetDimensions(double lenX, double lenY) {
    m_lenX = lenX;
    m_lenY = lenY;
}

// Offset the midsurface of the composite shell element.  A positive value shifts the element's midsurface upward
// along the elements zeta direction.  The offset should be provided in model units.

void ChElementShellANCF_3833_TR01::SetMidsurfaceOffset(const double offset) {
    m_midsurfoffset = offset;
}

// Set the value for the single term structural damping coefficient.

void ChElementShellANCF_3833_TR01::SetAlphaDamp(double a) {
    m_Alpha = a;
    if (std::abs(m_Alpha) > 1e-10)
        m_damping_enabled = true;
    else
        m_damping_enabled = false;
}

// -----------------------------------------------------------------------------
// Evaluate Strains and Stresses
// -----------------------------------------------------------------------------
// These functions are designed for single function calls.  If these values are needed at the same points in the element
// through out the simulation, then the adjusted normalized shape function derivative matrix (Sxi_D) for each query
// point should be cached and saved to increase the execution speed
// -----------------------------------------------------------------------------

// Get the Green-Lagrange strain tensor at the normalized element coordinates (xi, eta, zeta) [-1...1]

ChMatrix33<> ChElementShellANCF_3833_TR01::GetGreenLagrangeStrain(const double xi,
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

ChMatrix33<> ChElementShellANCF_3833_TR01::GetPK2Stress(const double layer,
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

    // Calculate the Green-Lagrange strain tensor at the current point in Voigt notation
    ChVectorN<double, 6> epsilon_combined;
    epsilon_combined(0) = 0.5 * (F.col(0).dot(F.col(0)) - 1);
    epsilon_combined(1) = 0.5 * (F.col(1).dot(F.col(1)) - 1);
    epsilon_combined(2) = 0.5 * (F.col(2).dot(F.col(2)) - 1);
    epsilon_combined(3) = F.col(1).dot(F.col(2));
    epsilon_combined(4) = F.col(0).dot(F.col(2));
    epsilon_combined(5) = F.col(0).dot(F.col(1));

    if (m_damping_enabled) {
        Matrix3xN ebardot;  // Element coordinate time derivatives in matrix form
        CalcCoordDerivMatrix(ebardot);

        // Calculate the time derivative of the Deformation Gradient at the current point
        ChMatrixNMc<double, 3, 3> Fdot = ebardot * Sxi_D;

        // Calculate the time derivative of the Green-Lagrange strain tensor in Voigt notation
        // and combine it with epsilon assuming a Linear Kelvin-Voigt Viscoelastic material model
        epsilon_combined(0) += m_Alpha * F.col(0).dot(Fdot.col(0));
        epsilon_combined(1) += m_Alpha * F.col(1).dot(Fdot.col(1));
        epsilon_combined(2) += m_Alpha * F.col(2).dot(Fdot.col(2));
        epsilon_combined(3) += m_Alpha * (F.col(1).dot(Fdot.col(2)) + Fdot.col(1).dot(F.col(2)));
        epsilon_combined(4) += m_Alpha * (F.col(0).dot(Fdot.col(2)) + Fdot.col(0).dot(F.col(2)));
        epsilon_combined(5) += m_Alpha * (F.col(0).dot(Fdot.col(1)) + Fdot.col(0).dot(F.col(1)));
    }

    // Get the stiffness tensor in 6x6 matrix form for the current layer and rotate it in the midsurface according
    // to the user specified angle.  Note that the matrix is reordered as well to match the Voigt notation used in
    // this element compared to what is used in ChMaterialShellANCF

    ChMatrixNM<double, 6, 6> D = m_layers[layer].GetMaterial()->Get_E_eps();
    RotateReorderStiffnessMatrix(D, m_layers[layer].Get_theta());

    ChVectorN<double, 6> sigmaPK2 = D * epsilon_combined;  // 2nd Piola Kirchhoff Stress tensor in Voigt notation

    ChMatrix33<> SPK2;
    SPK2(0, 0) = sigmaPK2(0);
    SPK2(1, 1) = sigmaPK2(1);
    SPK2(2, 2) = sigmaPK2(2);
    SPK2(1, 2) = sigmaPK2(3);
    SPK2(2, 1) = sigmaPK2(3);
    SPK2(0, 2) = sigmaPK2(4);
    SPK2(2, 0) = sigmaPK2(4);
    SPK2(0, 1) = sigmaPK2(5);
    SPK2(1, 0) = sigmaPK2(5);

    return SPK2;
}

// Get the von Mises stress value at the normalized **layer** coordinates (xi, eta, layer_zeta) at the current state
// of the element for the specified layer number (0 indexed) since the stress can be discontinuous at the layer
// boundary.  "layer_zeta" spans -1 to 1 from the bottom surface to the top surface

double ChElementShellANCF_3833_TR01::GetVonMissesStress(const double layer,
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

    // Calculate the Green-Lagrange strain tensor at the current point in Voigt notation
    ChVectorN<double, 6> epsilon_combined;
    epsilon_combined(0) = 0.5 * (F.col(0).dot(F.col(0)) - 1);
    epsilon_combined(1) = 0.5 * (F.col(1).dot(F.col(1)) - 1);
    epsilon_combined(2) = 0.5 * (F.col(2).dot(F.col(2)) - 1);
    epsilon_combined(3) = F.col(1).dot(F.col(2));
    epsilon_combined(4) = F.col(0).dot(F.col(2));
    epsilon_combined(5) = F.col(0).dot(F.col(1));

    if (m_damping_enabled) {
        Matrix3xN ebardot;  // Element coordinate time derivatives in matrix form
        CalcCoordDerivMatrix(ebardot);

        // Calculate the time derivative of the Deformation Gradient at the current point
        ChMatrixNMc<double, 3, 3> Fdot = ebardot * Sxi_D;

        // Calculate the time derivative of the Green-Lagrange strain tensor in Voigt notation
        // and combine it with epsilon assuming a Linear Kelvin-Voigt Viscoelastic material model
        epsilon_combined(0) += m_Alpha * F.col(0).dot(Fdot.col(0));
        epsilon_combined(1) += m_Alpha * F.col(1).dot(Fdot.col(1));
        epsilon_combined(2) += m_Alpha * F.col(2).dot(Fdot.col(2));
        epsilon_combined(3) += m_Alpha * (F.col(1).dot(Fdot.col(2)) + Fdot.col(1).dot(F.col(2)));
        epsilon_combined(4) += m_Alpha * (F.col(0).dot(Fdot.col(2)) + Fdot.col(0).dot(F.col(2)));
        epsilon_combined(5) += m_Alpha * (F.col(0).dot(Fdot.col(1)) + Fdot.col(0).dot(F.col(1)));
    }

    // Get the stiffness tensor in 6x6 matrix form for the current layer and rotate it in the midsurface according
    // to the user specified angle.  Note that the matrix is reordered as well to match the Voigt notation used in
    // this element compared to what is used in ChMaterialShellANCF

    ChMatrixNM<double, 6, 6> D = m_layers[layer].GetMaterial()->Get_E_eps();
    RotateReorderStiffnessMatrix(D, m_layers[layer].Get_theta());

    ChVectorN<double, 6> sigmaPK2 = D * epsilon_combined;  // 2nd Piola Kirchhoff Stress tensor in Voigt notation

    ChMatrixNM<double, 3, 3> SPK2;  // 2nd Piola Kirchhoff Stress tensor
    SPK2(0, 0) = sigmaPK2(0);
    SPK2(1, 1) = sigmaPK2(1);
    SPK2(2, 2) = sigmaPK2(2);
    SPK2(1, 2) = sigmaPK2(3);
    SPK2(2, 1) = sigmaPK2(3);
    SPK2(0, 2) = sigmaPK2(4);
    SPK2(2, 0) = sigmaPK2(4);
    SPK2(0, 1) = sigmaPK2(5);
    SPK2(1, 0) = sigmaPK2(5);

    // Convert from 2ndPK Stress to Cauchy Stress
    ChMatrix33<double> S = (F * SPK2 * F.transpose()) / F.determinant();
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

void ChElementShellANCF_3833_TR01::SetupInitial(ChSystem* system) {
    // Store the initial nodal coordinates. These values define the reference configuration of the element.
    CalcCoordMatrix(m_ebar0);
    CalcCoordVector(m_e0);

    // Compute and store the constant mass matrix and the matrix used to multiply the acceleration due to gravity to get
    // the generalized gravitational force vector for the element
    ComputeMassMatrixAndGravityForce();
}

// Fill the D vector with the current field values at the element nodes.

void ChElementShellANCF_3833_TR01::GetStateBlock(ChVectorDynamic<>& mD) {
    mD.segment(0, 3) = m_nodes[0]->GetPos().eigen();
    mD.segment(3, 3) = m_nodes[0]->GetD().eigen();
    mD.segment(6, 3) = m_nodes[0]->GetDD().eigen();
    mD.segment(9, 3) = m_nodes[1]->GetPos().eigen();
    mD.segment(12, 3) = m_nodes[1]->GetD().eigen();
    mD.segment(15, 3) = m_nodes[1]->GetDD().eigen();
    mD.segment(18, 3) = m_nodes[2]->GetPos().eigen();
    mD.segment(21, 3) = m_nodes[2]->GetD().eigen();
    mD.segment(24, 3) = m_nodes[2]->GetDD().eigen();
    mD.segment(27, 3) = m_nodes[3]->GetPos().eigen();
    mD.segment(30, 3) = m_nodes[3]->GetD().eigen();
    mD.segment(33, 3) = m_nodes[3]->GetDD().eigen();
    mD.segment(36, 3) = m_nodes[4]->GetPos().eigen();
    mD.segment(39, 3) = m_nodes[4]->GetD().eigen();
    mD.segment(42, 3) = m_nodes[4]->GetDD().eigen();
    mD.segment(45, 3) = m_nodes[5]->GetPos().eigen();
    mD.segment(48, 3) = m_nodes[5]->GetD().eigen();
    mD.segment(51, 3) = m_nodes[5]->GetDD().eigen();
    mD.segment(54, 3) = m_nodes[6]->GetPos().eigen();
    mD.segment(57, 3) = m_nodes[6]->GetD().eigen();
    mD.segment(60, 3) = m_nodes[6]->GetDD().eigen();
    mD.segment(63, 3) = m_nodes[7]->GetPos().eigen();
    mD.segment(66, 3) = m_nodes[7]->GetD().eigen();
    mD.segment(69, 3) = m_nodes[7]->GetDD().eigen();
}

// State update.

void ChElementShellANCF_3833_TR01::Update() {
    ChElementGeneric::Update();
}

// Return the mass matrix in full sparse form.

void ChElementShellANCF_3833_TR01::ComputeMmatrixGlobal(ChMatrixRef M) {
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

void ChElementShellANCF_3833_TR01::ComputeNodalMass() {
    m_nodes[0]->m_TotalMass += m_MassMatrix(0) + m_MassMatrix(3) + m_MassMatrix(6) + m_MassMatrix(9) +
                               m_MassMatrix(12) + m_MassMatrix(15) + m_MassMatrix(18) + m_MassMatrix(21);
    m_nodes[1]->m_TotalMass += m_MassMatrix(3) + m_MassMatrix(69) + m_MassMatrix(72) + m_MassMatrix(75) +
                               m_MassMatrix(78) + m_MassMatrix(81) + m_MassMatrix(84) + m_MassMatrix(87);
    m_nodes[2]->m_TotalMass += m_MassMatrix(6) + m_MassMatrix(72) + m_MassMatrix(129) + m_MassMatrix(132) +
                               m_MassMatrix(135) + m_MassMatrix(138) + m_MassMatrix(141) + m_MassMatrix(144);
    m_nodes[3]->m_TotalMass += m_MassMatrix(9) + m_MassMatrix(75) + m_MassMatrix(132) + m_MassMatrix(180) +
                               m_MassMatrix(183) + m_MassMatrix(186) + m_MassMatrix(189) + m_MassMatrix(192);
    m_nodes[4]->m_TotalMass += m_MassMatrix(12) + m_MassMatrix(78) + m_MassMatrix(135) + m_MassMatrix(183) +
                               m_MassMatrix(222) + m_MassMatrix(225) + m_MassMatrix(228) + m_MassMatrix(231);
    m_nodes[5]->m_TotalMass += m_MassMatrix(15) + m_MassMatrix(81) + m_MassMatrix(138) + m_MassMatrix(186) +
                               m_MassMatrix(225) + m_MassMatrix(255) + m_MassMatrix(258) + m_MassMatrix(261);
    m_nodes[6]->m_TotalMass += m_MassMatrix(18) + m_MassMatrix(84) + m_MassMatrix(141) + m_MassMatrix(189) +
                               m_MassMatrix(228) + m_MassMatrix(258) + m_MassMatrix(279) + m_MassMatrix(282);
    m_nodes[7]->m_TotalMass += m_MassMatrix(21) + m_MassMatrix(87) + m_MassMatrix(144) + m_MassMatrix(192) +
                               m_MassMatrix(231) + m_MassMatrix(261) + m_MassMatrix(282) + m_MassMatrix(294);
}

// Compute the generalized internal force vector for the current nodal coordinates and set the value in the Fi vector.

void ChElementShellANCF_3833_TR01::ComputeInternalForces(ChVectorDynamic<>& Fi) {
    assert(Fi.size() == 3 * NSF);

    Vector3N e;
    Vector3N edot;

    CalcCoordVector(e);
    CalcCoordDerivVector(edot);
    ComputeInternalForcesAtState(Fi, e, edot);
}

// Calculate the global matrix H as a linear combination of K, R, and M:
//   H = Mfactor * [M] + Kfactor * [K] + Rfactor * [R]

void ChElementShellANCF_3833_TR01::ComputeKRMmatricesGlobal(ChMatrixRef H,
                                                            double Kfactor,
                                                            double Rfactor,
                                                            double Mfactor) {
    assert((H.rows() == 3 * NSF) && (H.cols() == 3 * NSF));

    ChVectorDynamic<double> FiOrignal(3 * NSF);
    ChVectorDynamic<double> FiDelta(3 * NSF);
    Vector3N e;
    Vector3N edot;

    CalcCoordVector(e);
    CalcCoordDerivVector(edot);

    double delta = 1e-6;

    Matrix3Nx3N Jac;

    // Compute the Jacobian via numerical differentiation of the generalized internal force vector
    // Since the generalized force vector due to gravity is a constant, it doesn't affect this
    // Jacobian calculation
    ComputeInternalForcesAtState(FiOrignal, e, edot);
    for (unsigned int i = 0; i < e.size(); i++) {
        e(i) = e(i) + delta;
        ComputeInternalForcesAtState(FiDelta, e, edot);
        Jac.col(i).noalias() = -Kfactor / delta * (FiDelta - FiOrignal);
        e(i) = e(i) - delta;

        edot(i) = edot(i) + delta;
        ComputeInternalForcesAtState(FiDelta, e, edot);
        Jac.col(i).noalias() += -Rfactor / delta * (FiDelta - FiOrignal);
        edot(i) = edot(i) - delta;
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

// Compute the generalized force vector due to gravity using the efficient ANCF specific method
void ChElementShellANCF_3833_TR01::ComputeGravityForces(ChVectorDynamic<>& Fg, const ChVector<>& G_acc) {
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

void ChElementShellANCF_3833_TR01::EvaluateSectionFrame(const double xi,
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

void ChElementShellANCF_3833_TR01::EvaluateSectionPoint(const double xi, const double eta, ChVector<>& point) {
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, eta, 0, m_thicknessZ, m_midsurfoffset);

    Matrix3xN e_bar;
    CalcCoordMatrix(e_bar);

    // r = S*e written in compact form
    point = e_bar * Sxi_compact;
}

void ChElementShellANCF_3833_TR01::EvaluateSectionVelNorm(const double xi, const double eta, ChVector<>& Result) {
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

void ChElementShellANCF_3833_TR01::LoadableGetStateBlock_x(int block_offset, ChState& mD) {
    mD.segment(block_offset + 0, 3) = m_nodes[0]->GetPos().eigen();
    mD.segment(block_offset + 3, 3) = m_nodes[0]->GetD().eigen();
    mD.segment(block_offset + 6, 3) = m_nodes[0]->GetDD().eigen();

    mD.segment(block_offset + 9, 3) = m_nodes[1]->GetPos().eigen();
    mD.segment(block_offset + 12, 3) = m_nodes[1]->GetD().eigen();
    mD.segment(block_offset + 15, 3) = m_nodes[1]->GetDD().eigen();

    mD.segment(block_offset + 18, 3) = m_nodes[2]->GetPos().eigen();
    mD.segment(block_offset + 21, 3) = m_nodes[2]->GetD().eigen();
    mD.segment(block_offset + 24, 3) = m_nodes[2]->GetDD().eigen();

    mD.segment(block_offset + 27, 3) = m_nodes[3]->GetPos().eigen();
    mD.segment(block_offset + 30, 3) = m_nodes[3]->GetD().eigen();
    mD.segment(block_offset + 33, 3) = m_nodes[3]->GetDD().eigen();

    mD.segment(block_offset + 36, 3) = m_nodes[4]->GetPos().eigen();
    mD.segment(block_offset + 39, 3) = m_nodes[4]->GetD().eigen();
    mD.segment(block_offset + 42, 3) = m_nodes[4]->GetDD().eigen();

    mD.segment(block_offset + 45, 3) = m_nodes[5]->GetPos().eigen();
    mD.segment(block_offset + 48, 3) = m_nodes[5]->GetD().eigen();
    mD.segment(block_offset + 51, 3) = m_nodes[5]->GetDD().eigen();

    mD.segment(block_offset + 54, 3) = m_nodes[6]->GetPos().eigen();
    mD.segment(block_offset + 57, 3) = m_nodes[6]->GetD().eigen();
    mD.segment(block_offset + 60, 3) = m_nodes[6]->GetDD().eigen();

    mD.segment(block_offset + 63, 3) = m_nodes[7]->GetPos().eigen();
    mD.segment(block_offset + 66, 3) = m_nodes[7]->GetD().eigen();
    mD.segment(block_offset + 69, 3) = m_nodes[7]->GetDD().eigen();
}

// Gets all the DOFs packed in a single vector (velocity part).

void ChElementShellANCF_3833_TR01::LoadableGetStateBlock_w(int block_offset, ChStateDelta& mD) {
    mD.segment(block_offset + 0, 3) = m_nodes[0]->GetPos_dt().eigen();
    mD.segment(block_offset + 3, 3) = m_nodes[0]->GetD_dt().eigen();
    mD.segment(block_offset + 6, 3) = m_nodes[0]->GetDD_dt().eigen();

    mD.segment(block_offset + 9, 3) = m_nodes[1]->GetPos_dt().eigen();
    mD.segment(block_offset + 12, 3) = m_nodes[1]->GetD_dt().eigen();
    mD.segment(block_offset + 15, 3) = m_nodes[1]->GetDD_dt().eigen();

    mD.segment(block_offset + 18, 3) = m_nodes[2]->GetPos_dt().eigen();
    mD.segment(block_offset + 21, 3) = m_nodes[2]->GetD_dt().eigen();
    mD.segment(block_offset + 24, 3) = m_nodes[2]->GetDD_dt().eigen();

    mD.segment(block_offset + 27, 3) = m_nodes[3]->GetPos_dt().eigen();
    mD.segment(block_offset + 30, 3) = m_nodes[3]->GetD_dt().eigen();
    mD.segment(block_offset + 33, 3) = m_nodes[3]->GetDD_dt().eigen();

    mD.segment(block_offset + 36, 3) = m_nodes[4]->GetPos_dt().eigen();
    mD.segment(block_offset + 39, 3) = m_nodes[4]->GetD_dt().eigen();
    mD.segment(block_offset + 42, 3) = m_nodes[4]->GetDD_dt().eigen();

    mD.segment(block_offset + 45, 3) = m_nodes[5]->GetPos_dt().eigen();
    mD.segment(block_offset + 48, 3) = m_nodes[5]->GetD_dt().eigen();
    mD.segment(block_offset + 51, 3) = m_nodes[5]->GetDD_dt().eigen();

    mD.segment(block_offset + 54, 3) = m_nodes[6]->GetPos_dt().eigen();
    mD.segment(block_offset + 57, 3) = m_nodes[6]->GetD_dt().eigen();
    mD.segment(block_offset + 60, 3) = m_nodes[6]->GetDD_dt().eigen();

    mD.segment(block_offset + 63, 3) = m_nodes[7]->GetPos_dt().eigen();
    mD.segment(block_offset + 66, 3) = m_nodes[7]->GetD_dt().eigen();
    mD.segment(block_offset + 69, 3) = m_nodes[7]->GetDD_dt().eigen();
}

/// Increment all DOFs using a delta.

void ChElementShellANCF_3833_TR01::LoadableStateIncrement(const unsigned int off_x,
                                                          ChState& x_new,
                                                          const ChState& x,
                                                          const unsigned int off_v,
                                                          const ChStateDelta& Dv) {
    for (int i = 0; i < 8; i++) {
        this->m_nodes[i]->NodeIntStateIncrement(off_x + 9 * i, x_new, x, off_v + 9 * i, Dv);
    }
}

// Get the pointers to the contained ChVariables, appending to the mvars vector.

void ChElementShellANCF_3833_TR01::LoadableGetVariables(std::vector<ChVariables*>& mvars) {
    for (int i = 0; i < m_nodes.size(); ++i) {
        mvars.push_back(&m_nodes[i]->Variables());
        mvars.push_back(&m_nodes[i]->Variables_D());
        mvars.push_back(&m_nodes[i]->Variables_DD());
    }
}

// Evaluate N'*F, which is the projection of the applied point force and moment at the midsurface coordinates (xi,eta,0)
// This calculation takes a slightly different form for ANCF elements
// For this ANCF element, only the first 6 entries in F are used in the calculation.  The first three entries is
// the applied force in global coordinates and the second 3 entries is the applied moment in global space.

void ChElementShellANCF_3833_TR01::ComputeNF(
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

void ChElementShellANCF_3833_TR01::ComputeNF(
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

double ChElementShellANCF_3833_TR01::GetDensity() {
    double tot_density = 0;
    for (int kl = 0; kl < m_numLayers; kl++) {
        double rho = m_layers[kl].GetMaterial()->Get_rho();
        double layerthick = m_layers[kl].Get_thickness();
        tot_density += rho * layerthick;
    }
    return tot_density / m_thicknessZ;
}

// Calculate normal to the midsurface at coordinates (xi, eta).

ChVector<> ChElementShellANCF_3833_TR01::ComputeNormal(const double xi, const double eta) {
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

void ChElementShellANCF_3833_TR01::ComputeMassMatrixAndGravityForce() {
    // For this element, the mass matrix integrand is of order 9 in xi, 9 in eta, and 9 in zeta.
    // 5 GQ Points are needed in the xi, eta, and zeta directions for exact integration of the element's mass matrix,
    // even if the reference configuration is not straight. Since the major pieces of the generalized force due to
    // gravity can also be used to calculate the mass matrix, these calculations are performed at the same time.  Only
    // the matrix that scales the acceleration due to gravity is calculated at this time so that any changes to the
    // acceleration due to gravity in the system are correctly accounted for in the generalized internal force
    // calculation.

    ChQuadratureTables* GQTable = GetStaticGQTables();
    unsigned int GQ_idx_xi_eta = 4;  // 5 Point Gauss-Quadrature;
    unsigned int GQ_idx_zeta = 4;    // 5 Point Gauss-Quadrature;

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

// -----------------------------------------------------------------------------
// Elastic force calculation
// -----------------------------------------------------------------------------

void ChElementShellANCF_3833_TR01::ComputeInternalForcesAtState(ChVectorDynamic<>& Fi,
                                                                const Vector3N& e,
                                                                const Vector3N& edot) {
    // Set Qi to zero since the results from each GQ point will be added to this vector
    Vector3N Qi;
    Qi.setZero();

    // Straight & Normalized Internal Force Integrand is of order : 12 in xi, order : 12 in eta, and order : 4 in zeta.
    // This requires GQ 7 points along the xi & eta directions and 3 points along the zeta directions for "Full
    // Integration"

    ChQuadratureTables* GQTable = GetStaticGQTables();

    // Assume integration across the entire volume of the element (no splitting of the Poisson effect)
    for (size_t kl = 0; kl < m_numLayers; kl++) {
        double thickness = m_layers[kl].Get_thickness();
        double layer_midsurface_offset =
            -m_thicknessZ / 2 + m_layer_zoffsets[kl] + m_layers[kl].Get_thickness() / 2 + m_midsurfoffset;

        ChMatrixNM<double, 6, 6> D = m_layers[kl].GetMaterial()->Get_E_eps();
        RotateReorderStiffnessMatrix(D, m_layers[kl].Get_theta());

        for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[NP - 1].size(); it_xi++) {
            for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[NP - 1].size(); it_eta++) {
                for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[NT - 1].size(); it_zeta++) {
                    double GQ_weight = GQTable->Weight[NP - 1][it_xi] * GQTable->Weight[NP - 1][it_eta] *
                                       GQTable->Weight[NT - 1][it_zeta];
                    double xi = GQTable->Lroots[NP - 1][it_xi];
                    double eta = GQTable->Lroots[NP - 1][it_eta];
                    double zeta = GQTable->Lroots[NT - 1][it_zeta];
                    Vector3N QiPnt;
                    ComputeInternalForcesSingleGQPnt(QiPnt, xi, eta, zeta, thickness, layer_midsurface_offset, D, e,
                                                     edot);

                    Qi.noalias() += GQ_weight * QiPnt;
                }
            }
        }
    }

    Fi.noalias() = Qi;
}

void ChElementShellANCF_3833_TR01::ComputeInternalForcesSingleGQPnt(Vector3N& Qi,
                                                                    double xi,
                                                                    double eta,
                                                                    double zeta,
                                                                    double thickness,
                                                                    double zoffset,
                                                                    const ChMatrixNM<double, 6, 6>& D,
                                                                    const Vector3N& e,
                                                                    const Vector3N& edot) {
    // Calculate the Partial Derivative of the Normalized Shape function matrix with respect to xi at the current point
    Matrix3x3N Sxi_xi;
    Calc_Sxi_xi(Sxi_xi, xi, eta, zeta, thickness, zoffset);

    // Calculate the Partial Derivative of the Normalized Shape function matrix with respect to eta at the current point
    Matrix3x3N Sxi_eta;
    Calc_Sxi_eta(Sxi_eta, xi, eta, zeta, thickness, zoffset);

    // Calculate the Partial Derivative of the Normalized Shape function matrix with respect to zeta at the current
    // point
    Matrix3x3N Sxi_zeta;
    Calc_Sxi_zeta(Sxi_zeta, xi, eta, zeta, thickness, zoffset);

    // Calculate the Element Jacobian Matrix at the current point (xi, eta, zeta)
    ChMatrix33<double> J_0xi;
    J_0xi.row(0).noalias() = Sxi_xi * m_e0;
    J_0xi.row(1).noalias() = Sxi_eta * m_e0;
    J_0xi.row(2).noalias() = Sxi_zeta * m_e0;

    // Calculate the Inverse of the Element Jacobian Matrix at the current point (xi, eta, zeta)
    ChMatrix33<double> J_0xi_Inv = J_0xi.inverse();

    // Calculate the normalized shape function derivatives that have been modified by J_0xi_Inv to account for a
    // potentially non-straight reference configuration
    Matrix3x3N SxiF1 = J_0xi_Inv(0, 0) * Sxi_xi + J_0xi_Inv(1, 0) * Sxi_eta + J_0xi_Inv(2, 0) * Sxi_zeta;
    Matrix3x3N SxiF2 = J_0xi_Inv(0, 1) * Sxi_xi + J_0xi_Inv(1, 1) * Sxi_eta + J_0xi_Inv(2, 1) * Sxi_zeta;
    Matrix3x3N SxiF3 = J_0xi_Inv(0, 2) * Sxi_xi + J_0xi_Inv(1, 2) * Sxi_eta + J_0xi_Inv(2, 2) * Sxi_zeta;

    // Calculate the Green-Lagrange strain tensor in Voigt notation
    ChVectorN<double, 6> epsilon;
    epsilon(0) = 0.5 * (e.transpose() * SxiF1.transpose() * SxiF1 * e - 1);
    epsilon(1) = 0.5 * (e.transpose() * SxiF2.transpose() * SxiF2 * e - 1);
    epsilon(2) = 0.5 * (e.transpose() * SxiF3.transpose() * SxiF3 * e - 1);
    epsilon(3) = e.transpose() * SxiF2.transpose() * SxiF3 * e;
    epsilon(4) = e.transpose() * SxiF1.transpose() * SxiF3 * e;
    epsilon(5) = e.transpose() * SxiF1.transpose() * SxiF2 * e;

    // Calculate the partial derivative of the Green-Lagrange strain tensor in Voigt notation with respect to the
    // element position coordinates
    Matrix6x3N depsilon_de;
    depsilon_de.row(0).noalias() = e.transpose() * SxiF1.transpose() * SxiF1;
    depsilon_de.row(1).noalias() = e.transpose() * SxiF2.transpose() * SxiF2;
    depsilon_de.row(2).noalias() = e.transpose() * SxiF3.transpose() * SxiF3;
    depsilon_de.row(3).noalias() =
        e.transpose() * SxiF2.transpose() * SxiF3 + e.transpose() * SxiF3.transpose() * SxiF2;
    depsilon_de.row(4).noalias() =
        e.transpose() * SxiF1.transpose() * SxiF3 + e.transpose() * SxiF3.transpose() * SxiF1;
    depsilon_de.row(5).noalias() =
        e.transpose() * SxiF1.transpose() * SxiF2 + e.transpose() * SxiF2.transpose() * SxiF1;

    // Calculate the generalized internal force integrand for a Linear Kelvin-Voigt Viscoelastic material model
    Qi.noalias() = -depsilon_de.transpose() * D * (epsilon + m_Alpha * (depsilon_de * edot)) * J_0xi.determinant();
}

// -----------------------------------------------------------------------------
// Shape functions
// -----------------------------------------------------------------------------

// Nx1 Vector Form of the Normalized Shape Functions
// [s1; s2; s3; ...]

void ChElementShellANCF_3833_TR01::Calc_Sxi_compact(VectorN& Sxi_compact,
                                                    double xi,
                                                    double eta,
                                                    double zeta,
                                                    double thickness,
                                                    double zoffset) {
    Sxi_compact(0) = (-0.25) * (xi - 1) * (eta - 1) * (eta + xi + 1);
    Sxi_compact(1) =
        (0.125) * (xi - 1) * (eta - 1) * (eta + xi + 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_compact(2) = (-0.03125) * (xi - 1) * (eta - 1) * (eta + xi + 1) *
                     (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) *
                     (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_compact(3) = (0.25) * (xi + 1) * (eta - 1) * (eta - xi + 1);
    Sxi_compact(4) =
        (-0.125) * (xi + 1) * (eta - 1) * (eta - xi + 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_compact(5) = (0.03125) * (xi + 1) * (eta - 1) * (eta - xi + 1) *
                     (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) *
                     (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_compact(6) = (0.25) * (xi + 1) * (eta + 1) * (eta + xi - 1);
    Sxi_compact(7) =
        (-0.125) * (xi + 1) * (eta + 1) * (eta + xi - 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_compact(8) = (0.03125) * (xi + 1) * (eta + 1) * (eta + xi - 1) *
                     (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) *
                     (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_compact(9) = (-0.25) * (xi - 1) * (eta + 1) * (eta - xi - 1);
    Sxi_compact(10) =
        (0.125) * (xi - 1) * (eta + 1) * (eta - xi - 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_compact(11) = (-0.03125) * (xi - 1) * (eta + 1) * (eta - xi - 1) *
                      (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) *
                      (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_compact(12) = (0.5) * (xi - 1) * (xi + 1) * (eta - 1);
    Sxi_compact(13) =
        (-0.25) * (xi - 1) * (xi + 1) * (eta - 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_compact(14) = (0.0625) * (xi - 1) * (xi + 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) *
                      (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) * (eta - 1);
    Sxi_compact(15) = (-0.5) * (eta - 1) * (eta + 1) * (xi + 1);
    Sxi_compact(16) =
        (0.25) * (eta - 1) * (eta + 1) * (xi + 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_compact(17) = (-0.0625) * (eta - 1) * (eta + 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) *
                      (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) * (xi + 1);
    Sxi_compact(18) = (-0.5) * (xi - 1) * (xi + 1) * (eta + 1);
    Sxi_compact(19) =
        (0.25) * (xi - 1) * (xi + 1) * (eta + 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_compact(20) = (-0.0625) * (xi - 1) * (xi + 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) *
                      (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) * (eta + 1);
    Sxi_compact(21) = (0.5) * (eta - 1) * (eta + 1) * (xi - 1);
    Sxi_compact(22) =
        (-0.25) * (eta - 1) * (eta + 1) * (xi - 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_compact(23) = (0.0625) * (eta - 1) * (eta + 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) *
                      (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) * (xi - 1);
}

// Nx1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to xi
// [s1; s2; s3; ...]

void ChElementShellANCF_3833_TR01::Calc_Sxi_xi_compact(VectorN& Sxi_xi_compact,
                                                       double xi,
                                                       double eta,
                                                       double zeta,
                                                       double thickness,
                                                       double zoffset) {
    Sxi_xi_compact(0) = (-0.25) * (eta - 1) * (eta + 2 * xi);
    Sxi_xi_compact(1) =
        (0.125) * (eta - 1) * (eta + 2 * xi) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_xi_compact(2) = (-0.03125) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) *
                        (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) * (eta - 1) * (eta + 2 * xi);
    Sxi_xi_compact(3) = (0.25) * (eta - 1) * (eta - 2 * xi);
    Sxi_xi_compact(4) =
        (-0.125) * (eta - 1) * (eta - 2 * xi) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_xi_compact(5) = (0.03125) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) *
                        (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) * (eta - 1) * (eta - 2 * xi);
    Sxi_xi_compact(6) = (0.25) * (eta + 1) * (eta + 2 * xi);
    Sxi_xi_compact(7) =
        (-0.125) * (eta + 1) * (eta + 2 * xi) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_xi_compact(8) = (0.03125) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) *
                        (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) * (eta + 1) * (eta + 2 * xi);
    Sxi_xi_compact(9) = (-0.25) * (eta + 1) * (eta - 2 * xi);
    Sxi_xi_compact(10) =
        (0.125) * (eta + 1) * (eta - 2 * xi) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_xi_compact(11) = (-0.03125) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) *
                         (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) * (eta + 1) * (eta - 2 * xi);
    Sxi_xi_compact(12) = (xi) * (eta - 1);
    Sxi_xi_compact(13) = (-0.5) * (xi) * (eta - 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_xi_compact(14) = (0.125) * (xi) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) *
                         (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) * (eta - 1);
    Sxi_xi_compact(15) = (-0.5) * (eta - 1) * (eta + 1);
    Sxi_xi_compact(16) = (0.25) * (eta - 1) * (eta + 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_xi_compact(17) = (-0.0625) * (eta - 1) * (eta + 1) *
                         (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) *
                         (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_xi_compact(18) = (-1) * (xi) * (eta + 1);
    Sxi_xi_compact(19) = (0.5) * (xi) * (eta + 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_xi_compact(20) = (-0.125) * (xi) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) *
                         (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) * (eta + 1);
    Sxi_xi_compact(21) = (0.5) * (eta - 1) * (eta + 1);
    Sxi_xi_compact(22) = (-0.25) * (eta - 1) * (eta + 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_xi_compact(23) = (0.0625) * (eta - 1) * (eta + 1) *
                         (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) *
                         (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
}

// Nx1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to eta
// [s1; s2; s3; ...]

void ChElementShellANCF_3833_TR01::Calc_Sxi_eta_compact(VectorN& Sxi_eta_compact,
                                                        double xi,
                                                        double eta,
                                                        double zeta,
                                                        double thickness,
                                                        double zoffset) {
    Sxi_eta_compact(0) = (-0.25) * (xi - 1) * (2 * eta + xi);
    Sxi_eta_compact(1) =
        (0.125) * (xi - 1) * (2 * eta + xi) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_eta_compact(2) = (-0.03125) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) *
                         (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) * (xi - 1) * (2 * eta + xi);
    Sxi_eta_compact(3) = (0.25) * (xi + 1) * (2 * eta - xi);
    Sxi_eta_compact(4) =
        (-0.125) * (xi + 1) * (2 * eta - xi) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_eta_compact(5) = (0.03125) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) *
                         (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) * (xi + 1) * (2 * eta - xi);
    Sxi_eta_compact(6) = (0.25) * (xi + 1) * (2 * eta + xi);
    Sxi_eta_compact(7) =
        (-0.125) * (xi + 1) * (2 * eta + xi) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_eta_compact(8) = (0.03125) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) *
                         (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) * (xi + 1) * (2 * eta + xi);
    Sxi_eta_compact(9) = (-0.25) * (xi - 1) * (2 * eta - xi);
    Sxi_eta_compact(10) =
        (0.125) * (xi - 1) * (2 * eta - xi) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_eta_compact(11) = (-0.03125) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) *
                          (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) * (xi - 1) * (2 * eta - xi);
    Sxi_eta_compact(12) = (0.5) * (xi - 1) * (xi + 1);
    Sxi_eta_compact(13) = (-0.25) * (xi - 1) * (xi + 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_eta_compact(14) = (0.0625) * (xi - 1) * (xi + 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) *
                          (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_eta_compact(15) = (-1) * (eta) * (xi + 1);
    Sxi_eta_compact(16) = (0.5) * (eta) * (xi + 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_eta_compact(17) = (-0.125) * (eta) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) *
                          (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) * (xi + 1);
    Sxi_eta_compact(18) = (-0.5) * (xi - 1) * (xi + 1);
    Sxi_eta_compact(19) = (0.25) * (xi - 1) * (xi + 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_eta_compact(20) = (-0.0625) * (xi - 1) * (xi + 1) *
                          (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) *
                          (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_eta_compact(21) = (eta) * (xi - 1);
    Sxi_eta_compact(22) = (-0.5) * (eta) * (xi - 1) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_eta_compact(23) = (0.125) * (eta) * (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) *
                          (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta) * (xi - 1);
}

// Nx1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to zeta
// [s1; s2; s3; ...]

void ChElementShellANCF_3833_TR01::Calc_Sxi_zeta_compact(VectorN& Sxi_zeta_compact,
                                                         double xi,
                                                         double eta,
                                                         double zeta,
                                                         double thickness,
                                                         double zoffset) {
    Sxi_zeta_compact(0) = 0;
    Sxi_zeta_compact(1) = (-0.125) * (thickness) * (xi - 1) * (eta - 1) * (eta + xi + 1);
    Sxi_zeta_compact(2) = (0.0625) * (thickness) * (xi - 1) * (eta - 1) * (eta + xi + 1) *
                          (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_zeta_compact(3) = 0;
    Sxi_zeta_compact(4) = (0.125) * (thickness) * (xi + 1) * (eta - 1) * (eta - xi + 1);
    Sxi_zeta_compact(5) = (-0.0625) * (thickness) * (xi + 1) * (eta - 1) * (eta - xi + 1) *
                          (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_zeta_compact(6) = 0;
    Sxi_zeta_compact(7) = (0.125) * (thickness) * (xi + 1) * (eta + 1) * (eta + xi - 1);
    Sxi_zeta_compact(8) = (-0.0625) * (thickness) * (xi + 1) * (eta + 1) * (eta + xi - 1) *
                          (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_zeta_compact(9) = 0;
    Sxi_zeta_compact(10) = (-0.125) * (thickness) * (xi - 1) * (eta + 1) * (eta - xi - 1);
    Sxi_zeta_compact(11) = (0.0625) * (thickness) * (xi - 1) * (eta + 1) * (eta - xi - 1) *
                           (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_zeta_compact(12) = 0;
    Sxi_zeta_compact(13) = (0.25) * (thickness) * (xi - 1) * (xi + 1) * (eta - 1);
    Sxi_zeta_compact(14) = (-0.125) * (thickness) * (xi - 1) * (xi + 1) * (eta - 1) *
                           (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_zeta_compact(15) = 0;
    Sxi_zeta_compact(16) = (-0.25) * (thickness) * (eta - 1) * (eta + 1) * (xi + 1);
    Sxi_zeta_compact(17) = (0.125) * (thickness) * (eta - 1) * (eta + 1) * (xi + 1) *
                           (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_zeta_compact(18) = 0;
    Sxi_zeta_compact(19) = (-0.25) * (thickness) * (xi - 1) * (xi + 1) * (eta + 1);
    Sxi_zeta_compact(20) = (0.125) * (thickness) * (xi - 1) * (xi + 1) * (eta + 1) *
                           (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
    Sxi_zeta_compact(21) = 0;
    Sxi_zeta_compact(22) = (0.25) * (thickness) * (eta - 1) * (eta + 1) * (xi - 1);
    Sxi_zeta_compact(23) = (-0.125) * (thickness) * (eta - 1) * (eta + 1) * (xi - 1) *
                           (m_thicknessZ - 2 * zoffset - thickness - thickness * zeta);
}

// Nx3 compact form of the partial derivatives of Normalized Shape Functions with respect to xi, eta, and zeta by
// columns

void ChElementShellANCF_3833_TR01::Calc_Sxi_D(MatrixNx3c& Sxi_D,
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

// 3x3N Sparse Form of the partial derivatives of Normalized Shape Functions with respect to xi
// [s1*I_3x3, s2*I_3x3, s3*I_3x3, ...]
void ChElementShellANCF_3833_TR01::Calc_Sxi_xi(Matrix3x3N& Sxi_xi,
                                               double xi,
                                               double eta,
                                               double zeta,
                                               double thickness,
                                               double zoffset) {
    VectorN Sxi_xi_compact;
    Calc_Sxi_xi_compact(Sxi_xi_compact, xi, eta, zeta, thickness, zoffset);
    Sxi_xi.setZero();

    for (unsigned int s = 0; s < NSF; s++) {
        Sxi_xi(0, 0 + (3 * s)) = Sxi_xi_compact(s);
        Sxi_xi(1, 1 + (3 * s)) = Sxi_xi_compact(s);
        Sxi_xi(2, 2 + (3 * s)) = Sxi_xi_compact(s);
    }
}

// 3x3N Sparse Form of the partial derivatives of Normalized Shape Functions with respect to eta
// [s1*I_3x3, s2*I_3x3, s3*I_3x3, ...]
void ChElementShellANCF_3833_TR01::Calc_Sxi_eta(Matrix3x3N& Sxi_eta,
                                                double xi,
                                                double eta,
                                                double zeta,
                                                double thickness,
                                                double zoffset) {
    VectorN Sxi_eta_compact;
    Calc_Sxi_eta_compact(Sxi_eta_compact, xi, eta, zeta, thickness, zoffset);
    Sxi_eta.setZero();

    for (unsigned int s = 0; s < NSF; s++) {
        Sxi_eta(0, 0 + (3 * s)) = Sxi_eta_compact(s);
        Sxi_eta(1, 1 + (3 * s)) = Sxi_eta_compact(s);
        Sxi_eta(2, 2 + (3 * s)) = Sxi_eta_compact(s);
    }
}

// 3x3N Sparse Form of the partial derivatives of Normalized Shape Functions with respect to zeta
// [s1*I_3x3, s2*I_3x3, s3*I_3x3, ...]
void ChElementShellANCF_3833_TR01::Calc_Sxi_zeta(Matrix3x3N& Sxi_zeta,
                                                 double xi,
                                                 double eta,
                                                 double zeta,
                                                 double thickness,
                                                 double zoffset) {
    VectorN Sxi_zeta_compact;
    Calc_Sxi_zeta_compact(Sxi_zeta_compact, xi, eta, zeta, thickness, zoffset);
    Sxi_zeta.setZero();

    for (unsigned int s = 0; s < NSF; s++) {
        Sxi_zeta(0, 0 + (3 * s)) = Sxi_zeta_compact(s);
        Sxi_zeta(1, 1 + (3 * s)) = Sxi_zeta_compact(s);
        Sxi_zeta(2, 2 + (3 * s)) = Sxi_zeta_compact(s);
    }
}

// -----------------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------------

void ChElementShellANCF_3833_TR01::CalcCoordVector(Vector3N& e) {
    e.segment(0, 3) = m_nodes[0]->GetPos().eigen();
    e.segment(3, 3) = m_nodes[0]->GetD().eigen();
    e.segment(6, 3) = m_nodes[0]->GetDD().eigen();

    e.segment(9, 3) = m_nodes[1]->GetPos().eigen();
    e.segment(12, 3) = m_nodes[1]->GetD().eigen();
    e.segment(15, 3) = m_nodes[1]->GetDD().eigen();

    e.segment(18, 3) = m_nodes[2]->GetPos().eigen();
    e.segment(21, 3) = m_nodes[2]->GetD().eigen();
    e.segment(24, 3) = m_nodes[2]->GetDD().eigen();

    e.segment(27, 3) = m_nodes[3]->GetPos().eigen();
    e.segment(30, 3) = m_nodes[3]->GetD().eigen();
    e.segment(33, 3) = m_nodes[3]->GetDD().eigen();

    e.segment(36, 3) = m_nodes[4]->GetPos().eigen();
    e.segment(39, 3) = m_nodes[4]->GetD().eigen();
    e.segment(42, 3) = m_nodes[4]->GetDD().eigen();

    e.segment(45, 3) = m_nodes[5]->GetPos().eigen();
    e.segment(48, 3) = m_nodes[5]->GetD().eigen();
    e.segment(51, 3) = m_nodes[5]->GetDD().eigen();

    e.segment(54, 3) = m_nodes[6]->GetPos().eigen();
    e.segment(57, 3) = m_nodes[6]->GetD().eigen();
    e.segment(60, 3) = m_nodes[6]->GetDD().eigen();

    e.segment(63, 3) = m_nodes[7]->GetPos().eigen();
    e.segment(66, 3) = m_nodes[7]->GetD().eigen();
    e.segment(69, 3) = m_nodes[7]->GetDD().eigen();
}

void ChElementShellANCF_3833_TR01::CalcCoordMatrix(Matrix3xN& ebar) {
    ebar.col(0) = m_nodes[0]->GetPos().eigen();
    ebar.col(1) = m_nodes[0]->GetD().eigen();
    ebar.col(2) = m_nodes[0]->GetDD().eigen();

    ebar.col(3) = m_nodes[1]->GetPos().eigen();
    ebar.col(4) = m_nodes[1]->GetD().eigen();
    ebar.col(5) = m_nodes[1]->GetDD().eigen();

    ebar.col(6) = m_nodes[2]->GetPos().eigen();
    ebar.col(7) = m_nodes[2]->GetD().eigen();
    ebar.col(8) = m_nodes[2]->GetDD().eigen();

    ebar.col(9) = m_nodes[3]->GetPos().eigen();
    ebar.col(10) = m_nodes[3]->GetD().eigen();
    ebar.col(11) = m_nodes[3]->GetDD().eigen();

    ebar.col(12) = m_nodes[4]->GetPos().eigen();
    ebar.col(13) = m_nodes[4]->GetD().eigen();
    ebar.col(14) = m_nodes[4]->GetDD().eigen();

    ebar.col(15) = m_nodes[5]->GetPos().eigen();
    ebar.col(16) = m_nodes[5]->GetD().eigen();
    ebar.col(17) = m_nodes[5]->GetDD().eigen();

    ebar.col(18) = m_nodes[6]->GetPos().eigen();
    ebar.col(19) = m_nodes[6]->GetD().eigen();
    ebar.col(20) = m_nodes[6]->GetDD().eigen();

    ebar.col(21) = m_nodes[7]->GetPos().eigen();
    ebar.col(22) = m_nodes[7]->GetD().eigen();
    ebar.col(23) = m_nodes[7]->GetDD().eigen();
}

void ChElementShellANCF_3833_TR01::CalcCoordDerivVector(Vector3N& edot) {
    edot.segment(0, 3) = m_nodes[0]->GetPos_dt().eigen();
    edot.segment(3, 3) = m_nodes[0]->GetD_dt().eigen();
    edot.segment(6, 3) = m_nodes[0]->GetDD_dt().eigen();

    edot.segment(9, 3) = m_nodes[1]->GetPos_dt().eigen();
    edot.segment(12, 3) = m_nodes[1]->GetD_dt().eigen();
    edot.segment(15, 3) = m_nodes[1]->GetDD_dt().eigen();

    edot.segment(18, 3) = m_nodes[2]->GetPos_dt().eigen();
    edot.segment(21, 3) = m_nodes[2]->GetD_dt().eigen();
    edot.segment(24, 3) = m_nodes[2]->GetDD_dt().eigen();

    edot.segment(27, 3) = m_nodes[3]->GetPos_dt().eigen();
    edot.segment(30, 3) = m_nodes[3]->GetD_dt().eigen();
    edot.segment(33, 3) = m_nodes[3]->GetDD_dt().eigen();

    edot.segment(36, 3) = m_nodes[4]->GetPos_dt().eigen();
    edot.segment(39, 3) = m_nodes[4]->GetD_dt().eigen();
    edot.segment(42, 3) = m_nodes[4]->GetDD_dt().eigen();

    edot.segment(45, 3) = m_nodes[5]->GetPos_dt().eigen();
    edot.segment(48, 3) = m_nodes[5]->GetD_dt().eigen();
    edot.segment(51, 3) = m_nodes[5]->GetDD_dt().eigen();

    edot.segment(54, 3) = m_nodes[6]->GetPos_dt().eigen();
    edot.segment(57, 3) = m_nodes[6]->GetD_dt().eigen();
    edot.segment(60, 3) = m_nodes[6]->GetDD_dt().eigen();

    edot.segment(63, 3) = m_nodes[7]->GetPos_dt().eigen();
    edot.segment(66, 3) = m_nodes[7]->GetD_dt().eigen();
    edot.segment(69, 3) = m_nodes[7]->GetDD_dt().eigen();
}

void ChElementShellANCF_3833_TR01::CalcCoordDerivMatrix(Matrix3xN& ebardot) {
    ebardot.col(0) = m_nodes[0]->GetPos_dt().eigen();
    ebardot.col(1) = m_nodes[0]->GetD_dt().eigen();
    ebardot.col(2) = m_nodes[0]->GetDD_dt().eigen();

    ebardot.col(3) = m_nodes[1]->GetPos_dt().eigen();
    ebardot.col(4) = m_nodes[1]->GetD_dt().eigen();
    ebardot.col(5) = m_nodes[1]->GetDD_dt().eigen();

    ebardot.col(6) = m_nodes[2]->GetPos_dt().eigen();
    ebardot.col(7) = m_nodes[2]->GetD_dt().eigen();
    ebardot.col(8) = m_nodes[2]->GetDD_dt().eigen();

    ebardot.col(9) = m_nodes[3]->GetPos_dt().eigen();
    ebardot.col(10) = m_nodes[3]->GetD_dt().eigen();
    ebardot.col(11) = m_nodes[3]->GetDD_dt().eigen();

    ebardot.col(12) = m_nodes[4]->GetPos_dt().eigen();
    ebardot.col(13) = m_nodes[4]->GetD_dt().eigen();
    ebardot.col(14) = m_nodes[4]->GetDD_dt().eigen();

    ebardot.col(15) = m_nodes[5]->GetPos_dt().eigen();
    ebardot.col(16) = m_nodes[5]->GetD_dt().eigen();
    ebardot.col(17) = m_nodes[5]->GetDD_dt().eigen();

    ebardot.col(18) = m_nodes[6]->GetPos_dt().eigen();
    ebardot.col(19) = m_nodes[6]->GetD_dt().eigen();
    ebardot.col(20) = m_nodes[6]->GetDD_dt().eigen();

    ebardot.col(21) = m_nodes[7]->GetPos_dt().eigen();
    ebardot.col(22) = m_nodes[7]->GetD_dt().eigen();
    ebardot.col(23) = m_nodes[7]->GetDD_dt().eigen();
}

void ChElementShellANCF_3833_TR01::CalcCombinedCoordMatrix(Matrix6xN& ebar_ebardot) {
    ebar_ebardot.block<3, 1>(0, 0) = m_nodes[0]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 0) = m_nodes[0]->GetPos_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 1) = m_nodes[0]->GetD().eigen();
    ebar_ebardot.block<3, 1>(3, 1) = m_nodes[0]->GetD_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 2) = m_nodes[0]->GetDD().eigen();
    ebar_ebardot.block<3, 1>(3, 2) = m_nodes[0]->GetDD_dt().eigen();

    ebar_ebardot.block<3, 1>(0, 3) = m_nodes[1]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 3) = m_nodes[1]->GetPos_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 4) = m_nodes[1]->GetD().eigen();
    ebar_ebardot.block<3, 1>(3, 4) = m_nodes[1]->GetD_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 5) = m_nodes[1]->GetDD().eigen();
    ebar_ebardot.block<3, 1>(3, 5) = m_nodes[1]->GetDD_dt().eigen();

    ebar_ebardot.block<3, 1>(0, 6) = m_nodes[2]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 6) = m_nodes[2]->GetPos_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 7) = m_nodes[2]->GetD().eigen();
    ebar_ebardot.block<3, 1>(3, 7) = m_nodes[2]->GetD_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 8) = m_nodes[2]->GetDD().eigen();
    ebar_ebardot.block<3, 1>(3, 8) = m_nodes[2]->GetDD_dt().eigen();

    ebar_ebardot.block<3, 1>(0, 9) = m_nodes[3]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 9) = m_nodes[3]->GetPos_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 10) = m_nodes[3]->GetD().eigen();
    ebar_ebardot.block<3, 1>(3, 10) = m_nodes[3]->GetD_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 11) = m_nodes[3]->GetDD().eigen();
    ebar_ebardot.block<3, 1>(3, 11) = m_nodes[3]->GetDD_dt().eigen();

    ebar_ebardot.block<3, 1>(0, 12) = m_nodes[4]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 12) = m_nodes[4]->GetPos_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 13) = m_nodes[4]->GetD().eigen();
    ebar_ebardot.block<3, 1>(3, 13) = m_nodes[4]->GetD_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 14) = m_nodes[4]->GetDD().eigen();
    ebar_ebardot.block<3, 1>(3, 14) = m_nodes[4]->GetDD_dt().eigen();

    ebar_ebardot.block<3, 1>(0, 15) = m_nodes[5]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 15) = m_nodes[5]->GetPos_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 16) = m_nodes[5]->GetD().eigen();
    ebar_ebardot.block<3, 1>(3, 16) = m_nodes[5]->GetD_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 17) = m_nodes[5]->GetDD().eigen();
    ebar_ebardot.block<3, 1>(3, 17) = m_nodes[5]->GetDD_dt().eigen();

    ebar_ebardot.block<3, 1>(0, 18) = m_nodes[6]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 18) = m_nodes[6]->GetPos_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 19) = m_nodes[6]->GetD().eigen();
    ebar_ebardot.block<3, 1>(3, 19) = m_nodes[6]->GetD_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 20) = m_nodes[6]->GetDD().eigen();
    ebar_ebardot.block<3, 1>(3, 20) = m_nodes[6]->GetDD_dt().eigen();

    ebar_ebardot.block<3, 1>(0, 21) = m_nodes[7]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 21) = m_nodes[7]->GetPos_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 22) = m_nodes[7]->GetD().eigen();
    ebar_ebardot.block<3, 1>(3, 22) = m_nodes[7]->GetD_dt().eigen();
    ebar_ebardot.block<3, 1>(0, 23) = m_nodes[7]->GetDD().eigen();
    ebar_ebardot.block<3, 1>(3, 23) = m_nodes[7]->GetDD_dt().eigen();
}

// Calculate the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element

void ChElementShellANCF_3833_TR01::Calc_J_0xi(ChMatrix33<double>& J_0xi,
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

double ChElementShellANCF_3833_TR01::Calc_det_J_0xi(double xi,
                                                    double eta,
                                                    double zeta,
                                                    double thickness,
                                                    double zoffset) {
    ChMatrix33<double> J_0xi;
    Calc_J_0xi(J_0xi, xi, eta, zeta, thickness, zoffset);

    return (J_0xi.determinant());
}

void ChElementShellANCF_3833_TR01::RotateReorderStiffnessMatrix(ChMatrixNM<double, 6, 6>& D, double theta) {
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
ChQuadratureTables static_tables_3833_TR01(1, CH_QUADRATURE_STATIC_TABLES);
//#endif // !CH_QUADRATURE_STATIC_TABLES

ChQuadratureTables* ChElementShellANCF_3833_TR01::GetStaticGQTables() {
    return &static_tables_3833_TR01;
}

////////////////////////////////////////////////////////////////

// ============================================================================
// Implementation of ChElementShellANCF_3833_TR01::Layer methods
// ============================================================================

// Private constructor (a layer can be created only by adding it to an element)

ChElementShellANCF_3833_TR01::Layer::Layer(double thickness,
                                           double theta,
                                           std::shared_ptr<ChMaterialShellANCF> material)
    : m_thickness(thickness), m_theta(theta), m_material(material) {}

}  // namespace fea
}  // namespace chrono

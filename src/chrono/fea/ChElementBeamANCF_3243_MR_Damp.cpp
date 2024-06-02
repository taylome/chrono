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
// Fully Parameterized ANCF beam element with 2 nodes (24DOF). A Description of this element can be found in: Johannes
// Gerstmayr and Ahmed A Shabana. Efficient integration of the elastic forces and thin three - dimensional beam elements
// in the absolute nodal coordinate formulation. In Proceeding of Multibody Dynamics ECCOMAS Thematic Conference,
// Madrid, Spain, 2005.
// =============================================================================
//
// =============================================================================
// Two term Mooney-Rivlin Hyperelastic Material Law with penalty term for incompressibility with the option for a single
// coefficient nonlinear KV Damping
// =============================================================================
// A description of the material law and the selective reduced integration technique can be found in: Orzechowski, G., &
// Fraczek, J. (2015). Nearly incompressible nonlinear material models in the large deformation analysis of beams using
// ANCF. Nonlinear Dynamics, 82(1), 451-464.
//
// A description of the damping law can be found in: Kubler, L., Eberhard, P., & Geisler, J. (2003). Flexible multibody
// systems with large deformations and nonlinear structural damping using absolute nodal coordinates. Nonlinear
// Dynamics, 34(1), 31-52.
// =============================================================================

#include "chrono/fea/ChElementBeamANCF_3243_MR_Damp.h"
#include "chrono/physics/ChSystem.h"

namespace chrono {
namespace fea {

// ------------------------------------------------------------------------------
// Constructor
// ------------------------------------------------------------------------------

ChElementBeamANCF_3243_MR_Damp::ChElementBeamANCF_3243_MR_Damp()
    : m_lenX(0), m_thicknessY(0), m_thicknessZ(0) {
    m_nodes.resize(2);
}

// ------------------------------------------------------------------------------
// Set element nodes
// ------------------------------------------------------------------------------

void ChElementBeamANCF_3243_MR_Damp::SetNodes(std::shared_ptr<ChNodeFEAxyzDDD> nodeA,
                                           std::shared_ptr<ChNodeFEAxyzDDD> nodeB) {
    assert(nodeA);
    assert(nodeB);

    m_nodes[0] = nodeA;
    m_nodes[1] = nodeB;

    std::vector<ChVariables*> mvars;
    mvars.push_back(&m_nodes[0]->Variables());
    mvars.push_back(&m_nodes[0]->VariablesSlope1());
    mvars.push_back(&m_nodes[0]->VariablesSlope2());
    mvars.push_back(&m_nodes[0]->VariablesSlope3());
    mvars.push_back(&m_nodes[1]->Variables());
    mvars.push_back(&m_nodes[1]->VariablesSlope1());
    mvars.push_back(&m_nodes[1]->VariablesSlope2());
    mvars.push_back(&m_nodes[1]->VariablesSlope3());

    Kmatr.SetVariables(mvars);

    // Initial positions and slopes of the element nodes
    // These values define the reference configuration of the element
    CalcCoordMatrix(m_ebar0);
}

// -----------------------------------------------------------------------------
// Element Settings
// -----------------------------------------------------------------------------

// Specify the element dimensions (in the undeformed state - which is different than the reference configuration and it
// is a state the element potentially is never in).

void ChElementBeamANCF_3243_MR_Damp::SetDimensions(double lenX, double thicknessY, double thicknessZ) {
    m_lenX = lenX;
    m_thicknessY = thicknessY;
    m_thicknessZ = thicknessZ;
}

// Specify the element material.

void ChElementBeamANCF_3243_MR_Damp::SetMaterial(std::shared_ptr<ChMaterialBeamANCF_MR> beam_mat) {
    m_material = beam_mat;
}

// -----------------------------------------------------------------------------
// Evaluate Strains and Stresses
// -----------------------------------------------------------------------------
// These functions are designed for single function calls.  If these values are needed at the same points in the element
// through out the simulation, then the adjusted normalized shape function derivative matrix (Sxi_D) for each query
// point should be cached and saved to increase the execution speed
// -----------------------------------------------------------------------------

// Get the Green-Lagrange strain tensor at the normalized element coordinates (xi, eta, zeta) [-1...1]

ChMatrix33d ChElementBeamANCF_3243_MR_Damp::GetGreenLagrangeStrain(const double xi, const double eta, const double zeta) {
    MatrixNx3c Sxi_D;  // Matrix of normalized shape function derivatives
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    ChMatrix33d J_0xi;  // Element Jacobian between the reference configuration and normalized configuration
    J_0xi.noalias() = m_ebar0 * Sxi_D;

    Sxi_D = Sxi_D * J_0xi.inverse();  // Adjust the shape function derivative matrix to account for the potentially
                                      // distorted reference configuration

    Matrix3xN e_bar;  // Element coordinates in matrix form
    CalcCoordMatrix(e_bar);

    // Calculate the Deformation Gradient at the current point
    ChMatrixNM_col<double, 3, 3> F = e_bar * Sxi_D;

    ChMatrix33d I3x3;
    I3x3.setIdentity();
    return 0.5 * (F.transpose() * F - I3x3);
}

// Get the 2nd Piola-Kirchoff stress tensor at the normalized element coordinates (xi, eta, zeta) [-1...1] at the
// current state of the element.

ChMatrix33d ChElementBeamANCF_3243_MR_Damp::GetPK2Stress(const double xi, const double eta, const double zeta) {
    MatrixNx3c Sxi_D;  // Matrix of normalized shape function derivatives
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    ChMatrix33d J_0xi;  // Element Jacobian between the reference configuration and normalized configuration
    J_0xi.noalias() = m_ebar0 * Sxi_D;

    Sxi_D = Sxi_D * J_0xi.inverse();  // Adjust the shape function derivative matrix to account for the potentially
                                      // distorted reference configuration

    Matrix3xN e_bar;  // Element coordinates in matrix form
    CalcCoordMatrix(e_bar);

    // Calculate the Deformation Gradient at the current point
    ChMatrixNM_col<double, 3, 3> F = e_bar * Sxi_D;

    // Calculate the contribution to the 2nd Piola-Kirchoff stress tensor from the Mooney-Rivlin material law
     // Formula from: Cheng, J., & Zhang, L. T. (2018). A general approach to derive stress and elasticity tensors for hyperelastic isotropic and anisotropic biomaterials. International journal of computational methods, 15(04), 1850028.
     // https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6377211/
    double J = F.determinant();
    ChMatrix33d C = F.transpose()*F;
    ChMatrix33d Csquare = C * C;
    ChMatrix33d Cbar = std::pow(J, -2.0 / 3.0)*C;
    ChMatrix33d I3x3;
    I3x3.setIdentity();
    double J_m23 = std::pow(J, -2.0 / 3.0);
    double I1 = C.trace();
    double I1bar = std::pow(J, -2.0 / 3.0)*I1;
    double I2 = 0.5*(I1*I1 - Csquare.trace());
    double I2bar = std::pow(J, -4.0 / 3.0)*I2;
    double mu1 = 2 * GetMaterial()->Get_c10();
    double mu2 = 2 * GetMaterial()->Get_c01();

    ChMatrix33d SPK2 =
        (std::pow(J, 1.0 / 3.0) * (J - 1) * GetMaterial()->Get_k() - J_m23 / 3.0 * (mu1 * I1bar + 2 * mu2 * I2bar)) *
        Cbar.inverse() +
        J_m23 * (mu1 + mu2 * I1bar) * I3x3 - J_m23 * mu2 * Cbar;


    // Calculate the contribution to the 2nd Piola-Kirchoff stress tensor from the single coefficient nonlinear Kelvin-Voigt Viscoelastic material law

    Matrix3xN edot_bar;  // Element coordinates in matrix form
    CalcCoordDtMatrix(edot_bar);

    // Calculate the time derivative of the Deformation Gradient at the current point
    ChMatrixNM_col<double, 3, 3> Fdot = edot_bar * Sxi_D;

    // Calculate the time derivative of the Greens-Lagrange strain tensor at the current point
    ChMatrix33d Edot = 0.5 * (F.transpose()*Fdot + Fdot.transpose()*F);
    ChMatrix33d CInv = C.inverse();
    SPK2.noalias() += J * GetMaterial()->Get_mu()*CInv*Edot*CInv;

    return SPK2;
}

// Get the von Mises stress value at the normalized element coordinates (xi, eta, zeta) [-1...1] at the current
// state of the element.

double ChElementBeamANCF_3243_MR_Damp::GetVonMissesStress(const double xi, const double eta, const double zeta) {
    MatrixNx3c Sxi_D;  // Matrix of normalized shape function derivatives
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    ChMatrix33d J_0xi;  // Element Jacobian between the reference configuration and normalized configuration
    J_0xi.noalias() = m_ebar0 * Sxi_D;

    Sxi_D = Sxi_D * J_0xi.inverse();  // Adjust the shape function derivative matrix to account for the potentially
                                      // distorted reference configuration

    Matrix3xN ebar;  // Element coordinates in matrix form
    CalcCoordMatrix(ebar);

    // Calculate the Deformation Gradient at the current point
    ChMatrixNM_col<double, 3, 3> F = ebar * Sxi_D;

    // Calculate the contribution to the 2nd Piola-Kirchoff stress tensor from the Mooney-Rivlin material law
    // Formula from: Cheng, J., & Zhang, L. T. (2018). A general approach to derive stress and elasticity tensors for hyperelastic isotropic and anisotropic biomaterials. International journal of computational methods, 15(04), 1850028.
    // https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6377211/
    double J = F.determinant();
    ChMatrix33d C = F.transpose()*F;
    ChMatrix33d Csquare = C * C;
    ChMatrix33d Cbar = std::pow(J, -2.0 / 3.0)*C;
    ChMatrix33d I3x3;
    I3x3.setIdentity();
    double J_m23 = std::pow(J, -2.0 / 3.0);
    double I1 = C.trace();
    double I1bar = std::pow(J, -2.0 / 3.0)*I1;
    double I2 = 0.5*(I1*I1 - Csquare.trace());
    double I2bar = std::pow(J, -4.0 / 3.0)*I2;
    double mu1 = 2 * GetMaterial()->Get_c10();
    double mu2 = 2 * GetMaterial()->Get_c01();

    ChMatrix33d SPK2 =
        (std::pow(J, 1.0 / 3.0) * (J - 1) * GetMaterial()->Get_k() - J_m23 / 3.0 * (mu1 * I1bar + 2 * mu2 * I2bar)) *
        Cbar.inverse() +
        J_m23 * (mu1 + mu2 * I1bar) * I3x3 - J_m23 * mu2 * Cbar;


    // Calculate the contribution to the 2nd Piola-Kirchoff stress tensor from the single coefficient nonlinear Kelvin-Voigt Viscoelastic material law

    Matrix3xN edot_bar;  // Element coordinates in matrix form
    CalcCoordDtMatrix(edot_bar);

    // Calculate the time derivative of the Deformation Gradient at the current point
    ChMatrixNM_col<double, 3, 3> Fdot = edot_bar * Sxi_D;

    // Calculate the time derivative of the Greens-Lagrange strain tensor at the current point
    ChMatrix33d Edot = 0.5 * (F.transpose()*Fdot + Fdot.transpose()*F);
    ChMatrix33d CInv = C.inverse();
    SPK2.noalias() += J * GetMaterial()->Get_mu()*CInv*Edot*CInv;


    // Convert from 2ndPK Stress to Cauchy Stress
    ChMatrix33d S = (F * SPK2 * F.transpose()) / J;
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

void ChElementBeamANCF_3243_MR_Damp::SetupInitial(ChSystem* system) {
    // Store the initial nodal coordinates. These values define the reference configuration of the element.
    CalcCoordMatrix(m_ebar0);

    // Compute and store the constant mass matrix and the matrix used to multiply the acceleration due to gravity to get
    // the generalized gravitational force vector for the element
    ComputeMassMatrixAndGravityForce();

    // Compute Pre-computed matrices and vectors for the generalized internal force calcualtions
    PrecomputeInternalForceMatricesWeights();
}

// Fill the D vector with the current field values at the element nodes.

void ChElementBeamANCF_3243_MR_Damp::GetStateBlock(ChVectorDynamic<>& mD) {
    mD.segment(0, 3) = m_nodes[0]->GetPos().eigen();
    mD.segment(3, 3) = m_nodes[0]->GetSlope1().eigen();
    mD.segment(6, 3) = m_nodes[0]->GetSlope2().eigen();
    mD.segment(9, 3) = m_nodes[0]->GetSlope3().eigen();

    mD.segment(12, 3) = m_nodes[1]->GetPos().eigen();
    mD.segment(15, 3) = m_nodes[1]->GetSlope1().eigen();
    mD.segment(18, 3) = m_nodes[1]->GetSlope2().eigen();
    mD.segment(21, 3) = m_nodes[1]->GetSlope3().eigen();
}

// State update.

void ChElementBeamANCF_3243_MR_Damp::Update() {
    ChElementGeneric::Update();
}

// Return the mass matrix in full sparse form.

void ChElementBeamANCF_3243_MR_Damp::ComputeMmatrixGlobal(ChMatrixRef M) {
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

void ChElementBeamANCF_3243_MR_Damp::ComputeNodalMass() {
    m_nodes[0]->m_TotalMass += m_MassMatrix(0) + m_MassMatrix(4);
    m_nodes[1]->m_TotalMass += m_MassMatrix(4) + m_MassMatrix(26);
}

// Compute the generalized internal force vector for the current nodal coordinates and set the value in the Fi vector.

void ChElementBeamANCF_3243_MR_Damp::ComputeInternalForces(ChVectorDynamic<>& Fi) {

    //if (GetMaterial()->Get_mu() != 0) {
    ComputeInternalForceDamping(Fi);
    //}
    //else {
    //    ComputeInternalForceNoDamping(Fi);
    //}
}

// Calculate the global matrix H as a linear combination of K, R, and M:
//   H = Mfactor * [M] + Kfactor * [K] + Rfactor * [R]

void ChElementBeamANCF_3243_MR_Damp::ComputeKRMmatricesGlobal(ChMatrixRef H,
                                                           double Kfactor,
                                                           double Rfactor,
                                                           double Mfactor) {

    //if (GetMaterial()->Get_mu() != 0) {
    ComputeInternalJacobianDamping(H, Kfactor, Rfactor, Mfactor);
    //}
    //else {
    //ComputeInternalJacobianNoDamping(H, Kfactor, Mfactor);
    //}
}

// Compute the generalized force vector due to gravity using the efficient ANCF specific method
void ChElementBeamANCF_3243_MR_Damp::ComputeGravityForces(ChVectorDynamic<>& Fg, const ChVector3d& G_acc) {
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
// Interface to ChElementBeam base class (and similar methods)
// -----------------------------------------------------------------------------

void ChElementBeamANCF_3243_MR_Damp::EvaluateSectionFrame(const double xi, ChVector3d& point, ChQuaternion<>& rot) {
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, 0, 0);
    VectorN Sxi_xi_compact;
    Calc_Sxi_xi_compact(Sxi_xi_compact, xi, 0, 0);
    VectorN Sxi_eta_compact;
    Calc_Sxi_eta_compact(Sxi_eta_compact, xi, 0, 0);

    Matrix3xN e_bar;
    CalcCoordMatrix(e_bar);

    // r = S*e written in compact form
    point = e_bar * Sxi_compact;

    // Since ANCF does not use rotations, calculate an approximate
    // rotation based off the position vector gradients
    ChVector3d BeamAxisTangent = e_bar * Sxi_xi_compact * 2 / m_lenX;
    ChVector3d CrossSectionY = e_bar * Sxi_eta_compact * 2 / m_thicknessY;

    // Since the position vector gradients are not in general orthogonal,
    // set the Dx direction tangent to the beam axis and
    // compute the Dy and Dz directions by using a
    // Gram-Schmidt orthonormalization, guided by the cross section Y direction
    ChMatrix33d msect;
    msect.SetFromAxisX(BeamAxisTangent, CrossSectionY);

    rot = msect.GetQuaternion();
}

void ChElementBeamANCF_3243_MR_Damp::EvaluateSectionPoint(const double xi, ChVector3d& point) {
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, 0, 0);

    Matrix3xN e_bar;
    CalcCoordMatrix(e_bar);

    // r = S*e written in compact form
    point = e_bar * Sxi_compact;
}

void ChElementBeamANCF_3243_MR_Damp::EvaluateSectionVel(const double xi, ChVector3d& Result) {
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, 0, 0);

    Matrix3xN e_bardot;
    CalcCoordDtMatrix(e_bardot);

    // rdot = S*edot written in compact form
    Result = e_bardot * Sxi_compact;
}

// -----------------------------------------------------------------------------
// Functions for ChLoadable interface
// -----------------------------------------------------------------------------

// Gets all the DOFs packed in a single vector (position part).

void ChElementBeamANCF_3243_MR_Damp::LoadableGetStateBlockPosLevel(int block_offset, ChState& mD) {
    mD.segment(block_offset + 0, 3) = m_nodes[0]->GetPos().eigen();
    mD.segment(block_offset + 3, 3) = m_nodes[0]->GetSlope1().eigen();
    mD.segment(block_offset + 6, 3) = m_nodes[0]->GetSlope2().eigen();
    mD.segment(block_offset + 9, 3) = m_nodes[0]->GetSlope3().eigen();

    mD.segment(block_offset + 12, 3) = m_nodes[1]->GetPos().eigen();
    mD.segment(block_offset + 15, 3) = m_nodes[1]->GetSlope1().eigen();
    mD.segment(block_offset + 18, 3) = m_nodes[1]->GetSlope2().eigen();
    mD.segment(block_offset + 21, 3) = m_nodes[1]->GetSlope3().eigen();
}

// Gets all the DOFs packed in a single vector (velocity part).

void ChElementBeamANCF_3243_MR_Damp::LoadableGetStateBlockVelLevel(int block_offset, ChStateDelta& mD) {
    mD.segment(block_offset + 0, 3) = m_nodes[0]->GetPosDt().eigen();
    mD.segment(block_offset + 3, 3) = m_nodes[0]->GetSlope1Dt().eigen();
    mD.segment(block_offset + 6, 3) = m_nodes[0]->GetSlope2Dt().eigen();
    mD.segment(block_offset + 9, 3) = m_nodes[0]->GetSlope3Dt().eigen();

    mD.segment(block_offset + 12, 3) = m_nodes[1]->GetPosDt().eigen();
    mD.segment(block_offset + 15, 3) = m_nodes[1]->GetSlope1Dt().eigen();
    mD.segment(block_offset + 18, 3) = m_nodes[1]->GetSlope2Dt().eigen();
    mD.segment(block_offset + 21, 3) = m_nodes[1]->GetSlope3Dt().eigen();
}

/// Increment all DOFs using a delta.

void ChElementBeamANCF_3243_MR_Damp::LoadableStateIncrement(const unsigned int off_x,
                                                         ChState& x_new,
                                                         const ChState& x,
                                                         const unsigned int off_v,
                                                         const ChStateDelta& Dv) {
    m_nodes[0]->NodeIntStateIncrement(off_x, x_new, x, off_v, Dv);
    m_nodes[1]->NodeIntStateIncrement(off_x + 12, x_new, x, off_v + 12, Dv);
}

// Get the pointers to the contained ChVariables, appending to the mvars vector.

void ChElementBeamANCF_3243_MR_Damp::LoadableGetVariables(std::vector<ChVariables*>& mvars) {
    for (int i = 0; i < m_nodes.size(); ++i) {
        mvars.push_back(&m_nodes[i]->Variables());
        mvars.push_back(&m_nodes[i]->VariablesSlope1());
        mvars.push_back(&m_nodes[i]->VariablesSlope2());
        mvars.push_back(&m_nodes[i]->VariablesSlope3());
    }
}

// Evaluate N'*F, which is the projection of the applied point force and moment at the beam axis coordinates (xi,0,0)
// This calculation takes a slightly different form for ANCF elements
// For this ANCF element, only the first 6 entries in F are used in the calculation.  The first three entries is
// the applied force in global coordinates and the second 3 entries is the applied moment in global space.

void ChElementBeamANCF_3243_MR_Damp::ComputeNF(
    const double xi,             // parametric coordinate along the beam axis
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
    Calc_Sxi_compact(Sxi_compact, xi, 0, 0);
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
    Calc_Sxi_D(Sxi_D, xi, 0, 0);

    ChMatrix33d J_Cxi;
    ChMatrix33d J_Cxi_Inv;

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
    //  actual differential length and the normalized differential length.  The vector 2 norm is used to calculate this
    //  length ratio for potential use in Gauss-Quadrature or similar numeric integration.
    detJ = J_Cxi.col(0).norm();
}

// Evaluate N'*F, which is the projection of the applied point force and moment at the coordinates (xi,eta,zeta)
// This calculation takes a slightly different form for ANCF elements
// For this ANCF element, only the first 6 entries in F are used in the calculation.  The first three entries is
// the applied force in global coordinates and the second 3 entries is the applied moment in global space.

void ChElementBeamANCF_3243_MR_Damp::ComputeNF(
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

    ChMatrix33d J_Cxi;
    ChMatrix33d J_Cxi_Inv;

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

double ChElementBeamANCF_3243_MR_Damp::GetDensity() {
    return GetMaterial()->GetDensity();
}

// Calculate tangent to the centerline at coordinate xi [-1 to 1].

ChVector3d ChElementBeamANCF_3243_MR_Damp::ComputeTangent(const double xi) {
    VectorN Sxi_xi_compact;
    Calc_Sxi_xi_compact(Sxi_xi_compact, xi, 0, 0);

    Matrix3xN e_bar;
    CalcCoordMatrix(e_bar);

    // partial derivative of the position vector with respect to xi (normalized coordinate along the beam axis).  In
    // general, this will not be a unit vector
    ChVector3d BeamAxisTangent = e_bar * Sxi_xi_compact;

    return BeamAxisTangent.GetNormalized();
}

// -----------------------------------------------------------------------------
// Mass Matrix & Generalized Force Due to Gravity Calculation
// -----------------------------------------------------------------------------

void ChElementBeamANCF_3243_MR_Damp::ComputeMassMatrixAndGravityForce() {
    // For this element, the mass matrix integrand is of order 10 in xi, 3 in eta, and 3 in zeta.
    // 4 GQ Points are needed in the xi direction and 2 GQ Points are needed in the eta and zeta directions for
    // exact integration of the element's mass matrix, even if the reference configuration is not straight. Since the
    // major pieces of the generalized force due to gravity can also be used to calculate the mass matrix, these
    // calculations are performed at the same time.  Only the matrix that scales the acceleration due to gravity is
    // calculated at this time so that any changes to the acceleration due to gravity in the system are correctly
    // accounted for in the generalized internal force calculation.

    ChQuadratureTables* GQTable = GetStaticGQTables();
    unsigned int GQ_idx_xi = 5;        // 6 Point Gauss-Quadrature;
    unsigned int GQ_idx_eta_zeta = 1;  // 2 Point Gauss-Quadrature;

    // Mass Matrix in its compact matrix form.  Since the mass matrix is symmetric, just the upper diagonal entries will
    // be stored.
    MatrixNxN MassMatrixCompactSquare;

    // Set these to zeros since they will be incremented as the vector/matrix is calculated
    MassMatrixCompactSquare.setZero();
    m_GravForceScale.setZero();

    double rho = GetMaterial()->GetDensity();  // Density of the material for the element

    // Sum the contribution to the mass matrix and generalized force due to gravity at the current point
    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi].size(); it_xi++) {
        for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[GQ_idx_eta_zeta].size(); it_eta++) {
            for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[GQ_idx_eta_zeta].size(); it_zeta++) {
                double GQ_weight = GQTable->Weight[GQ_idx_xi][it_xi] * GQTable->Weight[GQ_idx_eta_zeta][it_eta] *
                                   GQTable->Weight[GQ_idx_eta_zeta][it_zeta];
                double xi = GQTable->Lroots[GQ_idx_xi][it_xi];
                double eta = GQTable->Lroots[GQ_idx_eta_zeta][it_eta];
                double zeta = GQTable->Lroots[GQ_idx_eta_zeta][it_zeta];
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

// Precalculate constant matrices for the internal force calculations

void ChElementBeamANCF_3243_MR_Damp::PrecomputeInternalForceMatricesWeights() {
    ChQuadratureTables* GQTable = GetStaticGQTables();

    m_SD.resize(NSF, 3 * NIP);
    m_kGQ_S.resize(1, NIP_S);
    m_kGQ_P.resize(1, NIP_P);

    // Precalculate the matrices of normalized shape function derivatives corrected for a potentially non-straight
    // reference configuration & GQ Weights times the determinant of the element Jacobian for later use in the
    // generalized internal force and Jacobian calculations.

    // First calculate the matrices and constants for the calculations excluding the volumetric penalty terms
    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[NP - 1].size(); it_xi++) {
        for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[NT - 1].size(); it_eta++) {
            for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[NT - 1].size(); it_zeta++) {
                double GQ_weight =
                    GQTable->Weight[NP - 1][it_xi] * GQTable->Weight[NT - 1][it_eta] * GQTable->Weight[NT - 1][it_zeta];
                double xi = GQTable->Lroots[NP - 1][it_xi];
                double eta = GQTable->Lroots[NT - 1][it_eta];
                double zeta = GQTable->Lroots[NT - 1][it_zeta];
                auto index = it_zeta + it_eta * GQTable->Lroots[NT - 1].size() +
                             it_xi * GQTable->Lroots[NT - 1].size() * GQTable->Lroots[NT - 1].size();
                ChMatrix33d J_0xi;  // Element Jacobian between the reference and normalized configurations
                MatrixNx3c Sxi_D;          // Matrix of normalized shape function derivatives

                Calc_Sxi_D(Sxi_D, xi, eta, zeta);
                J_0xi.noalias() = m_ebar0 * Sxi_D;

                // Adjust the shape function derivative matrix to account for a potentially deformed reference state
                m_kGQ_S(index) = -J_0xi.determinant() * GQ_weight;
                ChMatrixNM<double, NSF, 3> SD = Sxi_D * J_0xi.inverse();

                // Group all of the columns together in blocks with the shape function derivatives for the section that
                // does not include the volumetric penalty terms at the beginning of m_SD
                m_SD.col(index) = SD.col(0);
                m_SD.col(index + NIP_S) = SD.col(1);
                m_SD.col(index + 2 * NIP_S) = SD.col(2);

                index++;
            }
        }
    }

    // Next calculate the matrices and constants for the portion of the selective reduced integration that accounts for enforcing the incompressibility of the material.  This term is integrated across the volume of the element with one 1 point
    // GQ for the cross section directions.
    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[NP - 1].size(); it_xi++) {
        double GQ_weight = GQTable->Weight[NP - 1][it_xi] * 2 * 2;
        double xi = GQTable->Lroots[NP - 1][it_xi];
        double eta = 0;
        double zeta = 0;
        ChMatrix33d J_0xi;  // Element Jacobian between the reference configuration and normalized configuration
        MatrixNx3c Sxi_D;          // Matrix of normalized shape function derivatives

        Calc_Sxi_D(Sxi_D, xi, eta, zeta);
        J_0xi.noalias() = m_ebar0 * Sxi_D;

        // Adjust the shape function derivative matrix to account for a potentially deformed reference state
        m_kGQ_P(it_xi) = -J_0xi.determinant() * GQ_weight;
        ChMatrixNM<double, NSF, 3> SD = Sxi_D * J_0xi.inverse();

        // Group all of the columns together in blocks with the shape function derivative for the volumetric penalty terms at the
        // end of m_SD
        m_SD.col(3 * NIP_S + it_xi) = SD.col(0);
        m_SD.col(3 * NIP_S + NIP_P + it_xi) = SD.col(1);
        m_SD.col(3 * NIP_S + 2 * NIP_P + it_xi) = SD.col(2);
    }
}

// -----------------------------------------------------------------------------
// Elastic force calculation
// -----------------------------------------------------------------------------

void ChElementBeamANCF_3243_MR_Damp::ComputeInternalForceDamping(ChVectorDynamic<>& Fi) {
    assert(Fi.size() == 3 * NSF);

    // Retrieve the nodal coordinates and nodal coordinate time derivatives
    Matrix6xN ebar_ebardot;
    CalcCombinedCoordMatrix(ebar_ebardot);

    // =============================================================================
    // Calculate the deformation gradient and time derivative of the deformation gradient for all Gauss quadrature
    // points in a single matrix multiplication.  Note that since the shape function derivative matrix is ordered by
    // columns, the resulting deformation gradient will be ordered by block matrix (column vectors) components
    // Note that the indices of the components are in transposed order
    // S = No Penalty Terms, P = Penalty terms integrated just along the beam axis
    //      [F11_S     F12_S     F13_S     F11_P     F12_P     F13_P ]
    //      [F21_S     F22_S     F23_S     F21_P     F22_P     F23_P ]
    // FC = [F31_S     F32_S     F33_S     F31_P     F32_P     F33_P ]
    //      [Fdot11_S  Fdot12_S  Fdot13_S  Fdot11_P  Fdot12_P  Fdot13_P ]
    //      [Fdot21_S  Fdot22_S  Fdot23_S  Fdot21_P  Fdot22_P  Fdot23_P ]
    //      [Fdot31_S  Fdot32_S  Fdot33_S  Fdot31_P  Fdot32_P  Fdot33_P ]
    // =============================================================================

    ChMatrixNM<double, 6, 3 * NIP> FC = ebar_ebardot * m_SD;

    Eigen::Map<ArrayNIP_S> F11_S(FC.block<1, NIP_S>(0, 0 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> F12_S(FC.block<1, NIP_S>(0, 1 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> F13_S(FC.block<1, NIP_S>(0, 2 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_P> F11_P(FC.block<1, NIP_P>(0, 3 * NIP_S + 0 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> F12_P(FC.block<1, NIP_P>(0, 3 * NIP_S + 1 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> F13_P(FC.block<1, NIP_P>(0, 3 * NIP_S + 2 * NIP_P).data(), 1, NIP_P);

    Eigen::Map<ArrayNIP_S> F21_S(FC.block<1, NIP_S>(1, 0 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> F22_S(FC.block<1, NIP_S>(1, 1 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> F23_S(FC.block<1, NIP_S>(1, 2 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_P> F21_P(FC.block<1, NIP_P>(1, 3 * NIP_S + 0 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> F22_P(FC.block<1, NIP_P>(1, 3 * NIP_S + 1 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> F23_P(FC.block<1, NIP_P>(1, 3 * NIP_S + 2 * NIP_P).data(), 1, NIP_P);

    Eigen::Map<ArrayNIP_S> F31_S(FC.block<1, NIP_S>(2, 0 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> F32_S(FC.block<1, NIP_S>(2, 1 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> F33_S(FC.block<1, NIP_S>(2, 2 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_P> F31_P(FC.block<1, NIP_P>(2, 3 * NIP_S + 0 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> F32_P(FC.block<1, NIP_P>(2, 3 * NIP_S + 1 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> F33_P(FC.block<1, NIP_P>(2, 3 * NIP_S + 2 * NIP_P).data(), 1, NIP_P);

    Eigen::Map<ArrayNIP_S> Fdot11_S(FC.block<1, NIP_S>(3, 0 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> Fdot12_S(FC.block<1, NIP_S>(3, 1 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> Fdot13_S(FC.block<1, NIP_S>(3, 2 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_P> Fdot11_P(FC.block<1, NIP_P>(3, 3 * NIP_S + 0 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> Fdot12_P(FC.block<1, NIP_P>(3, 3 * NIP_S + 1 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> Fdot13_P(FC.block<1, NIP_P>(3, 3 * NIP_S + 2 * NIP_P).data(), 1, NIP_P);

    Eigen::Map<ArrayNIP_S> Fdot21_S(FC.block<1, NIP_S>(4, 0 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> Fdot22_S(FC.block<1, NIP_S>(4, 1 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> Fdot23_S(FC.block<1, NIP_S>(4, 2 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_P> Fdot21_P(FC.block<1, NIP_P>(4, 3 * NIP_S + 0 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> Fdot22_P(FC.block<1, NIP_P>(4, 3 * NIP_S + 1 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> Fdot23_P(FC.block<1, NIP_P>(4, 3 * NIP_S + 2 * NIP_P).data(), 1, NIP_P);

    Eigen::Map<ArrayNIP_S> Fdot31_S(FC.block<1, NIP_S>(5, 0 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> Fdot32_S(FC.block<1, NIP_S>(5, 1 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> Fdot33_S(FC.block<1, NIP_S>(5, 2 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_P> Fdot31_P(FC.block<1, NIP_P>(5, 3 * NIP_S + 0 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> Fdot32_P(FC.block<1, NIP_P>(5, 3 * NIP_S + 1 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> Fdot33_P(FC.block<1, NIP_P>(5, 3 * NIP_S + 2 * NIP_P).data(), 1, NIP_P);


    //Calculate the Right Cauchy-Green deformation tensor (symmetric):
    ArrayNIP_S C11 = F11_S * F11_S + F21_S * F21_S + F31_S * F31_S;
    ArrayNIP_S C22 = F12_S * F12_S + F22_S * F22_S + F32_S * F32_S;
    ArrayNIP_S C33 = F13_S * F13_S + F23_S * F23_S + F33_S * F33_S;
    ArrayNIP_S C12 = F11_S * F12_S + F21_S * F22_S + F31_S * F32_S;
    ArrayNIP_S C13 = F11_S * F13_S + F21_S * F23_S + F31_S * F33_S;
    ArrayNIP_S C23 = F12_S * F13_S + F22_S * F23_S + F32_S * F33_S;

    //Calculate the 1st invariant of the Right Cauchy-Green deformation tensor: trace(C)
    ArrayNIP_S I1 = C11 + C22 + C33;

    //Calculate the 2nd invariant of the Right Cauchy-Green deformation tensor: 1/2*(trace(C)^2-trace(C^2))
    ArrayNIP_S I2 = 0.5*(I1*I1 - C11 * C11 - C22 * C22 - C33 * C33) - C12 * C12 - C13 * C13 - C23 * C23;

    //Calculate the determinate of F (commonly denoted as J)
    ArrayNIP_S J = F11_S * (F22_S*F33_S - F23_S * F32_S) + F12_S * (F23_S*F31_S - F21_S * F33_S) + F13_S * (F21_S*F32_S - F22_S * F31_S);

    //Calculate the determinate of F to the -2/3 power -> J^(-2/3)
    ArrayNIP_S J_m2_3 = J.pow(-2.0 / 3.0);

    // Calculate the time derivative of the Greens-Lagrange strain tensor at the current point (symmetric):
    ArrayNIP_S Edot11 = F11_S * Fdot11_S + F21_S * Fdot21_S + F31_S * Fdot31_S;
    ArrayNIP_S Edot22 = F12_S * Fdot12_S + F22_S * Fdot22_S + F32_S * Fdot32_S;
    ArrayNIP_S Edot33 = F13_S * Fdot13_S + F23_S * Fdot23_S + F33_S * Fdot33_S;
    ArrayNIP_S Edot12 = 0.5*(F11_S * Fdot12_S + F12_S * Fdot11_S + F21_S * Fdot22_S + F22_S * Fdot21_S + F31_S * Fdot32_S + F32_S * Fdot31_S);
    ArrayNIP_S Edot13 = 0.5*(F11_S * Fdot13_S + F13_S * Fdot11_S + F21_S * Fdot23_S + F23_S * Fdot21_S + F31_S * Fdot33_S + F33_S * Fdot31_S);
    ArrayNIP_S Edot23 = 0.5*(F12_S * Fdot13_S + F13_S * Fdot12_S + F22_S * Fdot23_S + F23_S * Fdot22_S + F32_S * Fdot33_S + F33_S * Fdot32_S);

    //Inverse of C times detF^2 (since all the detF terms will be handled together)
    ArrayNIP_S CInv11 = C22 * C33 - C23 * C23;
    ArrayNIP_S CInv22 = C11 * C33 - C13 * C13;
    ArrayNIP_S CInv33 = C11 * C22 - C12 * C12;
    ArrayNIP_S CInv12 = C13 * C23 - C12 * C33;
    ArrayNIP_S CInv13 = C12 * C23 - C13 * C22;
    ArrayNIP_S CInv23 = C13 * C12 - C11 * C23;

    //Calculate the Stress from the viscosity law
    double mu = GetMaterial()->Get_mu();
    ArrayNIP_S kGQmu_over_J3 = mu * m_kGQ_S / (J*J*J);
    ArrayNIP_S SPK2_NLKV_1 = kGQmu_over_J3 * (Edot11*CInv11*CInv11 + 2.0 * Edot12*CInv11*CInv12 + 2.0 * Edot13*CInv11*CInv13 + Edot22 * CInv12*CInv12 + 2.0 * Edot23*CInv12*CInv13 + Edot33 * CInv13*CInv13);
    ArrayNIP_S SPK2_NLKV_2 = kGQmu_over_J3 * (Edot11*CInv12*CInv12 + 2.0 * Edot12*CInv12*CInv22 + 2.0 * Edot13*CInv12*CInv23 + Edot22 * CInv22*CInv22 + 2.0 * Edot23*CInv22*CInv23 + Edot33 * CInv23*CInv23);
    ArrayNIP_S SPK2_NLKV_3 = kGQmu_over_J3 * (Edot11*CInv13*CInv13 + 2.0 * Edot12*CInv13*CInv23 + 2.0 * Edot13*CInv13*CInv33 + Edot22 * CInv23*CInv23 + 2.0 * Edot23*CInv23*CInv33 + Edot33 * CInv33*CInv33);
    ArrayNIP_S SPK2_NLKV_4 = kGQmu_over_J3 * (Edot11*CInv12*CInv13 + Edot12 * (CInv12*CInv23 + CInv22 * CInv13) + Edot13 * (CInv12*CInv33 + CInv13 * CInv23) + Edot22 * CInv22*CInv23 + Edot23 * (CInv23*CInv23 + CInv22 * CInv33) + Edot33 * CInv23*CInv33);
    ArrayNIP_S SPK2_NLKV_5 = kGQmu_over_J3 * (Edot11*CInv11*CInv13 + Edot12 * (CInv11*CInv23 + CInv12 * CInv13) + Edot13 * (CInv13*CInv13 + CInv11 * CInv33) + Edot22 * CInv12*CInv23 + Edot23 * (CInv12*CInv33 + CInv13 * CInv23) + Edot33 * CInv13*CInv33);
    ArrayNIP_S SPK2_NLKV_6 = kGQmu_over_J3 * (Edot11*CInv11*CInv12 + Edot12 * (CInv12*CInv12 + CInv11 * CInv22) + Edot13 * (CInv11*CInv23 + CInv12 * CInv13) + Edot22 * CInv12*CInv22 + Edot23 * (CInv12*CInv23 + CInv22 * CInv13) + Edot33 * CInv13*CInv23);

    //Get the element's material properties
    double c10 = GetMaterial()->Get_c10();
    double c01 = GetMaterial()->Get_c01();
    double k = GetMaterial()->Get_k();

    //Calculate the scale factors used for calculating the transpose of the 1st Piola-Kirchhoff Stress tensors
    ArrayNIP_S M0 = 2.0 * m_kGQ_S * J_m2_3;
    ArrayNIP_S M2 = -c01 * M0 * J_m2_3;
    M0 *= c10;
    ArrayNIP_S M1 = M0 - M2 * I1;
    ArrayNIP_S M3 = (2*I2*M2 - I1*M0) / (3.0 * J);

    //Calculate the transpose of the 1st Piola-Kirchhoff Stress tensors grouped by Gauss-quadrature points for the terms excluding the volumetric penalty factor
    ChMatrixNM_col<double, 3 * NIP, 3> P_Block;
    P_Block.block<NIP_S, 1>(0 * NIP_S, 0) = M3 * (F22_S * F33_S - F23_S * F32_S) + (M1 + M2 * C11) * F11_S + M2 * (C12 * F12_S + C13 * F13_S) + F11_S * SPK2_NLKV_1 + F12_S * SPK2_NLKV_6 + F13_S * SPK2_NLKV_5;
    P_Block.block<NIP_S, 1>(1 * NIP_S, 0) = M3 * (F23_S * F31_S - F21_S * F33_S) + (M1 + M2 * C22) * F12_S + M2 * (C12 * F11_S + C23 * F13_S) + F11_S * SPK2_NLKV_6 + F12_S * SPK2_NLKV_2 + F13_S * SPK2_NLKV_4;
    P_Block.block<NIP_S, 1>(2 * NIP_S, 0) = M3 * (F21_S * F32_S - F22_S * F31_S) + (M1 + M2 * C33) * F13_S + M2 * (C13 * F11_S + C23 * F12_S) + F11_S * SPK2_NLKV_5 + F12_S * SPK2_NLKV_4 + F13_S * SPK2_NLKV_3;
    P_Block.block<NIP_S, 1>(0 * NIP_S, 1) = M3 * (F13_S * F32_S - F12_S * F33_S) + (M1 + M2 * C11) * F21_S + M2 * (C12 * F22_S + C13 * F23_S) + F21_S * SPK2_NLKV_1 + F22_S * SPK2_NLKV_6 + F23_S * SPK2_NLKV_5;
    P_Block.block<NIP_S, 1>(1 * NIP_S, 1) = M3 * (F11_S * F33_S - F13_S * F31_S) + (M1 + M2 * C22) * F22_S + M2 * (C12 * F21_S + C23 * F23_S) + F21_S * SPK2_NLKV_6 + F22_S * SPK2_NLKV_2 + F23_S * SPK2_NLKV_4;
    P_Block.block<NIP_S, 1>(2 * NIP_S, 1) = M3 * (F12_S * F31_S - F11_S * F32_S) + (M1 + M2 * C33) * F23_S + M2 * (C13 * F21_S + C23 * F22_S) + F21_S * SPK2_NLKV_5 + F22_S * SPK2_NLKV_4 + F23_S * SPK2_NLKV_3;
    P_Block.block<NIP_S, 1>(0 * NIP_S, 2) = M3 * (F12_S * F23_S - F13_S * F22_S) + (M1 + M2 * C11) * F31_S + M2 * (C12 * F32_S + C13 * F33_S) + F31_S * SPK2_NLKV_1 + F32_S * SPK2_NLKV_6 + F33_S * SPK2_NLKV_5;
    P_Block.block<NIP_S, 1>(1 * NIP_S, 2) = M3 * (F13_S * F21_S - F11_S * F23_S) + (M1 + M2 * C22) * F32_S + M2 * (C12 * F31_S + C23 * F33_S) + F31_S * SPK2_NLKV_6 + F32_S * SPK2_NLKV_2 + F33_S * SPK2_NLKV_4;
    P_Block.block<NIP_S, 1>(2 * NIP_S, 2) = M3 * (F11_S * F22_S - F12_S * F21_S) + (M1 + M2 * C33) * F33_S + M2 * (C13 * F31_S + C23 * F32_S) + F31_S * SPK2_NLKV_5 + F32_S * SPK2_NLKV_4 + F33_S * SPK2_NLKV_3;


    //Calculate the determinate of F (commonly denoted as J) for the volumentric penalty Gauss quadrature points
    ArrayNIP_P J_P = F11_P * (F22_P*F33_P - F23_P * F32_P) + F12_P * (F23_P*F31_P - F21_P * F33_P) + F13_P * (F21_P*F32_P - F22_P * F31_P);
    ArrayNIP_P M3_P = k * (J_P - 1.0)*m_kGQ_P;

    //Calculate the transpose of the 1st Piola-Kirchhoff Stress tensors grouped by Gauss-quadrature points for the terms accounting for the volumetric penalty factor
    P_Block.block<NIP_P, 1>(3 * NIP_S + 0 * NIP_P, 0) = M3_P * (F22_P * F33_P - F23_P * F32_P);
    P_Block.block<NIP_P, 1>(3 * NIP_S + 1 * NIP_P, 0) = M3_P * (F23_P * F31_P - F21_P * F33_P);
    P_Block.block<NIP_P, 1>(3 * NIP_S + 2 * NIP_P, 0) = M3_P * (F21_P * F32_P - F22_P * F31_P);
    P_Block.block<NIP_P, 1>(3 * NIP_S + 0 * NIP_P, 1) = M3_P * (F13_P * F32_P - F12_P * F33_P);
    P_Block.block<NIP_P, 1>(3 * NIP_S + 1 * NIP_P, 1) = M3_P * (F11_P * F33_P - F13_P * F31_P);
    P_Block.block<NIP_P, 1>(3 * NIP_S + 2 * NIP_P, 1) = M3_P * (F12_P * F31_P - F11_P * F32_P);
    P_Block.block<NIP_P, 1>(3 * NIP_S + 0 * NIP_P, 2) = M3_P * (F12_P * F23_P - F13_P * F22_P);
    P_Block.block<NIP_P, 1>(3 * NIP_S + 1 * NIP_P, 2) = M3_P * (F13_P * F21_P - F11_P * F23_P);
    P_Block.block<NIP_P, 1>(3 * NIP_S + 2 * NIP_P, 2) = M3_P * (F11_P * F22_P - F12_P * F21_P);

    //Calculate the generalized internal force vector in matrix form and then reshape it into its required vector layout
    MatrixNx3 QiCompact = m_SD * P_Block;
    Eigen::Map<ChVectorN<double, 3 * NSF>> QiReshaped(QiCompact.data(), QiCompact.size());
    Fi.noalias() = QiReshaped;
}

void ChElementBeamANCF_3243_MR_Damp::ComputeInternalForceNoDamping(ChVectorDynamic<>& Fi) {
    assert(Fi.size() == 3 * NSF);

    // Element coordinates in matrix form
    Matrix3xN ebar;
    CalcCoordMatrix(ebar);

    // =============================================================================
    // Calculate the deformation gradient for all Gauss quadrature
    // points in a single matrix multiplication.  Note that since the shape function derivative matrix is ordered by
    // columns, the resulting deformation gradient will be ordered by block matrix (column vectors) components
    // Note that the indices of the components are in transposed order
    // S = No Penalty Terms, P = Penalty terms integrated just along the beam axis
    //      [F11_S     F12_S     F13_S     F11_P     F12_P     F13_P ]
    // FC = [F21_S     F22_S     F23_S     F21_P     F22_P     F23_P ]
    //      [F31_S     F32_S     F33_S     F31_P     F32_P     F33_P ]
    // =============================================================================

    ChMatrixNM<double, 3, 3 * NIP> FC = ebar * m_SD;

    Eigen::Map<ArrayNIP_S> F11_S(FC.block<1, NIP_S>(0, 0 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> F12_S(FC.block<1, NIP_S>(0, 1 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> F13_S(FC.block<1, NIP_S>(0, 2 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_P> F11_P(FC.block<1, NIP_P>(0, 3 * NIP_S + 0 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> F12_P(FC.block<1, NIP_P>(0, 3 * NIP_S + 1 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> F13_P(FC.block<1, NIP_P>(0, 3 * NIP_S + 2 * NIP_P).data(), 1, NIP_P);

    Eigen::Map<ArrayNIP_S> F21_S(FC.block<1, NIP_S>(1, 0 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> F22_S(FC.block<1, NIP_S>(1, 1 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> F23_S(FC.block<1, NIP_S>(1, 2 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_P> F21_P(FC.block<1, NIP_P>(1, 3 * NIP_S + 0 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> F22_P(FC.block<1, NIP_P>(1, 3 * NIP_S + 1 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> F23_P(FC.block<1, NIP_P>(1, 3 * NIP_S + 2 * NIP_P).data(), 1, NIP_P);

    Eigen::Map<ArrayNIP_S> F31_S(FC.block<1, NIP_S>(2, 0 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> F32_S(FC.block<1, NIP_S>(2, 1 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> F33_S(FC.block<1, NIP_S>(2, 2 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_P> F31_P(FC.block<1, NIP_P>(2, 3 * NIP_S + 0 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> F32_P(FC.block<1, NIP_P>(2, 3 * NIP_S + 1 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> F33_P(FC.block<1, NIP_P>(2, 3 * NIP_S + 2 * NIP_P).data(), 1, NIP_P);

    //Calculate the Right Cauchy-Green deformation tensor (symmetric):
    ArrayNIP_S C11 = F11_S * F11_S + F21_S * F21_S + F31_S * F31_S;
    ArrayNIP_S C22 = F12_S * F12_S + F22_S * F22_S + F32_S * F32_S;
    ArrayNIP_S C33 = F13_S * F13_S + F23_S * F23_S + F33_S * F33_S;
    ArrayNIP_S C12 = F11_S * F12_S + F21_S * F22_S + F31_S * F32_S;
    ArrayNIP_S C13 = F11_S * F13_S + F21_S * F23_S + F31_S * F33_S;
    ArrayNIP_S C23 = F12_S * F13_S + F22_S * F23_S + F32_S * F33_S;

    //Calculate the 1st invariant of the Right Cauchy-Green deformation tensor: trace(C)
    ArrayNIP_S I1 = C11 + C22 + C33;

    //Calculate the 2nd invariant of the Right Cauchy-Green deformation tensor: 1/2*(trace(C)^2-trace(C^2))
    ArrayNIP_S I2 = 0.5*(I1*I1 - C11 * C11 - C22 * C22 - C33 * C33) - C12 * C12 - C13 * C13 - C23 * C23;

    //Calculate the determinate of F (commonly denoted as J)
    ArrayNIP_S J = F11_S * (F22_S*F33_S - F23_S * F32_S) + F12_S * (F23_S*F31_S - F21_S * F33_S) + F13_S * (F21_S*F32_S - F22_S * F31_S);

    //Calculate the determinate of F to the -2/3 power -> J^(-2/3)
    ArrayNIP_S J_m2_3 = J.pow(-2.0 / 3.0);

    //Get the element's material properties
    double c10 = GetMaterial()->Get_c10();
    double c01 = GetMaterial()->Get_c01();
    double k = GetMaterial()->Get_k();

    //Calculate the scale factors used for calculating the transpose of the 1st Piola-Kirchhoff Stress tensors
    ArrayNIP_S M0 = 2.0 * m_kGQ_S * J_m2_3;
    ArrayNIP_S M2 = -c01 * M0 * J_m2_3;
    M0 *= c10;
    ArrayNIP_S M1 = M0 - M2 * I1;
    ArrayNIP_S M3 = (2 * I2*M2 - I1 * M0) / (3.0 * J);

    ChMatrixNM_col<double, 3 * NIP, 3> P_Block;
    P_Block.block<NIP_S, 1>(0 * NIP_S, 0) = M3 * (F22_S * F33_S - F23_S * F32_S) + (M1 + M2 * C11) * F11_S + M2 * (C12 * F12_S + C13 * F13_S);
    P_Block.block<NIP_S, 1>(1 * NIP_S, 0) = M3 * (F23_S * F31_S - F21_S * F33_S) + (M1 + M2 * C22) * F12_S + M2 * (C12 * F11_S + C23 * F13_S);
    P_Block.block<NIP_S, 1>(2 * NIP_S, 0) = M3 * (F21_S * F32_S - F22_S * F31_S) + (M1 + M2 * C33) * F13_S + M2 * (C13 * F11_S + C23 * F12_S);
    P_Block.block<NIP_S, 1>(0 * NIP_S, 1) = M3 * (F13_S * F32_S - F12_S * F33_S) + (M1 + M2 * C11) * F21_S + M2 * (C12 * F22_S + C13 * F23_S);
    P_Block.block<NIP_S, 1>(1 * NIP_S, 1) = M3 * (F11_S * F33_S - F13_S * F31_S) + (M1 + M2 * C22) * F22_S + M2 * (C12 * F21_S + C23 * F23_S);
    P_Block.block<NIP_S, 1>(2 * NIP_S, 1) = M3 * (F12_S * F31_S - F11_S * F32_S) + (M1 + M2 * C33) * F23_S + M2 * (C13 * F21_S + C23 * F22_S);
    P_Block.block<NIP_S, 1>(0 * NIP_S, 2) = M3 * (F12_S * F23_S - F13_S * F22_S) + (M1 + M2 * C11) * F31_S + M2 * (C12 * F32_S + C13 * F33_S);
    P_Block.block<NIP_S, 1>(1 * NIP_S, 2) = M3 * (F13_S * F21_S - F11_S * F23_S) + (M1 + M2 * C22) * F32_S + M2 * (C12 * F31_S + C23 * F33_S);
    P_Block.block<NIP_S, 1>(2 * NIP_S, 2) = M3 * (F11_S * F22_S - F12_S * F21_S) + (M1 + M2 * C33) * F33_S + M2 * (C13 * F31_S + C23 * F32_S);


    //Calculate the determinate of F (commonly denoted as J) for the volumentric penalty Gauss quadrature points
    ArrayNIP_P J_P = F11_P * (F22_P*F33_P - F23_P * F32_P) + F12_P * (F23_P*F31_P - F21_P * F33_P) + F13_P * (F21_P*F32_P - F22_P * F31_P);
    ArrayNIP_P M3_P = k * (J_P - 1.0)*m_kGQ_P;

    P_Block.block<NIP_P, 1>(3 * NIP_S + 0 * NIP_P, 0) = M3_P * (F22_P * F33_P - F23_P * F32_P);
    P_Block.block<NIP_P, 1>(3 * NIP_S + 1 * NIP_P, 0) = M3_P * (F23_P * F31_P - F21_P * F33_P);
    P_Block.block<NIP_P, 1>(3 * NIP_S + 2 * NIP_P, 0) = M3_P * (F21_P * F32_P - F22_P * F31_P);
    P_Block.block<NIP_P, 1>(3 * NIP_S + 0 * NIP_P, 1) = M3_P * (F13_P * F32_P - F12_P * F33_P);
    P_Block.block<NIP_P, 1>(3 * NIP_S + 1 * NIP_P, 1) = M3_P * (F11_P * F33_P - F13_P * F31_P);
    P_Block.block<NIP_P, 1>(3 * NIP_S + 2 * NIP_P, 1) = M3_P * (F12_P * F31_P - F11_P * F32_P);
    P_Block.block<NIP_P, 1>(3 * NIP_S + 0 * NIP_P, 2) = M3_P * (F12_P * F23_P - F13_P * F22_P);
    P_Block.block<NIP_P, 1>(3 * NIP_S + 1 * NIP_P, 2) = M3_P * (F13_P * F21_P - F11_P * F23_P);
    P_Block.block<NIP_P, 1>(3 * NIP_S + 2 * NIP_P, 2) = M3_P * (F11_P * F22_P - F12_P * F21_P);

    //Calculate the generalized internal force vector in matrix form and then reshape it into its required vector layout
    MatrixNx3 QiCompact = m_SD * P_Block;
    Eigen::Map<ChVectorN<double, 3 * NSF>> QiReshaped(QiCompact.data(), QiCompact.size());
    Fi.noalias() = QiReshaped;
}

// -----------------------------------------------------------------------------
// Jacobians of internal forces
// -----------------------------------------------------------------------------

void ChElementBeamANCF_3243_MR_Damp::ComputeInternalJacobianDamping(ChMatrixRef H, double Kfactor, double Rfactor, double Mfactor) {
    assert((H.rows() == 3 * NSF) && (H.cols() == 3 * NSF));

    // Retrieve the nodal coordinates and nodal coordinate time derivatives
    Matrix6xN ebar_ebardot;
    CalcCombinedCoordMatrix(ebar_ebardot);

    // =============================================================================
    // Calculate the deformation gradient and time derivative of the deformation gradient for all Gauss quadrature
    // points in a single matrix multiplication.  Note that since the shape function derivative matrix is ordered by
    // columns, the resulting deformation gradient will be ordered by block matrix (column vectors) components
    // Note that the indices of the components are in transposed order
    // S = No Penalty Terms, P = Penalty terms integrated just along the beam axis
    //      [F11_S     F12_S     F13_S     F11_P     F12_P     F13_P ]
    //      [F21_S     F22_S     F23_S     F21_P     F22_P     F23_P ]
    // FC = [F31_S     F32_S     F33_S     F31_P     F32_P     F33_P ]
    //      [Fdot11_S  Fdot12_S  Fdot13_S  Fdot11_P  Fdot12_P  Fdot13_P ]
    //      [Fdot21_S  Fdot22_S  Fdot23_S  Fdot21_P  Fdot22_P  Fdot23_P ]
    //      [Fdot31_S  Fdot32_S  Fdot33_S  Fdot31_P  Fdot32_P  Fdot33_P ]
    // =============================================================================

    ChMatrixNM<double, 6, 3 * NIP> FC = ebar_ebardot * m_SD;

    Eigen::Map<ArrayNIP_S> F11_S(FC.block<1, NIP_S>(0, 0 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> F12_S(FC.block<1, NIP_S>(0, 1 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> F13_S(FC.block<1, NIP_S>(0, 2 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_P> F11_P(FC.block<1, NIP_P>(0, 3 * NIP_S + 0 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> F12_P(FC.block<1, NIP_P>(0, 3 * NIP_S + 1 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> F13_P(FC.block<1, NIP_P>(0, 3 * NIP_S + 2 * NIP_P).data(), 1, NIP_P);

    Eigen::Map<ArrayNIP_S> F21_S(FC.block<1, NIP_S>(1, 0 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> F22_S(FC.block<1, NIP_S>(1, 1 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> F23_S(FC.block<1, NIP_S>(1, 2 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_P> F21_P(FC.block<1, NIP_P>(1, 3 * NIP_S + 0 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> F22_P(FC.block<1, NIP_P>(1, 3 * NIP_S + 1 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> F23_P(FC.block<1, NIP_P>(1, 3 * NIP_S + 2 * NIP_P).data(), 1, NIP_P);

    Eigen::Map<ArrayNIP_S> F31_S(FC.block<1, NIP_S>(2, 0 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> F32_S(FC.block<1, NIP_S>(2, 1 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> F33_S(FC.block<1, NIP_S>(2, 2 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_P> F31_P(FC.block<1, NIP_P>(2, 3 * NIP_S + 0 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> F32_P(FC.block<1, NIP_P>(2, 3 * NIP_S + 1 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> F33_P(FC.block<1, NIP_P>(2, 3 * NIP_S + 2 * NIP_P).data(), 1, NIP_P);

    Eigen::Map<ArrayNIP_S> Fdot11_S(FC.block<1, NIP_S>(3, 0 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> Fdot12_S(FC.block<1, NIP_S>(3, 1 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> Fdot13_S(FC.block<1, NIP_S>(3, 2 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_P> Fdot11_P(FC.block<1, NIP_P>(3, 3 * NIP_S + 0 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> Fdot12_P(FC.block<1, NIP_P>(3, 3 * NIP_S + 1 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> Fdot13_P(FC.block<1, NIP_P>(3, 3 * NIP_S + 2 * NIP_P).data(), 1, NIP_P);

    Eigen::Map<ArrayNIP_S> Fdot21_S(FC.block<1, NIP_S>(4, 0 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> Fdot22_S(FC.block<1, NIP_S>(4, 1 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> Fdot23_S(FC.block<1, NIP_S>(4, 2 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_P> Fdot21_P(FC.block<1, NIP_P>(4, 3 * NIP_S + 0 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> Fdot22_P(FC.block<1, NIP_P>(4, 3 * NIP_S + 1 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> Fdot23_P(FC.block<1, NIP_P>(4, 3 * NIP_S + 2 * NIP_P).data(), 1, NIP_P);

    Eigen::Map<ArrayNIP_S> Fdot31_S(FC.block<1, NIP_S>(5, 0 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> Fdot32_S(FC.block<1, NIP_S>(5, 1 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> Fdot33_S(FC.block<1, NIP_S>(5, 2 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_P> Fdot31_P(FC.block<1, NIP_P>(5, 3 * NIP_S + 0 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> Fdot32_P(FC.block<1, NIP_P>(5, 3 * NIP_S + 1 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> Fdot33_P(FC.block<1, NIP_P>(5, 3 * NIP_S + 2 * NIP_P).data(), 1, NIP_P);

    ArrayNIP_S BA11 = F22_S * F33_S - F23_S * F32_S;
    ArrayNIP_S BA21 = F23_S * F31_S - F21_S * F33_S;
    ArrayNIP_S BA31 = F21_S * F32_S - F22_S * F31_S;
    ArrayNIP_S BA12 = F13_S * F32_S - F12_S * F33_S;
    ArrayNIP_S BA22 = F11_S * F33_S - F13_S * F31_S;
    ArrayNIP_S BA32 = F12_S * F31_S - F11_S * F32_S;
    ArrayNIP_S BA13 = F12_S * F23_S - F13_S * F22_S;
    ArrayNIP_S BA23 = F13_S * F21_S - F11_S * F23_S;
    ArrayNIP_S BA33 = F11_S * F22_S - F12_S * F21_S;

    ArrayNIP_P BA11_P = F22_P * F33_P - F23_P * F32_P;
    ArrayNIP_P BA21_P = F23_P * F31_P - F21_P * F33_P;
    ArrayNIP_P BA31_P = F21_P * F32_P - F22_P * F31_P;
    ArrayNIP_P BA12_P = F13_P * F32_P - F12_P * F33_P;
    ArrayNIP_P BA22_P = F11_P * F33_P - F13_P * F31_P;
    ArrayNIP_P BA32_P = F12_P * F31_P - F11_P * F32_P;
    ArrayNIP_P BA13_P = F12_P * F23_P - F13_P * F22_P;
    ArrayNIP_P BA23_P = F13_P * F21_P - F11_P * F23_P;
    ArrayNIP_P BA33_P = F11_P * F22_P - F12_P * F21_P;

    ChMatrixNM<double, 3 * NSF, 7 * NIP_S + NIP_P> Left;
    for (auto i = 0; i < NSF; i++) {
        Left.block<1, NIP_S>(3 * i + 0, 0).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * F11_S;
        Left.block<1, NIP_S>(3 * i + 0, 0 + 1 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * F12_S;
        Left.block<1, NIP_S>(3 * i + 0, 0 + 2 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * F13_S;
        Left.block<1, NIP_S>(3 * i + 0, 0 + 3 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * F13_S + m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * F12_S;
        Left.block<1, NIP_S>(3 * i + 0, 0 + 4 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * F13_S + m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * F11_S;
        Left.block<1, NIP_S>(3 * i + 0, 0 + 5 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * F12_S + m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * F11_S;
        Left.block<1, NIP_S>(3 * i + 0, 0 + 6 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * BA11 +
            m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * BA21 +
            m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * BA31;
        Left.block<1, NIP_P>(3 * i + 0, 0 + 7 * NIP_S).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 0 * NIP_P).array() * BA11_P +
            m_SD.block<1, NIP_P>(i, 3 * NIP_S + 1 * NIP_P).array() * BA21_P +
            m_SD.block<1, NIP_P>(i, 3 * NIP_S + 2 * NIP_P).array() * BA31_P;

        Left.block<1, NIP_S>(3 * i + 1, 0).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * F21_S;
        Left.block<1, NIP_S>(3 * i + 1, 0 + 1 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * F22_S;
        Left.block<1, NIP_S>(3 * i + 1, 0 + 2 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * F23_S;
        Left.block<1, NIP_S>(3 * i + 1, 0 + 3 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * F23_S + m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * F22_S;
        Left.block<1, NIP_S>(3 * i + 1, 0 + 4 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * F23_S + m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * F21_S;
        Left.block<1, NIP_S>(3 * i + 1, 0 + 5 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * F22_S + m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * F21_S;
        Left.block<1, NIP_S>(3 * i + 1, 0 + 6 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * BA12 +
            m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * BA22 +
            m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * BA32;
        Left.block<1, NIP_P>(3 * i + 1, 0 + 7 * NIP_S).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 0 * NIP_P).array() * BA12_P +
            m_SD.block<1, NIP_P>(i, 3 * NIP_S + 1 * NIP_P).array() * BA22_P +
            m_SD.block<1, NIP_P>(i, 3 * NIP_S + 2 * NIP_P).array() * BA32_P;

        Left.block<1, NIP_S>(3 * i + 2, 0).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * F31_S;
        Left.block<1, NIP_S>(3 * i + 2, 0 + 1 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * F32_S;
        Left.block<1, NIP_S>(3 * i + 2, 0 + 2 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * F33_S;
        Left.block<1, NIP_S>(3 * i + 2, 0 + 3 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * F33_S + m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * F32_S;
        Left.block<1, NIP_S>(3 * i + 2, 0 + 4 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * F33_S + m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * F31_S;
        Left.block<1, NIP_S>(3 * i + 2, 0 + 5 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * F32_S + m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * F31_S;
        Left.block<1, NIP_S>(3 * i + 2, 0 + 6 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * BA13 +
            m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * BA23 +
            m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * BA33;
        Left.block<1, NIP_P>(3 * i + 2, 0 + 7 * NIP_S).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 0 * NIP_P).array() * BA13_P +
            m_SD.block<1, NIP_P>(i, 3 * NIP_S + 1 * NIP_P).array() * BA23_P +
            m_SD.block<1, NIP_P>(i, 3 * NIP_S + 2 * NIP_P).array() * BA33_P;
    }

    // Calculate the time derivative of the Greens-Lagrange strain tensor at the current point (symmetric):
    ArrayNIP_S Edot11 = F11_S * Fdot11_S + F21_S * Fdot21_S + F31_S * Fdot31_S;
    ArrayNIP_S Edot22 = F12_S * Fdot12_S + F22_S * Fdot22_S + F32_S * Fdot32_S;
    ArrayNIP_S Edot33 = F13_S * Fdot13_S + F23_S * Fdot23_S + F33_S * Fdot33_S;
    ArrayNIP_S Edot12 = 0.5*(F11_S * Fdot12_S + F12_S * Fdot11_S + F21_S * Fdot22_S + F22_S * Fdot21_S + F31_S * Fdot32_S + F32_S * Fdot31_S);
    ArrayNIP_S Edot13 = 0.5*(F11_S * Fdot13_S + F13_S * Fdot11_S + F21_S * Fdot23_S + F23_S * Fdot21_S + F31_S * Fdot33_S + F33_S * Fdot31_S);
    ArrayNIP_S Edot23 = 0.5*(F12_S * Fdot13_S + F13_S * Fdot12_S + F22_S * Fdot23_S + F23_S * Fdot22_S + F32_S * Fdot33_S + F33_S * Fdot32_S);

    //Calculate the Right Cauchy-Green deformation tensor (symmetric):
    ArrayNIP_S C11 = F11_S * F11_S + F21_S * F21_S + F31_S * F31_S;
    ArrayNIP_S C22 = F12_S * F12_S + F22_S * F22_S + F32_S * F32_S;
    ArrayNIP_S C33 = F13_S * F13_S + F23_S * F23_S + F33_S * F33_S;
    ArrayNIP_S C12 = F11_S * F12_S + F21_S * F22_S + F31_S * F32_S;
    ArrayNIP_S C13 = F11_S * F13_S + F21_S * F23_S + F31_S * F33_S;
    ArrayNIP_S C23 = F12_S * F13_S + F22_S * F23_S + F32_S * F33_S;

    //Calculate the 1st invariant of the Right Cauchy-Green deformation tensor: trace(C)
    ArrayNIP_S I1 = C11 + C22 + C33;

    //Calculate the 2nd invariant of the Right Cauchy-Green deformation tensor: 1/2*(trace(C)^2-trace(C^2))
    ArrayNIP_S I2 = 0.5*(I1*I1 - C11 * C11 - C22 * C22 - C33 * C33) - C12 * C12 - C13 * C13 - C23 * C23;

    //Inverse of C times detF^2 (since all the detF terms will be handled together)
    ArrayNIP_S CInv11 = C22 * C33 - C23 * C23;
    ArrayNIP_S CInv22 = C11 * C33 - C13 * C13;
    ArrayNIP_S CInv33 = C11 * C22 - C12 * C12;
    ArrayNIP_S CInv12 = C13 * C23 - C12 * C33;
    ArrayNIP_S CInv13 = C12 * C23 - C13 * C22;
    ArrayNIP_S CInv23 = C13 * C12 - C11 * C23;

    ArrayNIP_S EdotC1_CInvC1 = Edot11 * CInv11 + Edot12 * CInv12 + Edot13 * CInv13;
    ArrayNIP_S EdotC2_CInvC1 = Edot12 * CInv11 + Edot22 * CInv12 + Edot23 * CInv13;
    ArrayNIP_S EdotC3_CInvC1 = Edot13 * CInv11 + Edot23 * CInv12 + Edot33 * CInv13;
    ArrayNIP_S EdotC1_CInvC2 = Edot11 * CInv12 + Edot12 * CInv22 + Edot13 * CInv23;
    ArrayNIP_S EdotC2_CInvC2 = Edot12 * CInv12 + Edot22 * CInv22 + Edot23 * CInv23;
    ArrayNIP_S EdotC3_CInvC2 = Edot13 * CInv12 + Edot23 * CInv22 + Edot33 * CInv23;
    ArrayNIP_S EdotC1_CInvC3 = Edot11 * CInv13 + Edot12 * CInv23 + Edot13 * CInv33;
    ArrayNIP_S EdotC2_CInvC3 = Edot12 * CInv13 + Edot22 * CInv23 + Edot23 * CInv33;
    ArrayNIP_S EdotC3_CInvC3 = Edot13 * CInv13 + Edot23 * CInv23 + Edot33 * CInv33;

    //Calculate the determinate of F (commonly denoted as J)
    ArrayNIP_S J = F11_S * (F22_S*F33_S - F23_S * F32_S) + F12_S * (F23_S*F31_S - F21_S * F33_S) + F13_S * (F21_S*F32_S - F22_S * F31_S);

    //Calculate the determinate of F to the -2/3 power -> J^(-2/3)
    ArrayNIP_S J_m2_3 = J.pow(-2.0 / 3.0);

    //Get the element's material properties
    double c10 = GetMaterial()->Get_c10();
    double c01 = GetMaterial()->Get_c01();
    double k = GetMaterial()->Get_k();
    double mu = GetMaterial()->Get_mu();

    //Calculate the first set of scale factors
    ArrayNIP_S kGQmu_over_J3 = mu * m_kGQ_S / (J*J*J);
    ArrayNIP_S KK = -Kfactor * kGQmu_over_J3;
    ArrayNIP_S KR = -Rfactor * kGQmu_over_J3;
    ArrayNIP_S SPK2_Scale = -3.0 / J;
    ArrayNIP_S M0 = (-Kfactor * 2.0) * m_kGQ_S * J_m2_3; //Temporary term used to build the other scale factors
    ArrayNIP_S M2 = (-2.0*c01) * M0 * J_m2_3; //Double M2 since 2*M2 is needed for the partial C11/de, C22/de, and C33/de terms.  It is reset to just M2 after that.
    ArrayNIP_S M4 = (-2.0 / 3.0) * M2 / J;
    M0 *= c10;
    ArrayNIP_S M5 = (-2.0 / 3.0) * M0 / J - I1 * M4;

    //Calculate the Stress from the viscosity law (scaled by -Kfactor * m_kGQ)
    ArrayNIP_S SPK2_NLKV_1 = KK * (Edot11*CInv11*CInv11 + 2.0 * Edot12*CInv11*CInv12 + 2.0 * Edot13*CInv11*CInv13 + Edot22 * CInv12*CInv12 + 2.0 * Edot23*CInv12*CInv13 + Edot33 * CInv13*CInv13);
    ArrayNIP_S SPK2_NLKV_2 = KK * (Edot11*CInv12*CInv12 + 2.0 * Edot12*CInv12*CInv22 + 2.0 * Edot13*CInv12*CInv23 + Edot22 * CInv22*CInv22 + 2.0 * Edot23*CInv22*CInv23 + Edot33 * CInv23*CInv23);
    ArrayNIP_S SPK2_NLKV_3 = KK * (Edot11*CInv13*CInv13 + 2.0 * Edot12*CInv13*CInv23 + 2.0 * Edot13*CInv13*CInv33 + Edot22 * CInv23*CInv23 + 2.0 * Edot23*CInv23*CInv33 + Edot33 * CInv33*CInv33);
    ArrayNIP_S SPK2_NLKV_4 = KK * (Edot11*CInv12*CInv13 + Edot12 * (CInv12*CInv23 + CInv22 * CInv13) + Edot13 * (CInv12*CInv33 + CInv13 * CInv23) + Edot22 * CInv22*CInv23 + Edot23 * (CInv23*CInv23 + CInv22 * CInv33) + Edot33 * CInv23*CInv33);
    ArrayNIP_S SPK2_NLKV_5 = KK * (Edot11*CInv11*CInv13 + Edot12 * (CInv11*CInv23 + CInv12 * CInv13) + Edot13 * (CInv13*CInv13 + CInv11 * CInv33) + Edot22 * CInv12*CInv23 + Edot23 * (CInv12*CInv33 + CInv13 * CInv23) + Edot33 * CInv13*CInv33);
    ArrayNIP_S SPK2_NLKV_6 = KK * (Edot11*CInv11*CInv12 + Edot12 * (CInv12*CInv12 + CInv11 * CInv22) + Edot13 * (CInv11*CInv23 + CInv12 * CInv13) + Edot22 * CInv12*CInv22 + Edot23 * (CInv12*CInv23 + CInv22 * CInv13) + Edot33 * CInv13*CInv23);

    ChMatrixNM<double, 3 * NSF, 7 * NIP_S + NIP_P> Right;

    //Calculate the Contribution from the dC11 / de terms
    {
        ArrayNIP_S ScaleNLKV_11cFdot = CInv11 * CInv11;
        ArrayNIP_S ScaleNLKV_22cFdot = CInv12 * CInv12;
        ArrayNIP_S ScaleNLKV_33cFdot = CInv13 * CInv13;
        ArrayNIP_S ScaleNLKV_12cFdot = CInv11 * CInv12;
        ArrayNIP_S ScaleNLKV_13cFdot = CInv11 * CInv13;
        ArrayNIP_S ScaleNLKV_23cFdot = CInv12 * CInv13;
        ArrayNIP_S ScaleNLKV_11cF = KR * ScaleNLKV_11cFdot;
        ArrayNIP_S ScaleNLKV_22cF = 4.0 * KK * (C33 * EdotC1_CInvC1 - C13 * EdotC3_CInvC1) + KR * ScaleNLKV_22cFdot;
        ArrayNIP_S ScaleNLKV_33cF = 4.0 * KK * (C22 * EdotC1_CInvC1 - C12 * EdotC2_CInvC1) + KR * ScaleNLKV_33cFdot;
        ArrayNIP_S ScaleNLKV_12cF = 2.0 * KK * (C23 * EdotC3_CInvC1 - C33 * EdotC2_CInvC1) + KR * ScaleNLKV_12cFdot;
        ArrayNIP_S ScaleNLKV_13cF = 2.0 * KK * (C23 * EdotC2_CInvC1 - C22 * EdotC3_CInvC1) + KR * ScaleNLKV_13cFdot;
        ArrayNIP_S ScaleNLKV_23cF = 2.0 * KK * (C12 * EdotC3_CInvC1 + C13 * EdotC2_CInvC1 - 2.0 * C23 * EdotC1_CInvC1) + KR * ScaleNLKV_23cFdot;
        ScaleNLKV_11cFdot *= KK;
        ScaleNLKV_22cFdot *= KK;
        ScaleNLKV_33cFdot *= KK;
        ScaleNLKV_12cFdot *= KK;
        ScaleNLKV_13cFdot *= KK;
        ScaleNLKV_23cFdot *= KK;

        ArrayNIP_S Partial_SPK2_NLKV_1_de_combined11 = SPK2_Scale * SPK2_NLKV_1 * BA22 + ScaleNLKV_11cFdot * Fdot11_S + ScaleNLKV_12cFdot * Fdot12_S + ScaleNLKV_13cFdot * Fdot13_S + ScaleNLKV_11cF * F11_S + ScaleNLKV_12cF * F12_S + ScaleNLKV_13cF * F13_S;
        ArrayNIP_S Partial_SPK2_NLKV_1_de_combined12 = SPK2_Scale * SPK2_NLKV_1 * BA23 + ScaleNLKV_11cFdot * Fdot21_S + ScaleNLKV_12cFdot * Fdot22_S + ScaleNLKV_13cFdot * Fdot23_S + ScaleNLKV_11cF * F21_S + ScaleNLKV_12cF * F22_S + ScaleNLKV_13cF * F23_S;
        ArrayNIP_S Partial_SPK2_NLKV_1_de_combined13 = SPK2_Scale * SPK2_NLKV_1 * BA13 + ScaleNLKV_11cFdot * Fdot31_S + ScaleNLKV_12cFdot * Fdot32_S + ScaleNLKV_13cFdot * Fdot33_S + ScaleNLKV_11cF * F31_S + ScaleNLKV_12cF * F32_S + ScaleNLKV_13cF * F33_S;
        ArrayNIP_S Partial_SPK2_NLKV_1_de_combined21 = SPK2_Scale * SPK2_NLKV_1 * BA32 + ScaleNLKV_12cFdot * Fdot11_S + ScaleNLKV_22cFdot * Fdot12_S + ScaleNLKV_23cFdot * Fdot13_S + ScaleNLKV_12cF * F11_S + ScaleNLKV_22cF * F12_S + ScaleNLKV_23cF * F13_S;
        ArrayNIP_S Partial_SPK2_NLKV_1_de_combined22 = SPK2_Scale * SPK2_NLKV_1 * BA33 + ScaleNLKV_12cFdot * Fdot21_S + ScaleNLKV_22cFdot * Fdot22_S + ScaleNLKV_23cFdot * Fdot23_S + ScaleNLKV_12cF * F21_S + ScaleNLKV_22cF * F22_S + ScaleNLKV_23cF * F23_S;
        ArrayNIP_S Partial_SPK2_NLKV_1_de_combined23 = SPK2_Scale * SPK2_NLKV_1 * BA23 + ScaleNLKV_12cFdot * Fdot31_S + ScaleNLKV_22cFdot * Fdot32_S + ScaleNLKV_23cFdot * Fdot33_S + ScaleNLKV_12cF * F31_S + ScaleNLKV_22cF * F32_S + ScaleNLKV_23cF * F33_S;
        ArrayNIP_S Partial_SPK2_NLKV_1_de_combined31 = SPK2_Scale * SPK2_NLKV_1 * BA31 + ScaleNLKV_13cFdot * Fdot11_S + ScaleNLKV_23cFdot * Fdot12_S + ScaleNLKV_33cFdot * Fdot13_S + ScaleNLKV_13cF * F11_S + ScaleNLKV_23cF * F12_S + ScaleNLKV_33cF * F13_S;
        ArrayNIP_S Partial_SPK2_NLKV_1_de_combined32 = SPK2_Scale * SPK2_NLKV_1 * BA32 + ScaleNLKV_13cFdot * Fdot21_S + ScaleNLKV_23cFdot * Fdot22_S + ScaleNLKV_33cFdot * Fdot23_S + ScaleNLKV_13cF * F21_S + ScaleNLKV_23cF * F22_S + ScaleNLKV_33cF * F23_S;
        ArrayNIP_S Partial_SPK2_NLKV_1_de_combined33 = SPK2_Scale * SPK2_NLKV_1 * BA33 + ScaleNLKV_13cFdot * Fdot31_S + ScaleNLKV_23cFdot * Fdot32_S + ScaleNLKV_33cFdot * Fdot33_S + ScaleNLKV_13cF * F31_S + ScaleNLKV_23cF * F32_S + ScaleNLKV_33cF * F33_S;

        ArrayNIP_S T0 = (M4 * C11 + M5);
        ArrayNIP_S T11 = T0 * BA11 + Partial_SPK2_NLKV_1_de_combined11;
        ArrayNIP_S T21 = T0 * BA21 - M2 * F12_S + Partial_SPK2_NLKV_1_de_combined21;
        ArrayNIP_S T31 = T0 * BA31 - M2 * F13_S + Partial_SPK2_NLKV_1_de_combined31;

        ArrayNIP_S T12 = T0 * BA12 + Partial_SPK2_NLKV_1_de_combined12;
        ArrayNIP_S T22 = T0 * BA22 - M2 * F22_S + Partial_SPK2_NLKV_1_de_combined22;
        ArrayNIP_S T32 = T0 * BA32 - M2 * F23_S + Partial_SPK2_NLKV_1_de_combined32;

        ArrayNIP_S T13 = T0 * BA13 + Partial_SPK2_NLKV_1_de_combined13;
        ArrayNIP_S T23 = T0 * BA23 - M2 * F32_S + Partial_SPK2_NLKV_1_de_combined23;
        ArrayNIP_S T33 = T0 * BA33 - M2 * F33_S + Partial_SPK2_NLKV_1_de_combined33;

        for (auto i = 0; i < NSF; i++) {
            Right.block<1, NIP_S>(3 * i + 0, 0).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T11 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T21 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T31;
            Right.block<1, NIP_S>(3 * i + 1, 0).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T12 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T22 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T32;
            Right.block<1, NIP_S>(3 * i + 2, 0).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T13 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T23 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T33;
        }
    }
    //Calculate the Contribution from the dC22 / de terms
    {
        ArrayNIP_S ScaleNLKV_11cFdot = CInv12 * CInv12;
        ArrayNIP_S ScaleNLKV_22cFdot = CInv22 * CInv22;
        ArrayNIP_S ScaleNLKV_33cFdot = CInv23 * CInv23;
        ArrayNIP_S ScaleNLKV_12cFdot = CInv12 * CInv22;
        ArrayNIP_S ScaleNLKV_13cFdot = CInv12 * CInv23;
        ArrayNIP_S ScaleNLKV_23cFdot = CInv22 * CInv23;
        ArrayNIP_S ScaleNLKV_11cF = 4.0 * KK * (C33 * EdotC2_CInvC2 - C23 * EdotC3_CInvC2) + KR * ScaleNLKV_11cFdot;
        ArrayNIP_S ScaleNLKV_22cF = KR * ScaleNLKV_22cFdot;
        ArrayNIP_S ScaleNLKV_33cF = 4.0 * KK * (C11 * EdotC2_CInvC2 - C12 * EdotC1_CInvC2) + KR * ScaleNLKV_33cFdot;
        ArrayNIP_S ScaleNLKV_12cF = 2.0 * KK * (C13 * EdotC3_CInvC2 - C33 * EdotC1_CInvC2) + KR * ScaleNLKV_12cFdot;
        ArrayNIP_S ScaleNLKV_13cF = 2.0 * KK * (C12 * EdotC3_CInvC2 + C23 * EdotC1_CInvC2 - 2.0 * C13 * EdotC2_CInvC2) + KR * ScaleNLKV_13cFdot;
        ArrayNIP_S ScaleNLKV_23cF = 2.0 * KK * (C13 * EdotC1_CInvC2 - C11 * EdotC3_CInvC2) + KR * ScaleNLKV_23cFdot;
        ScaleNLKV_11cFdot *= KK;
        ScaleNLKV_22cFdot *= KK;
        ScaleNLKV_33cFdot *= KK;
        ScaleNLKV_12cFdot *= KK;
        ScaleNLKV_13cFdot *= KK;
        ScaleNLKV_23cFdot *= KK;

        ArrayNIP_S Partial_SPK2_NLKV_2_de_combined11 = SPK2_Scale * SPK2_NLKV_2 * BA22 + ScaleNLKV_11cFdot * Fdot11_S + ScaleNLKV_12cFdot * Fdot12_S + ScaleNLKV_13cFdot * Fdot13_S + ScaleNLKV_11cF * F11_S + ScaleNLKV_12cF * F12_S + ScaleNLKV_13cF * F13_S;
        ArrayNIP_S Partial_SPK2_NLKV_2_de_combined12 = SPK2_Scale * SPK2_NLKV_2 * BA23 + ScaleNLKV_11cFdot * Fdot21_S + ScaleNLKV_12cFdot * Fdot22_S + ScaleNLKV_13cFdot * Fdot23_S + ScaleNLKV_11cF * F21_S + ScaleNLKV_12cF * F22_S + ScaleNLKV_13cF * F23_S;
        ArrayNIP_S Partial_SPK2_NLKV_2_de_combined13 = SPK2_Scale * SPK2_NLKV_2 * BA13 + ScaleNLKV_11cFdot * Fdot31_S + ScaleNLKV_12cFdot * Fdot32_S + ScaleNLKV_13cFdot * Fdot33_S + ScaleNLKV_11cF * F31_S + ScaleNLKV_12cF * F32_S + ScaleNLKV_13cF * F33_S;
        ArrayNIP_S Partial_SPK2_NLKV_2_de_combined21 = SPK2_Scale * SPK2_NLKV_2 * BA32 + ScaleNLKV_12cFdot * Fdot11_S + ScaleNLKV_22cFdot * Fdot12_S + ScaleNLKV_23cFdot * Fdot13_S + ScaleNLKV_12cF * F11_S + ScaleNLKV_22cF * F12_S + ScaleNLKV_23cF * F13_S;
        ArrayNIP_S Partial_SPK2_NLKV_2_de_combined22 = SPK2_Scale * SPK2_NLKV_2 * BA33 + ScaleNLKV_12cFdot * Fdot21_S + ScaleNLKV_22cFdot * Fdot22_S + ScaleNLKV_23cFdot * Fdot23_S + ScaleNLKV_12cF * F21_S + ScaleNLKV_22cF * F22_S + ScaleNLKV_23cF * F23_S;
        ArrayNIP_S Partial_SPK2_NLKV_2_de_combined23 = SPK2_Scale * SPK2_NLKV_2 * BA23 + ScaleNLKV_12cFdot * Fdot31_S + ScaleNLKV_22cFdot * Fdot32_S + ScaleNLKV_23cFdot * Fdot33_S + ScaleNLKV_12cF * F31_S + ScaleNLKV_22cF * F32_S + ScaleNLKV_23cF * F33_S;
        ArrayNIP_S Partial_SPK2_NLKV_2_de_combined31 = SPK2_Scale * SPK2_NLKV_2 * BA31 + ScaleNLKV_13cFdot * Fdot11_S + ScaleNLKV_23cFdot * Fdot12_S + ScaleNLKV_33cFdot * Fdot13_S + ScaleNLKV_13cF * F11_S + ScaleNLKV_23cF * F12_S + ScaleNLKV_33cF * F13_S;
        ArrayNIP_S Partial_SPK2_NLKV_2_de_combined32 = SPK2_Scale * SPK2_NLKV_2 * BA32 + ScaleNLKV_13cFdot * Fdot21_S + ScaleNLKV_23cFdot * Fdot22_S + ScaleNLKV_33cFdot * Fdot23_S + ScaleNLKV_13cF * F21_S + ScaleNLKV_23cF * F22_S + ScaleNLKV_33cF * F23_S;
        ArrayNIP_S Partial_SPK2_NLKV_2_de_combined33 = SPK2_Scale * SPK2_NLKV_2 * BA33 + ScaleNLKV_13cFdot * Fdot31_S + ScaleNLKV_23cFdot * Fdot32_S + ScaleNLKV_33cFdot * Fdot33_S + ScaleNLKV_13cF * F31_S + ScaleNLKV_23cF * F32_S + ScaleNLKV_33cF * F33_S;

        ArrayNIP_S T0 = (M4 * C22 + M5);
        ArrayNIP_S T11 = T0 * BA11 - M2 * F11_S + Partial_SPK2_NLKV_2_de_combined11;
        ArrayNIP_S T21 = T0 * BA21 + Partial_SPK2_NLKV_2_de_combined21;
        ArrayNIP_S T31 = T0 * BA31 - M2 * F13_S + Partial_SPK2_NLKV_2_de_combined31;

        ArrayNIP_S T12 = T0 * BA12 - M2 * F21_S + Partial_SPK2_NLKV_2_de_combined12;
        ArrayNIP_S T22 = T0 * BA22 + Partial_SPK2_NLKV_2_de_combined22;
        ArrayNIP_S T32 = T0 * BA32 - M2 * F23_S + Partial_SPK2_NLKV_2_de_combined32;

        ArrayNIP_S T13 = T0 * BA13 - M2 * F31_S + Partial_SPK2_NLKV_2_de_combined13;
        ArrayNIP_S T23 = T0 * BA23 + Partial_SPK2_NLKV_2_de_combined23;
        ArrayNIP_S T33 = T0 * BA33 - M2 * F33_S + Partial_SPK2_NLKV_2_de_combined33;

        for (auto i = 0; i < NSF; i++) {
            Right.block<1, NIP_S>(3 * i + 0, 0 + 1 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T11 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T21 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T31;
            Right.block<1, NIP_S>(3 * i + 1, 0 + 1 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T12 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T22 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T32;
            Right.block<1, NIP_S>(3 * i + 2, 0 + 1 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T13 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T23 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T33;
        }
    }
    //Calculate the Contribution from the dC33 / de terms
    {
        ArrayNIP_S ScaleNLKV_11cFdot = CInv13 * CInv13;
        ArrayNIP_S ScaleNLKV_22cFdot = CInv23 * CInv23;
        ArrayNIP_S ScaleNLKV_33cFdot = CInv33 * CInv33;
        ArrayNIP_S ScaleNLKV_12cFdot = CInv13 * CInv23;
        ArrayNIP_S ScaleNLKV_13cFdot = CInv13 * CInv33;
        ArrayNIP_S ScaleNLKV_23cFdot = CInv23 * CInv33;
        ArrayNIP_S ScaleNLKV_11cF = 4.0 * KK * (C22 * EdotC3_CInvC3 - C23 * EdotC2_CInvC3) + KR * ScaleNLKV_11cFdot;
        ArrayNIP_S ScaleNLKV_22cF = 4.0 * KK * (C11 * EdotC3_CInvC3 - C13 * EdotC1_CInvC3) + KR * ScaleNLKV_22cFdot;
        ArrayNIP_S ScaleNLKV_33cF = KR * ScaleNLKV_33cFdot;
        ArrayNIP_S ScaleNLKV_12cF = 2.0 * KK * (C13 * EdotC2_CInvC3 + C23 * EdotC1_CInvC3 - 2.0 * C12 * EdotC3_CInvC3) + KR * ScaleNLKV_12cFdot;
        ArrayNIP_S ScaleNLKV_13cF = 2.0 * KK * (C12 * EdotC2_CInvC3 - C22 * EdotC1_CInvC3) + KR * ScaleNLKV_13cFdot;
        ArrayNIP_S ScaleNLKV_23cF = 2.0 * KK * (C12 * EdotC1_CInvC3 - C11 * EdotC2_CInvC3) + KR * ScaleNLKV_23cFdot;
        ScaleNLKV_11cFdot *= KK;
        ScaleNLKV_22cFdot *= KK;
        ScaleNLKV_33cFdot *= KK;
        ScaleNLKV_12cFdot *= KK;
        ScaleNLKV_13cFdot *= KK;
        ScaleNLKV_23cFdot *= KK;

        ArrayNIP_S Partial_SPK2_NLKV_3_de_combined11 = SPK2_Scale * SPK2_NLKV_3 * BA22 + ScaleNLKV_11cFdot * Fdot11_S + ScaleNLKV_12cFdot * Fdot12_S + ScaleNLKV_13cFdot * Fdot13_S + ScaleNLKV_11cF * F11_S + ScaleNLKV_12cF * F12_S + ScaleNLKV_13cF * F13_S;
        ArrayNIP_S Partial_SPK2_NLKV_3_de_combined12 = SPK2_Scale * SPK2_NLKV_3 * BA23 + ScaleNLKV_11cFdot * Fdot21_S + ScaleNLKV_12cFdot * Fdot22_S + ScaleNLKV_13cFdot * Fdot23_S + ScaleNLKV_11cF * F21_S + ScaleNLKV_12cF * F22_S + ScaleNLKV_13cF * F23_S;
        ArrayNIP_S Partial_SPK2_NLKV_3_de_combined13 = SPK2_Scale * SPK2_NLKV_3 * BA13 + ScaleNLKV_11cFdot * Fdot31_S + ScaleNLKV_12cFdot * Fdot32_S + ScaleNLKV_13cFdot * Fdot33_S + ScaleNLKV_11cF * F31_S + ScaleNLKV_12cF * F32_S + ScaleNLKV_13cF * F33_S;
        ArrayNIP_S Partial_SPK2_NLKV_3_de_combined21 = SPK2_Scale * SPK2_NLKV_3 * BA32 + ScaleNLKV_12cFdot * Fdot11_S + ScaleNLKV_22cFdot * Fdot12_S + ScaleNLKV_23cFdot * Fdot13_S + ScaleNLKV_12cF * F11_S + ScaleNLKV_22cF * F12_S + ScaleNLKV_23cF * F13_S;
        ArrayNIP_S Partial_SPK2_NLKV_3_de_combined22 = SPK2_Scale * SPK2_NLKV_3 * BA33 + ScaleNLKV_12cFdot * Fdot21_S + ScaleNLKV_22cFdot * Fdot22_S + ScaleNLKV_23cFdot * Fdot23_S + ScaleNLKV_12cF * F21_S + ScaleNLKV_22cF * F22_S + ScaleNLKV_23cF * F23_S;
        ArrayNIP_S Partial_SPK2_NLKV_3_de_combined23 = SPK2_Scale * SPK2_NLKV_3 * BA23 + ScaleNLKV_12cFdot * Fdot31_S + ScaleNLKV_22cFdot * Fdot32_S + ScaleNLKV_23cFdot * Fdot33_S + ScaleNLKV_12cF * F31_S + ScaleNLKV_22cF * F32_S + ScaleNLKV_23cF * F33_S;
        ArrayNIP_S Partial_SPK2_NLKV_3_de_combined31 = SPK2_Scale * SPK2_NLKV_3 * BA31 + ScaleNLKV_13cFdot * Fdot11_S + ScaleNLKV_23cFdot * Fdot12_S + ScaleNLKV_33cFdot * Fdot13_S + ScaleNLKV_13cF * F11_S + ScaleNLKV_23cF * F12_S + ScaleNLKV_33cF * F13_S;
        ArrayNIP_S Partial_SPK2_NLKV_3_de_combined32 = SPK2_Scale * SPK2_NLKV_3 * BA32 + ScaleNLKV_13cFdot * Fdot21_S + ScaleNLKV_23cFdot * Fdot22_S + ScaleNLKV_33cFdot * Fdot23_S + ScaleNLKV_13cF * F21_S + ScaleNLKV_23cF * F22_S + ScaleNLKV_33cF * F23_S;
        ArrayNIP_S Partial_SPK2_NLKV_3_de_combined33 = SPK2_Scale * SPK2_NLKV_3 * BA33 + ScaleNLKV_13cFdot * Fdot31_S + ScaleNLKV_23cFdot * Fdot32_S + ScaleNLKV_33cFdot * Fdot33_S + ScaleNLKV_13cF * F31_S + ScaleNLKV_23cF * F32_S + ScaleNLKV_33cF * F33_S;

        ArrayNIP_S T0 = (M4 * C22 + M5);
        ArrayNIP_S T11 = T0 * BA11 - M2 * F11_S + Partial_SPK2_NLKV_3_de_combined11;
        ArrayNIP_S T21 = T0 * BA21 - M2 * F12_S + Partial_SPK2_NLKV_3_de_combined21;
        ArrayNIP_S T31 = T0 * BA31 + Partial_SPK2_NLKV_3_de_combined31;

        ArrayNIP_S T12 = T0 * BA12 - M2 * F21_S + Partial_SPK2_NLKV_3_de_combined12;
        ArrayNIP_S T22 = T0 * BA22 - M2 * F22_S + Partial_SPK2_NLKV_3_de_combined22;
        ArrayNIP_S T32 = T0 * BA32 + Partial_SPK2_NLKV_3_de_combined32;

        ArrayNIP_S T13 = T0 * BA13 - M2 * F31_S + Partial_SPK2_NLKV_3_de_combined13;
        ArrayNIP_S T23 = T0 * BA23 - M2 * F32_S + Partial_SPK2_NLKV_3_de_combined23;
        ArrayNIP_S T33 = T0 * BA33 + Partial_SPK2_NLKV_3_de_combined33;

        for (auto i = 0; i < NSF; i++) {
            Right.block<1, NIP_S>(3 * i + 0, 0 + 2 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T11 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T21 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T31;
            Right.block<1, NIP_S>(3 * i + 1, 0 + 2 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T12 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T22 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T32;
            Right.block<1, NIP_S>(3 * i + 2, 0 + 2 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T13 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T23 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T33;
        }
    }

    //Reset M2 to it's correct value instead of double that was used above
    M2 *= 0.5;
    // Calculate the Contribution from the dC23/de terms
    {
        ArrayNIP_S ScaleNLKV_11cFdot = CInv12 * CInv13;
        ArrayNIP_S ScaleNLKV_22cFdot = CInv22 * CInv23;
        ArrayNIP_S ScaleNLKV_33cFdot = CInv23 * CInv33;
        ArrayNIP_S ScaleNLKV_12cFdot = 0.5 * (CInv12 * CInv23 + CInv22 * CInv13);
        ArrayNIP_S ScaleNLKV_13cFdot = 0.5 * (CInv12 * CInv33 + CInv13 * CInv23);
        ArrayNIP_S ScaleNLKV_23cFdot = 0.5 * (CInv23 * CInv23 + CInv22 * CInv33);
        ArrayNIP_S ScaleNLKV_11cF = 2.0 * KK * (C33 * EdotC2_CInvC3 + C22 * EdotC3_CInvC2 - C23 * (EdotC2_CInvC2 + EdotC3_CInvC3)) + KR * ScaleNLKV_11cFdot;
        ArrayNIP_S ScaleNLKV_22cF = 2.0 * KK * (C11 * EdotC3_CInvC2 - C13 * EdotC1_CInvC2) + KR * ScaleNLKV_22cFdot;
        ArrayNIP_S ScaleNLKV_33cF = 2.0 * KK * (C11 * EdotC2_CInvC3 - C12 * EdotC1_CInvC3) + KR * ScaleNLKV_33cFdot;
        ArrayNIP_S ScaleNLKV_12cF = KK * (C13 * (EdotC2_CInvC2 + EdotC3_CInvC3) + C23 * EdotC1_CInvC2 - 2.0 * C12 * EdotC3_CInvC2 - C33 * EdotC1_CInvC3) + KR * ScaleNLKV_12cFdot;
        ArrayNIP_S ScaleNLKV_13cF = KK * (C12 * (EdotC2_CInvC2 + EdotC3_CInvC3) + C23 * EdotC1_CInvC3 - 2.0 * C13 * EdotC2_CInvC3 - C22 * EdotC1_CInvC2) + KR * ScaleNLKV_13cFdot;
        ArrayNIP_S ScaleNLKV_23cF = KK * (C12 * EdotC1_CInvC2 + C13 * EdotC1_CInvC3 - C11 * (EdotC2_CInvC2 + EdotC3_CInvC3)) + KR * ScaleNLKV_23cFdot;
        ScaleNLKV_11cFdot *= KK;
        ScaleNLKV_22cFdot *= KK;
        ScaleNLKV_33cFdot *= KK;
        ScaleNLKV_12cFdot *= KK;
        ScaleNLKV_13cFdot *= KK;
        ScaleNLKV_23cFdot *= KK;

        ArrayNIP_S Partial_SPK2_NLKV_4_de_combined11 = SPK2_Scale * SPK2_NLKV_4 * BA22 + ScaleNLKV_11cFdot * Fdot11_S + ScaleNLKV_12cFdot * Fdot12_S + ScaleNLKV_13cFdot * Fdot13_S + ScaleNLKV_11cF * F11_S + ScaleNLKV_12cF * F12_S + ScaleNLKV_13cF * F13_S;
        ArrayNIP_S Partial_SPK2_NLKV_4_de_combined12 = SPK2_Scale * SPK2_NLKV_4 * BA23 + ScaleNLKV_11cFdot * Fdot21_S + ScaleNLKV_12cFdot * Fdot22_S + ScaleNLKV_13cFdot * Fdot23_S + ScaleNLKV_11cF * F21_S + ScaleNLKV_12cF * F22_S + ScaleNLKV_13cF * F23_S;
        ArrayNIP_S Partial_SPK2_NLKV_4_de_combined13 = SPK2_Scale * SPK2_NLKV_4 * BA13 + ScaleNLKV_11cFdot * Fdot31_S + ScaleNLKV_12cFdot * Fdot32_S + ScaleNLKV_13cFdot * Fdot33_S + ScaleNLKV_11cF * F31_S + ScaleNLKV_12cF * F32_S + ScaleNLKV_13cF * F33_S;
        ArrayNIP_S Partial_SPK2_NLKV_4_de_combined21 = SPK2_Scale * SPK2_NLKV_4 * BA32 + ScaleNLKV_12cFdot * Fdot11_S + ScaleNLKV_22cFdot * Fdot12_S + ScaleNLKV_23cFdot * Fdot13_S + ScaleNLKV_12cF * F11_S + ScaleNLKV_22cF * F12_S + ScaleNLKV_23cF * F13_S;
        ArrayNIP_S Partial_SPK2_NLKV_4_de_combined22 = SPK2_Scale * SPK2_NLKV_4 * BA33 + ScaleNLKV_12cFdot * Fdot21_S + ScaleNLKV_22cFdot * Fdot22_S + ScaleNLKV_23cFdot * Fdot23_S + ScaleNLKV_12cF * F21_S + ScaleNLKV_22cF * F22_S + ScaleNLKV_23cF * F23_S;
        ArrayNIP_S Partial_SPK2_NLKV_4_de_combined23 = SPK2_Scale * SPK2_NLKV_4 * BA23 + ScaleNLKV_12cFdot * Fdot31_S + ScaleNLKV_22cFdot * Fdot32_S + ScaleNLKV_23cFdot * Fdot33_S + ScaleNLKV_12cF * F31_S + ScaleNLKV_22cF * F32_S + ScaleNLKV_23cF * F33_S;
        ArrayNIP_S Partial_SPK2_NLKV_4_de_combined31 = SPK2_Scale * SPK2_NLKV_4 * BA31 + ScaleNLKV_13cFdot * Fdot11_S + ScaleNLKV_23cFdot * Fdot12_S + ScaleNLKV_33cFdot * Fdot13_S + ScaleNLKV_13cF * F11_S + ScaleNLKV_23cF * F12_S + ScaleNLKV_33cF * F13_S;
        ArrayNIP_S Partial_SPK2_NLKV_4_de_combined32 = SPK2_Scale * SPK2_NLKV_4 * BA32 + ScaleNLKV_13cFdot * Fdot21_S + ScaleNLKV_23cFdot * Fdot22_S + ScaleNLKV_33cFdot * Fdot23_S + ScaleNLKV_13cF * F21_S + ScaleNLKV_23cF * F22_S + ScaleNLKV_33cF * F23_S;
        ArrayNIP_S Partial_SPK2_NLKV_4_de_combined33 = SPK2_Scale * SPK2_NLKV_4 * BA33 + ScaleNLKV_13cFdot * Fdot31_S + ScaleNLKV_23cFdot * Fdot32_S + ScaleNLKV_33cFdot * Fdot33_S + ScaleNLKV_13cF * F31_S + ScaleNLKV_23cF * F32_S + ScaleNLKV_33cF * F33_S;

        ArrayNIP_S T0 = M4 * C23;
        ArrayNIP_S T11 = T0 * BA11 + Partial_SPK2_NLKV_4_de_combined11;
        ArrayNIP_S T21 = T0 * BA21 + M2 * F13_S + Partial_SPK2_NLKV_4_de_combined21;
        ArrayNIP_S T31 = T0 * BA31 + M2 * F12_S + Partial_SPK2_NLKV_4_de_combined31;

        ArrayNIP_S T12 = T0 * BA12 + Partial_SPK2_NLKV_4_de_combined12;
        ArrayNIP_S T22 = T0 * BA22 + M2 * F23_S + Partial_SPK2_NLKV_4_de_combined22;
        ArrayNIP_S T32 = T0 * BA32 + M2 * F22_S + Partial_SPK2_NLKV_4_de_combined32;

        ArrayNIP_S T13 = T0 * BA13 + Partial_SPK2_NLKV_4_de_combined13;
        ArrayNIP_S T23 = T0 * BA23 + M2 * F33_S + Partial_SPK2_NLKV_4_de_combined23;
        ArrayNIP_S T33 = T0 * BA33 + M2 * F32_S + Partial_SPK2_NLKV_4_de_combined33;

        for (auto i = 0; i < NSF; i++) {
            Right.block<1, NIP_S>(3 * i + 0, 0 + 3 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T11 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T21 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T31;
            Right.block<1, NIP_S>(3 * i + 1, 0 + 3 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T12 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T22 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T32;
            Right.block<1, NIP_S>(3 * i + 2, 0 + 3 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T13 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T23 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T33;
        }
    }
    // Calculate the Contribution from the dC13/de terms
    {
        ArrayNIP_S ScaleNLKV_11cFdot = CInv11 * CInv13;
        ArrayNIP_S ScaleNLKV_22cFdot = CInv12 * CInv23;
        ArrayNIP_S ScaleNLKV_33cFdot = CInv13 * CInv33;
        ArrayNIP_S ScaleNLKV_12cFdot = 0.5 * (CInv11 * CInv23 + CInv12 * CInv13);
        ArrayNIP_S ScaleNLKV_13cFdot = 0.5 * (CInv11 * CInv33 + CInv13 * CInv13);
        ArrayNIP_S ScaleNLKV_23cFdot = 0.5 * (CInv12 * CInv33 + CInv13 * CInv23);
        ArrayNIP_S ScaleNLKV_11cF = 2.0 * KK * (C22 * EdotC3_CInvC1 - C23 * EdotC2_CInvC1) + KR * ScaleNLKV_11cFdot;
        ArrayNIP_S ScaleNLKV_22cF = 2.0 * KK * (C11 * EdotC3_CInvC1 + C33 * EdotC1_CInvC3 - C13 * (EdotC1_CInvC1 + EdotC3_CInvC3)) + KR * ScaleNLKV_22cFdot;
        ArrayNIP_S ScaleNLKV_33cF = 2.0 * KK * (C22 * EdotC1_CInvC3 - C12 * EdotC2_CInvC3) + KR * ScaleNLKV_33cFdot;
        ArrayNIP_S ScaleNLKV_12cF = KK * (C13 * EdotC2_CInvC1 + C23 * (EdotC1_CInvC1 + EdotC3_CInvC3) - 2.0 * C12 * EdotC3_CInvC1 - C33 * EdotC2_CInvC3) + KR * ScaleNLKV_12cFdot;
        ArrayNIP_S ScaleNLKV_13cF = KK * (C12 * EdotC2_CInvC1 + C23 * EdotC2_CInvC3 - C22 * (EdotC1_CInvC1 + EdotC3_CInvC3)) + KR * ScaleNLKV_13cFdot;
        ArrayNIP_S ScaleNLKV_23cF = KK * (C12 * (EdotC1_CInvC1 + EdotC3_CInvC3) + C13 * EdotC2_CInvC3 - 2.0 * C23 * EdotC1_CInvC3 - C11 * EdotC2_CInvC1) + KR * ScaleNLKV_23cFdot;
        ScaleNLKV_11cFdot *= KK;
        ScaleNLKV_22cFdot *= KK;
        ScaleNLKV_33cFdot *= KK;
        ScaleNLKV_12cFdot *= KK;
        ScaleNLKV_13cFdot *= KK;
        ScaleNLKV_23cFdot *= KK;

        ArrayNIP_S Partial_SPK2_NLKV_5_de_combined11 = SPK2_Scale * SPK2_NLKV_5 * BA22 + ScaleNLKV_11cFdot * Fdot11_S + ScaleNLKV_12cFdot * Fdot12_S + ScaleNLKV_13cFdot * Fdot13_S + ScaleNLKV_11cF * F11_S + ScaleNLKV_12cF * F12_S + ScaleNLKV_13cF * F13_S;
        ArrayNIP_S Partial_SPK2_NLKV_5_de_combined12 = SPK2_Scale * SPK2_NLKV_5 * BA23 + ScaleNLKV_11cFdot * Fdot21_S + ScaleNLKV_12cFdot * Fdot22_S + ScaleNLKV_13cFdot * Fdot23_S + ScaleNLKV_11cF * F21_S + ScaleNLKV_12cF * F22_S + ScaleNLKV_13cF * F23_S;
        ArrayNIP_S Partial_SPK2_NLKV_5_de_combined13 = SPK2_Scale * SPK2_NLKV_5 * BA13 + ScaleNLKV_11cFdot * Fdot31_S + ScaleNLKV_12cFdot * Fdot32_S + ScaleNLKV_13cFdot * Fdot33_S + ScaleNLKV_11cF * F31_S + ScaleNLKV_12cF * F32_S + ScaleNLKV_13cF * F33_S;
        ArrayNIP_S Partial_SPK2_NLKV_5_de_combined21 = SPK2_Scale * SPK2_NLKV_5 * BA32 + ScaleNLKV_12cFdot * Fdot11_S + ScaleNLKV_22cFdot * Fdot12_S + ScaleNLKV_23cFdot * Fdot13_S + ScaleNLKV_12cF * F11_S + ScaleNLKV_22cF * F12_S + ScaleNLKV_23cF * F13_S;
        ArrayNIP_S Partial_SPK2_NLKV_5_de_combined22 = SPK2_Scale * SPK2_NLKV_5 * BA33 + ScaleNLKV_12cFdot * Fdot21_S + ScaleNLKV_22cFdot * Fdot22_S + ScaleNLKV_23cFdot * Fdot23_S + ScaleNLKV_12cF * F21_S + ScaleNLKV_22cF * F22_S + ScaleNLKV_23cF * F23_S;
        ArrayNIP_S Partial_SPK2_NLKV_5_de_combined23 = SPK2_Scale * SPK2_NLKV_5 * BA23 + ScaleNLKV_12cFdot * Fdot31_S + ScaleNLKV_22cFdot * Fdot32_S + ScaleNLKV_23cFdot * Fdot33_S + ScaleNLKV_12cF * F31_S + ScaleNLKV_22cF * F32_S + ScaleNLKV_23cF * F33_S;
        ArrayNIP_S Partial_SPK2_NLKV_5_de_combined31 = SPK2_Scale * SPK2_NLKV_5 * BA31 + ScaleNLKV_13cFdot * Fdot11_S + ScaleNLKV_23cFdot * Fdot12_S + ScaleNLKV_33cFdot * Fdot13_S + ScaleNLKV_13cF * F11_S + ScaleNLKV_23cF * F12_S + ScaleNLKV_33cF * F13_S;
        ArrayNIP_S Partial_SPK2_NLKV_5_de_combined32 = SPK2_Scale * SPK2_NLKV_5 * BA32 + ScaleNLKV_13cFdot * Fdot21_S + ScaleNLKV_23cFdot * Fdot22_S + ScaleNLKV_33cFdot * Fdot23_S + ScaleNLKV_13cF * F21_S + ScaleNLKV_23cF * F22_S + ScaleNLKV_33cF * F23_S;
        ArrayNIP_S Partial_SPK2_NLKV_5_de_combined33 = SPK2_Scale * SPK2_NLKV_5 * BA33 + ScaleNLKV_13cFdot * Fdot31_S + ScaleNLKV_23cFdot * Fdot32_S + ScaleNLKV_33cFdot * Fdot33_S + ScaleNLKV_13cF * F31_S + ScaleNLKV_23cF * F32_S + ScaleNLKV_33cF * F33_S;

        ArrayNIP_S T0 = M4 * C13;
        ArrayNIP_S T11 = T0 * BA11 + M2 * F13_S + Partial_SPK2_NLKV_5_de_combined11;
        ArrayNIP_S T21 = T0 * BA21 + Partial_SPK2_NLKV_5_de_combined21;
        ArrayNIP_S T31 = T0 * BA31 + M2 * F11_S + Partial_SPK2_NLKV_5_de_combined31;

        ArrayNIP_S T12 = T0 * BA12 + M2 * F23_S + Partial_SPK2_NLKV_5_de_combined12;
        ArrayNIP_S T22 = T0 * BA22 + Partial_SPK2_NLKV_5_de_combined22;
        ArrayNIP_S T32 = T0 * BA32 + M2 * F21_S + Partial_SPK2_NLKV_5_de_combined32;

        ArrayNIP_S T13 = T0 * BA13 + M2 * F33_S + Partial_SPK2_NLKV_5_de_combined13;
        ArrayNIP_S T23 = T0 * BA23 + Partial_SPK2_NLKV_5_de_combined23;
        ArrayNIP_S T33 = T0 * BA33 + M2 * F31_S + Partial_SPK2_NLKV_5_de_combined33;

        for (auto i = 0; i < NSF; i++) {
            Right.block<1, NIP_S>(3 * i + 0, 0 + 4 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T11 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T21 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T31;
            Right.block<1, NIP_S>(3 * i + 1, 0 + 4 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T12 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T22 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T32;
            Right.block<1, NIP_S>(3 * i + 2, 0 + 4 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T13 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T23 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T33;
        }
    }
    // Calculate the Contribution from the dC12/de terms
    {
        ArrayNIP_S ScaleNLKV_11cFdot = CInv11 * CInv12;
        ArrayNIP_S ScaleNLKV_22cFdot = CInv12 * CInv22;
        ArrayNIP_S ScaleNLKV_33cFdot = CInv13 * CInv23;
        ArrayNIP_S ScaleNLKV_12cFdot = 0.5 * (CInv11 * CInv22 + CInv12 * CInv12);
        ArrayNIP_S ScaleNLKV_13cFdot = 0.5 * (CInv11 * CInv23 + CInv12 * CInv13);
        ArrayNIP_S ScaleNLKV_23cFdot = 0.5 * (CInv12 * CInv23 + CInv22 * CInv13);
        ArrayNIP_S ScaleNLKV_11cF = 2.0 * KK * (C33 * EdotC2_CInvC1 - C23 * EdotC3_CInvC1) + KR * ScaleNLKV_11cFdot;
        ArrayNIP_S ScaleNLKV_22cF = 2.0 * KK * (C33 * EdotC1_CInvC2 - C13 * EdotC3_CInvC2) + KR * ScaleNLKV_22cFdot;
        ArrayNIP_S ScaleNLKV_33cF = 2.0 * KK * (C11 * EdotC2_CInvC1 + C22 * EdotC1_CInvC2 - C12 * (EdotC1_CInvC1 + EdotC2_CInvC2)) + KR * ScaleNLKV_33cFdot;
        ArrayNIP_S ScaleNLKV_12cF = KK * (C13 * EdotC3_CInvC1 + C23 * EdotC3_CInvC2 - C33 * (EdotC1_CInvC1 + EdotC2_CInvC2)) + KR * ScaleNLKV_12cFdot;
        ArrayNIP_S ScaleNLKV_13cF = KK * (C12 * EdotC3_CInvC1 + C23 * (EdotC1_CInvC1 + EdotC2_CInvC2) - 2.0 * C13 * EdotC2_CInvC1 - C22 * EdotC3_CInvC2) + KR * ScaleNLKV_13cFdot;
        ArrayNIP_S ScaleNLKV_23cF = KK * (C12 * EdotC3_CInvC2 + C13 * (EdotC1_CInvC1 + EdotC2_CInvC2) - 2.0 * C23 * EdotC1_CInvC2 - C11 * EdotC3_CInvC1) + KR * ScaleNLKV_23cFdot;
        ScaleNLKV_11cFdot *= KK;
        ScaleNLKV_22cFdot *= KK;
        ScaleNLKV_33cFdot *= KK;
        ScaleNLKV_12cFdot *= KK;
        ScaleNLKV_13cFdot *= KK;
        ScaleNLKV_23cFdot *= KK;

        ArrayNIP_S Partial_SPK2_NLKV_6_de_combined11 = SPK2_Scale * SPK2_NLKV_6 * BA22 + ScaleNLKV_11cFdot * Fdot11_S + ScaleNLKV_12cFdot * Fdot12_S + ScaleNLKV_13cFdot * Fdot13_S + ScaleNLKV_11cF * F11_S + ScaleNLKV_12cF * F12_S + ScaleNLKV_13cF * F13_S;
        ArrayNIP_S Partial_SPK2_NLKV_6_de_combined12 = SPK2_Scale * SPK2_NLKV_6 * BA23 + ScaleNLKV_11cFdot * Fdot21_S + ScaleNLKV_12cFdot * Fdot22_S + ScaleNLKV_13cFdot * Fdot23_S + ScaleNLKV_11cF * F21_S + ScaleNLKV_12cF * F22_S + ScaleNLKV_13cF * F23_S;
        ArrayNIP_S Partial_SPK2_NLKV_6_de_combined13 = SPK2_Scale * SPK2_NLKV_6 * BA13 + ScaleNLKV_11cFdot * Fdot31_S + ScaleNLKV_12cFdot * Fdot32_S + ScaleNLKV_13cFdot * Fdot33_S + ScaleNLKV_11cF * F31_S + ScaleNLKV_12cF * F32_S + ScaleNLKV_13cF * F33_S;
        ArrayNIP_S Partial_SPK2_NLKV_6_de_combined21 = SPK2_Scale * SPK2_NLKV_6 * BA32 + ScaleNLKV_12cFdot * Fdot11_S + ScaleNLKV_22cFdot * Fdot12_S + ScaleNLKV_23cFdot * Fdot13_S + ScaleNLKV_12cF * F11_S + ScaleNLKV_22cF * F12_S + ScaleNLKV_23cF * F13_S;
        ArrayNIP_S Partial_SPK2_NLKV_6_de_combined22 = SPK2_Scale * SPK2_NLKV_6 * BA33 + ScaleNLKV_12cFdot * Fdot21_S + ScaleNLKV_22cFdot * Fdot22_S + ScaleNLKV_23cFdot * Fdot23_S + ScaleNLKV_12cF * F21_S + ScaleNLKV_22cF * F22_S + ScaleNLKV_23cF * F23_S;
        ArrayNIP_S Partial_SPK2_NLKV_6_de_combined23 = SPK2_Scale * SPK2_NLKV_6 * BA23 + ScaleNLKV_12cFdot * Fdot31_S + ScaleNLKV_22cFdot * Fdot32_S + ScaleNLKV_23cFdot * Fdot33_S + ScaleNLKV_12cF * F31_S + ScaleNLKV_22cF * F32_S + ScaleNLKV_23cF * F33_S;
        ArrayNIP_S Partial_SPK2_NLKV_6_de_combined31 = SPK2_Scale * SPK2_NLKV_6 * BA31 + ScaleNLKV_13cFdot * Fdot11_S + ScaleNLKV_23cFdot * Fdot12_S + ScaleNLKV_33cFdot * Fdot13_S + ScaleNLKV_13cF * F11_S + ScaleNLKV_23cF * F12_S + ScaleNLKV_33cF * F13_S;
        ArrayNIP_S Partial_SPK2_NLKV_6_de_combined32 = SPK2_Scale * SPK2_NLKV_6 * BA32 + ScaleNLKV_13cFdot * Fdot21_S + ScaleNLKV_23cFdot * Fdot22_S + ScaleNLKV_33cFdot * Fdot23_S + ScaleNLKV_13cF * F21_S + ScaleNLKV_23cF * F22_S + ScaleNLKV_33cF * F23_S;
        ArrayNIP_S Partial_SPK2_NLKV_6_de_combined33 = SPK2_Scale * SPK2_NLKV_6 * BA33 + ScaleNLKV_13cFdot * Fdot31_S + ScaleNLKV_23cFdot * Fdot32_S + ScaleNLKV_33cFdot * Fdot33_S + ScaleNLKV_13cF * F31_S + ScaleNLKV_23cF * F32_S + ScaleNLKV_33cF * F33_S;

        ArrayNIP_S T0 = M4 * C13;
        ArrayNIP_S T11 = T0 * BA11 + M2 * F12_S + Partial_SPK2_NLKV_6_de_combined11;
        ArrayNIP_S T21 = T0 * BA21 + M2 * F11_S + Partial_SPK2_NLKV_6_de_combined21;
        ArrayNIP_S T31 = T0 * BA31 + Partial_SPK2_NLKV_6_de_combined31;

        ArrayNIP_S T12 = T0 * BA12 + M2 * F22_S + Partial_SPK2_NLKV_6_de_combined12;
        ArrayNIP_S T22 = T0 * BA22 + M2 * F21_S + Partial_SPK2_NLKV_6_de_combined22;
        ArrayNIP_S T32 = T0 * BA32 + Partial_SPK2_NLKV_6_de_combined32;

        ArrayNIP_S T13 = T0 * BA13 + M2 * F32_S + Partial_SPK2_NLKV_6_de_combined13;
        ArrayNIP_S T23 = T0 * BA23 + M2 * F31_S + Partial_SPK2_NLKV_6_de_combined23;
        ArrayNIP_S T33 = T0 * BA33 + Partial_SPK2_NLKV_6_de_combined33;

        for (auto i = 0; i < NSF; i++) {
            Right.block<1, NIP_S>(3 * i + 0, 0 + 5 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T11 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T21 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T31;
            Right.block<1, NIP_S>(3 * i + 1, 0 + 5 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T12 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T22 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T32;
            Right.block<1, NIP_S>(3 * i + 2, 0 + 5 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T13 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T23 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T33;
        }
    }


    // Calculate the Contribution from the ddetF/de terms - No Penalty Terms
    {
        ArrayNIP_S M6 = (5.0 / 9.0*I1*M0 - 14.0 / 9.0*M2*I2) / (J*J);

        ArrayNIP_S T11 = M4 * (C11*F11_S + C12 * F12_S + C13 * F13_S) + M5 * F11_S + M6 * BA11;
        ArrayNIP_S T21 = M4 * (C12*F11_S + C22 * F12_S + C23 * F13_S) + M5 * F12_S + M6 * BA21;
        ArrayNIP_S T31 = M4 * (C13*F11_S + C23 * F12_S + C33 * F13_S) + M5 * F13_S + M6 * BA31;
        ArrayNIP_S T12 = M4 * (C11*F21_S + C12 * F22_S + C13 * F23_S) + M5 * F21_S + M6 * BA12;
        ArrayNIP_S T22 = M4 * (C12*F21_S + C22 * F22_S + C23 * F23_S) + M5 * F22_S + M6 * BA22;
        ArrayNIP_S T32 = M4 * (C13*F21_S + C23 * F22_S + C33 * F23_S) + M5 * F23_S + M6 * BA32;
        ArrayNIP_S T13 = M4 * (C11*F31_S + C12 * F32_S + C13 * F33_S) + M5 * F31_S + M6 * BA13;
        ArrayNIP_S T23 = M4 * (C12*F31_S + C22 * F32_S + C23 * F33_S) + M5 * F32_S + M6 * BA23;
        ArrayNIP_S T33 = M4 * (C13*F31_S + C23 * F32_S + C33 * F33_S) + M5 * F33_S + M6 * BA33;

        for (auto i = 0; i < NSF; i++) {
            Right.block<1, NIP_S>(3 * i + 0, 0 + 6 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T11 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T21 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T31;
            Right.block<1, NIP_S>(3 * i + 1, 0 + 6 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T12 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T22 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T32;
            Right.block<1, NIP_S>(3 * i + 2, 0 + 6 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T13 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T23 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T33;
        }
    }
    // Calculate the Contribution from the ddetF/de terms - Penalty Terms
    {
        ArrayNIP_P M6 = -Kfactor * k * m_kGQ_P;

        ArrayNIP_P T11 = M6 * BA11_P;
        ArrayNIP_P T21 = M6 * BA21_P;
        ArrayNIP_P T31 = M6 * BA31_P;

        ArrayNIP_P T12 = M6 * BA12_P;
        ArrayNIP_P T22 = M6 * BA22_P;
        ArrayNIP_P T32 = M6 * BA32_P;

        ArrayNIP_P T13 = M6 * BA13_P;
        ArrayNIP_P T23 = M6 * BA23_P;
        ArrayNIP_P T33 = M6 * BA33_P;

        for (auto i = 0; i < NSF; i++) {
            Right.block<1, NIP_P>(3 * i + 0, 0 + 7 * NIP_S).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 0 * NIP_P).array() * T11 +
                m_SD.block<1, NIP_P>(i, 3 * NIP_S + 1 * NIP_P).array() * T21 +
                m_SD.block<1, NIP_P>(i, 3 * NIP_S + 2 * NIP_P).array() * T31;
            Right.block<1, NIP_P>(3 * i + 1, 0 + 7 * NIP_S).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 0 * NIP_P).array() * T12 +
                m_SD.block<1, NIP_P>(i, 3 * NIP_S + 1 * NIP_P).array() * T22 +
                m_SD.block<1, NIP_P>(i, 3 * NIP_S + 2 * NIP_P).array() * T32;
            Right.block<1, NIP_P>(3 * i + 2, 0 + 7 * NIP_S).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 0 * NIP_P).array() * T13 +
                m_SD.block<1, NIP_P>(i, 3 * NIP_S + 1 * NIP_P).array() * T23 +
                m_SD.block<1, NIP_P>(i, 3 * NIP_S + 2 * NIP_P).array() * T33;
        }
    }
    Matrix3Nx3N Jac = Left * Right.transpose();

    //Calculate the contribution from the Mass Matrix and Expand/Expandij components
    ArrayNIP_S M1 = M0 - M2 * I1;
    ArrayNIP_S M3 = (2*I2*M2 - I1*M0) / (3.0 * J);

    ArrayNIP_S MC11 = M1 + M2 * C11 + SPK2_NLKV_1;
    ArrayNIP_S MC22 = M1 + M2 * C22 + SPK2_NLKV_2;
    ArrayNIP_S MC33 = M1 + M2 * C33 + SPK2_NLKV_3;
    ArrayNIP_S MC12 = M2 * C12 + SPK2_NLKV_6;
    ArrayNIP_S MC13 = M2 * C13 + SPK2_NLKV_5;
    ArrayNIP_S MC23 = M2 * C23 + SPK2_NLKV_4;

    ArrayNIP_S MF11_S = M3 * F11_S;
    ArrayNIP_S MF12_S = M3 * F12_S;
    ArrayNIP_S MF13_S = M3 * F13_S;
    ArrayNIP_S MF21_S = M3 * F21_S;
    ArrayNIP_S MF22_S = M3 * F22_S;
    ArrayNIP_S MF23_S = M3 * F23_S;
    ArrayNIP_S MF31_S = M3 * F31_S;
    ArrayNIP_S MF32_S = M3 * F32_S;
    ArrayNIP_S MF33_S = M3 * F33_S;

    //Calculate the determinate of F (commonly denoted as J) for the volumentric penalty Gauss quadrature points
    ArrayNIP_P J_P = F11_P * (F22_P*F33_P - F23_P * F32_P) + F12_P * (F23_P*F31_P - F21_P * F33_P) + F13_P * (F21_P*F32_P - F22_P * F31_P);
    ArrayNIP_P M3_P = (-Kfactor * k) * (J_P - 1.0)*m_kGQ_P;
    ArrayNIP_P MF11_P = M3_P * F11_P;
    ArrayNIP_P MF12_P = M3_P * F12_P;
    ArrayNIP_P MF13_P = M3_P * F13_P;
    ArrayNIP_P MF21_P = M3_P * F21_P;
    ArrayNIP_P MF22_P = M3_P * F22_P;
    ArrayNIP_P MF23_P = M3_P * F23_P;
    ArrayNIP_P MF31_P = M3_P * F31_P;
    ArrayNIP_P MF32_P = M3_P * F32_P;
    ArrayNIP_P MF33_P = M3_P * F33_P;

    unsigned int idx = 0;
    for (unsigned int i = 0; i < (NSF - 1); i++) {
        //Calculate the scaled row of SD
        ChVectorN<double, 3 * NIP_S> R;
        R.segment(0 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * MC11 +
            m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * MC12 +
            m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * MC13;
        R.segment(1 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * MC12 +
            m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * MC22 +
            m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * MC23;
        R.segment(2 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * MC13 +
            m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * MC23 +
            m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * MC33;

        //Calculate the scaled row of B2 - Non Penalty Terms
        ChVectorN<double, 3 * NIP_S> X_S;
        X_S.segment(0 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(i, 1 * NIP_S).array()*MF33_S - m_SD.block<1, NIP_S>(i, 2 * NIP_S).array()*MF32_S;
        X_S.segment(1 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(i, 2 * NIP_S).array()*MF31_S - m_SD.block<1, NIP_S>(i, 0 * NIP_S).array()*MF33_S;
        X_S.segment(2 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array()*MF32_S - m_SD.block<1, NIP_S>(i, 1 * NIP_S).array()*MF31_S;

        //Calculate the scaled row of B3 - Non Penalty Terms
        ChVectorN<double, 3 * NIP_S> Y_S;
        Y_S.segment(0 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(i, 2 * NIP_S).array()*MF22_S - m_SD.block<1, NIP_S>(i, 1 * NIP_S).array()*MF23_S;
        Y_S.segment(1 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array()*MF23_S - m_SD.block<1, NIP_S>(i, 2 * NIP_S).array()*MF21_S;
        Y_S.segment(2 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(i, 1 * NIP_S).array()*MF21_S - m_SD.block<1, NIP_S>(i, 0 * NIP_S).array()*MF22_S;

        //Calculate the scaled row of B4 - Non Penalty Terms
        ChVectorN<double, 3 * NIP_S> Z_S;
        Z_S.segment(0 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(i, 1 * NIP_S).array()*MF13_S - m_SD.block<1, NIP_S>(i, 2 * NIP_S).array()*MF12_S;
        Z_S.segment(1 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(i, 2 * NIP_S).array()*MF11_S - m_SD.block<1, NIP_S>(i, 0 * NIP_S).array()*MF13_S;
        Z_S.segment(2 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array()*MF12_S - m_SD.block<1, NIP_S>(i, 1 * NIP_S).array()*MF11_S;

        //Calculate the scaled row of B2 - Penalty Terms
        ChVectorN<double, 3 * NIP_P> X_P;
        X_P.segment(0 * NIP_P, NIP_P).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 1 * NIP_P).array()*MF33_P - m_SD.block<1, NIP_P>(i, 3 * NIP_S + 2 * NIP_P).array()*MF32_P;
        X_P.segment(1 * NIP_P, NIP_P).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 2 * NIP_P).array()*MF31_P - m_SD.block<1, NIP_P>(i, 3 * NIP_S + 0 * NIP_P).array()*MF33_P;
        X_P.segment(2 * NIP_P, NIP_P).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 0 * NIP_P).array()*MF32_P - m_SD.block<1, NIP_P>(i, 3 * NIP_S + 1 * NIP_P).array()*MF31_P;

        //Calculate the scaled row of B3 - Penalty Terms
        ChVectorN<double, 3 * NIP_P> Y_P;
        Y_P.segment(0 * NIP_P, NIP_P).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 2 * NIP_P).array()*MF22_P - m_SD.block<1, NIP_P>(i, 3 * NIP_S + 1 * NIP_P).array()*MF23_P;
        Y_P.segment(1 * NIP_P, NIP_P).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 0 * NIP_P).array()*MF23_P - m_SD.block<1, NIP_P>(i, 3 * NIP_S + 2 * NIP_P).array()*MF21_P;
        Y_P.segment(2 * NIP_P, NIP_P).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 1 * NIP_P).array()*MF21_P - m_SD.block<1, NIP_P>(i, 3 * NIP_S + 0 * NIP_P).array()*MF22_P;

        //Calculate the scaled row of B4 - Penalty Terms
        ChVectorN<double, 3 * NIP_P> Z_P;
        Z_P.segment(0 * NIP_P, NIP_P).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 1 * NIP_P).array()*MF13_P - m_SD.block<1, NIP_P>(i, 3 * NIP_S + 2 * NIP_P).array()*MF12_P;
        Z_P.segment(1 * NIP_P, NIP_P).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 2 * NIP_P).array()*MF11_P - m_SD.block<1, NIP_P>(i, 3 * NIP_S + 0 * NIP_P).array()*MF13_P;
        Z_P.segment(2 * NIP_P, NIP_P).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 0 * NIP_P).array()*MF12_P - m_SD.block<1, NIP_P>(i, 3 * NIP_S + 1 * NIP_P).array()*MF11_P;

        //double d_diag = Mfactor * m_MassMatrix(idx) + (scaled_SD_row_i.dot(m_SD.row(i)));
        double d_diag = Mfactor * m_MassMatrix(idx) + (R.dot(m_SD.block<1, 3 * NIP_S>(i, 0)));

        Jac(3 * i, 3 * i) += d_diag;
        Jac(3 * i + 1, 3 * i + 1) += d_diag;
        Jac(3 * i + 2, 3 * i + 2) += d_diag;
        idx++;

        for (unsigned int j = (i + 1); j < NSF; j++) {
            double d = Mfactor * m_MassMatrix(idx) + (R.dot(m_SD.block<1, 3 * NIP_S>(j, 0)));
            double B2 = X_S.dot(m_SD.block<1, 3 * NIP_S>(j, 0)) + X_P.dot(m_SD.block<1, 3 * NIP_P>(j, 3 * NIP_S));
            double B3 = Y_S.dot(m_SD.block<1, 3 * NIP_S>(j, 0)) + Y_P.dot(m_SD.block<1, 3 * NIP_P>(j, 3 * NIP_S));
            double B4 = Z_S.dot(m_SD.block<1, 3 * NIP_S>(j, 0)) + Z_P.dot(m_SD.block<1, 3 * NIP_P>(j, 3 * NIP_S));

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
    ChVectorN<double, 3 * NIP_S> R;
    R.segment(0 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(NSF - 1, 0 * NIP_S).array() * MC11 +
        m_SD.block<1, NIP_S>(NSF - 1, 1 * NIP_S).array() * MC12 +
        m_SD.block<1, NIP_S>(NSF - 1, 2 * NIP_S).array() * MC13;
    R.segment(1 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(NSF - 1, 0 * NIP_S).array() * MC12 +
        m_SD.block<1, NIP_S>(NSF - 1, 1 * NIP_S).array() * MC22 +
        m_SD.block<1, NIP_S>(NSF - 1, 2 * NIP_S).array() * MC23;
    R.segment(2 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(NSF - 1, 0 * NIP_S).array() * MC13 +
        m_SD.block<1, NIP_S>(NSF - 1, 1 * NIP_S).array() * MC23 +
        m_SD.block<1, NIP_S>(NSF - 1, 2 * NIP_S).array() * MC33;

    //double d_diag = Mfactor * m_MassMatrix(idx) + (scaled_SD_row_i.dot(m_SD.row(NSF - 1)));
    double d_diag = Mfactor * m_MassMatrix(idx) + (R.dot(m_SD.block<1, 3 * NIP_S>(NSF - 1, 0)));

    Jac(3 * (NSF - 1) + 0, 3 * (NSF - 1) + 0) += d_diag;
    Jac(3 * (NSF - 1) + 1, 3 * (NSF - 1) + 1) += d_diag;
    Jac(3 * (NSF - 1) + 2, 3 * (NSF - 1) + 2) += d_diag;

    H.noalias() = Jac;
}

void ChElementBeamANCF_3243_MR_Damp::ComputeInternalJacobianNoDamping(ChMatrixRef H, double Kfactor, double Mfactor) {
    assert((H.rows() == 3 * NSF) && (H.cols() == 3 * NSF));

    // Element coordinates in matrix form
    Matrix3xN ebar;
    CalcCoordMatrix(ebar);

    // =============================================================================
    // Calculate the deformation gradient for all Gauss quadrature
    // points in a single matrix multiplication.  Note that since the shape function derivative matrix is ordered by
    // columns, the resulting deformation gradient will be ordered by block matrix (column vectors) components
    // Note that the indices of the components are in transposed order
    // S = No Penalty Terms, P = Penalty terms integrated just along the beam axis
    //      [F11_S     F12_S     F13_S     F11_P     F12_P     F13_P ]
    // FC = [F21_S     F22_S     F23_S     F21_P     F22_P     F23_P ]
    //      [F31_S     F32_S     F33_S     F31_P     F32_P     F33_P ]
    // =============================================================================

    ChMatrixNM<double, 3, 3 * NIP> FC = ebar * m_SD;

    Eigen::Map<ArrayNIP_S> F11_S(FC.block<1, NIP_S>(0, 0 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> F12_S(FC.block<1, NIP_S>(0, 1 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> F13_S(FC.block<1, NIP_S>(0, 2 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_P> F11_P(FC.block<1, NIP_P>(0, 3 * NIP_S + 0 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> F12_P(FC.block<1, NIP_P>(0, 3 * NIP_S + 1 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> F13_P(FC.block<1, NIP_P>(0, 3 * NIP_S + 2 * NIP_P).data(), 1, NIP_P);

    Eigen::Map<ArrayNIP_S> F21_S(FC.block<1, NIP_S>(1, 0 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> F22_S(FC.block<1, NIP_S>(1, 1 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> F23_S(FC.block<1, NIP_S>(1, 2 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_P> F21_P(FC.block<1, NIP_P>(1, 3 * NIP_S + 0 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> F22_P(FC.block<1, NIP_P>(1, 3 * NIP_S + 1 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> F23_P(FC.block<1, NIP_P>(1, 3 * NIP_S + 2 * NIP_P).data(), 1, NIP_P);

    Eigen::Map<ArrayNIP_S> F31_S(FC.block<1, NIP_S>(2, 0 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> F32_S(FC.block<1, NIP_S>(2, 1 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_S> F33_S(FC.block<1, NIP_S>(2, 2 * NIP_S + 0 * NIP_P).data(), 1, NIP_S);
    Eigen::Map<ArrayNIP_P> F31_P(FC.block<1, NIP_P>(2, 3 * NIP_S + 0 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> F32_P(FC.block<1, NIP_P>(2, 3 * NIP_S + 1 * NIP_P).data(), 1, NIP_P);
    Eigen::Map<ArrayNIP_P> F33_P(FC.block<1, NIP_P>(2, 3 * NIP_S + 2 * NIP_P).data(), 1, NIP_P);

    ArrayNIP_S BA11 = F22_S * F33_S - F23_S * F32_S;
    ArrayNIP_S BA21 = F23_S * F31_S - F21_S * F33_S;
    ArrayNIP_S BA31 = F21_S * F32_S - F22_S * F31_S;
    ArrayNIP_S BA12 = F13_S * F32_S - F12_S * F33_S;
    ArrayNIP_S BA22 = F11_S * F33_S - F13_S * F31_S;
    ArrayNIP_S BA32 = F12_S * F31_S - F11_S * F32_S;
    ArrayNIP_S BA13 = F12_S * F23_S - F13_S * F22_S;
    ArrayNIP_S BA23 = F13_S * F21_S - F11_S * F23_S;
    ArrayNIP_S BA33 = F11_S * F22_S - F12_S * F21_S;

    ArrayNIP_P BA11_P = F22_P * F33_P - F23_P * F32_P;
    ArrayNIP_P BA21_P = F23_P * F31_P - F21_P * F33_P;
    ArrayNIP_P BA31_P = F21_P * F32_P - F22_P * F31_P;
    ArrayNIP_P BA12_P = F13_P * F32_P - F12_P * F33_P;
    ArrayNIP_P BA22_P = F11_P * F33_P - F13_P * F31_P;
    ArrayNIP_P BA32_P = F12_P * F31_P - F11_P * F32_P;
    ArrayNIP_P BA13_P = F12_P * F23_P - F13_P * F22_P;
    ArrayNIP_P BA23_P = F13_P * F21_P - F11_P * F23_P;
    ArrayNIP_P BA33_P = F11_P * F22_P - F12_P * F21_P;

    ChMatrixNM<double, 3 * NSF, 7 * NIP_S + NIP_P> Left;
    for (auto i = 0; i < NSF; i++) {
        Left.block<1, NIP_S>(3 * i + 0, 0).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * F11_S;
        Left.block<1, NIP_S>(3 * i + 0, 0 + 1 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * F12_S;
        Left.block<1, NIP_S>(3 * i + 0, 0 + 2 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * F13_S;
        Left.block<1, NIP_S>(3 * i + 0, 0 + 3 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * F13_S + m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * F12_S;
        Left.block<1, NIP_S>(3 * i + 0, 0 + 4 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * F13_S + m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * F11_S;
        Left.block<1, NIP_S>(3 * i + 0, 0 + 5 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * F12_S + m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * F11_S;
        Left.block<1, NIP_S>(3 * i + 0, 0 + 6 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * BA11 +
            m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * BA21 +
            m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * BA31;
        Left.block<1, NIP_P>(3 * i + 0, 0 + 7 * NIP_S).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 0 * NIP_P).array() * BA11_P +
            m_SD.block<1, NIP_P>(i, 3 * NIP_S + 1 * NIP_P).array() * BA21_P +
            m_SD.block<1, NIP_P>(i, 3 * NIP_S + 2 * NIP_P).array() * BA31_P;

        Left.block<1, NIP_S>(3 * i + 1, 0).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * F21_S;
        Left.block<1, NIP_S>(3 * i + 1, 0 + 1 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * F22_S;
        Left.block<1, NIP_S>(3 * i + 1, 0 + 2 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * F23_S;
        Left.block<1, NIP_S>(3 * i + 1, 0 + 3 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * F23_S + m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * F22_S;
        Left.block<1, NIP_S>(3 * i + 1, 0 + 4 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * F23_S + m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * F21_S;
        Left.block<1, NIP_S>(3 * i + 1, 0 + 5 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * F22_S + m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * F21_S;
        Left.block<1, NIP_S>(3 * i + 1, 0 + 6 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * BA12 +
            m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * BA22 +
            m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * BA32;
        Left.block<1, NIP_P>(3 * i + 1, 0 + 7 * NIP_S).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 0 * NIP_P).array() * BA12_P +
            m_SD.block<1, NIP_P>(i, 3 * NIP_S + 1 * NIP_P).array() * BA22_P +
            m_SD.block<1, NIP_P>(i, 3 * NIP_S + 2 * NIP_P).array() * BA32_P;

        Left.block<1, NIP_S>(3 * i + 2, 0).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * F31_S;
        Left.block<1, NIP_S>(3 * i + 2, 0 + 1 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * F32_S;
        Left.block<1, NIP_S>(3 * i + 2, 0 + 2 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * F33_S;
        Left.block<1, NIP_S>(3 * i + 2, 0 + 3 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * F33_S + m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * F32_S;
        Left.block<1, NIP_S>(3 * i + 2, 0 + 4 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * F33_S + m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * F31_S;
        Left.block<1, NIP_S>(3 * i + 2, 0 + 5 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * F32_S + m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * F31_S;
        Left.block<1, NIP_S>(3 * i + 2, 0 + 6 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * BA13 +
            m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * BA23 +
            m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * BA33;
        Left.block<1, NIP_P>(3 * i + 2, 0 + 7 * NIP_S).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 0 * NIP_P).array() * BA13_P +
            m_SD.block<1, NIP_P>(i, 3 * NIP_S + 1 * NIP_P).array() * BA23_P +
            m_SD.block<1, NIP_P>(i, 3 * NIP_S + 2 * NIP_P).array() * BA33_P;
    }

    //Calculate the Right Cauchy-Green deformation tensor (symmetric):
    ArrayNIP_S C11 = F11_S * F11_S + F21_S * F21_S + F31_S * F31_S;
    ArrayNIP_S C22 = F12_S * F12_S + F22_S * F22_S + F32_S * F32_S;
    ArrayNIP_S C33 = F13_S * F13_S + F23_S * F23_S + F33_S * F33_S;
    ArrayNIP_S C12 = F11_S * F12_S + F21_S * F22_S + F31_S * F32_S;
    ArrayNIP_S C13 = F11_S * F13_S + F21_S * F23_S + F31_S * F33_S;
    ArrayNIP_S C23 = F12_S * F13_S + F22_S * F23_S + F32_S * F33_S;

    //Calculate the 1st invariant of the Right Cauchy-Green deformation tensor: trace(C)
    ArrayNIP_S I1 = C11 + C22 + C33;

    //Calculate the 2nd invariant of the Right Cauchy-Green deformation tensor: 1/2*(trace(C)^2-trace(C^2))
    ArrayNIP_S I2 = 0.5*(I1*I1 - C11 * C11 - C22 * C22 - C33 * C33) - C12 * C12 - C13 * C13 - C23 * C23;

    //Calculate the determinate of F (commonly denoted as J)
    ArrayNIP_S J = F11_S * (F22_S*F33_S - F23_S * F32_S) + F12_S * (F23_S*F31_S - F21_S * F33_S) + F13_S * (F21_S*F32_S - F22_S * F31_S);

    //Calculate the determinate of F to the -2/3 power -> J^(-2/3)
    ArrayNIP_S J_m2_3 = J.pow(-2.0 / 3.0);

    //Get the element's material properties
    double c10 = GetMaterial()->Get_c10();
    double c01 = GetMaterial()->Get_c01();
    double k = GetMaterial()->Get_k();

    ArrayNIP_S M0 = (-Kfactor * 2.0) * m_kGQ_S * J_m2_3; //Temporary term used to build the other scale factors
    ArrayNIP_S M2 = (-2.0*c01) * M0 * J_m2_3; //Double M2 since 2*M2 is needed for the partial C11/de, C22/de, and C33/de terms.  It is reset to just M2 after that.
    ArrayNIP_S M4 = (-2.0 / 3.0) * M2 / J;
    M0 *= c10;
    ArrayNIP_S M5 = (-2.0 / 3.0) * M0 / J - I1 * M4;

    ChMatrixNM<double, 3 * NSF, 7 * NIP_S + NIP_P> Right;

    //Calculate the Contribution from the dC11 / de terms
    {
        ArrayNIP_S T0 = (M4 * C11 + M5);
        ArrayNIP_S T11 = T0 * BA11;
        ArrayNIP_S T21 = T0 * BA21 - M2 * F12_S;
        ArrayNIP_S T31 = T0 * BA31 - M2 * F13_S;

        ArrayNIP_S T12 = T0 * BA12;
        ArrayNIP_S T22 = T0 * BA22 - M2 * F22_S;
        ArrayNIP_S T32 = T0 * BA32 - M2 * F23_S;

        ArrayNIP_S T13 = T0 * BA13;
        ArrayNIP_S T23 = T0 * BA23 - M2 * F32_S;
        ArrayNIP_S T33 = T0 * BA33 - M2 * F33_S;

        for (auto i = 0; i < NSF; i++) {
            Right.block<1, NIP_S>(3 * i + 0, 0).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T11 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T21 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T31;
            Right.block<1, NIP_S>(3 * i + 1, 0).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T12 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T22 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T32;
            Right.block<1, NIP_S>(3 * i + 2, 0).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T13 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T23 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T33;
        }
    }
    //Calculate the Contribution from the dC22 / de terms
    {
        ArrayNIP_S T0 = (M4 * C22 + M5);
        ArrayNIP_S T11 = T0 * BA11 - M2 * F11_S;
        ArrayNIP_S T21 = T0 * BA21;
        ArrayNIP_S T31 = T0 * BA31 - M2 * F13_S;

        ArrayNIP_S T12 = T0 * BA12 - M2 * F21_S;
        ArrayNIP_S T22 = T0 * BA22;
        ArrayNIP_S T32 = T0 * BA32 - M2 * F23_S;

        ArrayNIP_S T13 = T0 * BA13 - M2 * F31_S;
        ArrayNIP_S T23 = T0 * BA23;
        ArrayNIP_S T33 = T0 * BA33 - M2 * F33_S;

        for (auto i = 0; i < NSF; i++) {
            Right.block<1, NIP_S>(3 * i + 0, 0 + 1 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T11 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T21 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T31;
            Right.block<1, NIP_S>(3 * i + 1, 0 + 1 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T12 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T22 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T32;
            Right.block<1, NIP_S>(3 * i + 2, 0 + 1 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T13 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T23 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T33;
        }
    }
    //Calculate the Contribution from the dC33 / de terms
    {
        ArrayNIP_S T0 = (M4 * C22 + M5);
        ArrayNIP_S T11 = T0 * BA11 - M2 * F11_S;
        ArrayNIP_S T21 = T0 * BA21 - M2 * F12_S;
        ArrayNIP_S T31 = T0 * BA31;

        ArrayNIP_S T12 = T0 * BA12 - M2 * F21_S;
        ArrayNIP_S T22 = T0 * BA22 - M2 * F22_S;
        ArrayNIP_S T32 = T0 * BA32;

        ArrayNIP_S T13 = T0 * BA13 - M2 * F31_S;
        ArrayNIP_S T23 = T0 * BA23 - M2 * F32_S;
        ArrayNIP_S T33 = T0 * BA33;

        for (auto i = 0; i < NSF; i++) {
            Right.block<1, NIP_S>(3 * i + 0, 0 + 2 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T11 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T21 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T31;
            Right.block<1, NIP_S>(3 * i + 1, 0 + 2 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T12 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T22 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T32;
            Right.block<1, NIP_S>(3 * i + 2, 0 + 2 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T13 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T23 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T33;
        }
    }

    //Reset M2 to it's correct value instead of double that was used above
    M2 *= 0.5;
    // Calculate the Contribution from the dC23/de terms
    {
        ArrayNIP_S T0 = M4 * C23;
        ArrayNIP_S T11 = T0 * BA11;
        ArrayNIP_S T21 = T0 * BA21 + M2 * F13_S;
        ArrayNIP_S T31 = T0 * BA31 + M2 * F12_S;

        ArrayNIP_S T12 = T0 * BA12;
        ArrayNIP_S T22 = T0 * BA22 + M2 * F23_S;
        ArrayNIP_S T32 = T0 * BA32 + M2 * F22_S;

        ArrayNIP_S T13 = T0 * BA13;
        ArrayNIP_S T23 = T0 * BA23 + M2 * F33_S;
        ArrayNIP_S T33 = T0 * BA33 + M2 * F32_S;

        for (auto i = 0; i < NSF; i++) {
            Right.block<1, NIP_S>(3 * i + 0, 0 + 3 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T11 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T21 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T31;
            Right.block<1, NIP_S>(3 * i + 1, 0 + 3 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T12 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T22 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T32;
            Right.block<1, NIP_S>(3 * i + 2, 0 + 3 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T13 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T23 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T33;
        }
    }
    // Calculate the Contribution from the dC13/de terms
    {
        ArrayNIP_S T0 = M4 * C13;
        ArrayNIP_S T11 = T0 * BA11 + M2 * F13_S;
        ArrayNIP_S T21 = T0 * BA21;
        ArrayNIP_S T31 = T0 * BA31 + M2 * F11_S;

        ArrayNIP_S T12 = T0 * BA12 + M2 * F23_S;
        ArrayNIP_S T22 = T0 * BA22;
        ArrayNIP_S T32 = T0 * BA32 + M2 * F21_S;

        ArrayNIP_S T13 = T0 * BA13 + M2 * F33_S;
        ArrayNIP_S T23 = T0 * BA23;
        ArrayNIP_S T33 = T0 * BA33 + M2 * F31_S;

        for (auto i = 0; i < NSF; i++) {
            Right.block<1, NIP_S>(3 * i + 0, 0 + 4 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T11 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T21 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T31;
            Right.block<1, NIP_S>(3 * i + 1, 0 + 4 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T12 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T22 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T32;
            Right.block<1, NIP_S>(3 * i + 2, 0 + 4 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T13 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T23 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T33;
        }
    }
    // Calculate the Contribution from the dC12/de terms
    {
        ArrayNIP_S T0 = M4 * C13;
        ArrayNIP_S T11 = T0 * BA11 + M2 * F12_S;
        ArrayNIP_S T21 = T0 * BA21 + M2 * F11_S;
        ArrayNIP_S T31 = T0 * BA31;

        ArrayNIP_S T12 = T0 * BA12 + M2 * F22_S;
        ArrayNIP_S T22 = T0 * BA22 + M2 * F21_S;
        ArrayNIP_S T32 = T0 * BA32;

        ArrayNIP_S T13 = T0 * BA13 + M2 * F32_S;
        ArrayNIP_S T23 = T0 * BA23 + M2 * F31_S;
        ArrayNIP_S T33 = T0 * BA33;

        for (auto i = 0; i < NSF; i++) {
            Right.block<1, NIP_S>(3 * i + 0, 0 + 5 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T11 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T21 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T31;
            Right.block<1, NIP_S>(3 * i + 1, 0 + 5 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T12 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T22 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T32;
            Right.block<1, NIP_S>(3 * i + 2, 0 + 5 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T13 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T23 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T33;
        }
    }
    // Calculate the Contribution from the ddetF/de terms - No Penalty Terms
    {
        ArrayNIP_S M6 = (5.0 / 9.0*I1*M0 - 14.0 / 9.0*M2*I2) / (J*J);

        ArrayNIP_S T11 = M4 * (C11*F11_S + C12 * F12_S + C13 * F13_S) + M5 * F11_S + M6 * BA11;
        ArrayNIP_S T21 = M4 * (C12*F11_S + C22 * F12_S + C23 * F13_S) + M5 * F12_S + M6 * BA21;
        ArrayNIP_S T31 = M4 * (C13*F11_S + C23 * F12_S + C33 * F13_S) + M5 * F13_S + M6 * BA31;
        ArrayNIP_S T12 = M4 * (C11*F21_S + C12 * F22_S + C13 * F23_S) + M5 * F21_S + M6 * BA12;
        ArrayNIP_S T22 = M4 * (C12*F21_S + C22 * F22_S + C23 * F23_S) + M5 * F22_S + M6 * BA22;
        ArrayNIP_S T32 = M4 * (C13*F21_S + C23 * F22_S + C33 * F23_S) + M5 * F23_S + M6 * BA32;
        ArrayNIP_S T13 = M4 * (C11*F31_S + C12 * F32_S + C13 * F33_S) + M5 * F31_S + M6 * BA13;
        ArrayNIP_S T23 = M4 * (C12*F31_S + C22 * F32_S + C23 * F33_S) + M5 * F32_S + M6 * BA23;
        ArrayNIP_S T33 = M4 * (C13*F31_S + C23 * F32_S + C33 * F33_S) + M5 * F33_S + M6 * BA33;

        for (auto i = 0; i < NSF; i++) {
            Right.block<1, NIP_S>(3 * i + 0, 0 + 6 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T11 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T21 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T31;
            Right.block<1, NIP_S>(3 * i + 1, 0 + 6 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T12 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T22 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T32;
            Right.block<1, NIP_S>(3 * i + 2, 0 + 6 * NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * T13 +
                m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * T23 +
                m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * T33;
        }
    }
    // Calculate the Contribution from the ddetF/de terms - Penalty Terms
    {
        ArrayNIP_P M6 = -Kfactor * k * m_kGQ_P;

        ArrayNIP_P T11 = M6 * BA11_P;
        ArrayNIP_P T21 = M6 * BA21_P;
        ArrayNIP_P T31 = M6 * BA31_P;

        ArrayNIP_P T12 = M6 * BA12_P;
        ArrayNIP_P T22 = M6 * BA22_P;
        ArrayNIP_P T32 = M6 * BA32_P;

        ArrayNIP_P T13 = M6 * BA13_P;
        ArrayNIP_P T23 = M6 * BA23_P;
        ArrayNIP_P T33 = M6 * BA33_P;

        for (auto i = 0; i < NSF; i++) {
            Right.block<1, NIP_P>(3 * i + 0, 0 + 7 * NIP_S).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 0 * NIP_P).array() * T11 +
                m_SD.block<1, NIP_P>(i, 3 * NIP_S + 1 * NIP_P).array() * T21 +
                m_SD.block<1, NIP_P>(i, 3 * NIP_S + 2 * NIP_P).array() * T31;
            Right.block<1, NIP_P>(3 * i + 1, 0 + 7 * NIP_S).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 0 * NIP_P).array() * T12 +
                m_SD.block<1, NIP_P>(i, 3 * NIP_S + 1 * NIP_P).array() * T22 +
                m_SD.block<1, NIP_P>(i, 3 * NIP_S + 2 * NIP_P).array() * T32;
            Right.block<1, NIP_P>(3 * i + 2, 0 + 7 * NIP_S).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 0 * NIP_P).array() * T13 +
                m_SD.block<1, NIP_P>(i, 3 * NIP_S + 1 * NIP_P).array() * T23 +
                m_SD.block<1, NIP_P>(i, 3 * NIP_S + 2 * NIP_P).array() * T33;
        }
    }

    Matrix3Nx3N Jac = Left * Right.transpose();

    //Calculate the contribution from the Mass Matrix and Expand/Expandij components
    ArrayNIP_S M1 = M0 - M2 * I1;
    ArrayNIP_S M3 = (2 * I2*M2 - I1 * M0) / (3.0 * J);

    ArrayNIP_S MC11 = M1 + M2 * C11;
    ArrayNIP_S MC22 = M1 + M2 * C22;
    ArrayNIP_S MC33 = M1 + M2 * C33;
    ArrayNIP_S MC12 = M2 * C12;
    ArrayNIP_S MC13 = M2 * C13;
    ArrayNIP_S MC23 = M2 * C23;

    ArrayNIP_S MF11_S = M3 * F11_S;
    ArrayNIP_S MF12_S = M3 * F12_S;
    ArrayNIP_S MF13_S = M3 * F13_S;
    ArrayNIP_S MF21_S = M3 * F21_S;
    ArrayNIP_S MF22_S = M3 * F22_S;
    ArrayNIP_S MF23_S = M3 * F23_S;
    ArrayNIP_S MF31_S = M3 * F31_S;
    ArrayNIP_S MF32_S = M3 * F32_S;
    ArrayNIP_S MF33_S = M3 * F33_S;

    //Calculate the determinate of F (commonly denoted as J) for the volumentric penalty Gauss quadrature points
    ArrayNIP_P J_P = F11_P * (F22_P*F33_P - F23_P * F32_P) + F12_P * (F23_P*F31_P - F21_P * F33_P) + F13_P * (F21_P*F32_P - F22_P * F31_P);
    ArrayNIP_P M3_P = (-Kfactor * k) * (J_P - 1.0)*m_kGQ_P;
    ArrayNIP_P MF11_P = M3_P * F11_P;
    ArrayNIP_P MF12_P = M3_P * F12_P;
    ArrayNIP_P MF13_P = M3_P * F13_P;
    ArrayNIP_P MF21_P = M3_P * F21_P;
    ArrayNIP_P MF22_P = M3_P * F22_P;
    ArrayNIP_P MF23_P = M3_P * F23_P;
    ArrayNIP_P MF31_P = M3_P * F31_P;
    ArrayNIP_P MF32_P = M3_P * F32_P;
    ArrayNIP_P MF33_P = M3_P * F33_P;

    unsigned int idx = 0;
    for (unsigned int i = 0; i < (NSF - 1); i++) {
        //Calculate the scaled row of SD
        ChVectorN<double, 3 * NIP_S> R;
        R.segment(0 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * MC11 +
            m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * MC12 +
            m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * MC13;
        R.segment(1 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * MC12 +
            m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * MC22 +
            m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * MC23;
        R.segment(2 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array() * MC13 +
            m_SD.block<1, NIP_S>(i, 1 * NIP_S).array() * MC23 +
            m_SD.block<1, NIP_S>(i, 2 * NIP_S).array() * MC33;

        //Calculate the scaled row of B2 - Non Penalty Terms
        ChVectorN<double, 3 * NIP_S> X_S;
        X_S.segment(0 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(i, 1 * NIP_S).array()*MF33_S - m_SD.block<1, NIP_S>(i, 2 * NIP_S).array()*MF32_S;
        X_S.segment(1 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(i, 2 * NIP_S).array()*MF31_S - m_SD.block<1, NIP_S>(i, 0 * NIP_S).array()*MF33_S;
        X_S.segment(2 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array()*MF32_S - m_SD.block<1, NIP_S>(i, 1 * NIP_S).array()*MF31_S;

        //Calculate the scaled row of B3 - Non Penalty Terms
        ChVectorN<double, 3 * NIP_S> Y_S;
        Y_S.segment(0 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(i, 2 * NIP_S).array()*MF22_S - m_SD.block<1, NIP_S>(i, 1 * NIP_S).array()*MF23_S;
        Y_S.segment(1 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array()*MF23_S - m_SD.block<1, NIP_S>(i, 2 * NIP_S).array()*MF21_S;
        Y_S.segment(2 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(i, 1 * NIP_S).array()*MF21_S - m_SD.block<1, NIP_S>(i, 0 * NIP_S).array()*MF22_S;

        //Calculate the scaled row of B4 - Non Penalty Terms
        ChVectorN<double, 3 * NIP_S> Z_S;
        Z_S.segment(0 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(i, 1 * NIP_S).array()*MF13_S - m_SD.block<1, NIP_S>(i, 2 * NIP_S).array()*MF12_S;
        Z_S.segment(1 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(i, 2 * NIP_S).array()*MF11_S - m_SD.block<1, NIP_S>(i, 0 * NIP_S).array()*MF13_S;
        Z_S.segment(2 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(i, 0 * NIP_S).array()*MF12_S - m_SD.block<1, NIP_S>(i, 1 * NIP_S).array()*MF11_S;

        //Calculate the scaled row of B2 - Penalty Terms
        ChVectorN<double, 3 * NIP_P> X_P;
        X_P.segment(0 * NIP_P, NIP_P).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 1 * NIP_P).array()*MF33_P - m_SD.block<1, NIP_P>(i, 3 * NIP_S + 2 * NIP_P).array()*MF32_P;
        X_P.segment(1 * NIP_P, NIP_P).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 2 * NIP_P).array()*MF31_P - m_SD.block<1, NIP_P>(i, 3 * NIP_S + 0 * NIP_P).array()*MF33_P;
        X_P.segment(2 * NIP_P, NIP_P).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 0 * NIP_P).array()*MF32_P - m_SD.block<1, NIP_P>(i, 3 * NIP_S + 1 * NIP_P).array()*MF31_P;

        //Calculate the scaled row of B3 - Penalty Terms
        ChVectorN<double, 3 * NIP_P> Y_P;
        Y_P.segment(0 * NIP_P, NIP_P).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 2 * NIP_P).array()*MF22_P - m_SD.block<1, NIP_P>(i, 3 * NIP_S + 1 * NIP_P).array()*MF23_P;
        Y_P.segment(1 * NIP_P, NIP_P).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 0 * NIP_P).array()*MF23_P - m_SD.block<1, NIP_P>(i, 3 * NIP_S + 2 * NIP_P).array()*MF21_P;
        Y_P.segment(2 * NIP_P, NIP_P).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 1 * NIP_P).array()*MF21_P - m_SD.block<1, NIP_P>(i, 3 * NIP_S + 0 * NIP_P).array()*MF22_P;

        //Calculate the scaled row of B4 - Penalty Terms
        ChVectorN<double, 3 * NIP_P> Z_P;
        Z_P.segment(0 * NIP_P, NIP_P).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 1 * NIP_P).array()*MF13_P - m_SD.block<1, NIP_P>(i, 3 * NIP_S + 2 * NIP_P).array()*MF12_P;
        Z_P.segment(1 * NIP_P, NIP_P).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 2 * NIP_P).array()*MF11_P - m_SD.block<1, NIP_P>(i, 3 * NIP_S + 0 * NIP_P).array()*MF13_P;
        Z_P.segment(2 * NIP_P, NIP_P).array() = m_SD.block<1, NIP_P>(i, 3 * NIP_S + 0 * NIP_P).array()*MF12_P - m_SD.block<1, NIP_P>(i, 3 * NIP_S + 1 * NIP_P).array()*MF11_P;

        //double d_diag = Mfactor * m_MassMatrix(idx) + (scaled_SD_row_i.dot(m_SD.row(i)));
        double d_diag = Mfactor * m_MassMatrix(idx) + (R.dot(m_SD.block<1, 3 * NIP_S>(i, 0)));

        Jac(3 * i, 3 * i) += d_diag;
        Jac(3 * i + 1, 3 * i + 1) += d_diag;
        Jac(3 * i + 2, 3 * i + 2) += d_diag;
        idx++;

        for (unsigned int j = (i + 1); j < NSF; j++) {
            double d = Mfactor * m_MassMatrix(idx) + (R.dot(m_SD.block<1, 3 * NIP_S>(j, 0)));
            double B2 = X_S.dot(m_SD.block<1, 3 * NIP_S>(j, 0)) + X_P.dot(m_SD.block<1, 3 * NIP_P>(j, 3 * NIP_S));
            double B3 = Y_S.dot(m_SD.block<1, 3 * NIP_S>(j, 0)) + Y_P.dot(m_SD.block<1, 3 * NIP_P>(j, 3 * NIP_S));
            double B4 = Z_S.dot(m_SD.block<1, 3 * NIP_S>(j, 0)) + Z_P.dot(m_SD.block<1, 3 * NIP_P>(j, 3 * NIP_S));

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
    ChVectorN<double, 3 * NIP_S> R;
    R.segment(0 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(NSF - 1, 0 * NIP_S).array() * MC11 +
        m_SD.block<1, NIP_S>(NSF - 1, 1 * NIP_S).array() * MC12 +
        m_SD.block<1, NIP_S>(NSF - 1, 2 * NIP_S).array() * MC13;
    R.segment(1 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(NSF - 1, 0 * NIP_S).array() * MC12 +
        m_SD.block<1, NIP_S>(NSF - 1, 1 * NIP_S).array() * MC22 +
        m_SD.block<1, NIP_S>(NSF - 1, 2 * NIP_S).array() * MC23;
    R.segment(2 * NIP_S, NIP_S).array() = m_SD.block<1, NIP_S>(NSF - 1, 0 * NIP_S).array() * MC13 +
        m_SD.block<1, NIP_S>(NSF - 1, 1 * NIP_S).array() * MC23 +
        m_SD.block<1, NIP_S>(NSF - 1, 2 * NIP_S).array() * MC33;

    //double d_diag = Mfactor * m_MassMatrix(idx) + (scaled_SD_row_i.dot(m_SD.row(NSF - 1)));
    double d_diag = Mfactor * m_MassMatrix(idx) + (R.dot(m_SD.block<1, 3 * NIP_S>(NSF - 1, 0)));

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

void ChElementBeamANCF_3243_MR_Damp::Calc_Sxi_compact(VectorN& Sxi_compact, double xi, double eta, double zeta) {
    Sxi_compact(0) = 0.25 * (xi * xi * xi - 3 * xi + 2);
    Sxi_compact(1) = 0.125 * m_lenX * (xi * xi * xi - xi * xi - xi + 1);
    Sxi_compact(2) = 0.25 * m_thicknessY * eta * (1 - xi);
    Sxi_compact(3) = 0.25 * m_thicknessZ * zeta * (1 - xi);
    Sxi_compact(4) = 0.25 * (-xi * xi * xi + 3 * xi + 2);
    Sxi_compact(5) = 0.125 * m_lenX * (xi * xi * xi + xi * xi - xi - 1);
    Sxi_compact(6) = 0.25 * m_thicknessY * eta * (1 + xi);
    Sxi_compact(7) = 0.25 * m_thicknessZ * zeta * (1 + xi);
}

// Nx1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to xi
// [s1; s2; s3; ...]

void ChElementBeamANCF_3243_MR_Damp::Calc_Sxi_xi_compact(VectorN& Sxi_xi_compact, double xi, double eta, double zeta) {
    Sxi_xi_compact(0) = 0.75 * (xi * xi - 1);
    Sxi_xi_compact(1) = 0.125 * m_lenX * (3 * xi * xi - 2 * xi - 1);
    Sxi_xi_compact(2) = -0.25 * m_thicknessY * eta;
    Sxi_xi_compact(3) = -0.25 * m_thicknessZ * zeta;
    Sxi_xi_compact(4) = 0.75 * (-xi * xi + 1);
    Sxi_xi_compact(5) = 0.125 * m_lenX * (3 * xi * xi + 2 * xi - 1);
    Sxi_xi_compact(6) = 0.25 * m_thicknessY * eta;
    Sxi_xi_compact(7) = 0.25 * m_thicknessZ * zeta;
}

// Nx1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to eta
// [s1; s2; s3; ...]

void ChElementBeamANCF_3243_MR_Damp::Calc_Sxi_eta_compact(VectorN& Sxi_eta_compact, double xi, double eta, double zeta) {
    Sxi_eta_compact(0) = 0.0;
    Sxi_eta_compact(1) = 0.0;
    Sxi_eta_compact(2) = 0.25 * m_thicknessY * (-xi + 1);
    Sxi_eta_compact(3) = 0.0;
    Sxi_eta_compact(4) = 0.0;
    Sxi_eta_compact(5) = 0.0;
    Sxi_eta_compact(6) = 0.25 * m_thicknessY * (xi + 1);
    Sxi_eta_compact(7) = 0.0;
}

// Nx1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to zeta
// [s1; s2; s3; ...]

void ChElementBeamANCF_3243_MR_Damp::Calc_Sxi_zeta_compact(VectorN& Sxi_zeta_compact, double xi, double eta, double zeta) {
    Sxi_zeta_compact(0) = 0.0;
    Sxi_zeta_compact(1) = 0.0;
    Sxi_zeta_compact(2) = 0.0;
    Sxi_zeta_compact(3) = 0.25 * m_thicknessZ * (-xi + 1);
    Sxi_zeta_compact(4) = 0.0;
    Sxi_zeta_compact(5) = 0.0;
    Sxi_zeta_compact(6) = 0.0;
    Sxi_zeta_compact(7) = 0.25 * m_thicknessZ * (xi + 1);
}

// Nx3 compact form of the partial derivatives of Normalized Shape Functions with respect to xi, eta, and zeta by
// columns

void ChElementBeamANCF_3243_MR_Damp::Calc_Sxi_D(MatrixNx3c& Sxi_D, double xi, double eta, double zeta) {
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

void ChElementBeamANCF_3243_MR_Damp::CalcCoordVector(Vector3N& e) {
    e.segment(0, 3) = m_nodes[0]->GetPos().eigen();
    e.segment(3, 3) = m_nodes[0]->GetSlope1().eigen();
    e.segment(6, 3) = m_nodes[0]->GetSlope2().eigen();
    e.segment(9, 3) = m_nodes[0]->GetSlope3().eigen();

    e.segment(12, 3) = m_nodes[1]->GetPos().eigen();
    e.segment(15, 3) = m_nodes[1]->GetSlope1().eigen();
    e.segment(18, 3) = m_nodes[1]->GetSlope2().eigen();
    e.segment(21, 3) = m_nodes[1]->GetSlope3().eigen();
}

void ChElementBeamANCF_3243_MR_Damp::CalcCoordMatrix(Matrix3xN& ebar) {
    ebar.col(0) = m_nodes[0]->GetPos().eigen();
    ebar.col(1) = m_nodes[0]->GetSlope1().eigen();
    ebar.col(2) = m_nodes[0]->GetSlope2().eigen();
    ebar.col(3) = m_nodes[0]->GetSlope3().eigen();

    ebar.col(4) = m_nodes[1]->GetPos().eigen();
    ebar.col(5) = m_nodes[1]->GetSlope1().eigen();
    ebar.col(6) = m_nodes[1]->GetSlope2().eigen();
    ebar.col(7) = m_nodes[1]->GetSlope3().eigen();
}

void ChElementBeamANCF_3243_MR_Damp::CalcCoordDtVector(Vector3N& edot) {
    edot.segment(0, 3) = m_nodes[0]->GetPosDt().eigen();
    edot.segment(3, 3) = m_nodes[0]->GetSlope1Dt().eigen();
    edot.segment(6, 3) = m_nodes[0]->GetSlope2Dt().eigen();
    edot.segment(9, 3) = m_nodes[0]->GetSlope3Dt().eigen();

    edot.segment(12, 3) = m_nodes[1]->GetPosDt().eigen();
    edot.segment(15, 3) = m_nodes[1]->GetSlope1Dt().eigen();
    edot.segment(18, 3) = m_nodes[1]->GetSlope2Dt().eigen();
    edot.segment(21, 3) = m_nodes[1]->GetSlope3Dt().eigen();
}

void ChElementBeamANCF_3243_MR_Damp::CalcCoordDtMatrix(Matrix3xN& ebardot) {
    ebardot.col(0) = m_nodes[0]->GetPosDt().eigen();
    ebardot.col(1) = m_nodes[0]->GetSlope1Dt().eigen();
    ebardot.col(2) = m_nodes[0]->GetSlope2Dt().eigen();
    ebardot.col(3) = m_nodes[0]->GetSlope3Dt().eigen();

    ebardot.col(4) = m_nodes[1]->GetPosDt().eigen();
    ebardot.col(5) = m_nodes[1]->GetSlope1Dt().eigen();
    ebardot.col(6) = m_nodes[1]->GetSlope2Dt().eigen();
    ebardot.col(7) = m_nodes[1]->GetSlope3Dt().eigen();
}

void ChElementBeamANCF_3243_MR_Damp::CalcCombinedCoordMatrix(Matrix6xN& ebar_ebardot) {
    ebar_ebardot.block<3, 1>(0, 0) = m_nodes[0]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 0) = m_nodes[0]->GetPosDt().eigen();
    ebar_ebardot.block<3, 1>(0, 1) = m_nodes[0]->GetSlope1().eigen();
    ebar_ebardot.block<3, 1>(3, 1) = m_nodes[0]->GetSlope1Dt().eigen();
    ebar_ebardot.block<3, 1>(0, 2) = m_nodes[0]->GetSlope2().eigen();
    ebar_ebardot.block<3, 1>(3, 2) = m_nodes[0]->GetSlope2Dt().eigen();
    ebar_ebardot.block<3, 1>(0, 3) = m_nodes[0]->GetSlope3().eigen();
    ebar_ebardot.block<3, 1>(3, 3) = m_nodes[0]->GetSlope3Dt().eigen();

    ebar_ebardot.block<3, 1>(0, 4) = m_nodes[1]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 4) = m_nodes[1]->GetPosDt().eigen();
    ebar_ebardot.block<3, 1>(0, 5) = m_nodes[1]->GetSlope1().eigen();
    ebar_ebardot.block<3, 1>(3, 5) = m_nodes[1]->GetSlope1Dt().eigen();
    ebar_ebardot.block<3, 1>(0, 6) = m_nodes[1]->GetSlope2().eigen();
    ebar_ebardot.block<3, 1>(3, 6) = m_nodes[1]->GetSlope2Dt().eigen();
    ebar_ebardot.block<3, 1>(0, 7) = m_nodes[1]->GetSlope3().eigen();
    ebar_ebardot.block<3, 1>(3, 7) = m_nodes[1]->GetSlope3Dt().eigen();
}

// Calculate the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element

void ChElementBeamANCF_3243_MR_Damp::Calc_J_0xi(ChMatrix33d& J_0xi, double xi, double eta, double zeta) {
    MatrixNx3c Sxi_D;
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    J_0xi = m_ebar0 * Sxi_D;
}

// Calculate the determinant of the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element

double ChElementBeamANCF_3243_MR_Damp::Calc_det_J_0xi(double xi, double eta, double zeta) {
    ChMatrix33d J_0xi;
    Calc_J_0xi(J_0xi, xi, eta, zeta);

    return (J_0xi.determinant());
}

////////////////////////////////////////////////////////////////

//#ifndef CH_QUADRATURE_STATIC_TABLES
#define CH_QUADRATURE_STATIC_TABLES 10
ChQuadratureTables static_tables_3243_MR_Damp(1, CH_QUADRATURE_STATIC_TABLES);
//#endif // !CH_QUADRATURE_STATIC_TABLES

ChQuadratureTables* ChElementBeamANCF_3243_MR_Damp::GetStaticGQTables() {
    return &static_tables_3243_MR_Damp;
}

}  // namespace fea
}  // namespace chrono

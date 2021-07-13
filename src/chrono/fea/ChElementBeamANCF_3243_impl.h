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
//
// A description of the Enhanced Continuum Mechanics based method can be found in: K. Nachbagauer, P. Gruber, and J.
// Gerstmayr. Structural and Continuum Mechanics Approaches for a 3D Shear Deformable ANCF Beam Finite Element :
// Application to Static and Linearized Dynamic Examples.J.Comput.Nonlinear Dynam, 8 (2) : 021004, 2012.
// =============================================================================
// The "Continuous Integration" style calculation for the generalized internal force is based on modifications to
// (including a new analytical Jacobian):  Gerstmayr, J., Shabana, A.A.: Efficient integration of the elastic forces and
// thin three-dimensional beam elements in the absolute nodal coordinate formulation.In: Proceedings of the Multibody
// Dynamics Eccomas thematic Conference, Madrid(2005).
//
// The "Pre-Integration" style calculation is based on modifications
// to Liu, Cheng, Qiang Tian, and Haiyan Hu. "Dynamics of a large scale rigid�flexible multibody system composed of
// composite laminated plates." Multibody System Dynamics 26, no. 3 (2011): 283-305.
//
// A report covering the detailed mathematics and implementation both of these generalized internal force calculations
// and their Jacobians can be found in: Taylor, M.: Technical Report TR-2020-09 Efficient CPU Based Calculations of the
// Generalized Internal Forces and Jacobian of the Generalized Internal Forces for ANCF Continuum Mechanics Elements
// with Linear Viscoelastic Materials, Simulation Based Engineering Lab, University of Wisconsin-Madison; 2021.
// =============================================================================
// This element class has been templatized by the number of Gauss quadrature points to use for the generalized internal
// force calculations and its Jacobian with the recommended values as the default.  Using fewer than 2 Gauss quadrature
// points along the beam axis (NP) or through each cross section direction (NT) will likely result in numerical issues
// with the element.
// =============================================================================

// ------------------------------------------------------------------------------
// Constructor
// ------------------------------------------------------------------------------
template <int NP, int NT>
ChElementBeamANCF_3243<NP, NT>::ChElementBeamANCF_3243()
    : m_method(IntFrcMethod::ContInt),
      m_gravity_on(false),
      m_lenX(0),
      m_thicknessY(0),
      m_thicknessZ(0),
      m_Alpha(0),
      m_damping_enabled(false) {
    m_nodes.resize(2);
}

// ------------------------------------------------------------------------------
// Set element nodes
// ------------------------------------------------------------------------------
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::SetNodes(std::shared_ptr<ChNodeFEAxyzDDD> nodeA,
                                              std::shared_ptr<ChNodeFEAxyzDDD> nodeB) {
    assert(nodeA);
    assert(nodeB);

    m_nodes[0] = nodeA;
    m_nodes[1] = nodeB;

    std::vector<ChVariables*> mvars;
    mvars.push_back(&m_nodes[0]->Variables());
    mvars.push_back(&m_nodes[0]->Variables_D());
    mvars.push_back(&m_nodes[0]->Variables_DD());
    mvars.push_back(&m_nodes[0]->Variables_DDD());
    mvars.push_back(&m_nodes[1]->Variables());
    mvars.push_back(&m_nodes[1]->Variables_D());
    mvars.push_back(&m_nodes[1]->Variables_DD());
    mvars.push_back(&m_nodes[1]->Variables_DDD());

    Kmatr.SetVariables(mvars);

    // Initial positions and slopes of the element nodes
    // These values define the reference configuration of the element
    CalcCoordMatrix(m_ebar0);

    // Check to see if SetupInitial has already been called (i.e. at least one set of precomputed matrices has been
    // populated).  If so, the precomputed matrices will need to be re-generated.  If not, this will be handled once
    // SetupInitial is called.
    if (m_SD_Dv.size() + m_O1.size() > 0) {
        PrecomputeInternalForceMatricesWeights();
    }
}

// -----------------------------------------------------------------------------
// Element Settings
// -----------------------------------------------------------------------------

// Specify the element dimensions.
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::SetDimensions(double lenX, double thicknessY, double thicknessZ) {
    m_lenX = lenX;
    m_thicknessY = thicknessY;
    m_thicknessZ = thicknessZ;

    // Check to see if SetupInitial has already been called (i.e. at least one set of precomputed matrices has been
    // populated).  If so, the precomputed matrices will need to be re-generated.  If not, this will be handled once
    // SetupInitial is called.
    if (m_SD_Dv.size() + m_O1.size() > 0) {
        PrecomputeInternalForceMatricesWeights();
    }
}

// Specify the element material.
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::SetMaterial(std::shared_ptr<ChMaterialBeamANCF_3243> beam_mat) {
    m_material = beam_mat;

    // Check to see if the Pre-Integration method is selected and if the precomputed matrices have already been
    // generated and thus need to be regenerated.  The Continuous Integration precomputed matrices do not include any
    // material properties.
    if (m_method == IntFrcMethod::PreInt) {
        if (m_O1.size() > 0) {
            PrecomputeInternalForceMatricesWeights();
        }
    }
}

// Set the value for the single term structural damping coefficient.
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::SetAlphaDamp(double a) {
    m_Alpha = a;
    if (std::abs(m_Alpha) > 1e-10)
        m_damping_enabled = true;
    else
        m_damping_enabled = false;
}

// Change the method used to compute the generalized internal force vector and its Jacobian.
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::SetIntFrcCalcMethod(IntFrcMethod method) {
    m_method = method;

    // Check to see if SetupInitial has already been called (i.e. at least one set of precomputed matrices has been
    // populated).  If so, the precomputed matrices will need to be generated if they do not already exist or
    // regenerated if they have to ensure they are in sync with any other element changes.
    if (m_SD_Dv.size() + m_O1.size() > 0) {
        PrecomputeInternalForceMatricesWeights();
    }
}

// -----------------------------------------------------------------------------
// Evaluate Strains and Stresses
// -----------------------------------------------------------------------------
// These functions are designed for single function calls.  If these values are needed at the same points in the element
// through out the simulation, then the adjusted normalized shape function derivative matrix (Sxi_D) for each query
// point should be cached and saved to increase the execution speed
// -----------------------------------------------------------------------------

// Get the Green-Lagrange strain tensor at the normalized element coordinates (xi, eta, zeta) [-1...1]
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::GetGreenLagrangeStrain(const double xi,
                                                            const double eta,
                                                            const double zeta,
                                                            ChMatrix33<>& E) {
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
    E = 0.5 * (F.transpose() * F - I3x3);
}

// Get the 2nd Piola-Kirchoff stress tensor at the normalized element coordinates (xi, eta, zeta) [-1...1] at the
// current state of the element.
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::GetPK2Stress(const double xi,
                                                  const double eta,
                                                  const double zeta,
                                                  ChMatrix33<>& SPK2) {
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

    const ChMatrixNM<double, 6, 6>& D = GetMaterial()->Get_D();

    ChVectorN<double, 6> sigmaPK2 = D * epsilon_combined;  // 2nd Piola Kirchhoff Stress tensor in Voigt notation

    SPK2(0, 0) = sigmaPK2(0);
    SPK2(1, 1) = sigmaPK2(1);
    SPK2(2, 2) = sigmaPK2(2);
    SPK2(1, 2) = sigmaPK2(3);
    SPK2(2, 1) = sigmaPK2(3);
    SPK2(0, 2) = sigmaPK2(4);
    SPK2(2, 0) = sigmaPK2(4);
    SPK2(0, 1) = sigmaPK2(5);
    SPK2(1, 0) = sigmaPK2(5);
}

// Get the von Mises stress value at the normalized element coordinates (xi, eta, zeta) [-1...1] at the current
// state of the element.
template <int NP, int NT>
double ChElementBeamANCF_3243<NP, NT>::GetVonMissesStress(const double xi, const double eta, const double zeta) {
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

    const ChMatrixNM<double, 6, 6>& D = GetMaterial()->Get_D();

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
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::SetupInitial(ChSystem* system) {
    // Store the initial nodal coordinates. These values define the reference configuration of the element.
    CalcCoordMatrix(m_ebar0);

    // Compute and store the constant mass matrix and gravitational force vector
    ComputeMassMatrixAndGravityForce(system->Get_G_acc());

    // Compute any required matrices and vectors for the generalized internal force and Jacobian calculations
    PrecomputeInternalForceMatricesWeights();
}

// Fill the D vector with the current field values at the element nodes.
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::GetStateBlock(ChVectorDynamic<>& mD) {
    mD.segment(0, 3) = m_nodes[0]->GetPos().eigen();
    mD.segment(3, 3) = m_nodes[0]->GetD().eigen();
    mD.segment(6, 3) = m_nodes[0]->GetDD().eigen();
    mD.segment(9, 3) = m_nodes[0]->GetDDD().eigen();

    mD.segment(12, 3) = m_nodes[1]->GetPos().eigen();
    mD.segment(15, 3) = m_nodes[1]->GetD().eigen();
    mD.segment(18, 3) = m_nodes[1]->GetDD().eigen();
    mD.segment(21, 3) = m_nodes[1]->GetDDD().eigen();
}

// State update.
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::Update() {
    ChElementGeneric::Update();
}

// Return the mass matrix in full sparse form.
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::ComputeMmatrixGlobal(ChMatrixRef M) {
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
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::ComputeNodalMass() {
    m_nodes[0]->m_TotalMass += m_MassMatrix(0) + m_MassMatrix(4);
    m_nodes[1]->m_TotalMass += m_MassMatrix(4) + m_MassMatrix(26);
}

// Compute the generalized internal force vector for the current nodal coordinates as set the value in the Fi vector.
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::ComputeInternalForces(ChVectorDynamic<>& Fi) {
    assert(Fi.size() == 3 * NSF);

    if (m_method == IntFrcMethod::ContInt) {
        if (m_damping_enabled) {  // If linear Kelvin-Voigt viscoelastic material model is enabled
            ComputeInternalForcesContIntDamping(Fi);
        } else {
            ComputeInternalForcesContIntNoDamping(Fi);
        }
    } else {
        ComputeInternalForcesContIntPreInt(Fi);
    }

    if (m_gravity_on) {
        Fi += m_GravForce;
    }
}

// Calculate the global matrix H as a linear combination of K, R, and M:
//   H = Mfactor * [M] + Kfactor * [K] + Rfactor * [R]
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::ComputeKRMmatricesGlobal(ChMatrixRef H,
                                                              double Kfactor,
                                                              double Rfactor,
                                                              double Mfactor) {
    assert((H.rows() == 3 * NSF) && (H.cols() == 3 * NSF));

    if (m_method == IntFrcMethod::ContInt) {
        if (m_damping_enabled) {  // If linear Kelvin-Voigt viscoelastic material model is enabled
            ComputeInternalJacobianContIntDamping(H, -Kfactor, -Rfactor, Mfactor);
        } else {
            ComputeInternalJacobianContIntNoDamping(H, -Kfactor, Mfactor);
        }
    } else {
        ComputeInternalJacobianPreInt(H, Kfactor, Rfactor, Mfactor);
    }
}

// -----------------------------------------------------------------------------
// Interface to ChElementBeam base class (and similar methods)
// -----------------------------------------------------------------------------

template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::EvaluateSectionFrame(const double xi, ChVector<>& point, ChQuaternion<>& rot) {
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
    ChVector<double> BeamAxisTangent = e_bar * Sxi_xi_compact;
    ChVector<double> CrossSectionY = e_bar * Sxi_eta_compact;

    // Since the position vector gradients are not in general orthogonal,
    // set the Dx direction tangent to the beam axis and
    // compute the Dy and Dz directions by using a
    // Gram-Schmidt orthonormalization, guided by the cross section Y direction
    ChMatrix33<> msect;
    msect.Set_A_Xdir(BeamAxisTangent, CrossSectionY);

    rot = msect.Get_A_quaternion();
}

template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::EvaluateSectionPoint(const double xi, ChVector<>& point) {
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, 0, 0);

    Matrix3xN e_bar;
    CalcCoordMatrix(e_bar);

    // r = S*e written in compact form
    point = e_bar * Sxi_compact;
}

template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::EvaluateSectionVel(const double xi, ChVector<>& Result) {
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, 0, 0);

    Matrix3xN e_bardot;
    CalcCoordDerivMatrix(e_bardot);

    // rdot = S*edot written in compact form
    Result = e_bardot * Sxi_compact;
}

// -----------------------------------------------------------------------------
// Functions for ChLoadable interface
// -----------------------------------------------------------------------------

// Gets all the DOFs packed in a single vector (position part).
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::LoadableGetStateBlock_x(int block_offset, ChState& mD) {
    mD.segment(block_offset + 0, 3) = m_nodes[0]->GetPos().eigen();
    mD.segment(block_offset + 3, 3) = m_nodes[0]->GetD().eigen();
    mD.segment(block_offset + 6, 3) = m_nodes[0]->GetDD().eigen();
    mD.segment(block_offset + 9, 3) = m_nodes[0]->GetDDD().eigen();

    mD.segment(block_offset + 12, 3) = m_nodes[1]->GetPos().eigen();
    mD.segment(block_offset + 15, 3) = m_nodes[1]->GetD().eigen();
    mD.segment(block_offset + 18, 3) = m_nodes[1]->GetDD().eigen();
    mD.segment(block_offset + 21, 3) = m_nodes[1]->GetDDD().eigen();
}

// Gets all the DOFs packed in a single vector (velocity part).
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::LoadableGetStateBlock_w(int block_offset, ChStateDelta& mD) {
    mD.segment(block_offset + 0, 3) = m_nodes[0]->GetPos_dt().eigen();
    mD.segment(block_offset + 3, 3) = m_nodes[0]->GetD_dt().eigen();
    mD.segment(block_offset + 6, 3) = m_nodes[0]->GetDD_dt().eigen();
    mD.segment(block_offset + 9, 3) = m_nodes[0]->GetDDD_dt().eigen();

    mD.segment(block_offset + 12, 3) = m_nodes[1]->GetPos_dt().eigen();
    mD.segment(block_offset + 15, 3) = m_nodes[1]->GetD_dt().eigen();
    mD.segment(block_offset + 18, 3) = m_nodes[1]->GetDD_dt().eigen();
    mD.segment(block_offset + 21, 3) = m_nodes[1]->GetDDD_dt().eigen();
}

/// Increment all DOFs using a delta.
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::LoadableStateIncrement(const unsigned int off_x,
                                                            ChState& x_new,
                                                            const ChState& x,
                                                            const unsigned int off_v,
                                                            const ChStateDelta& Dv) {
    m_nodes[0]->NodeIntStateIncrement(off_x, x_new, x, off_v, Dv);
    m_nodes[1]->NodeIntStateIncrement(off_x + 12, x_new, x, off_v + 12, Dv);
}

// Get the pointers to the contained ChVariables, appending to the mvars vector.
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::LoadableGetVariables(std::vector<ChVariables*>& mvars) {
    for (int i = 0; i < m_nodes.size(); ++i) {
        mvars.push_back(&m_nodes[i]->Variables());
        mvars.push_back(&m_nodes[i]->Variables_D());
        mvars.push_back(&m_nodes[i]->Variables_DD());
        mvars.push_back(&m_nodes[i]->Variables_DDD());
    }
}

// Evaluate N'*F, which is the projection of the applied point force and moment at the midsurface coordinates (xi,eta,0)
// This calculation takes a slightly form for ANCF elements
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::ComputeNF(
    const double xi,             // parametric coordinate along the beam axis
    ChVectorDynamic<>& Qi,       // Return result of Q = N'*F  here
    double& detJ,                // Return det[J] here
    const ChVectorDynamic<>& F,  // Input F vector, size is =n. field coords.
    ChVectorDynamic<>* state_x,  // if != 0, update state (pos. part) to this, then evaluate Q
    ChVectorDynamic<>* state_w   // if != 0, update state (speed part) to this, then evaluate Q
) {
    ComputeNF(xi, 0, 0, Qi, detJ, F, state_x, state_w);
}

// Evaluate N'*F, which is the projection of the applied point force and moment at the coordinates (xi,eta,zeta)
// This calculation takes a slightly form for ANCF elements
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::ComputeNF(
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
    //  configuration used in the internal force calculations
    detJ = J_Cxi.determinant();
}

// Return the element density (needed for ChLoaderVolumeGravity).
template <int NP, int NT>
double ChElementBeamANCF_3243<NP, NT>::GetDensity() {
    return GetMaterial()->Get_rho();
}

// Calculate tangent to the centerline at coordinate xi [-1 to 1].
template <int NP, int NT>
ChVector<> ChElementBeamANCF_3243<NP, NT>::ComputeTangent(const double xi) {
    VectorN Sxi_xi_compact;
    Calc_Sxi_xi_compact(Sxi_xi_compact, xi, 0, 0);

    Matrix3xN e_bar;
    CalcCoordMatrix(e_bar);

    // partial derivative of the position vector with respect to xi (normalized coordinate along the beam axis).  In
    // general, this will not be a unit vector
    ChVector<double> BeamAxisTangent = e_bar * Sxi_xi_compact;

    return BeamAxisTangent.GetNormalized();
}

// -----------------------------------------------------------------------------
// Mass Matrix & Generalized Force Due to Gravity Calculation
// -----------------------------------------------------------------------------
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::ComputeMassMatrixAndGravityForce(const ChVector<>& g_acc) {
    // For this element, the mass matrix integrand is of order 10 in xi, 3 in eta, and 3 in zeta.
    // 4 GQ Points are needed in the xi direction and 2 GQ Points are needed in the eta and zeta directions for
    // exact integration of the element's mass matrix, even if the reference configuration is not straight Since the
    // major pieces of the generalized force due to gravity can also be used to calculate the mass matrix, these
    // calculations are performed at the same time.

    ChQuadratureTables* GQTable = GetStaticGQTables();
    unsigned int GQ_idx_xi = 4;        // 5 Point Gauss-Quadrature;
    unsigned int GQ_idx_eta_zeta = 1;  // 2 Point Gauss-Quadrature;

    // Mass Matrix in its compact matrix form.  Since the mass matrix is symmetric, just the upper diagonal entries will
    // be stored.
    ChMatrixNM<double, NSF, NSF> MassMatrixCompactSquare;

    // Set these to zeros since they will be incremented as the vector/matrix is calculated
    MassMatrixCompactSquare.setZero();
    m_GravForce.setZero();

    double rho = GetMaterial()->Get_rho();  // Density of the material for the element

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

                ChMatrixNM<double, 3, 3 * NSF> Sxi;  // Normalized Shape Function Matrix in expanded full (sparse) form
                Sxi.setZero();
                for (unsigned int s = 0; s < Sxi_compact.size(); s++) {
                    Sxi(0, 0 + (3 * s)) = Sxi_compact(s);
                    Sxi(1, 1 + (3 * s)) = Sxi_compact(s);
                    Sxi(2, 2 + (3 * s)) = Sxi_compact(s);
                }

                m_GravForce += (GQ_weight * rho * det_J_0xi) * Sxi.transpose() * g_acc.eigen();
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

// Precalculate constant matrices and scalars for the internal force calculations
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::PrecomputeInternalForceMatricesWeights() {
    if (m_method == IntFrcMethod::ContInt)
        PrecomputeInternalForceMatricesWeightsContInt();
    else
        PrecomputeInternalForceMatricesWeightsPreInt();
}

// Precalculate constant matrices for the internal force calculations using the "Continuous Integration" style method
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::PrecomputeInternalForceMatricesWeightsContInt() {
    ChQuadratureTables* GQTable = GetStaticGQTables();
    unsigned int GQ_idx_xi = NP - 1;        // Gauss-Quadrature table index for xi
    unsigned int GQ_idx_eta_zeta = NT - 1;  // Gauss-Quadrature table index for eta and zeta

    m_SD_D0.resize(NSF, 3 * NIP_D0);
    m_kGQ_D0.resize(NIP_D0, 1);
    m_SD_Dv.resize(NSF, 3 * NIP_Dv);
    m_kGQ_Dv.resize(NIP_Dv, 1);

    ChMatrixNM<double, NSF, 3> SD_precompute_D;

    // Precalculate the matrices of normalized shape function derivatives corrected for a potentially non-straight
    // reference configuration & GQ Weights times the determinant of the element Jacobian for later Calculating the
    // portion of the Selective Reduced Integration that does account for the Poisson effect
    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi].size(); it_xi++) {
        for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[GQ_idx_eta_zeta].size(); it_eta++) {
            for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[GQ_idx_eta_zeta].size(); it_zeta++) {
                double GQ_weight = GQTable->Weight[GQ_idx_xi][it_xi] * GQTable->Weight[GQ_idx_eta_zeta][it_eta] *
                                   GQTable->Weight[GQ_idx_eta_zeta][it_zeta];
                double xi = GQTable->Lroots[GQ_idx_xi][it_xi];
                double eta = GQTable->Lroots[GQ_idx_eta_zeta][it_eta];
                double zeta = GQTable->Lroots[GQ_idx_eta_zeta][it_zeta];
                auto index = it_zeta + it_eta * GQTable->Lroots[GQ_idx_eta_zeta].size() +
                             it_xi * GQTable->Lroots[GQ_idx_eta_zeta].size() * GQTable->Lroots[GQ_idx_eta_zeta].size();
                ChMatrix33<double>
                    J_0xi;         // Element Jacobian between the reference configuration and normalized configuration
                MatrixNx3c Sxi_D;  // Matrix of normalized shape function derivatives

                Calc_Sxi_D(Sxi_D, xi, eta, zeta);
                J_0xi.noalias() = m_ebar0 * Sxi_D;

                // Adjust the shape function derivative matrix to account for a potentially deformed reference state
                SD_precompute_D = Sxi_D * J_0xi.inverse();
                m_kGQ_D0(index) = -J_0xi.determinant() * GQ_weight;

                // Group all of the columns together in blocks, and then layer by layer across the entire element
                m_SD_D0.col(index) = SD_precompute_D.col(0);
                m_SD_D0.col(index + NIP_D0) = SD_precompute_D.col(1);
                m_SD_D0.col(index + 2 * NIP_D0) = SD_precompute_D.col(2);

                index++;
            }
        }
    }

    // Precalculate the matrices of normalized shape function derivatives corrected for a potentially non-straight
    // reference configuration & GQ Weights times the determinant of the element Jacobian for later Calculate the
    // portion of the Selective Reduced Integration that account for the Poisson effect, but only on the beam axis
    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi].size(); it_xi++) {
        double GQ_weight = GQTable->Weight[GQ_idx_xi][it_xi] * 2 * 2;
        double xi = GQTable->Lroots[GQ_idx_xi][it_xi];
        double eta = 0;
        double zeta = 0;
        ChMatrix33<double> J_0xi;  // Element Jacobian between the reference configuration and normalized configuration
        MatrixNx3c Sxi_D;          // Matrix of normalized shape function derivatives

        Calc_Sxi_D(Sxi_D, xi, eta, zeta);
        J_0xi.noalias() = m_ebar0 * Sxi_D;

        // Adjust the shape function derivative matrix to account for a potentially deformed reference state
        SD_precompute_D = Sxi_D * J_0xi.inverse();
        m_kGQ_Dv(it_xi) = -J_0xi.determinant() * GQ_weight;

        // Group all of the columns together in blocks, and then layer by layer across the entire element
        m_SD_Dv.col(it_xi) = SD_precompute_D.col(0);
        m_SD_Dv.col(it_xi + NIP_Dv) = SD_precompute_D.col(1);
        m_SD_Dv.col(it_xi + 2 * NIP_Dv) = SD_precompute_D.col(2);
    }
}

// Precalculate constant matrices for the internal force calculations using the "Pre-Integration Integration" style
// method
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::PrecomputeInternalForceMatricesWeightsPreInt() {
    ChQuadratureTables* GQTable = GetStaticGQTables();
    unsigned int GQ_idx_xi = NP - 1;        // Gauss-Quadrature table index for xi
    unsigned int GQ_idx_eta_zeta = NT - 1;  // Gauss-Quadrature table index for eta and zeta

    m_O1.resize(NSF * NSF, NSF * NSF);
    m_O2.resize(NSF * NSF, NSF * NSF);
    m_K3Compact.resize(NSF, NSF);
    m_K13Compact.resize(NSF, NSF);

    m_K13Compact.setZero();
    m_K3Compact.setZero();
    m_O1.setZero();

    // =============================================================================
    // =============================================================================
    // Calculate the contribution to constant matrices needed to account for the integration across the volume of the
    // beam element, excluding the Poisson effect
    // =============================================================================
    // =============================================================================

    // Get the components of the stiffness tensor in 6x6 matrix form that exclude the Poisson effect
    const ChVectorN<double, 6>& D0 = GetMaterial()->Get_D0();
    ChMatrixNM<double, 6, 6> D;
    D.setZero();
    D.diagonal() = D0;

    // Setup the stiffness tensor in block matrix form to make it easier to iterate through all 4 subscripts
    ChMatrixNM<double, 9, 9> D_block;
    D_block << D(0, 0), D(0, 5), D(0, 4), D(0, 5), D(0, 1), D(0, 3), D(0, 4), D(0, 3), D(0, 2), D(5, 0), D(5, 5),
        D(5, 4), D(5, 5), D(5, 1), D(5, 3), D(5, 4), D(5, 3), D(5, 2), D(4, 0), D(4, 5), D(4, 4), D(4, 5), D(4, 1),
        D(4, 3), D(4, 4), D(4, 3), D(4, 2), D(5, 0), D(5, 5), D(5, 4), D(5, 5), D(5, 1), D(5, 3), D(5, 4), D(5, 3),
        D(5, 2), D(1, 0), D(1, 5), D(1, 4), D(1, 5), D(1, 1), D(1, 3), D(1, 4), D(1, 3), D(1, 2), D(3, 0), D(3, 5),
        D(3, 4), D(3, 5), D(3, 1), D(3, 3), D(3, 4), D(3, 3), D(3, 2), D(4, 0), D(4, 5), D(4, 4), D(4, 5), D(4, 1),
        D(4, 3), D(4, 4), D(4, 3), D(4, 2), D(3, 0), D(3, 5), D(3, 4), D(3, 5), D(3, 1), D(3, 3), D(3, 4), D(3, 3),
        D(3, 2), D(2, 0), D(2, 5), D(2, 4), D(2, 5), D(2, 1), D(2, 3), D(2, 4), D(2, 3), D(2, 2);

    // Pre-calculate the matrices of normalized shape function derivatives corrected for a potentially non-straight
    // reference configuration & GQ Weights times the determinate of the element Jacobian for later Calculating the
    // portion of the Selective Reduced Integration that does account for the Poisson effect
    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi].size(); it_xi++) {
        for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[GQ_idx_eta_zeta].size(); it_eta++) {
            for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[GQ_idx_eta_zeta].size(); it_zeta++) {
                double GQ_weight = GQTable->Weight[GQ_idx_xi][it_xi] * GQTable->Weight[GQ_idx_eta_zeta][it_eta] *
                                   GQTable->Weight[GQ_idx_eta_zeta][it_zeta];
                double xi = GQTable->Lroots[GQ_idx_xi][it_xi];
                double eta = GQTable->Lroots[GQ_idx_eta_zeta][it_eta];
                double zeta = GQTable->Lroots[GQ_idx_eta_zeta][it_zeta];

                ChMatrix33<double>
                    J_0xi;         // Element Jacobian between the reference configuration and normalized configuration
                MatrixNx3c Sxi_D;  // Matrix of normalized shape function derivatives

                Calc_Sxi_D(Sxi_D, xi, eta, zeta);
                J_0xi.noalias() = m_ebar0 * Sxi_D;

                MatrixNx3c Sxi_D_0xi = Sxi_D * J_0xi.inverse();
                double GQWeight_det_J_0xi = -J_0xi.determinant() * GQ_weight;

                // Shortcut for:
                // m_K3Compact += GQWeight_det_J_0xi * 0.5 * (Sxi_D_0xi*D11*Sxi_D_0xi.transpose() + Sxi_D_0xi *
                // D22*Sxi_D_0xi.transpose() + Sxi_D_0xi * D33*Sxi_D_0xi.transpose());
                m_K3Compact += GQWeight_det_J_0xi * 0.5 *
                               (D0(0) * Sxi_D_0xi.template block<NSF, 1>(0, 0) *
                                    Sxi_D_0xi.template block<NSF, 1>(0, 0).transpose() +
                                D0(1) * Sxi_D_0xi.template block<NSF, 1>(0, 1) *
                                    Sxi_D_0xi.template block<NSF, 1>(0, 1).transpose() +
                                D0(2) * Sxi_D_0xi.template block<NSF, 1>(0, 2) *
                                    Sxi_D_0xi.template block<NSF, 1>(0, 2).transpose());

                MatrixNxN scale;
                for (unsigned int n = 0; n < 3; n++) {
                    for (unsigned int c = 0; c < 3; c++) {
                        scale = Sxi_D_0xi * D_block.block<3, 3>(3 * n, 3 * c) * Sxi_D_0xi.transpose();
                        scale *= GQWeight_det_J_0xi;

                        MatrixNxN Sxi_D_0xi_n_Sxi_D_0xi_c_transpose =
                            Sxi_D_0xi.template block<NSF, 1>(0, n) * Sxi_D_0xi.template block<NSF, 1>(0, c).transpose();
                        for (unsigned int f = 0; f < NSF; f++) {
                            for (unsigned int t = 0; t < NSF; t++) {
                                m_O1.block<NSF, NSF>(NSF * t, NSF * f) +=
                                    scale(t, f) * Sxi_D_0xi_n_Sxi_D_0xi_c_transpose;
                            }
                        }
                    }
                }
            }
        }
    }

    // =============================================================================
    // =============================================================================
    // Calculate the contribution to constant matrices needed to account for the integration across the volume of the
    // beam element, including the Poisson effect only along the beam axis (selective reduced integration)
    // =============================================================================
    // =============================================================================

    // Get the components of the stiffness tensor in 6x6 matrix form that include the Poisson effect

    D.setZero();
    D.block(0, 0, 3, 3) = GetMaterial()->Get_Dv();

    // =============================================================================
    // Pull the required entries from the 4th order stiffness tensor that are needed for K3.  Note that D is written
    // in Voigt notation rather than a 4th other tensor so the correct entries from that matrix need to be pulled.
    // D11 = slice (matrix) of the 4th order tensor with last two subscripts = 1)
    // D22 = slice (matrix) of the 4th order tensor with last two subscripts = 2)
    // D33 = slice (matrix) of the 4th order tensor with last two subscripts = 3)
    // =============================================================================
    ChMatrixNM<double, 3, 3> D11;
    ChMatrixNM<double, 3, 3> D22;
    ChMatrixNM<double, 3, 3> D33;
    D11(0, 0) = D(0, 0);
    D11(1, 1) = D(1, 0);
    D11(2, 2) = D(2, 0);
    D11(1, 0) = D(5, 0);
    D11(0, 1) = D(5, 0);
    D11(2, 0) = D(4, 0);
    D11(0, 2) = D(4, 0);
    D11(2, 1) = D(3, 0);
    D11(1, 2) = D(3, 0);

    D22(0, 0) = D(0, 1);
    D22(1, 1) = D(1, 1);
    D22(2, 2) = D(2, 1);
    D22(1, 0) = D(5, 1);
    D22(0, 1) = D(5, 1);
    D22(2, 0) = D(4, 1);
    D22(0, 2) = D(4, 1);
    D22(2, 1) = D(3, 1);
    D22(1, 2) = D(3, 1);

    D33(0, 0) = D(0, 2);
    D33(1, 1) = D(1, 2);
    D33(2, 2) = D(2, 2);
    D33(1, 0) = D(5, 2);
    D33(0, 1) = D(5, 2);
    D33(2, 0) = D(4, 2);
    D33(0, 2) = D(4, 2);
    D33(2, 1) = D(3, 2);
    D33(1, 2) = D(3, 2);

    // Setup the stiffness tensor in block matrix form to make it easier to iterate through all 4 subscripts
    D_block << D(0, 0), D(0, 5), D(0, 4), D(0, 5), D(0, 1), D(0, 3), D(0, 4), D(0, 3), D(0, 2), D(5, 0), D(5, 5),
        D(5, 4), D(5, 5), D(5, 1), D(5, 3), D(5, 4), D(5, 3), D(5, 2), D(4, 0), D(4, 5), D(4, 4), D(4, 5), D(4, 1),
        D(4, 3), D(4, 4), D(4, 3), D(4, 2), D(5, 0), D(5, 5), D(5, 4), D(5, 5), D(5, 1), D(5, 3), D(5, 4), D(5, 3),
        D(5, 2), D(1, 0), D(1, 5), D(1, 4), D(1, 5), D(1, 1), D(1, 3), D(1, 4), D(1, 3), D(1, 2), D(3, 0), D(3, 5),
        D(3, 4), D(3, 5), D(3, 1), D(3, 3), D(3, 4), D(3, 3), D(3, 2), D(4, 0), D(4, 5), D(4, 4), D(4, 5), D(4, 1),
        D(4, 3), D(4, 4), D(4, 3), D(4, 2), D(3, 0), D(3, 5), D(3, 4), D(3, 5), D(3, 1), D(3, 3), D(3, 4), D(3, 3),
        D(3, 2), D(2, 0), D(2, 5), D(2, 4), D(2, 5), D(2, 1), D(2, 3), D(2, 4), D(2, 3), D(2, 2);

    // Loop over each Gauss quadrature point and sum the contribution to each constant matrix that will be used for
    // the internal force calculations
    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi].size(); it_xi++) {
        double GQ_weight = GQTable->Weight[GQ_idx_xi][it_xi] * 2 * 2;
        double xi = GQTable->Lroots[GQ_idx_xi][it_xi];
        double eta = 0;
        double zeta = 0;

        ChMatrix33<double> J_0xi;  // Element Jacobian between the reference configuration and normalized configuration
        MatrixNx3c Sxi_D;          // Matrix of normalized shape function derivatives

        Calc_Sxi_D(Sxi_D, xi, eta, zeta);
        J_0xi.noalias() = m_ebar0 * Sxi_D;

        MatrixNx3c Sxi_D_0xi = Sxi_D * J_0xi.inverse();
        double GQWeight_det_J_0xi = -J_0xi.determinant() * GQ_weight;

        m_K3Compact += GQWeight_det_J_0xi * 0.5 *
                       (Sxi_D_0xi * D11 * Sxi_D_0xi.transpose() + Sxi_D_0xi * D22 * Sxi_D_0xi.transpose() +
                        Sxi_D_0xi * D33 * Sxi_D_0xi.transpose());

        MatrixNxN scale;
        for (unsigned int n = 0; n < 3; n++) {
            for (unsigned int c = 0; c < 3; c++) {
                scale = Sxi_D_0xi * D_block.block<3, 3>(3 * n, 3 * c) * Sxi_D_0xi.transpose();
                scale *= GQWeight_det_J_0xi;

                MatrixNxN Sxi_D_0xi_n_Sxi_D_0xi_c_transpose =
                    Sxi_D_0xi.template block<NSF, 1>(0, n) * Sxi_D_0xi.template block<NSF, 1>(0, c).transpose();
                for (unsigned int f = 0; f < NSF; f++) {
                    for (unsigned int t = 0; t < NSF; t++) {
                        m_O1.block<NSF, NSF>(NSF * t, NSF * f) += scale(t, f) * Sxi_D_0xi_n_Sxi_D_0xi_c_transpose;
                    }
                }
            }
        }
    }

    // Since O2 is just a reordered version of O1, wait until O1 is completely calculated and then generate O2 from it
    for (unsigned int f = 0; f < NSF; f++) {
        for (unsigned int t = 0; t < NSF; t++) {
            m_O2.block<NSF, NSF>(NSF * t, NSF * f) = m_O1.block<NSF, NSF>(NSF * t, NSF * f).transpose();
        }
    }
}

// -----------------------------------------------------------------------------
// Elastic force calculation
// -----------------------------------------------------------------------------

template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::ComputeInternalForcesContIntDamping(ChVectorDynamic<>& Fi) {
    // Calculate the generalize internal force vector using the "Continuous Integration" style of method assuming a
    // linear viscoelastic material model (single term damping model).  For this style of method, the generalized
    // internal force vector is integrated across the volume of the element every time this calculation is performed.
    // For this element, this is likely more efficient than the "Pre-Integration" style calculation method.  Note that
    // the integrand for the generalize internal force vector for a straight and normalized element is of order : 8 in
    // xi, 4 in eta, and 4 in zeta. This requires GQ 5 points along the xi direction and 3 GQ points along the eta and
    // zeta directions for "Full Integration". However, very similar results can be obtained with fewer GQ point in each
    // direction, resulting in significantly fewer calculations.  Based on testing, this could be as low as 3x2x2

    MatrixNx6 ebar_ebardot;
    CalcCombinedCoordMatrix(ebar_ebardot);

    // =============================================================================
    // =============================================================================
    // Since the Enhanced Continuum Mechanics method is utilized for this element, first calculate the contribution to
    // the generalized internal force vector from the terms that do not include the Poisson effect.  This is integrated
    // across the entire volume of the element using the full number of Gauss quadrature points.
    // =============================================================================
    // =============================================================================

    // =============================================================================
    // Calculate the deformation gradient and time derivative of the deformation gradient for all Gauss quadrature
    // points in a single matrix multiplication.  Note that since the shape function derivative matrix is ordered by
    // columns, the resulting deformation gradient will be ordered by block matrix (column vectors) components
    // Note that the indices of the components are in transposed order
    //      [F11  F21  F31  F11dot  F21dot  F31dot ]
    // FC = [F12  F22  F32  F12dot  F22dot  F32dot ]
    //      [F13  F23  F33  F13dot  F23dot  F33dot ]
    // =============================================================================

    ChMatrixNMc<double, 3 * NIP_D0, 6> FC_D0 = m_SD_D0.transpose() * ebar_ebardot;

    // =============================================================================
    // Get the diagonal terms of the 6x6 matrix (do not include the Poisson effect
    // =============================================================================
    const ChVectorN<double, 6>& D0 = GetMaterial()->Get_D0();

    // =============================================================================
    // Calculate each individual value of the Green-Lagrange strain component by component across all the
    // Gauss-Quadrature points at a time for the current layer to better leverage vectorized CPU instructions.
    // Note that the scaled time derivatives of the Green-Lagrange strain are added to make the later calculation of
    // the 2nd Piola-Kirchoff stresses more efficient.  The combined result is then scaled by minus the Gauss
    // quadrature weight times the element Jacobian at the corresponding Gauss point (m_kGQ) again for efficiency.
    // Since only the diagonal terms in the 6x6 stiffness matrix are used, the 2nd Piola-Kirchoff can be calculated by
    // simply scaling the vector of scaled Green-Lagrange strains in Voight notation by the corresponding diagonal entry
    // in the stiffness matrix.
    // Results are written in Voigt notation: epsilon = [E11,E22,E33,2*E23,2*E13,2*E12]
    //  kGQ*SPK2 = kGQ*[SPK2_11,SPK2_22,SPK2_33,SPK2_23,SPK2_13,SPK2_12] = D * E_Combined
    // =============================================================================

    // Each entry in SPK2_1 = D11*kGQ*(E11+alpha*E11dot)
    //     = D11*kGQ*(1/2*(F11*F11+F21*F21+F31*F31-1)+alpha*(F11*F11dot+F21*F21dot+F31*F31dot))
    VectorNIP_D0 E_BlockDamping_D0 =
        FC_D0.template block<NIP_D0, 1>(0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 3)) +
        FC_D0.template block<NIP_D0, 1>(0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 4)) +
        FC_D0.template block<NIP_D0, 1>(0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 5));
    VectorNIP_D0 SPK2_1_Block_D0 =
        FC_D0.template block<NIP_D0, 1>(0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 0)) +
        FC_D0.template block<NIP_D0, 1>(0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 1)) +
        FC_D0.template block<NIP_D0, 1>(0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 2));
    SPK2_1_Block_D0.array() -= 1;
    SPK2_1_Block_D0 += (2 * m_Alpha) * E_BlockDamping_D0;
    SPK2_1_Block_D0.array() *= m_kGQ_D0.array();
    SPK2_1_Block_D0 *= (0.5 * D0(0));

    // Each entry in SPK2_2 = D22*kGQ*(E22+alpha*E22dot)
    //     = D22*kGQ*(1/2*(F12*F12+F22*F22+F32*F32-1)+alpha*(F12*F12dot+F22*F22dot+F32*F32dot))
    E_BlockDamping_D0.noalias() =
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 3)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 4)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 5));
    VectorNIP_D0 SPK2_2_Block_D0 =
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 0)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 1)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 2));
    SPK2_2_Block_D0.array() -= 1;
    SPK2_2_Block_D0 += (2 * m_Alpha) * E_BlockDamping_D0;
    SPK2_2_Block_D0.array() *= m_kGQ_D0.array();
    SPK2_2_Block_D0 *= (0.5 * D0(1));

    // Each entry in SPK2_3 = D33*kGQ*(E33+alpha*E33dot)
    //     = D33*kGQ*(1/2*(F13*F13+F23*F23+F33*F33-1)+alpha*(F13*F13dot+F23*F23dot+F33*F33dot))
    E_BlockDamping_D0.noalias() =
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 3)) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 4)) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 5));
    VectorNIP_D0 SPK2_3_Block_D0 =
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0)) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1)) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2));
    SPK2_3_Block_D0.array() -= 1;
    SPK2_3_Block_D0 += (2 * m_Alpha) * E_BlockDamping_D0;
    SPK2_3_Block_D0.array() *= m_kGQ_D0.array();
    SPK2_3_Block_D0 *= (0.5 * D0(2));

    // Each entry in SPK2_4 = D44*kGQ*(2*(E23+alpha*E23dot))
    //     = D44*kGQ*((F12*F13+F22*F23+F32*F33)
    //       +alpha*(F12dot*F13+F22dot*F23+F32dot*F33 + F12*F13dot+F22*F23dot+F32*F33dot))
    E_BlockDamping_D0.noalias() =
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 3)) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 4)) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 5)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 3)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 4)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 5));
    VectorNIP_D0 SPK2_4_Block_D0 =
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2));
    SPK2_4_Block_D0 += m_Alpha * E_BlockDamping_D0;
    SPK2_4_Block_D0.array() *= m_kGQ_D0.array();
    SPK2_4_Block_D0 *= D0(3);

    // Each entry in SPK2_5 = D55*kGQ*(2*(E13+alpha*E13dot))
    //     = D55*kGQ*((F11*F13+F21*F23+F31*F33)
    //       +alpha*(F11dot*F13+F21dot*F23+F31dot*F33 + F11*F13dot+F21*F23dot+F31*F33dot))
    E_BlockDamping_D0.noalias() =
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 3)) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 4)) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 5)) +
        FC_D0.template block<NIP_D0, 1>(0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 3)) +
        FC_D0.template block<NIP_D0, 1>(0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 4)) +
        FC_D0.template block<NIP_D0, 1>(0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 5));
    VectorNIP_D0 SPK2_5_Block_D0 =
        FC_D0.template block<NIP_D0, 1>(0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0)) +
        FC_D0.template block<NIP_D0, 1>(0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1)) +
        FC_D0.template block<NIP_D0, 1>(0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2));
    SPK2_5_Block_D0 += m_Alpha * E_BlockDamping_D0;
    SPK2_5_Block_D0.array() *= m_kGQ_D0.array();
    SPK2_5_Block_D0 *= D0(4);

    // Each entry in SPK2_6 = D66*kGQ*(2*(E12+alpha*E12dot))
    //     = D66*kGQ*((F11*F12+F21*F22+F31*F32)
    //       +alpha*(F11dot*F12+F21dot*F22+F31dot*F32 + F11*F12dot+F21*F22dot+F31*F32dot))
    E_BlockDamping_D0.noalias() =
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 3)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 4)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 5)) +
        FC_D0.template block<NIP_D0, 1>(0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 3)) +
        FC_D0.template block<NIP_D0, 1>(0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 4)) +
        FC_D0.template block<NIP_D0, 1>(0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 5));
    VectorNIP_D0 SPK2_6_Block_D0 =
        FC_D0.template block<NIP_D0, 1>(0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 0)) +
        FC_D0.template block<NIP_D0, 1>(0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 1)) +
        FC_D0.template block<NIP_D0, 1>(0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 2));
    SPK2_6_Block_D0 += m_Alpha * E_BlockDamping_D0;
    SPK2_6_Block_D0.array() *= m_kGQ_D0.array();
    SPK2_6_Block_D0 *= D0(5);

    // =============================================================================
    // Calculate the transpose of the 1st Piola-Kirchoff stresses in block tensor form whose entries have been
    // scaled by minus the Gauss quadrature weight times the element Jacobian at the corresponding Gauss point.
    // The entries are grouped by component in block matrices (column vectors)
    // P_Block = kGQ*P_transpose = kGQ*SPK2*F_transpose
    //           [kGQ*(P_transpose)_11  kGQ*(P_transpose)_12  kGQ*(P_transpose)_13 ]
    //         = [kGQ*(P_transpose)_21  kGQ*(P_transpose)_22  kGQ*(P_transpose)_23 ]
    //           [kGQ*(P_transpose)_31  kGQ*(P_transpose)_32  kGQ*(P_transpose)_33 ]
    // =============================================================================

    ChMatrixNMc<double, 3 * NIP_D0, 3> P_Block_D0;

    P_Block_D0.template block<NIP_D0, 1>(0, 0) =
        FC_D0.template block<NIP_D0, 1>(0, 0).cwiseProduct(SPK2_1_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 0).cwiseProduct(SPK2_6_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0).cwiseProduct(SPK2_5_Block_D0);
    P_Block_D0.template block<NIP_D0, 1>(0, 1) =
        FC_D0.template block<NIP_D0, 1>(0, 1).cwiseProduct(SPK2_1_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 1).cwiseProduct(SPK2_6_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1).cwiseProduct(SPK2_5_Block_D0);
    P_Block_D0.template block<NIP_D0, 1>(0, 2) =
        FC_D0.template block<NIP_D0, 1>(0, 2).cwiseProduct(SPK2_1_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 2).cwiseProduct(SPK2_6_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2).cwiseProduct(SPK2_5_Block_D0);

    P_Block_D0.template block<NIP_D0, 1>(NIP_D0, 0) =
        FC_D0.template block<NIP_D0, 1>(0, 0).cwiseProduct(SPK2_6_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 0).cwiseProduct(SPK2_2_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0).cwiseProduct(SPK2_4_Block_D0);
    P_Block_D0.template block<NIP_D0, 1>(NIP_D0, 1) =
        FC_D0.template block<NIP_D0, 1>(0, 1).cwiseProduct(SPK2_6_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 1).cwiseProduct(SPK2_2_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1).cwiseProduct(SPK2_4_Block_D0);
    P_Block_D0.template block<NIP_D0, 1>(NIP_D0, 2) =
        FC_D0.template block<NIP_D0, 1>(0, 2).cwiseProduct(SPK2_6_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 2).cwiseProduct(SPK2_2_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2).cwiseProduct(SPK2_4_Block_D0);

    P_Block_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0) =
        FC_D0.template block<NIP_D0, 1>(0, 0).cwiseProduct(SPK2_5_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 0).cwiseProduct(SPK2_4_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0).cwiseProduct(SPK2_3_Block_D0);
    P_Block_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1) =
        FC_D0.template block<NIP_D0, 1>(0, 1).cwiseProduct(SPK2_5_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 1).cwiseProduct(SPK2_4_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1).cwiseProduct(SPK2_3_Block_D0);
    P_Block_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2) =
        FC_D0.template block<NIP_D0, 1>(0, 2).cwiseProduct(SPK2_5_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 2).cwiseProduct(SPK2_4_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2).cwiseProduct(SPK2_3_Block_D0);

    // =============================================================================
    // Multiply the scaled first Piola-Kirchoff stresses by the shape function derivative matrix for the current
    // layer to get the generalized force vector in matrix form (in the correct order if its calculated in row-major
    // memory layout)
    // =============================================================================

    MatrixNx3 QiCompact = m_SD_D0 * P_Block_D0;

    // =============================================================================
    // =============================================================================
    // Since the Enhanced Continuum Mechanics method is utilized for this element, second calculate the contribution to
    // the generalized internal force vector from the terms that include the Poisson effect.  This is integrated
    // across the entire volume of the element using Gauss quadrature points only along the beam axis.  It is assumed
    // that an isotropic material or orthotropic material aligned with its principle axes is used
    // =============================================================================
    // =============================================================================

    // =============================================================================
    // Calculate the deformation gradient and time derivative of the deformation gradient for all Gauss quadrature
    // points in a single matrix multiplication.  Note that since the shape function derivative matrix is ordered by
    // columns, the resulting deformation gradient will be ordered by block matrix (column vectors) components
    // Note that the indices of the components are in transposed order
    //      [F11  F21  F31  F11dot  F21dot  F31dot ]
    // FC = [F12  F22  F32  F12dot  F22dot  F32dot ]
    //      [F13  F23  F33  F13dot  F23dot  F33dot ]
    // =============================================================================

    ChMatrixNMc<double, 3 * NIP_Dv, 6> FC_Dv = m_SD_Dv.transpose() * ebar_ebardot;

    // =============================================================================
    // Calculate each individual value of the Green-Lagrange strain component by component across all the
    // Gauss-Quadrature points at a time for the current layer to better leverage vectorized CPU instructions.
    // Note that the scaled time derivatives of the Green-Lagrange strain are added to make the later calculation of
    // the 2nd Piola-Kirchoff stresses more efficient.  The combined result is then scaled by minus the Gauss
    // quadrature weight times the element Jacobian at the corresponding Gauss point (m_kGQ) again for efficiency.
    // Results are written in Voigt notation: epsilon = [E11,E22,E33,2*E23,2*E13,2*E12]
    // Note that with the material assumption being used, only [E11,E22,E33] need to be calculated
    // =============================================================================

    // Each entry in E1 = kGQ*(E11+alpha*E11dot)
    //                  = kGQ*(1/2*(F11*F11+F21*F21+F31*F31-1)+alpha*(F11*F11dot+F21*F21dot+F31*F31dot))
    VectorNIP_Dv E_BlockDamping_Dv =
        FC_Dv.template block<NIP_Dv, 1>(0, 0).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(0, 3)) +
        FC_Dv.template block<NIP_Dv, 1>(0, 1).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(0, 4)) +
        FC_Dv.template block<NIP_Dv, 1>(0, 2).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(0, 5));
    VectorNIP_Dv E1_Block_Dv =
        FC_Dv.template block<NIP_Dv, 1>(0, 0).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(0, 0)) +
        FC_Dv.template block<NIP_Dv, 1>(0, 1).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(0, 1)) +
        FC_Dv.template block<NIP_Dv, 1>(0, 2).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(0, 2));
    E1_Block_Dv.array() -= 1;
    E1_Block_Dv *= 0.5;
    E1_Block_Dv += m_Alpha * E_BlockDamping_Dv;
    E1_Block_Dv.array() *= m_kGQ_Dv.array();

    // Each entry in E2 = kGQ*(E22+alpha*E22dot)
    //                  = kGQ*(1/2*(F12*F12+F22*F22+F32*F32-1)+alpha*(F12*F12dot+F22*F22dot+F32*F32dot))
    E_BlockDamping_Dv.noalias() =
        FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 0).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 3)) +
        FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 1).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 4)) +
        FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 2).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 5));
    VectorNIP_Dv E2_Block_Dv =
        FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 0).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 0)) +
        FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 1).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 1)) +
        FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 2).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 2));
    E2_Block_Dv.array() -= 1;
    E2_Block_Dv *= 0.5;
    E2_Block_Dv += m_Alpha * E_BlockDamping_Dv;
    E2_Block_Dv.array() *= m_kGQ_Dv.array();

    // Each entry in E3 = kGQ*(E33+alpha*E33dot)
    //                  = kGQ*(1/2*(F13*F13+F23*F23+F33*F33-1)+alpha*(F13*F13dot+F23*F23dot+F33*F33dot))
    E_BlockDamping_Dv.noalias() =
        FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 0).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 3)) +
        FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 1).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 4)) +
        FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 2).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 5));
    VectorNIP_Dv E3_Block_Dv =
        FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 0).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 0)) +
        FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 1).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 1)) +
        FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 2).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 2));
    E3_Block_Dv.array() -= 1;
    E3_Block_Dv *= 0.5;
    E3_Block_Dv += m_Alpha * E_BlockDamping_Dv;
    E3_Block_Dv.array() *= m_kGQ_Dv.array();

    // =============================================================================
    // Get the upper 3x3 block of the stiffness tensor in 6x6 matrix that accounts for the Poisson effect.  Note that
    // with the split of the stiffness matrix and the assumed materials, the entries in the 6x6 stiffness matrix outside
    // of the upper 3x3 block are all zeros.
    // =============================================================================

    const ChMatrix33<double>& Dv = GetMaterial()->Get_Dv();

    // =============================================================================
    // Calculate the 2nd Piola-Kirchoff stresses in Voigt notation across all the Gauss quadrature points in the
    // current layer at a time component by component.
    // Note that the Green-Largrange strain components have been scaled have already been combined with their scaled
    // time derivatives and minus the Gauss quadrature weight times the element Jacobian at the corresponding Gauss
    // point to make the calculation of the 2nd Piola-Kirchoff stresses more efficient.
    //  kGQ*SPK2 = kGQ*[SPK2_11,SPK2_22,SPK2_33,SPK2_23,SPK2_13,SPK2_12] = D * E_Combined
    // =============================================================================

    VectorNIP_Dv SPK2_1_Block_Dv = Dv(0, 0) * E1_Block_Dv + Dv(0, 1) * E2_Block_Dv + Dv(0, 2) * E3_Block_Dv;
    VectorNIP_Dv SPK2_2_Block_Dv = Dv(1, 0) * E1_Block_Dv + Dv(1, 1) * E2_Block_Dv + Dv(1, 2) * E3_Block_Dv;
    VectorNIP_Dv SPK2_3_Block_Dv = Dv(2, 0) * E1_Block_Dv + Dv(2, 1) * E2_Block_Dv + Dv(2, 2) * E3_Block_Dv;

    // =============================================================================
    // Calculate the transpose of the 1st Piola-Kirchoff stresses in block tensor form whose entries have been
    // scaled by minus the Gauss quadrature weight times the element Jacobian at the corresponding Gauss point.
    // The entries are grouped by component in block matrices (column vectors)
    // P_Block = kGQ*P_transpose = kGQ*SPK2*F_transpose
    //           [kGQ*(P_transpose)_11  kGQ*(P_transpose)_12  kGQ*(P_transpose)_13 ]
    //         = [kGQ*(P_transpose)_21  kGQ*(P_transpose)_22  kGQ*(P_transpose)_23 ]
    //           [kGQ*(P_transpose)_31  kGQ*(P_transpose)_32  kGQ*(P_transpose)_33 ]
    // =============================================================================

    ChMatrixNMc<double, 3 * NIP_Dv, 3> P_Block_Dv;

    P_Block_Dv.template block<NIP_Dv, 1>(0, 0) = FC_Dv.template block<NIP_Dv, 1>(0, 0).cwiseProduct(SPK2_1_Block_Dv);
    P_Block_Dv.template block<NIP_Dv, 1>(0, 1) = FC_Dv.template block<NIP_Dv, 1>(0, 1).cwiseProduct(SPK2_1_Block_Dv);
    P_Block_Dv.template block<NIP_Dv, 1>(0, 2) = FC_Dv.template block<NIP_Dv, 1>(0, 2).cwiseProduct(SPK2_1_Block_Dv);

    P_Block_Dv.template block<NIP_Dv, 1>(NIP_Dv, 0) =
        FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 0).cwiseProduct(SPK2_2_Block_Dv);
    P_Block_Dv.template block<NIP_Dv, 1>(NIP_Dv, 1) =
        FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 1).cwiseProduct(SPK2_2_Block_Dv);
    P_Block_Dv.template block<NIP_Dv, 1>(NIP_Dv, 2) =
        FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 2).cwiseProduct(SPK2_2_Block_Dv);

    P_Block_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 0) =
        FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 0).cwiseProduct(SPK2_3_Block_Dv);
    P_Block_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 1) =
        FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 1).cwiseProduct(SPK2_3_Block_Dv);
    P_Block_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 2) =
        FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 2).cwiseProduct(SPK2_3_Block_Dv);

    // =============================================================================
    // Multiply the scaled first Piola-Kirchoff stresses by the shape function derivative matrix for the current
    // layer to get the generalized force vector in matrix form (in the correct order if its calculated in row-major
    // memory layout)
    // =============================================================================

    QiCompact += m_SD_Dv * P_Block_Dv;

    // =============================================================================
    // Reshape the compact matrix form of the generalized internal force vector (stored using a row-major memory layout)
    // into its actual column vector format.  This is done by mathematically stacking the transpose of each row on top
    // of each other forming the column vector.  Due to the memory organization this is simply a reinterpretation of the
    // data
    // =============================================================================

    Eigen::Map<Vector3N> QiReshaped(QiCompact.data(), QiCompact.size());
    Fi = QiReshaped;
}

template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::ComputeInternalForcesContIntNoDamping(ChVectorDynamic<>& Fi) {
    // Calculate the generalize internal force vector using the "Continuous Integration" style of method assuming a
    // linear material model (no damping).  For this style of method, the generalized internal force vector is
    // integrated across the volume of the element every time this calculation is performed. For this element, this is
    // likely more efficient than the "Pre-Integration" style calculation method.  Note that the integrand for the
    // generalize internal force vector for a straight and normalized element is of order : 8 in xi, 4 in eta, and 4 in
    // zeta. This requires GQ 5 points along the xi direction and 3 GQ points along the eta and zeta directions for
    // "Full Integration". However, very similar results can be obtained with fewer GQ point in each direction,
    // resulting in significantly fewer calculations.  Based on testing, this could be as low as 3x2x2

    Matrix3xN e_bar;
    CalcCoordMatrix(e_bar);

    // =============================================================================
    // =============================================================================
    // Since the Enhanced Continuum Mechanics method is utilized for this element, first calculate the contribution to
    // the generalized internal force vector from the terms that do not include the Poisson effect.  This is integrated
    // across the entire volume of the element using the full number of Gauss quadrature points.
    // =============================================================================
    // =============================================================================

    // =============================================================================
    // Calculate the deformation gradient for all Gauss quadrature points in a single matrix multiplication.  Note
    // that since the shape function derivative matrix is ordered by columns, the resulting deformation gradient
    // will be ordered by block matrix (column vectors) components
    // Note that the indices of the components are in transposed order
    //      [F11  F21  F31 ]
    // FC = [F12  F22  F32 ]
    //      [F13  F23  F33 ]
    // =============================================================================

    ChMatrixNMc<double, 3 * NIP_D0, 3> FC_D0 = m_SD_D0.transpose() * e_bar.transpose();

    // =============================================================================
    // Get the diagonal terms of the 6x6 matrix (do not include the Poisson effect
    // =============================================================================
    const ChVectorN<double, 6>& D0 = GetMaterial()->Get_D0();

    // =============================================================================
    // Calculate each individual value of the Green-Lagrange strain component by component across all the
    // Gauss-Quadrature points at a time for the current layer to better leverage vectorized CPU instructions.
    // The result is then scaled by minus the Gauss quadrature weight times the element Jacobian at the
    // corresponding Gauss point (m_kGQ) for efficiency.
    // Since only the diagonal terms in the 6x6 stiffness matrix are used, the 2nd Piola-Kirchoff can be calculated by
    // simply scaling the vector of scaled Green-Lagrange strains in Voight notation by the corresponding diagonal entry
    // in the stiffness matrix.
    // Results are written in Voigt notation: epsilon = [E11,E22,E33,2*E23,2*E13,2*E12]
    //  kGQ*SPK2 = kGQ*[SPK2_11,SPK2_22,SPK2_33,SPK2_23,SPK2_13,SPK2_12] = D * E_Combined
    // =============================================================================

    // Each entry in SPK2_1 = D11*kGQ*E11 = D11*kGQ*1/2*(F11*F11+F21*F21+F31*F31-1)
    VectorNIP_D0 SPK2_1_Block_D0 =
        FC_D0.template block<NIP_D0, 1>(0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 0)) +
        FC_D0.template block<NIP_D0, 1>(0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 1)) +
        FC_D0.template block<NIP_D0, 1>(0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 2));
    SPK2_1_Block_D0.array() -= 1;
    SPK2_1_Block_D0.array() *= (0.5 * D0(0)) * m_kGQ_D0.array();

    // Each entry in SPK2_2 = D22*kGQ*E22 = D22*kGQ*1/2*(F12*F12+F22*F22+F32*F32-1)
    VectorNIP_D0 SPK2_2_Block_D0 =
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 0)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 1)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 2));
    SPK2_2_Block_D0.array() -= 1;
    SPK2_2_Block_D0.array() *= (0.5 * D0(1)) * m_kGQ_D0.array();

    // Each entry in SPK2_3 = D33*kGQ_D0*E33 = D33*kGQ*1/2*(F13*F13+F23*F23+F33*F33-1)
    VectorNIP_D0 SPK2_3_Block_D0 =
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0)) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1)) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2));
    SPK2_3_Block_D0.array() -= 1;
    SPK2_3_Block_D0.array() *= (0.5 * D0(2)) * m_kGQ_D0.array();

    // Each entry in SPK2_4 = D44*kGQ*2*E23 = D44*kGQ*(F12*F13+F22*F23+F32*F33)
    VectorNIP_D0 SPK2_4_Block_D0 =
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2));
    SPK2_4_Block_D0.array() *= D0(3) * m_kGQ_D0.array();

    // Each entry in SPK2_5 = D55*kGQ*2*E13 = D55*kGQ*(F11*F13+F21*F23+F31*F33)
    VectorNIP_D0 SPK2_5_Block_D0 =
        FC_D0.template block<NIP_D0, 1>(0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0)) +
        FC_D0.template block<NIP_D0, 1>(0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1)) +
        FC_D0.template block<NIP_D0, 1>(0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2));
    SPK2_5_Block_D0.array() *= D0(4) * m_kGQ_D0.array();

    // Each entry in SPK2_6 = D66*kGQ*2*E12 = D66*(F11*F12+F21*F22+F31*F32)
    VectorNIP_D0 SPK2_6_Block_D0 =
        FC_D0.template block<NIP_D0, 1>(0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 0)) +
        FC_D0.template block<NIP_D0, 1>(0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 1)) +
        FC_D0.template block<NIP_D0, 1>(0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 2));
    SPK2_6_Block_D0.array() *= D0(5) * m_kGQ_D0.array();

    // =============================================================================
    // Calculate the transpose of the 1st Piola-Kirchoff stresses in block tensor form whose entries have been
    // scaled by minus the Gauss quadrature weight times the element Jacobian at the corresponding Gauss point.
    // The entries are grouped by component in block matrices (column vectors)
    // P_Block = kGQ*P_transpose = kGQ*SPK2*F_transpose
    //           [kGQ*(P_transpose)_11  kGQ*(P_transpose)_12  kGQ*(P_transpose)_13 ]
    //         = [kGQ*(P_transpose)_21  kGQ*(P_transpose)_22  kGQ*(P_transpose)_23 ]
    //           [kGQ*(P_transpose)_31  kGQ*(P_transpose)_32  kGQ*(P_transpose)_33 ]
    // =============================================================================

    ChMatrixNMc<double, 3 * NIP_D0, 3> P_Block_D0;

    P_Block_D0.template block<NIP_D0, 1>(0, 0) =
        FC_D0.template block<NIP_D0, 1>(0, 0).cwiseProduct(SPK2_1_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 0).cwiseProduct(SPK2_6_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0).cwiseProduct(SPK2_5_Block_D0);
    P_Block_D0.template block<NIP_D0, 1>(0, 1) =
        FC_D0.template block<NIP_D0, 1>(0, 1).cwiseProduct(SPK2_1_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 1).cwiseProduct(SPK2_6_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1).cwiseProduct(SPK2_5_Block_D0);
    P_Block_D0.template block<NIP_D0, 1>(0, 2) =
        FC_D0.template block<NIP_D0, 1>(0, 2).cwiseProduct(SPK2_1_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 2).cwiseProduct(SPK2_6_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2).cwiseProduct(SPK2_5_Block_D0);

    P_Block_D0.template block<NIP_D0, 1>(NIP_D0, 0) =
        FC_D0.template block<NIP_D0, 1>(0, 0).cwiseProduct(SPK2_6_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 0).cwiseProduct(SPK2_2_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0).cwiseProduct(SPK2_4_Block_D0);
    P_Block_D0.template block<NIP_D0, 1>(NIP_D0, 1) =
        FC_D0.template block<NIP_D0, 1>(0, 1).cwiseProduct(SPK2_6_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 1).cwiseProduct(SPK2_2_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1).cwiseProduct(SPK2_4_Block_D0);
    P_Block_D0.template block<NIP_D0, 1>(NIP_D0, 2) =
        FC_D0.template block<NIP_D0, 1>(0, 2).cwiseProduct(SPK2_6_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 2).cwiseProduct(SPK2_2_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2).cwiseProduct(SPK2_4_Block_D0);

    P_Block_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0) =
        FC_D0.template block<NIP_D0, 1>(0, 0).cwiseProduct(SPK2_5_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 0).cwiseProduct(SPK2_4_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0).cwiseProduct(SPK2_3_Block_D0);
    P_Block_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1) =
        FC_D0.template block<NIP_D0, 1>(0, 1).cwiseProduct(SPK2_5_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 1).cwiseProduct(SPK2_4_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1).cwiseProduct(SPK2_3_Block_D0);
    P_Block_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2) =
        FC_D0.template block<NIP_D0, 1>(0, 2).cwiseProduct(SPK2_5_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 2).cwiseProduct(SPK2_4_Block_D0) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2).cwiseProduct(SPK2_3_Block_D0);

    // =============================================================================
    // Multiply the scaled first Piola-Kirchoff stresses by the shape function derivative matrix for the current
    // layer to get the generalized force vector in matrix form (in the correct order if its calculated in row-major
    // memory layout)
    // =============================================================================

    MatrixNx3 QiCompact = m_SD_D0 * P_Block_D0;

    // =============================================================================
    // =============================================================================
    // Since the Enhanced Continuum Mechanics method is utilized for this element, second calculate the contribution to
    // the generalized internal force vector from the terms that include the Poisson effect.  This is integrated
    // across the entire volume of the element using Gauss quadrature points only along the beam axis.  It is assumed
    // that an isotropic material or orthotropic material aligned with its principle axes is used
    // =============================================================================
    // =============================================================================

    // =============================================================================
    // Calculate the deformation gradient for all Gauss quadrature points in a single matrix multiplication.  Note
    // that since the shape function derivative matrix is ordered by columns, the resulting deformation gradient
    // will be ordered by block matrix (column vectors) components
    // Note that the indices of the components are in transposed order
    //      [F11  F21  F31 ]
    // FC = [F12  F22  F32 ]
    //      [F13  F23  F33 ]
    // =============================================================================

    ChMatrixNMc<double, 3 * NIP_Dv, 3> FC_Dv = m_SD_Dv.transpose() * e_bar.transpose();

    // =============================================================================
    // Calculate each individual value of the Green-Lagrange strain component by component across all the
    // Gauss-Quadrature points at a time for the current layer to better leverage vectorized CPU instructions.
    // The combined result is then scaled by minus the Gauss quadrature weight times the element Jacobian at the
    // corresponding Gauss point (m_kGQ) again for efficiency. Results are written in Voigt notation: epsilon =
    // [E11,E22,E33,2*E23,2*E13,2*E12] Note that with the material assumption being used, only [E11,E22,E33] need to be
    // calculated
    // =============================================================================

    // Each entry in E1 = kGQ*E11 = kGQ*1/2*(F11*F11+F21*F21+F31*F31-1)
    VectorNIP_Dv E1_Block_Dv =
        FC_Dv.template block<NIP_Dv, 1>(0, 0).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(0, 0)) +
        FC_Dv.template block<NIP_Dv, 1>(0, 1).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(0, 1)) +
        FC_Dv.template block<NIP_Dv, 1>(0, 2).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(0, 2));
    E1_Block_Dv.array() -= 1;
    E1_Block_Dv.array() *= 0.5 * m_kGQ_Dv.array();

    // Each entry in E2 = kGQ*E22 = kGQ*1/2*(F12*F12+F22*F22+F32*F32-1)
    VectorNIP_Dv E2_Block_Dv =
        FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 0).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 0)) +
        FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 1).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 1)) +
        FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 2).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 2));
    E2_Block_Dv.array() -= 1;
    E2_Block_Dv.array() *= 0.5 * m_kGQ_Dv.array();

    // Each entry in E3 = kGQ*E33 = kGQ*1/2*(F13*F13+F23*F23+F33*F33-1)
    VectorNIP_Dv E3_Block_Dv =
        FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 0).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 0)) +
        FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 1).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 1)) +
        FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 2).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 2));
    E3_Block_Dv.array() -= 1;
    E3_Block_Dv.array() *= 0.5 * m_kGQ_Dv.array();

    // =============================================================================
    // Get the upper 3x3 block of the stiffness tensor in 6x6 matrix that accounts for the Poisson effect.  Note that
    // with the split of the stiffness matrix and the assumed materials, the entries in the 6x6 stiffness matrix outside
    // of the upper 3x3 block are all zeros.
    // =============================================================================

    const ChMatrix33<double>& Dv = GetMaterial()->Get_Dv();

    // =============================================================================
    // Calculate the 2nd Piola-Kirchoff stresses in Voigt notation across all the Gauss quadrature points in the
    // current layer at a time component by component.
    // Note that the Green-Largrange strain components have been scaled have already been combined with their scaled
    // time derivatives and minus the Gauss quadrature weight times the element Jacobian at the corresponding Gauss
    // point to make the calculation of the 2nd Piola-Kirchoff stresses more efficient.
    //  kGQ*SPK2 = kGQ*[SPK2_11,SPK2_22,SPK2_33,SPK2_23,SPK2_13,SPK2_12] = D * E_Combined
    // =============================================================================

    VectorNIP_Dv SPK2_1_Block_Dv = Dv(0, 0) * E1_Block_Dv + Dv(0, 1) * E2_Block_Dv + Dv(0, 2) * E3_Block_Dv;
    VectorNIP_Dv SPK2_2_Block_Dv = Dv(1, 0) * E1_Block_Dv + Dv(1, 1) * E2_Block_Dv + Dv(1, 2) * E3_Block_Dv;
    VectorNIP_Dv SPK2_3_Block_Dv = Dv(2, 0) * E1_Block_Dv + Dv(2, 1) * E2_Block_Dv + Dv(2, 2) * E3_Block_Dv;

    // =============================================================================
    // Calculate the transpose of the 1st Piola-Kirchoff stresses in block tensor form whose entries have been
    // scaled by minus the Gauss quadrature weight times the element Jacobian at the corresponding Gauss point.
    // The entries are grouped by component in block matrices (column vectors)
    // P_Block = kGQ*P_transpose = kGQ*SPK2*F_transpose
    //           [kGQ*(P_transpose)_11  kGQ*(P_transpose)_12  kGQ*(P_transpose)_13 ]
    //         = [kGQ*(P_transpose)_21  kGQ*(P_transpose)_22  kGQ*(P_transpose)_23 ]
    //           [kGQ*(P_transpose)_31  kGQ*(P_transpose)_32  kGQ*(P_transpose)_33 ]
    // =============================================================================

    ChMatrixNMc<double, 3 * NIP_Dv, 3> P_Block_Dv;

    P_Block_Dv.template block<NIP_Dv, 1>(0, 0) = FC_Dv.template block<NIP_Dv, 1>(0, 0).cwiseProduct(SPK2_1_Block_Dv);
    P_Block_Dv.template block<NIP_Dv, 1>(0, 1) = FC_Dv.template block<NIP_Dv, 1>(0, 1).cwiseProduct(SPK2_1_Block_Dv);
    P_Block_Dv.template block<NIP_Dv, 1>(0, 2) = FC_Dv.template block<NIP_Dv, 1>(0, 2).cwiseProduct(SPK2_1_Block_Dv);

    P_Block_Dv.template block<NIP_Dv, 1>(NIP_Dv, 0) =
        FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 0).cwiseProduct(SPK2_2_Block_Dv);
    P_Block_Dv.template block<NIP_Dv, 1>(NIP_Dv, 1) =
        FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 1).cwiseProduct(SPK2_2_Block_Dv);
    P_Block_Dv.template block<NIP_Dv, 1>(NIP_Dv, 2) =
        FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 2).cwiseProduct(SPK2_2_Block_Dv);

    P_Block_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 0) =
        FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 0).cwiseProduct(SPK2_3_Block_Dv);
    P_Block_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 1) =
        FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 1).cwiseProduct(SPK2_3_Block_Dv);
    P_Block_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 2) =
        FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 2).cwiseProduct(SPK2_3_Block_Dv);

    // =============================================================================
    // Multiply the scaled first Piola-Kirchoff stresses by the shape function derivative matrix for the current
    // layer to get the generalized force vector in matrix form (in the correct order if its calculated in row-major
    // memory layout)
    // =============================================================================

    QiCompact += m_SD_Dv * P_Block_Dv;

    // =============================================================================
    // Reshape the compact matrix form of the generalized internal force vector (stored using a row-major memory layout)
    // into its actual column vector format.  This is done by mathematically stacking the transpose of each row on top
    // of each other forming the column vector.  Due to the memory organization this is simply a reinterpretation of the
    // data
    // =============================================================================

    Eigen::Map<Vector3N> QiReshaped(QiCompact.data(), QiCompact.size());
    Fi = QiReshaped;
}

template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::ComputeInternalForcesContIntPreInt(ChVectorDynamic<>& Fi) {
    // Calculate the generalize internal force vector using the "Pre-Integration" style of method assuming a
    // linear viscoelastic material model (single term damping model).  For this style of method, the components of the
    // generalized internal force vector and its Jacobian that need to be integrated across the volume are calculated
    // once prior to the start of the simulation.  This makes this method well suited for applications with many
    // discrete layers since the in-simulation calculations are independent of the number of Gauss quadrature points
    // used throughout the entire element.

    Matrix3xN ebar;
    Matrix3xN ebardot;

    CalcCoordMatrix(ebar);
    CalcCoordDerivMatrix(ebardot);

    // Calculate PI1 which is a combined form of the nodal coordinates.  It is calculated in matrix form and then later
    // reshaped into vector format (through a simple reinterpretation of the data)
    MatrixNxN PI1_matrix = 0.5 * ebar.transpose() * ebar;

    // If damping is enabled adjust PI1 to account for the extra terms.  This is the only modification required to
    // include damping in the generalized internal force calculation
    if (m_damping_enabled) {
        PI1_matrix += m_Alpha * ebardot.transpose() * ebar;
    }

    MatrixNxN K1_matrix;

    // Setup the reshaped/reinterpreted/mapped forms of PI1 and K1 to make the calculation of K1 simpler
    Eigen::Map<ChVectorN<double, NSF * NSF>> PI1(PI1_matrix.data(), PI1_matrix.size());
    Eigen::Map<ChVectorN<double, NSF * NSF>> K1_vec(K1_matrix.data(), K1_matrix.size());

    // Calculate the matrix K1 in mapped vector form and the resulting matrix will be in the correct form to combine
    // with K3
    K1_vec.noalias() = m_O1 * PI1;

    // Store the combined sum of K1 and K3 since it will be used again in the Jacobian calculation
    m_K13Compact.noalias() = K1_matrix - m_K3Compact;

    // Multiply the combined K1 and K3 matrix by the nodal coordinates in compact form and then remap it into the
    // required vector order that is the generalized internal force vector
    MatrixNx3 QiCompactLiu = m_K13Compact * ebar.transpose();
    Eigen::Map<Vector3N> QiReshapedLiu(QiCompactLiu.data(), QiCompactLiu.size());

    Fi = QiReshapedLiu;
}

// -----------------------------------------------------------------------------
// Jacobians of internal forces
// -----------------------------------------------------------------------------

template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::ComputeInternalJacobianContIntDamping(ChMatrixRef& H,
                                                                           double Kfactor,
                                                                           double Rfactor,
                                                                           double Mfactor) {
    // Calculate the Jacobian of the generalize internal force vector using the "Continuous Integration" style of method
    // assuming a linear viscoelastic material model (single term damping model).  For this style of method, the
    // Jacobian of the generalized internal force vector is integrated across the volume of the element every time this
    // calculation is performed. For this element, this is likely more efficient than the "Pre-Integration" style
    // calculation method.

    MatrixNx6 ebar_ebardot;
    CalcCombinedCoordMatrix(ebar_ebardot);

    // No values from the generalized internal force vector are cached for reuse in the Jacobian.  Instead these
    // quantities are recalculated again during the Jacobian calculations.  This both simplifies the code and speeds
    // up the generalized internal force calculation while only having a minimal impact on the performance of the
    // Jacobian calculation speed.  The Jacobian calculation is performed in two major pieces.  First is the
    // calculation of the potentially non-symmetric and non-sparse component.  The second pieces is the symmetric
    // and sparse component.

    // =============================================================================
    // Calculate the deformation gradient and time derivative of the deformation gradient for all Gauss quadrature
    // points in a single matrix multiplication.  Note that since the shape function derivative matrix is ordered by
    // columns, the resulting deformation gradient will be ordered by block matrix (column vectors) components
    // Note that the indices of the components are in transposed order
    //      [F11  F21  F31  F11dot  F21dot  F31dot ]
    // FC = [F12  F22  F32  F12dot  F22dot  F32dot ]
    //      [F13  F23  F33  F13dot  F23dot  F33dot ]
    // =============================================================================

    ChMatrixNMc<double, 3 * NIP_D0, 6> FC_D0 = m_SD_D0.transpose() * ebar_ebardot;

    //==============================================================================
    //==============================================================================
    // Calculate the potentially non-symmetric and non-sparse component of the Jacobian matrix
    //==============================================================================
    //==============================================================================

    // =============================================================================
    // Calculate the partial derivative of the Green-Largrange strains with respect to the nodal coordinates
    // (transposed).  This calculation is performed in blocks across all the Gauss quadrature points at the same
    // time.  This value should be store in row major memory layout to align with the access patterns for
    // calculating this matrix.
    // PE = [(d epsilon1/d e)GQPnt1' (d epsilon1/d e)GQPnt2' ... (d epsilon1/d e)GQPntNIP'...
    //       (d epsilon2/d e)GQPnt1' (d epsilon2/d e)GQPnt2' ... (d epsilon2/d e)GQPntNIP'...
    //       (d epsilon3/d e)GQPnt1' (d epsilon3/d e)GQPnt2' ... (d epsilon3/d e)GQPntNIP'...
    //       (d epsilon4/d e)GQPnt1' (d epsilon4/d e)GQPnt2' ... (d epsilon4/d e)GQPntNIP'...
    //       (d epsilon5/d e)GQPnt1' (d epsilon5/d e)GQPnt2' ... (d epsilon5/d e)GQPntNIP'...
    //       (d epsilon6/d e)GQPnt1' (d epsilon6/d e)GQPnt2' ... (d epsilon6/d e)GQPntNIP']
    // Note that each partial derivative block shown is placed to the left of the previous block.
    // The explanation of the calculation above is just too long to write it all on a single line.
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> PE;
    PE.resize(3 * NSF, 6 * NIP_D0);

    for (auto i = 0; i < NSF; i++) {
        PE.block<1, NIP_D0>(3 * i, 0 * NIP_D0) =
            m_SD_D0.block<1, NIP_D0>(i, 0 * NIP_D0)
                .cwiseProduct(FC_D0.template block<NIP_D0, 1>(0 * NIP_D0, 0).transpose());
        PE.block<1, NIP_D0>(3 * i, 1 * NIP_D0) =
            m_SD_D0.block<1, NIP_D0>(i, 1 * NIP_D0)
                .cwiseProduct(FC_D0.template block<NIP_D0, 1>(1 * NIP_D0, 0).transpose());
        PE.block<1, NIP_D0>(3 * i, 2 * NIP_D0) =
            m_SD_D0.block<1, NIP_D0>(i, 2 * NIP_D0)
                .cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0).transpose());
        PE.block<1, NIP_D0>(3 * i, 3 * NIP_D0) =
            m_SD_D0.block<1, NIP_D0>(i, 2 * NIP_D0)
                .cwiseProduct(FC_D0.template block<NIP_D0, 1>(1 * NIP_D0, 0).transpose()) +
            m_SD_D0.block<1, NIP_D0>(i, 1 * NIP_D0)
                .cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0).transpose());
        PE.block<1, NIP_D0>(3 * i, 4 * NIP_D0) =
            m_SD_D0.block<1, NIP_D0>(i, 2 * NIP_D0)
                .cwiseProduct(FC_D0.template block<NIP_D0, 1>(0 * NIP_D0, 0).transpose()) +
            m_SD_D0.block<1, NIP_D0>(i, 0 * NIP_D0)
                .cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0).transpose());
        PE.block<1, NIP_D0>(3 * i, 5 * NIP_D0) =
            m_SD_D0.block<1, NIP_D0>(i, 1 * NIP_D0)
                .cwiseProduct(FC_D0.template block<NIP_D0, 1>(0 * NIP_D0, 0).transpose()) +
            m_SD_D0.block<1, NIP_D0>(i, 0 * NIP_D0)
                .cwiseProduct(FC_D0.template block<NIP_D0, 1>(1 * NIP_D0, 0).transpose());

        PE.block<1, NIP_D0>((3 * i) + 1, 0) =
            m_SD_D0.block<1, NIP_D0>(i, 0 * NIP_D0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 1).transpose());
        PE.block<1, NIP_D0>((3 * i) + 1, NIP_D0) =
            m_SD_D0.block<1, NIP_D0>(i, 1 * NIP_D0)
                .cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 1).transpose());
        PE.block<1, NIP_D0>((3 * i) + 1, 2 * NIP_D0) =
            m_SD_D0.block<1, NIP_D0>(i, 2 * NIP_D0)
                .cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1).transpose());
        PE.block<1, NIP_D0>((3 * i) + 1, 3 * NIP_D0) =
            m_SD_D0.block<1, NIP_D0>(i, 2 * NIP_D0)
                .cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 1).transpose()) +
            m_SD_D0.block<1, NIP_D0>(i, 1 * NIP_D0)
                .cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1).transpose());
        PE.block<1, NIP_D0>((3 * i) + 1, 4 * NIP_D0) =
            m_SD_D0.block<1, NIP_D0>(i, 2 * NIP_D0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 1).transpose()) +
            m_SD_D0.block<1, NIP_D0>(i, 0 * NIP_D0)
                .cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1).transpose());
        PE.block<1, NIP_D0>((3 * i) + 1, 5 * NIP_D0) =
            m_SD_D0.block<1, NIP_D0>(i, 1 * NIP_D0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 1).transpose()) +
            m_SD_D0.block<1, NIP_D0>(i, 0 * NIP_D0)
                .cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 1).transpose());

        PE.block<1, NIP_D0>((3 * i) + 2, 0) =
            m_SD_D0.block<1, NIP_D0>(i, 0 * NIP_D0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 2).transpose());
        PE.block<1, NIP_D0>((3 * i) + 2, NIP_D0) =
            m_SD_D0.block<1, NIP_D0>(i, 1 * NIP_D0)
                .cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 2).transpose());
        PE.block<1, NIP_D0>((3 * i) + 2, 2 * NIP_D0) =
            m_SD_D0.block<1, NIP_D0>(i, 2 * NIP_D0)
                .cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2).transpose());
        PE.block<1, NIP_D0>((3 * i) + 2, 3 * NIP_D0) =
            m_SD_D0.block<1, NIP_D0>(i, 2 * NIP_D0)
                .cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 2).transpose()) +
            m_SD_D0.block<1, NIP_D0>(i, 1 * NIP_D0)
                .cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2).transpose());
        PE.block<1, NIP_D0>((3 * i) + 2, 4 * NIP_D0) =
            m_SD_D0.block<1, NIP_D0>(i, 2 * NIP_D0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 2).transpose()) +
            m_SD_D0.block<1, NIP_D0>(i, 0 * NIP_D0)
                .cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2).transpose());
        PE.block<1, NIP_D0>((3 * i) + 2, 5 * NIP_D0) =
            m_SD_D0.block<1, NIP_D0>(i, 1 * NIP_D0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 2).transpose()) +
            m_SD_D0.block<1, NIP_D0>(i, 0 * NIP_D0)
                .cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 2).transpose());
    }

    // =============================================================================
    // Combine the deformation gradient, time derivative of the deformation gradient, damping coefficient, Gauss
    // quadrature weighting times the element Jacobian, and Jacobian component scale factors into a scaled block
    // deformation gradient matrix Calculate the deformation gradient and time derivative of the deformation
    // gradient for all Gauss quadrature points in a single matrix multiplication.  Note that the resulting combined
    // deformation gradient block matrix will be ordered by block matrix (column vectors) components Note that the
    // indices of the components are in transposed order
    //            [kGQ*(Kfactor+alpha*Rfactor)*F11+alpha*Rfactor*F11dot ... similar for F21 & F31 blocks]
    // FCscaled = [kGQ*(Kfactor+alpha*Rfactor)*F12+alpha*Rfactor*F12dot ... similar for F22 & F32 blocks]
    //            [kGQ*(Kfactor+alpha*Rfactor)*F13+alpha*Rfactor*F13dot ... similar for F23 & F33 blocks]
    // =============================================================================

    ChMatrixNMc<double, 3 * NIP_D0, 3> FCscaled_D0 =
        (Kfactor + m_Alpha * Rfactor) * FC_D0.template block<3 * NIP_D0, 3>(0, 0) +
        (m_Alpha * Kfactor) * FC_D0.template block<3 * NIP_D0, 3>(0, 3);

    for (auto i = 0; i < 3; i++) {
        FCscaled_D0.template block<NIP_D0, 1>(0, i).array() *= m_kGQ_D0.array();
        FCscaled_D0.template block<NIP_D0, 1>(NIP_D0, i).array() *= m_kGQ_D0.array();
        FCscaled_D0.template block<NIP_D0, 1>(2 * NIP_D0, i).array() *= m_kGQ_D0.array();
    }

    // =============================================================================
    // Get the diagonal terms of the 6x6 matrix (do not include the Poisson effect
    // =============================================================================
    const ChVectorN<double, 6>& D0 = GetMaterial()->Get_D0();

    // =============================================================================
    // Calculate the combination of the scaled partial derivative of the Green-Largrange strains with respect to the
    // nodal coordinates, the scaled partial derivative of the time derivative of the Green-Largrange strains with
    // respect to the nodal coordinates, the scaled partial derivative of the Green-Largrange strains with respect
    // to the time derivative of the nodal coordinates, and the other parameters to correctly integrate across the
    // volume of the element.  This calculation is performed in blocks across all the Gauss quadrature points at the
    // same time.  This value should be store in row major memory layout to align with the access patterns for
    // calculating this matrix.
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DScaled_Combined_PE;
    DScaled_Combined_PE.resize(3 * NSF, 6 * NIP_D0);

    for (auto i = 0; i < NSF; i++) {
        DScaled_Combined_PE.block<1, NIP_D0>(3 * i, 0) =
            D0(0) * m_SD_D0.block<1, NIP_D0>(i, 0 * NIP_D0)
                        .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(0, 0).transpose());
        DScaled_Combined_PE.block<1, NIP_D0>(3 * i, NIP_D0) =
            D0(1) * m_SD_D0.block<1, NIP_D0>(i, 1 * NIP_D0)
                        .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(NIP_D0, 0).transpose());
        DScaled_Combined_PE.block<1, NIP_D0>(3 * i, 2 * NIP_D0) =
            D0(2) * m_SD_D0.block<1, NIP_D0>(i, 2 * NIP_D0)
                        .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0).transpose());
        DScaled_Combined_PE.block<1, NIP_D0>(3 * i, 3 * NIP_D0) =
            D0(3) * (m_SD_D0.block<1, NIP_D0>(i, 2 * NIP_D0)
                         .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(NIP_D0, 0).transpose()) +
                     m_SD_D0.block<1, NIP_D0>(i, 1 * NIP_D0)
                         .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0).transpose()));
        DScaled_Combined_PE.block<1, NIP_D0>(3 * i, 4 * NIP_D0) =
            D0(4) * (m_SD_D0.block<1, NIP_D0>(i, 2 * NIP_D0)
                         .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(0, 0).transpose()) +
                     m_SD_D0.block<1, NIP_D0>(i, 0 * NIP_D0)
                         .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0).transpose()));
        DScaled_Combined_PE.block<1, NIP_D0>(3 * i, 5 * NIP_D0) =
            D0(5) * (m_SD_D0.block<1, NIP_D0>(i, 1 * NIP_D0)
                         .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(0, 0).transpose()) +
                     m_SD_D0.block<1, NIP_D0>(i, 0 * NIP_D0)
                         .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(NIP_D0, 0).transpose()));

        DScaled_Combined_PE.block<1, NIP_D0>((3 * i) + 1, 0) =
            D0(0) * m_SD_D0.block<1, NIP_D0>(i, 0 * NIP_D0)
                        .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(0, 1).transpose());
        DScaled_Combined_PE.block<1, NIP_D0>((3 * i) + 1, NIP_D0) =
            D0(1) * m_SD_D0.block<1, NIP_D0>(i, 1 * NIP_D0)
                        .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(NIP_D0, 1).transpose());
        DScaled_Combined_PE.block<1, NIP_D0>((3 * i) + 1, 2 * NIP_D0) =
            D0(2) * m_SD_D0.block<1, NIP_D0>(i, 2 * NIP_D0)
                        .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1).transpose());
        DScaled_Combined_PE.block<1, NIP_D0>((3 * i) + 1, 3 * NIP_D0) =
            D0(3) * (m_SD_D0.block<1, NIP_D0>(i, 2 * NIP_D0)
                         .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(NIP_D0, 1).transpose()) +
                     m_SD_D0.block<1, NIP_D0>(i, 1 * NIP_D0)
                         .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1).transpose()));
        DScaled_Combined_PE.block<1, NIP_D0>((3 * i) + 1, 4 * NIP_D0) =
            D0(4) * (m_SD_D0.block<1, NIP_D0>(i, 2 * NIP_D0)
                         .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(0, 1).transpose()) +
                     m_SD_D0.block<1, NIP_D0>(i, 0 * NIP_D0)
                         .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1).transpose()));
        DScaled_Combined_PE.block<1, NIP_D0>((3 * i) + 1, 5 * NIP_D0) =
            D0(5) * (m_SD_D0.block<1, NIP_D0>(i, 1 * NIP_D0)
                         .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(0, 1).transpose()) +
                     m_SD_D0.block<1, NIP_D0>(i, 0 * NIP_D0)
                         .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(NIP_D0, 1).transpose()));

        DScaled_Combined_PE.block<1, NIP_D0>((3 * i) + 2, 0) =
            D0(0) * m_SD_D0.block<1, NIP_D0>(i, 0 * NIP_D0)
                        .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(0, 2).transpose());
        DScaled_Combined_PE.block<1, NIP_D0>((3 * i) + 2, NIP_D0) =
            D0(1) * m_SD_D0.block<1, NIP_D0>(i, 1 * NIP_D0)
                        .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(NIP_D0, 2).transpose());
        DScaled_Combined_PE.block<1, NIP_D0>((3 * i) + 2, 2 * NIP_D0) =
            D0(2) * m_SD_D0.block<1, NIP_D0>(i, 2 * NIP_D0)
                        .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2).transpose());
        DScaled_Combined_PE.block<1, NIP_D0>((3 * i) + 2, 3 * NIP_D0) =
            D0(3) * (m_SD_D0.block<1, NIP_D0>(i, 2 * NIP_D0)
                         .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(NIP_D0, 2).transpose()) +
                     m_SD_D0.block<1, NIP_D0>(i, 1 * NIP_D0)
                         .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2).transpose()));
        DScaled_Combined_PE.block<1, NIP_D0>((3 * i) + 2, 4 * NIP_D0) =
            D0(4) * (m_SD_D0.block<1, NIP_D0>(i, 2 * NIP_D0)
                         .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(0, 2).transpose()) +
                     m_SD_D0.block<1, NIP_D0>(i, 0 * NIP_D0)
                         .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2).transpose()));
        DScaled_Combined_PE.block<1, NIP_D0>((3 * i) + 2, 5 * NIP_D0) =
            D0(5) * (m_SD_D0.block<1, NIP_D0>(i, 1 * NIP_D0)
                         .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(0, 2).transpose()) +
                     m_SD_D0.block<1, NIP_D0>(i, 0 * NIP_D0)
                         .cwiseProduct(FCscaled_D0.template block<NIP_D0, 1>(NIP_D0, 2).transpose()));
    }

    // =============================================================================
    // Multiply the partial derivative block matrix by the final scaled and combined partial derivative block matrix
    // to obtain the potentially non-symmetric and non-sparse component of the Jacobian matrix
    // =============================================================================

    H = PE * DScaled_Combined_PE.transpose();

    //==============================================================================
    //==============================================================================
    // Calculate the sparse and symmetric component of the Jacobian matrix
    //==============================================================================
    //==============================================================================

    // =============================================================================
    // Calculate each individual value of the Green-Lagrange strain component by component across all the
    // Gauss-Quadrature points at a time for the current layer to better leverage vectorized CPU instructions.
    // Note that the scaled time derivatives of the Green-Lagrange strain are added to make the later calculation of
    // the 2nd Piola-Kirchoff stresses more efficient.  The combined result is then scaled by minus the Gauss
    // quadrature weight times the element Jacobian at the corresponding Gauss point (m_kGQ) again for efficiency.
    // Since only the diagonal terms in the 6x6 stiffness matrix are used, the 2nd Piola-Kirchoff can be calculated by
    // simply scaling the vector of scaled Green-Lagrange strains in Voight notation by the corresponding diagonal entry
    // in the stiffness matrix.
    // Results are written in Voigt notation: epsilon = [E11,E22,E33,2*E23,2*E13,2*E12]
    //  kGQ*SPK2 = kGQ*[SPK2_11,SPK2_22,SPK2_33,SPK2_23,SPK2_13,SPK2_12] = D * E_Combined
    // =============================================================================

    // Each entry in SPK2_1 = D11*kGQ*(E11+alpha*E11dot)
    //     = D11*kGQ*(1/2*(F11*F11+F21*F21+F31*F31-1)+alpha*(F11*F11dot+F21*F21dot+F31*F31dot))
    VectorNIP_D0 E_BlockDamping_D0 =
        FC_D0.template block<NIP_D0, 1>(0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 3)) +
        FC_D0.template block<NIP_D0, 1>(0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 4)) +
        FC_D0.template block<NIP_D0, 1>(0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 5));
    VectorNIP_D0 SPK2_1_Block_D0 =
        FC_D0.template block<NIP_D0, 1>(0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 0)) +
        FC_D0.template block<NIP_D0, 1>(0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 1)) +
        FC_D0.template block<NIP_D0, 1>(0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 2));
    SPK2_1_Block_D0.array() -= 1;
    SPK2_1_Block_D0 += (2 * m_Alpha) * E_BlockDamping_D0;
    SPK2_1_Block_D0.array() *= m_kGQ_D0.array();
    SPK2_1_Block_D0 *= (0.5 * D0(0));

    // Each entry in SPK2_2 = D22*kGQ*(E22+alpha*E22dot)
    //     = D22*kGQ*(1/2*(F12*F12+F22*F22+F32*F32-1)+alpha*(F12*F12dot+F22*F22dot+F32*F32dot))
    E_BlockDamping_D0.noalias() =
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 3)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 4)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 5));
    VectorNIP_D0 SPK2_2_Block_D0 =
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 0)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 1)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 2));
    SPK2_2_Block_D0.array() -= 1;
    SPK2_2_Block_D0 += (2 * m_Alpha) * E_BlockDamping_D0;
    SPK2_2_Block_D0.array() *= m_kGQ_D0.array();
    SPK2_2_Block_D0 *= (0.5 * D0(1));

    // Each entry in SPK2_3 = D33*kGQ*(E33+alpha*E33dot)
    //     = D33*kGQ*(1/2*(F13*F13+F23*F23+F33*F33-1)+alpha*(F13*F13dot+F23*F23dot+F33*F33dot))
    E_BlockDamping_D0.noalias() =
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 3)) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 4)) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 5));
    VectorNIP_D0 SPK2_3_Block_D0 =
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0)) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1)) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2));
    SPK2_3_Block_D0.array() -= 1;
    SPK2_3_Block_D0 += (2 * m_Alpha) * E_BlockDamping_D0;
    SPK2_3_Block_D0.array() *= m_kGQ_D0.array();
    SPK2_3_Block_D0 *= (0.5 * D0(2));

    // Each entry in SPK2_4 = D44*kGQ*(2*(E23+alpha*E23dot))
    //     = D44*kGQ*((F12*F13+F22*F23+F32*F33)
    //       +alpha*(F12dot*F13+F22dot*F23+F32dot*F33 + F12*F13dot+F22*F23dot+F32*F33dot))
    E_BlockDamping_D0.noalias() =
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 3)) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 4)) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 5)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 3)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 4)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 5));
    VectorNIP_D0 SPK2_4_Block_D0 =
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2));
    SPK2_4_Block_D0 += m_Alpha * E_BlockDamping_D0;
    SPK2_4_Block_D0.array() *= m_kGQ_D0.array();
    SPK2_4_Block_D0 *= D0(3);

    // Each entry in SPK2_5 = D55*kGQ*(2*(E13+alpha*E13dot))
    //     = D55*kGQ*((F11*F13+F21*F23+F31*F33)
    //       +alpha*(F11dot*F13+F21dot*F23+F31dot*F33 + F11*F13dot+F21*F23dot+F31*F33dot))
    E_BlockDamping_D0.noalias() =
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 3)) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 4)) +
        FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 5)) +
        FC_D0.template block<NIP_D0, 1>(0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 3)) +
        FC_D0.template block<NIP_D0, 1>(0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 4)) +
        FC_D0.template block<NIP_D0, 1>(0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 5));
    VectorNIP_D0 SPK2_5_Block_D0 =
        FC_D0.template block<NIP_D0, 1>(0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 0)) +
        FC_D0.template block<NIP_D0, 1>(0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 1)) +
        FC_D0.template block<NIP_D0, 1>(0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(2 * NIP_D0, 2));
    SPK2_5_Block_D0 += m_Alpha * E_BlockDamping_D0;
    SPK2_5_Block_D0.array() *= m_kGQ_D0.array();
    SPK2_5_Block_D0 *= D0(4);

    // Each entry in SPK2_6 = D66*kGQ*(2*(E12+alpha*E12dot))
    //     = D66*kGQ*((F11*F12+F21*F22+F31*F32)
    //       +alpha*(F11dot*F12+F21dot*F22+F31dot*F32 + F11*F12dot+F21*F22dot+F31*F32dot))
    E_BlockDamping_D0.noalias() =
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 3)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 4)) +
        FC_D0.template block<NIP_D0, 1>(NIP_D0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(0, 5)) +
        FC_D0.template block<NIP_D0, 1>(0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 3)) +
        FC_D0.template block<NIP_D0, 1>(0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 4)) +
        FC_D0.template block<NIP_D0, 1>(0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 5));
    VectorNIP_D0 SPK2_6_Block_D0 =
        FC_D0.template block<NIP_D0, 1>(0, 0).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 0)) +
        FC_D0.template block<NIP_D0, 1>(0, 1).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 1)) +
        FC_D0.template block<NIP_D0, 1>(0, 2).cwiseProduct(FC_D0.template block<NIP_D0, 1>(NIP_D0, 2));
    SPK2_6_Block_D0 += m_Alpha * E_BlockDamping_D0;
    SPK2_6_Block_D0.array() *= m_kGQ_D0.array();
    SPK2_6_Block_D0 *= D0(5);

    // =============================================================================
    // Multiply the shape function derivative matrix by the 2nd Piola-Kirchoff stresses for each corresponding Gauss
    // quadrature point
    // =============================================================================

    ChMatrixNM<double, NSF, 3 * NIP_D0> S_scaled_SD_D0;

    for (auto i = 0; i < NSF; i++) {
        S_scaled_SD_D0.template block<1, NIP_D0>(i, 0) =
            SPK2_1_Block_D0.transpose().cwiseProduct(m_SD_D0.block<1, NIP_D0>(i, 0 * NIP_D0)) +
            SPK2_6_Block_D0.transpose().cwiseProduct(m_SD_D0.block<1, NIP_D0>(i, 1 * NIP_D0)) +
            SPK2_5_Block_D0.transpose().cwiseProduct(m_SD_D0.block<1, NIP_D0>(i, 2 * NIP_D0));

        S_scaled_SD_D0.template block<1, NIP_D0>(i, NIP_D0) =
            SPK2_6_Block_D0.transpose().cwiseProduct(m_SD_D0.block<1, NIP_D0>(i, 0 * NIP_D0)) +
            SPK2_2_Block_D0.transpose().cwiseProduct(m_SD_D0.block<1, NIP_D0>(i, 1 * NIP_D0)) +
            SPK2_4_Block_D0.transpose().cwiseProduct(m_SD_D0.block<1, NIP_D0>(i, 2 * NIP_D0));

        S_scaled_SD_D0.template block<1, NIP_D0>(i, 2 * NIP_D0) =
            SPK2_5_Block_D0.transpose().cwiseProduct(m_SD_D0.block<1, NIP_D0>(i, 0 * NIP_D0)) +
            SPK2_4_Block_D0.transpose().cwiseProduct(m_SD_D0.block<1, NIP_D0>(i, 1 * NIP_D0)) +
            SPK2_3_Block_D0.transpose().cwiseProduct(m_SD_D0.block<1, NIP_D0>(i, 2 * NIP_D0));
    }

    // =============================================================================
    // Calculate just the non-sparse upper triangular entires of the sparse and symmetric component of the Jacobian
    // matrix, combine this with the scaled mass matrix, and then expand them out to full size by summing the
    // contribution into the correct locations of the full sized Jacobian matrix
    // =============================================================================

    // Add in the Mass Matrix Which is Stored in Compact Upper Triangular Form
    ChVectorN<double, (NSF * (NSF + 1)) / 2> ScaledMassMatrix = Mfactor * m_MassMatrix;

    unsigned int idx = 0;
    for (unsigned int i = 0; i < NSF; i++) {
        for (unsigned int j = i; j < NSF; j++) {
            double d = Kfactor * m_SD_D0.row(i) * S_scaled_SD_D0.row(j).transpose();
            d += ScaledMassMatrix(idx);

            H(3 * i, 3 * j) += d;
            H(3 * i + 1, 3 * j + 1) += d;
            H(3 * i + 2, 3 * j + 2) += d;
            if (i != j) {
                H(3 * j, 3 * i) += d;
                H(3 * j + 1, 3 * i + 1) += d;
                H(3 * j + 2, 3 * i + 2) += d;
            }
            idx++;
        }
    }

    // =============================================================================
    // Calculate the deformation gradient and time derivative of the deformation gradient for all Gauss quadrature
    // points in a single matrix multiplication.  Note that since the shape function derivative matrix is ordered by
    // columns, the resulting deformation gradient will be ordered by block matrix (column vectors) components
    // Note that the indices of the components are in transposed order
    //      [F11  F21  F31  F11dot  F21dot  F31dot ]
    // FC = [F12  F22  F32  F12dot  F22dot  F32dot ]
    //      [F13  F23  F33  F13dot  F23dot  F33dot ]
    // =============================================================================

    ChMatrixNMc<double, 3 * NIP_Dv, 6> FC_Dv = m_SD_Dv.transpose() * ebar_ebardot;

    //==============================================================================
    //==============================================================================
    // Calculate the potentially non-symmetric and non-sparse component of the Jacobian matrix
    //==============================================================================
    //==============================================================================

    // =============================================================================
    // Calculate the partial derivative of the Green-Largrange strains with respect to the nodal coordinates
    // (transposed).  This calculation is performed in blocks across all the Gauss quadrature points at the same
    // time.  This value should be store in row major memory layout to align with the access patterns for
    // calculating this matrix.
    // PE = [(d epsilon1/d e)GQPnt1' (d epsilon1/d e)GQPnt2' ... (d epsilon1/d e)GQPntNIP'...
    //       (d epsilon2/d e)GQPnt1' (d epsilon2/d e)GQPnt2' ... (d epsilon2/d e)GQPntNIP'...
    //       (d epsilon3/d e)GQPnt1' (d epsilon3/d e)GQPnt2' ... (d epsilon3/d e)GQPntNIP'...
    //       (d epsilon4/d e)GQPnt1' (d epsilon4/d e)GQPnt2' ... (d epsilon4/d e)GQPntNIP'...
    //       (d epsilon5/d e)GQPnt1' (d epsilon5/d e)GQPnt2' ... (d epsilon5/d e)GQPntNIP'...
    //       (d epsilon6/d e)GQPnt1' (d epsilon6/d e)GQPnt2' ... (d epsilon6/d e)GQPntNIP']
    // Note that each partial derivative block shown is placed to the left of the previous block.
    // The explanation of the calculation above is just too long to write it all on a single line.
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> PE_Dv;
    PE_Dv.resize(3 * NSF, 3 * NIP_Dv);

    for (auto i = 0; i < NSF; i++) {
        PE_Dv.block<1, NIP_Dv>(3 * i, 0 * NIP_Dv) =
            m_SD_Dv.block<1, NIP_Dv>(i, 0 * NIP_Dv)
                .cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(0 * NIP_Dv, 0).transpose());
        PE_Dv.block<1, NIP_Dv>(3 * i, 1 * NIP_Dv) =
            m_SD_Dv.block<1, NIP_Dv>(i, 1 * NIP_Dv)
                .cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(1 * NIP_Dv, 0).transpose());
        PE_Dv.block<1, NIP_Dv>(3 * i, 2 * NIP_Dv) =
            m_SD_Dv.block<1, NIP_Dv>(i, 2 * NIP_Dv)
                .cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 0).transpose());

        PE_Dv.block<1, NIP_Dv>((3 * i) + 1, 0) =
            m_SD_Dv.block<1, NIP_Dv>(i, 0 * NIP_Dv).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(0, 1).transpose());
        PE_Dv.block<1, NIP_Dv>((3 * i) + 1, NIP_Dv) =
            m_SD_Dv.block<1, NIP_Dv>(i, 1 * NIP_Dv)
                .cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 1).transpose());
        PE_Dv.block<1, NIP_Dv>((3 * i) + 1, 2 * NIP_Dv) =
            m_SD_Dv.block<1, NIP_Dv>(i, 2 * NIP_Dv)
                .cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 1).transpose());

        PE_Dv.block<1, NIP_Dv>((3 * i) + 2, 0) =
            m_SD_Dv.block<1, NIP_Dv>(i, 0 * NIP_Dv).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(0, 2).transpose());
        PE_Dv.block<1, NIP_Dv>((3 * i) + 2, NIP_Dv) =
            m_SD_Dv.block<1, NIP_Dv>(i, 1 * NIP_Dv)
                .cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 2).transpose());
        PE_Dv.block<1, NIP_Dv>((3 * i) + 2, 2 * NIP_Dv) =
            m_SD_Dv.block<1, NIP_Dv>(i, 2 * NIP_Dv)
                .cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 2).transpose());
    }

    // =============================================================================
    // Combine the deformation gradient, time derivative of the deformation gradient, damping coefficient, Gauss
    // quadrature weighting times the element Jacobian, and Jacobian component scale factors into a scaled block
    // deformation gradient matrix Calculate the deformation gradient and time derivative of the deformation
    // gradient for all Gauss quadrature points in a single matrix multiplication.  Note that the resulting combined
    // deformation gradient block matrix will be ordered by block matrix (column vectors) components Note that the
    // indices of the components are in transposed order
    //            [kGQ*(Kfactor+alpha*Rfactor)*F11+alpha*Rfactor*F11dot ... similar for F21 & F31 blocks]
    // FCscaled = [kGQ*(Kfactor+alpha*Rfactor)*F12+alpha*Rfactor*F12dot ... similar for F22 & F32 blocks]
    //            [kGQ*(Kfactor+alpha*Rfactor)*F13+alpha*Rfactor*F13dot ... similar for F23 & F33 blocks]
    // =============================================================================

    ChMatrixNMc<double, 3 * NIP_Dv, 3> FCscaled_Dv =
        (Kfactor + m_Alpha * Rfactor) * FC_Dv.template block<3 * NIP_Dv, 3>(0, 0) +
        (m_Alpha * Kfactor) * FC_Dv.template block<3 * NIP_Dv, 3>(0, 3);

    for (auto i = 0; i < 3; i++) {
        FCscaled_Dv.template block<NIP_Dv, 1>(0, i).array() *= m_kGQ_Dv.array();
        FCscaled_Dv.template block<NIP_Dv, 1>(NIP_Dv, i).array() *= m_kGQ_Dv.array();
        FCscaled_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, i).array() *= m_kGQ_Dv.array();
    }

    // =============================================================================
    // Calculate the combination of the scaled partial derivative of the Green-Largrange strains with respect to the
    // nodal coordinates, the scaled partial derivative of the time derivative of the Green-Largrange strains with
    // respect to the nodal coordinates, the scaled partial derivative of the Green-Largrange strains with respect
    // to the time derivative of the nodal coordinates, and the other parameters to correctly integrate across the
    // volume of the element.  This calculation is performed in blocks across all the Gauss quadrature points at the
    // same time.  This value should be store in row major memory layout to align with the access patterns for
    // calculating this matrix.
    // =============================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Scaled_Combined_PE_Dv;
    Scaled_Combined_PE_Dv.resize(3 * NSF, 3 * NIP_Dv);

    for (auto i = 0; i < NSF; i++) {
        Scaled_Combined_PE_Dv.block<1, NIP_Dv>(3 * i, 0) =
            m_SD_Dv.block<1, NIP_Dv>(i, 0 * NIP_Dv)
                .cwiseProduct(FCscaled_Dv.template block<NIP_Dv, 1>(0, 0).transpose());
        Scaled_Combined_PE_Dv.block<1, NIP_Dv>(3 * i, NIP_Dv) =
            m_SD_Dv.block<1, NIP_Dv>(i, 1 * NIP_Dv)
                .cwiseProduct(FCscaled_Dv.template block<NIP_Dv, 1>(NIP_Dv, 0).transpose());
        Scaled_Combined_PE_Dv.block<1, NIP_Dv>(3 * i, 2 * NIP_Dv) =
            m_SD_Dv.block<1, NIP_Dv>(i, 2 * NIP_Dv)
                .cwiseProduct(FCscaled_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 0).transpose());

        Scaled_Combined_PE_Dv.block<1, NIP_Dv>((3 * i) + 1, 0) =
            m_SD_Dv.block<1, NIP_Dv>(i, 0 * NIP_Dv)
                .cwiseProduct(FCscaled_Dv.template block<NIP_Dv, 1>(0, 1).transpose());
        Scaled_Combined_PE_Dv.block<1, NIP_Dv>((3 * i) + 1, NIP_Dv) =
            m_SD_Dv.block<1, NIP_Dv>(i, 1 * NIP_Dv)
                .cwiseProduct(FCscaled_Dv.template block<NIP_Dv, 1>(NIP_Dv, 1).transpose());
        Scaled_Combined_PE_Dv.block<1, NIP_Dv>((3 * i) + 1, 2 * NIP_Dv) =
            m_SD_Dv.block<1, NIP_Dv>(i, 2 * NIP_Dv)
                .cwiseProduct(FCscaled_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 1).transpose());

        Scaled_Combined_PE_Dv.block<1, NIP_Dv>((3 * i) + 2, 0) =
            m_SD_Dv.block<1, NIP_Dv>(i, 0 * NIP_Dv)
                .cwiseProduct(FCscaled_Dv.template block<NIP_Dv, 1>(0, 2).transpose());
        Scaled_Combined_PE_Dv.block<1, NIP_Dv>((3 * i) + 2, NIP_Dv) =
            m_SD_Dv.block<1, NIP_Dv>(i, 1 * NIP_Dv)
                .cwiseProduct(FCscaled_Dv.template block<NIP_Dv, 1>(NIP_Dv, 2).transpose());
        Scaled_Combined_PE_Dv.block<1, NIP_Dv>((3 * i) + 2, 2 * NIP_Dv) =
            m_SD_Dv.block<1, NIP_Dv>(i, 2 * NIP_Dv)
                .cwiseProduct(FCscaled_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 2).transpose());
    }

    // =============================================================================
    // Get the upper 3x3 block of the stiffness tensor in 6x6 matrix that accounts for the Poisson effect.  Note that
    // with the split of the stiffness matrix and the assumed materials, the entries in the 6x6 stiffness matrix outside
    // of the upper 3x3 block are all zeros.
    // =============================================================================

    const ChMatrix33<double>& Dv = GetMaterial()->Get_Dv();

    // =============================================================================
    // Multiply the scaled and combined partial derivative block matrix by the stiffness matrix for each individual
    // Gauss quadrature point
    // =============================================================================
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DScaled_Combined_PE_Dv;
    DScaled_Combined_PE_Dv.resize(3 * NSF, 3 * NIP_Dv);

    DScaled_Combined_PE_Dv.template block<3 * NSF, NIP_Dv>(0, 0) =
        Dv(0, 0) * Scaled_Combined_PE_Dv.block<3 * NSF, NIP_Dv>(0, 0) +
        Dv(0, 1) * Scaled_Combined_PE_Dv.block<3 * NSF, NIP_Dv>(0, NIP_Dv) +
        Dv(0, 2) * Scaled_Combined_PE_Dv.block<3 * NSF, NIP_Dv>(0, 2 * NIP_Dv);
    DScaled_Combined_PE_Dv.template block<3 * NSF, NIP_Dv>(0, NIP_Dv) =
        Dv(1, 0) * Scaled_Combined_PE_Dv.block<3 * NSF, NIP_Dv>(0, 0) +
        Dv(1, 1) * Scaled_Combined_PE_Dv.block<3 * NSF, NIP_Dv>(0, NIP_Dv) +
        Dv(1, 2) * Scaled_Combined_PE_Dv.block<3 * NSF, NIP_Dv>(0, 2 * NIP_Dv);
    DScaled_Combined_PE_Dv.template block<3 * NSF, NIP_Dv>(0, 2 * NIP_Dv) =
        Dv(2, 0) * Scaled_Combined_PE_Dv.block<3 * NSF, NIP_Dv>(0, 0) +
        Dv(2, 1) * Scaled_Combined_PE_Dv.block<3 * NSF, NIP_Dv>(0, NIP_Dv) +
        Dv(2, 2) * Scaled_Combined_PE_Dv.block<3 * NSF, NIP_Dv>(0, 2 * NIP_Dv);

    // =============================================================================
    // Multiply the partial derivative block matrix by the final scaled and combined partial derivative block matrix
    // to obtain the potentially non-symmetric and non-sparse component of the Jacobian matrix
    // =============================================================================

    H += PE_Dv * DScaled_Combined_PE_Dv.transpose();

    //==============================================================================
    //==============================================================================
    // Calculate the sparse and symmetric component of the Jacobian matrix
    //==============================================================================
    //==============================================================================

    // =============================================================================
    // Calculate each individual value of the Green-Lagrange strain component by component across all the
    // Gauss-Quadrature points at a time for the current layer to better leverage vectorized CPU instructions.
    // Note that the scaled time derivatives of the Green-Lagrange strain are added to make the later calculation of
    // the 2nd Piola-Kirchoff stresses more efficient.  The combined result is then scaled by minus the Gauss
    // quadrature weight times the element Jacobian at the corresponding Gauss point (m_kGQ) again for efficiency.
    // Results are written in Voigt notation: epsilon = [E11,E22,E33,2*E23,2*E13,2*E12]
    // Note that with the material assumption being used, only [E11,E22,E33] need to be calculated
    // =============================================================================

    // Each entry in E1 = kGQ*(E11+alpha*E11dot)
    //                  = kGQ*(1/2*(F11*F11+F21*F21+F31*F31-1)+alpha*(F11*F11dot+F21*F21dot+F31*F31dot))
    VectorNIP_Dv E_BlockDamping_Dv =
        FC_Dv.template block<NIP_Dv, 1>(0, 0).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(0, 3)) +
        FC_Dv.template block<NIP_Dv, 1>(0, 1).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(0, 4)) +
        FC_Dv.template block<NIP_Dv, 1>(0, 2).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(0, 5));
    VectorNIP_Dv E1_Block_Dv =
        FC_Dv.template block<NIP_Dv, 1>(0, 0).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(0, 0)) +
        FC_Dv.template block<NIP_Dv, 1>(0, 1).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(0, 1)) +
        FC_Dv.template block<NIP_Dv, 1>(0, 2).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(0, 2));
    E1_Block_Dv.array() -= 1;
    E1_Block_Dv *= 0.5;
    E1_Block_Dv += m_Alpha * E_BlockDamping_Dv;
    E1_Block_Dv.array() *= m_kGQ_Dv.array();

    // Each entry in E2 = kGQ*(E22+alpha*E22dot)
    //                  = kGQ*(1/2*(F12*F12+F22*F22+F32*F32-1)+alpha*(F12*F12dot+F22*F22dot+F32*F32dot))
    E_BlockDamping_Dv.noalias() =
        FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 0).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 3)) +
        FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 1).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 4)) +
        FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 2).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 5));
    VectorNIP_Dv E2_Block_Dv =
        FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 0).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 0)) +
        FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 1).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 1)) +
        FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 2).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(NIP_Dv, 2));
    E2_Block_Dv.array() -= 1;
    E2_Block_Dv *= 0.5;
    E2_Block_Dv += m_Alpha * E_BlockDamping_Dv;
    E2_Block_Dv.array() *= m_kGQ_Dv.array();

    // Each entry in E3 = kGQ*(E33+alpha*E33dot)
    //                  = kGQ*(1/2*(F13*F13+F23*F23+F33*F33-1)+alpha*(F13*F13dot+F23*F23dot+F33*F33dot))
    E_BlockDamping_Dv.noalias() =
        FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 0).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 3)) +
        FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 1).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 4)) +
        FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 2).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 5));
    VectorNIP_Dv E3_Block_Dv =
        FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 0).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 0)) +
        FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 1).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 1)) +
        FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 2).cwiseProduct(FC_Dv.template block<NIP_Dv, 1>(2 * NIP_Dv, 2));
    E3_Block_Dv.array() -= 1;
    E3_Block_Dv *= 0.5;
    E3_Block_Dv += m_Alpha * E_BlockDamping_Dv;
    E3_Block_Dv.array() *= m_kGQ_Dv.array();

    // =============================================================================
    // Calculate the 2nd Piola-Kirchoff stresses in Voigt notation across all the Gauss quadrature points in the
    // current layer at a time component by component.
    // Note that the Green-Largrange strain components have been scaled have already been combined with their scaled
    // time derivatives and minus the Gauss quadrature weight times the element Jacobian at the corresponding Gauss
    // point to make the calculation of the 2nd Piola-Kirchoff stresses more efficient.
    //  kGQ*SPK2 = kGQ*[SPK2_11,SPK2_22,SPK2_33,SPK2_23,SPK2_13,SPK2_12] = D * E_Combined
    // =============================================================================

    VectorNIP_Dv SPK2_1_Block_Dv = Dv(0, 0) * E1_Block_Dv + Dv(0, 1) * E2_Block_Dv + Dv(0, 2) * E3_Block_Dv;
    VectorNIP_Dv SPK2_2_Block_Dv = Dv(1, 0) * E1_Block_Dv + Dv(1, 1) * E2_Block_Dv + Dv(1, 2) * E3_Block_Dv;
    VectorNIP_Dv SPK2_3_Block_Dv = Dv(2, 0) * E1_Block_Dv + Dv(2, 1) * E2_Block_Dv + Dv(2, 2) * E3_Block_Dv;

    // =============================================================================
    // Multiply the shape function derivative matrix by the 2nd Piola-Kirchoff stresses for each corresponding Gauss
    // quadrature point
    // =============================================================================

    ChMatrixNM<double, NSF, 3 * NIP_Dv> S_scaled_SD_Dv;

    for (auto i = 0; i < NSF; i++) {
        S_scaled_SD_Dv.template block<1, NIP_Dv>(i, 0) =
            SPK2_1_Block_Dv.transpose().cwiseProduct(m_SD_Dv.block<1, NIP_Dv>(i, 0 * NIP_Dv));

        S_scaled_SD_Dv.template block<1, NIP_Dv>(i, NIP_Dv) =
            SPK2_2_Block_Dv.transpose().cwiseProduct(m_SD_Dv.block<1, NIP_Dv>(i, 1 * NIP_Dv));

        S_scaled_SD_Dv.template block<1, NIP_Dv>(i, 2 * NIP_Dv) =
            SPK2_3_Block_Dv.transpose().cwiseProduct(m_SD_Dv.block<1, NIP_Dv>(i, 2 * NIP_Dv));
    }

    // =============================================================================
    // Calculate just the non-sparse upper triangular entires of the sparse and symmetric component of the Jacobian
    // matrix, combine this with the scaled mass matrix, and then expand them out to full size by summing the
    // contribution into the correct locations of the full sized Jacobian matrix
    // =============================================================================

    idx = 0;
    for (unsigned int i = 0; i < NSF; i++) {
        for (unsigned int j = i; j < NSF; j++) {
            double d = Kfactor * m_SD_Dv.row(i) * S_scaled_SD_Dv.row(j).transpose();

            H(3 * i, 3 * j) += d;
            H(3 * i + 1, 3 * j + 1) += d;
            H(3 * i + 2, 3 * j + 2) += d;
            if (i != j) {
                H(3 * j, 3 * i) += d;
                H(3 * j + 1, 3 * i + 1) += d;
                H(3 * j + 2, 3 * i + 2) += d;
            }
            idx++;
        }
    }
}

template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::ComputeInternalJacobianContIntNoDamping(ChMatrixRef& H,
                                                                             double Kfactor,
                                                                             double Mfactor) {
    ComputeInternalJacobianContIntDamping(H, Kfactor, 0, Mfactor);

    //// Calculate the Jacobian of the generalize internal force vector using the "Continuous Integration" style of
    ///method / assuming a linear material model (no damping).  For this style of method, the Jacobian of the
    ///generalized / internal force vector is integrated across the volume of the element every time this calculation is
    ///performed. / For this element, this is likely more efficient than the "Pre-Integration" style calculation method.

    // Matrix3xN e_bar;
    // CalcCoordMatrix(e_bar);

    //// No values from the generalized internal force vector are cached for reuse in the Jacobian.  Instead these
    //// quantities are recalculated again during the Jacobian calculations.  This both simplifies the code and speeds
    //// up the generalized internal force calculation while only having a minimal impact on the performance of the
    //// Jacobian calculation speed.  The Jacobian calculation is performed in two major pieces.  First is the
    //// calculation of the potentially non-symmetric and non-sparse component.  The second pieces is the symmetric
    //// and sparse component.

    //// =============================================================================
    //// Calculate the deformation gradient for all Gauss quadrature points in a single matrix multiplication.  Note
    //// that since the shape function derivative matrix is ordered by columns, the resulting deformation gradient
    //// will be ordered by block matrix (column vectors) components
    //// Note that the indices of the components are in transposed order
    ////      [F11  F21  F31 ]
    //// FC = [F12  F22  F32 ]
    ////      [F13  F23  F33 ]
    //// =============================================================================
    // ChMatrixNMc<double, 3 * NIP, 3> FC = m_SD.transpose() * e_bar.transpose();

    ////==============================================================================
    ////==============================================================================
    //// Calculate the potentially non-symmetric and non-sparse component of the Jacobian matrix
    ////==============================================================================
    ////==============================================================================

    //// =============================================================================
    //// Calculate the partial derivative of the Green-Largrange strains with respect to the nodal coordinates
    //// (transposed).  This calculation is performed in blocks across all the Gauss quadrature points at the same
    //// time.  This value should be store in row major memory layout to align with the access patterns for
    //// calculating this matrix.
    //// PE = [(d epsilon1/d e)GQPnt1' (d epsilon1/d e)GQPnt2' ... (d epsilon1/d e)GQPntNIP'...
    ////       (d epsilon2/d e)GQPnt1' (d epsilon2/d e)GQPnt2' ... (d epsilon2/d e)GQPntNIP'...
    ////       (d epsilon3/d e)GQPnt1' (d epsilon3/d e)GQPnt2' ... (d epsilon3/d e)GQPntNIP'...
    ////       (d epsilon4/d e)GQPnt1' (d epsilon4/d e)GQPnt2' ... (d epsilon4/d e)GQPntNIP'...
    ////       (d epsilon5/d e)GQPnt1' (d epsilon5/d e)GQPnt2' ... (d epsilon5/d e)GQPntNIP'...
    ////       (d epsilon6/d e)GQPnt1' (d epsilon6/d e)GQPnt2' ... (d epsilon6/d e)GQPntNIP']
    //// Note that each partial derivative block shown is placed to the left of the previous block.
    //// The explanation of the calculation above is just too long to write it all on a single line.
    //// =============================================================================
    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> PE;
    // PE.resize(3 * NSF, 6 * NIP);

    // for (auto i = 0; i < NSF; i++) {
    //    PE.block<1, NIP>(3 * i, 0) =
    //        m_SD.block<1, NIP>(i, 0 * NIP).cwiseProduct(FC.template block<NIP, 1>(0, 0).transpose());
    //    PE.block<1, NIP>(3 * i, NIP) =
    //        m_SD.block<1, NIP>(i, 1 * NIP).cwiseProduct(FC.template block<NIP, 1>(NIP, 0).transpose());
    //    PE.block<1, NIP>(3 * i, 2 * NIP) =
    //        m_SD.block<1, NIP>(i, 2 * NIP).cwiseProduct(FC.template block<NIP, 1>(2 * NIP, 0).transpose());
    //    PE.block<1, NIP>(3 * i, 3 * NIP) =
    //        m_SD.block<1, NIP>(i, 2 * NIP).cwiseProduct(FC.template block<NIP, 1>(NIP, 0).transpose()) +
    //        m_SD.block<1, NIP>(i, 1 * NIP).cwiseProduct(FC.template block<NIP, 1>(2 * NIP, 0).transpose());
    //    PE.block<1, NIP>(3 * i, 4 * NIP) =
    //        m_SD.block<1, NIP>(i, 2 * NIP).cwiseProduct(FC.template block<NIP, 1>(0, 0).transpose()) +
    //        m_SD.block<1, NIP>(i, 0 * NIP).cwiseProduct(FC.template block<NIP, 1>(2 * NIP, 0).transpose());
    //    PE.block<1, NIP>(3 * i, 5 * NIP) =
    //        m_SD.block<1, NIP>(i, 1 * NIP).cwiseProduct(FC.template block<NIP, 1>(0, 0).transpose()) +
    //        m_SD.block<1, NIP>(i, 0 * NIP).cwiseProduct(FC.template block<NIP, 1>(NIP, 0).transpose());

    //    PE.block<1, NIP>((3 * i) + 1, 0) =
    //        m_SD.block<1, NIP>(i, 0 * NIP).cwiseProduct(FC.template block<NIP, 1>(0, 1).transpose());
    //    PE.block<1, NIP>((3 * i) + 1, NIP) =
    //        m_SD.block<1, NIP>(i, 1 * NIP).cwiseProduct(FC.template block<NIP, 1>(NIP, 1).transpose());
    //    PE.block<1, NIP>((3 * i) + 1, 2 * NIP) =
    //        m_SD.block<1, NIP>(i, 2 * NIP).cwiseProduct(FC.template block<NIP, 1>(2 * NIP, 1).transpose());
    //    PE.block<1, NIP>((3 * i) + 1, 3 * NIP) =
    //        m_SD.block<1, NIP>(i, 2 * NIP).cwiseProduct(FC.template block<NIP, 1>(NIP, 1).transpose()) +
    //        m_SD.block<1, NIP>(i, 1 * NIP).cwiseProduct(FC.template block<NIP, 1>(2 * NIP, 1).transpose());
    //    PE.block<1, NIP>((3 * i) + 1, 4 * NIP) =
    //        m_SD.block<1, NIP>(i, 2 * NIP).cwiseProduct(FC.template block<NIP, 1>(0, 1).transpose()) +
    //        m_SD.block<1, NIP>(i, 0 * NIP).cwiseProduct(FC.template block<NIP, 1>(2 * NIP, 1).transpose());
    //    PE.block<1, NIP>((3 * i) + 1, 5 * NIP) =
    //        m_SD.block<1, NIP>(i, 1 * NIP).cwiseProduct(FC.template block<NIP, 1>(0, 1).transpose()) +
    //        m_SD.block<1, NIP>(i, 0 * NIP).cwiseProduct(FC.template block<NIP, 1>(NIP, 1).transpose());

    //    PE.block<1, NIP>((3 * i) + 2, 0) =
    //        m_SD.block<1, NIP>(i, 0 * NIP).cwiseProduct(FC.template block<NIP, 1>(0, 2).transpose());
    //    PE.block<1, NIP>((3 * i) + 2, NIP) =
    //        m_SD.block<1, NIP>(i, 1 * NIP).cwiseProduct(FC.template block<NIP, 1>(NIP, 2).transpose());
    //    PE.block<1, NIP>((3 * i) + 2, 2 * NIP) =
    //        m_SD.block<1, NIP>(i, 2 * NIP).cwiseProduct(FC.template block<NIP, 1>(2 * NIP, 2).transpose());
    //    PE.block<1, NIP>((3 * i) + 2, 3 * NIP) =
    //        m_SD.block<1, NIP>(i, 2 * NIP).cwiseProduct(FC.template block<NIP, 1>(NIP, 2).transpose()) +
    //        m_SD.block<1, NIP>(i, 1 * NIP).cwiseProduct(FC.template block<NIP, 1>(2 * NIP, 2).transpose());
    //    PE.block<1, NIP>((3 * i) + 2, 4 * NIP) =
    //        m_SD.block<1, NIP>(i, 2 * NIP).cwiseProduct(FC.template block<NIP, 1>(0, 2).transpose()) +
    //        m_SD.block<1, NIP>(i, 0 * NIP).cwiseProduct(FC.template block<NIP, 1>(2 * NIP, 2).transpose());
    //    PE.block<1, NIP>((3 * i) + 2, 5 * NIP) =
    //        m_SD.block<1, NIP>(i, 1 * NIP).cwiseProduct(FC.template block<NIP, 1>(0, 2).transpose()) +
    //        m_SD.block<1, NIP>(i, 0 * NIP).cwiseProduct(FC.template block<NIP, 1>(NIP, 2).transpose());
    //}

    //// =============================================================================
    //// Scale the block deformation gradient matrix by the Kfactor weighting factor for the Jacobian and multiply
    //// each Gauss quadrature component by its Gauss quadrature weight times the element Jacobian (kGQ)
    //// =============================================================================
    // ChMatrixNMc<double, 3 * NIP, 3> FCscaled = Kfactor * FC;

    // for (auto i = 0; i < 3; i++) {
    //    FCscaled.template block<NIP, 1>(0, i).array() *= m_kGQ.array();
    //    FCscaled.template block<NIP, 1>(NIP, i).array() *= m_kGQ.array();
    //    FCscaled.template block<NIP, 1>(2 * NIP, i).array() *= m_kGQ.array();
    //}

    //// =============================================================================
    //// Calculate the scaled partial derivative of the Green-Largrange strains with respect to the nodal coordinates
    //// and the other parameters to correctly integrate across the volume of the element.  This calculation is
    //// performed in blocks across all the Gauss quadrature points at the same time.  This value should be store in
    //// row major memory layout to align with the access patterns for calculating this matrix.
    //// =============================================================================

    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Scaled_Combined_PE;
    // Scaled_Combined_PE.resize(3 * NSF, 6 * NIP);

    // for (auto i = 0; i < NSF; i++) {
    //    Scaled_Combined_PE.block<1, NIP>(3 * i, 0) =
    //        m_SD.block<1, NIP>(i, 0 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(0, 0).transpose());
    //    Scaled_Combined_PE.block<1, NIP>(3 * i, NIP) =
    //        m_SD.block<1, NIP>(i, 1 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(NIP, 0).transpose());
    //    Scaled_Combined_PE.block<1, NIP>(3 * i, 2 * NIP) =
    //        m_SD.block<1, NIP>(i, 2 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(2 * NIP, 0).transpose());
    //    Scaled_Combined_PE.block<1, NIP>(3 * i, 3 * NIP) =
    //        m_SD.block<1, NIP>(i, 2 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(NIP, 0).transpose()) +
    //        m_SD.block<1, NIP>(i, 1 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(2 * NIP, 0).transpose());
    //    Scaled_Combined_PE.block<1, NIP>(3 * i, 4 * NIP) =
    //        m_SD.block<1, NIP>(i, 2 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(0, 0).transpose()) +
    //        m_SD.block<1, NIP>(i, 0 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(2 * NIP, 0).transpose());
    //    Scaled_Combined_PE.block<1, NIP>(3 * i, 5 * NIP) =
    //        m_SD.block<1, NIP>(i, 1 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(0, 0).transpose()) +
    //        m_SD.block<1, NIP>(i, 0 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(NIP, 0).transpose());

    //    Scaled_Combined_PE.block<1, NIP>((3 * i) + 1, 0) =
    //        m_SD.block<1, NIP>(i, 0 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(0, 1).transpose());
    //    Scaled_Combined_PE.block<1, NIP>((3 * i) + 1, NIP) =
    //        m_SD.block<1, NIP>(i, 1 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(NIP, 1).transpose());
    //    Scaled_Combined_PE.block<1, NIP>((3 * i) + 1, 2 * NIP) =
    //        m_SD.block<1, NIP>(i, 2 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(2 * NIP, 1).transpose());
    //    Scaled_Combined_PE.block<1, NIP>((3 * i) + 1, 3 * NIP) =
    //        m_SD.block<1, NIP>(i, 2 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(NIP, 1).transpose()) +
    //        m_SD.block<1, NIP>(i, 1 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(2 * NIP, 1).transpose());
    //    Scaled_Combined_PE.block<1, NIP>((3 * i) + 1, 4 * NIP) =
    //        m_SD.block<1, NIP>(i, 2 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(0, 1).transpose()) +
    //        m_SD.block<1, NIP>(i, 0 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(2 * NIP, 1).transpose());
    //    Scaled_Combined_PE.block<1, NIP>((3 * i) + 1, 5 * NIP) =
    //        m_SD.block<1, NIP>(i, 1 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(0, 1).transpose()) +
    //        m_SD.block<1, NIP>(i, 0 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(NIP, 1).transpose());

    //    Scaled_Combined_PE.block<1, NIP>((3 * i) + 2, 0) =
    //        m_SD.block<1, NIP>(i, 0 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(0, 2).transpose());
    //    Scaled_Combined_PE.block<1, NIP>((3 * i) + 2, NIP) =
    //        m_SD.block<1, NIP>(i, 1 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(NIP, 2).transpose());
    //    Scaled_Combined_PE.block<1, NIP>((3 * i) + 2, 2 * NIP) =
    //        m_SD.block<1, NIP>(i, 2 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(2 * NIP, 2).transpose());
    //    Scaled_Combined_PE.block<1, NIP>((3 * i) + 2, 3 * NIP) =
    //        m_SD.block<1, NIP>(i, 2 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(NIP, 2).transpose()) +
    //        m_SD.block<1, NIP>(i, 1 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(2 * NIP, 2).transpose());
    //    Scaled_Combined_PE.block<1, NIP>((3 * i) + 2, 4 * NIP) =
    //        m_SD.block<1, NIP>(i, 2 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(0, 2).transpose()) +
    //        m_SD.block<1, NIP>(i, 0 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(2 * NIP, 2).transpose());
    //    Scaled_Combined_PE.block<1, NIP>((3 * i) + 2, 5 * NIP) =
    //        m_SD.block<1, NIP>(i, 1 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(0, 2).transpose()) +
    //        m_SD.block<1, NIP>(i, 0 * NIP).cwiseProduct(FCscaled.template block<NIP, 1>(NIP, 2).transpose());
    //}

    //// =============================================================================
    //// Get the stiffness tensor in 6x6 matrix form
    //// =============================================================================

    // const ChMatrixNM<double, 6, 6>& D = GetMaterial()->Get_D();

    //// =============================================================================
    //// Multiply the scaled and combined partial derivative block matrix by the stiffness matrix for each individual
    //// Gauss quadrature point
    //// =============================================================================
    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DScaled_Combined_PE;
    // DScaled_Combined_PE.resize(3 * NSF, 6 * NIP);

    // DScaled_Combined_PE.template block<3 * NSF, NIP>(0, 0) =
    //    D(0, 0) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 0) +
    //    D(0, 1) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, NIP) +
    //    D(0, 2) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 2 * NIP) +
    //    D(0, 3) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 3 * NIP) +
    //    D(0, 4) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 4 * NIP) +
    //    D(0, 5) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 5 * NIP);
    // DScaled_Combined_PE.template block<3 * NSF, NIP>(0, NIP) =
    //    D(1, 0) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 0) +
    //    D(1, 1) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, NIP) +
    //    D(1, 2) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 2 * NIP) +
    //    D(1, 3) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 3 * NIP) +
    //    D(1, 4) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 4 * NIP) +
    //    D(1, 5) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 5 * NIP);
    // DScaled_Combined_PE.template block<3 * NSF, NIP>(0, 2 * NIP) =
    //    D(2, 0) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 0) +
    //    D(2, 1) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, NIP) +
    //    D(2, 2) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 2 * NIP) +
    //    D(2, 3) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 3 * NIP) +
    //    D(2, 4) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 4 * NIP) +
    //    D(2, 5) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 5 * NIP);
    // DScaled_Combined_PE.template block<3 * NSF, NIP>(0, 3 * NIP) =
    //    D(3, 0) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 0) +
    //    D(3, 1) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, NIP) +
    //    D(3, 2) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 2 * NIP) +
    //    D(3, 3) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 3 * NIP) +
    //    D(3, 4) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 4 * NIP) +
    //    D(3, 5) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 5 * NIP);
    // DScaled_Combined_PE.template block<3 * NSF, NIP>(0, 4 * NIP) =
    //    D(4, 0) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 0) +
    //    D(4, 1) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, NIP) +
    //    D(4, 2) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 2 * NIP) +
    //    D(4, 3) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 3 * NIP) +
    //    D(4, 4) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 4 * NIP) +
    //    D(4, 5) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 5 * NIP);
    // DScaled_Combined_PE.template block<3 * NSF, NIP>(0, 5 * NIP) =
    //    D(5, 0) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 0) +
    //    D(5, 1) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, NIP) +
    //    D(5, 2) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 2 * NIP) +
    //    D(5, 3) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 3 * NIP) +
    //    D(5, 4) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 4 * NIP) +
    //    D(5, 5) * Scaled_Combined_PE.block<3 * NSF, NIP>(0, 5 * NIP);

    //// =============================================================================
    //// Multiply the partial derivative block matrix by the final scaled and combined partial derivative block matrix
    //// to obtain the potentially non-symmetric and non-sparse component of the Jacobian matrix
    //// =============================================================================

    // H = PE * DScaled_Combined_PE.transpose();

    ////==============================================================================
    ////==============================================================================
    //// Calculate the sparse and symmetric component of the Jacobian matrix
    ////==============================================================================
    ////==============================================================================

    //// =============================================================================
    //// Calculate each individual value of the Green-Lagrange strain component by component across all the
    //// Gauss-Quadrature points at a time for the current layer to better leverage vectorized CPU instructions.
    //// The result is then scaled by minus the Gauss quadrature weight times the element Jacobian at the
    //// corresponding Gauss point (m_kGQ) for efficiency.
    //// Results are written in Voigt notation: epsilon = [E11,E22,E33,2*E23,2*E13,2*E12]
    //// =============================================================================

    //// Each entry in E1 = kGQ*E11 = kGQ*1/2*(F11*F11+F21*F21+F31*F31-1)
    // VectorNIP E1_Block = FC.template block<NIP, 1>(0, 0).cwiseProduct(FC.template block<NIP, 1>(0, 0)) +
    //    FC.template block<NIP, 1>(0, 1).cwiseProduct(FC.template block<NIP, 1>(0, 1)) +
    //    FC.template block<NIP, 1>(0, 2).cwiseProduct(FC.template block<NIP, 1>(0, 2));
    // E1_Block.array() -= 1;
    // E1_Block.array() *= 0.5 * m_kGQ.array();

    //// Each entry in E2 = kGQ*E22 = kGQ*1/2*(F12*F12+F22*F22+F32*F32-1)
    // VectorNIP E2_Block = FC.template block<NIP, 1>(NIP, 0).cwiseProduct(FC.template block<NIP, 1>(NIP, 0)) +
    //    FC.template block<NIP, 1>(NIP, 1).cwiseProduct(FC.template block<NIP, 1>(NIP, 1)) +
    //    FC.template block<NIP, 1>(NIP, 2).cwiseProduct(FC.template block<NIP, 1>(NIP, 2));
    // E2_Block.array() -= 1;
    // E2_Block.array() *= 0.5 * m_kGQ.array();

    //// Each entry in E3 = kGQ*E33 = kGQ*1/2*(F13*F13+F23*F23+F33*F33-1)
    // VectorNIP E3_Block = FC.template block<NIP, 1>(2 * NIP, 0).cwiseProduct(FC.template block<NIP, 1>(2 * NIP, 0)) +
    //    FC.template block<NIP, 1>(2 * NIP, 1).cwiseProduct(FC.template block<NIP, 1>(2 * NIP, 1)) +
    //    FC.template block<NIP, 1>(2 * NIP, 2).cwiseProduct(FC.template block<NIP, 1>(2 * NIP, 2));
    // E3_Block.array() -= 1;
    // E3_Block.array() *= 0.5 * m_kGQ.array();

    //// Each entry in E4 = kGQ*2*E23 = kGQ*(F12*F13+F22*F23+F32*F33)
    // VectorNIP E4_Block = FC.template block<NIP, 1>(NIP, 0).cwiseProduct(FC.template block<NIP, 1>(2 * NIP, 0)) +
    //    FC.template block<NIP, 1>(NIP, 1).cwiseProduct(FC.template block<NIP, 1>(2 * NIP, 1)) +
    //    FC.template block<NIP, 1>(NIP, 2).cwiseProduct(FC.template block<NIP, 1>(2 * NIP, 2));
    // E4_Block.array() *= m_kGQ.array();

    //// Each entry in E5 = kGQ*2*E13 = kGQ*(F11*F13+F21*F23+F31*F33)
    // VectorNIP E5_Block = FC.template block<NIP, 1>(0, 0).cwiseProduct(FC.template block<NIP, 1>(2 * NIP, 0)) +
    //    FC.template block<NIP, 1>(0, 1).cwiseProduct(FC.template block<NIP, 1>(2 * NIP, 1)) +
    //    FC.template block<NIP, 1>(0, 2).cwiseProduct(FC.template block<NIP, 1>(2 * NIP, 2));
    // E5_Block.array() *= m_kGQ.array();

    //// Each entry in E6 = kGQ*2*E12 = (F11*F12+F21*F22+F31*F32)
    // VectorNIP E6_Block = FC.template block<NIP, 1>(0, 0).cwiseProduct(FC.template block<NIP, 1>(NIP, 0)) +
    //    FC.template block<NIP, 1>(0, 1).cwiseProduct(FC.template block<NIP, 1>(NIP, 1)) +
    //    FC.template block<NIP, 1>(0, 2).cwiseProduct(FC.template block<NIP, 1>(NIP, 2));
    // E6_Block.array() *= m_kGQ.array();

    //// =============================================================================
    //// Calculate the 2nd Piola-Kirchoff stresses in Voigt notation across all the Gauss quadrature points in the
    //// current layer at a time component by component.
    //// Note that the Green-Largrange strain components have been scaled have already been combined with their scaled
    //// time derivatives and minus the Gauss quadrature weight times the element Jacobian at the corresponding Gauss
    //// point to make the calculation of the 2nd Piola-Kirchoff stresses more efficient.
    ////  kGQ*SPK2 = kGQ*[SPK2_11,SPK2_22,SPK2_33,SPK2_23,SPK2_13,SPK2_12] = D * E_Combined
    //// =============================================================================

    // VectorNIP SPK2_1_Block = D(0, 0) * E1_Block + D(0, 1) * E2_Block + D(0, 2) * E3_Block + D(0, 3) * E4_Block +
    //    D(0, 4) * E5_Block + D(0, 5) * E6_Block;
    // VectorNIP SPK2_2_Block = D(1, 0) * E1_Block + D(1, 1) * E2_Block + D(1, 2) * E3_Block + D(1, 3) * E4_Block +
    //    D(1, 4) * E5_Block + D(1, 5) * E6_Block;
    // VectorNIP SPK2_3_Block = D(2, 0) * E1_Block + D(2, 1) * E2_Block + D(2, 2) * E3_Block + D(2, 3) * E4_Block +
    //    D(2, 4) * E5_Block + D(2, 5) * E6_Block;
    // VectorNIP SPK2_4_Block = D(3, 0) * E1_Block + D(3, 1) * E2_Block + D(3, 2) * E3_Block + D(3, 3) * E4_Block +
    //    D(3, 4) * E5_Block + D(3, 5) * E6_Block;
    // VectorNIP SPK2_5_Block = D(4, 0) * E1_Block + D(4, 1) * E2_Block + D(4, 2) * E3_Block + D(4, 3) * E4_Block +
    //    D(4, 4) * E5_Block + D(4, 5) * E6_Block;
    // VectorNIP SPK2_6_Block = D(5, 0) * E1_Block + D(5, 1) * E2_Block + D(5, 2) * E3_Block + D(5, 3) * E4_Block +
    //    D(5, 4) * E5_Block + D(5, 5) * E6_Block;

    //// =============================================================================
    //// Multiply the shape function derivative matrix by the 2nd Piola-Kirchoff stresses for each corresponding Gauss
    //// quadrature point
    //// =============================================================================

    // ChMatrixNM<double, NSF, 3 * NIP> S_scaled_SD;

    // for (auto i = 0; i < NSF; i++) {
    //    S_scaled_SD.template block<1, NIP>(i, 0) =
    //        SPK2_1_Block.transpose().cwiseProduct(m_SD.block<1, NIP>(i, 0 * NIP)) +
    //        SPK2_6_Block.transpose().cwiseProduct(m_SD.block<1, NIP>(i, 1 * NIP)) +
    //        SPK2_5_Block.transpose().cwiseProduct(m_SD.block<1, NIP>(i, 2 * NIP));

    //    S_scaled_SD.template block<1, NIP>(i, NIP) =
    //        SPK2_6_Block.transpose().cwiseProduct(m_SD.block<1, NIP>(i, 0 * NIP)) +
    //        SPK2_2_Block.transpose().cwiseProduct(m_SD.block<1, NIP>(i, 1 * NIP)) +
    //        SPK2_4_Block.transpose().cwiseProduct(m_SD.block<1, NIP>(i, 2 * NIP));

    //    S_scaled_SD.template block<1, NIP>(i, 2 * NIP) =
    //        SPK2_5_Block.transpose().cwiseProduct(m_SD.block<1, NIP>(i, 0 * NIP)) +
    //        SPK2_4_Block.transpose().cwiseProduct(m_SD.block<1, NIP>(i, 1 * NIP)) +
    //        SPK2_3_Block.transpose().cwiseProduct(m_SD.block<1, NIP>(i, 2 * NIP));
    //}

    //// =============================================================================
    //// Calculate just the non-sparse upper triangular entires of the sparse and symmetric component of the Jacobian
    //// matrix, combine this with the scaled mass matrix, and then expand them out to full size by summing the
    //// contribution into the correct locations of the full sized Jacobian matrix
    //// =============================================================================

    //// Add in the Mass Matrix Which is Stored in Compact Upper Triangular Form
    // ChVectorN<double, (NSF * (NSF + 1)) / 2> ScaledMassMatrix = Mfactor * m_MassMatrix;

    // unsigned int idx = 0;
    // for (unsigned int i = 0; i < NSF; i++) {
    //    for (unsigned int j = i; j < NSF; j++) {
    //        double d = Kfactor * m_SD.row(i) * S_scaled_SD.row(j).transpose();
    //        d += ScaledMassMatrix(idx);

    //        H(3 * i, 3 * j) += d;
    //        H(3 * i + 1, 3 * j + 1) += d;
    //        H(3 * i + 2, 3 * j + 2) += d;
    //        if (i != j) {
    //            H(3 * j, 3 * i) += d;
    //            H(3 * j + 1, 3 * i + 1) += d;
    //            H(3 * j + 2, 3 * i + 2) += d;
    //        }
    //        idx++;
    //    }
    //}
}

template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::ComputeInternalJacobianPreInt(ChMatrixRef& H,
                                                                   double Kfactor,
                                                                   double Rfactor,
                                                                   double Mfactor) {
    // Calculate the Jacobian of the generalize internal force vector using the "Pre-Integration" style of method
    // assuming a linear viscoelastic material model (single term damping model).  For this style of method, the
    // components of the generalized internal force vector and its Jacobian that need to be integrated across the volume
    // are calculated once prior to the start of the simulation.  This makes this method well suited for applications
    // with many discrete layers since the in-simulation calculations are independent of the number of Gauss quadrature
    // points used throughout the entire element.  Since computationally expensive quantities are required for both the
    // generalized internal force vector and its Jacobian, these values were cached for reuse during this Jacobian
    // calculation.

    Matrix3xN e_bar;
    Matrix3xN e_bar_dot;

    CalcCoordMatrix(e_bar);
    CalcCoordDerivMatrix(e_bar_dot);

    // Build the [9 x NSF^2] matrix containing the combined scaled nodal coordinates and their time derivatives.
    Matrix3xN temp = (Kfactor + m_Alpha * Rfactor) * e_bar + (m_Alpha * Kfactor) * e_bar_dot;
    ChMatrixNM<double, 1, NSF> tempRow0 = temp.template block<1, NSF>(0, 0);
    ChMatrixNM<double, 1, NSF> tempRow1 = temp.template block<1, NSF>(1, 0);
    ChMatrixNM<double, 1, NSF> tempRow2 = temp.template block<1, NSF>(2, 0);
    ChMatrixNMc<double, 9, NSF * NSF> PI2;

    for (unsigned int v = 0; v < NSF; v++) {
        PI2.template block<3, NSF>(0, NSF * v) = e_bar.template block<3, 1>(0, v) * tempRow0;
        PI2.template block<3, NSF>(3, NSF * v) = e_bar.template block<3, 1>(0, v) * tempRow1;
        PI2.template block<3, NSF>(6, NSF * v) = e_bar.template block<3, 1>(0, v) * tempRow2;
    }

    // Calculate the matrix containing the dense part of the Jacobian matrix in a reordered form. This is then reordered
    // from its [9 x NSF^2] form into its required [3*NSF x 3*NSF] form
    ChMatrixNM<double, 9, NSF* NSF> K2 = -PI2 * m_O2;

    for (unsigned int k = 0; k < NSF; k++) {
        for (unsigned int f = 0; f < NSF; f++) {
            H.block<3, 1>(3 * k, 3 * f) = K2.template block<3, 1>(0, NSF * f + k);
            H.block<3, 1>(3 * k, 3 * f + 1) = K2.template block<3, 1>(3, NSF * f + k);
            H.block<3, 1>(3 * k, 3 * f + 2) = K2.template block<3, 1>(6, NSF * f + k);
        }
    }

    // Add in the sparse (blocks times the 3x3 identity matrix) component of the Jacobian that was already calculated as
    // part of the generalized internal force calculations
    MatrixNxN K_K13Compact = -Kfactor * m_K13Compact;
    for (unsigned int i = 0; i < NSF; i++) {
        for (unsigned int j = 0; j < NSF; j++) {
            H(3 * i, 3 * j) += K_K13Compact(i, j);
            H(3 * i + 1, 3 * j + 1) += K_K13Compact(i, j);
            H(3 * i + 2, 3 * j + 2) += K_K13Compact(i, j);
        }
    }
}

// -----------------------------------------------------------------------------
// Shape functions
// -----------------------------------------------------------------------------

// Nx1 Vector Form of the Normalized Shape Functions
// [s1; s2; s3; ...]
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::Calc_Sxi_compact(VectorN& Sxi_compact, double xi, double eta, double zeta) {
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
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::Calc_Sxi_xi_compact(VectorN& Sxi_xi_compact, double xi, double eta, double zeta) {
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
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::Calc_Sxi_eta_compact(VectorN& Sxi_eta_compact,
                                                          double xi,
                                                          double eta,
                                                          double zeta) {
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
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::Calc_Sxi_zeta_compact(VectorN& Sxi_zeta_compact,
                                                           double xi,
                                                           double eta,
                                                           double zeta) {
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
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::Calc_Sxi_D(MatrixNx3c& Sxi_D, double xi, double eta, double zeta) {
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

template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::CalcCoordVector(Vector3N& e) {
    e.segment(0, 3) = m_nodes[0]->GetPos().eigen();
    e.segment(3, 3) = m_nodes[0]->GetD().eigen();
    e.segment(6, 3) = m_nodes[0]->GetDD().eigen();
    e.segment(9, 3) = m_nodes[0]->GetDDD().eigen();

    e.segment(12, 3) = m_nodes[1]->GetPos().eigen();
    e.segment(15, 3) = m_nodes[1]->GetD().eigen();
    e.segment(18, 3) = m_nodes[1]->GetDD().eigen();
    e.segment(21, 3) = m_nodes[1]->GetDDD().eigen();
}

template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::CalcCoordMatrix(Matrix3xN& ebar) {
    ebar.col(0) = m_nodes[0]->GetPos().eigen();
    ebar.col(1) = m_nodes[0]->GetD().eigen();
    ebar.col(2) = m_nodes[0]->GetDD().eigen();
    ebar.col(3) = m_nodes[0]->GetDDD().eigen();

    ebar.col(4) = m_nodes[1]->GetPos().eigen();
    ebar.col(5) = m_nodes[1]->GetD().eigen();
    ebar.col(6) = m_nodes[1]->GetDD().eigen();
    ebar.col(7) = m_nodes[1]->GetDDD().eigen();
}

template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::CalcCoordDerivVector(Vector3N& edot) {
    edot.segment(0, 3) = m_nodes[0]->GetPos_dt().eigen();
    edot.segment(3, 3) = m_nodes[0]->GetD_dt().eigen();
    edot.segment(6, 3) = m_nodes[0]->GetDD_dt().eigen();
    edot.segment(9, 3) = m_nodes[0]->GetDDD_dt().eigen();

    edot.segment(12, 3) = m_nodes[1]->GetPos_dt().eigen();
    edot.segment(15, 3) = m_nodes[1]->GetD_dt().eigen();
    edot.segment(18, 3) = m_nodes[1]->GetDD_dt().eigen();
    edot.segment(21, 3) = m_nodes[1]->GetDDD_dt().eigen();
}

template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::CalcCoordDerivMatrix(Matrix3xN& ebardot) {
    ebardot.col(0) = m_nodes[0]->GetPos_dt().eigen();
    ebardot.col(1) = m_nodes[0]->GetD_dt().eigen();
    ebardot.col(2) = m_nodes[0]->GetDD_dt().eigen();
    ebardot.col(3) = m_nodes[0]->GetDDD_dt().eigen();

    ebardot.col(4) = m_nodes[1]->GetPos_dt().eigen();
    ebardot.col(5) = m_nodes[1]->GetD_dt().eigen();
    ebardot.col(6) = m_nodes[1]->GetDD_dt().eigen();
    ebardot.col(7) = m_nodes[1]->GetDDD_dt().eigen();
}

template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::CalcCombinedCoordMatrix(MatrixNx6& ebar_ebardot) {
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
}

// Calculate the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element
template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::Calc_J_0xi(ChMatrix33<double>& J_0xi, double xi, double eta, double zeta) {
    MatrixNx3c Sxi_D;
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    J_0xi = m_ebar0 * Sxi_D;
}

// Calculate the determinant of the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element
template <int NP, int NT>
double ChElementBeamANCF_3243<NP, NT>::Calc_det_J_0xi(double xi, double eta, double zeta) {
    ChMatrix33<double> J_0xi;
    Calc_J_0xi(J_0xi, xi, eta, zeta);

    return (J_0xi.determinant());
}

template <int NP, int NT>
void ChElementBeamANCF_3243<NP, NT>::RotateReorderStiffnessMatrix(ChMatrixNM<double, 6, 6>& D, double theta) {
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
ChQuadratureTables static_tables_3243(1, CH_QUADRATURE_STATIC_TABLES);
//#endif // !CH_QUADRATURE_STATIC_TABLES

template <int NP, int NT>
ChQuadratureTables* ChElementBeamANCF_3243<NP, NT>::GetStaticGQTables() {
    return &static_tables_3243;
}

////////////////////////////////////////////////////////////////

// ============================================================================
// Implementation of ChMaterialBeamANCF_3243 methods
// ============================================================================

// Construct an isotropic material.
ChMaterialBeamANCF_3243::ChMaterialBeamANCF_3243(double rho,        // material density
                                                 double E,          // Young's modulus
                                                 double nu,         // Poisson ratio
                                                 const double& k1,  // Shear correction factor along beam local y axis
                                                 const double& k2   // Shear correction factor along beam local z axis
                                                 )
    : m_rho(rho) {
    double G = 0.5 * E / (1 + nu);
    Calc_D0_Dv(ChVector<>(E), ChVector<>(nu), ChVector<>(G), k1, k2);
}

// Construct a (possibly) orthotropic material.
ChMaterialBeamANCF_3243::ChMaterialBeamANCF_3243(double rho,            // material density
                                                 const ChVector<>& E,   // elasticity moduli (E_x, E_y, E_z)
                                                 const ChVector<>& nu,  // Poisson ratios (nu_xy, nu_xz, nu_yz)
                                                 const ChVector<>& G,   // shear moduli (G_xy, G_xz, G_yz)
                                                 const double& k1,  // Shear correction factor along beam local y axis
                                                 const double& k2   // Shear correction factor along beam local z axis
                                                 )
    : m_rho(rho) {
    Calc_D0_Dv(E, nu, G, k1, k2);
}

// Calculate the matrix form of two stiffness tensors used by the ANCF beam for selective reduced integration of the
// Poisson effect
void ChMaterialBeamANCF_3243::Calc_D0_Dv(const ChVector<>& E,
                                         const ChVector<>& nu,
                                         const ChVector<>& G,
                                         double k1,
                                         double k2) {
    // orthotropic material ref: http://homes.civil.aau.dk/lda/Continuum/material.pdf
    // except position of the shear terms is different to match the original ANCF reference paper

    double nu_12 = nu.x();
    double nu_13 = nu.y();
    double nu_23 = nu.z();
    double nu_21 = nu_12 * E.y() / E.x();
    double nu_31 = nu_13 * E.z() / E.x();
    double nu_32 = nu_23 * E.z() / E.y();
    double k = 1.0 - nu_23 * nu_32 - nu_12 * nu_21 - nu_13 * nu_31 - nu_12 * nu_23 * nu_31 - nu_21 * nu_32 * nu_13;

    // Component of Stiffness Tensor that does not contain the Poisson Effect
    m_D0(0) = E.x();
    m_D0(1) = E.y();
    m_D0(2) = E.z();
    m_D0(3) = G.z();
    m_D0(4) = G.y() * k1;
    m_D0(5) = G.x() * k2;

    // Remaining components of the Stiffness Tensor that contain the Poisson Effect
    m_Dv(0, 0) = E.x() * (1 - nu_23 * nu_32) / k - m_D0(0);
    m_Dv(1, 0) = E.y() * (nu_13 * nu_32 + nu_12) / k;
    m_Dv(2, 0) = E.z() * (nu_12 * nu_23 + nu_13) / k;

    m_Dv(0, 1) = E.x() * (nu_23 * nu_31 + nu_21) / k;
    m_Dv(1, 1) = E.y() * (1 - nu_13 * nu_31) / k - m_D0(1);
    m_Dv(2, 1) = E.z() * (nu_13 * nu_21 + nu_23) / k;

    m_Dv(0, 2) = E.x() * (nu_21 * nu_32 + nu_31) / k;
    m_Dv(1, 2) = E.y() * (nu_12 * nu_31 + nu_32) / k;
    m_Dv(2, 2) = E.z() * (1 - nu_12 * nu_21) / k - m_D0(2);

    m_D.diagonal() = m_D0;
    m_D.block<3, 3>(0, 0) += m_Dv;
}

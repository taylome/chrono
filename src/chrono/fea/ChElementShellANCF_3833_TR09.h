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
// Higher oreder ANCF shell element with 8 nodes. Description of this element (3833)
// and its internal forces may be found in: Henrik Ebel, Marko K Matikainen, 
// Vesa-Ville Hurskainen, and Aki Mikkola. Analysis of high-order quadrilateral
// plate elements based on the absolute nodal coordinate formulation for three -
// dimensional elasticity.Advances in Mechanical Engineering, 9(6) : 1687814017705069, 2017.
// =============================================================================
// Internal Force Calculation Method is based on:  Liu, Cheng, Qiang Tian, and
// Haiyan Hu. "Dynamics of a large scale rigid–flexible multibody system composed
// of composite laminated plates." Multibody System Dynamics 26, no. 3 (2011): 283-305.
// =============================================================================
// TR09 = a Liu style implementation of the element with pre-calculation
//     of the matrices needed for both the internal force (O1) and Jacobian Calculation (O2)
//
//  Mass Matrix = Constant, pre-calculated 8x8 matrix
//
//  Generalized Force due to gravity = Constant 24x1 Vector
//     (assumption that gravity is constant too)
//
//  Generalized Internal Force Vector = Calculated using the Liu method:
//     Math is based on the method presented by Liu, Tian, and Hu
//     "Full Integration" Number of GQ Integration Points (5x3x3)
//
//  Jacobian of the Generalized Internal Force Vector = Analytical Jacobian that
//     is based on the method presented by Liu, Tian, and Hu.
//     Internal force calculation results are cached for reuse during the Jacobian
//     calculations
//     O2 is precalculated and stored rather than calculated from O1
//
// =============================================================================

#ifndef CHELEMENTSHELLANCF3833TR09_H
#define CHELEMENTSHELLANCF3833TR09_H

#include <vector>

#include "chrono/core/ChQuadrature.h"
#include "chrono/fea/ChElementShell.h"
#include "chrono/fea/ChNodeFEAxyzDD.h"

namespace chrono {
namespace fea {

/// @addtogroup fea_elements
/// @{

/// Material definition.
/// This class implements material properties for an ANCF Shell.
class ChApi ChMaterialShellANCF_3833_TR09 {
  public:
    /// Construct an isotropic material.
    ChMaterialShellANCF_3833_TR09(double rho,        ///< material density
                                 double E,          ///< Young's modulus
                                 double nu          ///< Poisson ratio
    );

    /// Construct a (possibly) orthotropic material.
    ChMaterialShellANCF_3833_TR09(double rho,            ///< material density
                                 const ChVector<>& E,   ///< elasticity moduli (E_x, E_y, E_z)
                                 const ChVector<>& nu,  ///< Poisson ratios (nu_xy, nu_xz, nu_yz)
                                 const ChVector<>& G    ///< shear moduli (G_xy, G_xz, G_yz)
    );

    /// Return the material density.
    double Get_rho() const { return m_rho; }

    const ChMatrixNM<double, 6, 6>& Get_D() const { return m_D; }
    const ChMatrixNM<double, 6, 6>& Get_D0() const { return m_D0; }
    const ChMatrixNM<double, 6, 6>& Get_Dv() const { return m_Dv; }

  private:
    /// Calculate the matrix form of two stiffness tensors used by the ANCF shell for selective reduced integration of
    /// the Poisson effect as well as the composite stiffness tensors.
    void Calc_D0_Dv(const ChVector<>& E, const ChVector<>& nu, const ChVector<>& G);

    double m_rho;  ///< density
    ChMatrixNM<double, 6, 6>
        m_D;  ///< matrix of elastic coefficients
    ChMatrixNM<double, 6, 6>
        m_D0;  ///< matrix of elastic coefficients (split of diagonal terms for integration across the entire element)
    ChMatrixNM<double, 6, 6> m_Dv;  ///< matrix of elastic coefficients (remainder of split, upper 3x3 terms for
                                    ///< integration only on the midsurface)

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @addtogroup fea_elements
/// @{

/// ANCF shell element with four nodes.
///
/// The node numbering is in ccw fashion as in the following scheme:
/// <pre>
///         v
///         ^
///         |
/// D o-----G-----o C
///   |     |     |
/// --H-----+-----F----> u
///   |     |     |
/// A o-----E-----o B
/// </pre>

class ChApi ChElementShellANCF_3833_TR09 : public ChElementShell, public ChLoadableUV, public ChLoadableUVW {
  public:
      template <typename T, int M, int N>
      using ChMatrixNMc = Eigen::Matrix<T, M, N, Eigen::ColMajor>;

    using VectorN = ChVectorN<double, 24>;
    using Vector3N = ChVectorN<double, 72>;
    using Matrix3xN = ChMatrixNM<double, 3, 24>;
    using Matrix3x3N = ChMatrixNM<double, 3, 72>;
    using Matrix6x3N = ChMatrixNM<double, 6, 72>;
    using MatrixNxN = ChMatrixNM<double, 24, 24>;
    using Matrix3Nx3N = ChMatrixNM<double, 72, 72>;
    using MatrixNx3 = ChMatrixNM<double, 24, 3>;
    using MatrixNx3c = ChMatrixNMc<double, 24, 3>;

    ChElementShellANCF_3833_TR09();
    ~ChElementShellANCF_3833_TR09() {}

    /// Get the number of nodes used by this element.
    virtual int GetNnodes() override { return 8; }

    /// Get the number of coordinates in the field used by the referenced nodes.
    virtual int GetNdofs() override { return 8 * 9; }

    /// Get the number of coordinates from the n-th node used by this element.
    virtual int GetNodeNdofs(int n) override { return 9; }

    /// Specify the nodes of this element.
    void SetNodes(std::shared_ptr<ChNodeFEAxyzDD> nodeA,
                  std::shared_ptr<ChNodeFEAxyzDD> nodeB,
                  std::shared_ptr<ChNodeFEAxyzDD> nodeC,
                  std::shared_ptr<ChNodeFEAxyzDD> nodeD,
                  std::shared_ptr<ChNodeFEAxyzDD> nodeE,
                  std::shared_ptr<ChNodeFEAxyzDD> nodeF,
                  std::shared_ptr<ChNodeFEAxyzDD> nodeG,
                  std::shared_ptr<ChNodeFEAxyzDD> nodeH);

    /// Specify the element dimensions.
    void SetDimensions(double lenX, double lenY, double thicknessZ) {
        m_lenX = lenX;
        m_lenY = lenY;
        m_thicknessZ = thicknessZ;
    }

    /// Specify the element material.
    void SetMaterial(std::shared_ptr<ChMaterialShellANCF_3833_TR09> shell_mat) { m_material = shell_mat; }

    /// Access the n-th node of this element.
    virtual std::shared_ptr<ChNodeFEAbase> GetNodeN(int n) override { return m_nodes[n]; }

    /// Get a handle to the first node of this element.
    std::shared_ptr<ChNodeFEAxyzDD> GetNodeA() const { return m_nodes[0]; }

    /// Get a handle to the second node of this element.
    std::shared_ptr<ChNodeFEAxyzDD> GetNodeB() const { return m_nodes[1]; }

    /// Get a handle to the second node of this element.
    std::shared_ptr<ChNodeFEAxyzDD> GetNodeC() const { return m_nodes[2]; }

    /// Get a handle to the second node of this element.
    std::shared_ptr<ChNodeFEAxyzDD> GetNodeD() const { return m_nodes[3]; }

    /// Get a handle to the first node of this element.
    std::shared_ptr<ChNodeFEAxyzDD> GetNodeE() const { return m_nodes[4]; }

    /// Get a handle to the second node of this element.
    std::shared_ptr<ChNodeFEAxyzDD> GetNodeF() const { return m_nodes[5]; }

    /// Get a handle to the third node of this element.
    std::shared_ptr<ChNodeFEAxyzDD> GetNodeG() const { return m_nodes[6]; }

    /// Get a handle to the fourth node of this element.
    std::shared_ptr<ChNodeFEAxyzDD> GetNodeH() const { return m_nodes[7]; }
	
    /// Return the material.
    std::shared_ptr<ChMaterialShellANCF_3833_TR09> GetMaterial() const { return m_material; }

    /// Turn gravity on/off.
    void SetGravityOn(bool val) { m_gravity_on = val; }

    /// Set the structural damping.
    void SetAlphaDamp(double a);

    /// Get the element length in the X direction.
    double GetLengthX() const { return m_lenX; }

    /// Get the element length in the Y direction.
    double GetLengthY() { return m_lenY; }

    /// Get the total thickness of the shell element.
    double GetThicknessZ() { return m_thicknessZ; }

    /// Poisson effect selection. - Not currently used
    enum StrainFormulation {
        CMPoisson,   ///< Continuum-Mechanics formulation, including Poisson effects
        CMNoPoisson  ///< Continuum-Mechanics formulation, disregarding Poisson effects
    };

  public:
    // Interface to ChElementBase base class
    // -------------------------------------

    // Fill the D vector (column matrix) with the current field values at the nodes of the element, with proper
    // ordering. If the D vector has not the size of this->GetNdofs(), it will be resized.
    //  {x_a y_a z_a Dx_a Dx_a Dx_a x_b y_b z_b Dx_b Dy_b Dz_b}
    virtual void GetStateBlock(ChVectorDynamic<>& mD) override;

    // Set H as a linear combination of M, K, and R.
    //   H = Mfactor * [M] + Kfactor * [K] + Rfactor * [R],
    // where [M] is the mass matrix, [K] is the stiffness matrix, and [R] is the damping matrix.
    virtual void ComputeKRMmatricesGlobal(ChMatrixRef H,
                                          double Kfactor,
                                          double Rfactor = 0,
                                          double Mfactor = 0) override;

    // Set M as the global mass matrix.
    virtual void ComputeMmatrixGlobal(ChMatrixRef M) override;

    /// Add contribution of element inertia to total nodal masses
    virtual void ComputeNodalMass() override;

    /// Computes the internal forces.
    /// (E.g. the actual position of nodes is not in relaxed reference position) and set values in the Fi vector.
    virtual void ComputeInternalForces(ChVectorDynamic<>& Fi) override;

    /// Update the state of this element.
    virtual void Update() override;

    // Strain Formulation methods
    // --------------------------

    /// Set the strain formulation. - Not currently used
    void SetStrainFormulation(StrainFormulation model) { m_strain_form = model; }

    /// Get the strain formulation. - Not currently used
    StrainFormulation GetStrainFormulation() const { return m_strain_form; }

    // Interface to ChElementShell base class
    // --------------------------------------

    //// Dummy method definitions.
    //virtual void EvaluateSectionStrain(const double, chrono::ChVector<double>&) override {}

    //virtual void EvaluateSectionForceTorque(const double,
    //                                        chrono::ChVector<double>&,
    //                                        chrono::ChVector<double>&) override {}

    /// Gets the xyz displacement of a point on the shell midsurface,
    /// and the rotation RxRyRz of section plane, at abscissa '(xi,eta,0)'.
    virtual void EvaluateSectionDisplacement(const double xi, const double eta, ChVector<>& u_displ, ChVector<>& u_rotaz) override {}

    /// Gets the absolute xyz position of a point on the shell midsurface,
    /// and the absolute rotation of section plane, at abscissa '(xi,eta,0)'.
    /// Note, nodeA = (xi=-1, eta=-1)
    /// Note, nodeB = (xi=1, eta=-1)
    /// Note, nodeC = (xi=1, eta=1)
    /// Note, nodeD = (xi=-1, eta=1)
    /// Note, 'displ' is the displ.state of 2 nodes, ex. get it as GetStateBlock()
    /// Results are expressed in world reference
    virtual void EvaluateSectionFrame(const double xi, const double eta, ChVector<>& point, ChQuaternion<>& rot) override;

    virtual void EvaluateSectionPoint(const double u, const double v, ChVector<>& point) override;

    virtual void EvaluateSectionVelNorm(double U, double V, ChVector<>& Result) override;

    // Functions for ChLoadable interface
    // ----------------------------------

    /// Gets the number of DOFs affected by this element (position part).
    virtual int LoadableGet_ndof_x() override { return 8 * 9; }

    /// Gets the number of DOFs affected by this element (velocity part).
    virtual int LoadableGet_ndof_w() override { return 8 * 9; }

    /// Gets all the DOFs packed in a single vector (position part).
    virtual void LoadableGetStateBlock_x(int block_offset, ChState& mD) override;

    /// Gets all the DOFs packed in a single vector (velocity part).
    virtual void LoadableGetStateBlock_w(int block_offset, ChStateDelta& mD) override;

    /// Increment all DOFs using a delta.
    virtual void LoadableStateIncrement(const unsigned int off_x,
                                        ChState& x_new,
                                        const ChState& x,
                                        const unsigned int off_v,
                                        const ChStateDelta& Dv) override;

    /// Number of coordinates in the interpolated field, ex=3 for a
    /// tetrahedron finite element or a cable, = 1 for a thermal problem, etc.
    virtual int Get_field_ncoords() override { return 9; }

    /// Tell the number of DOFs blocks (ex. =1 for a body, =4 for a tetrahedron, etc.)
    virtual int GetSubBlocks() override { return 8; }

    /// Get the offset of the i-th sub-block of DOFs in global vector.
    virtual unsigned int GetSubBlockOffset(int nblock) override { return m_nodes[nblock]->NodeGetOffset_w(); }

    /// Get the size of the i-th sub-block of DOFs in global vector.
    virtual unsigned int GetSubBlockSize(int nblock) override { return 9; }

    // What is this used for?  What is the definition?
    // void EvaluateSectionVelNorm(double U, ChVector<>& Result);

    /// Get the pointers to the contained ChVariables, appending to the mvars vector.
    virtual void LoadableGetVariables(std::vector<ChVariables*>& mvars) override;

    /// Evaluate N'*F , where N is some type of shape function
    /// evaluated at U,V coordinates of the surface, each ranging in -1..+1
    /// F is a load, N'*F is the resulting generalized load
    /// Returns also det[J] with J=[dx/du,..], that might be useful in gauss quadrature.
    virtual void ComputeNF(const double U,              ///< parametric coordinate in surface
                           const double V,              ///< parametric coordinate in surface
                           ChVectorDynamic<>& Qi,       ///< Return result of Q = N'*F  here
                           double& detJ,                ///< Return det[J] here
                           const ChVectorDynamic<>& F,  ///< Input F vector, size is =n. field coords.
                           ChVectorDynamic<>* state_x,  ///< if != 0, update state (pos. part) to this, then evaluate
                           ChVectorDynamic<>* state_w   ///< if != 0, update state (speed part) to this, then evaluate
                           ) override;

    /// Evaluate N'*F , where N is some type of shape function
    /// evaluated at U,V,W coordinates of the volume, each ranging in -1..+1
    /// F is a load, N'*F is the resulting generalized load
    /// Returns also det[J] with J=[dx/du,..], that might be useful in gauss quadrature.
    virtual void ComputeNF(const double U,              ///< parametric coordinate in volume
                           const double V,              ///< parametric coordinate in volume
                           const double W,              ///< parametric coordinate in volume
                           ChVectorDynamic<>& Qi,       ///< Return result of N'*F  here, maybe with offset block_offset
                           double& detJ,                ///< Return det[J] here
                           const ChVectorDynamic<>& F,  ///< Input F vector, size is = n.field coords.
                           ChVectorDynamic<>* state_x,  ///< if != 0, update state (pos. part) to this, then evaluate Q
                           ChVectorDynamic<>* state_w   ///< if != 0, update state (speed part) to this, then evaluate Q
                           ) override;

    /// This is needed so that it can be accessed by ChLoaderVolumeGravity.
    /// Density is mass per unit surface.
    virtual double GetDensity() override;

    /// Gets the normal to the surface at the parametric coordinate U,V.
    /// Each coordinate ranging in -1..+1.
    virtual ChVector<> ComputeNormal(const double U, const double V) override;

  private:
    /// Initial setup. This is used to precompute matrices that do not change during the simulation, such as the local
    /// stiffness of each element (if any), the mass, etc.
    virtual void SetupInitial(ChSystem* system) override;

    /// Compute the mass matrix & generalized gravity force of the element.
    /// Note: in this 'basic' implementation, a constant rectangular cross-section and
    /// a constant density material are assumed
    void ComputeMassMatrixAndGravityForce(const ChVector<>& g_acc);

    void PrecomputeInternalForceMatricesWeights();

    // Internal computations
    // ---------------------

    /// Compute Jacobians of the internal forces.
    /// This function calculates a linear combination of the stiffness (K) and damping (R) matrices,
    ///     J = Kfactor * K + Rfactor * R
    /// for given coefficients Kfactor and Rfactor.
    /// This Jacobian will be further combined with the global mass matrix M and included in the global
    /// stiffness matrix H in the function ComputeKRMmatricesGlobal().
    void ComputeInternalJacobians(Matrix3Nx3N& JacobianMatrix, double Kfactor, double Rfactor);

    // Calculate the generalized internal force for the element given the provided current state coordinates
    void ComputeInternalForcesAtState(ChVectorDynamic<>& Fi, const MatrixNx3c& ebar, const MatrixNx3c& ebardot);

    // Return the pre-computed generalized force due to gravity
    void Get_GravityFrc(Vector3N& Gi) { Gi = m_GravForce; }

    // Calculate the current 72x1 vector of nodal coordinates.
    void CalcCoordVector(Vector3N& e);

    // Calculate the current 3x24 matrix of nodal coordinates.
    void CalcCoordMatrix(Matrix3xN& ebar);
    void CalcCoordMatrix(MatrixNx3c& ebar);

    // Calculate the current 72x1 vector of nodal coordinate time derivatives.
    void CalcCoordDerivVector(Vector3N& edot);

    // Calculate the current 3x24 matrix of nodal coordinate time derivatives.
    void CalcCoordDerivMatrix(Matrix3xN& ebardot);
    void CalcCoordDerivMatrix(MatrixNx3c& ebardot);

    // Calculate the 3x72 Sparse & Repetitive Normalized Shape Function Matrix
    void Calc_Sxi(Matrix3x3N& Sxi, double xi, double eta, double zeta);

    // Calculate the 24x1 Compact Vector of the Normalized Shape Functions
    void Calc_Sxi_compact(VectorN& Sxi_compact, double xi, double eta, double zeta);

    // Calculate the 3x72 Sparse & Repetitive Derivative of the Normalized Shape Function Matrix with respect to xi
    void Calc_Sxi_xi(Matrix3x3N& Sxi_xi, double xi, double eta, double zeta);

    // Calculate the 24x1 Compact Vector of the Derivative of the Normalized Shape Functions with respect to xi
    void Calc_Sxi_xi_compact(VectorN& Sxi_xi_compact, double xi, double eta, double zeta);

    // Calculate the 3x72 Sparse & Repetitive Derivative of the Normalized Shape Function Matrix with respect to eta
    void Calc_Sxi_eta(Matrix3x3N& Sxi_eta, double xi, double eta, double zeta);

    // Calculate the 24x1 Compact Vector of the Derivative of the Normalized Shape Functions with respect to eta
    void Calc_Sxi_eta_compact(VectorN& Sxi_eta_compact, double xi, double eta, double zeta);

    // Calculate the 3x72 Sparse & Repetitive Derivative of the Normalized Shape Function Matrix with respect to zeta
    void Calc_Sxi_zeta(Matrix3x3N& Sxi_zeta, double xi, double eta, double zeta);

    // Calculate the 24x1 Compact Vector of the Derivative of the Normalized Shape Functions with respect to zeta
    void Calc_Sxi_zeta_compact(VectorN& Sxi_zeta_compact, double xi, double eta, double zeta);

    // Calculate the 24x3 Compact Matrix of the Derivatives of the Normalized Shape Functions with respect to xi, eta,
    // and then zeta
    void Calc_Sxi_D(MatrixNx3c& Sxi_D, double xi, double eta, double zeta);

    // Calculate the element Jacobian of the reference configuration with respect to the normalized configuration
    void Calc_J_0xi(ChMatrix33<double>& J_0xi, double xi, double eta, double zeta);

    // Calculate the determinate of the element Jacobian of the reference configuration with respect to the normalized
    // configuration
    double Calc_det_J_0xi(double xi, double eta, double zeta);

    /// Access a statically-allocated set of tables, from 0 to a 10th order,
    /// with precomputed tables.
    static ChQuadratureTables* GetStaticGQTables();

    std::vector<std::shared_ptr<ChNodeFEAxyzDD> > m_nodes;    ///< element nodes
    double m_lenX;                                             ///< total element length along X
    double m_lenY;                                             ///< total element length along Y
    double m_thicknessZ;                                       ///< total element thickness along Z
    double m_Alpha;                                            ///< structural damping
    bool m_damping_enabled;                                    ///< Flag to run internal force damping calculations
    bool m_gravity_on;                                         ///< enable/disable gravity calculation
    Vector3N m_GravForce;                                      ///< Gravity Force
    MatrixNxN m_MassMatrix;                                    ///< mass matrix
    std::shared_ptr<ChMaterialShellANCF_3833_TR09> m_material;  ///< shell material
    StrainFormulation m_strain_form;                           ///< Strain formulation
    Matrix3xN m_ebar0;  ///< Element Position Coordinate Vector for the Reference Configuration
    // For Liu Method
    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 256, 256> m_O1;
    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 256, 256> m_O2;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> m_O1;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> m_O2;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> m_K3Compact;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> m_K13Compact;
    //ChMatrixNM<double, 64, 64> m_O1;
    //ChMatrixNMc<double, 64, 64> m_O2;
    //ChMatrixNM<double, 8, 8> m_K3Compact;
    //ChMatrixNM<double, 8, 8> m_K13Compact;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @} fea_elements

}  // end of namespace fea
}  // end of namespace chrono

#endif

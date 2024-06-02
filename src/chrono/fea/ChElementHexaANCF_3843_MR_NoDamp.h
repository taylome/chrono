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

#ifndef CH_ELEMENT_HEXA_ANCF_3843_MR_NoDamp_H
#define CH_ELEMENT_HEXA_ANCF_3843_MR_NoDamp_H

#include <vector>

#include "chrono/fea/ChElementHexahedron.h"
#include "chrono/fea/ChMaterialHexaANCF_MR.h"
#include "chrono/fea/ChElementANCF.h"
#include "chrono/fea/ChElementGeneric.h"
#include "chrono/fea/ChNodeFEAxyzDDD.h"
#include "chrono/physics/ChLoadable.h"

namespace chrono {
namespace fea {

/// @addtogroup fea_elements
/// @{

/// ANCF brick element with eight nodes.  The coordinates at each node are the position vector
/// and 3 position vector gradients.
///
/// The node numbering is in ccw fashion as in the following scheme:
/// <pre>
/// Bottom Layer:
///         v
///         ^
///         |
/// D o-----+-----o C
///   |     |     |
/// --+-----+-----+----> u
///   |     |     |
/// A o-----+-----o B
///
/// Top Layer:
///         v
///         ^
///         |
/// H o-----+-----o G
///   |     |     |
/// --+-----+-----+----> u
///   |     |     |
/// E o-----+-----o F
/// </pre>

class ChApi ChElementHexaANCF_3843_MR_NoDamp : public ChElementANCF, public ChElementHexahedron, public ChElementGeneric, public ChLoadableUVW {
  public:
    // Using fewer than 3 Gauss quadrature will likely result in numerical issues with the element.
    static const unsigned int NP = 4;              ///< number of Gauss quadrature along beam axis
    static const unsigned int NIP = NP * NP * NP;  ///< number of Gauss quadrature points
    static const unsigned int NSF = 32;            ///< number of shape functions

    // Short-cut for defining a Eigen arrays of length NIP
    using ArrayNIP = Eigen::Array<double, 1, NIP, Eigen::RowMajor>;

    // Short-cut for defining Eigen vectors and matrices of different sizes
    using VectorN = ChVectorN<double, NSF>;
    using Vector3N = ChVectorN<double, 3 * NSF>;
    using VectorNIP = ChVectorN<double, NIP>;
    using Matrix3xN = ChMatrixNM<double, 3, NSF>;
    using MatrixNx3 = ChMatrixNM<double, NSF, 3>;
    using MatrixNx3c = ChMatrixNM_col<double, NSF, 3>;
    using MatrixNx6 = ChMatrixNM<double, NSF, 6>;
    using MatrixNxN = ChMatrixNM<double, NSF, NSF>;
    using Matrix3Nx3N = ChMatrixNM<double, 3 * NSF, 3 * NSF>;
    using Matrix3x3N = ChMatrixNM<double, 3, 3 * NSF>;
    using Matrix6x3N = ChMatrixNM<double, 6, 3 * NSF>;
    using Matrix6xN = ChMatrixNM<double, 6, NSF>;

    ChElementHexaANCF_3843_MR_NoDamp();
    ~ChElementHexaANCF_3843_MR_NoDamp() {}

    /// Get the number of nodes used by this element.
    virtual unsigned int GetNumNodes() override { return 8; }

    /// Get the number of coordinates in the field used by the referenced nodes.
    virtual unsigned int GetNumCoordsPosLevel() override { return 8 * 12; }

    /// Get the number of active coordinates in the field used by the referenced nodes.
    virtual unsigned int GetNumCoordsPosLevelActive() override { return m_element_dof; }

    /// Get the number of coordinates from the n-th node used by this element.
    virtual unsigned int GetNodeNumCoordsPosLevel(unsigned int n) override {
        return m_nodes[n]->GetNumCoordsPosLevel();
    }

    /// Get the number of active coordinates from the n-th node used by this element.
    virtual unsigned int GetNodeNumCoordsPosLevelActive(unsigned int n) override {
        return m_nodes[n]->GetNumCoordsPosLevelActive();
    }

    /// Specify the nodes of this element.
    void SetNodes(std::shared_ptr<ChNodeFEAxyzDDD> nodeA,
                  std::shared_ptr<ChNodeFEAxyzDDD> nodeB,
                  std::shared_ptr<ChNodeFEAxyzDDD> nodeC,
                  std::shared_ptr<ChNodeFEAxyzDDD> nodeD,
                  std::shared_ptr<ChNodeFEAxyzDDD> nodeE,
                  std::shared_ptr<ChNodeFEAxyzDDD> nodeF,
                  std::shared_ptr<ChNodeFEAxyzDDD> nodeG,
                  std::shared_ptr<ChNodeFEAxyzDDD> nodeH);

    /// Specify the element dimensions.
    void SetDimensions(double lenX, double lenY, double lenZ);

    /// Specify the element material.
    void SetMaterial(std::shared_ptr<ChMaterialHexaANCF_MR> brick_mat);

    /// Return the material.
    std::shared_ptr<ChMaterialHexaANCF_MR> GetMaterial() const { return m_material; }

    /// Access the n-th node of this element.
    virtual std::shared_ptr<ChNodeFEAbase> GetNode(unsigned int n) override { return m_nodes[n]; }

    /// Return the specified hexahedron node (0 <= n <= 7).
    virtual std::shared_ptr<ChNodeFEAxyz> GetHexahedronNode(unsigned int n) override { return m_nodes[n]; }

    /// Get a handle to the first node of this element.
    std::shared_ptr<ChNodeFEAxyzDDD> GetNodeA() const { return m_nodes[0]; }

    /// Get a handle to the second node of this element.
    std::shared_ptr<ChNodeFEAxyzDDD> GetNodeB() const { return m_nodes[1]; }

    /// Get a handle to the third node of this element.
    std::shared_ptr<ChNodeFEAxyzDDD> GetNodeC() const { return m_nodes[2]; }

    /// Get a handle to the fourth node of this element.
    std::shared_ptr<ChNodeFEAxyzDDD> GetNodeD() const { return m_nodes[3]; }

    /// Get a handle to the 5th node of this element.
    std::shared_ptr<ChNodeFEAxyzDDD> GetNodeE() const { return m_nodes[4]; }

    /// Get a handle to the 6th node of this element.
    std::shared_ptr<ChNodeFEAxyzDDD> GetNodeF() const { return m_nodes[5]; }

    /// Get a handle to the 7th node of this element.
    std::shared_ptr<ChNodeFEAxyzDDD> GetNodeG() const { return m_nodes[6]; }

    /// Get a handle to the 8th node of this element.
    std::shared_ptr<ChNodeFEAxyzDDD> GetNodeH() const { return m_nodes[7]; }

    /// Get the element length in the X direction.
    double GetLengthX() const { return m_lenX; }

    /// Get the element length in the Y direction.
    double GetLengthY() const { return m_lenY; }

    /// Get the element length in the Z direction.
    double GetLengthZ() const { return m_lenZ; }

    /// Get the Green-Lagrange strain tensor at the normalized element coordinates (xi, eta, zeta) at the current state
    /// of the element.  Normalized element coordinates span from -1 to 1.
    ChMatrix33d GetGreenLagrangeStrain(const double xi, const double eta, const double zeta);

    /// Get the 2nd Piola-Kirchoff stress tensor at the normalized element coordinates (xi, eta, zeta) at the current
    /// state of the element.  Normalized element coordinates span from -1 to 1.
    ChMatrix33d GetPK2Stress(const double xi, const double eta, const double zeta);

    /// Get the von Mises stress value at the normalized element coordinates (xi, eta, zeta) at the current state
    /// of the element.  Normalized element coordinates span from -1 to 1.
    double GetVonMissesStress(const double xi, const double eta, const double zeta);

  public:
    // Interface to ChElementBase base class
    // -------------------------------------

    /// Fill the D vector (column matrix) with the current field values at the nodes of the element, with proper
    /// ordering. If the D vector has not the size of this->GetNumCoordsPosLevel(), it will be resized.
    ///  {Pos_a Du_a Dv_a Dw_a  Pos_b Du_b Dv_b Dw_b  Pos_c Du_c Dv_c Dw_c  Pos_d Du_d Dv_d Dw_d}
    virtual void GetStateBlock(ChVectorDynamic<>& mD) override;

    /// Update the state of this element.
    virtual void Update() override;

    /// Set M equal to the global mass matrix.
    virtual void ComputeMmatrixGlobal(ChMatrixRef M) override;

    /// Add contribution of element inertia to total nodal masses
    virtual void ComputeNodalMass() override;

    /// Compute the generalized internal force vector for the current nodal coordinates as set the value in the Fi
    /// vector.
    virtual void ComputeInternalForces(ChVectorDynamic<>& Fi) override;

    /// Set H as a linear combination of M, K, and R.
    ///   H = Mfactor * [M] + Kfactor * [K] + Rfactor * [R],
    /// where [M] is the mass matrix, [K] is the stiffness matrix, and [R] is the damping matrix.
    virtual void ComputeKRMmatricesGlobal(ChMatrixRef H,
                                          double Kfactor,
                                          double Rfactor = 0,
                                          double Mfactor = 0) override;

    /// Compute the generalized force vector due to gravity using the efficient ANCF specific method
    virtual void ComputeGravityForces(ChVectorDynamic<>& Fg, const ChVector3d& G_acc) override;

    // --------------------------------------

    /// Gets the xyz displacement of a point in the element, and the approximate rotation RxRyRz at that point
    /// '(xi,eta,zeta)'.
    virtual void EvaluateElementDisplacement(const double xi,
                                             const double eta,
                                             const double zeta,
                                             ChVector3d& u_displ,
                                             ChVector3d& u_rotaz) {}

    /// Gets the absolute xyz position of a point in the element, and the approximate rotation RxRyRz at that point
    /// '(xi,eta,zeta)'. Note, nodeA = (xi=-1, eta=-1, zeta=-1) Note, nodeB = (xi=1, eta=-1, zeta=-1) Note, nodeC =
    /// (xi=1, eta=1, zeta=-1) Note, nodeD = (xi=-1, eta=1, zeta=-1) Note, nodeE = (xi=-1, eta=-1, zeta=1) Note, nodeF =
    /// (xi=1, eta=-1, zeta=1) Note, nodeG = (xi=1, eta=1, zeta=1) Note, nodeH = (xi=-1, eta=1, zeta=1)
    virtual void EvaluateElementFrame(const double xi,
                                      const double eta,
                                      const double zeta,
                                      ChVector3d& point,
                                      ChQuaternion<>& rot);

    /// Gets the absolute xyz position of a point in the element specified in normalized coordinates
    virtual void EvaluateElementPoint(const double xi, const double eta, const double zeta, ChVector3d& point);

    /// Gets the absolute xyz velocity of a point in the element specified in normalized coordinates
    virtual void EvaluateElementVel(const double xi, const double eta, const double zeta, ChVector3d& Result);

    // Functions for ChLoadable interface
    // ----------------------------------

    /// Gets the number of DOFs affected by this element (position part).
    virtual unsigned int GetLoadableNumCoordsPosLevel() override { return 8 * 12; }

    /// Gets the number of DOFs affected by this element (velocity part).
    virtual unsigned int GetLoadableNumCoordsVelLevel() override { return 8 * 12; }

    /// Gets all the DOFs packed in a single vector (position part).
    virtual void LoadableGetStateBlockPosLevel(int block_offset, ChState& mD) override;

    /// Gets all the DOFs packed in a single vector (velocity part).
    virtual void LoadableGetStateBlockVelLevel(int block_offset, ChStateDelta& mD) override;

    /// Increment all DOFs using a delta.
    virtual void LoadableStateIncrement(const unsigned int off_x,
                                        ChState& x_new,
                                        const ChState& x,
                                        const unsigned int off_v,
                                        const ChStateDelta& Dv) override;

    /// Number of coordinates in the interpolated field, ex=3 for a
    /// tetrahedron finite element or a cable, = 1 for a thermal problem, etc.
    virtual unsigned int GetNumFieldCoords() override { return 12; }

    /// Tell the number of DOFs blocks (ex. =1 for a body, =4 for a tetrahedron, etc.)
    virtual unsigned int GetNumSubBlocks() override { return 8; }

    /// Get the offset of the i-th sub-block of DOFs in global vector.
    virtual unsigned int GetSubBlockOffset(unsigned int nblock) override {
        return m_nodes[nblock]->NodeGetOffsetVelLevel();
    }

    /// Get the size of the i-th sub-block of DOFs in global vector.
    virtual unsigned int GetSubBlockSize(unsigned int nblock) override { return 12; }

    /// Check if the specified sub-block of DOFs is active.
    virtual bool IsSubBlockActive(unsigned int nblock) const override { return !m_nodes[nblock]->IsFixed(); }

    /// Get the pointers to the contained ChVariables, appending to the mvars vector.
    virtual void LoadableGetVariables(std::vector<ChVariables*>& mvars) override;

    /// Evaluate N'*F , where N is some type of shape function
    /// evaluated at xi,eta,zeta coordinates of the volume, each ranging in -1..+1
    /// F is a load, N'*F is the resulting generalized load
    /// Returns also det[J] with J=[dx/du,..], that might be useful in gauss quadrature.
    /// For this ANCF element, only the first 6 entries in F are used in the calculation.  The first three entries is
    /// the applied force in global coordinates and the second 3 entries is the applied moment in global space.
    virtual void ComputeNF(const double xi,             ///< parametric coordinate in volume
                           const double eta,            ///< parametric coordinate in volume
                           const double zeta,           ///< parametric coordinate in volume
                           ChVectorDynamic<>& Qi,       ///< Return result of N'*F  here, maybe with offset block_offset
                           double& detJ,                ///< Return det[J] here
                           const ChVectorDynamic<>& F,  ///< Input F vector, size is = n.field coords.
                           ChVectorDynamic<>* state_x,  ///< if != 0, update state (pos. part) to this, then evaluate Q
                           ChVectorDynamic<>* state_w   ///< if != 0, update state (speed part) to this, then evaluate Q
                           ) override;

    /// This is needed so that it can be accessed by ChLoaderVolumeGravity.
    /// Density is the average mass per unit volume in the reference state of the element.
    virtual double GetDensity() override;

  private:
    /// Initial setup. This is used to precompute matrices that do not change during the simulation, such as the local
    /// stiffness of each element (if any), the mass, etc.
    virtual void SetupInitial(ChSystem* system) override;

    // Internal computations
    // ---------------------

    /// Compute the mass matrix & generalized gravity force of the element.
    /// Note: in this implementation, a constant density material is assumed
    void ComputeMassMatrixAndGravityForce();

    /// Precalculate constant matrices and scalars for the internal force calculations.
    void PrecomputeInternalForceMatricesWeights();

    /// Calculate the generalized internal force for the element at the current nodal coordinates and time derivatives
    /// of the nodal coordinates assuming damping is included
    void ComputeInternalForceDamping(ChVectorDynamic<>& Fi);

    /// Calculate the generalized internal force for the element at the current nodal coordinates and time derivatives
    /// of the nodal coordinates assuming no damping
    void ComputeInternalForceNoDamping(ChVectorDynamic<>& Fi);

    /// Calculate the calculate the Jacobian of the internal force integrand assuming damping is included. This function calculates
    ///     H = Kfactor * K + Rfactor * R + Mfactor * M
    /// for given coefficients Kfactor, Rfactor, and Mfactor.
    /// This Jacobian includes the global mass matrix (M) with the global stiffness (K) and damping matrix (R) in H.
    void ComputeInternalJacobianDamping(ChMatrixRef H, double Kfactor, double Rfactor, double Mfactor);

    /// Calculate the calculate the Jacobian of the internal force integrand assuming damping is not included. This function calculates
    ///     H = Kfactor * K + Mfactor * M
    /// for given coefficients Kfactor and Mfactor.
    /// This Jacobian includes the global mass matrix (M) with the global stiffness (K) in H.
    void ComputeInternalJacobianNoDamping(ChMatrixRef H, double Kfactor, double Mfactor);

    /// Calculate the current 3Nx1 vector of nodal coordinates.
    void CalcCoordVector(Vector3N& e);

    /// Calculate the current 3xN matrix of nodal coordinates.
    void CalcCoordMatrix(Matrix3xN& ebar);

    /// Calculate the current 3Nx1 vector of nodal coordinate time derivatives.
    void CalcCoordDtVector(Vector3N& edot);

    /// Calculate the current 3xN matrix of nodal coordinate time derivatives.
    void CalcCoordDtMatrix(Matrix3xN& ebardot);

    /// Calculate the current 6xN matrix of the nodal coordinates and nodal coordinate time derivatives.
    void CalcCombinedCoordMatrix(Matrix6xN& ebar_ebardot);

    /// Calculate the Nx1 Compact Vector of the Normalized Shape Functions (just the unique values)
    void Calc_Sxi_compact(VectorN& Sxi_compact, double xi, double eta, double zeta);

    /// Calculate the Nx1 Compact Vector of the Derivative of the Normalized Shape Functions with respect to xi
    void Calc_Sxi_xi_compact(VectorN& Sxi_xi_compact, double xi, double eta, double zeta);

    /// Calculate the Nx1 Compact Vector of the Derivative of the Normalized Shape Functions with respect to eta
    void Calc_Sxi_eta_compact(VectorN& Sxi_eta_compact, double xi, double eta, double zeta);

    /// Calculate the Nx1 Compact Vector of the Derivative of the Normalized Shape Functions with respect to zeta
    void Calc_Sxi_zeta_compact(VectorN& Sxi_zeta_compact, double xi, double eta, double zeta);

    /// Calculate the Nx3 Compact Matrix of the Derivatives of the Normalized Shape Functions with respect to xi, eta,
    /// and then zeta
    void Calc_Sxi_D(MatrixNx3c& Sxi_D, double xi, double eta, double zeta);

    /// Calculate the element Jacobian of the reference configuration with respect to the normalized configuration
    void Calc_J_0xi(ChMatrix33d& J_0xi, double xi, double eta, double zeta);

    /// Calculate the determinate of the element Jacobian of the reference configuration with respect to the normalized
    /// configuration
    double Calc_det_J_0xi(double xi, double eta, double zeta);

    /// Access a statically-allocated set of tables, from 0 to a 10th order, with precomputed tables.
    static ChQuadratureTables* GetStaticGQTables();

    std::shared_ptr<ChMaterialHexaANCF_MR> m_material;      ///< material model
    std::vector<std::shared_ptr<ChNodeFEAxyzDDD>> m_nodes;  ///< element nodes
    double m_lenX;                                          ///< total element length along X
    double m_lenY;                                          ///< total element length along Y
    double m_lenZ;                                          ///< total element length along Z
    VectorN m_GravForceScale;  ///< Gravity scaling matrix used to get the generalized force due to gravity
    Matrix3xN m_ebar0;         ///< Element Position Coordinate Vector for the Reference Configuration
    ChVectorN<double, (NSF * (NSF + 1)) / 2>
        m_MassMatrix;  /// Mass Matrix in extra compact form (Upper Triangular Part only)
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        m_SD;  ///< Precomputed corrected normalized
               ///< shape function derivative matrices
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        m_kGQ;  ///< Precomputed Gauss-Quadrature Weight & Element Jacobian
                ///< scale factors

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @} fea_elements

}  // end of namespace fea
}  // end of namespace chrono

#endif
